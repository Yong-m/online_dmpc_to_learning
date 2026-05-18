from __future__ import annotations
import itertools
import math
import torch
from torch.distributions import Normal, kl_divergence
from typing import Any, Mapping, Optional, Union, Dict, Tuple, Callable
from policyflow_torch.modules import Network, ContinuousNormalizingFlow
from policyflow_torch.agents import PolicyFlowBase
from policyflow_torch.storage import ReplayBuffer
from policyflow_torch.utils.kl_adaptive import KLAdaptiveLR


class PolicyFlow(PolicyFlowBase):
    def __init__(
        self,
        models: Mapping[
            str, Union[ContinuousNormalizingFlow | Network | torch.nn.Module]
        ],
        replay_buffer: ReplayBuffer,
        cfg: dict = dict(),
        device: Optional[Union[str, torch.device]] = None,
        optim_params: Optional[dict] = None,
    ) -> None:
        if optim_params is None:
            optim_params = {"weight_decay": 1e-5}

        super().__init__(models, replay_buffer, cfg, device)

        self._desired_kl = cfg.get("desired_kl", 0.01)
        self._learning_rate = cfg.get("learning_rate", 1e-4)
        self._critic_learning_rate = cfg.get("critic_learning_rate", self._learning_rate)
        self._discount_factor = cfg.get("discount_factor", 0.99)
        self._lambda = cfg.get("lam", 0.95)
        self._time_limit_bootstrap = cfg.get("time_limit_bootstrap", True)
        self._mini_batches = cfg.get("mini_batches", 1)
        self._learning_epochs = cfg.get("learning_epochs", 1)
        self._gaussian_entropy_loss_scale = cfg.get("gaussian_entropy_loss_scale", 0.0)
        self._brownian_reg_loss_scale = cfg.get("brownian_reg_loss_scale", 0.0)
        self._ratio_clip = cfg.get("ratio_clip", 0.2)
        self._clip_predicted_values = cfg.get("clip_predicted_values", True)
        self._value_clip = cfg.get("value_clip", 0.1)
        self._value_loss_scale = cfg.get("value_loss_scale", 1.0)
        self._grad_norm_clip = cfg.get("grad_norm_clip", 1.0)
        self._degenerate2gaussian = cfg.get("degenerate2gaussian", False)
        self._action_clip = cfg.get("action_clip", 1.0)  # 0 = disabled

        # Auxiliary loss: adds external loss (e.g. Chamfer) from condition network to total_loss.
        # The condition network stores _chamfer_loss during forward pass; this injects it
        # into the backward graph so gradients flow to TactilePoseEstimator.
        # Set to 0.0 to disable (default, original behavior).
        self._auxiliary_loss_scale = cfg.get("auxiliary_loss_scale", 0.0)

        # BC Regularization loss hook: callable(obs, delta_actions, action_dist, actions_prior) -> tensor or 0
        # Set externally via agent._bc_loss_fn = my_fn to enable two-phase BC guidance.
        # Signature: fn(sampled_actor_observations, sampled_delta_actions, action_distribution_new, sampled_actions_prior)
        self._bc_loss_fn: Optional[Callable] = None

        # BC reference model for KL-based regularization (set externally via set_bc_ref_model()).
        # This is a frozen copy of the warmup checkpoint actor; gradients do not flow through it.
        # KL(current_policy || ref_policy) is added to the PPO loss with a decaying coefficient.
        #
        # Coefficient schedule — two-phase decay (see `_bc_ref_kl_coef` for math):
        #   Phase 1 (t ≤ decay_steps): inverse-linear from coef0 to coef0/100.
        #   Phase 2 (t > decay_steps): harmonic continuation; at k×decay_steps
        #     the coefficient is coef0/(100·k), so 1/200 at 2×, 1/300 at 3×, …
        #   bc_ref_kl_min (default 0) is a hard floor; 0 → decay continues forever.
        #
        # cfg keys:
        #   bc_ref_kl_coef0     : initial KL coefficient (default 1.0)
        #   bc_ref_kl_decay_steps: PPO update steps to reach coef0/100 (default 100)
        #   bc_ref_kl_min       : floor coefficient (default 0.0 → no floor)
        self._bc_ref_model: Optional[object] = None   # frozen ContinuousNormalizingFlow
        self._bc_ref_kl_coef0       = cfg.get("bc_ref_kl_coef0",        1.0)
        self._bc_ref_kl_decay_steps = cfg.get("bc_ref_kl_decay_steps",  100.0)  # steps to reach 1/100
        self._bc_ref_kl_min         = cfg.get("bc_ref_kl_min",          0.0)
        self._bc_ref_kl_pose_only   = cfg.get("bc_ref_kl_pose_only",    False)  # KL on dims 0:6 only
        self._bc_ref_update_step    = 0   # counts PPO update() calls

        # PPO-EWMA config (Hilton et al., "Batch size-invariance for policy optimization")
        self._use_ewma = cfg.get("use_ewma", False)
        self._beta_prox = cfg.get("beta_prox", 0.889)
        self._kl_penalty = cfg.get("kl_penalty", 0.0)
        self._imp_samp_max = cfg.get("imp_samp_max", 0.0)

        # KL-based LR scheduling (disable for EWMA — constant LR recommended)
        self._use_kl_lr_schedule = cfg.get("use_kl_lr_schedule", True)

        # Actor LR linear warmup: ramp from _actor_lr_warmup_start to _learning_rate
        # over _actor_lr_warmup_steps PPO update() calls.
        # During warmup the actor param group LR is set manually after each KL-schedule step.
        # cfg keys:
        #   actor_lr_warmup_steps: number of PPO updates for warmup (default 0 = disabled)
        #   actor_lr_warmup_start: initial actor LR at step 0 (default 1e-7)
        self._actor_lr_warmup_steps = int(cfg.get("actor_lr_warmup_steps", 0))
        self._actor_lr_warmup_start = float(cfg.get("actor_lr_warmup_start", 1e-7))
        self._actor_lr_update_step  = 0   # counts PPO update() calls for warmup

        # Actor and critic are in separate param groups so their LRs can be controlled
        # independently (warmup for actor, normal for critic from the start).
        # Group 0 = actor, Group 1 = critic  (order referenced below).
        _actor_init_lr = (
            self._actor_lr_warmup_start
            if self._actor_lr_warmup_steps > 0
            else self._learning_rate
        )
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(self.model_dict["actor"].model.parameters()),
                    "lr": _actor_init_lr,
                    "name": "actor",
                },
                {
                    "params": list(self.model_dict["critic"].parameters()),
                    "lr": self._critic_learning_rate,
                    "name": "critic",
                },
            ],
            **optim_params,
        )
        if self._use_kl_lr_schedule:
            self.lr_schedule = KLAdaptiveLR(
                self.optimizer,
                **cfg.get(
                    "learning_rate_scheduler_kwargs",
                    {
                        "kl_threshold": self._desired_kl,
                    },
                ),
            )
        else:
            self.lr_schedule = None
        self._register_serializable("optimizer")

    def set_bc_ref_model(self, ref_actor: "ContinuousNormalizingFlow") -> None:
        """Attach a frozen reference actor (from warmup checkpoint) for KL regularization.

        The reference model is frozen (no gradient) and stays on the same device as the
        live actor.  Call this once after loading the warmup checkpoint.

        Args:
            ref_actor: A ContinuousNormalizingFlow whose weights will be used as the
                       reference distribution π_ref(a|s).  Its parameters are frozen here.
        """
        import copy
        self._bc_ref_model = copy.deepcopy(ref_actor)
        # Freeze all parameters — no gradients ever flow through the reference model.
        for p in self._bc_ref_model.model.parameters():
            p.requires_grad_(False)
        self._bc_ref_model.model.eval()
        # Move to same device as live actor
        self._bc_ref_model.model.to(self.device)
        print(f"[BC-Ref] Frozen reference actor attached for KL regularization "
              f"(coef0={self._bc_ref_kl_coef0}, decay_steps={self._bc_ref_kl_decay_steps}, "
              f"min={self._bc_ref_kl_min})")

    def _bc_ref_kl_coef(self) -> float:
        """Compute current KL coefficient using a two-phase decay schedule.

        Phase 1 (t ≤ decay_steps): inverse-linear from coef0 → coef0/100.
          denom = 1 + 99 * t / decay_steps
          → coef(0) = coef0, coef(decay_steps) = coef0/100.

        Phase 2 (t > decay_steps): harmonic continuation past coef0/100.
          denom = 100 * t / decay_steps
          → coef(k * decay_steps) = coef0 / (100 * k) for any k ≥ 1.
          e.g. coef0/100 at decay_steps, coef0/200 at 2×, coef0/300 at 3×.

        `bc_ref_kl_min` (default 0) acts as a hard floor; with 0 the decay
        continues indefinitely.
        """
        coef0       = self._bc_ref_kl_coef0
        coef_min    = self._bc_ref_kl_min  # 0 → no floor, decay continues forever
        decay_steps = self._bc_ref_kl_decay_steps
        t = self._bc_ref_update_step
        if t <= decay_steps:
            denom = 1.0 + 99.0 * t / decay_steps
        else:
            denom = 100.0 * t / decay_steps
        coef = coef0 / denom
        return max(coef_min, coef)

    def init_replay_buffer(
        self,
        critic_observation_size: Union[int, Tuple[int]],
        actor_observation_size: Union[int, Tuple[int]],
        action_size: Union[int, Tuple[int]],
    ) -> None:
        """Initialize the agent"""
        self.eval_mode()
        self._action_size = action_size
        self._register_serializable("_action_size")

        # create tensors in replay_buffer
        if self.replay_buffer is not None:
            self.replay_buffer.create_tensor(
                name="critic_observations",
                size=critic_observation_size,
                dtype=torch.float32,
            )
            self.replay_buffer.create_tensor(
                name="actor_observations",
                size=actor_observation_size,
                dtype=torch.float32,
            )
            self.replay_buffer.create_tensor(
                name="next_critic_observations",
                size=critic_observation_size,
                dtype=torch.float32,
            )
            self.replay_buffer.create_tensor(
                name="actions_prior", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="flow_x0", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="delta_actions", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="delta_actions_std", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="delta_actions_log_prob", size=1, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="rewards", size=1, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="terminated", size=1, dtype=torch.bool
            )
            self.replay_buffer.create_tensor(name="values", size=1, dtype=torch.float32)
            self.replay_buffer.create_tensor(
                name="returns", size=1, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="advantages", size=1, dtype=torch.float32
            )

    def draw_actions(
        self, observations_dict: Dict[str, torch.Tensor], env_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]:

        x0 = torch.randn(
            (
                observations_dict["actor_observations"].shape[0],
                self._action_size,
            ),
            device=self.device,
        )
        
        if self._degenerate2gaussian:
            x0 = torch.zeros_like(x0)

        actions_prior, std = self.model_dict["actor"].sample(
            x0=x0,
            condition=observations_dict["actor_observations"],
            n_samples=observations_dict["actor_observations"].shape[0],
        )
        delta_action_distribution = torch.distributions.Normal(
            torch.zeros_like(actions_prior), std
        )
        delta_actions = delta_action_distribution.sample().detach()
        actions = actions_prior.detach() + delta_actions
        if self._action_clip > 0:
            actions = torch.tanh(actions / self._action_clip) * self._action_clip
        delta_actions_logp = delta_action_distribution.log_prob(delta_actions).sum(-1)

        info = {
            "actions_prior": actions_prior.detach(),
            "delta_actions": delta_actions.detach(),
            "delta_actions_std": std.detach(),
            "delta_actions_log_prob": delta_actions_logp.detach(),
            "flow_x0": x0.clone(),
        }

        return actions, info

    def process_transition(
        self,
        observations_dict: Dict[str, torch.Tensor],
        environement_info: Dict[str, Any],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations_dict: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        actions_info: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        if self.replay_buffer is not None:
            # compute values
            values = self.model_dict["critic"](observations_dict["critic_observations"].flatten(start_dim=1))

            # time-limit (truncation) boostrapping
            truncated = environement_info.get("time_outs", torch.zeros_like(dones))
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # Lazily register per-env auxiliary tensors (zeros as fallback for non-TRO mode)
            _n_envs = rewards.shape[0]
            _rb_dev  = self.replay_buffer.device

            _spawn_dist = environement_info.get("spawn_dist", None)
            if _spawn_dist is None:
                _spawn_dist = torch.zeros(_n_envs, device=_rb_dev)
            if "spawn_dist" not in self.replay_buffer.tensors:
                self.replay_buffer.create_tensor("spawn_dist", 1)

            _tro_inner_q = environement_info.get("tro_inner_q", None)
            if _tro_inner_q is None:
                _tro_inner_q = torch.zeros((_n_envs, 12), device=_rb_dev)
            if "tro_inner_q" not in self.replay_buffer.tensors:
                self.replay_buffer.create_tensor("tro_inner_q", _tro_inner_q.shape[-1])

            # storage transition in replay_buffer
            self.replay_buffer.add_samples(
                critic_observations=observations_dict["critic_observations"],
                next_critic_observations=next_observations_dict["critic_observations"],
                actor_observations=observations_dict["actor_observations"],
                actions_prior=actions_info["actions_prior"],
                flow_x0=actions_info["flow_x0"],
                delta_actions=actions_info["delta_actions"],
                delta_actions_std=actions_info["delta_actions_std"],
                delta_actions_log_prob=actions_info["delta_actions_log_prob"],
                rewards=rewards,
                terminated=dones,
                values=values,
                spawn_dist=_spawn_dist,
                tro_inner_q=_tro_inner_q,
            )

    def compute_gae(self) -> torch.Tensor:
        """Compute the Generalized Advantage Estimator (GAE)"""
        rewards = self.replay_buffer.get_tensor_by_name("rewards")
        values = self.replay_buffer.get_tensor_by_name("values")
        dones = self.replay_buffer.get_tensor_by_name("terminated")
        next_critic_observations = self.replay_buffer.get_tensor_by_name(
            "next_critic_observations"
        )[-1]

        with torch.inference_mode():
            last_values = self.model_dict["critic"](
                next_critic_observations.flatten(start_dim=1)
            ).detach()

        advantage = 0
        advantages = torch.zeros_like(rewards)
        not_dones = dones.logical_not()
        memory_size = rewards.shape[0]

        # advantages computation
        for i in reversed(range(memory_size)):
            next_values = values[i + 1] if i < memory_size - 1 else last_values
            advantage = (
                rewards[i]
                - values[i]
                + self._discount_factor
                * not_dones[i]
                * (next_values + self._lambda * advantage)
            )
            advantages[i] = advantage
        # returns computation
        returns = advantages + values
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.replay_buffer.set_tensor_by_name("returns", returns)
        self.replay_buffer.set_tensor_by_name("advantages", advantages)

    def update(self) -> Dict[str, Union[float, torch.Tensor]]:
        self.compute_gae()
        self.train_mode()

        actor = self.model_dict["actor"]
        use_ewma = self._use_ewma and actor.model_proximal is not None

        cumulative_policy_loss = 0
        cumulative_gaussian_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_brownian_reg_loss = 0
        cumulative_kl_penalty_loss = 0
        cumulative_auxiliary_loss = 0
        cumulative_bc_loss = 0
        cumulative_bc_ref_kl_loss = 0
        delta_vel_max = -100.0
        delta_vel_min = 100.0

        # BC-reference KL coefficient for this update step
        _bc_ref_coef = self._bc_ref_kl_coef() if self._bc_ref_model is not None else 0.0

        kl_divergences = []
        clip_fractions = []

        # learning epochs
        for _ in range(self._learning_epochs):

            # sample mini-batches from replay_buffer
            sampled_batches = self.replay_buffer.sample_all(
                names=[
                    "critic_observations",
                    "actor_observations",
                    "flow_x0",
                    "actions_prior",
                    "delta_actions",
                    "delta_actions_std",
                    "delta_actions_log_prob",
                    "values",
                    "returns",
                    "advantages",
                    "spawn_dist",
                    "tro_inner_q",
                ],
                mini_batches=self._mini_batches,
            )

            # mini-batches loop
            for (
                sampled_critic_observations,
                sampled_actor_observations,
                sampled_flow_x0,
                sampled_actions_prior,
                sampled_delta_actions,
                sampled_delta_actions_std,
                sampled_delta_actions_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_spawn_dist,
                sampled_tro_inner_q,
            ) in sampled_batches:

                # ── compute flow variation ──────────────────────────────
                compute_breg = (
                    not self._degenerate2gaussian
                    and self._brownian_reg_loss_scale > 0
                )
                result = actor.compute_flow_variation(
                    x1=sampled_actions_prior,
                    condition=sampled_actor_observations,
                    x0=sampled_flow_x0,
                    compute_brownian_reg_loss=compute_breg,
                    return_proximal_info=use_ewma,
                )

                # Unpack results based on flags
                if compute_breg and use_ewma:
                    delta_vel, delta_std_new, brownian_reg_loss, proximal_info = result
                elif compute_breg:
                    delta_vel, delta_std_new, brownian_reg_loss = result
                    proximal_info = None
                elif use_ewma:
                    delta_vel, delta_std_new, proximal_info = result
                    brownian_reg_loss = 0
                else:
                    delta_vel, delta_std_new = result
                    proximal_info = None
                    brownian_reg_loss = 0

                # Scale brownian reg loss
                if compute_breg:
                    if "anneal_coef" in self.model_dict:
                        brownian_reg_loss = (
                            self._brownian_reg_loss_scale
                            * self.model_dict["anneal_coef"].forward()
                            * brownian_reg_loss
                        )
                    else:
                        brownian_reg_loss = (
                            self._brownian_reg_loss_scale * brownian_reg_loss
                        )

                # ── current policy distribution ─────────────────────────
                action_distribution_new = torch.distributions.Normal(
                    delta_vel, delta_std_new
                )
                actions_log_prob_new = action_distribution_new.log_prob(
                    sampled_delta_actions
                ).sum(-1)

                # ── entropy loss ────────────────────────────────────────
                if self._gaussian_entropy_loss_scale:
                    gaussian_entropy_loss = (
                        -self._gaussian_entropy_loss_scale
                        * action_distribution_new.entropy().sum(dim=-1).mean()
                    )
                else:
                    gaussian_entropy_loss = 0

                # ── policy loss ─────────────────────────────────────────
                kl_penalty_loss = 0

                if use_ewma and proximal_info is not None:
                    # ════════════════════════════════════════════════════
                    # Decoupled clipped objective (PPO-EWMA)
                    # Reference: Hilton et al., Appendix A (Algorithm 2)
                    # ppo-ewma/ppo_ewma/ppo.py compute_losses()
                    # ════════════════════════════════════════════════════
                    prox_delta_vel, prox_std = proximal_info

                    # Proximal policy log prob
                    prox_dist = torch.distributions.Normal(
                        prox_delta_vel, prox_std
                    )
                    log_prob_prox = prox_dist.log_prob(
                        sampled_delta_actions
                    ).sum(-1)

                    # log(π_θ / π_prox) — for clipping
                    logratio = actions_log_prob_new - log_prob_prox

                    # Adjusted behavior log prob (IS ratio clipping for stability)
                    logp_adj = sampled_delta_actions_log_prob
                    if self._imp_samp_max > 0:
                        logp_adj = torch.max(
                            sampled_delta_actions_log_prob,
                            actions_log_prob_new.detach()
                            - math.log(self._imp_samp_max),
                        )

                    # Unclipped surrogate: -A * (π_θ / π_behav)
                    pg_losses = -sampled_advantages * torch.exp(
                        actions_log_prob_new - logp_adj
                    )

                    # Clipped surrogate: -A * clip(π_θ/π_prox) * (π_prox/π_behav)
                    clipped_logratio = torch.clamp(
                        logratio,
                        math.log(1.0 - self._ratio_clip),
                        math.log(1.0 + self._ratio_clip),
                    )
                    pg_losses2 = -sampled_advantages * torch.exp(
                        clipped_logratio + log_prob_prox - logp_adj
                    )

                    policy_loss = torch.max(pg_losses, pg_losses2).mean()

                    # Optional KL penalty to proximal
                    if self._kl_penalty > 0:
                        kl_penalty_loss = (
                            self._kl_penalty * 0.5 * (logratio ** 2).mean()
                        )

                    # KL for lr scheduling: approximate KL to proximal
                    kl_divergences.append(
                        (0.5 * (logratio ** 2).mean()).detach()
                    )

                    # Clip fraction diagnostic
                    with torch.no_grad():
                        clip_fractions.append(
                            torch.logical_or(
                                logratio < math.log(1.0 - self._ratio_clip),
                                logratio > math.log(1.0 + self._ratio_clip),
                            ).float().mean().item()
                        )
                else:
                    # ════════════════════════════════════════════════════
                    # Standard PPO clipped objective (original behavior)
                    # ════════════════════════════════════════════════════
                    ratio = torch.exp(
                        actions_log_prob_new - sampled_delta_actions_log_prob
                    )
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # KL for lr scheduling (vs behavior, original)
                    kl_divergences.append(
                        self._compute_kl_divergence(
                            delta_vel,
                            delta_std_new,
                            torch.zeros_like(delta_vel),
                            sampled_delta_actions_std,
                        )
                    )

                # ── value regression loss ───────────────────────────────
                predicted_values = self.model_dict["critic"](
                    sampled_critic_observations
                )
                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(
                        predicted_values - sampled_values,
                        min=-self._value_clip,
                        max=self._value_clip,
                    )
                value_loss = self._value_loss_scale * torch.nn.functional.mse_loss(
                    sampled_returns, predicted_values
                )

                # ── BC reference KL regularization ─────────────────────
                # KL( π_current(a|s) || π_ref(a|s) ) with decaying coefficient.
                # Both distributions are Gaussian in flow-variation space (delta_vel, std).
                # π_ref is evaluated with no_grad since its parameters are frozen.
                bc_ref_kl_loss = 0
                if self._bc_ref_model is not None and _bc_ref_coef > 0:
                    with torch.no_grad():
                        ref_cond = self._bc_ref_model.model["condition"](
                            sampled_actor_observations
                        )
                        # Use the same (xt, t) that compute_flow_variation already used
                        # — we re-derive them here from sampled_flow_x0 and sampled_actions_prior.
                        # Draw a fresh random t for the reference query (same distribution).
                        _ref_t = torch.rand(
                            sampled_actions_prior.shape[0], device=self.device
                        )
                        _ref_alpha = _ref_t.unsqueeze(-1)
                        _ref_xt = (
                            (1.0 - _ref_alpha) * sampled_flow_x0
                            + _ref_alpha * sampled_actions_prior
                        )
                        ref_vel = self._bc_ref_model.model["flow"](
                            _ref_xt, _ref_t, ref_cond
                        )
                        ref_std = torch.ones_like(ref_vel) * self._bc_ref_model.model["variance"].std

                    # Re-query current model at the same (xt, t) for a fair comparison.
                    cur_cond = actor.model["condition"](sampled_actor_observations)
                    cur_vel  = actor.model["flow"](_ref_xt, _ref_t, cur_cond)
                    cur_std  = torch.ones_like(cur_vel) * actor.model["variance"].std

                    if self._bc_ref_kl_pose_only:
                        # KL only on translation (0:3) + rotation (3:6), skip finger (6:)
                        cur_dist = Normal(cur_vel[:, :6], cur_std[:, :6])
                        ref_dist = Normal(ref_vel.detach()[:, :6], ref_std.detach()[:, :6])
                    else:
                        cur_dist = Normal(cur_vel, cur_std)
                        ref_dist = Normal(ref_vel.detach(), ref_std.detach())
                    # Mean over action dims and batch
                    bc_ref_kl_loss = _bc_ref_coef * kl_divergence(cur_dist, ref_dist).sum(-1).mean()

                # ── optimization step ───────────────────────────────────
                total_loss = (
                    policy_loss
                    + gaussian_entropy_loss
                    + brownian_reg_loss
                    + value_loss
                    + kl_penalty_loss
                    + bc_ref_kl_loss
                )

                # ── BC Regularization Loss (Two-Phase Continuous Guidance) ────────────
                # Enabled by setting agent._bc_loss_fn externally (see train_xhand.py).
                # Hook signature: fn(obs, delta_actions, action_dist_normal, actions_prior) -> scalar tensor or 0
                _bc_loss = 0
                if self._bc_loss_fn is not None:
                    try:
                        _bc_loss = self._bc_loss_fn(
                            sampled_actor_observations,
                            sampled_delta_actions,
                            action_distribution_new,
                            sampled_actions_prior,
                            sampled_tro_inner_q,
                            sampled_spawn_dist,
                        )
                        # _bc_loss /= 1000.0  # Scale down to keep in balance with RL losses (tune as needed)
                        total_loss = total_loss + _bc_loss
                    except Exception as _bc_err:
                        import traceback as _tb
                        print(f"\n[BC Loss ERROR] _bc_loss_fn 예외 발생 — 이번 미니배치 BC loss 스킵")
                        print(f"  obs={sampled_actor_observations.shape}, acts={sampled_delta_actions.shape}")
                        _tb.print_exc()

                # === [OPTION: Auxiliary Chamfer Loss — gradient-based pose estimation] ===
                # Uncomment below to enable direct gradient training of TactilePoseEstimator.
                # Gradients flow: total_loss → chamfer_loss → pred_transformed_pc
                #   → TactilePoseEstimator (pred_translation, pred_quat) → TactileLinearEncoder
                # Requires auxiliary_loss_scale > 0 in agent config (e.g. 0.01~0.1).
                # Replace the total_loss above with the block below:
                #
                # auxiliary_loss = 0
                # if self._auxiliary_loss_scale > 0:
                #     nn_condition = actor.model["condition"]
                #     if hasattr(nn_condition, '_chamfer_loss') and nn_condition._chamfer_loss is not None:
                #         auxiliary_loss = self._auxiliary_loss_scale * nn_condition._chamfer_loss
                # total_loss = (
                #     policy_loss
                #     + gaussian_entropy_loss
                #     + brownian_reg_loss
                #     + value_loss
                #     + kl_penalty_loss
                #     + auxiliary_loss
                # )
                self.optimizer.zero_grad()
                total_loss.backward()
                if self._grad_norm_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        itertools.chain(
                            actor.model.parameters(),
                            self.model_dict["critic"].parameters(),
                        ),
                        self._grad_norm_clip,
                    )
                self.optimizer.step()

                # EWMA proximal update — after each gradient step
                if use_ewma:
                    actor.update_proximal()

                if actor.using_ema:
                    actor.ema_update()

                # ── accumulate diagnostics ──────────────────────────────
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._gaussian_entropy_loss_scale:
                    cumulative_gaussian_entropy_loss += gaussian_entropy_loss.item()
                if compute_breg:
                    cumulative_brownian_reg_loss += brownian_reg_loss.item()
                if self._kl_penalty > 0 and use_ewma:
                    cumulative_kl_penalty_loss += (
                        kl_penalty_loss.item()
                        if isinstance(kl_penalty_loss, torch.Tensor)
                        else kl_penalty_loss
                    )
                # [OPTION: Auxiliary loss diagnostics] Uncomment when auxiliary loss is enabled:
                # if self._auxiliary_loss_scale > 0 and isinstance(auxiliary_loss, torch.Tensor):
                #     cumulative_auxiliary_loss += auxiliary_loss.item()
                if self._bc_loss_fn is not None and isinstance(_bc_loss, torch.Tensor):
                    cumulative_bc_loss += _bc_loss.item()

                if isinstance(bc_ref_kl_loss, torch.Tensor):
                    cumulative_bc_ref_kl_loss += bc_ref_kl_loss.item()

                if torch.max(delta_vel) > delta_vel_max:
                    delta_vel_max = torch.max(delta_vel)
                if torch.min(delta_vel) < delta_vel_min:
                    delta_vel_min = torch.min(delta_vel)

            # update learning rate (per epoch)
            kl = torch.tensor(kl_divergences, device=self.device).mean()
            if self.lr_schedule is not None:
                self.lr_schedule.step(kl.item())

            # Actor LR warmup override — applied AFTER KLAdaptiveLR.step() so that
            # KLAdaptiveLR cannot override the warmup schedule during warmup.
            # Linear ramp: lr = start + (target - start) * (step / warmup_steps)
            if self._actor_lr_warmup_steps > 0 and self._actor_lr_update_step < self._actor_lr_warmup_steps:
                _warmup_frac  = self._actor_lr_update_step / self._actor_lr_warmup_steps
                _warmup_lr    = self._actor_lr_warmup_start + (self._learning_rate - self._actor_lr_warmup_start) * _warmup_frac
                self.optimizer.param_groups[0]["lr"] = _warmup_lr

        # Update model_last (behavior policy for next iteration)
        actor.update()

        # Advance step counters (one increment per update() call)
        self._actor_lr_update_step += 1
        if self._bc_ref_model is not None:
            self._bc_ref_update_step += 1

        n_updates = self._learning_epochs * self._mini_batches
        output = {
            "Loss/policy_loss": cumulative_policy_loss / n_updates,
            "Loss/gaussian_entropy_loss": cumulative_gaussian_entropy_loss / n_updates,
            "Loss/value_loss": cumulative_value_loss / n_updates,
            "Policy/mean_noise_std": delta_std_new.mean().item(),
            "Policy/delta_vel_max": delta_vel_max.item(),
            "Policy/delta_vel_min": delta_vel_min.item(),
            "Loss/actor_lr": self.optimizer.param_groups[0]["lr"],
            "Loss/critic_lr": self.optimizer.param_groups[1]["lr"],
            "Loss/kl": kl.item(),
        }

        if self._brownian_reg_loss_scale > 0:
            output["Loss/brownian_reg_loss"] = cumulative_brownian_reg_loss / n_updates
            if "anneal_coef" in self.model_dict:
                self.model_dict["anneal_coef"].step_update()
                output["Loss/brownian_reg_anneal_coef"] = (
                    self.model_dict["anneal_coef"].forward().item()
                )

        if use_ewma:
            output["Loss/kl_penalty_loss"] = cumulative_kl_penalty_loss / n_updates
            if clip_fractions:
                output["Policy/clip_fraction"] = sum(clip_fractions) / len(
                    clip_fractions
                )

        # [OPTION: Auxiliary loss logging] Uncomment when auxiliary loss is enabled:
        # if self._auxiliary_loss_scale > 0:
        #     output["Loss/auxiliary_chamfer_loss"] = cumulative_auxiliary_loss / n_updates

        if self._bc_loss_fn is not None:
            output["Loss/bc_loss"] = cumulative_bc_loss / n_updates

        if self._bc_ref_model is not None:
            output["Loss/bc_ref_kl_loss"] = cumulative_bc_ref_kl_loss / n_updates
            output["Loss/bc_ref_kl_coef"] = _bc_ref_coef

        if self._actor_lr_warmup_steps > 0:
            output["Policy/actor_lr_warmup_step"] = min(
                self._actor_lr_update_step, self._actor_lr_warmup_steps
            )

        self.eval_mode()
        self.replay_buffer.reset()

        return output

    def _compute_kl_divergence(
        self,
        actions_mean: torch.Tensor,
        actions_std: torch.Tensor,
        last_action_mean: torch.Tensor,
        last_action_std: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            std_ratio = actions_std / last_action_std + 1.0e-5
            std_drifted = torch.square(last_action_std) + torch.square(
                last_action_mean - actions_mean
            )
            kl = torch.sum(
                torch.log(std_ratio)
                + std_drifted / (2.0 * torch.square(actions_std))
                - 0.5,
                axis=-1,
            )  # type: ignore
            kl_mean = torch.mean(kl).detach()
        return kl_mean

    def eval_mode(self):
        self.set_mode("eval")

    def train_mode(self):
        self.set_mode("train")

    def to(self, device: str) -> PolicyFlowBase:
        for model in self.model_dict.values():
            if model is not None:
                model.to(device)
        return self

    def export_onnx(self) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
        pass

    def get_inference_policy(self, device: str = None) -> Callable:
        self.eval_mode()
        if device is not None:
            self.to(device)

        def actor_policy(obs_dict):
            x0 = torch.randn(
                (
                    obs_dict["actor_observations"].shape[0],
                    self._action_size,
                ),
                device=self.device,
            )
            if self._degenerate2gaussian:
                x0 = torch.zeros_like(x0)
            mean, std = self.model_dict["actor"].sample(
                x0=x0,
                condition=obs_dict["actor_observations"],
                n_samples=obs_dict["actor_observations"].shape[0],
            )
            action_distribution = torch.distributions.Normal(mean, std)
            actions = action_distribution.sample().detach()
            if self._action_clip > 0:
                actions = torch.tanh(actions / self._action_clip) * self._action_clip
            return actions

        return actor_policy
