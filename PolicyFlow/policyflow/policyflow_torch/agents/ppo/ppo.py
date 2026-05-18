from __future__ import annotations
import itertools
import torch
from typing import Any, Mapping, Optional, Union, Dict, Tuple, Callable
from policyflow_torch.modules import Network
from policyflow_torch.agents import ActorCriticBase
from policyflow_torch.storage import ReplayBuffer
from policyflow_torch.utils.kl_adaptive import KLAdaptiveLR


class PPO(ActorCriticBase):
    def __init__(
        self,
        models: Mapping[str, Network],
        replay_buffer: ReplayBuffer,
        cfg: dict = dict(),
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:

        super().__init__(models, replay_buffer, cfg, device)

        self._desired_kl = cfg.get("desired_kl", 0.01)
        self._learning_rate = cfg.get("learning_rate", 1e-3)
        self._discount_factor = cfg.get("discount_factor", 0.99)
        self._lambda = cfg.get("lam", 0.95)
        self._time_limit_bootstrap = cfg.get("time_limit_bootstrap", True)
        self._mini_batches = cfg.get("mini_batches", 1)
        self._learning_epochs = cfg.get("learning_epochs", 1)
        self._entropy_loss_scale = cfg.get("entropy_loss_scale", 0.0)
        self._ratio_clip = cfg.get("ratio_clip", 0.2)
        self._value_loss_scale = cfg.get("value_loss_scale", 1.0)
        self._clip_predicted_values = cfg.get("clip_predicted_values", True)
        self._value_clip = cfg.get("value_clip", 0.1)
        self._grad_norm_clip = cfg.get("grad_norm_clip", 1.0)

        self.optimizer = torch.optim.Adam(
            itertools.chain(
                self.model_dict["actor"].parameters(),
                self.model_dict["critic"].parameters(),
            ),
            lr=self._learning_rate,
        )
        self.po_lr_schedule = KLAdaptiveLR(
            self.optimizer,
            **cfg.get(
                "learning_rate_scheduler_kwargs",
                {
                    "kl_threshold": self._desired_kl,
                },
            ),
        )
        self._register_serializable("optimizer")

    def init_replay_buffer(
        self,
        critic_observation_size: Union[int, Tuple[int]],
        actor_observation_size: Union[int, Tuple[int]],
        action_size: Union[int, Tuple[int]],
    ) -> None:
        """Initialize the agent"""
        self.eval_mode()
        self._action_size = action_size
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
                name="actions", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="actions_std", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="actions_mean", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="rewards", size=1, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="terminated", size=1, dtype=torch.bool
            )
            self.replay_buffer.create_tensor(
                name="actions_log_prob", size=1, dtype=torch.float32
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
        
        mean, std = self.model_dict["actor"].forward(
            observations_dict["actor_observations"],
            compute_std=True,
        )
        action_distribution = torch.distributions.Normal(mean, std)
        actions = action_distribution.sample().detach()
        actions_logp = action_distribution.log_prob(actions).sum(-1)

        info = {
            "actions_log_prob": actions_logp.detach(),
            "actions_mean": mean.detach(),
            "actions_std": std.detach(),
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
            values = self.model_dict["critic"](
                observations_dict["critic_observations"].flatten(start_dim=1)
            )

            # time-limit (truncation) boostrapping
            truncated = environement_info.get("time_outs", torch.zeros_like(dones))
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated
                
            # storage transition in replay_buffer
            self.replay_buffer.add_samples(
                critic_observations=observations_dict["critic_observations"],
                next_critic_observations=next_observations_dict["critic_observations"],
                actor_observations=observations_dict["actor_observations"],
                actions_log_prob=actions_info["actions_log_prob"],
                actions_mean=actions_info["actions_mean"],
                actions_std=actions_info["actions_std"],
                actions=actions,
                rewards=rewards,
                terminated=dones,
                values=values,
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
        info = self.update_actor_critic()
        self.eval_mode()
        self.replay_buffer.reset()
        return info

    def update_actor_critic(self) -> Dict[str, Union[float, torch.Tensor]]:

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for _ in range(self._learning_epochs):
            kl_divergences = []

            # sample mini-batches from replay_buffer
            sampled_batches = self.replay_buffer.sample_all(
                names=[
                    "critic_observations",
                    "actor_observations",
                    "actions",
                    "actions_mean",
                    "actions_std",
                    "actions_log_prob",
                    "values",
                    "returns",
                    "advantages",
                ],
                mini_batches=self._mini_batches,
            )

            # mini-batches loop
            for (
                sampled_critic_observations,
                sampled_actor_observations,
                sampled_actions,
                sampled_actions_mean,
                sampled_actions_std,
                sampled_actions_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:
                mean, std = self.model_dict["actor"].forward(
                    sampled_actor_observations,
                    compute_std=True,
                )
                action_distribution = torch.distributions.Normal(mean, std)
                actions_log_prob_new = action_distribution.log_prob(
                    sampled_actions
                ).sum(-1)

                # compute approximate KL divergence
                with torch.no_grad():
                    kl_divergences.append(self._compute_kl_divergence(
                        mean, std, sampled_actions_mean, sampled_actions_std
                    ))

                # compute entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = (
                        -self._entropy_loss_scale
                        * action_distribution.entropy().sum(dim=-1).mean()
                    )
                else:
                    entropy_loss = 0

                # compute policy loss
                ratio = torch.exp(actions_log_prob_new - sampled_actions_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(
                    ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                )
                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # update value network
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

                # optimization step
                self.optimizer.zero_grad()
                (policy_loss + entropy_loss + value_loss).backward()

                if self._grad_norm_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        itertools.chain(
                            self.model_dict["critic"].parameters(),
                            self.model_dict["actor"].parameters(),
                        ),
                        self._grad_norm_clip,
                    )
                self.optimizer.step()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            kl = torch.tensor(kl_divergences, device=self.device).mean()
            self.po_lr_schedule.step(kl.item())

        output = {
            "Loss/policy_loss": cumulative_policy_loss
            / (self._learning_epochs * self._mini_batches),
            "Loss/entropy_loss": cumulative_entropy_loss
            / (self._learning_epochs * self._mini_batches),
            "Loss/value_loss": cumulative_value_loss
            / (self._learning_epochs * self._mini_batches),
            "Policy/policy_std": std.mean().item(),
            "Loss/learning_rate": self.po_lr_schedule.get_last_lr()[0],
        }

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

    def to(self, device: str) -> ActorCriticBase:
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
            mean, std = self.model_dict["actor"].forward(
                obs_dict["actor_observations"],
                compute_std=True,
            )
            action_distribution = torch.distributions.Normal(mean, std)
            actions = action_distribution.sample().detach()
            return actions

        return actor_policy
