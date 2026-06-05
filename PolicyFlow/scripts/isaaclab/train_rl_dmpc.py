"""train_rl_dmpc.py — PPO training for multi-drone navigation with DMPC-guided dense reward.

DMPC runs in parallel every control step to provide a velocity-alignment reward
(where should the drone be heading?).  The policy is a flow-matching actor
trained with PolicyFlow PPO.  At inference time only the RL policy runs — no
DMPC needed.

Architecture (per-agent shared policy):
  Actor condition: PerDroneConditionNet — shared NeighborEncoder per drone → (B*N, emb_dim)
  Actor flow:      FlowMlp(x_dim=3, emb_dim=emb_dim) — shared across all drones
  Wrapper:         PerAgentActorWrapper reshapes (B, N*3) ↔ (B*N, 3) for env↔CNF interface
  Critic:          flat MLP(obs_dim → scalar)

Usage::

    python train_rl_dmpc.py \\
        --num_envs 64 --num_drones 4 \\
        --max_iterations 5000 \\
        --experiment_name dmpc_rl_v1 [--headless]
"""

from __future__ import annotations

import argparse
import os
import sys
import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_POLICYFLOW_ROOT = _HERE.parent.parent / "policyflow"
_PROJECT_ROOT = _HERE.parent.parent.parent
for _p in (_POLICYFLOW_ROOT, _PROJECT_ROOT):
    _s = str(_p)
    if _p.exists() and _s not in sys.path:
        sys.path.insert(0, _s)

# ── CLI (must be parsed before AppLauncher) ───────────────────────────────────
parser = argparse.ArgumentParser(description="PPO training for multi-drone DMPC-RL.")
parser.add_argument("--num_envs",        type=int,   default=768)
parser.add_argument("--num_drones",      type=int,   default=10)
parser.add_argument("--max_iterations",  type=int,   default=500000)
parser.add_argument("--rollouts",        type=int,   default=100,
                    help="Env steps per PPO rollout per env.")
parser.add_argument("--save_interval",   type=int,   default=10)
parser.add_argument("--log_dir",         type=str,   default="runs/rl_dmpc")
parser.add_argument("--experiment_name", type=str,   default=None)
parser.add_argument("--resume",          type=str,   default=None)
parser.add_argument("--seed",            type=int,   default=42)
# Model
parser.add_argument("--emb_dim",    type=int,   default=128)
parser.add_argument("--hidden_dims", type=int,  nargs="*", default=[256, 256, 256])
parser.add_argument("--sample_steps", type=int, default=10)
# BC warmup phase
parser.add_argument("--bc_iters",            type=int,   default=0,
                    help="BC warmup iterations before RL (0 = skip).")
parser.add_argument("--bc_epochs_per_round", type=int,   default=10,
                    help="Gradient steps per rollout round during BC.")
parser.add_argument("--bc_batch_size",       type=int,   default=2048)
parser.add_argument("--bc_lr",              type=float, default=3e-4)
parser.add_argument("--bc_critic_value_loss_scale", type=float, default=1e-4)
parser.add_argument("--bc_awr_temperature", type=float, default=2.0)
parser.add_argument("--bc_awr_max_weight", type=float, default=20.0)
parser.add_argument("--bc_ref_kl_coef0",    type=float, default=1.0,
                    help="Initial KL coef toward frozen BC policy in RL phase.")
parser.add_argument("--bc_ref_kl_decay_steps", type=float, default=500.0,
                    help="PPO updates to decay KL coef to ~1/100 of initial.")
parser.add_argument("--bc_ref_kl_min", type=float, default=0.0,
                    help="Floor value for BC reference KL coefficient.")
parser.add_argument("--bc_critic_warmup_iters", type=int, default=10,
                    help="After BC warmup, freeze actor for this many PPO updates "
                         "so the critic learns before actor updates resume.")
# N curriculum
parser.add_argument("--n_min", type=int, default=0,
                    help="Min active drones for N curriculum (0 = disabled, use --num_drones throughout).") #1
# Phase-2 resume curriculum (critic reset and actor freeze only).  The harder
# env settings below are now enabled from phase 1 as well.
parser.add_argument("--curriculum", action="store_true", default=False,
                    help="On resume: reset critic and freeze actor for "
                         "--actor_freeze_iters PPO updates.")
parser.add_argument("--hover_vel_scale", type=float, default=5.0,
                    help="Hover velocity penalty scale used with --curriculum (per drone, *step_dt).")
parser.add_argument("--hover_proximity_dist", type=float, default=0.4,
                    help="Tanh half-width [m] for hover velocity penalty gate (with --curriculum).")
parser.add_argument("--dist_far_scale", type=float, default=45.0,
                    help="Far-field distance-to-goal reward scale.")
parser.add_argument("--dist_far_tanh_scale", type=float, default=1.2,
                    help="Far-field tanh denominator [m] for distance-to-goal reward.")
parser.add_argument("--dist_near_scale", type=float, default=30.0,
                    help="Near-goal distance-to-goal reward scale.")
parser.add_argument("--dist_tanh_scale", type=float, default=0.3,
                    help="Near-goal tanh denominator [m] for distance-to-goal reward. "
                         "Smaller = sharper gradient near goal.")
parser.add_argument("--actor_freeze_iters", type=int, default=10,
                    help="PPO updates to freeze the actor after critic reset (--curriculum). Default 10.")
parser.add_argument("--collision_terminate_dist", type=float, default=0.12,
                    help="Hard collision termination distance [m]. Separate from cfg.rmin, "
                         "which remains the DMPC/safety-margin penalty radius.")
# Resume with N change
parser.add_argument("--critic_warmup_iters", type=int, default=10,
                    help="When resuming with a different num_drones, freeze the actor for this many "
                         "PPO updates so the new critic can warm up before joint training.")
# Play / eval mode
parser.add_argument("--play", action="store_true", default=False,
                    help="Eval mode: load --resume checkpoint, run policy without training, "
                         "print success stats. Combine with --n_min to test with inactive drones.")
parser.add_argument("--eval_episodes", type=int, default=50,
                    help="Number of complete episodes to evaluate in --play mode.")
# Wandb
parser.add_argument("--wandb",           action="store_true", default=False)
parser.add_argument("--wandb_project",   type=str,   default="dmpc_rl")

from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = getattr(args_cli, "headless", True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ───────────────────────────────────────────────────────
import copy                     # noqa: E402
import gymnasium as gym         # noqa: E402
import torch                    # noqa: E402
import torch.nn as nn           # noqa: E402
import torch.nn.functional as F # noqa: E402

import quadcopter                # noqa: F401,E402  — registers all envs
from quadcopter.multi_drone_dmpc_rl_env import (  # noqa: E402
    MultiDroneDmpcRLEnv,
    MultiDroneDmpcRLEnvCfg,
)
from quadcopter.multi_drone_dmpc_env import (  # noqa: E402
    PER_DRONE_OWN_DIM,
    PER_DRONE_SDF_DIM,
    PER_NEIGHBOUR_DIM,
)

from policyflow_torch.env import IsaacLabEnvWrapper              # noqa: E402
from policyflow_torch.storage import ReplayBuffer                # noqa: E402
from policyflow_torch.modules import (                           # noqa: E402
    ContinuousNormalizingFlow,
    FlowMlp,
    NeighborEncoder,
)
from policyflow_torch.modules.normalizer import EmpiricalNormalization  # noqa: E402
from policyflow_torch.agents import PolicyFlow  # noqa: E402
from policyflow_torch.runners import IsaaclabRunner              # noqa: E402

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Per-drone condition net (shared weights, operates on single-drone obs)   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class PerDroneConditionNet(nn.Module):
    """Shared per-drone condition network.

    Input:  ``(B*N, own_dim + neigh_dim*(N-1))`` — per-drone observation slice.
    Output: ``(B*N, emb_dim)`` — per-drone embedding.

    The same weights process every drone in every environment.
    """

    def __init__(
        self,
        N: int,
        own_dim: int,
        neigh_dim: int,
        emb_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        self.N        = N
        self.own_dim  = own_dim
        self.neigh_dim = neigh_dim
        self.per_dim  = own_dim + neigh_dim * (N - 1)
        self.emb_dim  = emb_dim

        self.own_norm   = EmpiricalNormalization(shape=own_dim,   until=int(1e8))
        self.neigh_norm = EmpiricalNormalization(shape=neigh_dim, until=int(1e8))
        self.neighbor_enc = NeighborEncoder(
            own_dim=own_dim, neighbor_dim=neigh_dim,
            emb_dim=emb_dim, num_heads=4,
        )
        # Project [own_norm | neighbor_enc_output] → emb_dim
        in_d = own_dim + emb_dim
        layers: list[nn.Module] = []
        prev = in_d
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, emb_dim))
        self.proj = nn.Sequential(*layers)

    def forward(self, obs_per_drone: torch.Tensor) -> torch.Tensor:
        """obs_per_drone: (BN, per_dim) → (BN, emb_dim)."""
        BN = obs_per_drone.shape[0]
        own   = obs_per_drone[:, :self.own_dim]
        neigh = obs_per_drone[:, self.own_dim:]
        own_n = self.own_norm(own)

        N_neigh = self.N - 1
        if N_neigh > 0:
            neigh_n = self.neigh_norm(
                neigh.reshape(BN * N_neigh, self.neigh_dim)
            ).reshape(BN, N_neigh, self.neigh_dim)
        else:
            neigh_n = torch.zeros(BN, 0, self.neigh_dim, device=obs_per_drone.device)

        enc     = self.neighbor_enc(own_n, neigh_n)       # (BN, emb_dim)
        per_emb = torch.cat([own_n, enc], dim=-1)         # (BN, own+emb)
        return self.proj(per_emb)                         # (BN, emb_dim)

    def update_norm(self, obs_flat: torch.Tensor) -> None:
        """obs_flat: (B, N*per_dim) — warm-up norm from flat env obs."""
        B   = obs_flat.shape[0]
        per = obs_flat.reshape(B * self.N, self.per_dim)
        self.own_norm.update(per[:, :self.own_dim].detach())
        if self.N > 1:
            self.neigh_norm.update(per[:, self.own_dim:].reshape(-1, self.neigh_dim).detach())


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Per-agent actor wrapper (shared policy N-drone reshape)                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class PerAgentActorWrapper:
    """Wraps ContinuousNormalizingFlow(x_dims=3) for N-drone shared-policy execution.

    External interface (matches PolicyFlow agent expectations):
      sample / compute_flow_variation receive (B, N*per_dim) obs and (B, N*3) actions.

    Internal interface (passed to CNF):
      (B*N, per_dim) obs and (B*N, 3) actions — each drone processed identically.

    All other attributes/methods are forwarded to the wrapped CNF.
    """

    def __init__(self, cnf: ContinuousNormalizingFlow, N: int, per_drone_obs_dim: int):
        self._cnf             = cnf
        self.N                = N
        self.per_drone_obs_dim = per_drone_obs_dim

    # ── internal helpers ─────────────────────────────────────────────────────
    def _split_obs(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """(B, N*P) → (B*N, P)."""
        B = obs_flat.shape[0]
        return obs_flat.reshape(B * self.N, self.per_drone_obs_dim)

    # ── actor API (used by PolicyFlow agent) ─────────────────────────────────
    def sample(
        self,
        x0: torch.Tensor,
        condition: torch.Tensor,
        n_samples: int = 0,  # unused; internally computed as B*N
    ):
        """(B, N*3) x0 + (B, N*P) condition → (B, N*3) actions, (B, N*3) std."""
        B = condition.shape[0]
        actions_per, std_per = self._cnf.sample(
            x0        = x0.reshape(B * self.N, 3),
            condition = self._split_obs(condition),
            n_samples = B * self.N,
        )
        return actions_per.reshape(B, self.N * 3), std_per.reshape(B, self.N * 3)

    def compute_flow_variation(
        self,
        x1: torch.Tensor,
        condition: torch.Tensor,
        x0=None,
        **kwargs,
    ):
        """(batch, N*3) + (batch, N*P) → per-agent forward; outputs reshaped to (batch, N*3).

        Inactive drone slots (curriculum masking) have all-zero obs because the
        rotation matrix — always non-zero for a real drone — is zeroed along with
        every other feature.  Those slots are excluded from the CNF forward pass so
        they do not contribute to the actor loss or the proximal KL term.
        Per-drone output tensors (velocities) are scattered back to the full (BN, 3)
        shape with zeros at inactive positions before the final reshape.
        """
        batch = condition.shape[0]
        BN    = batch * self.N
        obs_per = self._split_obs(condition)                    # (BN, P)
        x1_per  = x1.reshape(BN, 3)                            # (BN, 3)
        x0_per  = x0.reshape(BN, 3) if x0 is not None else None

        # Detect inactive drones: all features (incl. rotation matrix) are zeroed.
        active      = obs_per.norm(dim=-1) > 1e-6              # (BN,) bool
        has_inactive = not active.all().item()

        if has_inactive:
            obs_in = obs_per[active]
            x1_in  = x1_per[active]
            x0_in  = x0_per[active] if x0_per is not None else None
        else:
            obs_in, x1_in, x0_in = obs_per, x1_per, x0_per

        result = self._cnf.compute_flow_variation(
            x1=x1_in, condition=obs_in, x0=x0_in, **kwargs
        )

        M = int(active.sum().item()) if has_inactive else BN

        # Precompute scatter index once for all (M, 3) output tensors.
        idx_3 = active.nonzero(as_tuple=True)[0].unsqueeze(1).expand(-1, 3) if has_inactive else None

        def _out(t: torch.Tensor, inactive_fill: float = 0.0) -> torch.Tensor:
            """Scatter active-only (M, 3) → (BN, 3) → (B, N*3) differentiably.

            Uses torch.Tensor.scatter (non-in-place) so the autograd graph is
            preserved from the CNF output to the policy loss.  In-place masked
            assignment (full[active] = t) would sever the gradient path because
            `full` is created without requires_grad.

            inactive_fill: value written at inactive drone positions.  Must be
            a valid float for std-like tensors to prevent NaN in Normal.log_prob.
            """
            if t.dim() >= 2 and t.shape[0] == M and has_inactive:
                base = t.new_full((BN, *t.shape[1:]), inactive_fill)
                t = base.scatter(0, idx_3, t)   # differentiable scatter
            if t.shape == (BN, 3):
                return t.reshape(batch, self.N * 3)
            return t

        if not isinstance(result, tuple):
            return _out(result)

        out = []
        for i, item in enumerate(result):
            if isinstance(item, torch.Tensor):
                if i == 1 and item.dim() >= 2 and item.shape[0] == M and has_inactive:
                    # result[1] is always std.  Inactive positions must receive a
                    # valid positive value so Normal(delta_vel, std).log_prob()
                    # does not produce NaN.  Use the mean of active stds (detached
                    # constant — no unintended gradient on variance parameter).
                    with torch.no_grad():
                        fill_std = float(item.mean().clamp(min=1e-3))
                    out.append(_out(item, inactive_fill=fill_std))
                else:
                    out.append(_out(item))
            elif isinstance(item, tuple):
                # proximal_info = (prox_delta_vel (M, 3) detached, prox_std scalar)
                vel = _out(item[0])
                std = item[1].repeat(self.N) if item[1].dim() == 1 else item[1]
                out.append((vel, std))
            else:
                out.append(item)   # None or scalar
        return tuple(out)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._cnf, name)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Critic network                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class DroneCriticNet(nn.Module):
    """Simple flat MLP critic.

    Input: ``(B, obs_dim)`` → scalar value ``(B,)``.
    The last-dim squeeze ensures compatibility with the replay buffer
    which squeezes size-1 tensors on retrieval.
    """

    def __init__(self, obs_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.obs_norm = EmpiricalNormalization(shape=obs_dim, until=int(1e8))
        layers: list[nn.Module] = []
        in_d = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.ELU()]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, hidden_state=None) -> torch.Tensor:
        return self.mlp(self.obs_norm(obs)).squeeze(-1)   # (B,)

    def update_norm(self, obs: torch.Tensor) -> None:
        self.obs_norm.update(obs.detach())


def _extract_actor_obs(obs) -> torch.Tensor:
    if isinstance(obs, dict):
        return obs["actor_observations"] if "actor_observations" in obs else obs["policy"]
    return obs


def _extract_critic_obs(obs) -> torch.Tensor:
    if isinstance(obs, dict):
        if "critic_observations" in obs:
            return obs["critic_observations"]
        if "critic" in obs:
            return obs["critic"]
        return obs["policy"]
    return obs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ PPO agent config (plain dict — avoids @configclass MISSING inheritance)  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
AGENT_CFG: dict = {
    "desired_kl":                  0.01,
    "use_kl_lr_schedule":          False,
    "use_kl_lr_schedule_critic_only": False,
    "discount_factor":             0.995,
    "lam":                         0.95,
    "time_limit_bootstrap":        True,

    "ratio_clip":                  0.2,
    "clip_predicted_values":       True,
    "value_clip":                  0.2,
    "value_loss_scale":            0.5,
    "grad_norm_clip":              1.0,
    "action_clip":                 1.0,

    "mini_batches":                4,
    "learning_epochs":             1,
    "learning_rate":               3e-4,
    "critic_learning_rate":        1e-3,

    "use_ewma":                    True,
    "beta_prox":                   0.95,
    "kl_penalty":                  0.0,
    "imp_samp_max":                0.0,

    "gaussian_entropy_loss_scale": 1e-5,
    "brownian_reg_loss_scale":     1e-4,
    "degenerate2gaussian":         False,
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ BC warmup phase                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def run_bc_phase(
    env,            # IsaacLabEnvWrapper
    cnf,            # ContinuousNormalizingFlow (actor's internal CNF)
    critic,         # DroneCriticNet
    N: int,
    per_drone_dim: int,
    actor_obs_dim: int,
    critic_obs_dim: int,
    device: torch.device,
    run_name: str,
) -> None:
    """Phase 1: collect DMPC rollouts, train actor via BC + critic via value regression.

    Saves a runner-compatible checkpoint at log_dir/run_name/bc_checkpoint.pt.
    """
    from quadcopter.dmpc_gpu_expert import GPUDMPCExpert
    from quadcopter.dmpc_expert import DMPCParams

    env_uw   = env.unwrapped
    num_envs = env.num_envs
    gamma    = AGENT_CFG["discount_factor"]

    env_cfg_uw = env_uw.cfg
    ts_bc = env_cfg_uw.sim.dt * env_cfg_uw.decimation
    bc_expert = GPUDMPCExpert(
        num_drones=N, num_envs=num_envs, device=device,
        params=DMPCParams(
            pmin=env_cfg_uw.pos_min,
            pmax=env_cfg_uw.pos_max,
            rmin=env_cfg_uw.rmin,
            ts=ts_bc,
            max_envs=num_envs,
        ),
    )
    obstacle_info = env_uw.get_obstacle_info() if hasattr(env_uw, "get_obstacle_info") else None
    bc_expert.reset(obstacle_info=obstacle_info)

    # Ring buffer (per-drone)
    BUF = 200_000
    buf_obs  = torch.zeros(BUF, per_drone_dim, device=device)
    buf_act  = torch.zeros(BUF, 3, device=device)
    buf_critic_obs = torch.zeros(BUF, critic_obs_dim, device=device)
    buf_ret  = torch.zeros(BUF, device=device)
    buf_ptr, buf_full = 0, False

    actor_opt  = torch.optim.Adam(list(cnf.model.parameters()), lr=args_cli.bc_lr)
    critic_opt = torch.optim.Adam(list(critic.parameters()),    lr=AGENT_CFG["critic_learning_rate"])

    obs, _ = env.reset()

    for it in range(args_cli.bc_iters):
        # ── Collect rollout with DMPC ────────────────────────────────────────
        actor_obs_list, critic_obs_list, act_list, rew_list, done_list = [], [], [], [], []

        for _ in range(args_cli.rollouts):
            st      = env_uw._stack_drone_state()
            ref_pos, ref_vel, ref_acc = bc_expert.compute(
                pos_w=st["pos_w"], vel_w=st["lin_vel_w"],
                goal_w=env_uw._goal_pos_w,
                env_origins=env_uw._terrain.env_origins,
            )
            env_act = env_uw.reference_to_action(ref_pos, ref_vel, ref_acc)  # (E, N*3)

            actor_obs_list.append(_extract_actor_obs(obs).clone())
            critic_obs_list.append(_extract_critic_obs(obs).clone())
            act_list.append(env_act.clone())

            obs, rew, done, _ = env.step(env_act)
            done_list.append(done.bool().clone())
            rew_list.append(rew.clone())

        # ── Compute discounted returns ───────────────────────────────────────
        with torch.no_grad():
            R = critic(_extract_critic_obs(obs))  # bootstrap last state
        returns = []
        for rew, done in zip(reversed(rew_list), reversed(done_list)):
            R = rew + gamma * R * (~done).float()
            returns.insert(0, R.clone())

        # ── Update obs normalisers ───────────────────────────────────────────
        all_actor_obs = torch.stack(actor_obs_list)    # (T, E, actor_obs_dim)
        all_critic_obs = torch.stack(critic_obs_list)  # (T, E, critic_obs_dim)
        flat_actor_for_norm = all_actor_obs.reshape(-1, actor_obs_dim)
        flat_critic_for_norm = all_critic_obs.reshape(-1, critic_obs_dim)
        cnf.model["condition"].update_norm(flat_actor_for_norm)
        critic.update_norm(flat_critic_for_norm)

        # ── Add to BC buffer ─────────────────────────────────────────────────
        for obs_s, critic_s, act_s, ret_s in zip(actor_obs_list, critic_obs_list, act_list, returns):
            obs_per = obs_s.reshape(-1, per_drone_dim)   # (E*N, per_dim)
            act_per = act_s.reshape(-1, 3)               # (E*N, 3)
            critic_per = critic_s.repeat_interleave(N, dim=0)
            ret_per = ret_s.repeat_interleave(N)
            n = obs_per.shape[0]
            end = buf_ptr + n
            if end <= BUF:
                buf_obs[buf_ptr:end] = obs_per
                buf_act[buf_ptr:end] = act_per
                buf_critic_obs[buf_ptr:end] = critic_per
                buf_ret[buf_ptr:end] = ret_per
                buf_ptr = end % BUF
                if buf_ptr == 0:
                    buf_full = True
            else:
                first = BUF - buf_ptr
                buf_obs[buf_ptr:]   = obs_per[:first]
                buf_act[buf_ptr:]   = act_per[:first]
                buf_critic_obs[buf_ptr:] = critic_per[:first]
                buf_ret[buf_ptr:] = ret_per[:first]
                buf_obs[:n - first] = obs_per[first:]
                buf_act[:n - first] = act_per[first:]
                buf_critic_obs[:n - first] = critic_per[first:]
                buf_ret[:n - first] = ret_per[first:]
                buf_ptr  = n - first
                buf_full = True

        buf_valid = BUF if buf_full else buf_ptr

        # ── AWR-BC actor update (rectified-flow matching) ────────────────────
        bc_loss_val = float("nan")
        bc_weight_val = float("nan")
        bc_adv_val = float("nan")
        if buf_valid >= args_cli.bc_batch_size:
            cnf.model["condition"].train()
            cnf.model["flow"].train()
            weight_sum = 0.0
            adv_sum = 0.0
            bc_batches = 0
            for _ in range(args_cli.bc_epochs_per_round):
                idx  = torch.randperm(buf_valid, device=device)[:args_cli.bc_batch_size]
                obs_b = buf_obs[idx]   # (B, per_dim)
                x1    = buf_act[idx]   # (B, 3)
                critic_b = buf_critic_obs[idx]
                ret_b = buf_ret[idx]
                x0    = torch.randn_like(x1)
                t     = torch.rand(x1.shape[0], device=device)
                xt    = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
                cond  = cnf.model["condition"](obs_b)
                v_pred = cnf.model["flow"](xt, t, cond)
                with torch.no_grad():
                    value_b = critic(critic_b)
                    adv_b = ret_b - value_b
                    weights = torch.exp(adv_b / max(args_cli.bc_awr_temperature, 1e-6))
                    weights = weights.clamp(max=args_cli.bc_awr_max_weight)
                    weight_sum += float(weights.mean().item())
                    adv_sum += float(adv_b.mean().item())
                    bc_batches += 1
                vel_mse = (v_pred - (x1 - x0)).pow(2).mean(dim=-1)
                bc_loss = (weights * vel_mse).mean()
                actor_opt.zero_grad()
                bc_loss.backward()
                torch.nn.utils.clip_grad_norm_(cnf.model.parameters(), 1.0)
                actor_opt.step()
                cnf.update_proximal()   # keep EWMA proximal in sync
            bc_loss_val = bc_loss.item()
            bc_weight_val = weight_sum / max(bc_batches, 1)
            bc_adv_val = adv_sum / max(bc_batches, 1)

        # ── Critic value regression ──────────────────────────────────────────
        flat_obs = all_critic_obs.reshape(-1, critic_obs_dim)
        flat_ret = torch.stack(returns).reshape(-1)
        perm = torch.randperm(flat_obs.shape[0], device=device)
        critic_loss_val = float("nan")
        for _ in range(args_cli.bc_epochs_per_round):
            for start in range(0, flat_obs.shape[0], args_cli.bc_batch_size):
                idx = perm[start : start + args_cli.bc_batch_size]
                if len(idx) == 0:
                    break
                critic_loss = (
                    args_cli.bc_critic_value_loss_scale
                    * F.mse_loss(critic(flat_obs[idx]), flat_ret[idx].detach())
                )
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
            critic_loss_val = critic_loss.item()

        print(
            f"[BC {it+1:4d}/{args_cli.bc_iters}] "
            f"bc={bc_loss_val:.4f}  critic={critic_loss_val:.4f}  "
            f"w={bc_weight_val:.3f}  adv={bc_adv_val:.3f}  buf={buf_valid}",
            flush=True,
        )
        if args_cli.wandb and _WANDB_AVAILABLE and _wandb.run is not None:
            _wandb.log(
                {
                    "BC/iteration": it + 1,
                    "BC/actor_loss": bc_loss_val,
                    "BC/critic_loss": critic_loss_val,
                    "BC/awr_weight": bc_weight_val,
                    "BC/advantage": bc_adv_val,
                    "BC/buffer_size": buf_valid,
                    "BC/rollout_steps": args_cli.rollouts,
                    "BC/epochs_per_round": args_cli.bc_epochs_per_round,
                },
                step=it + 1,
            )

    # PPO starts from the post-BC policy.  Keep behavior/proximal snapshots in
    # sync so the first RL update is not measured against the random init actor.
    cnf.update()
    cnf.reset_proximal()

    # ── Save BC checkpoint (runner-compatible format) ────────────────────────
    ckpt_dir  = Path(args_cli.log_dir) / run_name
    ckpt_path = ckpt_dir / "bc_checkpoint.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dicts": {
                "actor": {
                    "_type": "flow",
                    "state": {
                        "model": cnf.model.state_dict(),
                        "model_ema": cnf.model_ema.state_dict(),
                        "model_last": cnf.model_last.state_dict(),
                        **(
                            {"model_proximal": cnf.model_proximal.state_dict()}
                            if cnf.model_proximal is not None
                            else {}
                        ),
                    },
                },
                "critic": {"_type": "nn_module", "state": critic.state_dict()},
            },
            "iteration": 0,
        },
        ckpt_path,
    )
    print(f"[BC] checkpoint saved → {ckpt_path}", flush=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Play / eval loop                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _run_play(
    env,            # IsaacLabEnvWrapper
    actor: PerAgentActorWrapper,
    N: int,
    args_cli,
    device,
) -> None:
    """Evaluation loop — runs policy without gradient updates.

    Tracks per-episode success (terminated = all active drones reached goal)
    and breaks results down by number of active drones when N-curriculum is on.
    """
    from collections import defaultdict

    env_uw = env.unwrapped
    E = env.num_envs
    curriculum_on = getattr(env_uw, "_curriculum_enabled", False)

    actor._cnf.model.eval()

    ep_results: list[tuple[int, bool]] = []   # (n_active, success)
    total_done = 0
    target = args_cli.eval_episodes

    obs, _ = env.reset()

    print(f"\n[Play] running {target} episodes  "
          f"num_envs={E}  N={N}  n_min={args_cli.n_min}  curriculum={curriculum_on}",
          flush=True)

    with torch.no_grad():
        while total_done < target:
            x0 = torch.randn(E, N * 3, device=device)
            actions, _ = actor.sample(x0=x0, condition=_extract_actor_obs(obs))
            obs, _, done, env_info = env.step(actions)

            done_envs = done.nonzero(as_tuple=True)[0]
            for e in done_envs.tolist():
                n_active = int(env_uw._active_n[e].item()) if curriculum_on else N
                success  = bool(env_uw._just_succeeded[e].item())
                ep_results.append((n_active, success))
                total_done += 1
                if total_done >= target:
                    break

    # ── Summary ──────────────────────────────────────────────────────────────
    total_suc = sum(s for _, s in ep_results)
    print(f"\n[Play] Results: {total_suc}/{total_done} = {total_suc/total_done:.1%} success")

    if curriculum_on:
        by_n: dict[int, list[bool]] = defaultdict(list)
        for n_a, suc in ep_results:
            by_n[n_a].append(suc)
        print("[Play] Breakdown by active-drone count:")
        for n_a in sorted(by_n):
            suc_list = by_n[n_a]
            print(f"  N_active={n_a:2d}: {sum(suc_list):3d}/{len(suc_list):3d} "
                  f"= {sum(suc_list)/len(suc_list):.1%}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Resume helper                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _reset_critic_weights(critic: "DroneCriticNet") -> None:
    """Re-initialise critic MLP weights and clear observation normalisation stats.

    Called when entering a new curriculum phase so the critic can re-learn the
    value function under the updated reward/termination structure while the actor
    (frozen for actor_freeze_iters updates) stays behaviorally stable.
    """
    for m in critic.mlp.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
    # Reset running statistics so old phase-1 normalisation doesn't pollute phase-2.
    norm = critic.obs_norm
    norm._mean.zero_()
    norm._var.fill_(1.0)
    norm._std.fill_(1.0)
    norm.count.zero_()
    print("[Phase-2] Critic weights and obs-norm stats reset.", flush=True)


def _load_resume(runner, path: str, device) -> tuple[int, bool]:
    """Load a checkpoint, re-initialising the critic if obs_dim changed (N changed).

    Actor weights (NeighborEncoder + FlowMlp) are N-independent and always loaded.
    The critic's first layer is obs_dim-sized; when num_drones changes the shapes
    won't match, so critic loading is silently skipped and the critic stays fresh.

    Returns:
        (iteration, critic_reinit) — iteration from the checkpoint; critic_reinit
        is True when the critic was NOT loaded (shape mismatch detected).
    """
    import re

    content = torch.load(path, map_location=device, weights_only=False)
    critic_reinit = False

    if "model_state_dicts" in content:
        # ── Actor: N-independent parameters → always loadable ────────────
        actor_sd = content["model_state_dicts"].get("actor")
        if actor_sd is not None:
            try:
                runner._model_load_state_dict(runner._agent.model_dict["actor"], actor_sd)
                print("[Resume] Actor weights loaded.", flush=True)
            except RuntimeError as exc:
                print(f"[Resume][WARN] Actor load failed: {exc}. Using fresh actor.", flush=True)

        # ── Critic: obs_dim-dependent → skip on shape mismatch ───────────
        critic_sd = content["model_state_dicts"].get("critic")
        if critic_sd is not None:
            try:
                runner._model_load_state_dict(runner._agent.model_dict["critic"], critic_sd)
                print("[Resume] Critic weights loaded.", flush=True)
            except RuntimeError:
                print(
                    "[Resume] Critic shape mismatch (num_drones changed) — "
                    "reinitialising critic. Actor will be frozen for warmup.",
                    flush=True,
                )
                critic_reinit = True
        # Optimizer is intentionally NOT restored so both actor and critic
        # start with a fresh Adam state (correct when N changes).
    else:
        # Legacy checkpoint: fall back to standard load (may error if N changed).
        runner.load(path)

    iteration = content.get("iteration", 0)
    if iteration == 0:
        match = re.search(r"model_(\d+)\.pt", os.path.basename(path))
        if match:
            iteration = int(match.group(1))

    print(f"[Resume] Checkpoint: iter={iteration}  critic_reinit={critic_reinit}", flush=True)
    return iteration, critic_reinit


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Main                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main() -> None:
    import numpy as np
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # ── env ──────────────────────────────────────────────────────────────────
    env_cfg = MultiDroneDmpcRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.num_drones     = args_cli.num_drones
    if args_cli.n_min > 0:
        env_cfg.n_curriculum_min = args_cli.n_min
    # Harder env settings are enabled from phase 1 so the policy learns the
    # final termination/reward semantics immediately.
    env_cfg.terminate_on_collision_rl   = True
    env_cfg.collision_terminate_dist    = args_cli.collision_terminate_dist
    env_cfg.terminate_on_oob_rl         = True
    env_cfg.hover_vel_scale             = args_cli.hover_vel_scale
    env_cfg.hover_proximity_dist        = args_cli.hover_proximity_dist
    env_cfg.dmpc_guide_scale            = env_cfg.dmpc_guide_scale * 0.5
    env_cfg.dist_to_goal_far_scale      = args_cli.dist_far_scale
    env_cfg.dist_to_goal_far_tanh_scale = args_cli.dist_far_tanh_scale
    env_cfg.dist_to_goal_near_scale     = args_cli.dist_near_scale
    env_cfg.dist_to_goal_near_tanh_scale = args_cli.dist_tanh_scale
    # --curriculum now only controls resume-time critic reset / actor freeze.
    if args_cli.curriculum:
        pass
    env_cfg.__post_init__()           # recompute action_space / observation_space

    env_raw = gym.make("Isaac-MultiDrone-DMPC-RL-Direct-v0", cfg=env_cfg)
    assert isinstance(env_raw.unwrapped, MultiDroneDmpcRLEnv)
    env     = IsaacLabEnvWrapper(env_raw)
    device  = env.device

    N       = env_cfg.num_drones
    actor_obs_dim = gym.spaces.flatdim(env.unwrapped.single_observation_space["policy"])
    # The env returns a runtime "critic" observation with active_n appended, but
    # IsaacLab's static single_observation_space is built from cfg and only
    # declares "policy".  Keep the critic size explicit here.
    critic_obs_dim = actor_obs_dim + 1
    act_dim = gym.spaces.flatdim(env.unwrapped.single_action_space)
    assert act_dim == N * 3, f"Unexpected action dim {act_dim} for N={N}"

    # Per-drone obs dim: own state + SDF obstacle grid + (N-1) neighbour features
    sdf_dim = PER_DRONE_SDF_DIM if env_cfg.enable_sdf_obs else 0
    own_dim = PER_DRONE_OWN_DIM + sdf_dim
    per_drone_dim = own_dim + PER_NEIGHBOUR_DIM * (N - 1)
    assert actor_obs_dim == N * per_drone_dim, (
        f"actor_obs_dim {actor_obs_dim} != N*per_drone_dim {N}*{per_drone_dim} "
        f"(own={own_dim}, sdf={sdf_dim}, neigh={PER_NEIGHBOUR_DIM}*(N-1))"
    )
    assert critic_obs_dim == actor_obs_dim + 1, (
        f"critic_obs_dim {critic_obs_dim} != actor_obs_dim+1 {actor_obs_dim + 1}"
    )

    print(f"[Env] {args_cli.num_envs} envs × {N} drones  "
          f"actor_obs={actor_obs_dim} critic_obs={critic_obs_dim} "
          f"(per_drone={per_drone_dim}, sdf={sdf_dim})  act={act_dim}", flush=True)

    # ── actor (per-agent shared policy) ──────────────────────────────────────
    emb_dim     = args_cli.emb_dim
    hidden_dims = args_cli.hidden_dims

    nn_condition = PerDroneConditionNet(
        N=N, own_dim=own_dim,
        neigh_dim=PER_NEIGHBOUR_DIM,
        emb_dim=emb_dim, hidden_dims=hidden_dims,
    ).to(device)

    nn_flow = FlowMlp(
        x_dim=3, emb_dim=emb_dim,          # single drone: 3D velocity action
        hidden_dims=hidden_dims,
        activations=["elu"] * len(hidden_dims) + ["linear"],
    ).to(device)

    cnf = ContinuousNormalizingFlow(
        x_dims=3,                           # single drone action space
        nn_flow=nn_flow,
        nn_condition=nn_condition,
        sample_steps=args_cli.sample_steps,
        sample_step_schedule="uniform_continuous",
        interpolation_type="rectified_flow",
        device=device,
    )

    # Wrapper handles B×N reshaping; agent sees (B, N*3) / (B, N*P) interface
    actor = PerAgentActorWrapper(cnf, N=N, per_drone_obs_dim=per_drone_dim)

    # ── critic ────────────────────────────────────────────────────────────────
    critic = DroneCriticNet(obs_dim=critic_obs_dim, hidden_dims=hidden_dims).to(device)

    # ── agent ─────────────────────────────────────────────────────────────────
    if AGENT_CFG["use_ewma"]:
        actor.init_proximal(beta_prox=AGENT_CFG["beta_prox"])

    replay_buffer = ReplayBuffer(
        memory_size=args_cli.rollouts,
        num_envs=env.num_envs,
        device=device,
    )

    agent_cfg = {
        **AGENT_CFG,
        "bc_ref_kl_coef0": args_cli.bc_ref_kl_coef0,
        "bc_ref_kl_decay_steps": args_cli.bc_ref_kl_decay_steps,
        "bc_ref_kl_min": args_cli.bc_ref_kl_min,
    }

    models = {"critic": critic, "actor": actor}
    agent  = PolicyFlow(
        models=models,
        replay_buffer=replay_buffer,
        device=device,
        cfg=agent_cfg,
    )
    agent.init_replay_buffer(
        critic_observation_size=critic_obs_dim,
        actor_observation_size=actor_obs_dim,
        action_size=act_dim,            # N*3 — env action space
    )

    # ── runner cfg ────────────────────────────────────────────────────────────
    run_name = args_cli.experiment_name or datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    bc_step_offset = args_cli.bc_iters if args_cli.bc_iters > 0 and not args_cli.resume else 0
    bc_critic_step_offset = (
        args_cli.bc_critic_warmup_iters if bc_step_offset > 0 else 0
    )
    runner_cfg = {
        "max_iterations": args_cli.max_iterations + bc_step_offset + bc_critic_step_offset,
        "rollouts":       args_cli.rollouts,
        "save_interval":  args_cli.save_interval,
        "log_dir":        args_cli.log_dir,
        "experiment_name": run_name,
        "run_name":        run_name,
        "wandb_project":   args_cli.wandb_project if args_cli.wandb else "",
        # If BC warmup is used, BC collection/training matures the actor and
        # critic normalizers.  Keep them fixed during subsequent RL to preserve
        # the post-BC policy/anchor semantics.  Without BC, update online in RL.
        "update_model_normalizers": args_cli.bc_iters <= 0,
    }

    if args_cli.wandb and _WANDB_AVAILABLE:
        _wandb.init(
            project=args_cli.wandb_project,
            name=run_name,
            config={**vars(args_cli), **dict(env_cfg.__dict__)},
            reinit=False,
        )

    runner = IsaaclabRunner(env=env, agent=agent, cfg=runner_cfg)

    # ── checkpoint resume ─────────────────────────────────────────────────────
    start_iteration = 0
    if args_cli.resume:
        start_iteration, critic_reinit = _load_resume(runner, args_cli.resume, device)

        # ── Play mode: eval only, no training ────────────────────────────────
        if args_cli.play:
            _run_play(env=env, actor=actor, N=N, args_cli=args_cli, device=device)
            env.close()
            simulation_app.close()
            return

        # ── Phase-2: curriculum flag resets critic and freezes actor ─────
        # Takes precedence over the automatic N-change reinit path below.
        if args_cli.curriculum:
            _reset_critic_weights(critic)
            critic_reinit = True  # triggers actor freeze logic below
            freeze_iters  = args_cli.actor_freeze_iters
            print(
                f"[Phase-2] Critic reset. Freezing actor for {freeze_iters} PPO updates.",
                flush=True,
            )
        else:
            freeze_iters = args_cli.critic_warmup_iters if critic_reinit else 0

        # When the critic was reinitialised (N changed or --reset_critic),
        # freeze the actor so the new critic can converge against a stationary
        # actor before joint training resumes.
        warmup = freeze_iters if critic_reinit else 0
        if warmup > 0:
            print(
                f"[Resume] Freezing actor for {warmup} PPO updates (critic warmup).",
                flush=True,
            )
            for p in cnf.model.parameters():
                p.requires_grad_(False)
            runner.train(start_iteration=start_iteration, return_epochs=warmup)
            for p in cnf.model.parameters():
                p.requires_grad_(True)
            start_iteration += warmup
            print("[Resume] Actor unfrozen — starting full RL.", flush=True)
    elif args_cli.play:
        print("[Play] --play requires --resume <checkpoint>. Exiting.", flush=True)
        env.close()
        simulation_app.close()
        return

    # ── Phase 1: BC warmup (skipped when --resume is set) ────────────────────
    if args_cli.bc_iters > 0 and not args_cli.resume:
        run_bc_phase(
            env=env, cnf=cnf, critic=critic,
            N=N, per_drone_dim=per_drone_dim,
            actor_obs_dim=actor_obs_dim, critic_obs_dim=critic_obs_dim,
            device=device, run_name=run_name,
        )
        start_iteration = args_cli.bc_iters

        # Attach the post-BC actor as a frozen KL reference for PPO.  PolicyFlow's
        # BC-ref KL path is wrapper-aware for this per-drone actor.
        agent.set_bc_ref_model(actor)
        print(
            f"[BC→RL] KL anchor attached "
            f"(coef0={args_cli.bc_ref_kl_coef0}, decay={args_cli.bc_ref_kl_decay_steps}, "
            f"min={args_cli.bc_ref_kl_min})",
            flush=True,
        )

        # Critic warmup against a fixed actor before joint PPO updates.
        if args_cli.bc_critic_warmup_iters > 0:
            print(
                f"[BC→RL] Freezing actor for {args_cli.bc_critic_warmup_iters} "
                "PPO updates for critic warmup.",
                flush=True,
            )
            for p in cnf.model.parameters():
                p.requires_grad_(False)
            runner.train(start_iteration=start_iteration, return_epochs=args_cli.bc_critic_warmup_iters)
            for p in cnf.model.parameters():
                p.requires_grad_(True)
            start_iteration += args_cli.bc_critic_warmup_iters
            print("[BC→RL] Actor unfrozen — starting joint PPO with KL anchor.", flush=True)

    # ── Phase 2: RL ───────────────────────────────────────────────────────────
    runner.train(start_iteration=start_iteration, return_epochs=args_cli.max_iterations)

    if args_cli.wandb and _WANDB_AVAILABLE:
        _wandb.finish()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
