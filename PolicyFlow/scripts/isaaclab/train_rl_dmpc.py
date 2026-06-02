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
parser.add_argument("--bc_ref_kl_coef0",    type=float, default=1.0,
                    help="Initial KL coef toward frozen BC policy in RL phase.")
parser.add_argument("--bc_ref_kl_decay_steps", type=float, default=500.0,
                    help="PPO updates to decay KL coef to ~1/100 of initial.")
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
        """(batch, N*3) + (batch, N*P) → per-agent forward; outputs reshaped to (batch, N*3)."""
        batch = condition.shape[0]
        result = self._cnf.compute_flow_variation(
            x1        = x1.reshape(batch * self.N, 3),
            condition = self._split_obs(condition),
            x0        = x0.reshape(batch * self.N, 3) if x0 is not None else None,
            **kwargs,
        )

        per_agent_shape = (batch * self.N, 3)

        def _r(t: torch.Tensor) -> torch.Tensor:
            # Only reshape per-agent action tensors (BN, 3) → (B, N*3).
            # Scalar losses and other shapes pass through unchanged.
            if t.shape == per_agent_shape:
                return t.reshape(batch, self.N * 3)
            return t

        if not isinstance(result, tuple):
            return _r(result)

        out = []
        for item in result:
            if isinstance(item, torch.Tensor):
                out.append(_r(item))
            elif isinstance(item, tuple):
                # proximal_info = (vel_tensor (BN,3), std_vec (D,))
                # Tile per-drone std (3,) → (N*3,) so Normal broadcasts with (batch, N*3).
                vel = _r(item[0])
                std = item[1].repeat(self.N) if item[1].dim() == 1 else item[1]
                out.append((vel, std))
            else:
                out.append(item)   # None
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


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ PPO agent config (plain dict — avoids @configclass MISSING inheritance)  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
AGENT_CFG: dict = {
    "desired_kl":                  0.01,
    "use_kl_lr_schedule":          False,
    "use_kl_lr_schedule_critic_only": True,
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
    obs_dim: int,
    device: torch.device,
    run_name: str,
) -> None:
    """Phase 1: collect DMPC rollouts, train actor via BC + critic via value regression.

    Saves a runner-compatible checkpoint at log_dir/run_name/bc_checkpoint.pt.
    """
    from quadcopter.dmpc_expert import DMPCExpert, DMPCParams

    env_uw   = env.unwrapped
    num_envs = env.num_envs
    gamma    = AGENT_CFG["discount_factor"]

    bc_expert = DMPCExpert(num_drones=N, params=DMPCParams(), device=device, verbose=False)
    bc_expert.enable_gpu_admm(max_envs=num_envs)

    # Ring buffer (per-drone)
    BUF = 200_000
    buf_obs  = torch.zeros(BUF, per_drone_dim, device=device)
    buf_act  = torch.zeros(BUF, 3, device=device)
    buf_ptr, buf_full = 0, False

    actor_opt  = torch.optim.Adam(list(cnf.model.parameters()), lr=args_cli.bc_lr)
    critic_opt = torch.optim.Adam(list(critic.parameters()),    lr=AGENT_CFG["critic_learning_rate"])

    obs, _ = env.reset()

    for it in range(args_cli.bc_iters):
        # ── Collect rollout with DMPC ────────────────────────────────────────
        obs_list, act_list, rew_list, done_list = [], [], [], []

        for _ in range(args_cli.rollouts):
            st      = env_uw._stack_drone_state()
            ref_pos, ref_vel, ref_acc = bc_expert.compute(
                pos_w=st["pos_w"], vel_w=st["lin_vel_w"],
                goal_w=env_uw._goal_pos_w,
                env_origins=env_uw._terrain.env_origins,
            )
            env_act = env_uw.reference_to_action(ref_pos, ref_vel, ref_acc)  # (E, N*3)

            obs_list.append(obs.clone())
            act_list.append(env_act.clone())

            obs, rew, term, trunc, _ = env.step(env_act)
            done_list.append((term | trunc).clone())
            rew_list.append(rew.clone())

        # ── Compute discounted returns ───────────────────────────────────────
        with torch.no_grad():
            R = critic(obs)  # bootstrap last state
        returns = []
        for rew, done in zip(reversed(rew_list), reversed(done_list)):
            R = rew + gamma * R * (~done).float()
            returns.insert(0, R.clone())

        # ── Update obs normalisers ───────────────────────────────────────────
        all_obs = torch.stack(obs_list)              # (T, E, obs_dim)
        flat_for_norm = all_obs.reshape(-1, obs_dim)
        cnf.model["condition"].update_norm(flat_for_norm)
        critic.update_norm(flat_for_norm)

        # ── Add to BC buffer ─────────────────────────────────────────────────
        for obs_s, act_s in zip(obs_list, act_list):
            obs_per = obs_s.reshape(-1, per_drone_dim)   # (E*N, per_dim)
            act_per = act_s.reshape(-1, 3)               # (E*N, 3)
            n = obs_per.shape[0]
            end = buf_ptr + n
            if end <= BUF:
                buf_obs[buf_ptr:end] = obs_per
                buf_act[buf_ptr:end] = act_per
                buf_ptr = end % BUF
                if buf_ptr == 0:
                    buf_full = True
            else:
                first = BUF - buf_ptr
                buf_obs[buf_ptr:]   = obs_per[:first]
                buf_act[buf_ptr:]   = act_per[:first]
                buf_obs[:n - first] = obs_per[first:]
                buf_act[:n - first] = act_per[first:]
                buf_ptr  = n - first
                buf_full = True

        buf_valid = BUF if buf_full else buf_ptr

        # ── BC actor update (rectified-flow matching) ────────────────────────
        bc_loss_val = float("nan")
        if buf_valid >= args_cli.bc_batch_size:
            cnf.model["condition"].train()
            cnf.model["flow"].train()
            for _ in range(args_cli.bc_epochs_per_round):
                idx  = torch.randperm(buf_valid, device=device)[:args_cli.bc_batch_size]
                obs_b = buf_obs[idx]   # (B, per_dim)
                x1    = buf_act[idx]   # (B, 3)
                x0    = torch.randn_like(x1)
                t     = torch.rand(x1.shape[0], device=device)
                xt    = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
                cond  = cnf.model["condition"](obs_b)
                v_pred = cnf.model["flow"](xt, t, cond)
                bc_loss = (v_pred - (x1 - x0)).pow(2).mean()
                actor_opt.zero_grad()
                bc_loss.backward()
                torch.nn.utils.clip_grad_norm_(cnf.model.parameters(), 1.0)
                actor_opt.step()
                cnf.update_proximal()   # keep EWMA proximal in sync
            bc_loss_val = bc_loss.item()

        # ── Critic value regression ──────────────────────────────────────────
        flat_obs = all_obs.reshape(-1, obs_dim)
        flat_ret = torch.stack(returns).reshape(-1)
        perm = torch.randperm(flat_obs.shape[0], device=device)
        critic_loss_val = float("nan")
        for _ in range(args_cli.bc_epochs_per_round):
            for start in range(0, flat_obs.shape[0], args_cli.bc_batch_size):
                idx = perm[start : start + args_cli.bc_batch_size]
                if len(idx) == 0:
                    break
                critic_loss = F.mse_loss(critic(flat_obs[idx]), flat_ret[idx].detach())
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
            critic_loss_val = critic_loss.item()

        print(
            f"[BC {it+1:4d}/{args_cli.bc_iters}] "
            f"bc={bc_loss_val:.4f}  critic={critic_loss_val:.4f}  "
            f"buf={buf_valid}",
            flush=True,
        )

    # ── Save BC checkpoint (runner-compatible format) ────────────────────────
    ckpt_dir  = Path(args_cli.log_dir) / run_name
    ckpt_path = ckpt_dir / "bc_checkpoint.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dicts": {"actor": cnf.state_dict(), "critic": critic.state_dict()},
            "iteration": 0,
        },
        ckpt_path,
    )
    print(f"[BC] checkpoint saved → {ckpt_path}", flush=True)


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
    env_cfg.__post_init__()           # recompute action_space / observation_space

    env_raw = gym.make("Isaac-MultiDrone-DMPC-RL-Direct-v0", cfg=env_cfg)
    assert isinstance(env_raw.unwrapped, MultiDroneDmpcRLEnv)
    env     = IsaacLabEnvWrapper(env_raw)
    device  = env.device

    N       = env_cfg.num_drones
    obs_dim = gym.spaces.flatdim(env.unwrapped.single_observation_space["policy"])
    act_dim = gym.spaces.flatdim(env.unwrapped.single_action_space)
    assert act_dim == N * 3, f"Unexpected action dim {act_dim} for N={N}"

    # Per-drone obs dim: own state + (N-1) neighbour features
    per_drone_dim = PER_DRONE_OWN_DIM + PER_NEIGHBOUR_DIM * (N - 1)
    assert obs_dim == N * per_drone_dim, (
        f"obs_dim {obs_dim} != N*per_drone_dim {N}*{per_drone_dim}"
    )

    print(f"[Env] {args_cli.num_envs} envs × {N} drones  "
          f"obs={obs_dim} (per_drone={per_drone_dim})  act={act_dim}", flush=True)

    # ── actor (per-agent shared policy) ──────────────────────────────────────
    emb_dim     = args_cli.emb_dim
    hidden_dims = args_cli.hidden_dims

    nn_condition = PerDroneConditionNet(
        N=N, own_dim=PER_DRONE_OWN_DIM,
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
    critic = DroneCriticNet(obs_dim=obs_dim, hidden_dims=hidden_dims).to(device)

    # ── agent ─────────────────────────────────────────────────────────────────
    if AGENT_CFG["use_ewma"]:
        actor.init_proximal(beta_prox=AGENT_CFG["beta_prox"])

    replay_buffer = ReplayBuffer(
        memory_size=args_cli.rollouts,
        num_envs=env.num_envs,
        device=device,
    )

    models = {"critic": critic, "actor": actor}
    agent  = PolicyFlow(
        models=models,
        replay_buffer=replay_buffer,
        device=device,
        cfg=AGENT_CFG,
    )
    agent.init_replay_buffer(
        critic_observation_size=obs_dim,
        actor_observation_size=obs_dim,
        action_size=act_dim,            # N*3 — env action space
    )

    # ── runner cfg ────────────────────────────────────────────────────────────
    run_name = args_cli.experiment_name or datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    runner_cfg = {
        "max_iterations": args_cli.max_iterations,
        "rollouts":       args_cli.rollouts,
        "save_interval":  args_cli.save_interval,
        "log_dir":        args_cli.log_dir,
        "experiment_name": run_name,
        "run_name":        run_name,
        "wandb_project":   args_cli.wandb_project if args_cli.wandb else "",
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
        start_iteration = runner.load(args_cli.resume)
        print(f"[Resume] starting from iteration {start_iteration}", flush=True)

    # ── Phase 1: BC warmup (skipped when --resume is set) ────────────────────
    if args_cli.bc_iters > 0 and not args_cli.resume:
        run_bc_phase(
            env=env, cnf=cnf, critic=critic,
            N=N, per_drone_dim=per_drone_dim, obs_dim=obs_dim,
            device=device, run_name=run_name,
        )

        # Freeze BC actor as reference; attach KL loss to RL agent.
        frozen_cnf = copy.deepcopy(cnf)
        for p in frozen_cnf.model.parameters():
            p.requires_grad_(False)
        frozen_cnf.model.eval()

        kl_step = [0]

        def _bc_kl_loss(obs, delta_actions, *_):
            coef0 = args_cli.bc_ref_kl_coef0
            decay = args_cli.bc_ref_kl_decay_steps
            coef  = coef0 * (0.01 ** min(kl_step[0] / max(decay, 1.0), 1.0))
            kl_step[0] += 1
            if coef < 1e-5:
                return 0
            B = obs.shape[0]
            obs_per = obs.reshape(B * N, per_drone_dim)
            x1 = delta_actions.reshape(B * N, 3)
            x0 = torch.randn_like(x1)
            t  = torch.rand(x1.shape[0], device=device)
            xt = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
            cur_cond = cnf.model["condition"](obs_per)
            vel_cur  = cnf.model["flow"](xt, t, cur_cond)
            with torch.no_grad():
                vel_ref = frozen_cnf.model["flow"](xt, t, frozen_cnf.model["condition"](obs_per))
            return coef * (vel_cur - vel_ref).pow(2).mean()

        agent._bc_loss_fn = _bc_kl_loss
        print(
            f"[BC→RL] KL ref attached  "
            f"(coef0={args_cli.bc_ref_kl_coef0}, decay={args_cli.bc_ref_kl_decay_steps} iters)",
            flush=True,
        )

    # ── Phase 2: RL ───────────────────────────────────────────────────────────
    runner.train(start_iteration=start_iteration, return_epochs=args_cli.max_iterations)

    if args_cli.wandb and _WANDB_AVAILABLE:
        _wandb.finish()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
