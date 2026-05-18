"""online_bc_dmpc.py - Online expert collection + BC training for multi-drone DMPC.

Mirrors the structure of ``online_bc_curobo.py`` (which uses CuRobo as the
expert) but the expert here is the Python DMPC re-implementation in
``dmpc_expert.py`` (same directory).  The student is a simple MLP that maps
environment observations to the ``4 * num_drones``-dimensional thrust / moment
action of :class:`MultiDroneDmpcEnv`.

Pipeline per outer iteration:

  1. *Collect*: roll out the env for ``--steps_per_batch`` steps; each step
     queries the DMPC expert for the desired world-frame acceleration, maps it
     to a 4-D thrust / moment action via the env's differential-flatness
     controller, applies it, and stores ``(obs, expert_action)``.
  2. *Train*: run ``--bc_epochs_per_round`` BC updates (MSE on tanh-squashed
     actions) over a rolling buffer of the most recent transitions.
  3. *Eval*: every ``--eval_every_rounds`` rounds run a deterministic rollout
     using the student policy and log mean episode return / success rate.

The script lives inside the local PolicyFlow copy at
``online_dmpc_to_learning/PolicyFlow/scripts/isaaclab/`` and adds two paths to
``sys.path`` so it works in-place without ``pip install -e``:

* ``online_dmpc_to_learning/`` -- exposes the ``quadcopter`` package containing
  the multi-drone env (``Isaac-MultiDrone-DMPC-Direct-v0``) and DMPC expert.
* ``online_dmpc_to_learning/PolicyFlow/policyflow/`` -- exposes the local
  ``policyflow_torch`` package, from which we use
  :class:`EmpiricalNormalization` to keep observation statistics aligned with
  the rest of the PolicyFlow training stack.

Run from anywhere::

    python ~/git_branch/online_dmpc_to_learning/PolicyFlow/scripts/isaaclab/online_bc_dmpc.py \\
        --num_envs 32 \\
        --num_drones 4 \\
        --save_path runs/online_bc_dmpc/model.pt \\
        [--dmpc_max_envs 16] \\
        [--wandb]

``--dmpc_max_envs`` caps how many envs are planned by DMPC each step (the rest
fall back to a PD heuristic toward the goal). DMPC runs on CPU and is the slow
piece -- start with 8-16 envs while tuning.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

# Resolve the two repo-local paths we need to expose on sys.path:
#   _HERE              = online_dmpc_to_learning/PolicyFlow/scripts/isaaclab/
#   _POLICYFLOW_ROOT   = online_dmpc_to_learning/PolicyFlow/policyflow/
#   _PROJECT_ROOT      = online_dmpc_to_learning/
# Adding both makes ``import policyflow_torch`` and ``import quadcopter``
# resolve to the copies inside this repository, with no pip-install needed.
_HERE = Path(__file__).resolve().parent
_POLICYFLOW_ROOT = _HERE.parent.parent / "policyflow"
_PROJECT_ROOT = _HERE.parent.parent.parent
for _p in (_POLICYFLOW_ROOT, _PROJECT_ROOT):
    p_str = str(_p)
    if _p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

# ── argument parsing must happen before AppLauncher ───────────────────────────
parser = argparse.ArgumentParser(description="Online BC with DMPC expert (multi-drone).")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_drones", type=int, default=4)
parser.add_argument("--task", type=str, default="Isaac-MultiDrone-DMPC-Direct-v0")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=0)

# Collection / training schedule.
parser.add_argument("--n_rounds", type=int, default=200,
                    help="Total outer iterations (0 = run forever).")
parser.add_argument("--steps_per_batch", type=int, default=64,
                    help="Env steps per collection batch.")
parser.add_argument("--bc_epochs_per_round", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--buffer_capacity", type=int, default=200_000,
                    help="Max (obs, action) transitions in the rolling buffer.")
parser.add_argument("--min_buffer_transitions", type=int, default=2_000,
                    help="Skip BC updates until the buffer has at least this many transitions.")
parser.add_argument("--dmpc_max_envs", type=int, default=16,
                    help="Plan with DMPC for at most this many envs per step; the rest use the PD fallback.")

# Eval.
parser.add_argument("--eval_every_rounds", type=int, default=10)
parser.add_argument("--eval_steps", type=int, default=200)

# BC / model.
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--hidden_dims", type=int, nargs="*", default=[256, 256, 256])
parser.add_argument("--save_path", type=str, default="runs/online_bc_dmpc/model.pt")
parser.add_argument("--save_every_rounds", type=int, default=20)
parser.add_argument("--resume", type=str, default=None)

# Logging.
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--wandb_project", type=str, default="online_bc_dmpc")
parser.add_argument("--wandb_run_name", type=str, default=None)

# Isaac sim launch flags.
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless mode unless the user explicitly asked for a viewer -- BC
# collection is throughput-bound and rendering wastes time.
if not getattr(args_cli, "headless", False) and not getattr(args_cli, "enable_cameras", False):
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ───────────────────────────────────────────────────────
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import quadcopter  # noqa: F401, E402 - registers Isaac-MultiDrone-DMPC-Direct-v0
from quadcopter.multi_drone_dmpc_env import (  # noqa: E402
    MultiDroneDmpcEnv,
    MultiDroneDmpcEnvCfg,
)
from quadcopter.dmpc_expert import DMPCExpert, DMPCParams  # noqa: E402
# Use the local PolicyFlow copy for empirical observation normalisation. This
# keeps the obs-stat treatment consistent with other PolicyFlow training
# scripts (online_bc_curobo.py, train_xhand.py, ...).
from policyflow_torch.modules.normalizer import EmpiricalNormalization  # noqa: E402

try:
    import wandb as _wandb  # noqa: E402
    _WANDB_AVAILABLE = True
except Exception:
    _wandb = None
    _WANDB_AVAILABLE = False


# ── student policy ────────────────────────────────────────────────────────────
class MlpPolicy(nn.Module):
    """Deterministic MLP mapping obs -> tanh-squashed action in [-1, 1]^A.

    Wraps an :class:`EmpiricalNormalization` so observation statistics are
    tracked online (and persisted in the checkpoint), matching the convention
    used by the rest of the PolicyFlow training scripts.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.obs_norm = EmpiricalNormalization(shape=obs_dim, until=int(1e8))
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(self.obs_norm(obs)))


# ── replay buffer (rolling, on-device) ────────────────────────────────────────
class TransitionBuffer:
    """Ring buffer for (obs, action) transitions kept on the training device."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, obs_dim), device=device)
        self.act = torch.zeros((capacity, action_dim), device=device)
        self.size = 0
        self.ptr = 0

    def add(self, obs: torch.Tensor, act: torch.Tensor) -> None:
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.capacity
        self.obs[idx] = obs.to(self.device)
        self.act[idx] = act.to(self.device)
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[idx], self.act[idx]


# ── expert wrapper: DMPC accel → env action ──────────────────────────────────
def expert_action(
    env: MultiDroneDmpcEnv,
    expert: DMPCExpert,
    dmpc_env_mask: torch.Tensor,
) -> torch.Tensor:
    """Query the DMPC expert for the active envs; fall back to a PD action for the rest.

    Returns a ``(num_envs, num_drones * 4)`` tensor matching the env action space.
    """
    states = env.get_world_states()
    pos_w = states["pos_w"]
    vel_w = states["lin_vel_w"]
    goal_w = states["goal_w"]
    origins = env._terrain.env_origins

    dmpc_ids = dmpc_env_mask.nonzero(as_tuple=False).flatten()
    accel = torch.zeros_like(pos_w)
    if dmpc_ids.numel() > 0:
        accel = expert.compute(pos_w, vel_w, goal_w, env_origins=origins, env_ids=dmpc_ids)

    # Fallback PD for envs not planned by DMPC. This keeps throughput high and
    # gives the student a non-trivial signal even when DMPC is the bottleneck.
    if dmpc_ids.numel() < env.num_envs:
        rest_mask = ~dmpc_env_mask
        pd = 2.0 * (goal_w - pos_w) - 1.5 * vel_w
        pd = pd.clamp(-1.0, 1.0)
        accel[rest_mask] = pd[rest_mask]

    return env.acc_to_action(accel)


# ── main loop ────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = MultiDroneDmpcEnvCfg()
    env_cfg.num_drones = args_cli.num_drones
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.__post_init__()
    if getattr(args_cli, "device", None):
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped  # type: MultiDroneDmpcEnv
    device = env.device
    obs_dim = env.cfg.observation_space
    action_dim = env.cfg.action_space
    print(f"[online_bc_dmpc] obs_dim={obs_dim} action_dim={action_dim} num_envs={env.num_envs}")

    policy = MlpPolicy(obs_dim, action_dim, args_cli.hidden_dims).to(device)
    if args_cli.resume is not None and os.path.isfile(args_cli.resume):
        policy.load_state_dict(torch.load(args_cli.resume, map_location=device))
        print(f"[online_bc_dmpc] loaded checkpoint from {args_cli.resume}")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args_cli.lr)

    buffer = TransitionBuffer(args_cli.buffer_capacity, obs_dim, action_dim, device)

    expert = DMPCExpert(
        num_drones=env.cfg.num_drones,
        params=DMPCParams(
            pmin=env.cfg.pos_min,
            pmax=env.cfg.pos_max,
            rmin=env.cfg.rmin,
            h=env.cfg.sim.dt * env.cfg.decimation,
        ),
        device=device,
    )

    dmpc_max = min(args_cli.dmpc_max_envs, env.num_envs)
    rotation_offset = 0

    use_wandb = args_cli.wandb and _WANDB_AVAILABLE
    if use_wandb:
        _wandb.init(
            project=args_cli.wandb_project,
            name=args_cli.wandb_run_name,
            config=vars(args_cli),
        )

    obs_dict, _ = env.reset(seed=args_cli.seed)
    obs = obs_dict["policy"]

    recent_returns: deque[float] = deque(maxlen=20)
    episode_returns = torch.zeros(env.num_envs, device=device)
    round_idx = 0
    total_env_steps = 0
    n_rounds = args_cli.n_rounds if args_cli.n_rounds > 0 else 10_000_000

    save_dir = os.path.dirname(args_cli.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    try:
        while round_idx < n_rounds:
            # ── 1. Collection ──────────────────────────────────────────────
            t_collect = time.time()
            dmpc_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            ids = (torch.arange(dmpc_max, device=device) + rotation_offset) % env.num_envs
            dmpc_mask[ids] = True
            rotation_offset = (rotation_offset + dmpc_max) % env.num_envs

            collect_returns = []
            for _ in range(args_cli.steps_per_batch):
                with torch.no_grad():
                    action = expert_action(env, expert, dmpc_mask)
                # Update empirical obs statistics with the latest batch before
                # storing — this keeps the normaliser tracking the rollout
                # distribution online.
                policy.obs_norm.update(obs.detach())
                buffer.add(obs.detach(), action.detach())
                step_obs, reward, terminated, truncated, info = env.step(action)
                obs = step_obs["policy"]
                episode_returns += reward
                done = terminated | truncated
                if done.any():
                    finished = episode_returns[done].detach().cpu().tolist()
                    collect_returns.extend(finished)
                    episode_returns[done] = 0.0
                    expert.reset(done.nonzero(as_tuple=False).flatten())
                total_env_steps += env.num_envs
            collect_dt = time.time() - t_collect
            if collect_returns:
                recent_returns.extend(collect_returns)

            # ── 2. BC training ────────────────────────────────────────────
            bc_loss = float("nan")
            if buffer.size >= max(args_cli.min_buffer_transitions, args_cli.batch_size):
                losses = []
                for _ in range(args_cli.bc_epochs_per_round):
                    b_obs, b_act = buffer.sample(args_cli.batch_size)
                    pred = policy(b_obs)
                    loss = F.mse_loss(pred, b_act.clamp(-0.999, 0.999))
                    optimizer.zero_grad()
                    loss.backward()
                    if args_cli.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), args_cli.grad_clip)
                    optimizer.step()
                    losses.append(loss.item())
                bc_loss = float(np.mean(losses))

            # ── 3. Eval ───────────────────────────────────────────────────
            eval_metrics: dict[str, float] = {}
            if args_cli.eval_every_rounds > 0 and round_idx % args_cli.eval_every_rounds == 0:
                eval_metrics = evaluate(env, policy, args_cli.eval_steps)
                obs_dict, _ = env.reset()
                obs = obs_dict["policy"]
                episode_returns.zero_()
                expert.reset()

            # ── 4. Logging + checkpointing ────────────────────────────────
            mean_ret = float(np.mean(recent_returns)) if recent_returns else float("nan")
            msg = (
                f"[round {round_idx:04d}] "
                f"buf={buffer.size:>7d}  "
                f"bc_loss={bc_loss:.4f}  "
                f"mean_collect_ret={mean_ret:.2f}  "
                f"collect_dt={collect_dt:.1f}s  "
                f"steps={total_env_steps}"
            )
            if eval_metrics:
                msg += f"  eval_ret={eval_metrics.get('eval_return', float('nan')):.2f}"
                msg += f"  eval_succ={eval_metrics.get('eval_success_rate', 0.0):.2f}"
            print(msg, flush=True)

            if use_wandb:
                log = {
                    "round": round_idx,
                    "buffer_size": buffer.size,
                    "bc_loss": bc_loss,
                    "mean_collect_return": mean_ret,
                    "total_env_steps": total_env_steps,
                    "collect_dt_s": collect_dt,
                }
                log.update(eval_metrics)
                _wandb.log(log)

            if args_cli.save_every_rounds > 0 and round_idx % args_cli.save_every_rounds == 0:
                torch.save(policy.state_dict(), args_cli.save_path)

            round_idx += 1
    finally:
        torch.save(policy.state_dict(), args_cli.save_path)
        env.close()
        simulation_app.close()


@torch.no_grad()
def evaluate(env: MultiDroneDmpcEnv, policy: MlpPolicy, n_steps: int) -> dict[str, float]:
    """Deterministic student rollout. Returns mean per-episode return and the
    fraction of envs whose drones all ended within 0.3 m of their goals."""
    device = env.device
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    returns = torch.zeros(env.num_envs, device=device)
    finished_returns: list[float] = []

    for _ in range(n_steps):
        action = policy(obs)
        step_obs, reward, terminated, truncated, info = env.step(action)
        obs = step_obs["policy"]
        returns += reward
        done = terminated | truncated
        if done.any():
            finished_returns.extend(returns[done].detach().cpu().tolist())
            returns[done] = 0.0

    pos_w = torch.stack([r.data.root_pos_w for r in env._robots], dim=1)
    final_dist = torch.linalg.norm(pos_w - env._goal_pos_w, dim=-1)
    success = (final_dist.max(dim=-1).values < 0.3).float().mean().item()
    mean_ret = float(np.mean(finished_returns)) if finished_returns else float("nan")
    return {"eval_return": mean_ret, "eval_success_rate": success}


if __name__ == "__main__":
    main()
