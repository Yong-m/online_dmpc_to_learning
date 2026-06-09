"""train_rl_ippo.py — Vanilla PPO + IPPO for multi-drone navigation (no DMPC guide).

Differences from train_rl_dmpc.py:
  - Gaussian MLP actor instead of PolicyFlow (rectified flow)
  - DMPC expert fully disabled (dmpc_guide_enabled=False) — no bottleneck
  - Always IPPO (PerDroneEnvWrapper + PerDroneCriticNet)
  - No BC warmup

Usage::

    python train_rl_ippo.py \\
        --num_envs 768 --num_drones 10 --n_min 1 \\
        --max_iterations 500000 \\
        --experiment_name ippo_vanilla [--headless]
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

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Vanilla PPO/IPPO for multi-drone (no DMPC guide).")
parser.add_argument("--num_envs",        type=int,   default=768)
parser.add_argument("--num_drones",      type=int,   default=10)
parser.add_argument("--max_iterations",  type=int,   default=500000)
parser.add_argument("--rollouts",        type=int,   default=100)
parser.add_argument("--save_interval",   type=int,   default=10)
parser.add_argument("--log_dir",         type=str,   default="runs/rl_ippo")
parser.add_argument("--experiment_name", type=str,   default=None)
parser.add_argument("--resume",          type=str,   default=None)
parser.add_argument("--seed",            type=int,   default=42)
# Model
parser.add_argument("--emb_dim",     type=int,  default=128)
parser.add_argument("--hidden_dims", type=int,  nargs="*", default=[256, 256, 256])
# N curriculum
parser.add_argument("--n_min", type=int, default=0,
                    help="Min active drones for N curriculum (0=disabled).")
# Reward / env
parser.add_argument("--hover_vel_scale",          type=float, default=5.0)
parser.add_argument("--hover_proximity_dist",     type=float, default=0.4)
parser.add_argument("--dist_far_scale",           type=float, default=50.0)
parser.add_argument("--dist_far_tanh_scale",      type=float, default=0.8)
parser.add_argument("--collision_terminate_dist", type=float, default=0.12)
# Resume
parser.add_argument("--reset_critic", action="store_true", default=False,
                    help="On resume: reset critic weights.")
# Play / eval
parser.add_argument("--play",          action="store_true", default=False)
parser.add_argument("--eval_episodes", type=int, default=50)
# Wandb
parser.add_argument("--wandb",         action="store_true", default=False)
parser.add_argument("--wandb_project", type=str, default="ippo_vanilla")

from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = getattr(args_cli, "headless", True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ───────────────────────────────────────────────────────
import gymnasium as gym          # noqa: E402
import torch                     # noqa: E402
import torch.nn as nn            # noqa: E402

import quadcopter                # noqa: F401,E402
from quadcopter.multi_drone_dmpc_rl_env import (  # noqa: E402
    MultiDroneDmpcRLEnv,
    MultiDroneDmpcRLEnvCfg,
)
from quadcopter.multi_drone_dmpc_env import (  # noqa: E402
    PER_DRONE_OWN_DIM,
    PER_DRONE_SDF_DIM,
    PER_NEIGHBOUR_DIM,
)

from policyflow_torch.env import IsaacLabEnvWrapper         # noqa: E402
from policyflow_torch.storage import ReplayBuffer           # noqa: E402
from policyflow_torch.modules import NeighborEncoder        # noqa: E402
from policyflow_torch.modules.normalizer import EmpiricalNormalization  # noqa: E402
from policyflow_torch.agents import PPO                     # noqa: E402
from policyflow_torch.runners import IsaaclabRunner         # noqa: E402

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
# ║ Networks                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class PerDroneConditionNet(nn.Module):
    """Shared per-drone encoder: cross-attention over neighbors → emb_dim."""

    def __init__(self, N, own_dim, neigh_dim, emb_dim, hidden_dims):
        super().__init__()
        self.N         = N
        self.own_dim   = own_dim
        self.neigh_dim = neigh_dim
        self.per_dim   = own_dim + neigh_dim * (N - 1)
        self.emb_dim   = emb_dim

        self.own_norm   = EmpiricalNormalization(shape=own_dim,   until=int(1e8))
        self.neigh_norm = EmpiricalNormalization(shape=neigh_dim, until=int(1e8))
        self.neighbor_enc = NeighborEncoder(
            own_dim=own_dim, neighbor_dim=neigh_dim,
            emb_dim=emb_dim, num_heads=4,
        )
        in_d = own_dim + emb_dim
        layers: list[nn.Module] = []
        prev = in_d
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, emb_dim))
        self.proj = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """(BN, per_dim) → (BN, emb_dim)."""
        BN    = obs.shape[0]
        own   = obs[:, :self.own_dim]
        neigh = obs[:, self.own_dim:]
        own_n = self.own_norm(own)
        N_neigh = self.N - 1
        if N_neigh > 0:
            neigh_n = self.neigh_norm(
                neigh.reshape(BN * N_neigh, self.neigh_dim)
            ).reshape(BN, N_neigh, self.neigh_dim)
        else:
            neigh_n = torch.zeros(BN, 0, self.neigh_dim, device=obs.device)
        enc = self.neighbor_enc(own_n, neigh_n)
        return self.proj(torch.cat([own_n, enc], dim=-1))

    def update_norm(self, obs: torch.Tensor) -> None:
        """obs: (BN, per_dim) — update running stats."""
        self.own_norm.update(obs[:, :self.own_dim].detach())
        if self.N > 1:
            self.neigh_norm.update(
                obs[:, self.own_dim:].reshape(-1, self.neigh_dim).detach()
            )


class GaussianActor(nn.Module):
    """Per-drone Gaussian actor: obs → (mean, std).

    Exposes ``self.model = {"condition": self}`` so the runner's
    normalizer-update path (which looks for model["condition"].update_norm)
    works without modification.
    """

    def __init__(self, N, own_dim, neigh_dim, emb_dim, hidden_dims,
                 action_dim: int = 3, log_std_init: float = 0.0):
        super().__init__()
        self._cond = PerDroneConditionNet(
            N=N, own_dim=own_dim, neigh_dim=neigh_dim,
            emb_dim=emb_dim, hidden_dims=hidden_dims,
        )
        self.mean_head = nn.Sequential(
            nn.Linear(emb_dim, action_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init)
        )
        # Expose for runner's _update_model_normalizers
        self.model = {"condition": self}

    def forward(self, obs: torch.Tensor, compute_std: bool = False):
        """(B, per_dim) → mean (B, 3) [, std (B, 3)]."""
        emb  = self._cond(obs)
        mean = self.mean_head(emb)
        if compute_std:
            std = self.log_std.exp().expand_as(mean)
            return mean, std
        return mean

    def update_norm(self, obs: torch.Tensor) -> None:
        """Called by runner via model["condition"].update_norm(actor_obs)."""
        self._cond.update_norm(obs)


class PerDroneCriticNet(nn.Module):
    """Per-drone critic: same encoder as actor + linear value head."""

    def __init__(self, N, own_dim, neigh_dim, emb_dim, hidden_dims):
        super().__init__()
        self._encoder   = PerDroneConditionNet(
            N=N, own_dim=own_dim, neigh_dim=neigh_dim,
            emb_dim=emb_dim, hidden_dims=hidden_dims,
        )
        self.value_head = nn.Linear(emb_dim, 1)

    def forward(self, obs: torch.Tensor, hidden_state=None) -> torch.Tensor:
        return self.value_head(self._encoder(obs)).squeeze(-1)

    def update_norm(self, obs: torch.Tensor) -> None:
        enc = self._encoder
        enc.own_norm.update(obs[:, :enc.own_dim].detach())
        if enc.N > 1:
            enc.neigh_norm.update(
                obs[:, enc.own_dim:].reshape(-1, enc.neigh_dim).detach()
            )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ IPPO env wrapper                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class PerDroneEnvWrapper:
    """Flattens E×N multi-drone env into E*N single-drone virtual envs."""

    def __init__(self, env: IsaacLabEnvWrapper, N: int, per_drone_dim: int):
        self._env           = env
        self._N             = N
        self._per_drone_dim = per_drone_dim
        self.num_envs       = env.num_envs * N
        self.device         = env.device

    @property
    def unwrapped(self):
        return self._env.unwrapped

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    @property
    def max_episode_length(self):
        return self.unwrapped.max_episode_length

    def reset(self):
        obs, info = self._env.reset()
        return self._split_obs(obs), info

    def step(self, actions: torch.Tensor):
        E, N = self._env.num_envs, self._N
        obs, _, done, info = self._env.step(actions.reshape(E, N * 3))
        rewards  = self._env.unwrapped._per_drone_rewards_last.reshape(E * N)
        done_per = done.unsqueeze(-1).expand(E, N).reshape(E * N)
        if isinstance(info, dict) and "time_outs" in info:
            info = dict(info)
            info["time_outs"] = info["time_outs"].unsqueeze(-1).expand(E, N).reshape(E * N)
        return self._split_obs(obs), rewards, done_per, info

    def _split_obs(self, obs):
        E, N, P = self._env.num_envs, self._N, self._per_drone_dim
        key = "actor_observations" if isinstance(obs, dict) and "actor_observations" in obs else "policy"
        per_drone = (obs[key] if isinstance(obs, dict) else obs).reshape(E * N, P)
        return {"actor_observations": per_drone, "critic_observations": per_drone}

    def close(self):
        self._env.close()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ PPO config                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
AGENT_CFG: dict = {
    "desired_kl":           0.01,
    "discount_factor":      0.995,
    "lam":                  0.95,
    "time_limit_bootstrap": True,
    "ratio_clip":           0.2,
    "clip_predicted_values": True,
    "value_clip":           0.2,
    "value_loss_scale":     0.5,
    "grad_norm_clip":       1.0,
    "action_clip":          1.0,
    "mini_batches":         4,
    "learning_epochs":      1,
    "learning_rate":        3e-4,
    "entropy_loss_scale":   1e-3,
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Play / eval                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _run_play(env: IsaacLabEnvWrapper, actor: GaussianActor, N: int, device) -> None:
    from collections import defaultdict

    env_uw        = env.unwrapped
    E             = env.num_envs
    curriculum_on = getattr(env_uw, "_curriculum_enabled", False)
    P             = actor._cond.per_dim

    actor.eval()
    ep_results: list[tuple[int, bool]] = []
    total_done = 0
    target     = args_cli.eval_episodes

    obs_raw, _ = env.reset()
    key = "actor_observations" if isinstance(obs_raw, dict) and "actor_observations" in obs_raw else "policy"

    print(f"\n[Play] running {target} episodes  num_envs={E}  N={N}", flush=True)

    with torch.no_grad():
        while total_done < target:
            obs_flat = (obs_raw[key] if isinstance(obs_raw, dict) else obs_raw)  # (E, N*P)
            obs_per  = obs_flat.reshape(E * N, P)
            mean, std = actor(obs_per, compute_std=True)
            actions  = torch.distributions.Normal(mean, std).sample()
            actions  = torch.tanh(actions).reshape(E, N * 3)
            obs_raw, _, done, _ = env.step(actions)

            for e in done.nonzero(as_tuple=True)[0].tolist():
                n_active = int(env_uw._active_n[e].item()) if curriculum_on else N
                success  = bool(env_uw._just_succeeded[e].item())
                ep_results.append((n_active, success))
                total_done += 1
                if total_done >= target:
                    break

    total_suc = sum(s for _, s in ep_results)
    print(f"\n[Play] {total_suc}/{total_done} = {total_suc/total_done:.1%} success")
    if curriculum_on:
        by_n: dict[int, list[bool]] = defaultdict(list)
        for n_a, suc in ep_results:
            by_n[n_a].append(suc)
        for n_a in sorted(by_n):
            sl = by_n[n_a]
            print(f"  N_active={n_a:2d}: {sum(sl):3d}/{len(sl):3d} = {sum(sl)/len(sl):.1%}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Resume helpers                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _reset_critic_weights(critic: PerDroneCriticNet) -> None:
    for m in critic.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
    enc = critic._encoder
    for norm in (enc.own_norm, enc.neigh_norm):
        norm._mean.zero_()
        norm._var.fill_(1.0)
        norm._std.fill_(1.0)
        norm.count.zero_()
    print("[Resume] Critic weights reset.", flush=True)


def _load_resume(runner: IsaaclabRunner, path: str, device) -> tuple[int, bool]:
    import re
    content       = torch.load(path, map_location=device, weights_only=False)
    critic_reinit = False

    if "model_state_dicts" in content:
        for key in ("actor", "critic"):
            sd = content["model_state_dicts"].get(key)
            if sd is None:
                continue
            try:
                runner._model_load_state_dict(runner._agent.model_dict[key], sd)
                print(f"[Resume] {key} weights loaded.", flush=True)
            except RuntimeError as exc:
                print(f"[Resume][WARN] {key} load failed ({exc}) — using fresh weights.", flush=True)
                if key == "critic":
                    critic_reinit = True
    else:
        runner.load(path)

    iteration = content.get("iteration", 0)
    if iteration == 0:
        m = re.search(r"model_(\d+)\.pt", os.path.basename(path))
        if m:
            iteration = int(m.group(1))

    print(f"[Resume] iter={iteration}  critic_reinit={critic_reinit}", flush=True)
    return iteration, critic_reinit


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Main                                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main() -> None:
    import numpy as np
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # ── env ──────────────────────────────────────────────────────────────────
    env_cfg = MultiDroneDmpcRLEnvCfg()
    env_cfg.scene.num_envs            = args_cli.num_envs
    env_cfg.num_drones                = args_cli.num_drones
    env_cfg.dmpc_guide_enabled        = False   # no DMPC expert, no bottleneck
    env_cfg.terminate_on_collision_rl  = True
    env_cfg.collision_terminate_dist   = args_cli.collision_terminate_dist
    env_cfg.terminate_on_oob_rl        = True
    env_cfg.hover_vel_scale            = args_cli.hover_vel_scale
    env_cfg.hover_proximity_dist       = args_cli.hover_proximity_dist
    env_cfg.dist_to_goal_far_scale     = args_cli.dist_far_scale
    env_cfg.dist_to_goal_far_tanh_scale = args_cli.dist_far_tanh_scale
    if args_cli.n_min > 0:
        env_cfg.n_curriculum_min = args_cli.n_min
    env_cfg.__post_init__()

    env_raw = gym.make("Isaac-MultiDrone-DMPC-RL-Direct-v0", cfg=env_cfg)
    assert isinstance(env_raw.unwrapped, MultiDroneDmpcRLEnv)
    env    = IsaacLabEnvWrapper(env_raw)
    device = env.device

    N             = env_cfg.num_drones
    actor_obs_dim = gym.spaces.flatdim(env.unwrapped.single_observation_space["policy"])
    act_dim       = gym.spaces.flatdim(env.unwrapped.single_action_space)
    assert act_dim == N * 3

    sdf_dim       = PER_DRONE_SDF_DIM if env_cfg.enable_sdf_obs else 0
    own_dim       = PER_DRONE_OWN_DIM + sdf_dim
    per_drone_dim = own_dim + PER_NEIGHBOUR_DIM * (N - 1)
    assert actor_obs_dim == N * per_drone_dim

    print(f"[Env] {args_cli.num_envs} envs × {N} drones  "
          f"per_drone={per_drone_dim} (own={own_dim}, sdf={sdf_dim})  "
          f"DMPC=OFF  actor=Gaussian", flush=True)

    # ── networks ──────────────────────────────────────────────────────────────
    emb_dim     = args_cli.emb_dim
    hidden_dims = args_cli.hidden_dims

    actor  = GaussianActor(
        N=N, own_dim=own_dim, neigh_dim=PER_NEIGHBOUR_DIM,
        emb_dim=emb_dim, hidden_dims=hidden_dims, action_dim=3,
    ).to(device)

    critic = PerDroneCriticNet(
        N=N, own_dim=own_dim, neigh_dim=PER_NEIGHBOUR_DIM,
        emb_dim=emb_dim, hidden_dims=hidden_dims,
    ).to(device)

    # ── IPPO env wrapper ──────────────────────────────────────────────────────
    env_train = PerDroneEnvWrapper(env, N=N, per_drone_dim=per_drone_dim)
    rl_n_envs = env.num_envs * N
    print(f"[IPPO] {env.num_envs}×{N} = {rl_n_envs} virtual envs  "
          f"obs={per_drone_dim}  act=3", flush=True)

    # ── agent ─────────────────────────────────────────────────────────────────
    replay_buffer = ReplayBuffer(
        memory_size=args_cli.rollouts,
        num_envs=rl_n_envs,
        device=device,
    )
    agent = PPO(
        models={"actor": actor, "critic": critic},
        replay_buffer=replay_buffer,
        device=device,
        cfg=AGENT_CFG,
    )
    agent.init_replay_buffer(
        critic_observation_size=per_drone_dim,
        actor_observation_size=per_drone_dim,
        action_size=3,
    )

    # ── runner ────────────────────────────────────────────────────────────────
    run_name   = args_cli.experiment_name or datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    runner_cfg = {
        "max_iterations":           args_cli.max_iterations,
        "rollouts":                 args_cli.rollouts,
        "save_interval":            args_cli.save_interval,
        "log_dir":                  args_cli.log_dir,
        "experiment_name":          run_name,
        "wandb_project":            args_cli.wandb_project if args_cli.wandb else "",
        "update_model_normalizers": True,
    }

    if args_cli.wandb and _WANDB_AVAILABLE:
        _wandb.init(
            project=args_cli.wandb_project,
            name=run_name,
            config=vars(args_cli),
            reinit=False,
        )

    runner = IsaaclabRunner(env=env_train, agent=agent, cfg=runner_cfg)

    # ── resume ────────────────────────────────────────────────────────────────
    start_iteration = 0
    if args_cli.resume:
        start_iteration, critic_reinit = _load_resume(runner, args_cli.resume, device)

        if args_cli.play:
            _run_play(env=env, actor=actor, N=N, device=device)
            env.close()
            simulation_app.close()
            return

        if args_cli.reset_critic or critic_reinit:
            _reset_critic_weights(critic)

    elif args_cli.play:
        print("[Play] --play requires --resume. Exiting.", flush=True)
        env.close()
        simulation_app.close()
        return

    # ── training ──────────────────────────────────────────────────────────────
    runner.train(start_iteration=start_iteration, return_epochs=args_cli.max_iterations)

    if args_cli.wandb and _WANDB_AVAILABLE:
        _wandb.finish()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
