"""test_bc_dmpc.py - Visualise and record a trained multi-drone flow-matching
policy (or the DMPC expert for baseline comparison).

Usage examples::

    # Student policy from checkpoint
    python test_bc_dmpc.py --checkpoint runs/online_bc_dmpc/model.pt

    # Expert baseline (no checkpoint needed)
    python test_bc_dmpc.py --mode expert

    # Student with custom camera / resolution
    python test_bc_dmpc.py \\
        --checkpoint runs/online_bc_dmpc/model.pt \\
        --num_envs 4 --num_drones 4 \\
        --num_episodes 20 \\
        --video_out runs/test_videos \\
        --cam_eye 10 10 8 --cam_lookat 0 0 1 \\
        --resolution 1920 1080

The script runs enough env steps to complete ``--num_episodes`` complete
episodes (across all envs), then prints per-episode and aggregate statistics
(success rate, mean return, mean final distance to goal) and closes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_POLICYFLOW_ROOT = _HERE.parent.parent / "policyflow"
_PROJECT_ROOT = _HERE.parent.parent.parent
for _p in (_POLICYFLOW_ROOT, _PROJECT_ROOT):
    p_str = str(_p)
    if _p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Test / visualise a trained multi-drone policy.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a model.pt saved by online_bc_dmpc.py. "
                         "Required for --mode student (default).")
parser.add_argument("--mode", choices=["student", "expert"], default="student",
                    help="'student' runs the loaded policy; 'expert' runs the DMPC baseline.")

parser.add_argument("--num_envs", type=int, default=4,
                    help="Number of parallel environments (more = faster episode collection).")
parser.add_argument("--num_drones", type=int, default=4)
parser.add_argument("--task", type=str, default="Isaac-MultiDrone-DMPC-Direct-v0")
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--num_episodes", type=int, default=16,
                    help="Total complete episodes to collect across all envs before exiting.")
parser.add_argument("--max_steps", type=int, default=0,
                    help="Hard step cap (0 = unlimited; will stop at num_episodes first).")
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="Override env episode length in seconds.")
parser.add_argument("--no_terminate_on_bounds", action="store_true", default=False)

# Model architecture (must match checkpoint if loading student).
parser.add_argument("--hidden_dims", type=int, nargs="*", default=[256, 256, 256])
parser.add_argument("--emb_dim", type=int, default=64)
parser.add_argument("--sample_steps", type=int, default=10)
parser.add_argument("--action_clip", type=float, default=0.999,
                    help="Must match the value used during training.")

# Video.
parser.add_argument("--video_out", type=str, default="runs/test_videos",
                    help="Directory for output videos.")
parser.add_argument("--video_length", type=int, default=0,
                    help="Frames per video clip (0 = full rollout until first done).")
parser.add_argument("--no_video", action="store_true", default=False,
                    help="Disable video recording (viewer only).")
parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720],
                    metavar=("W", "H"))
parser.add_argument("--cam_eye", type=float, nargs=3, default=[8.0, 8.0, 6.0],
                    metavar=("X", "Y", "Z"))
parser.add_argument("--cam_lookat", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                    metavar=("X", "Y", "Z"))

# Alignment diagnostics.
parser.add_argument("--diag_alignment", action="store_true", default=False,
                    help="Print goal-action cosine similarity diagnostics every "
                         "--diag_interval steps. Student mode only.")
parser.add_argument("--diag_interval", type=int, default=10,
                    help="Print alignment diagnostics every N steps (default 10). "
                         "Only active with --diag_alignment.")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ──────────────────────────────────────────────────────
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import quadcopter  # noqa: F401, E402
from quadcopter.multi_drone_dmpc_env import (  # noqa: E402
    MultiDroneDmpcEnv,
    MultiDroneDmpcEnvCfg,
    per_drone_obs_dim,
    PER_DRONE_OWN_DIM,
    PER_NEIGHBOUR_DIM,
)

from policyflow_torch.modules import (  # noqa: E402
    ContinuousNormalizingFlow,
    ConditionMlp,
    FlowMlp,
    NeighborEncoder,
)
from policyflow_torch.modules.normalizer import EmpiricalNormalization  # noqa: E402

PER_DRONE_ACTION_DIM = 3


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Policy (copy from online_bc_dmpc.py — must stay in sync)                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class SharedDronePolicy(nn.Module):
    def __init__(
        self,
        per_drone_obs_dim: int,
        per_drone_action_dim: int,
        hidden_dims: list[int],
        emb_dim: int,
        sample_steps: int,
        device: torch.device,
        own_dim: int = PER_DRONE_OWN_DIM,
        neighbor_dim: int = PER_NEIGHBOUR_DIM,
        num_attn_heads: int = 4,
        action_clip: float = 0.999,
    ):
        super().__init__()
        self.P = per_drone_obs_dim
        self.A = per_drone_action_dim
        self.own_dim = own_dim
        self.neighbor_dim = neighbor_dim
        self.device = device
        self.action_clip = action_clip

        self.own_norm = EmpiricalNormalization(shape=own_dim, until=int(1e8))
        self.neigh_norm = EmpiricalNormalization(shape=neighbor_dim, until=int(1e8))

        self.neighbor_enc = NeighborEncoder(
            own_dim=own_dim,
            neighbor_dim=neighbor_dim,
            emb_dim=emb_dim,
            num_heads=num_attn_heads,
        )

        cond_input_dim = own_dim + emb_dim
        nn_condition = ConditionMlp(
            cond_dim=cond_input_dim,
            emb_dim=emb_dim,
            activations=["elu"] * len(hidden_dims) + ["linear"],
            hidden_dims=hidden_dims,
        )
        nn_flow = FlowMlp(
            x_dim=self.A,
            emb_dim=emb_dim,
            activations=["elu"] * len(hidden_dims) + ["linear"],
            hidden_dims=hidden_dims,
        )
        self.cnf = ContinuousNormalizingFlow(
            x_dims=self.A,
            nn_flow=nn_flow,
            nn_condition=nn_condition,
            sample_steps=sample_steps,
            sample_step_schedule="uniform_continuous",
            interpolation_type="rectified_flow",
            device=device,
        )
        self.cnf_model = self.cnf.model
        self.cnf_ema = self.cnf.model_ema
        self.cnf_last = self.cnf.model_last
        self.cnf.init_proximal(beta_prox=0.97)

    def _encode_obs(self, obs_flat: torch.Tensor) -> torch.Tensor:
        B = obs_flat.shape[0]
        own_raw = obs_flat[:, : self.own_dim]
        neigh_raw = obs_flat[:, self.own_dim :]
        own_n = self.own_norm(own_raw)
        N_neigh = neigh_raw.shape[1] // self.neighbor_dim
        if N_neigh > 0:
            neigh_2d = neigh_raw.reshape(B * N_neigh, self.neighbor_dim)
            neigh_n = self.neigh_norm(neigh_2d).reshape(B, N_neigh, self.neighbor_dim)
        else:
            neigh_n = torch.zeros(B, 0, self.neighbor_dim, device=obs_flat.device)
        neighbor_embed = self.neighbor_enc(own_n, neigh_n)
        return torch.cat([own_n, neighbor_embed], dim=-1)

    @torch.no_grad()
    def sample_action(self, per_drone_obs: torch.Tensor) -> torch.Tensor:
        """(E, N, P) → (E, N, A) action decoded as tanh(latent/clip)*clip."""
        E, N, P = per_drone_obs.shape
        flat = per_drone_obs.reshape(E * N, P)
        self.cnf.eval()
        cond_in = self._encode_obs(flat)
        x0 = torch.randn(cond_in.shape[0], self.A, device=cond_in.device)
        latent, _ = self.cnf.sample(x0=x0, condition=cond_in, n_samples=cond_in.shape[0])
        action = torch.tanh(latent / self.action_clip) * self.action_clip
        self.cnf.train()
        return action.reshape(E, N, self.A)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Expert helpers (needed for --mode expert)                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _expert_action(env: MultiDroneDmpcEnv, expert) -> torch.Tensor:
    """Run DMPC with proper pos/vel/acc references — identical to training."""
    states = env.get_world_states()
    ref_pos_w, ref_vel_w = expert.plan(
        pos_w=states["pos_w"],
        vel_w=states["lin_vel_w"],
        goal_w=states["goal_w"],
        env_origins=env._terrain.env_origins,
    )
    # Compute acceleration from Bezier 2nd derivative (same as training).
    ref_acc_w = torch.zeros_like(ref_pos_w)
    h_total = (expert.p.k_hor - 1) * expert.p.h
    for e in range(env.num_envs):
        for i in range(env.cfg.num_drones):
            st = expert._state.get((e, i))
            if st is None:
                continue
            steps_before = max(int(st["steps"]) - 1, 0)
            t_sub = ((steps_before % expert.n_substeps) + 1) * expert.p.ts
            t_sub = min(t_sub, h_total)
            acc = expert.bezier.sample_matrix(np.array([t_sub]), deriv=2) @ st["U"]
            ref_acc_w[e, i] = torch.from_numpy(acc.astype(np.float32)).to(env.device)
    return env.ref_to_action(ref_pos_w, ref_vel_w, ref_acc_w)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Statistics accumulator                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class EpisodeStats:
    def __init__(self):
        self.returns: list[float] = []
        self.success: list[bool] = []       # True = success termination
        self.final_dist: list[float] = []   # mean distance of all drones to goal

    def add(self, ret: float, succeeded: bool, dist: float):
        self.returns.append(ret)
        self.success.append(succeeded)
        self.final_dist.append(dist)

    def n(self) -> int:
        return len(self.returns)

    def summary(self) -> str:
        if not self.returns:
            return "(no completed episodes)"
        n = len(self.returns)
        sr = sum(self.success) / n
        mr = float(np.mean(self.returns))
        md = float(np.mean(self.final_dist))
        return (
            f"episodes={n}  success_rate={sr:.3f}  "
            f"mean_return={mr:.2f}  mean_final_dist={md:.3f}m"
        )


def _diag_policy(policy: SharedDronePolicy) -> None:
    """Print checkpoint diagnostics to catch common failure modes."""
    print("[diag] ── checkpoint diagnostics ──────────────────────────────")

    # 1. Obs normaliser stats (EmpiricalNormalization uses _mean / _std buffers).
    #    count=0 → normaliser never updated; std=1 everywhere → default init only.
    #    std≈0 in some dims → (obs-mean)/std blows up.
    own_mean  = policy.own_norm.mean    # property: _mean.squeeze(0)
    own_std   = policy.own_norm.std     # property: _std.squeeze(0)
    neigh_mean = policy.neigh_norm.mean
    neigh_std  = policy.neigh_norm.std
    n_own_seen   = int(policy.own_norm.count.item())
    n_neigh_seen = int(policy.neigh_norm.count.item())
    print(f"[diag]   own_norm   count={n_own_seen}  "
          f"mean=[{own_mean.min():.3f},{own_mean.max():.3f}]  "
          f"std=[{own_std.min():.3f},{own_std.max():.3f}]")
    print(f"[diag]   neigh_norm count={n_neigh_seen}  "
          f"mean=[{neigh_mean.min():.3f},{neigh_mean.max():.3f}]  "
          f"std=[{neigh_std.min():.3f},{neigh_std.max():.3f}]")
    if n_own_seen == 0:
        print("[diag]   WARNING: own_norm never updated (count=0) → "
              "normaliser not warmed up. Checkpoint saved before any obs.")
    elif own_std.max() > 0.99 and own_std.min() > 0.99:
        print("[diag]   WARNING: own_norm std ≈ 1 everywhere → "
              "normaliser barely updated (very early checkpoint).")
    if own_std.min() < 1e-3:
        print("[diag]   WARNING: own_norm std ≈ 0 in some dims → obs/std = inf.")

    # 2. Flow weights magnitude: if all near-zero, network was never trained.
    all_params = torch.cat([p.detach().flatten() for p in policy.cnf_model.parameters()])
    print(f"[diag]   flow weight norm={all_params.norm():.3f}  "
          f"abs_mean={all_params.abs().mean():.5f}  "
          f"max={all_params.abs().max():.3f}")
    if all_params.abs().mean() < 1e-4:
        print("[diag]   WARNING: flow weights near zero → network not trained.")

    print("[diag] ────────────────────────────────────────────────────────")


def _print_alignment_diag(
    per_drone_obs: torch.Tensor,  # (E, N, P) raw (unnormalised) per-drone obs
    action_pd: torch.Tensor,      # (E, N, 3) decoded v_ref in (-1, 1)
    step: int,
    prev_goal_dist: torch.Tensor | None = None,  # (N,) goal_dist from last diag call
) -> torch.Tensor:
    """Print per-drone goal-direction vs v_ref alignment for env 0.

    Obs layout (own dims 0-18):
      [0:3]   lin_vel_b
      [3:6]   ang_vel_b
      [6:9]   proj_gravity_b
      [9:12]  goal_b (body frame, raw metres)
      [12:15] prev_v_ref_b
      [15:19] time_emb

    Action: v_ref (A=3, normalised [-1, 1]).

    Interpretation guide:
      cos(goal, v_ref) > 0.7   v_ref pointing toward goal          ✓
      cos(goal, v_ref) ≈ 0     v_ref ignoring goal direction
      cos(goal, v_ref) < 0     v_ref pointing away from goal       ✗
      Δdist < 0 each step      drone is approaching goal            ✓
    """
    obs_e0 = per_drone_obs[0]   # (N, P)
    act_e0 = action_pd[0]       # (N, 3)

    goal_b = obs_e0[:, 9:12]    # (N, 3)
    v_ref  = act_e0[:, 0:3]     # (N, 3) normalised

    goal_dist = goal_b.norm(dim=-1)                         # (N,) metres
    cos_vr    = F.cosine_similarity(goal_b, v_ref, dim=-1)  # (N,)
    vr_mag    = v_ref.norm(dim=-1)                          # (N,)

    N_drones = obs_e0.shape[0]
    lines = [f"  [align step {step:>5d}]  (env 0 only)"]
    for i in range(N_drones):
        if prev_goal_dist is not None:
            ddist = goal_dist[i].item() - prev_goal_dist[i].item()
            ddist_str = f"  Δdist={ddist:+.3f}m"
        else:
            ddist_str = ""
        lines.append(
            f"    drone {i}: "
            f"goal_dist={goal_dist[i]:.2f}m{ddist_str}  "
            f"cos(goal,vref)={cos_vr[i]:+.3f}  "
            f"|vref|={vr_mag[i]:.3f}"
        )
    if prev_goal_dist is not None:
        ddist_mean = (goal_dist - prev_goal_dist).mean().item()
        ddist_mean_str = f"  Δdist={ddist_mean:+.3f}m"
    else:
        ddist_mean_str = ""
    lines.append(
        f"    mean :  "
        f"cos(goal,vref)={cos_vr.mean():+.3f}{ddist_mean_str}  "
        f"|vref|={vr_mag.mean():.3f}"
    )
    print("\n".join(lines), flush=True)
    return goal_dist.detach()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Main                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # ── validate args ────────────────────────────────────────────────────────
    if args_cli.mode == "student" and args_cli.checkpoint is None:
        parser.error("--checkpoint is required for --mode student")

    record_video = not args_cli.no_video

    # ── env setup ────────────────────────────────────────────────────────────
    env_cfg = MultiDroneDmpcEnvCfg()
    env_cfg.num_drones = args_cli.num_drones
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.no_terminate_on_bounds:
        env_cfg.terminate_on_bounds = False
    env_cfg.__post_init__()
    if getattr(args_cli, "device", None):
        env_cfg.sim.device = args_cli.device

    if record_video:
        env_cfg.viewer.eye = tuple(args_cli.cam_eye)
        env_cfg.viewer.lookat = tuple(args_cli.cam_lookat)
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.resolution = tuple(args_cli.resolution)

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if record_video else None,
    )

    if record_video:
        os.makedirs(args_cli.video_out, exist_ok=True)
        vid_len = args_cli.video_length if args_cli.video_length > 0 else int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.video_out,
            episode_trigger=lambda ep: True,   # record every episode
            video_length=vid_len,
            disable_logger=True,
            name_prefix=f"test_{args_cli.mode}",
        )
        print(f"[test] Recording videos → {args_cli.video_out}/  (clip length={vid_len} steps)")

    env_raw: MultiDroneDmpcEnv = env.unwrapped
    device = env_raw.device
    N = env_raw.cfg.num_drones
    P = per_drone_obs_dim(N)
    A = PER_DRONE_ACTION_DIM

    print(
        f"[test] mode={args_cli.mode}  num_envs={env_raw.num_envs}  num_drones={N}  "
        f"per_drone_obs={P}  per_drone_action={A}"
    )

    # ── policy / expert setup ────────────────────────────────────────────────
    expert = None
    policy = None

    if args_cli.mode == "student":
        policy = SharedDronePolicy(
            per_drone_obs_dim=P,
            per_drone_action_dim=A,
            hidden_dims=args_cli.hidden_dims,
            emb_dim=args_cli.emb_dim,
            sample_steps=args_cli.sample_steps,
            device=device,
            action_clip=args_cli.action_clip,
        ).to(device)
        ckpt = torch.load(args_cli.checkpoint, map_location=device)
        policy.load_state_dict(ckpt)
        policy.cnf.eval()
        print(f"[test] loaded checkpoint: {args_cli.checkpoint}")
        _diag_policy(policy)

    else:  # expert
        from quadcopter.dmpc_expert import DMPCExpert, DMPCParams  # noqa: E402
        expert = DMPCExpert(
            num_drones=N,
            params=DMPCParams(
                pmin=env_raw.cfg.pos_min,
                pmax=env_raw.cfg.pos_max,
                rmin=env_raw.cfg.rmin,
                ts=env_raw.cfg.sim.dt * env_raw.cfg.decimation,
            ),
            device=device,
        )
        print("[test] using DMPC expert baseline")

    # ── rollout ──────────────────────────────────────────────────────────────
    env.reset(seed=args_cli.seed)
    if expert is not None:
        expert.reset()
    last_episode_length_buf = env_raw.episode_length_buf.clone()

    returns = torch.zeros(env_raw.num_envs, device=device)
    stats = EpisodeStats()
    step = 0
    max_steps = args_cli.max_steps if args_cli.max_steps > 0 else int(1e9)
    _prev_goal_dist: torch.Tensor | None = None  # for Δdist tracking in alignment diag

    print(f"[test] running until {args_cli.num_episodes} episodes complete …")

    while stats.n() < args_cli.num_episodes and step < max_steps:
        with torch.no_grad():
            if args_cli.mode == "student":
                per_drone_obs = env_raw.get_per_drone_obs()          # (E, N, P)
                action_pd = policy.sample_action(per_drone_obs)       # (E, N, A)
                action_flat = action_pd.reshape(env_raw.num_envs, N * A)
                # Periodically print action stats to detect saturation / randomness.
                if step % 50 == 0:
                    am = action_pd.abs().mean().item()
                    amax = action_pd.abs().max().item()
                    print(f"  [step {step:>5d}] action abs_mean={am:.3f}  abs_max={amax:.3f}"
                          f"  (random≈0.5, expert≈0.1–0.3)", flush=True)
                if args_cli.diag_alignment and step % args_cli.diag_interval == 0:
                    _prev_goal_dist = _print_alignment_diag(
                        per_drone_obs, action_pd, step, _prev_goal_dist
                    )
            else:
                # Keep expert in sync with env resets.
                rewound = env_raw.episode_length_buf < last_episode_length_buf
                if rewound.any():
                    expert.reset(rewound.nonzero(as_tuple=False).flatten())
                action_flat = _expert_action(env_raw, expert)

        last_episode_length_buf = env_raw.episode_length_buf.clone()

        _, reward, terminated, truncated, _ = env.step(action_flat)
        returns += reward
        done = terminated | truncated

        # Expert: sync on explicit resets triggered inside env.step().
        if expert is not None:
            reset_env_ids = getattr(env_raw, "_last_reset_env_ids", None)
            if reset_env_ids is not None and reset_env_ids.numel() > 0:
                expert.reset(reset_env_ids)
                env_raw._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)
            if done.any():
                expert.reset(done.nonzero(as_tuple=False).flatten())

        if done.any():
            done_ids = done.nonzero(as_tuple=False).flatten()
            pos_w = torch.stack([r.data.root_pos_w for r in env_raw._robots], dim=1)
            for env_id in done_ids:
                dists = torch.linalg.norm(
                    pos_w[env_id] - env_raw._goal_pos_w[env_id], dim=-1
                )  # (N,)
                mean_dist = dists.mean().item()
                succeeded = bool(env_raw._just_succeeded[env_id].item())
                ret = returns[env_id].item()
                stats.add(ret, succeeded, mean_dist)
                print(
                    f"  ep {stats.n():>4d}  env={int(env_id):>2d}  "
                    f"ret={ret:>8.2f}  success={succeeded}  "
                    f"dist={mean_dist:.3f}m",
                    flush=True,
                )
            returns[done] = 0.0
            # Reset Δdist baseline when env 0 resets so the next diag doesn't
            # show a spurious large jump from post-reset to new start position.
            if args_cli.diag_alignment and (done[0] if done.numel() > 0 else False):
                _prev_goal_dist = None

        step += 1

    # ── final summary ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"[test] RESULTS ({args_cli.mode})")
    print(stats.summary())
    if record_video:
        print(f"[test] videos saved to: {args_cli.video_out}/")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
