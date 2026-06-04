"""collect_dmpc_data.py — Offline DMPC expert data collection.

Runs the DMPC expert across many IsaacLab environments in parallel and saves
rich per-agent .npz datasets for offline / decoupled policy training.
Each file contains one agent's trajectory from one episode — success is judged
per-agent using env._drone_just_succeeded rather than env-level success.
No learning happens here.

Each saved file (one per agent per episode) contains:

  Per Bezier window  (T = windows in episode):
    obs              (T, P)       this agent's obs at window start
    pos_w            (T, 3)       world position
    vel_w            (T, 3)       world velocity
    quat_w           (T, 4)       body→world quaternion
    ang_vel_b        (T, 3)       body-frame angular velocity
    goal_w           (T, 3)       goal world position
    R_goal           (T, 9)       goal-aligned rotation (world→goal), row-major
    bezier_U         (T, n_bez)   DMPC Bezier control vector (env-local frame)
    bezier_U_goal    (T, n_bez)   DMPC ctrl pts, goal-aligned relative frame (policy target)
    ref_pos_w        (T, 3)       pos ref at first sub-step
    ref_vel_w        (T, 3)       vel ref
    ref_acc_w        (T, 3)       acc ref
    mpc_replanned    (T,)         bool — QP was solved this window
    mpc_reset_mode   (T,)         bool
    mpc_fallback     (T,)         bool
    mpc_collision_ts (T,)         int  — first collision substep (-1 = none)

  Per substep within window  (S = n_substeps sub-steps):
    substep_ref_pos_w  (T, S, 3)
    substep_ref_vel_w  (T, S, 3)
    substep_ref_acc_w  (T, S, 3)
    substep_thrust_z   (T, S)     body-frame thrust z [N]
    substep_torque_b   (T, S, 3)  body-frame torques [N·m]

  Episode metadata (scalars / small arrays):
    init_pos_local  (3,)   drone start pos: xy relative to env origin, z absolute
    goal_local      (3,)   goal: xy relative to env origin, z absolute
    success         bool
    n_substeps      int
    bezier_k        int
    h_window        float32
    n_bez           int

Usage::

    python collect_dmpc_data.py \\
        --num_envs 64 --num_drones 4 \\
        --save_dir runs/dmpc_data \\
        --target_episodes 2000

    # save all episodes (not just successful)
    python collect_dmpc_data.py ... --save_all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# ── path setup ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_POLICYFLOW_ROOT = _HERE.parent.parent / "policyflow"
_PROJECT_ROOT = _HERE.parent.parent.parent
for _p in (_POLICYFLOW_ROOT, _PROJECT_ROOT):
    _s = str(_p)
    if _p.exists() and _s not in sys.path:
        sys.path.insert(0, _s)

# ── argument parser (must precede AppLauncher) ────────────────────────────
parser = argparse.ArgumentParser(description="Collect DMPC expert data for offline training.")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--num_drones", type=int, default=10)
parser.add_argument("--task", type=str, default="Isaac-MultiDrone-DMPC-Direct-v0")
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--save_dir", type=str, default=None,
                    help="Directory to write per-episode .npz files. "
                         "Optional when --test_only is set.")
parser.add_argument("--target_episodes", type=int, default=500000,
                    help="Stop after this many episodes saved (normal mode) "
                         "or completed (--test_only mode).")
parser.add_argument("--max_steps", type=int, default=0,
                    help="Hard env-step cap (0 = use --target_episodes only).")
parser.add_argument("--save_all", action="store_true", default=False,
                    help="Save all episodes, not just successful ones.")
parser.add_argument("--min_safe_dist", type=float, default=0.08,
                    help="Filter episodes where any drone pair came closer than this [m]. "
                         "Defaults to 0.08m (~Crazyflie body diameter). Use 0 to disable.")

parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--no_terminate_on_bounds", action="store_true", default=False)

# N curriculum (inactive drones)
parser.add_argument("--n_min", type=int, default=0,
                    help="Enable N-curriculum: env e uses (e %% N)+1 active drones. "
                         "Uses MultiDroneDmpcRLEnv so inactive drones are parked. "
                         "0 = disabled (all envs use --num_drones).")
parser.add_argument("--test_only", action="store_true", default=False,
                    help="Run expert and print success stats without saving .npz files. "
                         "Stops after --target_episodes total completed episodes.")

# GPU ADMM
parser.add_argument("--gpu_admm", action="store_true", default=False,
                    help="Use batched GPU ADMM instead of CPU OSQP threads.")
parser.add_argument("--admm_iters", type=int, default=50,
                    help="ADMM iterations per window (default 50).")
parser.add_argument("--gpu_expert", action="store_true", default=False,
                    help="Use GPUDMPCExpert (colleague's solver) instead of DMPCExpert+GPU-ADMM.")

# Video recording
parser.add_argument("--video", action="store_true", default=False,
                    help="Record video clips during collection.")
parser.add_argument("--video_length", type=int, default=70,
                    help="Frames per video clip.")
parser.add_argument("--video_interval", type=int, default=210,
                    help="Record a new clip every this many env steps.")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Enable cameras only when video recording is requested.
args_cli.enable_cameras = args_cli.video

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ──────────────────────────────────────────────────────
import gymnasium as gym          # noqa: E402
import torch                      # noqa: E402

import quadcopter                 # noqa: F401, E402  — registers the gym env
from quadcopter.multi_drone_dmpc_env import (  # noqa: E402
    MultiDroneDmpcEnv,
    MultiDroneDmpcEnvCfg,
    per_drone_obs_dim,
)
from quadcopter.multi_drone_dmpc_rl_env import (  # noqa: E402
    MultiDroneDmpcRLEnv,
    MultiDroneDmpcRLEnvCfg,
)
from quadcopter.dmpc_expert import DMPCExpert, DMPCParams  # noqa: E402
from quadcopter.dmpc_gpu_expert import GPUDMPCExpert       # noqa: E402


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Per-env episode collector                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class EpisodeCollector:
    """Accumulates Bezier-window data for one env and saves to a .npz file.

    Layout:
    - ``begin_window()``  — called once per Bezier window (at window start)
    - ``record_substep()``— called once per env step (n_substeps times per window)
    - ``save()``          — called at episode done to write the file
    - ``reset()``         — discard in-progress data (e.g. on mid-episode reset)
    """

    def __init__(self, env_idx: int, N: int, P: int, n_bez: int, n_substeps: int):
        self.env_idx = env_idx
        self.N = N
        self.P = P
        self.n_bez = n_bez
        self.n_substeps = n_substeps

        self._wins: list[dict[str, Any]] = []   # completed windows
        self._curr: dict[str, Any] | None = None  # window being filled
        self._sub_bufs: list[dict[str, np.ndarray]] = []  # substep accumulator

    # ── window lifecycle ────────────────────────────────────────────────────
    def begin_window(
        self,
        obs: "np.ndarray",                # (N, P)
        pos_w: "np.ndarray",              # (N, 3)
        vel_w: "np.ndarray",
        quat_w: "np.ndarray",             # (N, 4)
        ang_vel_b: "np.ndarray",
        goal_w: "np.ndarray",
        R_goal: "np.ndarray",             # (N, 9) row-major
        bezier_U: "np.ndarray",           # (N, n_bez)
        bezier_U_goal: "np.ndarray",      # (N, n_bez) goal-aligned relative frame
        ref_pos_w: "np.ndarray",          # (N, 3) — first sub-step ref
        ref_vel_w: "np.ndarray",
        ref_acc_w: "np.ndarray",
        mpc_replanned: "np.ndarray",      # (N,) bool
        mpc_reset_mode: "np.ndarray",
        mpc_fallback: "np.ndarray",
        mpc_collision_ts: "np.ndarray",   # (N,) int
    ) -> None:
        if self._curr is not None:
            self._close_window()
        self._curr = dict(
            obs=obs, pos_w=pos_w, vel_w=vel_w, quat_w=quat_w,
            ang_vel_b=ang_vel_b, goal_w=goal_w, R_goal=R_goal,
            bezier_U=bezier_U, bezier_U_goal=bezier_U_goal,
            ref_pos_w=ref_pos_w, ref_vel_w=ref_vel_w, ref_acc_w=ref_acc_w,
            mpc_replanned=mpc_replanned, mpc_reset_mode=mpc_reset_mode,
            mpc_fallback=mpc_fallback, mpc_collision_ts=mpc_collision_ts,
        )
        self._sub_bufs = []

    def record_substep(
        self,
        ref_pos: "np.ndarray",   # (N, 3)
        ref_vel: "np.ndarray",
        ref_acc: "np.ndarray",
        thrust_z: "np.ndarray",  # (N,)
        torque_b: "np.ndarray",  # (N, 3)
    ) -> None:
        self._sub_bufs.append(dict(
            ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc,
            thrust_z=thrust_z, torque_b=torque_b,
        ))
        if len(self._sub_bufs) >= self.n_substeps:
            self._close_window()

    def _close_window(self) -> None:
        if self._curr is None:
            return
        subs = self._sub_bufs
        if subs:
            # Pad to n_substeps if the episode ended mid-window.
            while len(subs) < self.n_substeps:
                subs.append(subs[-1].copy())
            self._curr["substep_ref_pos_w"] = np.stack([s["ref_pos"] for s in subs])  # (S, N, 3)
            self._curr["substep_ref_vel_w"] = np.stack([s["ref_vel"] for s in subs])
            self._curr["substep_ref_acc_w"] = np.stack([s["ref_acc"] for s in subs])
            self._curr["substep_thrust_z"]  = np.stack([s["thrust_z"] for s in subs])  # (S, N)
            self._curr["substep_torque_b"]  = np.stack([s["torque_b"] for s in subs])  # (S, N, 3)
        else:
            S, N = self.n_substeps, self.N
            self._curr["substep_ref_pos_w"] = np.zeros((S, N, 3), np.float32)
            self._curr["substep_ref_vel_w"] = np.zeros((S, N, 3), np.float32)
            self._curr["substep_ref_acc_w"] = np.zeros((S, N, 3), np.float32)
            self._curr["substep_thrust_z"]  = np.zeros((S, N), np.float32)
            self._curr["substep_torque_b"]  = np.zeros((S, N, 3), np.float32)
        self._wins.append(self._curr)
        self._curr = None
        self._sub_bufs = []

    # ── save / reset ────────────────────────────────────────────────────────
    def save(
        self,
        path: str,
        init_pos_local: "np.ndarray",  # (N, 3)
        goal_local: "np.ndarray",       # (N, 3)
        success: bool,
        h_window: float,
    ) -> bool:
        """Stack accumulated windows and write a compressed .npz file.

        Returns True if saved, False if there was nothing to save.
        """
        if self._curr is not None:
            self._close_window()
        T = len(self._wins)
        if T == 0:
            return False

        def _stack(key: str) -> np.ndarray:
            return np.stack([w[key] for w in self._wins], axis=0)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            # Per-window state (T, N, ...)
            obs              = _stack("obs"),
            pos_w            = _stack("pos_w"),
            vel_w            = _stack("vel_w"),
            quat_w           = _stack("quat_w"),
            ang_vel_b        = _stack("ang_vel_b"),
            goal_w           = _stack("goal_w"),
            R_goal           = _stack("R_goal"),
            # Per-window expert outputs
            bezier_U         = _stack("bezier_U"),
            bezier_U_goal    = _stack("bezier_U_goal"),
            ref_pos_w        = _stack("ref_pos_w"),
            ref_vel_w        = _stack("ref_vel_w"),
            ref_acc_w        = _stack("ref_acc_w"),
            mpc_replanned    = _stack("mpc_replanned"),
            mpc_reset_mode   = _stack("mpc_reset_mode"),
            mpc_fallback     = _stack("mpc_fallback"),
            mpc_collision_ts = _stack("mpc_collision_ts"),
            # Per-substep (T, S, N, ...)
            substep_ref_pos_w  = _stack("substep_ref_pos_w"),
            substep_ref_vel_w  = _stack("substep_ref_vel_w"),
            substep_ref_acc_w  = _stack("substep_ref_acc_w"),
            substep_thrust_z   = _stack("substep_thrust_z"),
            substep_torque_b   = _stack("substep_torque_b"),
            # Episode metadata
            init_pos_local  = init_pos_local,
            goal_local      = goal_local,
            success         = np.array(success,           dtype=np.bool_),
            n_substeps      = np.array(self.n_substeps,   dtype=np.int32),
            bezier_k        = np.array(self.n_bez,         dtype=np.int32),  # kept for compat
            h_window        = np.array(h_window,          dtype=np.float32),
            n_bez           = np.array(self.n_bez,         dtype=np.int32),
        )
        return True

    def save_agent(
        self,
        path: str,
        agent_idx: int,
        init_pos_agent: "np.ndarray",  # (3,)
        goal_agent: "np.ndarray",       # (3,)
        success: bool,
        h_window: float,
    ) -> bool:
        """Save a single agent's trajectory sliced from the accumulated windows.

        Returns True if saved, False if there was nothing to save.
        """
        if self._curr is not None:
            self._close_window()
        T = len(self._wins)
        if T == 0:
            return False

        i = agent_idx

        def _sa(key: str) -> np.ndarray:
            # window arrays have shape (N, ...) → stack → (T, N, ...) → [:,i] → (T, ...)
            return np.stack([w[key] for w in self._wins], axis=0)[:, i]

        def _sa_sub(key: str) -> np.ndarray:
            # substep arrays have shape (S, N, ...) → stack → (T, S, N, ...) → [:,:,i]
            return np.stack([w[key] for w in self._wins], axis=0)[:, :, i]

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            # Per-window state (T, ...)
            obs              = _sa("obs"),
            pos_w            = _sa("pos_w"),
            vel_w            = _sa("vel_w"),
            quat_w           = _sa("quat_w"),
            ang_vel_b        = _sa("ang_vel_b"),
            goal_w           = _sa("goal_w"),
            R_goal           = _sa("R_goal"),
            bezier_U         = _sa("bezier_U"),
            bezier_U_goal    = _sa("bezier_U_goal"),
            ref_pos_w        = _sa("ref_pos_w"),
            ref_vel_w        = _sa("ref_vel_w"),
            ref_acc_w        = _sa("ref_acc_w"),
            mpc_replanned    = _sa("mpc_replanned"),
            mpc_reset_mode   = _sa("mpc_reset_mode"),
            mpc_fallback     = _sa("mpc_fallback"),
            mpc_collision_ts = _sa("mpc_collision_ts"),
            # Per-substep (T, S, ...)
            substep_ref_pos_w  = _sa_sub("substep_ref_pos_w"),
            substep_ref_vel_w  = _sa_sub("substep_ref_vel_w"),
            substep_ref_acc_w  = _sa_sub("substep_ref_acc_w"),
            substep_thrust_z   = _sa_sub("substep_thrust_z"),
            substep_torque_b   = _sa_sub("substep_torque_b"),
            # Episode metadata
            init_pos_local  = init_pos_agent,
            goal_local      = goal_agent,
            success         = np.array(success,         dtype=np.bool_),
            n_substeps      = np.array(self.n_substeps, dtype=np.int32),
            bezier_k        = np.array(self.n_bez,       dtype=np.int32),  # kept for compat
            h_window        = np.array(h_window,        dtype=np.float32),
            n_bez           = np.array(self.n_bez,       dtype=np.int32),
        )
        return True

    def reset(self) -> None:
        self._wins.clear()
        self._curr = None
        self._sub_bufs = []


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Helpers                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _read_expert_state(
    expert: DMPCExpert | GPUDMPCExpert, E: int, N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read DMPC expert state for all (e, i) pairs after a plan() call."""
    if isinstance(expert, GPUDMPCExpert) or expert._gpu_solver is not None:
        # GPU path: direct numpy array slice — no Python loop
        return (
            expert._st_U[:E, :N].astype(np.float32),
            expert._st_replanned[:E, :N].copy(),
            expert._st_reset_mode[:E, :N].copy(),
            expert._st_fallback[:E, :N].copy(),
            expert._st_coll_ts[:E, :N].copy(),
        )
    # OSQP path: read from state dict
    bezier_U       = np.zeros((E, N, expert.n_bez), dtype=np.float32)
    mpc_replanned  = np.zeros((E, N), dtype=np.bool_)
    mpc_reset_mode = np.zeros((E, N), dtype=np.bool_)
    mpc_fallback   = np.zeros((E, N), dtype=np.bool_)
    mpc_col_ts     = np.full((E, N), -1, dtype=np.int32)
    for e in range(E):
        for i in range(N):
            st = expert._state.get((e, i))
            if st is None:
                continue
            bezier_U[e, i]       = st["U"].astype(np.float32)
            mpc_replanned[e, i]  = bool(st.get("last_replanned", False))
            mpc_reset_mode[e, i] = bool(st.get("last_reset_mode", False))
            mpc_fallback[e, i]   = bool(st.get("last_fallback", False))
            mpc_col_ts[e, i]     = int(st.get("collision_timestep", -1))
    return bezier_U, mpc_replanned, mpc_reset_mode, mpc_fallback, mpc_col_ts


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Main collection loop                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # ── validate args ────────────────────────────────────────────────────────
    if not args_cli.test_only and args_cli.save_dir is None:
        raise ValueError("--save_dir is required unless --test_only is set.")

    # ── env setup ────────────────────────────────────────────────────────────
    curriculum_on = args_cli.n_min > 0
    if curriculum_on:
        env_cfg = MultiDroneDmpcRLEnvCfg()
        env_cfg.n_curriculum_min = args_cli.n_min
        task_id = "Isaac-MultiDrone-DMPC-RL-Direct-v0"
    else:
        env_cfg = MultiDroneDmpcEnvCfg()
        task_id = args_cli.task

    env_cfg.num_drones = args_cli.num_drones
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.no_terminate_on_bounds and hasattr(env_cfg, "terminate_on_bounds"):
        env_cfg.terminate_on_bounds = False
    # Collision termination is handled via had_collision filter at save time,
    # not by terminating episodes early (which would interrupt the DMPC expert).
    env_cfg.terminate_on_collision = False
    env_cfg.__post_init__()
    if getattr(args_cli, "device", None):
        env_cfg.sim.device = args_cli.device

    env_gym = gym.make(
        task_id,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    env: MultiDroneDmpcEnv = env_gym.unwrapped  # direct reference for attribute access

    # ── optional video recording ──────────────────────────────────────────────
    if args_cli.video:
        video_dir = Path(args_cli.save_dir) / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        env_gym = gym.wrappers.RecordVideo(
            env_gym,
            video_folder=str(video_dir),
            # Skip step 0 — SDG pipeline hangs on first render before sim is warm.
            # Also only record from env 0 viewport by using episode_trigger if available;
            # step_trigger fires every video_interval steps *after* warmup.
            step_trigger=lambda step: step > 0 and step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[collect_dmpc_data] video → {video_dir}  "
              f"(every {args_cli.video_interval} steps, {args_cli.video_length} frames)")

    device = env.device
    E = env.num_envs
    N = env.cfg.num_drones
    P = per_drone_obs_dim(N)
    h_window: float = 0.1
    n_substeps = max(1, round(h_window / (env_cfg.sim.dt * env_cfg.decimation)))

    print(
        f"[collect_dmpc_data] num_envs={E}  num_drones={N}  "
        f"obs_dim={P}  n_substeps={n_substeps}  "
        f"h_window={h_window}s  dt={env_cfg.sim.dt*env_cfg.decimation:.3f}s"
    )

    # ── DMPC expert ───────────────────────────────────────────────────────────
    _params = DMPCParams(
        pmin=env_cfg.pos_min,
        pmax=env_cfg.pos_max,
        rmin=env_cfg.rmin,
        ts=env_cfg.sim.dt * env_cfg.decimation,
    )
    if args_cli.gpu_expert:
        expert = GPUDMPCExpert(
            num_drones=N,
            num_envs=E,
            params=_params,
            device=device,
        )
        print(f"[collect_dmpc_data] GPUDMPCExpert enabled  "
              f"(N={N}  E={E}  ts={_params.ts:.3f}s  rmin={_params.rmin}m)")
    else:
        expert = DMPCExpert(num_drones=N, params=_params, device=device)
        if args_cli.gpu_admm:
            expert.enable_gpu_admm(max_envs=E, admm_iters=args_cli.admm_iters)
            print(f"[collect_dmpc_data] GPU ADMM enabled  "
                  f"(max_B={E*N}  iters={args_cli.admm_iters}  alpha=1.6  coll=penalty)")

    # ── output dir ───────────────────────────────────────────────────────────
    if args_cli.test_only:
        save_dir = None
        ep_count = 0
        print(f"[collect_dmpc_data] test_only mode  target={args_cli.target_episodes} episodes"
              + (f"  n_min={args_cli.n_min}" if curriculum_on else ""))
    else:
        save_dir = Path(args_cli.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(save_dir.glob("ep_*.npz"))
        ep_count = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
        print(f"[collect_dmpc_data] save_dir={save_dir}  "
              f"target={args_cli.target_episodes}  starting from ep {ep_count}"
              + (f"  n_min={args_cli.n_min}" if curriculum_on else ""))

    # Per-N success tracking (active when curriculum_on or always for breakdown).
    from collections import defaultdict
    suc_by_n: dict[int, list[bool]] = defaultdict(list)

    # ── per-env collectors + step trackers ───────────────────────────────────
    collectors = [
        EpisodeCollector(e, N, P, expert.n_bez, n_substeps)
        for e in range(E)
    ]
    # Per-env: position and goal at the start of the current episode.
    ep_init_pos = torch.zeros(E, N, 3, device=device)
    ep_goal     = torch.zeros(E, N, 3, device=device)
    # Per-env step counter (reset to 0 on each episode start).
    env_step_cnt = torch.zeros(E, dtype=torch.long, device=device)

    # ── collision filter setup ────────────────────────────────────────────────
    col_threshold = float(args_cli.min_safe_dist)
    filter_collisions = (N > 1 and col_threshold > 0.0)
    if filter_collisions:
        print(f"[collect] collision filter: min_safe_dist={col_threshold:.3f}m")
    # Per-env flag: True if any physical collision occurred this episode.
    had_collision = torch.zeros(E, dtype=torch.bool, device=device)
    col_filtered  = 0   # total episodes dropped by collision filter

    # ── initial reset ─────────────────────────────────────────────────────────
    env_gym.reset(seed=args_cli.seed)
    expert.reset()
    if hasattr(env, "_last_reset_env_ids"):
        env._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)

    # Capture initial positions/goals for all envs.
    ep_init_pos[:] = env._init_pos_w.detach().clone()
    ep_goal[:]     = env._goal_pos_w.detach().clone()
    last_ep_len = env.episode_length_buf.clone()

    # ── collection loop ───────────────────────────────────────────────────────
    total_steps   = 0
    saved_count   = ep_count
    # test_only: stop by total episodes done; normal: stop by saved count.
    target        = ep_count + args_cli.target_episodes
    t0            = time.time()

    # Success rate tracking
    total_done    = 0   # per-drone episodes counted
    total_ep_done = 0   # whole env-episodes completed (for test_only stopping)
    total_success = 0
    recent_outcomes: deque[int] = deque(maxlen=100)  # rolling 100-ep window

    pbar = tqdm(
        total=args_cli.target_episodes,
        initial=0,
        desc="collecting",
        unit="ep",
        dynamic_ncols=True,
    )

    try:
        while (total_ep_done < args_cli.target_episodes if args_cli.test_only
               else saved_count < target):
            if args_cli.max_steps > 0 and total_steps >= args_cli.max_steps:
                print(f"[collect_dmpc_data] max_steps={args_cli.max_steps} reached.")
                break

            # ── detect mid-episode resets (IsaacLab internal) ──────────────
            rewound = env.episode_length_buf < last_ep_len
            if rewound.any():
                rw_ids = rewound.nonzero(as_tuple=False).flatten()
                expert.reset(rw_ids)
                for e in rw_ids.tolist():
                    collectors[e].reset()
                    env_step_cnt[e] = 0
                    had_collision[e] = False
                    ep_init_pos[e] = env._init_pos_w[e].detach().clone()
                    ep_goal[e]     = env._goal_pos_w[e].detach().clone()
            last_ep_len = env.episode_length_buf.clone()

            # ── get per-drone obs BEFORE expert plan ────────────────────────
            per_drone_obs = env.get_per_drone_obs()        # (E, N, P)
            states        = env.get_world_states()         # pos_w, vel_w, quat_w, ...

            # ── DMPC expert plan (handles sub-step scheduling internally) ───
            ref_pos_w, ref_vel_w, ref_acc_w = expert.plan(
                pos_w=states["pos_w"],
                vel_w=states["lin_vel_w"],
                goal_w=states["goal_w"],
                env_origins=env._terrain.env_origins,
            )
            action_flat = torch.cat(
                [ref_pos_w, ref_vel_w, ref_acc_w], dim=-1
            ).reshape(E, N * 9)

            # ── read expert internal state ──────────────────────────────────
            (bezier_U, mpc_replanned,
             mpc_reset_mode, mpc_fallback, mpc_col_ts) = _read_expert_state(expert, E, N)

            # ── identify window-start envs and fit free ctrl_pts ────────────
            win_envs = [
                e for e in range(E)
                if env_step_cnt[e].item() % n_substeps == 0
            ]
            if win_envs:
                # Compute goal-aligned R for window-start envs.
                R_goal_all = env._compute_goal_aligned_R(
                    states["pos_w"], states["goal_w"]
                )  # (E, N, 3, 3)

                n_ctrl = expert.n_bez // 3  # 18 control points (3 segments × 6 pts)

                for e in win_envs:
                    obs_e      = per_drone_obs[e].cpu().numpy().astype(np.float32)
                    R_goal_e   = R_goal_all[e].cpu().numpy().astype(np.float32)  # (N, 3, 3)
                    pos_w_e    = states["pos_w"][e].cpu().numpy().astype(np.float32)   # (N, 3)
                    origin_e   = env._terrain.env_origins[e].cpu().numpy().astype(np.float32)  # (3,)
                    pos_local_e = pos_w_e - origin_e[None, :]  # (N, 3) env-local

                    # Transform bezier_U (env-local absolute) → goal-aligned relative.
                    # U.reshape(N, n_ctrl, 3) gives ctrl pts in env-local frame.
                    # V_pts = (U_pts - pos_local) @ R^T  (row vectors: world→goal-aligned)
                    U_e = bezier_U[e]  # (N, n_bez) numpy
                    U_pts = U_e.reshape(N, n_ctrl, 3)
                    U_pts_rel = U_pts - pos_local_e[:, None, :]  # (N, n_ctrl, 3)
                    R_T = R_goal_e.transpose(0, 2, 1)  # (N, 3, 3) transposed
                    V_pts = np.matmul(U_pts_rel, R_T)  # (N, n_ctrl, 3) goal-aligned relative
                    bezier_U_goal_e = V_pts.reshape(N, expert.n_bez).astype(np.float32)

                    collectors[e].begin_window(
                        obs              = obs_e,
                        pos_w            = pos_w_e,
                        vel_w            = states["lin_vel_w"][e].cpu().numpy().astype(np.float32),
                        quat_w           = states["quat_w"][e].cpu().numpy().astype(np.float32),
                        ang_vel_b        = states["ang_vel_b"][e].cpu().numpy().astype(np.float32),
                        goal_w           = states["goal_w"][e].cpu().numpy().astype(np.float32),
                        R_goal           = R_goal_e.reshape(N, 9),
                        bezier_U         = U_e,
                        bezier_U_goal    = bezier_U_goal_e,
                        ref_pos_w        = ref_pos_w[e].cpu().numpy().astype(np.float32),
                        ref_vel_w        = ref_vel_w[e].cpu().numpy().astype(np.float32),
                        ref_acc_w        = ref_acc_w[e].cpu().numpy().astype(np.float32),
                        mpc_replanned    = mpc_replanned[e],
                        mpc_reset_mode   = mpc_reset_mode[e],
                        mpc_fallback     = mpc_fallback[e],
                        mpc_collision_ts = mpc_col_ts[e],
                    )

            # ── step env ────────────────────────────────────────────────────
            _, reward, terminated, truncated, _ = env_gym.step(action_flat)
            env_step_cnt += 1

            # ── physical collision tracking (post-step positions) ────────────
            if filter_collisions:
                pos_now = env.get_world_states()["pos_w"]          # (E, N, 3)
                diff    = pos_now.unsqueeze(2) - pos_now.unsqueeze(1)
                pdist   = torch.linalg.norm(diff, dim=-1)          # (E, N, N)
                eye_m   = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
                pdist   = pdist.masked_fill(eye_m, float("inf"))
                min_dist = pdist.amin(dim=(1, 2))                  # (E,)
                had_collision |= (min_dist < col_threshold)

            # ── record substep wrench + refs ─────────────────────────────────
            for e in range(E):
                thrust_z = env._thrust[e, :, 0, 2].cpu().numpy().astype(np.float32)   # (N,)
                torque_b = env._moment[e, :, 0, :].cpu().numpy().astype(np.float32)   # (N, 3)
                ref_9d_e = action_flat[e].reshape(N, 9)
                collectors[e].record_substep(
                    ref_pos  = ref_9d_e[:, 0:3].cpu().numpy().astype(np.float32),
                    ref_vel  = ref_9d_e[:, 3:6].cpu().numpy().astype(np.float32),
                    ref_acc  = ref_9d_e[:, 6:9].cpu().numpy().astype(np.float32),
                    thrust_z = thrust_z,
                    torque_b = torque_b,
                )

            # ── handle unexpected resets (non-done) ─────────────────────────
            reset_ids = getattr(env, "_last_reset_env_ids", None)
            env._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)
            done = terminated | truncated
            done_set = (
                set(done.nonzero(as_tuple=False).flatten().tolist())
                if done.any() else set()
            )
            if reset_ids is not None and reset_ids.numel() > 0:
                unexpected = torch.tensor(
                    [e for e in reset_ids.tolist() if e not in done_set],
                    dtype=torch.long, device=device,
                )
                if unexpected.numel() > 0:
                    expert.reset(unexpected)
                    for e in unexpected.tolist():
                        collectors[e].reset()
                        env_step_cnt[e] = 0
                        ep_init_pos[e] = env._init_pos_w[e].detach().clone()
                        ep_goal[e]     = env._goal_pos_w[e].detach().clone()

            # ── episode done: save per-agent and reset ───────────────────────
            if done.any():
                done_ids = done.nonzero(as_tuple=False).flatten()
                expert.reset(done_ids)
                for e in done_ids.tolist():
                    total_ep_done += 1
                    origin = env._terrain.env_origins[e].cpu()
                    ip = ep_init_pos[e].cpu().numpy().astype(np.float32).copy()
                    gp = ep_goal[e].cpu().numpy().astype(np.float32).copy()
                    ip[:, :2] -= origin[:2].numpy()
                    gp[:, :2] -= origin[:2].numpy()
                    ep_had_collision = bool(had_collision[e].item())
                    # active drone count for this env (1..N when curriculum, N otherwise)
                    n_active = int(env._active_n[e].item()) if curriculum_on else N
                    for i in range(N):
                        if i >= n_active:
                            break  # skip parked inactive drones
                        success_i = bool(env._drone_just_succeeded[e, i].item())
                        total_done    += 1
                        total_success += int(success_i)
                        recent_outcomes.append(int(success_i))
                        suc_by_n[n_active].append(success_i)
                        if not args_cli.test_only:
                            should_save = (args_cli.save_all or success_i) and not ep_had_collision
                            if should_save:
                                ep_path = str(save_dir / f"ep_{saved_count:07d}.npz")
                                saved = collectors[e].save_agent(
                                    ep_path, i, ip[i], gp[i], success_i, h_window
                                )
                                if saved:
                                    saved_count += 1
                                    pbar.update(1)
                    collectors[e].reset()
                    had_collision[e] = False
                    env_step_cnt[e] = 0
                    ep_init_pos[e] = env._init_pos_w[e].detach().clone()
                    ep_goal[e]     = env._goal_pos_w[e].detach().clone()

            total_steps += E

            # ── tqdm postfix (every step, lightweight) ────────────────────────
            elapsed   = time.time() - t0
            sps       = total_steps / max(elapsed, 1e-3)
            roll_n    = len(recent_outcomes)
            roll_pct  = sum(recent_outcomes) / roll_n * 100 if roll_n else 0.0
            total_pct = total_success / max(total_done, 1) * 100
            if args_cli.test_only:
                pbar.n = total_ep_done
                pbar.set_postfix(
                    sps=f"{sps:.0f}",
                    succ=f"{total_pct:.1f}%",
                    roll=f"{roll_pct:.1f}%",
                    ep=total_ep_done,
                    refresh=False,
                )
            else:
                ep_rate = (saved_count - ep_count) / max(elapsed, 1e-3)
                pbar.set_postfix(
                    sps=f"{sps:.0f}",
                    ep_s=f"{ep_rate:.2f}",
                    succ=f"{total_pct:.1f}%",
                    roll=f"{roll_pct:.1f}%",
                    done=total_done,
                    refresh=False,
                )

    except KeyboardInterrupt:
        print("\n[collect_dmpc_data] interrupted by user.")
    finally:
        pbar.close()
        total_pct = total_success / max(total_done, 1) * 100
        if args_cli.test_only:
            print(
                f"\n[collect_dmpc_data] test done.  "
                f"episodes={total_ep_done}  "
                f"success={total_success}/{total_done} ({total_pct:.1f}%)  "
                f"steps={total_steps}  elapsed={time.time()-t0:.1f}s"
            )
        else:
            print(
                f"\n[collect_dmpc_data] done.  "
                f"saved={saved_count - ep_count}  "
                f"success={total_success}/{total_done} ({total_pct:.1f}%)  "
                f"steps={total_steps}  elapsed={time.time()-t0:.1f}s"
            )
        if curriculum_on and suc_by_n:
            print("[collect_dmpc_data] Breakdown by active-drone count:")
            for n_a in sorted(suc_by_n):
                sl = suc_by_n[n_a]
                print(f"  N_active={n_a:2d}: {sum(sl):4d}/{len(sl):4d} = {sum(sl)/len(sl):.1%}")
        simulation_app.close()


if __name__ == "__main__":
    main()
