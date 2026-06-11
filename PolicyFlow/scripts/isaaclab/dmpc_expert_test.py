"""Benchmark CPU/GPU DMPC experts in IsaacLab.

This script runs the DMPC expert directly, without BC training or policy
sampling. It logs env-level success metrics, task completion times in simulated
seconds, and per-step optimization wall time measured around expert planning
for all environments in the current Isaac step.

Examples:

    # CPU OSQP expert, fixed task sequence
    python dmpc_expert_test.py --solver cpu --num_envs 4 --num_drones 10 --episodes 32 --seed 0

    # GPU ADMM expert, fixed number of env steps, with video
    python dmpc_expert_test.py --solver gpu --num_envs 16 --num_drones 10 --steps 1000 --video
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_POLICYFLOW_ROOT = _HERE.parent.parent / "policyflow"
_PROJECT_ROOT = _HERE.parent.parent.parent
for _p in (_POLICYFLOW_ROOT, _PROJECT_ROOT):
    _s = str(_p)
    if _p.exists() and _s not in sys.path:
        sys.path.insert(0, _s)

parser = argparse.ArgumentParser(description="Benchmark DMPC expert performance.")
parser.add_argument("--solver", choices=["cpu", "gpu"], default="cpu",
                    help="cpu = DMPCExpert/OSQP, gpu = GPUDMPCExpert/ADMM.")
parser.add_argument("--gpu_admm", action="store_true", default=False,
                    help="Alias for --solver gpu.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_drones", "--num_agents", dest="num_drones", type=int, default=4)
parser.add_argument("--episodes", "--num_episodes", dest="episodes", type=int, default=0,
                    help="Stop after this many completed env episodes. Use 0 to disable.")
parser.add_argument("--steps", "--total_env_steps", dest="steps", type=int, default=0,
                    help="Stop after this many Isaac env steps. If >0, this takes priority over --episodes.")
parser.add_argument("--task", type=str, default="Isaac-MultiDrone-DMPC-Direct-v0")
parser.add_argument("--seed", type=int, default=0,
                    help="Task seed. Use a negative value to leave env.reset unseeded.")
parser.add_argument("--rmin_collision_check", type=float, default=0.3)
parser.add_argument("--num_workers", type=int, default=None)
parser.add_argument("--progress_every", type=int, default=100)
parser.add_argument("--log_dir", type=str, default="runs/dmpc_expert_test")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--no_step_csv", action="store_true", default=False,
                    help="Do not write per-step timing CSV.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_dir", type=str, default="runs/dmpc_expert_test/videos")
parser.add_argument("--video_length", type=int, default=0,
                    help="Frames per clip. 0 uses one env episode length.")
parser.add_argument("--video_interval", type=int, default=0,
                    help="Start a video every N env steps. 0 records from step 0 only.")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.gpu_admm:
    args_cli.solver = "gpu"
args_cli.enable_cameras = args_cli.video

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import quadcopter  # noqa: F401, E402
from quadcopter.dmpc_expert import DMPCExpert, DMPCParams  # noqa: E402
from quadcopter.multi_drone_dmpc_env import MultiDroneDmpcEnv, MultiDroneDmpcEnvCfg  # noqa: E402


def _sync_if_cuda(device: str | torch.device) -> None:
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_p99": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_count": float(arr.size),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_p99": float(np.percentile(arr, 99)),
        f"{prefix}_max": float(np.max(arr)),
    }


@torch.no_grad()
def _static_obstacle_collision_mask(env: MultiDroneDmpcEnv, pos_w: torch.Tensor) -> torch.Tensor:
    if not getattr(env.cfg, "enable_static_obstacles", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_static_obstacle_ellipsoid_centers_scales_w"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    centers, scales = env._static_obstacle_ellipsoid_centers_scales_w()
    if centers.shape[1] == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    metric_rel = (pos_w[:, :, None, :] - centers[:, None, :, :]) / scales.reshape(1, 1, -1, 3).clamp(min=1e-6)
    metric_dist = torch.linalg.norm(metric_rel, dim=-1)
    return (metric_dist < 1.0).any(dim=(1, 2))


@torch.no_grad()
def _expert_action(env: MultiDroneDmpcEnv, expert: DMPCExpert) -> torch.Tensor:
    states = env.get_world_states()
    ref_pos_w, ref_vel_w, ref_acc_w = expert.plan(
        pos_w=states["pos_w"],
        vel_w=states["lin_vel_w"],
        goal_w=states["goal_w"],
        env_origins=env._terrain.env_origins,
    )
    return env.ref_to_action(ref_pos_w, ref_vel_w, ref_acc_w)


def _make_env() -> tuple[gym.Env, MultiDroneDmpcEnv]:
    env_cfg = MultiDroneDmpcEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.num_drones = args_cli.num_drones
    env_cfg.__post_init__()

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        os.makedirs(args_cli.video_dir, exist_ok=True)
        video_length = args_cli.video_length if args_cli.video_length > 0 else int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
        if args_cli.video_interval > 0:
            trigger = lambda step: step % args_cli.video_interval == 0
        else:
            trigger = lambda step: step == 0
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.video_dir,
            step_trigger=trigger,
            video_length=video_length,
            disable_logger=True,
            name_prefix=f"dmpc_expert_{args_cli.solver}",
        )
        print(f"[dmpc_expert_test] video_dir={args_cli.video_dir} video_length={video_length}", flush=True)
    return env, env.unwrapped


def _make_expert(env: MultiDroneDmpcEnv) -> DMPCExpert:
    params = DMPCParams(
        pmin=env.cfg.pos_min,
        pmax=env.cfg.pos_max,
        rmin=env.cfg.rmin,
        ts=env.cfg.sim.dt * env.cfg.decimation,
    )
    if args_cli.solver == "gpu":
        from quadcopter.dmpc_gpu_expert import GPUDMPCExpert  # noqa: E402
        expert = GPUDMPCExpert(
            num_drones=env.cfg.num_drones,
            num_envs=env.num_envs,
            params=params,
            device=env.device,
        )
    else:
        expert = DMPCExpert(
            num_drones=env.cfg.num_drones,
            num_envs=env.num_envs,
            params=params,
            num_workers=args_cli.num_workers,
            device=env.device,
        )
    return expert


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # Single-run evaluation: each env runs exactly ONE episode from step 0.
    # Completed envs (success / collision / OOB) receive zero actions (freeze).
    # Events are only tracked for still-active envs, so auto-resets of
    # completed envs do not corrupt metrics.
    if args_cli.seed >= 0:
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)

    env_gym, env = _make_env()
    expert = _make_expert(env)
    device = env.device
    E = env.num_envs
    N = env.cfg.num_drones

    # ── Reset — all envs start from step 0 ───────────────────────────────
    if args_cli.seed >= 0:
        env_gym.reset(seed=args_cli.seed)
    else:
        env_gym.reset()
    expert.reset(obstacle_info=env.get_obstacle_info() if hasattr(env, "get_obstacle_info") else None)
    if hasattr(expert, "reset_qp_timing_stats"):
        expert.reset_qp_timing_stats()
    if hasattr(env, "_last_reset_env_ids"):
        env._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)

    run_name = args_cli.run_name or time.strftime(f"{args_cli.solver}_E{E}_N{N}_%Y%m%d_%H%M%S")
    static_obstacle_enabled = getattr(env.cfg, "enable_static_obstacles", False)
    log_dir = args_cli.log_dir + ("_obs" if static_obstacle_enabled else "")
    log_dir = Path(log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    max_ep_steps = int(round(env.cfg.episode_length_s / (env.cfg.sim.dt * env.cfg.decimation)))

    print(
        f"[dmpc_expert_test] solver={args_cli.solver}  num_envs={E}  num_drones={N}  "
        f"max_ep_steps={max_ep_steps}  seed={args_cli.seed}  "
        f"static_obs={static_obstacle_enabled}  log_dir={log_dir}",
        flush=True,
    )

    # ── Per-env state ─────────────────────────────────────────────────────
    completed     = torch.zeros(E, dtype=torch.bool, device=device)
    succeeded     = torch.zeros(E, dtype=torch.bool, device=device)
    collided      = torch.zeros(E, dtype=torch.bool, device=device)
    obs_collided  = torch.zeros(E, dtype=torch.bool, device=device)
    oob_flag      = torch.zeros(E, dtype=torch.bool, device=device)
    success_step  = torch.full((E,), -1, dtype=torch.long, device=device)
    returns       = torch.zeros(E, device=device)
    eye_mask      = torch.eye(N, dtype=torch.bool, device=device) if N > 1 else None

    step_rows:   list[dict]  = []
    plan_ms_log: list[float] = []
    env_ms_log:  list[float] = []

    t_run0 = time.perf_counter()
    step   = 0

    for step in range(max_ep_steps):
        active = ~completed
        if not active.any():
            break

        # ── Expert planning ───────────────────────────────────────────────
        _sync_if_cuda(device)
        t_plan0 = time.perf_counter()
        action = _expert_action(env, expert)
        _sync_if_cuda(device)
        t_plan1 = time.perf_counter()
        plan_ms = (t_plan1 - t_plan0) * 1e3
        plan_ms_log.append(plan_ms)

        # Zero-out completed envs so their drones stay still
        action[completed] = 0.0

        t_env0 = time.perf_counter()
        _, reward, terminated, truncated, _ = env_gym.step(action)
        _sync_if_cuda(device)
        t_env1 = time.perf_counter()
        env_ms_log.append((t_env1 - t_env0) * 1e3)

        # Accumulate return only for active envs
        returns += reward * active.float()

        # ── Event detection — active envs only ────────────────────────────
        just_succeeded = getattr(
            env, "_just_succeeded", torch.zeros(E, dtype=torch.bool, device=device)
        ).clone() & active

        st    = env.get_world_states()
        pos_w = st["pos_w"]   # (E, N, 3)

        if N > 1:
            pair_diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
            pair_dist = torch.linalg.norm(pair_diff, dim=-1)
            pair_dist = pair_dist.masked_fill(eye_mask, float("inf"))
            just_collided = (pair_dist.amin(dim=(1, 2)) < args_cli.rmin_collision_check) & active
        else:
            just_collided = torch.zeros(E, dtype=torch.bool, device=device)

        just_obs_collided = _static_obstacle_collision_mask(env, pos_w) & active

        z        = pos_w[..., 2]
        just_oob = ((z < env.cfg.z_min) | (z > env.cfg.z_max)).any(dim=-1) & active

        # Record success step on first occurrence
        new_succ = just_succeeded & (success_step < 0)
        success_step[new_succ] = step

        succeeded    |= just_succeeded
        collided     |= just_collided
        obs_collided |= just_obs_collided
        oob_flag     |= just_oob

        # Complete on any terminal event
        natural_done = terminated | truncated
        completed |= just_succeeded | just_collided | just_obs_collided | just_oob | (natural_done & active)

        # Reset expert state for auto-reset envs (keeps internal solver clean)
        reset_ids = getattr(env, "_last_reset_env_ids", None)
        if reset_ids is None or reset_ids.numel() == 0:
            if natural_done.any():
                reset_ids = natural_done.nonzero(as_tuple=False).flatten()
        if reset_ids is not None and reset_ids.numel() > 0:
            expert.reset(
                obstacle_info=env.get_obstacle_info() if hasattr(env, "get_obstacle_info") else None,
                env_ids=reset_ids,
            )
            if hasattr(env, "_last_reset_env_ids"):
                env._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)

        if args_cli.progress_every > 0 and (step + 1) % args_cli.progress_every == 0:
            n_done = int(completed.sum().item())
            n_succ = int(succeeded.sum().item())
            print(
                f"[dmpc_expert_test] step={step + 1}/{max_ep_steps}  "
                f"completed={n_done}/{E}  succeeded={n_succ}/{E}  "
                f"plan_ms={plan_ms:.2f}",
                flush=True,
            )

        if not args_cli.no_step_csv:
            qp_stats = expert.get_qp_timing_stats() if hasattr(expert, "get_qp_timing_stats") else {}
            step_rows.append({
                "step":             step,
                "sim_time_s":       float((step + 1) * env.step_dt),
                "plan_wall_ms":     plan_ms,
                "env_step_wall_ms": env_ms_log[-1],
                "active_envs":      int(active.sum().item()),
                "completed_envs":   int(completed.sum().item()),
                "succeeded_so_far": int(succeeded.sum().item()),
                "expert_qp_avg_ms": float(qp_stats.get("avg_s", 0.0) * 1e3),
                "expert_qp_max_ms": float(qp_stats.get("max_s", 0.0) * 1e3),
            })

    # ── Per-env results ───────────────────────────────────────────────────
    env_rows:        list[dict]  = []
    success_times_s: list[float] = []
    dt = env.step_dt

    for eid in range(E):
        succ  = bool(succeeded[eid].item())
        coll  = bool(collided[eid].item())
        o_col = bool(obs_collided[eid].item())
        oob   = bool(oob_flag[eid].item())
        clean = succ and not coll and not o_col and not oob
        sst   = int(success_step[eid].item())
        t_s   = float(sst) * dt if (succ and sst >= 0) else float("nan")
        if not np.isnan(t_s):
            success_times_s.append(t_s)
        env_rows.append({
            "env_id":             eid,
            "success":            int(succ),
            "clean_success":      int(clean),
            "drone_collision":    int(coll),
            "obstacle_collision": int(o_col),
            "oob":                int(oob),
            "time_to_success_s":  t_s,
            "return":             float(returns[eid].item()),
        })

    # ── Summary ───────────────────────────────────────────────────────────
    runtime_s = time.perf_counter() - t_run0
    n_succ  = int(succeeded.sum().item())
    n_clean = int((succeeded & ~collided & ~obs_collided & ~oob_flag).sum().item())
    n_coll  = int(collided.sum().item())
    n_obs   = int(obs_collided.sum().item())
    n_oob   = int(oob_flag.sum().item())
    steps_run = step + 1

    qp_stats = expert.get_qp_timing_stats() if hasattr(expert, "get_qp_timing_stats") else {
        "avg_s": 0.0, "max_s": 0.0, "total_s": 0.0, "count": 0.0,
    }
    metrics = {
        "solver":                  args_cli.solver,
        "num_envs":                E,
        "num_drones":              N,
        "seed":                    args_cli.seed,
        "steps_run":               steps_run,
        "success_rate":            float(n_succ  / E),
        "clean_success_rate":      float(n_clean / E),
        "collision_rate":          float(n_coll  / E),
        "obstacle_collision_rate": float(n_obs   / E),
        "oob_rate":                float(n_oob   / E),
        "runtime_wall_s":          float(runtime_s),
        "steps_per_wall_s":        float(steps_run / max(runtime_s, 1e-9)),
        "expert_qp_avg_ms":        float(qp_stats.get("avg_s",   0.0) * 1e3),
        "expert_qp_max_ms":        float(qp_stats.get("max_s",   0.0) * 1e3),
        "expert_qp_total_s":       float(qp_stats.get("total_s", 0.0)),
        "expert_qp_count":         float(qp_stats.get("count",   0.0)),
    }
    metrics.update(_stats(plan_ms_log,     "plan_wall_ms"))
    metrics.update(_stats(env_ms_log,      "env_step_wall_ms"))
    metrics.update(_stats(success_times_s, "success_time_sim_s"))

    with (log_dir / "summary.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (log_dir / "args.json").open("w") as f:
        json.dump(vars(args_cli), f, indent=2)
    _write_csv(log_dir / "envs.csv", env_rows)
    if not args_cli.no_step_csv:
        _write_csv(log_dir / "steps.csv", step_rows)

    print(
        f"[dmpc_expert_test] done  envs={E}  "
        f"success={metrics['success_rate']:.3f}  clean={metrics['clean_success_rate']:.3f}  "
        f"collision={metrics['collision_rate']:.3f}  oob={metrics['oob_rate']:.3f}  "
        f"plan_ms_p50={metrics['plan_wall_ms_p50']:.2f}  "
        f"plan_ms_p95={metrics['plan_wall_ms_p95']:.2f}  "
        f"mean_success_time={metrics['success_time_sim_s_mean']:.2f}s  "
        f"log_dir={log_dir}",
        flush=True,
    )

    env_gym.close()
    simulation_app.close()


if __name__ == "__main__":
    main()