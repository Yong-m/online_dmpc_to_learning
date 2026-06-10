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
    if args_cli.seed >= 0:
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)

    env_gym, env = _make_env()
    expert = _make_expert(env)
    device = env.device
    E = env.num_envs
    N = env.cfg.num_drones

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
    static_obstacle_enabled = env.cfg.enable_static_obstacles
    log_dir = args_cli.log_dir + ("_obs" if static_obstacle_enabled else "")
    log_dir = Path(log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    stop_by_steps = args_cli.steps > 0
    max_steps = args_cli.steps if stop_by_steps else 10**12
    target_episodes = args_cli.episodes if args_cli.episodes > 0 else 0

    episode_returns = torch.zeros(E, device=device)
    episode_success = torch.zeros(E, dtype=torch.bool, device=device)
    episode_drone_collision = torch.zeros(E, dtype=torch.bool, device=device)
    episode_obstacle_collision = torch.zeros(E, dtype=torch.bool, device=device)
    episode_bounds_failure = torch.zeros(E, dtype=torch.bool, device=device)
    episode_success_step = torch.full((E,), -1, dtype=torch.long, device=device)
    episode_start_step = torch.zeros(E, dtype=torch.long, device=device)
    last_episode_length_buf = env.episode_length_buf.clone()

    step_rows: list[dict] = []
    episode_rows: list[dict] = []
    plan_wall_ms: list[float] = []
    env_step_wall_ms: list[float] = []
    total_step_wall_ms: list[float] = []
    success_times_s: list[float] = []
    task_durations_s: list[float] = []

    total_episodes = 0
    total_success = 0
    total_clean_success = 0
    total_drone_collision = 0
    total_obstacle_collision = 0
    total_bounds_failure = 0
    total_terminated = 0
    total_truncated = 0

    t_run0 = time.perf_counter()
    print(
        f"[dmpc_expert_test] solver={args_cli.solver} num_envs={E} num_drones={N} "
        f"episodes_target={target_episodes} steps_limit={args_cli.steps} seed={args_cli.seed} "
        f"static_obs={env.cfg.enable_static_obstacles} log_dir={log_dir}",
        flush=True,
    )

    step = 0
    while step < max_steps:
        if not stop_by_steps and target_episodes > 0 and total_episodes >= target_episodes:
            break
        rewound = env.episode_length_buf < last_episode_length_buf
        if rewound.any():
            ids = rewound.nonzero(as_tuple=False).flatten()
            expert.reset(obstacle_info=env.get_obstacle_info(), env_ids=ids)
            episode_returns[ids] = 0.0
            episode_success[ids] = False
            episode_drone_collision[ids] = False
            episode_obstacle_collision[ids] = False
            episode_bounds_failure[ids] = False
            episode_success_step[ids] = -1
            episode_start_step[ids] = step

        t_step0 = time.perf_counter()
        _sync_if_cuda(device)
        t_plan0 = time.perf_counter()
        action = _expert_action(env, expert)
        _sync_if_cuda(device)
        t_plan1 = time.perf_counter()

        t_env0 = time.perf_counter()
        _, reward, terminated, truncated, _ = env_gym.step(action)
        _sync_if_cuda(device)
        t_env1 = time.perf_counter()
        t_step1 = time.perf_counter()

        plan_ms = (t_plan1 - t_plan0) * 1e3
        env_ms = (t_env1 - t_env0) * 1e3
        step_ms = (t_step1 - t_step0) * 1e3
        plan_wall_ms.append(plan_ms)
        env_step_wall_ms.append(env_ms)
        total_step_wall_ms.append(step_ms)
        episode_returns += reward

        st = env.get_world_states()
        pos_w = st["pos_w"]
        goal_dist = torch.linalg.norm(pos_w - st["goal_w"], dim=-1)

        if N > 1:
            pair_diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
            pair_dist = torch.linalg.norm(pair_diff, dim=-1)
            eye = torch.eye(N, dtype=torch.bool, device=device)
            pair_dist = pair_dist.masked_fill(eye, float("inf"))
            episode_drone_collision |= pair_dist.amin(dim=(1, 2)) < args_cli.rmin_collision_check
        episode_obstacle_collision |= _static_obstacle_collision_mask(env, pos_w)
        z = pos_w[..., 2]
        episode_bounds_failure |= ((z < env.cfg.z_min) | (z > env.cfg.z_max)).any(dim=-1)

        success_done = getattr(env, "_just_succeeded", torch.zeros(E, dtype=torch.bool, device=device)).clone()
        newly_succeeded = success_done & ~episode_success
        episode_success_step[newly_succeeded] = step
        episode_success |= newly_succeeded

        natural_done = terminated | truncated
        done = natural_done | success_done
        if done.any():
            done_ids = done.nonzero(as_tuple=False).flatten()
            failure_terminated = terminated & ~success_done
            clean_success = episode_success & ~(episode_drone_collision | episode_obstacle_collision | episode_bounds_failure | failure_terminated)
            for env_id_t in done_ids:
                env_id = int(env_id_t.item())
                succeeded = bool(episode_success[env_id].item())
                clean = bool(clean_success[env_id].item())
                duration_s = float((step - int(episode_start_step[env_id].item()) + 1) * env.step_dt)
                task_durations_s.append(duration_s)
                t_success = float("nan")
                if succeeded and episode_success_step[env_id] >= 0:
                    t_success = float((episode_success_step[env_id] - episode_start_step[env_id]).item() * env.step_dt)
                    success_times_s.append(t_success)
                episode_rows.append({
                    "episode_index": total_episodes,
                    "env_id": env_id,
                    "end_step": step,
                    "duration_s": duration_s,
                    "time_to_success_s": t_success,
                    "success": int(succeeded),
                    "clean_success": int(clean),
                    "drone_collision": int(episode_drone_collision[env_id].item()),
                    "obstacle_collision": int(episode_obstacle_collision[env_id].item()),
                    "bounds_failure": int(episode_bounds_failure[env_id].item()),
                    "terminated": int(terminated[env_id].item()),
                    "truncated": int(truncated[env_id].item()),
                    "return": float(episode_returns[env_id].item()),
                    "goal_dist_mean": float(goal_dist[env_id].mean().item()),
                    "goal_dist_max": float(goal_dist[env_id].max().item()),
                })
            total_episodes += int(done_ids.numel())
            total_success += int(episode_success[done_ids].sum().item())
            total_clean_success += int(clean_success[done_ids].sum().item())
            total_drone_collision += int(episode_drone_collision[done_ids].sum().item())
            total_obstacle_collision += int(episode_obstacle_collision[done_ids].sum().item())
            total_bounds_failure += int(episode_bounds_failure[done_ids].sum().item())
            total_terminated += int(failure_terminated[done_ids].sum().item())
            total_truncated += int(truncated[done_ids].sum().item())

        reset_env_ids = getattr(env, "_last_reset_env_ids", None)
        ids = None
        if reset_env_ids is not None and reset_env_ids.numel() > 0:
            ids = reset_env_ids
            expert.reset(obstacle_info=env.get_obstacle_info(), env_ids=ids)
            env._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)
        elif natural_done.any():
            ids = natural_done.nonzero(as_tuple=False).flatten()
            expert.reset(obstacle_info=env.get_obstacle_info(), env_ids=ids)

        if ids is not None and ids.numel() > 0:
            episode_returns[ids] = 0.0
            episode_success[ids] = False
            episode_drone_collision[ids] = False
            episode_obstacle_collision[ids] = False
            episode_bounds_failure[ids] = False
            episode_success_step[ids] = -1
            episode_start_step[ids] = step + 1

        qp_stats = expert.get_qp_timing_stats() if hasattr(expert, "get_qp_timing_stats") else {}
        step_rows.append({
            "step": step,
            "sim_time_s": float((step + 1) * env.step_dt),
            "plan_wall_ms_all_envs": plan_ms,
            "env_step_wall_ms": env_ms,
            "total_step_wall_ms": step_ms,
            "completed_episodes": total_episodes,
            "success_rate_so_far": float(total_success / max(total_episodes, 1)),
            "clean_success_rate_so_far": float(total_clean_success / max(total_episodes, 1)),
            "goal_dist_mean": float(goal_dist.max(dim=-1).values.mean().item()),
            "goal_dist_max": float(goal_dist.max().item()),
            "expert_qp_avg_ms": float(qp_stats.get("avg_s", 0.0) * 1e3),
            "expert_qp_max_ms": float(qp_stats.get("max_s", 0.0) * 1e3),
            "expert_qp_count": float(qp_stats.get("count", 0.0)),
        })

        if args_cli.progress_every > 0 and ((step + 1) % args_cli.progress_every == 0 or step == 0):
            denom = max(total_episodes, 1)
            print(
                f"[dmpc_expert_test] step={step + 1} episodes={total_episodes} "
                f"success={total_success / denom:.3f} clean={total_clean_success / denom:.3f} "
                f"plan_ms_mean={np.mean(plan_wall_ms):.2f} plan_ms_last={plan_ms:.2f} "
                f"goal_max_mean={goal_dist.max(dim=-1).values.mean().item():.3f}",
                flush=True,
            )

        last_episode_length_buf = env.episode_length_buf.clone()
        step += 1

    runtime_s = time.perf_counter() - t_run0
    denom = max(total_episodes, 1)
    qp_stats = expert.get_qp_timing_stats() if hasattr(expert, "get_qp_timing_stats") else {
        "avg_s": 0.0, "max_s": 0.0, "total_s": 0.0, "count": 0.0
    }
    metrics = {
        "solver": args_cli.solver,
        "num_envs": E,
        "num_drones": N,
        "seed": args_cli.seed,
        "steps": step,
        "completed_episodes": total_episodes,
        "success_rate": float(total_success / denom),
        "clean_success_rate": float(total_clean_success / denom),
        "drone_collision_rate": float(total_drone_collision / denom),
        "obstacle_collision_rate": float(total_obstacle_collision / denom),
        "bounds_failure_rate": float(total_bounds_failure / denom),
        "terminated_rate": float(total_terminated / denom),
        "truncated_rate": float(total_truncated / denom),
        "runtime_wall_s": float(runtime_s),
        "steps_per_wall_s": float(step / max(runtime_s, 1e-9)),
        "episodes_per_wall_s": float(total_episodes / max(runtime_s, 1e-9)),
        "expert_qp_avg_ms": float(qp_stats.get("avg_s", 0.0) * 1e3),
        "expert_qp_max_ms": float(qp_stats.get("max_s", 0.0) * 1e3),
        "expert_qp_total_s": float(qp_stats.get("total_s", 0.0)),
        "expert_qp_count": float(qp_stats.get("count", 0.0)),
    }
    metrics.update(_stats(plan_wall_ms, "plan_wall_ms_all_envs"))
    metrics.update(_stats(env_step_wall_ms, "env_step_wall_ms"))
    metrics.update(_stats(total_step_wall_ms, "total_step_wall_ms"))
    metrics.update(_stats(success_times_s, "success_time_sim_s"))
    metrics.update(_stats(task_durations_s, "episode_duration_sim_s"))

    with (log_dir / "summary.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (log_dir / "args.json").open("w") as f:
        json.dump(vars(args_cli), f, indent=2)
    _write_csv(log_dir / "episodes.csv", episode_rows)
    if not args_cli.no_step_csv:
        _write_csv(log_dir / "steps.csv", step_rows)

    print(
        "[dmpc_expert_test] done "
        f"episodes={total_episodes} success={metrics['success_rate']:.3f} "
        f"clean={metrics['clean_success_rate']:.3f} "
        f"mean_plan_ms={metrics['plan_wall_ms_all_envs_mean']:.2f} "
        f"p95_plan_ms={metrics['plan_wall_ms_all_envs_p95']:.2f} "
        f"mean_success_time={metrics['success_time_sim_s_mean']:.2f}s "
        f"log_dir={log_dir}",
        flush=True,
    )

    env_gym.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
