"""Evaluate a trained RL policy on the multi-drone navigation task.

Mirrors dmpc_expert_test.py but uses a loaded RL model checkpoint instead of the
DMPC expert. Supports both PolicyFlow (train_rl_dmpc.py) and Gaussian PPO
(train_rl_ippo.py) checkpoints; actor type is detected automatically from the
checkpoint.

Training phases (all use train_rl_dmpc.py → PolicyFlow actor):
  Phase 1: soft curriculum  →  config/phase1.pt
  Phase 2: hard termination →  config/phase2.pt
  Phase 3: per-drone IPPO critic, resumed from Phase 2

Gaussian PPO (train_rl_ippo.py) is a separate comparison baseline.

Examples:

    # Evaluate Phase 2 PolicyFlow checkpoint
    python rl_model_test.py --checkpoint config/phase2.pt --num_envs 16 --num_drones 10

    # Evaluate Gaussian PPO baseline checkpoint
    python rl_model_test.py --checkpoint runs/rl_ippo/exp/checkpoints/model_best.pt \\
        --num_envs 16 --num_drones 10

    # Stop after 200 completed episodes, save to custom dir
    python rl_model_test.py --checkpoint runs/rl_dmpc/exp/checkpoints/model_best.pt \\
        --num_envs 32 --num_drones 10 --episodes 200 --log_dir runs/rl_model_test
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

parser = argparse.ArgumentParser(description="Evaluate a trained RL policy.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to runner checkpoint (.pt file).")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_drones", "--num_agents", dest="num_drones", type=int, default=10)
parser.add_argument("--episodes", "--num_episodes", dest="episodes", type=int, default=0,
                    help="Stop after this many completed env episodes. 0 = disabled.")
parser.add_argument("--steps", "--total_env_steps", dest="steps", type=int, default=0,
                    help="Stop after this many env steps. Overrides --episodes if > 0.")
parser.add_argument("--emb_dim", type=int, default=128)
parser.add_argument("--hidden_dims", type=int, nargs="*", default=[256, 256, 256])
parser.add_argument("--sample_steps", type=int, default=10,
                    help="ODE steps for PolicyFlow sampling (ignored for Gaussian).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--rmin_collision_check", type=float, default=0.3)
parser.add_argument("--progress_every", type=int, default=100)
parser.add_argument("--log_dir", type=str, default="runs/rl_model_test")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--no_step_csv", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_dir", type=str, default="runs/rl_model_test/videos")
parser.add_argument("--video_length", type=int, default=0)
parser.add_argument("--video_interval", type=int, default=0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = args_cli.video

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
import torch.nn as nn        # noqa: E402

import quadcopter            # noqa: F401, E402
from quadcopter.multi_drone_dmpc_env import (  # noqa: E402
    PER_DRONE_OWN_DIM,
    PER_DRONE_SDF_DIM,
    PER_NEIGHBOUR_DIM,
)
from quadcopter.multi_drone_dmpc_rl_env import MultiDroneDmpcRLEnv, MultiDroneDmpcRLEnvCfg  # noqa: E402
from policyflow_torch.modules import (                             # noqa: E402
    ContinuousNormalizingFlow,
    FlowMlp,
    NeighborEncoder,
)
from policyflow_torch.modules.normalizer import EmpiricalNormalization  # noqa: E402
from policyflow_torch.env.isaaclab_wrapper import IsaacLabEnvWrapper  # noqa: E402


# ── Shared per-drone encoder (mirrors train_rl_ippo.py / train_rl_dmpc.py) ──

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


# ── Gaussian actor (matches train_rl_ippo.py) ─────────────────────────────

class GaussianActor(nn.Module):
    """Gaussian policy: PerDroneConditionNet encoder + mean head + learnable log_std."""

    def __init__(self, N, own_dim, neigh_dim, emb_dim, hidden_dims, action_dim=3, log_std_init=0.0):
        super().__init__()
        self._cond = PerDroneConditionNet(
            N=N, own_dim=own_dim, neigh_dim=neigh_dim,
            emb_dim=emb_dim, hidden_dims=hidden_dims,
        )
        self.mean_head = nn.Sequential(nn.Linear(emb_dim, action_dim), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_head(self._cond(obs))


# ── Checkpoint helpers ────────────────────────────────────────────────────

def _load_checkpoint(path: str, device):
    content = torch.load(path, map_location=device, weights_only=False)
    # New format: model_state_dicts
    if "model_state_dicts" in content:
        return content["model_state_dicts"].get("actor"), content
    # Legacy format: nested agent dict
    agent_state = content.get("agent", {})
    legacy = agent_state.get("model_dict", {})
    return legacy.get("actor"), content


def _detect_actor_type(actor_sd: dict) -> str:
    """Returns 'policyflow' or 'gaussian' based on checkpoint keys."""
    if actor_sd is None:
        raise ValueError("Checkpoint does not contain an actor state dict.")
    if actor_sd.get("_type") == "nn_module":
        return "gaussian"
    return "policyflow"


def _load_model_sd(model, sd: dict):
    """Load state dict using the same logic as IsaaclabRunner._model_load_state_dict."""
    if sd.get("_type") == "nn_module":
        model.load_state_dict(sd["state"])
    else:
        for attr, state in sd["state"].items():
            m = getattr(model, attr, None)
            if m is not None:
                m.load_state_dict(state)


def _build_policyflow_actor(N, per_drone_dim, emb_dim, hidden_dims, sample_steps, actor_sd, device):
    own_dim  = PER_DRONE_OWN_DIM   # base own-state without SDF; condition net splits internally
    neigh_dim = PER_NEIGHBOUR_DIM

    condition_net = PerDroneConditionNet(
        N=N, own_dim=per_drone_dim - PER_NEIGHBOUR_DIM * (N - 1),
        neigh_dim=neigh_dim, emb_dim=emb_dim, hidden_dims=hidden_dims,
    )
    flow_net = FlowMlp(
        x_dim=3,
        emb_dim=emb_dim,
        hidden_dims=hidden_dims,
        activations=["elu"] * len(hidden_dims) + ["linear"],
    )
    cnf = ContinuousNormalizingFlow(
        x_dims=3,
        nn_flow=flow_net,
        nn_condition=condition_net,
        sample_steps=sample_steps,
        sample_step_schedule="uniform_continuous",
        interpolation_type="rectified_flow",
        device=device,
    )
    _load_model_sd(cnf, actor_sd)
    cnf.model.eval()
    cnf.model_ema.eval()
    return cnf


def _build_gaussian_actor(N, per_drone_dim, emb_dim, hidden_dims, actor_sd, device):
    own_dim   = per_drone_dim - PER_NEIGHBOUR_DIM * (N - 1)
    neigh_dim = PER_NEIGHBOUR_DIM
    actor = GaussianActor(
        N=N, own_dim=own_dim, neigh_dim=neigh_dim,
        emb_dim=emb_dim, hidden_dims=hidden_dims,
    ).to(device)
    _load_model_sd(actor, actor_sd)
    actor.eval()
    return actor


# ── Policy inference ─────────────────────────────────────────────────────

@torch.no_grad()
def _policy_action(obs_per: torch.Tensor, actor, actor_type: str) -> torch.Tensor:
    """obs_per: (E*N, per_drone_dim) → actions: (E*N, 3)."""
    EN = obs_per.shape[0]
    if actor_type == "policyflow":
        x0 = torch.randn(EN, 3, device=obs_per.device)
        actions, _ = actor.sample(x0=x0, condition=obs_per, n_samples=EN)
        return actions
    else:
        return actor(obs_per)


# ── Stats helper ─────────────────────────────────────────────────────────

def _stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_{k}": float("nan")
                for k in ("count", "mean", "std", "min", "p50", "p90", "p95", "p99", "max")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_count": float(arr.size),
        f"{prefix}_mean":  float(np.mean(arr)),
        f"{prefix}_std":   float(np.std(arr)),
        f"{prefix}_min":   float(np.min(arr)),
        f"{prefix}_p50":   float(np.percentile(arr, 50)),
        f"{prefix}_p90":   float(np.percentile(arr, 90)),
        f"{prefix}_p95":   float(np.percentile(arr, 95)),
        f"{prefix}_p99":   float(np.percentile(arr, 99)),
        f"{prefix}_max":   float(np.max(arr)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sync_if_cuda(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


# ── Main ─────────────────────────────────────────────────────────────────
#
# Single-run evaluation: each env runs exactly ONE episode from step 0.
#   - Completed envs (success / collision / OOB) receive zero actions (freeze).
#   - Events are only tracked for still-active envs, so auto-resets of
#     completed envs do not corrupt the metrics.
#   - Loop ends when all E envs are completed or max episode steps elapsed.
#
# Logged:
#   summary.json  — success_rate, clean_success_rate, collision/oob rates,
#                   inference time stats, success_time_sim_s stats
#   args.json     — CLI arguments
#   envs.csv      — one row per env: success, clean_success, collision, oob,
#                   time_to_success_s, return
#   steps.csv     — one row per step: infer_wall_ms, active/completed counts
#                   (disable with --no_step_csv)

def main() -> None:
    if args_cli.seed >= 0:
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)

    # ── Environment ───────────────────────────────────────────────────────
    env_cfg = MultiDroneDmpcRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.num_drones     = args_cli.num_drones
    env_cfg.dmpc_guide_enabled = False
    env_cfg.__post_init__()

    render_mode = "rgb_array" if args_cli.video else None
    env_raw = gym.make("Isaac-MultiDrone-DMPC-RL-Direct-v0", cfg=env_cfg, render_mode=render_mode)
    assert isinstance(env_raw.unwrapped, MultiDroneDmpcRLEnv)

    if args_cli.video:
        os.makedirs(args_cli.video_dir, exist_ok=True)
        ep_len = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
        vlen = args_cli.video_length if args_cli.video_length > 0 else ep_len
        trigger = (lambda step: step % args_cli.video_interval == 0) if args_cli.video_interval > 0 \
                  else (lambda step: step == 0)
        env_raw = gym.wrappers.RecordVideo(
            env_raw, video_folder=args_cli.video_dir,
            step_trigger=trigger, video_length=vlen,
            disable_logger=True, name_prefix="rl_model",
        )

    env = IsaacLabEnvWrapper(env_raw)
    device = env.device
    E = args_cli.num_envs
    N = args_cli.num_drones

    # ── Obs / action dims ─────────────────────────────────────────────────
    sdf_dim       = PER_DRONE_SDF_DIM if env_cfg.enable_sdf_obs else 0
    own_dim       = PER_DRONE_OWN_DIM + sdf_dim
    per_drone_dim = own_dim + PER_NEIGHBOUR_DIM * (N - 1)

    # ── Load model ────────────────────────────────────────────────────────
    actor_sd, content = _load_checkpoint(args_cli.checkpoint, device)
    actor_type = _detect_actor_type(actor_sd)
    iteration  = content.get("iteration", 0)

    if actor_type == "policyflow":
        actor = _build_policyflow_actor(
            N, per_drone_dim, args_cli.emb_dim, args_cli.hidden_dims,
            args_cli.sample_steps, actor_sd, device,
        )
    else:
        actor = _build_gaussian_actor(
            N, per_drone_dim, args_cli.emb_dim, args_cli.hidden_dims,
            actor_sd, device,
        )

    print(
        f"[rl_model_test] actor_type={actor_type}  checkpoint_iter={iteration}  "
        f"num_envs={E}  num_drones={N}  per_drone_dim={per_drone_dim}  "
        f"emb_dim={args_cli.emb_dim}  hidden_dims={args_cli.hidden_dims}",
        flush=True,
    )

    # ── Logging dirs ──────────────────────────────────────────────────────
    run_name = args_cli.run_name or time.strftime(
        f"{actor_type}_E{E}_N{N}_iter{iteration}_%Y%m%d_%H%M%S"
    )
    log_dir = Path(args_cli.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Reset — all envs start from step 0 ───────────────────────────────
    if args_cli.seed >= 0:
        env_raw.reset(seed=args_cli.seed)
    else:
        env_raw.reset()
    obs, _ = env.reset()

    env_uw = env.unwrapped
    max_ep_steps = int(round(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)))

    # ── Per-env state ─────────────────────────────────────────────────────
    completed    = torch.zeros(E, dtype=torch.bool, device=device)
    succeeded    = torch.zeros(E, dtype=torch.bool, device=device)
    collided     = torch.zeros(E, dtype=torch.bool, device=device)
    oob_flag     = torch.zeros(E, dtype=torch.bool, device=device)
    success_step = torch.full((E,), -1, dtype=torch.long, device=device)
    returns      = torch.zeros(E, device=device)
    eye_mask     = torch.eye(N, dtype=torch.bool, device=device) if N > 1 else None

    step_rows:    list[dict]  = []
    infer_ms_log: list[float] = []
    env_ms_log:   list[float] = []

    t_run0 = time.perf_counter()
    step   = 0

    for step in range(max_ep_steps):
        active = ~completed
        if not active.any():
            break

        # ── Inference ─────────────────────────────────────────────────────
        obs_tensor = obs["actor_observations"] if isinstance(obs, dict) else obs
        obs_per    = obs_tensor.reshape(E * N, per_drone_dim)

        _sync_if_cuda(device)
        t_inf0 = time.perf_counter()
        actions_per = _policy_action(obs_per, actor, actor_type)
        _sync_if_cuda(device)
        t_inf1 = time.perf_counter()
        infer_ms = (t_inf1 - t_inf0) * 1e3
        infer_ms_log.append(infer_ms)

        # Zero-out completed envs so their drones stay still
        actions_env = actions_per.reshape(E, N * 3)
        actions_env[completed] = 0.0

        t_env0 = time.perf_counter()
        obs, reward, done, info = env.step(actions_env)
        _sync_if_cuda(device)
        t_env1 = time.perf_counter()
        env_ms_log.append((t_env1 - t_env0) * 1e3)

        # Accumulate return only for active envs
        returns += reward * active.float()

        # ── Event detection — active envs only ────────────────────────────
        just_succeeded = env_uw._just_succeeded.clone() & active

        st    = env_uw._stack_drone_state()
        pos_w = st["pos_w"]   # (E, N, 3)

        if N > 1:
            diff      = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
            pair_dist = torch.linalg.norm(diff, dim=-1)
            pair_dist = pair_dist.masked_fill(eye_mask, float("inf"))
            just_collided = (pair_dist.amin(dim=(1, 2)) < args_cli.rmin_collision_check) & active
        else:
            just_collided = torch.zeros(E, dtype=torch.bool, device=device)

        z        = pos_w[..., 2]
        just_oob = ((z < env_cfg.z_min) | (z > env_cfg.z_max)).any(dim=-1) & active

        # Record success step on first occurrence
        new_succ = just_succeeded & (success_step < 0)
        success_step[new_succ] = step

        succeeded |= just_succeeded
        collided  |= just_collided
        oob_flag  |= just_oob

        # Complete on success or env-level termination/timeout; collision/OOB flagged only
        timeout  = info.get("time_outs", torch.zeros(E, dtype=torch.bool, device=device))
        completed |= just_succeeded | (done.bool() & active) | (timeout & active)

        if args_cli.progress_every > 0 and (step + 1) % args_cli.progress_every == 0:
            n_done = int(completed.sum().item())
            n_succ = int(succeeded.sum().item())
            print(
                f"[rl_model_test] step={step + 1}/{max_ep_steps}  "
                f"completed={n_done}/{E}  succeeded={n_succ}/{E}  "
                f"infer_ms={infer_ms:.2f}",
                flush=True,
            )

        if not args_cli.no_step_csv:
            step_rows.append({
                "step":             step,
                "sim_time_s":       float((step + 1) * env_uw.step_dt),
                "infer_wall_ms":    infer_ms,
                "env_step_wall_ms": env_ms_log[-1],
                "active_envs":      int(active.sum().item()),
                "completed_envs":   int(completed.sum().item()),
                "succeeded_so_far": int(succeeded.sum().item()),
            })

    # Any envs still active at max_ep_steps → timed out (failure)

    # ── Per-env results ───────────────────────────────────────────────────
    env_rows:       list[dict]  = []
    success_times_s: list[float] = []
    dt = env_uw.step_dt

    for eid in range(E):
        succ  = bool(succeeded[eid].item())
        coll  = bool(collided[eid].item())
        oob   = bool(oob_flag[eid].item())
        clean = succ and not coll and not oob
        sst   = int(success_step[eid].item())
        t_s   = float(sst) * dt if (succ and sst >= 0) else float("nan")
        if not np.isnan(t_s):
            success_times_s.append(t_s)
        env_rows.append({
            "env_id":            eid,
            "success":           int(succ),
            "clean_success":     int(clean),
            "drone_collision":   int(coll),
            "oob":               int(oob),
            "time_to_success_s": t_s,
            "return":            float(returns[eid].item()),
        })

    # ── Summary ───────────────────────────────────────────────────────────
    runtime_s = time.perf_counter() - t_run0
    n_succ    = int(succeeded.sum().item())
    n_clean   = int((succeeded & ~collided & ~oob_flag).sum().item())
    n_coll    = int(collided.sum().item())
    n_oob     = int(oob_flag.sum().item())
    steps_run = step + 1

    metrics = {
        "actor_type":           actor_type,
        "checkpoint":           str(args_cli.checkpoint),
        "checkpoint_iteration": iteration,
        "num_envs":             E,
        "num_drones":           N,
        "seed":                 args_cli.seed,
        "steps_run":            steps_run,
        "success_rate":         float(n_succ  / E),
        "clean_success_rate":   float(n_clean / E),
        "collision_rate":       float(n_coll  / E),
        "oob_rate":             float(n_oob   / E),
        "runtime_wall_s":       float(runtime_s),
        "steps_per_wall_s":     float(steps_run / max(runtime_s, 1e-9)),
    }
    metrics.update(_stats(infer_ms_log,    "infer_wall_ms"))
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
        f"[rl_model_test] done  envs={E}  "
        f"success={metrics['success_rate']:.3f}  clean={metrics['clean_success_rate']:.3f}  "
        f"collision={metrics['collision_rate']:.3f}  oob={metrics['oob_rate']:.3f}  "
        f"infer_ms_p50={metrics['infer_wall_ms_p50']:.2f}  "
        f"infer_ms_p95={metrics['infer_wall_ms_p95']:.2f}  "
        f"mean_success_time={metrics['success_time_sim_s_mean']:.2f}s  "
        f"log_dir={log_dir}",
        flush=True,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
