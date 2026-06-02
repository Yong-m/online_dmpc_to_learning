"""eval_bc_dmpc.py — Evaluate an offline-trained drone BC policy in Isaac Sim.

Loads a checkpoint saved by offline_bc_dmpc.py, runs the policy in the
multi-drone DMPC environment, and writes results as JSON.

The policy outputs bezier_U_goal (n_bez=54D, goal-aligned relative frame).
At each window start the inference path is:
  1. Sample V = policy(obs)  →  (E, N, n_bez) goal-aligned relative ctrl pts
  2. Convert to U_local:
       V_pts[n, k] = V.reshape(n_ctrl, 3)
       U_local_pts = V_pts @ R + pos_local   (goal-aligned → world-relative → env-local abs)
  3. Evaluate Bezier at each sub-step using pre-computed sample matrices.

Usage (standalone)::

    python eval_bc_dmpc.py \\
        --checkpoint runs/offline_bc/model_best.pt \\
        --num_envs 32 --num_drones 4 \\
        --output_json /tmp/eval_result.json \\
        --headless

Called automatically by offline_bc_dmpc.py when --eval_every > 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_POLICYFLOW_ROOT = _HERE.parent.parent / "policyflow"
_PROJECT_ROOT = _HERE.parent.parent.parent
for _p in (_POLICYFLOW_ROOT, _PROJECT_ROOT):
    _s = str(_p)
    if _p.exists() and _s not in sys.path:
        sys.path.insert(0, _s)

# ── CLI (must be parsed before AppLauncher) ──────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate offline BC drone policy in sim.")
parser.add_argument("--checkpoint",  type=str, required=True,
                    help="Path to .pt checkpoint saved by offline_bc_dmpc.py.")
parser.add_argument("--num_envs",    type=int, default=32)
parser.add_argument("--num_drones",  type=int, default=4)
parser.add_argument("--task",        type=str, default="Isaac-MultiDrone-DMPC-Direct-v0")
parser.add_argument("--eval_steps",  type=int, default=400,
                    help="Total env steps for evaluation rollout.")
parser.add_argument("--n_substeps",  type=int, default=2,
                    help="Bezier sub-steps per window (overridden by checkpoint if present).")
parser.add_argument("--output_json", type=str, default=None,
                    help="Write JSON result to this path. If omitted, prints to stdout.")
parser.add_argument("--seed",        type=int, default=0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = getattr(args_cli, "headless", True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ──────────────────────────────────────────────────────
import gymnasium as gym  # noqa: E402
import numpy as np       # noqa: E402
import torch             # noqa: E402

import quadcopter  # noqa: F401,E402  registers Isaac-MultiDrone-DMPC-Direct-v0
from quadcopter.multi_drone_dmpc_env import (  # noqa: E402
    MultiDroneDmpcEnv,
    MultiDroneDmpcEnvCfg,
)
from quadcopter.dmpc_expert import BezierBasis, DMPCParams  # noqa: E402

# Import policy class from sibling offline_bc_dmpc.py
_here_str = str(_HERE)
if _here_str not in sys.path:
    sys.path.insert(0, _here_str)
from offline_bc_dmpc import OfflineDronePolicy  # noqa: E402


# ── Evaluation loop ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    env: MultiDroneDmpcEnv,
    policy: OfflineDronePolicy,
    n_steps: int,
    N: int,
    n_substeps: int = 2,
) -> dict[str, float]:
    """Run the offline policy in simulation and return success metrics.

    Policy outputs bezier_U_goal (n_bez=54D, goal-aligned relative frame).
    Each window: V → U_local → sample ref_pos/vel/acc per substep via BezierBasis.
    """
    device  = env.device
    E       = env.num_envs
    n_bez   = policy.n_bez
    n_ctrl  = n_bez // 3   # 18 control points

    # Build BezierBasis and precompute per-substep sample matrices.
    dmpc_p  = DMPCParams()
    bezier  = BezierBasis(p=dmpc_p.deg, l=dmpc_p.num_segments,
                          t_segment=dmpc_p.t_segment, dim=dmpc_p.dim)
    ts      = dmpc_p.ts  # control period = 0.02 s
    S_pos = [torch.from_numpy(
                bezier.sample_matrix(np.array([(k + 1) * ts]), deriv=0).astype(np.float32)
             ).to(device) for k in range(n_substeps)]   # each (3, n_bez)
    S_vel = [torch.from_numpy(
                bezier.sample_matrix(np.array([(k + 1) * ts]), deriv=1).astype(np.float32)
             ).to(device) for k in range(n_substeps)]
    S_acc = [torch.from_numpy(
                bezier.sample_matrix(np.array([(k + 1) * ts]), deriv=2).astype(np.float32)
             ).to(device) for k in range(n_substeps)]

    window_refs: list[torch.Tensor] | None = None
    sub_step = 0

    env.reset()
    per_drone_obs = env.get_per_drone_obs()
    returns          = torch.zeros(E, device=device)
    finished_returns: list[float] = []
    success_count    = 0
    episode_count    = 0

    env_origins = env._terrain.env_origins  # (E, 3)

    for _ in range(n_steps):
        if sub_step == 0 or window_refs is None:
            drone_st = env._stack_drone_state()
            pos_w0   = drone_st["pos_w"]       # (E, N, 3)

            R = env._compute_goal_aligned_R(pos_w0, env._goal_pos_w)  # (E, N, 3, 3)

            # Policy outputs V: (E, N, n_bez) in goal-aligned relative frame.
            V = policy.sample_action(per_drone_obs)  # (E, N, n_bez)
            V_flat = V.reshape(E * N, n_ctrl, 3)      # (E*N, 18, 3)
            R_flat = R.reshape(E * N, 3, 3)            # (E*N, 3, 3)

            # Convert goal-aligned relative → env-local absolute:
            #   U_local_pts = V_pts @ R + pos_local
            # (row-vector form: v_goal @ R = u_world_rel)
            origins_exp = env_origins.unsqueeze(1).expand(E, N, 3).reshape(E * N, 3)
            pos_local   = pos_w0.reshape(E * N, 3) - origins_exp  # (E*N, 3)
            U_pts_rel   = torch.bmm(V_flat, R_flat)               # (E*N, 18, 3)
            U_local     = (U_pts_rel + pos_local.unsqueeze(1)).reshape(E * N, n_bez)  # (E*N, n_bez)

            window_refs = []
            for j in range(n_substeps):
                # ref_pos in env-local → add origin for world frame
                ref_pos_local = (U_local @ S_pos[j].T).reshape(E, N, 3)  # (E, N, 3)
                ref_vel_w     = (U_local @ S_vel[j].T).reshape(E, N, 3)
                ref_acc_w     = (U_local @ S_acc[j].T).reshape(E, N, 3)
                ref_pos_w     = ref_pos_local + env_origins.unsqueeze(1)  # (E, N, 3)
                window_refs.append(torch.cat([ref_pos_w, ref_vel_w, ref_acc_w], dim=-1))  # (E, N, 9)

        ref_9d      = window_refs[sub_step]   # (E, N, 9)
        action_flat = ref_9d.reshape(E, N * 9)
        _, reward, terminated, truncated, _ = env.step(action_flat)

        per_drone_obs = env.get_per_drone_obs()
        returns      += reward
        done          = terminated | truncated

        if done.any():
            success_count  += int(env._just_succeeded[done].sum().item())
            episode_count  += int(done.sum().item())
            finished_returns.extend(returns[done].detach().cpu().tolist())
            returns[done]   = 0.0
            sub_step        = 0
            window_refs     = None
            continue

        sub_step = (sub_step + 1) % n_substeps

    success_rate = success_count / max(episode_count, 1)
    mean_ret     = float(np.mean(finished_returns)) if finished_returns else float("nan")
    return {
        "eval_success_rate":  success_rate,
        "eval_return":        mean_ret,
        "eval_episodes":      episode_count,
        "eval_success_count": success_count,
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = Path(args_cli.checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(str(ckpt_path), map_location=device)

    n_bez       = int(ckpt.get("n_bez",        54))
    n_substeps  = int(ckpt.get("n_substeps",   args_cli.n_substeps))
    own_dim     = int(ckpt.get("own_dim",      37))
    neigh_dim   = int(ckpt.get("neighbor_dim",  9))
    hidden_dims = list(ckpt.get("hidden_dims", [256, 256, 256]))
    emb_dim     = int(ckpt.get("emb_dim",      64))
    samp_steps  = int(ckpt.get("sample_steps", 10))
    ctrl_max    = float(ckpt.get("ctrl_pts_max", 2.0))

    policy = OfflineDronePolicy(
        n_bez        = n_bez,
        hidden_dims  = hidden_dims,
        emb_dim      = emb_dim,
        sample_steps = samp_steps,
        device       = device,
        ctrl_pts_max = ctrl_max,
        own_dim      = own_dim,
        neighbor_dim = neigh_dim,
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    print(f"[Eval] loaded checkpoint: {ckpt_path}  epoch={ckpt.get('epoch', '?')}  "
          f"n_bez={n_bez}  substeps={n_substeps}", flush=True)

    # ── build env ────────────────────────────────────────────────────────────
    env_cfg = MultiDroneDmpcEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.num_drones     = args_cli.num_drones
    env_raw = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    assert isinstance(env_raw, MultiDroneDmpcEnv)
    N = env_raw.cfg.num_drones

    print(f"[Eval] env ready — {args_cli.num_envs} envs × {N} drones  "
          f"steps={args_cli.eval_steps}  substeps={n_substeps}", flush=True)

    # ── run ──────────────────────────────────────────────────────────────────
    metrics = evaluate(env_raw, policy, args_cli.eval_steps, N, n_substeps)

    print(
        f"[Eval] success_rate={metrics['eval_success_rate']:.3f}  "
        f"return={metrics['eval_return']:.2f}  "
        f"episodes={metrics['eval_episodes']}",
        flush=True,
    )

    # ── write output ─────────────────────────────────────────────────────────
    if args_cli.output_json:
        Path(args_cli.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args_cli.output_json, "w") as f:
            json.dump(metrics, f)
        print(f"[Eval] results written to {args_cli.output_json}", flush=True)
    else:
        print(json.dumps(metrics, indent=2))

    env_raw.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
