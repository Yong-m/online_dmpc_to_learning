#!/usr/bin/env python3
"""Plot first-env DMPC expert logs saved by online_bc_dmpc.py.

Example:
    python plot_dmpc_log.py runs/online_bc_dmpc/dmpc_debug.npz --agent 0 --out dmpc_debug.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description="Plot DMPC expert planning/tracking log.")
parser.add_argument("log_path", type=Path, help=".npz file produced by --dmpc_log_path")
parser.add_argument("--agent", type=int, default=0, help="Drone index in the first env to plot.")
parser.add_argument("--horizon_stride", type=int, default=10,
                    help="Plot every Nth planned/predicted horizon in the 3D panel.")
parser.add_argument("--out", type=Path, default=None, help="Optional image path. If omitted, show interactively.")
args = parser.parse_args()


data = np.load(args.log_path)
step = data["step"]
pos = data["pos_w"][:, args.agent]
vel = data["vel_w"][:, args.agent]
goal = data["goal_w"][:, args.agent]
ref_pos = data["ref_pos_w"][:, args.agent]
ref_vel = data["ref_vel_w"][:, args.agent]
cmd_pos = data["desired_pos_cmd_w"][:, args.agent] if "desired_pos_cmd_w" in data else ref_pos
cmd_vel = data["desired_vel_cmd_w"][:, args.agent]
cmd_acc = data["desired_acc_cmd_w"][:, args.agent] if "desired_acc_cmd_w" in data else None
planned = data["planned_ref_pos_w"][:, args.agent]
predicted = data["predicted_pos_w"][:, args.agent]
mpc_replanned = data["mpc_replanned"][:, args.agent].astype(bool) if "mpc_replanned" in data else None
mpc_reset_mode = data["mpc_reset_mode"][:, args.agent].astype(bool) if "mpc_reset_mode" in data else None
mpc_fallback = data["mpc_fallback"][:, args.agent].astype(bool) if "mpc_fallback" in data else None
mpc_collision_timestep = data["mpc_collision_timestep"][:, args.agent].astype(int) if "mpc_collision_timestep" in data else None
mpc_collision_pos = data["mpc_collision_pos_w"][:, args.agent] if "mpc_collision_pos_w" in data else None

err_goal = np.linalg.norm(goal - pos, axis=-1)
err_ref = np.linalg.norm(ref_pos - pos, axis=-1)

fig = plt.figure(figsize=(16, 14), constrained_layout=True)
grid = fig.add_gridspec(4, 2)
ax3d = fig.add_subplot(grid[:, 0], projection="3d")
ax_pos = fig.add_subplot(grid[0, 1])
ax_vel = fig.add_subplot(grid[1, 1])
ax_acc = fig.add_subplot(grid[2, 1])
ax_ll = fig.add_subplot(grid[3, 1])

ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="black", lw=2.0, label="actual state")
ax3d.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], color="tab:blue", lw=1.5, label="emitted ref sample")
ax3d.scatter(goal[0, 0], goal[0, 1], goal[0, 2], marker="*", s=150, color="tab:green", label="initial goal")
ax3d.scatter(goal[-1, 0], goal[-1, 1], goal[-1, 2], marker="X", s=80, color="tab:red", label="final goal")

stride = max(1, args.horizon_stride)
for idx in range(0, len(step), stride):
    alpha = 0.2 + 0.6 * idx / max(1, len(step) - 1)
    ax3d.plot(planned[idx, :, 0], planned[idx, :, 1], planned[idx, :, 2],
              color="tab:orange", alpha=alpha, lw=1.0)
    ax3d.plot(predicted[idx, :, 0], predicted[idx, :, 1], predicted[idx, :, 2],
              color="tab:purple", alpha=alpha, lw=1.0, ls="--")
ax3d.plot([], [], [], color="tab:orange", lw=1.0, label="planned ref horizon")
ax3d.plot([], [], [], color="tab:purple", lw=1.0, ls="--", label="predicted state horizon")
if mpc_collision_pos is not None:
    valid_collision_pos = np.isfinite(mpc_collision_pos).all(axis=-1)
    if valid_collision_pos.any():
        ax3d.scatter(mpc_collision_pos[valid_collision_pos, 0],
                     mpc_collision_pos[valid_collision_pos, 1],
                     mpc_collision_pos[valid_collision_pos, 2],
                     marker="x", s=55, color="red", linewidths=1.8,
                     label="collision-predicted pos")
ax3d.set_title(f"First env, drone {args.agent}: planning and tracking")
ax3d.set_xlabel("x [m]")
ax3d.set_ylabel("y [m]")
ax3d.set_zlabel("z [m]")
ax3d.legend(loc="best")

labels = ["x", "y", "z"]
colors = ["tab:red", "tab:green", "tab:blue"]
for dim, (label, color) in enumerate(zip(labels, colors)):
    ax_pos.plot(step, pos[:, dim], color=color, lw=2.0, label=f"actual {label}")
    ax_pos.plot(step, ref_pos[:, dim], color=color, lw=1.2, ls="--", label=f"expert ref {label}")
    ax_pos.plot(step, cmd_pos[:, dim], color=color, lw=1.0, ls="-.", label=f"env cmd pos {label}")
    ax_pos.plot(step, goal[:, dim], color=color, lw=0.9, ls=":", label=f"goal {label}")
if mpc_collision_pos is not None:
    valid_collision_pos = np.isfinite(mpc_collision_pos).all(axis=-1)
    if valid_collision_pos.any():
        collision_markers = ["x", "+", "1"]
        for dim, (label, color) in enumerate(zip(labels, colors)):
            ax_pos.scatter(
                step[valid_collision_pos],
                mpc_collision_pos[valid_collision_pos, dim],
                color=color, marker=collision_markers[dim], s=42, linewidths=1.4,
                label=f"collision pos {label}", zorder=5,
            )
if mpc_replanned is not None:
    repl_steps = step[mpc_replanned]
    reset_steps = step[mpc_reset_mode] if mpc_reset_mode is not None else np.asarray([], dtype=step.dtype)
    fallback_steps = step[mpc_fallback] if mpc_fallback is not None else np.asarray([], dtype=step.dtype)
    for n, s in enumerate(repl_steps):
        ax_pos.axvline(s, color="tab:gray", lw=0.7, alpha=0.25, label="MPC replan" if n == 0 else None)
    for n, s in enumerate(reset_steps):
        ax_pos.axvline(s, color="tab:red", lw=1.1, alpha=0.5, label="MPC reset-mode" if n == 0 else None)
    for n, s in enumerate(fallback_steps):
        ax_pos.axvline(s, color="tab:purple", lw=1.2, alpha=0.55, label="MPC fallback" if n == 0 else None)
collision_pos_valid = None
if mpc_collision_pos is not None:
    collision_pos_valid = np.isfinite(mpc_collision_pos).all(axis=-1)
    collision_pos_steps = step[collision_pos_valid]
    for n, s in enumerate(collision_pos_steps):
        ax_pos.axvline(s, color="red", lw=1.0, alpha=0.35, label="collision pos logged" if n == 0 else None)
if mpc_collision_timestep is not None:
    ca_steps = step[mpc_collision_timestep >= 0]
    for n, s in enumerate(ca_steps):
        ax_pos.axvline(s, color="tab:orange", lw=1.0, alpha=0.45, label="CA detected" if n == 0 else None)
ax_pos_twin = ax_pos.twinx()
ax_pos_twin.plot(step, err_goal, color="black", lw=1.5, alpha=0.7, label="||goal-pos||")
ax_pos_twin.plot(step, err_ref, color="tab:gray", lw=1.2, alpha=0.8, label="||ref-pos||")
ax_pos.set_title("Position tracking")
ax_pos.set_xlabel("collection step")
ax_pos.set_ylabel("position [m]")
ax_pos_twin.set_ylabel("error [m]")
lines, names = ax_pos.get_legend_handles_labels()
lines2, names2 = ax_pos_twin.get_legend_handles_labels()
ax_pos.legend(lines + lines2, names + names2, ncols=2, fontsize=8, loc="best")
ax_pos.grid(True, alpha=0.3)

for dim, (label, color) in enumerate(zip(labels, colors)):
    ax_vel.plot(step, vel[:, dim], color=color, lw=2.0, label=f"actual vel {label}")
    ax_vel.plot(step, cmd_vel[:, dim], color=color, lw=1.2, ls="--", label=f"cmd vel {label}")
    ax_vel.plot(step, ref_vel[:, dim], color=color, lw=0.9, ls=":", label=f"ref vel {label}")
ax_vel.set_title("Velocity command and tracking")
ax_vel.set_xlabel("collection step")
ax_vel.set_ylabel("velocity [m/s]")
ax_vel.grid(True, alpha=0.3)
ax_vel.legend(ncols=3, fontsize=8, loc="best")

if cmd_acc is not None:
    for dim, (label, color) in enumerate(zip(labels, colors)):
        ax_acc.plot(step, cmd_acc[:, dim], color=color, lw=1.6, label=f"cmd acc {label}")
    ax_acc.set_title("Env-side desired acceleration")
else:
    ax_acc.text(0.5, 0.5, "desired_acc_cmd_w not present in this log", ha="center", va="center")
    ax_acc.set_title("Env-side desired acceleration")
ax_acc.set_xlabel("collection step")
ax_acc.set_ylabel("acceleration [m/s^2]")
ax_acc.grid(True, alpha=0.3)
ax_acc.legend(ncols=3, fontsize=8, loc="best")

if "ll_thrust_b_z" in data:
    thrust = data["ll_thrust_b_z"][:, args.agent]
    tau = data["ll_tau_des_b"][:, args.agent]
    e_r = data["ll_e_R"][:, args.agent]
    ang = data["ll_angvel_b"][:, args.agent]
    ax_ll.plot(step, thrust, color="black", lw=1.7, label="thrust z [N]")
    for dim, (label, color) in enumerate(zip(labels, colors)):
        ax_ll.plot(step, tau[:, dim], color=color, lw=1.3, label=f"tau {label} [Nm]")
        ax_ll.plot(step, e_r[:, dim], color=color, lw=0.9, ls="--", label=f"e_R {label}")
        # ax_ll.plot(step, ang[:, dim], color=color, lw=0.8, ls=":", label=f"omega {label}")
    ax_ll.set_title("Low-level control debug")
else:
    ax_ll.text(0.5, 0.5, "low-level debug fields not present in this log", ha="center", va="center")
    ax_ll.set_title("Low-level control debug")
if mpc_collision_timestep is not None:
    ax_kc = ax_ll.twinx()
    kc_plot = np.where(mpc_collision_timestep >= 0, mpc_collision_timestep, np.nan)
    ax_kc.step(step, kc_plot, where="post", color="tab:orange", lw=1.6, label="collision kc")
    ax_kc.scatter(step[mpc_collision_timestep >= 0], mpc_collision_timestep[mpc_collision_timestep >= 0],
                  color="tab:orange", s=18, zorder=5)
    ax_kc.set_ylabel("collision horizon kc")
    lines_kc, names_kc = ax_kc.get_legend_handles_labels()
    lines_ll, names_ll = ax_ll.get_legend_handles_labels()
    ax_ll.legend(lines_ll + lines_kc, names_ll + names_kc, ncols=3, fontsize=7, loc="best")
else:
    ax_ll.legend(ncols=3, fontsize=7, loc="best")

ax_ll.set_xlabel("collection step")
ax_ll.grid(True, alpha=0.3)

if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"saved {args.out}")
else:
    plt.show()
