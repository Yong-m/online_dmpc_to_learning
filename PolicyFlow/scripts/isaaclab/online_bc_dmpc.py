"""online_bc_dmpc.py - Online BC with a DMPC expert and PolicyFlow's
flow-matching student for the multi-drone DMPC env.

This mirrors the *PolicyFlow* training stack used by
``online_bc_curobo.py`` -- a rectified-flow policy built from
``ContinuousNormalizingFlow`` + ``FlowMlp`` + ``EmpiricalNormalization`` --
adapted to a multi-drone setting with two specific design choices that follow
the paper "Online Trajectory Generation With Distributed Model Predictive
Control for Multi-Robot Motion Planning" (Luis et al., RA-L 2020):

1.  **Decentralised, parameter-shared student.** Each drone runs the same
    flow-matching network on its own per-drone slice of the observation. The
    env exposes ``get_per_drone_obs() -> (E, N, P)``; we reshape to
    ``(E*N, P)``, run the shared condition + flow network, and reshape the
    resulting per-drone action ``(E*N, A)`` (``A = 9`` -- normalised
    ``[delta_p, v_ref, a_ref]`` per drone) back to ``(E, N*A)`` before sending to the env. The
    same weights govern every drone, which is the natural permutation-
    equivariant choice for a homogeneous team.

2.  **Full PolicyFlow flow-matching head.** Action targets are ``atanh``-mapped
    to the latent unit cube, paired with Gaussian ``x0``, and the model
    regresses the rectified-flow velocity ``(x1 - x0)`` -- the same loss used
    throughout PolicyFlow. Sampling at rollout time is the Heun-style ODE
    integration of ``ContinuousNormalizingFlow.sample``. EMA / behaviour /
    proximal copies of the flow weights are kept around (matches the curobo
    script), making it easy to plug in PPO-EWMA / decoupled objectives later.

Pipeline per outer round:

  1. *Collect*. Roll out ``--steps_per_batch`` env steps. For every step we
     query the DMPC expert for a **world-frame position + velocity reference**
     for *every* drone in *every* env, convert it to the env's 9-D normalised
     reference action via :py:meth:`MultiDroneDmpcEnv.ref_to_action`, apply it,
     and store ``(per_drone_obs, atanh(per_drone_action))`` for every drone
     (each step contributes ``E * N`` BC samples).
  2. *Train*. ``--bc_epochs_per_round`` flow-matching updates on a uniform
     sample from the rolling per-drone buffer; loss is rectified-flow
     velocity-MSE.
  3. *Eval*. Every ``--eval_every_rounds`` rounds run a student rollout and
     log mean episode return + success rate.

Run from anywhere::

    python ~/git_branch/online_dmpc_to_learning/PolicyFlow/scripts/isaaclab/online_bc_dmpc.py \\
        --num_envs 32 --num_drones 4 \\
        --save_path runs/online_bc_dmpc/model.pt [--wandb]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

# Make the in-repo PolicyFlow + quadcopter packages importable without pip:
#   _POLICYFLOW_ROOT = online_dmpc_to_learning/PolicyFlow/policyflow/
#   _PROJECT_ROOT    = online_dmpc_to_learning/
_HERE = Path(__file__).resolve().parent
_POLICYFLOW_ROOT = _HERE.parent.parent / "policyflow"
_PROJECT_ROOT = _HERE.parent.parent.parent
for _p in (_POLICYFLOW_ROOT, _PROJECT_ROOT):
    p_str = str(_p)
    if _p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

parser = argparse.ArgumentParser(description="Online BC (flow-matching) with DMPC expert (multi-drone).")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_drones", type=int, default=4)
parser.add_argument("--task", type=str, default="Isaac-MultiDrone-DMPC-Direct-v0")
parser.add_argument("--seed", type=int, default=None) # Original default: 0

parser.add_argument("--n_rounds", type=int, default=200,
                    help="Total outer iterations (0 = run forever).")
parser.add_argument("--steps_per_batch", type=int, default=64,
                    help="Env steps per collection batch.")
parser.add_argument("--bc_epochs_per_round", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--buffer_capacity", type=int, default=400_000,
                    help="Max per-drone (obs, action_latent) transitions in the rolling buffer.")
parser.add_argument("--min_buffer_transitions", type=int, default=4_000,
                    help="Skip BC updates until the buffer has at least this many transitions.")

parser.add_argument("--eval_every_rounds", type=int, default=10)
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--expert_test_only", action="store_true", default=False,
                    help="Run DMPC expert only, skip replay-buffer collection and BC training.")
parser.add_argument("--expert_test_steps", type=int, default=None,
                    help="Expert-only rollout length. Defaults to 3 full episodes per env.")
parser.add_argument("--success_goal_tol", type=float, default=0.01,
                    help="Per-drone goal distance threshold for env-level success.")
parser.add_argument("--success_dwell_steps", type=int, default=10,
                    help="Consecutive steps all drones must stay within goal tolerance.")
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="Override env episode length in seconds for debug rollouts.")
parser.add_argument("--no_randomize_episode_start", action="store_true", default=False,
                    help="Start episodes at t=0 instead of randomizing episode_length_buf.")
parser.add_argument("--no_terminate_on_bounds", action="store_true", default=False,
                    help="Disable z-bound termination for fixed-target debug rollouts.")
parser.add_argument("--action_source", choices=["dmpc"], default="dmpc",
                    help="Compatibility flag for run_dmpc_logged_test.sh; master supports DMPC only.")
parser.add_argument("--dmpc_log_path", type=str, default=None,
                    help="Optional .npz path for first-env DMPC debug logging.")
parser.add_argument("--dmpc_log_every", type=int, default=1,
                    help="Save one DMPC debug sample every N expert steps.")

# Flow-matching / model.
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--hidden_dims", type=int, nargs="*", default=[256, 256, 256])
parser.add_argument("--emb_dim", type=int, default=64,
                    help="Both the time and condition embeddings live in this dim. "
                         "FlowMlp adds them, so they must share a size.")
parser.add_argument("--sample_steps", type=int, default=10,
                    help="Number of Heun ODE integration steps used for sampling.")
parser.add_argument("--action_clip", type=float, default=0.999,
                    help="Atanh-cap so the inverse mapping into latent space stays finite.")
parser.add_argument("--save_path", type=str, default="runs/online_bc_dmpc/model.pt")
parser.add_argument("--save_every_rounds", type=int, default=20)
parser.add_argument("--resume", type=str, default=None)

parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--wandb_project", type=str, default="online_bc_dmpc")
parser.add_argument("--wandb_run_name", type=str, default=None)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# if not getattr(args_cli, "headless", False) and not getattr(args_cli, "enable_cameras", False):
#     args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ─────────────────────────────────────────────────────
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import quadcopter  # noqa: F401, E402 - registers Isaac-MultiDrone-DMPC-Direct-v0
from quadcopter.multi_drone_dmpc_env import (  # noqa: E402
    MultiDroneDmpcEnv,
    MultiDroneDmpcEnvCfg,
    per_drone_obs_dim,
)
from quadcopter.dmpc_expert import DMPCExpert, DMPCParams  # noqa: E402

# Full PolicyFlow flow-matching stack (same imports as online_bc_curobo.py).
from policyflow_torch.modules import (  # noqa: E402
    ContinuousNormalizingFlow,
    ConditionMlp,
    FlowMlp,
)
from policyflow_torch.modules.normalizer import EmpiricalNormalization  # noqa: E402

try:
    import wandb as _wandb  # noqa: E402
    _WANDB_AVAILABLE = True
except Exception:
    _wandb = None
    _WANDB_AVAILABLE = False


# Per-drone action dimension exposed by ``MultiDroneDmpcEnv``: a normalised
# [delta_p_w, v_ref_w, a_ref_w] command in [-1, 1]^9. The env
# unpacks it before the low-level wrench controller.
PER_DRONE_ACTION_DIM = 9


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Decentralised per-drone flow-matching policy                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class SharedDronePolicy(nn.Module):
    """One PolicyFlow rectified-flow network applied independently per-drone.

    ``forward`` is not used at training time; the BC loop calls
    :meth:`flow_match_loss` directly, which produces the exact rectified-flow
    velocity-MSE loss used in ``online_bc_curobo.py``. At inference time
    :meth:`sample_action` integrates the Heun-style ODE owned by
    :class:`ContinuousNormalizingFlow` to draw a per-drone action.

    Per-drone obs (``P``) is empirically-normalised before either branch.
    """

    def __init__(
        self,
        per_drone_obs_dim: int,
        per_drone_action_dim: int,
        hidden_dims: list[int],
        emb_dim: int,
        sample_steps: int,
        device: torch.device,
    ):
        super().__init__()
        self.P = per_drone_obs_dim
        self.A = per_drone_action_dim
        self.device = device

        # Empirical normaliser on the per-drone obs (shape = P).
        self.obs_norm = EmpiricalNormalization(shape=self.P, until=int(1e8))

        # PolicyFlow building blocks.
        nn_condition = ConditionMlp(
            cond_dim=self.P,
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
        # ``ContinuousNormalizingFlow`` is not itself an nn.Module -- expose its
        # ModuleDicts under ``self`` so ``state_dict()`` / ``.to(device)`` /
        # ``.parameters()`` work transparently.
        self.cnf_model = self.cnf.model
        self.cnf_ema = self.cnf.model_ema
        self.cnf_last = self.cnf.model_last

    # ── training-side helpers ──────────────────────────────────────────────
    def update_obs_norm(self, per_drone_obs_flat: torch.Tensor) -> None:
        self.obs_norm.update(per_drone_obs_flat.detach())

    def flow_match_loss(
        self,
        obs_flat: torch.Tensor,        # (B, P)
        action_latent: torch.Tensor,   # (B, A) -- atanh'd target
    ) -> torch.Tensor:
        """Rectified-flow velocity-MSE on a flat per-drone batch.

        Matches the inner loop of ``run_bc_round`` in ``online_bc_curobo.py``.
        """
        obs_n = self.obs_norm(obs_flat)
        x1 = action_latent
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        cond_emb = self.cnf.model["condition"](obs_n)
        vel_pred = self.cnf.model["flow"](xt, t, cond_emb)
        vel_target = (x1 - x0).detach()
        return (vel_pred - vel_target).pow(2).mean()

    # ── inference: sample an action conditioned on per-drone obs ───────────
    @torch.no_grad()
    def sample_action(self, per_drone_obs: torch.Tensor) -> torch.Tensor:
        """Integrate the rectified-flow ODE to draw a per-drone action.

        Args:
            per_drone_obs: ``(E, N, P)``.

        Returns:
            ``(E, N, A)`` tanh-squashed action in ``[-1, 1]``.
        """
        E, N, P = per_drone_obs.shape
        flat = per_drone_obs.reshape(E * N, P)
        obs_n = self.obs_norm(flat)
        x0 = torch.randn(obs_n.shape[0], self.A, device=obs_n.device)
        latent, _ = self.cnf.sample(x0=x0, condition=obs_n, n_samples=obs_n.shape[0])
        return torch.tanh(latent).reshape(E, N, self.A)

    # ── PolicyFlow bookkeeping mirroring the curobo script ─────────────────
    def step_after_optim(self) -> None:
        """EMA update of the flow weights. Call after each optimiser step."""
        self.cnf.ema_update()

    def refresh_behaviour(self) -> None:
        """Snapshot the current weights into ``model_last`` so the next BC
        round trains against a fixed behaviour policy reference."""
        self.cnf.update()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Per-drone replay buffer                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class PerDroneBuffer:
    """Ring buffer of per-drone ``(obs, action_latent)`` pairs.

    Stores the *atanh-mapped* action so the BC loop can apply the rectified-
    flow loss without re-running atanh every batch.
    """

    def __init__(self, capacity: int, P: int, A: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, P), device=device)
        self.act_latent = torch.zeros((capacity, A), device=device)
        self.size = 0
        self.ptr = 0

    def add(self, obs_per_drone: torch.Tensor, act_latent_per_drone: torch.Tensor) -> None:
        flat_obs = obs_per_drone.reshape(-1, obs_per_drone.shape[-1]).to(self.device)
        flat_act = act_latent_per_drone.reshape(-1, act_latent_per_drone.shape[-1]).to(self.device)
        n = flat_obs.shape[0]
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.capacity
        self.obs[idx] = flat_obs
        self.act_latent[idx] = flat_act
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[idx], self.act_latent[idx]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Expert wrapper: DMPC position/velocity/acceleration → env action        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def expert_reference_acceleration(
    env: MultiDroneDmpcEnv,
    expert: DMPCExpert,
    ref_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Recover the acceleration sample matching the latest DMPC reference.

    ``DMPCExpert.compute`` currently returns position and velocity. It also
    keeps each agent's Bezier control points, so we sample the second
    derivative at the same substep used for the returned reference.
    """
    ref_acc_w = torch.zeros_like(ref_pos_w)
    h_total = (expert.p.k_hor - 1) * expert.p.h
    for e in range(env.num_envs):
        for i in range(env.cfg.num_drones):
            st = expert._state.get((e, i))
            if st is None:
                continue
            steps_before_sample = max(int(st["steps"]) - 1, 0)
            t_sub = ((steps_before_sample % expert.n_substeps) + 1) * expert.p.ts
            t_sub = min(t_sub, h_total)
            ref_acc = expert.bezier.sample_matrix(np.array([t_sub]), deriv=2) @ st["U"]
            ref_acc_w[e, i] = torch.from_numpy(ref_acc.astype(np.float32)).to(env.device)
    return ref_acc_w


def hover_action(
    env: MultiDroneDmpcEnv,
    debug_logger: "DmpcExpertLogger | None" = None,
    debug_step: int = 0,
) -> torch.Tensor:
    """Generate a hold-position command at the reset initial position."""
    states = env.get_world_states()
    ref_pos_w = env._init_pos_w.clone()
    ref_vel_w = torch.zeros_like(ref_pos_w)
    ref_acc_w = torch.zeros_like(ref_pos_w)
    action = env.ref_to_action(ref_pos_w, ref_vel_w, ref_acc_w)
    if debug_logger is not None:
        debug_logger.add(debug_step, states, ref_pos_w, ref_vel_w, ref_acc_w, action)
    return action


def greedy_action(
    env: MultiDroneDmpcEnv,
    speed: float,
    pos_gain: float,
    step: int,
    debug_logger: "DmpcExpertLogger | None" = None,
    debug_step: int = 0,
) -> torch.Tensor:
    """Generate a smooth minimum-jerk trajectory from reset position to goal.

    The quintic time scaling gives zero velocity and zero acceleration at
    both endpoints. ``speed`` bounds the peak speed approximately.
    ``pos_gain`` is kept for CLI compatibility and is not used here.
    """
    states = env.get_world_states()
    init_w = env._init_pos_w
    goal_w = states["goal_w"]
    delta_w = goal_w - init_w
    dist = torch.linalg.norm(delta_w, dim=-1, keepdim=True).clamp(min=1e-6)
    # For alpha(s)=10s^3-15s^4+6s^5, max alpha_dot is 1.875 at s=0.5.
    duration = (1.875 * dist / max(speed, 1e-6)).clamp(min=env.step_dt)
    t = torch.full_like(dist, float(step) * env.step_dt)
    s = (t / duration).clamp(0.0, 1.0)
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s
    alpha = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
    alpha_dot = (30.0 * s2 - 60.0 * s3 + 30.0 * s4) / duration
    alpha_ddot = (60.0 * s - 180.0 * s2 + 120.0 * s3) / (duration * duration)
    ref_pos_w = init_w + alpha * delta_w
    ref_vel_w = alpha_dot * delta_w
    ref_acc_w = alpha_ddot * delta_w
    action = env.ref_to_action(ref_pos_w, ref_vel_w, ref_acc_w)
    if debug_logger is not None:
        debug_logger.add(debug_step, states, ref_pos_w, ref_vel_w, ref_acc_w, action)
    return action


def _push_first_env_debug_trajectories(
    env: MultiDroneDmpcEnv,
    expert: DMPCExpert,
    states: dict[str, torch.Tensor],
) -> None:
    """Push first-env MPC horizons to Isaac Sim debug markers."""
    if not hasattr(env, "set_debug_trajectories"):
        return
    env_idx = 0
    N = env.cfg.num_drones
    K = expert.p.k_hor
    device = env.device
    origin = env._terrain.env_origins[env_idx].detach().cpu().numpy().astype(np.float64)
    pos_w = states["pos_w"][env_idx].detach().cpu().numpy().astype(np.float64)
    vel_w = states["lin_vel_w"][env_idx].detach().cpu().numpy().astype(np.float64)
    planned_np = np.full((N, K, 3), np.nan, dtype=np.float32)
    predicted_np = np.full((N, K, 3), np.nan, dtype=np.float32)
    collision_points: list[np.ndarray] = []
    for i in range(N):
        st = expert._state.get((env_idx, i))
        if st is None:
            continue
        U = st["U"]
        planned_np[i] = ((expert.M_pos_hor @ U).reshape(K, 3) + origin).astype(np.float32)
        x0_local = np.concatenate([pos_w[i] - origin, vel_w[i]])
        u_samples = expert.M_pos_hor @ U
        pred_stack = expert._A0_stack @ x0_local + expert._Lam_stack @ u_samples
        predicted_np[i] = (pred_stack.reshape(K, 6)[:, :3] + origin).astype(np.float32)
        for kc in st.get("collision_sample_indices", []):
            if 0 <= int(kc) < K:
                collision_points.append(planned_np[i, int(kc)])
    planned = torch.from_numpy(planned_np).to(device)
    predicted = torch.from_numpy(predicted_np).to(device)
    valid_plan = torch.isfinite(planned).all(dim=-1)
    valid_pred = torch.isfinite(predicted).all(dim=-1)
    n_short = max(1, min(int(getattr(env.cfg, "debug_short_horizon_steps", 3)), K))
    planned_short = planned[:, :n_short]
    predicted_short = predicted[:, :n_short]
    valid_plan_short = torch.isfinite(planned_short).all(dim=-1)
    valid_pred_short = torch.isfinite(predicted_short).all(dim=-1)
    planned_segments = []
    predicted_segments = []
    seg_ids = np.minimum((expert._times_hor / expert.bezier.T_seg).astype(np.int64), expert.bezier.l - 1)
    for seg_idx in range(expert.bezier.l):
        seg_indices = np.nonzero(seg_ids == seg_idx)[0].tolist()
        if not seg_indices:
            planned_segments.append(torch.empty(0, 1, 3, device=device))
            predicted_segments.append(torch.empty(0, 1, 3, device=device))
            continue
        planned_seg = planned[:, seg_indices]
        predicted_seg = predicted[:, seg_indices]
        valid_planned_seg = torch.isfinite(planned_seg).all(dim=-1)
        valid_predicted_seg = torch.isfinite(predicted_seg).all(dim=-1)
        planned_segments.append(planned_seg[valid_planned_seg].reshape(-1, 1, 3))
        predicted_segments.append(predicted_seg[valid_predicted_seg].reshape(-1, 1, 3))

    if collision_points:
        collision = torch.from_numpy(np.stack(collision_points, axis=0).astype(np.float32)).to(device)
    else:
        collision = torch.empty(0, 3, device=device)
    env.set_debug_trajectories(
        planned[valid_plan].reshape(-1, 1, 3),
        predicted[valid_pred].reshape(-1, 1, 3),
        planned_short[valid_plan_short].reshape(-1, 1, 3),
        predicted_short[valid_pred_short].reshape(-1, 1, 3),
        collision,
        planned_segment_pos_w=planned_segments,
        predicted_segment_pos_w=predicted_segments,
    )


def expert_action(
    env: MultiDroneDmpcEnv,
    expert: DMPCExpert,
    debug_logger: "DmpcExpertLogger | None" = None,
    debug_step: int = 0,
) -> torch.Tensor:
    """Run DMPC for every drone in every env and map its plan to the env action.

    DMPC returns the paper's ``u_i`` position-reference sample. We pack
    that sample, its first derivative, and its second derivative into the
    env's 9-D normalised ``[delta_p, v_ref, a_ref]`` action.

    Returns ``(num_envs, num_drones * 9)``.
    """
    states = env.get_world_states()
    pos_w = states["pos_w"]
    vel_w = states["lin_vel_w"]
    goal_w = states["goal_w"]
    origins = env._terrain.env_origins

    ref_pos_w, ref_vel_w = expert.plan(
        pos_w=pos_w, vel_w=vel_w, goal_w=goal_w, env_origins=origins,
    )
    ref_acc_w = expert_reference_acceleration(env, expert, ref_pos_w)
    _push_first_env_debug_trajectories(env, expert, states)
    action = env.ref_to_action(ref_pos_w, ref_vel_w, ref_acc_w)
    if debug_logger is not None:
        debug_logger.add(debug_step, states, ref_pos_w, ref_vel_w, ref_acc_w, action)
    return action


class DmpcExpertLogger:
    """Collect first-env DMPC expert traces and save them as compressed NumPy arrays."""

    def __init__(self, path: str, env: MultiDroneDmpcEnv, expert: DMPCExpert):
        self.path = path
        self.env = env
        self.expert = expert
        self.records: list[dict[str, np.ndarray | int]] = []

    def add(
        self,
        step: int,
        states: dict[str, torch.Tensor],
        ref_pos_w: torch.Tensor,
        ref_vel_w: torch.Tensor,
        ref_acc_w: torch.Tensor,
        action_flat: torch.Tensor,
    ) -> None:
        env_idx = 0
        N = self.env.cfg.num_drones
        K = self.expert.p.k_hor
        origin = self.env._terrain.env_origins[env_idx].detach().cpu().numpy().astype(np.float64)
        pos_w = states["pos_w"][env_idx].detach().cpu().numpy().astype(np.float64)
        vel_w = states["lin_vel_w"][env_idx].detach().cpu().numpy().astype(np.float64)
        goal_w = states["goal_w"][env_idx].detach().cpu().numpy().astype(np.float64)
        action_norm = action_flat.view(self.env.num_envs, N, PER_DRONE_ACTION_DIM)[env_idx]
        action_norm_np = action_norm.detach().cpu().numpy().astype(np.float64)
        desired_pos_w = pos_w + action_norm_np[:, 0:3] * float(self.env.cfg.delta_pos_max)
        desired_vel_w = action_norm_np[:, 3:6] * float(self.env.cfg.v_max)
        desired_acc_w = action_norm_np[:, 6:9] * float(self.env.cfg.accel_action_max)

        planned_ref_pos_w = np.full((N, K, 3), np.nan, dtype=np.float64)
        planned_ref_vel_w = np.full((N, K, 3), np.nan, dtype=np.float64)
        planned_ref_acc_w = np.full((N, K, 3), np.nan, dtype=np.float64)
        predicted_pos_w = np.full((N, K, 3), np.nan, dtype=np.float64)
        predicted_vel_w = np.full((N, K, 3), np.nan, dtype=np.float64)
        control_points = np.full((N, self.expert.n_bez), np.nan, dtype=np.float64)
        mpc_replanned = np.zeros(N, dtype=np.bool_)
        mpc_reset_mode = np.zeros(N, dtype=np.bool_)
        mpc_fallback = np.zeros(N, dtype=np.bool_)
        mpc_collision_timestep = np.full(N, -1, dtype=np.int32)
        mpc_collision_pos_w = np.full((N, 3), np.nan, dtype=np.float64)

        for i in range(N):
            st = self.expert._state.get((env_idx, i))
            if st is None:
                continue
            mpc_replanned[i] = bool(st.get("last_replanned", False))
            mpc_reset_mode[i] = bool(st.get("last_reset_mode", False))
            mpc_fallback[i] = bool(st.get("last_fallback", False))
            mpc_collision_timestep[i] = int(st.get("collision_timestep", -1))
            U = st["U"]
            control_points[i] = U
            planned_ref_pos_w[i] = (self.expert.M_pos_hor @ U).reshape(K, 3) + origin
            planned_ref_vel_w[i] = (self.expert.M_vel_hor @ U).reshape(K, 3)
            planned_ref_acc_w[i] = (self.expert.M_acc_hor @ U).reshape(K, 3)
            x0_local = np.concatenate([pos_w[i] - origin, vel_w[i]])
            u_samples = self.expert.M_pos_hor @ U
            pred_stack = self.expert._A0_stack @ x0_local + self.expert._Lam_stack @ u_samples
            pred_state = pred_stack.reshape(K, 6)
            predicted_pos_w[i] = pred_state[:, :3] + origin
            predicted_vel_w[i] = pred_state[:, 3:6]
            collision_indices = st.get("collision_sample_indices", [])
            if collision_indices:
                collision_idx = int(collision_indices[0])
                if 0 <= collision_idx < K:
                    mpc_collision_pos_w[i] = planned_ref_pos_w[i, collision_idx]

        ll = getattr(self.env, "_last_low_level_debug", {})
        ll_payload = {}
        for key in ("acc_cmd_w", "F_des_w", "z_wb", "z_des", "e_R", "angvel_b", "tau_des_b"):
            value = ll.get(key) if isinstance(ll, dict) else None
            if value is None:
                ll_payload["ll_" + key] = np.full((N, 3), np.nan, dtype=np.float32)
            else:
                ll_payload["ll_" + key] = value[env_idx].detach().cpu().numpy().astype(np.float32)
        thrust_value = ll.get("thrust_b_z") if isinstance(ll, dict) else None
        if thrust_value is None:
            ll_payload["ll_thrust_b_z"] = np.full((N,), np.nan, dtype=np.float32)
        else:
            ll_payload["ll_thrust_b_z"] = thrust_value[env_idx].detach().cpu().numpy().astype(np.float32)

        self.records.append(
            {
                **ll_payload,
                "step": int(step),
                "pos_w": pos_w.astype(np.float32),
                "vel_w": vel_w.astype(np.float32),
                "goal_w": goal_w.astype(np.float32),
                "ref_pos_w": ref_pos_w[env_idx].detach().cpu().numpy().astype(np.float32),
                "ref_vel_w": ref_vel_w[env_idx].detach().cpu().numpy().astype(np.float32),
                "ref_acc_w": ref_acc_w[env_idx].detach().cpu().numpy().astype(np.float32),
                "action_normalized": action_norm_np.astype(np.float32),
                "desired_pos_cmd_w": desired_pos_w.astype(np.float32),
                "desired_vel_cmd_w": desired_vel_w.astype(np.float32),
                "desired_acc_cmd_w": desired_acc_w.astype(np.float32),
                "planned_ref_pos_w": planned_ref_pos_w.astype(np.float32),
                "planned_ref_vel_w": planned_ref_vel_w.astype(np.float32),
                "planned_ref_acc_w": planned_ref_acc_w.astype(np.float32),
                "predicted_pos_w": predicted_pos_w.astype(np.float32),
                "predicted_vel_w": predicted_vel_w.astype(np.float32),
                "control_points": control_points.astype(np.float32),
                "mpc_replanned": mpc_replanned,
                "mpc_reset_mode": mpc_reset_mode,
                "mpc_fallback": mpc_fallback,
                "mpc_collision_timestep": mpc_collision_timestep,
                "mpc_collision_pos_w": mpc_collision_pos_w.astype(np.float32),
            }
        )

    def save(self) -> None:
        if not self.records:
            return
        out_dir = os.path.dirname(self.path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        keys = [k for k in self.records[0] if k != "step"]
        payload = {"step": np.asarray([r["step"] for r in self.records], dtype=np.int64)}
        for key in keys:
            payload[key] = np.stack([r[key] for r in self.records], axis=0)
        np.savez_compressed(self.path, **payload)


def action_to_latent(action: torch.Tensor, clip: float) -> torch.Tensor:
    """Map a tanh-squashed action in ``[-1, 1]^A`` to the unconstrained atanh
    latent the flow regresses against. Clipping keeps the inverse finite when
    the expert saturates the action box."""
    return torch.atanh(action.clamp(-clip, clip))

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
def run_expert_test(
    env: MultiDroneDmpcEnv,
    expert: DMPCExpert,
    n_steps: int,
    goal_tol: float,
    dwell_steps: int,
    debug_logger: DmpcExpertLogger | None = None,
    log_every: int = 1,
) -> dict[str, float]:
    """Run repeated DMPC-only episodes and aggregate env-level success metrics."""
    device = env.device
    env.reset(seed=args_cli.seed) if args_cli.seed is not None else env.reset()
    expert.reset(obstacle_info=env.get_obstacle_info())
    if hasattr(env, "_last_reset_env_ids"):
        env._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)
    last_episode_length_buf = env.episode_length_buf.clone()

    E = env.num_envs
    N = env.cfg.num_drones
    episode_returns = torch.zeros(E, device=device)
    episode_dwell = torch.zeros(E, dtype=torch.long, device=device)
    episode_success = torch.zeros(E, dtype=torch.bool, device=device)
    episode_drone_collision = torch.zeros(E, dtype=torch.bool, device=device)
    episode_obstacle_collision = torch.zeros(E, dtype=torch.bool, device=device)
    episode_bounds_failure = torch.zeros(E, dtype=torch.bool, device=device)
    episode_success_step = torch.full((E,), -1, dtype=torch.long, device=device)
    episode_start_step = torch.zeros(E, dtype=torch.long, device=device)

    total_episodes = 0
    total_success = 0
    total_clean_success = 0
    total_drone_collision = 0
    total_obstacle_collision = 0
    total_bounds_failure = 0
    total_terminated = 0
    total_truncated = 0
    completed_returns: list[float] = []
    completed_success_times: list[float] = []

    t0 = time.time()
    for step in range(n_steps):
        rewound = env.episode_length_buf < last_episode_length_buf
        if rewound.any():
            ids = rewound.nonzero(as_tuple=False).flatten()
            expert.reset(obstacle_info=env.get_obstacle_info(), env_ids=ids)
            episode_returns[ids] = 0.0
            episode_dwell[ids] = 0
            episode_success[ids] = False
            episode_drone_collision[ids] = False
            episode_obstacle_collision[ids] = False
            episode_bounds_failure[ids] = False
            episode_success_step[ids] = -1
            episode_start_step[ids] = step

        log_this_step = debug_logger is not None and log_every > 0 and step % log_every == 0
        action = expert_action(
            env, expert,
            debug_logger=debug_logger if log_this_step else None,
            debug_step=step,
        )
        _, reward, terminated, truncated, _ = env.step(action)
        episode_returns += reward

        st = env.get_world_states()
        pos_w = st["pos_w"]
        goal_dist = torch.linalg.norm(pos_w - st["goal_w"], dim=-1)
        all_at_goal = goal_dist.max(dim=-1).values < goal_tol
        episode_dwell = torch.where(all_at_goal, episode_dwell + 1, torch.zeros_like(episode_dwell))

        if N > 1:
            pair_diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
            pair_dist = torch.linalg.norm(pair_diff, dim=-1)
            eye = torch.eye(N, dtype=torch.bool, device=device)
            pair_dist = pair_dist.masked_fill(eye, float("inf"))
            episode_drone_collision |= pair_dist.amin(dim=(1, 2)) < env.cfg.rmin

        episode_obstacle_collision |= _static_obstacle_collision_mask(env, pos_w)
        z = pos_w[..., 2]
        episode_bounds_failure |= ((z < env.cfg.z_min) | (z > env.cfg.z_max)).any(dim=-1)

        goal_reached = episode_dwell >= dwell_steps
        newly_succeeded = goal_reached & ~episode_success
        episode_success_step[newly_succeeded] = step
        episode_success |= newly_succeeded

        natural_done = terminated | truncated
        success_done = newly_succeeded
        done = natural_done | success_done
        if done.any():
            done_ids = done.nonzero(as_tuple=False).flatten()
            total_episodes += int(done_ids.numel())
            clean_success = episode_success & ~(episode_drone_collision | episode_obstacle_collision | episode_bounds_failure | terminated)
            total_success += int(episode_success[done_ids].sum().item())
            total_clean_success += int(clean_success[done_ids].sum().item())
            total_drone_collision += int(episode_drone_collision[done_ids].sum().item())
            total_obstacle_collision += int(episode_obstacle_collision[done_ids].sum().item())
            total_bounds_failure += int(episode_bounds_failure[done_ids].sum().item())
            total_terminated += int(terminated[done_ids].sum().item())
            total_truncated += int(truncated[done_ids].sum().item())
            completed_returns.extend(episode_returns[done_ids].detach().cpu().tolist())
            success_done_ids = done_ids[episode_success[done_ids]]
            success_ids = success_done_ids.detach().cpu().tolist()
            done_ids_list = done_ids.detach().cpu().tolist()
            print(
                f"[expert_test] episode_end step={step} done_env_ids={done_ids_list} "
                f"success_env_ids={success_ids}",
                flush=True,
            )
            if success_done_ids.numel() > 0:
                success_times = (episode_success_step[success_done_ids] - episode_start_step[success_done_ids]).float() * env.step_dt
                completed_success_times.extend(success_times.detach().cpu().tolist())

        reset_env_ids = getattr(env, "_last_reset_env_ids", None)
        if reset_env_ids is not None and reset_env_ids.numel() > 0:
            expert.reset(obstacle_info=env.get_obstacle_info(), env_ids=reset_env_ids)
            env._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)
            ids = reset_env_ids
        elif natural_done.any():
            ids = natural_done.nonzero(as_tuple=False).flatten()
            expert.reset(obstacle_info=env.get_obstacle_info(), env_ids=ids)
        else:
            ids = None

        manual_success_ids = success_done.nonzero(as_tuple=False).flatten()
        if manual_success_ids.numel() > 0:
            env._reset_idx(manual_success_ids)
            env.scene.write_data_to_sim()
            env.sim.forward()
            expert.reset(obstacle_info=env.get_obstacle_info(), env_ids=manual_success_ids)
            if hasattr(env, "_last_reset_env_ids"):
                env._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)
            ids = manual_success_ids if ids is None else torch.unique(torch.cat([ids, manual_success_ids]))

        if ids is not None and ids.numel() > 0:
            episode_returns[ids] = 0.0
            episode_dwell[ids] = 0
            episode_success[ids] = False
            episode_drone_collision[ids] = False
            episode_obstacle_collision[ids] = False
            episode_bounds_failure[ids] = False
            episode_success_step[ids] = -1
            episode_start_step[ids] = step + 1

        last_episode_length_buf = env.episode_length_buf.clone()

    active_incomplete = int((~episode_success & ~episode_drone_collision & ~episode_obstacle_collision & ~episode_bounds_failure).sum().item())
    denom = max(total_episodes, 1)
    metrics = {
        "expert_success_rate": total_success / denom,
        "expert_clean_success_rate": total_clean_success / denom,
        "expert_drone_collision_rate": total_drone_collision / denom,
        "expert_obstacle_collision_rate": total_obstacle_collision / denom,
        "expert_bounds_failure_rate": total_bounds_failure / denom,
        "expert_terminated_rate": total_terminated / denom,
        "expert_truncated_rate": total_truncated / denom,
        "expert_incomplete_envs": float(active_incomplete),
        "expert_completed_episodes": float(total_episodes),
        "expert_mean_return": float(np.mean(completed_returns)) if completed_returns else float("nan"),
        "expert_mean_time_to_success_s": float(np.mean(completed_success_times)) if completed_success_times else float("nan"),
        "expert_runtime_s": time.time() - t0,
        "expert_num_envs": float(E),
        "expert_steps": float(n_steps),
    }
    print(
        "[expert_test] "
        f"episodes={total_episodes}  "
        f"success={metrics['expert_success_rate']:.3f}  "
        f"clean_success={metrics['expert_clean_success_rate']:.3f}  "
        f"drone_col={metrics['expert_drone_collision_rate']:.3f}  "
        f"obs_col={metrics['expert_obstacle_collision_rate']:.3f}  "
        f"bounds={metrics['expert_bounds_failure_rate']:.3f}  "
        f"incomplete_envs={active_incomplete}  "
        f"mean_t_success={metrics['expert_mean_time_to_success_s']:.2f}s",
        flush=True,
    )
    return metrics

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Main training loop                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main():
    if args_cli.seed is not None:
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)

    env_cfg = MultiDroneDmpcEnvCfg()
    env_cfg.num_drones = args_cli.num_drones
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    elif args_cli.expert_test_only:
        env_cfg.episode_length_s = 60.0
    if args_cli.no_randomize_episode_start and hasattr(env_cfg, "randomize_episode_start"):
        env_cfg.randomize_episode_start = False
    if args_cli.no_terminate_on_bounds and hasattr(env_cfg, "terminate_on_bounds"):
        env_cfg.terminate_on_bounds = False
    env_cfg.__post_init__()
    if getattr(args_cli, "device", None):
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped  # type: MultiDroneDmpcEnv
    
    if args_cli.seed is not None:
        env.reset(seed=args_cli.seed)
    else:
        env.reset()

    device = env.device
    E = env.num_envs
    N = env.cfg.num_drones
    P = per_drone_obs_dim(N)
    A = PER_DRONE_ACTION_DIM
    print(
        f"[online_bc_dmpc] num_envs={env.num_envs}  num_drones={N}  "
        f"per_drone_obs={P}  per_drone_action={A}  "
        f"obs_total={env.cfg.observation_space}  action_total={env.cfg.action_space}"
    )

    policy = SharedDronePolicy(
        per_drone_obs_dim=P,
        per_drone_action_dim=A,
        hidden_dims=args_cli.hidden_dims,
        emb_dim=args_cli.emb_dim,
        sample_steps=args_cli.sample_steps,
        device=device,
    ).to(device)
    if args_cli.resume is not None and os.path.isfile(args_cli.resume):
        policy.load_state_dict(torch.load(args_cli.resume, map_location=device))
        print(f"[online_bc_dmpc] loaded checkpoint from {args_cli.resume}")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args_cli.lr)

    buffer = PerDroneBuffer(args_cli.buffer_capacity, P, A, device)

    # The DMPC expert distinguishes the MPC replanning period ``h`` (default
    # 0.1 s, as in the paper) from the control / subsample rate ``ts``. We
    # bind ``ts`` to the env step (sim.dt * decimation) so every env step the
    # planner emits one sample of the current Bezier, and leave ``h`` at its
    # paper default so a full QP solve only fires every ``h / ts`` env steps.
    expert = DMPCExpert(
        num_drones=N,
        num_envs=E,
        params=DMPCParams(
            pmin=env.cfg.pos_min,
            pmax=env.cfg.pos_max,
            rmin=env.cfg.rmin,
            ts=env.cfg.sim.dt * env.cfg.decimation,
        ),
        device=device,
    )

    dmpc_logger = (
        DmpcExpertLogger(args_cli.dmpc_log_path, env, expert)
        if args_cli.dmpc_log_path is not None
        else None
    )
    if dmpc_logger is not None:
        print(f"[online_bc_dmpc] logging first-env DMPC traces to {args_cli.dmpc_log_path}")

    use_wandb = args_cli.wandb and _WANDB_AVAILABLE
    if use_wandb:
        _wandb.init(
            project=args_cli.wandb_project,
            name=args_cli.wandb_run_name,
            config=vars(args_cli),
        )


    expert.reset(obstacle_info=env.get_obstacle_info())
    if args_cli.expert_test_only:
        n_test_steps = args_cli.expert_test_steps if args_cli.expert_test_steps is not None else 3 * int(env.max_episode_length)
        try:
            run_expert_test(
                env=env,
                expert=expert,
                n_steps=n_test_steps,
                goal_tol=args_cli.success_goal_tol,
                dwell_steps=args_cli.success_dwell_steps,
                debug_logger=dmpc_logger,
                log_every=args_cli.dmpc_log_every,
            )
        finally:
            if dmpc_logger is not None:
                dmpc_logger.save()
                print(f"[online_bc_dmpc] saved DMPC log to {args_cli.dmpc_log_path}", flush=True)
            env.close()
            simulation_app.close()
        return

    if hasattr(env, "_last_reset_env_ids"):
        env._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)
    last_episode_length_buf = env.episode_length_buf.clone()
    per_drone_obs = env.get_per_drone_obs()  # (E, N, P)

    recent_returns: deque[float] = deque(maxlen=20)
    episode_returns = torch.zeros(env.num_envs, device=device)
    round_idx = 0
    total_env_steps = 0
    expert_collect_step = 0
    n_rounds = args_cli.n_rounds if args_cli.n_rounds > 0 else 10_000_000

    save_dir = os.path.dirname(args_cli.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    try:
        while round_idx < n_rounds:
            # ── 1. Collection ────────────────────────────────────────────
            t_collect = time.time()

            collect_returns: list[float] = []
            for _ in range(args_cli.steps_per_batch):
                with torch.no_grad():
                    # Keep the DMPC expert in lockstep with IsaacLab resets.
                    # Normally the done branch below handles this, but this
                    # also catches full/manual resets or env counters rewound
                    # before this expert query.
                    rewound = env.episode_length_buf < last_episode_length_buf
                    if rewound.any():
                        expert.reset(obstacle_info=env.get_obstacle_info(),
                                     env_ids=rewound.nonzero(as_tuple=False).flatten())
                    last_episode_length_buf = env.episode_length_buf.clone()

                    log_this_step = (
                        dmpc_logger is not None
                        and args_cli.dmpc_log_every > 0
                        and expert_collect_step % args_cli.dmpc_log_every == 0
                    )
                    action_flat = expert_action(
                        env, expert,
                        debug_logger=dmpc_logger if log_this_step else None,
                        debug_step=expert_collect_step,
                    )  # (E, N*A)

                action_per_drone = action_flat.view(env.num_envs, N, A)
                act_latent = action_to_latent(action_per_drone, args_cli.action_clip)

                # Update obs stats from the live rollout distribution and add to
                # the per-drone buffer (E*N samples per env step).
                policy.update_obs_norm(per_drone_obs.reshape(-1, P).detach())
                buffer.add(per_drone_obs.detach(), act_latent.detach())

                _, reward, terminated, truncated, _ = env.step(action_flat)
                per_drone_obs = env.get_per_drone_obs()
                episode_returns += reward
                done = terminated | truncated
                reset_env_ids = getattr(env, "_last_reset_env_ids", None)
                if reset_env_ids is not None and reset_env_ids.numel() > 0:
                    expert.reset(obstacle_info=env.get_obstacle_info(),
                                 env_ids=reset_env_ids)
                    env._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)
                elif done.any():
                    expert.reset(obstacle_info=env.get_obstacle_info(),
                                 env_ids=done.nonzero(as_tuple=False).flatten())
                if done.any():
                    finished = episode_returns[done].detach().cpu().tolist()
                    collect_returns.extend(finished)
                    episode_returns[done] = 0.0
                last_episode_length_buf = env.episode_length_buf.clone()
                total_env_steps += env.num_envs
                expert_collect_step += 1
            collect_dt = time.time() - t_collect
            if collect_returns:
                recent_returns.extend(collect_returns)

            # ── 2. Flow-matching BC training ─────────────────────────────
            bc_loss = float("nan")
            if buffer.size >= max(args_cli.min_buffer_transitions, args_cli.batch_size):
                losses = []
                for _ in range(args_cli.bc_epochs_per_round):
                    b_obs, b_lat = buffer.sample(args_cli.batch_size)
                    loss = policy.flow_match_loss(b_obs, b_lat)
                    optimizer.zero_grad()
                    loss.backward()
                    if args_cli.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), args_cli.grad_clip)
                    optimizer.step()
                    policy.step_after_optim()
                    losses.append(loss.item())
                policy.refresh_behaviour()
                bc_loss = float(np.mean(losses))

            # ── 3. Eval ─────────────────────────────────────────────────
            eval_metrics: dict[str, float] = {}
            if args_cli.eval_every_rounds > 0 and round_idx % args_cli.eval_every_rounds == 0:
                eval_metrics = evaluate(env, policy, args_cli.eval_steps, N, A)
                env.reset()
                expert.reset(obstacle_info=env.get_obstacle_info())
                if hasattr(env, "_last_reset_env_ids"):
                    env._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)
                last_episode_length_buf = env.episode_length_buf.clone()
                per_drone_obs = env.get_per_drone_obs()
                episode_returns.zero_()

            # ── 4. Logging + checkpointing ──────────────────────────────
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
        if dmpc_logger is not None:
            dmpc_logger.save()
            print(f"[online_bc_dmpc] saved DMPC log to {args_cli.dmpc_log_path}", flush=True)
        env.close()
        simulation_app.close()


@torch.no_grad()
def evaluate(
    env: MultiDroneDmpcEnv,
    policy: SharedDronePolicy,
    n_steps: int,
    N: int,
    A: int,
) -> dict[str, float]:
    """Deterministic student rollout. Returns mean per-episode return and the
    fraction of envs whose drones all ended within 0.3 m of their goals."""
    device = env.device
    env.reset()
    per_drone_obs = env.get_per_drone_obs()
    returns = torch.zeros(env.num_envs, device=device)
    finished_returns: list[float] = []
    print("Evaluation")
    for _ in range(n_steps):
        action_pd = policy.sample_action(per_drone_obs)             # (E, N, A)
        action_flat = action_pd.reshape(env.num_envs, N * A)        # (E, N*A)
        _, reward, terminated, truncated, _ = env.step(action_flat)
        per_drone_obs = env.get_per_drone_obs()
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
