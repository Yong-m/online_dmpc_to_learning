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
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--n_rounds", type=int, default=200,
                    help="Total outer iterations (0 = run forever).")
parser.add_argument("--steps_per_batch", type=int, default=200,
                    help="Env steps per collection batch.")
parser.add_argument("--bc_epochs_per_round", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--buffer_capacity", type=int, default=400_000,
                    help="Max per-drone (obs, action_latent) transitions in the rolling buffer.")
parser.add_argument("--min_buffer_transitions", type=int, default=4_000,
                    help="Skip BC updates until the buffer has at least this many transitions.")

parser.add_argument("--eval_every_rounds", type=int, default=2)
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="Override env episode length in seconds for debug rollouts.")
parser.add_argument("--no_terminate_on_bounds", action="store_true", default=False,
                    help="Disable z-bound termination for fixed-target debug rollouts.")
parser.add_argument("--action_source", choices=["dmpc"], default="dmpc",
                    help="Compatibility flag for run_dmpc_logged_test.sh; master supports DMPC only.")
parser.add_argument("--dmpc_log_path", type=str, default=None,
                    help="Optional .npz path for first-env DMPC debug logging.")
parser.add_argument("--dmpc_log_every", type=int, default=1,
                    help="Save one DMPC debug sample every N expert steps.")

parser.add_argument("--traj_save_dir", type=str, default=None,
                    help="Directory to save successful DMPC expert trajectories. "
                         "If omitted, trajectories are not saved to disk.")
parser.add_argument("--traj_load_dir", type=str, default=None,
                    help="Directory of previously saved trajectories to pre-populate "
                         "the BC buffer at startup (direct obs/act_latent load, no env).")
parser.add_argument("--traj_replay_dir", type=str, default=None,
                    help="Like --traj_load_dir but re-runs trajectories in the env to "
                         "regenerate obs (use when obs definition has changed).")
parser.add_argument("--traj_max_load", type=int, default=None,
                    help="Cap on the number of trajectory files to load/replay.")

# Flow-matching / model.
parser.add_argument("--lr", type=float, default=5e-4)
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
parser.add_argument("--save_every_rounds", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)

parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--wandb_project", type=str, default="online_bc_dmpc")
parser.add_argument("--wandb_run_name", type=str, default=None)

parser.add_argument("--video", action="store_true", default=False,
                    help="Record collection videos (one mp4 per round).")
parser.add_argument("--video_length", type=int, default=0,
                    help="Video clip length in steps (0 = steps_per_batch).")
parser.add_argument("--video_root", type=str, default="runs/online_bc_dmpc/videos",
                    help="Output directory for videos.")
parser.add_argument("--video_resolution", type=int, nargs=2, default=[1280, 720],
                    help="Recording resolution (W H).")
parser.add_argument("--cam_eye", type=float, nargs=3, default=[8.0, 8.0, 6.0],
                    help="Viewer camera position (m), world frame.")
parser.add_argument("--cam_lookat", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                    help="Viewer camera lookat (m), world frame.")

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
import torch.nn.functional as F  # noqa: E402

import quadcopter  # noqa: F401, E402 - registers Isaac-MultiDrone-DMPC-Direct-v0
from quadcopter.multi_drone_dmpc_env import (  # noqa: E402
    MultiDroneDmpcEnv,
    MultiDroneDmpcEnvCfg,
    per_drone_obs_dim,
    PER_DRONE_OWN_DIM,
    PER_NEIGHBOUR_DIM,
)
from quadcopter.dmpc_expert import DMPCExpert, DMPCParams  # noqa: E402

# Full PolicyFlow flow-matching stack (same imports as online_bc_curobo.py).
from policyflow_torch.modules import (  # noqa: E402
    ContinuousNormalizingFlow,
    ConditionMlp,
    FlowMlp,
    NeighborEncoder,
)
from policyflow_torch.modules.normalizer import EmpiricalNormalization  # noqa: E402

try:
    import wandb as _wandb  # noqa: E402
    _WANDB_AVAILABLE = True
except Exception:
    _wandb = None
    _WANDB_AVAILABLE = False


# Per-drone action dimension: wrench [f_z, tau_x, tau_y, tau_z] in body frame.
PER_DRONE_ACTION_DIM = 3  # goal-aligned v_ref [vx, vy, vz] normalised to [-1, 1]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Decentralised per-drone flow-matching policy                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class SharedDronePolicy(nn.Module):
    """Per-drone flow-matching policy with a cross-attention neighbor encoder.

    Architecture:
        per_drone_obs (P) → [own_norm | neigh_norm] → NeighborEncoder
                         → [own(OWN_DIM) ‖ neighbor_embed(emb_dim)]
                         → ConditionMlp → ContinuousNormalizingFlow → v_ref(A=3)

    Own-state and neighbor observations are normalised with separate
    EmpiricalNormalization instances (each of fixed dimension), so the policy
    generalises across different values of N without retraining the normaliser.
    The NeighborEncoder uses cross-attention (own as query, neighbors as
    keys/values) to produce a fixed-size embedding regardless of N.
    """

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
    ):
        super().__init__()
        self.P = per_drone_obs_dim
        self.A = per_drone_action_dim
        self.own_dim = own_dim
        self.neighbor_dim = neighbor_dim
        self.device = device

        # Separate normalizers so stats are independent of N.
        self.own_norm = EmpiricalNormalization(shape=own_dim, until=int(1e8))
        self.neigh_norm = EmpiricalNormalization(shape=neighbor_dim, until=int(1e8))
        # Wrench action normalizer: empirical z-score for [f_z, tau_x, tau_y, tau_z].
        self.act_norm = EmpiricalNormalization(shape=per_drone_action_dim, until=int(1e8))

        # Cross-attention: own attends over variable-length neighbor set.
        self.neighbor_enc = NeighborEncoder(
            own_dim=own_dim,
            neighbor_dim=neighbor_dim,
            emb_dim=emb_dim,
            num_heads=num_attn_heads,
        )

        # Condition net receives [own(own_dim) ‖ neighbor_embed(emb_dim)].
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
        # Expose CNF ModuleDicts so state_dict() / .parameters() see them.
        self.cnf_model = self.cnf.model
        self.cnf_ema = self.cnf.model_ema
        self.cnf_last = self.cnf.model_last
        # Proximal copy for PPO-EWMA (same as curobo). Updated after each
        # gradient step via update_proximal(); acts as trust-region anchor.
        self.cnf.init_proximal(beta_prox=0.97)

    # ── obs encoding ───────────────────────────────────────────────────────
    def _encode_obs(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """Normalize and encode a flat per-drone obs batch.

        Args:
            obs_flat: ``(B, P)`` raw (un-normalized) per-drone observations.

        Returns:
            ``(B, own_dim + emb_dim)`` condition input for the flow net.
        """
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

    # ── training-side helpers ──────────────────────────────────────────────
    def update_obs_norm(self, per_drone_obs_flat: torch.Tensor) -> None:
        """Update running obs normalizer stats from a flat per-drone obs batch."""
        own_raw = per_drone_obs_flat[:, : self.own_dim].detach()
        neigh_raw = per_drone_obs_flat[:, self.own_dim :].detach()
        self.own_norm.update(own_raw)
        N_neigh = neigh_raw.shape[1] // self.neighbor_dim
        if N_neigh > 0:
            self.neigh_norm.update(neigh_raw.reshape(-1, self.neighbor_dim))

    def update_act_norm(self, wrench_flat: torch.Tensor) -> None:
        """Update running action (wrench) normalizer stats."""
        self.act_norm.update(wrench_flat.detach())

    def encode_action(self, wrench_flat: torch.Tensor) -> torch.Tensor:
        """Normalize raw wrench (B, A) → latent (B, A) via z-score."""
        return self.act_norm(wrench_flat)

    def decode_action(self, latent_flat: torch.Tensor) -> torch.Tensor:
        """Denormalize latent (B, A) → raw wrench (B, A)."""
        return self.act_norm.inverse(latent_flat)

    def flow_match_loss(
        self,
        obs_flat: torch.Tensor,    # (B, P) raw per-drone obs
        action_raw: torch.Tensor,  # (B, A=4) raw wrench [f_z, tau_x, tau_y, tau_z]
    ) -> torch.Tensor:
        """Rectified-flow velocity-MSE loss for wrench prediction.

        Wrench is z-score normalized before being used as the flow target x1,
        so x1 is approximately N(0,1) — ideal for rectified flow.
        """
        self.cnf.model["condition"].train()
        self.cnf.model["flow"].train()
        cond_in = self._encode_obs(obs_flat)
        x1 = self.encode_action(action_raw)  # normalize wrench → latent
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        cond_emb = self.cnf.model["condition"](cond_in)
        vel_pred = self.cnf.model["flow"](xt, t, cond_emb)
        vel_target = (x1 - x0).detach()
        return (vel_pred - vel_target).pow(2).mean()

    # ── inference ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample_action(self, per_drone_obs: torch.Tensor) -> torch.Tensor:
        """Integrate the rectified-flow ODE and decode to raw wrench.

        Args:
            per_drone_obs: ``(E, N, P)`` raw observations.

        Returns:
            ``(E, N, A=4)`` raw wrench ``[f_z, tau_x, tau_y, tau_z]`` (body frame).
        """
        E, N, P = per_drone_obs.shape
        flat = per_drone_obs.reshape(E * N, P)
        self.cnf.eval()
        cond_in = self._encode_obs(flat)
        x0 = torch.randn(cond_in.shape[0], self.A, device=cond_in.device)
        latent, _ = self.cnf.sample(x0=x0, condition=cond_in, n_samples=cond_in.shape[0])
        wrench = self.decode_action(latent)  # denormalize latent → raw wrench
        self.cnf.train()
        return wrench.reshape(E, N, self.A)

    # ── PolicyFlow bookkeeping ─────────────────────────────────────────────
    def step_after_optim(self) -> None:
        """Update proximal model after each gradient step (curobo pattern)."""
        self.cnf.update_proximal()

    def refresh_behaviour(self) -> None:
        """Sync model_last ← model (needed when compute_flow_variation is used)."""
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
# ║ Per-env episode buffer (filter truncated episodes before main buffer)    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class PerEnvEpisodeBuffer:
    """Accumulates per-env (obs, act_latent) transitions and flushes them to
    the main ``PerDroneBuffer`` only when an episode ends with
    ``terminated=True`` (success or crash).  Truncated episodes (timeout) are
    silently discarded, keeping the training buffer free of time-cut data.

    Memory: ``num_envs × max_ep_len × N × (P+A) × 4 bytes``.  For typical
    settings (E=32, T=200, N=4, P=48, A=9) this is about 6 MB.
    """

    def __init__(
        self,
        num_envs: int,
        max_ep_len: int,
        N: int,
        P: int,
        A: int,
        device: torch.device,
    ):
        self._E = num_envs
        self._T = max_ep_len
        self._device = device
        # Pre-allocate; extra steps beyond max_ep_len are silently dropped.
        self._obs = torch.zeros(num_envs, max_ep_len, N, P, device=device)
        self._act = torch.zeros(num_envs, max_ep_len, N, A, device=device)
        self._len = torch.zeros(num_envs, dtype=torch.long, device=device)

    def push(self, obs: torch.Tensor, act_latent: torch.Tensor) -> None:
        """Append one step for every env.

        Args:
            obs:        ``(E, N, P)`` raw per-drone observations.
            act_latent: ``(E, N, A)`` atanh-mapped expert actions.
        """
        E = self._E
        idx = self._len.clamp(max=self._T - 1)   # (E,)
        env_idx = torch.arange(E, device=self._device)
        self._obs[env_idx, idx] = obs
        self._act[env_idx, idx] = act_latent
        self._len = (self._len + 1).clamp(max=self._T)

    def flush(
        self,
        main_buffer: PerDroneBuffer,
        env_ids: torch.Tensor,
        terminated: torch.Tensor,
        drone_success: torch.Tensor | None = None,
    ) -> int:
        """Flush done envs.

        Args:
            main_buffer:   Destination ring buffer.
            env_ids:       1-D tensor of env indices that are done.
            terminated:    Bool tensor of same length — used when ``drone_success``
                           is None; only terminated envs are flushed.
            drone_success: Optional ``(len(env_ids), N)`` bool tensor.  When
                           provided, ``terminated`` is ignored and each drone is
                           flushed independently based on its individual success.
                           Drones where the mask is False are silently skipped.

        Returns:
            Number of per-drone transitions flushed.
        """
        flushed = 0
        for i, e in enumerate(env_ids.tolist()):
            t = int(self._len[e].item())
            if t == 0:
                self._len[e] = 0
                continue
            if drone_success is not None:
                dmask = drone_success[i]          # (N,) bool
                good = dmask.nonzero(as_tuple=False).flatten()
                if good.numel() > 0:
                    main_buffer.add(
                        self._obs[e, :t][:, good],    # (T, k, P)
                        self._act[e, :t][:, good],    # (T, k, A)
                    )
                    flushed += t * good.numel()
            elif terminated[i]:
                main_buffer.add(self._obs[e, :t], self._act[e, :t])
                flushed += t * self._obs.shape[2]  # t * N
            self._len[e] = 0
        return flushed

    def clear(self, env_ids: torch.Tensor) -> None:
        """Discard episodes for the given envs without flushing (e.g. external reset)."""
        self._len[env_ids] = 0


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
    # Convert world-frame v_ref to goal-aligned 3D action for the v_ref cascade.
    action = env.vref_to_action(ref_vel_w)
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
        desired_vel_w = action_norm_np[:, 0:3] * float(self.env.cfg.v_max)

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
                "desired_vel_cmd_w": desired_vel_w.astype(np.float32),
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


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Success trajectory recorder / replayer                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class SuccessTrajectoryRecorder:
    """Records successful DMPC expert episodes to disk.

    Per episode saves (in a single compressed .npz):
      - init_pos_local (N, 3): drone start positions relative to env origin (XY)
      - goal_local     (N, 3): goal positions relative to env origin (XY)
      - act_latent     (T, N, A): atanh-mapped goal-frame v_ref sequence
      - obs            (T, N, P): per-drone observations at each step

    Local-frame storage makes files portable across different env tile layouts.
    """

    def __init__(
        self,
        save_dir: str,
        num_envs: int,
        max_ep_len: int,
        N: int,
        P: int,
        A: int,
        device: torch.device,
    ):
        self._dir = Path(save_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._E, self._T = num_envs, max_ep_len
        self._N, self._P, self._A = N, P, A
        self._device = device

        self._obs = torch.zeros(num_envs, max_ep_len, N, P, device=device)
        self._act = torch.zeros(num_envs, max_ep_len, N, A, device=device)
        self._init_pos = torch.zeros(num_envs, N, 3, device=device)
        self._goal     = torch.zeros(num_envs, N, 3, device=device)
        self._len = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._traj_count = int(
            max((int(f.stem.split("_")[1]) for f in self._dir.glob("traj_*.npz")),
                default=-1) + 1
        )

    def push(
        self,
        obs: torch.Tensor,           # (E, N, P)
        act_latent: torch.Tensor,    # (E, N, A)
        env_raw: "MultiDroneDmpcEnv",
    ) -> None:
        """Stage one step. Captures initial conditions on the first step of each episode."""
        new_ep = self._len == 0
        if new_ep.any():
            new_ids = new_ep.nonzero(as_tuple=False).flatten()
            origins = env_raw._terrain.env_origins[new_ids]  # (n, 3)
            ip = env_raw._init_pos_w[new_ids].detach().clone()
            gp = env_raw._goal_pos_w[new_ids].detach().clone()
            ip[..., :2] -= origins[:, None, :2]
            gp[..., :2] -= origins[:, None, :2]
            self._init_pos[new_ids] = ip
            self._goal[new_ids] = gp

        idx = self._len.clamp(max=self._T - 1)
        env_idx = torch.arange(self._E, device=self._device)
        self._obs[env_idx, idx] = obs.detach()
        self._act[env_idx, idx] = act_latent.detach()
        self._len = (self._len + 1).clamp(max=self._T)

    def save_successes(
        self,
        env_ids: torch.Tensor,
        just_succeeded: torch.Tensor,  # bool, same length as env_ids
    ) -> int:
        """Flush successful episodes to disk. Returns the number saved."""
        saved = 0
        for e, succ in zip(env_ids.tolist(), just_succeeded.tolist()):
            if not succ:
                continue
            t = int(self._len[e].item())
            if t == 0:
                continue
            np.savez_compressed(
                str(self._dir / f"traj_{self._traj_count:06d}.npz"),
                obs=self._obs[e, :t].cpu().numpy().astype(np.float32),
                act_latent=self._act[e, :t].cpu().numpy().astype(np.float32),
                init_pos_local=self._init_pos[e].cpu().numpy().astype(np.float32),
                goal_local=self._goal[e].cpu().numpy().astype(np.float32),
            )
            self._traj_count += 1
            saved += 1
        return saved

    def clear(self, env_ids: torch.Tensor) -> None:
        self._len[env_ids] = 0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Replay pool — replaces DMPC for a fraction of envs                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class ReplayPool:
    """Dedicate the last ``num_envs // replay_denom`` envs to replaying saved
    trajectories, reducing DMPC solver invocations.

    Lifecycle:
      - Activates once the trajectory directory contains ≥ ``num_envs`` files.
      - On activation (and after each episode end), loads the next trajectory
        (round-robin) and pins the env to its initial conditions.
      - Each call to ``step(env_id)`` returns the pre-recorded action for that
        step.  When the trajectory runs out (episode still in progress), returns
        ``(None, None)`` so the DMPC action takes over for the remainder.
    """

    def __init__(
        self,
        num_envs: int,
        traj_dir: str,
        action_clip: float,
        device: torch.device,
        replay_denom: int = 5,
    ):
        self.n_replay = max(1, num_envs // replay_denom)
        # Use the highest-indexed envs as replay envs (env 0 is kept for debug).
        self.replay_ids: set[int] = set(range(num_envs - self.n_replay, num_envs))
        self._dir = Path(traj_dir)
        self.action_clip = action_clip
        self.device = device
        self._min_trajs = num_envs             # activate when ≥ num_envs files exist
        self._files: list[Path] = []
        self._rr_idx = 0                       # round-robin file pointer
        self._state: dict[int, dict] = {}      # env_id → {actions, act_latent, step, T}
        self.active = False
        self._just_activated = False

    def refresh(self) -> int:
        """Rescan traj_dir.  Returns file count.  Sets ``active`` + ``_just_activated``
        the first time enough files are available."""
        self._files = sorted(self._dir.glob("traj_*.npz"))
        n = len(self._files)
        if not self.active and n >= self._min_trajs:
            self.active = True
            self._just_activated = True
            print(
                f"[ReplayPool] activated — {n} trajectories, "
                f"{self.n_replay} replay envs: {sorted(self.replay_ids)}",
                flush=True,
            )
        return n

    def is_replay_env(self, env_id: int) -> bool:
        return self.active and env_id in self.replay_ids

    def load_next(self, env_id: int, env_raw: "MultiDroneDmpcEnv") -> None:
        """Load the next trajectory (round-robin) and pin ``env_id`` to its
        initial conditions via ``env_raw._pinned_reset_state``."""
        if not self._files:
            return
        f = self._files[self._rr_idx % len(self._files)]
        self._rr_idx = (self._rr_idx + 1) % len(self._files)

        data = np.load(str(f))
        act_lat = torch.from_numpy(data["act_latent"]).to(self.device)   # (T, N, A)
        actions = torch.tanh(act_lat / self.action_clip) * self.action_clip

        origin = env_raw._terrain.env_origins[env_id]   # (3,)
        ip = torch.from_numpy(data["init_pos_local"]).to(self.device).clone()
        gp = torch.from_numpy(data["goal_local"]).to(self.device).clone()
        ip[..., :2] += origin[:2]
        gp[..., :2] += origin[:2]
        env_raw._pinned_reset_state[env_id] = {"init_pos": ip, "goal": gp}

        self._state[env_id] = {
            "actions": actions,
            "act_latent": act_lat,
            "step": 0,
            "T": act_lat.shape[0],
        }

    def step(
        self, env_id: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Advance one step.  Returns (action (N,A), act_latent (N,A)) or
        (None, None) when the trajectory is exhausted (falls back to DMPC)."""
        s = self._state.get(env_id)
        if s is None or s["step"] >= s["T"]:
            return None, None
        t = s["step"]
        s["step"] += 1
        return s["actions"][t], s["act_latent"][t]

    def on_episode_done(
        self,
        env_id: int,
        env_raw: "MultiDroneDmpcEnv",
    ) -> None:
        """Called when a replay env's episode ends.  Loads the next trajectory
        and force-resets the env to its initial conditions (double-reset)."""
        self._state.pop(env_id, None)
        self.load_next(env_id, env_raw)
        # _pinned_reset_state is set; apply it now with an explicit _reset_idx call.
        env_raw._reset_idx(torch.tensor([env_id], device=env_raw.device))


def load_trajs_to_buffer(
    traj_dir: str,
    buffer: "PerDroneBuffer",
    device: torch.device,
    max_trajs: int | None = None,
) -> int:
    """Load saved trajectory files and add (obs, act_latent) pairs to the BC buffer.

    Does NOT require the env — feeds saved data directly.  Use this when the
    observation definition has not changed since the trajectories were saved.
    """
    files = sorted(Path(traj_dir).glob("traj_*.npz"))
    if max_trajs is not None:
        files = files[:max_trajs]
    loaded = 0
    for f in files:
        data = np.load(str(f))
        obs = torch.from_numpy(data["obs"]).to(device)         # (T, N, P)
        act = torch.from_numpy(data["act_latent"]).to(device)  # (T, N, A)
        buffer.add(obs, act)
        loaded += 1
    print(f"[traj_load] loaded {loaded} trajectories from {traj_dir}", flush=True)
    return loaded


def replay_trajs_in_env(
    traj_dir: str,
    env_raw: "MultiDroneDmpcEnv",
    buffer: "PerDroneBuffer",
    action_clip: float,
    max_trajs: int | None = None,
) -> int:
    """Replay saved trajectories inside the running env to regenerate fresh obs.

    For each saved trajectory:
      1. Pin env 0 to the saved initial positions / goal.
      2. Reset env 0 via _reset_idx.
      3. Step through the saved actions and collect (new_obs, act_latent) pairs.

    This is useful when the observation definition has changed since the
    trajectories were saved, since it regenerates obs from the current code.
    All other envs continue running normally.
    """
    files = sorted(Path(traj_dir).glob("traj_*.npz"))
    if max_trajs is not None:
        files = files[:max_trajs]
    if not files:
        return 0

    device = env_raw.device
    E = env_raw.num_envs
    N = env_raw.cfg.num_drones
    A = PER_DRONE_ACTION_DIM
    replayed = 0

    for traj_file in files:
        data = np.load(str(traj_file))
        init_local = torch.from_numpy(data["init_pos_local"]).to(device)  # (N, 3)
        goal_local  = torch.from_numpy(data["goal_local"]).to(device)      # (N, 3)
        act_lat     = torch.from_numpy(data["act_latent"]).to(device)      # (T, N, A)
        T = act_lat.shape[0]

        # Decode latent → normalised v_ref for env stepping.
        actions = torch.tanh(act_lat / action_clip) * action_clip  # (T, N, A)

        # Pin env 0 to saved initial conditions (world frame = local + origin XY).
        origin = env_raw._terrain.env_origins[0]
        ip = init_local.clone(); ip[..., :2] += origin[:2]
        gp = goal_local.clone(); gp[..., :2] += origin[:2]
        env_raw._pinned_reset_state[0] = {"init_pos": ip, "goal": gp}
        env_raw._reset_idx(torch.tensor([0], device=device))

        per_drone_obs = env_raw.get_per_drone_obs()  # (E, N, P)

        for t in range(T):
            # Broadcast trajectory action to all envs (only env 0 is pinned,
            # others run freely but we only collect from env 0).
            act_t = actions[t].unsqueeze(0).expand(E, -1, -1)   # (E, N, A)
            act_flat = act_t.reshape(E, N * A)
            env_raw.step(act_flat)
            new_obs = env_raw.get_per_drone_obs()

            # Add env-0 step to buffer using the *pre-step* obs.
            buffer.add(per_drone_obs[:1], act_lat[t : t + 1])   # (1, N, P/A)
            per_drone_obs = new_obs

        replayed += 1
        print(f"  [replay] {traj_file.name}  T={T}", flush=True)

    print(f"[traj_replay] replayed {replayed} trajectories", flush=True)
    return replayed


def action_to_latent(action: torch.Tensor, clip: float) -> torch.Tensor:
    """Map a normalised action to the flow latent — curobo-style encoding.

    Forward (inference):  action = tanh(latent / clip) * clip
    Inverse (this fn):    latent = atanh(action / clip) * clip

    Both must use the same clip value.  For the drone env action ∈ [-1,1],
    use clip ≈ 1 (default 0.999) to stay within the invertible range.
    """
    safe_cap = 0.999999
    scaled = (action / max(clip, 1e-6)).clamp(-safe_cap, safe_cap)
    return torch.atanh(scaled) * clip


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Main training loop                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = MultiDroneDmpcEnvCfg()
    env_cfg.num_drones = args_cli.num_drones
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.no_terminate_on_bounds and hasattr(env_cfg, "terminate_on_bounds"):
        env_cfg.terminate_on_bounds = False
    env_cfg.__post_init__()
    if getattr(args_cli, "device", None):
        env_cfg.sim.device = args_cli.device

    if args_cli.video:
        env_cfg.viewer.eye = tuple(args_cli.cam_eye)
        env_cfg.viewer.lookat = tuple(args_cli.cam_lookat)
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.resolution = tuple(args_cli.video_resolution)

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        os.makedirs(args_cli.video_root, exist_ok=True)
        _vid_len = int(args_cli.video_length) if args_cli.video_length > 0 else args_cli.steps_per_batch
        _sps = args_cli.steps_per_batch
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.video_root,
            step_trigger=lambda step, _sps=_sps: step % _sps == 0,
            video_length=_vid_len,
            disable_logger=True,
            name_prefix="dmpc_collect",
        )
    env_raw = env.unwrapped  # type: MultiDroneDmpcEnv
    device = env_raw.device
    N = env_raw.cfg.num_drones
    P = per_drone_obs_dim(N)
    A = PER_DRONE_ACTION_DIM
    print(
        f"[online_bc_dmpc] num_envs={env_raw.num_envs}  num_drones={N}  "
        f"per_drone_obs={P}  per_drone_action={A}  "
        f"obs_total={env_raw.cfg.observation_space}  action_total={env_raw.cfg.action_space}"
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
        params=DMPCParams(
            pmin=env_raw.cfg.pos_min,
            pmax=env_raw.cfg.pos_max,
            rmin=env_raw.cfg.rmin,
            ts=env_raw.cfg.sim.dt * env_raw.cfg.decimation,
        ),
        device=device,
    )

    dmpc_logger = (
        DmpcExpertLogger(args_cli.dmpc_log_path, env_raw, expert)
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

    env.reset(seed=args_cli.seed)
    expert.reset()
    if hasattr(env_raw, "_last_reset_env_ids"):
        env_raw._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)
    last_episode_length_buf = env_raw.episode_length_buf.clone()
    per_drone_obs = env_raw.get_per_drone_obs()  # (E, N, P)

    max_ep_len = int(env_raw.max_episode_length) + 10
    ep_buf = PerEnvEpisodeBuffer(env_raw.num_envs, max_ep_len, N, P, A, device)

    # Optional trajectory recorder + replay pool.
    traj_save_dir = args_cli.traj_save_dir
    traj_recorder: SuccessTrajectoryRecorder | None = None
    replay_pool: ReplayPool | None = None
    if traj_save_dir is not None:
        traj_recorder = SuccessTrajectoryRecorder(
            traj_save_dir, env_raw.num_envs, max_ep_len, N, P, A, device
        )
        replay_pool = ReplayPool(
            num_envs=env_raw.num_envs,
            traj_dir=traj_save_dir,
            action_clip=args_cli.action_clip,
            device=device,
        )
        print(
            f"[online_bc_dmpc] traj_save_dir={traj_save_dir}  "
            f"replay_pool: {replay_pool.n_replay}/{env_raw.num_envs} envs "
            f"(activates after {replay_pool._min_trajs} saved trajectories)"
        )

    # Pre-populate BC buffer from saved trajectories before training begins.
    if args_cli.traj_load_dir is not None:
        load_trajs_to_buffer(args_cli.traj_load_dir, buffer, device, args_cli.traj_max_load)
    if args_cli.traj_replay_dir is not None:
        replay_trajs_in_env(
            args_cli.traj_replay_dir, env_raw, buffer,
            args_cli.action_clip, args_cli.traj_max_load,
        )
        # Re-sync env state after replay.
        env.reset()
        expert.reset()
        if hasattr(env_raw, "_last_reset_env_ids"):
            env_raw._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)
        last_episode_length_buf = env_raw.episode_length_buf.clone()
        per_drone_obs = env_raw.get_per_drone_obs()

    recent_returns: deque[float] = deque(maxlen=20)
    episode_returns = torch.zeros(env_raw.num_envs, device=device)
    collect_success_count = 0
    collect_episode_count = 0
    round_idx = 0
    total_env_steps = 0
    expert_collect_step = 0
    n_rounds = args_cli.n_rounds if args_cli.n_rounds > 0 else 10_000_000

    save_dir = os.path.dirname(args_cli.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    _stem = os.path.splitext(args_cli.save_path)
    best_model_path = _stem[0] + "_best" + _stem[1]
    best_eval_success = -1.0

    try:
        while round_idx < n_rounds:
            # ── 1. Collection ────────────────────────────────────────────
            t_collect = time.time()

            # Activate replay pool once enough trajectories have been saved.
            if replay_pool is not None:
                replay_pool.refresh()
                if replay_pool._just_activated:
                    replay_pool._just_activated = False
                    for _rid in sorted(replay_pool.replay_ids):
                        replay_pool.load_next(_rid, env_raw)
                        env_raw._reset_idx(torch.tensor([_rid], device=device))
                        expert.reset(torch.tensor([_rid], device=device))
                        ep_buf.clear(torch.tensor([_rid], device=device))
                        if traj_recorder is not None:
                            traj_recorder.clear(torch.tensor([_rid], device=device))
                    per_drone_obs = env_raw.get_per_drone_obs()

            collect_returns: list[float] = []
            for _collect_step in range(args_cli.steps_per_batch):
                print(
                    f"  [collect {_collect_step + 1:>4d}/{args_cli.steps_per_batch}]",
                    end="\r", flush=True,
                )
                with torch.no_grad():
                    # Keep the DMPC expert in lockstep with IsaacLab resets.
                    # Normally the done branch below handles this, but this
                    # also catches full/manual resets or env counters rewound
                    # before this expert query.
                    rewound = env_raw.episode_length_buf < last_episode_length_buf
                    if rewound.any():
                        expert.reset(rewound.nonzero(as_tuple=False).flatten())
                    last_episode_length_buf = env_raw.episode_length_buf.clone()

                    log_this_step = (
                        dmpc_logger is not None
                        and args_cli.dmpc_log_every > 0
                        and expert_collect_step % args_cli.dmpc_log_every == 0
                    )
                    action_flat = expert_action(
                        env_raw, expert,
                        debug_logger=dmpc_logger if log_this_step else None,
                        debug_step=expert_collect_step,
                    )  # (E, N*A)

                obs_before = per_drone_obs  # obs at time t

                # Update obs normaliser stats before stepping.
                policy.update_obs_norm(obs_before.reshape(-1, P).detach())

                _, reward, terminated, truncated, _ = env.step(action_flat)

                # action_flat is goal-aligned v_ref ∈ [-1,1]^3 per drone.
                action_per_drone = action_flat.view(env_raw.num_envs, N, A).detach()
                policy.update_act_norm(action_per_drone.reshape(-1, A))
                ep_buf.push(obs_before.detach(), action_per_drone)
                if traj_recorder is not None:
                    traj_recorder.push(obs_before.detach(), action_per_drone, env_raw)
                per_drone_obs = env_raw.get_per_drone_obs()
                episode_returns += reward
                done = terminated | truncated
                reset_env_ids = getattr(env_raw, "_last_reset_env_ids", None)
                if reset_env_ids is not None and reset_env_ids.numel() > 0:
                    expert.reset(reset_env_ids)
                    ep_buf.clear(reset_env_ids)
                    if traj_recorder is not None:
                        traj_recorder.clear(reset_env_ids)
                    env_raw._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)
                if done.any():
                    done_ids = done.nonzero(as_tuple=False).flatten()
                    expert.reset(done_ids)
                    # Per-drone flush: use individual drone success signal so that
                    # even partial successes (not all drones) contribute training data.
                    drone_succ = env_raw._drone_just_succeeded[done_ids]  # (M, N)
                    ep_buf.flush(buffer, done_ids, terminated[done_ids],
                                 drone_success=drone_succ)
                    # Save successful episodes (skip replay envs — would create duplicates).
                    if traj_recorder is not None:
                        _not_replay = torch.tensor(
                            [not (replay_pool is not None
                                  and replay_pool.is_replay_env(int(e)))
                             for e in done_ids.tolist()],
                            dtype=torch.bool, device=device,
                        )
                        traj_recorder.save_successes(
                            done_ids,
                            env_raw._just_succeeded[done_ids] & _not_replay,
                        )
                        traj_recorder.clear(done_ids)
                    # For replay envs: double-reset to next pinned trajectory.
                    _replay_reset_any = False
                    for _eid in done_ids.tolist():
                        if replay_pool is not None and replay_pool.is_replay_env(_eid):
                            replay_pool.on_episode_done(_eid, env_raw)
                            expert.reset(torch.tensor([_eid], device=device))
                            ep_buf.clear(torch.tensor([_eid], device=device))
                            if traj_recorder is not None:
                                traj_recorder.clear(torch.tensor([_eid], device=device))
                            _replay_reset_any = True
                    if _replay_reset_any:
                        # Refresh obs so replay envs' initial-state obs are correct.
                        per_drone_obs = env_raw.get_per_drone_obs()
                if done.any():
                    collect_success_count += int(env_raw._just_succeeded[done].sum().item())
                    collect_episode_count += int(done.sum().item())
                    finished = episode_returns[done].detach().cpu().tolist()
                    collect_returns.extend(finished)
                    episode_returns[done] = 0.0
                last_episode_length_buf = env_raw.episode_length_buf.clone()
                total_env_steps += env_raw.num_envs
                expert_collect_step += 1
            print()  # end \r overwrite
            collect_dt = time.time() - t_collect

            # Flush in-progress episodes (spanning multiple batches) without
            # per-drone filtering — this keeps data flowing early in training
            # when few drones have succeeded yet.  Per-drone filtering applies
            # only to completed episodes (done handling above).
            in_progress = (ep_buf._len > 0).nonzero(as_tuple=False).flatten()
            if in_progress.numel() > 0:
                ep_buf.flush(
                    buffer,
                    in_progress,
                    torch.ones(in_progress.numel(), dtype=torch.bool, device=device),
                )

            if collect_returns:
                recent_returns.extend(collect_returns)

            # ── 2. Flow-matching BC training ─────────────────────────────
            bc_loss = float("nan")
            if buffer.size >= max(args_cli.min_buffer_transitions, args_cli.batch_size):
                losses = []
                for epoch_i in range(args_cli.bc_epochs_per_round):
                    b_obs, b_lat = buffer.sample(args_cli.batch_size)
                    loss = policy.flow_match_loss(b_obs, b_lat)  # sets train mode internally
                    optimizer.zero_grad()
                    loss.backward()
                    if args_cli.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), args_cli.grad_clip)
                    optimizer.step()
                    policy.step_after_optim()  # update_proximal (curobo pattern)
                    losses.append(loss.item())
                    print(
                        f"  [BC epoch {epoch_i + 1}/{args_cli.bc_epochs_per_round}] "
                        f"loss={loss.item():.5f}",
                        flush=True,
                    )
                # Sync model_last for future compute_flow_variation / PPO-EWMA use.
                policy.refresh_behaviour()
                bc_loss = float(np.mean(losses))

            # ── 3. Eval ─────────────────────────────────────────────────
            eval_metrics: dict[str, float] = {}
            if args_cli.eval_every_rounds > 0 and round_idx % args_cli.eval_every_rounds == 0:
                eval_metrics = evaluate(env_raw, policy, args_cli.eval_steps, N, A)
                env.reset()
                expert.reset()
                if hasattr(env_raw, "_last_reset_env_ids"):
                    env_raw._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)
                last_episode_length_buf = env_raw.episode_length_buf.clone()
                per_drone_obs = env_raw.get_per_drone_obs()
                episode_returns.zero_()

            # ── 4. Logging + checkpointing ──────────────────────────────
            expert_success_rate = collect_success_count / max(collect_episode_count, 1)
            collect_success_count = 0
            collect_episode_count = 0

            mean_ret = float(np.mean(recent_returns)) if recent_returns else float("nan")
            msg = (
                f"[round {round_idx:04d}] "
                f"buf={buffer.size:>7d}  "
                f"bc_loss={bc_loss:.4f}  "
                f"expert_succ={expert_success_rate:.2f}  "
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
                    "expert/success_rate": expert_success_rate,
                    "mean_collect_return": mean_ret,
                    "total_env_steps": total_env_steps,
                    "collect_dt_s": collect_dt,
                }
                log.update(eval_metrics)
                _wandb.log(log)

            if args_cli.save_every_rounds > 0 and round_idx % args_cli.save_every_rounds == 0:
                torch.save(policy.state_dict(), args_cli.save_path)

            if eval_metrics:
                eval_succ = eval_metrics.get("eval_success_rate", 0.0)
                if eval_succ > best_eval_success:
                    best_eval_success = eval_succ
                    torch.save(policy.state_dict(), best_model_path)
                    print(f"  [best] eval_success_rate={eval_succ:.3f} → saved {best_model_path}", flush=True)

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
    """Deterministic student rollout. Returns mean per-episode return and
    episode success rate (same criterion as training: all drones within
    success_dist_threshold for success_hold_s seconds → _just_succeeded).

    Mirrors curobo's evaluate_actor: explicit eval/train mode wrapping around
    the rollout so BatchNorm / Dropout layers behave correctly.
    """
    device = env.device
    policy.cnf.eval()
    try:
        env.reset()
        per_drone_obs = env.get_per_drone_obs()
        returns = torch.zeros(env.num_envs, device=device)
        finished_returns: list[float] = []
        success_count = 0
        episode_count = 0
        for _ in range(n_steps):
            action_pd = policy.sample_action(per_drone_obs)         # (E, N, A) — cnf.eval() inside
            action_flat = action_pd.reshape(env.num_envs, N * A)    # (E, N*A)
            _, reward, terminated, truncated, _ = env.step(action_flat)
            per_drone_obs = env.get_per_drone_obs()
            returns += reward
            done = terminated | truncated
            if done.any():
                success_count += int(env._just_succeeded[done].sum().item())
                episode_count += int(done.sum().item())
                finished_returns.extend(returns[done].detach().cpu().tolist())
                returns[done] = 0.0
    finally:
        policy.cnf.train()

    success_rate = success_count / max(episode_count, 1)
    mean_ret = float(np.mean(finished_returns)) if finished_returns else float("nan")
    return {"eval_return": mean_ret, "eval_success_rate": success_rate}


if __name__ == "__main__":
    main()
