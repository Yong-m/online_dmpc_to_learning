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
    resulting per-drone action ``(E*N, A)`` (``A = 3`` -- normalised desired
    velocity per drone) back to ``(E, N*A)`` before sending to the env. The
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
     for *every* drone in *every* env, convert it to the env's 3-D normalised
     velocity action via :py:meth:`MultiDroneDmpcEnv.ref_to_action`, apply it,
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
parser.add_argument("--dmpc_log_path", type=str, default=None,
                    help="Optional .npz path for first-env DMPC expert debug logs.")
parser.add_argument("--dmpc_log_every", type=int, default=1,
                    help="Log every N collection steps when --dmpc_log_path is set.")

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
# desired-velocity command in [-1, 1]^3. The env internally turns it into the
# 4-D thrust + moment command via its cascaded position controller.
PER_DRONE_ACTION_DIM = 3


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
# ║ Expert wrapper: DMPC position + velocity reference → env action          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def expert_action(
    env: MultiDroneDmpcEnv,
    expert: DMPCExpert,
    debug_logger: "DmpcExpertLogger | None" = None,
    debug_step: int = 0,
) -> torch.Tensor:
    """Run DMPC for every drone in every env and map its plan to the env action.

    DMPC returns the paper's ``u_i`` (a position reference) together with its
    first derivative as a feed-forward velocity. We pass both into the env's
    cascaded P/PD position controller (:py:meth:`MultiDroneDmpcEnv.ref_to_action`)
    which closes the inner loop and lands on the env's 3-D normalised velocity
    command (the env then turns that into a 4-D thrust+moment internally).

    Returns ``(num_envs, num_drones * 3)``.
    """
    states = env.get_world_states()
    pos_w = states["pos_w"]
    vel_w = states["lin_vel_w"]
    goal_w = states["goal_w"]
    origins = env._terrain.env_origins

    ref_pos_w, ref_vel_w = expert.plan(
        pos_w=pos_w, vel_w=vel_w, goal_w=goal_w, env_origins=origins,
    )
    action = env.ref_to_action(ref_pos_w, ref_vel_w)
    if debug_logger is not None:
        debug_logger.add(debug_step, states, ref_pos_w, ref_vel_w, action)
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
        desired_vel_w = action_norm_np * float(self.env.cfg.v_max)
        desired_pos_w = pos_w + desired_vel_w * float(self.env.step_dt)
        desired_acc_w = (
            float(self.env.cfg.pos_track_kp) * (desired_pos_w - pos_w)
            + float(self.env.cfg.pos_track_kd) * (desired_vel_w - vel_w)
        )
        accel_clip = float(self.env.cfg.track_accel_clip)
        desired_acc_w = np.clip(desired_acc_w, -accel_clip, accel_clip)

        planned_ref_pos_w = np.full((N, K, 3), np.nan, dtype=np.float64)
        planned_ref_vel_w = np.full((N, K, 3), np.nan, dtype=np.float64)
        predicted_pos_w = np.full((N, K, 3), np.nan, dtype=np.float64)
        predicted_vel_w = np.full((N, K, 3), np.nan, dtype=np.float64)
        control_points = np.full((N, self.expert.n_bez), np.nan, dtype=np.float64)

        for i in range(N):
            st = self.expert._state.get((env_idx, i))
            if st is None:
                continue
            U = st["U"]
            control_points[i] = U
            planned_ref_pos_w[i] = (self.expert.M_pos_hor @ U).reshape(K, 3) + origin
            planned_ref_vel_w[i] = (self.expert.M_vel_hor @ U).reshape(K, 3)
            x0_local = np.concatenate([pos_w[i] - origin, vel_w[i]])
            u_samples = self.expert.M_pos_hor @ U
            pred_stack = self.expert._A0_stack @ x0_local + self.expert._Lam_stack @ u_samples
            pred_state = pred_stack.reshape(K, 6)
            predicted_pos_w[i] = pred_state[:, :3] + origin
            predicted_vel_w[i] = pred_state[:, 3:6]

        self.records.append(
            {
                "step": int(step),
                "pos_w": pos_w.astype(np.float32),
                "vel_w": vel_w.astype(np.float32),
                "goal_w": goal_w.astype(np.float32),
                "ref_pos_w": ref_pos_w[env_idx].detach().cpu().numpy().astype(np.float32),
                "ref_vel_w": ref_vel_w[env_idx].detach().cpu().numpy().astype(np.float32),
                "action_normalized": action_norm_np.astype(np.float32),
                "desired_pos_cmd_w": desired_pos_w.astype(np.float32),
                "desired_vel_cmd_w": desired_vel_w.astype(np.float32),
                "desired_acc_cmd_w": desired_acc_w.astype(np.float32),
                "planned_ref_pos_w": planned_ref_pos_w.astype(np.float32),
                "planned_ref_vel_w": planned_ref_vel_w.astype(np.float32),
                "predicted_pos_w": predicted_pos_w.astype(np.float32),
                "predicted_vel_w": predicted_vel_w.astype(np.float32),
                "control_points": control_points.astype(np.float32),
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


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Main training loop                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = MultiDroneDmpcEnvCfg()
    env_cfg.num_drones = args_cli.num_drones
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.__post_init__()
    if getattr(args_cli, "device", None):
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped  # type: MultiDroneDmpcEnv
    device = env.device
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

    env.reset(seed=args_cli.seed)
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
                if done.any():
                    finished = episode_returns[done].detach().cpu().tolist()
                    collect_returns.extend(finished)
                    episode_returns[done] = 0.0
                    expert.reset(done.nonzero(as_tuple=False).flatten())
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
                per_drone_obs = env.get_per_drone_obs()
                episode_returns.zero_()
                expert.reset()

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
