"""offline_bc_dmpc.py — Offline Behavior Cloning from saved DMPC expert data.

Loads per-agent .npz trajectories collected by collect_dmpc_data.py and trains
a flow-matching policy (same architecture as online_bc_dmpc.py).

Loss per batch:
    total = flow_match(free_ctrl_pts)
          + λ_vel * MSE(vel_head(cond_emb), substep_ref_vel_w[:,0,:])
          + λ_acc * MSE(acc_head(cond_emb), substep_ref_acc_w[:,0,:])

The velocity and acceleration auxiliary heads share the condition encoder with
the flow net.  Their gradients force the encoder to capture derivative structure
beyond what the flow loss alone provides — this is the regularization between
position, velocity and acceleration.

Every --eval_every epochs the policy runs in the Isaac Sim environment and
success rate is measured.  The best checkpoint (by eval success rate) is saved
separately.

Usage::

    python offline_bc_dmpc.py \\
        --data_dir runs/dmpc_data \\
        --save_path runs/offline_bc/model.pt \\
        --num_envs 32 --num_drones 10 \\
        --n_epochs 200 --eval_every 10 [--wandb]
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
    _s = str(_p)
    if _p.exists() and _s not in sys.path:
        sys.path.insert(0, _s)

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Offline BC (flow-matching) from saved DMPC data.")

# Data
parser.add_argument("--data_dir", type=str, required=True,
                    help="Directory containing per-agent .npz files from collect_dmpc_data.py.")
parser.add_argument("--max_files", type=int, default=None,
                    help="Cap on number of .npz files to load (None = all).")
parser.add_argument("--val_fraction", type=float, default=0.05,
                    help="Fraction of files held out as validation set.")

# Env (for eval)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_drones", type=int, default=10)
parser.add_argument("--task", type=str, default="Isaac-MultiDrone-DMPC-Direct-v0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--no_terminate_on_bounds", action="store_true", default=False)

# Training
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--lambda_vel", type=float, default=0.1,
                    help="Weight for velocity auxiliary loss.")
parser.add_argument("--lambda_acc", type=float, default=0.05,
                    help="Weight for acceleration auxiliary loss.")

# Eval
parser.add_argument("--eval_every", type=int, default=10,
                    help="Run Isaac Sim eval every this many epochs.")
parser.add_argument("--eval_steps", type=int, default=400,
                    help="Env steps per eval run.")

# Model
parser.add_argument("--hidden_dims", type=int, nargs="*", default=[256, 256, 256])
parser.add_argument("--emb_dim", type=int, default=64)
parser.add_argument("--sample_steps", type=int, default=10)
parser.add_argument("--bezier_k", type=int, default=6)
parser.add_argument("--ctrl_pts_max", type=float, default=0.3,
                    help="Scale for normalising free position ctrl_pts to ~[-1, 1].")

# Checkpointing
parser.add_argument("--save_path", type=str, default="runs/offline_bc/model.pt")
parser.add_argument("--save_every", type=int, default=10,
                    help="Save latest checkpoint every this many epochs.")
parser.add_argument("--resume", type=str, default=None)

# Logging
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--wandb_project", type=str, default="offline_bc_dmpc")
parser.add_argument("--wandb_run_name", type=str, default=None)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False   # cameras only needed if recording eval video

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ──────────────────────────────────────────────────────
import gymnasium as gym       # noqa: E402
import torch                   # noqa: E402
import torch.nn as nn          # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import Dataset, DataLoader, random_split  # noqa: E402

import quadcopter              # noqa: F401, E402
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

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except Exception:
    _wandb = None
    _WANDB_AVAILABLE = False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Dataset                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class DMPCOfflineDataset(Dataset):
    """Per-window samples loaded from per-agent .npz files.

    Each item:
        obs         (P,)        raw per-drone observation
        ctrl_pts    (K_free*3,) free position ctrl_pts, goal-aligned delta [m]
        ref_vel_w   (3,)        first-substep velocity reference, world frame [m/s]
        ref_acc_w   (3,)        first-substep acceleration reference, world frame [m/s²]
    """

    def __init__(self, files: list[Path]):
        super().__init__()
        obs_list:      list[np.ndarray] = []
        ctrl_list:     list[np.ndarray] = []
        ref_vel_list:  list[np.ndarray] = []
        ref_acc_list:  list[np.ndarray] = []

        for fpath in files:
            d = np.load(fpath)
            T = d["obs"].shape[0]
            obs_list.append(d["obs"].astype(np.float32))                     # (T, P)
            ctrl_list.append(
                d["free_ctrl_pts"].reshape(T, -1).astype(np.float32)         # (T, K_free*3)
            )
            # substep_ref_vel/acc_w: (T, S, 3) — use first substep
            if "substep_ref_vel_w" in d:
                ref_vel_list.append(d["substep_ref_vel_w"][:, 0, :].astype(np.float32))
                ref_acc_list.append(d["substep_ref_acc_w"][:, 0, :].astype(np.float32))
            else:
                ref_vel_list.append(d["ref_vel_w"].astype(np.float32))
                ref_acc_list.append(d["ref_acc_w"].astype(np.float32))

        self.obs       = torch.from_numpy(np.concatenate(obs_list,     axis=0))
        self.ctrl_pts  = torch.from_numpy(np.concatenate(ctrl_list,    axis=0))
        self.ref_vel_w = torch.from_numpy(np.concatenate(ref_vel_list, axis=0))
        self.ref_acc_w = torch.from_numpy(np.concatenate(ref_acc_list, axis=0))

        print(f"[Dataset] {len(files)} files → {len(self.obs)} windows  "
              f"obs={tuple(self.obs.shape[1:])}  ctrl={tuple(self.ctrl_pts.shape[1:])}")

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "obs":      self.obs[idx],
            "ctrl_pts": self.ctrl_pts[idx],
            "ref_vel":  self.ref_vel_w[idx],
            "ref_acc":  self.ref_acc_w[idx],
        }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Policy with derivative-regularisation auxiliary heads                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class OfflineDronePolicy(nn.Module):
    """Flow-matching policy + auxiliary vel/acc heads for derivative regularisation.

    Primary output: K_free position ctrl_pts in goal-aligned delta frame.
    Auxiliary:      linear heads on the condition embedding that predict
                    first-substep (vel_w, acc_w) in world frame.  Their loss
                    flows back into the shared condition encoder, forcing it to
                    capture the velocity/acceleration structure of trajectories.
    """

    def __init__(
        self,
        obs_dim: int,
        bezier_k: int,
        hidden_dims: list[int],
        emb_dim: int,
        sample_steps: int,
        device: torch.device,
        ctrl_pts_max: float = 0.3,
        own_dim: int = PER_DRONE_OWN_DIM,
        neighbor_dim: int = PER_NEIGHBOUR_DIM,
    ):
        super().__init__()
        self.P = obs_dim
        self.bezier_k      = bezier_k
        self.bezier_k_free = bezier_k - 2
        self.A = self.bezier_k_free * 3   # flow latent dim
        self.own_dim      = own_dim
        self.neighbor_dim = neighbor_dim
        self.device       = device

        # Obs normalizers
        self.own_norm   = EmpiricalNormalization(shape=own_dim,      until=int(1e8))
        self.neigh_norm = EmpiricalNormalization(shape=neighbor_dim, until=int(1e8))

        # Scale for normalising free ctrl_pts to ~[-1, 1]
        self.register_buffer("_ctrl_pts_max",
                             torch.tensor(ctrl_pts_max, dtype=torch.float32))

        # Cross-attention neighbour encoder
        self.neighbor_enc = NeighborEncoder(
            own_dim=own_dim, neighbor_dim=neighbor_dim,
            emb_dim=emb_dim, num_heads=4,
        )
        cond_input_dim = own_dim + emb_dim

        # Condition net + flow net
        nn_condition = ConditionMlp(
            cond_dim=cond_input_dim, emb_dim=emb_dim,
            activations=["elu"] * len(hidden_dims) + ["linear"],
            hidden_dims=hidden_dims,
        )
        nn_flow = FlowMlp(
            x_dim=self.A, emb_dim=emb_dim,
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
        self.cnf_ema   = self.cnf.model_ema
        self.cnf_last  = self.cnf.model_last
        self.cnf.init_proximal(beta_prox=0.97)

        # Auxiliary heads: predict world-frame vel / acc at first substep.
        # They take the condition *embedding* (post condition-net, dim=emb_dim).
        self.vel_head = nn.Linear(emb_dim, 3)
        self.acc_head = nn.Linear(emb_dim, 3)

    # ── obs encoding ───────────────────────────────────────────────────────
    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, P) → (B, own_dim + emb_dim) condition input."""
        B = obs.shape[0]
        own_raw   = obs[:, :self.own_dim]
        neigh_raw = obs[:, self.own_dim:]
        own_n = self.own_norm(own_raw)
        N_neigh = neigh_raw.shape[1] // self.neighbor_dim
        if N_neigh > 0:
            neigh_n = self.neigh_norm(
                neigh_raw.reshape(B * N_neigh, self.neighbor_dim)
            ).reshape(B, N_neigh, self.neighbor_dim)
        else:
            neigh_n = torch.zeros(B, 0, self.neighbor_dim, device=obs.device)
        return torch.cat([own_n, self.neighbor_enc(own_n, neigh_n)], dim=-1)

    def update_obs_norm(self, obs: torch.Tensor) -> None:
        own_raw   = obs[:, :self.own_dim].detach()
        neigh_raw = obs[:, self.own_dim:].detach()
        self.own_norm.update(own_raw)
        N_neigh = neigh_raw.shape[1] // self.neighbor_dim
        if N_neigh > 0:
            self.neigh_norm.update(neigh_raw.reshape(-1, self.neighbor_dim))

    def encode_action(self, ctrl_pts_flat: torch.Tensor) -> torch.Tensor:
        return ctrl_pts_flat / self._ctrl_pts_max

    def decode_action(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self._ctrl_pts_max

    # ── training loss ──────────────────────────────────────────────────────
    def compute_loss(
        self,
        obs: torch.Tensor,        # (B, P)
        ctrl_pts: torch.Tensor,   # (B, K_free*3) in goal-aligned delta [m]
        ref_vel: torch.Tensor,    # (B, 3) world-frame, first substep [m/s]
        ref_acc: torch.Tensor,    # (B, 3) world-frame, first substep [m/s²]
        lambda_vel: float = 0.1,
        lambda_acc: float = 0.05,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        self.update_obs_norm(obs)

        # ── 1. Primary: rectified-flow matching on free ctrl_pts ──────────
        cond_in = self._encode_obs(obs)
        x1      = self.encode_action(ctrl_pts)              # (B, A) normalised
        x0      = torch.randn_like(x1)
        t       = torch.rand(x1.shape[0], device=x1.device)
        xt      = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        cond_emb = self.cnf.model["condition"](cond_in)    # (B, emb_dim)
        vel_pred = self.cnf.model["flow"](xt, t, cond_emb)
        vel_tgt  = (x1 - x0).detach()
        flow_loss = (vel_pred - vel_tgt).pow(2).mean()

        # ── 2. Derivative regularisation via auxiliary heads ──────────────
        # cond_emb already computed — reuse for zero extra forward pass cost.
        vel_aux = self.vel_head(cond_emb)                  # (B, 3) predicted vel_w
        acc_aux = self.acc_head(cond_emb)                  # (B, 3) predicted acc_w

        # Normalise targets so loss scale is ~independent of units.
        vel_scale = ref_vel.detach().norm(dim=-1, keepdim=True).clamp(min=0.1)
        acc_scale = ref_acc.detach().norm(dim=-1, keepdim=True).clamp(min=0.1)
        vel_loss  = ((vel_aux - ref_vel) / vel_scale).pow(2).mean()
        acc_loss  = ((acc_aux - ref_acc) / acc_scale).pow(2).mean()

        total = flow_loss + lambda_vel * vel_loss + lambda_acc * acc_loss
        return total, {
            "loss/total":    float(total.item()),
            "loss/flow":     float(flow_loss.item()),
            "loss/vel_aux":  float(vel_loss.item()),
            "loss/acc_aux":  float(acc_loss.item()),
        }

    # ── inference (same as online_bc_dmpc.py) ─────────────────────────────
    @torch.no_grad()
    def sample_action(self, per_drone_obs: torch.Tensor) -> torch.Tensor:
        """(E, N, P) → (E, N, K_free*3) free ctrl_pts in goal-aligned delta [m]."""
        E, N, P = per_drone_obs.shape
        flat = per_drone_obs.reshape(E * N, P)
        self.cnf.eval()
        cond_in = self._encode_obs(flat)
        x0      = torch.randn(cond_in.shape[0], self.A, device=cond_in.device)
        latent, _ = self.cnf.sample(x0=x0, condition=cond_in, n_samples=cond_in.shape[0])
        free_ctrl = self.decode_action(latent)
        self.cnf.train()
        return free_ctrl.reshape(E, N, self.A)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Bezier helpers (copied from online_bc_dmpc.py)                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import math as _math  # noqa: E402


def eval_bezier(ctrl_pts: torch.Tensor, tau: float) -> torch.Tensor:
    K = ctrl_pts.shape[-2]
    out = torch.zeros(*ctrl_pts.shape[:-2], 3, device=ctrl_pts.device, dtype=ctrl_pts.dtype)
    for j in range(K):
        c = _math.comb(K - 1, j) * (tau ** j) * ((1.0 - tau) ** (K - 1 - j))
        out = out + c * ctrl_pts[..., j, :]
    return out


def eval_bezier_full(ctrl_pts: torch.Tensor, tau: float, h: float):
    """(..., K, 3) → pos, vel [m/s], acc [m/s²] at normalised time tau."""
    K = ctrl_pts.shape[-2]
    vel_ctrl = (K - 1) / h * (ctrl_pts[..., 1:, :] - ctrl_pts[..., :-1, :])
    acc_ctrl = (K - 2) / h * (vel_ctrl[..., 1:, :] - vel_ctrl[..., :-1, :])
    return eval_bezier(ctrl_pts, tau), eval_bezier(vel_ctrl, tau), eval_bezier(acc_ctrl, tau)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Eval (runs in Isaac Sim)                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def evaluate(
    env: MultiDroneDmpcEnv,
    policy: OfflineDronePolicy,
    n_steps: int,
    n_substeps: int,
    h_window: float,
) -> dict[str, float]:
    """Deterministic student rollout.  Returns eval_success_rate and eval_return."""
    device   = env.device
    E        = env.num_envs
    N        = env.cfg.num_drones
    K        = policy.bezier_k
    K_free   = policy.bezier_k_free
    policy.cnf.eval()

    window_refs: list[torch.Tensor] | None = None
    sub_step = 0
    returns  = torch.zeros(E, device=device)
    finished_returns: list[float] = []
    success_count = episode_count = 0

    try:
        env.reset()
        per_drone_obs = env.get_per_drone_obs()

        for _ in range(n_steps):
            if sub_step == 0 or window_refs is None:
                drone_st = env._stack_drone_state()
                pos_w0   = drone_st["pos_w"]       # (E, N, 3)
                vel_w0   = drone_st["lin_vel_w"]   # (E, N, 3)
                R   = env._compute_goal_aligned_R(pos_w0, env._goal_pos_w)  # (E, N, 3, 3)
                R_T = R.transpose(-1, -2)

                P_1 = torch.bmm(
                    R.reshape(E * N, 3, 3), vel_w0.reshape(E * N, 3, 1)
                ).reshape(E, N, 3) * (h_window / (K - 1))
                P_0 = torch.zeros_like(P_1)

                P_free = policy.sample_action(per_drone_obs).reshape(E, N, K_free, 3)
                full_ctrl = torch.cat(
                    [P_0.unsqueeze(-2), P_1.unsqueeze(-2), P_free], dim=-2
                )  # (E, N, K, 3) goal-aligned delta

                window_refs = []
                R_T_flat = R_T.reshape(E * N, 3, 3)
                for j in range(n_substeps):
                    tau_j = (j + 1) / n_substeps
                    dp_g, v_g, a_g = eval_bezier_full(full_ctrl, tau_j, h_window)
                    dp_w = torch.bmm(R_T_flat, dp_g.reshape(E * N, 3, 1)).reshape(E, N, 3)
                    v_w  = torch.bmm(R_T_flat, v_g.reshape(E * N, 3, 1)).reshape(E, N, 3)
                    a_w  = torch.bmm(R_T_flat, a_g.reshape(E * N, 3, 1)).reshape(E, N, 3)
                    window_refs.append(
                        torch.cat([pos_w0 + dp_w, v_w, a_w], dim=-1)  # (E, N, 9)
                    )

            ref_9d  = window_refs[sub_step]
            _, reward, terminated, truncated, _ = env.step(ref_9d.reshape(E, N * 9))
            per_drone_obs = env.get_per_drone_obs()
            returns += reward
            done = terminated | truncated

            if done.any():
                success_count += int(env._just_succeeded[done].sum().item())
                episode_count += int(done.sum().item())
                finished_returns.extend(returns[done].detach().cpu().tolist())
                returns[done] = 0.0
                sub_step = 0
                window_refs = None
                continue

            sub_step = (sub_step + 1) % n_substeps
    finally:
        policy.cnf.train()

    success_rate = success_count / max(episode_count, 1)
    mean_ret     = float(np.mean(finished_returns)) if finished_returns else float("nan")
    return {"eval_success_rate": success_rate, "eval_return": mean_ret,
            "eval_episodes": episode_count}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Main                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load dataset ────────────────────────────────────────────────────────
    all_files = sorted(Path(args_cli.data_dir).glob("ep_*.npz"))
    if not all_files:
        all_files = sorted(Path(args_cli.data_dir).glob("*.npz"))
    if args_cli.max_files:
        all_files = all_files[:args_cli.max_files]
    assert all_files, f"No .npz files found in {args_cli.data_dir}"

    np.random.shuffle(all_files := list(all_files))
    n_val    = max(1, int(len(all_files) * args_cli.val_fraction))
    val_files   = all_files[:n_val]
    train_files = all_files[n_val:]
    print(f"[Data] train={len(train_files)}  val={len(val_files)} files")

    train_ds = DMPCOfflineDataset(train_files)
    val_ds   = DMPCOfflineDataset(val_files)

    train_loader = DataLoader(
        train_ds, batch_size=args_cli.batch_size,
        shuffle=True, drop_last=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args_cli.batch_size,
        shuffle=False, drop_last=False, num_workers=2, pin_memory=True,
    )

    # Infer dims from first file
    sample_file = np.load(train_files[0])
    obs_dim   = int(sample_file["obs"].shape[1])
    bezier_k  = int(sample_file["bezier_k"])
    n_substeps = int(sample_file["n_substeps"])
    h_window  = float(sample_file["h_window"])
    print(f"[Config] obs_dim={obs_dim}  bezier_k={bezier_k}  "
          f"n_substeps={n_substeps}  h_window={h_window:.3f}s")

    # ── policy ──────────────────────────────────────────────────────────────
    policy = OfflineDronePolicy(
        obs_dim=obs_dim,
        bezier_k=bezier_k,
        hidden_dims=args_cli.hidden_dims,
        emb_dim=args_cli.emb_dim,
        sample_steps=args_cli.sample_steps,
        device=device,
        ctrl_pts_max=args_cli.ctrl_pts_max,
    ).to(device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args_cli.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args_cli.n_epochs, eta_min=args_cli.lr * 0.1
    )

    start_epoch  = 0
    best_success = -1.0
    save_path    = Path(args_cli.save_path)
    best_path    = save_path.parent / (save_path.stem + "_best" + save_path.suffix)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if args_cli.resume:
        ckpt = torch.load(args_cli.resume, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt.get("epoch", 0)
        best_success = ckpt.get("best_success", -1.0)
        print(f"[Resume] epoch={start_epoch}  best_success={best_success:.3f}")

    # ── Isaac Sim env (used only for eval) ──────────────────────────────────
    env_cfg = MultiDroneDmpcEnvCfg()
    env_cfg.num_drones = args_cli.num_drones
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.no_terminate_on_bounds:
        env_cfg.terminate_on_bounds = False
    env_cfg.__post_init__()
    env_gym = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env: MultiDroneDmpcEnv = env_gym.unwrapped

    # ── wandb ───────────────────────────────────────────────────────────────
    if args_cli.wandb and _WANDB_AVAILABLE:
        _wandb.init(
            project=args_cli.wandb_project,
            name=args_cli.wandb_run_name,
            config=vars(args_cli),
        )

    # ── training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args_cli.n_epochs):
        policy.train()
        epoch_metrics: dict[str, list[float]] = {
            "loss/total": [], "loss/flow": [], "loss/vel_aux": [], "loss/acc_aux": []
        }

        for batch in train_loader:
            obs      = batch["obs"].to(device)
            ctrl_pts = batch["ctrl_pts"].to(device)
            ref_vel  = batch["ref_vel"].to(device)
            ref_acc  = batch["ref_acc"].to(device)

            loss, metrics = policy.compute_loss(
                obs, ctrl_pts, ref_vel, ref_acc,
                lambda_vel=args_cli.lambda_vel,
                lambda_acc=args_cli.lambda_acc,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args_cli.grad_clip)
            optimizer.step()
            policy.cnf.update_proximal()

            for k, v in metrics.items():
                epoch_metrics[k].append(v)

        scheduler.step()

        avg = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}

        # ── validation loss ─────────────────────────────────────────────
        val_losses: list[float] = []
        policy.eval()
        with torch.no_grad():
            for batch in val_loader:
                obs      = batch["obs"].to(device)
                ctrl_pts = batch["ctrl_pts"].to(device)
                ref_vel  = batch["ref_vel"].to(device)
                ref_acc  = batch["ref_acc"].to(device)
                _, vm = policy.compute_loss(
                    obs, ctrl_pts, ref_vel, ref_acc,
                    lambda_vel=args_cli.lambda_vel,
                    lambda_acc=args_cli.lambda_acc,
                )
                val_losses.append(vm["loss/total"])
        policy.train()
        avg["val/loss"] = float(np.mean(val_losses))

        log_str = (f"[epoch {epoch+1:4d}/{args_cli.n_epochs}]"
                   f"  loss={avg['loss/total']:.4f}"
                   f"  flow={avg['loss/flow']:.4f}"
                   f"  vel={avg['loss/vel_aux']:.4f}"
                   f"  acc={avg['loss/acc_aux']:.4f}"
                   f"  val={avg['val/loss']:.4f}")

        # ── eval every N epochs ──────────────────────────────────────────
        eval_metrics: dict[str, float] = {}
        if args_cli.eval_every > 0 and (epoch + 1) % args_cli.eval_every == 0:
            eval_metrics = evaluate(env, policy, args_cli.eval_steps, n_substeps, h_window)
            succ = eval_metrics["eval_success_rate"]
            log_str += (f"  eval_succ={succ:.3f}"
                        f"  eval_ret={eval_metrics['eval_return']:.2f}"
                        f"  eval_eps={eval_metrics['eval_episodes']}")
            avg.update(eval_metrics)

            if succ > best_success:
                best_success = succ
                torch.save({
                    "policy":       policy.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "epoch":        epoch + 1,
                    "best_success": best_success,
                    "obs_dim":      obs_dim,
                    "bezier_k":     bezier_k,
                    "n_substeps":   n_substeps,
                    "h_window":     h_window,
                }, str(best_path))
                log_str += f"  ← best saved"

        print(log_str, flush=True)
        if args_cli.wandb and _WANDB_AVAILABLE:
            _wandb.log({"epoch": epoch + 1, **avg})

        # ── periodic checkpoint ──────────────────────────────────────────
        if (epoch + 1) % args_cli.save_every == 0:
            torch.save({
                "policy":       policy.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "epoch":        epoch + 1,
                "best_success": best_success,
                "obs_dim":      obs_dim,
                "bezier_k":     bezier_k,
                "n_substeps":   n_substeps,
                "h_window":     h_window,
            }, str(save_path))

    env_gym.close()
    if args_cli.wandb and _WANDB_AVAILABLE:
        _wandb.finish()
    print(f"Training done. best_success={best_success:.3f}  saved → {best_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
