"""offline_bc_dmpc.py — Offline Behavior Cloning from saved DMPC expert data.

Loads per-agent .npz trajectories collected by collect_dmpc_data.py and trains
a flow-matching policy (same architecture as online_bc_dmpc.py).

Loss per batch:
    total = flow_match(bezier_U_goal, temporal_weights)
          + λ_vel * MSE(vel_head(cond_emb), substep_ref_vel_w[:,0,:])
          + λ_acc * MSE(acc_head(cond_emb), substep_ref_acc_w[:,0,:])

Pure offline training — no Isaac Sim.  Use online_bc_dmpc.py (or a separate
eval script) to test the saved checkpoint in simulation.

Usage::

    python offline_bc_dmpc.py \\
        --data_dir runs/dmpc_data \\
        --save_path runs/offline_bc/model.pt \\
        --n_epochs 200 [--wandb]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_POLICYFLOW_ROOT = _HERE.parent.parent / "policyflow"
_PROJECT_ROOT = _HERE.parent.parent.parent
for _p in (_POLICYFLOW_ROOT, _PROJECT_ROOT):
    _s = str(_p)
    if _p.exists() and _s not in sys.path:
        sys.path.insert(0, _s)

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


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline BC (flow-matching) from saved DMPC data.")

    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of files held out as validation set.")

    # Training
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_vel", type=float, default=0.1)
    parser.add_argument("--lambda_acc", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)

    # Model
    parser.add_argument("--hidden_dims", type=int, nargs="*", default=[256, 256, 256])
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--ctrl_pts_max", type=float, default=2.0,
                        help="Scale for normalising bezier_U_goal to ~[-1,1]. "
                             "Set to ~95th-percentile of per-element abs value in your dataset.")
    parser.add_argument("--ctrl_clip_threshold", type=float, default=1.0,
                        help="Drop windows where any ctrl_pt element exceeds this [m]. "
                             "Filters physically impossible DMPC plans (>10 m/s in 0.1s).")
    parser.add_argument("--own_dim", type=int, default=37,
                        help="PER_DRONE_OWN_DIM from multi_drone_dmpc_env.py.")
    parser.add_argument("--neighbor_dim", type=int, default=9,
                        help="PER_NEIGHBOUR_DIM from multi_drone_dmpc_env.py.")

    # Checkpointing
    parser.add_argument("--save_path", type=str, default="runs/offline_bc/model.pt")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)

    # Evaluation (calls eval_bc_dmpc.py in a subprocess)
    parser.add_argument("--eval_every", type=int, default=0,
                        help="Run sim eval every N epochs (0=disabled). Also runs on new best.")
    parser.add_argument("--eval_on_best", action="store_true", default=True,
                        help="Always run eval when a new best checkpoint is saved.")
    parser.add_argument("--eval_script", type=str, default=None,
                        help="Path to eval_bc_dmpc.py. Defaults to sibling file.")
    parser.add_argument("--eval_num_envs", type=int, default=32)
    parser.add_argument("--eval_num_drones", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=400)
    parser.add_argument("--eval_task", type=str, default="Isaac-MultiDrone-DMPC-Direct-v0")
    parser.add_argument("--eval_timeout", type=int, default=600,
                        help="Max seconds to wait for one eval subprocess.")

    # Logging
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="offline_bc_dmpc")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Dataset                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class DMPCOfflineDataset(Dataset):
    def __init__(self, files: list[Path]):
        super().__init__()
        obs_list:      list[np.ndarray] = []
        ctrl_list:     list[np.ndarray] = []
        ref_vel_list:  list[np.ndarray] = []
        ref_acc_list:  list[np.ndarray] = []

        for fpath in tqdm(files, desc="Loading", unit="file", leave=False, dynamic_ncols=True):
            d = np.load(fpath)
            obs_list.append(d["obs"].astype(np.float32))
            ctrl_list.append(d["bezier_U_goal"].astype(np.float32))  # (T, n_bez)
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

        print(f"[Dataset] {len(files)} files → {len(self.obs):,} windows  "
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
# ║ Policy                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class OfflineDronePolicy(nn.Module):
    def __init__(
        self,
        n_bez: int,
        hidden_dims: list[int],
        emb_dim: int,
        sample_steps: int,
        device: torch.device,
        ctrl_pts_max: float = 2.0,
        own_dim: int = 37,
        neighbor_dim: int = 9,
        n_segments: int = 3,
    ):
        super().__init__()
        self.n_bez         = n_bez
        self.A             = n_bez
        self.own_dim       = own_dim
        self.neighbor_dim  = neighbor_dim
        self.device        = device

        self.own_norm   = EmpiricalNormalization(shape=own_dim,      until=int(1e8))
        self.neigh_norm = EmpiricalNormalization(shape=neighbor_dim, until=int(1e8))
        self.register_buffer("_ctrl_pts_max",
                             torch.tensor(ctrl_pts_max, dtype=torch.float32))

        # Temporal weights: higher weight for earlier segments (closer in time).
        # n_bez = n_segments × n_per_seg. Weight decays linearly across segments.
        n_per_seg = n_bez // n_segments
        seg_weights = torch.linspace(float(n_segments), 1.0, n_segments)  # [3,2,1] for 3 segs
        seg_weights = seg_weights * (n_segments / seg_weights.sum())        # normalize mean=1
        flow_weights = seg_weights.repeat_interleave(n_per_seg)             # (n_bez,)
        self.register_buffer("_flow_weights", flow_weights)

        self.neighbor_enc = NeighborEncoder(
            own_dim=own_dim, neighbor_dim=neighbor_dim,
            emb_dim=emb_dim, num_heads=4,
        )
        cond_input_dim = own_dim + emb_dim

        self.cnf = ContinuousNormalizingFlow(
            x_dims=self.A,
            nn_flow=FlowMlp(
                x_dim=self.A, emb_dim=emb_dim,
                activations=["elu"] * len(hidden_dims) + ["linear"],
                hidden_dims=hidden_dims,
            ),
            nn_condition=ConditionMlp(
                cond_dim=cond_input_dim, emb_dim=emb_dim,
                activations=["elu"] * len(hidden_dims) + ["linear"],
                hidden_dims=hidden_dims,
            ),
            sample_steps=sample_steps,
            sample_step_schedule="uniform_continuous",
            interpolation_type="rectified_flow",
            device=device,
        )
        self.cnf.init_proximal(beta_prox=0.97)

        self.vel_head = nn.Linear(emb_dim, 3)
        self.acc_head = nn.Linear(emb_dim, 3)

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        own_n   = self.own_norm(obs[:, :self.own_dim])
        neigh_r = obs[:, self.own_dim:]
        N_neigh = neigh_r.shape[1] // self.neighbor_dim
        if N_neigh > 0:
            neigh_n = self.neigh_norm(
                neigh_r.reshape(B * N_neigh, self.neighbor_dim)
            ).reshape(B, N_neigh, self.neighbor_dim)
        else:
            neigh_n = torch.zeros(B, 0, self.neighbor_dim, device=obs.device)
        return torch.cat([own_n, self.neighbor_enc(own_n, neigh_n)], dim=-1)

    def _update_obs_norm(self, obs: torch.Tensor) -> None:
        own_r   = obs[:, :self.own_dim].detach()
        neigh_r = obs[:, self.own_dim:].detach()
        self.own_norm.update(own_r)
        N_neigh = neigh_r.shape[1] // self.neighbor_dim
        if N_neigh > 0:
            self.neigh_norm.update(neigh_r.reshape(-1, self.neighbor_dim))

    @torch.no_grad()
    def sample_action(self, per_drone_obs: torch.Tensor) -> torch.Tensor:
        """ODE integration → bezier_U_goal (E, N, n_bez) in goal-aligned relative frame [m].

        Args:
            per_drone_obs: ``(E, N, P)`` raw per-drone observations.

        Returns:
            ``(E, N, n_bez)`` Bezier control points in goal-aligned relative frame [m].
        """
        E, N, P = per_drone_obs.shape
        flat = per_drone_obs.reshape(E * N, P)
        self.cnf.eval()
        cond_in = self._encode_obs(flat)
        x0 = torch.randn(cond_in.shape[0], self.A, device=cond_in.device)
        latent, _ = self.cnf.sample(x0=x0, condition=cond_in, n_samples=cond_in.shape[0])
        bezier_U_goal = latent * self._ctrl_pts_max
        self.cnf.train()
        return bezier_U_goal.reshape(E, N, self.A)

    def compute_loss(
        self,
        obs: torch.Tensor,
        ctrl_pts: torch.Tensor,
        ref_vel: torch.Tensor,
        ref_acc: torch.Tensor,
        lambda_vel: float = 0.1,
        lambda_acc: float = 0.05,
        update_norm: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if update_norm:
            self._update_obs_norm(obs)

        cond_in  = self._encode_obs(obs)
        x1       = ctrl_pts / self._ctrl_pts_max
        x0       = torch.randn_like(x1)
        t        = torch.rand(x1.shape[0], device=x1.device)
        xt       = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        cond_emb = self.cnf.model["condition"](cond_in)
        vel_pred = self.cnf.model["flow"](xt, t, cond_emb)
        target   = (x1 - x0).detach()
        # Weight earlier (closer-in-time) segments more heavily.
        flow_loss = ((vel_pred - target).pow(2) * self._flow_weights).mean()

        vel_aux   = self.vel_head(cond_emb)
        acc_aux   = self.acc_head(cond_emb)
        vel_scale = ref_vel.detach().norm(dim=-1, keepdim=True).clamp(min=0.1)
        acc_scale = ref_acc.detach().norm(dim=-1, keepdim=True).clamp(min=0.1)
        vel_loss  = ((vel_aux - ref_vel) / vel_scale).pow(2).mean()
        acc_loss  = ((acc_aux - ref_acc) / acc_scale).pow(2).mean()

        total = flow_loss + lambda_vel * vel_loss + lambda_acc * acc_loss
        return total, {
            "loss/total":   float(total),
            "loss/flow":    float(flow_loss),
            "loss/vel_aux": float(vel_loss),
            "loss/acc_aux": float(acc_loss),
        }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Main                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── dataset ──────────────────────────────────────────────────────────────
    all_files = sorted(Path(args.data_dir).glob("ep_*.npz"))
    if not all_files:
        all_files = sorted(Path(args.data_dir).glob("*.npz"))
    if args.max_files:
        all_files = all_files[:args.max_files]
    assert all_files, f"No .npz files found in {args.data_dir}"

    all_files = list(all_files)
    np.random.shuffle(all_files)
    n_val       = max(1, int(len(all_files) * args.val_fraction))
    val_files   = all_files[:n_val]
    train_files = all_files[n_val:]
    print(f"[Data] train={len(train_files)}  val={len(val_files)} files")

    train_ds = DMPCOfflineDataset(train_files)
    val_ds   = DMPCOfflineDataset(val_files)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, drop_last=False,
                              num_workers=2, pin_memory=True)

    # Infer dims from first file
    d0         = np.load(all_files[0])
    obs_dim    = int(d0["obs"].shape[1])
    n_bez      = int(d0["n_bez"])
    n_substeps = int(d0["n_substeps"])
    h_window   = float(d0["h_window"])
    print(f"[Config] obs_dim={obs_dim}  n_bez={n_bez}  "
          f"n_substeps={n_substeps}  h_window={h_window:.3f}s")

    # ── policy ───────────────────────────────────────────────────────────────
    policy = OfflineDronePolicy(
        n_bez=n_bez,
        hidden_dims=args.hidden_dims,
        emb_dim=args.emb_dim,
        sample_steps=args.sample_steps,
        device=device,
        ctrl_pts_max=args.ctrl_pts_max,
        own_dim=args.own_dim,
        neighbor_dim=args.neighbor_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=args.lr * 0.1
    )

    start_epoch  = 0
    best_val     = float("inf")
    save_path    = Path(args.save_path)
    best_path    = save_path.parent / (save_path.stem + "_best" + save_path.suffix)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"[Resume] epoch={start_epoch}  best_val={best_val:.4f}")

    def _save(path: Path, epoch: int) -> None:
        torch.save({
            "policy":      policy.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "epoch":       epoch,
            "best_val":    best_val,
            # env / data config
            "obs_dim":     obs_dim,
            "n_bez":       n_bez,
            "n_substeps":  n_substeps,
            "h_window":    h_window,
            # model architecture (needed to rebuild at eval time)
            "own_dim":     args.own_dim,
            "neighbor_dim": args.neighbor_dim,
            "hidden_dims": args.hidden_dims,
            "emb_dim":     args.emb_dim,
            "sample_steps": args.sample_steps,
            "ctrl_pts_max": args.ctrl_pts_max,
        }, str(path))

    # ── eval subprocess helper ────────────────────────────────────────────────
    _eval_script = args.eval_script or str(Path(__file__).parent / "eval_bc_dmpc.py")

    def run_eval(checkpoint_path: Path) -> dict[str, float] | None:
        """Launch eval_bc_dmpc.py in a subprocess. Returns metrics or None."""
        import json
        import subprocess
        import tempfile
        if not Path(_eval_script).exists():
            print(f"[Eval] script not found: {_eval_script}", flush=True)
            return None
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
        cmd = [
            sys.executable, _eval_script,
            "--checkpoint",  str(checkpoint_path),
            "--num_envs",    str(args.eval_num_envs),
            "--num_drones",  str(args.eval_num_drones),
            "--eval_steps",  str(args.eval_steps),
            "--task",        args.eval_task,
            "--n_substeps",  str(n_substeps),
            "--output_json", json_path,
            "--headless",
        ]
        print(f"[Eval] launching: {' '.join(cmd)}", flush=True)
        try:
            proc = subprocess.run(
                cmd, timeout=args.eval_timeout,
                capture_output=False,  # show stdout/stderr live
            )
            if proc.returncode != 0:
                print(f"[Eval] subprocess exited with code {proc.returncode}", flush=True)
                return None
            with open(json_path) as f:
                return json.load(f)
        except subprocess.TimeoutExpired:
            print(f"[Eval] timed out after {args.eval_timeout}s", flush=True)
            return None
        except Exception as e:
            print(f"[Eval] error: {e}", flush=True)
            return None
        finally:
            Path(json_path).unlink(missing_ok=True)

    # ── wandb ────────────────────────────────────────────────────────────────
    if args.wandb and _WANDB_AVAILABLE:
        _wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                    config=vars(args))

    # ── training loop ─────────────────────────────────────────────────────────
    global_step = start_epoch * len(train_loader)
    epoch_bar = tqdm(range(start_epoch, args.n_epochs), desc="Epochs",
                     unit="ep", dynamic_ncols=True)

    for epoch in epoch_bar:
        # ── train ──────────────────────────────────────────────────────────
        policy.train()
        train_metrics: dict[str, list[float]] = {
            "loss/total": [], "loss/flow": [], "loss/vel_aux": [], "loss/acc_aux": []
        }
        batch_bar = tqdm(train_loader, desc=f"  train ep{epoch+1}",
                         unit="batch", leave=False, dynamic_ncols=True)
        for batch in batch_bar:
            obs      = batch["obs"].to(device)
            ctrl_pts = batch["ctrl_pts"].to(device)
            ref_vel  = batch["ref_vel"].to(device)
            ref_acc  = batch["ref_acc"].to(device)

            loss, m = policy.compute_loss(obs, ctrl_pts, ref_vel, ref_acc,
                                          lambda_vel=args.lambda_vel,
                                          lambda_acc=args.lambda_acc,
                                          update_norm=True)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            policy.cnf.update_proximal()
            global_step += 1

            for k, v in m.items():
                train_metrics[k].append(v)
            batch_bar.set_postfix(loss=f"{m['loss/total']:.4f}",
                                  flow=f"{m['loss/flow']:.4f}")

            if args.wandb and _WANDB_AVAILABLE:
                _wandb.log({"step/loss":    m["loss/total"],
                            "step/flow":    m["loss/flow"],
                            "step/vel_aux": m["loss/vel_aux"],
                            "step/acc_aux": m["loss/acc_aux"]},
                           step=global_step)

        scheduler.step()
        t_avg = {k: float(np.mean(v)) for k, v in train_metrics.items()}
        t_avg["lr"] = float(scheduler.get_last_lr()[0])

        # ── val ────────────────────────────────────────────────────────────
        policy.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  val", unit="batch",
                              leave=False, dynamic_ncols=True):
                obs      = batch["obs"].to(device)
                ctrl_pts = batch["ctrl_pts"].to(device)
                ref_vel  = batch["ref_vel"].to(device)
                ref_acc  = batch["ref_acc"].to(device)
                _, vm = policy.compute_loss(obs, ctrl_pts, ref_vel, ref_acc,
                                            lambda_vel=args.lambda_vel,
                                            lambda_acc=args.lambda_acc,
                                            update_norm=False)
                val_losses.append(vm["loss/total"])
        policy.train()
        val_loss = float(np.mean(val_losses))

        # ── best checkpoint ────────────────────────────────────────────────
        is_new_best = val_loss < best_val
        if is_new_best:
            best_val = val_loss
            _save(best_path, epoch + 1)
            tqdm.write(f"  [best] epoch={epoch+1}  val={val_loss:.4f} → {best_path}")

        # ── sim eval (optional) ───────────────────────────────────────────
        eval_metrics: dict[str, float] = {}
        should_eval = (
            (args.eval_every > 0 and (epoch + 1) % args.eval_every == 0)
            or (args.eval_on_best and is_new_best)
        )
        if should_eval and Path(_eval_script).exists():
            tqdm.write(f"  [eval] running sim eval at epoch {epoch+1} …")
            _em = run_eval(best_path)
            if _em is not None:
                eval_metrics = _em
                tqdm.write(
                    f"  [eval] success={eval_metrics.get('eval_success_rate', 0):.3f}"
                    f"  return={eval_metrics.get('eval_return', float('nan')):.2f}"
                )

        # ── epoch bar postfix ──────────────────────────────────────────────
        epoch_bar.set_postfix(
            loss=f"{t_avg['loss/total']:.4f}",
            val=f"{val_loss:.4f}",
            lr=f"{t_avg['lr']:.2e}",
        )

        if args.wandb and _WANDB_AVAILABLE:
            log_dict = {"epoch":             epoch + 1,
                        "epoch/loss_total":  t_avg["loss/total"],
                        "epoch/loss_flow":   t_avg["loss/flow"],
                        "epoch/loss_vel":    t_avg["loss/vel_aux"],
                        "epoch/loss_acc":    t_avg["loss/acc_aux"],
                        "epoch/val_loss":    val_loss,
                        "epoch/lr":          t_avg["lr"]}
            if eval_metrics:
                log_dict["eval/success_rate"] = eval_metrics.get("eval_success_rate", 0.0)
                log_dict["eval/return"]        = eval_metrics.get("eval_return", float("nan"))
                log_dict["eval/episodes"]      = eval_metrics.get("eval_episodes", 0)
            _wandb.log(log_dict, step=global_step)

        # ── periodic checkpoint ────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            _save(save_path, epoch + 1)

    if args.wandb and _WANDB_AVAILABLE:
        _wandb.finish()
    print(f"Done. best_val={best_val:.4f}  saved → {best_path}")


if __name__ == "__main__":
    main()
