from .dmpc_expert import DMPCExpert, DMPCParams
import numpy as np
import torch
from typing import Dict
import time

# ───────────────────────────────────────────────────────────────────────────
# Batched GPU ADMM QP solver
# ───────────────────────────────────────────────────────────────────────────
# Solution status codes for the custom GPU ADMM solver.
SOLVED = 1
SOLVED_INACCURATE = 2
MAX_ITER = -2
NONFINITE = -3

ADMM_SOLVED = SOLVED
ADMM_SOLVED_INACCURATE = SOLVED_INACCURATE
ADMM_MAX_ITER = MAX_ITER
ADMM_NONFINITE = NONFINITE

class BatchedGPUDMPC:
    """Batched ADMM solver with optional fixed soft-collision slots.

    Decision variable is ``x = [U]`` when ``max_collision_slots == 0`` and
    ``x = [U; eps]`` otherwise.  Soft collision rows use the same sign
    convention as the CPU OSQP path: ``A_col U - eps >= rhs`` with ``eps <= 0``.
    Dynamic collision row coefficients make the KKT matrix batch-dependent, so
    this solver uses batched Cholesky each call instead of one fixed factor.
    """

    def __init__(
        self,
        n_bez: int,
        H_bez: np.ndarray,
        A_cont: np.ndarray,
        M_deriv_0: list[np.ndarray],
        M_pos_hor: np.ndarray,
        M_acc_hor: np.ndarray,
        pmin_vec: np.ndarray,
        pmax_vec: np.ndarray,
        amin_vec: np.ndarray,
        amax_vec: np.ndarray,
        max_B: int,
        rho: float = 1.0,
        admm_iters: int = 50,
        alpha: float = 1.6,
        device: str | torch.device = "cuda",
        reg: float = 1e-6,
        eps_abs: float = 1e-3,
        eps_rel: float = 1e-3,
        max_collision_slots: int = 0,
        quad_coll: float = 1.0,
        lin_coll: float = -1.0e3,
    ) -> None:
        self.device = torch.device(device)
        self.n_bez = int(n_bez)
        self.max_collision_slots = max(0, int(max_collision_slots))
        self.n_var = self.n_bez + self.max_collision_slots
        self.rho = float(rho)
        self.admm_iters = int(admm_iters)
        self.alpha = float(alpha)
        self.max_B = int(max_B)
        self.reg = float(reg)
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)

        n_eq_cont = A_cont.shape[0]
        A_init_np = [M.astype(np.float64) for M in M_deriv_0]
        n_eq_init = sum(M.shape[0] for M in A_init_np)
        self.n_eq_cont = n_eq_cont
        self.n_eq_init = n_eq_init
        self.n_eq = n_eq_cont + n_eq_init

        A_ineq_np = np.vstack([M_pos_hor, M_acc_hor]).astype(np.float64)
        self.n_ineq_box = A_ineq_np.shape[0]
        self.n_slack_rows = self.max_collision_slots
        self.n_fixed = self.n_eq + self.n_ineq_box + self.n_slack_rows
        self.n_coll = self.max_collision_slots
        self.m = self.n_fixed + self.n_coll

        A_fixed_u = np.vstack([A_cont.astype(np.float64)] + A_init_np + [A_ineq_np])
        # Allocate the full ADMM constraint matrix, including dynamic soft-collision
        # rows.  The fixed rows are populated here; collision rows are filled per
        # solve() call from A_col_batch.
        A_fixed_np = np.zeros((self.m, self.n_var), dtype=np.float64)
        A_fixed_np[: A_fixed_u.shape[0], : self.n_bez] = A_fixed_u
        if self.max_collision_slots > 0:
            row0 = self.n_eq + self.n_ineq_box
            A_fixed_np[row0 : row0 + self.max_collision_slots, self.n_bez :] = np.eye(self.max_collision_slots)

        H_np = np.zeros((self.n_var, self.n_var), dtype=np.float64)
        H_np[: self.n_bez, : self.n_bez] = H_bez.astype(np.float64)
        if self.max_collision_slots > 0:
            H_np[self.n_bez :, self.n_bez :] = 2.0 * float(quad_coll) * np.eye(self.max_collision_slots)
        self._H = torch.from_numpy(0.5 * (H_np + H_np.T)).float().to(self.device)
        self._A_fixed = torch.from_numpy(A_fixed_np).float().to(self.device)

        q_slack = np.zeros(self.n_var, dtype=np.float32)
        if self.max_collision_slots > 0:
            q_slack[self.n_bez :] = float(lin_coll)
        self._q_slack = torch.from_numpy(q_slack).to(self.device)

        l_ineq = np.concatenate([pmin_vec, amin_vec]).astype(np.float32)
        u_ineq = np.concatenate([pmax_vec, amax_vec]).astype(np.float32)
        self._l_ineq = torch.from_numpy(l_ineq).to(self.device)
        self._u_ineq = torch.from_numpy(u_ineq).to(self.device)

        self._z = torch.zeros(max_B, self.m, dtype=torch.float32, device=self.device)
        self._y = torch.zeros(max_B, self.m, dtype=torch.float32, device=self.device)
        self._last_admm_stats: dict[str, np.ndarray] = {}

    def reset_warm_start(self, flat_indices: torch.Tensor | None = None) -> None:
        if flat_indices is None:
            self._z.zero_()
            self._y.zero_()
        else:
            idx = flat_indices.to(self.device)
            self._z[idx] = 0.0
            self._y[idx] = 0.0

    @torch.no_grad()
    def solve(
        self,
        q_batch: torch.Tensor,
        b_eq_batch: torch.Tensor,
        flat_indices: torch.Tensor,
        A_col_batch: torch.Tensor | None = None,
        l_col_batch: torch.Tensor | None = None,
        active_col_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = q_batch.shape[0]
        rho = self.rho
        alpha = self.alpha
        inv_rho = 1.0 / rho
        one_minus_alpha = 1.0 - alpha

        A = self._A_fixed.unsqueeze(0).expand(B, -1, -1).clone()
        l = torch.empty(B, self.m, dtype=torch.float32, device=self.device)
        u = torch.empty(B, self.m, dtype=torch.float32, device=self.device)
        l[:, : self.n_eq] = b_eq_batch
        u[:, : self.n_eq] = b_eq_batch
        box0 = self.n_eq
        box1 = box0 + self.n_ineq_box
        l[:, box0:box1].copy_(self._l_ineq)
        u[:, box0:box1].copy_(self._u_ineq)
        if self.max_collision_slots > 0:
            slack0 = box1
            slack1 = slack0 + self.max_collision_slots
            l[:, slack0:slack1] = -float("inf")
            u[:, slack0:slack1] = 0.0
            coll0 = self.n_fixed
            coll1 = coll0 + self.max_collision_slots
            l[:, coll0:coll1] = -float("inf")
            u[:, coll0:coll1] = float("inf")
            if A_col_batch is not None and l_col_batch is not None and active_col_batch is not None:
                S = min(self.max_collision_slots, A_col_batch.shape[1])
                A[:, coll0 : coll0 + S, : self.n_bez] = A_col_batch[:, :S]
                eye = torch.eye(self.max_collision_slots, device=self.device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
                A[:, coll0:coll1, self.n_bez:] = -eye
                active = active_col_batch[:, :S]
                l[:, coll0 : coll0 + S] = torch.where(active, l_col_batch[:, :S], l[:, coll0 : coll0 + S])
        q = torch.zeros(B, self.n_var, dtype=torch.float32, device=self.device)
        q[:, : self.n_bez] = q_batch
        q += self._q_slack

        z = self._z[flat_indices].clone()
        y = self._y[flat_indices].clone()
        z_prev = z.clone()
        x = torch.zeros(B, self.n_var, dtype=torch.float32, device=self.device)
        Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)

        At = A.transpose(1, 2).contiguous()
        KKT = self._H.unsqueeze(0) + rho * torch.bmm(At, A)
        eye = torch.eye(self.n_var, dtype=torch.float32, device=self.device).unsqueeze(0)
        KKT = 0.5 * (KKT + KKT.transpose(1, 2)) + self.reg * eye
        try:
            L = torch.linalg.cholesky(KKT)
        except RuntimeError:
            jitter = 10.0 * self.reg * eye
            L = torch.linalg.cholesky(KKT + jitter)
        L_T = L.transpose(1, 2).contiguous()

        for _ in range(self.admm_iters):
            rhs = -q + torch.bmm(At, (rho * z - y).unsqueeze(-1)).squeeze(-1)
            fwd = torch.linalg.solve_triangular(L, rhs.unsqueeze(-1), upper=False)
            x = torch.linalg.solve_triangular(L_T, fwd, upper=True).squeeze(-1)
            Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
            z_prev = z
            z_hat = alpha * Ax + one_minus_alpha * z_prev
            z = torch.clamp(z_hat + y * inv_rho, l, u)
            y = y + rho * (z_hat - z)

        self._z[flat_indices] = z
        self._y[flat_indices] = y

        pri_vec = Ax - z
        dual_vec = rho * torch.bmm(At, (z - z_prev).unsqueeze(-1)).squeeze(-1)
        pri_res = torch.linalg.vector_norm(pri_vec, ord=float("inf"), dim=1)
        dual_res = torch.linalg.vector_norm(dual_vec, ord=float("inf"), dim=1)
        ax_norm = torch.maximum(
            torch.linalg.vector_norm(Ax, ord=float("inf"), dim=1),
            torch.linalg.vector_norm(z, ord=float("inf"), dim=1),
        )
        q_norm = torch.linalg.vector_norm(q, ord=float("inf"), dim=1)
        aty_norm = torch.linalg.vector_norm(torch.bmm(At, y.unsqueeze(-1)).squeeze(-1), ord=float("inf"), dim=1)
        pri_tol = self.eps_abs + self.eps_rel * ax_norm
        dual_tol = self.eps_abs + self.eps_rel * torch.maximum(q_norm, aty_norm)
        finite = torch.isfinite(x).all(dim=1) & torch.isfinite(pri_res) & torch.isfinite(dual_res)
        solved = finite & (pri_res <= pri_tol) & (dual_res <= dual_tol)
        inaccurate = finite & (~solved) & (pri_res <= 10.0 * pri_tol) & (dual_res <= 10.0 * dual_tol)
        status = torch.full((B,), ADMM_MAX_ITER, dtype=torch.int32, device=self.device)
        status = torch.where(inaccurate, torch.full_like(status, ADMM_SOLVED_INACCURATE), status)
        status = torch.where(solved, torch.full_like(status, ADMM_SOLVED), status)
        status = torch.where(finite, status, torch.full_like(status, ADMM_NONFINITE))

        pri_ratio = pri_res / torch.clamp(pri_tol, min=1.0e-12)
        dual_ratio = dual_res / torch.clamp(dual_tol, min=1.0e-12)
        self._last_admm_stats = {
            "pri_res": pri_res.detach().cpu().numpy(),
            "dual_res": dual_res.detach().cpu().numpy(),
            "pri_tol": pri_tol.detach().cpu().numpy(),
            "dual_tol": dual_tol.detach().cpu().numpy(),
            "pri_ratio": pri_ratio.detach().cpu().numpy(),
            "dual_ratio": dual_ratio.detach().cpu().numpy(),
            "status": status.detach().cpu().numpy(),
        }
        return x[:, : self.n_bez], status




class GPUDMPCExpert(DMPCExpert):
    def __init__(self, num_drones: int, num_envs: int, params: DMPCParams | None = None, device="cpu",
                num_workers: int | None = None,):
        super().__init__(num_drones, num_envs, params, device, num_workers)
        self._precompute_static_obstacle_coeffs(None)
        
        # Batched GPU ADMM solver (None until enable_gpu_admm() is called).
        self._gpu_solver: BatchedGPUDMPC | None = None
        self._enable_gpu_admm(
            max_envs=self.p.max_envs,
            rho=self.p.rho,
            admm_iters=self.p.admm_iters,
            alpha=self.p.alpha
        )

    def _enable_gpu_admm(
        self,
        max_envs: int,
        rho: float = 1.0,
        admm_iters: int = 50,
        alpha: float = 1.6,
        device: str | torch.device | None = None,
    ) -> None:
        """Enable batched GPU ADMM solver.

        Collision constraints are handled as a penalty term added to the linear
        cost q (equivalent to eliminating the slack variable from the original
        soft-constraint formulation).  All agents are solved on GPU — no CPU fallback.

        Args:
            max_envs:    Upper bound on the number of parallel environments.
            rho:         ADMM penalty.  Default 1.0.
            admm_iters:  ADMM iterations per window.  Default 50.
            alpha:       Over-relaxation factor.  Range (1, 2).  Default 1.6.
            device:      Target device.  Defaults to CUDA if available, else CPU.
        """
        if device is None:
            device = (
                self.device
                if self.device.type == "cuda"
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        def _make_solver(H_bez: np.ndarray) -> BatchedGPUDMPC:
            return BatchedGPUDMPC(
                n_bez=self.n_bez,
                H_bez=H_bez,
                A_cont=self.A_continuity,
                M_deriv_0=self._M_deriv_0,
                M_pos_hor=self.M_pos_hor,
                M_acc_hor=self.M_acc_hor,
                pmin_vec=self._pmin,
                pmax_vec=self._pmax,
                amin_vec=self._amin,
                amax_vec=self._amax,
                max_B=max_envs * self.N,
                rho=rho,
                admm_iters=admm_iters,
                alpha=alpha,
                device=device,
                eps_abs=self.p.admm_eps_abs,
                eps_rel=self.p.admm_eps_rel,
                reg=self.p.reg,
                max_collision_slots=self.p.admm_collision_slots,
                quad_coll=self.p.admm_quad_coll,
                lin_coll=self.p.admm_lin_coll,
            )

        self._gpu_solvers: dict[str, BatchedGPUDMPC] = {
            "free": _make_solver(self._H_bez_free),
            "obs": _make_solver(self._H_bez_obs),
            "emergency": _make_solver(self._H_bez_emergency),
        }
        # Compatibility handle for existing code that only needs device/shape metadata.
        self._gpu_solver = self._gpu_solvers["free"]

        # Numpy state arrays — GPU hot path reads/writes these instead of
        # the Python dict, eliminating per-agent Python loop overhead.
        K    = self.p.k_hor
        deg1 = self.p.deg_poly + 1
        N_   = self.N
        self._st_has        = np.zeros((max_envs, N_), dtype=bool)
        self._st_U          = np.zeros((max_envs, N_, self.n_bez), dtype=np.float64)
        self._st_u_pred     = np.zeros((max_envs, N_, K, 3),       dtype=np.float64)
        self._st_seeds      = np.zeros((max_envs, N_, deg1, 3),    dtype=np.float64)
        self._st_steps      = np.zeros((max_envs, N_),             dtype=np.int32)
        # Flag arrays for data logging — written per-replan, reset every compute() call
        self._st_replanned  = np.zeros((max_envs, N_), dtype=bool)
        self._st_reset_mode = np.zeros((max_envs, N_), dtype=bool)
        self._st_fallback   = np.zeros((max_envs, N_), dtype=bool)
        self._st_coll_ts    = np.full((max_envs, N_), -1, dtype=np.int32)
        self._st_emergency  = np.zeros((max_envs, N_), dtype=bool)
        self._st_fallback_type = np.full((max_envs, N_), "", dtype=object)
        self._st_priority_hold = np.zeros((max_envs, N_), dtype=bool)
        self._st_priority_scores = np.zeros((max_envs, N_), dtype=np.float64)
        self._st_solve_status = np.zeros((max_envs, N_), dtype=np.int32)

        # Per-agent neighbor index table: row i = indices of all j ≠ i  (N, N-1)
        self._neighbor_idx = np.array(
            [[j for j in range(N_) if j != i] for i in range(N_)],
            dtype=np.intp,
        )

        # Pre-cache float32 GPU tensors for Phase C — fixed matrices that never
        # change between calls, uploaded once here to avoid repeated CPU→GPU transfers.
        _dev = self._gpu_solver.device
        self._t_Fx    = torch.from_numpy(self._Fx_pos_pred.astype(np.float32)).to(_dev)
        self._t_Qd_free = torch.from_numpy(self._Q_diag_free.astype(np.float32)).to(_dev)
        self._t_Qd_obs = torch.from_numpy(self._Q_diag_obs.astype(np.float32)).to(_dev)
        self._t_Qd_emergency = torch.from_numpy(self._Q_diag_emergency.astype(np.float32)).to(_dev)
        self._t_Qd    = self._t_Qd_free
        self._t_Gp    = torch.from_numpy(self._G_pos_pred.astype(np.float32)).to(_dev)
        self._t_Phi   = torch.from_numpy(
            self.M_pos_hor.astype(np.float32).reshape(K, 3, self.n_bez)
        ).to(_dev)  # (K, 3, n_bez)
        self._t_thinv = torch.from_numpy(
            np.diag(self._Theta_inv).astype(np.float32)
        ).to(_dev)  # (3,)

    def reset(self, env_ids: torch.Tensor | None = None, obstacle_info: Dict | None = None) -> None:
        """Drop per-agent cached state for the given envs (all by default)."""
        self._precompute_static_obstacle_coeffs(obstacle_info)
        if env_ids is None:
            self._state.clear()
            self._priority_hold.clear()
            for solver in self._gpu_solvers.values():
                solver.reset_warm_start()
            self._st_has[:]        = False
            self._st_steps[:]      = 0
            self._st_replanned[:]  = False
            self._st_reset_mode[:] = False
            self._st_fallback[:]   = False
            self._st_coll_ts[:]    = -1
            self._st_emergency[:]  = False
            self._st_fallback_type[:] = ""
            self._st_priority_hold[:] = False
            self._st_priority_scores[:] = 0.0
            self._st_solve_status[:] = 0
            return
        
        ids_list = [int(i) for i in env_ids.detach().cpu().tolist()]
        ids = set(ids_list)
        self._state = {k: v for k, v in self._state.items() if k[0] not in ids}
        self._priority_hold = {k: v for k, v in self._priority_hold.items() if k[0] not in ids}

        ids_np = np.array(ids_list, dtype=np.intp)
        self._st_has[ids_np]        = False
        self._st_steps[ids_np]      = 0
        self._st_replanned[ids_np]  = False
        self._st_reset_mode[ids_np] = False
        self._st_fallback[ids_np]   = False
        self._st_coll_ts[ids_np]    = -1
        self._st_emergency[ids_np]  = False
        self._st_fallback_type[ids_np] = ""
        self._st_priority_hold[ids_np] = False
        self._st_priority_scores[ids_np] = 0.0
        self._st_solve_status[ids_np] = 0
        flat = [e * self.N + i for e in ids for i in range(self.N)]
        if flat:
            flat_t = torch.tensor(flat, dtype=torch.long)
            for solver in self._gpu_solvers.values():
                solver.reset_warm_start(flat_t)

    def _compute_priority_scores_array(
        self,
        pos_local: np.ndarray,
        vel: np.ndarray,
        goal_local: np.ndarray,
        env_arr: np.ndarray,
    ) -> np.ndarray:
        """Vectorized priority scores for the GPU path, with goal hysteresis."""
        scores = np.zeros((len(env_arr), self.N), dtype=np.float64)
        if not self.p.enable_priority:
            return scores

        pos_e = pos_local[env_arr]
        vel_e = vel[env_arr]
        goal_e = goal_local[env_arr]
        to_goal = goal_e - pos_e
        goal_dist = np.linalg.norm(to_goal, axis=-1)

        hold_prev = self._st_priority_hold[env_arr]
        hold_new = np.where(
            hold_prev,
            goal_dist <= self.p.priority_goal_exit_dist,
            goal_dist <= self.p.priority_goal_enter_dist,
        )
        self._st_priority_hold[env_arr] = hold_new

        vel_norm = np.linalg.norm(vel_e, axis=-1)
        denom = np.maximum(vel_norm * goal_dist, self.p.priority_velocity_eps)
        align = np.sum(vel_e * to_goal, axis=-1) / denom
        align = np.clip(align, -1.0, 1.0)
        active = (goal_dist > self.p.priority_velocity_eps) & (vel_norm > self.p.priority_velocity_eps)
        align = np.where(active, align, 0.0)

        scores = (
            self.p.priority_goal_dist_weight * goal_dist
            + self.p.priority_velocity_align_weight * align
        )
        scores = np.where(hold_new, self.p.priority_hold_score, scores)
        return scores.astype(np.float64)

    def _static_membership_batch(
        self,
        e_arr: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return batched static-obstacle membership and first violated obstacle id."""
        B = len(e_arr)
        if self._num_static_obstacles <= 0 or self._obstacle_pos_local is None:
            return np.zeros(B, dtype=bool), np.full(B, -1, dtype=np.int32)

        pts = np.asarray(points, dtype=np.float64).reshape(B, -1, 3)
        obs = self._obstacle_pos_local[e_arr]
        theta = self._obs_theta_inv[e_arr]
        diff = pts[:, :, None, :] - obs[:, None, :, :]
        scaled = np.einsum("bmoc,bocd->bmod", diff, theta)
        metric = np.linalg.norm(scaled, axis=-1)
        thresh = (self._obs_rmin[e_arr] * self.p.obs_g_factor)[:, None, :]
        inside_by_obs = (metric < thresh).any(axis=1)
        inside = inside_by_obs.any(axis=1)
        violated = np.where(inside, inside_by_obs.argmax(axis=1), -1).astype(np.int32)
        return inside, violated

    def _plan_batched_gpu(
        self,
        e_arr: np.ndarray,              # (B,) env indices
        i_arr: np.ndarray,              # (B,) agent indices
        nbr_preds_arr: np.ndarray,      # (E, N, N-1, K, 3) — prebuilt by compute()
        valid_nbr_arr: np.ndarray,      # (E, N, N-1) bool
        priority_scores_arr: np.ndarray,
        nbr_priority_scores_arr: np.ndarray,
        pos_np_local: np.ndarray,
        vel_np: np.ndarray,
        goal_np_local: np.ndarray,
    ) -> None:
        """Solve a batch of replanning QPs on GPU.

        All phases are fully vectorised — zero per-agent Python loops:
        - state read     : numpy fancy indexing into _st_* arrays
        - goal-tracking q: one batched matmul
        - collision      : numpy broadcast over (B, K-1, N-1, 3)
        - collision grad : einsum
        - result write   : batch fancy-index assignment to _st_* arrays
        """
        _t0 = time.perf_counter()

        K      = self.p.k_hor
        n_bez  = self.n_bez
        N      = self.N
        gs     = self._gpu_solver
        n_eq   = gs.n_eq
        n_eq_c = gs.n_eq_cont
        rmin   = self.p.rmin
        g_thr  = self.p.agent_g_factor * rmin
        qc     = self.p.admm_quad_coll
        B      = len(e_arr)
        self._gpu_plan_calls = getattr(self, "_gpu_plan_calls", 0) + 1
        _do_log = (self._gpu_plan_calls % 100 == 1)

        # ── Phase A+B: zero-loop state read + vectorised init ─────────────
        # All reads are numpy fancy indexing into _st_* arrays — no Python loop.
        deg1   = self.p.deg_poly + 1
        eps_v  = self.p.eps_velocity
        amin_v = np.asarray(self.p.amin, dtype=np.float64)
        amax_v = np.asarray(self.p.amax, dtype=np.float64)
        h_dyn  = self.p.h

        pos_b  = pos_np_local[e_arr, i_arr]    # (B, 3)
        vel_b  = vel_np[e_arr, i_arr]          # (B, 3)
        goal_b = goal_np_local[e_arr, i_arr]   # (B, 3)
        x0_batch   = np.concatenate([pos_b, vel_b], axis=-1)  # (B, 6)
        goal_batch = goal_b                                    # (B, 3)

        has_prev       = self._st_has[e_arr, i_arr]            # (B,) bool
        prev_seeds_all = self._st_seeds[e_arr, i_arr]          # (B, deg1, 3)
        own_pred_batch = self._st_u_pred[e_arr, i_arr].copy()  # (B, K, 3)
        U_prev_b       = self._st_U[e_arr, i_arr]              # (B, n_bez)

        # nbr arrays from prebuilt (E, N, N-1, K, 3) — fancy index, no loop
        nbr_batch = nbr_preds_arr[e_arr, i_arr]   # (B, N-1, K, 3)
        valid_nbr = valid_nbr_arr[e_arr, i_arr]   # (B, N-1)
        own_priority = priority_scores_arr[e_arr, i_arr]  # (B,)
        nbr_priority = nbr_priority_scores_arr[e_arr, i_arr]  # (B, N-1)
        nbr_ids = self._neighbor_idx[i_arr]  # (B, N-1)
        N_max     = N - 1

        # PD rollout for new agents — vectorised over all new agents simultaneously
        new_mask = ~has_prev
        if new_mask.any():
            p_new    = pos_b[new_mask].copy()         # (n_new, 3)
            v_new    = np.zeros_like(p_new)           # (n_new, 3)
            goal_new = goal_b[new_mask]               # (n_new, 3)
            for k in range(K):
                a = np.clip(2.0*(goal_new - p_new) - 1.5*v_new, amin_v, amax_v)
                p_new = p_new + v_new*h_dyn + 0.5*a*(h_dyn*h_dyn)
                v_new = v_new + a*h_dyn
                own_pred_batch[new_mask, k] = p_new

        # Vectorised event-triggered init (Eq. 15-17)
        err   = pos_b - prev_seeds_all[:, 0]
        denom = -(vel_b + np.sign(vel_b + 1e-12) * eps_v)
        denom = np.where(np.abs(denom) < 1e-6,
                         np.sign(denom) * 1e-6 + 1e-6, denom)
        f_vec   = err**5 / denom
        in_band = ((f_vec > self.p.f_min) & (f_vec < self.p.f_max)).all(-1)
        do_reset = (~has_prev) | (~in_band)

        seeds_batch = prev_seeds_all.copy()
        seeds_batch[do_reset, 0]  = pos_b[do_reset]
        seeds_batch[do_reset, 1:] = 0.0

        n_emergency = max(1, min(self.p.emergency_prediction_steps, K))
        emergency_points = np.concatenate(
            [pos_b[:, None, :], own_pred_batch[:, :n_emergency, :]], axis=1
        )
        emergency_mode, violated_obs = self._static_membership_batch(e_arr, emergency_points)
        emergency_active = emergency_mode if self.p.enable_emergency else np.zeros_like(emergency_mode, dtype=bool)

        n_seeds    = n_eq - n_eq_c
        seeds_flat = seeds_batch.reshape(B, n_seeds).astype(np.float32)

        _t_init = time.perf_counter() - _t0

        # ── Phase B: nbr arrays already built — no loop ────────────────────
        _t1 = time.perf_counter()
        own_batch = own_pred_batch                  # (B, K, 3)
        _t_assemble = time.perf_counter() - _t1

        # ── Phase C: q computation on GPU ─────────────────────────────────
        _t2 = time.perf_counter()
        dev = gs.device

        # Upload per-call inputs once
        x0_t    = torch.from_numpy(x0_batch.astype(np.float32)).to(dev)     # (B, 6)
        goal_t  = torch.from_numpy(goal_batch.astype(np.float32)).to(dev)   # (B, 3)
        own_t   = torch.from_numpy(own_batch.astype(np.float32)).to(dev)    # (B, K, 3)
        nbr_t   = torch.from_numpy(nbr_batch.astype(np.float32)).to(dev)    # (B, N-1, K, 3)
        valid_t = torch.from_numpy(valid_nbr).to(dev)                        # (B, N-1)
        own_prio_t = torch.from_numpy(own_priority.astype(np.float32)).to(dev)  # (B,)
        nbr_prio_t = torch.from_numpy(nbr_priority.astype(np.float32)).to(dev)  # (B, N-1)
        nbr_ids_t = torch.from_numpy(nbr_ids.astype(np.int64)).to(dev)          # (B, N-1)
        i_ids_t = torch.from_numpy(i_arr.astype(np.int64)).to(dev)              # (B,)

        # C1 — affine residual shared by all cost modes. The Q diagonal is
        # selected after collision/emergency mode detection below.
        residual = x0_t @ self._t_Fx.T - goal_t.repeat(1, K)   # (B, 3K)
        q_penalty_t = torch.zeros(B, n_bez, dtype=torch.float32, device=dev)
        soft_slots = max(0, int(self.p.admm_collision_slots))
        use_soft_slots = soft_slots > 0
        A_col_np = np.zeros((B, soft_slots, n_bez), dtype=np.float32) if use_soft_slots else None
        l_col_np = np.zeros((B, soft_slots), dtype=np.float32) if use_soft_slots else None
        active_col_np = np.zeros((B, soft_slots), dtype=bool) if use_soft_slots else None
        soft_slot_counts = np.zeros(B, dtype=np.int32)
        soft_slot_overflow = 0

        # C2 — collision detection: own[k] vs nbr[k+1] for k in [0, K-1)
        # own_t[:,:-1]: (B,K-1,3) → (B,K-1,1,3)
        # nbr_t[:,:,1:].permute(0,2,1,3): (B,N-1,K-1,3) → (B,K-1,N-1,3)
        diff  = (own_t[:, :-1, None, :]
                 - nbr_t[:, :, 1:, :].permute(0, 2, 1, 3))    # (B,K-1,N-1,3)
        d     = diff * self._t_thinv                            # Theta-scaled
        norms = torch.sqrt((d * d).sum(-1))                     # (B,K-1,N-1)
        norms = torch.where(valid_t[:, None, :], norms,
                            torch.full_like(norms, float('inf')))

        coll_any = (norms < rmin).any(-1)                       # (B, K-1)
        has_coll = coll_any.any(-1)                             # (B,)
        kc_arr   = torch.where(has_coll, coll_any.long().argmax(-1),
                               has_coll.new_full((B,), -1))     # (B,) 0-indexed

        # C3 — collision penalty gradient for agents that have a collision
        if has_coll.any():
            coll_idx = has_coll.nonzero(as_tuple=True)[0]      # (B_c,)
            B_c  = coll_idx.shape[0]
            kc_c = kc_arr[coll_idx] + 1                        # 1-indexed (B_c,)

            d_kc = d[coll_idx, kc_c - 1, :, :]                # (B_c, N-1, 3)
            n_kc = norms[coll_idx, kc_c - 1, :]               # (B_c, N-1)

            ns   = torch.where(n_kc > 1e-6, n_kc, torch.ones_like(n_kc))
            udir = d_kc / ns.unsqueeze(-1)                     # (B_c, N-1, 3)
            ut   = udir * self._t_thinv                        # (B_c, N-1, 3)

            Phi_kc = self._t_Phi[kc_c - 1]                    # (B_c, 3, n_bez)
            dg_dU  = torch.einsum('bnk,bkm->bnm', ut, Phi_kc) # (B_c, N-1, n_bez)

            # neighbour positions at timestep kc_c[b] for each j
            jo     = torch.arange(N_max, device=dev).unsqueeze(0).expand(B_c, N_max)
            nbr_kc = nbr_t[
                coll_idx.unsqueeze(1).expand(B_c, N_max),
                jo,
                kc_c.unsqueeze(1).expand(B_c, N_max),
                :
            ]                                                   # (B_c, N-1, 3)

            rhs    = (ut * nbr_kc).sum(-1) + rmin              # (B_c, N-1)
            active = valid_t[coll_idx] & (n_kc < g_thr)        # (B_c, N-1)
            if self.p.enable_priority:
                own_p = own_prio_t[coll_idx].unsqueeze(1)
                nbr_p = nbr_prio_t[coll_idx]
                nbr_id = nbr_ids_t[coll_idx]
                own_id = i_ids_t[coll_idx].unsqueeze(1)
                margin = torch.minimum(
                    torch.full_like(nbr_p, 1e-3),
                    0.01 * torch.maximum(torch.abs(own_p), torch.abs(nbr_p)),
                )
                prioritized = (own_p > nbr_p + margin) | ((torch.abs(own_p - nbr_p) < margin) & (own_id > nbr_id))
                active = active & (~prioritized)

            U_prev_c = torch.from_numpy(
                U_prev_b[coll_idx.cpu().numpy()].astype(np.float32)
            ).to(dev)                                           # (B_c, n_bez)
            pred  = torch.einsum('bnm,bm->bn', dg_dU, U_prev_c)  # (B_c, N-1)
            viol  = torch.where(active,
                                torch.clamp(rhs - pred, min=0.0),
                                torch.zeros_like(pred))

            if use_soft_slots:
                coll_idx_np = coll_idx.cpu().numpy()
                dg_np = dg_dU.detach().cpu().numpy().astype(np.float32)
                rhs_np = rhs.detach().cpu().numpy().astype(np.float32)
                active_np = active.detach().cpu().numpy()
                for rr, b_local in enumerate(coll_idx_np):
                    for jj in np.nonzero(active_np[rr])[0]:
                        slot = int(soft_slot_counts[b_local])
                        if slot >= soft_slots:
                            soft_slot_overflow += 1
                            continue
                        A_col_np[b_local, slot] = dg_np[rr, jj]
                        l_col_np[b_local, slot] = rhs_np[rr, jj]
                        active_col_np[b_local, slot] = True
                        soft_slot_counts[b_local] += 1
            else:
                pg = -2.0 * qc * torch.einsum('bn,bnm->bm', viol, dg_dU)  # (B_c, n_bez)
                q_penalty_t[coll_idx] += pg

        # C4 — static-obstacle penalty. Unlike inter-agent CA, static obstacle
        # penalties are added for every predicted guard violation timestep.
        static_has_coll_np = np.zeros(B, dtype=bool)
        static_sample_lists: list[list[int]] = [[] for _ in range(B)]
        if self._num_static_obstacles > 0 and self._obstacle_pos_local is not None:
            O = self._num_static_obstacles
            obs_pos_np = self._obstacle_pos_local[e_arr].astype(np.float32)  # (B,O,3)
            obs_theta_np = self._obs_theta_inv[e_arr].astype(np.float32)     # (B,O,3,3)
            obs_rmin_np = self._obs_rmin[e_arr].astype(np.float32)           # (B,O)
            obs_g_np = self._obs_g_thres[e_arr].astype(np.float32)           # (B,O)
            if self.p.enable_emergency and violated_obs.size == B:
                rows = np.arange(B)
                valid_v = (violated_obs >= 0) & emergency_active
                obs_rmin_np[rows[valid_v], violated_obs[valid_v]] *= self.p.emergency_repel_factor

            obs_pos_t = torch.from_numpy(obs_pos_np).to(dev)
            obs_theta_t = torch.from_numpy(obs_theta_np).to(dev)
            obs_rmin_t = torch.from_numpy(obs_rmin_np).to(dev)
            obs_g_t = torch.from_numpy(obs_g_np).to(dev)

            obs_diff = own_t[:, 1:, None, :] - obs_pos_t[:, None, :, :]  # (B,K-1,O,3)
            obs_d = torch.einsum('bkoc,bocd->bkod', obs_diff, obs_theta_t)
            obs_norms = torch.sqrt((obs_d * obs_d).sum(-1))
            obs_active = obs_norms < obs_g_t[:, None, :]

            if obs_active.any():
                b_idx, k_rel, o_idx = obs_active.nonzero(as_tuple=True)
                static_has_coll_np = obs_active.any(dim=(1, 2)).cpu().numpy()
                active_np = obs_active.cpu().numpy()
                for b_local in np.nonzero(static_has_coll_np)[0]:
                    static_sample_lists[b_local] = sorted((np.nonzero(active_np[b_local])[0] + 1).astype(int).tolist())

                d_sel = obs_d[b_idx, k_rel, o_idx, :]       # (R,3)
                n_sel = obs_norms[b_idx, k_rel, o_idx]      # (R,)
                ns = torch.where(n_sel > 1e-6, n_sel, torch.ones_like(n_sel))
                unit = d_sel / ns.unsqueeze(-1)
                theta_sel = obs_theta_t[b_idx, o_idx, :, :] # (R,3,3)
                ut = torch.einsum('rc,rcd->rd', unit, theta_sel)
                Phi_obs = self._t_Phi[k_rel + 1]            # (R,3,n_bez)
                dg_obs = torch.einsum('rk,rkm->rm', ut, Phi_obs)
                rhs_obs = (ut * obs_pos_t[b_idx, o_idx, :]).sum(-1) + obs_rmin_t[b_idx, o_idx]

                U_prev_obs = torch.from_numpy(U_prev_b.astype(np.float32)).to(dev)
                if use_soft_slots:
                    b_np = b_idx.cpu().numpy()
                    dg_np = dg_obs.detach().cpu().numpy().astype(np.float32)
                    rhs_np = rhs_obs.detach().cpu().numpy().astype(np.float32)
                    for rr, b_local in enumerate(b_np):
                        slot = int(soft_slot_counts[b_local])
                        if slot >= soft_slots:
                            soft_slot_overflow += 1
                            continue
                        A_col_np[b_local, slot] = dg_np[rr]
                        l_col_np[b_local, slot] = rhs_np[rr]
                        active_col_np[b_local, slot] = True
                        soft_slot_counts[b_local] += 1
                else:
                    pred_obs = torch.einsum('rm,rm->r', dg_obs, U_prev_obs[b_idx])
                    viol_obs = torch.clamp(rhs_obs - pred_obs, min=0.0)
                    pg_obs = -2.0 * qc * viol_obs.unsqueeze(-1) * dg_obs
                    q_penalty_t.index_add_(0, b_idx, pg_obs)

        has_coll_np = has_coll.cpu().numpy()
        mode_np = np.where(emergency_active, 2, np.where(has_coll_np | static_has_coll_np, 1, 0)).astype(np.int8)
        q_t = torch.empty(B, n_bez, dtype=torch.float32, device=dev)
        for mode_code, q_diag in (
            (0, self._t_Qd_free),
            (1, self._t_Qd_obs),
            (2, self._t_Qd_emergency),
        ):
            rows_np = np.nonzero(mode_np == mode_code)[0]
            if rows_np.size == 0:
                continue
            rows_t = torch.from_numpy(rows_np.astype(np.int64)).to(dev)
            q_t[rows_t] = 2.0 * (residual[rows_t] * q_diag) @ self._t_Gp
        q_t += q_penalty_t

        _t_q = time.perf_counter() - _t2

        # ── Phase D: GPU solve ────────────────────────────────────────────
        _t3 = time.perf_counter()
        b_eq  = np.zeros((B, n_eq), dtype=np.float32)
        b_eq[:, n_eq_c:] = seeds_flat
        idx_np = (e_arr * N + i_arr).astype(np.int64)

        # q_t already on GPU from Phase C. Split the batch by mode so each
        # group uses the matching pre-factorized Hessian/KKT matrix.
        beq_t = torch.from_numpy(b_eq).to(dev)
        idx_t = torch.from_numpy(idx_np).to(dev)
        U_raw = np.empty((B, n_bez), dtype=np.float32)
        solve_status = np.full(B, MAX_ITER, dtype=np.int32)
        admm_diag = {
            "pri_res": np.full(B, np.nan, dtype=np.float32),
            "dual_res": np.full(B, np.nan, dtype=np.float32),
            "pri_tol": np.full(B, np.nan, dtype=np.float32),
            "dual_tol": np.full(B, np.nan, dtype=np.float32),
            "pri_ratio": np.full(B, np.nan, dtype=np.float32),
            "dual_ratio": np.full(B, np.nan, dtype=np.float32),
        }
        status_by_mode: dict[str, tuple[int, int, int, int]] = {}
        for mode_code, solver_name in ((0, "free"), (1, "obs"), (2, "emergency")):
            rows_np = np.nonzero(mode_np == mode_code)[0]
            if rows_np.size == 0:
                continue
            rows_t = torch.from_numpy(rows_np.astype(np.int64)).to(dev)
            solver = self._gpu_solvers[solver_name]
            if use_soft_slots:
                A_col_t = torch.from_numpy(A_col_np[rows_np]).to(dev)
                l_col_t = torch.from_numpy(l_col_np[rows_np]).to(dev)
                active_col_t = torch.from_numpy(active_col_np[rows_np]).to(dev)
                U_t, solve_status_t = solver.solve(q_t[rows_t], beq_t[rows_t], idx_t[rows_t], A_col_t, l_col_t, active_col_t)
            else:
                U_t, solve_status_t = solver.solve(q_t[rows_t], beq_t[rows_t], idx_t[rows_t])
            U_raw[rows_np] = U_t.cpu().numpy()
            solve_status[rows_np] = solve_status_t.cpu().numpy().astype(np.int32)
            stats = getattr(solver, "_last_admm_stats", {})
            for key in admm_diag:
                if key in stats:
                    admm_diag[key][rows_np] = np.asarray(stats[key], dtype=np.float32)
            status_mode = solve_status[rows_np]
            status_by_mode[solver_name] = (
                int((status_mode == SOLVED).sum()),
                int((status_mode == SOLVED_INACCURATE).sum()),
                int((status_mode == MAX_ITER).sum()),
                int((status_mode == NONFINITE).sum()),
            )
        _t_gpu = time.perf_counter() - _t3

        fallback_mask = (solve_status < 0) | (~np.isfinite(U_raw).all(axis=1))
        fallback_type = np.full(B, "", dtype=object)
        if self.p.enable_admm_fallback and fallback_mask.any():
            shifted_M = self.bezier.sample_matrix(
                np.minimum(self._times_hor + self.p.h, self._times_hor[-1]), deriv=0
            )
            for b in np.nonzero(fallback_mask)[0]:
                U_fb = None
                u_fb = None
                prefer_previous = (not emergency_active[b]) and has_prev[b]
                if prefer_previous and np.isfinite(U_prev_b[b]).all():
                    shifted = (shifted_M @ U_prev_b[b]).reshape(K, 3)
                    inside, _ = self._static_membership_batch(
                        e_arr[b:b + 1], shifted[: max(1, self.p.emergency_prediction_steps)][None, :, :]
                    )
                    if not inside[0]:
                        try:
                            U_fb = np.linalg.lstsq(self.M_pos_hor, shifted.reshape(-1), rcond=None)[0]
                            u_fb = shifted
                            fallback_type[b] = "shift_previous"
                        except np.linalg.LinAlgError:
                            U_fb = None
                            u_fb = None

                if U_fb is None or u_fb is None:
                    p_hold = pos_b[b].copy()
                    p_cur = pos_b[b].copy()
                    v_cur = vel_b[b].copy()
                    u_fb = np.zeros((K, 3), dtype=np.float64)
                    for k in range(K):
                        a = np.clip(2.0 * (p_hold - p_cur) - 2.0 * v_cur, amin_v, amax_v)
                        p_cur = p_cur + v_cur * self.p.h + 0.5 * a * self.p.h ** 2
                        v_cur = v_cur + a * self.p.h
                        u_fb[k] = p_cur
                    try:
                        U_fb = np.linalg.lstsq(self.M_pos_hor, u_fb.reshape(-1), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        U_fb = np.zeros(n_bez, dtype=np.float64)
                    fallback_type[b] = "brake_hover"

                U_raw[b] = U_fb.astype(np.float32)

        # ── Phase E: result extraction (batched matmuls + batch numpy write)
        _t4 = time.perf_counter()
        U_f64     = U_raw.astype(np.float64)                         # (B, n_bez)
        u_pred_b  = (U_f64 @ self.M_pos_hor.T).reshape(B, K, 3)    # (B, K, 3)
        deg1      = self.p.deg_poly + 1
        seeds_new = np.stack(
            [(U_f64 @ self._M_deriv_h[r].T).reshape(B, 3)
             for r in range(deg1)], axis=1
        )  # (B, deg1, 3)

        # has_coll / kc_arr are torch tensors from Phase C — convert to numpy
        has_coll_np = has_coll.cpu().numpy()
        kc_arr_np   = kc_arr.cpu().numpy()
        static_first = np.asarray([samples[0] if samples else -1 for samples in static_sample_lists], dtype=np.int32)
        kc_meta = np.where(has_coll_np, kc_arr_np + 1, static_first)   # 1-indexed coll ts

        # Batch write all fields to numpy arrays — zero per-agent loop
        self._st_U[e_arr, i_arr]          = U_f64
        self._st_u_pred[e_arr, i_arr]     = u_pred_b
        self._st_seeds[e_arr, i_arr]      = seeds_new
        self._st_steps[e_arr, i_arr]      = 0
        self._st_has[e_arr, i_arr]        = True
        self._st_replanned[e_arr, i_arr]  = True
        self._st_reset_mode[e_arr, i_arr] = do_reset
        self._st_fallback[e_arr, i_arr]   = fallback_mask
        self._st_fallback_type[e_arr, i_arr] = fallback_type
        self._st_emergency[e_arr, i_arr]  = emergency_active
        self._st_coll_ts[e_arr, i_arr]    = kc_meta
        self._st_solve_status[e_arr, i_arr] = solve_status
        _t_extract = time.perf_counter() - _t4

        if _do_log:
            n_coll = int(has_coll_np.sum())
            n_free = int((mode_np == 0).sum())
            n_obs = int((mode_np == 1).sum())
            n_emerg = int((mode_np == 2).sum())
            n_solved = int((solve_status > 0).sum())
            n_fail = int((solve_status < 0).sum())
            n_total = n_fail + n_solved
            n_inaccurate = int((solve_status == SOLVED_INACCURATE).sum())
            n_soft_rows = int(soft_slot_counts.sum()) if use_soft_slots else 0
            n_fb_shift = int((fallback_type == "shift_previous").sum())
            n_fb_brake = int((fallback_type == "brake_hover").sum())
            n_fb_other = int(fallback_mask.sum()) - n_fb_shift - n_fb_brake
            n_max_iter = int((solve_status == MAX_ITER).sum())
            n_nonfinite = int((solve_status == NONFINITE).sum())
            slot_max = int(soft_slot_counts.max()) if use_soft_slots and soft_slot_counts.size else 0
            slot_mean = float(soft_slot_counts.mean()) if use_soft_slots and soft_slot_counts.size else 0.0

            def _nan_stats(vals: np.ndarray) -> tuple[float, float, float]:
                finite_vals = vals[np.isfinite(vals)]
                if finite_vals.size == 0:
                    return float("nan"), float("nan"), float("nan")
                return float(finite_vals.min()), float(finite_vals.mean()), float(finite_vals.max())

            pri_r_min, pri_r_mean, pri_r_max = _nan_stats(admm_diag["pri_ratio"])
            dual_r_min, dual_r_mean, dual_r_max = _nan_stats(admm_diag["dual_ratio"])
            pri_min, pri_mean, pri_max = _nan_stats(admm_diag["pri_res"])
            dual_min, dual_mean, dual_max = _nan_stats(admm_diag["dual_res"])
            tol_p_min, tol_p_mean, tol_p_max = _nan_stats(admm_diag["pri_tol"])
            tol_d_min, tol_d_mean, tol_d_max = _nan_stats(admm_diag["dual_tol"])
            mode_status = "/".join(
                f"{name}:s{vals[0]} i{vals[1]} m{vals[2]} n{vals[3]}"
                for name, vals in status_by_mode.items()
            )
            total  = (_t_init + _t_assemble + _t_q + _t_gpu + _t_extract) * 1e3
            print(
                f"[DMPC gpu_admm] call={self._gpu_plan_calls}  B={B}  coll={n_coll}"
                f"  mode=free:{n_free}/obs:{n_obs}/emg:{n_emerg}"
                f"  solved={n_solved}/{n_total}  fail={n_fail}/{n_total}  inaccurate={n_inaccurate}/{n_total}"
                f"  max_iter={n_max_iter}/{n_total}  nonfinite={n_nonfinite}/{n_total}"
                f"  soft_slots={n_soft_rows}/{B * soft_slots if use_soft_slots else 0}  max={slot_max}  mean={slot_mean:.1f}  overflow={soft_slot_overflow}"
                f"  fallback=shift:{n_fb_shift}/brake:{n_fb_brake}/other:{n_fb_other}"
                f"  rho={self.p.rho:g}  alpha={self.p.alpha:g}  iters={self.p.admm_iters}"
                f"  pri_ratio=min/mean/max:{pri_r_min:.2g}/{pri_r_mean:.2g}/{pri_r_max:.2g}"
                f"  dual_ratio=min/mean/max:{dual_r_min:.2g}/{dual_r_mean:.2g}/{dual_r_max:.2g}"
                f"  pri=min/mean/max:{pri_min:.2g}/{pri_mean:.2g}/{pri_max:.2g}"
                f"  dual=min/mean/max:{dual_min:.2g}/{dual_mean:.2g}/{dual_max:.2g}"
                f"  tol_p/tol_d_mean={tol_p_mean:.2g}/{tol_d_mean:.2g}"
                f"  status_by_mode={mode_status}"
                f"  init={_t_init*1e3:.1f}ms"
                f"  assemble={_t_assemble*1e3:.1f}ms"
                f"  q+coll={_t_q*1e3:.1f}ms"
                f"  gpu={_t_gpu*1e3:.1f}ms"
                f"  extract={_t_extract*1e3:.1f}ms"
                f"  total={total:.1f}ms",
                flush=True,
            )

    def _sync_state_dict_from_arrays(self, env_list: list[int]) -> None:
        """Mirror GPU array state into the base-class state dict for script helpers."""
        K = self.p.k_hor
        for e in env_list:
            for i in range(self.N):
                if not self._st_has[e, i]:
                    continue
                coll_ts = int(self._st_coll_ts[e, i])
                collision_indices = [max(coll_ts - 1, 0)] if coll_ts >= 0 else []
                self._state[(e, i)] = {
                    "U": self._st_U[e, i].copy(),
                    "u_pred": self._st_u_pred[e, i].copy(),
                    "seeds": self._st_seeds[e, i].copy(),
                    "steps": int(self._st_steps[e, i]),
                    "last_replanned": bool(self._st_replanned[e, i]),
                    "last_reset_mode": bool(self._st_reset_mode[e, i]),
                    "last_fallback": bool(self._st_fallback[e, i]),
                    "last_emergency_mode": bool(self._st_emergency[e, i]),
                    "fallback_type": str(self._st_fallback_type[e, i]),
                    "collision_timestep": coll_ts,
                    "collision_sample_indices": [idx for idx in collision_indices if idx < K],
                    "priority_scores": self._st_priority_scores[e, self._neighbor_idx[i]].copy(),
                    "solve_status": int(self._st_solve_status[e, i]),
                }

    # ───────────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def compute(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        goal_w: torch.Tensor,
        env_origins: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one control step.

        Internally replans the Bezier curve every ``h`` seconds and samples it
        at the env control rate ``Ts`` in between, matching Algorithm 1 of the
        paper.

        Args:
            pos_w / vel_w: ``(E, N, 3)`` world-frame state.
            goal_w: ``(E, N, 3)`` world-frame goals.
            env_origins: ``(E, 3)`` env-local origin offset. The workspace
                ``pmin/pmax`` is interpreted in env-local coordinates so all
                envs share the same QP geometry.
            env_ids: optional 1-D tensor selecting a subset of envs to plan
                for. For envs not in ``env_ids`` we output ``(pos_w, 0)``.

        Returns:
            ``(ref_pos_w, ref_vel_w)`` each ``(E, N, 3)``.
        """
        E = pos_w.shape[0]
        device = self.device
        ref_pos = pos_w.clone()
        ref_vel = torch.zeros_like(vel_w)

        env_iter = range(E) if env_ids is None else env_ids.detach().cpu().tolist()

        pos_np = pos_w.detach().cpu().numpy().astype(np.float64)
        vel_np = vel_w.detach().cpu().numpy().astype(np.float64)
        goal_np = goal_w.detach().cpu().numpy().astype(np.float64)
        if env_origins is not None:
            origins_np = env_origins.detach().cpu().numpy().astype(np.float64)
            pos_np_local = pos_np - origins_np[:, None, :]
            goal_np_local = goal_np - origins_np[:, None, :]
            if self._obstacle_pos_w is not None:
                self._obstacle_pos_local = self._obstacle_pos_w - origins_np[:, None, :]
        else:
            origins_np = None
            pos_np_local = pos_np
            goal_np_local = goal_np
            self._obstacle_pos_local = self._obstacle_pos_w

        ts = self.p.ts
        h_total = (self.p.k_hor - 1) * self.p.h
        K = self.p.k_hor
        N = self.N
        env_list = list(env_iter)
        _tc0 = time.perf_counter()
        E_n = len(env_list)
        env_arr  = np.array(env_list, dtype=np.intp)           # (E_n,)
        e_all    = np.repeat(env_arr, N)                        # (E_n*N,)
        i_all    = np.tile(np.arange(N, dtype=np.intp), E_n)   # (E_n*N,)

        # sub-timers (set in GPU block, 0 for OSQP path)
        _t_flag = _t_nbr = _t_check = 0.0

        # Phase 1a: reset flags — numpy batch, no Python loop
        _ta = time.perf_counter()
        self._st_replanned[env_arr, :]  = False
        self._st_reset_mode[env_arr, :] = False
        self._st_fallback[env_arr, :]   = False
        self._st_emergency[env_arr, :]  = False
        self._st_fallback_type[env_arr, :] = ""
        _t_flag = time.perf_counter() - _ta

        priority_scores_local = self._compute_priority_scores_array(
            pos_np_local, vel_np, goal_np_local, env_arr
        )
        priority_scores_arr = np.zeros((E, N), dtype=np.float64)
        priority_scores_arr[env_arr] = priority_scores_local
        self._st_priority_scores[env_arr] = priority_scores_local

        # Phase 1b: build nbr_preds_arr via numpy fancy indexing — no loop
        # _st_u_pred[e,i] = (K,3) plan or zeros for uninitialised agents.
        # For uninitialised agents: use stationary position tiled K times.
        _tb = time.perf_counter()
        upred_all = self._st_u_pred[env_arr]         # (E_n, N, K, 3)
        has_all   = self._st_has[env_arr]            # (E_n, N)
        pos_tiled = np.repeat(
            pos_np_local[env_arr, :, np.newaxis, :], K, axis=2
        )  # (E_n, N, K, 3)
        upred_safe = np.where(
            has_all[:, :, np.newaxis, np.newaxis], upred_all, pos_tiled
        )  # (E_n, N, K, 3)
        # nbr_preds_arr[e_idx, i] = predictions of the N-1 neighbours of i
        nbr_preds_local = upred_safe[:, self._neighbor_idx]   # (E_n,N,N-1,K,3)
        valid_nbr_local = has_all[:, self._neighbor_idx]       # (E_n, N, N-1)
        nbr_priority_local = priority_scores_local[:, self._neighbor_idx]
        nbr_preds_arr = np.zeros((E, N, N - 1, K, 3), dtype=upred_safe.dtype)
        valid_nbr_arr = np.zeros((E, N, N - 1), dtype=bool)
        nbr_priority_scores_arr = np.zeros((E, N, N - 1), dtype=np.float64)
        nbr_preds_arr[env_arr] = nbr_preds_local
        valid_nbr_arr[env_arr] = valid_nbr_local
        nbr_priority_scores_arr[env_arr] = nbr_priority_local
        _t_nbr = time.perf_counter() - _tb

        # Phase 2: replan condition via numpy — no dict loop
        _tc_chk = time.perf_counter()
        needs_replan = (
            (~self._st_has[e_all, i_all]) |
            (self._st_steps[e_all, i_all] % self.n_substeps == 0)
        )  # (E_n*N,)
        replan_e_arr = e_all[needs_replan]
        replan_i_arr = i_all[needs_replan]
        _t_check = time.perf_counter() - _tc_chk

        _tc1 = time.perf_counter()

        if replan_e_arr.size > 0:
            self._plan_batched_gpu(
                replan_e_arr, replan_i_arr,
                nbr_preds_arr, valid_nbr_arr,
                priority_scores_arr, nbr_priority_scores_arr,
                pos_np_local, vel_np, goal_np_local,
            )

        _tc2 = time.perf_counter()

        # Phase 3: sample references — no per-agent Python loop
        _S_pos = getattr(self, "_S_pos_cache", None)
        if _S_pos is None or len(_S_pos) != self.n_substeps:
            self._S_pos_cache = [
                self.bezier.sample_matrix(
                    np.array([min((k + 1) * ts, h_total)]), deriv=0
                ).astype(np.float32) for k in range(self.n_substeps)
            ]
            self._S_vel_cache = [
                self.bezier.sample_matrix(
                    np.array([min((k + 1) * ts, h_total)]), deriv=1
                ).astype(np.float32) for k in range(self.n_substeps)
            ]

        all_U   = self._st_U[e_all, i_all].astype(np.float32)    # (E_n*N, n_bez)
        all_sub = (self._st_steps[e_all, i_all] % self.n_substeps).astype(np.int32)
        self._st_steps[e_all, i_all] += 1                        # batch increment
        self._sync_state_dict_from_arrays(env_list)
        if origins_np is not None:
            all_ori = np.repeat(origins_np[env_arr], N, axis=0).astype(np.float32)
        else:
            all_ori = np.zeros((E_n * N, 3), dtype=np.float32)

        # One batched matmul per unique substep index.
        _tp3 = time.perf_counter()
        all_pos = np.empty((E_n * N, 3), dtype=np.float32)
        all_vel = np.empty((E_n * N, 3), dtype=np.float32)
        for k in range(self.n_substeps):
            mask = all_sub == k
            if not mask.any():
                continue
            # S_pos/vel: (1, n_bez) → squeeze for broadcast
            all_pos[mask] = (all_U[mask] @ self._S_pos_cache[k].T).reshape(-1, 3) + all_ori[mask]
            all_vel[mask] = (all_U[mask] @ self._S_vel_cache[k].T).reshape(-1, 3)
        _t_phase3 = time.perf_counter() - _tp3

        # Write results back to ref tensors — two bulk CUDA transfers.
        _tw = time.perf_counter()
        all_pos_t = torch.from_numpy(all_pos).to(device).reshape(E_n, N, 3)
        all_vel_t = torch.from_numpy(all_vel).to(device).reshape(E_n, N, 3)
        if env_ids is None:
            ref_pos[:] = all_pos_t
            ref_vel[:] = all_vel_t
        else:
            env_t = torch.tensor(env_list, dtype=torch.long, device=device)
            ref_pos[env_t] = all_pos_t
            ref_vel[env_t] = all_vel_t
        _t_write = time.perf_counter() - _tw

        _tc3 = time.perf_counter()

        # ── per-call timing log (every 10 calls) ──────────────────────────
        self._compute_calls = getattr(self, "_compute_calls", 0) + 1
        if self._compute_calls % 100 == 1:
            print(
                f"[DMPC timing] call={self._compute_calls}"
                f"  flag={_t_flag*1e3:.1f}ms"
                f"  nbr={_t_nbr*1e3:.1f}ms"
                f"  check={_t_check*1e3:.1f}ms"
                f"  replan={(_tc2-_tc1)*1e3:.1f}ms"
                f"  phase3={_t_phase3*1e3:.1f}ms"
                f"  write={_t_write*1e3:.1f}ms"
                f"  total={(_tc3-_tc0)*1e3:.1f}ms",
                flush=True,
            )

        return ref_pos, ref_vel