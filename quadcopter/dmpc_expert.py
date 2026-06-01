# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Faithful Python re-implementation of the online DMPC expert from

    C. E. Luis, M. Vukosavljev, A. P. Schoellig.
    "Online Trajectory Generation with Distributed Model Predictive Control for
    Multi-Robot Motion Planning." IEEE RA-L 5(2):604-611, April 2020.

and its C++ reference (``git_branch/online_dmpc_to_learning/cpp``).

When the paper and the C++ implementation diverge, this file follows the
**paper** first and back-fills C++-only refinements that are compatible with it.
Concretely:

* Collision avoidance is **input-space** as in the paper Section III.E (the
  paper reports it beat state-space in their experiments); the C++ reference
  uses state-space, so this file intentionally differs there.
* Event-triggered replanning uses the paper's ``f_max = 0.8`` threshold; the
  C++ reference uses ``0.08`` (likely a typo) and is *not* followed here.
* The C++ ``H_f`` / ``H_o`` cost-mode switching is preserved: we precompute
  both Hessians and pick the obs profile when at least one collision row was
  added to the QP, matching ``generator.cpp::setErrorPenaltyMatrices`` /
  ``buildQP``.
* Reference-signal continuity is enforced on *all* ``deg_poly + 1`` initial
  derivatives across replans (C++ ``_x0_ref`` carry-over), not just position.

Implementation choices mirror the paper / config.json:

* **Trajectory parameterisation.** ``l`` concatenated Bezier curves of degree
  ``p`` per agent; decision variables are the control points (Section III.A).
  Sampling and derivative matrices are built once via the Bernstein-to-power
  basis transform so position / velocity / acceleration samples of the input
  trajectory are exact linear functions of the control points (Section III.D).

* **Agent prediction model.** A 6-state, 3-input discrete linear system
  obtained by ZOH discretisation of the identified second-order closed-loop
  tracking dynamics for the Crazyflie (``zeta_xy / tau_xy`` etc., from
  ``config.json``). Predicted positions over the horizon are linear in the
  Bezier control points: ``X = A0 x0 + Lambda M_pos U`` (Eq. 4 of the paper).

* **Input continuity** (Section III.C) is enforced as equality constraints up
  to ``deg_poly`` derivatives at every Bezier segment boundary.

* **Physical limits** (Section III.D, "third alternative"): exact samples of
  the input *and its second derivative* at every MPC step are constrained to
  the workspace ``[pmin, pmax]`` / acceleration ``[amin, amax]`` boxes via
  linear inequalities -- no convex-hull conservatism.

* **On-demand collision avoidance in the input space** (Section III.E).
  After each replanning round agents broadcast their planned input trajectory
  ``Pi_i = u_i samples``. At the next round, agent ``i`` checks for the first
  collision step ``k_c`` against neighbour ``j``'s broadcast and adds the
  half-space cut

  ``eta^T Theta^{-1} (u_i[k_c - 1] - Pi_j[k_c]) >= rmin + eps_ij``

  with linearisation direction taken from the seed predictions. The
  ellipsoidal metric uses ``Theta = diag(1, 1, height_scaling)`` from the
  config so the avoidance volume is consistent with the C++ default.

* **Soft slack** ``eps_ij <= 0`` is added as a decision variable per neighbour;
  the cost contributes ``quad_coll * eps^2 + lin_coll * eps`` (Eq. 13).

* **Event-triggered replanning** (Section IV, Eq. 15-17): the activation
  function ``f_n[k_t]`` decides whether to seed the optimisation with the
  next sample of the previous solution (normal operation) or with the
  measured state + zero higher derivatives (disturbed operation).

* **MPC step ``h`` vs. control rate ``Ts``** (Section V). The expert replans
  every ``h`` seconds and emits a sample of the first Bezier segment every
  ``Ts`` seconds in between -- exactly matching the paper / C++ time bases.

Public entry point::

    expert = DMPCExpert(num_drones=N, params=DMPCParams(), device=env.device)
    ref_pos, ref_vel = expert.compute(pos_w, vel_w, goal_w, env_origins)
    action = env.ref_to_action(ref_pos, ref_vel)
    env.step(action)
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import torch
from scipy.special import comb, factorial

try:
    import osqp
except ImportError as e:  # pragma: no cover
    raise ImportError("DMPC expert requires osqp. Install with `pip install osqp`.") from e


# ───────────────────────────────────────────────────────────────────────────
# Parameters (mirror cpp/config/config.json)
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class DMPCParams:
    # Horizon / discretisation
    k_hor: int = 20 #16
    h: float = 0.1          # MPC replanning period (s)
    ts: float = 0.02        # subsample / control period (s)

    # Bezier parameterisation
    deg: int = 5            # polynomial degree p
    num_segments: int = 3   # number of concatenated curves l
    deg_poly: int = 3       # continuity order
    t_segment: float | None = None  # Auto: ((k_hor - 1) * h) / num_segments    # originally 0.5
    dim: int = 3

    def __post_init__(self):
        if self.num_segments <= 0:
            raise ValueError("num_segments must be positive")
        if self.t_segment is None:
            self.t_segment = (self.k_hor - 1) * self.h / self.num_segments

    # Identified second-order closed-loop tracking dynamics
    zeta_xy: float = 0.6502
    tau_xy: float = 0.3815
    zeta_z: float = 0.9103
    tau_z: float = 0.3

    # Workspace + actuation limits
    pmin: tuple[float, float, float] = (-1.5, -1.5, 0.2)
    pmax: tuple[float, float, float] = (1.5, 1.5, 2.2)
    amin: tuple[float, float, float] = (-1.5, -1.5, -1.5)#(-2.0, -2.0, -2.0) #(-4.0,-4.0,-4.0) # (-1.0, -1.0, -1.0)
    amax: tuple[float, float, float] = (1.5, 1.5, 1.5)#(2.0, 2.0, 2.0) # (4.0, 4.0, 4.0) # (1.0, 1.0, 1.0)
    # Cost weights. ``s_free`` / ``spd_f`` weight the *last spd_f* steps when no
    # collision is predicted; ``s_obs`` / ``spd_o`` weight the *last spd_o* steps
    # when a collision row is added to the QP (mirrors the C++ free/obs cost
    # mode switching in generator.cpp::setErrorPenaltyMatrices).
    s_free: float = 100.0
    spd_f: int = 3
    s_obs: float = 100.0
    spd_o: int = 1
    acc_cost: float = 1.0 #10.0 #8e-3
    lin_coll: float = -1.0e3
    quad_coll: float = 1.0

    # Collision parameters
    rmin: float = 0.5 #0.3
    height_scaling: float = 2.0
    g_factor: float = 2.0

    # Event-triggered replanning
    f_min: float = -0.01
    f_max: float = 0.8
    eps_velocity: float = 0.01

    # OSQP options
    osqp_eps_abs: float = 1e-3
    osqp_eps_rel: float = 1e-3
    osqp_max_iter: int = 200 #200
    osqp_polish: bool = False


# ───────────────────────────────────────────────────────────────────────────
# Bezier basis (l segments of degree p)
# ───────────────────────────────────────────────────────────────────────────
class BezierBasis:
    """Linear sampling / continuity matrices for concatenated Bezier curves.

    The decision vector for one agent is ``U = [vec_P_seg0, vec_P_seg1, ...]``
    where ``vec_P_seg = [P_0_x, P_0_y, P_0_z, P_1_x, ...]`` for the ``p+1``
    control points of that segment. Time inside segment ``s`` is expressed via
    the local coordinate ``tau = (t - s*T_seg) / T_seg in [0, 1]``.
    """

    def __init__(self, p: int, l: int, t_segment: float, dim: int = 3):
        self.p = p
        self.l = l
        self.T_seg = t_segment
        self.dim = dim
        self.n_per_seg = (p + 1) * dim
        self.n_var = l * self.n_per_seg

        # Bernstein -> power-basis transform:
        #   B_{m,p}(tau) = sum_k B2P[m, k] tau^k
        # with B2P[m, k] = (-1)^{k-m} C(p, k) C(k, m) for k >= m.
        B2P = np.zeros((p + 1, p + 1))
        for m in range(p + 1):
            for k in range(m, p + 1):
                B2P[m, k] = ((-1) ** (k - m)) * comb(p, k) * comb(k, m)
        self.B2P = B2P

    # Row (length p+1) such that  r @ [P_0, ..., P_p] = S^{(c)}(tau * T_seg).
    def sampling_row(self, tau: float, deriv: int = 0) -> np.ndarray:
        p = self.p
        T = self.T_seg
        v = np.zeros(p + 1)
        if deriv > p:
            return v @ self.B2P.T
        for k in range(deriv, p + 1):
            v[k] = (factorial(k) / factorial(k - deriv)) * (tau ** (k - deriv)) / (T ** deriv)
        return v @ self.B2P.T

    def sample_matrix(self, times: np.ndarray, deriv: int = 0) -> np.ndarray:
        """Stacked sampling matrix for the c-th derivative at ``len(times)``
        time instants, mapping ``U -> samples in R^{dim*K}``."""
        K = len(times)
        M = np.zeros((self.dim * K, self.n_var))
        I_d = np.eye(self.dim)
        for idx, t in enumerate(times):
            s = min(int(t / self.T_seg + 1e-9), self.l - 1)
            tau = (t - s * self.T_seg) / self.T_seg
            tau = max(0.0, min(1.0, tau))
            row = self.sampling_row(tau, deriv)
            block = np.kron(row[None, :], I_d)
            col = s * self.n_per_seg
            M[idx * self.dim : (idx + 1) * self.dim, col : col + self.n_per_seg] = block
        return M

    def continuity_matrix(self, deg_poly: int) -> np.ndarray:
        """Equality matrix enforcing C^{deg_poly}-continuity at every segment
        boundary. Returns ``A`` such that ``A @ U = 0``."""
        if self.l < 2:
            return np.zeros((0, self.n_var))
        nb = self.l - 1
        rows = (deg_poly + 1) * nb
        A = np.zeros((self.dim * rows, self.n_var))
        I_d = np.eye(self.dim)
        for b in range(nb):
            for c in range(deg_poly + 1):
                end_row = self.sampling_row(1.0, c)
                start_row = self.sampling_row(0.0, c)
                rr = b * (deg_poly + 1) + c
                col_b = b * self.n_per_seg
                col_b1 = (b + 1) * self.n_per_seg
                row_slice = slice(rr * self.dim, (rr + 1) * self.dim)
                A[row_slice, col_b : col_b + self.n_per_seg] = np.kron(end_row[None, :], I_d)
                A[row_slice, col_b1 : col_b1 + self.n_per_seg] = -np.kron(start_row[None, :], I_d)
        return A


# ───────────────────────────────────────────────────────────────────────────
# Second-order tracking dynamics + propagator
# ───────────────────────────────────────────────────────────────────────────
def _build_2nd_order_dynamics(p: DMPCParams) -> tuple[np.ndarray, np.ndarray]:
    """Identified Crazyflie closed-loop tracking dynamics, ZOH-discretised at
    step ``h``. State ``x = [px, py, pz, vx, vy, vz]``; input ``u = position
    reference``. Each axis is a second-order system parameterised by
    ``(zeta, tau)``; xy share the same parameters."""

    def per_axis(zeta: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
        omega = 1.0 / tau
        A = np.array([[0.0, 1.0], [-(omega ** 2), -2.0 * zeta * omega]])
        B = np.array([[0.0], [omega ** 2]])
        return A, B

    A_xy, B_xy = per_axis(p.zeta_xy, p.tau_xy)
    A_z, B_z = per_axis(p.zeta_z, p.tau_z)

    A_c = np.zeros((6, 6))
    B_c = np.zeros((6, 3))
    # x axis: idx 0 = pos, 3 = vel
    A_c[0, 0] = A_xy[0, 0]; A_c[0, 3] = A_xy[0, 1]
    A_c[3, 0] = A_xy[1, 0]; A_c[3, 3] = A_xy[1, 1]
    B_c[0, 0] = B_xy[0, 0]; B_c[3, 0] = B_xy[1, 0]
    # y axis: idx 1 = pos, 4 = vel
    A_c[1, 1] = A_xy[0, 0]; A_c[1, 4] = A_xy[0, 1]
    A_c[4, 1] = A_xy[1, 0]; A_c[4, 4] = A_xy[1, 1]
    B_c[1, 1] = B_xy[0, 0]; B_c[4, 1] = B_xy[1, 0]
    # z axis: idx 2 = pos, 5 = vel
    A_c[2, 2] = A_z[0, 0]; A_c[2, 5] = A_z[0, 1]
    A_c[5, 2] = A_z[1, 0]; A_c[5, 5] = A_z[1, 1]
    B_c[2, 2] = B_z[0, 0]; B_c[5, 2] = B_z[1, 0]

    # ZOH discretisation via the augmented matrix exponential.
    M = np.zeros((9, 9))
    M[:6, :6] = A_c
    M[:6, 6:] = B_c
    M_d = sla.expm(M * p.h)
    return M_d[:6, :6], M_d[:6, 6:]


def _build_state_propagator(A_d: np.ndarray, B_d: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``X = [x[1], ..., x[K]] = A0 x[0] + Lambda U``."""
    nx, nu = A_d.shape[0], B_d.shape[1]
    A0 = np.zeros((nx * K, nx))
    Lam = np.zeros((nx * K, nu * K))
    A_pow = [np.eye(nx)]
    for _ in range(K + 1):
        A_pow.append(A_d @ A_pow[-1])
    for k in range(K):
        A0[k * nx : (k + 1) * nx] = A_pow[k + 1]
        for j in range(k + 1):
            Lam[k * nx : (k + 1) * nx, j * nu : (j + 1) * nu] = A_pow[k - j] @ B_d
    return A0, Lam


# ───────────────────────────────────────────────────────────────────────────
# Batched GPU ADMM QP solver
# ───────────────────────────────────────────────────────────────────────────

class BatchedGPUDMPC:
    """Batched ADMM solver for the constant-structure DMPC QP.

    All (env, agent) pairs without active collision constraints share the same
    constraint matrix  A_fixed = [A_cont ; M_deriv_0[0..d] ; M_pos_hor ; M_acc_hor]
    and (in the free-cost mode) the same Hessian H.  The KKT factor L from
    H + rho*A^T*A  is computed once at init; every GPU solve then needs only
    two batched triangular solves plus element-wise clamp/axpy ops—all on GPU.

    Warm-start tensors (z, y) are maintained per (env, agent) across replanning
    windows, stored at flat index  e * N_agents + i  in pre-allocated buffers.

    Agents with active collision constraints must be handled by the caller
    (e.g., fall back to CPU OSQP).

    ADMM iteration (OSQP formulation):
        x-update:  (H + rho A^T A) x  =  -q + A^T (rho z - y)
        z-update:  z  =  clip( A x + y/rho,  l,  u )
        y-update:  y  =  y + rho (A x - z)

    Equality constraints are handled as  l_i = u_i = b_eq_i  (box of width 0).
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
    ) -> None:
        self.device = torch.device(device)
        self.n_bez = n_bez
        self.rho = rho
        self.admm_iters = admm_iters
        self.alpha = alpha  # OSQP-style over-relaxation (1.6 is OSQP default)
        self.max_B = max_B

        # ── Partition constraint rows ─────────────────────────────────────
        n_eq_cont = A_cont.shape[0]
        A_init_np = [M.astype(np.float64) for M in M_deriv_0]
        n_eq_init = sum(M.shape[0] for M in A_init_np)
        self.n_eq_cont = n_eq_cont
        self.n_eq_init = n_eq_init
        self.n_eq = n_eq_cont + n_eq_init

        A_ineq_np = np.vstack([M_pos_hor, M_acc_hor]).astype(np.float64)
        self.n_ineq = A_ineq_np.shape[0]
        self.m = self.n_eq + self.n_ineq

        # ── Constant A_fixed (m, n_bez) ───────────────────────────────────
        A_fixed_np = np.vstack(
            [A_cont.astype(np.float64)] + A_init_np + [A_ineq_np]
        )

        # ── KKT = H + rho * A^T A, Cholesky-factored in float64 ──────────
        KKT_np = H_bez.astype(np.float64) + rho * (A_fixed_np.T @ A_fixed_np)
        KKT_np = 0.5 * (KKT_np + KKT_np.T) + reg * np.eye(n_bez)
        KKT_t = torch.from_numpy(KKT_np).to(dtype=torch.float64, device=self.device)
        self._L = torch.linalg.cholesky(KKT_t).float()  # (n, n), lower-triangular float32
        self._L_T = self._L.T.contiguous()

        self._A = torch.from_numpy(A_fixed_np).float().to(self.device)  # (m, n)

        # ── Constant inequality bounds ────────────────────────────────────
        l_ineq = np.concatenate([pmin_vec, amin_vec]).astype(np.float32)
        u_ineq = np.concatenate([pmax_vec, amax_vec]).astype(np.float32)
        self._l_ineq = torch.from_numpy(l_ineq).to(self.device)
        self._u_ineq = torch.from_numpy(u_ineq).to(self.device)

        # ── Warm-start buffers (max_B, m) ─────────────────────────────────
        self._z = torch.zeros(max_B, self.m, dtype=torch.float32, device=self.device)
        self._y = torch.zeros(max_B, self.m, dtype=torch.float32, device=self.device)

    def reset_warm_start(self, flat_indices: torch.Tensor | None = None) -> None:
        """Zero warm-start for the given flat indices (all if None)."""
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
        q_batch: torch.Tensor,       # (B, n_bez) float32 on device
        b_eq_batch: torch.Tensor,    # (B, n_eq) float32 on device
        flat_indices: torch.Tensor,  # (B,) int64
    ) -> torch.Tensor:               # (B, n_bez) float32
        """Warm-started batched ADMM solve.  Returns primal solutions (B, n_bez)."""
        B = q_batch.shape[0]
        rho = self.rho
        A = self._A       # (m, n)
        L = self._L       # (n, n) lower-triangular
        L_T = self._L_T   # (n, n) upper-triangular

        # Per-problem bounds (B, m): equality rows have l=u=b_eq
        l = torch.empty(B, self.m, dtype=torch.float32, device=self.device)
        u = torch.empty(B, self.m, dtype=torch.float32, device=self.device)
        l[:, : self.n_eq] = b_eq_batch
        u[:, : self.n_eq] = b_eq_batch
        l[:, self.n_eq :].copy_(self._l_ineq)
        u[:, self.n_eq :].copy_(self._u_ineq)

        z = self._z[flat_indices].clone()  # (B, m)
        y = self._y[flat_indices].clone()
        inv_rho = 1.0 / rho
        alpha = self.alpha
        one_minus_alpha = 1.0 - alpha

        for _ in range(self.admm_iters):
            # x-update: KKT x = -q + A^T(rho*z - y)  →  two triangular solves
            rhs = (-q_batch + (rho * z - y) @ A).T.contiguous()  # (n, B)
            fwd = torch.linalg.solve_triangular(L, rhs, upper=False)  # (n, B)
            x = torch.linalg.solve_triangular(L_T, fwd, upper=True).T  # (B, n)

            # Over-relaxed z-update (Boyd 2011, Eq 3.21-3.23 with unscaled dual y):
            #   z_hat = alpha*Ax + (1-alpha)*z_prev   [over-relaxed proxy for Ax]
            #   z_new = clip(z_hat + y/rho, l, u)
            #   y    += rho * (z_hat - z_new)          [= standard ADMM when alpha=1]
            Ax = x @ A.T                                       # (B, m)
            z_hat = alpha * Ax + one_minus_alpha * z           # uses OLD z
            z = torch.clamp(z_hat + y * inv_rho, l, u)        # NEW z
            y = y + rho * (z_hat - z)

        self._z[flat_indices] = z
        self._y[flat_indices] = y
        return x


# ───────────────────────────────────────────────────────────────────────────
# DMPC expert
# ───────────────────────────────────────────────────────────────────────────
class DMPCExpert:
    """Distributed MPC expert producing position-reference trajectories for
    multiple agents, plus a method to evaluate the trajectory at the env
    control rate via :py:meth:`compute`."""

    def __init__(self, num_drones: int, params: DMPCParams | None = None, device="cpu",
                 num_workers: int | None = None):
        self.N = num_drones
        self.p = params or DMPCParams()
        if self.p.t_segment is None:
            self.p.t_segment = (self.p.k_hor - 1) * self.p.h / self.p.num_segments
        self.device = torch.device(device)

        # Number of control steps between replanning rounds (Ts samples per h).
        self.n_substeps = max(1, int(round(self.p.h / self.p.ts)))

        # Thread pool for parallel env processing (one thread per env).
        _nw = num_workers if num_workers is not None else min(os.cpu_count() or 4, 16)
        self._pool = ThreadPoolExecutor(max_workers=_nw) if _nw > 1 else None

        # ── Bezier basis + sampling matrices ──────────────────────────────
        self.bezier = BezierBasis(
            p=self.p.deg, l=self.p.num_segments,
            t_segment=self.p.t_segment, dim=self.p.dim,
        )
        self.n_bez = self.bezier.n_var
        K = self.p.k_hor
        h = self.p.h
        self._times_hor = np.array([k * h for k in range(K)])
        self.M_pos_hor = self.bezier.sample_matrix(self._times_hor, deriv=0)
        self.M_vel_hor = self.bezier.sample_matrix(self._times_hor, deriv=1)
        self.M_acc_hor = self.bezier.sample_matrix(self._times_hor, deriv=2)
        # Velocity sampling at each sub-step within one h-window: used by
        # get_vref_ctrl_pts() to fit Bezier control points for BC supervision.
        self._sub_times = np.array([(k + 1) * self.p.ts for k in range(self.n_substeps)])
        self.M_vel_sub = self.bezier.sample_matrix(self._sub_times, deriv=1)  # (3*n_sub, n_bez)
        self._vref_ctrl_cache: dict[int, np.ndarray] = {}  # K → Phi_pinv (K, n_sub)
        # Sampling matrices for derivatives 0..deg_poly at t = 0 (used to pin
        # the new Bezier's initial state to seeds from the previous solve, so
        # the reference signal stays C^{deg_poly}-continuous across replans,
        # cf. C++ generator.cpp::buildQP).
        self._M_deriv_0 = [
            self.bezier.sample_matrix(np.array([0.0]), deriv=r)
            for r in range(self.p.deg_poly + 1)
        ]
        # Same set sampled at t = h: extracts the seeds for the *next* solve
        # from the *current* solution (paper Section IV, Eq. 17 normal branch).
        self._M_deriv_h = [
            self.bezier.sample_matrix(np.array([self.p.h]), deriv=r)
            for r in range(self.p.deg_poly + 1)
        ]

        # Continuity equality matrix.
        self.A_continuity = self.bezier.continuity_matrix(self.p.deg_poly)
        self.n_continuity = self.A_continuity.shape[0]

        # ── Predicted-state propagator (input -> predicted state) ─────────
        self._A_d, self._B_d = _build_2nd_order_dynamics(self.p)
        self._A0_stack, self._Lam_stack = _build_state_propagator(self._A_d, self._B_d, K)
        # E_K extracts position from every 6-D state in the K-stack.
        E_K = np.zeros((3 * K, 6 * K))
        for k in range(K):
            E_K[3 * k : 3 * k + 3, 6 * k : 6 * k + 3] = np.eye(3)
        # Predicted position is linear in U_bez (via input samples) + affine in
        # x0:   p_pred = G @ U_bez + Fx @ x0.
        self._G_pos_pred = E_K @ self._Lam_stack @ self.M_pos_hor   # (3K, n_bez)
        self._Fx_pos_pred = E_K @ self._A0_stack                     # (3K, 6)

        # ── Constant cost on U_bez (slack quadratic added per-call) ───────
        self._build_constant_cost()

        # ── Constant box-constraint vectors ───────────────────────────────
        self._amin = np.tile(np.asarray(self.p.amin), K)
        self._amax = np.tile(np.asarray(self.p.amax), K)
        self._pmin = np.tile(np.asarray(self.p.pmin), K)
        self._pmax = np.tile(np.asarray(self.p.pmax), K)
        # Theta scaling for ellipsoid distances.
        self._Theta = np.diag([1.0, 1.0, self.p.height_scaling])
        self._Theta_inv = np.linalg.inv(self._Theta)

        # ── Per-agent persistent state ────────────────────────────────────
        # Indexed by (env_idx, agent_idx). Each entry stores:
        #   "U"          last solution, ndarray (n_bez,)
        #   "u_pred"     last predicted input samples (K, 3) for broadcast
        #   "seeds"      list of (deg_poly+1) length-3 ndarrays = the input
        #                Bezier's derivative values at t=h of the previous
        #                solve. Used as the next solve's initial conditions
        #                so the reference is C^{deg_poly}-continuous.
        #   "steps"      subsample counter modulo n_substeps
        self._state: dict[tuple[int, int], dict] = {}

        # Batched GPU ADMM solver (None until enable_gpu_admm() is called).
        self._gpu_solver: BatchedGPUDMPC | None = None

    # ───────────────────────────────────────────────────────────────────
    # Constant cost
    # ───────────────────────────────────────────────────────────────────
    def _build_constant_cost(self) -> None:
        """Pre-build the free / obs mode Hessians and `Q` diagonals.

        Mirrors `setErrorPenaltyMatrices` in the C++ reference: when no
        collision row is added to the QP we weight the *last spd_f* steps of
        the predicted-position error by `s_free`; when a collision row is
        added we switch to the obs profile (`s_obs` over the last `spd_o`
        steps). Both modes share the energy term on the squared acceleration.
        """
        K = self.p.k_hor
        G = self._G_pos_pred
        H_energy = self._build_energy_quadratic()

        def _make_mode(weight: float, spd: int) -> tuple[np.ndarray, np.ndarray]:
            q_weights = np.zeros(K)
            kappa = max(0, min(spd, K))
            for k in range(K - kappa, K):
                q_weights[k] = weight
            Q_diag = np.repeat(q_weights, 3)  # length 3K
            H_error = 2.0 * (G.T * Q_diag) @ G
            H = H_error + 2.0 * self.p.acc_cost * H_energy
            H = 0.5 * (H + H.T)
            return Q_diag, H

        self._Q_diag_free, self._H_bez_free = _make_mode(self.p.s_free, self.p.spd_f)
        self._Q_diag_obs, self._H_bez_obs = _make_mode(self.p.s_obs, self.p.spd_o)

    def _build_energy_quadratic(self) -> np.ndarray:
        """``H`` such that ``U^T H U = integral ||u''(t)||^2 dt``."""
        n = self.n_bez
        H = np.zeros((n, n))
        n_quad = 8
        xs, ws = np.polynomial.legendre.leggauss(n_quad)
        xs = 0.5 * (xs + 1.0)
        ws = 0.5 * ws
        T = self.bezier.T_seg
        for s in range(self.bezier.l):
            seg_H = np.zeros((self.bezier.n_per_seg, self.bezier.n_per_seg))
            for x, w in zip(xs, ws):
                row = self.bezier.sampling_row(x, deriv=2)
                block = np.kron(row[None, :], np.eye(self.bezier.dim))
                seg_H += w * (block.T @ block) * T
            c = s * self.bezier.n_per_seg
            H[c : c + self.bezier.n_per_seg, c : c + self.bezier.n_per_seg] = seg_H
        return H

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
        else:
            origins_np = None
            pos_np_local = pos_np
            goal_np_local = goal_np

        ts = self.p.ts
        h_total = (self.p.k_hor - 1) * self.p.h
        K = self.p.k_hor
        N = self.N
        env_list = list(env_iter)

        import time as _time
        _tc0 = _time.perf_counter()
        E_n = len(env_list)
        env_arr  = np.array(env_list, dtype=np.intp)           # (E_n,)
        e_all    = np.repeat(env_arr, N)                        # (E_n*N,)
        i_all    = np.tile(np.arange(N, dtype=np.intp), E_n)   # (E_n*N,)

        # sub-timers (set in GPU block, 0 for OSQP path)
        _t_flag = _t_nbr = _t_check = 0.0

        if self._gpu_solver is not None:
            # ── GPU path: fully vectorised, no per-agent Python loops ─────

            # Phase 1a: reset flags — numpy batch, no Python loop
            _ta = _time.perf_counter()
            self._st_replanned[env_arr, :]  = False
            self._st_reset_mode[env_arr, :] = False
            self._st_fallback[env_arr, :]   = False
            _t_flag = _time.perf_counter() - _ta

            # Phase 1b: build nbr_preds_arr via numpy fancy indexing — no loop
            # _st_u_pred[e,i] = (K,3) plan or zeros for uninitialised agents.
            # For uninitialised agents: use stationary position tiled K times.
            _tb = _time.perf_counter()
            upred_all = self._st_u_pred[env_arr]         # (E_n, N, K, 3)
            has_all   = self._st_has[env_arr]            # (E_n, N)
            pos_tiled = np.repeat(
                pos_np_local[env_arr, :, np.newaxis, :], K, axis=2
            )  # (E_n, N, K, 3)
            upred_safe = np.where(
                has_all[:, :, np.newaxis, np.newaxis], upred_all, pos_tiled
            )  # (E_n, N, K, 3)
            # nbr_preds_arr[e_idx, i] = predictions of the N-1 neighbours of i
            nbr_preds_arr = upred_safe[:, self._neighbor_idx]   # (E_n,N,N-1,K,3)
            valid_nbr_arr = has_all[:, self._neighbor_idx]       # (E_n, N, N-1)
            _t_nbr = _time.perf_counter() - _tb

            # Phase 2: replan condition via numpy — no dict loop
            _tc_chk = _time.perf_counter()
            needs_replan = (
                (~self._st_has[e_all, i_all]) |
                (self._st_steps[e_all, i_all] % self.n_substeps == 0)
            )  # (E_n*N,)
            replan_e_arr = e_all[needs_replan]
            replan_i_arr = i_all[needs_replan]
            _t_check = _time.perf_counter() - _tc_chk

            _tc1 = _time.perf_counter()

            if replan_e_arr.size > 0:
                self._plan_batched_gpu(
                    replan_e_arr, replan_i_arr,
                    nbr_preds_arr, valid_nbr_arr,
                    pos_np_local, vel_np, goal_np_local,
                )

            _tc2 = _time.perf_counter()

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
            if origins_np is not None:
                all_ori = np.repeat(origins_np[env_arr], N, axis=0).astype(np.float32)
            else:
                all_ori = np.zeros((E_n * N, 3), dtype=np.float32)

        else:
            # ── OSQP / CPU path: unchanged serial logic ───────────────────
            nbr_preds: dict[tuple[int, int], np.ndarray] = {}
            for e in env_list:
                for i in range(N):
                    st = self._state.get((e, i))
                    if st is not None:
                        st["last_replanned"] = False
                        st["last_reset_mode"] = False
                        st["last_fallback"] = False
                    nbr_preds[(e, i)] = self._gather_neighbour_predictions(e, i, pos_np_local)

            _tc1 = _time.perf_counter()

            replan_tasks = [
                (e, i) for e in env_list for i in range(N)
                if (self._state.get((e, i)) is None
                    or self._state[(e, i)]["steps"] % self.n_substeps == 0)
            ]

            def _do_replan(ei: tuple[int, int]) -> None:
                e, i = ei
                self._replan_agent(
                    e, i,
                    pos_local=pos_np_local[e, i],
                    vel=vel_np[e, i],
                    goal_local=goal_np_local[e, i],
                    nbr_pred=nbr_preds[(e, i)],
                )

            if self._pool is not None and len(replan_tasks) > 1:
                list(self._pool.map(_do_replan, replan_tasks))
            else:
                for task in replan_tasks:
                    _do_replan(task)

            _tc2 = _time.perf_counter()

            # Phase 3: serial U collection (OSQP path)
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

            all_U   = np.empty((E_n * N, self.n_bez), dtype=np.float32)
            all_sub = np.empty(E_n * N, dtype=np.int32)
            all_ori = np.empty((E_n * N, 3), dtype=np.float32)
            flat = 0
            for e in env_list:
                origin_e = (origins_np[e] if origins_np is not None
                            else np.zeros(3, dtype=np.float64))
                for i in range(N):
                    st = self._state[(e, i)]
                    all_U[flat]   = st["U"].astype(np.float32)
                    all_sub[flat] = st["steps"] % self.n_substeps
                    all_ori[flat] = origin_e
                    st["steps"] += 1
                    flat += 1

        # One batched matmul per unique substep index.
        _tp3 = _time.perf_counter()
        all_pos = np.empty((E_n * N, 3), dtype=np.float32)
        all_vel = np.empty((E_n * N, 3), dtype=np.float32)
        for k in range(self.n_substeps):
            mask = all_sub == k
            if not mask.any():
                continue
            # S_pos/vel: (1, n_bez) → squeeze for broadcast
            all_pos[mask] = (all_U[mask] @ self._S_pos_cache[k].T).reshape(-1, 3) + all_ori[mask]
            all_vel[mask] = (all_U[mask] @ self._S_vel_cache[k].T).reshape(-1, 3)
        _t_phase3 = _time.perf_counter() - _tp3

        # Write results back to ref tensors — two bulk CUDA transfers.
        _tw = _time.perf_counter()
        all_pos_t = torch.from_numpy(all_pos).to(device).reshape(E_n, N, 3)
        all_vel_t = torch.from_numpy(all_vel).to(device).reshape(E_n, N, 3)
        if env_ids is None:
            ref_pos[:] = all_pos_t
            ref_vel[:] = all_vel_t
        else:
            env_t = torch.tensor(env_list, dtype=torch.long, device=device)
            ref_pos[env_t] = all_pos_t
            ref_vel[env_t] = all_vel_t
        _t_write = _time.perf_counter() - _tw

        _tc3 = _time.perf_counter()

        # ── per-call timing log (every 10 calls) ──────────────────────────
        self._compute_calls = getattr(self, "_compute_calls", 0) + 1
        if self._compute_calls % 10 == 1:
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

    def plan(self, *args, **kwargs):
        """Alias for :py:meth:`compute`. The BC script + readers in the paper's
        notation prefer "plan" -- semantically identical."""
        return self.compute(*args, **kwargs)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Drop per-agent cached state for the given envs (all by default)."""
        if env_ids is None:
            self._state.clear()
            if self._gpu_solver is not None:
                self._gpu_solver.reset_warm_start()
                self._st_has[:]        = False
                self._st_steps[:]      = 0
                self._st_replanned[:]  = False
                self._st_reset_mode[:] = False
                self._st_fallback[:]   = False
                self._st_coll_ts[:]    = -1
            return
        ids_list = [int(i) for i in env_ids.detach().cpu().tolist()]
        ids = set(ids_list)
        self._state = {k: v for k, v in self._state.items() if k[0] not in ids}
        if self._gpu_solver is not None:
            ids_np = np.array(ids_list, dtype=np.intp)
            self._st_has[ids_np]        = False
            self._st_steps[ids_np]      = 0
            self._st_replanned[ids_np]  = False
            self._st_reset_mode[ids_np] = False
            self._st_fallback[ids_np]   = False
            self._st_coll_ts[ids_np]    = -1
            flat = [e * self.N + i for e in ids for i in range(self.N)]
            if flat:
                self._gpu_solver.reset_warm_start(
                    torch.tensor(flat, dtype=torch.long)
                )

    # ───────────────────────────────────────────────────────────────────
    # GPU ADMM batch solver
    # ───────────────────────────────────────────────────────────────────

    def enable_gpu_admm(
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
        self._gpu_solver = BatchedGPUDMPC(
            n_bez=self.n_bez,
            H_bez=self._H_bez_free,
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
        )

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

        # Per-agent neighbor index table: row i = indices of all j ≠ i  (N, N-1)
        self._neighbor_idx = np.array(
            [[j for j in range(N_) if j != i] for i in range(N_)],
            dtype=np.intp,
        )

        # Pre-cache float32 GPU tensors for Phase C — fixed matrices that never
        # change between calls, uploaded once here to avoid repeated CPU→GPU transfers.
        _dev = self._gpu_solver.device
        self._t_Fx    = torch.from_numpy(self._Fx_pos_pred.astype(np.float32)).to(_dev)
        self._t_Qd    = torch.from_numpy(self._Q_diag_free.astype(np.float32)).to(_dev)
        self._t_Gp    = torch.from_numpy(self._G_pos_pred.astype(np.float32)).to(_dev)
        self._t_Phi   = torch.from_numpy(
            self.M_pos_hor.astype(np.float32).reshape(K, 3, self.n_bez)
        ).to(_dev)  # (K, 3, n_bez)
        self._t_thinv = torch.from_numpy(
            np.diag(self._Theta_inv).astype(np.float32)
        ).to(_dev)  # (3,)

    def _plan_batched_gpu(
        self,
        e_arr: np.ndarray,              # (B,) env indices
        i_arr: np.ndarray,              # (B,) agent indices
        nbr_preds_arr: np.ndarray,      # (E, N, N-1, K, 3) — prebuilt by compute()
        valid_nbr_arr: np.ndarray,      # (E, N, N-1) bool
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
        import time as _time
        _t0 = _time.perf_counter()

        K      = self.p.k_hor
        n_bez  = self.n_bez
        N      = self.N
        gs     = self._gpu_solver
        n_eq   = gs.n_eq
        n_eq_c = gs.n_eq_cont
        rmin   = self.p.rmin
        g_thr  = self.p.g_factor * rmin
        qc     = self.p.quad_coll
        B      = len(e_arr)
        self._gpu_plan_calls = getattr(self, "_gpu_plan_calls", 0) + 1
        _do_log = (self._gpu_plan_calls % 10 == 1)

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

        n_seeds    = n_eq - n_eq_c
        seeds_flat = seeds_batch.reshape(B, n_seeds).astype(np.float32)

        _t_init = _time.perf_counter() - _t0

        # ── Phase B: nbr arrays already built — no loop ────────────────────
        _t1 = _time.perf_counter()
        own_batch = own_pred_batch                  # (B, K, 3)
        _t_assemble = _time.perf_counter() - _t1

        # ── Phase C: q computation on GPU ─────────────────────────────────
        _t2 = _time.perf_counter()
        dev = gs.device

        # Upload per-call inputs once
        x0_t    = torch.from_numpy(x0_batch.astype(np.float32)).to(dev)     # (B, 6)
        goal_t  = torch.from_numpy(goal_batch.astype(np.float32)).to(dev)   # (B, 3)
        own_t   = torch.from_numpy(own_batch.astype(np.float32)).to(dev)    # (B, K, 3)
        nbr_t   = torch.from_numpy(nbr_batch.astype(np.float32)).to(dev)    # (B, N-1, K, 3)
        valid_t = torch.from_numpy(valid_nbr).to(dev)                        # (B, N-1)

        # C1 — goal-tracking linear cost
        residual = x0_t @ self._t_Fx.T - goal_t.repeat(1, K)   # (B, 3K)
        q_t = 2.0 * (residual * self._t_Qd) @ self._t_Gp       # (B, n_bez)

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

            U_prev_c = torch.from_numpy(
                U_prev_b[coll_idx.cpu().numpy()].astype(np.float32)
            ).to(dev)                                           # (B_c, n_bez)
            pred  = torch.einsum('bnm,bm->bn', dg_dU, U_prev_c)  # (B_c, N-1)
            viol  = torch.where(active,
                                torch.clamp(rhs - pred, min=0.0),
                                torch.zeros_like(pred))

            pg = -2.0 * qc * torch.einsum('bn,bnm->bm', viol, dg_dU)  # (B_c, n_bez)
            q_t[coll_idx] += pg

        _t_q = _time.perf_counter() - _t2

        # ── Phase D: GPU solve ────────────────────────────────────────────
        _t3 = _time.perf_counter()
        b_eq  = np.zeros((B, n_eq), dtype=np.float32)
        b_eq[:, n_eq_c:] = seeds_flat
        idx_np = (e_arr * N + i_arr).astype(np.int64)

        # q_t already on GPU from Phase C
        beq_t = torch.from_numpy(b_eq).to(dev)
        idx_t = torch.from_numpy(idx_np).to(dev)
        U_raw = gs.solve(q_t, beq_t, idx_t).cpu().numpy()  # (B, n_bez) float32
        _t_gpu = _time.perf_counter() - _t3

        # ── Phase E: result extraction (batched matmuls + batch numpy write)
        _t4 = _time.perf_counter()
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
        kc_meta = np.where(has_coll_np, kc_arr_np + 1, -1)   # 1-indexed coll ts

        # Batch write all fields to numpy arrays — zero per-agent loop
        self._st_U[e_arr, i_arr]          = U_f64
        self._st_u_pred[e_arr, i_arr]     = u_pred_b
        self._st_seeds[e_arr, i_arr]      = seeds_new
        self._st_steps[e_arr, i_arr]      = 0
        self._st_has[e_arr, i_arr]        = True
        self._st_replanned[e_arr, i_arr]  = True
        self._st_reset_mode[e_arr, i_arr] = do_reset
        self._st_fallback[e_arr, i_arr]   = False
        self._st_coll_ts[e_arr, i_arr]    = kc_meta
        _t_extract = _time.perf_counter() - _t4

        if _do_log:
            n_coll = int(has_coll_np.sum())
            total  = (_t_init + _t_assemble + _t_q + _t_gpu + _t_extract) * 1e3
            print(
                f"[DMPC gpu_admm] call={self._gpu_plan_calls}  B={B}  coll={n_coll}"
                f"  init={_t_init*1e3:.1f}ms"
                f"  assemble={_t_assemble*1e3:.1f}ms"
                f"  q+coll={_t_q*1e3:.1f}ms"
                f"  gpu={_t_gpu*1e3:.1f}ms"
                f"  extract={_t_extract*1e3:.1f}ms"
                f"  total={total:.1f}ms",
                flush=True,
            )


    def get_vref_ctrl_pts(
        self,
        env_list: "Iterable[int]",
        K: int = 4,
    ) -> torch.Tensor:
        """Fit K-point Bezier control points to the velocity profile of each agent.

        Must be called right after :py:meth:`compute`/plan().  For each (e, i),
        evaluates v_ref at ``n_substeps`` time points within the current h-window
        and fits a degree-(K-1) Bezier via least-squares.

        Args:
            env_list: iterable of env indices to process.
            K: number of control points (degree = K-1). Default 4 (cubic).

        Returns:
            Float32 tensor of shape ``(E_max, N, K, 3)`` on ``self.device``,
            where ``E_max = max(env_list) + 1``.  Agents with no DMPC state
            (just reset) are left as zero.
        """
        n_sub = self.n_substeps
        if K not in self._vref_ctrl_cache:
            taus = np.arange(1, n_sub + 1, dtype=np.float64) / n_sub  # (n_sub,)
            Phi = np.column_stack([
                comb(K - 1, j, exact=True) * (taus ** j) * ((1 - taus) ** (K - 1 - j))
                for j in range(K)
            ]).astype(np.float32)  # (n_sub, K)
            self._vref_ctrl_cache[K] = np.linalg.pinv(Phi).astype(np.float32)  # (K, n_sub)
        Phi_pinv = self._vref_ctrl_cache[K]  # (K, n_sub)

        env_list = list(env_list)
        E_max = max(env_list) + 1
        ctrl_pts = np.zeros((E_max, self.N, K, 3), dtype=np.float32)

        # M_vel_sub: (3*n_sub, n_bez); reshape output to (n_sub, 3).
        for e in env_list:
            for i in range(self.N):
                st = self._state.get((e, i))
                if st is None:
                    continue
                v_refs = (self.M_vel_sub @ st["U"]).reshape(n_sub, 3).astype(np.float32)  # (n_sub, 3)
                ctrl_pts[e, i] = Phi_pinv @ v_refs  # (K, 3)

        return torch.from_numpy(ctrl_pts).to(self.device)

    def get_pos_ctrl_pts(
        self,
        env_list: "Iterable[int]",
        pos_w: "torch.Tensor",       # (E, N, 3) world frame
        vel_w: "torch.Tensor",       # (E, N, 3) world frame
        goal_w: "torch.Tensor",      # (E, N, 3) world frame
        env_origins: "torch.Tensor", # (E, 3)
        K: int = 6,
        h: float = 0.1,
        n_fit: int = 20,
    ) -> "torch.Tensor":
        """Fit K-point position Bezier free ctrl_pts to DMPC plan (goal-aligned delta frame).

        The first two ctrl_pts are pinned for C0/C1 continuity:
          P_0 = (0,0,0)  — current position = origin in delta frame
          P_1 = R @ vel_current * h / (K-1)  — C1 velocity continuity

        The remaining K-2 free ctrl_pts are returned.

        Args:
            env_list:    Env indices to process.
            pos_w:       Current drone positions ``(E, N, 3)`` world frame.
            vel_w:       Current drone velocities ``(E, N, 3)`` world frame.
            goal_w:      Goal positions ``(E, N, 3)`` world frame.
            env_origins: Per-env origin offsets ``(E, 3)``.
            K:           Total number of position ctrl_pts (default 6).
            h:           Bezier window duration in seconds (default 0.1).
            n_fit:       Sample count for the least-squares fit (default 20).

        Returns:
            Float32 tensor ``(E_max, N, K-2, 3)`` on ``self.device``.
        """
        env_list = list(env_list)
        E_max = max(env_list) + 1
        K_free = K - 2
        free_ctrl = np.zeros((E_max, self.N, K_free, 3), dtype=np.float32)

        pos_np   = pos_w.detach().cpu().numpy().astype(np.float64)
        vel_np   = vel_w.detach().cpu().numpy().astype(np.float64)
        goal_np  = goal_w.detach().cpu().numpy().astype(np.float64)
        orig_np  = env_origins.detach().cpu().numpy().astype(np.float64)

        # Sampling taus in (0, 1] for the K-1 degree Bezier.
        taus = np.linspace(1.0 / n_fit, 1.0, n_fit, dtype=np.float64)  # (n_fit,)
        t_abs = taus * h  # absolute times in (0, h] for DMPC sampling

        # Bernstein basis matrix: Phi[m, j] = B_{j, K-1}(taus[m])
        Phi_all = np.column_stack([
            comb(K - 1, j, exact=True) * (taus ** j) * ((1 - taus) ** (K - 1 - j))
            for j in range(K)
        ])  # (n_fit, K)
        Phi_free   = Phi_all[:, 2:].astype(np.float32)    # (n_fit, K-2) — free ctrl_pts
        Phi_pinned = Phi_all[:, :2].astype(np.float32)    # (n_fit, 2)   — P_0, P_1

        for e in env_list:
            origin_e = orig_np[e]
            for i in range(self.N):
                if self._gpu_solver is not None:
                    if not self._st_has[e, i]:
                        continue
                    U = self._st_U[e, i]
                else:
                    st = self._state.get((e, i))
                    if st is None:
                        continue
                    U = st["U"]

                # Sample DMPC position Bezier at n_fit time points.
                pos_dmpc = (
                    self.bezier.sample_matrix(t_abs, deriv=0) @ U
                ).reshape(n_fit, 3) + origin_e  # (n_fit, 3) world frame

                # Current drone state.
                p_curr = pos_np[e, i]    # (3,)
                v_curr = vel_np[e, i]    # (3,)
                g_curr = goal_np[e, i]   # (3,)

                # Goal-aligned rotation R: world → goal frame.
                delta = g_curr - p_curr
                horiz = np.hypot(delta[0], delta[1])
                if horiz < 1e-3:
                    R = np.eye(3, dtype=np.float64)
                else:
                    cos_t, sin_t = delta[0] / horiz, delta[1] / horiz
                    R = np.array([[cos_t, sin_t, 0.0],
                                  [-sin_t, cos_t, 0.0],
                                  [0.0, 0.0, 1.0]], dtype=np.float64)

                # Pinned ctrl_pts in goal-aligned delta frame.
                # P_0 = (0,0,0); P_1 = R @ v_curr * h / (K-1).
                P_1 = R @ v_curr * (h / (K - 1))  # (3,)

                # DMPC positions → goal-aligned delta frame.
                delta_w = pos_dmpc - p_curr  # (n_fit, 3)
                p_goal  = (R @ delta_w.T).T  # (n_fit, 3)

                # Residual after subtracting P_0 (= 0) and P_1 contributions.
                residual = p_goal - np.outer(Phi_pinned[:, 1], P_1)  # (n_fit, 3)

                # Least-squares: Phi_free @ P_free = residual  → P_free (K-2, 3).
                P_free, _, _, _ = np.linalg.lstsq(
                    Phi_free.astype(np.float64), residual, rcond=None
                )
                free_ctrl[e, i] = P_free.astype(np.float32)

        return torch.from_numpy(free_ctrl).to(self.device)

    # ───────────────────────────────────────────────────────────────────
    # Neighbour broadcast retrieval
    # ───────────────────────────────────────────────────────────────────
    def _gather_neighbour_predictions(
        self, env_idx: int, agent_idx: int, pos_local: np.ndarray
    ) -> np.ndarray:
        """Return broadcast input predictions ``(N-1, K, 3)`` for the
        neighbours of ``agent_idx`` in env ``env_idx``. Agents without a
        cached plan yet are treated as stationary at their current position."""
        K = self.p.k_hor
        preds: list[np.ndarray] = []
        for j in range(self.N):
            if j == agent_idx:
                continue
            sj = self._state.get((env_idx, j))
            if sj is None:
                preds.append(np.tile(pos_local[env_idx, j], (K, 1)))
            else:
                preds.append(sj["u_pred"].copy())
        return np.stack(preds, axis=0) if preds else np.zeros((0, K, 3))

    # ───────────────────────────────────────────────────────────────────
    # Per-agent replanning: build + solve one QP.
    # ───────────────────────────────────────────────────────────────────
    def _replan_agent(
        self,
        env_idx: int,
        agent_idx: int,
        pos_local: np.ndarray,
        vel: np.ndarray,
        goal_local: np.ndarray,
        nbr_pred: np.ndarray,
    ) -> None:
        K = self.p.k_hor
        n_bez = self.n_bez
        n_nbr = nbr_pred.shape[0]

        prev = self._state.get((env_idx, agent_idx))
        seeds, _reset_mode = self._event_triggered_init(
            prev=prev, pos=pos_local, vel=vel,
        )

        # ── Collision constraints first: their presence selects the cost mode.
        own_pred_seed = self._seed_input_prediction(prev, seeds[0], goal_local)
        coll_A, coll_l, coll_u, collision_timestep, n_slack = self._build_collision_constraints(
            own_pred=own_pred_seed,
            nbr_pred=nbr_pred,
        )
        collision_mode = coll_A is not None

        # Pick free / obs cost mode (mirrors C++ generator.cpp::buildQP).
        if collision_mode:
            H_bez_const = self._H_bez_obs
            Q_diag = self._Q_diag_obs
        else:
            H_bez_const = self._H_bez_free
            Q_diag = self._Q_diag_free
            # n_slack = 0 #!!!!

        # Variable layout: [U_bez (n_bez), slacks (n_slack)]
        n_total = n_bez + n_slack

        # ── Cost
        H = np.zeros((n_total, n_total))
        H[:n_bez, :n_bez] = H_bez_const
        if n_slack > 0:
            H[n_bez:, n_bez:] = 2.0 * self.p.quad_coll * np.eye(n_slack)

        # Linear term on U_bez:  2 G^T Q (Fx x0 - pd_stack)
        x0 = np.concatenate([pos_local, vel])
        pd_stack = np.tile(goal_local, K)
        residual = self._Fx_pos_pred @ x0 - pd_stack
        q_lin_bez = 2.0 * (self._G_pos_pred.T @ (Q_diag * residual))
        q = np.zeros(n_total)
        q[:n_bez] = q_lin_bez
        if n_slack > 0:
            q[n_bez:] = self.p.lin_coll

        # ── Equality constraints
        A_eq_rows = []
        b_eq_rows = []
        if self.n_continuity > 0:
            A_eq_rows.append(np.hstack([self.A_continuity, np.zeros((self.n_continuity, n_slack))]))
            b_eq_rows.append(np.zeros(self.n_continuity))
        # Pin the first deg_poly+1 derivatives at t = 0 to the seeds carried
        # over from the previous solve (or zeros above order 0 in disturbed
        # mode, per Eq. 17). This enforces C^{deg_poly} continuity of the
        # reference signal across replanning rounds, matching C++ generator
        # buildQP. Without this only position is pinned and the QP is free to
        # introduce velocity/acceleration jumps.


        for r in range(self.p.deg_poly + 1):
            A_eq_rows.append(np.hstack([self._M_deriv_0[r], np.zeros((3, n_slack))]))
            b_eq_rows.append(seeds[r].copy())

        # ── Inequality constraints
        A_ineq_rows = []
        l_ineq = []
        u_ineq = []
        # Position-reference workspace box: pmin <= M_pos U <= pmax
        A_ineq_rows.append(np.hstack([self.M_pos_hor, np.zeros((3 * K, n_slack))]))
        l_ineq.append(self._pmin.copy())
        u_ineq.append(self._pmax.copy())
        # Acceleration-of-reference box: amin <= M_acc U <= amax
        A_ineq_rows.append(np.hstack([self.M_acc_hor, np.zeros((3 * K, n_slack))]))
        l_ineq.append(self._amin.copy())
        u_ineq.append(self._amax.copy())
        if n_slack > 0:
            # eps <= 0   (and -inf < eps)
            A_ineq_rows.append(np.hstack([np.zeros((n_slack, n_bez)), np.eye(n_slack)]))
            l_ineq.append(np.full(n_slack, -np.inf))
            u_ineq.append(np.zeros(n_slack))

        if coll_A is not None:
            A_ineq_rows.append(coll_A)
            l_ineq.append(coll_l)
            u_ineq.append(coll_u)

        # Assemble and solve
        A_eq = np.vstack(A_eq_rows)
        b_eq = np.concatenate(b_eq_rows)
        A_in = np.vstack(A_ineq_rows)
        l_in = np.concatenate(l_ineq)
        u_in = np.concatenate(u_ineq)

        A_full = sp.vstack([sp.csc_matrix(A_eq), sp.csc_matrix(A_in)], format="csc")
        l_full = np.concatenate([b_eq, l_in])
        u_full = np.concatenate([b_eq, u_in])
        P = sp.csc_matrix(H)

        solver = osqp.OSQP()
        try:
            solver.setup(
                P=P, q=q, A=A_full, l=l_full, u=u_full,
                verbose=False,
                eps_abs=self.p.osqp_eps_abs,
                eps_rel=self.p.osqp_eps_rel,
                max_iter=self.p.osqp_max_iter,
                polish=self.p.osqp_polish,
            )
        except ValueError:
            self._fallback_state(env_idx, agent_idx, seeds[0], goal_local)
            self._state[(env_idx, agent_idx)]["last_replanned"] = True
            self._state[(env_idx, agent_idx)]["last_reset_mode"] = bool(_reset_mode)
            self._state[(env_idx, agent_idx)]["last_fallback"] = True
            self._state[(env_idx, agent_idx)]["collision_timestep"] = int(collision_timestep)
            self._state[(env_idx, agent_idx)]["collision_sample_indices"] = [max(int(collision_timestep) - 1, 0)] if collision_timestep >= 0 else []
            return

        if prev is not None and prev["U"].shape[0] == n_bez:
            warm = np.concatenate([prev["U"], np.zeros(n_slack)])
            solver.warm_start(warm)
        res = solver.solve()
        if res.info.status_val not in (1, 2):
            # print(f"QP failed for env {env_idx} agent {agent_idx} with status {res.info.status}")
            self._fallback_state(env_idx, agent_idx, seeds[0], goal_local)
            self._state[(env_idx, agent_idx)]["last_replanned"] = True
            self._state[(env_idx, agent_idx)]["last_reset_mode"] = bool(_reset_mode)
            self._state[(env_idx, agent_idx)]["last_fallback"] = True
            self._state[(env_idx, agent_idx)]["collision_timestep"] = int(collision_timestep)
            self._state[(env_idx, agent_idx)]["collision_sample_indices"] = [max(int(collision_timestep) - 1, 0)] if collision_timestep >= 0 else []
            return

        U_sol = res.x[:n_bez]
        u_pred = (self.M_pos_hor @ U_sol).reshape(K, 3)
        new_seeds = np.stack(
            [(self._M_deriv_h[r] @ U_sol).reshape(3)
             for r in range(self.p.deg_poly + 1)]
        )  # (deg1, 3) ndarray
        self._state[(env_idx, agent_idx)] = {
            "U": U_sol,
            "u_pred": u_pred,
            "seeds": new_seeds,
            "steps": 0,
            "last_replanned": True,
            "last_reset_mode": bool(_reset_mode),
            "last_fallback": False,
            "collision_timestep": int(collision_timestep),
            "collision_sample_indices": [max(int(collision_timestep) - 1, 0)] if collision_timestep >= 0 else [],
        }

    # ───────────────────────────────────────────────────────────────────
    # Event-triggered initial condition (Section IV)
    # ───────────────────────────────────────────────────────────────────
    def _event_triggered_init(
        self,
        prev: dict | None,
        pos: np.ndarray,
        vel: np.ndarray,
    ) -> tuple[list[np.ndarray], bool]:
        """Returns ``(seeds, reset_mode)`` per Eq. 15-17.

        ``seeds`` is a list of length ``deg_poly + 1`` containing the desired
        value of the new Bezier's derivatives at ``t = 0``:

        * Normal branch (Eq. 17, top): seeds come from the previous solve
          evaluated at ``t = h``, so the reference is C^{deg_poly}-continuous
          across the replanning instant.
        * Disturbed branch (Eq. 17, bottom): position seeded to the current
          measured position, higher-order derivatives zeroed -- the reference
          is intentionally reset to the robot's state.
        """
        zero_higher = [np.zeros(3) for _ in range(self.p.deg_poly)]
        if prev is None:
            # print("No previous plan, using disturbed initialisation.")
            return [pos.copy(), *zero_higher], True

        u_prev_at_kt = prev["seeds"][0]
        err = pos - u_prev_at_kt
        # Eq. 15: f_n[k_t] = (p - u)^5 / (-(v + sgn(v) eps))
        denom = -(vel + np.sign(vel + 1e-12) * self.p.eps_velocity)
        denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6 + 1e-6, denom)
        f_vec = (err ** 5) / denom

        in_band = bool(np.logical_and(f_vec > self.p.f_min, f_vec < self.p.f_max).all())

        # return [s.copy() for s in prev["seeds"]], False
    

        if in_band:
            # print("In-band, using normal initialisation.")
            return [s.copy() for s in prev["seeds"]], False
        return [pos.copy(), *zero_higher], True

    def _seed_input_prediction(
        self,
        prev: dict | None,
        u_init: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        """Provide a coarse predicted-input trajectory used as the
        linearisation seed for on-demand collision avoidance. Uses the prior
        plan when available, otherwise a simple PD-toward-goal rollout."""
        K = self.p.k_hor
        if prev is not None:
            return prev["u_pred"].copy()
        traj = np.zeros((K, 3))
        p = u_init.copy()
        v = np.zeros(3)
        for k in range(K):
            a = np.clip(2.0 * (goal - p) - 1.5 * v, np.asarray(self.p.amin), np.asarray(self.p.amax))
            p = p + v * self.p.h + 0.5 * a * self.p.h ** 2
            v = v + a * self.p.h
            traj[k] = p
        return traj

    # ───────────────────────────────────────────────────────────────────
    # On-demand input-space collision constraints (Section III.E)
    # ───────────────────────────────────────────────────────────────────
    def _build_collision_constraints(
        self,
        own_pred: np.ndarray,
        nbr_pred: np.ndarray,
        # n_slack: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | int | int]:
        """Construct collision-avoidance inequalities. For each neighbour ``j``
        find the first step ``k_c`` where the predicted (input-space) distance
        is below ``g(rmin)`` and add one linearised cut. The slack variable
        for neighbour ``j`` is ``eps_{ij}`` (one per neighbour); if no
        collision is predicted, no row is added and the slack stays at zero."""
        K = self.p.k_hor
        rmin = self.p.rmin
        g_thresh = self.p.g_factor * rmin
        Theta_inv = self._Theta_inv
        n_nbr = nbr_pred.shape[0]
        kc = -1
        n_slack = 0
        if n_nbr == 0:
            return None, None, None, kc, n_slack

        rows = []
        lbs = []
        ubs = []
        # collision_sample_indices: list[int] = []
        
        # Collision detection
        # 1. Determine k_c,i: the first timestep where there exists a neighbor j s.t. d < rmin
        # 2. Determine Omega_i: the set of neighbors j s.t. d < g_thresh at k_c,i
        
        # 1. Determine k_c,i
        for k in range(1, K):
            for j in range(n_nbr):
                d = Theta_inv @ (own_pred[k - 1] - nbr_pred[j, k])
                if np.linalg.norm(d) < rmin:
                    kc = k
                    break
            if kc > 0:
                break
        
        # 2. Determine Omega_i and construct constraint
        if kc >= 0:
            for j in range(n_nbr):
                d = Theta_inv @ (own_pred[kc - 1] - nbr_pred[j, kc])
                if np.linalg.norm(d) < g_thresh:
                    # construct collision constraint
                    Phi_kc = self.M_pos_hor[3 * (kc - 1) : 3 * (kc - 1) + 3, :]
                    unit_diff = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else np.array([1.0, 0.0, 0.0])
                    dg_dU = unit_diff.transpose() @ Theta_inv @ Phi_kc
                    rhs_lower = unit_diff.transpose() @ Theta_inv @ nbr_pred[j, kc] + rmin

                    row = np.zeros(self.n_bez) # + n_slack)
                    row = dg_dU
                    # row[: self.n_bez] = dg_dU
                    # row[self.n_bez + j] = -1
                    
                    rows.append(row)
                    lbs.append(rhs_lower)
                    ubs.append(np.inf)

                    n_slack += 1

            A_col = np.hstack([np.vstack(rows), -np.eye(n_slack)])
            return A_col, np.asarray(lbs), np.asarray(ubs), kc, n_slack

        else:
            return None, None, None, kc, n_slack
    def _collision_check_batch(
        self,
        own_pred_list: list[np.ndarray],
        nbr_pred_list: list[np.ndarray],
    ) -> list[tuple]:
        """Vectorised collision check for a batch of agents.

        Replaces calling :meth:`_build_collision_constraints` B times in a
        Python loop.  All distance computations are done in one numpy broadcast
        over shape ``(B, K-1, N_max, 3)``.

        Returns a list of ``(A_coll_U, b_coll, kc, n_slack)`` per agent, where
        ``A_coll_U`` is ``(n_slack, n_bez)`` float32 or ``None``.
        """
        B = len(own_pred_list)
        K = self.p.k_hor
        rmin = self.p.rmin
        g_thresh = self.p.g_factor * rmin
        theta_inv_diag = np.diag(self._Theta_inv)       # (3,)

        N_max = max((p.shape[0] for p in nbr_pred_list), default=0)
        if N_max == 0:
            return [(None, None, -1, 0)] * B

        # Pad neighbor predictions to uniform shape (B, N_max, K, 3).
        own_batch = np.stack(own_pred_list, axis=0)      # (B, K, 3)
        nbr_batch = np.zeros((B, N_max, K, 3), dtype=np.float64)
        valid_nbr = np.zeros((B, N_max), dtype=bool)
        for b, nbr in enumerate(nbr_pred_list):
            n = nbr.shape[0]
            if n > 0:
                nbr_batch[b, :n] = nbr
                valid_nbr[b, :n] = True

        # Vectorised distance: d[b,k,j] = Theta_inv*(own[b,k-1] - nbr[b,j,k])
        # own[:,:-1,:] : (B, K-1, 3)   →  expand to (B, K-1, 1, 3)
        # nbr[:,:,1:,:]: (B,N_max,K-1,3) → transpose to (B, K-1, N_max, 3)
        diff = (own_batch[:, :-1, np.newaxis, :]
                - nbr_batch[:, :, 1:, :].transpose(0, 2, 1, 3))  # (B,K-1,N_max,3)
        d = diff * theta_inv_diag                                  # broadcast last dim
        norms = np.sqrt((d * d).sum(axis=-1))                      # (B, K-1, N_max)
        # Invalid neighbors never trigger collision.
        norms = np.where(valid_nbr[:, np.newaxis, :], norms, np.inf)

        # First collision step per agent.
        coll_at_step = norms < rmin                      # (B, K-1, N_max)
        coll_any_nbr = coll_at_step.any(axis=-1)         # (B, K-1)
        has_coll     = coll_any_nbr.any(axis=-1)         # (B,)
        # argmax finds the FIRST True along axis; -1 when no collision.
        kc_arr = np.where(has_coll, coll_any_nbr.argmax(axis=-1), -1)  # (B,)

        results: list[tuple] = []
        for b in range(B):
            if not has_coll[b]:
                results.append((None, None, -1, 0))
                continue

            kc = int(kc_arr[b]) + 1           # convert 0-indexed → 1-indexed kc
            d_kc    = d[b, kc - 1]            # (N_max, 3)
            norms_kc = norms[b, kc - 1]       # (N_max,)

            active = np.where(valid_nbr[b] & (norms_kc < g_thresh))[0]
            if len(active) == 0:
                results.append((None, None, kc, 0))
                continue

            Phi_kc = self.M_pos_hor[3 * (kc - 1):3 * (kc - 1) + 3, :]  # (3, n_bez)
            rows, lbs = [], []
            for j in active:
                dj   = d_kc[j]
                nj   = norms_kc[j]
                udir = dj / nj if nj > 1e-6 else np.array([1., 0., 0.])
                dg_dU    = udir @ self._Theta_inv @ Phi_kc           # (n_bez,)
                rhs_lower = udir @ self._Theta_inv @ nbr_batch[b, j, kc] + rmin
                rows.append(dg_dU.astype(np.float32))
                lbs.append(float(rhs_lower))

            n_slack  = len(rows)
            A_coll_U = np.vstack(rows)                    # (n_slack, n_bez)
            b_coll   = np.array(lbs, dtype=np.float32)
            results.append((A_coll_U, b_coll, kc, n_slack))

        return results

    # ───────────────────────────────────────────────────────────────────
    # Fallback (solver failure)
    # ───────────────────────────────────────────────────────────────────
    def _fallback_state(
        self,
        env_idx: int,
        agent_idx: int,
        pos_seed: np.ndarray,
        goal: np.ndarray,
    ) -> None:
        """If the QP fails, freeze the agent's plan to a PD-toward-goal seed
        fitted onto the Bezier basis so subsequent samples remain
        well-defined."""
        K = self.p.k_hor
        h = self.p.h
        p = pos_seed.copy()
        v = np.zeros(3)
        u_pred = np.zeros((K, 3))
        for k in range(K):
            a = np.clip(2.0 * (goal - p) - 1.5 * v, np.asarray(self.p.amin), np.asarray(self.p.amax))
            p = p + v * h + 0.5 * a * h ** 2
            v = v + a * h
            u_pred[k] = p
        # Least-squares fit Bezier control points to the K samples.
        try:
            U = np.linalg.lstsq(self.M_pos_hor, u_pred.reshape(-1), rcond=None)[0]
        except np.linalg.LinAlgError:
            U = np.zeros(self.n_bez)
        seeds = np.stack(
            [(self._M_deriv_h[r] @ U).reshape(3)
             for r in range(self.p.deg_poly + 1)]
        )  # (deg1, 3) ndarray
        self._state[(env_idx, agent_idx)] = {
            "U": U,
            "u_pred": u_pred,
            "seeds": seeds,
            "steps": 0,
            "collision_timestep": -1,
            "collision_sample_indices": [],
        }
