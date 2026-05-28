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

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import torch
from scipy.special import comb, factorial

from typing import Dict, List

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
    lin_coll: float = -1.0e2 #-1.0e5
    quad_coll: float = 1.0 #1.0

    # Collision parameters
    rmin: float = 0.5 #0.3
    height_scaling: float = 2.0
    g_factor: float = 2.0
    # height_scaling_obs: float = 4.0

    # Event-triggered replanning
    f_min: float = -0.01
    f_max: float = 0.8
    eps_velocity: float = 0.01

    # OSQP options
    osqp_eps_abs: float = 1e-2 # 1e-3
    osqp_eps_rel: float = 1e-2 # 1e-3
    osqp_max_iter: int = 400 #200
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
# DMPC expert
# ───────────────────────────────────────────────────────────────────────────
class DMPCExpert:
    """Distributed MPC expert producing position-reference trajectories for
    multiple agents, plus a method to evaluate the trajectory at the env
    control rate via :py:meth:`compute`."""

    def __init__(self, num_drones: int, num_envs: int, params: DMPCParams | None = None, device="cpu"):
        self.N = num_drones
        self.E = num_envs
        self.p = params or DMPCParams()
        if self.p.t_segment is None:
            self.p.t_segment = (self.p.k_hor - 1) * self.p.h / self.p.num_segments
        self.device = torch.device(device)

        # Number of control steps between replanning rounds (Ts samples per h).
        self.n_substeps = max(1, int(round(self.p.h / self.p.ts)))

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
    
    def _precompute_static_obstacle_coeffs(self, obstacle_info: Dict | None):
        # Precompute coefficients (e.g., covering ellipsoid) as obstacle support grows.
        self._obstacle_info = obstacle_info
        self._num_static_obstacles = 0
        self._obstacle_pos_w = None
        self._obstacle_pos_local = None

        if obstacle_info is None:
            return

        obstacle_pos_w = obstacle_info.get("ellipsoid_pos_w", obstacle_info.get("pos_w"))
        if obstacle_pos_w is None:
            return

        self._num_static_obstacles = int(obstacle_pos_w.shape[1])
        self._obstacle_pos_w = obstacle_pos_w.detach().cpu().numpy()

        params = obstacle_info.get("params", {})
        ellipsoid_axes = params.get("ellipsoid_axes")
        if ellipsoid_axes is not None:
            ellipsoid_axes = np.asarray(ellipsoid_axes, dtype=np.float64)
            if ellipsoid_axes.ndim == 1:
                ellipsoid_axes = np.tile(ellipsoid_axes[None, :], (self._num_static_obstacles, 1))
            if ellipsoid_axes.shape[0] != self._num_static_obstacles:
                repeats = int(np.ceil(self._num_static_obstacles / max(ellipsoid_axes.shape[0], 1)))
                ellipsoid_axes = np.tile(ellipsoid_axes, (repeats, 1))[: self._num_static_obstacles]
            theta_inv = np.zeros((self.E, self._num_static_obstacles, 3, 3), dtype=np.float64)
            for j in range(self._num_static_obstacles):
                theta_inv[:, j, :, :] = np.diag(1.0 / np.maximum(ellipsoid_axes[j], 1e-6))
            self._obs_theta_inv = theta_inv
            self._obs_rmin = np.ones((self.E, self._num_static_obstacles), dtype=np.float64)
            self._obs_g_thres = self.p.g_factor * self._obs_rmin
            return

        shape_name = obstacle_info.get("shape_name")
        if shape_name == "SphereCfg":
            # TODO: set heterogeneous obstacles & parse info.
            self._obs_theta_inv = np.asarray(
                [[np.eye(3) for _ in range(self._num_static_obstacles)] for _ in range(self.E)]
            )
            self._obs_rmin = np.asarray(
                [[params.get("radius") + 0.25 for _ in range(self._num_static_obstacles)] for _ in range(self.E)]
            )
            self._obs_g_thres = self.p.g_factor * self._obs_rmin
        elif shape_name == "CuboidCfg":
            size = np.asarray(params.get("size"), dtype=np.float64)
            # Fallback: cover each cuboid by one ellipsoid. Prefer the split
            # ellipsoid set above for narrow passages.
            margin = float(params.get("ellipsoid_margin", 0.25))
            semi_axes = np.sqrt(3.0) * 0.5 * size + margin
            theta_inv = np.diag(1.0 / np.maximum(semi_axes, 1e-6))
            self._obs_theta_inv = np.tile(
                theta_inv[None, None, :, :],
                (self.E, self._num_static_obstacles, 1, 1),
            )
            self._obs_rmin = np.ones((self.E, self._num_static_obstacles), dtype=np.float64)
            self._obs_g_thres = self.p.g_factor * self._obs_rmin
        else:
            # TODO: support other shapes
            raise NotImplementedError(f"Unsupported obstacle type: {shape_name}")

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
        E, N = pos_w.shape[0], pos_w.shape[1]
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
        for e in env_iter:
            for i in range(N):
                st = self._state.get((e, i))
                if st is not None:
                    st["last_replanned"] = False
                    st["last_reset_mode"] = False
                    st["last_fallback"] = False

            # First sweep: replan for any agent whose subsample counter rolled
            # over. We do this in two passes (replan-first, sample-second) so
            # every agent in this env has up-to-date broadcasts when neighbours
            # query ``u_pred``.

            # gather prediction first
            nbr_preds = [self._gather_neighbour_predictions(e, i, pos_np_local) for i in range(N)]
            
            # replan with up-to-date neighbour predictions
            for i in range(N):
                key = (e, i)
                st = self._state.get(key)
                if st is None or st["steps"] % self.n_substeps == 0:
                    nbr_pred = nbr_preds[i]
                    self._replan_agent(
                        e, i,
                        pos_local=pos_np_local[e, i],
                        vel=vel_np[e, i],
                        goal_local=goal_np_local[e, i],
                        nbr_pred=nbr_pred,
                    )

            for i in range(N):
                key = (e, i)
                st = self._state[key]
                t_sub = ((st["steps"] % self.n_substeps) + 1) * ts
                t_sub = min(t_sub, h_total)
                ref_pos_local = (
                    self.bezier.sample_matrix(np.array([t_sub]), deriv=0) @ st["U"]
                )
                ref_vel_local = (
                    self.bezier.sample_matrix(np.array([t_sub]), deriv=1) @ st["U"]
                )

                origin_e = origins_np[e] if origins_np is not None else np.zeros(3)
                ref_pos[e, i] = torch.from_numpy(
                    (ref_pos_local + origin_e).astype(np.float32)
                ).to(device)
                ref_vel[e, i] = torch.from_numpy(
                    ref_vel_local.astype(np.float32)
                ).to(device)
                st["steps"] += 1
        return ref_pos, ref_vel

    def plan(self, *args, **kwargs):
        """Alias for :py:meth:`compute`. The BC script + readers in the paper's
        notation prefer "plan" -- semantically identical."""
        return self.compute(*args, **kwargs)

    def reset(self, obstacle_info: Dict | None = None, env_ids: torch.Tensor | None = None) -> None:
        """Drop per-agent cached state for the given envs (all by default)."""
        self._precompute_static_obstacle_coeffs(obstacle_info)
        if env_ids is None:
            self._state.clear()
            return
        ids = set(int(i) for i in env_ids.detach().cpu().tolist())
        self._state = {k: v for k, v in self._state.items() if k[0] not in ids}

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
        coll_A, coll_l, coll_u, inter_agent_col_t, static_collision_timesteps, n_slack = self._build_collision_constraints(
            env_idx=env_idx,
            own_pred=own_pred_seed,
            nbr_pred=nbr_pred,
        )
        collision_sample_indices = [max(int(inter_agent_col_t) - 1, 0)] if inter_agent_col_t >= 0 else []
        for kc_o in static_collision_timesteps:
            for kc in kc_o:
                collision_sample_indices.append(max(int(kc) - 1, 0))
        collision_sample_indices = sorted(set(collision_sample_indices))
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
            self._state[(env_idx, agent_idx)]["collision_timestep"] = int(inter_agent_col_t)
            self._state[(env_idx, agent_idx)]["collision_sample_indices"] = collision_sample_indices
            return

        if prev is not None and prev["U"].shape[0] == n_bez:
            warm = np.concatenate([prev["U"], np.zeros(n_slack)])
            solver.warm_start(warm)
        res = solver.solve()
        if res.info.status_val not in (1, 2):
            print(f"QP failed for env {env_idx} agent {agent_idx} with status {res.info.status}")
            self._fallback_state(env_idx, agent_idx, seeds[0], goal_local)
            self._state[(env_idx, agent_idx)]["last_replanned"] = True
            self._state[(env_idx, agent_idx)]["last_reset_mode"] = bool(_reset_mode)
            self._state[(env_idx, agent_idx)]["last_fallback"] = True
            self._state[(env_idx, agent_idx)]["collision_timestep"] = int(inter_agent_col_t)
            self._state[(env_idx, agent_idx)]["collision_sample_indices"] = collision_sample_indices
            return

        U_sol = res.x[:n_bez]
        u_pred = (self.M_pos_hor @ U_sol).reshape(K, 3)
        new_seeds = [
            (self._M_deriv_h[r] @ U_sol).reshape(3)
            for r in range(self.p.deg_poly + 1)
        ]
        self._state[(env_idx, agent_idx)] = {
            "U": U_sol,
            "u_pred": u_pred,
            "seeds": new_seeds,
            "steps": 0,
            "last_replanned": True,
            "last_reset_mode": bool(_reset_mode),
            "last_fallback": False,
            "collision_timestep": int(inter_agent_col_t),
            "collision_sample_indices": collision_sample_indices,
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
        env_idx: int,
        own_pred: np.ndarray,
        nbr_pred: np.ndarray,
        # n_slack: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | int | List[List[int]] | int]:
        """Construct collision-avoidance inequalities. For each neighbour ``j``
        find the first step ``k_c`` where the predicted (input-space) distance
        is below ``g(rmin)`` and add one linearised cut. The slack variable
        for neighbour ``j`` is ``eps_{ij}`` (one per neighbour); if no
        collision is predicted, no row is added and the slack stays at zero."""
        K = self.p.k_hor
        rmin = self.p.rmin
        g_thresh = self.p.g_factor * rmin
        Theta_inv = self._Theta_inv # for inter-agent CA
        n_nbr = nbr_pred.shape[0]
        n_obs = self._num_static_obstacles

        inter_agent_kc = -1
        l_obs_kc = []
        n_slack = 0
        if n_nbr + n_obs == 0:
            return None, None, None, inter_agent_kc, l_obs_kc, n_slack

        rows = []
        lbs = []
        ubs = []
        # collision_sample_indices: list[int] = []
        
        # Collision detection
        # 1. Determine k_c,i: the first timestep where there exists a neighbor j s.t. d < rmin
        # 2. Determine Omega_i: the set of neighbors j s.t. d < g_thresh at k_c,i

        # 1. Determine k_c,i
        for k in range(1, K):
            # Inter-agent
            for j in range(n_nbr):
                d = Theta_inv @ (own_pred[k - 1] - nbr_pred[j, k])
                if np.linalg.norm(d) < rmin:
                    inter_agent_kc = k
                    break
            if inter_agent_kc > 0:
                break
        
        # Static obstacle
        for o in range(n_obs):
            obs_kc_o = []
            for k in range(1, K):
                Theta_inv_o = self._obs_theta_inv[env_idx, o]
                d = Theta_inv_o @ (own_pred[k - 1] - self._obstacle_pos_local[env_idx, o])
                if np.linalg.norm(d) < self._obs_g_thres[env_idx, o]: # more conservative collision checking for static obstacles #self._obs_rmin[env_idx, o]:
                    obs_kc_o.append(k)
            l_obs_kc.append(obs_kc_o)
        
        # 2. Determine Omega_i and construct constraint
        if inter_agent_kc >= 0:
            Phi_kc = self.M_pos_hor[3 * (inter_agent_kc - 1) : 3 * (inter_agent_kc - 1) + 3, :]
            # Inter-agent
            for j in range(n_nbr):
                d = Theta_inv @ (own_pred[inter_agent_kc - 1] - nbr_pred[j, inter_agent_kc])
                if np.linalg.norm(d) < g_thresh:
                    # construct collision constraint
                    unit_diff = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else np.array([1.0, 0.0, 0.0])
                    dg_dU = unit_diff.transpose() @ Theta_inv @ Phi_kc
                    rhs_lower = unit_diff.transpose() @ Theta_inv @ nbr_pred[j, inter_agent_kc] + rmin

                    row = np.zeros(self.n_bez) # + n_slack)
                    row = dg_dU
                    # row[: self.n_bez] = dg_dU
                    # row[self.n_bez + j] = -1
                    
                    rows.append(row)
                    lbs.append(rhs_lower)
                    ubs.append(np.inf)

                    n_slack += 1

        # static obstacle
        for (o, obs_kc_o) in enumerate(l_obs_kc):
            Theta_inv_o = self._obs_theta_inv[env_idx, o]
            obstacle_pos = self._obstacle_pos_local[env_idx, o]
            for kc_oi in obs_kc_o:
                Phi_kc_oi = self.M_pos_hor[3 * (kc_oi - 1) : 3 * (kc_oi - 1) + 3, :]
                d = Theta_inv_o @ (own_pred[kc_oi - 1] - obstacle_pos)
                if np.linalg.norm(d) < self._obs_g_thres[env_idx, o]:
                    unit_diff = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else np.array([1.0, 0.0, 0.0])
                    dg_dU = unit_diff.transpose() @ Theta_inv_o @ Phi_kc_oi
                    rhs_lower = unit_diff.transpose() @ Theta_inv_o @ obstacle_pos + self._obs_rmin[env_idx, o]

                    row = np.zeros(self.n_bez) # + n_slack)
                    row = dg_dU

                    rows.append(row)
                    lbs.append(rhs_lower)
                    ubs.append(np.inf)
                    n_slack += 1 

        if rows:
            A_col = np.hstack([np.vstack(rows), -np.eye(n_slack)])
            return A_col, np.asarray(lbs), np.asarray(ubs), inter_agent_kc, l_obs_kc, n_slack
        else:
            return None, None, None, inter_agent_kc, l_obs_kc, n_slack
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
        seeds = [
            (self._M_deriv_h[r] @ U).reshape(3)
            for r in range(self.p.deg_poly + 1)
        ]
        self._state[(env_idx, agent_idx)] = {
            "U": U,
            "u_pred": u_pred,
            "seeds": seeds,
            "steps": 0,
            "collision_timestep": -1,
            "collision_sample_indices": [],
        }
