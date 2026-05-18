# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Python re-implementation of the DMPC expert from
``git_branch/online_dmpc_to_learning/cpp``.

This is a *minimal MVP*: each agent solves its own QP over an ``k_hor``-step
horizon with acceleration controls of a 3-D double integrator, position /
acceleration box constraints, and **on-demand** linearised collision constraints
against the predicted trajectories of every other drone in the same env.

Compared with the published C++ implementation the simplifications are:

* No Bezier-curve parametrisation: decision variables are direct acceleration
  samples ``u_0,...,u_{k_hor-1}``. This makes the problem a standard QP that
  ``osqp`` solves quickly (a few ms per agent on CPU).
* Collision avoidance uses spheres (``rmin``) instead of vertically scaled
  ellipsoids.
* Hard-constraint mode only - infeasible solves fall back to a PD command
  toward the goal so the rollout can keep advancing.

The output of :py:meth:`DMPCExpert.compute` is the desired *world-frame*
acceleration for every drone, in the shape ``(num_envs, num_drones, 3)``. Feed
that into :py:meth:`MultiDroneDmpcEnv.acc_to_action` to obtain the 4-D thrust /
moment action that the BC student is trained to imitate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

try:
    import osqp
except ImportError as e:  # pragma: no cover - osqp is a runtime requirement
    raise ImportError(
        "DMPC expert requires osqp. Install with `pip install osqp`."
    ) from e


@dataclass
class DMPCParams:
    """Hyper-parameters mirroring ``cpp/config/config.json``."""

    k_hor: int = 16
    h: float = 0.1
    rmin: float = 0.3
    amin: float = -1.0
    amax: float = 1.0
    pmin: tuple[float, float, float] = (-1.5, -1.5, 0.2)
    pmax: tuple[float, float, float] = (1.5, 1.5, 2.2)
    # Cost weights.
    goal_weight: float = 100.0
    acc_weight: float = 8.0e-3
    smoothness_weight: float = 1.0e-2
    # Extra margin (m) added to rmin for the linearised collision check.
    collision_margin: float = 0.05
    # OSQP options.
    osqp_eps_abs: float = 1e-3
    osqp_eps_rel: float = 1e-3
    osqp_max_iter: int = 200


class DMPCExpert:
    """Distributed MPC expert producing world-frame acceleration commands.

    Usage::

        expert = DMPCExpert(num_drones=N, params=DMPCParams(), device=env.device)
        states = env.get_world_states()
        accel = expert.compute(states["pos_w"], states["lin_vel_w"], states["goal_w"])
        action = env.acc_to_action(accel)
        env.step(action)
    """

    def __init__(self, num_drones: int, params: DMPCParams | None = None, device: str | torch.device = "cpu"):
        self.N = num_drones
        self.p = params or DMPCParams()
        self.device = torch.device(device)

        # Build the propagation matrices once. For a 3-D double integrator under
        # piecewise-constant acceleration of duration ``h``, the position at the
        # end of step k+1 is x0_pos + (k+1)*h*x0_vel + sum_{j<=k} c_{k,j} u_j
        # with c_{k,j} = ((k-j)+0.5) * h^2 (computed below).
        K, h = self.p.k_hor, self.p.h
        a0_pos = np.zeros((K, 3, 6))  # state x0 = [pos, vel] in R^6
        phi_pos = np.zeros((K, K, 3, 3))
        for k in range(K):
            t = (k + 1) * h
            a0_pos[k, :, :3] = np.eye(3)
            a0_pos[k, :, 3:] = np.eye(3) * t
            for j in range(k + 1):
                coef = ((k - j) + 0.5) * h * h
                phi_pos[k, j] = np.eye(3) * coef
        self._A0_pos = a0_pos
        self._Phi_pos = phi_pos

        # Stack to (3K, 6) and (3K, 3K) for QP construction.
        self._A0_pos_stack = a0_pos.reshape(3 * K, 6)
        Phi = np.zeros((3 * K, 3 * K))
        for k in range(K):
            for j in range(k + 1):
                Phi[3 * k : 3 * k + 3, 3 * j : 3 * j + 3] = phi_pos[k, j]
        self._Phi_pos_stack = Phi  # (3K, 3K)

        self._build_constant_qp_pieces()

        # Warm-start cache per (env, drone).
        self._last_u: dict[tuple[int, int], np.ndarray] = {}

    # ── QP construction ────────────────────────────────────────────────────
    def _build_constant_qp_pieces(self) -> None:
        K = self.p.k_hor
        n = 3 * K

        Phi = self._Phi_pos_stack
        Phi_last = Phi[-3:, :]  # (3, n) - effect on terminal position
        A0_last = self._A0_pos_stack[-3:, :]  # (3, 6)

        # First-difference operator for smoothness regularisation.
        D = np.zeros((3 * (K - 1), n))
        for k in range(K - 1):
            D[3 * k : 3 * k + 3, 3 * k : 3 * k + 3] = -np.eye(3)
            D[3 * k : 3 * k + 3, 3 * (k + 1) : 3 * (k + 1) + 3] = np.eye(3)

        H = (
            2.0 * self.p.goal_weight * (Phi_last.T @ Phi_last)
            + 2.0 * self.p.acc_weight * np.eye(n)
            + 2.0 * self.p.smoothness_weight * (D.T @ D)
        )
        H = 0.5 * (H + H.T)
        self._H_const = sp.csc_matrix(H)
        self._Phi_last = Phi_last
        self._A0_last = A0_last
        self._n = n

        # Box constraints.
        self._u_box = (np.full(n, self.p.amin), np.full(n, self.p.amax))
        self._pos_box = (
            np.tile(np.array(self.p.pmin), K),
            np.tile(np.array(self.p.pmax), K),
        )

    # ── public API ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def compute(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        goal_w: torch.Tensor,
        env_origins: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run DMPC for the first control step of every drone in every env.

        The DMPC formulation uses *env-local* coordinates so the workspace box
        ``pmin/pmax`` matches the original config regardless of where each env
        sits in the world. Pass ``env_origins`` (``(E, 3)``) to subtract them
        before planning; goal positions are translated to the same frame.

        Args:
            pos_w: ``(E, N, 3)`` world-frame positions.
            vel_w: ``(E, N, 3)`` world-frame linear velocities.
            goal_w: ``(E, N, 3)`` world-frame goal positions.
            env_origins: optional ``(E, 3)`` offset subtracted from positions
                and goals before planning. The returned acceleration is frame-
                agnostic so no inverse transform is needed.
            env_ids: optional subset of envs to plan for.

        Returns:
            ``(E, N, 3)`` first-step acceleration command.
        """
        E, N = pos_w.shape[0], pos_w.shape[1]
        out = torch.zeros((E, N, 3), device=self.device, dtype=torch.float32)
        env_iter = range(E) if env_ids is None else env_ids.detach().cpu().tolist()

        pos_np = pos_w.detach().cpu().numpy().astype(np.float64)
        vel_np = vel_w.detach().cpu().numpy().astype(np.float64)
        goal_np = goal_w.detach().cpu().numpy().astype(np.float64)
        if env_origins is not None:
            origins_np = env_origins.detach().cpu().numpy().astype(np.float64)
            pos_np = pos_np - origins_np[:, None, :]
            goal_np = goal_np - origins_np[:, None, :]

        for e in env_iter:
            traj_pred = self._initial_trajectories(pos_np[e], vel_np[e], goal_np[e])
            for i in range(N):
                u = self._solve_agent(
                    pos_np[e, i],
                    vel_np[e, i],
                    goal_np[e, i],
                    neighbour_traj=np.delete(traj_pred, i, axis=0),
                    env_idx=e,
                    agent_idx=i,
                )
                if u is None:
                    a = self._pd_toward_goal(pos_np[e, i], vel_np[e, i], goal_np[e, i])
                else:
                    a = u[:3]
                out[e, i] = torch.from_numpy(a.astype(np.float32)).to(self.device)
        return out

    # ── internals ──────────────────────────────────────────────────────────
    def _initial_trajectories(self, pos: np.ndarray, vel: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Predict ``k_hor`` future positions using a simple PD toward each goal.
        Used to seed the on-demand collision constraints."""
        K = self.p.k_hor
        N = pos.shape[0]
        traj = np.empty((N, K, 3))
        p, v = pos.copy(), vel.copy()
        h = self.p.h
        for k in range(K):
            err = goal - p
            a = np.clip(2.0 * err - 1.5 * v, self.p.amin, self.p.amax)
            p = p + v * h + 0.5 * a * h * h
            v = v + a * h
            traj[:, k] = p
        return traj

    def _pd_toward_goal(self, pos: np.ndarray, vel: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.clip(2.0 * (goal - pos) - 1.5 * vel, self.p.amin, self.p.amax)

    def _solve_agent(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        goal: np.ndarray,
        neighbour_traj: np.ndarray,
        env_idx: int,
        agent_idx: int,
    ) -> np.ndarray | None:
        """Solve the QP for one agent. Returns the optimal acceleration sequence
        ``(3K,)`` or ``None`` if OSQP did not return a usable solution."""
        K = self.p.k_hor
        n = self._n
        x0 = np.concatenate([pos, vel])

        # Linear cost term: q = -2 * goal_w * (pf - A0_last x0)^T Phi_last
        residual = goal - self._A0_last @ x0
        q = -2.0 * self.p.goal_weight * (self._Phi_last.T @ residual)

        A_rows = [sp.eye(n, format="csc")]
        l_rows = [self._u_box[0]]
        u_rows = [self._u_box[1]]

        A_rows.append(sp.csc_matrix(self._Phi_pos_stack))
        l_rows.append(self._pos_box[0] - self._A0_pos_stack @ x0)
        u_rows.append(self._pos_box[1] - self._A0_pos_stack @ x0)

        # Seed our own predicted trajectory for choosing half-space normals.
        own_pred = self._initial_trajectories(pos[None, :], vel[None, :], goal[None, :])[0]

        n_neigh = neighbour_traj.shape[0]
        if n_neigh > 0:
            rmin_eff = self.p.rmin + self.p.collision_margin
            for k in range(K):
                A0_k = self._A0_pos_stack[3 * k : 3 * k + 3, :]
                Phi_k = self._Phi_pos_stack[3 * k : 3 * k + 3, :]
                base_pred = A0_k @ x0  # (3,) - position contribution from x0 only
                for j in range(n_neigh):
                    diff = own_pred[k] - neighbour_traj[j, k]
                    dist = np.linalg.norm(diff)
                    if dist > 3.0 * rmin_eff:
                        continue
                    eta = diff / max(dist, 1e-4)
                    row = eta @ Phi_k  # (n,)
                    lb = rmin_eff + eta @ neighbour_traj[j, k] - eta @ base_pred
                    A_rows.append(sp.csc_matrix(row[None, :]))
                    l_rows.append(np.array([lb]))
                    u_rows.append(np.array([np.inf]))

        A = sp.vstack(A_rows, format="csc")
        l = np.concatenate(l_rows)
        u = np.concatenate(u_rows)

        prob = osqp.OSQP()
        try:
            prob.setup(
                P=self._H_const,
                q=q,
                A=A,
                l=l,
                u=u,
                verbose=False,
                eps_abs=self.p.osqp_eps_abs,
                eps_rel=self.p.osqp_eps_rel,
                max_iter=self.p.osqp_max_iter,
            )
        except ValueError:
            return None

        warm = self._last_u.get((env_idx, agent_idx))
        if warm is not None and warm.shape[0] == n:
            prob.warm_start(warm)
        res = prob.solve()
        # osqp status codes: 1 = solved, 2 = solved_inaccurate.
        if res.info.status_val not in (1, 2):
            return None

        self._last_u[(env_idx, agent_idx)] = res.x.copy()
        return res.x

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Drop cached warm-start solutions for the given envs (all by default)."""
        if env_ids is None:
            self._last_u.clear()
            return
        ids = set(int(i) for i in env_ids.detach().cpu().tolist())
        self._last_u = {k: v for k, v in self._last_u.items() if k[0] not in ids}
