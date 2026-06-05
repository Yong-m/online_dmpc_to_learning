"""multi_drone_dmpc_rl_env.py — RL env with DMPC-guided dense reward.

Inherits from MultiDroneDmpcEnv and overrides rewards/dones for PPO training.

Reward design:
  - r_guide     (dense)  DMPC velocity alignment per drone per step
  - r_dist      (dense)  tanh-shaped distance-to-goal shaping
  - r_goal_all  (sparse) +goal_reward when ALL drones simultaneously reach goal
  - r_collision (dense)  graduated penalty: max at 0 m, zero at rmin
  - r_oob       (dense)  flat penalty per OOB drone (same scale as max collision)
  - r_ang_vel   (dense)  small angular-velocity smoothness penalty

Termination / truncation:
  - terminated: ALL drones simultaneously hold goal for success_hold_s (env-wise)
  - truncated:  timeout only — collision and OOB handled via penalties

Spawn safety:
  - Initial positions and target positions are checked cross-pair:
    init[i] vs goal[j] for all i,j must be > rmin (goal z is perturbed if not).
    goal[i] vs goal[j] pairs are also checked.

N curriculum:
  - When n_curriculum_min < num_drones, env e permanently uses (e % N) + 1 active drones.
  - Inactive drones are parked above the workspace with zeroed actions.
  - Their obs slots and reward contributions are masked to zero.
"""

from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.utils import configclass

from quadcopter.multi_drone_dmpc_env import (
    MultiDroneDmpcEnv,
    MultiDroneDmpcEnvCfg,
    HISTORY_STEPS,
    PER_DRONE_OWN_DIM,
    PER_NEIGHBOUR_DIM,
)


# ── Env config ────────────────────────────────────────────────────────────────
@configclass
class MultiDroneDmpcRLEnvCfg(MultiDroneDmpcEnvCfg):
    # Disable hard collision termination; use truncation instead.
    terminate_on_collision: bool = False
    # Disable z-bounds termination; handle in truncation below.
    terminate_on_bounds: bool = False

    # ── Success criterion (override parent's 0.05 m / 1.0 s) ────────────
    success_dist_threshold: float = 0.1   # metres; loosened for RL learning
    success_hold_s: float = 0.5           # seconds = 10 steps @ 20 Hz

    # ── Sparse reward ───────────────────────────────────────────────────
    goal_reward: float = 10000.0          # all drones simultaneously reach goal
    per_drone_goal_reward: float = 1000.0  # per drone individually holding goal

    # ── Dense rewards ───────────────────────────────────────────────────
    dmpc_guide_scale: float = 20.0        # DMPC velocity-alignment reward (per drone, *step_dt)
    dmpc_guide_sigma_v: float = 1.0      # velocity bandwidth [m/s]
    dist_to_goal_far_scale: float = 30.0     # broad goal-attraction shaping (per drone, *step_dt)
    dist_to_goal_far_tanh_scale: float = 1.2 # larger denominator keeps gradient farther from goal
    dist_to_goal_near_scale: float = 20.0    # near-goal precision shaping (per drone, *step_dt)
    dist_to_goal_near_tanh_scale: float = 0.8 # smaller denominator sharpens near-goal gradient

    # ang_vel_reward_scale is inherited from parent (-0.01); kept as-is.

    # ── Collision penalty (graduated by distance) ────────────────────────
    # At dist=0 → collision_step_penalty * step_dt; at dist=rmin → 0.
    collision_step_penalty: float = -100.0  # multiplied by (violation/rmin) * step_dt

    # ── OOB penalty (graduated by penetration depth) ──────────────────────
    # 0 inside bounds; grows linearly with L2 penetration distance;
    # reaches oob_step_penalty * step_dt at oob_ref_dist beyond the wall.
    # Averaged over drones so scale is N-independent (max = collision max).
    oob_step_penalty: float = -50.0  # penalty at oob_ref_dist penetration
    oob_ref_dist: float = 0.5         # penetration depth [m] for full penalty

    # ── Attitude (tilt) penalty ──────────────────────────────────────────
    # 1 - cos(tilt): 0 when upright, up to 2 when inverted.
    # Summed over drones, multiplied by step_dt.
    tilt_reward_scale: float = -5.0

    # ── DMPC guide ───────────────────────────────────────────────────────
    dmpc_guide_enabled: bool = True

    # ── Phase-2 curriculum terminations ──────────────────────────────────
    # When enabled, collisions and OOB are treated as terminated (V=0) rather
    # than soft penalties.  Enable when resuming from a converged phase-1 run.
    terminate_on_collision_rl: bool = False  # terminate when any active pair dist < collision_terminate_dist
    collision_terminate_dist: float = 0.12    # hard terminate distance [m], separate from DMPC rmin
    terminate_on_oob_rl: bool = False        # terminate when any active drone leaves workspace

    # ── Hover incentive near goal ─────────────────────────────────────────
    # Penalises linear velocity scaled by proximity to goal.  Off by default.
    hover_vel_scale: float = 0.0        # penalty weight (per drone, *step_dt)
    hover_proximity_dist: float = 0.4   # tanh half-width [m] for proximity gate

    # ── N curriculum ─────────────────────────────────────────────────────
    # Static per-env drone count: env e uses (e % num_drones) + 1 active drones.
    # This covers 1..num_drones uniformly across envs.
    # Inactive drones (index >= active_n) are permanently parked above the
    # workspace with zero actions; their obs/reward slots are masked to zero.
    # Set n_curriculum_min = 0 (default) to disable (all envs use num_drones).
    n_curriculum_min: int = 0  # any value 1..num_drones-1 enables curriculum; 0 = disabled


# ── Env class ─────────────────────────────────────────────────────────────────
class MultiDroneDmpcRLEnv(MultiDroneDmpcEnv):
    """Multi-drone env for PPO training with DMPC as a dense reward guide."""

    cfg: MultiDroneDmpcRLEnvCfg

    def __init__(self, cfg: MultiDroneDmpcRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # DMPC expert for guide reward.
        self._expert = None
        self._dmpc_ref_vel_w = torch.zeros(self.num_envs, self.N, 3, device=self.device)

        # Per-drone success already rewarded this episode — prevents repeat firing.
        self._drone_success_rewarded = torch.zeros(
            self.num_envs, self.N, dtype=torch.bool, device=self.device
        )

        if cfg.dmpc_guide_enabled:
            self._init_dmpc_expert()

        # N curriculum: static per-env active drone count assigned once at init.
        # env_id % N + 1 gives 1..N covering the full range uniformly.
        # When n_curriculum_min == 0 (disabled) every env uses all N drones.
        n_min = cfg.n_curriculum_min
        self._curriculum_enabled = (0 < n_min < self.N)
        self._n_curriculum_min = max(1, n_min) if self._curriculum_enabled else self.N
        if self._curriculum_enabled:
            # Deterministic: env e uses (e % N) + 1 active drones.
            env_ids_all = torch.arange(self.num_envs, device=self.device)
            self._active_n = (env_ids_all % self.N) + 1          # (E,) in 1..N
        else:
            self._active_n = torch.full(
                (self.num_envs,), self.N, device=self.device, dtype=torch.long
            )

        # Remove parent keys that are never updated (parent _get_rewards is overridden).
        for _k in ("lin_vel", "ang_vel", "distance_to_goal", "collision"):
            self._episode_sums.pop(_k, None)

        # RL-specific episode-sum accumulators.
        self._episode_sums["guide"]           = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums["collision_rl"]    = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums["dist_to_goal_rl"] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums["oob_rl"]          = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums["per_drone_goal"]  = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums["goal_all"]        = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums["ang_vel_rl"]      = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums["tilt_rl"]         = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums["hover_rl"]        = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums["total_rl"]        = torch.zeros(self.num_envs, device=self.device)

    # ── Curriculum helpers ────────────────────────────────────────────────────
    @property
    def _active_mask(self) -> torch.Tensor:
        """(E, N) bool — True when drone is active this episode."""
        drone_idx = torch.arange(self.N, device=self.device).unsqueeze(0)  # (1, N)
        return drone_idx < self._active_n.unsqueeze(1)                      # (E, N)

    def _mask_curriculum_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Zero obs slots for inactive drones (own state + neighbor slots).

        obs: (E, N, per_drone_obs_dim) — modified in-place clone.
        """
        active = self._active_mask  # (E, N)

        # Zero inactive drone own obs
        obs = obs * active.unsqueeze(-1)

        # Zero neighbor slots in each drone's obs that correspond to inactive drones.
        # Neighbor ordering for drone i: [0,1,...,i-1, i+1,...,N-1] (j-th entry skips i).
        own_width = PER_DRONE_OWN_DIM + self.per_drone_sdf_dim
        for i in range(self.N):
            k = 0
            for j in range(self.N):
                if j == i:
                    continue
                s = own_width + k * PER_NEIGHBOUR_DIM
                # active[:, j]: (E,) — 0 when drone j is inactive
                obs[:, i, s : s + PER_NEIGHBOUR_DIM] *= active[:, j].unsqueeze(-1)
                k += 1

        return obs

    # ── Obs override ─────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        per_drone = self.get_per_drone_obs()  # (E, N, per_drone_obs_dim)
        if self._curriculum_enabled:
            per_drone = self._mask_curriculum_obs(per_drone)
        policy = per_drone.reshape(self.num_envs, -1)
        active_n = self._active_n.to(dtype=policy.dtype).unsqueeze(-1)
        critic = torch.cat([policy, active_n], dim=-1)
        return {"policy": policy, "critic": critic}

    # ── Action override ───────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._curriculum_enabled:
            inactive = ~self._active_mask                              # (E, N)
            act = actions.view(self.num_envs, self.N, -1)             # (E, N, act_per_drone)
            inactive3 = inactive.unsqueeze(-1)                         # (E, N, 1)

            if act.shape[-1] == 9:
                # N×9 full-reference mode: zero-filling ref_pos sends drones to world origin.
                # Instead keep ref_pos = _last_ref_pos_w (set to park position on reset)
                # and zero velocity + acceleration.
                act9 = act.clone()
                act9[:, :, 0:3] = torch.where(inactive3, self._last_ref_pos_w, act9[:, :, 0:3])
                act9[:, :, 3:9] = act9[:, :, 3:9].masked_fill(inactive3, 0.0)
                actions = act9.view(self.num_envs, -1)
            else:
                # N×3 v_ref mode: zero velocity → cascade integrates from park position safely.
                actions = act.masked_fill(inactive3, 0.0).view(self.num_envs, -1)
        super()._pre_physics_step(actions)

    def _init_dmpc_expert(self) -> None:
        from quadcopter.dmpc_gpu_expert import GPUDMPCExpert
        from quadcopter.dmpc_expert import DMPCParams
        ts = self.cfg.sim.dt * self.cfg.decimation   # actual env control period
        self._expert = GPUDMPCExpert(
            num_drones=self.N,
            num_envs=self.num_envs,
            params=DMPCParams(
                pmin=self.cfg.pos_min,
                pmax=self.cfg.pos_max,
                rmin=self.cfg.rmin,
                ts=ts,
                max_envs=self.num_envs,
            ),
            device=self.device,
        )

    # ── DMPC helper ──────────────────────────────────────────────────────────
    def _update_dmpc_ref(self) -> None:
        if self._expert is None:
            return
        st = self._stack_drone_state()
        _, ref_vel_w, _ = self._expert.compute(
            pos_w=st["pos_w"],
            vel_w=st["lin_vel_w"],
            goal_w=self._goal_pos_w,
            env_origins=self._terrain.env_origins,
        )
        self._dmpc_ref_vel_w = ref_vel_w.detach()

    # ── Rewards ──────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        self._update_dmpc_ref()

        st = self._stack_drone_state()
        pos_w     = st["pos_w"]       # (E, N, 3)
        lin_vel_w = st["lin_vel_w"]   # (E, N, 3)
        ang_vel_b = st["ang_vel_b"]   # (E, N, 3)
        quat_w    = st["quat_w"]      # (E, N, 4) — [w, x, y, z]

        # Active-drone mask and per-drone normaliser.
        # When curriculum is off: amask = all-ones, inv_n = 1 (sum, matching original behaviour).
        # When curriculum is on:  amask zeros inactive slots, inv_n = 1/active_n
        #                         so each reward term is a per-drone average.
        if self._curriculum_enabled:
            amask   = self._active_mask.float()             # (E, N) — 1=active 0=inactive
            inv_n   = 1.0 / amask.sum(dim=-1).clamp(min=1.0)  # (E,)
        else:
            amask   = torch.ones(self.num_envs, self.N, device=self.device)
            inv_n   = torch.ones(self.num_envs, device=self.device)  # sum unchanged

        # ── Dense: DMPC velocity alignment ──────────────────────────────
        if self._expert is not None:
            vel_err = (lin_vel_w - self._dmpc_ref_vel_w).norm(dim=-1)   # (E, N)
            r_guide_per = torch.exp(
                -(vel_err ** 2) / (2.0 * self.cfg.dmpc_guide_sigma_v ** 2)
            )
            r_guide = (r_guide_per * amask).sum(dim=-1) * inv_n * self.cfg.dmpc_guide_scale * self.step_dt
        else:
            r_guide = torch.zeros(self.num_envs, device=self.device)

        # ── Dense: distance-to-goal shaping ─────────────────────────────
        # Split into broad far-field attraction and sharper near-goal precision.
        dist_to_goal = (self._goal_pos_w - pos_w).norm(dim=-1)          # (E, N)
        r_dist_far_per = 1.0 - torch.tanh(
            dist_to_goal / self.cfg.dist_to_goal_far_tanh_scale
        )
        r_dist_near_per = 1.0 - torch.tanh(
            dist_to_goal / self.cfg.dist_to_goal_near_tanh_scale
        )
        r_dist_per = (
            self.cfg.dist_to_goal_far_scale * r_dist_far_per
            + self.cfg.dist_to_goal_near_scale * r_dist_near_per
        )
        r_dist = (
            (r_dist_per * amask).sum(dim=-1)
            * inv_n
            * self.step_dt
        )

        # ── Dense: OOB penalty (graduated by penetration depth) ──────────
        pos_local = pos_w - self._terrain.env_origins.unsqueeze(1)  # (E, N, 3)
        viol_lo = (self._pos_min - pos_local).clamp(min=0.0)        # (E, N, 3)
        viol_hi = (pos_local - self._pos_max).clamp(min=0.0)        # (E, N, 3)
        viol_dist = torch.maximum(viol_lo, viol_hi).norm(dim=-1)    # (E, N)
        viol_frac = (viol_dist / self.cfg.oob_ref_dist).clamp(0.0, 1.0)  # (E, N)
        r_oob = (viol_frac * amask).sum(dim=-1) * inv_n * self.cfg.oob_step_penalty * self.step_dt

        # ── Dense: graduated collision penalty ───────────────────────────
        # Only penalise pairs where BOTH drones are active.
        if self.N > 1:
            diff      = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)          # (E, N, N, 3)
            pair_dist = diff.norm(dim=-1)                                  # (E, N, N)
            eye_m     = torch.eye(self.N, device=self.device, dtype=torch.bool).unsqueeze(0)
            pair_dist = pair_dist.masked_fill(eye_m, float("inf"))
            if self._curriculum_enabled:
                # Mask out pairs involving inactive drones.
                both_active = amask.unsqueeze(2) * amask.unsqueeze(1)     # (E, N, N)
                pair_dist   = pair_dist.masked_fill(both_active < 0.5, float("inf"))
            min_pair  = pair_dist.amin(dim=(1, 2))                        # (E,)
            viol_frac = (self.cfg.rmin - min_pair).clamp(min=0.0) / self.cfg.rmin
            r_coll    = viol_frac * self.cfg.collision_step_penalty * self.step_dt
        else:
            r_coll = torch.zeros(self.num_envs, device=self.device)

        # ── Sparse: per-drone individual success (fires exactly once per drone) ──
        newly_succeeded = self._drone_just_succeeded & ~self._drone_success_rewarded
        if self._curriculum_enabled:
            # Only reward active drones.
            newly_succeeded = newly_succeeded & self._active_mask
        self._drone_success_rewarded |= newly_succeeded
        r_per_drone = (
            newly_succeeded.float().sum(dim=-1) * self.cfg.per_drone_goal_reward
        )

        # ── Sparse: all-drone env-wise success ───────────────────────────
        r_goal_all = self._just_succeeded.float() * self.cfg.goal_reward

        # ── Dense: angular velocity smoothness ───────────────────────────
        ang_pen = (ang_vel_b.square().sum(dim=-1) * amask).sum(dim=-1) * inv_n
        r_ang   = ang_pen * self.cfg.ang_vel_reward_scale * self.step_dt

        # ── Dense: attitude (tilt) penalty ───────────────────────────────
        # cos(tilt) = body-z · world-z = w²-x²-y²+z² for quat [w,x,y,z].
        w = quat_w[..., 0]; x = quat_w[..., 1]
        y = quat_w[..., 2]; z = quat_w[..., 3]
        cos_tilt = w*w - x*x - y*y + z*z                                  # (E, N)
        tilt_pen = ((1.0 - cos_tilt) * amask).sum(dim=-1) * inv_n         # (E,)
        r_tilt   = tilt_pen * self.cfg.tilt_reward_scale * self.step_dt

        # ── Dense: hover incentive (linear velocity near goal) ───────────
        # Penalises velocity weighted by closeness to goal so the policy learns
        # to decelerate and hold position once it arrives.
        if self.cfg.hover_vel_scale != 0.0:
            proximity = 1.0 - torch.tanh(dist_to_goal / self.cfg.hover_proximity_dist)  # (E, N)
            vel_mag   = lin_vel_w.norm(dim=-1)                                            # (E, N)
            r_hover   = -(vel_mag * proximity * amask).sum(dim=-1) * inv_n * self.cfg.hover_vel_scale * self.step_dt
        else:
            r_hover = torch.zeros(self.num_envs, device=self.device)

        total_unscaled = r_guide + r_dist + r_coll + r_oob + r_per_drone + r_goal_all + r_ang + r_tilt + r_hover
        total = total_unscaled * 0.1

        self._episode_sums["guide"]           += r_guide
        self._episode_sums["collision_rl"]    += r_coll
        self._episode_sums["dist_to_goal_rl"] += r_dist
        self._episode_sums["oob_rl"]          += r_oob
        self._episode_sums["per_drone_goal"]  += r_per_drone
        self._episode_sums["goal_all"]        += r_goal_all
        self._episode_sums["ang_vel_rl"]      += r_ang
        self._episode_sums["tilt_rl"]         += r_tilt
        self._episode_sums["hover_rl"]        += r_hover
        self._episode_sums["total_rl"]        += total

        return total

    # ── Dones ─────────────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Save success counters before super() modifies them with an all-N check.
        # When curriculum is active we'll restore and redo with active-only check.
        if self._curriculum_enabled:
            saved_succ  = self._success_steps.clone()
            saved_drone = self._drone_success_steps.clone()

        _, time_out = super()._get_dones()
        # terminate_on_collision=False, terminate_on_bounds=False → "died" is
        # always zero in this env cfg; terminated = succeeded from super().
        # We discard super()'s terminated and recompute below.

        pos_w = torch.stack([r.data.root_pos_w for r in self._robots], dim=1)

        if self._curriculum_enabled:
            # Restore counters, then redo success check counting only active drones.
            self._success_steps.copy_(saved_succ)
            self._drone_success_steps.copy_(saved_drone)

            dist = torch.linalg.norm(pos_w - self._goal_pos_w, dim=-1)   # (E, N)
            active = self._active_mask                                     # (E, N)
            # Treat inactive drones as always "at goal" (dist → 0).
            dist_check = dist.masked_fill(~active, 0.0)

            all_close = (dist_check < self.cfg.success_dist_threshold).all(dim=-1)
            self._success_steps[all_close] += 1
            self._success_steps[~all_close] = 0
            hold_steps = max(1, round(self.cfg.success_hold_s / self.step_dt))
            succeeded = self._success_steps >= hold_steps
            self._just_succeeded[:] = succeeded

            per_close = dist_check < self.cfg.success_dist_threshold      # (E, N)
            self._drone_success_steps[per_close] += 1
            self._drone_success_steps[~per_close] = 0
            self._drone_just_succeeded[:] = (
                self._drone_success_steps >= hold_steps
            )
            terminated = succeeded
        else:
            # No curriculum: super()'s success check was correct.
            terminated = self._just_succeeded.clone()

        # Hard floor: physically unrecoverable, truncate with bootstrap.
        on_ground = (pos_w[..., 2] < self.cfg.z_min).any(dim=-1)

        # ── Phase-2 curriculum: hard terminations (off by default) ───────
        # Collision termination: any active drone pair closer than the physical
        # hard-stop threshold.  This is intentionally separate from cfg.rmin,
        # which remains the DMPC/safety-margin and penalty radius.
        if self.cfg.terminate_on_collision_rl and self.N > 1:
            diff      = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)           # (E, N, N, 3)
            pair_dist = diff.norm(dim=-1)                                   # (E, N, N)
            eye_m     = torch.eye(self.N, device=self.device, dtype=torch.bool).unsqueeze(0)
            pair_dist = pair_dist.masked_fill(eye_m, float("inf"))
            if self._curriculum_enabled:
                both_active = self._active_mask.unsqueeze(2) * self._active_mask.unsqueeze(1)
                pair_dist   = pair_dist.masked_fill(both_active < 0.5, float("inf"))
            collision_term = (pair_dist < self.cfg.collision_terminate_dist).any(dim=(1, 2))
        else:
            collision_term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # OOB termination: any active drone outside workspace bounds.
        if self.cfg.terminate_on_oob_rl:
            pos_local = pos_w - self._terrain.env_origins.unsqueeze(1)    # (E, N, 3)
            oob_per   = (
                (pos_local < torch.tensor(self.cfg.pos_min, device=self.device)) |
                (pos_local > torch.tensor(self.cfg.pos_max, device=self.device))
            ).any(dim=-1)                                                   # (E, N)
            if self._curriculum_enabled:
                oob_per = oob_per & self._active_mask
            oob_term = oob_per.any(dim=-1)                                 # (E,)
        else:
            oob_term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return terminated | collision_term | oob_term, time_out | on_ground

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self._robots[0]._ALL_INDICES
        if self._expert is not None:
            obstacle_info = self.get_obstacle_info() if hasattr(self, "get_obstacle_info") else None
            self._expert.reset(env_ids, obstacle_info=obstacle_info)
        self._drone_success_rewarded[env_ids] = False

        # Fix active drone goals first (before inactive drones are parked).
        self._fix_spawn_collisions(env_ids)

        # ── N curriculum ────────────────────────────────────────────────
        if self._curriculum_enabled:
            self._apply_n_curriculum(env_ids)

    def _apply_n_curriculum(self, env_ids: torch.Tensor) -> None:
        """Park inactive drones above the workspace for this episode.

        active_n per env is fixed at initialisation (env_id % N + 1) and does
        NOT change across resets.  Only the physical placement and goal/history
        bookkeeping are updated here so the parked drones hover in place.
        """
        n = len(env_ids)
        # _active_n is already set per-env at init — do NOT overwrite it here.

        drone_idx = torch.arange(self.N, device=self.device).unsqueeze(0)  # (1, N)
        inactive = drone_idx >= self._active_n[env_ids].unsqueeze(1)       # (n, N) bool

        # Park positions: keep each drone's init XY (circle-spaced, already
        # collision-free) but lift Z above the workspace.  This avoids stacking
        # multiple inactive drones at the same point (physics instability) and
        # keeps parked drones far from active drone goals (min Z gap ≥ 3 m).
        park_z = self.cfg.z_max + 3.0
        init_xy = self._init_pos_w[env_ids][..., :2]                       # (n, N, 2)
        park_pos_per_drone = torch.cat([
            init_xy,
            torch.full((n, self.N, 1), park_z, device=self.device),
        ], dim=-1)                                                          # (n, N, 3)

        # Teleport each inactive drone to its per-drone park position.
        identity_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).unsqueeze(0).repeat(n, 1)                                        # (n, 4) contiguous
        for k in range(self.N):
            local_inactive = inactive[:, k]                                # (n,) bool
            if not local_inactive.any():
                continue
            global_inactive = env_ids[local_inactive]
            root = self._robots[k].data.default_root_state[global_inactive].clone()
            root[:, :3] = park_pos_per_drone[local_inactive, k]
            root[:, 3:7] = identity_quat[local_inactive]
            root[:, 7:] = 0.0
            self._robots[k].write_root_pose_to_sim(root[:, :7], global_inactive)
            self._robots[k].write_root_velocity_to_sim(root[:, 7:], global_inactive)

        park_exp = park_pos_per_drone                                      # (n, N, 3) — per-drone already
        inactive3 = inactive.unsqueeze(-1)                                 # (n, N, 1)

        # Goals and reference positions → park position (PID stays put).
        self._goal_pos_w[env_ids] = torch.where(
            inactive3, park_exp, self._goal_pos_w[env_ids]
        )
        self._init_pos_w[env_ids] = torch.where(
            inactive3, park_exp, self._init_pos_w[env_ids]
        )
        self._last_ref_pos_w[env_ids] = torch.where(
            inactive3, park_exp, self._last_ref_pos_w[env_ids]
        )
        # Pos history → fill all HISTORY_STEPS slots with park position.
        park_hist = park_exp.unsqueeze(2).expand(-1, self.N, HISTORY_STEPS, -1)
        inactive_hist = inactive.unsqueeze(-1).unsqueeze(-1).expand_as(park_hist)
        self._pos_history[env_ids] = torch.where(
            inactive_hist, park_hist, self._pos_history[env_ids]
        )

        # Mark inactive drones as already-rewarded (no per-drone bonus for them).
        self._drone_success_rewarded[env_ids] |= inactive

    def _fix_spawn_collisions(self, env_ids: torch.Tensor) -> None:
        """Perturb goal z-heights until all init-goal and goal-goal pairs are > rmin apart.

        Checks two kinds of proximity:
          1. init[i] vs goal[j] (original): circular reset places init[i] at goal[j]'s xy.
          2. goal[i] vs goal[j]: independent z sampling can bring goals within rmin in 3D.
        """
        if len(env_ids) == 0:
            return

        rmin   = self.cfg.rmin
        z_lo   = self._pos_min[2] + 0.1
        z_hi   = self._pos_max[2] - 0.1

        # Use stored init positions (inactive drone positions will be overridden by curriculum later).
        init_pos = self._init_pos_w[env_ids]  # (n, N, 3)
        goal_pos = self._goal_pos_w[env_ids].clone()  # (n, N, 3)

        eye_g = torch.eye(self.N, device=self.device, dtype=torch.bool)  # (N, N)

        for _ in range(30):
            any_bad = False

            # 1) init[i] vs goal[j]
            cross_dist = (
                init_pos.unsqueeze(2) - goal_pos.unsqueeze(1)
            ).norm(dim=-1)  # (n, N, N)
            bad = cross_dist < rmin
            if bad.any():
                any_bad = True
                bad_e, _, bad_j = bad.nonzero(as_tuple=True)
                goal_pos[bad_e, bad_j, 2] = (
                    goal_pos[bad_e, bad_j, 2] + rmin * 0.6
                ).clamp(z_lo, z_hi)

            # 2) goal[i] vs goal[j] (skip self-pairs)
            goal_dist = (
                goal_pos.unsqueeze(2) - goal_pos.unsqueeze(1)
            ).norm(dim=-1)  # (n, N, N)
            goal_dist = goal_dist.masked_fill(eye_g.unsqueeze(0), float("inf"))
            bad_g = goal_dist < rmin
            if bad_g.any():
                any_bad = True
                bad_e, _, bad_j = bad_g.nonzero(as_tuple=True)
                goal_pos[bad_e, bad_j, 2] = (
                    goal_pos[bad_e, bad_j, 2] + rmin * 0.6
                ).clamp(z_lo, z_hi)

            if not any_bad:
                break

        self._goal_pos_w[env_ids] = goal_pos


# ── Gym registration ──────────────────────────────────────────────────────────
gym.register(
    id="Isaac-MultiDrone-DMPC-RL-Direct-v0",
    entry_point="quadcopter.multi_drone_dmpc_rl_env:MultiDroneDmpcRLEnv",
    disable_env_checker=True,
    kwargs={
        "cfg": MultiDroneDmpcRLEnvCfg(),
    },
)
