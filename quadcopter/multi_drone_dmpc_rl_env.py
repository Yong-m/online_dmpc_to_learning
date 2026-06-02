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
"""

from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.utils import configclass

from quadcopter.multi_drone_dmpc_env import (
    MultiDroneDmpcEnv,
    MultiDroneDmpcEnvCfg,
)


# ── Env config ────────────────────────────────────────────────────────────────
@configclass
class MultiDroneDmpcRLEnvCfg(MultiDroneDmpcEnvCfg):
    # Disable hard collision termination; use truncation instead.
    terminate_on_collision: bool = False
    # Disable z-bounds termination; handle in truncation below.
    terminate_on_bounds: bool = False

    # ── Sparse reward ───────────────────────────────────────────────────
    goal_reward: float = 10000.0          # all drones simultaneously reach goal
    per_drone_goal_reward: float = 1000.0  # per drone individually holding goal

    # ── Dense rewards ───────────────────────────────────────────────────
    dmpc_guide_scale: float = 20.0        # DMPC velocity-alignment reward (per drone, *step_dt)
    dmpc_guide_sigma_v: float = 1.0      # velocity bandwidth [m/s]
    dist_to_goal_scale: float = 50.0     # tanh distance shaping (per drone, *step_dt)

    # ang_vel_reward_scale is inherited from parent (-0.01); kept as-is.

    # ── Collision penalty (graduated by distance) ────────────────────────
    # At dist=0 → collision_step_penalty * step_dt; at dist=rmin → 0.
    collision_step_penalty: float = -50.0  # multiplied by (violation/rmin) * step_dt

    # ── OOB penalty (graduated by penetration depth) ──────────────────────
    # 0 inside bounds; grows linearly with L2 penetration distance;
    # reaches oob_step_penalty * step_dt at oob_ref_dist beyond the wall.
    # Averaged over drones so scale is N-independent (max = collision max).
    oob_step_penalty: float = -50.0  # penalty at oob_ref_dist penetration
    oob_ref_dist: float = 0.5         # penetration depth [m] for full penalty

    # ── DMPC guide ───────────────────────────────────────────────────────
    dmpc_guide_enabled: bool = True


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
        self._episode_sums["total_rl"]        = torch.zeros(self.num_envs, device=self.device)

    def _init_dmpc_expert(self) -> None:
        from quadcopter.dmpc_expert import DMPCExpert, DMPCParams
        self._expert = DMPCExpert(
            num_drones=self.N,
            params=DMPCParams(),
            device=self.device,
            verbose=False,
        )
        self._expert.enable_gpu_admm(max_envs=self.num_envs)

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

        # ── Dense: DMPC velocity alignment ──────────────────────────────
        if self._expert is not None:
            vel_err = (lin_vel_w - self._dmpc_ref_vel_w).norm(dim=-1)   # (E, N)
            r_guide_per = torch.exp(
                -(vel_err ** 2) / (2.0 * self.cfg.dmpc_guide_sigma_v ** 2)
            )
            r_guide = r_guide_per.sum(dim=-1) * self.cfg.dmpc_guide_scale * self.step_dt
        else:
            r_guide = torch.zeros(self.num_envs, device=self.device)

        # ── Dense: distance-to-goal shaping ─────────────────────────────
        dist_to_goal = (self._goal_pos_w - pos_w).norm(dim=-1)          # (E, N)
        r_dist = (
            (1.0 - torch.tanh(dist_to_goal / 0.8)).sum(dim=-1)
            * self.cfg.dist_to_goal_scale
            * self.step_dt
        )

        # ── Dense: OOB penalty (graduated by penetration depth) ──────────
        # Per-drone L2 penetration distance; averaged over drones so scale
        # is N-independent.  0 inside; linearly grows to oob_step_penalty
        # at oob_ref_dist beyond the wall → smooth recovery gradient.
        pos_local = pos_w - self._terrain.env_origins.unsqueeze(1)  # (E, N, 3)
        viol_lo = (self._pos_min - pos_local).clamp(min=0.0)        # (E, N, 3)
        viol_hi = (pos_local - self._pos_max).clamp(min=0.0)        # (E, N, 3)
        viol_dist = torch.maximum(viol_lo, viol_hi).norm(dim=-1)    # (E, N)
        viol_frac = (viol_dist / self.cfg.oob_ref_dist).clamp(0.0, 1.0)  # (E, N) ∈ [0,1]
        r_oob = viol_frac.mean(dim=-1) * self.cfg.oob_step_penalty * self.step_dt

        # ── Dense: graduated collision penalty ───────────────────────────
        # penalty = |collision_step_penalty| * violation_fraction * step_dt
        # violation_fraction = clamp(rmin - dist, 0) / rmin ∈ [0, 1]
        if self.N > 1:
            diff      = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)          # (E, N, N, 3)
            pair_dist = diff.norm(dim=-1)                                  # (E, N, N)
            eye_m     = torch.eye(self.N, device=self.device, dtype=torch.bool).unsqueeze(0)
            pair_dist = pair_dist.masked_fill(eye_m, float("inf"))
            min_pair  = pair_dist.amin(dim=(1, 2))                        # (E,)
            viol_frac = (self.cfg.rmin - min_pair).clamp(min=0.0) / self.cfg.rmin
            r_coll    = viol_frac * self.cfg.collision_step_penalty * self.step_dt
        else:
            r_coll = torch.zeros(self.num_envs, device=self.device)

        # ── Sparse: per-drone individual success (fires exactly once per drone) ──
        # _drone_just_succeeded stays True every step after hold; gate with
        # _drone_success_rewarded so the bonus fires only on first achievement.
        newly_succeeded = self._drone_just_succeeded & ~self._drone_success_rewarded
        self._drone_success_rewarded |= newly_succeeded
        r_per_drone = (
            newly_succeeded.float().sum(dim=-1) * self.cfg.per_drone_goal_reward
        )

        # ── Sparse: all-drone env-wise success ───────────────────────────
        # _just_succeeded is set by _get_dones() on the previous step.
        r_goal_all = self._just_succeeded.float() * self.cfg.goal_reward

        # ── Dense: angular velocity smoothness ───────────────────────────
        ang_pen = ang_vel_b.square().sum(dim=-1).sum(dim=-1)
        r_ang   = ang_pen * self.cfg.ang_vel_reward_scale * self.step_dt

        total_unscaled = r_guide + r_dist + r_coll + r_oob + r_per_drone + r_goal_all + r_ang
        total = total_unscaled * 0.01

        self._episode_sums["guide"]           += r_guide
        self._episode_sums["collision_rl"]    += r_coll
        self._episode_sums["dist_to_goal_rl"] += r_dist
        self._episode_sums["oob_rl"]          += r_oob
        self._episode_sums["per_drone_goal"]  += r_per_drone
        self._episode_sums["goal_all"]        += r_goal_all
        self._episode_sums["ang_vel_rl"]      += r_ang
        self._episode_sums["total_rl"]        += total

        return total

    # ── Dones ─────────────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()

        # Hard floor: physically unrecoverable, truncate with bootstrap.
        pos_w = torch.stack([r.data.root_pos_w for r in self._robots], dim=1)
        on_ground = (pos_w[..., 2] < self.cfg.z_min).any(dim=-1)

        return terminated, time_out | on_ground

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self._robots[0]._ALL_INDICES
        if self._expert is not None:
            self._expert.reset(env_ids)
        self._drone_success_rewarded[env_ids] = False
        self._fix_spawn_collisions(env_ids)

    def _fix_spawn_collisions(self, env_ids: torch.Tensor) -> None:
        """Perturb goal z-heights until all init-goal cross-pairs are > rmin apart.

        In the circular-exchange reset, init[i] and goal[i+N/2] share the same
        xy position.  Independent z sampling can accidentally place them within
        rmin of each other.  We push goal z up iteratively until all cross-pair
        3-D distances exceed rmin.
        """
        if len(env_ids) == 0:
            return

        rmin   = self.cfg.rmin
        z_lo   = self._pos_min[2] + 0.1
        z_hi   = self._pos_max[2] - 0.1

        # Current drone positions after reset (= init positions).
        init_pos = torch.stack(
            [r.data.root_pos_w[env_ids] for r in self._robots], dim=1
        )  # (n, N, 3)
        goal_pos = self._goal_pos_w[env_ids].clone()  # (n, N, 3)

        for _ in range(30):
            # (n, N_init, N_goal, 3) cross-pair vectors
            cross_dist = (
                init_pos.unsqueeze(2) - goal_pos.unsqueeze(1)
            ).norm(dim=-1)  # (n, N, N)

            bad = cross_dist < rmin  # (n, N, N)
            if not bad.any():
                break

            # For each bad (env, init_drone, goal_drone), nudge goal_drone z up.
            bad_e, _, bad_j = bad.nonzero(as_tuple=True)
            goal_pos[bad_e, bad_j, 2] = (
                goal_pos[bad_e, bad_j, 2] + rmin * 0.6
            ).clamp(z_lo, z_hi)

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
