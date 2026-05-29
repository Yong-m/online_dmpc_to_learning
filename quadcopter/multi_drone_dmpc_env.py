# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-drone DMPC environment (Luis et al. 2020 reference setup).

Mirrors the layout of the standalone ``quadcopter_env.py`` (one Crazyflie per
env, goal-reaching) but instantiates *N* Crazyflies and aligns the action /
observation interface with the multi-robot motion-planning setting of the
``online_dmpc`` paper. Design choices:

* **Action = 3-D velocity reference per drone**: normalised ``v_ref_w`` in
  ``[-1, 1]^3``.  The env integrates one step ahead to produce the position
  reference ``ref_pos_w = pos_w + v_ref_w * v_max * dt`` and drives the
  cascade controller with ``(ref_pos, ref_vel, ref_acc=0)``.  Total env
  action dimension is ``3 * num_drones``.

* **Per-drone observation including neighbour inputs.** ``get_per_drone_obs()``
  returns ``(num_envs, num_drones, per_drone_obs_dim)``.  Each slice contains
  the drone's own body-frame state + goal + previous v_ref + sinusoidal
  episode-progress embedding, and for every neighbour ``j``: its body-frame
  relative position, relative velocity, and previous v_ref.

* **Decentralised policy ready.** The flat observation in ``"policy"`` is a
  concatenation of the per-drone slices in fixed drone order.  The BC script
  reshapes it to ``(num_envs * num_drones, per_drone_obs_dim)`` and applies a
  *single shared MLP* to every drone in parallel.
"""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, matrix_from_quat

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG, SPHERE_MARKER_CFG  # isort: skip


# Per-drone observation layout sizes.
HISTORY_STEPS = 5        # own position history length (steps)
TIME_EMB_DIM = 4         # sinusoidal episode-progress embedding
PER_DRONE_OWN_DIM = 37   # goal_rel_w(3) + lin_vel_w(3) + R_wb(9) + ang_vel_b(3)
                          # + past_rel_pos_w(HISTORY_STEPS*3) + time_emb(TIME_EMB_DIM)
PER_NEIGHBOUR_DIM = 9    # rel_pos_w(3) + rel_vel_w(3) + neigh_goal_dir_w(3)


def per_drone_obs_dim(num_drones: int) -> int:
    return PER_DRONE_OWN_DIM + PER_NEIGHBOUR_DIM * (num_drones - 1)


class MultiDroneDmpcEnvWindow(BaseEnvWindow):
    """Window manager for the multi-drone DMPC environment."""

    def __init__(self, env: "MultiDroneDmpcEnv", window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class MultiDroneDmpcEnvCfg(DirectRLEnvCfg):
    # ── env ──
    num_drones: int = 4
    episode_length_s: float = 10.0
    decimation: int = 5
    action_space: int = 3 * 4   # 3 (v_ref) per drone, overwritten in __post_init__
    observation_space: int = 4 * 39  # overwritten in __post_init__
    state_space: int = 0
    debug_vis: bool = True
    debug_short_horizon_steps: int = 3
    randomize_episode_start: bool = False
    terminate_on_bounds: bool = True
    collision_free_two_drone_reset: bool = False # For DMPC debugging
    success_dist_threshold: float = 0.05   # metres; all drones must be within this
    success_hold_s: float = 1.0            # seconds all drones must hold the threshold
    goal_blend_radius: float = 0.5         # metres; blend ref_pos→goal within this radius

    ui_window_class_type = MultiDroneDmpcEnvWindow

    # ── simulation ──
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ── scene ──
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=6.0, replicate_physics=True, clone_in_fabric=True
    )

    # ── drone dynamics ──
    robot_template: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Drone_{idx}"
    )
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01

    # ── DMPC workspace bounds (xyz). Matches cpp/config/config.json. ──
    pos_min: tuple[float, float, float] = (-1.5, -1.5, 0.2)
    pos_max: tuple[float, float, float] = (1.5, 1.5, 2.2)
    rmin: float = 0.3
    # Action normalisation for the reference-command interface. The per-drone
    # action is [delta_p_w, v_ref_w, a_ref_w] in normalised [-1, 1] units.
    delta_pos_max: float = 0.3
    v_max: float = 2.0
    accel_action_max: float = 1.5  # matches DMPC QP hard constraint: amin/amax = ±1.5 m/s²

    # ── position-reference tracker gains ──
    pos_track_kp: float = 8.0 #5.0 #6.0
    pos_track_kd: float = 16.0 #4.5
    track_accel_clip: float = 4.0 # 4.0 #(of no use now)
    att_track_kp: float = 0.01
    att_track_kd: float = 0.0008

    # ── reward scales ──
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 15.0
    collision_reward_scale: float = -50.0
    z_min: float = 0.1
    z_max: float = 2.5

    def __post_init__(self):
        self.action_space = 3 * self.num_drones
        self.observation_space = self.num_drones * per_drone_obs_dim(self.num_drones)


class MultiDroneDmpcEnv(DirectRLEnv):
    cfg: MultiDroneDmpcEnvCfg

    def __init__(self, cfg: MultiDroneDmpcEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.N = self.cfg.num_drones
        self.per_drone_obs_dim = per_drone_obs_dim(self.N)
        device = self.device

        # Per-drone action buffer (v_ref 3D or wrench 4D depending on call site).
        self._actions = torch.zeros(self.num_envs, self.N, 3, device=device)
        # Position history for own-motion feature: (E, N, HISTORY_STEPS, 3), world frame.
        self._pos_history = torch.zeros(self.num_envs, self.N, HISTORY_STEPS, 3, device=device)
        # Last valid goal-aligned rotation matrix per drone, used as fallback when the
        # drone is horizontally co-located with its goal (degenerate frame).
        self._prev_R = torch.eye(3, device=device) \
            .unsqueeze(0).unsqueeze(0) \
            .expand(self.num_envs, self.N, 3, 3).clone()
        self._last_ref_pos_w = torch.zeros(self.num_envs, self.N, 3, device=device)
        self._last_ref_vel_w = torch.zeros(self.num_envs, self.N, 3, device=device)
        self._last_ref_acc_w = torch.zeros(self.num_envs, self.N, 3, device=device)
        # Per-drone thrust / moment buffers ultimately applied to PhysX.
        self._thrust = torch.zeros(self.num_envs, self.N, 1, 3, device=device)
        self._moment = torch.zeros(self.num_envs, self.N, 1, 3, device=device)

        # Per-drone goal positions in world frame.
        self._goal_pos_w = torch.zeros(self.num_envs, self.N, 3, device=device)
        # Initial positions saved at reset (handy for diagnostics).
        self._init_pos_w = torch.zeros(self.num_envs, self.N, 3, device=device)

        # Per-drone collision radius used in the neighbour observation.
        # Defaults to cfg.rmin for all drones; set individual entries to model
        # obstacles as virtual agents with a larger radius.
        self._drone_rmin = torch.full((self.N,), self.cfg.rmin, device=device)

        # One-shot override for _reset_idx: maps env_id (int) →
        # {"init_pos": (N,3), "goal": (N,3)} in WORLD frame.
        # Set before a reset call; consumed (popped) inside _reset_idx.
        self._pinned_reset_state: dict[int, dict[str, torch.Tensor]] = {}

        # Consecutive steps all drones have been within success_dist_threshold.
        self._success_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        # Set to True for envs that triggered success termination this step.
        # Cleared at the start of each _get_dones call; safe to read after env.step() returns.
        self._just_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        # Per-drone success tracking: consecutive steps each individual drone has held threshold.
        self._drone_success_steps = torch.zeros(self.num_envs, self.N, dtype=torch.long, device=device)
        # Per-drone flag: True for drone n in env e if that drone individually held threshold.
        self._drone_just_succeeded = torch.zeros(self.num_envs, self.N, dtype=torch.bool, device=device)

        # First-env live DMPC horizon visualization buffers. Shapes are
        # (num_drones, horizon, 3); empty until the expert pushes a plan.
        self._debug_planned_pos_w = torch.empty(0, self.N, 3, device=device)
        self._debug_predicted_pos_w = torch.empty(0, self.N, 3, device=device)
        self._debug_planned_short_pos_w = torch.empty(0, self.N, 3, device=device)
        self._debug_predicted_short_pos_w = torch.empty(0, self.N, 3, device=device)
        self._debug_planned_segment_pos_w = []
        self._debug_predicted_segment_pos_w = []
        self._debug_collision_pos_w = torch.empty(0, 3, device=device)
        self._last_reset_env_ids = torch.empty(0, dtype=torch.long, device=device)

        # Logging.
        self._episode_sums = {
            key: torch.zeros(self.num_envs, device=device)
            for key in ("lin_vel", "ang_vel", "distance_to_goal", "collision")
        }

        # Robot mass / gravity.
        self._body_id = self._robots[0].find_bodies("body")[0]
        robot_mass = self._robots[0].root_physx_view.get_masses()[0].sum()
        self._robot_mass = float(robot_mass)
        gravity = torch.tensor(self.sim.cfg.gravity, device=device).norm()
        self._gravity_magnitude = float(gravity)
        self._robot_weight = self._robot_mass * self._gravity_magnitude

        self._pos_min = torch.tensor(self.cfg.pos_min, device=device)
        self._pos_max = torch.tensor(self.cfg.pos_max, device=device)

        self.set_debug_vis(self.cfg.debug_vis)

    # ── scene setup ────────────────────────────────────────────────────────
    def _setup_scene(self):
        self._robots: list[Articulation] = []
        for i in range(self.cfg.num_drones):
            robot_cfg = self.cfg.robot_template.replace(
                prim_path=self.cfg.robot_template.prim_path.format(idx=i)
            )
            robot = Articulation(robot_cfg)
            self._robots.append(robot)
            self.scene.articulations[f"drone_{i}"] = robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── physics step ───────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        """Dispatch to wrench-direct or v_ref→cascade path based on action size.

        * ``actions.shape[-1] == N * 4``: wrench mode — ``[f_z, τ_x, τ_y, τ_z]``
          per drone (body frame) applied directly, bypassing the cascade.
        * ``actions.shape[-1] == N * 3``: v_ref mode — normalised velocity
          reference in goal-aligned frame, converted via the cascade controller.
        """
        E, N = self.num_envs, self.N

        if actions.shape[-1] == N * 4:
            # ── Wrench-direct mode ────────────────────────────────────────
            w = actions.view(E, N, 4)
            self._actions = w.clone()
            self._thrust[:, :, 0, 2] = w[:, :, 0]    # body-frame thrust (z)
            self._moment[:, :, 0, :] = w[:, :, 1:]   # body-frame torques
            self._last_low_level_debug = {
                "thrust_b_z": w[:, :, 0].detach().clone(),
                "tau_des_b":  w[:, :, 1:].detach().clone(),
            }
            return

        # ── v_ref mode (cascade) ──────────────────────────────────────────
        a = actions.view(E, N, 3).clamp(-1.0, 1.0)
        self._actions = a.clone()

        st = self._stack_drone_state()
        pos_w = st["pos_w"]

        # v_ref is expressed in each drone's goal-aligned frame → rotate to world frame.
        R = self._compute_goal_aligned_R(pos_w, self._goal_pos_w)   # (E, N, 3, 3)
        self._prev_R = R.detach()  # keep fallback updated for next call
        v_ref_goal = a * self.cfg.v_max                               # (E, N, 3)
        ref_vel_w = torch.bmm(
            R.transpose(-1, -2).reshape(E * N, 3, 3),
            v_ref_goal.reshape(E * N, 3, 1),
        ).reshape(E, N, 3)

        # Integrate v_ref: advance reference from last reference position.
        ref_pos_w = self._last_ref_pos_w + ref_vel_w * self.step_dt

        ref_acc_w = torch.zeros_like(ref_vel_w)

        self._last_ref_pos_w = ref_pos_w.detach().clone()
        self._last_ref_vel_w = ref_vel_w.detach().clone()
        self._last_ref_acc_w = ref_acc_w.detach().clone()

        wrench_b = self._ref_to_thrust_moment(ref_pos_w, ref_vel_w, ref_acc_w)
        self._thrust[:, :, 0, 2] = wrench_b[:, :, 0]
        self._moment[:, :, 0, :] = wrench_b[:, :, 1:]

    def _apply_action(self):
        for i, robot in enumerate(self._robots):
            robot.set_external_force_and_torque(
                self._thrust[:, i], self._moment[:, i], body_ids=self._body_id, is_global=False
            )

    # ── observation helpers ────────────────────────────────────────────────
    def _stack_drone_state(self) -> dict[str, torch.Tensor]:
        return {
            "pos_w": torch.stack([r.data.root_pos_w for r in self._robots], dim=1),
            "quat_w": torch.stack([r.data.root_quat_w for r in self._robots], dim=1),
            "lin_vel_w": torch.stack([r.data.root_lin_vel_w for r in self._robots], dim=1),
            "lin_vel_b": torch.stack([r.data.root_lin_vel_b for r in self._robots], dim=1),
            "ang_vel_b": torch.stack([r.data.root_ang_vel_b for r in self._robots], dim=1),
            "ang_vel_w": torch.stack([r.data.root_ang_vel_w for r in self._robots], dim=1),
            "proj_gravity_b": torch.stack([r.data.projected_gravity_b for r in self._robots], dim=1),
        }

    def _compute_goal_aligned_R(
        self, pos_w: torch.Tensor, goal_w: torch.Tensor
    ) -> torch.Tensor:
        """Per-drone goal-aligned rotation matrix: world → goal frame.

        Frame definition:
          x = horizontal direction toward goal (xy-plane projection)
          z = global +z (up)
          y = z × x  (right-hand rule)

        Falls back to ``self._prev_R`` per (env, drone) when the drone's XY
        position coincides with the goal's XY (degenerate: no horizontal
        component to define x).  Does NOT update ``_prev_R`` — callers are
        responsible for that.

        Args:
            pos_w:  ``(E, N, 3)`` drone positions in world frame.
            goal_w: ``(E, N, 3)`` goal positions in world frame.

        Returns:
            R: ``(E, N, 3, 3)`` — apply as ``v_frame = R @ v_world``.
        """
        delta = goal_w - pos_w            # (E, N, 3)
        dx = delta[..., 0]                # (E, N)
        dy = delta[..., 1]
        horiz_dist = (dx * dx + dy * dy).sqrt()  # (E, N)

        safe = horiz_dist.clamp(min=1e-4)
        cos_t = dx / safe                 # (E, N)
        sin_t = dy / safe

        zeros = torch.zeros_like(cos_t)
        ones  = torch.ones_like(cos_t)

        # R rows: x_frame, y_frame, z_frame expressed in world coords.
        # v_frame = R @ v_world
        #   R = [[ cos,  sin, 0],
        #        [-sin,  cos, 0],
        #        [   0,    0, 1]]
        R = torch.stack(
            [cos_t, sin_t, zeros, -sin_t, cos_t, zeros, zeros, zeros, ones],
            dim=-1,
        ).view(*cos_t.shape, 3, 3)  # (E, N, 3, 3)

        # Fallback for degenerate cases (drone directly above/below goal).
        deg = (horiz_dist < 1e-3).unsqueeze(-1).unsqueeze(-1)  # (E, N, 1, 1)
        return torch.where(deg, self._prev_R, R)

    def get_per_drone_obs(self) -> torch.Tensor:
        """Build the per-drone observation tensor in world frame.

        Returns:
            ``(E, N, per_drone_obs_dim)`` layout for drone ``i``:

            * **0-2**   goal position relative to drone, world frame
            * **3-5**   linear velocity, world frame
            * **6-14**  rotation matrix body→world, row-major flattened (R_wb)
            * **15-17** angular velocity, body frame
            * **18-32** past 5-step own positions relative to current, world frame
            * **33-36** sinusoidal episode-progress embedding (TIME_EMB_DIM=4)
            * For each neighbour ``j``:
              * relative position world frame (3)
              * relative velocity world frame (3)
              * unit direction from neighbour to its goal, world frame (3)
        """
        E, N = self.num_envs, self.N
        st = self._stack_drone_state()
        pos_w     = st["pos_w"]       # (E, N, 3)
        quat_w    = st["quat_w"]      # (E, N, 4)
        lin_vel_w = st["lin_vel_w"]   # (E, N, 3)
        ang_vel_b = st["ang_vel_b"]   # (E, N, 3)

        rot_wb    = matrix_from_quat(quat_w)          # (E, N, 3, 3)
        R_wb_flat = rot_wb.reshape(E, N, 9)           # (E, N, 9) row-major

        # Episode-progress sinusoidal embedding (E, TIME_EMB_DIM).
        t = (self.episode_length_buf.float() / self.max_episode_length).clamp(0.0, 1.0)
        t_emb = torch.stack([
            torch.sin(torch.pi * t),
            torch.cos(torch.pi * t),
            torch.sin(2.0 * torch.pi * t),
            torch.cos(2.0 * torch.pi * t),
        ], dim=-1)  # (E, 4)

        out = torch.empty((E, N, self.per_drone_obs_dim), device=self.device, dtype=pos_w.dtype)

        for i in range(N):
            # Own features (all world frame except ang_vel_b).
            goal_rel_w  = self._goal_pos_w[:, i] - pos_w[:, i]  # (E, 3)
            lin_vel_w_i = lin_vel_w[:, i]                        # (E, 3)
            R_wb_flat_i = R_wb_flat[:, i]                        # (E, 9)
            ang_vel_b_i = ang_vel_b[:, i]                        # (E, 3)

            # Past HISTORY_STEPS positions relative to current, world frame.
            past_feats = []
            for k in range(HISTORY_STEPS):
                past_feats.append(self._pos_history[:, i, k] - pos_w[:, i])  # (E, 3)
            past_rel_w = torch.cat(past_feats, dim=-1)  # (E, HISTORY_STEPS*3)

            own = torch.cat(
                [goal_rel_w, lin_vel_w_i, R_wb_flat_i, ang_vel_b_i, past_rel_w, t_emb],
                dim=-1,
            )  # (E, 37)
            out[:, i, :PER_DRONE_OWN_DIM] = own

            # Neighbour features (world frame).
            offset = PER_DRONE_OWN_DIM
            for j in range(N):
                if j == i:
                    continue
                rel_pos_w = pos_w[:, j] - pos_w[:, i]              # (E, 3)
                rel_vel_w = lin_vel_w[:, j] - lin_vel_w[:, i]      # (E, 3)
                ng_delta  = self._goal_pos_w[:, j] - pos_w[:, j]   # (E, 3)
                ng_dir_w  = ng_delta / ng_delta.norm(dim=-1, keepdim=True).clamp(min=1e-4)
                out[:, i, offset     : offset + 3] = rel_pos_w
                out[:, i, offset + 3 : offset + 6] = rel_vel_w
                out[:, i, offset + 6 : offset + 9] = ng_dir_w
                offset += PER_NEIGHBOUR_DIM  # 9

        # Shift position history and record current positions.
        self._pos_history[:, :, 1:, :] = self._pos_history[:, :, :-1, :].clone()
        self._pos_history[:, :, 0, :] = pos_w.detach()

        return out

    def _get_observations(self) -> dict:
        per_drone = self.get_per_drone_obs()
        flat = per_drone.reshape(self.num_envs, -1)
        return {"policy": flat}

    # ── rewards / dones ────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        st = self._stack_drone_state()
        pos_w = st["pos_w"]
        lin_vel_b = st["lin_vel_b"]
        ang_vel_b = st["ang_vel_b"]

        dist = torch.linalg.norm(self._goal_pos_w - pos_w, dim=-1)
        dist_mapped = (1.0 - torch.tanh(dist / 0.8)).sum(dim=-1)

        lin_vel_pen = (lin_vel_b.square().sum(dim=-1)).sum(dim=-1)
        ang_vel_pen = (ang_vel_b.square().sum(dim=-1)).sum(dim=-1)

        if self.N > 1:
            diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
            pair_dist = torch.linalg.norm(diff, dim=-1)
            eye = torch.eye(self.N, device=self.device, dtype=torch.bool)
            pair_dist = pair_dist.masked_fill(eye, float("inf"))
            min_pair = pair_dist.amin(dim=(1, 2))
            collision_pen = torch.clamp(self.cfg.rmin - min_pair, min=0.0)
        else:
            collision_pen = torch.zeros(self.num_envs, device=self.device)

        rewards = {
            "lin_vel": lin_vel_pen * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel_pen * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": dist_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "collision": collision_pen * self.cfg.collision_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for k, v in rewards.items():
            self._episode_sums[k] += v
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._just_succeeded[:] = False  # clear previous step's flag
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pos_w = torch.stack([r.data.root_pos_w for r in self._robots], dim=1)
        z = pos_w[..., 2]
        if self.cfg.terminate_on_bounds:
            died = ((z < self.cfg.z_min) | (z > self.cfg.z_max)).any(dim=-1)
        else:
            died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Success termination: all drones within threshold for hold duration.
        dist = torch.linalg.norm(pos_w - self._goal_pos_w, dim=-1)  # (E, N)
        all_close = (dist < self.cfg.success_dist_threshold).all(dim=-1)  # (E,)
        self._success_steps[all_close] += 1
        self._success_steps[~all_close] = 0
        hold_steps = max(1, round(self.cfg.success_hold_s / self.step_dt))
        succeeded = self._success_steps >= hold_steps
        # Expose for the collection loop to read after env.step() returns.
        # NOT cleared in _reset_idx — next _get_dones call clears it at entry.
        self._just_succeeded[:] = succeeded

        # Per-drone success: each drone independently tracked regardless of others.
        per_close = dist < self.cfg.success_dist_threshold  # (E, N)
        self._drone_success_steps[per_close] += 1
        self._drone_success_steps[~per_close] = 0
        self._drone_just_succeeded[:] = self._drone_success_steps >= hold_steps

        return died | succeeded, time_out

    # ── resets ─────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES
        self._last_reset_env_ids = env_ids.detach().clone()

        pos_w = torch.stack([r.data.root_pos_w for r in self._robots], dim=1)
        final_dist = torch.linalg.norm(self._goal_pos_w[env_ids] - pos_w[env_ids], dim=-1).mean()
        extras = dict()
        for k in self._episode_sums:
            extras["Episode_Reward/" + k] = (
                self._episode_sums[k][env_ids].mean() / self.max_episode_length_s
            )
            self._episode_sums[k][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        self.extras["log"].update(
            {
                "Episode_Termination/died": torch.count_nonzero(self.reset_terminated[env_ids]).item(),
                "Episode_Termination/time_out": torch.count_nonzero(self.reset_time_outs[env_ids]).item(),
                "Metrics/final_distance_to_goal": final_dist.item(),
            }
        )

        for robot in self._robots:
            robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and self.cfg.randomize_episode_start:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._last_ref_vel_w[env_ids] = 0.0
        self._last_ref_acc_w[env_ids] = 0.0
        self._success_steps[env_ids] = 0
        self._drone_success_steps[env_ids] = 0
        self._drone_just_succeeded[env_ids] = False
        self._prev_R[env_ids] = torch.eye(3, device=self.device)

        # Random-exchange scenario (DMPC paper Section IV/VI): each drone on
        # one side of a random circle, goal on the diametrically opposite side.
        n = len(env_ids)
        device = self.device
        if self.N == 2 and self.cfg.collision_free_two_drone_reset:
            init_xy = torch.tensor([[-1.0, -0.55], [-1.0, 0.55]], device=device).unsqueeze(0).repeat(n, 1, 1)
            goal_xy = torch.tensor([[1.0, -0.55], [1.0, 0.55]], device=device).unsqueeze(0).repeat(n, 1, 1)
            z0 = torch.ones(n, self.N, device=device)
            zg = torch.ones(n, self.N, device=device)
        else:
            radius = torch.empty(n, self.N, device=device).uniform_(0.8, 1.3)
            theta_base = torch.empty(n, 1, device=device).uniform_(0.0, 2.0 * torch.pi)
            theta_offsets = 2.0 * torch.pi * torch.arange(self.N, device=device).float().unsqueeze(0) / self.N
            theta0 = theta_base + theta_offsets
            z0 = torch.empty(n, self.N, device=device).uniform_(0.6, 1.4)

            init_xy = torch.stack([radius * torch.cos(theta0), radius * torch.sin(theta0)], dim=-1)
            goal_xy = -init_xy
            zg = torch.empty(n, self.N, device=device).uniform_(0.6, 1.4)

        origins = self._terrain.env_origins[env_ids]
        init_pos = torch.cat([init_xy, z0.unsqueeze(-1)], dim=-1)
        init_pos[..., :2] += origins[:, None, :2]
        goal_pos = torch.cat([goal_xy, zg.unsqueeze(-1)], dim=-1)
        goal_pos[..., :2] += origins[:, None, :2]

        # One-shot override: replay mode can pin specific envs to exact positions.
        if self._pinned_reset_state:
            for local_i, env_id in enumerate(env_ids.tolist()):
                if env_id in self._pinned_reset_state:
                    pin = self._pinned_reset_state.pop(env_id)
                    init_pos[local_i] = pin["init_pos"].to(device)
                    goal_pos[local_i] = pin["goal"].to(device)

        self._goal_pos_w[env_ids] = goal_pos
        self._init_pos_w[env_ids] = init_pos
        self._last_ref_pos_w[env_ids] = init_pos  # seed integration at drone start position
        # Fill history with init_pos so first-step relative displacements start at zero.
        self._pos_history[env_ids] = init_pos.unsqueeze(2).expand(n, self.N, HISTORY_STEPS, 3)

        for i, robot in enumerate(self._robots):
            default_root = robot.data.default_root_state[env_ids].clone()
            default_root[:, :3] = init_pos[:, i]
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            # joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
            robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ── helpers exposed for the DMPC expert ────────────────────────────────
    def get_world_states(self) -> dict[str, torch.Tensor]:
        """World-frame states + goals used by :class:`DMPCExpert`."""
        st = self._stack_drone_state()
        st["goal_w"] = self._goal_pos_w
        st["last_ref_pos_w"] = self._last_ref_pos_w
        st["last_ref_vel_w"] = self._last_ref_vel_w
        st["last_ref_acc_w"] = self._last_ref_acc_w
        return st

    def reference_to_action(
        self,
        ref_pos_w: torch.Tensor,
        ref_vel_w: torch.Tensor | None = None,
        _ref_acc_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pack world-frame velocity reference into normalised 3-D action.

        Args:
            ref_pos_w: ``(num_envs, num_drones, 3)`` desired positions (unused,
                kept for API compatibility with the DMPC expert).
            ref_vel_w: desired velocities in m/s; zeros if omitted.
            _ref_acc_w: ignored (kept for API compatibility).

        Returns:
            ``(num_envs, num_drones * 3)`` normalised v_ref in ``[-1, 1]``.
        """
        if ref_vel_w is None:
            ref_vel_w = torch.zeros_like(ref_pos_w)
        E, N = ref_pos_w.shape[0], ref_pos_w.shape[1]
        pos_w = self._stack_drone_state()["pos_w"]
        R = self._compute_goal_aligned_R(pos_w, self._goal_pos_w)  # (E, N, 3, 3)
        # Rotate world-frame v_ref into each drone's goal-aligned frame.
        v_ref_goal = torch.bmm(
            R.reshape(E * N, 3, 3),
            ref_vel_w.reshape(E * N, 3, 1),
        ).reshape(E, N, 3)
        vel_norm = (v_ref_goal / max(self.cfg.v_max, 1e-6)).clamp(-1.0, 1.0)
        return vel_norm.reshape(E, N * 3)

    def velocity_to_action(self, v_cmd_w: torch.Tensor) -> torch.Tensor:
        """Compatibility helper: pack velocity-only commands with zero acceleration."""
        st = self._stack_drone_state()
        ref_pos_w = st["pos_w"] + v_cmd_w * self.step_dt
        return self.reference_to_action(ref_pos_w, v_cmd_w, torch.zeros_like(v_cmd_w))

    def ref_to_action(
        self,
        ref_pos_w: torch.Tensor,
        ref_vel_w: torch.Tensor | None = None,
        ref_acc_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for :meth:`reference_to_action` used by the DMPC expert."""
        return self.reference_to_action(ref_pos_w, ref_vel_w, ref_acc_w)

    def vref_to_action(self, ref_vel_w: torch.Tensor) -> torch.Tensor:
        """Rotate world-frame velocity reference into goal-aligned frame and normalise.

        Returns ``(num_envs, num_drones * 3)`` in ``[-1, 1]`` — the 3-D action
        consumed by the v_ref cascade path in :meth:`_pre_physics_step`.
        """
        st = self._stack_drone_state()
        pos_w = st["pos_w"]
        E, N = pos_w.shape[0], pos_w.shape[1]
        R = self._compute_goal_aligned_R(pos_w, self._goal_pos_w)   # (E, N, 3, 3)
        # world → goal frame: v_goal = R @ ref_vel_w
        v_goal = torch.bmm(
            R.reshape(E * N, 3, 3),
            ref_vel_w.reshape(E * N, 3, 1),
        ).reshape(E, N, 3)
        vel_norm = (v_goal / max(self.cfg.v_max, 1e-6)).clamp(-1.0, 1.0)
        return vel_norm.reshape(E, N * 3)

    # ── internal: cascaded position controller -> thrust/moment ────────────
    def _ref_to_thrust_moment(
        self,
        ref_pos_w: torch.Tensor,
        ref_vel_w: torch.Tensor | None = None,
        ref_acc_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cascaded P-controller (paper Fig. 2 inner loop) + differential
        flatness. Returns physical body-frame ``[force_z_N, tau_x, tau_y, tau_z]``."""
        st = self._stack_drone_state()
        pos_w = st["pos_w"]
        vel_w = st["lin_vel_w"]
        quat_w = st["quat_w"]
        angvel_b = st["ang_vel_b"]

        if ref_vel_w is None:
            ref_vel_w = torch.zeros_like(ref_pos_w)
        if ref_acc_w is None:
            ref_acc_w = torch.zeros_like(ref_pos_w)
        rot_wb = matrix_from_quat(quat_w)  # (E, N, 3, 3)

        # yaw_des = torch.atan2(ref_vel_w[..., 1], ref_vel_w[..., 0])
        yaw_des = torch.zeros_like(ref_vel_w[..., 0])
    
        # Simplified geometric controller (Mellinger & Kumar 2011)
        # thrust
        acc_cmd_w = (
            self.cfg.pos_track_kp * (ref_pos_w - pos_w)
            + self.cfg.pos_track_kd * (ref_vel_w - vel_w)
            + ref_acc_w
        )
        F_des_w = self._robot_mass * (
            acc_cmd_w + self._gravity_magnitude * torch.tensor([0.0, 0.0, 1.0], device=self.device)
        ) # (E, N, 3)
        z_wb = rot_wb[:, :, :, 2]  # (E, N, 3)
        f = torch.einsum("...i,...i->...", F_des_w, z_wb)  # (E, N), desired thrust magnitude
        # max_thrust = self.cfg.thrust_to_weight * self._robot_weight
        # f = f.clamp(0.0, max_thrust)
        # Temporarily leave thrust unclipped while debugging the ideal wrench controller.

        # attitude
        z_des = F_des_w / torch.norm(F_des_w, dim=-1, keepdim=True).clamp(min=1e-6)  # (E, N, 3)
        x_c = torch.stack([torch.cos(yaw_des), torch.sin(yaw_des), torch.zeros_like(yaw_des)], dim=-1)  # (E, N, 3)
        y_c = torch.cross(z_des, x_c, dim=-1)  # (E, N, 3)
        y_des = y_c / torch.norm(y_c, dim=-1, keepdim=True).clamp(min=1e-6)  # (E, N, 3)
        x_des = torch.cross(y_des, z_des, dim=-1)  # (E, N, 3)
        rot_des = torch.stack([x_des, y_des, z_des], dim=-1)  # (E, N, 3, 3)

        def vee(S: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                [S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]],
                dim=-1,
            )

        e_R = vee(0.5 * (torch.einsum("...ij,...ik->...jk", rot_des, rot_wb) - torch.einsum("...ij,...ik->...jk", rot_wb, rot_des))) # (E, N, 3)
        e_w = angvel_b # set desired angular velocity to zero for now

        tau_des_b = -self.cfg.att_track_kp * e_R - self.cfg.att_track_kd * e_w  # (E, N, 3)
        # tau_des_b = - self.cfg.att_track_kp * e_R
        # tau_des_b = torch.zeros_like(e_R)  # placeholder while we iterate on the controller implementation
        self._last_low_level_debug = {
            "acc_cmd_w": acc_cmd_w.detach().clone(),
            "F_des_w": F_des_w.detach().clone(),
            "thrust_b_z": f.detach().clone(),
            "z_wb": z_wb.detach().clone(),
            "z_des": z_des.detach().clone(),
            "e_R": e_R.detach().clone(),
            "angvel_b": angvel_b.detach().clone(),
            "tau_des_b": tau_des_b.detach().clone(),
        }
        return torch.cat([f.unsqueeze(-1), tau_des_b], dim=-1)  # (E, N, 4)

        # #########

        # # ref_acc_w is intentionally accepted but not yet used here; this is
        # # the hook for the physical wrench controller update.
        # desired_accel = (
        #     self.cfg.pos_track_kp * (ref_pos_w - pos_w)
        #     + self.cfg.pos_track_kd * (ref_vel_w - vel_w)
        # )
        # clip = self.cfg.track_accel_clip
        # desired_accel = desired_accel.clamp(-clip, clip)
        # return _acc_to_thrust_moment_action(self, desired_accel)

    # ── live MPC debug visualization ───────────────────────────────────────
    def set_debug_trajectories(
        self,
        planned_pos_w: torch.Tensor | None = None,
        predicted_pos_w: torch.Tensor | None = None,
        planned_short_pos_w: torch.Tensor | None = None,
        predicted_short_pos_w: torch.Tensor | None = None,
        collision_pos_w: torch.Tensor | None = None,
        planned_segment_pos_w: list[torch.Tensor] | None = None,
        predicted_segment_pos_w: list[torch.Tensor] | None = None,
    ) -> None:
        """Update first-env MPC horizon markers.

        Args:
            planned_pos_w: ``(N, K, 3)`` planned reference positions.
            predicted_pos_w: ``(N, K, 3)`` predicted state positions.
            planned_short_pos_w: early planned reference samples to color separately.
            predicted_short_pos_w: early predicted state samples to color separately.
            collision_pos_w: ``(M, 3)`` predicted collision points.
            planned_segment_pos_w: planned-reference samples grouped by Bezier segment.
            predicted_segment_pos_w: predicted-state samples grouped by Bezier segment.
        """
        if planned_pos_w is None:
            self._debug_planned_pos_w = torch.empty(0, self.N, 3, device=self.device)
        else:
            self._debug_planned_pos_w = planned_pos_w.detach().to(self.device).clone()
        if predicted_pos_w is None:
            self._debug_predicted_pos_w = torch.empty(0, self.N, 3, device=self.device)
        else:
            self._debug_predicted_pos_w = predicted_pos_w.detach().to(self.device).clone()
        if planned_short_pos_w is None:
            self._debug_planned_short_pos_w = torch.empty(0, self.N, 3, device=self.device)
        else:
            self._debug_planned_short_pos_w = planned_short_pos_w.detach().to(self.device).clone()
        if predicted_short_pos_w is None:
            self._debug_predicted_short_pos_w = torch.empty(0, self.N, 3, device=self.device)
        else:
            self._debug_predicted_short_pos_w = predicted_short_pos_w.detach().to(self.device).clone()
        if planned_segment_pos_w is None:
            self._debug_planned_segment_pos_w = []
        else:
            self._debug_planned_segment_pos_w = [p.detach().to(self.device).clone() for p in planned_segment_pos_w]
        if predicted_segment_pos_w is None:
            self._debug_predicted_segment_pos_w = []
        else:
            self._debug_predicted_segment_pos_w = [p.detach().to(self.device).clone() for p in predicted_segment_pos_w]
        if collision_pos_w is None:
            self._debug_collision_pos_w = torch.empty(0, 3, device=self.device)
        else:
            self._debug_collision_pos_w = collision_pos_w.detach().to(self.device).clone()

    # ── debug viz ──────────────────────────────────────────────────────────
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/multi_drone_goal"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
            if not hasattr(self, "mpc_plan_visualizer"):
                plan_cfg = SPHERE_MARKER_CFG.copy()
                plan_cfg.markers["sphere"].radius = 0.01
                plan_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.55, 0.0)
                plan_cfg.prim_path = "/Visuals/Command/multi_drone_mpc_plan"
                self.mpc_plan_visualizer = VisualizationMarkers(plan_cfg)
            if not hasattr(self, "mpc_pred_visualizer"):
                pred_cfg = SPHERE_MARKER_CFG.copy()
                pred_cfg.markers["sphere"].radius = 0.008
                pred_cfg.markers["sphere"].visual_material.diffuse_color = (0.55, 0.25, 1.0)
                pred_cfg.prim_path = "/Visuals/Command/multi_drone_mpc_prediction"
                self.mpc_pred_visualizer = VisualizationMarkers(pred_cfg)
            if not hasattr(self, "mpc_plan_segment_visualizers"):
                plan_colors = [(0.0, 0.9, 1.0), (1.0, 0.85, 0.0), (1.0, 0.2, 0.75)]
                pred_colors = [(0.1, 1.0, 0.25), (0.35, 0.65, 1.0), (0.85, 0.45, 1.0)]
                self.mpc_plan_segment_visualizers = []
                self.mpc_pred_segment_visualizers = []
                for seg_idx, color in enumerate(plan_colors):
                    seg_cfg = SPHERE_MARKER_CFG.copy()
                    seg_cfg.markers["sphere"].radius = 0.018
                    seg_cfg.markers["sphere"].visual_material.diffuse_color = color
                    seg_cfg.prim_path = f"/Visuals/Command/multi_drone_mpc_plan_segment_{seg_idx}"
                    self.mpc_plan_segment_visualizers.append(VisualizationMarkers(seg_cfg))
                for seg_idx, color in enumerate(pred_colors):
                    seg_cfg = SPHERE_MARKER_CFG.copy()
                    seg_cfg.markers["sphere"].radius = 0.014
                    seg_cfg.markers["sphere"].visual_material.diffuse_color = color
                    seg_cfg.prim_path = f"/Visuals/Command/multi_drone_mpc_prediction_segment_{seg_idx}"
                    self.mpc_pred_segment_visualizers.append(VisualizationMarkers(seg_cfg))
            self.mpc_plan_visualizer.set_visibility(True)
            self.mpc_pred_visualizer.set_visibility(True)
            for visualizer in self.mpc_plan_segment_visualizers:
                visualizer.set_visibility(True)
            for visualizer in self.mpc_pred_segment_visualizers:
                visualizer.set_visibility(True)
            if not hasattr(self, "mpc_collision_visualizer"):
                coll_cfg = SPHERE_MARKER_CFG.copy()
                coll_cfg.markers["sphere"].radius = 0.012
                coll_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
                coll_cfg.prim_path = "/Visuals/Command/multi_drone_mpc_collision"
                self.mpc_collision_visualizer = VisualizationMarkers(coll_cfg)
            self.mpc_collision_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "mpc_plan_visualizer"):
                self.mpc_plan_visualizer.set_visibility(False)
            if hasattr(self, "mpc_pred_visualizer"):
                self.mpc_pred_visualizer.set_visibility(False)
            if hasattr(self, "mpc_plan_segment_visualizers"):
                for visualizer in self.mpc_plan_segment_visualizers:
                    visualizer.set_visibility(False)
            if hasattr(self, "mpc_pred_segment_visualizers"):
                for visualizer in self.mpc_pred_segment_visualizers:
                    visualizer.set_visibility(False)
            if hasattr(self, "mpc_collision_visualizer"):
                self.mpc_collision_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._goal_pos_w.reshape(-1, 3))
        if hasattr(self, "mpc_plan_visualizer") and self._debug_planned_pos_w.numel() > 0:
            self.mpc_plan_visualizer.visualize(self._debug_planned_pos_w.reshape(-1, 3))
        if hasattr(self, "mpc_pred_visualizer") and self._debug_predicted_pos_w.numel() > 0:
            self.mpc_pred_visualizer.visualize(self._debug_predicted_pos_w.reshape(-1, 3))
        if hasattr(self, "mpc_plan_segment_visualizers"):
            for visualizer, points in zip(self.mpc_plan_segment_visualizers, self._debug_planned_segment_pos_w):
                if points.numel() > 0:
                    visualizer.visualize(points.reshape(-1, 3))
        if hasattr(self, "mpc_pred_segment_visualizers"):
            for visualizer, points in zip(self.mpc_pred_segment_visualizers, self._debug_predicted_segment_pos_w):
                if points.numel() > 0:
                    visualizer.visualize(points.reshape(-1, 3))
        if hasattr(self, "mpc_collision_visualizer") and self._debug_collision_pos_w.numel() > 0:
            self.mpc_collision_visualizer.visualize(self._debug_collision_pos_w.reshape(-1, 3))


# ───────────────────────────────────────────────────────────────────────────
# Low-level controller: desired world-frame acceleration → (E, N, 4) action
# ───────────────────────────────────────────────────────────────────────────
def _acc_to_thrust_moment_action(env: MultiDroneDmpcEnv, accel_w: torch.Tensor) -> torch.Tensor:
    """Differential-flatness mapping desired body-COM acceleration -> 4-D
    action (thrust normalised in ``[-1, 1]`` with 0 = hover, plus three
    normalised moment components)."""
    device = env.device
    E, N = env.num_envs, env.N
    g = env._gravity_magnitude
    mass = env._robot_mass

    g_world = torch.tensor([0.0, 0.0, g], device=device)
    a_total_w = accel_w + g_world[None, None, :]

    quat_w = torch.stack([r.data.root_quat_w for r in env._robots], dim=1)
    w, x, y, z = quat_w.unbind(dim=-1)
    body_z_w = torch.stack(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
        dim=-1,
    )

    thrust_mag = (mass * (a_total_w * body_z_w).sum(dim=-1)).clamp(min=0.0)
    max_thrust = env.cfg.thrust_to_weight * env._robot_weight
    a0 = (2.0 * thrust_mag / max(max_thrust, 1e-6) - 1.0).clamp(-1.0, 1.0)

    a_norm = torch.linalg.norm(a_total_w, dim=-1, keepdim=True).clamp(min=1e-4)
    b_z_des = a_total_w / a_norm
    err_w = torch.cross(body_z_w, b_z_des, dim=-1)

    ang_vel_b = torch.stack([r.data.root_ang_vel_b for r in env._robots], dim=1)
    err_b = quat_apply_inverse(quat_w.reshape(-1, 4), err_w.reshape(-1, 3)).reshape(E, N, 3)

    moment_cmd = env.cfg.att_track_kp * err_b - env.cfg.att_track_kd * ang_vel_b
    norm = (moment_cmd / max(env.cfg.moment_scale, 1e-6)).clamp(-1.0, 1.0)

    return torch.stack([a0, norm[..., 0], norm[..., 1], norm[..., 2]], dim=-1)
