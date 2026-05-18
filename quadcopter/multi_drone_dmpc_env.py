# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-drone DMPC environment.

Mirrors the layout of the standalone ``quadcopter_env.py`` (one Crazyflie, 4-D
thrust+moment action, goal-reaching) but instantiates *N* Crazyflies per
environment, exposes per-drone goal positions, and adds inter-drone collision
penalties so the workload matches the multi-robot motion-planning setting of
the C++ ``online_dmpc`` reference.

Action space per env: ``num_drones * 4`` — concatenation of each drone's
``[thrust_norm, mx_norm, my_norm, mz_norm]``.

Observation space per env: ``num_drones * (12 + 3*(num_drones-1))`` — for each
drone we concatenate its own ``[lin_vel_b, ang_vel_b, projected_gravity_b,
goal_pos_b]`` followed by the body-frame relative positions of every other
drone (fixed order). The student policy thus has access to the same information
the DMPC expert uses.
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
from isaaclab.utils.math import quat_rotate_inverse, subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


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
    decimation: int = 2
    # action / obs sizes filled in __post_init__ based on num_drones
    action_space: int = 4 * 4
    observation_space: int = 4 * (12 + 3 * 3)
    state_space: int = 0
    debug_vis: bool = True

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
    # Increase env_spacing so the per-env workspace (~3 m diameter, matching the
    # DMPC config [-1.5, 1.5] in xy) fits without neighbours from another env
    # bleeding into observations / collisions.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=6.0, replicate_physics=True, clone_in_fabric=True
    )

    # ── drone dynamics ──
    # Template ArticulationCfg used to spawn each of the ``num_drones`` drones.
    # The prim path placeholder ``{idx}`` is filled in at scene-build time.
    robot_template: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Drone_{idx}"
    )
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01

    # ── DMPC workspace bounds (xyz). Matches cpp/config/config.json. ──
    pos_min: tuple[float, float, float] = (-1.5, -1.5, 0.2)
    pos_max: tuple[float, float, float] = (1.5, 1.5, 2.2)
    # Inter-drone collision boundary (rmin from the DMPC paper).
    rmin: float = 0.3

    # ── reward scales ──
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 15.0
    collision_reward_scale: float = -50.0
    # Termination: clip when any drone leaves the workspace altitude band.
    z_min: float = 0.1
    z_max: float = 2.5

    def __post_init__(self):
        # Update action/obs space sizes once num_drones is finalised.
        self.action_space = 4 * self.num_drones
        self.observation_space = self.num_drones * (12 + 3 * (self.num_drones - 1))


class MultiDroneDmpcEnv(DirectRLEnv):
    cfg: MultiDroneDmpcEnvCfg

    def __init__(self, cfg: MultiDroneDmpcEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.N = self.cfg.num_drones
        device = self.device

        # Action / force buffers.
        self._actions = torch.zeros(self.num_envs, self.N, 4, device=device)
        self._thrust = torch.zeros(self.num_envs, self.N, 1, 3, device=device)
        self._moment = torch.zeros(self.num_envs, self.N, 1, 3, device=device)

        # Per-drone goal positions in world frame.
        self._goal_pos_w = torch.zeros(self.num_envs, self.N, 3, device=device)
        # Per-drone initial positions in world frame (saved for diagnostics + DMPC seeding).
        self._init_pos_w = torch.zeros(self.num_envs, self.N, 3, device=device)

        # Logging.
        self._episode_sums = {
            key: torch.zeros(self.num_envs, device=device)
            for key in ("lin_vel", "ang_vel", "distance_to_goal", "collision")
        }

        # Robot mass / weight (same per drone).
        self._body_id = self._robots[0].find_bodies("body")[0]
        robot_mass = self._robots[0].root_physx_view.get_masses()[0].sum()
        self._robot_mass = float(robot_mass)
        gravity = torch.tensor(self.sim.cfg.gravity, device=device).norm()
        self._gravity_magnitude = float(gravity)
        self._robot_weight = self._robot_mass * self._gravity_magnitude

        # Workspace bounds tensors for convenience.
        self._pos_min = torch.tensor(self.cfg.pos_min, device=device)
        self._pos_max = torch.tensor(self.cfg.pos_max, device=device)

        self.set_debug_vis(self.cfg.debug_vis)

    # ── scene setup ────────────────────────────────────────────────────────
    def _setup_scene(self):
        # Spawn ``num_drones`` Crazyflies per env. We deliberately use distinct
        # prim names so each behaves as an independent Articulation; the
        # InteractiveScene clone logic still replicates them per env.
        self._robots: list[Articulation] = []
        for i in range(self.cfg.num_drones):
            robot_cfg = self.cfg.robot_template.replace(
                prim_path=self.cfg.robot_template.prim_path.format(idx=i)
            )
            robot = Articulation(robot_cfg)
            self._robots.append(robot)
            self.scene.articulations[f"drone_{i}"] = robot

        # Terrain + clone.
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
        # ``actions`` arrives flattened as (num_envs, num_drones*4).
        a = actions.view(self.num_envs, self.N, 4).clamp(-1.0, 1.0)
        self._actions = a.clone()
        # Map normalised commands to per-drone thrust + moment vectors.
        self._thrust[:, :, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (a[:, :, 0] + 1.0) / 2.0
        )
        self._moment[:, :, 0, :] = self.cfg.moment_scale * a[:, :, 1:]

    def _apply_action(self):
        for i, robot in enumerate(self._robots):
            robot.set_external_force_and_torque(
                self._thrust[:, i], self._moment[:, i], body_ids=self._body_id
            )

    # ── observations / rewards ─────────────────────────────────────────────
    def _per_drone_state_world(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (pos_w, quat_w, lin_vel_w, ang_vel_w, lin_vel_b) each shape (E, N, 3 or 4)."""
        pos_w = torch.stack([r.data.root_pos_w for r in self._robots], dim=1)
        quat_w = torch.stack([r.data.root_quat_w for r in self._robots], dim=1)
        lin_vel_w = torch.stack([r.data.root_lin_vel_w for r in self._robots], dim=1)
        ang_vel_w = torch.stack([r.data.root_ang_vel_w for r in self._robots], dim=1)
        lin_vel_b = torch.stack([r.data.root_lin_vel_b for r in self._robots], dim=1)
        return pos_w, quat_w, lin_vel_w, ang_vel_w, lin_vel_b

    def _get_observations(self) -> dict:
        E, N = self.num_envs, self.N
        pos_w, quat_w, _, _, _ = self._per_drone_state_world()
        obs_per_drone = []
        for i in range(N):
            robot = self._robots[i]
            goal_b, _ = subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, self._goal_pos_w[:, i]
            )
            base = torch.cat(
                [
                    robot.data.root_lin_vel_b,
                    robot.data.root_ang_vel_b,
                    robot.data.projected_gravity_b,
                    goal_b,
                ],
                dim=-1,
            )  # (E, 12)
            # Body-frame relative position of every other drone.
            neighbour_rel = []
            for j in range(N):
                if j == i:
                    continue
                rel_w = pos_w[:, j] - robot.data.root_pos_w
                rel_b = quat_rotate_inverse(robot.data.root_quat_w, rel_w)
                neighbour_rel.append(rel_b)
            if neighbour_rel:
                base = torch.cat([base] + neighbour_rel, dim=-1)
            obs_per_drone.append(base)
        obs = torch.cat(obs_per_drone, dim=-1)  # (E, N*(12 + 3*(N-1)))
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pos_w, _, _, _, lin_vel_b = self._per_drone_state_world()
        ang_vel_b = torch.stack([r.data.root_ang_vel_b for r in self._robots], dim=1)

        # Goal-tracking, per drone, then summed.
        dist = torch.linalg.norm(self._goal_pos_w - pos_w, dim=-1)  # (E, N)
        dist_mapped = (1.0 - torch.tanh(dist / 0.8)).sum(dim=-1)

        lin_vel_pen = (lin_vel_b.square().sum(dim=-1)).sum(dim=-1)
        ang_vel_pen = (ang_vel_b.square().sum(dim=-1)).sum(dim=-1)

        # Pairwise collision penalty: counts close pairs (distance < rmin).
        if self.N > 1:
            diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)  # (E, N, N, 3)
            pair_dist = torch.linalg.norm(diff, dim=-1)
            eye = torch.eye(self.N, device=self.device, dtype=torch.bool)
            pair_dist = pair_dist.masked_fill(eye, float("inf"))
            min_pair = pair_dist.amin(dim=(1, 2))  # closest pair per env
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
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pos_w = torch.stack([r.data.root_pos_w for r in self._robots], dim=1)
        z = pos_w[..., 2]
        died = ((z < self.cfg.z_min) | (z > self.cfg.z_max)).any(dim=-1)
        return died, time_out

    # ── resets ─────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES

        # Logging.
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
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0

        # Sample initial + goal positions on opposite sides of a circle (the
        # "random exchange" scenario from the DMPC paper). Each drone gets its
        # own random angle; goal angle = init angle + pi (so trajectories cross).
        n = len(env_ids)
        device = self.device
        radius = torch.empty(n, self.N, device=device).uniform_(0.8, 1.3)
        theta0 = torch.empty(n, self.N, device=device).uniform_(0.0, 2.0 * torch.pi)
        z0 = torch.empty(n, self.N, device=device).uniform_(0.6, 1.4)

        init_xy = torch.stack([radius * torch.cos(theta0), radius * torch.sin(theta0)], dim=-1)
        goal_xy = -init_xy  # opposite side of the circle
        zg = torch.empty(n, self.N, device=device).uniform_(0.6, 1.4)

        # Translate to the per-env origin.
        origins = self._terrain.env_origins[env_ids]  # (n, 3)
        init_pos = torch.cat([init_xy, z0.unsqueeze(-1)], dim=-1)
        init_pos[..., :2] += origins[:, None, :2]
        goal_pos = torch.cat([goal_xy, zg.unsqueeze(-1)], dim=-1)
        goal_pos[..., :2] += origins[:, None, :2]

        self._goal_pos_w[env_ids] = goal_pos
        self._init_pos_w[env_ids] = init_pos

        for i, robot in enumerate(self._robots):
            default_root = robot.data.default_root_state[env_ids].clone()
            default_root[:, :3] = init_pos[:, i]
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ── helpers exposed for the DMPC expert ────────────────────────────────
    def get_world_states(self) -> dict[str, torch.Tensor]:
        """Returns world-frame positions / velocities / orientations for every
        drone, plus per-drone goals. Used by the DMPC expert."""
        pos_w, quat_w, lin_vel_w, ang_vel_w, _ = self._per_drone_state_world()
        return {
            "pos_w": pos_w,
            "quat_w": quat_w,
            "lin_vel_w": lin_vel_w,
            "ang_vel_w": ang_vel_w,
            "goal_w": self._goal_pos_w,
        }

    def acc_to_action(self, accel_w: torch.Tensor) -> torch.Tensor:
        """Convert desired world-frame accelerations into the 4-D thrust/moment
        action used by this env (per drone), via a differential-flatness style
        low-level controller.

        Args:
            accel_w: ``(num_envs, num_drones, 3)`` desired body-COM accelerations
                in world frame, *not* including gravity.

        Returns:
            ``(num_envs, num_drones * 4)`` action tensor with each component in
            ``[-1, 1]``, ready to feed into :py:meth:`step`.
        """
        return _acc_to_thrust_moment_action(self, accel_w)

    # ── debug viz ──────────────────────────────────────────────────────────
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/multi_drone_goal"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Flatten (E, N, 3) -> (E*N, 3) for visualisation.
        self.goal_pos_visualizer.visualize(self._goal_pos_w.reshape(-1, 3))


# ───────────────────────────────────────────────────────────────────────────
# Low-level controller: desired world-frame acceleration → 4-D Crazyflie action
# ───────────────────────────────────────────────────────────────────────────
def _acc_to_thrust_moment_action(env: MultiDroneDmpcEnv, accel_w: torch.Tensor) -> torch.Tensor:
    """Differential-flatness-style mapping.

    Given a desired world-frame acceleration ``a_des`` for the drone COM, we
    compute the thrust magnitude as the projection of ``m * (a_des + g_comp)``
    onto the current body-z axis, and synthesise a body-rate command via a
    proportional controller on the desired body z-axis. The output is shaped
    so that 0.0 maps to hover thrust and ±1.0 to ±the limit moment used by the
    env, mirroring the standard ``quadcopter_env.py`` action convention.
    """
    device = env.device
    E, N = env.num_envs, env.N
    g = env._gravity_magnitude
    mass = env._robot_mass

    # Gravity compensation: a_total = a_des + g·ẑ_w  (world z up, gravity down).
    g_world = torch.tensor([0.0, 0.0, g], device=device)
    a_total_w = accel_w + g_world[None, None, :]  # (E, N, 3)

    # Current body z-axis in world frame from quaternion (w, x, y, z).
    quat_w = torch.stack([r.data.root_quat_w for r in env._robots], dim=1)  # (E, N, 4)
    w, x, y, z = quat_w.unbind(dim=-1)
    body_z_w = torch.stack(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
        dim=-1,
    )  # (E, N, 3)

    # Thrust magnitude = m * <a_total, b_z>, clamped to >= 0.
    thrust_mag = (mass * (a_total_w * body_z_w).sum(dim=-1)).clamp(min=0.0)  # (E, N)

    # Map to normalised [-1, 1] action component using thrust_to_weight scaling.
    # quadcopter_env: thrust_z = thrust_to_weight * weight * (a+1)/2
    # → a = 2 * thrust_z / (thrust_to_weight * weight) - 1
    max_thrust = env.cfg.thrust_to_weight * env._robot_weight
    a0 = (2.0 * thrust_mag / max(max_thrust, 1e-6) - 1.0).clamp(-1.0, 1.0)

    # Desired body-z direction = a_total / ||a_total||.
    a_norm = torch.linalg.norm(a_total_w, dim=-1, keepdim=True).clamp(min=1e-4)
    b_z_des = a_total_w / a_norm  # (E, N, 3)
    # Tilt error: cross product between current b_z and desired b_z, expressed
    # in world frame, then mapped to the body frame as a proxy moment.
    err_w = torch.cross(body_z_w, b_z_des, dim=-1)  # (E, N, 3)

    # Use a damping term on current body angular velocity.
    ang_vel_b = torch.stack([r.data.root_ang_vel_b for r in env._robots], dim=1)  # (E, N, 3)

    # Rotate world-frame tilt error into body frame for moment command.
    err_b = quat_rotate_inverse(quat_w.reshape(-1, 4), err_w.reshape(-1, 3)).reshape(E, N, 3)

    kp_att, kd_att = 8.0, 0.6
    moment_cmd = kp_att * err_b - kd_att * ang_vel_b  # raw, body frame

    # Normalise to [-1, 1] action range. The env multiplies by moment_scale, so
    # the natural normalisation is moment_cmd / moment_scale, then clamp.
    norm = (moment_cmd / max(env.cfg.moment_scale, 1e-6)).clamp(-1.0, 1.0)

    out = torch.stack([a0, norm[..., 0], norm[..., 1], norm[..., 2]], dim=-1)  # (E, N, 4)
    return out.reshape(E, N * 4)
