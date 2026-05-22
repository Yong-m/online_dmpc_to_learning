# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-drone DMPC environment (Luis et al. 2020 reference setup).

Mirrors the layout of the standalone ``quadcopter_env.py`` (one Crazyflie per
env, goal-reaching) but instantiates *N* Crazyflies and aligns the action /
observation interface with the multi-robot motion-planning setting of the
``online_dmpc`` paper. Design choices:

* **Action = 3-D desired velocity per drone** (paper's ``u_i`` is a position
  reference; we expose it as a normalised velocity command, which yields a
  per-step position-reference delta of ``v_cmd * ts``). Total env action
  dimension is ``3 * num_drones``. The env internally turns the 3-D command
  into the standard 4-D Crazyflie thrust + moment via a cascaded P-controller
  + differential flatness (`ref_to_action`).

* **Per-drone observation including neighbour inputs.** ``get_per_drone_obs()``
  returns ``(num_envs, num_drones, per_drone_obs_dim)``. Each slice contains
  the drone's own body-frame state + goal, and for every neighbour ``j``: its
  body-frame relative position, relative velocity, **and the neighbour's most
  recent applied input (velocity reference)**. The DMPC paper (Section III.E)
  finds input-space avoidance superior to state-space because the input is
  forward-looking; we surface that same signal to the student.

* **Decentralised policy ready.** The flat observation in ``"policy"`` is a
  concatenation of the per-drone slices in fixed drone order. The BC script
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
from isaaclab.utils.math import quat_rotate_inverse, subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


# Per-drone observation layout sizes.
PER_DRONE_OWN_DIM = 12   # lin_vel_b (3) + ang_vel_b (3) + projected_gravity_b (3) + goal_b (3)
PER_NEIGHBOUR_DIM = 9    # rel_pos_b (3) + rel_vel_b (3) + neighbour_last_input_b (3)


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
    decimation: int = 2
    action_space: int = 3 * 4   # 3 per drone, overwritten in __post_init__
    observation_space: int = 4 * 39  # overwritten in __post_init__
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
    # Velocity normalisation for the action interface. ``action * v_max`` gives
    # the desired world-frame velocity, and ``ref_pos_w = current_pos +
    # desired_velocity * step_dt`` is the target tracked by the inner P loop.
    v_max: float = 5.0 #2.0

    # ── position-reference tracker gains ──
    pos_track_kp: float = 6.0 #6.0
    pos_track_kd: float = 4.5 #4.5
    track_accel_clip: float = 4.0 # 4.0
    att_track_kp: float = 8.0
    att_track_kd: float = 0.6

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

        # Per-drone normalised desired velocity action and a buffer of the
        # most-recent reference velocity ``v_cmd_w`` actually applied.
        self._actions = torch.zeros(self.num_envs, self.N, 3, device=device)
        self._last_ref_vel_w = torch.zeros(self.num_envs, self.N, 3, device=device)
        # Per-drone thrust / moment buffers ultimately applied to PhysX.
        self._thrust = torch.zeros(self.num_envs, self.N, 1, 3, device=device)
        self._moment = torch.zeros(self.num_envs, self.N, 1, 3, device=device)

        # Per-drone goal positions in world frame.
        self._goal_pos_w = torch.zeros(self.num_envs, self.N, 3, device=device)
        # Initial positions saved at reset (handy for diagnostics).
        self._init_pos_w = torch.zeros(self.num_envs, self.N, 3, device=device)

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
        """``actions`` is ``(num_envs, 3 * num_drones)`` of normalised desired
        velocities in ``[-1, 1]^3`` per drone. We turn it into a position
        reference one ``step_dt`` ahead of the current state, run the cascaded
        P-controller in :py:meth:`ref_to_action`, and stash the world-frame
        reference velocity so neighbours can observe it next step."""
        a = actions.view(self.num_envs, self.N, 3).clamp(-1.0, 1.0)
        self._actions = a.clone()
        v_cmd_w = a * self.cfg.v_max                          # (E, N, 3)
        self._last_ref_vel_w = v_cmd_w.detach().clone()

        # Convert to a 4-D thrust + moment command via the position-tracking
        # cascade. ref_pos_w is one control step ahead of the current state.
        st = self._stack_drone_state()
        pos_w = st["pos_w"]
        ref_pos_w = pos_w + v_cmd_w * self.step_dt
        thrust_moment = self._ref_to_thrust_moment(ref_pos_w, v_cmd_w)
        a4 = thrust_moment  # (E, N, 4)
        self._thrust[:, :, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (a4[:, :, 0] + 1.0) / 2.0
        )
        self._moment[:, :, 0, :] = self.cfg.moment_scale * a4[:, :, 1:]

    def _apply_action(self):
        for i, robot in enumerate(self._robots):
            robot.set_external_force_and_torque(
                self._thrust[:, i], self._moment[:, i], body_ids=self._body_id, is_local=True
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

    def get_per_drone_obs(self) -> torch.Tensor:
        """Build the per-drone observation tensor used by the decentralised
        student policy.

        Returns:
            ``(num_envs, num_drones, per_drone_obs_dim)`` with layout for
            drone ``i`` (body-frame coordinates expressed in drone ``i``'s body
            frame):

            * **0-2**   ``lin_vel_b``
            * **3-5**   ``ang_vel_b``
            * **6-8**   ``projected_gravity_b``
            * **9-11**  goal position (body frame)
            * For each neighbour ``j`` in fixed order (drones ``0..N-1``
              skipping ``i``):
              * relative position (body frame, 3),
              * relative linear velocity (body frame, 3),
              * neighbour's last applied input -- its desired velocity command,
                rotated into drone ``i``'s body frame (3).
        """
        E, N = self.num_envs, self.N
        st = self._stack_drone_state()
        pos_w = st["pos_w"]
        quat_w = st["quat_w"]
        lin_vel_w = st["lin_vel_w"]

        out = torch.empty(
            (E, N, self.per_drone_obs_dim), device=self.device, dtype=pos_w.dtype
        )
        for i in range(N):
            robot = self._robots[i]
            quat_i = quat_w[:, i]
            goal_b, _ = subtract_frame_transforms(
                robot.data.root_pos_w, quat_i, self._goal_pos_w[:, i]
            )
            own = torch.cat(
                [
                    st["lin_vel_b"][:, i],
                    st["ang_vel_b"][:, i],
                    st["proj_gravity_b"][:, i],
                    goal_b,
                ],
                dim=-1,
            )  # (E, 12)
            out[:, i, : PER_DRONE_OWN_DIM] = own

            offset = PER_DRONE_OWN_DIM
            for j in range(N):
                if j == i:
                    continue
                rel_pos_w = pos_w[:, j] - pos_w[:, i]
                rel_vel_w = lin_vel_w[:, j] - lin_vel_w[:, i]
                rel_pos_b = quat_rotate_inverse(quat_i, rel_pos_w)
                rel_vel_b = quat_rotate_inverse(quat_i, rel_vel_w)
                # Neighbour input: their applied velocity reference, mapped
                # into drone i's body frame. This is the "u_j" they are
                # executing right now in paper notation.
                neigh_ref_b = quat_rotate_inverse(quat_i, self._last_ref_vel_w[:, j])
                out[:, i, offset : offset + 3] = rel_pos_b
                out[:, i, offset + 3 : offset + 6] = rel_vel_b
                out[:, i, offset + 6 : offset + 9] = neigh_ref_b
                offset += PER_NEIGHBOUR_DIM
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
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pos_w = torch.stack([r.data.root_pos_w for r in self._robots], dim=1)
        z = pos_w[..., 2]
        died = ((z < self.cfg.z_min) | (z > self.cfg.z_max)).any(dim=-1)
        return died, time_out

    # ── resets ─────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES

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
        self._last_ref_vel_w[env_ids] = 0.0

        # Random-exchange scenario (DMPC paper Section IV/VI): each drone on
        # one side of a random circle, goal on the diametrically opposite side.
        n = len(env_ids)
        device = self.device
        radius = torch.empty(n, self.N, device=device).uniform_(0.8, 1.3)
        theta0 = torch.empty(n, self.N, device=device).uniform_(0.0, 2.0 * torch.pi)
        z0 = torch.empty(n, self.N, device=device).uniform_(0.6, 1.4)

        init_xy = torch.stack([radius * torch.cos(theta0), radius * torch.sin(theta0)], dim=-1)
        goal_xy = -init_xy
        zg = torch.empty(n, self.N, device=device).uniform_(0.6, 1.4)

        origins = self._terrain.env_origins[env_ids]
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
        """World-frame states + goals used by :class:`DMPCExpert`."""
        st = self._stack_drone_state()
        st["goal_w"] = self._goal_pos_w
        st["last_ref_vel_w"] = self._last_ref_vel_w
        return st

    def velocity_to_action(self, v_cmd_w: torch.Tensor) -> torch.Tensor:
        """Convert a desired *world-frame velocity command* into the env's
        normalised 3-D action ``[-1, 1]^3`` per drone.

        Args:
            v_cmd_w: ``(num_envs, num_drones, 3)`` in m/s.

        Returns:
            ``(num_envs, num_drones * 3)`` action tensor in ``[-1, 1]``.
        """
        E, N = v_cmd_w.shape[0], v_cmd_w.shape[1]
        norm = (v_cmd_w / max(self.cfg.v_max, 1e-6)).clamp(-1.0, 1.0)
        return norm.reshape(E, N * 3)

    def ref_to_action(
        self,
        ref_pos_w: torch.Tensor,
        ref_vel_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert a *world-frame position reference* (and optional velocity
        reference) into the env's normalised 3-D velocity action.

        The env's pre-physics step interprets the action as ``v_cmd_w = a *
        v_max`` and sets ``ref_pos_internal = pos_w + v_cmd_w * step_dt``. The
        natural inverse is therefore ``v_cmd_w = (ref_pos_w - pos_w) / step_dt``;
        if ``ref_vel_w`` is supplied we blend the two so the inner P-controller
        also receives a non-trivial feed-forward velocity. This is the function
        the DMPC expert calls to drop its position-reference plan into the env.

        Args:
            ref_pos_w: ``(num_envs, num_drones, 3)`` desired positions.
            ref_vel_w: ``(num_envs, num_drones, 3)`` optional feed-forward
                velocity. When given, the command is the average of the
                position-based command and ``ref_vel_w`` so that fast straight-
                line tracking still works when the position error is small.

        Returns:
            ``(num_envs, num_drones * 3)`` action tensor in ``[-1, 1]``.
        """
        st = self._stack_drone_state()
        pos_w = st["pos_w"]
        step_dt = self.step_dt
        v_from_pos = (ref_pos_w - pos_w) / max(step_dt, 1e-6)
        if ref_vel_w is None:
            v_cmd_w = v_from_pos
        else:
            v_cmd_w = 0.5 * (v_from_pos + ref_vel_w)
        return self.velocity_to_action(v_cmd_w)

    # ── internal: cascaded position controller -> thrust/moment ────────────
    def _ref_to_thrust_moment(
        self, ref_pos_w: torch.Tensor, ref_vel_w: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Cascaded P-controller (paper Fig. 2 inner loop) + differential
        flatness. Returns ``(E, N, 4)`` normalised thrust + moment commands."""
        st = self._stack_drone_state()
        pos_w = st["pos_w"]
        vel_w = st["lin_vel_w"]
        if ref_vel_w is None:
            ref_vel_w = torch.zeros_like(ref_pos_w)
        desired_accel = (
            self.cfg.pos_track_kp * (ref_pos_w - pos_w)
            + self.cfg.pos_track_kd * (ref_vel_w - vel_w)
        )
        clip = self.cfg.track_accel_clip
        desired_accel = desired_accel.clamp(-clip, clip)
        return _acc_to_thrust_moment_action(self, desired_accel)

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
        self.goal_pos_visualizer.visualize(self._goal_pos_w.reshape(-1, 3))


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
    err_b = quat_rotate_inverse(quat_w.reshape(-1, 4), err_w.reshape(-1, 3)).reshape(E, N, 3)

    moment_cmd = env.cfg.att_track_kp * err_b - env.cfg.att_track_kd * ang_vel_b
    norm = (moment_cmd / max(env.cfg.moment_scale, 1e-6)).clamp(-1.0, 1.0)

    return torch.stack([a0, norm[..., 0], norm[..., 1], norm[..., 2]], dim=-1)
