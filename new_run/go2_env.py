"""
Go2 Base Environment – Genesis Physics Simulator
Basis-Klasse ohne Rewards. Rewards werden in reward_wrapper.py definiert.
"""

import math
import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(shape, device=device, dtype=torch.float32) + lower


class Go2Env:

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device=None):
        self.num_envs    = num_envs
        self.num_obs     = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device      = device if device else gs.device

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = 0.02   # 50 Hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg     = env_cfg
        self.obs_cfg     = obs_cfg
        self.reward_cfg  = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales  = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"].copy()

        # ------------------------------------------------------------------
        # Scene
        # ------------------------------------------------------------------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.4),
                camera_fov=40,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            ),
        )

        self.scene.build(n_envs=num_envs)

        # DOF-Indizes
        self.motor_dofs = [
            self.robot.get_joint(name).dofs_idx_local[0]
            for name in env_cfg["joint_names"]
        ]

        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # Konstanten
        self.global_gravity   = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        self.base_init_pos    = torch.tensor(env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat   = torch.tensor(env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][n] for n in env_cfg["joint_names"]],
            device=self.device,
        )

        self._init_buffers()

        # Reward-Funktionen registrieren (sucht _reward_{name} per getattr)
        self.reward_functions = {}
        self.episode_sums     = {}
        for name in list(self.reward_scales.keys()):
            self.reward_scales[name] *= self.dt
            fn = getattr(self, f"_reward_{name}", None)
            if fn is not None:
                self.reward_functions[name] = fn
                self.episode_sums[name] = torch.zeros(num_envs, device=self.device)
            else:
                print(f"[Go2Env] Warnung: _reward_{name} nicht gefunden – wird ignoriert.")

        self.extras = {"observations": {}}

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _init_buffers(self):
        n, dev = self.num_envs, self.device
        self.base_pos   = torch.zeros((n, 3), device=dev)
        self.base_quat  = torch.zeros((n, 4), device=dev)
        self.base_lin_vel = torch.zeros((n, 3), device=dev)
        self.base_ang_vel = torch.zeros((n, 3), device=dev)
        self.base_euler   = torch.zeros((n, 3), device=dev)
        self.projected_gravity = torch.zeros((n, 3), device=dev)

        self.dof_pos  = torch.zeros((n, self.num_actions), device=dev)
        self.dof_vel  = torch.zeros((n, self.num_actions), device=dev)
        self.last_dof_vel = torch.zeros((n, self.num_actions), device=dev)

        # Drehmomente (aus PD-Formel berechnet, in step() gesetzt)
        self.torques = torch.zeros((n, self.num_actions), device=dev)

        self.actions      = torch.zeros((n, self.num_actions), device=dev)
        self.last_actions = torch.zeros((n, self.num_actions), device=dev)

        self.commands = torch.zeros((n, self.num_commands), device=dev)

        self.episode_length_buf = torch.zeros(n, device=dev, dtype=torch.int32)
        self.reset_buf = torch.ones(n, device=dev, dtype=torch.bool)
        self.rew_buf   = torch.zeros(n, device=dev)
        self.obs_buf   = torch.zeros((n, self.num_obs), device=dev)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions):
        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )

        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        # Drehmomente approximieren (PD-Gesetz) – vor scene.step() da dof_pos noch aktuell
        self.torques = (
            self.env_cfg["kp"] * (target_dof_pos - self.dof_pos)
            - self.env_cfg["kd"] * self.dof_vel
        )

        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        self.episode_length_buf += 1
        self._update_state()
        self._compute_rewards()
        self._check_termination()

        # Kommandos periodisch neu sampeln
        resample_ids = (
            self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0
        ).nonzero(as_tuple=False).flatten()
        self._resample_commands(resample_ids)

        reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids)

        self._update_observation()
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        self.extras["observations"]["critic"] = self.obs_buf
        self.extras["time_outs"] = (self.episode_length_buf >= self.max_episode_length).float()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------
    # State / Observation / Reward
    # ------------------------------------------------------------------

    def _update_state(self):
        self.base_pos  = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()

        inv_q = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_q)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_q)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_q)

        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True, degrees=True,
        )

        self.dof_pos = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel = self.robot.get_dofs_velocity(self.motor_dofs)

    def _update_observation(self):
        """
        Obs-Vektor (45 Dimensionen):
          3  – Winkelgeschwindigkeit (Body-Frame)
          3  – projizierte Schwerkraft
          3  – Kommandos (vx, vy, yaw)
          12 – Gelenkposition (relativ zur Standardstellung)
          12 – Gelenkgeschwindigkeit
          12 – letzte Aktion
        """
        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.obs_scales.get("commands", 1.0),
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
        ], dim=-1)

    def _compute_rewards(self):
        self.rew_buf.zero_()
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

    def _check_termination(self):
        self.reset_buf = self.episode_length_buf >= self.max_episode_length
        roll  = self.env_cfg.get("termination_if_roll_greater_than", 30)
        pitch = self.env_cfg.get("termination_if_pitch_greater_than", 30)
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > roll
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > pitch
        min_h = self.env_cfg.get("termination_if_height_lower_than", 0.15)
        self.reset_buf |= self.base_pos[:, 2] < min_h

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        self.commands[env_ids, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (n,), self.device)
        self.commands[env_ids, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (n,), self.device)
        self.commands[env_ids, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"],   (n,), self.device)

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        pos  = self.base_init_pos.unsqueeze(0).expand(n, -1)
        quat = self.base_init_quat.unsqueeze(0).expand(n, -1)
        dofs = self.default_dof_pos.unsqueeze(0).expand(n, -1)

        self.robot.set_pos(pos,  zero_velocity=True, envs_idx=env_ids)
        self.robot.set_quat(quat, zero_velocity=True, envs_idx=env_ids)
        self.robot.set_dofs_position(dofs, self.motor_dofs, zero_velocity=True, envs_idx=env_ids)

        self.base_pos[env_ids]   = self.base_init_pos
        self.base_quat[env_ids]  = self.base_init_quat
        self.base_lin_vel[env_ids] = 0
        self.base_ang_vel[env_ids] = 0
        self.dof_pos[env_ids]    = self.default_dof_pos
        self.dof_vel[env_ids]    = 0
        self.actions[env_ids]    = 0
        self.last_actions[env_ids] = 0
        self.last_dof_vel[env_ids] = 0
        self.torques[env_ids]    = 0
        self.episode_length_buf[env_ids] = 0

        self.extras["episode"] = {}
        for key, val in self.episode_sums.items():
            self.extras["episode"][f"rew_{key}"] = (
                val[env_ids].mean() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][env_ids] = 0

        self._resample_commands(env_ids)

    # ------------------------------------------------------------------
    # RSL-RL Interface
    # ------------------------------------------------------------------

    def reset(self):
        self.reset_buf.fill_(True)
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._update_observation()
        return self.obs_buf, None

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None