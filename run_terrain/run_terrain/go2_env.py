import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.engine.scene import Scene
from genesis.engine.solvers import RigidSolver


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, eval=False):
        self.show_viewer = show_viewer
        self.eval = eval
        self.device = gs.device

        self._initialize_env_parameters(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)
        self._setup_scene(show_viewer)
        self._add_terrain()
        self._add_and_configure_robot()
        self._set_camera()
        self.scene.build(n_envs=num_envs)
        self._setup_motor_joints()
        self._find_link_indices()
        self._setup_reward_functions()
        self._initialize_buffers()
        # Resample initial commands now that buffers exist
        self.resample_commands(torch.arange(num_envs, device=self.device))
        self.reset()

    # -------------------------------------------------------------------------
    # Initialisation helpers
    # -------------------------------------------------------------------------

    def _initialize_env_parameters(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg):
        """Initialize all scalar / config parameters."""
        self.num_envs: int            = num_envs
        self.num_obs: int             = obs_cfg["num_obs"]
        self.num_privileged_obs: int  = obs_cfg.get("num_priviliged_obs", None)
        self.num_actions: int         = env_cfg["num_actions"]

        self.simulate_action_latency: bool = True   # 1-step latency as on real robot
        self.dt: float                     = 0.02
        self.max_episode_length: int       = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg      = env_cfg
        self.use_terrain  = env_cfg.get("use_terrain", False)
        self.obs_cfg      = obs_cfg
        self.reward_cfg   = reward_cfg
        self.obs_scales   = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.command_cfg  = command_cfg

        # --- Range-based command API -----------------------------------------
        self.num_commands: int    = command_cfg.get("num_commands", 3)
        self.lin_vel_x_range: list = list(command_cfg.get("lin_vel_x_range", [0.0, 2.0]))
        self.lin_vel_y_range: list = list(command_cfg.get("lin_vel_y_range", [-0.3, 0.3]))
        self.ang_vel_range: list   = list(command_cfg.get("ang_vel_range",   [-1.0, 1.0]))
        self.resampling_time: int  = int(command_cfg.get("resampling_time_s", 4.0) / self.dt)

        # --- Recording -------------------------------------------------------
        self.headless: bool               = not self.show_viewer
        self._recording: bool             = False
        self._recorded_frames_behind: list = []
        self._recorded_frames_side: list   = []
        self.num_frames: int              = 500

        # Will be overwritten properly in _add_terrain
        self.base_init_pos = torch.tensor([0.0, 0.0, 0.42], device=self.device)

    def _setup_scene(self, show_viewer):
        self.scene: Scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                tolerance=1e-8,
                enable_collision=True,
                enable_joint_limit=True,
                
            ),
            show_viewer=show_viewer,
        )
        for solver in self.scene.sim.solvers:
            if isinstance(solver, RigidSolver):
                self.rigid_solver = solver
                break


    def _add_terrain(self):
        """Add terrain or plane to the scene"""
        if self.use_terrain:
            self._add_complex_terrain()
        else:
            self._add_simple_plane()
            self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)

    def _add_complex_terrain(self):
        """Add complex terrain with height field and distribute environments."""
        self.terrain_cfg = self.env_cfg['terrain_cfg']
        self.terrain = self.scene.add_entity(
            gs.morphs.Terrain(
                n_subterrains=self.terrain_cfg['n_subterrains'],
                horizontal_scale=self.terrain_cfg['horizontal_scale'],
                vertical_scale=self.terrain_cfg['vertical_scale'],
                subterrain_size=self.terrain_cfg['subterrain_size'],
                subterrain_types=self.terrain_cfg['subterrain_types'],
                randomize=self.terrain_cfg.get('randomize', False),
            ),
        )
        
        terrain_margin_x = self.terrain_cfg['n_subterrains'][0] * self.terrain_cfg['subterrain_size'][0]
        terrain_margin_y = self.terrain_cfg['n_subterrains'][1] * self.terrain_cfg['subterrain_size'][1]
        self.terrain_margin = torch.tensor(
            [terrain_margin_x, terrain_margin_y], device=self.device, dtype=gs.tc_float
        )
        
        height_field = self.terrain.geoms[0].metadata["height_field"]
        self.height_field = torch.tensor(
            height_field, device=self.device, dtype=gs.tc_float
        ) * self.terrain_cfg['vertical_scale']

        # WICHTIG: base_init_pos als (num_envs, 3) initialisieren, um Roboter aufzuteilen
        self.base_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        subterrain_width_x = self.terrain_cfg['subterrain_size'][0]
        subterrain_height_y = self.terrain_cfg['subterrain_size'][1]
        num_rows = self.terrain_cfg['n_subterrains'][1]
        y_start_center = (num_rows * subterrain_height_y) / 2

        for env_id in range(self.num_envs):
            row_idx = env_id % num_rows
            y_pos = row_idx * subterrain_height_y + (subterrain_height_y / 2)
            
            if self.terrain_cfg.get('randomize', False):
                x_pos = subterrain_width_x / 2
                self.base_init_pos[env_id] = torch.tensor([x_pos, y_pos, 0.45], device=self.device)
            else: 
                self.base_init_pos[env_id] = torch.tensor([0.30, y_pos, 0.35], device=self.device)

        self.height_patch_n_x = 5
        self.height_patch_n_y = 5
        self.height_patch_step_x = 0.4
        self.height_patch_step_y = 0.4
        self.height_patch_n_points = self.height_patch_n_x * self.height_patch_n_y

        local_positions = []
        for i in range(self.height_patch_n_x):
            x = i * self.height_patch_step_x
            for j in range(self.height_patch_n_y):
                y = -((self.height_patch_n_y - 1) * self.height_patch_step_y) / 2 + j * self.height_patch_step_y
                local_positions.append([x, y, 0])
        self.height_patch_local_positions = torch.tensor(
            local_positions, device=self.device, dtype=gs.tc_float
        )

        self.num_obs += self.height_patch_n_points
        if self.num_privileged_obs is not None:
            self.num_privileged_obs += self.height_patch_n_points
        
        self.relative_heights = torch.zeros((self.num_envs, self.height_patch_n_points), device=self.device, dtype=gs.tc_float)
        self.reset_environment_at_random_terrain = self.terrain_cfg.get('reset_environment_at_random_terrain', False)


    def _add_simple_plane(self):
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.reset_environment_at_random_terrain = False
        self.target_increased = False
        # Konsistente Dimension (num_envs, 3) auch für Plane halten
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device).repeat(self.num_envs, 1)

    def _add_and_configure_robot(self):
        self.base_init_quat    = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        
        # FIX: Genesis verlangt hier nur EINE (x,y,z) Koordinate für den Bauplan des Roboters.
        dummy_archetype_pos = [0.0, 0.0, 0.45]
        
        self.robot: RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                links_to_keep=self.env_cfg["links_to_keep"],
                pos=dummy_archetype_pos,  # <--- HIER MUSS dummy_archetype_pos REIN, NICHT base_init_pos!
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

    def _setup_motor_joints(self):
        self.motor_dofs: list[int] = [
            self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]
        ]
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

    def _setup_reward_functions(self):
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt  # scale by dt: reward/step → reward/second
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

    def _initialize_buffers(self):
        """Allocate every state tensor used for observations, rewards, and control."""
        # Velocity / orientation
        self.base_lin_vel      = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel      = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity    = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)

        # Observations
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.privileged_obs_buf = (
            None if self.num_privileged_obs is None
            else torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=gs.tc_float)
        )

        # Rewards / resets / episode counters
        self.rew_buf            = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf          = torch.ones ((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        # Actions (3 history levels for smoothness reward)
        self.actions           = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions      = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)  # for 2nd-order smoothness

        # Joint states
        self.dof_pos      = torch.zeros_like(self.actions)
        self.dof_vel      = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.torques      = torch.zeros_like(self.actions)  # measured joint torques

        # Base pose
        self.base_pos      = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.base_quat     = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)

        # Default joint positions
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device, dtype=gs.tc_float,
        )

        # Contact forces  shape: (num_envs, n_links, 3)
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )

        # Velocity commands
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)

        # Misc
        self.extras = dict()

        self._initialize_feet_buffers()

    def _initialize_feet_buffers(self):
        self.foot_positions   = torch.zeros((self.num_envs, 4, 3), device=self.device, dtype=gs.tc_float)
        self.foot_quaternions = torch.zeros((self.num_envs, 4, 4), device=self.device, dtype=gs.tc_float)
        self.foot_velocities  = torch.zeros((self.num_envs, 4, 3), device=self.device, dtype=gs.tc_float)
        # Boolean contact mask and cumulative air-time
        self.feet_contact  = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.bool)
        self.feet_air_time = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)

    def _find_link_indices(self):
        def find_link_indices(names):
            return [
                link.idx - self.robot.link_start
                for link in self.robot.links
                if any(name in link.name for name in names)
            ]

        self.termination_contact_link_indices = find_link_indices(self.env_cfg["termination_contact_link_names"])
        self.penalized_contact_link_indices   = find_link_indices(self.env_cfg["penalized_contact_link_names"])
        self.feet_link_indices                = find_link_indices(self.env_cfg["feet_link_names"])
        self.feet_link_indices_world_frame    = [i + 1 for i in self.feet_link_indices]

    # -------------------------------------------------------------------------
    # Commands
    # -------------------------------------------------------------------------

    def resample_commands(self, env_ids):
        n = len(env_ids)
        if n == 0:
            return
        # Untere Grenze bleibt bei 0.5, obere wächst mit Curriculum
        v_max = self.lin_vel_x_range[1]
        v_min = self.lin_vel_x_range[0]
        self.commands[env_ids, 0] = torch.empty(n, device=self.device).uniform_(v_min, v_max)
        self.commands[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*self.lin_vel_y_range)
        self.commands[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*self.ang_vel_range)
        self.commands[env_ids, :2] *= (torch.abs(self.commands[env_ids, :2]) > 0.2).float()

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------

    def step(self, actions):
        self._process_actions(actions)
        self._update_robot_state()
        self._check_termination()
        self._compute_rewards()

        # KORREKTUR: Berechne die Terrain-Höhen vor den Observations!
        if self.use_terrain:
            self._compute_relative_heights()

        resample_ids = (self.episode_length_buf % self.resampling_time == 0).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self.resample_commands(resample_ids)

        self._compute_observations()
        self._render_headless()

        # Shift action history
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:]      = self.actions[:]
        self.last_dof_vel[:]      = self.dof_vel[:]

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def _process_actions(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

    def _update_robot_state(self):
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:]  = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler   = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )

        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:]    = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:]    = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity  = transform_by_quat(self.global_gravity,  inv_base_quat)

        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.torques[:] = self.robot.get_dofs_force(self.motor_dofs)  # for torque penalty

        self.foot_positions[:]   = self.rigid_solver.get_links_pos(self.feet_link_indices_world_frame)
        self.foot_quaternions[:] = self.rigid_solver.get_links_quat(self.feet_link_indices_world_frame)
        self.foot_velocities[:]  = self.rigid_solver.get_links_vel(self.feet_link_indices_world_frame)

        self.link_contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(), device=self.device, dtype=gs.tc_float,
        )
        # Boolean foot contact: any foot link with contact force > 1 N
        self.feet_contact = (
            torch.norm(self.link_contact_forces[:, self.feet_link_indices, :], dim=-1) > 1.0
        )  # shape: (num_envs, 4)

    def _check_termination(self):
        self.reset_buf  = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        base_contact = (
            torch.norm(self.link_contact_forces[:, self.termination_contact_link_indices[0], :], dim=-1) > 1e-3
        )
        self.reset_buf |= base_contact

        self._handle_timeouts()
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())


    def _handle_timeouts(self):
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

    def _compute_rewards(self):
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf          += rew
            self.episode_sums[name] += rew

    def _get_heights(self):
        """Berechnet die relative Geländehöhe im Grid um die Roboterbasis."""
        if not self.use_terrain:
            return torch.zeros((self.num_envs, self.height_patch_n_points), device=self.device)

        # 1. Globale Positionen der Scan-Punkte berechnen (Basis-Position + lokales Grid)
        # self.base_pos: (num_envs, 3) -> erweitert auf (num_envs, 1, 3)
        # lokales Grid: (height_patch_n_points, 3)
        points = self.base_pos.unsqueeze(1) + self.height_patch_local_positions.unsqueeze(0)
        
        # Extrahieren der X und Y Koordinaten (num_envs, n_points)
        px = points[..., 0]
        py = points[..., 1]

        # 2. Koordinaten in Indizes für die height_field Matrix umrechnen
        h_scale = self.terrain_cfg['horizontal_scale']
        
        # Sicherstellen, dass wir innerhalb der Matrixgrenzen bleiben
        max_idx_x = self.height_field.shape[0] - 1
        max_idx_y = self.height_field.shape[1] - 1

        grid_x = torch.clamp((px / h_scale).long(), 0, max_idx_x)
        grid_y = torch.clamp((py / h_scale).long(), 0, max_idx_y)

        # 3. Absoluthöhen aus dem Heightfield extrahieren
        # Da grid_x und grid_y (num_envs, n_points) groß sind, nutzen wir advanced indexing
        absolute_heights = self.height_field[grid_x, grid_y]

        # 4. Relative Höhe berechnen (Bodenhöhe minus aktuelle Rumpfhöhe des Roboters)
        # Negativ bedeutet: Der Boden liegt unterhalb des Roboterrumpfes
        relative_heights = absolute_heights - self.base_pos[:, 2].unsqueeze(1)
        
        return relative_heights

    def _compute_relative_heights(self) -> None:
        """Compute heights relative to the base of the robot for the height field patch in front of the robot."""
        local_positions = self.height_patch_local_positions.unsqueeze(0).repeat(self.num_envs, 1, 1)
        base_quat_repeated = self.base_quat.unsqueeze(1).repeat(1, 25, 1)

        base_quat_flat = base_quat_repeated.view(-1, 4)
        local_positions_flat = local_positions.view(-1, 3)

        rotated_positions_flat = transform_by_quat(local_positions_flat, base_quat_flat)
        rotated_positions = rotated_positions_flat.view(self.num_envs, 25, 3)
        world_positions = self.base_pos.unsqueeze(1) + rotated_positions

        clipped_positions = world_positions[:, :, :2].clamp(
            min=torch.zeros(2, device=self.device),
            max=self.terrain_margin
        )

        grid_indices = (clipped_positions / self.terrain_cfg['horizontal_scale'] - 0.5).floor().int()
        width, height = self.height_field.shape

        grid_indices[:, :, 0] = torch.clamp(grid_indices[:, :, 0], 0, width - 1)
        grid_indices[:, :, 1] = torch.clamp(grid_indices[:, :, 1], 0, height - 1)

        grid_x = grid_indices[:, :, 0]
        grid_y = grid_indices[:, :, 1]

        heights = self.height_field[grid_x, grid_y]
        self.relative_heights = heights - self.base_pos[:, 2].unsqueeze(1)

    # -------------------------------------------------------------------------
    # Observations
    # -------------------------------------------------------------------------

    def _compute_observations(self):
        # Berechne die skalierten Commands (3 Dimensionen: x-vel, y-vel, yaw-vel)
        scaled_commands = self.commands.clone()
        scaled_commands[:, :2] *= self.obs_scales['lin_vel']
        scaled_commands[:, 2]  *= self.obs_scales['ang_vel']

        obs_list = [
            self.base_lin_vel * self.obs_scales['lin_vel'], # 3
            self.base_ang_vel * self.obs_scales["ang_vel"], # 3
            self.projected_gravity, # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"], # 12
            self.dof_vel * self.obs_scales["dof_vel"], # 12
            self.actions, # 12
            self.base_pos - self.last_base_pos, # 3
            scaled_commands, # 3  <--- DIESE ZEILE HAT GEFEHLT!
        ]
        
        if self.use_terrain:
            self.relative_heights = self._get_heights()
            obs_list.append(self.relative_heights)
        else:
            empty_heights = torch.zeros((self.num_envs, self.height_patch_n_points), device=self.device)
            obs_list.append(empty_heights)

        self.obs_buf = torch.cat(obs_list, dim=-1)

        if self.num_privileged_obs is not None:
            priv_list = [
                self.base_lin_vel   * self.obs_scales["lin_vel"],                     
                self.base_ang_vel   * self.obs_scales["ang_vel"],                     
                self.projected_gravity,                                                
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],   
                self.dof_vel        * self.obs_scales["dof_vel"],                     
                self.last_dof_vel   * self.obs_scales["dof_vel"],                     
                self.actions,                                                          
                self.last_actions,                                                     
                self.commands,                                                         
                self.base_pos - self.last_base_pos,                                   
            ]
            
            # Auch für das Critic-Netzwerk anhängen
            if self.use_terrain:
                priv_list.append(self.relative_heights)
                
            self.privileged_obs_buf = torch.clip(torch.cat(priv_list, dim=-1), -100.0, 100.0)


    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self._reset_robot_state(envs_idx)
        self._reset_buffers(envs_idx)
        self._update_episode_stats(envs_idx)
        self.resample_commands(envs_idx)

    def _reset_robot_state(self, envs_idx):
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        
        # KORREKTUR: Indizierter Zugriff auf die Startpositionen
        self.base_pos[envs_idx] = self.base_init_pos[envs_idx]

        self.last_base_pos[envs_idx] = self.base_pos[envs_idx].clone()
        self.base_quat[envs_idx]     = self.base_init_quat.unsqueeze(0)
        self.robot.set_pos (self.base_pos[envs_idx],  zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

    def _reset_buffers(self, envs_idx):
        self.last_actions[envs_idx]       = 0.0
        self.last_last_actions[envs_idx]  = 0.0
        self.last_dof_vel[envs_idx]       = 0.0
        self.feet_air_time[envs_idx]      = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx]          = True

    def _update_episode_stats(self, envs_idx):
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # -------------------------------------------------------------------------
    # Curriculum
    # -------------------------------------------------------------------------

    def increase_x_target(self, delta):
        """Erhöht die obere Grenze des Geschwindigkeits-Samplings nachhaltig."""
        self.lin_vel_x_range[1] = min(self.lin_vel_x_range[1] + delta, 5.0)  # Ziel: bis zu 5 m/s
        # Optional: Schiebe das Minimum leicht nach, damit er langsames Laufen verlernt
        self.lin_vel_x_range[0] = min(self.lin_vel_x_range[0] + delta * 0.3, 1.5)
        print(f"[Velocity Curriculum] Neuer Zielbereich x-Geschwindigkeit: {self.lin_vel_x_range[0]:.2f} bis {self.lin_vel_x_range[1]:.2f} m/s")

    def update_terrain_curriculum(self, iteration: int, interval: int):
        """
        NEU: Verschiebt die Startsektoren der Umgebungen auf schwierigere Terrains
        und aktualisiert den virtuellen Skalierungsfaktor für das Logging.
        """
        if not self.use_terrain:
            return

        max_levels = self.terrain_cfg['n_subterrains'][0]  # Spaltenanzahl (Schwierigkeitsstufen)
        current_level = min(iteration // interval, max_levels - 1)
        
        # Simuliere die ansteigende vertikale Komponente für das WandB-Logging
        self.terrain_cfg['curriculum_vertical_scale'] = self.terrain_cfg['vertical_scale'] * (1.0 + 0.15 * current_level)
        
        subterrain_width_x = self.terrain_cfg['subterrain_size'][0]
        subterrain_height_y = self.terrain_cfg['subterrain_size'][1]
        num_rows = self.terrain_cfg['n_subterrains'][1]
        
        for env_id in range(self.num_envs):
            row_idx = env_id % num_rows
            
            # Berechne Startpunkt innerhalb der aktuellen Schwierigkeitsstufe (Spalte = current_level)
            x_start = current_level * subterrain_width_x + 0.5
            y_start = row_idx * subterrain_height_y + (subterrain_height_y / 2)
            
            # Höhenfeld-Lookup um Stürze oder Clipping beim Spawn zu verhindern
            grid_x = max(0, min(int(x_start / self.terrain_cfg['horizontal_scale']), self.height_field.shape[0] - 1))
            grid_y = max(0, min(int(y_start / self.terrain_cfg['horizontal_scale']), self.height_field.shape[1] - 1))
            z_terrain = self.height_field[grid_x, grid_y].item()
            
            # Sicheres Z-Niveau über dem konkreten Punkt setzen
            self.base_init_pos[env_id] = torch.tensor([x_start, y_start, z_terrain + 0.40], device=self.device)
            
        print(f"[Terrain Curriculum] Iteration {iteration}: Wechsel auf Terrain-Stufe {current_level}/{max_levels - 1}")

    # -------------------------------------------------------------------------
    # Camera / recording
    # -------------------------------------------------------------------------

    def _set_camera(self):
        self._floating_camera_behind = self.scene.add_camera(
            pos=np.array([-1.5, 0.0, 5.0]),
            lookat=np.array([0, 0, 0.1]),
            fov=45, GUI=False, res=(720, 720),
        )
        if self.eval:
            self._floating_camera_side = self.scene.add_camera(
                pos=np.array([0.0, -2.5, 1.5]),
                lookat=np.array([0, 0, 0.3]),
                fov=45, GUI=False, res=(720, 720),
            )

    def _render_headless(self):
        if self._recording and len(self._recorded_frames_behind) < self.num_frames:
            robot_pos = np.array(self.base_pos[0].cpu())
            self._floating_camera_behind.set_pose(
                pos=robot_pos + np.array([-1.5, 0.0, 2.5]),
                lookat=robot_pos + np.array([0.3, 0.0, 0.0]),
            )
            frame_behind, _, _, _ = self._floating_camera_behind.render()
            self._recorded_frames_behind.append(frame_behind)
            if self.eval:
                self._floating_camera_side.set_pose(
                    pos=robot_pos + np.array([0.0, -2.5, 1.0]),
                    lookat=robot_pos + np.array([0.0, 0.0, -0.1]),
                )
                frame_side, _, _, _ = self._floating_camera_side.render()
                self._recorded_frames_side.append(frame_side)

    def get_recorded_frames(self):
        print(f"Recorded {len(self._recorded_frames_behind)} behind frames")
        done = len(self._recorded_frames_behind) >= self.num_frames - 1
        if done:
            frames_behind = self._recorded_frames_behind
            self._recorded_frames_behind = []
            self._recording = False
            if self.eval:
                frames_side = self._recorded_frames_side
                self._recorded_frames_side = []
                return frames_behind, frames_side
            return frames_behind
        return None

    def start_recording(self, record_internal=True):
        self._recorded_frames_behind = []
        self._recorded_frames_side   = []
        self._recording = True
        if not record_internal:
            self._floating_camera_behind.start_recording()
            if self.eval:
                self._floating_camera_side.start_recording()

    def stop_recording(self, save_path_behind=None, save_path_side=None):
        self._recorded_frames_behind = []
        self._recorded_frames_side   = []
        self._recording = False
        if save_path_behind is not None:
            self._floating_camera_behind.stop_recording(save_path_behind, fps=int(1 / self.dt))
        if save_path_side is not None and self.eval:
            self._floating_camera_side.stop_recording(save_path_side, fps=int(1 / self.dt))