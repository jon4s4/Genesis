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
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg = None, show_viewer=False, device="cuda", eval=False):
        self.device = torch.device(device)
        self.show_viewer = show_viewer
        self.eval = eval
        self.num_frames = 1993 if self.eval else 241 #save shorter clips during training and longer clips during evaluation

        # Configuration parameters
        self._initialize_env_parameters(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)
        
        # Create simulation scene
        self._setup_scene(self.show_viewer)
        
        # Add terrain or plane
        self._add_terrain()
        
        # Add robot and configure it
        self._add_and_configure_robot()
        
        # Set up camera for recording
        self._set_camera()
        
        # Build the scene with specified number of environments
        self.scene.build(n_envs=num_envs)
        
        # Configure motor joints
        self._setup_motor_joints()
        
        # Prepare reward functions
        self._setup_reward_functions()
        
        # Initialize state buffers
        self._initialize_buffers()
        
        # Find link indices for different parts of the robot
        self._find_link_indices()

    def _initialize_env_parameters(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg):
        """Initialize environment parameters from configuration"""
        self.num_envs: int = num_envs
        self.num_obs: int = obs_cfg["num_obs"] #45
        self.num_privileged_obs: int = obs_cfg.get("num_priviliged_obs", None)  # privileged informations only for trainig for the critic network 
        self.num_actions: int = env_cfg["num_actions"] #12

        self.simulate_action_latency: bool = True  # there is a 1 step latency on real robot
        self.dt: float = 0.02  # control frequency on real robot is 50hz (0.02 = 1/50)
        self.max_episode_length: int = math.ceil(env_cfg["episode_length_s"] / self.dt) # 40/0.02 = 1500 steps; maximum number of environment steps allowed in one episode before a forced reset

        self.env_cfg: dict = env_cfg
        self.use_terrain = self.env_cfg.get('use_terrain', False)
        self.obs_cfg: dict = obs_cfg
        self.reward_cfg: dict = reward_cfg
        self.obs_scales: dict = obs_cfg["obs_scales"]
        self.reward_scales: dict = reward_cfg["reward_scales"]
        self.command_cfg: dict = command_cfg
        
        if self.command_cfg is not None:
            self.num_commands = self.command_cfg["num_commands"]
            self.lin_vel_x_range = self.command_cfg["lin_vel_x_range"]
            self.lin_vel_y_range = self.command_cfg["lin_vel_y_range"]
            self.ang_vel_range = self.command_cfg["ang_vel_range"]
            self.resampling_time = int(self.command_cfg.get("resampling_time_s", 4.0) / self.dt)
            
            self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)


        # Camera and recording related variables
        self.headless: bool = not self.show_viewer
        self._recording: bool = False
        self._recorded_frames_behind: list = []
        self._recorded_frames_side: list = []

    def _setup_scene(self, show_viewer):
        """Set up the simulation scene with appropriate options"""
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
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # Get reference to rigid solver
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
        """Add complex terrain with height field"""
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
        y_start = (self.terrain_cfg['n_subterrains'][1] * self.terrain_cfg['subterrain_size'][1]) / 2

        if self.terrain_cfg.get('randomize', False):
            x_start = (self.terrain_cfg['subterrain_size'][0]) / 2 #start at middle of first terrain 
            self.base_init_pos = torch.tensor([x_start, y_start, 0.45], device=self.device)
        else: 
            self.base_init_pos = torch.tensor([0.30, y_start, 0.35], device=self.device) # start at the beginning of the terrain(x starts at 0 but we add small margin, 0.35 is approx the height of the robot)

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
        
        self.relative_heights = torch.zeros((self.num_envs, self.height_patch_n_points), device=self.device, dtype=gs.tc_float) # init here
        self.reset_environment_at_random_terrain = self.terrain_cfg.get('reset_environment_at_random_terrain', False)
        # self.target_increased: bool = False # flag to indicate if the target has been increased in the curriculum learning


    def _add_simple_plane(self):
        """Add simple plane to the scene"""
        self.scene.add_entity(
            gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True),
        )

    def _add_and_configure_robot(self):
        """Add robot to the scene and configure its initial state"""
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat) # inverse of the initial orientation quaternion of the robot's base
        
        self.robot: RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                links_to_keep=self.env_cfg['links_to_keep'],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

    def _setup_motor_joints(self):
        """Set up motor joints and PD control parameters"""
        # names to indices
        self.motor_dofs: list[int] = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]# mapping from joint names to indices

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs) #apply kp to each joint
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)# apply kd to each joint

    def _setup_reward_functions(self):
        """Set up reward functions and scale rewards by dt"""
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt # we scale the rewards by dt bc 1 step != 1 second in the simulation
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

    def _initialize_buffers(self):
        """Initialize all state buffers used for observation and control"""
        # Initialize buffers for robot state
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        
        
        # Initialize observation buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.privileged_obs_buf = (
            None
            if self.num_privileged_obs is None
            else torch.zeros(
                (self.num_envs, self.num_privileged_obs),
                device=self.device,
                dtype=gs.tc_float,
            )
        )
        
        # Initialize control buffers
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)# buffer to store episode length for each environment
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        
        # Initialize joint state buffers
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        
        # Initialize pose buffers
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        # Default joint positions
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        
        # Contact forces
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )
        
        # Extra information for logging
        self.extras = dict()
        
        # For terrain heights if using terrain
        if self.use_terrain:
            self._initialize_terrain_heights()
            
        # Initialize feet-related buffers
        self._initialize_feet_buffers()

    def _initialize_terrain_heights(self):
        """Initialize terrain heights for terrain-based environments"""
        clipped_base_pos = self.base_pos[:, :2].clamp(min=torch.zeros(2, device=self.device), max=self.terrain_margin)
        height_field_ids = (clipped_base_pos / self.terrain_cfg['horizontal_scale'] - 0.5).floor().int()
        height_field_ids.clamp(min=0)
        self.terrain_heights = self.height_field[height_field_ids[:, 0], height_field_ids[:, 1]]

    def _initialize_feet_buffers(self):
        """Initialize buffers for foot positions, orientations, and velocities"""
        self.foot_positions = torch.ones(
            self.num_envs, 4, 3, device=self.device, dtype=gs.tc_float,
        ) #not used for now
        self.foot_quaternions = torch.ones(
            self.num_envs, 4, 4, device=self.device, dtype=gs.tc_float,
        )# not used for now
        self.foot_velocities = torch.ones(
            self.num_envs, 4, 3, device=self.device, dtype=gs.tc_float,
        )

    def _find_link_indices(self):
        """Find indices of important links like feet, contact points, etc."""
        def find_link_indices(names):
            """Finds the indices of the links in the robot that match the given names."""
            link_indices = list()

            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.termination_contact_link_indices = find_link_indices(
            self.env_cfg['termination_contact_link_names']
        ) # not used for now
        self.penalized_contact_link_indices = find_link_indices(
            self.env_cfg['penalized_contact_link_names']
        ) # indices of the links that we want to penalize for contact forces
        self.feet_link_indices = find_link_indices(
            self.env_cfg['feet_link_names']
        ) # indices of feet links

        ### foot related stuff ###
        self.feet_link_indices_world_frame = [i+1 for i in self.feet_link_indices]

    def step(self, actions):
        """Execute one step of simulation with the given actions"""
        self._process_actions(actions)
        self._update_robot_state()
        self._check_termination()
        self._compute_rewards()
        if self.use_terrain:
            self._compute_relative_heights() # compute the height field patch in front of the robot

        self._compute_observations()
        self._render_headless()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _process_actions(self, actions):
        """Process and apply actions to the robot"""
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]) #clippimng to prevent extreme values
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions  #simulate action latency
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos # transform normalized policy actions into target joint positions
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)# send target positions to robot's PID controllers
        self.scene.step() # advance the simulation by one step

    def _update_robot_state(self):
        """Update all robot state variables after simulation step"""
        # update buffers
        self.episode_length_buf += 1 # increment episode length
        self.last_base_pos[:] = self.base_pos[:] # store previous base position
        self.base_pos[:] = self.robot.get_pos() # get fresh robot base position
        self.base_quat[:] = self.robot.get_quat() # get fresh robot base orientation
        self.base_euler = quat_to_xyz( 
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        ) # get fresh robot base orientation in euler angles (roll, pitch, yaw)
        
        # Transform velocities to robot frame
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat) # transform world lin vel to robot's base frame
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)# transform world ang vel to robot's base frame
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat) # project gravity vector to robot's base frame
        
        # Update joint states
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)# update joint positions
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)# update joint velocities
        
        # Update foot states
        self.foot_positions[:] = self.rigid_solver.get_links_pos(self.feet_link_indices_world_frame)# update foot positions
        self.foot_quaternions[:] = self.rigid_solver.get_links_quat(self.feet_link_indices_world_frame) # update foot orientations
        self.foot_velocities[:] = self.rigid_solver.get_links_vel(self.feet_link_indices_world_frame)# update foot velocities
        
        # Update contact forces
        self.link_contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )# net force applied on each links due to direct external contacts, shape (num_envs, num_links, 3)

    def _check_termination(self):
        """Check termination conditions and reset environments if needed"""
        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length # if we reached 1500 steps, we need to reset the environment
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]# if pitch(forward/backward tilt) is greater than x degrees, we need to reset the environment
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]# if roll(side-to-side tilt) is greater than x degrees, we need to reset the environment
        base_contact = torch.norm(
                self.link_contact_forces[:, self.termination_contact_link_indices[0], :], dim=-1
            ) > 1e-3
        self.reset_buf |= base_contact # reset if the base is in contact with the ground

        if self.use_terrain:
            self._check_terrain_boundaries()

        # Handle timeouts
        self._handle_timeouts()
        
        # Reset environments that need resetting
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten()) # reset the environments that need to be reset

    def _check_terrain_boundaries(self):
        """Check if robot is within terrain boundaries for terrain-based environments"""
        # Reset if robot goes outside the terrain boundaries
        self.reset_buf |= (self.base_pos[:, 0] < 0.2)  # X min boundary
        self.reset_buf |= (self.base_pos[:, 1] < 0.2)  # Y min boundary
        self.reset_buf |= (self.base_pos[:, 0] > self.terrain_margin[0] - 0.1)  # X max boundary
        self.reset_buf |= (self.base_pos[:, 1] > self.terrain_margin[1] - 0.1)  # Y max boundary

    def _handle_timeouts(self):
        """Handle episode timeouts for PPO"""
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()# environments that have reached the max episode length
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0 # used by PPO to handle terminated vs. truncated episodes differently

    def _compute_rewards(self):
        """Compute rewards for all environments"""
        # TODO: think about introducing a curriculum penalty coefficient that increases the difficulty of the task over time(starts low)
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew


    def _compute_relative_heights(self) -> None:
        """Compute heights relative to the base of the robot for the height field patch in front of the robot."""
        # Expand local positions to all environments
        local_positions = self.height_patch_local_positions.unsqueeze(0).repeat(self.num_envs, 1, 1)  # Shape: (num_envs, 25, 3)

        # Repeat the quaternion for each point in the patch
        base_quat_repeated = self.base_quat.unsqueeze(1).repeat(1, 25, 1)  # Shape: (num_envs, 25, 4)

        # Flatten for transform_by_quat
        base_quat_flat = base_quat_repeated.view(-1, 4)  # Shape: (num_envs * 25, 4) = (102400, 4)
        local_positions_flat = local_positions.view(-1, 3)  # Shape: (num_envs * 25, 3) = (102400, 3)

        # Rotate local positions to world frame
        rotated_positions_flat = transform_by_quat(local_positions_flat, base_quat_flat)  # Shape: (num_envs * 25, 3)

        # Reshape back to (num_envs, 25, 3)
        rotated_positions = rotated_positions_flat.view(self.num_envs, 25, 3)  # Shape: (4096, 25, 3)

        # Translate to world coordinates by adding base position
        world_positions = self.base_pos.unsqueeze(1) + rotated_positions  # Shape: (num_envs, 25, 3)

        # Clamp positions to terrain boundaries (x, y only); #TODO: this seems unnecessary 
        clipped_positions = world_positions[:, :, :2].clamp(
            min=torch.zeros(2, device=self.device),
            max=self.terrain_margin
        )  # Shape: (num_envs, 25, 2)

        # Convert world positions to height field grid indices
        grid_indices = (clipped_positions / self.terrain_cfg['horizontal_scale'] - 0.5).floor().int()  # Shape: (num_envs, 25, 2)

        # Get height field dimensions
        width, height,  = self.height_field.shape

        # Clamp grid indices: x to [0, 95], y to [0, 47]; #TODO: this seems unnecessary
        grid_indices[:, :, 0] = torch.clamp(grid_indices[:, :, 0], 0, width - 1)  # x-indices from world x
        grid_indices[:, :, 1] = torch.clamp(grid_indices[:, :, 1], 0, height - 1) # y-indices from world y

        # Extract grid coordinates
        grid_x = grid_indices[:, :, 0]  # x-indices for rows (world x)
        grid_y = grid_indices[:, :, 1]  # y-indices for columns (world y)

        # Get terrain heights at these indices
        heights = self.height_field[grid_x, grid_y]  # Shape: (num_envs, 25)
        # Compute relative heights (terrain height - robot base height)
        relative_heights = heights - self.base_pos[:, 2].unsqueeze(1)  # Shape: (num_envs, 25)#

        self.relative_heights = relative_heights

    def _compute_observations(self):
        """Compute observations for agent"""
        scaled_commands = self.commands.clone()
        scaled_commands[:, 0] *= self.obs_scales['lin_vel']
        scaled_commands[:, 1] *= self.obs_scales['lin_vel']
        scaled_commands[:, 2] *= self.obs_scales['ang_vel']
        
        obs = [ 
                self.base_lin_vel * self.obs_scales['lin_vel'],                     # 3
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3, the robot's angular velocity in its base frame(3d)
                self.projected_gravity,  # 3, gravity vector in the robot's base frame, indicating its orientation
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12, current joint angles relative to default
                self.dof_vel * self.obs_scales["dof_vel"],  # 12, current joint velocities 
                self.actions,  # 12 # current actions issued by the policy
                self.commands,
                self.base_pos - self.last_base_pos,  # 3, difference between previous and current base position 
        ]
        if self.use_terrain:
            obs.append(self.relative_heights) # 25, height field patch in front of the robot
        self.obs_buf = torch.cat(obs, axis=-1)  # Shape: (num_envs, num_obs)


        # clip observations to prevent extreme values
        clip_obs = 100.0
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        if self.num_privileged_obs is not None:

            privileged_obs = [
                self.base_lin_vel * self.obs_scales['lin_vel'],                     # 3
                self.base_ang_vel * self.obs_scales['ang_vel'],                     # 3
                self.projected_gravity,                                             # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'], # 12, current joint angles relative to default
                self.dof_vel * self.obs_scales['dof_vel'],  # 12, current joint velocities
                self.last_dof_vel * self.obs_scales['dof_vel'],  # 12, previous joint velocities
                self.actions, # 12, current actions issued by the policy
                self.last_actions, # 12, previous actions
                self.base_pos - self.last_base_pos, # 3, difference between previous and current base position
            ]
            if self.use_terrain:
                privileged_obs.append(self.relative_heights)
   
            self.privileged_obs_buf = torch.cat(privileged_obs, axis=-1)
            
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

    def get_observations(self):
        """Get current observations"""
        return self.obs_buf

    def get_privileged_observations(self):
        """Get current privileged observations"""
        return self.privileged_obs_buf

    def reset_idx(self, envs_idx):
        """Reset specified environments"""
        if len(envs_idx) == 0:
            return

        self._reset_robot_state(envs_idx)
        self._reset_buffers(envs_idx)
        self._update_episode_stats(envs_idx)

    def _reset_robot_state(self, envs_idx):
        """Reset robot state for specified environments"""
        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        if self.reset_environment_at_random_terrain and self.target_increased:
            # Randomly select row indices for each environment to reset
            row_idx = torch.randint(0, self.terrain_cfg['n_subterrains'][1], (len(envs_idx),), device=self.device)
            subterrain_size_y = self.terrain_cfg['subterrain_size'][1]
            y_start = (row_idx + 0.5) * subterrain_size_y  # Center of the selected row

            self.base_pos[envs_idx, 0] = 0.40 # Fixed x position for all environments
            self.base_pos[envs_idx, 1] = y_start # Random y position based on selected row  
            self.base_pos[envs_idx, 2] = 0.40  # Fixed height above flat terrain
            self.last_base_pos[envs_idx] = self.base_pos[envs_idx].clone()
            # TODO: Do we maybe also need to adjust the base orientation (quat) based on the terrain?

        else:     
            self.base_pos[envs_idx] = self.last_base_pos[envs_idx] = self.base_init_pos 
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

    def _reset_buffers(self, envs_idx):
        """Reset buffers for specified environments"""
        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

    def _update_episode_stats(self, envs_idx):
        """Update episode statistics for specified environments"""
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        """Reset all environments"""
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None
    

    def increase_x_target(self, delta):
        """Increase the x target velocity by delta"""
        mask: torch.Tensor = self.commands[:, 0] < 5.0 # mask to select environments where the x target velocity is less than 2.5
        self.commands[mask, 0] += delta
        # self.target_increased = True # set the flag to indicate that the target has been increased
        print("Increased x target velocity by", delta)

    def resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids, 0] = gs_rand_float(*self.lin_vel_x_range, (len(env_ids),), self.device)
        self.commands[env_ids, 1] = gs_rand_float(*self.lin_vel_y_range, (len(env_ids),), self.device)
        self.commands[env_ids, 2] = gs_rand_float(*self.ang_vel_range, (len(env_ids),), self.device)

    def _set_camera(self):
        '''Set camera positions and directions for recording'''
        # Elevated behind view (original)
        self._floating_camera_behind = self.scene.add_camera(
            pos=np.array([-1.5, 0.0, 5.0]),  # Behind and elevated
            lookat=np.array([0, 0, 0.1]),    # Looking at the robot
            fov=45,                          
            GUI=False,
            res=(720, 720),               
        )
        
        # Side view for feet
        if self.eval:
            self._floating_camera_side = self.scene.add_camera(
                pos=np.array([0.0, -2.5, 1.5]),     # Side view: to the right and lower
                lookat=np.array([0, 0, 0.3]),       # Looking at robot's center/legs
                fov=45,                              
                GUI=False,
                res=(720, 720),                      
            )

    def _render_headless(self):
        '''Render frames for recording when in headless mode'''
        if self._recording and len(self._recorded_frames_behind) < self.num_frames:
            robot_pos = np.array(self.base_pos[0].cpu())
            
            # Behind camera
            self._floating_camera_behind.set_pose(
                pos=robot_pos + np.array([-1.5, 0.0, 2.5]),  # Position camera behind and above robot
                lookat=robot_pos + np.array([0.3, 0, 0.0])   # Look slightly ahead of the robot
            )
            frame_behind, _, _, _ = self._floating_camera_behind.render()
            self._recorded_frames_behind.append(frame_behind)
            
            # Side camera
            if self.eval:
                self._floating_camera_side.set_pose(
                    pos=robot_pos + np.array([0.0, -2.5, 1.0]),  # Side view following robot
                    lookat=robot_pos + np.array([0.0, 0, -0.1])  # Look at robot's feet level
                )
                frame_side, _, _, _ = self._floating_camera_side.render()
                self._recorded_frames_side.append(frame_side)

    def get_recorded_frames(self):
        '''Return the recorded frames and reset recording state'''
        print("We have recorded", len(self._recorded_frames_behind), "behind frames")
        if len(self._recorded_frames_behind) == self.num_frames - 1 and self.eval:
            frames_behind = self._recorded_frames_behind
            frames_side = self._recorded_frames_side
            self._recorded_frames_behind = []
            self._recorded_frames_side = []
            self._recording = False
            return frames_behind, frames_side
        elif len(self._recorded_frames_behind) == self.num_frames - 1 and not self.eval:
            frames_behind = self._recorded_frames_behind
            self._recorded_frames_behind = []
            self._recording = False
            return frames_behind

    def start_recording(self, record_internal=True):
        '''Start recording frames'''
        self._recorded_frames_behind = []
        self._recorded_frames_side = []
        self._recording = True
        if not record_internal:
            self._floating_camera_behind.start_recording()
            if self.eval:
                self._floating_camera_side.start_recording()

    def stop_recording(self, save_path_behind=None, save_path_side=None):
        '''Stop recording and optionally save to files'''
        self._recorded_frames_behind = []
        self._recorded_frames_side = []
        self._recording = False
        if save_path_behind is not None:
            print("fps", int(1 / self.dt))
            self._floating_camera_behind.stop_recording(save_path_behind, fps=int(1 / self.dt))
        if save_path_side is not None and self.eval:
            self._floating_camera_side.stop_recording(save_path_side, fps=int(1 / self.dt))
