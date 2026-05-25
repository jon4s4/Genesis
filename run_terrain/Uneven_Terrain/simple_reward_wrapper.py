from simple_go2_env import Go2Env
import torch 

# yaw is from left to right, pitch is from up to down, roll is rotating 

class WalkRandomTerrain(Go2Env):

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_lin_vel_y(self):
        # Penalize y axis base linear velocity
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_termination(self):
        # penalize non timeout termination (falling over, collision)
        non_timeout_reset = (self.reset_buf == 1) & (self.episode_length_buf <= self.max_episode_length)
        return non_timeout_reset.float()
    
    def _reward_sideway_movement(self):
        # Penalize sideway movement away from the starting point
        return torch.clamp(torch.abs(self.base_pos[:, 1] - self.base_init_pos[1]), max=2)
    
    def _reward_x_progress(self):
        # Reward for moving forward (to prevent model from standing still)
        return torch.clamp(self.base_pos[:, 0] - self.base_init_pos[0], max=1)
    
    def _reward_paper_lateral_drift(self):
        """
        Lateral drift penalty: |y|
        Bestraft absolute seitliche Abweichung vom Startpunkt.
        """
        return torch.abs(self.base_pos[:, 1] - self.base_init_pos[1])
    
    def _reward_base_height(self):
        """
        AUF TERRAIN: Relative Höhe über dem Boden statt absoluter Höhe!

        Nutzt den mittleren Height-Map-Patch direkt unter dem Roboter.
        Ziel: ~0.34m über dem Boden (normale Stand-Höhe des Go2).
        """
        if self.use_terrain:
            # Mittlerer Patch in der 3x3 Grid = Index 4
            center_idx = (self.height_patch_n_x // 2) * self.height_patch_n_y + (self.height_patch_n_y // 2)
            # relative_heights[i, j] = terrain_height - base_pos.z  → negativ wenn Roboter über dem Boden
            height_above_ground = -self.relative_heights[:, center_idx]
            target_height = 0.34
            return torch.abs(height_above_ground - target_height)
        else:
            return torch.abs(self.base_pos[:, 2] - 0.34)