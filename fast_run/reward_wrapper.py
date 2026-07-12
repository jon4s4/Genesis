from go2_env import Go2Env
import torch 

# yaw is from left to right, pitch is from up to down, roll is rotating 

class RunFlatTerrain(Go2Env):

    def _reward_tracking_lin_vel_x(self):
        """Tracking of linear velocity commands (x axis)"""
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)"""
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity"""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_base_height(self):
        """Penalize deviation from target base height"""
        target = self.reward_cfg.get("base_height_target", 0.3)
        return torch.square(self.base_pos[:, 2] - target)

    def _reward_action_rate(self):
        """Penalize changes in actions"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        """Penalize joint poses far away from default pose"""
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_heading(self):
        """Replaces sideway_movement penalty, used for high speed"""
        yaw = self.base_euler[:, 2]
        yaw_rad = yaw * (torch.pi / 180.0)
        return torch.abs(yaw_rad)
    
