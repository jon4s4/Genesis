from go2_env import Go2Env
import torch 

# yaw is from left to right, pitch is from up to down, roll is rotating 

class RunCurveTerrain(Go2Env):

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

class RunCurve(RunCurveTerrain):

    def _reward_progress(self):
        # rewards progress when running in the direction the robot is heading
        yaw_rad = self.base_euler[:, 2] * (torch.pi / 180.0)
        heading_dir = torch.stack([torch.cos(yaw_rad), torch.sin(yaw_rad)], dim=1)  # (num_envs, 2)
        step_delta = self.base_pos[:, :2] - self.last_base_pos[:, :2]
        forward_progress = torch.sum(step_delta * heading_dir, dim=1)
        return torch.clamp(forward_progress, max=1.0)

    def _reward_lateral_drift(self):
        # penalizes sideway movement, modified for curve running but only if angular velocity is between -0.5 and 0.5
        straight_mask = (torch.abs(self.commands[:, 2]) < 0.05).float()  # ~0 rad/s = "gerade"
        lateral_vel_sq = torch.square(self.base_lin_vel[:, 1])
        return lateral_vel_sq * straight_mask
