from go2_env import Go2Env
import torch


class WalkRandomTerrain(Go2Env):

    # ------------------------------------------------------------------
    # Velocity tracking
    # ------------------------------------------------------------------

    def _reward_tracking_lin_vel_x(self):
        error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        """Tracking der Ziel-Drehrate — deckt yaw_drift mit ab."""
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    def _reward_alive(self):
        return (~self.reset_buf.bool()).float()

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_lin_vel_y(self):
        return torch.square(self.base_lin_vel[:, 1])

    # ------------------------------------------------------------------
    # Joint / action quality
    # ------------------------------------------------------------------

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)


    # ------------------------------------------------------------------
    # Feet / contact
    # ------------------------------------------------------------------

    def _reward_penalized_contact(self):
        penalized_forces = self.link_contact_forces[:, self.penalized_contact_link_indices, :]
        return torch.sum((torch.norm(penalized_forces, dim=-1) > 0.1).float(), dim=1)

