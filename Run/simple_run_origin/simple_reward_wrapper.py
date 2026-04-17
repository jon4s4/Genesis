from simple_go2_env import Go2Env
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

    def _reward_smoothness(self):
        return torch.sum(
            torch.square(self.actions - 2.0 * self.last_actions + self.last_last_actions), dim=1
        )

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    # ------------------------------------------------------------------
    # Feet / contact
    # ------------------------------------------------------------------

    def _reward_feet_air_time(self):
        """Belohnt angemessene Flugphase pro Schritt."""
        contact = self.feet_contact
        first_contact = (self.feet_air_time > 0) & contact
        self.feet_air_time += self.dt
        target_air_time = (
            0.18 + 0.18 * torch.abs(self.commands[:, 0]).unsqueeze(1)
        ).clamp(max=0.45)
        reward = torch.sum(
            (self.feet_air_time - target_air_time).clip(max=0.0) * first_contact.float(),
            dim=1,
        )
        self.feet_air_time *= (~contact).float()
        return reward

    def _reward_feet_slip(self):
        contact = self.feet_contact.float()
        feet_xy_speed = torch.norm(self.foot_velocities[:, :, :2], dim=-1)
        return torch.sum(feet_xy_speed * contact, dim=1)

    def _reward_penalized_contact(self):
        penalized_forces = self.link_contact_forces[:, self.penalized_contact_link_indices, :]
        return torch.sum((torch.norm(penalized_forces, dim=-1) > 0.1).float(), dim=1)

    def _reward_hip_symmetry(self):
        """Verhindert asymmetrisches Ausschlagen der Hüften."""
        right_hips = self.dof_pos[:, [0, 6]]  # FR, RR
        left_hips  = self.dof_pos[:, [3, 9]]  # FL, RL
        return torch.sum(torch.square(right_hips + left_hips), dim=1)
