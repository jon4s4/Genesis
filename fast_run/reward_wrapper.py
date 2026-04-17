from go2_env import Go2Env
import torch

class SprintFlatTerrain(Go2Env):

    # ------------------------------------------------------------------
    # Velocity tracking
    # ------------------------------------------------------------------

    def _reward_tracking_lin_vel_x(self):
        error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    def _reward_lin_vel_z(self):
        # Bestraft vertikales Hüpfen des Torsos — wichtig beim schnellen Laufen
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_lin_vel_y(self):
        # Bestraft seitliches Abdriften
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_ang_vel_xy(self):
        # Bestraft Nicken und Rollen des Torsos
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    # ------------------------------------------------------------------
    # Joint / action quality
    # ------------------------------------------------------------------

    def _reward_action_rate(self):
        # Bestraft ruckartige Gelenkbewegungen → flüssigerer Gang
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # ------------------------------------------------------------------
    # Feet / contact  (optional, aber hilfreich für Stabilität)
    # ------------------------------------------------------------------

    def _reward_penalized_contact(self):
        # Bestraft Bodenkontakt von Oberschenkel/Unterschenkel/Torso
        penalized_forces = self.link_contact_forces[:, self.penalized_contact_link_indices, :]
        return torch.sum((torch.norm(penalized_forces, dim=-1) > 0.1).float(), dim=1)
    
    # def _reward_feet_air_time(self):
    #     # Belohne den Roboter, wenn die Füße in der Luft sind (längere Flugphase = schnelleres Rennen)
    #     # self.feet_air_time wird in der go2_env.py meistens mitgeführt
    #     return torch.sum(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) < 1.0, dim=1)

    def _reward_ang_vel_xy(self):
        """Penalise roll and pitch angular velocity."""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """Penalise tilted base via gravity projection onto x/y."""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        """Penalise large joint torques (energy efficiency)."""
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_action_rate(self):
        """Penalise abrupt action changes (1st-order)."""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_smoothness(self):
        """Penalise jerk — 2nd-order action differences."""
        return torch.sum(
            torch.square(self.actions - 2.0 * self.last_actions + self.last_last_actions), dim=1
        )

    def _reward_feet_air_time(self):
        """Reward appropriate foot air-time for dynamic trotting / galloping."""
        contact = self.feet_contact                          # (num_envs, 4)  bool
        first_contact = (self.feet_air_time > 0) & contact
        self.feet_air_time += self.dt

        # Target scales with commanded forward speed
        target_air_time = 0.18 + 0.18 * torch.abs(self.commands[:, 0]).unsqueeze(1)

        reward = torch.sum(
            (self.feet_air_time - target_air_time).clip(max=0.0) * first_contact.float(),
            dim=1,
        )
        self.feet_air_time *= (~contact).float()             # reset counter on touch-down
        return reward
    
    def _reward_penalized_contact(self):
        """Penalise contact on thigh and calf links."""
        penalized_forces = self.link_contact_forces[:, self.penalized_contact_link_indices, :]
        return torch.sum(
            (torch.norm(penalized_forces, dim=-1) > 0.1).float(), dim=1
    )

    def _reward_feet_slip(self):
        """Penalise lateral foot velocity while foot is in contact (slipping)."""
        contact = self.feet_contact.float() # (num_envs, 4)
        feet_xy_speed = torch.norm(self.foot_velocities[:, :, :2], dim=-1)  # (num_envs, 4)
        return torch.sum(feet_xy_speed * contact, dim=1)
    
    def _reward_similar_to_default(self):
        """Penalise joint poses far from the default standing pose."""
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_alive(self):
        """Belohne jeden Schritt, in dem der Roboter nicht terminiert."""
        return (~self.reset_buf.bool()).float()

    def _reward_joint_motion(self):
        """Encourage all joints to participate."""
        joint_vel_abs = torch.abs(self.dof_vel)
        mean_vel = joint_vel_abs.mean(dim=1, keepdim=True)
        underused = torch.clamp(mean_vel * 0.2 - joint_vel_abs, min=0.0)
        return underused.sum(dim=1)
    
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
