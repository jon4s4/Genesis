from go2_env import Go2Env
import torch

class SprintFlatTerrain(Go2Env):

    # ------------------------------------------------------------------
    # 1. Hauptziel: Geschwindigkeit & Richtung
    # ------------------------------------------------------------------

    def _reward_tracking_lin_vel_x(self):
        """Belohnt das exakte Halten der Vorwärts-Zielgeschwindigkeit."""
        error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        """Belohnt das Einhalten der gewünschten Drehgeschwindigkeit (Gieren/Yaw)."""
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    # ------------------------------------------------------------------
    # 2. Stabilität & Haltung (Sollte mit negativen Gewichten versehen werden)
    # ------------------------------------------------------------------

    def _reward_lin_vel_z(self):
        """Bestraft vertikales Hüpfen des Torsos — wichtig für effizientes Rennen."""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_lin_vel_y(self):
        """Bestraft seitliches Abdriften (besser als absolute Positionsbestrafung)."""
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_ang_vel_xy(self):
        """Bestraft Nicken (Pitch) und Rollen (Roll) des Torsos."""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """Bestraft eine geneigte Basis (Körper sollte waagerecht bleiben)."""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # ------------------------------------------------------------------
    # 3. Energie & Geschmeidigkeit (Kleine negative Gewichte!)
    # ------------------------------------------------------------------

    def _reward_action_rate(self):
        """Bestraft ruckartige Gelenkbewegungen von einem Step zum nächsten."""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_smoothness(self):
        """Bestraft abrupte Beschleunigungsänderungen in den Gelenken (Jerk)."""
        return torch.sum(
            torch.square(self.actions - 2.0 * self.last_actions + self.last_last_actions), dim=1
        )

    def _reward_torques(self):
        """Bestraft zu hohen Drehmomentverbrauch (fördert Energieeffizienz)."""
        return torch.sum(torch.square(self.torques), dim=1)

    # ------------------------------------------------------------------
    # 4. Füße & Kontakt (Entscheidend für Trab/Galopp)
    # ------------------------------------------------------------------

    def _reward_feet_air_time(self):
        """Belohnt angemessene Flugzeiten der Füße (zwingt den Roboter in den Trab/Galopp)."""
        contact = self.feet_contact                          # (num_envs, 4) bool
        first_contact = (self.feet_air_time > 0) & contact
        self.feet_air_time += self.dt

        # Ziel-Flugzeit skaliert mit der Vorwärtsgeschwindigkeit
        target_air_time = 0.18 + 0.3 * torch.abs(self.commands[:, 0]).unsqueeze(1)

        reward = torch.sum(
            (self.feet_air_time - target_air_time).clip(max=0.0) * first_contact.float(),
            dim=1,
        )
        self.feet_air_time *= (~contact).float()
        return reward

    def _reward_feet_slip(self):
        """Bestraft das Rutschen der Füße auf dem Boden."""
        contact = self.feet_contact.float() # (num_envs, 4)
        feet_xy_speed = torch.norm(self.foot_velocities[:, :, :2], dim=-1)  # (num_envs, 4)
        return torch.sum(feet_xy_speed * contact, dim=1)
    
    def _reward_penalized_contact(self):
        """Bestraft Bodenkontakt von allem, was kein Fuß ist (Oberschenkel, Unterschenkel, Torso)."""
        penalized_forces = self.link_contact_forces[:, self.penalized_contact_link_indices, :]
        return torch.sum((torch.norm(penalized_forces, dim=-1) > 0.1).float(), dim=1)

    # ------------------------------------------------------------------
    # 5. Überleben & Bestrafung bei Sturz
    # ------------------------------------------------------------------

    def _reward_alive(self):
        """Gibt einen kleinen Bonus für jeden überlebten Step (hält den Roboter am Anfang aufrecht)."""
        return (~self.reset_buf.bool()).float()

    def _reward_termination(self):
        """Gibt eine dicke Strafe, wenn der Roboter umfällt (nicht bei Timeouts!)."""
        non_timeout_reset = (self.reset_buf == 1) & (self.episode_length_buf <= self.max_episode_length)
        return non_timeout_reset.float()