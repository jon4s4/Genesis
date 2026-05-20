from go2_env import Go2Env
import torch

class SprintFlatTerrain(Go2Env):
    """
    Optimiert für Terrain-Laufen mit lockerem Höhen- und Orientierungsfeedback.
    """

    # ------------------------------------------------------------------
    # 1. Hauptziel: Geschwindigkeit & Richtung (WICHTIG!)
    # ------------------------------------------------------------------

    def _reward_tracking_lin_vel_x(self):
        """Belohnt das exakte Halten der Vorwärts-Zielgeschwindigkeit."""
        error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])


    def _reward_paper_velocity(self):
        error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.clamp(1.0 - error, min=0.0)

    def _reward_paper_energy_penalty(self):
        """Energy penalty: |tau * q_dot|"""
        power = torch.abs(self.torques * self.dof_vel)
        return torch.sum(power, dim=1)

    def _reward_paper_orientation(self):
        """Orientation penalty: ||q - (1, 0, 0, 0)||"""
        target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        return torch.norm(self.base_quat - target_quat, dim=1)

    def _reward_paper_lateral_drift(self):
        """Lateral drift penalty: |y| - Reduziert für Terrain."""
        return torch.abs(self.base_pos[:, 1] - self.base_init_pos[1])

    def _reward_lin_vel_y(self):
        """Bestraft seitliches Abdriften."""
        return torch.square(self.base_lin_vel[:, 1])
    

    def _reward_feet_air_time(self):
        """
        Belohnt optimale Flugzeit für effizientes Laufen.
        Auf Terrain ist dies weniger kritisch als auf flachem Terrain.
        """
        contact = self.feet_contact
        first_contact = (self.feet_air_time > 0) & contact
        self.feet_air_time += self.dt

        # Optimale Flugzeit: bei höherer Geschwindigkeit länger
        target_air_time = 0.15 + 0.15 * torch.abs(self.commands[:, 0]).unsqueeze(1)

        reward = torch.sum(
            self.feet_air_time.clip(max=target_air_time) * first_contact.float(),
            dim=1,
        )
        # Nur aktiv bei Vorwärtsbewegung
        reward *= (torch.abs(self.commands[:, 0]) > 0.1).float()
        self.feet_air_time *= (~contact).float()
        return reward
    

    def _reward_penalized_contact(self):
        """
        Bestraft Bodenkontakt außer bei den Füßen.
        WICHTIG für Terrain-Stabilität!
        """
        penalized_forces = self.link_contact_forces[:, self.penalized_contact_link_indices, :]
        contact_magnitude = torch.norm(penalized_forces, dim=-1)
        # Bestrafung wenn Kraft > 1 N (nicht nur > 0.1 N)
        return torch.sum((contact_magnitude > 1.0).float(), dim=1)


    def _reward_tracking_ang_vel(self):
        """Belohnt Drehgeschwindigkeit (Gieren)."""
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    # ------------------------------------------------------------------
    # 2. Stabilität & Haltung (LOCKERER auf Terrain!)
    # ------------------------------------------------------------------

    def _reward_lin_vel_z(self):
        """Bestraft vertikales Hüpfen - aber weniger streng auf Terrain."""
        # Auf Terrain ist etwas Hüpfen erlaubt
        return torch.square(self.base_lin_vel[:, 2])


    def _reward_ang_vel_xy(self):
        """Bestraft Nicken (Pitch) und Rollen (Roll)."""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """
        Bestraft Neigung des Körpers.
        Auf Terrain können kleine Neigungen ok sein.
        """
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # ------------------------------------------------------------------
    # 3. Energie & Geschmeidigkeit
    # ------------------------------------------------------------------

    def _reward_action_rate(self):
        """Bestraft ruckartige Gelenkbewegungen."""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_smoothness(self):
        """Bestraft abrupte Beschleunigungen (Jerk)."""
        return torch.sum(
            torch.square(self.actions - 2.0 * self.last_actions + self.last_last_actions), dim=1
        )

    def _reward_torques(self):
        """Bestraft hohen Drehmomentverbrauch."""
        return torch.sum(torch.square(self.torques), dim=1)

    # ------------------------------------------------------------------
    # 4. Füße & Kontakt (KRITISCH für Terrain!)
    # ------------------------------------------------------------------

    def _reward_feet_slip(self):
        """
        Bestraft Fuß-Rutschen - WICHTIG auf Terrain!
        Gleitende Füße verlieren Traktion.
        """
        contact = self.feet_contact.float()
        feet_xy_speed = torch.norm(self.foot_velocities[:, :, :2], dim=-1)
        
        # Sanfte Bestrafung: erlaubt bis zu 0.3 m/s Slip
        slip_penalty = torch.clamp(feet_xy_speed - 0.3, min=0.0)
        return torch.sum(slip_penalty * contact, dim=1)


    # ------------------------------------------------------------------
    # 5. Überleben & Bestrafung bei Sturz
    # ------------------------------------------------------------------

    def _reward_alive(self):
        """Bonus für jeden überlebten Schritt."""
        return (~self.reset_buf.bool()).float()

    def _reward_termination(self):
        """Strafe für Sturz (nicht für Timeout)."""
        non_timeout_reset = (self.reset_buf == 1) & (self.episode_length_buf <= self.max_episode_length)
        return non_timeout_reset.float()
    
    # ==================================================================
    # Paper-Based Rewards
    # ==================================================================

    def _reward_paper_height(self):
        """
        Height penalty: |z - 0.3|
        AUF TERRAIN LOCKERER! Roboter kann 0.25-0.35m Höhe haben.
        """
        target_height = 0.30
        height_error = torch.abs(self.base_pos[:, 2] - target_height)
        # Nur bestrafen wenn > 0.1m vom Ziel entfernt
        return torch.clamp(height_error - 0.1, min=0.0)

    # ==================================================================
    # NEUE REWARDS FÜR TERRAIN
    # ==================================================================

    def _reward_terrain_adaptation(self):
        """
        Bonus für stabiles Laufen auf unebenem Terrain.
        Gemessen als: (kein Sturz) + (kontrollierte Orientation).
        """
        # Nicht gestürzt (Roll/Pitch klein)
        roll = 2 * torch.asin(torch.clamp(self.base_quat[:, 2], -1, 1))
        pitch = 2 * torch.asin(torch.clamp(self.base_quat[:, 1], -1, 1))
        
        # Bonus wenn beide < 20°
        not_tilted = ((torch.abs(roll) < 0.35) & (torch.abs(pitch) < 0.35)).float()
        return not_tilted * 0.1  # Kleiner Bonus

    def _reward_height_map_feedback(self):
        """
        Bonus für erfolgreiche Höhenpassung.
        (Nur wenn height_map im obs verfügbar ist)
        """
        if hasattr(self, 'height_map') and self.height_map is not None:
            # Roboter wird belohnt wenn er Höhenänderungen adaptiert
            # durch stabile Velocity bei variierender Höhe
            height_variance = torch.var(self.height_map.view(self.num_envs, -1), dim=1)
            velocity_stability = torch.abs(self.base_lin_vel[:, 0] - self.commands[:, 0])
            
            # Je höher die Gelände-Varianz, desto wichtiger ist Stabilität
            reward = torch.exp(-0.5 * velocity_stability * (1.0 + height_variance))
            return reward
        else:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_climb_bonus(self):
        """
        Bonus wenn Roboter bergauf läuft.
        Fördert aggressive Navigations-Strategien.
        """
        if hasattr(self, 'height_map') and self.height_map is not None:
            # Lokale Höhenänderung vorne
            front_height = torch.mean(self.height_map[:, :, 3:6, 3:6], dim=(2, 3))  # Vorne-Mitte
            center_height = torch.mean(self.height_map[:, :, 2:4, 2:4], dim=(2, 3))  # Mitte
            
            height_ahead = front_height - center_height
            # Bonus wenn der Roboter bergauf mit Vorwärtsgeschwindigkeit läuft
            climbing = (height_ahead > 0.05) & (self.base_lin_vel[:, 0] > 1.0)
            return climbing.float() * 0.05
        else:
            return torch.zeros(self.num_envs, device=self.device)
