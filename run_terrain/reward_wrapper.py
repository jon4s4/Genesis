from go2_env import Go2Env
import torch

class SprintFlatTerrain(Go2Env):

    # ------------------------------------------------------------------
    # 1. Hauptziel: Geschwindigkeit
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
    # 2. Stabilität (TERRAIN-ANGEPASST)
    # ------------------------------------------------------------------

    def _reward_lin_vel_y(self):
        """Bestraft seitliches Abdriften."""
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_lin_vel_z(self):
        """
        Bestraft vertikales Hüpfen.
        AUF TERRAIN schwächer gewichten als auf flachem Boden,
        da der Roboter bei Steigungen zwangsläufig vertikal beschleunigt!
        Empfohlen: -0.5 statt -2.0
        """
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """
        Bestraft Nicken (Pitch) und Rollen (Roll).
        AUF TERRAIN leicht lockerer, da Steigungen Roll/Pitch erfordern.
        Empfohlen: -0.05 statt -0.1
        """
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """
        Bestraft Neigung via projected gravity.
        Besser als paper_orientation auf Terrain: toleriert Steigungen automatisch,
        weil der Gravitationsvektor im Körperframe bei echtem Aufrichten stabil bleibt.
        Empfohlen: -1.0
        """
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

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

    # ------------------------------------------------------------------
    # 3. Energie & Geschmeidigkeit
    # ------------------------------------------------------------------

    def _reward_action_rate(self):
        """
        Bestraft ruckartige Gelenkbewegungen.
        AUF TERRAIN wichtiger: Roboter neigt zu hektischen Korrekturen.
        Empfohlen: -0.01 statt -0.005
        """
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_smoothness(self):
        """Bestraft Jerk (Beschleunigungsänderung)."""
        return torch.sum(
            torch.square(self.actions - 2.0 * self.last_actions + self.last_last_actions), dim=1
        )

    def _reward_torques(self):
        """
        Bestraft Drehmomentverbrauch.
        AUF TERRAIN: Roboter braucht mehr Kraft → schwächer gewichten.
        Empfohlen: -0.0001 statt -0.0002
        """
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_sideway_movement(self):
        # Penalize sideway movement away from the starting point
        return torch.clamp(torch.abs(self.base_pos[:, 1] - self.base_init_pos[1]), max=2)
    # ------------------------------------------------------------------
    # 4. Füße & Kontakt
    # ------------------------------------------------------------------

    def _reward_feet_air_time(self):
        """
        Belohnt optimalen Schrittrhythmus.
        AUF TERRAIN: Kürzere Ziel-Flugzeit als auf flat terrain.
        """
        contact = self.feet_contact
        first_contact = (self.feet_air_time > 0) & contact
        self.feet_air_time += self.dt

        # AUF TERRAIN: 0.12 statt 0.18 als Basis
        target_air_time = 0.12 + 0.15 * torch.abs(
            self.commands[:, 0]
        ).unsqueeze(1)

        reward = torch.sum(
            self.feet_air_time.clip(max=target_air_time) * first_contact.float(),
            dim=1,
        )
        reward *= (torch.abs(self.commands[:, 0]) > 0.1).float()
        self.feet_air_time *= (~contact).float()
        return reward

    def _reward_feet_slip(self):
        """
        Bestraft Fuß-Rutschen.
        AUF TERRAIN sehr wichtig: schlechte Traktion führt zu Instabilität.
        Empfohlen: -0.1 (stärker als auf flat)
        """
        contact = self.feet_contact.float()
        feet_xy_speed = torch.norm(self.foot_velocities[:, :, :2], dim=-1)
        return torch.sum(feet_xy_speed * contact, dim=1)

    def _reward_penalized_contact(self):
        """
        Bestraft unerwünschte Kollisionen (Oberschenkel, Unterschenkel, Torso).
        AUF TERRAIN wichtiger als auf flat!
        Empfohlen: -1.0
        """
        penalized_forces = self.link_contact_forces[:, self.penalized_contact_link_indices, :]
        return torch.sum((torch.norm(penalized_forces, dim=-1) > 0.1).float(), dim=1)

    # ------------------------------------------------------------------
    # 5. Überleben & Sturz
    # ------------------------------------------------------------------

    def _reward_alive(self):
        """Bonus für jeden überlebten Step."""
        return (~self.reset_buf.bool()).float()

    def _reward_termination(self):
        """Strafe bei Sturz (nicht bei Timeout)."""
        non_timeout_reset = (self.reset_buf == 1) & (self.episode_length_buf <= self.max_episode_length)
        return non_timeout_reset.float()