from go2_env import Go2Env
import torch 

# yaw is from left to right, pitch is from up to down, roll is rotating 

class RunFlatTerrain(Go2Env):

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

    def _reward_base_height(self):
        """Penalize deviation from target base height"""
        target = self.reward_cfg.get("base_height_target", 0.3)
        return torch.square(self.base_pos[:, 2] - target)

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
    
    def _reward_heading(self):
        # Bestraft Abweichung vom geraden Heading (yaw relativ zur Start-Orientierung).
        # Ersetzt _reward_sideway_movement: dort wurde die absolute Y-Position bestraft,
        # die bei hohem Tempo (großer Strecke pro Episode) sofort in den clamp(max=2)
        # läuft und damit kein Gradienten-Signal mehr liefert. Yaw direkt zu bestrafen
        # bekämpft die Ursache (Drift-Richtung) statt das Symptom (akkumulierte Distanz).
        # abs() statt square(): bei kleinen Drift-Winkeln (1-10°) liefert das quadrierte
        # Radiant-Signal ein zu schwaches Gradienten-Signal relativ zu anderen Reward-Termen.
        yaw = self.base_euler[:, 2]  # bereits relativ zur init-Orientierung (transform mit inv_base_init_quat)
        yaw_rad = yaw * (torch.pi / 180.0)
        return torch.abs(yaw_rad)
    
    def _reward_x_progress(self):
        # Reward for moving forward (to prevent model from standing still)
        return torch.clamp(self.base_pos[:, 0] - self.base_init_pos[0], max=1)
    
    def _reward_feet_slip(self):
        # Berechne die Kontaktkraft für jeden Fuß (Norm über X, Y, Z Kräfte)
        foot_forces = torch.norm(self.link_contact_forces[:, self.feet_link_indices, :], dim=-1)
        
        # Prüfe, ob der Fuß den Boden berührt (Schwellenwert z.B. > 1.0 Newton)
        contact = foot_forces > 1.0
        
        # Berechne die quadratische Geschwindigkeit der Füße in der X-Y Ebene
        foot_vel_xy = torch.sum(torch.square(self.foot_velocities[:, :, :2]), dim=-1)
        
        # Bestrafe die Geschwindigkeit, ABER nur für die Füße, die Bodenkontakt haben
        return torch.sum(contact * foot_vel_xy, dim=1)
    
    def _reward_dof_vel(self):
        # Bestrafe hohe Gelenkgeschwindigkeiten für eine flüssigere Bewegung
        return torch.sum(torch.square(self.dof_vel), dim=1)
