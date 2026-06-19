from go2_env import Go2Env
import torch 

class SlalomObstacleTerrain(Go2Env):

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
    
    def _reward_obstacle_avoidance(self):
        # Bestrafe den Roboter extrem, wenn er einer Stange zu nahe kommt
        # Berechne Distanz zu allen Stangen
        dists = torch.norm(self.base_pos[:, :2].unsqueeze(1) - self.pole_positions, dim=-1)
        min_dist = torch.min(dists, dim=1)[0]
        
        # Wenn Distanz kleiner als 0.4m (Kollisionsradius), gib starken negativen Reward
        return torch.where(min_dist < 0.4, -1.0, 0.0)

    def _reward_slalom_progress(self):
        # Belohne die tatsächliche Geschwindigkeit in X-Richtung
        # Das ist stabiler als die absolute Position
        return self.base_lin_vel[:, 0]

    def _reward_obstacle_collision(self):
        # Berechne Distanz zur nächsten Stange (hast du in obs berechnet)
        # Nutze relative_pole_pos, die wir in _compute_observations hinzugefügt haben
        # Wir nehmen den Vektor zur nächsten Stange (Spalte 8-9 in obs_buf oder direkt berechnen)
        
        # Beispiel via direkter Distanzberechnung zu allen Stangen:
        dists = torch.norm(self.base_pos[:, :2].unsqueeze(1) - self.pole_positions, dim=-1)
        min_dist = torch.min(dists, dim=1)[0]
        
        # Sanftere Strafe: Je näher er kommt, desto mehr Abzug (Gradient statt hartem Cut)
        # Alles unter 0.4m gilt als kritisch
        collision_penalty = torch.where(min_dist < 0.4, 1.0 - (min_dist / 0.4), torch.zeros_like(min_dist))
        return -collision_penalty
    
    def _reward_corridor_boundary(self):
        """Bestraft den Roboter, wenn er die Slalomgasse verlässt."""
        max_allowed_y = 1.0  
        
        excess_y = torch.clamp(torch.abs(self.base_pos[:, 1]) - max_allowed_y, min=0.0)
        
        # NEU: Cappe den maximalen excess_y Wert bei z.B. 1.5 Metern.
        # Verhindert, dass ein weggeschleuderter Roboter astronomische Strafen generiert.
        excess_y = torch.clamp(excess_y, max=1.5)
        
        return torch.square(excess_y)