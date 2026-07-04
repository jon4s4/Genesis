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


class RunCurve(RunFlatTerrain):
    """
    Erweitert RunFlatTerrain um Kurvenlaufen (commands[:, 2] = Soll-Gierrate ω in rad/s).

    Kernidee: commands[:, 2] ist KEIN abruptes Richtungsziel, sondern eine konstante
    Drehrate, die über die ganze Episode (bzw. bis zum nächsten Resampling) gehalten
    wird. Bei konstantem v_x und konstantem ω läuft der Roboter geometrisch exakt auf
    einem Kreisbogen mit Radius r = v_x / ω - das ist der "schöne Bogen" ohne abrupten
    Richtungswechsel. Den abrupten Wechsel verhindern wir also nicht durch eine neue
    Reward-Formel, sondern dadurch, dass ω selbst über die Episode konstant bleibt und
    der command_cfg-Range so gewählt ist, dass r bei Zielgeschwindigkeit nicht zu klein
    (= zu scharfe Kurve) wird.

    Wichtigste Änderungen gegenüber RunFlatTerrain:
    - _reward_heading entfernt/überschrieben (würde aktives Kurvenlaufen direkt
      bestrafen, da es jede Yaw-Abweichung von der Start-Orientierung straft).
    - _reward_tracking_ang_vel wird (wie in der Basisklasse bereits vorbereitet)
      scharf genutzt, um commands[:, 2] zu verfolgen.
    - _reward_x_progress (nur globale X-Achse) ersetzt durch _reward_progress, das
      Fortschritt entlang der tatsächlichen Bewegungsbahn belohnt (heading-unabhängig),
      sonst würde Vorwärtslaufen in einer Kurve fälschlich nicht belohnt werden.
    - _reward_lateral_drift NUR aktiv, wenn ω-Kommando ~0 ist (gerade Strecke);
      während aktivem Kurvenkommando deaktiviert, weil seitliche Bewegung dort
      gewünscht/notwendig ist.
    """

    def _reward_heading(self):
        # Überschreibt die Basisklasse: KEIN Straf-Reward auf Yaw mehr, da Yaw bei
        # Kurvenlaufen aktiv und gewünscht von der Start-Orientierung abweicht.
        # Bleibt als Methode bestehen (falls reward_scales sie referenziert), gibt
        # aber konstant 0 zurück. In get_cfgs() sollte "heading" idealerweise ganz
        # aus reward_scales entfernt werden (siehe train.py).
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _reward_progress(self):
        # Belohnt Fortschritt ENTLANG der aktuellen Blickrichtung (heading), nicht die
        # rohe euklidische XY-Distanz. Reiner norm(step_delta) wäre exploitable: seitliches
        # Wackeln/Rutschen würde genauso belohnt wie echtes Vorwärtskommen. Stattdessen
        # projizieren wir die Positionsänderung auf den aktuellen Heading-Vektor - das
        # entspricht exakt base_lin_vel[:, 0] integriert über einen Step, ist aber direkt
        # aus Weltkoordinaten berechnet und bleibt damit auch bei Kurven (Y-Anteil in
        # Weltkoordinaten) korrekt forward-bezogen. Ersetzt _reward_x_progress, das nur
        # globale X-Bewegung zählte und bei aktivem Kurvenkommando kein Signal mehr gibt.
        yaw_rad = self.base_euler[:, 2] * (torch.pi / 180.0)
        heading_dir = torch.stack([torch.cos(yaw_rad), torch.sin(yaw_rad)], dim=1)  # (num_envs, 2)
        step_delta = self.base_pos[:, :2] - self.last_base_pos[:, :2]
        forward_progress = torch.sum(step_delta * heading_dir, dim=1)
        return torch.clamp(forward_progress, max=1.0)

    def _reward_lateral_drift(self):
        # Bestraft seitliche Geschwindigkeit im Körper-Frame (base_lin_vel[:, 1]),
        # ABER nur wenn kein/kaum Kurven-Kommando aktiv ist. Bei aktivem ω-Kommando
        # erzeugt das Kurvenlaufen selbst eine gewollte Kombination aus Vorwärts- und
        # geringer Lateralbewegung (durch Yaw-Rotation), die hier nicht bestraft werden
        # soll - sonst widerspricht dieser Reward direkt dem Kurven-Tracking-Reward.
        straight_mask = (torch.abs(self.commands[:, 2]) < 0.05).float()  # ~0 rad/s = "gerade"
        lateral_vel_sq = torch.square(self.base_lin_vel[:, 1])
        return lateral_vel_sq * straight_mask

    # _reward_tracking_lin_vel_x wird unverändert von RunFlatTerrain übernommen.
    # base_lin_vel[:, 0] ist bereits die Vorwärtsgeschwindigkeit im Körper-Frame
    # (siehe _update_robot_state: transform_by_quat mit inv_base_quat), bleibt also
    # auch während aktiver Drehung (Kurve) eine korrekte "wie schnell laufe ich
    # gerade nach vorne"-Messung - kein Override nötig.
