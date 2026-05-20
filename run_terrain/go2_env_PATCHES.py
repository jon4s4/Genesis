"""
HINZUFÜGEN zu go2_env.py - Diese Methoden fehlen noch für vollständigen Terrain-Support
"""

# ============================================================================
# 1. In der __init__ Methode nach self._initialize_buffers() hinzufügen:
# ============================================================================

# Nach Zeile 30 (nach self._initialize_buffers()):
# self._setup_height_map()

# ============================================================================
# 2. NEUE METHODE: _setup_height_map
# ============================================================================

def _setup_height_map(self):
    """Initialize height map buffer for terrain observations."""
    self.height_map = torch.zeros(
        (self.num_envs, 1, 8, 8),  # (batch, channels, height, width)
        device=self.device,
        dtype=gs.tc_float
    )
    self.height_map_radius = 0.5  # Look at 0.5m around robot
    self.height_map_resolution = 8  # 8x8 grid

# ============================================================================
# 3. NEUE METHODE: _update_height_map (in step() aufrufen)
# ============================================================================

def _update_height_map(self):
    """
    Aktualisiere die lokale Height Map um jeden Roboter.
    Dies ist eine vereinfachte Version - die exakte Implementierung 
    hängt von Genesis terrain APIs ab.
    """
    if not self.use_terrain:
        return
    
    # Sampling-Punkte um den Roboter generieren
    num_cells = self.height_map_resolution
    radius = self.height_map_radius
    
    # Grid von -radius bis +radius
    x_offsets = torch.linspace(-radius, radius, num_cells, device=self.device)
    y_offsets = torch.linspace(-radius, radius, num_cells, device=self.device)
    
    height_samples = torch.zeros(
        (self.num_envs, num_cells, num_cells),
        device=self.device,
        dtype=gs.tc_float
    )
    
    # Für jeden Roboter
    for env_idx in range(self.num_envs):
        for i, x_offset in enumerate(x_offsets):
            for j, y_offset in enumerate(y_offsets):
                # Sampling-Position im globalen Frame
                sample_pos_x = self.base_pos[env_idx, 0] + x_offset
                sample_pos_y = self.base_pos[env_idx, 1] + y_offset
                
                # Height an dieser Position abtasten
                # ACHTUNG: Das hängt von Genesis terrain implementation ab!
                # Diese Zeile ist PSEUDOCODE und muss angepasst werden:
                height = self.scene.get_terrain_height(
                    env_idx, sample_pos_x, sample_pos_y
                )
                height_samples[env_idx, i, j] = height
    
    # Normalisieren und in Buffer speichern
    self.height_map = (height_samples - self.base_pos[:, 2:3]).unsqueeze(1)

# ============================================================================
# 4. MODIFIZIERTE METHODE: _compute_observations
# ============================================================================

"""
ERSETZE die _compute_observations() Methode mit dieser Version:

def _compute_observations(self):
    self.base_lin_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
    self.base_lin_vel = transform_by_quat(self.inv_base_init_quat, self.base_lin_vel)
    self.base_ang_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)  # TYPO?
    self.base_ang_vel = transform_by_quat(self.inv_base_init_quat, self.base_ang_vel)
    self.projected_gravity = transform_by_quat(self.inv_base_init_quat, self.global_gravity)

    obs_list = [
        self.base_lin_vel * self.obs_scales["lin_vel"],                    
        self.base_ang_vel * self.obs_scales["ang_vel"],                    
        self.projected_gravity,                                                    
        (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],       
        self.dof_vel * self.obs_scales["dof_vel"],                                
        self.actions,                                                              
        self.commands,                                                             
        self.base_pos - self.last_base_pos,
    ]
    
    # === NEU: Height Map hinzufügen ===
    if self.use_terrain and self.height_map is not None:
        obs_list.append(
            self.height_map.view(self.num_envs, -1) * self.obs_scales.get("height_map", 1.0)
        )
    
    self.obs_buf = torch.clip(torch.cat(obs_list, dim=-1), -100.0, 100.0)

    if self.num_privileged_obs is not None:
        priv_list = [
            self.base_lin_vel   * self.obs_scales["lin_vel"],                     
            self.base_ang_vel   * self.obs_scales["ang_vel"],                     
            self.projected_gravity,                                                
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],   
            self.dof_vel        * self.obs_scales["dof_vel"],                     
            self.last_dof_vel   * self.obs_scales["dof_vel"],                     
            self.actions,                                                          
            self.last_actions,                                                     
            self.commands,                                                         
            self.base_pos - self.last_base_pos,
        ]
        
        # === NEU: Height Map auch in privileged obs ===
        if self.use_terrain and self.height_map is not None:
            priv_list.append(
                self.height_map.view(self.num_envs, -1) * self.obs_scales.get("height_map", 1.0)
            )
        
        self.privileged_obs_buf = torch.clip(torch.cat(priv_list, dim=-1), -100.0, 100.0)
"""

# ============================================================================
# 5. NEUE METHODE: increase_terrain_difficulty (für Curriculum)
# ============================================================================

def increase_terrain_difficulty(self):
    """
    Erhöhe die Terrain-Schwierigkeit graduell.
    Wird vom OnPolicyRunner aufgerufen für Curriculum Learning.
    """
    if not self.use_terrain:
        return
    
    terrain_cfg = self.env_cfg.get('terrain_cfg', {})
    current_scale = terrain_cfg.get('curriculum_vertical_scale', 0.005)
    max_scale = terrain_cfg.get('curriculum_max_vertical_scale', 0.05)
    
    # 20% Anstieg, aber mit maximum
    new_scale = min(current_scale * 1.2, max_scale)
    
    # Update in config
    self.env_cfg['terrain_cfg']['curriculum_vertical_scale'] = new_scale
    
    print(f"\n{'='*60}")
    print(f"🏔️  TERRAIN DIFFICULTY INCREASED")
    print(f"   vertical_scale: {current_scale:.5f} → {new_scale:.5f}")
    print(f"   Progress: {100 * (new_scale - 0.005) / (max_scale - 0.005):.1f}%")
    print(f"{'='*60}\n")

# ============================================================================
# 6. NEUE METHODE: update_curriculum (in OnPolicyRunner aufrufen)
# ============================================================================

def update_terrain_curriculum(self, mean_reward, iteration, curriculum_interval=500):
    """
    Zentralisierte Curriculum-Control für Terrain.
    
    Args:
        mean_reward: Durchschnittlicher Reward über letzte Episoden
        iteration: Aktuelle Trainings-Iteration
        curriculum_interval: Alle N Iterationen ein Update
    """
    if not self.use_terrain:
        return
    
    if iteration % curriculum_interval == 0 and iteration > 0:
        # Strategie 1: Periodisch schwieriger machen
        self.increase_terrain_difficulty()
        
        # Strategie 2: Basierend auf Performance (optional)
        # if mean_reward > 0.8:  # Wenn Roboter gut läuft
        #     self.increase_terrain_difficulty()

# ============================================================================
# 7. MODIFIZIERTE METHODE: step (Height Map Update hinzufügen)
# ============================================================================

"""
In der step() Methode, nach dem Simulator-Schritt hinzufügen:

# Nach self.robot.control_dofs_position(self.actions, self.motor_dofs):
self.scene.step()

# === NEU: Height Map aktualisieren ===
self._update_height_map()

# Rest der step() Methode...
"""

# ============================================================================
# 8. VERBESSERUNG: Bessere Terrain-Generierung in _add_terrain
# ============================================================================

"""
ERSETZE _add_terrain() mit dieser verbesserten Version:

def _add_terrain(self):
    if self.use_terrain:
        terrain_cfg = self.env_cfg.get('terrain_cfg', {})
        
        # Genesis terrain options
        terrain_options = {
            'num_rows': terrain_cfg.get('n_subterrains', (8, 8))[0],
            'num_cols': terrain_cfg.get('n_subterrains', (8, 8))[1],
            'subterrain_size': terrain_cfg.get('subterrain_size', (25.0, 12.0)),
            'terrain_type': terrain_cfg.get('subterrain_types', 'random'),
            'vertical_scale': terrain_cfg.get('vertical_scale', 0.005),
            'horizontal_scale': terrain_cfg.get('horizontal_scale', 0.25),
        }
        
        # Erstelle Terrain
        terrain = gs.morphs.Terrain(
            name='terrain',
            terrain_type=terrain_options['terrain_type'],
            num_rows=terrain_options['num_rows'],
            num_cols=terrain_options['num_cols'],
            subterrain_size=terrain_options['subterrain_size'],
            vertical_scale=terrain_options['vertical_scale'],
            horizontal_scale=terrain_options['horizontal_scale'],
        )
        
        self.scene.add_entity(terrain)
        
        # Start-Position anpassen
        pos = self.env_cfg.get("base_init_pos", [0.0, 0.0, 0.42])
        self.base_init_pos = torch.tensor(pos, device=self.device)
    else:
        # Flaches Terrain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        pos = self.env_cfg.get("base_init_pos", [0.0, 0.0, 0.42])
        self.base_init_pos = torch.tensor(pos, device=self.device)
"""

# ============================================================================
# ZUSAMMENFASSUNG DER ÄNDERUNGEN
# ============================================================================

"""
Zu modifizierende Zeilen in go2_env.py:

1. LINE 30: Nach _initialize_buffers() hinzufügen:
   self._setup_height_map()

2. Neue Methode hinzufügen: _setup_height_map()

3. Neue Methode hinzufügen: _update_height_map()

4. step() Methode: Nach self.scene.step() hinzufügen:
   self._update_height_map()

5. _compute_observations() ersetzen mit neuer Version (Height Map)

6. _add_terrain() optional verbessern (siehe oben)

7. Neue Methode hinzufügen: increase_terrain_difficulty()

8. Neue Methode hinzufügen: update_terrain_curriculum()
"""
