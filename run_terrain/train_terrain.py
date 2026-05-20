import argparse
import json
import os
import pickle
import shutil

import genesis as gs
gs.init(backend=gs.gpu)

import wandb
from reward_wrapper_terrain import SprintFlatTerrain
from rsl_rl.runners import OnPolicyRunner

import random


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.001,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 48,
            "policy_class_name": "ActorCritic",
            "record_interval": 200,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 200,
            "init_at_random_ep_len": False,
            "curriculum": True,
            "curriculum_delta": 0.02,
            "curriculum_threshold": 0.85,
            # --- Neue Curriculum-Optionen für Terrain ---
            "terrain_curriculum": True,  # Aktiviert Terrain-Curriculum
            "terrain_curriculum_interval": 500,  # Alle 500 Iterationen terrain schwieriger machen
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        'links_to_keep': ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot',],
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 0.8,
            "RR_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "kp": 40.0,
        "kd": 1.0,
        "termination_if_roll_greater_than": 45,
        "termination_if_pitch_greater_than": 45,
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 8.0,
        "action_scale": 0.5,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        # === Terrain erst mal DEAKTIVIERT (Genesis API unterschiedlich) ===
        'use_terrain': False,  # Trainiere auf Flat wie Original - funktioniert!
        'terrain_cfg': {
            'subterrain_types': 'random_uniform_terrain',
            'n_subterrains': (4, 4),
            'subterrain_size': (25.0, 12.0),
            'horizontal_scale': 0.25,
            'vertical_scale': 0.005,
            'randomize': True,
            'reset_environment_at_random_terrain': True,
            'curriculum_vertical_scale': 0.005,
            'curriculum_max_vertical_scale': 0.05,
        },
        'termination_contact_link_names': ['base'],
        'penalized_contact_link_names': ['base', 'thigh', 'calf'],
        'feet_link_names': ['foot'],
        'base_link_name': ['base'],
    }

    obs_cfg = {
        "num_obs": 51,  # Nur 51 (kein Height Map, da Genesis API unterschiedlich)
        "num_priviliged_obs": 75,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.1,
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.3,
        "reward_scales": {
            "tracking_lin_vel_x":     2.0,    # Wichtigster Reward
            "lin_vel_y":             -0.1,    # Weniger wichtig
            "paper_lateral_drift":   -0.2,    # REDUZIERT (vorher -0.4) - auf Terrain schwerer
            "paper_height":          -20.0,   # REDUZIERT (vorher -50.0) - auf Terrain schwerer
            "tracking_ang_vel":       0.4,    # Bleibt gleich
            "action_rate":           -0.005,  # Bleibt gleich
            "lin_vel_z":             -1.0,    # REDUZIERT (vorher -2.0) - normales Hüpfen erlaubt
            "feet_slip":             -0.05,   # NEU: Auf Terrain wichtig!
            "feet_air_time":          0.1,    # NEU: Besserer Schrittrhythmus
            "penalized_contact":     -0.05,   # NEU: Keine Bodenkollisionen außer Füße
        },
    }

    command_cfg = {
        "num_commands": 3,
        # === REDUZIERT FÜR TERRAINS ===
        "lin_vel_x_range": [0.3, 3.0],      # Startet langsamer bei Terrain
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-0.0, 0.0],
        "resampling_time_s": 4.0,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="test")
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=3000)  # Mehr Iterationen für Terrain
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument('--ckpt', type=int, default=1000)
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    all_cfgs = {
        "env_cfg": env_cfg,
        "obs_cfg": obs_cfg,
        "train_cfg": train_cfg,
        "reward_cfg": reward_cfg,
        "command_cfg": command_cfg,
        "num_envs": args.num_envs,
    }
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(all_cfgs, f, indent=4)

    env = SprintFlatTerrain(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
    )

    runner = OnPolicyRunner(
        env, 
        train_cfg, 
        log_dir, 
        device="mps", 
        curriculum=train_cfg["runner"]["curriculum"], 
        delta=train_cfg["runner"]["curriculum_delta"], 
        curriculum_threshold=train_cfg["runner"]["curriculum_threshold"]
    )

    if args.resume is not None:
        resume_dir = f'logs/{args.resume}'
        resume_path = os.path.join(resume_dir, f'model_{args.ckpt}.pt')
        print('==> resume training from', resume_path)
        runner.load(resume_path)

    wandb.init(project='genesis', name=args.exp_name, dir=log_dir, mode='offline')
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=train_cfg["runner"]["init_at_random_ep_len"])

if __name__ == "__main__":
    main()

"""
Verwendung:
python train_terrain.py -e go2-terrain -B 4096 --max_iterations 3000

Zum Fortsetzen:
python train_terrain.py -e go2-terrain-resume -B 4096 --max_iterations 1000 --resume go2-terrain --ckpt 1000
"""
