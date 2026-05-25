import argparse
import json
import os
import pickle
import shutil

import genesis as gs
gs.init(logging_level="warning")

import wandb
from simple_reward_wrapper import WalkRandomTerrain
from rsl_rl.runners import OnPolicyRunner
import random


# def create_random_terrains(seed: int = 42):
#     """Create a 5x5 terrain configuration with reproducible randomization."""
#     random.seed(seed)
#     terrain = [["fractal_terrain", "fractal_terrain", "fractal_terrain"]]

#     all_terrains = [
#         "wave_terrain",
#         # "fractal_terrain",
#         "pyramid_sloped_terrain",
#         "pyramid_stairs_terrain",
#         "flat_terrain"
#     ]
    
#     for _ in range(2):
#         shuffled_terrains = all_terrains.copy()
#         random.shuffle(shuffled_terrains)
#         terrain.append(shuffled_terrains[:3]) # take the first 3 terrains from the shuffled list

#     return terrain

def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2, # control how much the policy is allowed to change at each update step (increase => faster learning but riskier, decrease => slower but more stable)
            "desired_kl": 0.01, # how much change do I want between the old and new policy (using an adaptive schedule in this implementation)
            "entropy_coef": 0.01, # rewards randomness in action selection (might make sense to set it higher in early training and lower it later)
            "gamma": 0.99, # determines how much agent values future rewards 
            "lam": 0.95, # lambda parameter for GAE (Generalized Advantage Estimation); higher means advantages depend more on long-term returns, lower means more on short-term returns
            "learning_rate": 0.001, # is adaptive 
            "max_grad_norm": 1.0, # gradient clipping (to prevent exploding gradients)
            "num_learning_epochs": 5, # how often do we reuse one rollout batch 
            "num_mini_batches":4, # how many chunks the data is split into during training ((num_envs * num_steps_per_env) / num_mini_batches) 
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0, # weight of value loss in the total loss function; so 1.0 means value loss is equally important as policy loss
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],# try fewer
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
            "num_steps_per_env": 24, # how many steps to take in each environment before updating the policy (maybe increase this bc we have longer episodes now and could make more sense to sample more from the enviornment before updating the policy)
            "policy_class_name": "ActorCritic",
            "record_interval": 100,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 200,
            "init_at_random_ep_len": False,
            "curriculum": True, # whether to use curriculum learning
            "curriculum_delta": 0.05, # how much to increase the target linear velocity during curriculum learning
            "curriculum_threshold": 0.85 # the threshold for the mean of the last 20 tracking rewards to increase the target linear velocity 
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        'links_to_keep': ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot',],
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
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
        # PD
        "kp": 50.0, # proportional gain that multiplies the instantaneous position error (desired − actual joint angle) to produce a corrective torque
        "kd": 1.5, #  derivative gain that multiplies the time-derivative of the position error (angular velocity error) to generate a damping torque opposing motion
        # termination
        "termination_if_roll_greater_than": 20,  # degree
        "termination_if_pitch_greater_than": 30,  # degree
        # base pose
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s":40.0,
        # "resampling_time_s": 4.0, used for resampling commands and dynamics randomization
        "action_scale": 0.5, # this is smth like the amplitude knob that converts the policy's dimesionless output into real angles
        "simulate_action_latency": True,
        "clip_actions": 10.0, # self.actions = torch.clip(actions, -clip_actions, clip_actions), so it prevents the actions from going outside the range of -100 to 100 (which is too high)
        'use_terrain': True,
        'terrain_cfg': {
            'subterrain_types': 'fractal_terrain', #create_random_terrains(), # 5x5 grid of random subterrain types that each start with flat terrain
            'n_subterrains': (3, 1),
            'subterrain_size': (12.0, 12.0),
            'horizontal_scale': 0.25, # determines the number of scales per tile, so here 12/0.25 = 48 per tile
            'vertical_scale': 0.005,
            'randomize': False,
            'reset_environment_at_random_terrain': False, # whether to reset the environment at a random terrain
        },
        'termination_contact_link_names': ['base'],
        'penalized_contact_link_names': ['base', 'thigh', 'calf'],
        'feet_link_names': ['foot'],
        'base_link_name': ['base'],
    }

    obs_cfg = {
        "num_obs": 51,
        "num_priviliged_obs": 72,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.3,
        "reward_scales": {
            "tracking_lin_vel_x":  2.0,
            "lin_vel_y":          -0.1,
            "paper_lateral_drift":    -0.4,
            "paper_height":           -50.0,
            "tracking_ang_vel":    	0.4,
            "action_rate":        	-0.005,
            "lin_vel_z":          	-2.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 5.0],  # Intervall für Vorwärts (nicht bei 0 starten!)
        "lin_vel_y_range": [-0.0, 0.0], # Seitwärts erstmal blockieren
        "ang_vel_range": [-0.0, 0.0],   # Drehung erstmal blockieren
        "resampling_time_s": 4.0,       # Alle 4 Sekunden neues Kommando
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="test")
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=5)
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

    env = WalkRandomTerrain(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0", curriculum=train_cfg["runner"]["curriculum"], delta=train_cfg["runner"]["curriculum_delta"], curriculum_threshold=train_cfg["runner"]["curriculum_threshold"])

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
To only see one of the GPUs: export CUDA_VISIBLE_DEVICES=1 (or 0)
python train_walk_random_terrain.py -e go2-all-terrains-v0 -B 4096 --max_iterations 2000

python train_walk_random_terrain.py -e test -B 4096 --max_iterations 312
resume : 
python train_uneven.py -e go2-uneven-v4-resume -B 4096 --max_iterations 1000 --resume go2-uneven-v4 --ckpt 1000
"""
