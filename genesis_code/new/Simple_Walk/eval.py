import argparse
import os
import pickle
import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from simple_reward_wrapper import WalkRandomTerrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-uneven")
    parser.add_argument("-r", "--record",   action="store_true", default=True)
    parser.add_argument("--ckpt",           type=int, default=100)
    # Optional: override the commanded forward speed for evaluation
    parser.add_argument("--vel_x",          type=float, default=2.0,  help="Commanded forward velocity (m/s)")
    parser.add_argument("--vel_y",          type=float, default=0.0,  help="Commanded lateral velocity (m/s)")
    parser.add_argument("--ang_vel",        type=float, default=0.0,  help="Commanded yaw rate (rad/s)")
    args = parser.parse_args()

    # Use CPU backend to avoid Vulkan crashes
    gs.init(backend=gs.gpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    # Disable all reward computation during evaluation
    reward_cfg["reward_scales"] = {}

    # Optionally fix the terrain for reproducible evaluation
    if "terrain_cfg" in env_cfg:
        env_cfg["terrain_cfg"]["randomize"] = False
        # Uncomment to test specific terrain sequences:
        # env_cfg["terrain_cfg"]["n_subterrains"] = (4, 1)
        # env_cfg["terrain_cfg"]["subterrain_types"] = [
        #     ["wave_terrain"], ["pyramid_sloped_terrain"],
        #     ["pyramid_stairs_terrain"], ["stairs_terrain"],
        # ]

    env = WalkRandomTerrain(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        eval=True,
    )

    # Match the device used during training (CPU)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="mps")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="mps")

    # Fix the evaluation command so we can test a specific speed
    env.reset()
    env.commands[:, 0] = args.vel_x    # forward velocity
    env.commands[:, 1] = args.vel_y    # lateral velocity
    env.commands[:, 2] = args.ang_vel  # yaw rate
    # Disable random resampling during eval by making resampling_time huge
    env.resampling_time = int(1e9)

    obs = env.get_observations()
    n_frames = 0

    if args.record:
        env.start_recording(record_internal=False)

    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            n_frames += 1

            if args.record and n_frames == 2000:
                env.stop_recording(
                    save_path_behind=f"{args.exp_name}_{args.ckpt}_behind.mp4",
                    save_path_side=f"{args.exp_name}_{args.ckpt}_side.mp4",
                )
                print(f"Saved recordings for checkpoint {args.ckpt}.")
                break


if __name__ == "__main__":
    main()

"""
Usage examples:
  python eval.py -e go2-fast-v1 -r --ckpt 1000 --vel_x 2.5
  python eval.py -e go2-fast-v1 -r --ckpt 2000 --vel_x 3.0 --ang_vel 0.3
"""
