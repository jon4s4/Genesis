import argparse
import os
import pickle
import torch
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from simple_reward_wrapper import WalkRandomTerrain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-uneven")
    parser.add_argument("-r", "--record", action="store_true", default=True)
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}
    env_cfg["terrain_cfg"]["randomize"]=False
    # Here you can modify the terrain configuration to test different terrains
    # env_cfg["terrain_cfg"]["n_subterrains"] = (4, 1) 
    # env_cfg["terrain_cfg"]["subterrain_types"] = [["wave_terrain"], ["pyramid_sloped_terrain"], ["pyramid_stairs_terrain"], ["stairs_terrain"]]
    env = WalkRandomTerrain(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        command_cfg=command_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
        eval=True
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device='cuda:0')
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    ### recording ###
    env.reset()
    obs = env.get_observations()
    with torch.no_grad():
        stop = False
        n_frames = 0
        if args.record:
            env.start_recording(record_internal=False)
        while not stop:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            n_frames += 1
            if args.record:
                if n_frames == 2000:
                    env.stop_recording(
                        f"{args.exp_name}_{args.ckpt}_different_terrains_behind_view.mp4",
                        f"{args.exp_name}_{args.ckpt}_different_terrains_side_view.mp4"
                    )
                    exit()


if __name__ == "__main__":
    main()

"""
python eval_walk_random_terrain.py -e go2-fractal-adaptive-curriculum-big-smaller-threshold-manual-increase -r --ckpt 2800
python eval_walk_random_terrain.py -e tesuto -r --ckpt 312
"""
