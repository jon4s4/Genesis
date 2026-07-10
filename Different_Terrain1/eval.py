import argparse
import os
import pickle
import matplotlib.pyplot as plt
import torch

import genesis as gs
gs.init(backend=gs.gpu, precision="64")  # Fix aus letztem Turn: muss zum Training passen
from rsl_rl.runners import OnPolicyRunner
from reward_wrapper import RunFractalTerrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="test")
    parser.add_argument("-r", "--record",   action="store_true", default=True)
    parser.add_argument("--ckpt",           type=int, default=100)
    parser.add_argument("--vel_x",          type=float, default=2.0)
    parser.add_argument("--vel_y",          type=float, default=0.0)
    parser.add_argument("--ang_vel",        type=float, default=0.0)
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    reward_cfg["reward_scales"] = {}

    # --- NEU: Multi-Terrain-Override für diesen Eval-Run ---
    env_cfg["terrain_cfg"]["n_subterrains"] = (3, 1)
    env_cfg["terrain_cfg"]["subterrain_size"] = (12.0, 12.0)  # 15m je Segment -> 45m gesamt
    env_cfg["terrain_cfg"]["subterrain_types"] = [
        ["fractal_terrain"],
        ["pyramid_stairs_terrain"],
        ["flat_terrain"],
    ]
    env_cfg["terrain_cfg"]["randomize"] = False
    env_cfg["terrain_cfg"]["subterrain_parameters"] = {
        "pyramid_stairs_terrain": {
            "step_width": 0.4,
            "step_height": -0.04,  # Vorzeichen/Schlüssel lokal verifizieren!
        }
    }
    # ---------------------------------------------------------

    env = RunFractalTerrain(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        eval=True,
        device="cuda",
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda")
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device="cuda")

    env.reset()
    device = env.commands.device
    current_commands = torch.zeros(3, device=device)
    target_commands = torch.tensor([args.vel_x, args.vel_y, args.ang_vel], device=device)
    accel_limits = torch.tensor([1.0, 0.0, 0.0], device=device)
    max_step_change = accel_limits * env.dt

    env.resampling_time = int(1e9)
    obs = env.get_observations()
    n_frames = 0

    if args.record:
        env.start_recording(record_internal=False)

    actual_speeds, target_speeds = [], []

    with torch.no_grad():
        while True:
            diff = target_commands - current_commands
            step_change = torch.clamp(diff, -max_step_change, max_step_change)
            current_commands += step_change
            env.commands[:, :3] = current_commands

            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            n_frames += 1

            actual_speeds.append(env.base_lin_vel[0, 0].item())
            target_speeds.append(env.commands[0, 0].item())

            # Genug Frames für ~45m bei vel_x m/s: anpassen falls Episode zu kurz/lang
            if args.record and n_frames == 1000:
                env.stop_recording(
                    save_path_behind=f"{args.exp_name}_{args.ckpt}_behind.mp4",
                    save_path_side=f"{args.exp_name}_{args.ckpt}_side.mp4",
                )
                print(f"Saved recordings for checkpoint {args.ckpt}.")
                break

    plt.figure(figsize=(10, 5))
    plt.plot(target_speeds, label="Ziel", linestyle="--", color="gray")
    plt.plot(actual_speeds, label="Tatsächlich", color="blue")
    plt.xlabel("Simulationsschritte")
    plt.ylabel("Geschwindigkeit x (m/s)")
    plt.title("Multi-Terrain Eval: Fraktal → Stufen → Flach")
    plt.legend(); plt.grid(True)
    plt.savefig(f"speed_plot_{args.ckpt}.png")


if __name__ == "__main__":
    main()
