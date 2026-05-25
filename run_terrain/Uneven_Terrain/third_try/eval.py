import argparse
import os
import pickle
import matplotlib.pyplot as plt
import torch

import genesis as gs
gs.init(backend=gs.metal)
from rsl_rl.runners import OnPolicyRunner
from simple_reward_wrapper import WalkRandomTerrain
 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="test")
    parser.add_argument("-r", "--record",   action="store_true", default=True)
    parser.add_argument("--ckpt",           type=int, default=100)
    # Optional: override the commanded forward speed for evaluation
    parser.add_argument("--vel_x",          type=float, default=2.0,  help="Commanded forward velocity (m/s)")
    parser.add_argument("--vel_y",          type=float, default=0.0,  help="Commanded lateral velocity (m/s)")
    parser.add_argument("--ang_vel",        type=float, default=0.0,  help="Commanded yaw rate (rad/s)")
    args = parser.parse_args()


    log_dir = f"fourth_try/logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    # Disable all reward computation during evaluation
    reward_cfg["reward_scales"] = {}

    # Optionally fix the terrain for reproducible evaluation
    if "terrain_cfg" in env_cfg:
        env_cfg["terrain_cfg"]["randomize"] = False

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

    env.reset()
    
    # --- NEU: Ramping Setup ---
    # Wir starten bei 0 und definieren das Ziel
    device = env.commands.device
    current_commands = torch.zeros(3, device=device)
    target_commands = torch.tensor([args.vel_x, args.vel_y, args.ang_vel], device=device)
    
    # Maximale Beschleunigung pro Sekunde (kannst du anpassen)
    # [x_accel, y_accel, yaw_accel]
    accel_limits = torch.tensor([2.0, 1.0, 1.5], device=device) 
    
    # Maximal erlaubte Änderung pro Simulationsschritt (dt)
    max_step_change = accel_limits * env.dt 
    # --------------------------

    # Disable random resampling during eval by making resampling_time huge
    env.resampling_time = int(1e9)

    obs = env.get_observations()
    n_frames = 0

    if args.record:
        env.start_recording(record_internal=False)

    actual_speeds = []
    target_speeds = []

    with torch.no_grad():
        while True:
            # --- NEU: Ramping Logik ---
            # Berechne die Differenz zum Ziel
            diff = target_commands - current_commands
            
            # Limitiere die Änderung auf unsere maximale Beschleunigung pro Schritt
            step_change = torch.clamp(diff, -max_step_change, max_step_change)
            
            # Wende die Änderung an
            current_commands += step_change
            env.commands[:, :3] = current_commands
            # --------------------------

            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

            n_frames += 1

            actual_speeds.append(env.base_lin_vel[0, 0].item())
            target_speeds.append(env.commands[0, 0].item())

            if args.record and n_frames == 1000:
                env.stop_recording(
                    save_path_behind=f"{args.exp_name}_{args.ckpt}_behind.mp4",
                    save_path_side=f"{args.exp_name}_{args.ckpt}_side.mp4",
                )
                print(f"Saved recordings for checkpoint {args.ckpt}.")
                break

    plt.figure(figsize=(10, 5))
    plt.plot(target_speeds, label="Ziel-Geschwindigkeit (Command)", linestyle="--", color="gray")
    plt.plot(actual_speeds, label="Tatsächliche Geschwindigkeit", color="blue")
    plt.xlabel("Simulationsschritte")
    plt.ylabel("Geschwindigkeit in x-Richtung (m/s)")
    plt.title("Beschleunigungsprofil des Go2")
    plt.legend()
    plt.grid(True)
    plt.savefig("speed_plot.png")

if __name__ == "__main__":
    main()


"""
Run evaluation:
python fourth_try/eval.py -e test --ckpt 2000 --vel_x 3.0
"""