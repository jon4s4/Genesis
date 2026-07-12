import argparse
import math
import os
import pickle
import matplotlib.pyplot as plt
import torch

import genesis as gs
gs.init(backend=gs.gpu)
from rsl_rl.runners import OnPolicyRunner
from reward_wrapper import RunCurve



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="test")
    parser.add_argument("-r", "--record",   action="store_true", default=True)
    parser.add_argument("--ckpt",           type=int, default=100)
    # Optional: override the commanded forward speed for evaluation
    parser.add_argument("--vel_x",            type=float, default=4.0,  help="Commanded forward velocity (m/s)")
    parser.add_argument("--vel_y",            type=float, default=0.0,  help="Commanded lateral velocity (m/s)")
    # ang_vel statt target_heading_deg: commands[:, 2] ist eine GIERRATE (rad/s),
    # kein absoluter Zielwinkel (siehe RunCurve / command_cfg["ang_vel_range"] in train.py).
    # Bei konstantem vel_x und konstantem ang_vel läuft der Roboter einen Kreisbogen mit
    # Radius r = vel_x / ang_vel. Positiv = Linkskurve, negativ = Rechtskurve.
    parser.add_argument("--ang_vel",         type=float, default=0.0,
                         help="Kommandierte Gierrate (rad/s). Positiv = Linkskurve, "
                              "negativ = Rechtskurve. r = vel_x / ang_vel.")
    parser.add_argument("--straight_duration_s", type=float, default=5.0,
                         help="Wie viele Sekunden zu Beginn geradeaus (ang_vel=0) gelaufen "
                              "wird, bevor das Kurven-Kommando gesetzt wird.")
    args = parser.parse_args()


    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    # Disable all reward computation during evaluation
    reward_cfg["reward_scales"] = {}

    # Optionally fix the terrain for reproducible evaluation
    if "terrain_cfg" in env_cfg:
        env_cfg["terrain_cfg"]["randomize"] = False

    env = RunCurve(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        eval=True,
        device="cuda",
    )

    # Match the device used during training (CPU)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda")

    env.reset()

    # --- Ramping Setup für vel_x/vel_y/ang_vel ---
    # commands[:, 2] ist eine Gierrate (rad/s), keine Zielrichtung. Ein "Ziel-Heading"
    # gibt es in diesem Setup nicht mehr - der Bogen entsteht dadurch, dass ang_vel über
    # die Zeit konstant gehalten wird, während der Roboter sich bewegt. Wir rampen
    # ang_vel trotzdem sanft ein wie vel_x/vel_y, damit der Übergang von Geradeauslauf zu
    # Kurve nicht als Sprung in der Beobachtung erscheint (das WICHTIGSTE: das Resampling
    # in go2_env.py läuft alle 4s UNABHÄNGIG vom eval-Flag weiter und würde unser
    # Kommando sonst überschreiben - wir müssen es daher in jedem Step neu erzwingen).
    device = env.commands.device
    current_cmd = torch.zeros(3, device=device)
    target_cmd = torch.tensor([args.vel_x, args.vel_y, 0.0], device=device)  # ang_vel-Ziel kommt erst nach straight_duration_s

    # Maximale Beschleunigung pro Sekunde für [x_accel, y_accel, ang_accel]
    accel_limits = torch.tensor([1.0, 0.0, 0.5], device=device)
    max_step_change = accel_limits * env.dt

    straight_duration_steps = int(args.straight_duration_s / env.dt)
    curve_command_active = False
    # --------------------------

    # WICHTIG: Das periodische Resampling in go2_env.py (alle 4s, in step()) ist NICHT
    # an das eval-Flag gekoppelt und bleibt aktiv. Wir setzen unser Kommando deshalb nach
    # jedem env.step() erneut, statt uns (wie ein früherer Eval-Stand das annahm) darauf
    # zu verlassen, dass es während eval automatisch deaktiviert wird.

    obs = env.get_observations()
    n_frames = 0

    if args.record:
        env.start_recording(record_internal=False)

    actual_speeds = []
    target_speeds = []
    actual_ang_vels = []
    target_ang_vels = []
    current_yaws_deg = []
    trajectory_xy = []  # (x, y) Weltposition pro Step, für die Bogen-Visualisierung

    with torch.no_grad():
        while True:
            # --- Verzögerte Kurveneinleitung ---
            # Die ersten straight_duration_steps Schritte läuft der Roboter mit ang_vel=0
            # (Geradeauslauf), damit zunächst eine stabile Gangart etabliert wird. Danach
            # wird das Kurven-Kommando (args.ang_vel) als neues Rampingziel gesetzt.
            if not curve_command_active and n_frames >= straight_duration_steps:
                target_cmd[2] = args.ang_vel
                curve_command_active = True
            # --------------------------

            # --- Ramping Logik (vel_x, vel_y, ang_vel) ---
            diff = target_cmd - current_cmd
            step_change = torch.clamp(diff, -max_step_change, max_step_change)
            current_cmd += step_change
            env.commands[:, :3] = current_cmd
            # Erzwingt das Kommando gegen das periodische Resampling in go2_env.py,
            # das unabhängig vom eval-Flag weiterläuft (siehe Kommentar oben).
            # --------------------------

            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

            n_frames += 1

            actual_speeds.append(env.base_lin_vel[0, 0].item())
            target_speeds.append(env.commands[0, 0].item())
            actual_ang_vels.append(env.base_ang_vel[0, 2].item() * (180.0 / math.pi))
            target_ang_vels.append(env.commands[0, 2].item() * (180.0 / math.pi))

            w, x, y, z = env.base_quat[0, 0], env.base_quat[0, 1], env.base_quat[0, 2], env.base_quat[0, 3]
            current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            current_yaws_deg.append(current_yaw.item() * (180.0 / math.pi))

            trajectory_xy.append((env.base_pos[0, 0].item(), env.base_pos[0, 1].item()))

            if args.record and n_frames == 1000:
                env.stop_recording(
                    save_path_behind=f"{args.exp_name}_{args.ckpt}_{args.vel_x}_{args.ang_vel}_behind.mp4",
                    save_path_side=f"{args.exp_name}_{args.ckpt}_{args.vel_x}_{args.ang_vel}_side.mp4",
                )
                print(f"Saved recordings for checkpoint {args.ckpt}.")
                break

    # --- Plot 1: Beschleunigungsprofil (Geschwindigkeit) ---
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(target_speeds, label="Commanded target velocity", linestyle="--", color="gray")
    ax1.plot(actual_speeds, label="Actual velocity", color="blue")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Velocity x (m/s)")
    ax1.set_title("Acceleration profil of Go2")
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()
    fig1.savefig(f"speed_plot_{args.ckpt}_{args.vel_x}_{args.ang_vel}.png")
    plt.close(fig1)

    # --- Plot 2: Gierraten-Tracking ---
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(target_ang_vels, label="Commanded target yaw-rate", linestyle="--", color="gray")
    ax2.plot(actual_ang_vels, label="Actual yaw-rate", color="orange")
    ax2.axvline(straight_duration_steps, label="Point of yaw-rate command", linestyle=":", color="red", alpha=0.7)
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("yaw-rate (deg/s)")
    radius_str = f"r ≈ {args.vel_x / args.ang_vel:.2f} m" if abs(args.ang_vel) > 1e-6 else "straight forward"
    ax2.set_title(f"Yaw-Rate-Tracking of Go2 ({args.straight_duration_s:.0f}s forward, then ang_vel={args.ang_vel:+.2f} rad/s, {radius_str})")
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig(f"ang_vel_plot_{args.ckpt}_{args.vel_x}_{args.ang_vel}.png")
    plt.close(fig2)

    # --- Plot 3: Trajektorie (Draufsicht) ---
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    traj = torch.tensor(trajectory_xy, device="cpu")
    ax3.plot(traj[:, 0], traj[:, 1], color="green")
    ax3.scatter([traj[0, 0]], [traj[0, 1]], color="black", marker="o", label="Start", zorder=5)
    curve_marker_idx = min(straight_duration_steps, len(traj) - 1)
    ax3.scatter([traj[curve_marker_idx, 0]], [traj[curve_marker_idx, 1]],
                color="red", marker="x", label="Curve initiated", zorder=5)
    ax3.set_xlabel("World-X (m)")
    ax3.set_ylabel("World-Y (m)")
    ax3.set_title("Trajectory")
    ax3.legend()
    ax3.grid(True)
    ax3.set_aspect("equal", adjustable="datalim")
    fig3.tight_layout()
    fig3.savefig(f"trajectory_plot_{args.ckpt}_{args.vel_x}_{args.ang_vel}.png")
    plt.close(fig3)

    final_ang_vel_error = abs(target_ang_vels[-1] - actual_ang_vels[-1])
    print(f"Final yaw-rate error: {final_ang_vel_error:.2f} deg/s")

if __name__ == "__main__":
    main()


"""
Run evaluation:
python eval.py -e test --ckpt 3000 --vel_x 4.0

Geradeaus, dann sanfte Linkskurve nach 5s (Default-Dauer):
python eval.py -e test --ckpt 3000 --vel_x 5.0 --ang_vel 0.6

Geradeaus, dann engere Rechtskurve, Kurve schon nach 3s einleiten:
python eval.py -e test --ckpt 3000 --vel_x 4.0 --ang_vel -0.8 --straight_duration_s 3.0
"""
