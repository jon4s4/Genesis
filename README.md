# Fast running robotic dog using locomotion policies

This repository contains the code and evaluation resources for my Bachelor's thesis on high-velocity locomotion policies for the Unitree Go2 quadrupedal robot.

### Experiments Conducted
1. **Flat-Terrain Speed Experiment**: Training the Go2 to reach a top forward velocity of 8 m/s on perfectly flat ground.
2. **Multi-Terrain Experiment**: Navigating a structured $3 \times 3$ grid of flat, fractal, and pyramid staircase terrains at speeds up to 4.5 m/s.
3. **High-Speed Curve Running (Flat Terrain)**: Implementing a goal-conditioned policy that commands the robot to execute circular arcs (up to 0.8 rad/s) while maintaining high forward velocities (up to 7 m/s).
4. **High-Speed Curve Running (Diverse Terrain)**: Combining multi-terrain navigation with curve-running capabilities, maintaining up to 4 m/s on complex surfaces with a commanded angular velocity of 0.8 rad/s.

## Requirements

* **Python:** 3.12.3
* **Genesis Simulator:** v0.4.7
* **Hardware:** NVIDIA RTX-5090 GPU (recommended for computing 32,768 parallel environments)

## Visual Results
Visual materials, including velocity tracking graphs, trajectories, and videos showing the robot in bird's eye view and from the side during evaluation, can be found in the respective experiment folders.
