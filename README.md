# SLOT: Autonomous Navigation via Offline Behavior-Regularized DDQN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-blue.svg)](https://docs.ros.org/en/humble/index.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-Offline_RL-EE4C2C.svg)](https://pytorch.org/)

Official code repository for the paper:

**"Autonomous Navigation of a Soft-Legged Omnidirectional Tetrapod via Offline Behavior-Regularized Double Deep Q-Network"**

This repository contains the complete hierarchical navigation pipeline for **SLOT**. The system decouples global spatial planning, using a staged 3D RRT, from local reactive execution, using an offline-trained, purely visual Double DQN policy.

---

## 🗂️ Repository Structure

```text
SLOT-Offline-RL/
│
├── 1_Global_Planner_3D_RRT/
│   └── 3D RRT engine for global trajectory generation
│
├── 2_AR_Video_Overlays/
│   └── OpenCV-based AR visualization and trajectory overlays
│
├── 3_Robot_Deployment/
│   └── Jetson deployment code for autonomous navigation
│
└── README.md
```

### `1_Global_Planner_3D_RRT/`

Contains the **3D Rapidly-exploring Random Tree (RRT)** engine used to generate smoothed, kinematically feasible global trajectories. The resulting environment maps and trajectories are exported as JSON files for deployment.

### `2_AR_Video_Overlays/`

Contains OpenCV-based scripts for projecting mathematical RRT paths and emergent RL obstacle-avoidance maneuvers onto real-world deployment footage using homography.

### `3_Robot_Deployment/`

Contains the code deployed directly on the **NVIDIA Jetson Xavier NX**. This module integrates:

- Isaac ROS Visual SLAM
- Intel RealSense D435i depth camera
- PyTorch DDQN policy
- Dynamixel motor control
- Global waypoint navigation
- Local reactive obstacle avoidance

---

# 🛠️ 1. Installation & Setup

## Laptop / PC

The planner and visualization modules require standard Python scientific-computing and computer-vision libraries.

### Clone the Repository

```bash
git clone https://github.com/Saumya-Karan/SLOT-Offline-RL.git
cd SLOT-Offline-RL
```

### Install Dependencies

```bash
pip install numpy scipy matplotlib opencv-python
```

---

## Robot: NVIDIA Jetson Xavier NX

The deployment pipeline requires **PyTorch** for inference, the **Dynamixel SDK** for motor control, and **ROS 2 Humble** for sensor processing.

### 1. Install ROS 2 and Isaac ROS

Ensure that the following are installed and configured on the Jetson:

- ROS 2 Humble
- Isaac ROS Visual SLAM
- Isaac ROS development Docker environment
- Intel RealSense ROS packages

### 2. Install Python Dependencies

Inside the Jetson environment:

```bash
pip install torch torchvision dynamixel-sdk numpy
```

### 3. Grant Serial Port Permissions

Grant access to the U2D2 motor controller:

```bash
sudo chmod a+rw /dev/ttyUSB0
```

---

## Hardware Configuration

### Dynamixel Motor IDs

Ensure the Dynamixel **XL430-W250-T** motors are configured with the following IDs:

| Position | Dynamixel ID |
|----------|--------------|
| Front-Left (FL) | `3` |
| Front-Right (FR) | `2` |
| Back-Left (BL) | `5` |
| Back-Right (BR) | `4` |

---

# 🚀 2. Step-by-Step Usage Guide

## Phase A: Generating the Global Map

**Run on Laptop / PC**

Before autonomous execution, the 3D topology of the environment must be generated and exported for the robot.

### 1. Navigate to the Planner Directory

```bash
cd 1_Global_Planner_3D_RRT/3D_Mathematical_Plots/
```

### 2. Generate a Trajectory

Run the desired trajectory script. For example:

```bash
python3 plot_traj1.py
```

### 3. Visualize and Export

A 3D Matplotlib window will open displaying:

- The generated RRT tree
- The smoothed global trajectory
- The environment topology

Once the visualization window is closed, the script automatically generates:

```text
outputs/full_3d_map.json
```

### 4. Transfer Required Files to the Jetson

Transfer the following files to:

```text
3_Robot_Deployment/
```

- `full_3d_map.json`
- `slot_pytorch_ddqn.pth`

---

# Phase B: Hardware Bringup

**Run on NVIDIA Jetson Xavier NX**

> **Note:** The Intel RealSense D435i is configured to operate at **640 × 480 resolution and 15 FPS**, with synchronized streams, to reduce USB bus saturation during simultaneous depth sensing and Visual SLAM.

### 1. Start the Isaac ROS Development Container

Open **Terminal 1**:

```bash
cd ${ISAAC_ROS_WS}/src/isaac_ros_common
./scripts/run_dev.sh ${ISAAC_ROS_WS}
```

### 2. Launch the Camera and Visual SLAM Pipeline

Inside the Isaac ROS container:

```bash
./vslam_launch.sh
```

This launches the RealSense camera and Isaac ROS Visual SLAM pipeline using the configured parameters required for stable state-space tracking.

---

# Phase C: Autonomous Execution

**Run on NVIDIA Jetson Xavier NX**

Once the camera and SLAM pipeline are running, launch the hierarchical navigation supervisor.

### 1. Navigate to the Deployment Directory

Open **Terminal 2**:

```bash
cd ~/SLOT-Offline-RL/3_Robot_Deployment/
```

### 2. Start Autonomous Navigation

```bash
python3 slot_deployment.py
```

### Expected Behavior

The robot loads the global trajectory from:

```text
full_3d_map.json
```

The navigation system then uses Visual SLAM to estimate the robot's state and steer toward the planned waypoints.

During execution, the RealSense depth grid continuously monitors the local environment. If an unmapped obstacle or overhead obstruction is detected, the local **PyTorch DDQN policy** can override the global supervisor and execute an appropriate reactive maneuver, such as:

- Lateral strafing
- Crawling
- Local obstacle avoidance

The system additionally uses **proprioceptive action masking and dead-reckoning** to safely clear the robot's physical blind spot before handing control back to the global navigation layer.

This creates a hierarchical navigation architecture in which:

```text
                 Global 3D RRT
                       │
                       ▼
              Global Waypoints
                       │
                       ▼
              Navigation Supervisor
                       │
             ┌─────────┴─────────┐
             │                   │
       Normal Operation     Obstacle Detected
             │                   │
             ▼                   ▼
        Follow Path          DDQN Policy
                                 │
                                 ▼
                         Reactive Maneuver
                                 │
                                 ▼
                         Blind-Spot Clearing
                                 │
                                 ▼
                         Return to Global Path
```

---

# 🎥 3. Generating Augmented Reality (AR) Videos

The repository also provides tools for visualizing the generated 3D RRT trajectories and emergent RL behaviors over real-world deployment footage.

## 1. Add Raw Footage

Place the raw `.mp4` recording inside:

```text
2_AR_Video_Overlays/raw_media/
```

## 2. Run the Corresponding Overlay Script

For Trajectory 1:

```bash
cd 2_AR_Video_Overlays/
python3 overlay_traj1.py
```

## 3. Compute the Homography

The script will prompt you to select:

1. The four floor corners
2. The relevant obstacle corners

These points are used to calculate the **homography matrix** and generate the corresponding occlusion mask.

The resulting AR visualization will be saved to:

```text
2_AR_Video_Overlays/outputs/AR_videos/
```

---

# 🧠 System Overview

The complete SLOT navigation pipeline combines classical planning with learned reactive control:

```text
                    Environment
                         │
                         ▼
                ┌─────────────────┐
                │   3D RRT Planner │
                └────────┬────────┘
                         │
                         ▼
                  Global Trajectory
                         │
                         ▼
                ┌─────────────────┐
                │ Navigation       │
                │ Supervisor       │
                └────────┬────────┘
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Visual SLAM              Depth Camera
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                Local Environment
                      State
                         │
                         ▼
                ┌─────────────────┐
                │ Offline DDQN    │
                │ Reactive Policy │
                └────────┬────────┘
                         │
                         ▼
                Reactive Maneuver
                         │
                         ▼
                  Motor Commands
                         │
                         ▼
                       SLOT
```

---

# 📄 Citation

If you use this repository or the associated methodology in your research, please cite:

```bibtex
@article{slot_offline_rl,
  title={Autonomous Navigation of a Soft-Legged Omnidirectional Tetrapod via Offline Behavior-Regularized Double Deep Q-Network},
  author={Saumya Karan and collaborators},
  journal={},
  year={}
}
```

---

# 📜 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.
