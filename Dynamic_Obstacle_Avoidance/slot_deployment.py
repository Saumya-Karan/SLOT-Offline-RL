import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import torch
import torch.nn as nn
import numpy as np
import threading
import time
import json
import math
from dynamixel_sdk import * # Your raw serial control

# ==========================================
# 1. PYTORCH MODEL (Your exact offline DDQN)
# ==========================================
class SLOT_DDQN(nn.Module):
    def __init__(self):
        super(SLOT_DDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7) # 7 Macro-Actions (Matches your paper)
        )
    def forward(self, x):
        return self.net(x)

# Globals for ROS Thread
global_depth_grid = np.zeros(25, dtype=np.float32)
global_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}

# ==========================================
# 2. ROS 2 BACKGROUND THREAD (SLAM + Depth)
# ==========================================
class SlotSensors(Node):
    def __init__(self):
        super().__init__('slot_sensor_node')
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(
            Odometry, '/visual_slam/tracking/odometry', self.odom_callback, qos_profile_sensor_data)
        
    def depth_callback(self, msg):
        global global_depth_grid
        # TODO: Insert your 5x5 pooling logic here to populate global_depth_grid
        pass

    def odom_callback(self, msg):
        global global_pose
        global_pose['x'] = msg.pose.pose.position.x
        global_pose['y'] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        global_pose['yaw'] = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

def start_ros_thread():
    rclpy.init()
    node = SlotSensors()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# ==========================================
# 3. DYNAMIXEL MOTOR CONTROL (Immortal GUI)
# ==========================================
# TODO: Initialize PortHandler and PacketHandler here

def execute_gait(action_id):
    """ Sends hex commands and returns (forward_distance_walked) for the 40cm blind-spot counter """
    dist_walked = 0.0
    try:
        if action_id == 0:
            print(">> GAIT: Forward Walk")
            dist_walked = 0.05 # Approximated forward distance per step
            # packetHandler.write4ByteTxRx(...)
        elif action_id == 1:
            print(">> GAIT: Left Walk (Strafe)")
        elif action_id == 2:
            print(">> GAIT: Right Walk (Strafe)")
        elif action_id == 5:
            print(">> GAIT: Crawl")
            dist_walked = 0.04
        # Add other gaits...
    except Exception as e:
        print(f"HARDWARE STALL CAUGHT: {e}")
    return dist_walked

# ==========================================
# 4. MAIN AUTOPILOT (The Brain + Supervisor)
# ==========================================
def main():
    threading.Thread(target=start_ros_thread, daemon=True).start()
    time.sleep(2) # Warm up sensors

    # Load Brain
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLOT_DDQN().to(device)
    model.load_state_dict(torch.load("slot_pytorch_ddqn.pth", map_location=device))
    model.eval()

    # Load 3D RRT Map
    with open("full_3d_map.json", "r") as f:
        map_data = json.load(f)
    path = map_data["planned_path"]
    
    current_wp_idx = 0
    
    # --- THE 40cm SAFETY FILTER VARIABLES ---
    ROBOT_LENGTH = 0.40  # 40cm Hardcoded robot length
    fwd_clearance = 0.0  # Tracks forward progress while dodging
    is_dodging = False   # State flag

    print("SLOT IS ALIVE. MAP LOADED. COMMENCING AUTONOMOUS NAVIGATION...")

    while True:
        if current_wp_idx >= len(path):
            print("GOAL REACHED! Mission Complete.")
            break
            
        target_x, target_y, _ = path[current_wp_idx]
        bot_x, bot_y, bot_yaw = global_pose['x'], global_pose['y'], global_pose['yaw']

        # A. CHECK WAYPOINT PROGRESSION
        dist = math.sqrt((target_x - bot_x)**2 + (target_y - bot_y)**2)
        if dist < 0.15: # 15cm tolerance
            current_wp_idx = min(current_wp_idx + 5, len(path) - 1)
            continue

        # B. CALCULATE SLAM ERROR TO THE RRT LINE
        angle_to_wp = math.atan2(target_y - bot_y, target_x - bot_x)
        heading_error = (angle_to_wp - bot_yaw + math.pi) % (2 * math.pi) - math.pi
        lateral_deviation = dist * math.sin(heading_error) 

        # C. GET AI PREDICTION (What does the camera see?)
        state_tensor = torch.tensor(global_depth_grid, dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            ai_action = torch.argmax(q_values).item()

        # D. THE SUPERVISOR LOGIC
        
        # Scenario 1: AI sees a threat and overrides! (e.g., Strafe Left, Crawl, etc.)
        if ai_action != 0: 
            final_action = ai_action
            is_dodging = True
            fwd_clearance = 0.0 # Reset counter because we are currently dodging!
            print(">> AI VISUAL OVERRIDE: Evading Threat! <<")

        # Scenario 2: AI sees open space (Action 0), BUT we are actively recovering from a dodge
        elif is_dodging:
            if fwd_clearance < ROBOT_LENGTH: 
                # THE 40cm BLIND SPOT FILTER
                final_action = 0 # Force Forward Walk
                print(f"Safety Filter Active: Clearing hind legs... ({fwd_clearance:.2f}m / {ROBOT_LENGTH}m)")
            else:
                # 40cm cleared! The robot can safely return to the path.
                if lateral_deviation > 0.10: # Drifting Right, must Strafe Left
                    final_action = 1 # Left Walk Macro Action
                    print("Blind Spot Cleared: Strafing Left to return to path.")
                elif lateral_deviation < -0.10: # Drifting Left, must Strafe Right
                    final_action = 2 # Right Walk Macro Action
                    print("Blind Spot Cleared: Strafing Right to return to path.")
                else:
                    is_dodging = False
                    final_action = 0
                    print("Successfully recovered to global RRT path.")

        # Scenario 3: AI sees open space, and we are perfectly on the RRT path
        else:
            # Let the Supervisor Steer toward the RRT Waypoint
            if heading_error > 0.25:
                final_action = 3 # Left Turn to adjust heading
            elif heading_error < -0.25:
                final_action = 4 # Right Turn to adjust heading
            else:
                final_action = 0 # Forward Walk

        # E. FIRE MOTORS
        dist_walked = execute_gait(final_action)
        
        # Only count forward progress for our 40cm filter if we are actively walking forward
        if final_action == 0: 
            fwd_clearance += dist_walked 

        time.sleep(1.2) # Fixed control loop

if __name__ == "__main__":
    main()