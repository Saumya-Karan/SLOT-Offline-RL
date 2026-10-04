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
from dynamixel_sdk import * # Raw serial control

# ==========================================
# 1. PYTORCH MODEL (Your exact 25D -> 6 Action MLP)
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
            nn.Linear(64, 6) # 0:Fwd, 1:L-Turn, 2:R-Turn, 3:L-Strafe, 4:R-Strafe, 5:Crawl
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
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.odom_callback, qos_profile_sensor_data)
        
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
PORT_NAME = '/dev/ttyUSB0'
BAUDRATE = 2000000
portHandler = PortHandler(PORT_NAME)
packetHandler = PacketHandler(2.0)

if not portHandler.openPort() or not portHandler.setBaudRate(BAUDRATE):
    print("CRITICAL: Failed to open U2D2 port!")

def execute_gait(action_id):
    """ Sends hex commands and returns (forward_distance_walked) for the blind-spot counter """
    dist_walked = 0.0
    try:
        if action_id == 0:
            print(">> GAIT: Forward Walk")
            dist_walked = 0.05 # Approximated forward distance per step
            # TODO: packetHandler.write4ByteTxRx(...)
        elif action_id == 3:
            print(">> GAIT: Left Strafe")
        elif action_id == 4:
            print(">> GAIT: Right Strafe")
        elif action_id == 5:
            print(">> GAIT: Crawl")
            dist_walked = 0.04
        # Add Turn Gaits here...
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
    fwd_clearance = 0.0  # Tracks how far we walk forward while dodging
    is_dodging = False   # State flag for obstacle avoidance

    print("SLOT IS ALIVE. MAP LOADED. COMMENCING AUTONOMOUS NAVIGATION...")

    while True:
        if current_wp_idx >= len(path):
            print("GOAL REACHED! Mission Complete.")
            break
            
        target_x, target_y, _ = path[current_wp_idx]
        bot_x, bot_y, bot_yaw = global_pose['x'], global_pose['y'], global_pose['yaw']

        # --- A. CHECK WAYPOINT PROGRESSION ---
        dist = math.sqrt((target_x - bot_x)**2 + (target_y - bot_y)**2)
        if dist < 0.15: # 15cm tolerance
            current_wp_idx = min(current_wp_idx + 5, len(path) - 1)
            continue

        # --- B. CALCULATE LATERAL DEVIATION (Cross-track error) ---
        angle_to_wp = math.atan2(target_y - bot_y, target_x - bot_x)
        heading_error = (angle_to_wp - bot_yaw + math.pi) % (2 * math.pi) - math.pi
        lateral_deviation = dist * math.sin(heading_error) # +ve means robot is too far right

        # --- C. GET AI PREDICTION (What does the camera see?) ---
        state_tensor = torch.tensor(global_depth_grid, dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            ai_action = torch.argmax(q_values).item()

        # --- D. THE DYNAMIC AVOIDANCE SUPERVISOR LOGIC ---
        
        # Scenario 1: AI sees a threat! (Outputs Strafe or Crawl)
        if ai_action in [3, 4, 5]: 
            final_action = ai_action
            is_dodging = True
            fwd_clearance = 0.0 # Reset forward counter because we are actively dodging laterally
            print(">> AI VISUAL OVERRIDE: Evading Threat! <<")

        # Scenario 2: AI sees open space, BUT we are currently off the path (Dodging Recovery)
        elif is_dodging:
            # We must walk forward to clear the blind spot (Proprioceptive Action Masking!)
            if fwd_clearance < 0.50: # Must walk 0.5m forward to clear the box length + body length
                final_action = 0 # FORCED Forward Walk
                print(f"Safety Filter Active: Clearing blind spot... ({fwd_clearance:.2f}m / 0.50m)")
            else:
                # Blind spot cleared! Use the "Rubber Band" to strafe back to the path
                if lateral_deviation > 0.10: # We drifted right, so strafe left
                    final_action = 3 
                    print("Rubber Band: Strafing Left to return to path.")
                elif lateral_deviation < -0.10: # We drifted left, so strafe right
                    final_action = 4
                    print("Rubber Band: Strafing Right to return to path.")
                else:
                    # We are back on the path!
                    is_dodging = False
                    final_action = 0
                    print("Successfully recovered to global path.")

        # Scenario 3: AI sees open space, and we are perfectly on the RRT path
        else:
            # Just follow the SLAM GPS towards the waypoint
            if heading_error > 0.25:
                final_action = 1 # Left Turn to adjust heading
            elif heading_error < -0.25:
                final_action = 2 # Right Turn to adjust heading
            else:
                final_action = 0 # Forward Walk

        # --- E. EXECUTE HARDWARE ---
        dist_walked = execute_gait(final_action)
        fwd_clearance += dist_walked # Update our blind spot counter!

        time.sleep(1.2) # Fixed control loop

if __name__ == "__main__":
    main()