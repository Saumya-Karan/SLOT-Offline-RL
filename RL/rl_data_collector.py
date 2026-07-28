import sys, time, threading, csv, os
import tkinter as tk
import numpy as np
from dynamixel_sdk import *
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
import math

# ==========================================
# 1. ROS 2 SENSORS
# ==========================================
class SlotSensors(Node):
    def __init__(self):
        super().__init__('rl_data_collector')
        self.sub_depth = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_cb, qos_profile_sensor_data)
        self.sub_slam = self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.slam_cb, 10)
        self.grid = [5.0] * 25
        self.alpha = 0.20
        self.robot_x = 0.0
        self.robot_y = 0.0

    def depth_cb(self, msg):
        img = np.ndarray(shape=(msg.height, msg.width), dtype=np.uint16, buffer=msg.data).astype(float) / 1000.0
        h, w = msg.height // 5, msg.width // 5
        raw = []
        for r in range(5):
            for c in range(5):
                cell = img[r*h:(r+1)*h, c*w:(c+1)*w]
                valid = cell[(cell > 0.1) & (cell <= 5.0)]
                raw.append(round(np.percentile(valid, 5) if len(valid) > cell.size*0.02 else 5.0, 2))
        for i in range(25): self.grid[i] = round((self.alpha * raw[i]) + ((1 - self.alpha) * self.grid[i]), 2)

    def slam_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

ros_node = None
def ros_thread():
    global ros_node; rclpy.init(); ros_node = SlotSensors(); rclpy.spin(ros_node); ros_node.destroy_node(); rclpy.shutdown()
threading.Thread(target=ros_thread, daemon=True).start()

# ==========================================
# 2. OFFLINE RL LOGGER & REFEREE
# ==========================================
CSV_FILE = "slot_rl_dataset3_with_omni.csv"

cur_traj = 1
cur_step = 1

# Check the CSV to resume Trajectory ID if script was restarted
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'r') as f:
        lines = f.readlines()
        if len(lines) > 1:
            try:
                cur_traj = int(lines[-1].split(',')[0]) + 1
            except:
                pass
else:
    with open(CSV_FILE, 'w', newline='') as f:
        headers = ['Traj_ID', 'Step_Num'] + [f"Grid_{i}" for i in range(25)] + ['Action', 'Reward', 'Survival']
        csv.writer(f).writerow(headers)

# --- NEW: TARE VARIABLES ---
slam_offset_x = 0.0
slam_offset_y = 0.0
wp1_cleared = False
wp2_cleared = False

def log_and_execute(action_id, gait_function):
    global ros_node, wp1_cleared, wp2_cleared, cur_traj, cur_step, slam_offset_x, slam_offset_y
    if ros_node is None: return

    state = ros_node.grid.copy()
    
    # RELATIVE SLAM POSITION
    rx = ros_node.robot_x - slam_offset_x
    ry = ros_node.robot_y - slam_offset_y
    
    reward = 0.0
    survival = 1

    # --- DEFINE TARGETS (X, Y, Radius in meters) ---
    # Change these X and Y values to match your exact room!
    goal_x, goal_y, goal_r = 0.40, -0.50, 0.25      # 25cm radius around Goal
    wp2_x, wp2_y, wp2_r    = 0.80, -0.50, 0.25      # 25cm radius around Corner
    wp1_x, wp1_y, wp1_r    = 0.80,  0.00,  0.25     # 25cm radius around Bar
    
    # Calculate exact Euclidean distance (radius) to each target
    dist_goal = math.hypot(rx - goal_x, ry - goal_y)
    dist_wp2  = math.hypot(rx - wp2_x, ry - wp2_y)
    dist_wp1  = math.hypot(rx - wp1_x, ry - wp1_y)

    # --- THE RADIUS REFEREE ---
    front_zone = min(state[7], state[11], state[12], state[13], state[17])
    
    if dist_goal <= goal_r and wp2_cleared:
        reward = +200.0
        print(f"?? GOAL REACHED! Distance: {dist_goal:.2f}m. Hit Reset!")
        
    elif dist_wp2 <= wp2_r and wp1_cleared and not wp2_cleared:
        reward = +50.0
        wp2_cleared = True
        print(f"?? WAYPOINT 2 CLEARED! Distance: {dist_wp2:.2f}m.")

    elif dist_wp1 <= wp1_r and not wp1_cleared:
        reward = +50.0
        wp1_cleared = True
        print(f"?? WAYPOINT 1 CLEARED! Distance: {dist_wp1:.2f}m.")
        
    else:
        reward = +1.0
        print(f"? Safe Step | Act: {action_id} | Pos: X={rx:.2f}, Y={ry:.2f}")

    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([cur_traj, cur_step] + state + [action_id, reward, survival])
    
    cur_step += 1
    gait_function()
def manual_crash():
    global ros_node, cur_traj, cur_step
    if ros_node is None: return
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([cur_traj, cur_step] + ros_node.grid.copy() + ["crash", -100.0, 0])

def reset_episode():
    global wp1_cleared, wp2_cleared, cur_traj, cur_step, slam_offset_x, slam_offset_y, ros_node
    
    wp1_cleared = False
    wp2_cleared = False
    cur_traj += 1   
    cur_step = 1    
    
    # "TARE" the SLAM to true zero!
    if ros_node is not None:
        slam_offset_x = ros_node.robot_x
        slam_offset_y = ros_node.robot_y
        
    print(f"\n?? EPISODE RESET! Starting Trajectory #{cur_traj}.")
    print(f"?? SLAM Zeroed. (Internal offset X:{slam_offset_x:.2f}, Y:{slam_offset_y:.2f})\n")
# ==========================================
# 3. DYNAMIXEL KINEMATICS
# ==========================================
port = PortHandler('/dev/ttyUSB0'); pkt = PacketHandler(2.0)
if port.openPort() and port.setBaudRate(2000000):
    for i in [3,2,5,4]: pkt.write1ByteTxRx(port, i, 64, 1)

ul, ulr, ulc, relc, defc, rel, defo = 250, 280, 220, 160, 115, 105, 175
DXL_IDS = {"FL": 3, "FR": 2, "BL": 5, "BR": 4}

def ms(ang, id): 
    pkt.write4ByteTxRx(port, id, 116, int((max(10, min(350, ang))/360.0)*4095))

def act_fwd():
    ms(rel-20, 3); ms(ul+70, 2); ms(ul, 5); ms(ul, 4); root.after(420)
    ms(ul-30, 3); ms(rel+20, 2); ms(ul, 5); ms(ul, 4); root.after(420)

def act_l_turn():
    ms(200, 3); ms(ul, 2); ms(defo, 5); ms(rel, 4); root.update(); time.sleep(0.42)
    ms(200, 3); ms(defo, 2); ms(rel, 5); ms(230, 4); root.update(); time.sleep(0.42)
    ms(200, 3); ms(rel, 2); ms(ul, 5); ms(defo, 4); root.update(); time.sleep(0.42)

def act_r_turn():
    ms(rel, 4); ms(rel, 2); ms(ul, 3); ms(ul, 5); root.update(); time.sleep(0.42)
    ms(rel, 4); ms(ul, 2); ms(ul, 3); ms(rel, 5); root.update(); time.sleep(0.42)

def act_l_walk():
    ms(ul, 2); ms(ul, 4); ms(ul, 3); ms(rel, 5); root.update(); time.sleep(0.42)
    ms(ul, 2); ms(ul, 4); ms(ul, 5); ms(rel, 3); root.update(); time.sleep(0.42)

def act_r_walk():
    ms(ul, 3); ms(ul, 5); ms(ul, 2); ms(rel, 4); root.update(); time.sleep(0.42)
    ms(ul, 3); ms(ul, 5); ms(ul, 4); ms(rel, 2); root.update(); time.sleep(0.42)

def act_crawl():
    print(">>> EXECUTING HARDCODED CRAWL MACRO <<<")
    for _ in range(3): # Hardcoded safety loop to clear the 40cm bar length!
        ms(defo, 2); ms(defo, 5); ms(ul, 3); ms(rel, 4); root.update(); time.sleep(0.40)
        ms(ul, 4); ms(rel, 3); ms(defo, 5); ms(defo, 2); root.update(); time.sleep(0.40)
        ms(defo, 4); ms(defo, 3); ms(rel, 5); ms(ul, 2); root.update(); time.sleep(0.40)
        ms(rel, 2); ms(ul, 5); ms(defo, 3); ms(defo, 4); root.update(); time.sleep(0.40)
        
def act_omni_walk():
    # ADD omni gait according to the angle needed
    ms(ul, 3); ms(ul, 5); ms(ul, 2); ms(rel, 4); root.update(); time.sleep(0.42)
    ms(ul, 3); ms(ul, 5); ms(ul, 4); ms(rel, 2); root.update(); time.sleep(0.42)

# ==========================================
# 4. GUI
# ==========================================
root = tk.Tk(); root.title("Offline RL Data Collector"); root.geometry("450x750"); root.configure(bg="#2b2b2b")

tk.Label(root, text="DATA COLLECTION MODE", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="white").pack(pady=10)

tk.Button(root, text="[0] FORWARD", width=25, height=2, command=lambda: log_and_execute(0, act_fwd)).pack(pady=2)
tk.Button(root, text="[1] LEFT TURN", width=25, height=2, command=lambda: log_and_execute(1, act_l_turn)).pack(pady=2)
tk.Button(root, text="[2] RIGHT TURN", width=25, height=2, command=lambda: log_and_execute(2, act_r_turn)).pack(pady=2)
tk.Button(root, text="[3] LEFT WALK", width=25, height=2, command=lambda: log_and_execute(3, act_l_walk)).pack(pady=2)
tk.Button(root, text="[4] RIGHT WALK", width=25, height=2, command=lambda: log_and_execute(4, act_r_walk)).pack(pady=2)
tk.Button(root, text="[5] CRAWL (MACRO)", width=25, height=2, bg="#0fb9b1", command=lambda: log_and_execute(5, act_crawl)).pack(pady=2)
tk.Button(root, text="[6] OMNI WALK", width=25, height=2, command=lambda: log_and_execute(4, act_omni_walk)).pack(pady=2)


tk.Button(root, text="?? LOG FATAL CRASH & TERMINATE", width=30, height=2, bg="darkred", fg="white", font=("Arial", 10, "bold"), command=manual_crash).pack(pady=15)
tk.Button(root, text="?? RESET EPISODE (To 0,0)", width=30, height=2, bg="#3b4d61", fg="white", font=("Arial", 10, "bold"), command=reset_episode).pack(pady=5)

def kill():
    for i in [3,2,5,4]: pkt.write1ByteTxRx(port, i, 64, 0)
    root.destroy(); sys.exit()

tk.Button(root, text="? EXIT & KILL MOTORS", bg="red", fg="white", width=30, command=kill).pack(pady=20)
root.mainloop()