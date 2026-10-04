import time, sys, threading, csv, os
import tkinter as tk
import numpy as np
import torch
import torch.nn as nn
from dynamixel_sdk import *
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data

# ==========================================
# 1. ROS 2 VISION NODE
# ==========================================
class DepthGridNode(Node):
    def __init__(self):
        super().__init__('eval_autopilot')
        self.sub = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.callback, qos_profile_sensor_data)
        self.grid = [5.0] * 25; self.alpha = 0.20 
    def callback(self, msg):
        img = np.ndarray(shape=(msg.height, msg.width), dtype=np.uint16, buffer=msg.data).astype(float) / 1000.0
        h, w = msg.height // 5, msg.width // 5
        raw = [round(np.percentile(img[r*h:(r+1)*h, c*w:(c+1)*w][(img[r*h:(r+1)*h, c*w:(c+1)*w] > 0.1) & (img[r*h:(r+1)*h, c*w:(c+1)*w] <= 5.0)], 5) if len(img[r*h:(r+1)*h, c*w:(c+1)*w][(img[r*h:(r+1)*h, c*w:(c+1)*w] > 0.1) & (img[r*h:(r+1)*h, c*w:(c+1)*w] <= 5.0)]) > (h*w*0.02) else 5.0, 2) for r in range(5) for c in range(5)]
        for i in range(25): self.grid[i] = round((self.alpha * raw[i]) + ((1.0 - self.alpha) * self.grid[i]), 2)

ros_node = None
def ros_thread():
    global ros_node; rclpy.init(); ros_node = DepthGridNode(); rclpy.spin(ros_node); ros_node.destroy_node(); rclpy.shutdown()
threading.Thread(target=ros_thread, daemon=True).start()

# ==========================================
# 2. LOAD PYTORCH BRAIN (.pth)
# ==========================================
class DDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x): return self.net(x)

print("Loading Pure PyTorch DDQN Brain...")
model = DDQN(25, 6)
try:
    model.load_state_dict(torch.load("slot_pytorch_ddqn_test.pth", map_location=torch.device('cpu')))
    model.eval() # Set to evaluation mode
    print("?? PyTorch Brain Loaded Successfully!")
except Exception as e:
    print(f"? Error loading model: {e}"); sys.exit()

# ==========================================
# 3. KINEMATICS & IMMORTAL SHIELD
# ==========================================
port = PortHandler('/dev/ttyUSB0'); pkt = PacketHandler(2.0)
DXL_IDS = {"FL": 3, "FR": 2, "BL": 5, "BR": 4}

def init_motors():
    try:
        if port.openPort() and port.setBaudRate(2000000):
            for i in DXL_IDS.values(): pkt.write1ByteTxRx(port, i, 64, 1)
            print("? Motors Connected & Torqued!")
            try: lbl_status.config(text="? MOTORS CONNECTED", fg="#2ecc71")
            except: pass
    except Exception as e:
        print(f"? Motor Init Error: {e}")

def reconnect_motors():
    print("? Attempting Hardware Reconnection...")
    try: port.closePort(); time.sleep(0.5); init_motors()
    except Exception as e: print(f"? Reconnect Failed: {e}")

def ms(ang, id): 
    try: pkt.write4ByteTxRx(port, id, 116, int((max(10, min(350, ang))/360.0)*4095))
    except Exception: print(f"?? MOTOR FREEZE ON ID {id}! Press 'RECONNECT MOTORS'.")

# GAITS
ul, rel, defo = 250, 105, 175
def act_fwd(): ms(rel-20, 3); ms(ul+70, 2); ms(220, 5); ms(220, 4); root.after(420); ms(ul-30, 3); ms(rel+20, 2); ms(220, 5); ms(220, 4); root.after(420)
def act_l_turn(): ms(ul,4); ms(defo,3); ms(rel,2); ms(ul,5); time.sleep(0.42); ms(ul,4); ms(ul,3); ms(defo,2); ms(rel,5); time.sleep(0.42); ms(ul,4); ms(rel,3); ms(ul,2); ms(defo,5); time.sleep(0.42)
def act_r_turn(): ms(ul,5); ms(defo,2); ms(rel,3); ms(ul,4); time.sleep(0.42); ms(ul,5); ms(ul,2); ms(defo,3); ms(rel,4); time.sleep(0.42); ms(ul,5); ms(rel,2); ms(ul,3); ms(defo,4); time.sleep(0.42)
def act_l_walk(): ms(ul,2); ms(ul,4); ms(ul,3); ms(rel,5); time.sleep(0.42); ms(ul,2); ms(ul,4); ms(ul,5); ms(rel,3); time.sleep(0.42)
def act_r_walk(): ms(ul,3); ms(ul,5); ms(ul,2); ms(rel,4); time.sleep(0.42); ms(ul,3); ms(ul,5); ms(ul,4); ms(rel,2); time.sleep(0.42)
def act_crawl():
    print(">>> AI LOCKED: Executing Crawl Sequence <<<")
    for _ in range(30): ms(defo,2); ms(defo,5); ms(ul,3); ms(rel,4); time.sleep(0.4); ms(ul,4); ms(rel,3); ms(defo,5); ms(defo,2); time.sleep(0.4); ms(defo,4); ms(defo,3); ms(rel,5); ms(ul,2); time.sleep(0.4); ms(rel,2); ms(ul,5); ms(defo,3); ms(defo,4); time.sleep(0.4)

action_dict = {0: ("FORWARD", act_fwd), 1: ("LEFT TURN", act_l_turn), 2: ("RIGHT TURN", act_r_turn), 3: ("LEFT WALK", act_l_walk), 4: ("RIGHT WALK", act_r_walk), 5: ("CRAWL", act_crawl)}

# ==========================================
# 4. DEPLOYMENT LOGGER & AUTO-PILOT
# ==========================================
SUM_LOG, STEP_LOG = "deploy_summary.csv", "deploy_steps.csv"
for log, headers in [(SUM_LOG, ['Traj_ID', 'Trial', 'Environment', 'Result', 'Time_Sec', 'Steps', 'Avg_Q']), 
                     (STEP_LOG, ['Traj_ID', 'Trial', 'Step_Num', 'Time_sec', 'Action', 'Max_Q'])]:
    if not os.path.exists(log):
        with open(log, 'w', newline='') as f: csv.writer(f).writerow(headers)

is_running = False; start_time = 0; step_count = 0; q_history = []; current_id, current_trial, current_env = 1, 1, "Track_A"

def start_run():
    global is_running, start_time, step_count, q_history, current_id, current_trial, current_env
    try: current_id = int(entry_id.get()); current_trial = int(entry_trial.get()); current_env = entry_env.get()
    except: pass
    is_running = True; start_time = time.time(); step_count = 0; q_history = []
    lbl_status.config(text=f"?? RUNNING Traj {current_id} (Trial {current_trial})", fg="#2ecc71")
    run_ai()

def log_result(result_string):
    global is_running
    if not is_running: return
    is_running = False
    elapsed = round(time.time() - start_time, 2)
    avg_q = round(np.mean(q_history), 2) if q_history else 0.0
    with open(SUM_LOG, 'a', newline='') as f: csv.writer(f).writerow([current_id, current_trial, current_env, result_string, elapsed, step_count, avg_q])
    lbl_status.config(text=f"?? LOGGED: {result_string} ({elapsed}s)", fg="#e74c3c")

def log_waypoint():
    print("?? HUMAN REFEREE: Waypoint Cleared!")
    lbl_status.config(text=f"?? WAYPOINT LOGGED!", fg="#f39c12")

def run_ai():
    global is_running, step_count
    if not is_running or ros_node is None: return
    
    obs = np.array(ros_node.grid, dtype=np.float32)
    obs = np.clip(obs, 0.0, 5.0) # Hardware Safety Clamp
    
    # PYTORCH INFERENCE
    with torch.no_grad():
        obs_tensor = torch.tensor(obs).unsqueeze(0) # Add batch dimension [1, 25]
        q_values = model(obs_tensor)
        best_q, action = torch.max(q_values, dim=1)
    
    q_val, act_val = round(best_q.item(), 2), int(action.item())
    q_history.append(q_val)
    step_count += 1
    
    # Analyze the 4 zones just like the environment does!
    top_avg = np.mean(obs[0:5])
    front_min = np.min([obs[7], obs[11], obs[12], obs[13], obs[17]])
    left_avg = np.mean([obs[10], obs[11], obs[15], obs[16]])
    right_avg = np.mean([obs[13], obs[14], obs[18], obs[19]])
    
    name, func = action_dict[act_val]
    print(f"S{step_count} | Fwd:{front_min:.1f}m Top:{top_avg:.1f}m L:{left_avg:.1f}m R:{right_avg:.1f}m | Q:{q_val} -> ACT: {name}")
    
    with open(STEP_LOG, 'a', newline='') as f:
        csv.writer(f).writerow([current_id, current_trial, step_count, round(time.time() - start_time, 2), name, q_val])
    
    func() # Physical Movement
    if is_running: root.after(10, run_ai)

# --- GUI ---
root = tk.Tk(); root.title("Immortal PyTorch Auto-Pilot"); root.geometry("500x550"); root.configure(bg="#2b2b2b")
tk.Label(root, text="Deployment Eval", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="white").pack(pady=5)
f1 = tk.Frame(root, bg="#2b2b2b"); f1.pack(pady=5)
tk.Label(f1, text="Traj ID:", bg="#2b2b2b", fg="white").grid(row=0,column=0)
entry_id = tk.Entry(f1, width=5); entry_id.grid(row=0,column=1); entry_id.insert(0, "1")
tk.Label(f1, text="  Trial:", bg="#2b2b2b", fg="white").grid(row=0,column=2)
entry_trial = tk.Entry(f1, width=5); entry_trial.grid(row=0,column=3); entry_trial.insert(0, "1")
tk.Label(root, text="Environment (e.g. 'Corner'):", bg="#2b2b2b", fg="white").pack()
entry_env = tk.Entry(root, width=20); entry_env.pack(); entry_env.insert(0, "Track_A")

tk.Button(root, text="? START RUN", bg="#27ae60", fg="white", font=("Arial", 12, "bold"), width=30, command=start_run).pack(pady=10)
tk.Button(root, text="?? MARK WAYPOINT (+50)", bg="#3498db", fg="white", width=30, command=log_waypoint).pack(pady=2)
tk.Button(root, text="?? MARK GOAL (Success)", bg="#f1c40f", font=("Arial", 10, "bold"), width=30, command=lambda: log_result("Goal")).pack(pady=2)
tk.Button(root, text="?? MARK CRASH (Fail)", bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), width=30, command=lambda: log_result("Crash")).pack(pady=2)
lbl_status = tk.Label(root, text="IDLE", font=("Arial", 12, "bold"), bg="#2b2b2b", fg="#bdc3c7"); lbl_status.pack(pady=10)
tk.Button(root, text="? RECONNECT MOTORS", bg="#f39c12", font=("Arial", 10, "bold"), width=30, command=reconnect_motors).pack(pady=5)

def kill():
    for i in [3,2,5,4]: pkt.write1ByteTxRx(port, i, 64, 0)
    root.destroy(); sys.exit()
tk.Button(root, text="? EXIT & KILL MOTORS", bg="darkred", fg="white", width=30, command=kill).pack(pady=5)

init_motors()
root.mainloop()