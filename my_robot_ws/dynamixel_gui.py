import sys
import time
import math
import tkinter as tk
from tkinter import messagebox
from dynamixel_sdk import *  # Dynamixel SDK library


# Control table addresses for XL-430
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

# Protocol version
PROTOCOL_VERSION = 2.0

# Default setting

DXL_IDS = {"FL": 3, "FR": 2, "BL": 5, "BR": 4}
BAUDRATE = 2000000
DEVICENAME = '/dev/ttyUSB0'  # Change this to match your setup

TORQUE_ENABLE = 1  # Enable torque
TORQUE_DISABLE = 0  # Disable torque

# Position limits (0 to 360 degrees mapped to 0 to 4095)
DXL_MIN_POSITION_VALUE = 0  # 0 degrees
DXL_MAX_POSITION_VALUE = 4095  # 360 degrees
ul = 210         # Perfect Standing Height
ulr = 280        # Original upper limit for left turn
ulc = 220
relc = 160
defc = 115
rel = 105        # Perfect Step Height
defo = 175       # Relaxed position

class SLOT_PID:
    def __init__(self, kp, ki, kd, limit=30):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.cumError = 0.0
        self.lastError = 0.0
        self.previousTime = time.time()
        self.limit = limit # Maximum degrees it can correct (e.g., +/- 30 deg)

    def compute(self, current_val, setpoint):
        currentTime = time.time()
        elapsedTime = currentTime - self.previousTime
        if elapsedTime <= 0: elapsedTime = 0.01 # Prevent divide-by-zero
        
        error = setpoint - current_val
        self.cumError += error * elapsedTime
        rateError = (error - self.lastError) / elapsedTime
        
        out = (self.kp * error) + (self.ki * self.cumError) + (self.kd * rateError)
        
        self.lastError = error
        self.previousTime = currentTime
        
        # Clamp the output so it doesn't break the legs!
        return max(-self.limit, min(self.limit, out))

# Create the 3 controllers using Saumya's exact V1 constants
pitch_pid = SLOT_PID(kp=10.0, ki=0.07, kd=3.0)
roll_pid = SLOT_PID(kp=10.0, ki=0.07, kd=3.0)
yaw_pid = SLOT_PID(kp=5.0, ki=0.0, kd=0.0)

#---------------------------------------------------------------------------------------------------------

def angle_to_position(angle):
    angle = max(10, min(350, angle))  # Restrict input to 10° - 350°
    return int((angle / 360.0) * 4095)

# Initialize PortHandler and PacketHandler
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if not portHandler.openPort():
    messagebox.showerror("Error", "Failed to open the port!")
    sys.exit()

if not portHandler.setBaudRate(BAUDRATE):
    messagebox.showerror("Error", "Failed to set the baudrate!")
    sys.exit()

# Enable torque for each Dynamixel
for dxl_id in DXL_IDS.values():
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

def move_servo(angle, dxl_id):
    goal_position = angle_to_position(angle)
    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, goal_position)

def move_servos():
    angle = int(angle_slider.get())
    for dxl_id in DXL_IDS.values():
        move_servo(angle, dxl_id)
    root.after(2000, update_positions)

def forward_gait():
    """Implements the forward walking motion by controlling the servos."""
    
    # Step 1
    move_servo(rel, DXL_IDS["FL"]) # 85
    move_servo(ul, DXL_IDS["FR"])  # 280
    move_servo(ul, DXL_IDS["BL"])
    move_servo(ul, DXL_IDS["BR"])
    root.after(420)  # Delay 1 sec
    
    # Step 2
    move_servo(ul, DXL_IDS["FL"])  # 150
    move_servo(rel, DXL_IDS["FR"]) # 125
    move_servo(ul, DXL_IDS["BL"])
    move_servo(ul, DXL_IDS["BR"])
    root.after(420)  # Delay 1 sec

def update_positions():
    for leg, dxl_id in DXL_IDS.items():
        dxl_present_position, _, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_POSITION)
        dxl_present_position &= 0xFFFFFFFF  # Handle negative values
        position_labels[leg].config(text=f"{leg} Position: {dxl_present_position}")

def backward_gait():
    """Implements the backward walking motion, perfectly mirrored from forward_gait."""
    
    # Step 1: Front legs anchor. BL reaches, BR anchors.
    move_servo(ul, DXL_IDS["FL"])
    move_servo(ul, DXL_IDS["FR"])
    move_servo(rel+7, DXL_IDS["BL"])
    move_servo(ul, DXL_IDS["BR"])
    root.after(420)  
    
    # Step 2: Front legs anchor. BL anchors, BR reaches.
    move_servo(ul, DXL_IDS["FL"])
    move_servo(ul, DXL_IDS["FR"])
    move_servo(ul+7, DXL_IDS["BL"])
    move_servo(rel, DXL_IDS["BR"])
    root.after(420)

def right_turn():
    """
    Spins Anti-Clockwise (Left).
    Anchor is Back-Right (BR). 
    Uses 3-step rowing to throw the heavy front end without dragging.
    """
    # Step 1: Anchor BR. Reach BL. Plant FL. Lift FR.
    move_servo(ul, DXL_IDS["BR"])    # THE ANCHOR (Locked)
    move_servo(rel, DXL_IDS["BL"])   # Reaching 
    move_servo(ul, DXL_IDS["FL"])    # Pulls the chassis
    move_servo(defo, DXL_IDS["FR"])  # Lifting (so it doesn't drag)
    root.after(420)
    
    # Step 2: Anchor BR. Plant BL. Lift FL. Reach FR.
    move_servo(ul, DXL_IDS["BR"])    # THE ANCHOR 
    move_servo(ul, DXL_IDS["BL"])    # Pulls the chassis
    move_servo(defo, DXL_IDS["FL"])  # Lifting (so it doesn't drag)
    move_servo(rel, DXL_IDS["FR"])   # Reaching 
    root.after(420)

    # Step 3: Anchor BR. Lift BL. Reach FL. Plant FR.
    move_servo(ul, DXL_IDS["BR"])    # THE ANCHOR
    move_servo(defo, DXL_IDS["BL"])  # Lifting (so it doesn't drag)
    move_servo(rel, DXL_IDS["FL"])   # Reaching
    move_servo(ul, DXL_IDS["FR"])    # Pulls the chassis
    root.after(420)


def left_turn():
    """
    Spins Clockwise (Right).
    Exact mirror of right_turn().
    Anchor is Back-Left (BL).
    """

    # Step 1: Anchor BL. Reach BR. Plant FR. Lift FL.
    move_servo(ul-80, DXL_IDS["BL"])      # Anchor
    move_servo(rel, DXL_IDS["BR"])     # Reach
    move_servo(ul+40, DXL_IDS["FR"])      # Pull
    move_servo(defo, DXL_IDS["FL"])    # Lift
    root.after(420)

    # Step 2: Anchor BL. Plant BR. Lift FR. Reach FL.
    move_servo(ul-80, DXL_IDS["BL"])      # Anchor
    move_servo(ul+40, DXL_IDS["BR"])      # Pull
    move_servo(defo, DXL_IDS["FR"])    # Lift
    move_servo(rel, DXL_IDS["FL"])     # Reach
    root.after(420)

    # Step 3: Anchor BL. Lift BR. Reach FR. Plant FL.
    move_servo(ul-80, DXL_IDS["BL"])      # Anchor
    move_servo(defo, DXL_IDS["BR"])    # Lift
    move_servo(rel, DXL_IDS["FR"])     # Reach
    move_servo(ul-30, DXL_IDS["FL"])      # Pull
    root.after(420)
    
def crawl_forward():
    # Phase 1: FR & BL are stable anchors (defo). FL pulls (ul), BR steps (rel).
    move_servo(defo, DXL_IDS["FR"])
    move_servo(defo, DXL_IDS["BL"])
    move_servo(ul, DXL_IDS["FL"])
    move_servo(rel, DXL_IDS["BR"])
    root.after(400)  # Sped up from 1000ms to keep momentum

    # Phase 2: BR pulls (ul), FL steps (rel).
    move_servo(ul+45, DXL_IDS["BR"])
    move_servo(rel, DXL_IDS["FL"])
    move_servo(defo, DXL_IDS["BL"])
    move_servo(defo, DXL_IDS["FR"])
    root.after(400)

    # Phase 3: FL & BR are stable anchors (defo). FR pulls (ul), BL steps (rel).
    move_servo(defo, DXL_IDS["BR"])
    move_servo(defo, DXL_IDS["FL"])
    move_servo(rel, DXL_IDS["BL"])
    move_servo(ul, DXL_IDS["FR"])
    root.after(400)

    # Phase 4: BL pulls (ul), FR steps (rel).
    move_servo(rel, DXL_IDS["FR"])
    move_servo(ul, DXL_IDS["BL"])
    move_servo(defo, DXL_IDS["FL"])
    move_servo(defo, DXL_IDS["BR"])
    root.after(400)
    
def crawl_backward():
    """Tactical stable crawl in the backward direction."""
    # Phase 1: FR & BL anchor (defo). BR pulls (ul), FL steps (rel).
    move_servo(defo, DXL_IDS["FR"])
    move_servo(defo, DXL_IDS["BL"])
    move_servo(rel, DXL_IDS["FL"])  # Stepping
    move_servo(ul, DXL_IDS["BR"])   # Pulling backward
    root.after(400)

    # Phase 2: FL pulls (ul), BR steps (rel).
    move_servo(rel, DXL_IDS["BR"])  # Stepping
    move_servo(ul, DXL_IDS["FL"])   # Pulling backward
    move_servo(defo, DXL_IDS["BL"])
    move_servo(defo, DXL_IDS["FR"])
    root.after(400)

    # Phase 3: FL & BR anchor (defo). BL pulls (ul), FR steps (rel).
    move_servo(defo, DXL_IDS["BR"])
    move_servo(defo, DXL_IDS["FL"])
    move_servo(ul, DXL_IDS["BL"])   # Pulling backward
    move_servo(rel, DXL_IDS["FR"])  # Stepping
    root.after(400)

    # Phase 4: FR pulls (ul), BL steps (rel).
    move_servo(ul, DXL_IDS["FR"])   # Pulling backward
    move_servo(rel, DXL_IDS["BL"])  # Stepping
    move_servo(defo, DXL_IDS["FL"])
    move_servo(defo, DXL_IDS["BR"])
    root.after(400)

def right_walk():
    """Strafe to the Left using the Squat Trick"""
    # Step 1: BR reaches. Others squat and pull (200).
    move_servo(ul, DXL_IDS["FL"])
    move_servo(ul, DXL_IDS["BL"])
    move_servo(ul, DXL_IDS["FR"])
    move_servo(rel, DXL_IDS["BR"])
    root.after(420)
    
    # Step 2: FR reaches. Others squat and pull (200).
    move_servo(ul, DXL_IDS["FL"])
    move_servo(ul, DXL_IDS["BL"])
    move_servo(ul, DXL_IDS["BR"])
    move_servo(rel, DXL_IDS["FR"])
    root.after(420)

def left_walk():
    """Strafe to the Right using the Squat Trick"""
    # Step 1: BL reaches. Others squat and pull (200).
    move_servo(ul, DXL_IDS["FR"])
    move_servo(ul, DXL_IDS["BR"])
    move_servo(ul, DXL_IDS["FL"])
    move_servo(rel, DXL_IDS["BL"])
    root.after(420)
    
    # Step 2: FL reaches. Others squat and pull (200).
    move_servo(ul, DXL_IDS["FR"])
    move_servo(ul, DXL_IDS["BR"])
    move_servo(ul, DXL_IDS["BL"])
    move_servo(rel, DXL_IDS["FL"])
    root.after(420)

def walk_30_deg():
    """30 Deg (More Forward). Back legs LOCKED. FL pulls deep to drag nose forward."""
    # Step 1: FR reaches.
    move_servo(ul, DXL_IDS["FL"])  
    move_servo(rel, DXL_IDS["FR"]) 
    move_servo(ul, DXL_IDS["BL"])  
    move_servo(ul, DXL_IDS["BR"])  
    root.update()
    time.sleep(0.42)
    
    # Step 2: FR plants. Back legs stay LOCKED at ul (250) to stop curving!
    move_servo(170, DXL_IDS["FL"]) # Deep pull to drag it more forward (30)
    move_servo(ul, DXL_IDS["FR"])  # Full plant
    move_servo(180, DXL_IDS["BL"])  
    move_servo(180, DXL_IDS["BR"])  
    root.update()
    time.sleep(0.42)

def walk_fr():
    """45 Deg. Using the 1-to-1 tick-to-degree ratio: FL at 115!"""
    # Step 1: FR reaches
    move_servo(ul, DXL_IDS["FL"])  
    move_servo(rel, DXL_IDS["FR"]) 
    move_servo(ul, DXL_IDS["BL"])  
    move_servo(ul, DXL_IDS["BR"])  
    root.update()
    time.sleep(0.42)
    
    # Step 2: The Golden 45° Pull
    move_servo(115, DXL_IDS["FL"]) # Exactly 10 ticks (degrees) shifted from the 55° mark!
    move_servo(ul, DXL_IDS["FR"])  # Push!
    move_servo(rel, DXL_IDS["BL"]) # Belly Drag
    move_servo(rel, DXL_IDS["BR"]) # Belly Drag
    root.update()
    time.sleep(0.42)

def walk_60_deg():
    """60 Deg (More Right). Uses FL for a short forward push to stop clockwise turning!"""
    # Step 1: FR does a FULL reach (105). FL does a SHORT reach (200). 
    move_servo(ul, DXL_IDS["FL"]) # Short reach to prepare the forward push
    move_servo(rel, DXL_IDS["FR"]) # Full reach
    move_servo(ul, DXL_IDS["BL"])  # Locked
    move_servo(ul, DXL_IDS["BR"])  # Locked
    root.update()
    time.sleep(0.42)
    
    # Step 2: BOTH front legs push to 250!
    # FL gives the forward push you suggested, perfectly canceling the clockwise turn!
    move_servo(150, DXL_IDS["FL"])  # PUSH!
    move_servo(ul, DXL_IDS["FR"])  # PUSH!
    move_servo(150, DXL_IDS["BL"])  # Locked
    move_servo(150, DXL_IDS["BR"])  # Locked
    root.update()
    time.sleep(0.42)

def walk_120_deg():
    """120 Deg (Backward-Right, more Right). Mirror of 60° pulling backwards."""
    # Step 1: BR reaches
    move_servo(ul, DXL_IDS["FL"])  
    move_servo(ul, DXL_IDS["FR"]) 
    move_servo(ul, DXL_IDS["BL"])  
    move_servo(rel, DXL_IDS["BR"]) # BR reaches back! 
    root.update()
    time.sleep(0.42)
    
    # Step 2: BR plants. BL, FL, and FR pull to 150!
    move_servo(150, DXL_IDS["FL"]) # Drag
    move_servo(150, DXL_IDS["FR"]) # Drag
    move_servo(150, DXL_IDS["BL"]) # Steer
    move_servo(ul, DXL_IDS["BR"])  # Full plant
    root.update()
    time.sleep(0.42)

def walk_br():
    """135 Deg (Backward-Right 45°). Mirror of 45°."""
    # Step 1: BR reaches
    move_servo(ul, DXL_IDS["FL"])  
    move_servo(ul, DXL_IDS["FR"]) 
    move_servo(ul, DXL_IDS["BL"])  
    move_servo(rel, DXL_IDS["BR"]) 
    root.update()
    time.sleep(0.42)
    
    # Step 2: BR plants. BL steers at 115. Front does Belly Drag!
    move_servo(rel, DXL_IDS["FL"]) # Belly Drag
    move_servo(rel, DXL_IDS["FR"]) # Belly Drag
    move_servo(115, DXL_IDS["BL"]) # 45° Steering depth
    move_servo(ul, DXL_IDS["BR"])  # Full plant
    root.update()
    time.sleep(0.42)

def walk_150_deg():
    """150 Deg (Backward-Right, more Back). Mirror of 30° pulling backwards."""
    # Step 1: BR reaches
    move_servo(ul, DXL_IDS["FL"])  
    move_servo(ul, DXL_IDS["FR"]) 
    move_servo(ul, DXL_IDS["BL"])  
    move_servo(rel, DXL_IDS["BR"]) 
    root.update()
    time.sleep(0.42)
    
    # Step 2: BR plants. BL steers hard backward.
    move_servo(180, DXL_IDS["FL"]) # Drag
    move_servo(180, DXL_IDS["FR"]) # Drag
    move_servo(170, DXL_IDS["BL"]) # Deep steer back
    move_servo(ul, DXL_IDS["BR"])  # Full plant
    root.update()
    time.sleep(0.42)

def walk_210_deg():
    """210 Deg (Backward-Left, more Back). Mirror of 30°."""
    # Step 1: BL reaches
    move_servo(ul, DXL_IDS["FL"])  
    move_servo(ul, DXL_IDS["FR"]) 
    move_servo(rel, DXL_IDS["BL"]) # BL reaches back! 
    move_servo(ul, DXL_IDS["BR"])  
    root.update()
    time.sleep(0.42)
    
    # Step 2: BL plants. BR steers back.
    move_servo(180, DXL_IDS["FL"]) # Drag
    move_servo(180, DXL_IDS["FR"]) # Drag
    move_servo(ul, DXL_IDS["BL"])  # Full plant
    move_servo(170, DXL_IDS["BR"]) # Deep steer back
    root.update()
    time.sleep(0.42)

def walk_bl():
    """225 Deg (Backward-Left 45°). Mirror of 45°."""
    # Step 1: BL reaches
    move_servo(ul, DXL_IDS["FL"])  
    move_servo(ul, DXL_IDS["FR"]) 
    move_servo(rel, DXL_IDS["BL"])  
    move_servo(ul, DXL_IDS["BR"]) 
    root.update()
    time.sleep(0.42)
    
    # Step 2: BL plants. BR steers 115. Front Belly Drag.
    move_servo(rel, DXL_IDS["FL"]) # Belly drag
    move_servo(rel, DXL_IDS["FR"]) # Belly drag
    move_servo(ul, DXL_IDS["BL"])  # Full plant
    move_servo(115, DXL_IDS["BR"]) # 45° Steering depth
    root.update()
    time.sleep(0.42)

def walk_240_deg():
    """240 Deg (Backward-Left, more Left). Mirror of 60°."""
    # Step 1: BL reaches
    move_servo(ul, DXL_IDS["FL"])  
    move_servo(ul, DXL_IDS["FR"]) 
    move_servo(rel, DXL_IDS["BL"])  
    move_servo(ul, DXL_IDS["BR"]) 
    root.update()
    time.sleep(0.42)
    
    # Step 2: BL plants. BR, FL, and FR pull to 150!
    move_servo(150, DXL_IDS["FL"]) # Drag
    move_servo(150, DXL_IDS["FR"]) # Drag
    move_servo(ul, DXL_IDS["BL"])  # Full plant
    move_servo(150, DXL_IDS["BR"]) # Steer
    root.update()
    time.sleep(0.42)

def walk_330_deg():
    """330 Deg (Forward-Left, more Forward). Mirror of 30°."""
    # Step 1: FL reaches.
    move_servo(rel, DXL_IDS["FL"])  
    move_servo(ul, DXL_IDS["FR"]) 
    move_servo(ul, DXL_IDS["BL"])  
    move_servo(ul, DXL_IDS["BR"])  
    root.update()
    time.sleep(0.42)
    
    # Step 2: FL plants. FR steers. Back legs drag lightly.
    move_servo(ul, DXL_IDS["FL"])  # Full plant
    move_servo(170, DXL_IDS["FR"]) # Deep pull to steer forward
    move_servo(180, DXL_IDS["BL"]) # Drag 
    move_servo(180, DXL_IDS["BR"]) # Drag 
    root.update()
    time.sleep(0.42)

def walk_fl():
    """315 Deg (Forward-Left 45°). Mirror of walk_fr."""
    # Step 1: FL reaches
    move_servo(rel, DXL_IDS["FL"])  
    move_servo(ul, DXL_IDS["FR"]) 
    move_servo(ul, DXL_IDS["BL"])  
    move_servo(ul, DXL_IDS["BR"])  
    root.update()
    time.sleep(0.42)
    
    # Step 2: FL plants. FR steers to 115. Back legs full belly drag.
    move_servo(ul, DXL_IDS["FL"])  # Full plant
    move_servo(115, DXL_IDS["FR"]) # 45° Steering depth
    move_servo(rel, DXL_IDS["BL"]) # Belly Drag
    move_servo(rel, DXL_IDS["BR"]) # Belly Drag
    root.update()
    time.sleep(0.42)

def walk_300_deg():
    """300 Deg (Forward-Left, more Left). Mirror of 60°."""
    # Step 1: FL reaches 
    move_servo(rel, DXL_IDS["FL"]) 
    move_servo(ul, DXL_IDS["FR"]) 
    move_servo(ul, DXL_IDS["BL"])  
    move_servo(ul, DXL_IDS["BR"])  
    root.update()
    time.sleep(0.42)
    
    # Step 2: FL plants. FR, BL, and BR all squat to 150 to pull left!
    move_servo(ul, DXL_IDS["FL"])  # Full plant
    move_servo(150, DXL_IDS["FR"]) # Steer
    move_servo(150, DXL_IDS["BL"]) # Drag
    move_servo(150, DXL_IDS["BR"]) # Drag
    root.update()
    time.sleep(0.42)



def close_app():
    for dxl_id in DXL_IDS.values():
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    portHandler.closePort()
    root.destroy()

# ==========================================
# GUI SETUP & D-PAD LAYOUT (NEW)
# ==========================================
root = tk.Tk()
root.title("SLOT Bot Controller")
root.geometry("550x450") # Wide enough to fit everything nicely!

# Top Frame for Sliders
top_frame = tk.Frame(root)
top_frame.pack(pady=10)

angle_slider = tk.Scale(top_frame, from_=0, to=360, orient=tk.HORIZONTAL, label="Angle (10-350)", length=200)
angle_slider.grid(row=0, column=0, columnspan=2, pady=5)
move_button = tk.Button(top_frame, text="Move Servos", command=move_servos)
move_button.grid(row=1, column=0, columnspan=2)

# Position Labels Frame
pos_frame = tk.Frame(root)
pos_frame.pack(pady=5)
position_labels = {}
col = 0
for leg in ["FL", "FR", "BL", "BR"]:
    lbl = tk.Label(pos_frame, text=f"{leg}: -")
    lbl.grid(row=0, column=col, padx=10)
    position_labels[leg] = lbl
    col += 1

# ==========================================
# PROFESSIONAL UI COLOR PALETTE (Hex Codes)
# ==========================================
BG_MAIN = "#2b2b2b"          # Dark gray background for the frame
BTN_CARDINAL = "#3b4d61"     # Dark slate blue for main directions (0, 90, 180, 270)
BTN_CORNER = "#4b6584"       # Medium slate for 45, 135, 225, 315
BTN_FINE = "#778ca3"         # Lighter slate for the 30, 60, 120 etc.
BTN_ACTION = "#0fb9b1"       # Teal for Crawls and Turns
BTN_EXIT = "#eb3b5a"         # Muted crimson for exit
TEXT_COLOR = "#ffffff"       # White text

FONT_MAIN = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")

# ==========================================
# 5x5 OMNIDIRECTIONAL COMPASS GRID
# ==========================================
compass_frame = tk.Frame(root, bg=BG_MAIN, padx=10, pady=10)
compass_frame.pack(pady=15)

tk.Label(compass_frame, text="SLOT OMNIDIRECTIONAL CONTROL MATRIX", 
         font=("Segoe UI", 12, "bold"), bg=BG_MAIN, fg="#a5b1c2").grid(row=0, column=0, columnspan=5, pady=(0, 15))

# --- BUTTON CONFIGURATION HELPER ---
# This dictionary makes all buttons uniform in size and styling
btn_style = {"width": 11, "height": 2, "fg": TEXT_COLOR, "font": FONT_MAIN, "relief": "ridge", "bd": 1}
bold_style = {"width": 11, "height": 2, "fg": TEXT_COLOR, "font": FONT_BOLD, "relief": "ridge", "bd": 2}

# --- Row 1: Upper Forward Arch ---
btn_315 = tk.Button(compass_frame, text="↖ 315 (FL)", bg=BTN_CORNER, command=walk_fl, **btn_style)
btn_330 = tk.Button(compass_frame, text="⇖ 330", bg=BTN_FINE, command=walk_330_deg, **btn_style)
btn_0   = tk.Button(compass_frame, text="⇑ 0 (FWD)", bg=BTN_CARDINAL, command=forward_gait, **bold_style)
btn_30  = tk.Button(compass_frame, text="⇗ 30", bg=BTN_FINE, command=walk_30_deg, **btn_style)
btn_45  = tk.Button(compass_frame, text="↗ 45 (FR)", bg=BTN_CORNER, command=walk_fr, **btn_style)

btn_315.grid(row=1, column=0, padx=4, pady=4)
btn_330.grid(row=1, column=1, padx=4, pady=4)
btn_0.grid(row=1, column=2, padx=4, pady=4)
btn_30.grid(row=1, column=3, padx=4, pady=4)
btn_45.grid(row=1, column=4, padx=4, pady=4)

# --- Row 2: Upper Mid Arch & Forward Crawl ---
btn_300 = tk.Button(compass_frame, text="⇖ 300", bg=BTN_FINE, command=walk_300_deg, **btn_style)
btn_c_f = tk.Button(compass_frame, text="⇡ Crawl FWD", bg=BTN_ACTION, command=crawl_forward, **bold_style)
btn_60  = tk.Button(compass_frame, text="⇗ 60", bg=BTN_FINE, command=walk_60_deg, **btn_style)

btn_300.grid(row=2, column=0, padx=4, pady=4)
btn_c_f.grid(row=2, column=2, padx=4, pady=4) # Placed in the previously empty center-top slot!
btn_60.grid(row=2, column=4, padx=4, pady=4)

# --- Row 3: Equator (Left, Center, Right, Turns) ---
btn_270 = tk.Button(compass_frame, text="⇐ 270 (L)", bg=BTN_CARDINAL, command=left_walk, **bold_style)
btn_tl  = tk.Button(compass_frame, text="↶ L Turn", bg=BTN_ACTION, command=left_turn, **bold_style)

# A non-clickable center dot to anchor the compass aesthetically
lbl_center = tk.Label(compass_frame, text="●", bg=BG_MAIN, fg="#4b6584", font=("Arial", 16))

btn_tr  = tk.Button(compass_frame, text="↷ R Turn", bg=BTN_ACTION, command=right_turn, **bold_style)
btn_90  = tk.Button(compass_frame, text="⇒ 90 (R)", bg=BTN_CARDINAL, command=right_walk, **bold_style)

btn_270.grid(row=3, column=0, padx=4, pady=4)
btn_tl.grid(row=3, column=1, padx=4, pady=4)
lbl_center.grid(row=3, column=2, padx=4, pady=4) # True center of the compass
btn_tr.grid(row=3, column=3, padx=4, pady=4)
btn_90.grid(row=3, column=4, padx=4, pady=4)

# --- Row 4: Lower Mid Arch & Backward Crawl ---
btn_240 = tk.Button(compass_frame, text="⇙ 240", bg=BTN_FINE, command=walk_240_deg, **btn_style)
btn_c_b = tk.Button(compass_frame, text="⇣ Crawl BWD", bg=BTN_ACTION, command=crawl_backward, **bold_style)
btn_120 = tk.Button(compass_frame, text="⇘ 120", bg=BTN_FINE, command=walk_120_deg, **btn_style)

btn_240.grid(row=4, column=0, padx=4, pady=4)
btn_c_b.grid(row=4, column=2, padx=4, pady=4) # Placed in the previously empty center-bottom slot!
btn_120.grid(row=4, column=4, padx=4, pady=4)

# --- Row 5: Lower Backward Arch ---
btn_225 = tk.Button(compass_frame, text="↙ 225 (BL)", bg=BTN_CORNER, command=walk_bl, **btn_style)
btn_210 = tk.Button(compass_frame, text="⇙ 210", bg=BTN_FINE, command=walk_210_deg, **btn_style)
btn_180 = tk.Button(compass_frame, text="⇓ 180 (BWD)", bg=BTN_CARDINAL, command=backward_gait, **bold_style)
btn_150 = tk.Button(compass_frame, text="⇘ 150", bg=BTN_FINE, command=walk_150_deg, **btn_style)
btn_135 = tk.Button(compass_frame, text="↘ 135 (BR)", bg=BTN_CORNER, command=walk_br, **btn_style)

btn_225.grid(row=5, column=0, padx=4, pady=4)
btn_210.grid(row=5, column=1, padx=4, pady=4)
btn_180.grid(row=5, column=2, padx=4, pady=4)
btn_150.grid(row=5, column=3, padx=4, pady=4)
btn_135.grid(row=5, column=4, padx=4, pady=4)

# ==========================================
# EXIT FRAME
# ==========================================
bottom_frame = tk.Frame(root)
bottom_frame.pack(pady=10)

btn_exit = tk.Button(bottom_frame, text="⚠ EXIT & DISABLE TORQUE", width=30, height=2, 
                     bg=BTN_EXIT, fg=TEXT_COLOR, font=FONT_BOLD, relief="ridge", bd=2, command=close_app)
btn_exit.pack()

root.mainloop()