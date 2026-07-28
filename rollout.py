print("Importing Required libraries")
import time
import torch
import serial
import zmq
import json
import threading
import numpy as np
import datetime
from collections import deque
from redq import REDQ as TD3
from dynamixel_sdk import *
import sys
from scipy.spatial.transform import Rotation as R
import multiprocessing
w1,w2 = 10,0.1
print("import complete, initizing zmq")
ctx = zmq.Context()
FRAME_STACKING = 4
ACT_SIZE = 4
OBS_SIZE = 21* FRAME_STACKING
MIN_POS = 800
MAX_POS = 3800
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132  # Address for current position (for X-series)
ADDR_PRESENT_LOAD = 126      # Address for load/torque
ADDR_REALTIME_TICK = 120
# Protocol version
PROTOCOL_VERSION = 2.0

# Default setting
DXL_IDS = {"FL": 3, "FR": 2, "BL": 5, "BR": 4}  # Servo IDs
BAUDRATE =  2000000
DEVICENAME = '/dev/ttyUSB0'  # Change this to match your setup

TORQUE_ENABLE = 1  # Enable torque
TORQUE_DISABLE = 0  # Disable torque

# Position limits (0 to 360 degrees mapped to 0 to 4095)
DXL_MIN_POSITION_VALUE = 0  # 0 degrees
DXL_MAX_POSITION_VALUE = 4095  # 360 degrees

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
print("Dynamixel parameters set sucessfully")
dynamixel_lock = multiprocessing.Lock()

if not portHandler.openPort():
    print("Error: Failed to open the port!", file=sys.stderr)
    sys.exit(1)
print("Port opened")
if not portHandler.setBaudRate(2000000):
    print("Error: Failed to set the baudrate!", file=sys.stderr)
    sys.exit(1)
print("Port baud rate is set")
last_vslam_data = None
last_ant_meas = None
last_vslam_meas = None
last_frame_ant_meas = None
last_frame_vslam_meas = None
last_frame_jpos = None

for dxl_id in DXL_IDS.values():
         packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)


# adjust your endpoint
#past_obses = deque([np.zeros(OBS_SIZE//FRAME_STACKING)]*FRAME_STACKING, maxlen=FRAME_STACKING)

def collect_and_distribute_measurements(child_conn):
    context = zmq.Context()
    vslam_socket = context.socket(zmq.REP)
    vslam_socket.bind("tcp://127.0.0.1:5555")  # Child process now owns this socket
    ultra_serial = serial.Serial('/dev/ttyACM0', 9600, timeout=0.1)
    time.sleep(2)  # Wait for Arduino to reboot and stabiliz
    ultra_serial.reset_input_buffer()
    print("Ultrasonic is initiziled")
    print("observations collection started")
    last_ant_time = 0
    last_ant_meas = None
    last_vslam_meas = None
    last_vslam_data = None

    while True:
        last_ant_meas = get_dyn_data()
        last_vslam_meas = vslam_data(vslam_socket, ultra_serial)  # Pass socket explicitly
        vslam_socket.send_json({"status": "received"})
        #print(last_vslam_meas)
        if child_conn.poll():
            child_conn.recv()
            child_conn.send([last_ant_meas, last_vslam_meas])

def vslam_data(vslam_socket, ultra_serial):
    global last_vslam_data
    message = vslam_socket.recv_json()
    current_time = time.time()
    current_epoch_ms = int(current_time * 1000)

    x = message["x"]
    y = message["y"]
    vslam_z = message["z"]
    qx = message["qx"]
    qy = message["qy"]
    qz = message["qz"]
    qw = message["qw"]

    r = R.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)


    z = vslam_z
    if ultra_serial:
       try:
           ultra_serial.reset_input_buffer()
           line = ultra_serial.readline().decode('utf-8').strip()
           if line:
                #ultrasonic_z = float(line) / 100.0
                ultrasonic_z = (float(line) - 5.2) / 100.0
                z = ultrasonic_z
#                print("the ultrasonic god has answered with",z)
       except Exception as e:
            print(f"[ULTRASONIC ERROR] {e}")
    if last_vslam_data is not None:
        dt = current_time - last_vslam_data["time"]
        xvel = (x - last_vslam_data["x"]) / dt
        yvel = (y - last_vslam_data["y"]) / dt
        zvel = (z - last_vslam_data["z"]) / dt
    else:
        xvel = yvel = zvel = 0.0

    last_vslam_data = {
        "x": x,
        "y": y,
        "z": z,
        "time": current_time
    }

    return {
        "x": x,
        "y": y,
        "z": z,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "xvel": xvel,
        "yvel": yvel,
        "zvel": zvel,
        "server_epoch_ms": current_epoch_ms
    }


def get_dyn_data():
    #print("d1")
    dyn_data = {}
    timestamp = time.time()

    with dynamixel_lock:
        for i, dxl_id in enumerate([2, 3, 4, 5]):
            position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
                portHandler, dxl_id, ADDR_PRESENT_POSITION
            )
            if dxl_comm_result != 0:
                print(f"[COMM ERROR] ID {dxl_id}: {packetHandler.getTxRxResult(dxl_comm_result)}")
                dyn_data[f"s{i+1}_angle"] = None
            elif dxl_error != 0:
                print(f"[DYNAMIXEL ERROR] ID {dxl_id}: {packetHandler.getRxPacketError(dxl_error)}")
                dyn_data[f"s{i+1}_angle"] = None
            else:
                dyn_data[f"s{i+1}_angle"] = position

    dyn_data["ant_time"] = timestamp

    return dyn_data

def tget_dyn_data():
    dyn_data = {}
    timestamp = time.time()

    with dynamixel_lock:
        for i, dxl_id in enumerate([2, 3, 4, 5]):
            position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
                portHandler, dxl_id, ADDR_PRESENT_POSITION
            )

            if dxl_comm_result != 0:
                print(f"[COMM ERROR] ID {dxl_id}: {packetHandler.getTxRxResult(dxl_comm_result)}")
                dyn_data[f"s{i+1}_angle"] = None
                continue

            if dxl_error != 0:
                print(f"[DYNAMIXEL ERROR] ID {dxl_id}: {packetHandler.getRxPacketError(dxl_error)}")

                # Reboot the servo
                print(f"[REBOOTING] ID {dxl_id} due to error...")
                rb_result, rb_error = packetHandler.reboot(portHandler, dxl_id)

                if rb_result != COMM_SUCCESS:
                    print(f"[REBOOT FAIL] ID {dxl_id}: {packetHandler.getTxRxResult(rb_result)}")
                elif rb_error != 0:
                    print(f"[REBOOT ERROR] ID {dxl_id}: {packetHandler.getRxPacketError(rb_error)}")
                else:
                    print(f"[REBOOT SUCCESS] ID {dxl_id}")
                    # Add delay after reboot
                    time.sleep(1)
                    # Call your custom reset function
                    reset()
                    time.sleep(2)

                dyn_data[f"s{i+1}_angle"] = None
                continue

            dyn_data[f"s{i+1}_angle"] = position

    dyn_data["ant_time"] = timestamp
    return dyn_data


class EnvironmentHandler():
    def __init__(self):
        print("pont3")
        self.running = True
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.p = multiprocessing.Process(target=collect_and_distribute_measurements, args=(self.child_conn,))
        self.p.start()
        self.zero_j_cnt = 0
        self.zero_c_cnt = 0
        self.past_obses = deque([np.zeros(OBS_SIZE//FRAME_STACKING)]*FRAME_STACKING, maxlen=FRAME_STACKING)




    def get_obs(self):
     global last_ant_meas, last_vslam_meas, last_frame_ant_meas, last_frame_vslam_meas, last_frame_jpos

     self.parent_conn.send([])
     last_ant_meas, last_vslam_meas = self.parent_conn.recv()
     #print(last_ant_meas)
     while last_ant_meas is None and last_vslam_meas is None:
        time.sleep(0.01)

     default_dt = 0.05  # s

     camera_dt = (last_vslam_meas['server_epoch_ms'] - last_frame_vslam_meas['server_epoch_ms']) / 1000 if last_frame_vslam_meas is not None else None
     joint_dt = (float(last_ant_meas['ant_time']) - float(last_frame_ant_meas['ant_time'])) / 1000 if last_frame_ant_meas is not None else None

     # Handle camera_dt
     if camera_dt == 0:
        self.zero_c_cnt += 1
        if self.zero_c_cnt > 3:
            print("observations stuck, quitting (camera)")
            quit()
        camera_dt = default_dt
     else:
        self.zero_c_cnt = 0

     # Handle joint_dt
     if joint_dt == 0:
        self.zero_j_cnt += 1
        if self.zero_j_cnt > 3:
            print("observations stuck, quitting (serial)")
            quit()
        joint_dt = default_dt
     else:
        self.zero_j_cnt = 0

     # Get torso velocity (from camera)
     x_vel = last_vslam_meas["xvel"]
     y_vel = last_vslam_meas["yvel"]
     z_vel = last_vslam_meas["zvel"]

     # === Dynamixel-based joint processing ===
     # Read angles for 4 motors: s1_angle to s4_angle
     angles = [float(last_ant_meas["s%d_angle" % (i + 1)]) for i in range(4)]

     # Normalize angles from [0, 4000] to [-1, 1]
     jpos = np.clip((np.array(angles) - 2000) / 2000.0, -1.0, 1.0)

     # Compute joint velocities
     jpos_vel = (last_frame_jpos - jpos) / joint_dt if last_frame_jpos is not None else np.zeros((4,))

     # Torso position and orientation components
     torso_pos_and_angle = np.array([
        x_vel, y_vel, z_vel,
        last_vslam_meas["z"],

        (last_frame_vslam_meas["roll"] - last_vslam_meas["roll"]) / camera_dt if last_frame_vslam_meas is not None else 0,
        (last_frame_vslam_meas["pitch"] - last_vslam_meas["pitch"]) / camera_dt if last_frame_vslam_meas is not None else 0,
        (last_frame_vslam_meas["yaw"] - last_vslam_meas["yaw"]) / camera_dt if last_frame_vslam_meas is not None else 0,

        np.sin(last_vslam_meas["roll"] / 180. * np.pi),
        np.sin(last_vslam_meas["pitch"] / 180. * np.pi),
        np.sin(last_vslam_meas["yaw"] / 180. * np.pi),
        np.cos(last_vslam_meas["roll"] / 180. * np.pi),
        np.cos(last_vslam_meas["pitch"] / 180. * np.pi),
        np.cos(last_vslam_meas["yaw"] / 180. * np.pi),
     ])

     # Final observation: 13 (torso) + 4 (jpos) + 4 (jvel) = 21
     obs = np.concatenate([torso_pos_and_angle, jpos, jpos_vel])

     # Append to past_obses (for temporal stacking)
     self.past_obses.append(obs)
     obs = np.concatenate(self.past_obses)


     # Store current frame data for next time step
     last_frame_ant_meas = last_ant_meas
     last_frame_vslam_meas = last_vslam_meas
     last_frame_jpos = jpos

     # Extra info (x, y, z of camera)
     info = np.array([last_vslam_meas["x"], last_vslam_meas["y"], last_vslam_meas["z"]])
     #print(obs)
     return obs, info



    def apply_controls(self, a):
     print("action will be applied")

     assert len(a) == 4, "Action array must have 4 values (fl, fr, bl, br)"
    # print(a)
     a = np.clip(a, -1, 1)
     
     scaled = ((a + 1) / 2) * (MAX_POS - MIN_POS) + MIN_POS
     scaled = np.clip(scaled.astype(int), MIN_POS, MAX_POS)
     #scaled = np.clip(((a + 1) * 2047.5).astype(int), 120, 3200)
     servo_ids = ["FL", "FR", "BL", "BR"]

     with dynamixel_lock:
        for i in range(4):
            dxl_id = DXL_IDS[servo_ids[i]]

            packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, scaled[i])
        print("Servos moved sucessfully to ",scaled)


    def reset_tracking(self):
        """ reset tracking state and tracking camera pose """
        global last_frame_ant_meas, last_frame_vslam_meas, last_frame_jpos
        print("Reseting now")
        last_frame_ant_meas = None
        last_frame_vslam_meas = None
        last_frame_jpos = None
        self.past_obses.clear()
        self.past_obses.extend([np.zeros(OBS_SIZE // FRAME_STACKING)] * FRAME_STACKING)
       # past_obses = deque([np.zeros(OBS_SIZE//FRAME_STACKING)]*FRAME_STACKING, maxlen=FRAME_STACKING)



    def reset_servos(self):
        apply_controls([-1,-1,-1,-1])

    def detach_servos(self):
       for dxl_id in DXL_IDS.values():
           packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
       print("servos are disabled sucessfully")

    def attach_servos(self):
       for dxl_id in DXL_IDS.values():
         packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
       print("Servos are attached, ready to fire")



def reset():
    """ reset robot joints and everything before rollout """
    env.reset_tracking()
    env.reset_servos()
    #print("reset has been called")

def detach_servos():
    """ cut torque to servos to save power """
    print("detachhing the servos")
    env.detach_servos()

def attach_servos():
    """ enable torque to servos to start actuation """
    env.attach_servos()

def get_state():
    """ get current state of joints and realsense data """
    print("Getting the current state")
    return env.get_obs()


def apply_controls(pid_setpoints):
    """apply controls to the robot"""
    env.apply_controls(pid_setpoints)

def cfompute_reward_walk(state, action, next_state):
    """ compute reward based on state changes and action applied """
    
    forward_vel = next_state[63]
    body_height= next_state[66]
    print(forward_vel,body_height*100)
    forward_reward = w1 * forward_vel
    height_penalty = w2 * torch.clamp(torch.tensor(0.06 - body_height), min=0.0)
    reward = forward_reward - (height_penalty/30) 
    return reward

def compute_reward_walk(state, action, next_state):
    """ compute reward based on state changes and action applied """

    forward_vel = next_state[63]
    reward = forward_vel

    print("Calculated Reward", reward)
    

    return reward


def rollout(agent, length=50, train=False, random=False, task='walk'):
    """ rollout policy for fixed length and collect data to buffer """
    global last_camera_meas
    print("rollout has started")
    reset()
    time.sleep(0.2) # tracking reset takes some time


    state, info = get_state()
    x, y, z = info
    x_start = x
    y_start = y
    time.sleep(0.05)
    episode_return = 0
    last_time = datetime.datetime.utcnow()
    for t in range(length):
        print("Roll out step:", t)
        now = datetime.datetime.utcnow()
        interval = (now - last_time).total_seconds()
        last_time = now

        print("rollout t", t, "time", now, "dt", interval)
        if random:
            action = np.random.uniform(-1, 1, ACT_SIZE)
            print("random mode will be executed")
        else:
            action = agent.act(state, train=train)
            print("the action from network is ",action)
            print("actor mode will be executed")

        print("controls will be applied")
        apply_controls(action)

        next_state, info = get_state()
        x, y, z = info

        # --- Termination condition ---
        if abs(x-x_start) > 0.30:
          print(f"Termination: x={x:.3f} m is out of bounds! Adding 100 reward.")
          reward = 100
          episode_return += reward
          reset()
          break  # End the episode early
        elif abs(y-y_start) > 0.15:
            print(f"Termination: y={y:.3f} m is out of bounds! Subtracting 100 reward.")
            reward = -100
            episode_return += reward
            reset()
            break  # End the episode early

        if task == 'walk':
            print("reward will be calculated")
            reward = compute_reward_walk(state, action, next_state)
        else:
            print("unknown task %s" % task)
            quit()

        not_done = True
        agent.replay_buffer.append([state, action, [reward], next_state, [not_done]])
        agent.info_buffer.append(info)
        print("Roll out step complete")
        episode_return += reward
        print("for this step immediate reward is", reward)
        state = next_state
        print("state is updated")

        # aim for a 0.05s cycle time, i.e. 20Hz, so sleep however much is still remaining
        time.sleep(max(0.05 - (datetime.datetime.utcnow() - last_time).total_seconds(), 0))

    time.sleep(0.2)
    print("resetting now ")
    reset()
    print("wait for 5 seconds")
    time.sleep(10) # allow servos to turn for 1.5 second

    print("The episode return", episode_return)
    return episode_return

if __name__ == '__main__':


    print("Main loop stared, initializition done")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    td3 = TD3(device, OBS_SIZE, ACT_SIZE)
    td3.info_buffer = []
    print("binding to pc")
    socket = ctx.socket(zmq.REP)
    socket.bind('tcp://*:5556')
    print("Binding sucessful")
    #ultra_serial = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)
    #print("ultrasonic sensor ready")
  #  ultra_serial = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.1)
    env = EnvironmentHandler()
    time.sleep(1.5)
    env.reset_servos()
    time.sleep(0.5)
    #env.detach_servos()
    #time.sleep(1)


    print("Running")

    while True:
        print("checking for actions and weights")
        (task, actor_weights) = socket.recv_pyobj()

        #task = "walk"
        #actor_weights = None
        print("Received actor weights", actor_weights, "task", task)

        # collect new data
        if actor_weights is None:
            print("No weights, defaulting to random")
            rollout(td3, random=True, task=task)
        else:
            print("weights recieved, starting")
            td3.actor.load_state_dict(actor_weights)
            rollout(td3, task=task)

        print("Replay buffer is done")
        print("Sending Data to main PC")
        print(td3.replay_buffer)
        socket.send_pyobj((td3.replay_buffer, td3.info_buffer))

        # reset replay buffer
        td3.replay_buffer.clear()
        td3.info_buffer.clear()
