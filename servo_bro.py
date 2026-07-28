import sys
import time
import argparse
from dynamixel_sdk import *

# ----------------- DYNAMIXEL SETTINGS -----------------
ADDR_TORQUE_ENABLE = 64
ADDR_TORQUE_DISABLE = 0
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_LOAD = 126
ADDR_PRESENT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146
ADDR_HARDWARE_ERROR_STATUS = 70

PROTOCOL_VERSION = 2.0
#DXL_IDS = {"FL": 5, "FR": 4, "BL": 3, "BR": 2} old
DXL_IDS = {"FL": 4, "FR": 2, "BL": 3, "BR": 5}
BAUDRATE = 2000000
DEVICENAME = '/dev/ttyUSB0'

# ----------------- ARGUMENT PARSING -----------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "mode",
    choices=["off", "enable", "debug", "status", "reset", "test"],
    help="Operation mode: off, enable, debug, status, reset, test"
)
parser.add_argument(
    "--angle",
    type=int,
    help="Target angle (120 to 4000) for test mode"
)
args = parser.parse_args()

# ----------------- DYNAMIXEL INIT -----------------
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print("Error: Failed to open the port!", file=sys.stderr)
    sys.exit(1)
if not portHandler.setBaudRate(BAUDRATE):
    print("Error: Failed to set the baudrate!", file=sys.stderr)
    sys.exit(1)

# ----------------- FUNCTION DEFINITIONS -----------------
def turn_off_torque():
    print("Turning off torque for all servos...")
    for name, dxl_id in DXL_IDS.items():
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Failed to disable torque for {name} (ID {dxl_id}): {packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Error disabling torque for {name} (ID {dxl_id}): {packetHandler.getRxPacketError(dxl_error)}")
        else:
            print(f"Torque disabled for {name} (ID {dxl_id})")

def enable_torque():
    print("Enabling torque for all servos...")
    for name, dxl_id in DXL_IDS.items():
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Failed to enable torque for {name} (ID {dxl_id}): {packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Error enabling torque for {name} (ID {dxl_id}): {packetHandler.getRxPacketError(dxl_error)}")
        else:
            print(f"Torque enabled for {name} (ID {dxl_id})")

def debug_hardware_errors():
    print("Checking hardware errors for all servos...")
    error_types = [
        ("Input Voltage Error", 0),
        ("Overheating Error", 1),
        ("Motor Encoder Error", 2),
        ("Electrical Shock Error", 3),
        ("Overload Error", 4),
    ]
    for name, dxl_id in DXL_IDS.items():
        error_result, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, dxl_id, ADDR_HARDWARE_ERROR_STATUS)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Failed to read error for {name} (ID {dxl_id}): {packetHandler.getTxRxResult(dxl_comm_result)}")
            continue
        if dxl_error != 0:
            print(f"Error reading error status for {name} (ID {dxl_id}): {packetHandler.getRxPacketError(dxl_error)}")
            continue
        if error_result == 0:
            print(f"{name} (ID {dxl_id}): No hardware errors.")
        else:
            print(f"{name} (ID {dxl_id}):")
            for err_name, bit in error_types:
                if error_result & (1 << bit):
                    print(f"  - {err_name}")

def print_status():
    print("Servo status (position, load, voltage, temperature):")
    for name, dxl_id in DXL_IDS.items():
        # Torque status
        torque, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            torque_str = "N/A"
        elif dxl_error != 0:
            torque_str = "Error"
        else:
            torque_str = "ENABLED" if torque else "DISABLED"
        pos, _, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_POSITION)
        load, _, _ = packetHandler.read2ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_LOAD)
        voltage, _, _ = packetHandler.read1ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_VOLTAGE)
        temp, _, _ = packetHandler.read1ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_TEMPERATURE)
        print(f"{name} (ID {dxl_id}):")
        print(f"  Torque: {torque_str}")
        print(f"  Position: {pos}")
        print(f"  Load: {load}")
        print(f"  Voltage: {voltage / 10.0:.1f} V")
        print(f"  Temperature: {temp} °C")

def reset_servos():
    print("Moving all servos to position 120...")
    for name, dxl_id in DXL_IDS.items():
        # Enable torque (in case it's off)
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            print(f"Failed to enable torque for {name} (ID {dxl_id})")
            continue
        # Move to position 120
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, 120)
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            print(f"Failed to move {name} (ID {dxl_id}) to 120")
        else:
            print(f"{name} (ID {dxl_id}) moved to 120")
    print("All servos moved to 120.")

def test_move_servos(angle):
    if not (120 <= angle <= 4000):
        print("Angle must be between 120 and 4000!")
        return
    print(f"Enabling torque for all servos and moving to position {angle}...")
    for name, dxl_id in DXL_IDS.items():
        # Enable torque
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            print(f"Failed to enable torque for {name} (ID {dxl_id})")
            continue
        # Move to angle
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, angle)
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            print(f"Failed to move {name} (ID {dxl_id}) to {angle}")
        else:
            print(f"{name} (ID {dxl_id}) moved to {angle}")
    print("Test move complete.")

# ----------------- MAIN LOGIC -----------------
try:
    if args.mode == "off":
        turn_off_torque()
    elif args.mode == "enable":
        enable_torque()
    elif args.mode == "debug":
        debug_hardware_errors()
    elif args.mode == "status":
        print_status()
    elif args.mode == "reset":
        reset_servos()
    elif args.mode == "test":
        if args.angle is None:
            print("You must specify --angle for test mode (120 to 4000).")
        else:
            test_move_servos(args.angle)
finally:
    portHandler.closePort()
