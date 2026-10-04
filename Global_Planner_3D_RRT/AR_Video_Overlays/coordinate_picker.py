import cv2
import numpy as np
import os

# ==========================================
# COORDINATE PICKER
# Click 4 floor corners (same order as your main script: Top-L -> Top-R -> Bot-R -> Bot-L),
# then click anywhere else (W1, W2, obstacle corners, etc.) to print its real-world (x,y) in meters.
# ==========================================

# EDIT THIS to match your main script exactly:
real_world_pts = np.array([[-0.4, 1.4], [0.6, 1.4], [0.6, -0.1], [-0.4, -0.1]], dtype=np.float32)
image_path = "raw_media/traj1_raw.png"  # EDIT if needed

img = cv2.imread(os.path.abspath(image_path))
if img is None:
    print(f"Error: could not read {image_path}")
    exit()

img_h, img_w = img.shape[:2]
display_h = 800
display_w = int(img_w * (display_h / img_h))
clone = img.copy()

cv2.namedWindow("Calibrate then Click", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Calibrate then Click", display_w, display_h)

image_pts = []
H_inv = None

def to_world(px, py):
    pt = np.array([[[px, py]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H_inv)
    return world[0][0][0], world[0][0][1]

def click(event, x, y, flags, params):
    global H_inv
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(image_pts) < 4:
        image_pts.append((x, y))
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibrate then Click", clone)
        if len(image_pts) == 4:
            H, _ = cv2.findHomography(real_world_pts, np.array(image_pts, dtype=np.float32))
            H_inv = np.linalg.inv(H)
            print("\nCalibration done. Now click any point to get its real-world (x, y) in meters.\n")
    else:
        wx, wy = to_world(x, y)
        print(f"Pixel ({x}, {y})  ->  world (x={wx:.3f}, y={wy:.3f})")
        cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Calibrate then Click", clone)

cv2.setMouseCallback("Calibrate then Click", click)
cv2.imshow("Calibrate then Click", clone)

print("PHASE 1: Click 4 floor corners in order Top-L -> Top-R -> Bot-R -> Bot-L")
print("(must be the SAME 4 points/order you use in your main script)")
cv2.waitKey(0)
cv2.destroyAllWindows()