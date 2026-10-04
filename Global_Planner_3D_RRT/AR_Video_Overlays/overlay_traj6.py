import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

import cv2
import numpy as np
from scipy.interpolate import splprep, splev

# ==========================================
# TRAJECTORY 0 COORDINATES
# ==========================================
start = (0.0, 0.0, 0.15) 
goal  = (0.80, 0.0, 0.15) 

emergent_path = [
    (0.0, 0.0, 0.15),
    (0.20, 0.0, 0.15),
    (0.20, 0.30, 0.15),  # Left Dodge
    (0.55, 0.30, 0.15), 
    (0.55, 0.0, 0.15),   # Right Return
    (0.80, 0.0, 0.15)    
]

# 0.8m x 0.8m calibration square for OpenCV
real_world_pts = np.array([[0.8, 0.4], [0.8, -0.4], [0.0, -0.4], [0.0, 0.4]], dtype=np.float32)
image_pts = []
box_mask_pts = []

video_path = "raw_media/traj6_raw.mp4" 
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret: 
    print("Error: Could not read video. Check 'raw_media/traj6_raw.mp4'!")
    exit()

def click_floor(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(image_pts) < 4:
        image_pts.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if len(image_pts) > 1:
            cv2.line(frame, image_pts[-2], image_pts[-1], (0, 255, 255), 2)
        if len(image_pts) == 4:
            cv2.line(frame, image_pts[3], image_pts[0], (0, 255, 255), 2)
        cv2.imshow("Click Calibration Square", frame)

cv2.namedWindow("Click Calibration Square", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Click Calibration Square", 1280, 720)
cv2.imshow("Click Calibration Square", frame)
cv2.setMouseCallback("Click Calibration Square", click_floor)

print("PHASE 1: Click 1.Top-L -> 2.Top-R -> 3.Bot-R -> 4.Bot-L")
while len(image_pts) < 4: cv2.waitKey(10)
cv2.waitKey(600)
cv2.destroyAllWindows()

def click_box(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(box_mask_pts) < 4:
        box_mask_pts.append((x, y))
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("Click Orange Box", frame)

cv2.namedWindow("Click Orange Box", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Click Orange Box", 1280, 720)
cv2.imshow("Click Orange Box", frame)
cv2.setMouseCallback("Click Orange Box", click_box)

print("PHASE 2: Click the 4 corners of the orange box to mask it.")
while len(box_mask_pts) < 4: cv2.waitKey(10)
cv2.waitKey(600)
cv2.destroyAllWindows()

mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
cv2.fillPoly(mask, [np.array(box_mask_pts, dtype=np.int32)], 0)

H, _ = cv2.findHomography(real_world_pts, np.array(image_pts, dtype=np.float32))

def project_to_pixel(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(pt, H)
    return int(proj[0][0][0]), int(proj[0][0][1])

print("Processing Emergent Path...")
em_x, em_y, em_z = [p[0] for p in emergent_path], [p[1] for p in emergent_path], [p[2] for p in emergent_path]
pts_em = np.array([em_x, em_y, em_z])
tck_em, _ = splprep(pts_em, s=0.015, k=3)
sm_em_x, sm_em_y, sm_em_z = splev(np.linspace(0, 1, 300), tck_em)

print("Rendering Final AR Video...")
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'AR_videos'))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'traj6_AR_output.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
overlay = np.zeros_like(frame)

# Draw the ORIGINAL RRT PATH (Yellow dashed/dotted line)
orig_p1 = project_to_pixel(0.0, 0.0)
orig_p2 = project_to_pixel(0.80, 0.0)
cv2.line(overlay, orig_p1, orig_p2, (0, 255, 255), 2)

# Draw the EMERGENT PATH (Thick Green Line)
for i in range(len(sm_em_x)-1):
    p1 = project_to_pixel(sm_em_x[i], sm_em_y[i])
    p2 = project_to_pixel(sm_em_x[i+1], sm_em_y[i+1])
    cv2.line(overlay, p1, p2, (0, 255, 0), 4)

overlay = cv2.bitwise_and(overlay, overlay, mask=mask)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ret, frame = cap.read()
    if not ret: break
    
    final_frame = cv2.addWeighted(frame, 1.0, overlay, 0.8, 0)
    
    # Draw Start and Goal Dots
    px_s = project_to_pixel(start[0], start[1])
    px_g = project_to_pixel(goal[0], goal[1])
    cv2.circle(final_frame, px_s, 9, (255, 255, 255), -1); cv2.circle(final_frame, px_s, 6, (0, 255, 255), -1)
    cv2.circle(final_frame, px_g, 9, (255, 255, 255), -1); cv2.circle(final_frame, px_g, 6, (0, 255, 0), -1)
        
    out.write(final_frame)

cap.release()
out.release()
print(f"\nSUCCESS! Check {out_path}")