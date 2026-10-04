import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import numpy as np
import colorsys
from scipy.interpolate import splprep, splev
from Global_Planner_3D_RRT.core.rrt_3d_engine import True3DRRT, Node

# ==========================================
# EXACT TRAJECTORY 3 SEQUENCE
# ==========================================
start = (0.0, 0.0, 0.15) 
wp1   = (0.0, 0.42, 0.15) 
wp2   = (0.45, 0.42, 0.08) # W2: Squat down here!
post_crawl = (0.75, 0.42, 0.08) 
goal  = (0.80, 0.42, 0.15) 

obstacles = [
    (0.60, 0.675, 0.0, 0.22, 0.0, 0.15), 
    (0.60, 0.675, 0.62, 0.85, 0.0, 0.15), 
    (0.60, 0.675, 0.0, 0.85, 0.15, 0.19)
]

# We STILL use the 4 corners of the room just for the OpenCV Math to work
calibration_pts = np.array([[0.0, 0.0], [0.0, 0.42], [0.80, 0.42], [0.80, 0.0]], dtype=np.float32)

# But these are the 4 dots we will actually draw on the video!
visual_path_pts = [start, wp1, wp2, goal]

image_pts = []
bar_mask_pts = []

video_path = "raw_media/traj3_raw.mp4" 
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret: 
    print("Error: Could not read video. Check 'raw_media/traj3_raw.mp4'!")
    exit()

def click_floor(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(image_pts) < 4:
        image_pts.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if len(image_pts) > 1:
            cv2.line(frame, image_pts[-2], image_pts[-1], (0, 255, 255), 2)
        if len(image_pts) == 4:
            cv2.line(frame, image_pts[3], image_pts[0], (0, 255, 255), 2)
        cv2.imshow("1. Click Floor Rectangle", frame)

cv2.namedWindow("1. Click Floor Rectangle", cv2.WINDOW_NORMAL)
cv2.resizeWindow("1. Click Floor Rectangle", 1280, 720)
cv2.imshow("1. Click Floor Rectangle", frame)
cv2.setMouseCallback("1. Click Floor Rectangle", click_floor)

print("PHASE 1: Form a rectangle to calibrate the camera!")
print("Click 1.Start(Bottom-L) -> 2.WP1(Top-L) -> 3.Goal(Top-R) -> 4.Empty floor(Bottom-R)")
while len(image_pts) < 4: cv2.waitKey(10)
cv2.waitKey(600)
cv2.destroyAllWindows()

def click_bar(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(bar_mask_pts) < 4:
        bar_mask_pts.append((x, y))
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("2. Click Bar", frame)

cv2.namedWindow("2. Click Bar", cv2.WINDOW_NORMAL)
cv2.resizeWindow("2. Click Bar", 1280, 720)
cv2.imshow("2. Click Bar", frame)
cv2.setMouseCallback("2. Click Bar", click_bar)

print("PHASE 2: Click the 4 corners of the physical silver bar.")
while len(bar_mask_pts) < 4: cv2.waitKey(10)
cv2.waitKey(600)
cv2.destroyAllWindows()

mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
cv2.fillPoly(mask, [np.array(bar_mask_pts, dtype=np.int32)], 0)
H, _ = cv2.findHomography(calibration_pts, np.array(image_pts, dtype=np.float32))

def project_to_pixel(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(pt, H)
    return int(proj[0][0][0]), int(proj[0][0][1])

print("Calculating Dense 3D RRT Trees...")
full_path, all_trees = [], []
stages = [(start, wp1), (wp1, wp2), (wp2, post_crawl), (post_crawl, goal)]

for i, (s, g) in enumerate(stages):
    rrt = True3DRRT(s, g, obstacles)
    path, tree = rrt.plan()
    if path:
        if i > 0: path = path[1:]
        full_path.extend(path)
        all_trees.append(tree)

print("Applying B-Spline Smoothing...")
pts = np.array([[p[0] for p in full_path], [p[1] for p in full_path], [p[2] for p in full_path]])
_, idx = np.unique(pts, axis=1, return_index=True)
pts = pts[:, np.sort(idx)]
tck, _ = splprep(pts, s=0.001, k=2)
smooth_x, smooth_y, smooth_z = splev(np.linspace(0, 1, 300), tck)

print("Rendering Final AR Video...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('outputs/AR_videos/traj3_AR_output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
overlay = np.zeros_like(frame)

for tree in all_trees:
    for i, node in enumerate(tree):
        if node.parent:
            p1 = project_to_pixel(node.x, node.y)
            p2 = project_to_pixel(node.parent.x, node.parent.y)
            rgb = colorsys.hsv_to_rgb(0.6 - (i/len(tree))*0.6, 1.0, 1.0)
            color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
            cv2.line(overlay, p1, p2, color, 1)

for i in range(len(smooth_x)-1):
    p1 = project_to_pixel(smooth_x[i], smooth_y[i])
    p2 = project_to_pixel(smooth_x[i+1], smooth_y[i+1])
    cv2.line(overlay, p1, p2, (0, 255, 0), 4)

overlay = cv2.bitwise_and(overlay, overlay, mask=mask)

# Node Colors: Start(Yellow), W1(Cyan), W2(Cyan), Goal(Green)
node_colors = [(0, 255, 255), (255, 255, 0), (255, 255, 0), (0, 255, 0)]

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ret, frame = cap.read()
    if not ret: break
    
    final_frame = cv2.addWeighted(frame, 1.0, overlay, 0.8, 0)
    
    # Draw Start, WP1, WP2, Goal!
    for idx, pt in enumerate(visual_path_pts):
        px = project_to_pixel(pt[0], pt[1])
        cv2.circle(final_frame, px, 9, (255, 255, 255), -1)
        cv2.circle(final_frame, px, 6, node_colors[idx], -1)
        
    out.write(final_frame)

cap.release()
out.release()
print("\nSUCCESS! Check 'outputs/AR_videos/traj3_AR_output.mp4'")