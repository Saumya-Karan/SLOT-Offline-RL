import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import colorsys
from scipy.interpolate import splprep, splev
from core.rrt_3d_engine import True3DRRT, Node

# ==========================================
# TRAJECTORY 2 COORDINATES
# ==========================================
start   = (0.15, 0.0, 0.15) 
wp_fork = (0.15, 0.65, 0.15)  
goal    = (0.45, 0.70, 0.15) 
w1_alt  = (0.15, 0.70, 0.15)  # Drawn strictly for visual reference!

obstacles = [
    (-0.30, -0.22, 0.50, 0.72, 0.0, 0.16), 
    ( 0.22,  0.30, 0.50, 0.72, 0.0, 0.16), 
    (-0.30,  0.30, 0.50, 0.54, 0.16, 0.20)
]

# 0.9m x 0.8m Calibration Rectangle
real_world_pts = np.array([[-0.3, 0.8], [0.6, 0.8], [0.6, 0.0], [-0.3, 0.0]], dtype=np.float32)
visual_path_pts = [start, goal, w1_alt] # We will draw W1 just to show the viewer the alternate path!

image_pts = []
box_mask_pts = []

video_path = "raw_media/traj7_raw.mp4" 
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret: 
    print("Error: Could not read video. Check 'raw_media/traj7_raw.mp4'!")
    exit()

def click_floor(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(image_pts) < 4:
        image_pts.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if len(image_pts) > 1: cv2.line(frame, image_pts[-2], image_pts[-1], (0, 255, 255), 2)
        if len(image_pts) == 4: cv2.line(frame, image_pts[3], image_pts[0], (0, 255, 255), 2)
        cv2.imshow("1. Click Floor Rectangle", frame)

cv2.namedWindow("1. Click Floor Rectangle", cv2.WINDOW_NORMAL)
cv2.resizeWindow("1. Click Floor Rectangle", 1280, 720)
cv2.imshow("1. Click Floor Rectangle", frame)
cv2.setMouseCallback("1. Click Floor Rectangle", click_floor)

print("PHASE 1: Click 1.Top-L(Near left support) -> 2.Top-R(In right gap) -> 3.Bot-R -> 4.Bot-L")
while len(image_pts) < 4: cv2.waitKey(10)
cv2.waitKey(600)
cv2.destroyAllWindows()

def click_box(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(box_mask_pts) < 4:
        box_mask_pts.append((x, y))
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("2. Click Right Support", frame)

cv2.namedWindow("2. Click Right Support", cv2.WINDOW_NORMAL)
cv2.resizeWindow("2. Click Right Support", 1280, 720)
cv2.imshow("2. Click Right Support", frame)
cv2.setMouseCallback("2. Click Right Support", click_box)

print("PHASE 2: Click 4 corners of the RIGHT SUPPORT BOX to mask it.")
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

print("Calculating Dense 3D RRT Trees...")
full_path, all_trees = [], []
stages = [(start, wp_fork), (wp_fork, goal)]

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
tck, _ = splprep(pts, s=0.015, k=3)
smooth_x, smooth_y, smooth_z = splev(np.linspace(0, 1, 300), tck)

print("Rendering Final AR Video...")
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'AR_videos'))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'traj7_AR_output.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
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
    cv2.line(overlay, p1, p2, (0, 255, 0), 5)

overlay = cv2.bitwise_and(overlay, overlay, mask=mask)

# Node Colors: Start (Yellow), Goal (Green), W1_Ignored (Cyan)
node_colors = [(0, 255, 255), (0, 255, 0), (255, 255, 0)]

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ret, frame = cap.read()
    if not ret: break
    
    final_frame = cv2.addWeighted(frame, 1.0, overlay, 0.8, 0)
    
    for idx, pt in enumerate(visual_path_pts):
        px = project_to_pixel(pt[0], pt[1])
        cv2.circle(final_frame, px, 9, (255, 255, 255), -1)
        cv2.circle(final_frame, px, 6, node_colors[idx], -1)
        
    out.write(final_frame)

cap.release()
out.release()
print(f"\nSUCCESS! Check '{out_path}'")