import sys
import os
# Tell Python to look in the parent folder (SLOT_V3) for modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import numpy as np
import colorsys
from scipy.interpolate import splprep, splev
from Global_Planner_3D_RRT.core.rrt_3d_engine import True3DRRT, Node

# ==========================================
# 1. 3D RRT PLANNER MATH
# ==========================================

# ==========================================
# 2. COORDINATES & OBSTACLES
# ==========================================
start = (0.0, 0.0, 0.15) 
pre_crawl  = (0.0, 0.30, 0.08) 
post_crawl = (0.0, 0.50, 0.08) 
wp1   = (0.0, 0.75, 0.15) 
wp2   = (0.6, 0.75, 0.15) 
goal  = (0.6, 0.1, 0.15)  
obstacles = [
    (-0.275, -0.20, 0.35, 0.45, 0.0, 0.15),  # Left Support
    ( 0.20,  0.275, 0.35, 0.45, 0.0, 0.15),  # Right Support
    (-0.275, 0.275, 0.35, 0.45, 0.15, 0.19)  # The Bar
]

# ==========================================
# 3. VIDEO UI: HOMOGRAPHY & MASKING
# ==========================================
real_world_pts = np.array([[0.0, 0.0], [0.0, 0.75], [0.6, 0.75], [0.6, 0.1]], dtype=np.float32)
image_pts = []
bar_mask_pts = []

# LOAD VIDEO
video_path = "raw_media/traj2_raw.mp4" 
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret: exit()

def click_floor(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(image_pts) < 4:
        image_pts.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("1. Click 4 Floor Points", frame)

cv2.imshow("1. Click 4 Floor Points", frame)
cv2.setMouseCallback("1. Click 4 Floor Points", click_floor)
print("PHASE 1: Click Start, WP1, WP2, Goal on the floor.")
while len(image_pts) < 4: cv2.waitKey(10)
cv2.waitKey(400)
cv2.destroyAllWindows()

def click_bar(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(bar_mask_pts) < 4:
        bar_mask_pts.append((x, y))
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("2. Click 4 Corners of the Bar", frame)

cv2.imshow("2. Click 4 Corners of the Bar", frame)
cv2.setMouseCallback("2. Click 4 Corners of the Bar", click_bar)
print("PHASE 2: Click the 4 corners of the physical silver bar to mask it.")
while len(bar_mask_pts) < 4: cv2.waitKey(10)
cv2.waitKey(400)
cv2.destroyAllWindows()

mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
cv2.fillPoly(mask, [np.array(bar_mask_pts, dtype=np.int32)], 0)

H, _ = cv2.findHomography(real_world_pts, np.array(image_pts, dtype=np.float32))

def project_to_pixel(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(pt, H)
    return int(proj[0][0][0]), int(proj[0][0][1])

# ==========================================
# 4. RUN RRT & RENDER VIDEO
# ==========================================
print("Calculating Dense 3D RRT Trees...")
full_path, all_trees = [], []
stages = [(start, pre_crawl), (pre_crawl, post_crawl), (post_crawl, wp1), (wp1, wp2), (wp2, goal)]

for i, (s, g) in enumerate(stages):
    rrt = True3DRRT(s, g, obstacles)
    path, tree = rrt.plan()
    if path:
        if i > 0: path = path[1:]
        full_path.extend(path)
        all_trees.append(tree)

print("Applying B-Spline Smoothing...")
x_vals, y_vals, z_vals = [p[0] for p in full_path], [p[1] for p in full_path], [p[2] for p in full_path]
pts = np.array([x_vals, y_vals, z_vals])
_, idx = np.unique(pts, axis=1, return_index=True)
pts = pts[:, np.sort(idx)]
tck, _ = splprep(pts, s=0.005, k=1)
smooth_x, smooth_y, smooth_z = splev(np.linspace(0, 1, 300), tck)

print("Rendering Final AR Video...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('outputs/AR_videos/traj2_AR_output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

overlay = np.zeros_like(frame)

# Draw Dense Branches
for tree in all_trees:
    for i, node in enumerate(tree):
        if node.parent:
            p1 = project_to_pixel(node.x, node.y)
            p2 = project_to_pixel(node.parent.x, node.parent.y)
            rgb = colorsys.hsv_to_rgb(0.6 - (i/len(tree))*0.6, 1.0, 1.0)
            color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
            cv2.line(overlay, p1, p2, color, 1)

# Draw Path
for i in range(len(smooth_x)-1):
    p1 = project_to_pixel(smooth_x[i], smooth_y[i])
    p2 = project_to_pixel(smooth_x[i+1], smooth_y[i+1])
    cv2.line(overlay, p1, p2, (0, 255, 0), 4)

# Apply Bar Mask (Erases the lines where the bar is)
overlay = cv2.bitwise_and(overlay, overlay, mask=mask)

# Node Colors (BGR format for OpenCV)
node_colors = [
    (0, 255, 255),  # Start: Yellow
    (255, 255, 0),  # WP1: Cyan
    (255, 255, 0),  # WP2: Cyan
    (0, 255, 0)     # Goal: Green
]

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Blend the overlay onto the video
    final_frame = cv2.addWeighted(frame, 1.0, overlay, 0.8, 0)
    
    # Draw the Waypoint Dots ON TOP of everything
    for idx, pt in enumerate(real_world_pts):
        px = project_to_pixel(pt[0], pt[1])
        cv2.circle(final_frame, px, 9, (255, 255, 255), -1) # White outline
        cv2.circle(final_frame, px, 6, node_colors[idx], -1) # Colored center
        
    out.write(final_frame)

cap.release()
out.release()
print("\nSUCCESS! Check 'output/AR_videos/traj2_AR_output.mp4'")