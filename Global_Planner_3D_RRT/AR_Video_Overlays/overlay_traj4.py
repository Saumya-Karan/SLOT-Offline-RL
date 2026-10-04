import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import numpy as np
import colorsys
from scipy.interpolate import splprep, splev
from Global_Planner_3D_RRT.core.rrt_3d_engine import True3DRRT, Node

# ==========================================
# TRAJECTORY 4 COORDINATES
# ==========================================
start = (0.0, 0.0, 0.15) 
pre_crawl  = (0.12, 0.12, 0.08) 
post_crawl = (0.35, 0.35, 0.08) 
wp1   = (0.45, 0.50, 0.15)      
wp2   = (0.80, 0.50, 0.15)      
goal  = (0.80, 0.10, 0.15)   
turn_anchor = (
    wp2[0] + (goal[0] - wp2[0]) * 0.15,
    wp2[1] + (goal[1] - wp2[1]) * 0.15,
    wp2[2]
)   

obstacles = [
    (0.05, 0.15, 0.35, 0.45, 0.0, 0.15), 
    (0.35, 0.45, 0.05, 0.15, 0.0, 0.15), 
    (0.05, 0.20, 0.30, 0.45, 0.15, 0.19),
    (0.15, 0.30, 0.20, 0.35, 0.15, 0.19),
    (0.25, 0.40, 0.10, 0.25, 0.15, 0.19),
    (0.35, 0.45, 0.05, 0.15, 0.15, 0.19)
]

# These 4 points naturally form a convex trapezoid, perfect for OpenCV!
real_world_pts = np.array([[0.0, 0.0], [0.45, 0.50], [0.80, 0.50], [0.80, 0.10]], dtype=np.float32)
visual_path_pts = [start, wp1, wp2, goal]

image_pts = []
bar_mask_pts = []

video_path = "raw_media/traj4_raw.mp4" 
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret: 
    print("Error: Could not read video. Check 'raw_media/traj4_raw.mp4'!")
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

print("PHASE 1: Click 1.Start -> 2.W1 -> 3.W2 -> 4.Goal")
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

print("PHASE 2: Click the 4 corners of the 45-degree physical silver bar.")
while len(bar_mask_pts) < 4: cv2.waitKey(10)
cv2.waitKey(600)
cv2.destroyAllWindows()

mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
cv2.fillPoly(mask, [np.array(bar_mask_pts, dtype=np.int32)], 0)
H, _ = cv2.findHomography(real_world_pts, np.array(image_pts, dtype=np.float32))

def project_to_pixel(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(pt, H)
    return int(proj[0][0][0]), int(proj[0][0][1])


# ==========================================
# PATH SHORTCUTTING (removes RRT loops/detours)
# ==========================================
def shortcut_path(path, obstacles, resolution=0.01):
    """
    Removes unnecessary loops/detours from a raw RRT path by greedily
    connecting the farthest reachable point with a straight, collision-free
    segment, reusing the same collision logic as the RRT engine itself.
    """
    checker = True3DRRT(path[0], path[-1], obstacles,
                         collision_check_resolution=resolution)

    pruned = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            a = Node(*path[i])
            b = Node(*path[j])
            if checker.is_segment_free(a, b):
                break
            j -= 1
        pruned.append(path[j])
        i = j
    return pruned


print("Calculating Dense 3D RRT Trees...")
full_path, all_trees = [], []
stages = [(start, pre_crawl), (pre_crawl, post_crawl), (post_crawl, wp1),
          (wp1, wp2), (wp2, turn_anchor), (turn_anchor, goal)]

for i, (s, g) in enumerate(stages):
    rrt = True3DRRT(s, g, obstacles)
    path, tree = rrt.plan()
    if path:
        if i > 0: path = path[1:]
        full_path.extend(path)
        all_trees.append(tree)

print("Shortcutting raw RRT path to remove loops/detours...")
full_path = shortcut_path(full_path, obstacles)
print(f"Shortcut path length: {len(full_path)} points")

print("Applying B-Spline Smoothing...")
pts = np.array([[p[0] for p in full_path], [p[1] for p in full_path], [p[2] for p in full_path]])
_, idx = np.unique(pts, axis=1, return_index=True)
pts = pts[:, np.sort(idx)]
print(f"Unique points for spline fit: {pts.shape[1]}")

# splprep needs m > k points (m = number of points, k = spline degree).
# Fall back to a lower-degree spline if shortcutting left too few points.
if pts.shape[1] < 2:
    raise RuntimeError(
        f"Shortcutting collapsed the path to {pts.shape[1]} unique point(s); "
        "start and goal are mutually visible in a straight line with no "
        "intermediate structure to smooth. Check your stages/obstacles, or "
        "skip the spline and draw a straight line directly."
    )
k_deg = 3
if pts.shape[1] <= k_deg:
    k_deg = pts.shape[1] - 1
    print(f"Not enough points for cubic spline, falling back to k={k_deg}")

tck, _ = splprep(pts, s=0.03, k=k_deg)
smooth_x, smooth_y, smooth_z = splev(np.linspace(0, 1, 300), tck)

print("Rendering Final AR Video...")
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'AR_videos'))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'traj4_AR_output.mp4')

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
    cv2.line(overlay, p1, p2, (0, 255, 0), 4)

overlay = cv2.bitwise_and(overlay, overlay, mask=mask)

node_colors = [(0, 255, 255), (255, 255, 0), (255, 255, 0), (0, 255, 0)]

# Snap each waypoint marker onto the actual rendered spline curve, rather
# than its raw input coordinate, so the dots visually sit on the green line
# even after shortcutting/smoothing has shifted the path slightly.
spline_pts_xy = np.stack([smooth_x, smooth_y], axis=1)  # shape (300, 2)
snapped_visual_pts = []
for pt in visual_path_pts:
    target = np.array([pt[0], pt[1]])
    dists = np.linalg.norm(spline_pts_xy - target, axis=1)
    nearest_idx = np.argmin(dists)
    snapped_visual_pts.append((smooth_x[nearest_idx], smooth_y[nearest_idx]))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ret, frame = cap.read()
    if not ret: break
    
    final_frame = cv2.addWeighted(frame, 1.0, overlay, 0.8, 0)
    
    for idx, pt in enumerate(snapped_visual_pts):
        px = project_to_pixel(pt[0], pt[1])
        cv2.circle(final_frame, px, 9, (255, 255, 255), -1)
        cv2.circle(final_frame, px, 6, node_colors[idx], -1)
        
    out.write(final_frame)

cap.release()
out.release()
print(f"\nSUCCESS! Check {out_path}")