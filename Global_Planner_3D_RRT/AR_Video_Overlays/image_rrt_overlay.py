import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import colorsys
from scipy.interpolate import splprep, splev
from core.rrt_3d_engine import True3DRRT, Node

# ==========================================
# 1. EXACT INCH-TO-CM COORDINATES
# ==========================================
# Start = 0m | Bar = 0.30m | Box = 0.86m | Goal = 1.39m
start = (-0.10, 0.15, 0.15) 
pre_crawl  = (-0.10, 0.45, 0.08)  # Squat before bar
post_crawl = (0.0, 0.55, 0.08)  # Clear the bar
obs_detect = (0.0, 0.65, 0.15)  # Sees orange box at 0.86m
dodge_right= (0.25, 0.95, 0.15) # Dodges RIGHT (towards monitors)
fwd_clear  = (0.25, 1.15, 0.15) # Walks forward past the box
return_left= (0.0, 1.15, 0.15)  # Snaps back to path
goal = (-0.10, 1.29, 0.15)        # Goal at the Black Wall

# The RRT only knows about the Bar!
rrt_obstacles = [
    (-0.25, -0.15, 0.25, 0.35, 0.0, 0.15), # Left Support
    (0.15, 0.25, 0.25, 0.35, 0.0, 0.15),   # Right Support
    (-0.25, 0.25, 0.25, 0.35, 0.15, 0.19)  # The Bar
]

# We define a massive 1.0m x 1.6m calibration rectangle for the floor
real_world_pts = np.array([[-0.5, 1.4], [0.5, 1.4], [0.5, -0.2], [-0.5, -0.2]], dtype=np.float32)

# ==========================================
# 2. LOAD IMAGE & DYNAMIC UI (No Stretching!)
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(
    script_dir,
    "..",
    "raw_media",
    "traj6_raw.jpeg"
)

image_path = os.path.abspath(image_path)
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image at {image_path}!")
    exit()

img_h, img_w = img.shape[:2]
display_h = 800
display_w = int(img_w * (display_h / img_h))
clone = img.copy()

def create_window(name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, display_w, display_h)

# --- A. CLICK THE FLOOR ---
image_pts = []
def click_floor(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(image_pts) < 4:
        image_pts.append((x, y))
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        if len(image_pts) > 1: cv2.line(clone, image_pts[-2], image_pts[-1], (0, 255, 255), 2)
        if len(image_pts) == 4: cv2.line(clone, image_pts[3], image_pts[0], (0, 255, 255), 2)
        cv2.imshow("1. Click Floor Rectangle", clone)

create_window("1. Click Floor Rectangle")
cv2.imshow("1. Click Floor Rectangle", clone)
cv2.setMouseCallback("1. Click Floor Rectangle", click_floor)

print("PHASE 1: Click 4 Floor Corners (1.Top-L -> 2.Top-R -> 3.Bot-R -> 4.Bot-L)")
while len(image_pts) < 4: cv2.waitKey(10)
cv2.waitKey(500)
cv2.destroyAllWindows()

# --- B. CLICK THE WHITE PIPE (Mask 1) ---
bar_mask_pts = []
def click_bar(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(bar_mask_pts) < 4:
        bar_mask_pts.append((x, y))
        cv2.circle(clone, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("2. Click White Pipe", clone)

create_window("2. Click White Pipe")
cv2.imshow("2. Click White Pipe", clone)
cv2.setMouseCallback("2. Click White Pipe", click_bar)

print("PHASE 2: Click 4 corners of the WHITE PIPE to mask it.")
while len(bar_mask_pts) < 4: cv2.waitKey(10)
cv2.waitKey(500)
cv2.destroyAllWindows()

# --- C. CLICK THE ORANGE BOX (Mask 2) ---
obs_mask_pts = []
def click_obs(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(obs_mask_pts) < 4:
        obs_mask_pts.append((x, y))
        cv2.circle(clone, (x, y), 5, (0, 165, 255), -1)
        cv2.imshow("3. Click Orange Box", clone)

create_window("3. Click Orange Box")
cv2.imshow("3. Click Orange Box", clone)
cv2.setMouseCallback("3. Click Orange Box", click_obs)

print("PHASE 3: Click 4 corners of the ORANGE BOX to mask it.")
while len(obs_mask_pts) < 4: cv2.waitKey(10)
cv2.waitKey(500)
cv2.destroyAllWindows()

# Create combined mask
mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
cv2.fillPoly(mask, [np.array(bar_mask_pts, dtype=np.int32)], 0)
cv2.fillPoly(mask, [np.array(obs_mask_pts, dtype=np.int32)], 0)

# Calculate Homography
H, _ = cv2.findHomography(real_world_pts, np.array(image_pts, dtype=np.float32))
def project_to_pixel(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(pt, H)
    return int(proj[0][0][0]), int(proj[0][0][1])

# ==========================================
# 3. CALCULATE PATHS & RRT TREES
# ==========================================
print("Calculating Dense 3D RRT Trees and Paths...")

# Original Nominal Plan — drawn as a pure straight-line reference (start -> goal),
# not the noisy RRT-sampled path. The RRT tree is still computed above (all_trees)
# so the branch visualization is unaffected; only the reference line itself is straightened.
rrt_stages = [(start, pre_crawl), (pre_crawl, post_crawl), (post_crawl, goal)]
all_trees = []
for s, g in rrt_stages:
    rrt = True3DRRT(s, g, rrt_obstacles)
    _, tree = rrt.plan()
    all_trees.append(tree)

sm_orig_x = np.linspace(start[0], goal[0], 300)
sm_orig_y = np.linspace(start[1], goal[1], 300)

# Emergent RL Plan (Dodging RIGHT)
emergent_path = [start, pre_crawl, post_crawl, obs_detect, dodge_right, fwd_clear, return_left, goal]
em_pts = np.array([[p[0] for p in emergent_path], [p[1] for p in emergent_path], [p[2] for p in emergent_path]])
tck_em, _ = splprep(em_pts, s=0.02, k=3) # Smooth sweeping curves for dodging
sm_em_x, sm_em_y, _ = splev(np.linspace(0, 1, 300), tck_em)

# ==========================================
# 4. RENDER FINAL IMAGE
# ==========================================
overlay = np.zeros_like(img)

# DRAW RRT TREE BRANCHES!
for tree in all_trees:
    for i, node in enumerate(tree):
        if node.parent:
            p1 = project_to_pixel(node.x, node.y)
            p2 = project_to_pixel(node.parent.x, node.parent.y)
            rgb = colorsys.hsv_to_rgb(0.6 - (i/len(tree))*0.6, 1.0, 1.0) # Blue to Red gradient
            color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
            cv2.line(overlay, p1, p2, color, 2 )

# Draw Pink Straight Line (Original Nominal Plan)
for i in range(len(sm_orig_x) - 1):
    p1 = project_to_pixel(sm_orig_x[i], sm_orig_y[i])
    p2 = project_to_pixel(sm_orig_x[i+1], sm_orig_y[i+1])
    # Hot Pink (recommended)
    cv2.line(overlay, p1, p2, (255, 0, 255), 10)

# Draw Thick Green Line (Emergent Dodging Path)
for i in range(len(sm_em_x)-1):
    p1 = project_to_pixel(sm_em_x[i], sm_em_y[i])
    p2 = project_to_pixel(sm_em_x[i+1], sm_em_y[i+1])
    cv2.line(overlay, p1, p2, (0, 255, 0), 6) 

# Mask out the physical obstacles
overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
final_img = cv2.addWeighted(img, 1.0, overlay, 0.8, 0)

# Draw Start (Yellow) and Goal (Green) Dots ON TOP
cv2.circle(final_img, project_to_pixel(start[0], start[1]), 12, (255, 255, 255), -1)
cv2.circle(final_img, project_to_pixel(start[0], start[1]), 8, (0, 255, 255), -1) 

cv2.circle(final_img, project_to_pixel(goal[0], goal[1]), 12, (255, 255, 255), -1)
cv2.circle(final_img, project_to_pixel(goal[0], goal[1]), 8, (0, 255, 0), -1) 

# SAVE IMAGE
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "traj6_AR_output.jpg")

cv2.imwrite(out_path, final_img)
print(f"\nSUCCESS! High-Res Image saved to '{out_path}'")

create_window("Final Annotated Figure")
cv2.imshow("Final Annotated Figure", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()