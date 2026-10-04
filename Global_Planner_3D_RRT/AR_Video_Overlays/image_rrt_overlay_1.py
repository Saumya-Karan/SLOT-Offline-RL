import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import colorsys
from scipy.interpolate import splprep, splev
from core.rrt_3d_engine import True3DRRT, Node

# ==========================================
# 1. COORDINATES - derived from your diagram
# ==========================================
# ASSUMPTIONS (verify/edit these against the real measured setup):
#  - origin at START, x = lateral, y = forward (toward goal), z = height
#  - diagonal bar: 60cm long, 30 degrees off vertical, placed just past start
#  - crawl-bar row: 40cm gap between pylons (matches your earlier 8cm-pylon
#    pattern), positioned ~0.75m past the diagonal bar
#  - W1/W2 fork: 55cm apart, goal centered between them

start = (0.213, 0.297, 0.15)
diag_clear = (0.05, 0.45, 0.15)      # between start and the bar crossing
pre_crawl  = (-0.025, 0.55, 0.08)    # squat before the cubical bar
post_crawl = (-0.025, 0.70, 0.08)    # clear the bar (~matches measured mid-crawl point 0.634)
w1_marker  = (-0.106, 1.053, 0.15)   # measured via coordinate_picker
w2_marker  = (0.093, 1.303, 0.15)    # measured via coordinate_picker
goal = (0.292, 1.093, 0.15)          # measured via coordinate_picker
apex = ((w1_marker[0] + w2_marker[0]) / 2,
        max(w1_marker[1], w2_marker[1]),
        0.15)
apex2 = (w2_marker[0] + 0.15,
         (w2_marker[1]),
         0.15)

# The 30 degree diagonal bar (60cm) - APPROXIMATED as an axis-aligned
# bounding box per your choice, since the RRT engine only does axis-aligned
# boxes. dx = 0.60*sin(30) = 0.30m, dy = 0.60*cos(30) = 0.52m, plus a small
# thickness margin (5cm) since the real bar has physical width.
obstacles = [
    (-0.20, 0.40, 0.10, 0.35, 0.0, 0.05),      # Diagonal bar (bounding-box approx)
    (-0.15, -0.05, 0.60, 0.70, 0.0, 0.15),     # Left crawl-bar support pylon
    ( 0.05,  0.15, 0.60, 0.70, 0.0, 0.15),     # Right crawl-bar support pylon
    (-0.15,  0.15, 0.60, 0.64, 0.15, 0.19),    # The cubical bar (4cm thick)
]

# Calibration rectangle - same as used in coordinate_picker.py, kept
# consistent so the derived W1/W2/goal values above stay valid.
real_world_pts = np.array([[-0.4, 1.4], [0.6, 1.4], [0.6, -0.1], [-0.4, -0.1]], dtype=np.float32)

# ==========================================
# 2. LOAD IMAGE & DYNAMIC UI (No Stretching!)
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.abspath(os.path.join(script_dir, "..", "raw_media", "traj1_raw.png"))

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

# --- B. CLICK THE CRAWL BAR (Mask 1) ---
bar_mask_pts = []
def click_bar(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(bar_mask_pts) < 4:
        bar_mask_pts.append((x, y))
        cv2.circle(clone, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("2. Click Crawl Bar", clone)

create_window("2. Click Crawl Bar")
cv2.imshow("2. Click Crawl Bar", clone)
cv2.setMouseCallback("2. Click Crawl Bar", click_bar)

print("PHASE 2: Click 4 corners of the CRAWL BAR to mask it.")
while len(bar_mask_pts) < 4: cv2.waitKey(10)
cv2.waitKey(500)
cv2.destroyAllWindows()

# --- C. CLICK THE DIAGONAL BAR (Mask 2) ---
diag_mask_pts = []
def click_diag(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(diag_mask_pts) < 4:
        diag_mask_pts.append((x, y))
        cv2.circle(clone, (x, y), 5, (0, 165, 255), -1)
        cv2.imshow("3. Click Diagonal Bar", clone)

create_window("3. Click Diagonal Bar")
cv2.imshow("3. Click Diagonal Bar", clone)
cv2.setMouseCallback("3. Click Diagonal Bar", click_diag)

print("PHASE 3: Click 4 corners of the DIAGONAL BAR to mask it.")
while len(diag_mask_pts) < 4: cv2.waitKey(10)
cv2.waitKey(500)
cv2.destroyAllWindows()

# Create combined mask
mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
cv2.fillPoly(mask, [np.array(bar_mask_pts, dtype=np.int32)], 0)
cv2.fillPoly(mask, [np.array(diag_mask_pts, dtype=np.int32)], 0)

# Calculate Homography
H, _ = cv2.findHomography(real_world_pts, np.array(image_pts, dtype=np.float32))
def project_to_pixel(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(pt, H)
    return int(proj[0][0][0]), int(proj[0][0][1])

# ==========================================
# 3. CALCULATE PATH & RRT TREES
# ==========================================
print("Calculating Dense 3D RRT Trees and Path...")

stages = [(start, diag_clear), (diag_clear, pre_crawl), (pre_crawl, post_crawl),
          (post_crawl, w1_marker), (w1_marker, apex), (apex, w2_marker),
          (w2_marker, apex2), (apex2, goal)]
full_path, all_trees = [], []
for s, g in stages:
    rrt = True3DRRT(s, g, obstacles)
    p, tree = rrt.plan()
    if p:
        full_path.extend(p)
        all_trees.append(tree)

pts = np.array([[p[0] for p in full_path], [p[1] for p in full_path], [p[2] for p in full_path]])
_, idx = np.unique(pts, axis=1, return_index=True)
pts = pts[:, np.sort(idx)]
tck, _ = splprep(pts, s=0.06, k=3)
sm_x, sm_y, _ = splev(np.linspace(0, 1, 300), tck)

# ==========================================
# 4. RENDER FINAL IMAGE (Green path + RRT branches only, no red line)
# ==========================================
overlay = np.zeros_like(img)

# DRAW RRT TREE BRANCHES
for tree in all_trees:
    for i, node in enumerate(tree):
        if node.parent:
            p1 = project_to_pixel(node.x, node.y)
            p2 = project_to_pixel(node.parent.x, node.parent.y)
            rgb = colorsys.hsv_to_rgb(0.6 - (i / len(tree)) * 0.6, 1.0, 1.0)
            color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            cv2.line(overlay, p1, p2, color, 1)

# Thick Green Line (the actual trajectory)
for i in range(len(sm_x) - 1):
    p1 = project_to_pixel(sm_x[i], sm_y[i])
    p2 = project_to_pixel(sm_x[i + 1], sm_y[i + 1])
    cv2.line(overlay, p1, p2, (0, 255, 0), 6)

# Mask out the physical obstacles
overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
final_img = cv2.addWeighted(img, 1.0, overlay, 0.8, 0)

# Draw Start (Yellow) and Goal (Green) Dots ON TOP
cv2.circle(final_img, project_to_pixel(start[0], start[1]), 12, (255, 255, 255), -1)
cv2.circle(final_img, project_to_pixel(start[0], start[1]), 8, (0, 255, 255), -1)

cv2.circle(final_img, project_to_pixel(goal[0], goal[1]), 12, (255, 255, 255), -1)
cv2.circle(final_img, project_to_pixel(goal[0], goal[1]), 8, (0, 255, 0), -1)

# W1 / W2 reference markers (blue, matching your diagram)
cv2.circle(final_img, project_to_pixel(w1_marker[0], w1_marker[1]), 10, (255, 255, 255), -1)
cv2.circle(final_img, project_to_pixel(w1_marker[0], w1_marker[1]), 7, (255, 165, 0), -1)
cv2.circle(final_img, project_to_pixel(w2_marker[0], w2_marker[1]), 10, (255, 255, 255), -1)
cv2.circle(final_img, project_to_pixel(w2_marker[0], w2_marker[1]), 7, (255, 165, 0), -1)

# SAVE IMAGE
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "traj1_AR_output.jpg")

cv2.imwrite(out_path, final_img)
print(f"\nSUCCESS! High-Res Image saved to '{out_path}'")

create_window("Final Annotated Figure")
cv2.imshow("Final Annotated Figure", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()