import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import colorsys
from scipy.interpolate import splprep, splev
from core.rrt_3d_engine import True3DRRT, Node

# ==========================================
# TRAJECTORY 1 COORDINATES (Fixed for COM)
# ==========================================
start = (0.35, -0.05, 0.15) # Shifted to center on the robot's COM!
wp1   = (0.10, 0.30, 0.15)    # In front of bar
pre_crawl  = (0.0, 0.30, 0.08) 
post_crawl = (0.0, 0.50, 0.08) 
wp2   = (0.40, 0.75, 0.15)   # Stood up past bar
wp3   = (0.40, 0.65, 0.15)   # Top right corner
goal  = (0.40, 0.45, 0.15)   # Shifted to perfectly hit the COM at the end

obstacles = [
    (-0.275, -0.20, 0.35, 0.45, 0.0, 0.15), 
    ( 0.20,  0.275, 0.35, 0.45, 0.0, 0.15), 
    (-0.275, 0.275, 0.35, 0.45, 0.15, 0.19),
    (-0.30, -0.25, -0.15, 0.00, 0.0, 0.05),
    (-0.15, -0.10, -0.35, -0.15, 0.0, 0.05),
    ( 0.00,  0.05, -0.55, -0.35, 0.0, 0.05),
    ( 0.25,  0.30, -0.15, 0.00, 0.0, 0.05),
    ( 0.40,  0.45, -0.35, -0.15, 0.0, 0.05),
    ( 0.55,  0.60, -0.55, -0.35, 0.0, 0.05)
]

# We use a massive bounding box for the 4 Calibration Clicks
real_world_pts = np.array([[-0.30, -0.52], [-0.30, 0.75], [0.80, 0.75], [0.80, -0.52]], dtype=np.float32)

# ONLY DRAW THESE 4 DOTS ON THE VIDEO (Start, WP1, WP2, Goal)
visual_path_pts = [start, wp1, wp3, goal]
node_colors = [(0, 255, 255), (255, 255, 0), (255, 255, 0), (0, 255, 0)] # Yellow, Cyan, Cyan, Green

image_pts = []
bar_mask_pts = []

video_path = "raw_media/traj1_raw.mp4" 
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret: 
    print("Error: Could not read video. Check 'raw_media/traj1_raw.mp4'!")
    exit()

# FIX: your video is portrait, but every cv2.resizeWindow() below was
# hardcoded to a fixed 1280x720 (landscape) size, which force-stretched the
# frame to fit that box. Derive display size from the ACTUAL frame's aspect
# ratio instead, and reuse it everywhere.
frame_h, frame_w = frame.shape[:2]
DISPLAY_H = 900
DISPLAY_W = int(frame_w * (DISPLAY_H / frame_h))

def make_window(name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, DISPLAY_W, DISPLAY_H)

def click_floor(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(image_pts) < 4:
        image_pts.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if len(image_pts) > 1:
            cv2.line(frame, image_pts[-2], image_pts[-1], (0, 255, 255), 2)
        if len(image_pts) == 4:
            cv2.line(frame, image_pts[3], image_pts[0], (0, 255, 255), 2)
        cv2.imshow("1. Click Floor Rectangle", frame)

make_window("1. Click Floor Rectangle")
cv2.imshow("1. Click Floor Rectangle", frame)
cv2.setMouseCallback("1. Click Floor Rectangle", click_floor)

print("PHASE 1: Click 1.Bot-L -> 2.Top-L -> 3.Top-R -> 4.Bot-R")
while len(image_pts) < 4: cv2.waitKey(10)
cv2.waitKey(600)
cv2.destroyAllWindows()

def click_bar(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(bar_mask_pts) < 4:
        bar_mask_pts.append((x, y))
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("2. Click Bar", frame)

make_window("2. Click Bar")
cv2.imshow("2. Click Bar", frame)
cv2.setMouseCallback("2. Click Bar", click_bar)

print("PHASE 2: Click the 4 corners of the physical silver bar.")
while len(bar_mask_pts) < 4: cv2.waitKey(10)
cv2.waitKey(600)
cv2.destroyAllWindows()

mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
cv2.fillPoly(mask, [np.array(bar_mask_pts, dtype=np.int32)], 0)
H, _ = cv2.findHomography(real_world_pts, np.array(image_pts, dtype=np.float32))

def project_to_pixel(x, y, current_H):
    pt = np.array([[[x, y]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(pt, current_H)
    return int(proj[0][0][0]), int(proj[0][0][1])

print("Calculating Dense 3D RRT Trees...")
full_path, all_trees = [], []
stages = [(start, wp1), (wp1, pre_crawl), (pre_crawl, post_crawl), (post_crawl, wp2), (wp2, wp3), (wp3, goal)]

def direct_segment_free(a, b, obstacles, resolution=0.005):
    """Sweep-check a straight a->b line against obstacles at fine resolution.
    Mirrors True3DRRT.is_segment_free but usable before constructing a tree."""
    dist = np.linalg.norm(np.array(b) - np.array(a))
    if dist == 0:
        return True
    n_steps = max(1, int(np.ceil(dist / resolution)))
    for i in range(n_steps + 1):
        t = i / n_steps
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        z = a[2] + t * (b[2] - a[2])
        for (xmin, xmax, ymin, ymax, zmin, zmax) in obstacles:
            if (xmin <= x <= xmax) and (ymin <= y <= ymax) and (zmin <= z <= zmax):
                return False
    return True

for i, (s, g) in enumerate(stages):
    # FIX: if a straight line between waypoints is already collision-free,
    # use it directly instead of running RRT. RRT doesn't return the shortest
    # path - it returns whatever branch first randomly gets close enough to
    # the goal, which on short open-space legs (like wp2->wp3->goal here)
    # produces visible wandering/loops once spline-fit. Only invoke RRT when
    # a real obstacle actually blocks the direct line.
    if direct_segment_free(s, g, obstacles):
        path = [s, g]
        tree = [Node(*s), Node(*g)]
        tree[1].parent = tree[0]
    else:
        rrt = True3DRRT(s, g, obstacles)
        path, tree = rrt.plan()
    if path:
        if i > 0: path = path[1:]
        full_path.extend(path)
        all_trees.append(tree)

pts = np.array([[p[0] for p in full_path], [p[1] for p in full_path], [p[2] for p in full_path]])
_, idx = np.unique(pts, axis=1, return_index=True)
pts = pts[:, np.sort(idx)]
# FIX: raised smoothing factor (0.005 -> 0.02) so the AR overlay traces the
# intended trajectory shape rather than hugging every jittery raw sample
# point (which is what visually produced the loop/squiggle).
tck, _ = splprep(pts, s=0.02, k=min(2, pts.shape[1]-1))
smooth_x, smooth_y, smooth_z = splev(np.linspace(0, 1, 300), tck)

print("\n=======================================================")
print("PHASE 3: RENDERING VIDEO")
print("--> IMPORTANT: Watch the video rendering on your screen.")
print("--> If the camera angle suddenly cuts/shifts, PRESS THE SPACEBAR!")
print("=======================================================\n")

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'AR_videos'))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'traj1_AR_output.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

make_window("Rendering...")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# FIX: replace manual spacebar re-click with automatic per-frame homography
# tracking. Instead of assuming the camera is static and only re-clicking at
# hard cuts, we track the 4 floor corners every frame with Lucas-Kanade
# optical flow and recompute H continuously. This fixes slow pans/tilts too,
# not just sudden cuts, and needs zero manual intervention during playback.
lk_params = dict(winSize=(31, 31), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

tracked_corners = np.array(image_pts, dtype=np.float32).reshape(-1, 1, 2)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# FIX: detect HARD CUTS separately from continuous shake. A tilt/shake changes
# a frame gradually (optical flow handles that fine); a hard cut causes a huge
# sudden jump in overall pixel difference. Only pause for manual re-click when
# that jump is detected - shake never triggers it.
CUT_DIFF_THRESHOLD = 40.0  # mean abs pixel diff (0-255 scale); tune if needed

def click_recalibrate(frame_img):
    """Blocking re-click of the 4 floor corners, used only on detected hard cuts."""
    pts = []
    def _cb(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(frame_img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("HARD CUT DETECTED: Re-click 4 Floor Points", frame_img)
    make_window("HARD CUT DETECTED: Re-click 4 Floor Points")
    cv2.imshow("HARD CUT DETECTED: Re-click 4 Floor Points", frame_img)
    cv2.setMouseCallback("HARD CUT DETECTED: Re-click 4 Floor Points", _cb)
    print("\n[HARD CUT] Click 1.Bot-L -> 2.Top-L -> 3.Top-R -> 4.Bot-R for the new angle!")
    while len(pts) < 4: cv2.waitKey(10)
    cv2.waitKey(400)
    cv2.destroyWindow("HARD CUT DETECTED: Re-click 4 Floor Points")
    print("Resuming render...\n")
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Hard-cut check BEFORE trying to track - tracking across a real cut is
    # meaningless (the "previous" floor position has no relation to the new
    # shot), so we must catch this first and re-click, not feed it to LK flow.
    mean_diff = np.mean(cv2.absdiff(gray, prev_gray))
    if mean_diff > CUT_DIFF_THRESHOLD:
        tracked_corners = click_recalibrate(frame.copy())
        H, _ = cv2.findHomography(real_world_pts, tracked_corners.reshape(-1, 2))
        prev_gray = gray
        overlay = np.zeros_like(frame)
    else:
        # Continuous shake/tilt: track the 4 floor corners with optical flow.
        new_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, tracked_corners, None, **lk_params)
        if new_corners is not None and status.sum() == 4:
            tracked_corners = new_corners
            H, _ = cv2.findHomography(real_world_pts, tracked_corners.reshape(-1, 2))
        else:
            print("WARNING: lost track of a floor corner this frame, reusing last homography.")
        prev_gray = gray

    overlay = np.zeros_like(frame)

    # Draw Trees
    for tree in all_trees:
        for i, node in enumerate(tree):
            if node.parent:
                p1 = project_to_pixel(node.x, node.y, H)
                p2 = project_to_pixel(node.parent.x, node.parent.y, H)
                rgb = colorsys.hsv_to_rgb(0.6 - (i/len(tree))*0.6, 1.0, 1.0)
                color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
                cv2.line(overlay, p1, p2, color, 1)

    # Draw Path
    for i in range(len(smooth_x)-1):
        p1 = project_to_pixel(smooth_x[i], smooth_y[i], H)
        p2 = project_to_pixel(smooth_x[i+1], smooth_y[i+1], H)
        cv2.line(overlay, p1, p2, (0, 255, 0), 4)

    overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
    final_frame = cv2.addWeighted(frame, 1.0, overlay, 0.8, 0)
    
    # Draw ONLY the 4 main dots
    for idx, pt in enumerate(visual_path_pts):
        px = project_to_pixel(pt[0], pt[1], H)
        cv2.circle(final_frame, px, 9, (255, 255, 255), -1)
        cv2.circle(final_frame, px, 6, node_colors[idx], -1)
        
    out.write(final_frame)
    cv2.imshow("Rendering...", final_frame)
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nSUCCESS! Check '{out_path}'")