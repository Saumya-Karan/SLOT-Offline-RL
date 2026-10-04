import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splprep, splev
from core.rrt_3d_engine import True3DRRT, Node

def draw_obstacle(ax, obs, color, alpha=0.9):
    xmin, xmax, ymin, ymax, zmin, zmax = obs
    vertices = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]
    ])
    faces = [[vertices[0], vertices[1], vertices[2], vertices[3]], [vertices[4], vertices[5], vertices[6], vertices[7]], 
             [vertices[0], vertices[1], vertices[5], vertices[4]], [vertices[2], vertices[3], vertices[7], vertices[6]], 
             [vertices[1], vertices[2], vertices[6], vertices[5]], [vertices[4], vertices[7], vertices[3], vertices[0]]]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1.0, edgecolors='black', alpha=alpha))

# ==========================================
# TRAJECTORY 1 COORDINATES (Omni -> U-Shape)
# ==========================================
# Math derived from your 60cm length at a 30-degree angle
start = (0.30, -0.52, 0.15) 
wp0   = (0.0, 0.0, 0.15)      # Re-aligns in front of the bar
pre_crawl  = (0.0, 0.30, 0.08) 
post_crawl = (0.0, 0.50, 0.08) 
wp1   = (0.0, 0.75, 0.15)     # Stands up after bar
wp2   = (0.7, 0.75, 0.15)     # Turns right behind monitor box
goal  = (0.7, 0.10, 0.15)     # Walks backward to goal

obstacles = [
    # 1. The Archway
    (-0.275, -0.20, 0.35, 0.45, 0.0, 0.15),  # Left Support
    ( 0.20,  0.275, 0.35, 0.45, 0.0, 0.15),  # Right Support
    (-0.275, 0.275, 0.35, 0.45, 0.15, 0.19), # The Bar
    
    # 2. Diagonal Floor Rails (AABB Staircase matching 30-deg angle)
    # Left Rail
    (-0.30, -0.25, -0.15, 0.00, 0.0, 0.05),
    (-0.15, -0.10, -0.35, -0.15, 0.0, 0.05),
    ( 0.00,  0.05, -0.55, -0.35, 0.0, 0.05),
    # Right Rail
    ( 0.25,  0.30, -0.15, 0.00, 0.0, 0.05),
    ( 0.40,  0.45, -0.35, -0.15, 0.0, 0.05),
    ( 0.55,  0.60, -0.55, -0.35, 0.0, 0.05),
    
    # 3. Surrounding Walls
    (-0.35, -0.30, -0.60, 0.85, 0.0, 0.40), # Left White Board
    (-0.35, 0.90, 0.85, 0.95, 0.0, 0.50),   # Front Black Board
    (0.275, 0.50, 0.10, 0.60, 0.0, 0.40)    # Center Monitor Box
]

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Sequence explicitly forces Omni -> Straight -> Turn -> Turn
stages = [(start, wp0), (wp0, pre_crawl), (pre_crawl, post_crawl), (post_crawl, wp1), (wp1, wp2), (wp2, goal)]
full_path = []

print("Running Trajectory 1 RRT Engine...")
for i, (s, g) in enumerate(stages):
    rrt = True3DRRT(s, g, obstacles)
    path, tree = rrt.plan()
    if path:
        if i > 0: path = path[1:]
        full_path.extend(path)
        for node in tree:
            if node.parent:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y], [node.z, node.parent.z], 
                        color='dodgerblue', alpha=0.3, linewidth=1.0) 

# Tight B-Spline Smoothing (k=2, s=0.005) for Holonomic Corners
if full_path:
    pts = np.array([[p[0] for p in full_path], [p[1] for p in full_path], [p[2] for p in full_path]])
    _, idx = np.unique(pts, axis=1, return_index=True)
    pts = pts[:, np.sort(idx)]
    tck, _ = splprep(pts, s=0.005, k=2)
    smooth_x, smooth_y, smooth_z = splev(np.linspace(0, 1, 300), tck)
    ax.plot(smooth_x, smooth_y, smooth_z, color='black', linewidth=4.5, label="Smoothed Trajectory")

# Draw Environment
for i in range(3): draw_obstacle(ax, obstacles[i], color='dimgray', alpha=0.9)
for i in range(3, 9): draw_obstacle(ax, obstacles[i], color='silver', alpha=0.9) # Floor rails
draw_obstacle(ax, obstacles[9], color='ghostwhite', alpha=0.4) 
draw_obstacle(ax, obstacles[10], color='#222222', alpha=0.8)    
draw_obstacle(ax, obstacles[11], color='wheat', alpha=0.4)      

# Draw Nodes
ax.scatter(*start, color='gold', s=200, label='START', zorder=5)
ax.scatter(*wp0, color='deepskyblue', s=200, label='W0', zorder=5)
ax.scatter(*wp1, color='deepskyblue', s=200, label='W1', zorder=5)
ax.scatter(*wp2, color='deepskyblue', s=200, label='W2', zorder=5)
ax.scatter(*goal, color='limegreen', s=200, label='GOAL', zorder=5)

ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)'); ax.set_zlabel('Z (meters)')
ax.set_xlim(-0.4, 0.8); ax.set_ylim(-0.6, 0.9); ax.set_zlim(0.0, 0.4) 
ax.set_title("Trajectory 1: Omni Corridor to U-Shape Crawl")
ax.legend()

# EXPORT JSON MAP
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
os.makedirs(out_dir, exist_ok=True)
json_path = os.path.join(out_dir, "traj1_map.json")

full_map_data = {
    "start": start, "goal": goal,
    "obstacles": [{"obstacle_id": i, "bounds": {"xmin": o[0], "xmax": o[1], "ymin": o[2], "ymax": o[3], "zmin": o[4], "zmax": o[5]}} for i, o in enumerate(obstacles)],
    "planned_path": [[round(x, 3), round(y, 3), round(z, 3)] for x, y, z in zip(smooth_x, smooth_y, smooth_z)]
}
with open(json_path, "w") as f: json.dump(full_map_data, f, indent=4)
print(f"SUCCESS! Map exported to '{json_path}'!")
plt.show()