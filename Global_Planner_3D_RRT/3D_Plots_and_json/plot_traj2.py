import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splprep, splev

# Tell Python to look in the parent folder to find the 'core' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.rrt_3d_engine import True3DRRT

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
# COORDINATES (Exactly matching your setup)
# ==========================================
start = (0.0, 0.0, 0.15) 
pre_crawl  = (0.0, 0.30, 0.08) # Drops height cleanly BEFORE the bar
post_crawl = (0.0, 0.50, 0.08) # Stays low until PAST the bar
wp1   = (0.0, 0.75, 0.15) # Stands up at W1
wp2   = (0.6, 0.75, 0.15) # Turns to W2
goal  = (0.6, 0.1, 0.15)  # Heads to Goal

# ONLY THE ARCHWAY REMAINS (No artificial walls)
obstacles = [
    (-0.275, -0.20, 0.35, 0.45, 0.0, 0.15),  # Left Support
    ( 0.20,  0.275, 0.35, 0.45, 0.0, 0.15),  # Right Support
    (-0.275, 0.275, 0.35, 0.45, 0.15, 0.19)  # The Bar
]

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Sequence of stages
stages = [(start, pre_crawl), (pre_crawl, post_crawl), (post_crawl, wp1), (wp1, wp2), (wp2, goal)]
full_path = []

print("Generating Flawless 3D Trajectory...")

for i, (s, g) in enumerate(stages):
    rrt = True3DRRT(s, g, obstacles)
    path, tree = rrt.plan()
    
    if path:
        if i > 0: path = path[1:] # Prevent duplicate points
        full_path.extend(path)
        
        # DRAW THE EXPLORATION BRANCHES
        for node in tree:
            if node.parent:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y], [node.z, node.parent.z], 
                        color='dodgerblue', alpha=0.3, linewidth=1.0) 

# TWEAK 2: TIGHTENED B-SPLINE SMOOTHING
if full_path:
    x_vals, y_vals, z_vals = [p[0] for p in full_path], [p[1] for p in full_path], [p[2] for p in full_path]
    pts = np.array([x_vals, y_vals, z_vals])
    _, idx = np.unique(pts, axis=1, return_index=True)
    pts = pts[:, np.sort(idx)]
    
    # s=0.005 keeps the trajectory glued closely to the intended path without wild swoops
    tck, _ = splprep(pts, s=0.005, k=3)
    smooth_x, smooth_y, smooth_z = splev(np.linspace(0, 1, 300), tck)
    
    # Plot final path
    ax.plot(smooth_x, smooth_y, smooth_z, color='black', linewidth=4.5, label="Optimal Smoothed Trajectory")

# Draw only the Archway Obstacles
draw_obstacle(ax, obstacles[0], color='silver', alpha=1.0)
draw_obstacle(ax, obstacles[1], color='silver', alpha=1.0)
draw_obstacle(ax, obstacles[2], color='dimgray', alpha=1.0)

# Draw the main Nodes
ax.scatter(*start, color='gold', s=200, label='START', zorder=5)
ax.scatter(*wp1, color='deepskyblue', s=200, label='W1', zorder=5)
ax.scatter(*wp2, color='deepskyblue', s=200, label='W2', zorder=5)
ax.scatter(*goal, color='limegreen', s=200, label='GOAL', zorder=5)

ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')
ax.set_xlim(-0.4, 0.8)
ax.set_ylim(-0.1, 0.9)
ax.set_zlim(0.0, 0.3) 
ax.set_title("Trajectory 1: 3D RRT Planner")
ax.legend()

# ==========================================
# EXPORT THE MAP FOR THE JETSON (STEP 3)
# ==========================================
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
os.makedirs(out_dir, exist_ok=True)
json_path = os.path.join(out_dir, "full_3d_map.json")

full_map_data = {
    "start": start,
    "goal": goal,
    "obstacles": [
        {"obstacle_id": i, "bounds": {"xmin": o[0], "xmax": o[1], "ymin": o[2], "ymax": o[3], "zmin": o[4], "zmax": o[5]}} 
        for i, o in enumerate(obstacles)
    ],
    "planned_path": [[round(x, 3), round(y, 3), round(z, 3)] for x, y, z in zip(smooth_x, smooth_y, smooth_z)]
}

with open(json_path, "w") as f:
    json.dump(full_map_data, f, indent=4)

print(f"\nSUCCESS! 3D Map and Trajectory exported to '{json_path}'!")

# Show the paper graph
plt.show()