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
# TRAJECTORY 2: THE "FORK IN THE ROAD"
# ==========================================
start   = (0.0, 0.0, 0.15) 
wp_fork = (0.0, 0.35, 0.15)  # The decision point!
goal    = (0.45, 0.85, 0.15) # The chosen Right Door
w1_alt  = (0.0, 0.85, 0.15)  # The ignored Left Door (W1 in your diagram)

# Exactly mapped from your Canva blueprint
obstacles = [
    # 1. Left Support (8cm wide, 22cm deep, 16cm high)
    (-0.30, -0.22, 0.50, 0.72, 0.0, 0.16), 
    # 2. Right Support
    ( 0.22,  0.30, 0.50, 0.72, 0.0, 0.16), 
    # 3. The Bar (44cm gap, 4cm thick, resting on top)
    (-0.30,  0.30, 0.50, 0.54, 0.16, 0.20),
    
    # ENVIRONMENT WALLS
    (-0.50, -0.30, 0.00, 1.00, 0.0, 0.40), # Left White Board
    ( 0.65,  1.00, 0.40, 1.00, 0.0, 0.50), # Right Monitor Box
    (-0.50,  0.65, 0.95, 1.00, 0.0, 0.50)  # Back Black Board
]

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Sequence: Start -> Approach the Fork -> Take the Right Door (Goal)
stages = [(start, wp_fork), (wp_fork, goal)]
full_path = []

print("Running Trajectory 2 RRT Engine...")
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

# B-Spline Smoothing
if full_path:
    pts = np.array([[p[0] for p in full_path], [p[1] for p in full_path], [p[2] for p in full_path]])
    _, idx = np.unique(pts, axis=1, return_index=True)
    pts = pts[:, np.sort(idx)]
    tck, _ = splprep(pts, s=0.015, k=3)
    smooth_x, smooth_y, smooth_z = splev(np.linspace(0, 1, 300), tck)
    ax.plot(smooth_x, smooth_y, smooth_z, color='black', linewidth=4.5, label="Smoothed Trajectory")

# Draw Environment
draw_obstacle(ax, obstacles[0], color='silver', alpha=1.0)
draw_obstacle(ax, obstacles[1], color='silver', alpha=1.0)
draw_obstacle(ax, obstacles[2], color='dimgray', alpha=1.0)
draw_obstacle(ax, obstacles[3], color='ghostwhite', alpha=0.3) 
draw_obstacle(ax, obstacles[4], color='wheat', alpha=0.3)      
draw_obstacle(ax, obstacles[5], color='#222222', alpha=0.8)    

# Draw Nodes
ax.scatter(*start, color='gold', s=200, label='START', zorder=5)
ax.scatter(*goal, color='limegreen', s=200, label='GOAL (Right Door)', zorder=5)
ax.scatter(*w1_alt, color='deepskyblue', s=200, label='W1 (Ignored Left Door)', zorder=5)

ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)'); ax.set_zlabel('Z (meters)')
ax.set_xlim(-0.4, 0.8); ax.set_ylim(-0.1, 1.0); ax.set_zlim(0.0, 0.4) 
ax.set_title("Trajectory 2: Resolving Perceptual Symmetry (The Fork in the Road)")
ax.legend()

# EXPORT JSON MAP
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
os.makedirs(out_dir, exist_ok=True)
json_path = os.path.join(out_dir, "traj2_map.json")

full_map_data = {
    "start": start, "goal": goal,
    "obstacles": [{"obstacle_id": i, "bounds": {"xmin": o[0], "xmax": o[1], "ymin": o[2], "ymax": o[3], "zmin": o[4], "zmax": o[5]}} for i, o in enumerate(obstacles)],
    "planned_path": [[round(x, 3), round(y, 3), round(z, 3)] for x, y, z in zip(smooth_x, smooth_y, smooth_z)]
}
with open(json_path, "w") as f: json.dump(full_map_data, f, indent=4)
print(f"SUCCESS! Map exported to '{json_path}'!")
plt.show()