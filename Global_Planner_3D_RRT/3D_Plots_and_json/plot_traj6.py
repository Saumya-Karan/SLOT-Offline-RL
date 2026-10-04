import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splprep, splev
from core.rrt_3d_engine import True3DRRT, Node

def draw_obstacle(ax, obs, color, alpha=0.9, hatch=None):
    xmin, xmax, ymin, ymax, zmin, zmax = obs
    vertices = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]
    ])
    faces = [[vertices[0], vertices[1], vertices[2], vertices[3]], [vertices[4], vertices[5], vertices[6], vertices[7]], 
             [vertices[0], vertices[1], vertices[5], vertices[4]], [vertices[2], vertices[3], vertices[7], vertices[6]], 
             [vertices[1], vertices[2], vertices[6], vertices[5]], [vertices[4], vertices[7], vertices[3], vertices[0]]]
    poly = Poly3DCollection(faces, facecolors=color, linewidths=1.0, edgecolors='black', alpha=alpha)
    if hatch:
        poly.set_hatch(hatch)
    ax.add_collection3d(poly)

# ==========================================
# TRAJECTORY 0: DYNAMIC OBSTACLE AVOIDANCE
# ==========================================
start = (0.0, 0.0, 0.15) 
goal  = (0.80, 0.0, 0.15) 

# 1. RRT knows nothing (Empty Room)
rrt_obstacles = []

# 2. EXACT ORANGE BOX (26cm wide, 10cm deep, 10cm high)
dynamic_obstacle = (0.35, 0.45, -0.13, 0.13, 0.0, 0.10) 

# 3. The Emergent Path (Left Dodge!)
emergent_path = [
    (0.0, 0.0, 0.15),
    (0.20, 0.0, 0.15),   # Stops before the box
    (0.20, 0.30, 0.15),  # Strafes LEFT to dodge
    (0.55, 0.30, 0.15),  # Walks FORWARD past the box
    (0.55, 0.0, 0.15),   # Strafes RIGHT back to the path
    (0.80, 0.0, 0.15)    # Reaches Goal
]

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

print("Calculating Original RRT Plan...")
rrt = True3DRRT(start, goal, rrt_obstacles)
rrt_path, tree = rrt.plan()

if rrt_path:
    pts = np.array([[p[0] for p in rrt_path], [p[1] for p in rrt_path], [p[2] for p in rrt_path]])
    _, idx = np.unique(pts, axis=1, return_index=True)
    pts = pts[:, np.sort(idx)]
    tck, _ = splprep(pts, s=0.005, k=2)
    smooth_x, smooth_y, smooth_z = splev(np.linspace(0, 1, 300), tck)
    
    # Draw Original Plan (Faint dashed line)
    ax.plot(smooth_x, smooth_y, smooth_z, color='black', linewidth=2, linestyle='--', label="Original RRT Plan")

# Draw the exact Orange Box
draw_obstacle(ax, dynamic_obstacle, color='darkorange', alpha=0.8, hatch='//')

# Draw the Emergent RL Path (Bright Green)
em_x, em_y, em_z = [p[0] for p in emergent_path], [p[1] for p in emergent_path], [p[2] for p in emergent_path]
pts_em = np.array([em_x, em_y, em_z])
tck_em, _ = splprep(pts_em, s=0.005, k=3)
sm_em_x, sm_em_y, sm_em_z = splev(np.linspace(0, 1, 300), tck_em)

ax.plot(sm_em_x, sm_em_y, sm_em_z, color='limegreen', linewidth=4.5, label="Emergent RL Execution")

ax.scatter(*start, color='gold', s=200, label='START', zorder=5)
ax.scatter(*goal, color='limegreen', s=200, label='GOAL', zorder=5)

ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)'); ax.set_zlabel('Z (meters)')
ax.set_xlim(-0.1, 0.9); ax.set_ylim(-0.4, 0.4); ax.set_zlim(0.0, 0.3) 
ax.set_title("Trajectory 6: Dynamic Obstacle Avoidance")
ax.legend()

plt.show()