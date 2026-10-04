import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splprep, splev
from Global_Planner_3D_RRT.core.rrt_3d_engine import True3DRRT, Node

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
# TRAJECTORY 5 COORDINATES (Straight Crawl -> Right Strafe)
# ==========================================
start = (0.0, 0.0, 0.15) 
pre_crawl  = (0.0, 0.25, 0.08)  # Squat before bar
post_crawl = (0.0, 0.55, 0.08)  # Clear the bar
wp1   = (0.0, 0.65, 0.15)       # Stand up at W1
goal  = (0.35, 0.65, 0.15)      # Strafe RIGHT into the pocket

obstacles = [
    # 1. Left Support
    (-0.275, -0.20, 0.35, 0.45, 0.0, 0.15), 
    # 2. Right Support
    ( 0.20,  0.275, 0.35, 0.45, 0.0, 0.15), 
    # 3. The Bar
    (-0.275, 0.275, 0.35, 0.45, 0.15, 0.19),
    
    # ENVIRONMENT WALLS
    (-0.35, -0.275, -0.10, 0.80, 0.0, 0.40), # Left White Board
    (-0.35, 0.60, 0.80, 0.90, 0.0, 0.50),    # Back Black Board
    (0.40, 0.60, -0.10, 0.50, 0.0, 0.40)     # Right Monitor Box (Stops at Y=0.50 so robot can strafe behind it!)
]

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

stages = [(start, pre_crawl), (pre_crawl, post_crawl), (post_crawl, wp1), (wp1, goal)]
full_path = []

print("Running Trajectory 5 RRT Engine...")
for i, (s, g) in enumerate(stages):
    rrt = True3DRRT(s, g, obstacles)
    path, tree = rrt.plan()
    if path:
        if i > 0: path = path[1:]
        full_path.extend(path)
        for node in tree:
            if node.parent:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y], [node.z, node.parent.z], 
                        color='dodgerblue', alpha=0.4, linewidth=1.0) 

# Tight B-Spline Smoothing (k=2, s=0.005) for Holonomic Corners
if full_path:
    pts = np.array([[p[0] for p in full_path], [p[1] for p in full_path], [p[2] for p in full_path]])
    _, idx = np.unique(pts, axis=1, return_index=True)
    pts = pts[:, np.sort(idx)]
    tck, _ = splprep(pts, s=0.005, k=2)
    smooth_x, smooth_y, smooth_z = splev(np.linspace(0, 1, 300), tck)
    ax.plot(smooth_x, smooth_y, smooth_z, color='black', linewidth=4.5, label="Smoothed Trajectory")

draw_obstacle(ax, obstacles[0], color='silver', alpha=1.0)
draw_obstacle(ax, obstacles[1], color='silver', alpha=1.0)
draw_obstacle(ax, obstacles[2], color='dimgray', alpha=1.0)
draw_obstacle(ax, obstacles[3], color='ghostwhite', alpha=0.4) 
draw_obstacle(ax, obstacles[4], color='#222222', alpha=0.8)    
draw_obstacle(ax, obstacles[5], color='wheat', alpha=0.4)      

ax.scatter(*start, color='gold', s=200, label='START', zorder=5)
ax.scatter(*wp1, color='deepskyblue', s=200, label='W1', zorder=5)
ax.scatter(*goal, color='limegreen', s=200, label='GOAL', zorder=5)

ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)'); ax.set_zlabel('Z (meters)')
ax.set_xlim(-0.3, 0.6); ax.set_ylim(-0.1, 0.9); ax.set_zlim(0.0, 0.4) 
ax.set_title("Trajectory 5: Crawl -> Right Strafe")
ax.legend()
plt.show()