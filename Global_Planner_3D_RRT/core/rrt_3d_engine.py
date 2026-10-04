import numpy as np

class Node:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.parent = None


class True3DRRT:
    def __init__(self, start, goal, obstacles, step_size=0.02, max_iter=50000,
                 collision_check_resolution=0.01, goal_bias=0.10, goal_tolerance=0.02,
                 bounds_margin=0.15, x_bounds=None, y_bounds=None, z_bounds=None):
        self.start, self.goal = Node(*start), Node(*goal)
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        self.collision_check_resolution = collision_check_resolution
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance
        self.tree = [self.start]

        if x_bounds is None:
            xs = [start[0], goal[0]] + [o[0] for o in obstacles] + [o[1] for o in obstacles]
            x_bounds = (min(xs) - bounds_margin, max(xs) + bounds_margin)
        if y_bounds is None:
            ys = [start[1], goal[1]] + [o[2] for o in obstacles] + [o[3] for o in obstacles]
            y_bounds = (min(ys) - bounds_margin, max(ys) + bounds_margin)
        if z_bounds is None:
            zs = [start[2], goal[2]] + [o[4] for o in obstacles] + [o[5] for o in obstacles]
            z_bounds = (max(0.0, min(zs) - bounds_margin), max(zs) + bounds_margin)
        self.x_bounds, self.y_bounds, self.z_bounds = x_bounds, y_bounds, z_bounds

        if not self.is_point_free(self.start.x, self.start.y, self.start.z):
            raise ValueError(f"START point {start} is inside an obstacle or below floor.")
        if not self.is_point_free(self.goal.x, self.goal.y, self.goal.z):
            raise ValueError(f"GOAL point {goal} is inside an obstacle or below floor.")

    def is_point_free(self, x, y, z):
        if z < 0:
            return False
        for (xmin, xmax, ymin, ymax, zmin, zmax) in self.obstacles:
            if (xmin <= x <= xmax) and (ymin <= y <= ymax) and (zmin <= z <= zmax):
                return False
        return True

    def is_collision_free(self, node):
        return self.is_point_free(node.x, node.y, node.z)

    def is_segment_free(self, a, b):
        dist = np.linalg.norm([b.x - a.x, b.y - a.y, b.z - a.z])
        if dist == 0:
            return self.is_point_free(a.x, a.y, a.z)
        n_steps = max(1, int(np.ceil(dist / self.collision_check_resolution)))
        for i in range(n_steps + 1):
            t = i / n_steps
            x = a.x + t * (b.x - a.x)
            y = a.y + t * (b.y - a.y)
            z = a.z + t * (b.z - a.z)
            if not self.is_point_free(x, y, z):
                return False
        return True

    def plan(self):
        for it in range(self.max_iter):
            if np.random.rand() < self.goal_bias:
                rand_node = Node(self.goal.x, self.goal.y, self.goal.z)
            else:
                rand_node = Node(np.random.uniform(*self.x_bounds),
                                  np.random.uniform(*self.y_bounds),
                                  np.random.uniform(*self.z_bounds))

            nearest = min(self.tree, key=lambda n: np.linalg.norm(
                [n.x - rand_node.x, n.y - rand_node.y, n.z - rand_node.z]))
            vec = np.array([rand_node.x - nearest.x, rand_node.y - nearest.y, rand_node.z - nearest.z])
            dist = np.linalg.norm(vec)
            if dist == 0:
                continue

            vec = (vec / dist) * min(self.step_size, dist)
            new_node = Node(nearest.x + vec[0], nearest.y + vec[1], nearest.z + vec[2])
            new_node.parent = nearest

            if self.is_segment_free(nearest, new_node):
                self.tree.append(new_node)

                dist_to_goal = np.linalg.norm(
                    [new_node.x - self.goal.x, new_node.y - self.goal.y, new_node.z - self.goal.z])

                if dist_to_goal < self.goal_tolerance and self.is_segment_free(new_node, self.goal):
                    self.goal.parent = new_node
                    self.tree.append(self.goal)
                    path, curr = [], self.goal
                    while curr is not None:
                        path.append((curr.x, curr.y, curr.z))
                        curr = curr.parent
                    return path[::-1], self.tree

        print(f"RRT FAILED after {self.max_iter} iterations: "
              f"start={self.start.x, self.start.y, self.start.z}, "
              f"goal={self.goal.x, self.goal.y, self.goal.z}, tree_size={len(self.tree)}")
        return None, self.tree