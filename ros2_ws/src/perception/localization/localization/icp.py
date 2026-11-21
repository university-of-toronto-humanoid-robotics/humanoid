"""Lines a LiDAR scan up with the course map: point-to-plane ICP over x, y, z and yaw.

The scan comes in FAST-LIO's odom frame, which gravity_align has already levelled, so roll and pitch are right and
only four numbers are unknown: where odom's origin is on the course (x, y, z) and which way it faces (yaw).

One round of ICP:
  1. move the scan by the current guess
  2. for each scan point, find the nearest map point
  3. drop pairs further apart than `max_dist`: things that aren't in the map (people, moved crates, the robot)
  4. solve for the small move that best puts each scan point on its map point's surface (along the normal)
Repeat until the move is tiny. A point on a flat floor only says "this high"; walls and step edges say where
along the floor, so both are needed.

No ROS in here, so it can be tested on its own.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

INLIER = 0.10       # m from the map surface that counts a scan point as matched
MIN_POINTS = 100    # scan points over the course needed to judge a fit at all


@dataclass
class Pose:
    """odom's pose in the course frame: a turn by yaw about z, then a shift by (x, y, z)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0

    def apply(self, points):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        out = np.empty_like(points)
        out[:, 0] = c * points[:, 0] - s * points[:, 1] + self.x
        out[:, 1] = s * points[:, 0] + c * points[:, 1] + self.y
        out[:, 2] = points[:, 2] + self.z
        return out

    def compose(self, other):
        """self after other: the pose that first applies other, then self."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return Pose(c * other.x - s * other.y + self.x, s * other.x + c * other.y + self.y,
                    other.z + self.z, wrap(self.yaw + other.yaw))

    def inverse(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return Pose(-(c * self.x + s * self.y), -(-s * self.x + c * self.y), -self.z, -self.yaw)

    def distance(self, other):
        """(m apart in x and y, rad apart in yaw)."""
        return float(np.hypot(self.x - other.x, self.y - other.y)), abs(wrap(self.yaw - other.yaw))


def wrap(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class Result:
    pose: Pose
    fitness: float      # share of the scan over the course that is within INLIER of the map surface
    rmse: float         # m, of those inliers
    iterations: int


class CourseMatcher:

    def __init__(self, map_points, map_normals):
        self.points = np.asarray(map_points, dtype=float)
        self.normals = np.asarray(map_normals, dtype=float)
        self.tree = cKDTree(self.points)
        self.low, self.high = self.points[:, :2].min(axis=0), self.points[:, :2].max(axis=0)

    def score(self, scan, pose):
        """How well scan fits at pose. Only points over the course count: what lies beyond it isn't in the map."""
        moved = pose.apply(scan)
        over = np.all((moved[:, :2] >= self.low) & (moved[:, :2] <= self.high), axis=1)
        if over.sum() < MIN_POINTS:
            return 0.0, np.inf
        dist, idx = self.tree.query(moved[over], distance_upper_bound=INLIER * 3)
        found = np.isfinite(dist)
        residual = np.abs(np.einsum('ij,ij->i', moved[over][found] - self.points[idx[found]], self.normals[idx[found]]))
        inlier = residual < INLIER
        fitness = inlier.sum() / over.sum()
        rmse = float(np.sqrt(np.mean(residual[inlier] ** 2))) if inlier.any() else np.inf
        return float(fitness), rmse

    def align(self, scan, guess, max_dist=0.3, iterations=30, damping=1e-3):
        """Refines guess until scan sits on the map. scan: (N, 3) in odom."""
        pose = Pose(**vars(guess))
        n_done = 0
        for n_done in range(1, iterations + 1):
            moved = pose.apply(scan)
            dist, idx = self.tree.query(moved, distance_upper_bound=max_dist)
            found = np.isfinite(dist)
            if found.sum() < 10:
                break
            q, m, n = moved[found], self.points[idx[found]], self.normals[idx[found]]
            residual = np.einsum('ij,ij->i', q - m, n)
            # A turn by d_yaw about the vertical through odom's origin, which is at (pose.x, pose.y) on the course,
            # moves each point by d_yaw * (-(qy - y), qx - x, 0).
            lever = np.column_stack([-(q[:, 1] - pose.y), q[:, 0] - pose.x])
            jac = np.column_stack([n[:, 0], n[:, 1], n[:, 2], np.einsum('ij,ij->i', lever, n[:, :2])])
            # Huber weights, so a few badly matched points can't pull the answer far.
            w = np.minimum(1.0, INLIER / np.maximum(np.abs(residual), 1e-9))
            h = jac.T @ (jac * w[:, None])
            # Damping keeps a direction the scan can't pin down (a long plain corridor) where it was.
            step = np.linalg.solve(h + damping * len(q) * np.eye(4), -jac.T @ (w * residual))
            dx, dy, dz, dyaw = step
            # The turn is about odom's origin, so it doesn't move the origin itself.
            pose = Pose(pose.x + dx, pose.y + dy, pose.z + dz, wrap(pose.yaw + dyaw))
            if np.hypot(dx, dy) < 1e-4 and abs(dz) < 1e-4 and abs(dyaw) < 1e-5:
                break
        fitness, rmse = self.score(scan, pose)
        return Result(pose, fitness, rmse, n_done)

    def search(self, scan, guess, radius=0.5, yaw_range=np.radians(20), xy_step=0.25, yaw_step=np.radians(10)):
        """Finds odom's pose when guess is only roughly right: tries a grid of starts around it and keeps the best.

        Each start is aligned coarsely (wide pairing) and then finely; the one that fits the most points wins.
        """
        offsets = np.arange(-radius, radius + 1e-9, xy_step)
        yaws = np.arange(-yaw_range, yaw_range + 1e-9, yaw_step)
        sample = scan[np.random.default_rng(0).permutation(len(scan))[:1500]]
        best = None
        for dyaw in yaws:
            for dx in offsets:
                for dy in offsets:
                    start = Pose(guess.x + dx, guess.y + dy, guess.z, wrap(guess.yaw + dyaw))
                    r = self.align(sample, start, max_dist=0.6, iterations=10)
                    r = self.align(sample, r.pose, max_dist=0.25, iterations=10)
                    if best is None or r.fitness > best.fitness:
                        best = r
        return self.align(scan, best.pose, max_dist=0.2, iterations=30)
