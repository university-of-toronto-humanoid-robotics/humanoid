"""Route files (navigation/routes/*.yaml): the waypoints the robot walks through, on the course map.

  speed: 0.4                    # m/s, walking between waypoints
  waypoints:
    - {x: 10.5, y: 4.2}                         # walk through
    - {x: 10.5, y: 4.9, yaw: 90}                # stop here, facing +y (deg in the course frame)
    - {x: 11.0, y: 4.9, yaw: 0, wait: 2.0}      # stop, then wait 2 s before going on
    - {x: 11.5, y: 3.5, tolerance: 0.05}        # stop more precisely than the default

A waypoint with yaw, wait or tolerance is a stop: the robot lines up on it (position, and heading if yaw is given)
before going on. Any other waypoint is walked through without slowing down. The last waypoint is always a stop.
The frame is the course frame (sim/competition/README.md); x and y are the pelvis's position.

No ROS in here, so it can be tested on its own.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

DEFAULT_TOLERANCE = 0.10    # m, how close a stop must be reached
DEFAULT_YAW_TOLERANCE = 5.0  # deg


@dataclass
class Waypoint:
    x: float
    y: float
    yaw: Optional[float] = None      # rad, or None for any heading
    stop: bool = False
    wait: float = 0.0                # s to stand at a stop before going on
    tolerance: float = DEFAULT_TOLERANCE
    yaw_tolerance: float = np.radians(DEFAULT_YAW_TOLERANCE)

    @property
    def xy(self):
        return np.array([self.x, self.y])


@dataclass
class Route:
    waypoints: list
    speed: float = 0.4
    name: str = ''


def make_waypoint(spec):
    yaw = spec.get('yaw')
    return Waypoint(
        x=float(spec['x']), y=float(spec['y']),
        yaw=None if yaw is None else np.radians(float(yaw)),
        stop=any(k in spec for k in ('yaw', 'wait', 'tolerance')),
        wait=float(spec.get('wait', 0.0)),
        tolerance=float(spec.get('tolerance', DEFAULT_TOLERANCE)),
        yaw_tolerance=np.radians(float(spec.get('yaw_tolerance', DEFAULT_YAW_TOLERANCE))))


def make_route(spec, name=''):
    waypoints = [make_waypoint(w) for w in spec['waypoints']]
    if not waypoints:
        raise ValueError(f'Route "{name}" has no waypoints')
    waypoints[-1].stop = True
    return Route(waypoints, float(spec.get('speed', 0.4)), name)


def load_route(path):
    path = Path(path)
    with open(path) as f:
        return make_route(yaml.safe_load(f), path.stem)


def obstacles_near(route, heights, x0, y0, res, start=None, radius=0.3, too_tall=0.25, step=0.02):
    """Places along the route that pass within radius (m) of something too_tall (m) above the floor under them.

    The navigator doesn't steer around anything: it walks the straight lines between waypoints. So a route has to
    clear the course by itself, and this checks it against the course's height grid. Stairs (20 cm risers), the
    step-over beams and the door sill are lower than too_tall, so routes may cross them. Returns [(x, y, height)].
    """
    points = ([np.asarray(start, dtype=float)] if start is not None else []) + [w.xy for w in route.waypoints]
    r = int(np.ceil(radius / res))
    di, dj = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing='ij')
    disk = np.hypot(di, dj) * res <= radius
    di, dj = di[disk], dj[disk]
    hits = []
    for a, b in zip(points[:-1], points[1:]):
        n = max(int(np.linalg.norm(b - a) / step), 1)
        for xy in a + (b - a) * (np.arange(n + 1) / n)[:, None]:
            i, j = int(round((xy[0] - x0) / res)), int(round((xy[1] - y0) / res))
            if not (0 <= i < heights.shape[0] and 0 <= j < heights.shape[1]):
                continue
            ii, jj = np.clip(i + di, 0, heights.shape[0] - 1), np.clip(j + dj, 0, heights.shape[1] - 1)
            above = heights[ii, jj] - heights[i, j]
            if above.max() > too_tall:
                hits.append((float(xy[0]), float(xy[1]), float(above.max())))
    return hits
