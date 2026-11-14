"""CAD-frame route geometry and a look-ahead velocity follower.

The geometry check screens the robot centre against a 2D obstacle mask. It
cannot prove that a full humanoid body clears walls, rails, or overhangs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HEIGHTMAP = ROOT / "sim" / "competition" / "course_heightmap.npz"
ROUTE = HERE / "mobility_route.json"
COURSE_START_XY = np.array([11.0, 4.21])


def obstacle_mask(heights: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Fixed walls and fixtures, with elevated stairs/landing left traversable."""
  stairs = (2.42 <= x) & (x <= 3.72) & (4.62 <= y) & (y <= 5.27)
  landing = (1.22 <= x) & (x <= 2.44) & (3.02 <= y) & (y <= 5.32)
  inside_lane = (1.22 <= x) & (x <= 12.19) & (3.00 <= y) & (y <= 5.42)
  return (~inside_lane) | ((heights > 0.55) & ~(stairs | landing))


@dataclass
class CourseMap:
  heights: np.ndarray
  x0: float
  y0: float
  res: float
  clearance: np.ndarray

  @classmethod
  def load(cls, path: Path = HEIGHTMAP) -> CourseMap:
    with np.load(path) as data:
      heights = data["heights"]
      x0, y0, res = (float(data[k]) for k in ("x0", "y0", "res"))
    x = x0 + np.arange(heights.shape[0])[:, None] * res
    y = y0 + np.arange(heights.shape[1])[None, :] * res
    clearance = distance_transform_edt(~obstacle_mask(heights, x, y)) * res
    return cls(heights, x0, y0, res, clearance)

  def indices(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy)
    i = np.rint((xy[..., 0] - self.x0) / self.res).astype(int)
    j = np.rint((xy[..., 1] - self.y0) / self.res).astype(int)
    if ((i < 0) | (i >= self.heights.shape[0]) | (j < 0) | (j >= self.heights.shape[1])).any():
      raise ValueError("Route point leaves the CAD height map")
    return i, j

  def clearance_at(self, xy: np.ndarray) -> np.ndarray:
    i, j = self.indices(xy)
    return self.clearance[i, j]

  def heights_at(self, xy: np.ndarray) -> np.ndarray:
    i, j = self.indices(xy)
    return self.heights[i, j]


def load_route(path: Path = ROUTE) -> dict:
  route = json.loads(path.read_text())
  if route["return_path"] != "reverse_outbound":
    raise ValueError("Only reverse_outbound is supported")
  points = route["waypoints"]
  if len(points) < 2 or len({p["id"] for p in points}) != len(points):
    raise ValueError("Route must have at least two waypoints with unique IDs")
  return route


def _line(a: np.ndarray, b: np.ndarray, step: float) -> np.ndarray:
  n = max(2, math.ceil(float(np.linalg.norm(b - a)) / step) + 1)
  return np.linspace(a, b, n)


def _corner_curve(before: np.ndarray, corner: np.ndarray, after: np.ndarray,
                  cutback: float, step: float) -> np.ndarray:
  incoming = (corner - before) / np.linalg.norm(corner - before)
  outgoing = (after - corner) / np.linalg.norm(after - corner)
  a, b = corner - cutback * incoming, corner + cutback * outgoing
  n = max(6, math.ceil(2 * cutback / step) + 1)
  t = np.linspace(0.0, 1.0, n)[:, None]
  return (1 - t) ** 2 * a + 2 * (1 - t) * t * corner + t ** 2 * b


@dataclass
class RoutePath:
  xy: np.ndarray
  s: np.ndarray
  goal_s: dict[str, float]

  @property
  def length(self) -> float:
    return float(self.s[-1])

  def point_at(self, distance: float) -> np.ndarray:
    distance = float(np.clip(distance, 0.0, self.length))
    return np.array([np.interp(distance, self.s, self.xy[:, axis]) for axis in (0, 1)])

  def nearest(self, xy: np.ndarray, previous_s: float, window: float = 1.0) -> tuple[float, float]:
    """Search only nearby arc length so the ramp Z and return cannot skip ahead."""
    keep = (self.s >= max(0.0, previous_s - 0.1)) & (self.s <= previous_s + window)
    candidates = np.flatnonzero(keep)
    nearest = int(candidates[np.argmin(np.linalg.norm(self.xy[candidates] - xy, axis=1))])
    return max(previous_s, float(self.s[nearest])), float(np.linalg.norm(self.xy[nearest] - xy))


def build_paths(route: dict, course: CourseMap) -> tuple[RoutePath, RoutePath, list[str]]:
  """Smooth free via-point corners and reject any resulting path near a wall."""
  points = route["waypoints"]
  coords = np.array([[p["x"], p["y"]] for p in points], dtype=float)
  if (np.linalg.norm(np.diff(coords, axis=0), axis=1) < 1e-6).any():
    raise ValueError("Consecutive waypoints must differ")
  margin = float(route["wall_clearance_m"])
  step = course.res / 2
  curves: dict[int, np.ndarray] = {}
  smoothed = []
  for i in range(1, len(points) - 1):
    p = points[i]
    if p["kind"] != "via" or not p.get("smooth", True):
      continue
    before, corner, after = coords[i - 1:i + 2]
    in_len = float(np.linalg.norm(corner - before))
    out_len = float(np.linalg.norm(after - corner))
    direction_change = np.dot((corner - before) / in_len, (after - corner) / out_len)
    if direction_change > 0.995:
      continue
    cutback = min(0.25, in_len * 0.25, out_len * 0.25)
    while cutback >= 0.02:
      curve = _corner_curve(before, corner, after, cutback, step)
      if float(course.clearance_at(curve).min()) >= margin:
        curves[i] = curve
        smoothed.append(p["id"])
        break
      cutback *= 0.5

  path = [coords[0]]
  for i in range(1, len(points)):
    end = curves[i][0] if i in curves else coords[i]
    path.extend(_line(path[-1], end, step)[1:])
    if i in curves:
      path.extend(curves[i][1:])
  xy = np.asarray(path)
  keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-9]
  xy = xy[keep]
  minimum = float(course.clearance_at(xy).min())
  if minimum < margin:
    raise ValueError(f"Smoothed route clearance {minimum:.2f} m is below {margin:.2f} m")
  s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
  goal_s = {}
  for p, coord in zip(points, coords):
    if p["kind"] in {"start", "goal", "end"}:
      idx = int(np.argmin(np.linalg.norm(xy - coord, axis=1)))
      if np.linalg.norm(xy[idx] - coord) > 1e-6:
        raise ValueError(f"Target {p['id']} moved off the route during smoothing")
      goal_s[p["id"]] = float(s[idx])
  outbound = RoutePath(xy, s, goal_s)
  reverse_xy = xy[::-1].copy()
  reverse_s = outbound.length - s[::-1]
  reverse_goal_s = {name: outbound.length - distance for name, distance in goal_s.items()}
  return outbound, RoutePath(reverse_xy, reverse_s, reverse_goal_s), smoothed


@dataclass
class Command:
  vx: float
  vy: float
  yaw_rate: float
  cross_track: float
  progress: float
  phase: str
  next_goal: str | None


class RouteFollower:
  """Pure-pursuit-style look-ahead commands in the robot's body frame."""

  def __init__(self, route: dict, outbound: RoutePath, returning: RoutePath) -> None:
    self.route = route
    self.paths = {"outbound": outbound, "return": returning}
    self.radius = float(route["goal_radius_m"])
    self.reset()

  def reset(self) -> None:
    self.phase = "outbound"
    self.progress_s = 0.0
    self.goal_index = 0

  def _goals(self) -> list[dict]:
    points = self.route["waypoints"]
    if self.phase == "outbound":
      return [p for p in points[1:] if p["kind"] in {"goal", "end"}]
    return [p for p in reversed(points[:-1]) if p["kind"] in {"goal", "start"}]

  def update(self, cad_xy: np.ndarray, yaw: float) -> Command:
    if self.phase == "complete":
      return Command(0, 0, 0, 0, 1, "complete", None)
    goals = self._goals()
    while self.goal_index < len(goals):
      goal = goals[self.goal_index]
      if np.linalg.norm(cad_xy - [goal["x"], goal["y"]]) > self.radius:
        break
      self.goal_index += 1
    if self.goal_index == len(goals):
      if self.phase == "outbound":
        self.phase = "return"
        self.progress_s = 0.0
        self.goal_index = 0
        goals = self._goals()
      else:
        self.phase = "complete"
        return Command(0, 0, 0, 0, 1, "complete", None)

    path = self.paths[self.phase]
    self.progress_s, cross_track = path.nearest(cad_xy, self.progress_s)
    next_goal = goals[self.goal_index]
    goal_id = next_goal["id"]
    goal_xy = np.array([next_goal["x"], next_goal["y"]])
    goal_distance = float(np.linalg.norm(goal_xy - cad_xy))
    lookahead = 0.30 if 2.3 <= cad_xy[0] <= 3.8 or 9.3 <= cad_xy[0] <= 10.8 else 0.50
    target = path.point_at(min(self.progress_s + lookahead, path.goal_s[goal_id]))
    heading_error = math.atan2(math.sin(math.atan2(target[1] - cad_xy[1], target[0] - cad_xy[0]) - yaw),
                               math.cos(math.atan2(target[1] - cad_xy[1], target[0] - cad_xy[0]) - yaw))
    speed = 0.35
    if 2.3 <= cad_xy[0] <= 3.8 or 9.3 <= cad_xy[0] <= 10.8:
      speed = 0.18
    elif 6.1 <= cad_xy[0] <= 8.6:
      speed = 0.25
    speed *= float(np.clip(goal_distance / 0.40, 0.4, 1.0))
    if abs(heading_error) > 0.65:
      vx = vy = 0.0
    else:
      vx = float(np.clip(speed * math.cos(heading_error), 0.0, 1.0))
      vy = float(np.clip(speed * math.sin(heading_error), -0.3, 0.3))
    yaw_rate = float(np.clip(2.0 * heading_error, -1.0, 1.0))
    fraction = (self.progress_s + (path.length if self.phase == "return" else 0.0)) / (2 * path.length)
    return Command(vx, vy, yaw_rate, cross_track, float(fraction), self.phase, goal_id)
