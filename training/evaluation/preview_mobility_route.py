"""Draw and screen an editable Mobility Trial route on the competition height map.

Run from the repository root with a Python environment containing numpy, scipy,
and matplotlib. This is a static 2D preview, not a robot controller or a full
collision test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
from route import CourseMap, build_paths, load_route, obstacle_mask


EVALUATION = Path(__file__).resolve().parent
ROOT = EVALUATION.parents[1]
COURSE = ROOT / "sim" / "competition"
DEFAULT_ROUTE = EVALUATION / "mobility_route.json"
DEFAULT_OUTPUT = EVALUATION / "mobility_route_preview.png"


def sample_segment(a: dict, b: dict, step: float) -> tuple[np.ndarray, np.ndarray]:
  length = float(np.hypot(b["x"] - a["x"], b["y"] - a["y"]))
  t = np.linspace(0.0, 1.0, max(2, int(np.ceil(length / step)) + 1))
  return a["x"] + t * (b["x"] - a["x"]), a["y"] + t * (b["y"] - a["y"])


def route_clearances(
  points: list[dict], clearance: np.ndarray, x0: float, y0: float, res: float
) -> list[tuple[str, str, float]]:
  results = []
  for a, b in zip(points, points[1:]):
    x, y = sample_segment(a, b, res / 2)
    i = np.rint((x - x0) / res).astype(int)
    j = np.rint((y - y0) / res).astype(int)
    if (i < 0).any() or (i >= clearance.shape[0]).any() or (j < 0).any() or (j >= clearance.shape[1]).any():
      raise ValueError(f"Segment {a['id']} -> {b['id']} leaves the CAD height map")
    results.append((a["id"], b["id"], float(clearance[i, j].min())))
  return results


def draw_panel(ax, points: list[dict], path_xy: np.ndarray, heights: np.ndarray, blocked: np.ndarray,
               x0: float, y0: float, res: float, title: str, color: str,
               radius: float) -> None:
  extent = (x0, x0 + heights.shape[0] * res, y0, y0 + heights.shape[1] * res)
  ax.imshow(heights.T, origin="lower", extent=extent, cmap="terrain", vmin=-0.05, vmax=1.15)
  ax.imshow(np.ma.masked_where(~blocked, blocked).T, origin="lower", extent=extent,
            cmap="Greys", vmin=0, vmax=1, alpha=0.45)
  xy = np.array([[p["x"], p["y"]] for p in points])
  ax.plot(path_xy[:, 0], path_xy[:, 1], color=color, linewidth=2.2, zorder=4)
  for a, b in zip(xy, xy[1:]):
    if np.linalg.norm(b - a) >= 0.25:
      ax.add_patch(FancyArrowPatch(a + 0.42 * (b - a), a + 0.67 * (b - a),
                                   arrowstyle="-|>", mutation_scale=10,
                                   color=color, linewidth=1.5, zorder=6))
  for index, p in enumerate(points):
    if p["kind"] in {"goal", "end"}:
      ax.add_patch(Circle((p["x"], p["y"]), radius, facecolor="#ffe65c",
                          edgecolor="#332500", linewidth=1, alpha=0.55, zorder=5))
    ax.scatter(p["x"], p["y"], s=21, c=color, edgecolors="white", linewidths=0.5, zorder=7)
    dy = 11 if index % 2 == 0 else -13
    ax.annotate(p["id"], (p["x"], p["y"]), xytext=(0, dy), textcoords="offset points",
                ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8})
  ax.set_title(title)
  ax.set_xlim(1.05, 13.3)
  ax.set_ylim(2.78, 5.65)
  ax.set_aspect("equal")
  ax.set_xticks(np.arange(2, 13, 1))
  ax.set_yticks(np.arange(3, 5.6, 0.5))
  ax.grid(alpha=0.18)
  ax.set_xlabel("CAD x (m)")
  ax.set_ylabel("CAD y (m)")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--route", type=Path, default=DEFAULT_ROUTE)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()

  route = load_route(args.route)
  points = route["waypoints"]
  ids = [p["id"] for p in points]
  if len(ids) != len(set(ids)) or len(points) < 2:
    raise ValueError("Route needs at least two waypoints with unique IDs")
  radius = float(route["goal_radius_m"])
  margin = float(route["wall_clearance_m"])
  course = CourseMap.load(COURSE / "course_heightmap.npz")
  heights, x0, y0, res, clearance = (
    course.heights, course.x0, course.y0, course.res, course.clearance
  )
  x = x0 + np.arange(heights.shape[0])[:, None] * res
  y = y0 + np.arange(heights.shape[1])[None, :] * res
  blocked = obstacle_mask(heights, x, y)
  segments = route_clearances(points, clearance, x0, y0, res)
  outbound, returning, smoothed = build_paths(route, course)

  print("Waypoint centres (CAD frame, metres):")
  for p in points:
    print(f"  {p['id']:>3}  ({p['x']:5.2f}, {p['y']:4.2f})  {p['name']}")
  print(f"\n2D wall/fixture clearance screen (required >= {margin:.2f} m):")
  for start, end, distance in segments:
    status = "OK" if distance >= margin else "FAIL"
    print(f"  {start:>3} -> {end:<3}  {distance:.2f} m  {status}")
  failures = [(a, b) for a, b, d in segments if d < margin]
  goal_failures = []
  print(f"\nEstimated target circles (radius {radius:.2f} m):")
  for p in points:
    if p["kind"] not in {"goal", "end"}:
      continue
    i = round((p["x"] - x0) / res)
    j = round((p["y"] - y0) / res)
    distance = float(clearance[i, j])
    status = "OK" if distance >= radius else "FAIL"
    print(f"  {p['id']:>3}  centre clearance {distance:.2f} m  {status}")
    if distance < radius:
      goal_failures.append(p["id"])

  fig, axes = plt.subplots(2, 1, figsize=(17, 9), constrained_layout=True)
  draw_panel(axes[0], points, outbound.xy, heights, blocked, x0, y0, res,
             "Outbound: Start → A → B → C → D → End", "#00d8ff", radius)
  draw_panel(axes[1], list(reversed(points)), returning.xy, heights, blocked, x0, y0, res,
             "Return: End → D → C → B → A → Start", "#ff6a00", radius)
  fig.suptitle("Provisional Mobility Trial route — circle locations estimated from rules diagram",
               fontsize=13, fontweight="bold")
  fig.savefig(args.output, dpi=180)
  plt.close(fig)
  print(f"\nPreview: {args.output}")
  if failures or goal_failures:
    raise SystemExit(f"Route clearance failed: segments={failures}, goals={goal_failures}")
  print(f"Minimum sampled clearance: {min(d for _, _, d in segments):.2f} m")
  print(f"Smoothed corners: {', '.join(smoothed) or 'none'}")


if __name__ == "__main__":
  main()
