"""Check the generated MuJoCo course against the CAD height grid.

1. Rays: heights MuJoCo sees match the height grid (catches flips and offsets).
   The course is built from boxes (tools/cad/course_boxes.py), so this also
   checks the box conversion.
2. Landmarks: course features sit where, and as high as, we measured in the CAD.
3. Contact: a G1-shaped foot (7 thin capsules) dropped on the floor, each
   stair and the landing comes to rest flat, at the right height.

Needs only mujoco + numpy.

Usage:
  python tools/cad/check_course.py
Exit code 1 if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
COURSE_DIR = REPO_ROOT / "sim" / "competition"

# (name, x, y, height above deck in m), course frame. Measured from the CAD on
# 2026-10-28; see sim/competition/README.md.
LANDMARKS = [
  ("flat section floor", 5.00, 4.80, 0.000),
  ("start area floor", 11.00, 4.60, 0.000),
  ("stair 1", 3.56, 4.96, 0.2032),
  ("stair 2", 3.33, 4.96, 0.4064),
  ("stair 3", 3.08, 4.96, 0.6096),
  ("stair 4", 2.84, 4.96, 0.8128),
  ("top step", 2.60, 4.96, 1.0160),
  ("landing", 1.80, 3.80, 1.0224),
  ("step-over section floor", 9.20, 4.21, 0.0263),
  ("door sill", 9.70, 4.21, 0.0771),
  ("step-over beam (top)", 8.87, 3.95, 0.1279),
  ("ramp ridge", 6.70, 3.60, 0.189),
  ("ramp valley", 7.30, 3.60, 0.032),
  ("lane wall (top)", 5.00, 5.45, 0.8128),
]
LANDMARK_TOL = 0.006  # m
RAY_TOL = 0.003  # m (ramp tiles are plane fits, within ~1.5 mm of the CAD)
# Above TALL nothing is walked on, so course_boxes.py rounds heights up to 5 cm.
TALL, TALL_ROUNDING = 1.1, 0.05 + 1e-6
REST_TOL = 0.006  # m
DROP_SPOTS = ["flat section floor", "stair 1", "stair 2", "stair 3", "stair 4", "landing",
              "step-over section floor"]
# G1 sole: 7 capsules (radius 1 cm) copied from g1.xml, centered on the foot.
# A flat box would be a bad test: MuJoCo keeps at most 50 contacts per geom
# pair, and a wide box on a 1 cm heightfield touches far more cells than that.
FOOT_RADIUS = 0.01
FOOT_CAPSULES = [  # (x_from, x_to, y) in meters
  (0.061, 0.011, -0.026), (-0.083, 0.084, -0.018), (-0.091, 0.091, -0.010),
  (-0.093, 0.093, 0.000), (-0.091, 0.091, 0.010), (-0.083, 0.084, 0.018),
  (0.061, 0.011, 0.026),
]


class Grid:
  def __init__(self, path: Path):
    d = np.load(path)
    self.h = d["heights"]
    self.x0, self.y0, self.res = float(d["x0"]), float(d["y0"]), float(d["res"])

  def index(self, x: float, y: float) -> tuple[int, int]:
    return int(round((x - self.x0) / self.res)), int(round((y - self.y0) / self.res))

  def at(self, x: float, y: float) -> float:
    return float(self.h[self.index(x, y)])


def ray_height(model, data, x: float, y: float) -> float:
  geomid = np.zeros(1, dtype=np.int32)
  top = 10.0
  dist = mujoco.mj_ray(model, data, np.array([x, y, top]), np.array([0.0, 0.0, -1.0]),
                       None, 1, -1, geomid)
  return top - dist if dist >= 0 else float("nan")


def check_rays(model, data, grid: Grid, n: int = 3000) -> bool:
  rng = np.random.default_rng(0)
  h = grid.h
  # Test cells away from edges, where the surface is locally planar (floors,
  # treads, ramp tiles). Right at an edge the box and the grid cell can differ
  # by up to one cell, which is expected. Cells below the deck are outside the
  # testbed, where the floor plane deliberately covers the real ground.
  pad = np.pad(h.astype(np.float64), 1, mode="edge")
  d2x = np.abs(pad[2:, 1:-1] - 2 * h + pad[:-2, 1:-1])
  d2y = np.abs(pad[1:-1, 2:] - 2 * h + pad[1:-1, :-2])
  cand = np.argwhere((d2x < 0.001) & (d2y < 0.001) & (h > -0.0005))
  pick = cand[rng.choice(len(cand), size=min(n, len(cand)), replace=False)]
  errs = []
  for i, j in pick:
    x = grid.x0 + (i + 0.3) * grid.res  # aim inside the cell, not on its edge
    y = grid.y0 + (j + 0.2) * grid.res
    diff = ray_height(model, data, x, y) - h[i, j]
    if h[i, j] >= TALL and 0 <= diff < TALL_ROUNDING:
      diff = 0.0  # tall obstacles are rounded up on purpose
    errs.append(abs(diff))
  errs = np.array(errs)
  ok = bool(np.all(errs < RAY_TOL))
  print(f"[{'PASS' if ok else 'FAIL'}] rays: {len(errs)} points, max error {errs.max() * 1000:.2f} mm")
  if not ok:
    worst = np.argsort(errs)[-3:]
    for w in worst:
      i, j = pick[w]
      print(f"       off by {errs[w] * 1000:.1f} mm at x={grid.x0 + i * grid.res:.2f} "
            f"y={grid.y0 + j * grid.res:.2f}")
  return ok


def check_landmarks(model, data, grid: Grid) -> bool:
  all_ok = True
  for name, x, y, want in LANDMARKS:
    g, r = grid.at(x, y), ray_height(model, data, x, y)
    ok = abs(g - want) < LANDMARK_TOL and abs(r - want) < LANDMARK_TOL
    all_ok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name:24s} want {want:6.3f}  grid {g:6.3f}  mujoco {r:6.3f}")
  return all_ok


def check_drops(grid: Grid) -> bool:
  spec = mujoco.MjSpec.from_file(str(COURSE_DIR / "course.xml"))
  spots = {name: (x, y, want) for name, x, y, want in LANDMARKS if name in DROP_SPOTS}
  for k, (name, (x, y, want)) in enumerate(spots.items()):
    body = spec.worldbody.add_body(name=f"foot{k}", pos=[x, y, want + FOOT_RADIUS + 0.05])
    body.add_freejoint()
    for x_from, x_to, cy in FOOT_CAPSULES:
      body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[FOOT_RADIUS, 0, 0],
                    fromto=[x_from, cy, 0, x_to, cy, 0], mass=0.15)
  model = spec.compile()
  data = mujoco.MjData(model)
  while data.time < 1.5:
    mujoco.mj_step(model, data)
  all_ok = True
  for k, (name, (x, y, want)) in enumerate(spots.items()):
    foot = data.body(f"foot{k}")
    z = foot.xpos[2] - FOOT_RADIUS
    tilt = np.degrees(np.arccos(np.clip(foot.xmat[8], -1, 1)))
    ok = abs(z - want) < REST_TOL and tilt < 2.0
    all_ok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] foot resting on {name:24s} at {z:6.3f} "
          f"(want {want:6.3f}), tilt {tilt:4.1f} deg")
  return all_ok


def main() -> int:
  grid = Grid(COURSE_DIR / "course_heightmap.npz")
  model = mujoco.MjModel.from_xml_path(str(COURSE_DIR / "course.xml"))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  results = [check_rays(model, data, grid), check_landmarks(model, data, grid), check_drops(grid)]
  ok = all(results)
  print("\nAll checks passed." if ok else "\nSome checks FAILED.")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
