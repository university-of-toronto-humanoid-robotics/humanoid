"""Build the competition course for MuJoCo from the organizers' CAD.

Turns the testbed model (COLLADA) into a 2.5D height grid, the top surface as
seen from above, then into collision boxes (see course_boxes.py). Writes, into
sim/competition/:

  course_heightmap.npz  1 cm height grid + metadata (the "perfect height map")
  course.xml            MuJoCo scene: floor plane + boxes built from the grid
  course_heightmap.png  top-down picture (only if matplotlib is installed)

Course frame: the CAD's x/y in meters, z shifted so the main deck floor is z=0.
heights[i, j] is the surface height at x = x0 + i*res, y = y0 + j*res.

Two simplifications, both fine for walking:
  - Parts that start more than --cap meters above the deck (door header, tops
    of frames) are dropped, so doorways stay open.
  - Everything else is cut at --cap, so walls read as --cap tall.

Usage:
  python tools/cad/fetch_testbed.py              # once, downloads the CAD
  python tools/cad/build_course.py               # CAD -> grid -> course.xml
  python tools/cad/build_course.py --from-grid   # rebuild course.xml from the saved grid
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path

import numpy as np

from course_boxes import boxes_xml, grid_to_boxes

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DAE = REPO_ROOT / "sim" / "competition" / "cad" / "LocoManipulationTestbed.dae"
DEFAULT_OUT = REPO_ROOT / "sim" / "competition"
PAD = 0.10  # m of outside ground kept around the testbed
CHUNK_POINTS = 4_000_000  # bounds memory while rasterizing


def read_unit_scale(dae: Path) -> float:
  """Meters per CAD unit, from the COLLADA <unit meter="..."> tag."""
  with dae.open("rb") as f:
    head = f.read(16384).decode("utf-8", "ignore")
  m = re.search(r'<unit[^>]*meter="([0-9.eE+-]+)"', head)
  return float(m.group(1)) if m else 1.0


def load_triangles(dae: Path) -> np.ndarray:
  """All triangles of the scene in world coordinates (meters). Shape (T, 3, 3)."""
  import trimesh

  scene = trimesh.load(str(dae), force="scene")
  if hasattr(scene, "to_geometry"):
    mesh = scene.to_geometry()
  else:
    mesh = scene.dump(concatenate=True)
  v = np.asarray(mesh.vertices, dtype=np.float64) * read_unit_scale(dae)
  f = np.asarray(mesh.faces, dtype=np.int64)
  length = np.ptp(v[:, 0])
  if not 5.0 < length < 50.0:
    raise ValueError(f"Testbed is {length:.1f} m long after unit scaling; check units.")
  return v[f]


def keep_max(hm: np.ndarray, x0: float, y0: float, res: float, pts: np.ndarray) -> None:
  """hm[cell] = max(hm[cell], z) for each (x, y, z) point; points off-grid are ignored."""
  ix = np.rint((pts[:, 0] - x0) / res).astype(np.int64)
  iy = np.rint((pts[:, 1] - y0) / res).astype(np.int64)
  nx, ny = hm.shape
  ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
  np.maximum.at(hm, (ix[ok], iy[ok]), pts[ok, 2])


def splat_vertices(hm, x0, y0, res, tris, cap_abs) -> None:
  verts = tris.reshape(-1, 3).copy()
  verts[:, 2] = np.minimum(verts[:, 2], cap_abs)
  keep_max(hm, x0, y0, res, verts)


def splat_edges(hm, x0, y0, res, tris, cap_abs) -> None:
  """Sample points along every triangle edge (catches thin walls and top edges)."""
  a = tris.reshape(-1, 3)
  b = np.roll(tris, -1, axis=1).reshape(-1, 3)
  n = np.floor(np.linalg.norm((b - a)[:, :2], axis=1) / (res * 0.5)).astype(np.int64)
  keep = n > 0
  a, b, n = a[keep], b[keep], n[keep]
  cum = np.cumsum(n)
  start = 0
  while start < len(n):
    done = cum[start - 1] if start > 0 else 0
    end = max(int(np.searchsorted(cum, done + CHUNK_POINTS, side="right")), start + 1)
    nn = n[start:end]
    edge = np.repeat(np.arange(start, end), nn)
    k = np.arange(nn.sum()) - np.repeat(np.cumsum(nn) - nn, nn) + 1  # 1..n per edge
    t = (k / (n[edge] + 1))[:, None]
    pts = a[edge] + (b[edge] - a[edge]) * t
    pts[:, 2] = np.minimum(pts[:, 2], cap_abs)
    keep_max(hm, x0, y0, res, pts)
    start = end


def fill_interiors(hm, x0, y0, res, tris, cap_abs) -> None:
  """Exact top-down rasterization: every cell center inside a triangle gets its height."""
  xy = tris[:, :, :2]
  v0 = xy[:, 1] - xy[:, 0]
  v1 = xy[:, 2] - xy[:, 0]
  den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
  i0 = np.ceil((xy[:, :, 0].min(1) - x0) / res - 1e-9).astype(np.int64)
  i1 = np.floor((xy[:, :, 0].max(1) - x0) / res + 1e-9).astype(np.int64)
  j0 = np.ceil((xy[:, :, 1].min(1) - y0) / res - 1e-9).astype(np.int64)
  j1 = np.floor((xy[:, :, 1].max(1) - y0) / res + 1e-9).astype(np.int64)
  span = np.maximum(i1 - i0, j1 - j0) + 1
  todo = (np.abs(den) > 1e-12) & (i1 >= i0) & (j1 >= j0)
  nx, ny = hm.shape

  # Group triangles by how many cells their bounding box spans (1, 2, 4, 8, ...),
  # then test all candidate cells of a group at once.
  size = 1
  while todo.any():
    group = np.nonzero(todo & (span <= size))[0]
    todo[group] = False
    per_chunk = max(1, CHUNK_POINTS // (size * size))
    di, dj = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    di, dj = di.ravel(), dj.ravel()
    for s in range(0, len(group), per_chunk):
      g = group[s : s + per_chunk]
      ii = i0[g, None] + di[None]
      jj = j0[g, None] + dj[None]
      px = x0 + ii * res - xy[g, 0, 0, None]
      py = y0 + jj * res - xy[g, 0, 1, None]
      w1 = (px * v1[g, 1, None] - py * v1[g, 0, None]) / den[g, None]
      w2 = (v0[g, 0, None] * py - v0[g, 1, None] * px) / den[g, None]
      w0 = 1.0 - w1 - w2
      inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
      inside &= (ii <= i1[g, None]) & (jj <= j1[g, None])
      inside &= (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
      z = w0 * tris[g, 0, 2, None] + w1 * tris[g, 1, 2, None] + w2 * tris[g, 2, 2, None]
      np.maximum.at(hm, (ii[inside], jj[inside]), np.minimum(z[inside], cap_abs))
    size *= 2


def find_deck_height(hm: np.ndarray) -> float:
  """The main floor is the most common height in the grid."""
  vals = hm[np.isfinite(hm)]
  vals = vals[(vals > 0.05) & (vals < 0.3)]
  uniq, counts = np.unique(np.round(vals, 3), return_counts=True)
  mode = uniq[np.argmax(counts)]
  return float(np.median(vals[np.abs(vals - mode) <= 0.001]))


def cad_to_grid(dae: Path, res: float, cap: float) -> dict:
  """Rasterize the CAD into the course height grid. Returns the npz fields."""
  t0 = time.time()
  print(f"Loading {dae.name} ...")
  tris = load_triangles(dae)
  print(f"  {len(tris):,} triangles ({time.time() - t0:.0f} s)")

  lo = tris[:, :, :2].reshape(-1, 2).min(0) - PAD
  hi = tris[:, :, :2].reshape(-1, 2).max(0) + PAD
  x0, y0 = (float(v) for v in np.floor(lo / res) * res)
  nx = int(np.ceil((hi[0] - x0) / res)) + 1
  ny = int(np.ceil((hi[1] - y0) / res)) + 1

  # The cap is measured from the deck, so find the deck on a coarse pass first.
  coarse = np.full((nx // 4 + 1, ny // 4 + 1), -np.inf)
  fill_interiors(coarse, x0, y0, res * 4, tris, np.inf)
  deck_z = find_deck_height(coarse)
  cap_abs = deck_z + cap
  print(f"  deck at CAD z = {deck_z:.4f} m; cutting at {cap_abs:.3f} m")

  tris = tris[tris[:, :, 2].min(1) < cap_abs]  # drop overhangs
  hm = np.full((nx, ny), -np.inf)
  fill_interiors(hm, x0, y0, res, tris, cap_abs)
  splat_edges(hm, x0, y0, res, tris, cap_abs)
  splat_vertices(hm, x0, y0, res, tris, cap_abs)
  ground_z = float(tris[:, :, 2].min())
  empty = ~np.isfinite(hm)
  hm[empty] = ground_z
  print(f"  grid {nx} x {ny} at {res * 100:.0f} cm, {empty.mean() * 100:.1f}% empty -> ground "
        f"({time.time() - t0:.0f} s)")
  return dict(
    heights=(hm - deck_z).astype(np.float32),
    x0=np.float64(x0),
    y0=np.float64(y0),
    res=np.float64(res),
    deck_z_cad=np.float64(deck_z),
    cap=np.float64(cap),
    source_sha256=hashlib.sha256(dae.read_bytes()).hexdigest(),
    frame="CAD x/y (m); z = CAD z - deck_z_cad. heights[i, j] at (x0 + i*res, y0 + j*res).",
  )


COURSE_XML = """<mujoco model="competition_course">
  <!-- Generated by tools/cad/build_course.py from the organizers' CAD. Do not edit by hand.
       Course frame: CAD x/y in meters, z shifted so the main deck floor is z=0.
       Source CAD sha256: {sha} -->
  <compiler angle="radian"/>

  <visual>
    <global offwidth="1600" offheight="900"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture name="deck" type="2d" builtin="checker" rgb1="0.62 0.56 0.48" rgb2="0.56 0.50 0.42"
             width="300" height="300" mark="edge" markrgb="0.4 0.36 0.3"/>
    <material name="floor" texture="deck" texrepeat="1.6404 1.6404" texuniform="true"/>
    <material name="low" rgba="0.80 0.62 0.38 1"/>
    <material name="ramp" rgba="0.70 0.55 0.35 1"/>
    <material name="beam" rgba="0.85 0.45 0.20 1"/>
    <material name="mid" rgba="0.55 0.58 0.62 1"/>
    <material name="tall" rgba="0.42 0.44 0.48 1"/>
{meshes}
  </asset>

  <worldbody>
    <light pos="{cx:.3f} {cy:.3f} 6" dir="0.2 0.3 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" pos="{cx:.4f} {cy:.4f} 0" size="30 30 0.1" material="floor"/>
{boxes}
  </worldbody>
</mujoco>
"""


def save_png(path: Path, heights: np.ndarray, x0: float, y0: float, res: float) -> None:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("matplotlib not installed; skipping the picture")
    return
  nx, ny = heights.shape
  extent = [x0 - res / 2, x0 + (nx - 0.5) * res, y0 - res / 2, y0 + (ny - 0.5) * res]
  fig, ax = plt.subplots(figsize=(16, 16 * ny / nx + 1.2))
  im = ax.imshow(np.clip(heights.T, -0.05, 1.1), origin="lower", extent=extent, cmap="turbo")
  ax.set_xlabel("x (m)")
  ax.set_ylabel("y (m)")
  ax.set_title("Competition course height map (course frame, deck = 0)")
  fig.colorbar(im, ax=ax, label="height (m), clipped at 1.1", shrink=0.8)
  fig.tight_layout()
  fig.savefig(path, dpi=80)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--dae", type=Path, default=DEFAULT_DAE)
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  parser.add_argument("--res", type=float, default=0.01, help="grid spacing in meters")
  parser.add_argument("--cap", type=float, default=1.8, help="max height above the deck (m)")
  parser.add_argument("--from-grid", action="store_true",
                      help="skip the CAD and rebuild course.xml from the saved grid")
  args = parser.parse_args()
  out: Path = args.out
  grid_path = out / "course_heightmap.npz"

  t0 = time.time()
  if args.from_grid:
    grid = dict(np.load(grid_path))
  else:
    if not args.dae.exists():
      raise SystemExit(f"{args.dae} not found. Run: python tools/cad/fetch_testbed.py")
    grid = cad_to_grid(args.dae, args.res, args.cap)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(grid_path, **grid)
    save_png(out / "course_heightmap.png", grid["heights"], float(grid["x0"]), float(grid["y0"]),
             float(grid["res"]))

  heights, x0, y0, res = grid["heights"], float(grid["x0"]), float(grid["y0"]), float(grid["res"])
  boxes = grid_to_boxes(heights, x0, y0, res)
  kinds = {k: sum(b.kind == k for b in boxes) for k in ("ramp", "beam", "flat")}
  nx, ny = heights.shape
  meshes, geoms = boxes_xml(boxes)
  xml = COURSE_XML.format(
    sha=str(grid["source_sha256"]),
    cx=x0 + (nx - 1) * res / 2,
    cy=y0 + (ny - 1) * res / 2,
    meshes=meshes,
    boxes=geoms,
  )
  (out / "course.xml").write_text(xml, encoding="utf-8")
  print(f"course.xml: {len(boxes)} boxes ({kinds['ramp']} ramp, {kinds['beam']} beam, "
        f"{kinds['flat']} flat) ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
  main()
