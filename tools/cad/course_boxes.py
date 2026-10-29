"""Turn the course height grid into MuJoCo boxes (fast, exact collision geometry).

A heightfield at 1 cm would be the most direct copy of the grid, but every
robot geom then touches hundreds of cells: MuJoCo slows to a crawl and mjlab
runs out of contacts. Boxes are cheap and exact for this course, because it
is built from lumber:

  - Sloped 15 deg ramp tiles -> one wedge per planar patch (plane fit): a
    convex mesh with vertical sides and a sloped top. (A tilted box would lean
    out over the floor at the tile's high end.)
  - The two diagonal step-over beams -> one rotated box each (line fit).
  - Everything else (stairs, landing, plates, sill, walls, counter, ...) is
    flat-topped, so it is cut into axis-aligned boxes straight from the grid.

The floor (deck, z = 0) is a plane, so floor cells need no box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

FLOOR_TOL = 0.0005  # m; cells this close to z=0 are floor
BOX_BOTTOM = -0.02  # m; boxes start just below the floor plane
TALL = 1.1  # m; above this nothing is walked on, so heights are rounded up to 5 cm
SLOPE_RANGE = (0.15, 0.45)  # height gradient treated as a ramp (8.5 to 24 deg)
RAMP_MIN_AREA = 0.05  # m^2
RAMP_FIT_TOL = 0.003  # m
BEAM_TOP = 0.1279  # m above the deck, measured in the CAD
BEAM_REGION = ((8.5, 9.8), (2.95, 5.5))  # x, y range of the step-over section


@dataclass
class Box:
  pos: tuple[float, float, float]
  size: tuple[float, float, float]  # half sizes
  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  kind: str = "flat"  # flat | ramp | beam
  vertices: tuple | None = None  # ramps only: wedge corners relative to pos (a convex mesh)


def _cell_xy(ii, jj, x0, y0, res):
  return x0 + ii * res, y0 + jj * res


def find_ramps(h: np.ndarray, x0: float, y0: float, res: float) -> tuple[list[Box], np.ndarray]:
  """Planar sloped patches -> tilted boxes. Returns the boxes and the cells they cover."""
  gx, gy = np.gradient(h.astype(np.float64), res)
  g = np.hypot(gx, gy)
  sloped = (g > SLOPE_RANGE[0]) & (g < SLOPE_RANGE[1])
  # Split by downhill direction so neighbouring tiles that face different ways separate.
  direction = np.where(np.abs(gx) > np.abs(gy), np.where(gx > 0, 1, 2), np.where(gy > 0, 3, 4))
  covered = np.zeros(h.shape, dtype=bool)
  boxes = []
  nx, ny = h.shape
  for d in (1, 2, 3, 4):
    labels, n = ndimage.label(sloped & (direction == d))
    for k in range(1, n + 1):
      ii, jj = np.nonzero(labels == k)
      if len(ii) * res * res < RAMP_MIN_AREA:
        continue
      x, y = _cell_xy(ii, jj, x0, y0, res)
      A = np.column_stack([x, y, np.ones_like(x)])
      (a, b, c), *_ = np.linalg.lstsq(A, h[ii, jj], rcond=None)

      def fits(i_sl, j_sl):
        I, J = np.meshgrid(np.arange(nx)[i_sl], np.arange(ny)[j_sl], indexing="ij")
        X, Y = _cell_xy(I, J, x0, y0, res)
        return np.all(np.abs(h[i_sl, j_sl] - (a * X + b * Y + c)) < RAMP_FIT_TOL)

      # The gradient mask misses the ridge/valley rows; grow the patch back to its edges.
      i0, i1, j0, j1 = ii.min(), ii.max(), jj.min(), jj.max()
      for _ in range(6):
        if i0 > 0 and fits(slice(i0 - 1, i0), slice(j0, j1 + 1)):
          i0 -= 1
        if i1 < nx - 1 and fits(slice(i1 + 1, i1 + 2), slice(j0, j1 + 1)):
          i1 += 1
        if j0 > 0 and fits(slice(i0, i1 + 1), slice(j0 - 1, j0)):
          j0 -= 1
        if j1 < ny - 1 and fits(slice(i0, i1 + 1), slice(j1 + 1, j1 + 2)):
          j1 += 1

      # The wedge fills the whole rectangle. Things standing on the ramp (a post)
      # still get their own boxes; anything *below* the ramp surface would be
      # buried, so refuse to build a wedge over it.
      I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
      X, Y = _cell_xy(I, J, x0, y0, res)
      dz = h[i0 : i1 + 1, j0 : j1 + 1] - (a * X + b * Y + c)
      on_plane = np.abs(dz) < RAMP_FIT_TOL
      if on_plane.mean() < 0.85 or (dz < -RAMP_FIT_TOL).sum() > 0.01 * dz.size:
        continue
      covered[i0 : i1 + 1, j0 : j1 + 1] |= on_plane

      xa, xb = x0 + (i0 - 0.5) * res, x0 + (i1 + 0.5) * res
      ya, yb = y0 + (j0 - 0.5) * res, y0 + (j1 + 0.5) * res
      cx, cy = (xa + xb) / 2, (ya + yb) / 2
      corners = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
      tops = [a * x + b * y + c for x, y in corners]
      verts = [(x - cx, y - cy, BOX_BOTTOM) for x, y in corners]
      verts += [(x - cx, y - cy, z) for (x, y), z in zip(corners, tops)]
      half_z = (max(tops) - BOX_BOTTOM) / 2
      boxes.append(Box((cx, cy, 0.0), ((xb - xa) / 2, (yb - ya) / 2, half_z), kind="ramp",
                       vertices=tuple(verts)))
  return boxes, covered


def find_beams(h: np.ndarray, x0: float, y0: float, res: float) -> tuple[list[Box], np.ndarray]:
  """The V of two diagonal 4x4 beams -> two rotated boxes."""
  (xa, xb), (ya, yb) = BEAM_REGION
  i0, i1 = int((xa - x0) / res), int((xb - x0) / res)
  j0, j1 = int((ya - y0) / res), int((yb - y0) / res)
  mask = np.zeros(h.shape, dtype=bool)
  mask[i0:i1, j0:j1] = np.abs(h[i0:i1, j0:j1] - BEAM_TOP) < 0.003
  labels, n = ndimage.label(mask)
  if n == 0:
    return [], mask
  sizes = ndimage.sum(mask, labels, range(1, n + 1))
  v_shape = labels == (np.argmax(sizes) + 1)
  ii, jj = np.nonzero(v_shape)
  pts = np.column_stack(_cell_xy(ii, jj, x0, y0, res))
  apex_y = pts[np.argmin(pts[:, 0]), 1]  # the V points toward -x
  boxes = []
  for side in (pts[:, 1] < apex_y, pts[:, 1] >= apex_y):
    q = pts[side]
    c = q.mean(0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    u = vt[0]
    t = (q - c) @ u
    length = t.max() - t.min() + res
    width = len(q) * res * res / length  # area / length; the raster edge is jagged
    mid = c + u * (t.max() + t.min()) / 2
    yaw = np.arctan2(u[1], u[0])
    zc = (BEAM_TOP + BOX_BOTTOM) / 2
    boxes.append(Box((mid[0], mid[1], zc), (length / 2, width / 2, (BEAM_TOP - BOX_BOTTOM) / 2),
                     (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)), "beam"))
  return boxes, v_shape


def flat_boxes(h: np.ndarray, x0: float, y0: float, res: float, skip: np.ndarray) -> list[Box]:
  """Cut every non-floor cell not in `skip` into axis-aligned boxes of equal height."""
  q = np.where(h < TALL, np.round(h / 0.001), np.ceil(h / 0.05) * 50).astype(np.int64)
  todo = (h > FLOOR_TOL) & ~skip
  nx, ny = h.shape
  boxes = []
  for i, j in zip(*np.nonzero(todo)):
    if not todo[i, j]:
      continue
    v = q[i, j]
    j2 = j
    while j2 + 1 < ny and todo[i, j2 + 1] and q[i, j2 + 1] == v:
      j2 += 1
    i2 = i
    while i2 + 1 < nx and np.all(todo[i2 + 1, j : j2 + 1]) and np.all(q[i2 + 1, j : j2 + 1] == v):
      i2 += 1
    todo[i : i2 + 1, j : j2 + 1] = False
    top = v * 0.001
    xa, xb = x0 + (i - 0.5) * res, x0 + (i2 + 0.5) * res
    ya, yb = y0 + (j - 0.5) * res, y0 + (j2 + 0.5) * res
    boxes.append(Box(((xa + xb) / 2, (ya + yb) / 2, (top + BOX_BOTTOM) / 2),
                     ((xb - xa) / 2, (yb - ya) / 2, (top - BOX_BOTTOM) / 2)))
  return boxes


def grid_to_boxes(h: np.ndarray, x0: float, y0: float, res: float) -> list[Box]:
  ramps, ramp_cells = find_ramps(h, x0, y0, res)
  beams, beam_cells = find_beams(h, x0, y0, res)
  return ramps + beams + flat_boxes(h, x0, y0, res, ramp_cells | beam_cells)


def material_for(box: Box) -> str:
  if box.kind in ("ramp", "beam"):
    return box.kind
  top = box.pos[2] + box.size[2]
  return "low" if top < 0.3 else ("mid" if top < TALL else "tall")


def boxes_xml(boxes: list[Box]) -> tuple[str, str]:
  """MJCF lines for the <asset> section (ramp meshes) and the <worldbody> geoms."""
  assets, geoms = [], []
  for k, b in enumerate(boxes):
    name = f"{b.kind}{k}"
    pos = f'pos="{b.pos[0]:.4f} {b.pos[1]:.4f} {b.pos[2]:.4f}"'
    if b.vertices is not None:
      vertex = " ".join(f"{v:.5f}" for corner in b.vertices for v in corner)
      assets.append(f'    <mesh name="{name}" vertex="{vertex}"/>')
      geoms.append(f'    <geom name="{name}" type="mesh" mesh="{name}" {pos} material="{material_for(b)}"/>')
      continue
    quat = "" if b.quat == (1.0, 0.0, 0.0, 0.0) else (' quat="%.6f %.6f %.6f %.6f"' % b.quat)
    geoms.append(
      f'    <geom name="{name}" type="box" {pos} '
      f'size="{b.size[0]:.4f} {b.size[1]:.4f} {b.size[2]:.4f}"{quat} material="{material_for(b)}"/>'
    )
  return "\n".join(assets), "\n".join(geoms)
