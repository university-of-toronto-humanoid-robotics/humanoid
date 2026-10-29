"""Terrains for the G1 competition tasks.

  comp_rough_terrains_cfg(): mjlab's rough-terrain mix, with stairs as tall as
    the competition's (up to 23 cm risers, 24 and 30 cm treads).
  course_terrains_cfg(): the competition course itself, built from the boxes in
    sim/competition/course.xml, with spawn points spread along the whole path.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  FlatPatchSamplingCfg,
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  HfWaveTerrainCfg,
  SubTerrainCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane

REPO_ROOT = Path(__file__).resolve().parents[2]
COURSE_DIR = REPO_ROOT / "sim" / "competition"

STAIR_HEIGHTS = (0.05, 0.23)  # m, easiest to hardest curriculum row (course risers: 0.203)


def comp_rough_terrains_cfg() -> TerrainGeneratorCfg:
  """Rough terrain with competition-height stairs. "up" spawns at the bottom of a
  stepped pit, "down" on top of a stepped pyramid."""
  stairs = dict(step_height_range=STAIR_HEIGHTS, platform_width=3.0, border_width=1.0)
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    add_lights=True,
    sub_terrains={
      "flat": BoxFlatTerrainCfg(proportion=0.1),
      "stairs_up_24cm": BoxInvertedPyramidStairsTerrainCfg(proportion=0.2, step_width=0.24, **stairs),
      "stairs_down_24cm": BoxPyramidStairsTerrainCfg(proportion=0.2, step_width=0.24, **stairs),
      "stairs_up_30cm": BoxInvertedPyramidStairsTerrainCfg(proportion=0.1, step_width=0.30, **stairs),
      "stairs_down_30cm": BoxPyramidStairsTerrainCfg(proportion=0.1, step_width=0.30, **stairs),
      "slope": HfPyramidSlopedTerrainCfg(
        proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
      ),
      "slope_inv": HfPyramidSlopedTerrainCfg(
        proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
        inverted=True,
      ),
      "random_rough": HfRandomUniformTerrainCfg(
        proportion=0.1, noise_range=(0.02, 0.08), noise_step=0.02, border_width=0.25
      ),
      "wave": HfWaveTerrainCfg(
        proportion=0.05, amplitude_range=(0.0, 0.15), num_waves=4, border_width=0.25
      ),
    },
  )


# Course frame (see sim/competition/README.md): x along the course, deck at z = 0.
COURSE_START = (11.0, 4.21)  # middle of the start area
COURSE_CROP = ((1.10, 12.30), (2.85, 5.60))  # keep boxes inside the lanes
COURSE_MARGIN = 0.5  # m of flat floor around the crop
LANES = ((1.25, 12.15), (3.0, 5.42))  # where spawns may go
MAX_STEP = 0.21  # m; tallest step a spawn may need to reach it from the start (riser 0.203)
_mesh_ids = itertools.count()


def course_spawn_points(heightmap: Path, n: int, rng: np.random.Generator) -> np.ndarray:
  """Flat spots reachable from the start area, spread evenly along x. [n, 3], course frame."""
  d = np.load(heightmap)
  h = d["heights"].astype(np.float64)
  x0, y0, res = float(d["x0"]), float(d["y0"]), float(d["res"])
  nx, ny = h.shape

  # Reachable: connected to the start by height changes no taller than a stair riser
  # (keeps spawns off the counter, the crate, and wall tops).
  idx = np.arange(nx * ny).reshape(nx, ny)
  ok_x = np.abs(np.diff(h, axis=0)) <= MAX_STEP
  ok_y = np.abs(np.diff(h, axis=1)) <= MAX_STEP
  src = np.concatenate([idx[:-1][ok_x], idx[:, :-1][ok_y]])
  dst = np.concatenate([idx[1:][ok_x], idx[:, 1:][ok_y]])
  graph = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(nx * ny, nx * ny))
  _, labels = connected_components(graph, directed=False)
  si = int(round((COURSE_START[0] - x0) / res))
  sj = int(round((COURSE_START[1] - y0) / res))
  reachable = (labels == labels[si * ny + sj]).reshape(nx, ny)

  # Flat under both feet (12 cm around), nothing over 25 cm taller within 20 cm.
  feet, body = 2 * round(0.12 / res) + 1, 2 * round(0.20 / res) + 1
  flat = ndimage.maximum_filter(h, feet) - ndimage.minimum_filter(h, feet) <= 0.01
  clear = ndimage.maximum_filter(h, body) - h <= 0.25
  x = x0 + np.arange(nx)[:, None] * res
  y = y0 + np.arange(ny)[None, :] * res
  inside = (x > LANES[0][0]) & (x < LANES[0][1]) & (y > LANES[1][0]) & (y < LANES[1][1])
  ii, jj = np.nonzero(reachable & flat & clear & inside)

  # The same number of spawns in every 25 cm of course length, so the short
  # stairs get as many as the big start area.
  bins = np.floor((x0 + ii * res) / 0.25).astype(np.int64)
  per_bin = int(np.ceil(n / len(np.unique(bins))))
  picks = np.concatenate([
    rng.choice(np.nonzero(bins == b)[0], size=per_bin, replace=bool((bins == b).sum() < per_bin))
    for b in np.unique(bins)
  ])
  picks = rng.permutation(picks)[:n]
  return np.column_stack([x0 + ii[picks] * res, y0 + jj[picks] * res, h[ii[picks], jj[picks]]])


def _color(top: float) -> tuple[float, float, float, float]:
  if top < 0.3:
    return (0.80, 0.62, 0.38, 1.0)
  return (0.55, 0.58, 0.62, 1.0) if top < 1.1 else (0.42, 0.44, 0.48, 1.0)


@dataclass(kw_only=True)
class CompetitionCourseTerrainCfg(SubTerrainCfg):
  """The competition course from the organizers' CAD (built by tools/cad/build_course.py)."""

  course_xml: Path = COURSE_DIR / "course.xml"
  heightmap: Path = COURSE_DIR / "course_heightmap.npz"
  num_spawns: int = 400
  size: tuple[float, float] = (
    COURSE_CROP[0][1] - COURSE_CROP[0][0] + 2 * COURSE_MARGIN,
    COURSE_CROP[1][1] - COURSE_CROP[1][0] + 2 * COURSE_MARGIN,
  )

  @staticmethod
  def offset() -> np.ndarray:
    """Course frame -> this sub-terrain's local frame (its corner is the origin)."""
    return np.array([COURSE_MARGIN - COURSE_CROP[0][0], COURSE_MARGIN - COURSE_CROP[1][0], 0.0])

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty  # The course has one difficulty.
    body = spec.body("terrain")
    off = self.offset()
    floor = make_plane(body, self.size, 0.0, center_zero=False)[0]
    geometries = [TerrainGeometry(geom=floor, color=(0.62, 0.56, 0.48, 1.0))]

    course = mujoco.MjSpec.from_file(str(self.course_xml))
    vertices = {m.name: list(m.uservert) for m in course.meshes}
    (xa, xb), (ya, yb) = COURSE_CROP
    for g in course.geoms:
      pos = np.array(g.pos, dtype=np.float64)
      if g.name == "floor" or not (xa <= pos[0] <= xb and ya <= pos[1] <= yb):
        continue
      if g.type == mujoco.mjtGeom.mjGEOM_MESH:  # ramp wedge
        verts = vertices[g.meshname]
        mesh = spec.add_mesh(name=f"course_wedge_{next(_mesh_ids)}", uservert=verts)
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, pos=pos + off)
        top = pos[2] + max(verts[2::3])
      else:
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(g.size), pos=pos + off,
                             quat=list(g.quat))
        top = pos[2] + g.size[2]
      geometries.append(TerrainGeometry(geom=geom, color=_color(top)))

    spawns = course_spawn_points(self.heightmap, self.num_spawns, rng) + off
    origin = np.array([*COURSE_START, 0.0]) + off
    return TerrainOutput(origin=origin, geometries=geometries, flat_patches={"spawn": spawns})


def course_terrains_cfg() -> TerrainGeneratorCfg:
  """A single copy of the course. All robots share it; mjwarp runs each robot in
  its own world, so they never bump into each other."""
  course = CompetitionCourseTerrainCfg(proportion=1.0)
  course.flat_patch_sampling = {"spawn": FlatPatchSamplingCfg(num_patches=course.num_spawns)}
  return TerrainGeneratorCfg(
    size=course.size,
    border_width=3.0,
    num_rows=1,
    num_cols=1,
    curriculum=False,
    add_lights=True,
    sub_terrains={"course": course},
  )
