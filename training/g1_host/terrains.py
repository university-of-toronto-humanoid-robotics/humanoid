"""The worlds the robot lies in: HoST's four (ground, platform, wall, slope) and ours
(steeper ramps for the course's 15 deg ones, the course's stairs).

Every tile is 8 x 8 m with its feature in a fixed place and orientation, and a
"spawn" list of points where a robot's pelvis is dropped (events.reset_posture adds
the posture, a little jitter and the drop height). As in HoST, the feature faces
-x: the platform's free edge, the wall and the uphill side of slopes and stairs.
A robot lying on its back with yaw 0 has its head toward -x.

  platform  box 4 x 7 m, 22-46 cm tall (HoST's code; the paper's text says 20-92 cm).
            Pelvis 0.45 m from the +x edge: trunk on it, legs over the edge.
  wall      incline behind a sitting robot, 14-84 deg, 1 m tall.
  slope     incline 1.1-14.6 deg (HoST); "ramp" variant up to 17 deg.
  stairs    5 risers of 10-21.5 cm on 24 cm treads up to a 1.22 m landing (course:
            20.3 cm), 3 m wide; robots dropped onto the flight.

Rows are difficulty (the generator's curriculum mode: one terrain type per column),
robots are spread over all rows, and nothing moves them between rows: HoST also
samples difficulty at random and has no terrain curriculum.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

from mjlab.terrains import FlatPatchSamplingCfg, SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane

from . import params as P

NUM_SPAWNS = 8
_mesh_ids = itertools.count()
_FLOOR = (0.42, 0.40, 0.37, 1.0)
_FEATURE = (0.36, 0.42, 0.52, 1.0)


def _lerp(a: float, b: float, t: float) -> float:
  return a + (b - a) * min(max(t, 0.0), 1.0)


def _pick(values: tuple[float, ...], d: float) -> float:
  return values[min(int(d * len(values)), len(values) - 1)]


class _Tile:
  def __init__(self, spec: mujoco.MjSpec):
    self.spec, self.body = spec, spec.body("terrain")
    self.geoms: list[TerrainGeometry] = [
      TerrainGeometry(geom=make_plane(self.body, (P.TILE, P.TILE), 0.0, center_zero=False)[0], color=_FLOOR)
    ]

  def box(self, x0: float, x1: float, y0: float, y1: float, top: float) -> None:
    geom = self.body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=((x1 - x0) / 2, (y1 - y0) / 2, (top + 0.02) / 2),
      pos=((x0 + x1) / 2, (y0 + y1) / 2, (top - 0.02) / 2),
    )
    self.geoms.append(TerrainGeometry(geom=geom, color=_FEATURE))

  def prism(self, xz: list[tuple[float, float]], y0: float, y1: float) -> None:
    """A convex prism: the polygon `xz` (in the x-z plane) extruded from y0 to y1."""
    verts = [v for (x, z) in xz for y in (y0, y1) for v in (x, y, z)]
    name = f"host_prism_{next(_mesh_ids)}"
    self.spec.add_mesh(name=name, uservert=verts)
    geom = self.body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)
    self.geoms.append(TerrainGeometry(geom=geom, color=_FEATURE))


def _spawns(x: float, y_range: tuple[float, float], z: float) -> np.ndarray:
  ys = np.linspace(*y_range, NUM_SPAWNS)
  return np.array([[x, y, z] for y in ys])


@dataclass(kw_only=True)
class _HostTerrainCfg(SubTerrainCfg):
  size: tuple[float, float] = (P.TILE, P.TILE)
  flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = field(
    default_factory=lambda: {"spawn": FlatPatchSamplingCfg(num_patches=NUM_SPAWNS)}
  )


@dataclass(kw_only=True)
class FlatTerrainCfg(_HostTerrainCfg):
  def function(self, difficulty, spec, rng) -> TerrainOutput:
    t = _Tile(spec)
    spawns = _spawns(4.0, (2.5, 5.5), 0.0)
    return TerrainOutput(origin=spawns[0], geometries=t.geoms, flat_patches={"spawn": spawns})


@dataclass(kw_only=True)
class PlatformTerrainCfg(_HostTerrainCfg):
  height_range: tuple[float, float] = P.PLATFORM_HEIGHT

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    t = _Tile(spec)
    h = _lerp(*self.height_range, difficulty)
    edge = 5.0
    t.box(1.0, edge, 0.5, 7.5, h)
    spawns = _spawns(edge - 0.45, (1.5, 6.5), h)
    return TerrainOutput(origin=spawns[0], geometries=t.geoms, flat_patches={"spawn": spawns})


@dataclass(kw_only=True)
class WallTerrainCfg(_HostTerrainCfg):
  slopes: tuple[float, ...] = P.WALL_SLOPES

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    t = _Tile(spec)
    foot, height = 3.0, P.WALL_HEIGHT
    top = foot - height / _pick(self.slopes, difficulty)
    t.prism([(foot, 0.0), (top, height), (top - 0.3, height), (top - 0.3, 0.0)], 0.5, 7.5)
    spawns = _spawns(foot + 0.12, (1.5, 6.5), 0.0)
    return TerrainOutput(origin=spawns[0], geometries=t.geoms, flat_patches={"spawn": spawns})


@dataclass(kw_only=True)
class SlopeTerrainCfg(_HostTerrainCfg):
  slopes: tuple[float, ...] = P.SLOPE_SLOPES
  """Rise/run values to pick from by difficulty (HoST's list)."""
  max_deg: float | None = None
  """If set, the angle grows linearly from 1 deg to this instead (our ramps)."""

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    t = _Tile(spec)
    grade = math.tan(_lerp(1.0, self.max_deg, difficulty) * P.DEG) if self.max_deg else _pick(self.slopes, difficulty)
    low_x, high_x = 7.0, 1.0
    t.prism([(low_x, 0.0), (high_x, (low_x - high_x) * grade), (high_x, -0.02), (low_x, -0.02)], 0.5, 7.5)
    x = 4.5
    spawns = _spawns(x, (1.5, 6.5), (low_x - x) * grade)
    return TerrainOutput(origin=spawns[0], geometries=t.geoms, flat_patches={"spawn": spawns})


@dataclass(kw_only=True)
class StairsTerrainCfg(_HostTerrainCfg):
  riser_range: tuple[float, float] = P.STAIR_RISER

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    t = _Tile(spec)
    riser, tread, n = _lerp(*self.riser_range, difficulty), P.STAIR_TREAD, P.STAIR_RISERS
    bottom = 6.0  # first riser; the flight climbs toward -x to the landing
    landing_end = bottom - (n - 1) * tread - P.STAIR_LANDING
    y0, y1 = 4.0 - P.STAIR_WIDTH / 2, 4.0 + P.STAIR_WIDTH / 2
    for i in range(1, n + 1):  # slab i: from its riser back to the landing's end
      t.box(landing_end, bottom - (i - 1) * tread, y0, y1, i * riser)
    x = bottom - 2.5 * tread  # middle of the flight
    step = math.floor((bottom - x) / tread) + 1
    spawns = _spawns(x, (3.4, 4.6), step * riser)
    return TerrainOutput(origin=spawns[0], geometries=t.geoms, flat_patches={"spawn": spawns})


# --- Worlds ----------------------------------------------------------------------


def world(sub_terrains: dict[str, SubTerrainCfg], num_rows: int = 8, cols_per_type: int = 2) -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    size=(P.TILE, P.TILE),
    border_width=4.0,
    num_rows=num_rows,
    num_cols=cols_per_type * len(sub_terrains),
    curriculum=True,  # one terrain type per column, difficulty by row (no level changes: see above)
    sub_terrains=sub_terrains,
    add_lights=True,
  )


def terrain_kinds(cfg: TerrainGeneratorCfg) -> list[str]:
  """Sub-terrain name of each column, the way mjlab assigns them in curriculum mode."""
  names = list(cfg.sub_terrains)
  props = np.array([cfg.sub_terrains[n].proportion for n in names], dtype=np.float64)
  cum = np.cumsum(props / props.sum())
  return [names[int(np.where(j / cfg.num_cols + 0.001 < cum)[0][0])] for j in range(cfg.num_cols)]


def kind_per_env(env, kinds: tuple[str, ...]) -> torch.Tensor:
  """Index into `kinds` of each robot's terrain (cached). Plane worlds: kinds[0]."""
  cache = getattr(env, "_host_kind_cache", None)
  if cache is not None and cache[0] == kinds:
    return cache[1]
  terrain = env.scene.terrain
  gen = env.cfg.scene.terrain.terrain_generator if env.cfg.scene.terrain is not None else None
  if gen is None or getattr(terrain, "terrain_types", None) is None:
    idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  else:
    cols = terrain_kinds(gen)
    lookup = torch.tensor([kinds.index(c) if c in kinds else 0 for c in cols], device=env.device)
    idx = lookup[terrain.terrain_types.long()]
  env._host_kind_cache = (kinds, idx)
  return idx
