"""The 3D worlds for the new tasks (Flat, Ramp, Stairs, Open, Rough, VeryRough, Competition).

Lane worlds (Ramp, Stairs, Open) are built from the competition's own parts, with
the dimensions measured from the organizers' CAD (sim/competition/README.md):

  ramps   0.61 m square tiles sloped 15 deg, in a "hill" (up, down, up, down) or
          each tile facing its own way (the course's pinwheel half)
  stairs  risers of 20.3 cm, treads of 24 cm, up to a landing and down again
  beams   10 cm tall, 9 cm wide step-over beams: straight, diagonal or a V
  sill    the 7.7 cm door sill, with door posts when the lane is narrow

A lane is a walled corridor along x, like the course. Each tile (8 m of lane)
holds a random sequence of parts with flat gaps between them. Tiles of one column
join end to end, so walking along a lane gives an endless random course that gets
harder row by row (the terrain curriculum moves robots between rows).

Everything scales with the tile's difficulty (0 = nearly flat, 1 = a little
harder than the competition), so robots that can only walk still get somewhere.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  BoxRandomGridTerrainCfg,
  FlatPatchSamplingCfg,
  HfDiscreteObstaclesTerrainCfg,
  HfPerlinNoiseTerrainCfg,
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  HfWaveTerrainCfg,
  SubTerrainCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane

# --- Competition dimensions (m), from the CAD -----------------------------------
RAMP_TILE = 0.61  # side of one ramp tile
RAMP_ANGLE = 15.0  # degrees
RAMP_BASE = 0.04  # the ramp tiles' low edge sits this far above the deck
STAIR_RISER = 0.203
STAIR_TREAD = 0.24
STAIR_RISERS = 5
STAIR_WIDTH = 0.77  # between the side panels
LANDING_LENGTH = 1.22
BEAM_HEIGHT = 0.102
BEAM_WIDTH = 0.09
BEAM_PLATE = 0.026  # the step-over section's floor is raised this much
SILL_HEIGHT = 0.077
SILL_LENGTH = 0.11
DOOR_OPENING = 0.93
LANE_WIDTH = 2.44
LANE_WALL_HEIGHT = 0.81

# --- How hard the hardest row is (a little past the competition) -----------------
MAX_RAMP_ANGLE = 17.0
MAX_RISER = 0.215
MAX_BEAM = 0.115
MAX_SILL = 0.085
EASY_TREAD = 0.34  # treads on the easiest row; they shrink to STAIR_TREAD

TILE_SIZE = (8.0, 3.2)  # m: length of lane in one tile, width of the tile (lane + walls)
MAZE_GATE_X = ((2.00, 2.25), (5.25, 5.85))
MAZE_LEAF_SIDES = (1, None)  # an open door leaf, then a plain offset wall opening
MAZE_LEAF_REACH_X = 0.75
MAZE_LEAF_SWING_Y = 0.58
MAZE_LEAF_THICKNESS = 0.04


def maze_lane_width(d):
  return LANE_WIDTH


def maze_gate_width(d):
  return 1.43 - 0.53 * d  # 1.40 m at level 0, 0.93 m at level 9


def maze_gate_offset(d):
  return 0.10 + 0.40 * d  # alternating offset makes each doorway require steering


def maze_leaf_endpoints(x0, center, opening, side):
  """Hinge at one jamb; the open leaf projects toward the approaching robot."""
  hinge_y = center - side * opening / 2
  return (x0, hinge_y), (x0 - MAZE_LEAF_REACH_X, hinge_y + side * MAZE_LEAF_SWING_Y)


_BOTTOM = -0.02  # parts start just below the floor
_mesh_ids = itertools.count()

# Kept dark: the lights are bright, and pale tops all render white in the head
# camera (play.py --fpv), which hides the step edges from the operator.
_COLORS = {
  "floor": (0.42, 0.40, 0.37),
  "wall": (0.24, 0.26, 0.31),
  "ramp": (0.55, 0.40, 0.22),
  "stairs": (0.36, 0.40, 0.47),
  "beam": (0.62, 0.28, 0.10),
  "sill": (0.50, 0.34, 0.16),
}
# Ramp tiles are shaded by the way they face, stair treads alternately, so that
# neighbouring surfaces can be told apart in the camera image.
_RAMP_SHADE = {"+x": 1.0, "-x": 0.75, "+y": 1.15, "-y": 0.88}


def _color(kind: str, shade: float = 1.0) -> tuple[float, float, float, float]:
  r, g, b = (min(c * shade, 1.0) for c in _COLORS[kind])
  return (r, g, b, 1.0)


def _lerp(a: float, b: float, t: float) -> float:
  return a + (b - a) * min(max(t, 0.0), 1.0)


@dataclass
class _Part:
  """One planned part: how much lane it takes, how tall it is, and how to build it at x."""

  length: float
  top: float
  build: Callable[[float], None]
  landing: tuple[float, float] | None = None  # stairs: landing (from, to) relative to the part's start


class _Tile:
  """Collects the geoms of one tile."""

  def __init__(self, spec: mujoco.MjSpec, yc: float, w: float, beam_group: int = 0):
    self.spec, self.body = spec, spec.body("terrain")
    # Keep wall geoms in a named subtree so a contact sensor can distinguish
    # them from the floor, stairs, ramps, and step-over beams.
    self.wall_body = spec.body("lane_walls") or self.body.add_body(name="lane_walls")
    self.yc, self.w = yc, w
    self.beam_group = beam_group
    self.geoms: list[TerrainGeometry] = []
    self.meshes: dict[tuple, str] = {}

  def across(self) -> tuple[float, float]:
    """The lane's full width, reaching a little into the walls."""
    return self.yc - self.w / 2 - 0.05, self.yc + self.w / 2 + 0.05

  def box(self, kind: str, x0: float, x1: float, y0: float, y1: float, top: float, shade: float = 1.0) -> None:
    geom = (self.wall_body if kind == "wall" else self.body).add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=((x1 - x0) / 2, (y1 - y0) / 2, (top - _BOTTOM) / 2),
      pos=((x0 + x1) / 2, (y0 + y1) / 2, (top + _BOTTOM) / 2),
    )
    self.geoms.append(TerrainGeometry(geom=geom, color=_color(kind, shade)))

  def panel(self, a: tuple[float, float], b: tuple[float, float], top: float) -> None:
    """A thin, upright collision panel, like the course's swung-open door leaf."""
    yaw = math.atan2(b[1] - a[1], b[0] - a[0])
    geom = self.wall_body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(math.dist(a, b) / 2, MAZE_LEAF_THICKNESS / 2, (top - _BOTTOM) / 2),
      pos=((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (top + _BOTTOM) / 2),
      quat=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
    )
    self.geoms.append(TerrainGeometry(geom=geom, color=_color("wall", 1.35)))

  def beam(self, a: tuple[float, float], b: tuple[float, float], top: float) -> None:
    """A step-over beam from a to b (a rotated box)."""
    yaw = math.atan2(b[1] - a[1], b[0] - a[0])
    geom = self.body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(math.hypot(b[0] - a[0], b[1] - a[1]) / 2 + 0.06, BEAM_WIDTH / 2, (top - _BOTTOM) / 2),
      pos=((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (top + _BOTTOM) / 2),
      quat=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
      group=self.beam_group,
    )
    self.geoms.append(TerrainGeometry(geom=geom, color=_color("beam")))

  def wedge(self, cx: float, cy: float, ax: float, ay: float, direction: str, low: float, high: float) -> None:
    """A ramp tile: vertical sides, sloped top rising toward `direction` (+x, -x, +y, -y)."""
    key = (round(ax, 4), round(ay, 4), direction, round(low, 4), round(high, 4))
    if key not in self.meshes:
      corners = [(-ax, -ay), (ax, -ay), (ax, ay), (-ax, ay)]
      axis, sign = (0 if direction[1] == "x" else 1), (1.0 if direction[0] == "+" else -1.0)
      half = ax if axis == 0 else ay
      tops = [low + (high - low) * (sign * c[axis] + half) / (2 * half) for c in corners]
      verts = [v for (x, y) in corners for v in (x, y, _BOTTOM)]
      verts += [v for (x, y), z in zip(corners, tops, strict=True) for v in (x, y, z)]
      self.meshes[key] = self.spec.add_mesh(name=f"lane_wedge_{next(_mesh_ids)}", uservert=verts).name
    geom = self.body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=self.meshes[key], pos=(cx, cy, 0.0))
    self.geoms.append(TerrainGeometry(geom=geom, color=_color("ramp", _RAMP_SHADE[direction])))


def _plan_ramps(t: _Tile, rng: np.random.Generator, d: float) -> _Part:
  low = RAMP_BASE * d
  high = low + RAMP_TILE * math.tan(math.radians(_lerp(4.0, MAX_RAMP_ANGLE, d)))
  n_x = int(rng.integers(2, 5))  # the course has 4 tiles along the lane
  n_y = math.ceil(t.w / RAMP_TILE)
  half = RAMP_TILE / 2
  hill = rng.random() < 0.5
  # Each tile's facing; None = flat at mid height (the course has a few of those).
  faces = [[None if rng.random() < 0.15 else str(rng.choice(["+x", "-x", "+y", "-y"])) for _ in range(n_y)]
           for _ in range(n_x)]

  def build(x: float) -> None:
    for i in range(n_x):
      cx = x + (i + 0.5) * RAMP_TILE
      if hill:  # ridges across the lane: one wide wedge per row, up then down
        t.wedge(cx, t.yc, half, n_y * half, "+x" if i % 2 == 0 else "-x", low, high)
        continue
      for j in range(n_y):
        cy = t.yc + (j + 0.5 - n_y / 2) * RAMP_TILE
        if faces[i][j] is None:
          t.box("ramp", cx - half, cx + half, cy - half, cy + half, (low + high) / 2, shade=0.6)
        else:
          t.wedge(cx, cy, half, half, faces[i][j], low, high)

  return _Part(n_x * RAMP_TILE, 0.0, build)


def _plan_stairs(t: _Tile, rng: np.random.Generator, d: float) -> _Part:
  n = 2 + round((STAIR_RISERS - 2) * d)
  riser = _lerp(0.06, MAX_RISER, d)
  tread = max(STAIR_TREAD, _lerp(EASY_TREAD, STAIR_TREAD, d) + rng.uniform(-0.01, 0.01))
  landing = rng.uniform(1.0, 1.4)  # the course's is 1.22
  length = 2 * (n - 1) * tread + landing

  def build(x: float) -> None:
    for i in range(1, n + 1):  # slab i is the i-th step on both sides; the top one is the landing
      t.box("stairs", x + (i - 1) * tread, x + length - (i - 1) * tread, *t.across(), i * riser,
            shade=1.0 if i % 2 else 0.7)

  return _Part(length, n * riser, build, landing=((n - 1) * tread, length - (n - 1) * tread))


def _plan_beams(t: _Tile, rng: np.random.Generator, d: float) -> _Part:
  plate = BEAM_PLATE if d > 0.3 else 0.0
  top = plate + _lerp(0.03, MAX_BEAM, d)
  kind = str(rng.choice(["straight", "diagonal", "vee"]))
  if kind == "diagonal" and t.w > 1.5:
    kind = "vee"  # one diagonal across a wide lane would fill the whole tile
  reach = {"straight": 0.0, "diagonal": t.w, "vee": t.w / 2}[kind]  # 45 degrees (the course's: about 47)
  flip = rng.random() < 0.5  # the V may open either way along the lane
  length = reach + 0.6  # room to step before and after

  def build(x: float) -> None:
    if plate:
      t.box("sill", x, x + length, *t.across(), plate)
    x0, y0, y1 = x + 0.3, t.yc - t.w / 2, t.yc + t.w / 2
    ends = {
      "straight": [((x0, y0), (x0, y1))],
      "diagonal": [((x0, y0), (x0 + reach, y1))],
      "vee": [((x0, t.yc), (x0 + reach, y0)), ((x0, t.yc), (x0 + reach, y1))],
    }[kind]
    for a, b in ends:
      if flip:
        a, b = (2 * x0 + reach - a[0], a[1]), (2 * x0 + reach - b[0], b[1])
      t.beam(a, b, top)

  return _Part(length, 0.0, build)


def _plan_sill(t: _Tile, rng: np.random.Generator, d: float) -> _Part:
  del rng
  height = _lerp(0.02, MAX_SILL, d)

  def build(x: float) -> None:
    t.box("sill", x, x + SILL_LENGTH, *t.across(), height)
    # Door posts narrow the lane to the door's opening. Only in lanes not much wider
    # than the door: in a wide lane a robot walking straight would just hit the wall.
    if DOOR_OPENING < t.w <= 1.5:
      lo, hi = t.across()
      t.box("wall", x, x + SILL_LENGTH, lo, t.yc - DOOR_OPENING / 2, LANE_WALL_HEIGHT)
      t.box("wall", x, x + SILL_LENGTH, t.yc + DOOR_OPENING / 2, hi, LANE_WALL_HEIGHT)

  return _Part(SILL_LENGTH, 0.0, build)


_PLANNERS = {"ramps": _plan_ramps, "stairs": _plan_stairs, "beams": _plan_beams, "sill": _plan_sill}


@dataclass(kw_only=True)
class LaneTerrainCfg(SubTerrainCfg):
  """One tile of lane: a random sequence of competition parts between two walls."""

  features: dict[str, float] = field(
    default_factory=lambda: {"ramps": 1.0, "stairs": 1.0, "beams": 0.5, "sill": 0.5}
  )
  """Which parts may appear (ramps, stairs, beams, sill), and how often relative to each other."""
  lane_width_range: tuple[float, float] = (0.9, LANE_WIDTH)
  """Lane width between the walls. Harder tiles lean toward the narrow end."""
  beam_group: int = 0
  """Geometry group used by beam-only foot sensors in the height-map task."""
  num_spawns: int = 8
  """Flat spots per tile where robots may start (between the parts, and on stair landings)."""
  size: tuple[float, float] = TILE_SIZE

  def __post_init__(self) -> None:
    self.flat_patch_sampling = {"spawn": FlatPatchSamplingCfg(num_patches=self.num_spawns, patch_radius=0.25)}

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    d = float(difficulty)
    length, width = self.size
    lo, hi = self.lane_width_range
    w = lo + (hi - lo) * rng.uniform() * (1.0 - 0.6 * d)  # harder tiles: narrower lanes
    tile = _Tile(spec, width / 2, w, beam_group=self.beam_group)
    tile.geoms.append(
      TerrainGeometry(geom=make_plane(tile.body, self.size, 0.0, center_zero=False)[0], color=_color("floor"))
    )
    names = list(self.features)
    weights = np.array([self.features[k] for k in names], dtype=np.float64)
    weights /= weights.sum()

    # Parts along the lane with flat gaps between them, until the tile is full.
    # `flats` and `landings` are (x from, x to, floor height): places to spawn.
    x = rng.uniform(0.8, 1.2)
    gap_start = 0.0
    flats: list[tuple[float, float, float]] = []
    landings: list[tuple[float, float, float]] = []
    tallest = 0.0
    while True:
      part = _PLANNERS[names[int(rng.choice(len(names), p=weights))]](tile, rng, d)
      if x + part.length + 0.8 > length:  # no room left before the end of the tile
        break
      flats.append((gap_start, x, 0.0))
      part.build(x)
      if part.landing is not None:
        landings.append((x + part.landing[0] + 0.35, x + part.landing[1] - 0.35, part.top))
      tallest = max(tallest, part.top)
      gap_start = x + part.length
      x = gap_start + rng.uniform(0.8, 1.4)
    flats.append((gap_start, length, 0.0))

    # Walls: solid from the lane's edge to the tile's edge, so tiles of different
    # widths meet without gaps. Taller next to stairs (the course has side panels).
    wall = max(LANE_WALL_HEIGHT, tallest + 0.45)
    tile.box("wall", 0.0, length, 0.0, tile.yc - w / 2, wall)
    tile.box("wall", 0.0, length, tile.yc + w / 2, width, wall)

    # Spawn points: on the flat gaps, and on stair landings (to practise going down).
    spots = [(a + 0.3, b - 0.3, z) if b - a > 0.7 else ((a + b) / 2, (a + b) / 2, z) for a, b, z in flats]
    spots += [s for s in landings if s[1] > s[0]]
    side = max(w / 2 - 0.3, 0.0)
    picks = rng.integers(0, len(spots), self.num_spawns)
    spawns = np.array([
      [rng.uniform(spots[k][0], spots[k][1]), tile.yc + rng.uniform(-side, side), spots[k][2]] for k in picks
    ])
    origin = np.array([flats[0][1] / 2, tile.yc, 0.0])
    return TerrainOutput(origin=origin, geometries=tile.geoms, flat_patches={"spawn": spawns})


def _lanes(sub_terrains: dict[str, SubTerrainCfg], num_cols: int) -> TerrainGeneratorCfg:
  # Even rows and odd columns: the world origin (where the simulator sizes its
  # contact buffers, before the first reset) is then on a flat lane floor.
  return TerrainGeneratorCfg(
    size=TILE_SIZE, border_width=10.0, num_rows=10, num_cols=num_cols, add_lights=True, sub_terrains=sub_terrains
  )


def ramp_world_cfg() -> TerrainGeneratorCfg:
  """Lanes full of the course's ramp tiles, up to 17 degrees (course: 15)."""
  return _lanes({"ramps": LaneTerrainCfg(features={"ramps": 1.0}, lane_width_range=(2 * RAMP_TILE, LANE_WIDTH))}, 7)


def stairs_world_cfg() -> TerrainGeneratorCfg:
  """Narrow lanes of stairs up to a landing and down again, up to 21.5 cm risers
  on 24 cm treads (course: 20.3 on 24), between side walls as close as 0.77 m."""
  return _lanes({"stairs": LaneTerrainCfg(features={"stairs": 1.0}, lane_width_range=(STAIR_WIDTH, 1.5))}, 7)


def open_world_cfg(beam_group: int = 0) -> TerrainGeneratorCfg:
  """Every part of the course in random order: ramps, stairs, step-over beams and
  the door sill, in lanes from 0.9 m to the course's 2.44 m wide. Some lanes are
  all stairs or all ramps, so neither is ever rare."""
  return _lanes(
    {
      "mixed": LaneTerrainCfg(proportion=0.5, beam_group=beam_group),
      "stairs": LaneTerrainCfg(proportion=0.25, features={"stairs": 1.0}, lane_width_range=(STAIR_WIDTH, 1.5)),
      "ramps": LaneTerrainCfg(proportion=0.25, features={"ramps": 1.0}, lane_width_range=(2 * RAMP_TILE, LANE_WIDTH)),
    },
    11,
  )


@dataclass(kw_only=True)
class RplLaneTerrainCfg(SubTerrainCfg):
  """One forward traversal of a single competition feature per curriculum tile."""

  feature: str
  beam_group: int = 0
  size: tuple[float, float] = TILE_SIZE
  # Filled in generator call order (column, then row), including its random difficulty jitter.
  stair_specs: list[tuple[int, float, float, float]] = field(default_factory=list, init=False, repr=False)

  def __post_init__(self) -> None:
    if self.feature not in {"ascend", "descend", "ramps", "obstacles"}:
      raise ValueError(f"Unknown RPL terrain feature: {self.feature}")
    self.flat_patch_sampling = {"spawn": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.25)}

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    d = float(difficulty)
    length, width = self.size
    if self.feature in {"ascend", "descend"}:
      lane_width = _lerp(1.5, STAIR_WIDTH, d)
    else:
      lane_width = _lerp(LANE_WIDTH, 1.2, d)
    tile = _Tile(spec, width / 2, lane_width, beam_group=self.beam_group)
    tile.geoms.append(
      TerrainGeometry(geom=make_plane(tile.body, self.size, 0.0, center_zero=False)[0], color=_color("floor"))
    )
    start_height = 0.0
    highest = 0.0
    if self.feature in {"ascend", "descend"}:
      n = 2 + round((STAIR_RISERS - 2) * d)
      riser = _lerp(0.06, 0.225, d)  # row 9 always exceeds the course's 20.3 cm
      tread = _lerp(EASY_TREAD, STAIR_TREAD, d)
      if self.feature == "descend":
        self.stair_specs.append((n, riser, tread, lane_width))
      highest = n * riser
      x0 = 2.0
      if self.feature == "ascend":
        # Nested slabs form upward risers and a long landing before the goal.
        for i in range(1, n + 1):
          tile.box("stairs", x0 + (i - 1) * tread, length, *tile.across(), i * riser,
                   shade=1.0 if i % 2 else 0.7)
      else:
        start_height = highest
        tile.box("stairs", 0.0, x0, *tile.across(), highest)
        for i in range(1, n):
          tile.box("stairs", x0 + (i - 1) * tread, x0 + i * tread,
                   *tile.across(), (n - i) * riser,
                   shade=1.0 if i % 2 else 0.7)
    elif self.feature == "ramps":
      _plan_ramps(tile, rng, d).build(2.0)
      highest = RAMP_BASE * d + RAMP_TILE * math.tan(math.radians(_lerp(4.0, MAX_RAMP_ANGLE, d)))
    else:
      _plan_beams(tile, rng, d).build(2.0)
      _plan_sill(tile, rng, d).build(5.0)
      highest = max(BEAM_PLATE + MAX_BEAM, MAX_SILL)

    wall_height = max(LANE_WALL_HEIGHT, highest + 0.45)
    tile.box("wall", 0.0, length, 0.0, tile.yc - lane_width / 2, wall_height)
    tile.box("wall", 0.0, length, tile.yc + lane_width / 2, width, wall_height)
    side = max(0.0, lane_width / 2 - 0.35)
    spawns = np.array([
      [rng.uniform(0.35, 0.55), tile.yc + rng.uniform(-side, side), start_height]
      for _ in range(8)
    ])
    return TerrainOutput(
      origin=np.array([0.45, tile.yc, start_height]),
      geometries=tile.geoms,
      flat_patches={"spawn": spawns},
    )


def rpl_world_cfg(feature: str, beam_group: int = 0) -> TerrainGeneratorCfg:
  """Ten curriculum rows, from easy features to slightly beyond course dimensions."""
  return _lanes({feature: RplLaneTerrainCfg(feature=feature, beam_group=beam_group)}, 7)


@dataclass(kw_only=True)
class MazeTerrainCfg(SubTerrainCfg):
  """One corridor tile with offset doorways, a ramp and step-over beams."""

  beam_group: int = 1
  preview_start_only: bool = False
  size: tuple[float, float] = TILE_SIZE

  def __post_init__(self) -> None:
    self.flat_patch_sampling = {"spawn": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.25)}

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    # Quantize the corridor shape to the curriculum row. This lets the command
    # generator and wall reward reproduce it without extra actor observations.
    level = min(9, int(difficulty * 10))
    d = (level + 0.5) / 10
    length, width = self.size
    lane_width = maze_lane_width(d)
    opening = maze_gate_width(d)
    offset = maze_gate_offset(d)
    tile = _Tile(spec, width / 2, lane_width, beam_group=self.beam_group)
    tile.geoms.append(
      TerrainGeometry(geom=make_plane(tile.body, self.size, 0.0, center_zero=False)[0], color=_color("floor"))
    )
    lo, hi = tile.yc - lane_width / 2, tile.yc + lane_width / 2
    tile.box("wall", 0, length, 0, lo, LANE_WALL_HEIGHT)
    tile.box("wall", 0, length, hi, width, LANE_WALL_HEIGHT)
    # The first passage resembles the course's open door; the second is a
    # plain offset wall opening so every tile practices both situations.
    for (x0, x1), center, side in zip(
      MAZE_GATE_X, (tile.yc + offset, tile.yc - offset), MAZE_LEAF_SIDES, strict=True
    ):
      lower, upper = center - opening / 2, center + opening / 2
      tile.box("wall", x0, x1, 0, lower, 1.5, shade=1.35)
      tile.box("wall", x0, x1, upper, width, 1.5, shade=1.35)
      tile.box("sill", x0, x1, lower, upper, _lerp(0.02, MAX_SILL, difficulty))
      if side is not None:
        tile.panel(*maze_leaf_endpoints(x0, center, opening, side), top=1.5)

    # A short two-tile hill and two step-over beams occupy the open gaps.
    low = RAMP_BASE * difficulty
    high = low + RAMP_TILE * math.tan(math.radians(_lerp(4.0, MAX_RAMP_ANGLE, difficulty)))
    for i, direction in enumerate(("+x", "-x")):
      tile.wedge(3.55 + (i + 0.5) * RAMP_TILE, tile.yc,
                 RAMP_TILE / 2, lane_width / 2, direction, low, high)
    beam_top = _lerp(0.03, MAX_BEAM, difficulty)
    for x in (3.15, 6.85):
      tile.beam((x, lo), (x, hi), beam_top)

    # Flat start spots away from the constrictions and obstacles.
    start_locations = ((0.4, 0.0),) if self.preview_start_only else ((0.4, 0.0), (3.15, offset))
    spots = np.array([
      [start_locations[k][0] + rng.uniform(-0.08, 0.08),
       tile.yc + start_locations[k][1] + rng.uniform(-0.08, 0.08), 0.0]
      for k in rng.integers(0, len(start_locations), size=8)
    ])
    return TerrainOutput(origin=np.array([0.4, tile.yc, 0.0]), geometries=tile.geoms,
                         flat_patches={"spawn": spots})


def maze_world_cfg(preview_start_only: bool = False) -> TerrainGeneratorCfg:
  """Curriculum of winding corridors; ramps and beams, with no stairs."""
  return _lanes({"maze": MazeTerrainCfg(preview_start_only=preview_start_only)}, 11)


def rough_world_cfg() -> TerrainGeneratorCfg:
  """Rough ground as it really is: bumpy floors, gentle slopes, curbs. Open
  fields (no lanes); the hardest row has 4 cm bumps, 14 degree slopes, 8 cm curbs."""
  # Order and column count matter: the middle column (where the simulator sizes its
  # contact buffers, with one robot at the world origin) must be built from boxes
  # with a flat rim, here the curbs. On a bumpy heightfield that robot alone makes
  # more contacts than the 48 each robot gets in training.
  return TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=13, add_lights=True,
    sub_terrains={
      "flat": BoxFlatTerrainCfg(proportion=0.15),
      "bumps": HfRandomUniformTerrainCfg(
        proportion=0.25, noise_range=(0.005, 0.04), noise_step=0.005, border_width=0.25
      ),
      "curbs_down": BoxPyramidStairsTerrainCfg(
        proportion=0.1, step_height_range=(0.0, 0.08), step_width=0.5, platform_width=3.0, border_width=1.0
      ),
      "curbs_up": BoxInvertedPyramidStairsTerrainCfg(
        proportion=0.1, step_height_range=(0.0, 0.08), step_width=0.5, platform_width=3.0, border_width=1.0
      ),
      "slope": HfPyramidSlopedTerrainCfg(
        proportion=0.1, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25
      ),
      "slope_inv": HfPyramidSlopedTerrainCfg(
        proportion=0.1, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25, inverted=True
      ),
      "rolling": HfPerlinNoiseTerrainCfg(proportion=0.2, height_range=(0.0, 0.25), border_width=0.5),
    },
  )


def very_rough_world_cfg() -> TerrainGeneratorCfg:
  """Much rougher than anything expected: 12 cm bumps, 27 degree slopes, 20 cm
  waves, 20 cm steps, scattered blocks. For robustness, not for the competition."""
  stairs = dict(step_height_range=(0.0, 0.2), step_width=0.3, platform_width=3.0, border_width=1.0)
  # As in rough_world_cfg: the middle column (the stairs) is built from boxes.
  return TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=15, add_lights=True,
    sub_terrains={
      "flat": BoxFlatTerrainCfg(proportion=0.1),
      "bumps": HfRandomUniformTerrainCfg(
        proportion=0.15, noise_range=(0.02, 0.12), noise_step=0.02, border_width=0.25
      ),
      "rolling": HfPerlinNoiseTerrainCfg(proportion=0.1, height_range=(0.0, 0.7), border_width=0.5),
      "slope": HfPyramidSlopedTerrainCfg(
        proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
      ),
      "stairs_down": BoxPyramidStairsTerrainCfg(proportion=0.1, **stairs),
      "stairs_up": BoxInvertedPyramidStairsTerrainCfg(proportion=0.1, **stairs),
      "slope_inv": HfPyramidSlopedTerrainCfg(
        proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25, inverted=True
      ),
      "wave": HfWaveTerrainCfg(proportion=0.1, amplitude_range=(0.0, 0.2), num_waves=4, border_width=0.25),
      "blocks": HfDiscreteObstaclesTerrainCfg(
        proportion=0.075, obstacle_width_range=(0.3, 1.0), obstacle_height_range=(0.05, 0.25),
        num_obstacles=40, border_width=0.25,
      ),
      "grid": BoxRandomGridTerrainCfg(
        proportion=0.075, grid_width=0.45, grid_height_range=(0.0, 0.2), platform_width=1.5
      ),
    },
  )
