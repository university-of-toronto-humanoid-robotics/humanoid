"""Terrain sensing for the G1 competition policy.

Two fixes over the stock mjlab height scan:

  - Rays start high above the pelvis. mjlab starts each ray at the frame's own
    height, so anything taller than the pelvis (the upper stairs, the 1 m
    landing, walls) has rays starting inside it. Those rays miss and read as
    deep holes. The scan value is still "pelvis height above the cell".
  - Rays only hit the terrain (geom group 0), never the robot's own meshes,
    so the sim scan matches a height map built from the CAD.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp

from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensor, RayCastSensorCfg

SCAN_SIZE = (1.6, 1.0)  # m, length (along the heading) x width
SCAN_RESOLUTION = 0.05  # m -> 33 x 21 = 693 points
SCAN_FORWARD = 0.3  # m; grid center ahead of the pelvis: 0.5 m behind to 1.1 m ahead
SCAN_LIFT = 2.0  # m; rays start this far above the pelvis
TERRAIN_GROUPS = (0,)  # terrain geoms; robot visuals are group 2, robot collisions group 3
FOOT_SENSOR_NAMES = ("left_foot_ground", "right_foot_ground")
BEAM_SENSOR_NAMES = ("left_foot_beam", "right_foot_beam")
SOLE_SENSOR_NAMES = ("left_sole_support", "right_sole_support")
FOOT_ENV_CONTACT_NAME = "feet_environment_contact"
FOOT_PATCH_SIZE = (0.2, 0.08)  # m, along x across; covers the sole (foot site is at its center)
FOOT_PATCH_RESOLUTION = 0.04  # m -> 6 x 3 rays


def grid_shape(size: tuple[float, float], resolution: float) -> tuple[int, int]:
  """(rows, columns) of a grid pattern's rays = (points across, points along).
  Rays are ordered row by row (mjlab's meshgrid with indexing="xy")."""
  return round(size[1] / resolution) + 1, round(size[0] / resolution) + 1


@dataclass
class RaisedGridPatternCfg(GridPatternCfg):
  """mjlab's grid of downward rays, shifted forward and started higher up."""

  forward: float = 0.0
  """Shift of the grid center along the frame's heading, in meters."""
  lift: float = 0.0
  """Ray start height above the frame, in meters."""

  def generate_rays(self, mj_model, device):
    offsets, directions = super().generate_rays(mj_model, device)
    offsets[:, 0] += self.forward
    offsets[:, 2] += self.lift
    return offsets, directions


def terrain_scan_cfg() -> RayCastSensorCfg:
  """The policy's height scan, attached to the pelvis and turning with its yaw."""
  return RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    ray_alignment="yaw",
    pattern=RaisedGridPatternCfg(
      size=SCAN_SIZE, resolution=SCAN_RESOLUTION, forward=SCAN_FORWARD, lift=SCAN_LIFT
    ),
    max_distance=6.0,
    exclude_parent_body=True,
    include_geom_groups=TERRAIN_GROUPS,
    debug_vis=True,
  )


def foot_ground_sensor_cfgs() -> tuple[RayCastSensorCfg, ...]:
  """A small patch of rays under each foot (6 x 3 over the sole), to measure
  how high the foot is above the ground right below it."""
  return tuple(
    RayCastSensorCfg(
      name=name,
      frame=ObjRef(type="site", name=f"{side}_foot", entity="robot"),
      ray_alignment="yaw",
      pattern=RaisedGridPatternCfg(size=FOOT_PATCH_SIZE, resolution=FOOT_PATCH_RESOLUTION, lift=0.5),
      max_distance=3.0,
      exclude_parent_body=True,
      include_geom_groups=TERRAIN_GROUPS,
      debug_vis=False,
    )
    for side, name in zip(("left", "right"), FOOT_SENSOR_NAMES, strict=True)
  )


def foot_beam_sensor_cfgs() -> tuple[RayCastSensorCfg, ...]:
  """Ray patches over the soles that see only step-over beams (geom group 1)."""
  return tuple(
    RayCastSensorCfg(
      name=name,
      frame=ObjRef(type="site", name=f"{side}_foot", entity="robot"),
      ray_alignment="yaw",
      pattern=RaisedGridPatternCfg(size=FOOT_PATCH_SIZE, resolution=FOOT_PATCH_RESOLUTION, lift=0.5),
      max_distance=3.0,
      exclude_parent_body=True,
      include_geom_groups=(1,),
      debug_vis=False,
    )
    for side, name in zip(("left", "right"), BEAM_SENSOR_NAMES, strict=True)
  )


@dataclass
class SoleGridPatternCfg(GridPatternCfg):
  """Exactly 10 longitudinal by 5 lateral samples inside the G1 sole capsules."""

  length: float = 0.18
  width: float = 0.064
  lift: float = 0.45

  def generate_rays(self, mj_model, device):
    del mj_model
    x = torch.linspace(-self.length / 2, self.length / 2, 10, device=device)
    y = torch.linspace(-self.width / 2, self.width / 2, 5, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    offsets = torch.stack((xx.flatten(), yy.flatten(), torch.full_like(xx.flatten(), self.lift)), dim=-1)
    directions = torch.zeros_like(offsets)
    directions[:, 2] = -1.0
    return offsets, directions


class SoleRayCastSensor(RayCastSensor):
  """Follow the full sole pose for sample positions, but cast vertically in world."""

  sole_points_w: torch.Tensor | None = None

  def prepare_rays(self) -> None:
    super().prepare_rays()
    assert self._cached_world_rays is not None
    assert self._cached_world_origins is not None
    assert self._cached_frame_mat is not None
    assert self._ray_vec is not None
    pattern = self.cfg.pattern
    assert isinstance(pattern, SoleGridPatternCfg)
    world_down = torch.zeros_like(self._cached_world_rays)
    world_down[..., 2] = -1.0
    wp.to_torch(self._ray_vec).view_as(world_down).copy_(world_down)
    self._cached_world_rays = world_down
    self.sole_points_w = self._cached_world_origins - self._cached_frame_mat[:, :, 2].unsqueeze(1) * pattern.lift


@dataclass
class SoleRayCastSensorCfg(RayCastSensorCfg):
  def build(self) -> SoleRayCastSensor:
    return SoleRayCastSensor(self)


def rpl_foot_safety_sensor_cfgs() -> tuple[ContactSensorCfg | SoleRayCastSensorCfg, ...]:
  """One world-frame net force per foot, plus ground-truth 50-point sole rays."""
  contacts = ContactSensorCfg(
    name=FOOT_ENV_CONTACT_NAME,
    primary=ContactMatch(
      mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"
    ),
    secondary=ContactMatch(mode="subtree", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1,
  )
  rays = tuple(
    SoleRayCastSensorCfg(
      name=name,
      frame=ObjRef(type="site", name=f"{side}_foot", entity="robot"),
      ray_alignment="base",
      pattern=SoleGridPatternCfg(),
      max_distance=3.0,
      exclude_parent_body=True,
      include_geom_groups=(0, 1),
      debug_vis=False,
    )
    for side, name in zip(("left", "right"), SOLE_SENSOR_NAMES, strict=True)
  )
  return (contacts, *rays)
