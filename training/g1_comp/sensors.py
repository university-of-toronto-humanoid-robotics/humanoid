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

from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

SCAN_SIZE = (1.6, 1.0)  # m, length (along the heading) x width
SCAN_RESOLUTION = 0.05  # m -> 33 x 21 = 693 points
SCAN_FORWARD = 0.3  # m; grid center ahead of the pelvis: 0.5 m behind to 1.1 m ahead
SCAN_LIFT = 2.0  # m; rays start this far above the pelvis
TERRAIN_GROUPS = (0,)  # terrain geoms; robot visuals are group 2, robot collisions group 3
FOOT_SENSOR_NAMES = ("left_foot_ground", "right_foot_ground")


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
      pattern=RaisedGridPatternCfg(size=(0.2, 0.08), resolution=0.04, lift=0.5),
      max_distance=3.0,
      exclude_parent_body=True,
      include_geom_groups=TERRAIN_GROUPS,
      debug_vis=False,
    )
    for side, name in zip(("left", "right"), FOOT_SENSOR_NAMES, strict=True)
  )
