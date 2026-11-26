"""Batched approximation of the ROS /height_scan path.

The training terrain ray sensor provides exact candidate surface samples. This
module gates them by head-LiDAR visibility, retains observations in a moving
world-aligned map, then applies height_scan/scan.py's fallback order. It does not
simulate individual Livox rays, FAST-LIO drift, or the CuPy map's variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ObjRef, RayCastSensorCfg
from mjlab.utils.lab_api.math import quat_apply

from ..sensors import RaisedGridPatternCfg, TERRAIN_GROUPS


@dataclass(frozen=True)
class StudentMapCfg:
  resolution: float = 0.05
  cells: int = 100  # 5 x 5 m, matching g1_elevation_mapping.yaml
  map_hz: float = 3.0  # filtered GridMap publication rate
  self_filter_radius: float = 0.6
  max_down_angle_deg: float = 52.0  # upside-down Mid-360 ground limit
  max_range: float = 5.0
  near_radius: float = 0.5
  sole_depth: float = 0.035
  inpaint_passes: int = 16
  probe_resolution: float = 0.10  # candidate hits over the entire 5 m map
  observation_delay_s: float = 0.0  # use visible hits on the first map update
  hit_keep_probability: float = 1.0  # effective map coverage after visibility filtering
  height_bias_std_m: float = 0.065  # spatially correlated error on each map update
  height_noise_std_m: float = 0.015  # independent error on retained height hits
  pose_noise_xy_std_m: float = 0.020  # shared XY misregistration of one update


def visibility(head: torch.Tensor, hits: torch.Tensor, hit_valid: torch.Tensor,
               cfg: StudentMapCfg, intermediate: torch.Tensor | None = None) -> torch.Tensor:
  """Visibility of candidate ground hits; ``intermediate`` is [B,N,K] terrain z.

  Intermediate heights are sampled along each head-to-hit ray. A higher
  surface blocks a farther hit. Missing intermediate heights do not block it.
  """
  delta = hits - head[:, None, :]
  radius = torch.linalg.vector_norm(delta[..., :2], dim=-1)
  down_limit = torch.deg2rad(torch.tensor(cfg.max_down_angle_deg, device=hits.device))
  visible = (hit_valid & (radius >= cfg.self_filter_radius)
             & (torch.linalg.vector_norm(delta, dim=-1) <= cfg.max_range)
             & (delta[..., 2] >= -radius * torch.tan(down_limit)))
  if intermediate is not None:
    fractions = torch.linspace(0.125, 0.875, intermediate.shape[-1], device=hits.device)
    line_z = head[:, None, 2, None] + delta[..., 2, None] * fractions
    visible &= ~torch.any(torch.isfinite(intermediate) & (intermediate > line_z + 0.03), dim=-1)
  return visible


def fill_unknown(observed: torch.Tensor, valid: torch.Tensor, passes: int) -> tuple[torch.Tensor, torch.Tensor]:
  """GPU-batched spatial fill on a [B,H,W] scan; preserve measured cells."""
  value, known = torch.nan_to_num(observed).unsqueeze(1), valid.unsqueeze(1)
  for _ in range(passes):
    count = F.avg_pool2d(known.float(), 3, 1, 1) * 9
    mean = F.avg_pool2d(value * known, 3, 1, 1) * 9 / count.clamp_min(1)
    new = ~known & (count > 0)
    value = torch.where(new, mean, value)
    known = known | new
  return value[:, 0], known[:, 0]


def select_ground(observed: torch.Tensor, valid: torch.Tensor, filled: torch.Tensor,
                  fill_valid: torch.Tensor, near: torch.Tensor, foot_height: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
  """ROS precedence: observed, near feet, distant inpaint, feet fallback.

  Source: 0=observed, 1=foot estimate, 2=inpaint.
  """
  from_feet = ~valid & (near | ~fill_valid)
  ground = torch.where(valid, observed, torch.where(from_feet, foot_height, filled))
  source = torch.where(valid, 0, torch.where(from_feet, 1, 2))
  return ground, source


class StudentHeightMap:
  """Observation term with per-environment moving map and reset support."""

  def __init__(self, cfg: ObservationTermCfg, env):
    cfg: StudentMapCfg = cfg.params["map_cfg"]
    if cfg.cells * cfg.resolution != 5.0:
      raise ValueError("Student map must cover 5 m at the configured resolution")
    if not 0 < cfg.hit_keep_probability <= 1:
      raise ValueError("hit_keep_probability must be in (0, 1]")
    if min(cfg.observation_delay_s, cfg.height_bias_std_m,
           cfg.height_noise_std_m, cfg.pose_noise_xy_std_m) < 0:
      raise ValueError("Map noise standard deviations must be nonnegative")
    self.cfg = cfg
    self.n = env.num_envs
    self.device = env.device
    self.height = torch.zeros((self.n, 1, cfg.cells, cfg.cells), device=env.device)
    self.valid = torch.zeros_like(self.height, dtype=torch.bool)
    self.first_hit_time = torch.full_like(self.height, -1e6)
    self.filled_height = torch.zeros_like(self.height)
    self.fill_valid = torch.zeros_like(self.valid)
    self.center = torch.zeros((self.n, 2), device=env.device)
    self.previous_height = torch.zeros_like(self.height)
    self.previous_valid = torch.zeros_like(self.valid)
    self.previous_filled_height = torch.zeros_like(self.filled_height)
    self.previous_fill_valid = torch.zeros_like(self.fill_valid)
    self.previous_center = torch.zeros_like(self.center)
    self.fresh = torch.ones(self.n, dtype=torch.bool, device=env.device)
    self.last_step = -1
    self.last_map_time = torch.full((self.n,), -1e6, device=env.device)
    self.last_source = None
    self.last_observed = None
    self.map_age_s = None
    self._rows, self._cols = 21, 33
    self._probe_side = round(cfg.cells * cfg.resolution / cfg.probe_resolution) + 1
    across = torch.linspace(-0.5, 0.5, self._rows, device=env.device)
    along = torch.linspace(-0.5, 1.1, self._cols, device=env.device)
    y, x = torch.meshgrid(across, along, indexing="ij")
    self.offsets = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
    cells = torch.arange(cfg.cells, device=env.device) - (cfg.cells - 1) / 2
    gy, gx = torch.meshgrid(cells, cells, indexing="ij")
    self.map_offsets = torch.stack((gx, gy), dim=-1) * cfg.resolution

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.height[ids] = 0
    self.valid[ids] = False
    self.first_hit_time[ids] = -1e6
    self.filled_height[ids] = 0
    self.fill_valid[ids] = False
    self.fresh[ids] = True
    self.last_map_time[ids] = -1e6
    self.previous_height[ids] = 0
    self.previous_valid[ids] = False
    self.previous_filled_height[ids] = 0
    self.previous_fill_valid[ids] = False

  def _sample(self, data: torch.Tensor, xy: torch.Tensor, center: torch.Tensor,
              mode: str = "nearest") -> torch.Tensor:
    grid = (xy - center[:, None, :]) / (self.cfg.resolution * (self.cfg.cells - 1) / 2)
    return F.grid_sample(data, grid.view(self.n, 1, -1, 2), mode=mode,
                         padding_mode="zeros", align_corners=True).view(self.n, -1)

  def _move_map(self, new_center: torch.Tensor) -> None:
    points = new_center[:, None, None, :] + self.map_offsets[None]
    grid = (points - self.center[:, None, None, :]) / (self.cfg.resolution * (self.cfg.cells - 1) / 2)
    self.height = F.grid_sample(self.height, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    self.valid = F.grid_sample(self.valid.float(), grid, mode="nearest", padding_mode="zeros",
                               align_corners=True) > 0.5
    moved_first = F.grid_sample(self.first_hit_time, grid, mode="nearest",
                                padding_mode="zeros", align_corners=True)
    in_old_map = (grid.abs() <= 1).all(dim=-1).unsqueeze(1)
    self.first_hit_time = torch.where(in_old_map, moved_first,
                                      torch.full_like(moved_first, -1e6))
    self.center = new_center

  def _occlusion_samples(self, head: torch.Tensor, hits: torch.Tensor,
                         scan_height: torch.Tensor, pelvis: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    fractions = torch.linspace(0.125, 0.875, 6, device=self.device)
    xy = head[:, None, None, :2] + (hits[..., :2] - head[:, None, :2]).unsqueeze(2) * fractions[None, None, :, None]
    relative = xy - pelvis[:, None, None, :2]
    c, s = yaw.cos()[:, None, None], yaw.sin()[:, None, None]
    along = c * relative[..., 0] + s * relative[..., 1]
    across = -s * relative[..., 0] + c * relative[..., 1]
    grid = torch.stack((along / (self.cfg.cells * self.cfg.resolution / 2),
                        across / (self.cfg.cells * self.cfg.resolution / 2)), dim=-1)
    source = scan_height.view(self.n, 1, self._probe_side, self._probe_side)
    sample = F.grid_sample(source, grid.view(self.n, 1, -1, 2), mode="nearest",
                           padding_mode="zeros", align_corners=True)
    sample = sample.view(self.n, hits.shape[1], -1)
    inside = (grid.abs() <= 1).all(dim=-1)
    return torch.where(inside, sample, torch.full_like(sample, float("nan")))

  def __call__(self, env, sensor_name: str, map_cfg: StudentMapCfg,
               return_history: bool = False) -> torch.Tensor:
    del map_cfg
    robot = env.scene["robot"]
    names = list(robot.body_names)
    pelvis = robot.data.body_link_pose_w[:, names.index("pelvis")]
    torso = robot.data.body_link_pose_w[:, names.index("torso_link")]
    head_offset = torch.tensor([0.0002835, 0.00003, 0.428434], device=self.device).expand(self.n, -1)
    head = torso[:, :3] + quat_apply(torso[:, 3:7], head_offset)
    q = pelvis[:, 3:7]
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1 - 2 * (q[:, 2].square() + q[:, 3].square()))
    sensor = env.scene[sensor_name]
    hits = sensor.data.hit_pos_w
    hit_valid = sensor.data.distances >= 0
    now = env.common_step_counter * env.step_dt
    if self.last_step != env.common_step_counter or self.fresh.any():
      update = (now - self.last_map_time >= 1 / self.cfg.map_hz) | self.fresh
      if update.any():
        older = update & ~self.fresh
        self.previous_height[older] = self.height[older]
        self.previous_valid[older] = self.valid[older]
        self.previous_filled_height[older] = self.filled_height[older]
        self.previous_fill_valid[older] = self.fill_valid[older]
        self.previous_center[older] = self.center[older]
        initial = update & self.fresh
        new_center = torch.round(pelvis[:, :2] / self.cfg.resolution) * self.cfg.resolution
        self._move_map(new_center)
        # Initial maps must not inherit observations from an earlier episode.
        self.height[self.fresh] = 0
        self.valid[self.fresh] = False
        self.first_hit_time[self.fresh] = -1e6
        true_z = torch.where(hit_valid, hits[..., 2], torch.full_like(hits[..., 2], float("nan")))
        intermediate = self._occlusion_samples(head, hits, true_z, pelvis, yaw)
        seen = visibility(head, hits, hit_valid, self.cfg, intermediate) & update[:, None]
        if self.cfg.hit_keep_probability < 1:
          seen &= torch.rand(seen.shape, device=self.device) < self.cfg.hit_keep_probability
        hit_xy = hits[..., :2]
        if self.cfg.pose_noise_xy_std_m:
          hit_xy = hit_xy + torch.randn((self.n, 1, 2), device=self.device) * self.cfg.pose_noise_xy_std_m
        index = torch.round((hit_xy - self.center[:, None, :] ) / self.cfg.resolution
                            + (self.cfg.cells - 1) / 2).long()
        inside = (index >= 0).all(dim=-1) & (index < self.cfg.cells).all(dim=-1)
        b, n = torch.where(seen & inside)
        if len(b):
          flat = index[b, n, 1] * self.cfg.cells + index[b, n, 0]
          measured_z = hits[..., 2]
          if self.cfg.height_bias_std_m:
            # Smooth patches across the probe footprint, not white noise on
            # the final 693 policy observations.
            bias = F.interpolate(
              torch.randn((self.n, 1, 9, 9), device=self.device),
              size=(self._probe_side, self._probe_side), mode="bilinear",
              align_corners=True).reshape(self.n, -1)
            measured_z = measured_z + bias * self.cfg.height_bias_std_m
          if self.cfg.height_noise_std_m:
            measured_z = measured_z + torch.randn(measured_z.shape, device=self.device) * self.cfg.height_noise_std_m
          self.height.view(self.n, -1)[b, flat] = measured_z[b, n]
          first = self.first_hit_time.view(self.n, -1)
          previous = first[b, flat]
          first[b, flat] = torch.where(previous < -1e5, now, previous)
        if self.cfg.observation_delay_s:
          self.valid |= ((self.first_hit_time > -1e5) &
                         (now - self.first_hit_time >= self.cfg.observation_delay_s))
        elif len(b):
          self.valid.view(self.n, -1)[b, flat] = True
        filled, fill_valid = fill_unknown(self.height[:, 0], self.valid[:, 0],
                                          self.cfg.inpaint_passes)
        self.filled_height = filled.unsqueeze(1)
        self.fill_valid = fill_valid.unsqueeze(1)
        self.previous_height[initial] = self.height[initial]
        self.previous_valid[initial] = self.valid[initial]
        self.previous_filled_height[initial] = self.filled_height[initial]
        self.previous_fill_valid[initial] = self.fill_valid[initial]
        self.previous_center[initial] = self.center[initial]
        self.last_map_time[update] = now
        self.fresh[update] = False
      self.last_step = env.common_step_counter

    # The ROS scan is pelvis-yaw aligned and uses the latest pose even when the
    # filtered map has not been republished.
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    xy = torch.stack((pelvis[:, None, 0] + c * self.offsets[:, 0] - s * self.offsets[:, 1],
                      pelvis[:, None, 1] + s * self.offsets[:, 0] + c * self.offsets[:, 1]), dim=-1)
    feet = robot.data.body_link_pos_w[:, [names.index("left_ankle_roll_link"),
                                          names.index("right_ankle_roll_link")], 2]
    foot_height = feet.min(dim=1).values[:, None] - self.cfg.sole_depth
    near = torch.linalg.vector_norm(self.offsets, dim=-1) < self.cfg.near_radius - 1e-6
    def project(height, valid_map, filled_map, fill_valid_map, center):
      observed = self._sample(height, xy, center)
      valid = self._sample(valid_map.float(), xy, center) > 0.5
      filled = self._sample(filled_map, xy, center)
      fill_valid = self._sample(fill_valid_map.float(), xy, center) > 0.5
      return select_ground(observed, valid, filled, fill_valid, near[None], foot_height)

    ground, source = project(self.height, self.valid, self.filled_height, self.fill_valid, self.center)
    self.last_source = source
    self.last_observed = source == 0
    self.map_age_s = now - self.last_map_time
    current = (pelvis[:, 2, None] - ground).clamp(-1.0, 2.0)
    if not return_history:
      return current
    previous_ground, _ = project(self.previous_height, self.previous_valid,
                                 self.previous_filled_height, self.previous_fill_valid,
                                 self.previous_center)
    previous = (pelvis[:, 2, None] - previous_ground).clamp(-1.0, 2.0)
    # The ROS scan is flattened lateral-major (21x33). The CNN expects
    # longitudinal rows and lateral columns (33x21).
    current = current.reshape(self.n, 21, 33).transpose(1, 2).reshape(self.n, -1)
    previous = previous.reshape(self.n, 21, 33).transpose(1, 2).reshape(self.n, -1)
    return torch.cat((current, previous), dim=-1)


def attach_student_map(cfg, map_cfg: StudentMapCfg | None = None):
  """Replace only a copied RPL config's actor scan; keep the expert config intact."""
  import copy

  result = copy.deepcopy(cfg)
  map_cfg = map_cfg or StudentMapCfg()
  result.scene.sensors = (result.scene.sensors or ()) + (student_probe_cfg(map_cfg),)
  if "height_scan" not in result.observations["actor"].terms:
    raise ValueError("Expected an actor height_scan term")
  result.observations["actor"].terms["height_scan"] = ObservationTermCfg(
    func=StudentHeightMap, params={"sensor_name": "student_map_probe", "map_cfg": map_cfg},
    clip=(-1.0, 2.0),
  )
  return result


def student_probe_cfg(map_cfg: StudentMapCfg) -> RayCastSensorCfg:
  return RayCastSensorCfg(
    name="student_map_probe",
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    ray_alignment="yaw",
    pattern=RaisedGridPatternCfg(
      size=(map_cfg.cells * map_cfg.resolution,) * 2,
      resolution=map_cfg.probe_resolution, lift=2.0,
    ),
    max_distance=6.0,
    exclude_parent_body=True,
    include_geom_groups=TERRAIN_GROUPS,
    debug_vis=False,
  )
