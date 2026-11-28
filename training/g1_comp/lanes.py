"""Commands and curriculum for the lane worlds (Ramp, Stairs, Open, Competition).

A lane is a corridor along x. With mjlab's uniform heading commands most robots
would be told to walk into a wall, so here most are told to walk along the lane
(either way), with some sideways speed and heading error left in, as a teleoperator
would steer. The curriculum then measures how far each robot got from where it
started, not from the tile's corner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.lab_api.math import wrap_to_pi

from . import worlds

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class LaneVelocityCommandCfg(UniformVelocityCommandCfg):
  lane_prob: float = 0.0
  """Share of heading-controlled robots whose target heading is along the lane
  (+x or -x). 0 gives mjlab's uniform command (used on open ground)."""
  heading_jitter: float = 0.25
  """Lane headings are off by up to this much, in rad (an imperfect operator)."""
  keep_direction_prob: float = 0.7
  """Chance a new command keeps the robot's current lane direction, so most robots
  cross a whole part instead of turning back halfway."""

  def build(self, env: ManagerBasedRlEnv) -> LaneVelocityCommand:
    return LaneVelocityCommand(self, env)


class LaneVelocityCommand(UniformVelocityCommand):
  cfg: LaneVelocityCommandCfg

  def __init__(self, cfg: LaneVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.lane_direction = torch.zeros(self.num_envs, device=self.device)  # 0 (+x) or pi (-x)

  def _random_direction(self, n: int) -> torch.Tensor:
    return torch.where(torch.rand(n, device=self.device) < 0.5, 0.0, math.pi)

  def reset(self, env_ids):
    self.lane_direction[env_ids] = self._random_direction(len(env_ids))
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    if self.cfg.lane_prob <= 0.0:
      return
    n = len(env_ids)
    keep = torch.rand(n, device=self.device) < self.cfg.keep_direction_prob
    direction = torch.where(keep, self.lane_direction[env_ids], self._random_direction(n))
    self.lane_direction[env_ids] = direction
    jitter = (torch.rand(n, device=self.device) * 2 - 1) * self.cfg.heading_jitter
    along = torch.rand(n, device=self.device) < self.cfg.lane_prob
    self.heading_target[env_ids] = torch.where(
      along, wrap_to_pi(direction + jitter), self.heading_target[env_ids]
    )


@dataclass(kw_only=True)
class RplForwardCommandCfg(LaneVelocityCommandCfg):
  """Sample walking speeds while keeping the target heading along positive lane x."""

  def build(self, env: ManagerBasedRlEnv) -> RplForwardCommand:
    return RplForwardCommand(self, env)


class RplForwardCommand(LaneVelocityCommand):
  def _random_direction(self, n: int) -> torch.Tensor:
    return torch.zeros(n, device=self.device)


def rpl_feature_progress(env: ManagerBasedRlEnv, env_ids: torch.Tensor, goal_distance: float = 5.6) -> torch.Tensor:
  """Promote only after reaching the far side of the assigned feature."""
  terrain = env.scene.terrain
  assert terrain is not None
  start = getattr(env, "_lane_start", None)
  if start is not None:
    pos = env.scene["robot"].data.root_link_pos_w[env_ids, :2]
    forward = pos[:, 0] - start[env_ids, 0]
    move_up = forward >= goal_distance
    move_down = forward < goal_distance / 3
    terrain.update_env_origins(env_ids, move_up, move_down & ~move_up)
  return torch.mean(terrain.terrain_levels.float())


def rpl_feature_complete(env: ManagerBasedRlEnv, goal_distance: float = 5.6) -> torch.Tensor:
  """End an episode at the far-side landing, before it enters the next tile."""
  start = getattr(env, "_lane_start", None)
  if start is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  return env.scene["robot"].data.root_link_pos_w[:, 0] - start[:, 0] >= goal_distance


def rpl_outside_tile(env: ManagerBasedRlEnv, half_width: float) -> torch.Tensor:
  """Reset a robot before it walks backward into the neighbouring terrain tile."""
  start = getattr(env, "_lane_start", None)
  if start is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  terrain = env.scene.terrain
  assert terrain is not None and terrain.env_origins is not None
  pos = env.scene["robot"].data.root_link_pos_w[:, :2]
  return (pos[:, 0] - start[:, 0] < -0.30) | (
    (pos[:, 1] - terrain.env_origins[:, 1]).abs() > half_width
  )


@dataclass(kw_only=True)
class MazeVelocityCommandCfg(LaneVelocityCommandCfg):
  """Visit ordered position targets through the door and wall passages."""

  num_rows: int = 10
  tile_length: float = 8.0
  fixed_level: int | None = None
  single_tile_route: bool = False
  waypoint_radius: float = 0.18
  passage_radius: float = 0.08
  cruise_speed_range: tuple[float, float] = (0.40, 0.45)
  passage_speed_range: tuple[float, float] = (0.20, 0.25)
  clearance_radius: float = 0.25
  open_lookahead: float = 0.50
  passage_lookahead: float = 0.30
  speed_kp: float = 0.8
  speed_filter_seconds: float = 0.30

  def build(self, env: ManagerBasedRlEnv) -> MazeVelocityCommand:
    return MazeVelocityCommand(self, env)


MAZE_ROUTE_X = (0.4, 1.1, 1.45, 1.95, 2.125, 2.7, 3.3,
                4.2, 4.8, 5.2, 5.55, 6.35, 7.45)
MAZE_ROUTE_SWING_M = 0.44
# y = offset * first coefficient + MAZE_ROUTE_SWING_M * second coefficient.
# The approach points take the robot around the open leaf before the opening.
MAZE_ROUTE_OFFSET = (0, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, 0)
MAZE_ROUTE_SWING = (0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def maze_route_tile_points(level: int, origin_y: float) -> list[tuple[float, float]]:
  """The same local waypoint coordinates used by the command term, for previews."""
  offset = worlds.maze_gate_offset((level + 0.5) / 10)
  return [
    (x, origin_y + offset * factor + MAZE_ROUTE_SWING_M * swing)
    for x, factor, swing in zip(MAZE_ROUTE_X, MAZE_ROUTE_OFFSET, MAZE_ROUTE_SWING, strict=True)
  ]


class MazeVelocityCommand(LaneVelocityCommand):
  cfg: MazeVelocityCommandCfg

  def create_gui(self, name, server, get_env_idx) -> None:
    """This task follows its fixed route; manual velocity sliders would override it."""
    del name, server, get_env_idx

  def __init__(self, cfg: MazeVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.terrain = env.scene.terrain
    self.route_x = torch.tensor(MAZE_ROUTE_X, device=self.device)
    self.route_offset = torch.tensor(MAZE_ROUTE_OFFSET, device=self.device)
    self.route_swing = torch.tensor(MAZE_ROUTE_SWING, device=self.device)
    self.route_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.route_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self.clearance_samples = torch.linspace(0.0, 1.0, 17, device=self.device)[None, :, None]
    self.active_target_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.active_target_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.target_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.target_is_alignment = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.cruise_speed = torch.zeros(self.num_envs, device=self.device)
    self.passage_speed = torch.zeros(self.num_envs, device=self.device)
    self.filtered_vx = torch.zeros(self.num_envs, device=self.device)
    self.speed_filter_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def reset(self, env_ids):
    self.route_pending[env_ids] = True
    self.cruise_speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.cruise_speed_range
    )
    self.passage_speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.passage_speed_range
    )
    self.filtered_vx[env_ids] = 0.0
    self.speed_filter_valid[env_ids] = False
    return super().reset(env_ids)

  def _target_position(self, route_index: torch.Tensor | None = None) -> torch.Tensor:
    assert self.terrain is not None and self.terrain.env_origins is not None
    if route_index is None:
      route_index = self.route_index
    n = len(MAZE_ROUTE_X)
    row, point = route_index // n, route_index % n
    level = torch.full_like(row, self.cfg.fixed_level) if self.cfg.fixed_level is not None else row
    offset = worlds.maze_gate_offset((level.float() + 0.5) / 10)
    x = row * self.cfg.tile_length - self.cfg.num_rows * self.cfg.tile_length / 2 + self.route_x[point]
    y = self.terrain.env_origins[:, 1] + offset * self.route_offset[point] + MAZE_ROUTE_SWING_M * self.route_swing[point]
    return torch.stack((x, y), dim=1)

  def _last_route_index(self) -> torch.Tensor:
    """Keep the expert's route in its assigned curriculum tile."""
    n = len(MAZE_ROUTE_X)
    if self.cfg.single_tile_route:
      return (self.route_index // n + 1) * n - 1
    return torch.full_like(self.route_index, self.cfg.num_rows * n - 1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # The route and speeds sampled at reset stay fixed throughout an episode.
    # CommandManager resampling must not change them mid-route.
    self.lane_direction[env_ids] = 0.0
    self.is_heading_env[env_ids] = True
    self.is_standing_env[env_ids] = False
    self.vel_command_b[env_ids] = 0.0

  def _segment_clear(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """Check wall clearance, allowing a robot already too close to move away."""
    assert self.terrain is not None and self.terrain.env_origins is not None
    samples = start[:, None, :] + self.clearance_samples * (end - start)[:, None, :]
    grid_x = (samples[..., 0] + self.cfg.num_rows * self.cfg.tile_length / 2).clamp(
      0, self.cfg.num_rows * self.cfg.tile_length - 0.001
    )
    row = torch.floor(grid_x / self.cfg.tile_length).long()
    x = grid_x - row * self.cfg.tile_length
    y = samples[..., 1] - self.terrain.env_origins[:, None, 1]
    level = torch.full_like(row, self.cfg.fixed_level) if self.cfg.fixed_level is not None else row
    d = (level.float() + 0.5) / 10
    opening = worlds.maze_gate_width(d)
    offset = worlds.maze_gate_offset(d)
    clearance = worlds.maze_lane_width(d) / 2 - y.abs()
    for (x0, x1), center, side in zip(
      worlds.MAZE_GATE_X, (offset, -offset), worlds.MAZE_LEAF_SIDES, strict=True
    ):
      at_gate = (x >= x0 - self.cfg.clearance_radius) & (x <= x1 + self.cfg.clearance_radius)
      gate_clearance = torch.minimum(y - (center - opening / 2), center + opening / 2 - y)
      clearance = torch.minimum(clearance, torch.where(at_gate, gate_clearance, torch.inf))
      if side is not None:
        (ax, ay), (bx, by) = worlds.maze_leaf_endpoints(x0, center, opening, side)
        dx, dy = bx - ax, by - ay
        along = (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)).clamp(0, 1)
        leaf_distance = torch.sqrt((x - (ax + along * dx)).square() + (y - (ay + along * dy)).square())
        clearance = torch.minimum(clearance, leaf_distance - worlds.MAZE_LEAF_THICKNESS / 2)
    sufficient = (clearance >= self.cfg.clearance_radius).all(dim=1)
    start_clearance = clearance[:, 0]
    escaping = (
      (start_clearance < self.cfg.clearance_radius)
      & (clearance[:, 1:] >= start_clearance[:, None] - 0.005).all(dim=1)
      & (clearance[:, -1] > start_clearance + 0.02)
    )
    return sufficient | escaping

  def _visible_target(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose a visible forward route point, never one behind the robot."""
    target = self._target_position()
    active_index = self.route_index.clone()
    visible = (target[:, 0] >= pos[:, 0] - 0.05) & self._segment_clear(pos, target)
    last = self._last_route_index()
    for steps_forward in range(1, 5):
      future_index = torch.minimum(self.route_index + steps_forward, last)
      future = self._target_position(future_index)
      clear = (future[:, 0] > pos[:, 0] + 0.05) & self._segment_clear(pos, future)
      use = ~visible & clear
      target = torch.where(use[:, None], future, target)
      active_index = torch.where(use, future_index, active_index)
      visible |= clear
    return target, visible, active_index

  def _lookahead_target(self, pos: torch.Tensor, near_passage: torch.Tensor) -> torch.Tensor:
    """Project onto the current route segment, then look ahead to its endpoint.

    The next waypoint is a hard limit: looking through a door before reaching
    its approach point could steer straight through the open leaf.
    """
    end = self._target_position()
    start = self._target_position((self.route_index - 1).clamp(min=0))
    segment = end - start
    length = torch.linalg.vector_norm(segment, dim=1).clamp(min=1e-6)
    along = ((pos - start) * segment).sum(dim=1) / length
    along = along.clamp(min=0.0)
    lookahead = torch.where(
      near_passage,
      torch.full_like(length, self.cfg.passage_lookahead),
      torch.full_like(length, self.cfg.open_lookahead),
    )
    fraction = ((along + lookahead) / length).clamp(max=1.0)
    return start + fraction[:, None] * segment

  def _update_command(self) -> None:
    assert self.terrain is not None and self.terrain.env_origins is not None
    pos = self.robot.data.root_link_pos_w[:, :2]
    grid_x = (pos[:, 0] + self.cfg.num_rows * self.cfg.tile_length / 2).clamp(
      0, self.cfg.num_rows * self.cfg.tile_length - 0.001
    )
    row = torch.floor(grid_x / self.cfg.tile_length).long()
    local_x = grid_x - row * self.cfg.tile_length
    next_point = torch.searchsorted(self.route_x, local_x.contiguous(), right=True)
    start_index = (row * len(MAZE_ROUTE_X) + next_point).clamp(max=self.cfg.num_rows * len(MAZE_ROUTE_X) - 1)
    self.route_index = torch.where(self.route_pending, start_index, self.route_index)
    self.route_pending[:] = False
    # A point can advance only when the robot reaches it. Looking past a door
    # before reaching the approach point would command a path through the leaf.
    last = self._last_route_index()
    for _ in range(4):
      target = self._target_position()
      point = self.route_index % len(MAZE_ROUTE_X)
      passage = ((point >= 1) & (point <= 4)) | ((point >= 8) & (point <= 10))
      radius = torch.where(passage, self.cfg.passage_radius, self.cfg.waypoint_radius)
      reached = torch.linalg.vector_norm(target - pos, dim=1) < radius
      passed = pos[:, 0] > target[:, 0] + 0.05
      self.route_index += ((reached | passed) & (self.route_index < last)).long()
    point = self.route_index % len(MAZE_ROUTE_X)
    near_passage = ((point >= 1) & (point <= 5)) | ((point >= 7) & (point <= 11))
    lookahead = self._lookahead_target(pos, near_passage)
    lookahead_clear = self._segment_clear(pos, lookahead)
    # If the look-ahead point is blocked, try a visible route waypoint before
    # aligning sideways. This keeps a drifted robot from targeting behind it.
    fallback, fallback_clear, fallback_index = self._visible_target(pos)
    target = torch.where(lookahead_clear[:, None], lookahead, fallback)
    visible = lookahead_clear | fallback_clear
    active_index = torch.where(lookahead_clear, self.route_index, fallback_index)
    # A direct shortcut can graze the open leaf even while the waypoint itself
    # is reachable. Move laterally on the near side first, then aim forward.
    alignment = torch.stack((pos[:, 0] + 0.01, target[:, 1]), dim=1)
    align_clear = self._segment_clear(pos, alignment)
    aligning = ~visible & align_clear & ((target[:, 1] - pos[:, 1]).abs() > 0.05)
    target = torch.where(aligning[:, None], alignment, target)
    visible |= aligning
    self.active_target_w[:] = target
    self.active_target_index[:] = active_index
    self.target_visible[:] = visible
    self.target_is_alignment[:] = aligning
    heading_to_target = torch.atan2(target[:, 1] - pos[:, 1], target[:, 0] - pos[:, 0])
    self.heading_target[:] = torch.where(visible, heading_to_target, self.robot.data.heading_w)
    self.heading_error[:] = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
    # Match the course follower's body-frame steering and filtered speed
    # feedback. Keep the speeds bounded for the maze's narrow passages.
    speed = torch.where(near_passage, self.passage_speed, self.cruise_speed)
    facing = self.heading_error.abs() <= 0.65
    moving = visible & facing
    target_vx = torch.where(moving, speed * self.heading_error.cos().clamp(min=0), 0.0)
    target_vy = torch.where(moving, (speed * self.heading_error.sin()).clamp(-0.3, 0.3), 0.0)
    alpha = 1.0 - math.exp(-self._env.step_dt / self.cfg.speed_filter_seconds)
    filtered = torch.where(
      self.speed_filter_valid, self.filtered_vx + alpha *
      (self.robot.data.root_link_lin_vel_b[:, 0] - self.filtered_vx), target_vx
    )
    self.filtered_vx[:] = torch.where(moving, filtered, 0.0)
    self.speed_filter_valid[:] = moving
    max_vx = torch.where(near_passage, 0.30, 0.60)
    self.vel_command_b[:, 0] = torch.where(
      moving, (target_vx + self.cfg.speed_kp * (target_vx - filtered)).clamp(min=0).minimum(max_vx), 0.0
    )
    self.vel_command_b[:, 1] = target_vy
    self.vel_command_b[:, 2] = torch.where(visible, (2.0 * self.heading_error).clamp(-1.0, 1.0), 0.0)


def mark_start(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Reset event (after the robot is placed): remember where each robot started."""
  # From the joint state just written by the reset: body positions (root_link_pos_w)
  # are only refreshed at the next simulator step and still show the old episode.
  data = env.scene["robot"].data
  pos = data.data.qpos[:, data.indexing.free_joint_q_adr[:2]]
  if not hasattr(env, "_lane_start"):
    env._lane_start = pos.clone()
  if env_ids is None:
    env._lane_start[:] = pos
  else:
    env._lane_start[env_ids] = pos[env_ids]


def set_initial_terrain_level(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, level: int) -> None:
  """Place every robot on one curriculum row at startup; later resets adapt normally."""
  del env_ids
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  assert terrain.env_origins is not None
  terrain.terrain_levels[:] = level
  terrain.env_origins[:] = terrain.terrain_origins[level, terrain.terrain_types]


def lane_terrain_levels(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str, promote_distance: float = 3.5
) -> torch.Tensor:
  """Terrain curriculum for lanes: a robot that ended its episode `promote_distance`
  from where it started moves to a harder row; one that covered less than half of
  what its command asked for moves to an easier one (mjlab's rule, from the start
  point instead of the tile origin)."""
  terrain = env.scene.terrain
  assert terrain is not None
  start = getattr(env, "_lane_start", None)
  if start is not None:
    pos = env.scene["robot"].data.root_link_pos_w[env_ids, :2]
    distance = torch.norm(pos - start[env_ids], dim=1)
    command = env.command_manager.get_command(command_name)
    move_up = distance > promote_distance
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    terrain.update_env_origins(env_ids, move_up, move_down & ~move_up)
  return torch.mean(terrain.terrain_levels.float())


def spawn_at_lane_start(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Start fixed-difficulty playback at the first tile of its long lane."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  terrain.terrain_levels[env_ids] = 0
  terrain.terrain_types[env_ids] = 0
  terrain.env_origins[env_ids] = terrain.terrain_origins[0, 0]


def outside_lane_bounds(
  env: ManagerBasedRlEnv, half_length: float, half_width: float
) -> torch.Tensor:
  """End an episode when the root leaves the lane grid or its corridor."""
  terrain = env.scene.terrain
  assert terrain is not None and terrain.env_origins is not None
  pos = env.scene["robot"].data.root_link_pos_w[:, :2]
  return (pos[:, 0].abs() > half_length) | (
    (pos[:, 1] - terrain.env_origins[:, 1]).abs() > half_width
  )
