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
class MazeVelocityCommandCfg(LaneVelocityCommandCfg):
  """Visit ordered position targets through the door and wall passages."""

  num_rows: int = 10
  tile_length: float = 8.0
  fixed_level: int | None = None
  waypoint_radius: float = 0.18
  passage_radius: float = 0.08
  cruise_speed: float = 0.45
  passage_speed: float = 0.25
  clearance_radius: float = 0.25

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

  def reset(self, env_ids):
    self.route_pending[env_ids] = True
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

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # The route and speed profile stay fixed throughout an episode. The
    # CommandManager may call this every few seconds, but nothing is sampled.
    self.lane_direction[env_ids] = 0.0
    self.is_heading_env[env_ids] = True
    self.is_standing_env[env_ids] = False
    self.vel_command_b[env_ids] = 0.0

  def _segment_clear(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """Check a straight centre path with body clearance against lane walls and passages."""
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
    safe = y.abs() <= worlds.maze_lane_width(d) / 2 - self.cfg.clearance_radius
    for (x0, x1), center, side in zip(
      worlds.MAZE_GATE_X, (offset, -offset), worlds.MAZE_LEAF_SIDES, strict=True
    ):
      at_gate = (x >= x0 - self.cfg.clearance_radius) & (x <= x1 + self.cfg.clearance_radius)
      within_opening = (y >= center - opening / 2 + self.cfg.clearance_radius) & (
        y <= center + opening / 2 - self.cfg.clearance_radius
      )
      safe &= ~at_gate | within_opening
      if side is not None:
        (ax, ay), (bx, by) = worlds.maze_leaf_endpoints(x0, center, opening, side)
        dx, dy = bx - ax, by - ay
        along = (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)).clamp(0, 1)
        leaf_distance = torch.sqrt((x - (ax + along * dx)).square() + (y - (ay + along * dy)).square())
        safe &= leaf_distance >= self.cfg.clearance_radius + worlds.MAZE_LEAF_THICKNESS / 2
    return safe.all(dim=1)

  def _visible_target(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose a visible forward route point, never one behind the robot."""
    target = self._target_position()
    active_index = self.route_index.clone()
    visible = (target[:, 0] >= pos[:, 0] - 0.05) & self._segment_clear(pos, target)
    last = self.cfg.num_rows * len(MAZE_ROUTE_X) - 1
    for steps_forward in range(1, 5):
      future_index = (self.route_index + steps_forward).clamp(max=last)
      future = self._target_position(future_index)
      clear = (future[:, 0] > pos[:, 0] + 0.05) & self._segment_clear(pos, future)
      use = ~visible & clear
      target = torch.where(use[:, None], future, target)
      active_index = torch.where(use, future_index, active_index)
      visible |= clear
    return target, visible, active_index

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
    last = self.cfg.num_rows * len(MAZE_ROUTE_X) - 1
    for _ in range(4):
      target = self._target_position()
      point = self.route_index % len(MAZE_ROUTE_X)
      passage = ((point >= 1) & (point <= 4)) | ((point >= 8) & (point <= 10))
      radius = torch.where(passage, self.cfg.passage_radius, self.cfg.waypoint_radius)
      reached = torch.linalg.vector_norm(target - pos, dim=1) < radius
      passed = pos[:, 0] > target[:, 0] + 0.05
      self.route_index += ((reached | passed) & (self.route_index < last)).long()
    # A drifted robot may have passed its waypoint laterally. Keep the target
    # ahead and stop rather than steering back into a wall.
    target, visible, active_index = self._visible_target(pos)
    self.active_target_w[:] = target
    self.active_target_index[:] = active_index
    self.target_visible[:] = visible
    heading_to_target = torch.atan2(target[:, 1] - pos[:, 1], target[:, 0] - pos[:, 0])
    self.heading_target[:] = torch.where(visible, heading_to_target, self.robot.data.heading_w)
    super()._update_command()
    # Turn first if facing away from the safe route; stop forward if blocked.
    point = self.route_index % len(MAZE_ROUTE_X)
    near_passage = ((point >= 1) & (point <= 5)) | ((point >= 7) & (point <= 11))
    speed = torch.where(
      near_passage,
      torch.full_like(self.heading_error, self.cfg.passage_speed),
      torch.full_like(self.heading_error, self.cfg.cruise_speed),
    )
    self.vel_command_b[:, 0] = speed * self.heading_error.cos().clamp(0, 1) * visible.float()
    self.vel_command_b[:, 1] = 0.0


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
