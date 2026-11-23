"""Bounded, ordered tread-contact shaping for the RPL stair-descent expert."""

from __future__ import annotations

import math

import numpy as np
import torch

from . import worlds
from .sensors import FOOT_ENV_CONTACT_NAME, SOLE_SENSOR_NAMES


def tread_support(
  sole_points: torch.Tensor, hit_heights: torch.Tensor, upward_force: torch.Tensor,
  origins: torch.Tensor, specs: torch.Tensor, body_weight: float,
  edge_margin: float = 0.005, height_tolerance: float = 0.025,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-foot, per-tread estimated support; shapes [env, 2, max_treads].

  The contact sensor has one net force per foot, so force cannot be attributed
  to an individual tread. The sole's pose and ray hit must also match that tread.
  """
  count = specs[:, 0].long()
  riser, depth, lane_width = specs[:, 1], specs[:, 2], specs[:, 3]
  k = torch.arange(worlds.STAIR_RISERS, device=sole_points.device)
  valid = k[None, :] < count[:, None]
  start = 2.0 + depth[:, None] * k[None, :]
  end = torch.where(k[None, :] == count[:, None] - 1, worlds.TILE_SIZE[0], start + depth[:, None])
  top = torch.clamp(count[:, None] - 1 - k[None, :], min=0) * riser[:, None]

  # Tile origin is (0.45, tile_width/2, start_height) in local coordinates.
  x = sole_points[..., 0] - origins[:, None, None, 0] + 0.45
  y = sole_points[..., 1] - origins[:, None, None, 1]
  inside = ((x[..., None] >= start[:, None, None, :] + edge_margin)
            & (x[..., None] <= end[:, None, None, :] - edge_margin)
            & (y.abs()[..., None] <= lane_width[:, None, None, None] / 2 - edge_margin))
  on_top = ((sole_points[..., 2, None] - top[:, None, None, :]).abs() <= height_tolerance)
  hit_top = ((hit_heights[..., None] - top[:, None, None, :]).abs() <= 0.015)
  coverage = (inside & on_top & hit_top & valid[:, None, None, :]).float().mean(dim=2)

  minimum = 0.05 * body_weight
  full = 0.20 * body_weight
  quality = ((upward_force - minimum) / (full - minimum)).clamp(0.0, 1.0)
  return coverage * quality[..., None], coverage


class SequentialTreadReward:
  """One-shot credit for sustained tread support, with bounded contact shaping.

  Skips rebase the target after a qualified later landing and get no credit for
  the skipped or landing tread. The actor's observations are unchanged.
  """

  def __init__(self, cfg, env):
    del cfg
    self.env = env
    terrain = env.scene.terrain
    assert terrain is not None and terrain.cfg.terrain_generator is not None
    generator = terrain.cfg.terrain_generator
    sub = generator.sub_terrains["descend"]
    assert isinstance(sub, worlds.RplLaneTerrainCfg)
    expected = generator.num_rows * generator.num_cols
    if len(sub.stair_specs) != expected:
      raise RuntimeError(f"Expected {expected} generated stair specifications, got {len(sub.stair_specs)}")
    specs = np.asarray(sub.stair_specs, dtype=np.float32).reshape(generator.num_cols, generator.num_rows, 4)
    self.specs = torch.as_tensor(specs.transpose(1, 0, 2).copy(), device=env.device)
    robot = env.scene["robot"]
    self.body_weight = float(env.sim.mj_model.body_subtreemass[robot.indexing.root_body_id]) * 9.81
    if self.body_weight <= 0:
      raise ValueError("Robot mass must be positive")
    n = env.num_envs
    self.target = torch.zeros(n, dtype=torch.long, device=env.device)
    self.target_foot = torch.full((n,), -1, dtype=torch.long, device=env.device)
    self.duration = torch.zeros(n, dtype=torch.long, device=env.device)
    self.later_tread = torch.full((n,), -1, dtype=torch.long, device=env.device)
    self.later_foot = torch.full((n,), -1, dtype=torch.long, device=env.device)
    self.later_duration = torch.zeros(n, dtype=torch.long, device=env.device)
    self.last_foot = torch.full((n,), -1, dtype=torch.long, device=env.device)
    self.window = torch.zeros(n, dtype=torch.long, device=env.device)
    self.window_opened = torch.zeros(n, dtype=torch.bool, device=env.device)
    self.completed = torch.zeros(n, dtype=torch.long, device=env.device)
    self.skipped = torch.zeros(n, dtype=torch.long, device=env.device)
    self.support_sum = torch.zeros(n, device=env.device)
    self.step_count = torch.zeros(n, device=env.device)
    self.qualified_duration_sum = torch.zeros(n, device=env.device)
    self.alternated = torch.zeros(n, device=env.device)
    self.transitions = torch.zeros(n, device=env.device)
    self.clean_finish = torch.zeros(n, dtype=torch.bool, device=env.device)
    self.active_count = torch.ones(n, device=env.device)
    self.needs_init = torch.ones(n, dtype=torch.bool, device=env.device)

  def __call__(
    self, env, min_support_ratio: float, min_contact_duration: float,
    contact_weight: float, completion_bonus: float, alternation_factor: float,
    window_duration: float,
  ) -> torch.Tensor:
    terrain = env.scene.terrain
    assert terrain is not None
    specs = self.specs[terrain.terrain_levels.long(), terrain.terrain_types.long()]
    count = specs[:, 0].long()
    self.active_count.copy_(count)
    origins = terrain.env_origins
    sole, hits = [], []
    for name in SOLE_SENSOR_NAMES:
      sensor = env.scene[name]
      assert sensor.sole_points_w is not None
      sole.append(sensor.sole_points_w)
      hits.append(torch.where(sensor.data.distances >= 0, sensor.data.hit_pos_w[..., 2], -torch.inf))
    contact = env.scene[FOOT_ENV_CONTACT_NAME]
    assert contact.data.force is not None
    upward = (-contact.data.force[..., 2]).clamp_min(0)
    score, coverage = tread_support(
      torch.stack(sole, dim=1), torch.stack(hits, dim=1), upward,
      origins, specs, self.body_weight,
    )
    qualified = (coverage >= min_support_ratio) & (upward[..., None] >= 0.08 * self.body_weight)
    valid = self.target < count
    # Current resets begin on the top landing; this also handles a reset on a tread.
    # Apply unconditionally to avoid a CUDA->CPU synchronization on every RL step.
    root = env.scene["robot"].data.root_link_pos_w
    local_x = root[:, 0] - origins[:, 0] + 0.45
    initial = torch.floor((local_x - 2.0) / specs[:, 2]).long().clamp_min(0)
    self.target = torch.where(self.needs_init, torch.minimum(initial, count - 1), self.target)
    self.needs_init[:] = False
    target_idx = self.target.clamp(max=worlds.STAIR_RISERS - 1)
    gather = target_idx[:, None, None].expand(-1, 2, 1)
    target_score = score.gather(2, gather).squeeze(-1)
    target_qualified = qualified.gather(2, gather).squeeze(-1) & valid[:, None]

    factors = torch.where(
      (self.last_foot[:, None] < 0) | (self.last_foot[:, None] != torch.arange(2, device=env.device)),
      1.0, alternation_factor,
    )
    weighted = target_score * factors
    chosen_score = weighted.max(dim=1).values
    qualified_score = torch.where(target_qualified, weighted, -1.0)
    chosen = qualified_score.argmax(dim=1)
    chosen_valid = qualified_score.max(dim=1).values >= 0
    same = chosen == self.target_foot
    self.duration = torch.where(chosen_valid, torch.where(same, self.duration + 1, 1), 0)
    self.target_foot = torch.where(chosen_valid, chosen, -1)
    required = max(1, math.ceil(min_contact_duration / env.step_dt))
    completed = chosen_valid & (self.duration >= required)

    opens = (chosen_score > 0) & ~self.window_opened & valid
    self.window = torch.where(opens, max(1, math.ceil(window_duration / env.step_dt)), self.window)
    self.window_opened |= opens
    shaping = torch.where((self.window > 0) & valid, chosen_score, 0.0)
    self.window = torch.clamp(self.window - 1, min=0)
    self.support_sum += chosen_score
    self.step_count += 1

    # Look only for *qualified* contacts on later treads; rebase after sustained support.
    indices = torch.arange(worlds.STAIR_RISERS, device=env.device)
    later_mask = indices[None, None, :] > self.target[:, None, None]
    later_scores = torch.where(qualified & later_mask, score, -1.0)
    flat = later_scores.flatten(1).argmax(dim=1)
    later_foot = flat // worlds.STAIR_RISERS
    later_idx = flat % worlds.STAIR_RISERS
    later_valid = later_scores.flatten(1).gather(1, flat[:, None]).squeeze(1) >= 0
    same_later = (later_foot == self.later_foot) & (later_idx == self.later_tread)
    self.later_duration = torch.where(later_valid, torch.where(same_later, self.later_duration + 1, 1), 0)
    self.later_foot = torch.where(later_valid, later_foot, -1)
    self.later_tread = torch.where(later_valid, later_idx, -1)
    recovered = later_valid & (self.later_duration >= required) & ~completed

    old_foot = self.last_foot.clone()
    self.alternated += (completed & (old_foot >= 0) & (old_foot != chosen)).float()
    self.transitions += (completed & (old_foot >= 0)).float()
    self.qualified_duration_sum += completed.float() * self.duration.float() * env.step_dt
    self.last_foot = torch.where(completed, chosen, self.last_foot)
    self.last_foot = torch.where(recovered, later_foot, self.last_foot)
    self.completed += completed.long()
    self.skipped += torch.where(recovered, later_idx - self.target, 0)
    self.target = torch.where(completed, self.target + 1, self.target)
    self.target = torch.where(recovered, later_idx + 1, self.target)
    changed = completed | recovered
    self.duration = torch.where(changed, 0, self.duration)
    self.target_foot = torch.where(changed, -1, self.target_foot)
    self.later_duration = torch.where(changed, 0, self.later_duration)
    self.window = torch.where(changed, 0, self.window)
    self.window_opened = torch.where(changed, False, self.window_opened)

    root_x = env.scene["robot"].data.root_link_pos_w[:, 0]
    start = getattr(env, "_lane_start", None)
    if start is not None:
      finish = root_x - start[:, 0] >= 5.6
      self.clean_finish = finish & (self.completed == count) & (self.skipped == 0)
    # The manager scales reward rates by step_dt; divide a one-time bonus by dt.
    return contact_weight * shaping + (completion_bonus / env.step_dt) * completed.float()

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    log = self.env.extras["log"]
    # Curriculum may already have moved resetting envs to a new row.
    count = self.active_count.clamp_min(1)
    log["Episode_Metrics/clean_descent"] = self.clean_finish[env_ids].float().mean()
    log["Episode_Metrics/tread_completion_rate"] = (self.completed[env_ids] / count[env_ids]).mean()
    log["Episode_Metrics/skipped_treads"] = self.skipped[env_ids].float().mean()
    log["Episode_Metrics/mean_tread_support"] = (
      self.support_sum[env_ids] / self.step_count[env_ids].clamp_min(1)
    ).mean()
    log["Episode_Metrics/alternating_contact_fraction"] = (
      self.alternated[env_ids] / self.transitions[env_ids].clamp_min(1)
    ).mean()
    log["Episode_Metrics/qualified_contact_duration"] = (
      self.qualified_duration_sum[env_ids] / self.completed[env_ids].clamp_min(1)
    ).mean()
    log["Episode_Metrics/target_tread_idx"] = self.target[env_ids].float().mean()
    for field in ("target", "duration", "later_duration", "window", "completed", "skipped",
                  "support_sum", "step_count", "qualified_duration_sum", "alternated", "transitions"):
      getattr(self, field)[env_ids] = 0
    for field in ("target_foot", "later_tread", "later_foot", "last_foot"):
      getattr(self, field)[env_ids] = -1
    self.window_opened[env_ids] = False
    self.clean_finish[env_ids] = False
    self.needs_init[env_ids] = True

  def debug_vis(self, visualizer) -> None:
    terrain = self.env.scene.terrain
    assert terrain is not None
    for i in visualizer.get_env_indices(self.env.num_envs):
      spec = self.specs[terrain.terrain_levels[i], terrain.terrain_types[i]]
      k = int(self.target[i].item())
      n = int(spec[0].item())
      if k >= n:
        continue
      depth = float(spec[2].item())
      x = float(terrain.env_origins[i, 0].item()) - 0.45 + 2.0 + (k + 0.5) * depth
      z = float(spec[1].item()) * (n - 1 - k) + 0.08
      center = (x, float(terrain.env_origins[i, 1].item()), z)
      visualizer.add_sphere(np.asarray(center), 0.07, (0.1, 0.9, 0.3, 0.9), label=f"Target tread {k}")
