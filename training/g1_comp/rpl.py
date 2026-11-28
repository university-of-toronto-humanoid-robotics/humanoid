"""Terrain-specific RPL tasks using the original G1 height-map checkpoint layout."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from . import lanes, mdp, worlds
from .sequential_tread import SequentialTreadReward
from .env_cfgs import g1_rough_heightmap_env_cfg, g1_rough_heightmap_ppo_runner_cfg
from .sensors import FOOT_ENV_CONTACT_NAME, SOLE_SENSOR_NAMES, rpl_foot_safety_sensor_cfgs
from .tasks import _use_lanes, _use_world, open_heightmap_env_cfg, open_maze_heightmap_env_cfg


RPL_TASKS = {
  "RPL-Ascend-Stairs": "ascend",
  "RPL-Descend-Stairs": "descend",
  "RPL-Ramps": "ramps",
  "RPL-Obstacles": "obstacles",
}


def _add_foot_safety(
  cfg: ManagerBasedRlEnvCfg, *, stumble_enabled: bool, stumble_ratio: float,
  stumble_weight: float, min_contact_force: float, foothold_enabled: bool,
  foothold_epsilon: float, foothold_contact_force: float, foothold_weight: float,
) -> None:
  """Shared RPL stumble and sole-support objectives for terrain experts."""
  if stumble_weight > 0 or foothold_weight > 0:
    raise ValueError("Safety penalties need nonpositive reward weights")
  if stumble_enabled or foothold_enabled:
    safety_sensors = rpl_foot_safety_sensor_cfgs()
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
      safety_sensors if foothold_enabled else safety_sensors[:1]
    )
  cfg.metrics = dict(cfg.metrics or {})
  if stumble_enabled:
    stumble_params = {
      "sensor_name": FOOT_ENV_CONTACT_NAME,
      "stumble_ratio": stumble_ratio,
      "min_contact_force": min_contact_force,
    }
    cfg.rewards["penalty_stumble"] = RewardTermCfg(
      func=mdp.penalty_stumble, weight=stumble_weight, params=stumble_params
    )
    cfg.metrics["stumble_rate"] = MetricsTermCfg(func=mdp.RplStumbleRate, params=stumble_params)
  if foothold_enabled:
    foothold_params = {
      "contact_sensor_name": FOOT_ENV_CONTACT_NAME,
      "sole_sensor_names": SOLE_SENSOR_NAMES,
      "foothold_epsilon": foothold_epsilon,
      "contact_force_threshold": foothold_contact_force,
    }
    cfg.rewards["penalty_foothold"] = RewardTermCfg(
      func=mdp.penalty_foothold, weight=foothold_weight, params=foothold_params
    )
    cfg.metrics["unsupported_sole_samples"] = MetricsTermCfg(
      func=mdp.unsupported_sole_samples, params=foothold_params
    )
  cfg.metrics["traversal_success"] = MetricsTermCfg(func=mdp.rpl_traversal_success)


def rpl_env_cfg(
  feature: str,
  play: bool = False,
  *,
  stumble_enabled: bool = True,
  stumble_ratio: float = 4.0,
  stumble_weight: float = -10.0,
  min_contact_force: float = 5.0,
  foothold_enabled: bool = True,
  foothold_epsilon: float = 0.05,
  foothold_contact_force: float = 1.0,
  foothold_weight: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """Specialize only the world and commands; retain the 791-input policy and rewards.

  Stairs and ramps inherit G1-Rough-HeightMap's rewards. Obstacles inherit
  G1-Open-HeightMap's arm-posture and beam-clearance rewards as well.
  """
  if feature not in RPL_TASKS.values():
    raise ValueError(f"Unknown RPL feature: {feature}")
  # Descent uses stronger foothold support; callers can still override it.
  if foothold_weight is None:
    foothold_weight = -5.0 if feature == "descend" else -1.0
  cfg = open_heightmap_env_cfg(play=play) if feature == "obstacles" else g1_rough_heightmap_env_cfg(play=play)
  if feature == "ramps":
    cfg.rewards["arm_joint_velocity"] = RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-0.001,
      params={"asset_cfg": SceneEntityCfg(
        "robot", joint_names=r"^(left|right)_(shoulder|elbow|wrist).*"
      )},
    )

  _add_foot_safety(
    cfg, stumble_enabled=stumble_enabled, stumble_ratio=stumble_ratio,
    stumble_weight=stumble_weight, min_contact_force=min_contact_force,
    foothold_enabled=foothold_enabled, foothold_epsilon=foothold_epsilon,
    foothold_contact_force=foothold_contact_force, foothold_weight=foothold_weight,
  )
  if feature == "descend":
    cfg.rewards["sequential_tread_contact"] = RewardTermCfg(
      func=SequentialTreadReward,
      weight=1.0,
      params={"min_support_ratio": 0.50, "min_contact_duration": 0.20,
              "contact_weight": 0.5, "completion_bonus": 0.5,
              "alternation_factor": 0.5, "window_duration": 0.30},
    )

  old = cfg.commands["twist"]
  assert isinstance(old, UniformVelocityCommandCfg)
  cfg.commands["twist"] = lanes.RplForwardCommandCfg(
    entity_name=old.entity_name,
    resampling_time_range=old.resampling_time_range,
    debug_vis=old.debug_vis,
    heading_command=True,
    heading_control_stiffness=old.heading_control_stiffness,
    rel_standing_envs=0.05,
    rel_heading_envs=1.0,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.2, 0.8), lin_vel_y=(-0.1, 0.1),
      ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi),
    ),
    viz=old.viz,
    lane_prob=1.0,
    heading_jitter=0.10,
    keep_direction_prob=1.0,
  )
  cfg.curriculum.pop("command_vel", None)
  world = worlds.rpl_world_cfg(feature, beam_group=1 if feature == "obstacles" else 0)
  _use_world(cfg, world, play)
  if play:
    # The generic lane preview concatenates ten tiles. An ascending tile ends
    # high, so the next tile would create an accidental descending cliff.
    # Show one complete feature; training still uses ten curriculum rows.
    world.num_rows = world.num_cols = 1
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.sim.nconmax = 512
    cfg.viewer.distance = 3.0
  _use_lanes(cfg, play)
  # _use_lanes gives the general-purpose lane tasks bidirectional commands;
  # these experts must encounter their feature from one defined side.
  twist = cfg.commands["twist"]
  assert isinstance(twist, lanes.RplForwardCommandCfg)
  twist.lane_prob = 1.0
  twist.rel_heading_envs = 1.0
  twist.rel_standing_envs = 0.05
  twist.heading_jitter = 0.10
  twist.ranges = UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(0.2, 0.8), lin_vel_y=(-0.1, 0.1),
    ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi),
  )
  cfg.events["reset_base"] = EventTermCfg(
    func=envs_mdp.reset_root_state_from_flat_patches,
    mode="reset",
    params={"patch_name": "spawn", "pose_range": {"yaw": (-0.1, 0.1)}, "velocity_range": {}},
  )
  cfg.terminations["lane_bounds"] = TerminationTermCfg(
    func=lanes.rpl_outside_tile,
    params={"half_width": worlds.LANE_WIDTH / 2 + 0.15},
  )
  # A successful traversal is an episode boundary, not a fall (-200 reward).
  cfg.terminations["rpl_feature_complete"] = TerminationTermCfg(
    func=lanes.rpl_feature_complete, time_out=True
  )
  if not play:
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=lanes.rpl_feature_progress)
  return cfg


def rpl_maze_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat corridors with doors, the RPL safety rewards, and full-route progression."""
  cfg = open_maze_heightmap_env_cfg(play=play, flat_only=True)
  twist = cfg.commands["twist"]
  assert isinstance(twist, lanes.MazeVelocityCommandCfg)
  twist.single_tile_route = True
  world = cfg.scene.terrain.terrain_generator
  assert world is not None
  # One geometry tile per difficulty level is sufficient; batched robots live
  # in independent MuJoCo worlds and can share the same curriculum tile.
  world.num_cols = 1
  if play:
    # Training keeps ten difficulty rows, but playback renders only one tile.
    level = twist.fixed_level if twist.fixed_level is not None else 0
    world.num_rows = 1
    world.difficulty_range = (level / 10, (level + 1) / 10)
    twist.num_rows = 1
    twist.fixed_level = level
    cfg.rewards["wall_proximity"].params["num_rows"] = 1
    cfg.rewards["wall_proximity"].params["fixed_level"] = level
  # At 0.20–0.25 m/s through the doorways, 20 s is too short to traverse.
  cfg.episode_length_s = 60.0
  _add_foot_safety(
    cfg, stumble_enabled=True, stumble_ratio=4.0, stumble_weight=-10.0,
    min_contact_force=5.0, foothold_enabled=True, foothold_epsilon=0.05,
    foothold_contact_force=1.0, foothold_weight=-1.0,
  )
  # Always enter from the near end, facing the two offset wall passages.
  cfg.events["reset_base"].params["pose_range"] = {"yaw": (-0.1, 0.1)}
  cfg.terminations["lane_bounds"] = TerminationTermCfg(
    func=lanes.rpl_outside_tile,
    params={"half_width": worlds.LANE_WIDTH / 2 + 0.15},
  )
  # From x=0.4, 6.8 m passes both openings and ends before the tile edge.
  goal_distance = 6.8
  cfg.terminations["rpl_feature_complete"] = TerminationTermCfg(
    func=lanes.rpl_feature_complete, params={"goal_distance": goal_distance}, time_out=True,
  )
  cfg.metrics["traversal_success"].params["goal_distance"] = goal_distance
  if not play:
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=lanes.rpl_feature_progress, params={"goal_distance": goal_distance},
    )
  return cfg


def rpl_descend_slower_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The descent expert with lighter stumble and stronger tread and landing rewards."""
  cfg = rpl_env_cfg("descend", play=play, stumble_weight=-2.0)
  cfg.rewards["sequential_tread_contact"].weight = 5.0
  cfg.rewards["soft_landing"].weight = -0.003
  return cfg


def rpl_runner_cfg(task_id: str) -> RslRlOnPolicyRunnerCfg:
  cfg = g1_rough_heightmap_ppo_runner_cfg()
  cfg.experiment_name = task_id.lower().replace("-", "_")
  return cfg
