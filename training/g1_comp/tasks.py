"""The G1 tasks, one per world. Train on one, fine-tune on another: every task
here has the same inputs and outputs, so any model runs (and continues) on any task.

  Flat         flat ground. The base model: learn to walk and follow commands.
  Ramp         lanes of the course's 15 degree ramp tiles.
  Stairs       narrow lanes of the course's stairs, up to a landing and down.
  Open         all the course's parts (ramps, stairs, beams, door sill) in random
               order, in endless lanes. The generalist: one model for everything.
  Rough        realistic rough ground (bumps, gentle slopes, curbs).
  VeryRough    much rougher than expected, for robustness.
  Competition  the competition course itself, from the organizers' CAD.

  <Task>-Crouch  the same world, plus crouching: more robots stand still, and most
               of those are told to lower the pelvis (down to 0.42 m).

What a robot is doing at any moment (the "division"; constants below):
  - walking a commanded velocity: 90% (75% in -Crouch tasks). In lanes most are
    told to walk along the lane; on open ground in any direction.
  - standing still: 10% (25% in -Crouch tasks, 60% of those crouching).
  - arms moved by someone else (random motion, as the IK will do): half of the
    standing time and 5% of the walking time.

Built on Unitree's G1 rough task (robot, motors, most rewards, randomization), with
the height map, sim-to-real noise and rewards of our older G1-Comp-* tasks
(env_cfgs.py), minus what broke those: no fixed-schedule "skills ramp", far fewer
standing robots, and the arm takeover is rare while walking.
"""

from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Callable
from functools import partial

from mjlab.actuator import DelayedActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainGeneratorCfg
from src.assets.robots.unitree_g1.g1_constants import FULL_COLLISION
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from . import lanes, mdp, worlds
from .actions import ArmOverrideJointPositionActionCfg
from .camera import fpv_enabled, head_camera_cfg
from .commands import BaseHeightCommandCfg
from .map_noise import MapNoiseCfg
from .scan import terrain_heights, terrain_heights_clean
from .sensors import (
  FOOT_PATCH_RESOLUTION,
  FOOT_PATCH_SIZE,
  FOOT_SENSOR_NAMES,
  SCAN_RESOLUTION,
  SCAN_SIZE,
  foot_ground_sensor_cfgs,
  grid_shape,
  terrain_scan_cfg,
)
from .terrains import course_terrains_cfg

# --- What the robots are told to do ---------------------------------------------
STANDING_SHARE = 0.10  # robots told to stand still (Unitree: 0.05)
CROUCH_STANDING_SHARE = 0.25  # the same, in -Crouch tasks
CROUCH_PROB = 0.6  # -Crouch tasks: chance a standing robot is also told to crouch
ARM_SHARE_WALKING = 0.05  # share of walking time the arms follow random motion
ARM_SHARE_STANDING = 0.5  # share of standing time (that is when the IK presses buttons)
YAW_RATE_SHARE = 0.3  # robots given a fixed turning rate (a joystick) instead of a heading to face
# Speeds (m/s, rad/s). Open ground: any direction. Lanes: mostly forward along the lane.
OPEN_SPEEDS = dict(lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0))
LANE_SPEEDS = dict(lin_vel_x=(-0.2, 0.8), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0))
LANE_SHARE = 0.9  # lanes: share of robots told to walk along the lane (the rest: any heading)
START_LEVEL = 1  # hardest terrain row a robot may start on (rows 0-9); they earn the rest

# --- Height map and rewards (as in the G1-Comp-* tasks; see training/README.md) ----
# The height map reads terrain height relative to the feet (scan.py): 0 on flat
# ground. Clipped at 0.9 m: taller is a wall, deeper is a drop, whatever the number.
SCAN_CLIP = (-0.9, 0.9)
FOOT_CLEARANCE_TARGET = 0.12  # m above the ground under the swing foot (Unitree: 0.10)
FOOT_SITES = ("left_foot", "right_foot")
HEIGHT_TRACKING_STD = 0.15
HEIGHT_TRACKING_WEIGHT = 2.0
TORSO_ANG_VEL_WEIGHT = -0.1  # Unitree: -0.05. The LiDAR is in the head.
EDGE_HEIGHT = 0.05  # m between neighbouring points under a sole that counts as a step edge
EDGE_WEIGHT = -0.2
STUMBLE_WEIGHT = -0.3
ARM_IDLE_WEIGHT = -0.01  # keeps the policy's arm outputs near zero while the arms are overridden
INIT_NOISE_STD = 0.5  # PPO's starting action noise (Unitree: 1.0)

# --- Contacts ----------------------------------------------------------------------
# Unitree lets all 33 collision shapes of the robot touch everything. Every pair of
# (robot shape, terrain shape) costs time each step, and a world has thousands of
# terrain shapes. Only the legs should ever touch the world, so the feet, shins,
# ankle braces and thighs (20 shapes) keep full contact, and the pelvis, hips, torso,
# head, arms and hands (13 shapes) only collide with the robot itself (arms still
# can't pass through the body). G1_FULL_CONTACT=1 gives Unitree's contacts back.
LEGS_ONLY_WORLD_CONTACT = os.environ.get("G1_FULL_CONTACT", "0") != "1"
_LEG_GEOMS = r"^(left|right)_(foot[1-7]|shin|linkage_brace|thigh)_collision$"
_WORLD, _SELF = 1, 2  # contact bits: terrain is on bit 1 (MuJoCo's default)

# --- Sim-to-real (training only; play shows the nominal robot) ---------------------
MAP_NOISE = MapNoiseCfg()  # what a SLAM map gets wrong (map_noise.py)
COMMAND_DELAY_STEPS = (0, 6)  # joint commands arrive 0-30 ms late (5 ms steps), per episode
PD_GAIN_SCALE = (0.9, 1.1)  # each joint's motor gains, per episode


def _delay_commands(cfg: ManagerBasedRlEnvCfg) -> None:
  robot = cfg.scene.entities["robot"]
  assert robot.articulation is not None
  actuators = tuple(
    a if isinstance(a, DelayedActuatorCfg) else DelayedActuatorCfg(
      base_cfg=a,
      delay_target="position",
      delay_min_lag=COMMAND_DELAY_STEPS[0],
      delay_max_lag=COMMAND_DELAY_STEPS[1],
      delay_hold_prob=1.0,  # keep the episode's lag (set at reset)
    )
    for a in robot.articulation.actuators
  )
  # replace(): the articulation config is shared with Unitree's own tasks.
  robot.articulation = dataclasses.replace(robot.articulation, actuators=actuators)
  cfg.events["command_delay"] = EventTermCfg(
    func=dr.sync_actuator_delays, mode="reset", params={"lag_range": COMMAND_DELAY_STEPS}
  )


def _base_cfg(play: bool, crouch: bool) -> ManagerBasedRlEnvCfg:
  """The robot, its inputs, commands and rewards. The world is added by each task."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  if LEGS_ONLY_WORLD_CONTACT:
    bits = {_LEG_GEOMS: _WORLD | _SELF, ".*_collision": _SELF}
    cfg.scene.entities["robot"].collisions = (
      dataclasses.replace(FULL_COLLISION, contype=dict(bits), conaffinity=dict(bits)),
    )

  # Sensors: our height scan replaces Unitree's; a ground patch under each foot.
  others = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
  cfg.scene.sensors = (terrain_scan_cfg(), *foot_ground_sensor_cfgs(), *others)
  # First-person view (play.py --fpv / --ros): the head camera. Never while training.
  if play and fpv_enabled():
    cfg.scene.sensors += (head_camera_cfg(),)

  # Inputs. The policy's scan has SLAM-like errors while training; the critic's is clean.
  actor, critic = cfg.observations["actor"].terms, cfg.observations["critic"].terms
  actor["height_scan"] = ObservationTermCfg(
    func=terrain_heights,
    params={
      "sensor_name": "terrain_scan",
      "grid_shape": grid_shape(SCAN_SIZE, SCAN_RESOLUTION),
      "resolution": SCAN_RESOLUTION,
      "noise": MAP_NOISE,
      "corrupt": not play,
    },
    clip=SCAN_CLIP,
  )
  critic["height_scan"] = ObservationTermCfg(
    func=terrain_heights_clean, params={"sensor_name": "terrain_scan"}, clip=SCAN_CLIP
  )
  critic["foot_height"] = ObservationTermCfg(func=mdp.foot_clearance, params={"sensor_names": FOOT_SENSOR_NAMES})
  critic["foot_contact_forces"] = ObservationTermCfg(
    func=mdp.foot_contact_forces_yaw, params={"sensor_name": "feet_ground_contact"}
  )
  for terms in (actor, critic):
    terms["height_command"] = ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "base_height"}
    )
    terms["arm_override"] = ObservationTermCfg(func=mdp.arm_override_flag)
  critic["base_height"] = ObservationTermCfg(func=mdp.measured_base_height)

  # Commands: velocity (speeds fixed: no widening to 2 m/s later), then the crouch
  # height (after "twist": it reads the velocity command).
  old = cfg.commands["twist"]
  assert isinstance(old, UniformVelocityCommandCfg)
  cfg.commands["twist"] = lanes.LaneVelocityCommandCfg(
    entity_name=old.entity_name,
    resampling_time_range=old.resampling_time_range,
    debug_vis=old.debug_vis,
    heading_command=True,
    heading_control_stiffness=old.heading_control_stiffness,
    rel_standing_envs=CROUCH_STANDING_SHARE if crouch else STANDING_SHARE,
    rel_heading_envs=1.0 - YAW_RATE_SHARE,
    ranges=UniformVelocityCommandCfg.Ranges(heading=(-math.pi, math.pi), **OPEN_SPEEDS),
    viz=old.viz,
  )
  cfg.curriculum.pop("command_vel", None)
  cfg.commands["base_height"] = BaseHeightCommandCfg(
    resampling_time_range=(2.0, 5.0), crouch_prob=CROUCH_PROB if crouch else 0.0
  )

  # Actions: all 29 joints; the arms are sometimes moved by random motion instead.
  # Off while playing, so you see the policy's own arms.
  joints = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = ArmOverrideJointPositionActionCfg(
    entity_name=joints.entity_name,
    actuator_names=joints.actuator_names,
    scale=joints.scale,
    use_default_offset=True,
    preserve_order=joints.preserve_order,
    override_prob=0.0 if play else ARM_SHARE_WALKING,
    standing_override_prob=0.0 if play else ARM_SHARE_STANDING,
  )

  # Rewards: Unitree's, with posture terms that follow the crouch command and ignore
  # overridden arms, foot clearance measured from the ground under the foot, and
  # small penalties for unsafe steps.
  rewards = cfg.rewards
  rewards["base_height"] = RewardTermCfg(
    func=mdp.base_height_tracking, weight=HEIGHT_TRACKING_WEIGHT, params={"std": HEIGHT_TRACKING_STD}
  )
  rewards["pose"] = RewardTermCfg(
    func=mdp.crouch_aware_posture,
    weight=rewards["pose"].weight,
    params={**rewards["pose"].params, "height_command_name": "base_height"},
  )
  rewards["stand_still"] = RewardTermCfg(
    func=mdp.crouch_aware_stand_still,
    weight=rewards["stand_still"].weight,
    params={**rewards["stand_still"].params, "height_command_name": "base_height"},
  )
  rewards["foot_clearance"] = RewardTermCfg(
    func=mdp.feet_clearance_terrain,
    weight=rewards["foot_clearance"].weight,
    params={
      "target_height": FOOT_CLEARANCE_TARGET,
      "sensor_names": FOOT_SENSOR_NAMES,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
    },
  )
  rewards["body_ang_vel"].weight = TORSO_ANG_VEL_WEIGHT
  rewards["foot_on_edge"] = RewardTermCfg(
    func=mdp.foot_on_edge,
    weight=EDGE_WEIGHT,
    params={
      "sensor_names": FOOT_SENSOR_NAMES,
      "patch_shape": grid_shape(FOOT_PATCH_SIZE, FOOT_PATCH_RESOLUTION),
      "contact_sensor_name": "feet_ground_contact",
      "edge_height": EDGE_HEIGHT,
    },
  )
  rewards["foot_stumble"] = RewardTermCfg(
    func=mdp.foot_stumble, weight=STUMBLE_WEIGHT, params={"contact_sensor_name": "feet_ground_contact"}
  )
  rewards["arm_idle"] = RewardTermCfg(func=mdp.arm_idle, weight=ARM_IDLE_WEIGHT)

  # Sim-to-real: late joint commands and off-nominal motor gains, drawn per episode.
  if not play:
    _delay_commands(cfg)
    cfg.events["pd_gains"] = EventTermCfg(
      func=dr.pd_gains, mode="reset", params={"kp_range": PD_GAIN_SCALE, "kd_range": PD_GAIN_SCALE}
    )
  return cfg


def _use_world(cfg: ManagerBasedRlEnvCfg, world: TerrainGeneratorCfg, play: bool) -> None:
  """Put a generated world under the robots. Training: robots start on the easy
  rows and move up as they succeed. Play: a small random sample of the world."""
  world.curriculum = not play
  if play:
    world.num_rows, world.num_cols = 4, 5
    # Contacts allocated per robot (training keeps Unitree's 48: memory grows with it).
    cfg.sim.nconmax = 128
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = world
  cfg.scene.terrain.max_init_terrain_level = START_LEVEL


def _use_lanes(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
  """Lane worlds: start on the tile's flat spots, walk along the lane, and move
  between rows by distance covered from the start."""
  twist = cfg.commands["twist"]
  assert isinstance(twist, lanes.LaneVelocityCommandCfg)
  twist.lane_prob = LANE_SHARE
  twist.rel_heading_envs = 0.9  # turning on the spot in a 0.8 m corridor: rarely
  twist.ranges = UniformVelocityCommandCfg.Ranges(heading=(-math.pi, math.pi), **LANE_SPEEDS)
  spawn = EventTermCfg(
    func=envs_mdp.reset_root_state_from_flat_patches,
    mode="reset",
    params={"patch_name": "spawn", "pose_range": {"yaw": (-math.pi, math.pi)}, "velocity_range": {}},
  )
  # Order matters: pick the tile (play only), place the robot, then note where it starts.
  events = dict(cfg.events)
  first = {k: events.pop(k) for k in ("randomize_terrain",) if k in events}
  events.pop("reset_base", None)
  cfg.events = {
    **first,
    "reset_base": spawn,
    **events,
    "mark_start": EventTermCfg(func=lanes.mark_start, mode="reset"),
  }
  if not play:
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=lanes.lane_terrain_levels, params={"command_name": "twist"}
    )


# --- The tasks --------------------------------------------------------------------


def flat_env_cfg(play: bool = False, crouch: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat ground, with Unitree's flat-ground simulator settings. The height map
  stays (it reads flat), so the model has the same inputs as on every other task."""
  cfg = _base_cfg(play, crouch)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.events.pop("randomize_terrain", None)
  cfg.curriculum.pop("terrain_levels", None)
  cfg.sim.njmax, cfg.sim.nconmax = 300, None
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  return cfg


def ramp_env_cfg(play: bool = False, crouch: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _base_cfg(play, crouch)
  _use_world(cfg, worlds.ramp_world_cfg(), play)
  _use_lanes(cfg, play)
  return cfg


def stairs_env_cfg(play: bool = False, crouch: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _base_cfg(play, crouch)
  _use_world(cfg, worlds.stairs_world_cfg(), play)
  _use_lanes(cfg, play)
  return cfg


def open_env_cfg(play: bool = False, crouch: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _base_cfg(play, crouch)
  _use_world(cfg, worlds.open_world_cfg(), play)
  _use_lanes(cfg, play)
  return cfg


def rough_env_cfg(play: bool = False, crouch: bool = False) -> ManagerBasedRlEnvCfg:
  """Open fields: commands in any direction, Unitree's terrain curriculum."""
  cfg = _base_cfg(play, crouch)
  _use_world(cfg, worlds.rough_world_cfg(), play)
  return cfg


def very_rough_env_cfg(play: bool = False, crouch: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _base_cfg(play, crouch)
  _use_world(cfg, worlds.very_rough_world_cfg(), play)
  return cfg


def competition_env_cfg(play: bool = False, crouch: bool = False) -> ManagerBasedRlEnvCfg:
  """The course itself (one copy, shared: each robot is simulated in its own
  world, so they never touch). Robots start anywhere along it. No curriculum."""
  cfg = _base_cfg(play, crouch)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = course_terrains_cfg()
  cfg.events.pop("randomize_terrain", None)
  cfg.sim.nconmax = 128  # walls, stair side panels
  _use_lanes(cfg, play)
  cfg.curriculum.pop("terrain_levels", None)
  return cfg


def runner_cfg(task_id: str) -> RslRlOnPolicyRunnerCfg:
  """Unitree's PPO settings; runs are saved under logs/rsl_rl/<task name, lower case>/."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = task_id.lower().replace("-", "_")
  cfg.actor.distribution_cfg["init_std"] = INIT_NOISE_STD
  cfg.algorithm.class_name = "g1_comp.policy:CompPPO"  # mirror augmentation + fixed input scales
  return cfg


_WORLDS: dict[str, Callable[..., ManagerBasedRlEnvCfg]] = {
  "Flat": flat_env_cfg,
  "Ramp": ramp_env_cfg,
  "Stairs": stairs_env_cfg,
  "Open": open_env_cfg,
  "Rough": rough_env_cfg,
  "VeryRough": very_rough_env_cfg,
  "Competition": competition_env_cfg,
}

# Task name -> config factory taking play=False/True.
TASKS: dict[str, Callable[..., ManagerBasedRlEnvCfg]] = {}
for _name, _make in _WORLDS.items():
  TASKS[_name] = partial(_make, crouch=False)
  TASKS[f"{_name}-Crouch"] = partial(_make, crouch=True)
