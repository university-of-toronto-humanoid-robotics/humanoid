"""G1 competition task configs, built on Unitree's G1 rough-terrain task.

Changes from Unitree-G1-Rough:
  - Height scan: 1.6 x 1.0 m at 5 cm (693 points), 0.5 m behind to 1.1 m ahead,
    rays started 2 m above the pelvis, terrain only, clipped (see SCAN_CLIP).
  - Foot clearance reward and critic foot height measured from the ground
    under each foot, not from world z = 0 (which punished being up the stairs).
  - Terrain: competition-height stairs (G1-Comp-Rough) or the course (G1-Comp-Course).
  - Crouch: a "base_height" command (pelvis height above the feet) the robot
    tracks while standing still; posture rewards relax the leg pitch joints while
    crouching (commands.py, mdp.py).
  - Arm override: about 30% of the time the arms follow a smooth random motion
    instead of the policy (as if IK had taken over); the policy is told when
    (actions.py). Posture rewards ignore the arms then.
Everything else (rewards, gains, randomization, PPO) is Unitree's.
"""

from __future__ import annotations

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from . import mdp
from .actions import ArmOverrideJointPositionActionCfg
from .commands import BaseHeightCommandCfg
from .sensors import FOOT_SENSOR_NAMES, foot_ground_sensor_cfgs, terrain_scan_cfg
from .terrains import comp_rough_terrains_cfg, course_terrains_cfg

SCAN_NOISE = 0.02  # m, uniform noise on the policy's scan (we assume a near-perfect CAD map)
# The scan reads "pelvis height above the cell". Below -1 m (ground over 1.8 m above
# the feet) means "too tall to ever climb"; above 2 m means "very deep drop".
SCAN_CLIP = (-1.0, 2.0)
FOOT_CLEARANCE_TARGET = 0.12  # m above the ground under the swing foot (Unitree: 0.10)
FOOT_SITES = ("left_foot", "right_foot")
STANDING_SHARE = 0.3  # share of robots told to stand still (Unitree: 0.05); crouching happens here
ARM_OVERRIDE_SHARE = 0.3  # share of time the arms follow random motion instead of the policy
CROUCH_PROB = 0.7  # chance a standing robot is also told to crouch
# Crouch height reward: exp(-(error / std)^2). At 0.05 m the reward was ~0 once the
# robot lagged the command by 10 cm, so a first test learned nothing; 0.15 m gives
# a signal across the whole crouch range.
HEIGHT_TRACKING_STD = 0.15
HEIGHT_TRACKING_WEIGHT = 2.0
# Standing, crouching and arm override ramp in between these training iterations,
# after the policy has learned to walk (env vars let the cluster job change them).
SKILLS_START = int(os.environ.get("G1_COMP_SKILLS_START", 500))
SKILLS_FULL = int(os.environ.get("G1_COMP_SKILLS_FULL", 2000))
# PPO's initial action noise (Unitree: 1.0). With 29 joints and Unitree's action
# smoothness penalty, 1.0 made surviving cost more than falling early in training.
INIT_NOISE_STD = 0.5


def g1_comp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_rough_env_cfg(play=play)

  # Terrain: competition-height stairs. Same play settings as Unitree's.
  gen = comp_rough_terrains_cfg()
  gen.curriculum = not play
  if play:
    gen.num_rows, gen.num_cols, gen.border_width = 5, 5, 10.0
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = gen

  # Sensors: our scan replaces Unitree's; add a ground patch under each foot.
  others = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
  cfg.scene.sensors = (terrain_scan_cfg(), *foot_ground_sensor_cfgs(), *others)

  # Observations: scan in meters, clipped; small noise for the policy only.
  for group, noise in (("actor", Unoise(n_min=-SCAN_NOISE, n_max=SCAN_NOISE)), ("critic", None)):
    cfg.observations[group].terms["height_scan"] = ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=noise,
      clip=SCAN_CLIP,
    )
  cfg.observations["critic"].terms["foot_height"] = ObservationTermCfg(
    func=mdp.foot_clearance, params={"sensor_names": FOOT_SENSOR_NAMES}
  )

  # Crouch command (after "twist": it reads the velocity command). It starts off,
  # like the arm override and the extra standing robots: skills_curriculum ramps
  # them in (during play they are on).
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.rel_standing_envs = STANDING_SHARE if play else 0.05
  cfg.commands["base_height"] = BaseHeightCommandCfg(
    resampling_time_range=(2.0, 5.0), crouch_prob=CROUCH_PROB if play else 0.0
  )

  # Arms sometimes driven by random motion instead of the policy.
  old = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = ArmOverrideJointPositionActionCfg(
    entity_name=old.entity_name,
    actuator_names=old.actuator_names,
    scale=old.scale,
    use_default_offset=True,
    preserve_order=old.preserve_order,
    override_prob=ARM_OVERRIDE_SHARE if play else 0.0,
  )
  if not play:
    cfg.curriculum["skills"] = CurriculumTermCfg(
      func=mdp.skills_curriculum,
      params={
        "start_iteration": SKILLS_START,
        "full_iteration": SKILLS_FULL,
        "standing_share": STANDING_SHARE,
        "arm_override_share": ARM_OVERRIDE_SHARE,
        "crouch_prob": CROUCH_PROB,
      },
    )

  # New observations: the commanded height and the arm-override flag (policy and
  # critic), plus the measured height (critic only).
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms["height_command"] = ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "base_height"}
    )
    terms["arm_override"] = ObservationTermCfg(func=mdp.arm_override_flag)
  cfg.observations["critic"].terms["base_height"] = ObservationTermCfg(func=mdp.measured_base_height)

  # Rewards: track the commanded height; posture terms that allow crouching and
  # ignore overridden arms.
  cfg.rewards["base_height"] = RewardTermCfg(
    func=mdp.base_height_tracking, weight=HEIGHT_TRACKING_WEIGHT, params={"std": HEIGHT_TRACKING_STD}
  )
  cfg.rewards["pose"] = RewardTermCfg(
    func=mdp.crouch_aware_posture,
    weight=cfg.rewards["pose"].weight,
    params={**cfg.rewards["pose"].params, "height_command_name": "base_height"},
  )
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=mdp.crouch_aware_stand_still,
    weight=cfg.rewards["stand_still"].weight,
    params={**cfg.rewards["stand_still"].params, "height_command_name": "base_height"},
  )

  # Reward: swing-foot clearance above the ground under the foot.
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=mdp.feet_clearance_terrain,
    weight=cfg.rewards["foot_clearance"].weight,
    params={
      "target_height": FOOT_CLEARANCE_TARGET,
      "sensor_names": FOOT_SENSOR_NAMES,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
    },
  )
  return cfg


def g1_comp_course_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The same robot setup on the competition course, spawning anywhere along it."""
  cfg = g1_comp_rough_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = course_terrains_cfg()
  # More contacts allowed per robot than on rough terrain (48): walls, stair side
  # panels, and before the first reset every robot starts in the middle of the ramps.
  cfg.sim.nconmax = 128

  # One copy of the course: no terrain curriculum; spawn on the course's flat spots.
  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum.pop("command_vel", None)
  cfg.events.pop("randomize_terrain", None)
  cfg.events["reset_base"] = EventTermCfg(
    func=envs_mdp.reset_root_state_from_flat_patches,
    mode="reset",
    params={"patch_name": "spawn", "pose_range": {"yaw": (-math.pi, math.pi)}, "velocity_range": {}},
  )

  # Course-sized speeds (the lanes are 2.4 m wide).
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (-0.5, 1.0)
  twist.ranges.lin_vel_y = (-0.3, 0.3)
  twist.ranges.ang_vel_z = (-1.0, 1.0)
  return cfg


def g1_comp_crouch_test_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Quick check of crouching and arm override on a small GPU: flat ground, most
  robots standing, both skills on from the start. Same observations and actions
  as the other tasks, so its checkpoints play on them too."""
  cfg = g1_comp_rough_env_cfg(play=True)  # play=True: skills on, no curriculum
  if not play:  # undo the play-only settings that matter for training
    rough = g1_comp_rough_env_cfg(play=False)
    cfg.episode_length_s = rough.episode_length_s
    cfg.observations["actor"].enable_corruption = True
    cfg.events["push_robot"] = rough.events["push_robot"]
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.events.pop("randomize_terrain", None)
  cfg.curriculum = {}
  cfg.sim.njmax, cfg.sim.nconmax = 300, None  # Unitree's flat-ground settings
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.commands["twist"].rel_standing_envs = 0.6
  cfg.commands["base_height"].crouch_prob = 0.8
  return cfg


def g1_comp_ppo_runner_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  cfg.actor.distribution_cfg["init_std"] = INIT_NOISE_STD
  return cfg
