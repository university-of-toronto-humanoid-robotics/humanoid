"""The HoST tasks. Same inputs (456) and outputs (23) in all of them, so a model
from one continues on any other (train.py --from).

  HoST's, as in the paper (one policy per terrain there):
    HoST-Ground     on the back, flat ground
    HoST-Prone      face down, flat ground (HoST's ground_prone)
    HoST-Platform   on the back on a 22-46 cm platform, legs over the edge
    HoST-Wall       sitting against an incline of 14-84 deg
    HoST-Slope      on the back on a 1-15 deg slope, head uphill
  Ours:
    HoST-Stairs        on the back or a side across the course's stairs, any direction
    HoST-Supine-Mixed  the supine family everywhere: flat (back and sides), platform,
                       wall, ramps up to 17 deg (head up or down), stairs
    HoST-Prone-Mixed   the prone family: face down and sides, flat, ramps, stairs
    HoST-All           everything in one policy, plus random falls (an experiment: the
                       paper found supine + prone together trained worse)

On the robot the two families are picked from the IMU after the fall (body-frame
gravity x < 0: lying on the back -> supine policy, else prone).

The settings are HoST's (params.py has every number and where it comes from).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import partial

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.viewer import ViewerConfig

from . import mdp
from . import params as P
from . import robot, terrains
from .actions import HostJointActionCfg
from .events import Recipe, force_and_bound, randomize_actuators, reset_posture

PI = math.pi
SIDES = ("side_left", "side_right")


@dataclass
class Spec:
  """What makes one task: its world and robots, and HoST's per-terrain settings."""

  kinds: tuple[str, ...]
  recipes: tuple[Recipe, ...]
  world: TerrainGeneratorCfg | None = None  # None: flat plane
  ref: str = "host"  # height reference (mdp.py)
  stages: tuple[float, float, float] = (P.H_STAGE1, P.H_STAGE2, P.BASE_HEIGHT_TARGET)
  prone: bool = False
  group_weights: tuple[float, ...] = P.GROUP_WEIGHTS
  shoulder: tuple[float, float] = (-0.02, 0.02)
  pull_needs_upright: bool = True
  kp_kd_scale: tuple[tuple[float, float], tuple[float, float]] = (P.KP_SCALE, P.KD_SCALE)
  l2c2_value: float = P.L2C2_VALUE
  extra_terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)


def _kind_weights(kind: str, prone: bool) -> dict[str, float]:
  w = dict(P.W)
  if kind != "ground":
    w.update(P.W_PSW)
  if kind in ("platform", "wall"):
    w.update(P.W_PLATFORM_WALL)
  if prone:
    w.update(P.W_PRONE)
  return w


def _rewards(s: Spec) -> dict[str, RewardTermCfg]:
  h1, h2, target = s.stages
  ref = {"ref": s.ref}
  terms = {
    "task_head_height": (mdp.head_height_reward, ref),
    "task_orientation": (mdp.orientation_reward, {}),
    "regu_dof_acc": (mdp.dof_acc, {}),
    "regu_action_rate": (mdp.action_rate, {}),
    "regu_smoothness": (mdp.action_smoothness, {}),
    "regu_torques": (mdp.torques, {}),
    "regu_joint_power": (mdp.joint_power, {}),
    "regu_dof_vel": (mdp.dof_vel, {}),
    "regu_tracking": (mdp.tracking_error, {}),
    "regu_dof_pos_limits": (mdp.dof_pos_limits, {}),
    "regu_dof_vel_limits": (mdp.dof_vel_limits, {}),
    "style_waist_deviation": (mdp.waist_deviation, {}),
    "style_hip_yaw_deviation": (mdp.hip_yaw_deviation, {}),
    "style_hip_roll_deviation": (mdp.hip_roll_deviation, {}),
    "style_hip_pitch_deviation": (mdp.hip_pitch_deviation, {}),
    "style_shoulder_roll_deviation": (mdp.shoulder_roll_deviation,
                                      {"left_below": s.shoulder[0], "right_above": s.shoulder[1]}),
    "style_left_foot_displacement": (mdp.foot_displacement, {"side": 0, "h2": h2, **ref}),
    "style_right_foot_displacement": (mdp.foot_displacement, {"side": 1, "h2": h2, **ref}),
    "style_knee_deviation": (mdp.knee_deviation, {}),
    "style_shank_orientation": (mdp.thigh_orientation, {}) if s.prone else (mdp.shank_orientation, {"h1": h1, **ref}),
    "style_ankle_parallel": (mdp.ankle_parallel, {}),
    "style_feet_distance": (mdp.feet_distance, {}),
    "style_feet_stumble": (mdp.feet_stumble, {}),
    "style_ang_vel_xy": (mdp.ang_vel_xy, {"threshold": h1, **ref}),
    "style_lower_body_deviation": (mdp.lower_body_deviation, {}),
    "post_ang_vel_xy": (mdp.ang_vel_xy, {"threshold": h2, **ref}),
    "post_lin_vel_xy": (mdp.lin_vel_xy, {"h2": h2, **ref}),
    "post_feet_height_var": (mdp.feet_height_var, {"h2": h2, **ref}),
    "post_upper_pose": (mdp.upper_pose, {"h2": h2, **ref}),
    "post_orientation": (mdp.upright, {"h2": h2, **ref}),
    "post_base_height": (mdp.base_height_reward, {"target": target, "h2": h2, **ref}),
    "post_lower_body_deviation": (mdp.lower_body_deviation, {}),
  }
  weight_key = {"style_left_foot_displacement": "style_foot_displacement",
                "style_right_foot_displacement": "style_foot_displacement"}
  per_kind = [_kind_weights(k, s.prone) for k in s.kinds]
  out: dict[str, RewardTermCfg] = {}
  for name, (func, params) in terms.items():
    if name.startswith("task_"):
      out[name] = RewardTermCfg(func=func, weight=1.0, params=dict(params))
      continue
    ws = tuple(w[weight_key.get(name, name)] for w in per_kind)
    if all(w == 0.0 for w in ws):
      continue
    if len(set(ws)) == 1:
      out[name] = RewardTermCfg(func=func, weight=ws[0], params=dict(params))
    else:  # mixed worlds: the weight depends on the robot's terrain
      out[name] = RewardTermCfg(func=mdp.per_terrain, weight=1.0,
                                params={"term": func, "kinds": s.kinds, "weights": ws, **params})
  return out


def _events(s: Spec, play: bool) -> dict[str, EventTermCfg]:
  ev: dict[str, EventTermCfg] = {}
  if not play:  # randomized once per robot at startup (HoST: at creation)
    def torso():  # one instance per term: mjlab resolves (and checks) each in place
      return SceneEntityCfg("robot", body_names=("torso_link",))

    others = SceneEntityCfg("robot", body_names=(r"^(?!torso_link$).*",))
    geoms = SceneEntityCfg("robot", geom_names=(".*_collision",))
    ev["link_mass"] = EventTermCfg(func=dr.body_mass, mode="startup",
                                   params={"asset_cfg": others, "ranges": P.LINK_MASS_SCALE, "operation": "scale"})
    ev["payload"] = EventTermCfg(func=dr.body_mass, mode="startup",
                                 params={"asset_cfg": torso(), "ranges": P.PAYLOAD_KG, "operation": "add"})
    ev["com"] = EventTermCfg(func=dr.body_com_offset, mode="startup", params={
      "asset_cfg": torso(), "operation": "add",
      "ranges": {0: (-P.COM_OFFSET_XY, P.COM_OFFSET_XY), 1: (-P.COM_OFFSET_XY, P.COM_OFFSET_XY),
                 2: (-P.COM_OFFSET_Z, P.COM_OFFSET_Z)}})
    ev["friction"] = EventTermCfg(func=dr.geom_friction, mode="startup", params={
      "asset_cfg": geoms, "ranges": P.FRICTION, "operation": "abs", "shared_random": True})
  ev["reset_posture"] = EventTermCfg(func=reset_posture, mode="reset", params={"kinds": s.kinds, "recipes": s.recipes})
  if not play:
    ev["actuators"] = EventTermCfg(func=randomize_actuators, mode="reset",
                                   params={"kp_range": s.kp_kd_scale[0], "kd_range": s.kp_kd_scale[1]})
  return ev


def make_env_cfg(s: Spec, play: bool = False) -> ManagerBasedRlEnvCfg:
  h1, h2, target = s.stages
  sensors = [
    ContactSensorCfg(
      name=mdp.FEET_CONTACT,
      primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    )
  ]
  if s.ref == "ground":
    sensors.append(mdp.ground_ray_cfg())
  if s.world is None:
    terrain = TerrainEntityCfg(terrain_type="plane", env_spacing=3.0)
  else:
    world = replace(s.world, num_rows=min(s.world.num_rows, 4)) if play else replace(s.world)
    terrain = TerrainEntityCfg(terrain_type="generator", terrain_generator=world,
                               max_init_terrain_level=world.num_rows - 1)

  action = HostJointActionCfg(
    entity_name="robot",
    actuator_names=robot.policy_joints(),
    use_default_offset=False,
    scale=1.0,
    beta_start=P.BETA_EVAL if play else P.BETA_START,
    pull=not play,
    pull_needs_upright=s.pull_needs_upright,
  )
  proprio = ObservationTermCfg(func=mdp.host_proprio, params={"noise": not play}, history_length=P.HISTORY)
  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "blew_up": TerminationTermCfg(func=mdp.blew_up),
    **s.extra_terminations,
  }
  curriculum = {} if play else {
    "force_and_bound": CurriculumTermCfg(func=force_and_bound, params={"head_ref": s.ref, "h2": h2, "base_ref": s.ref})
  }
  return ManagerBasedRlEnvCfg(
    decimation=P.DECIMATION,
    episode_length_s=P.EPISODE_S,
    scene=SceneCfg(terrain=terrain, entities={"robot": robot.robot_cfg()}, sensors=tuple(sensors), num_envs=1,
                   extent=2.0),
    observations={"actor": ObservationGroupCfg(terms={"proprio": proprio}, concatenate_terms=True)},
    actions={"joint_pos": action},
    events=_events(s, play),
    rewards=_rewards(s),
    terminations=terminations,
    curriculum=curriculum,
    # Contacts per world. Stairs: at startup mjlab counts contacts with the robot at the
    # model file's own pose (standing at the world origin), which can be inside the steps.
    sim=SimulationCfg(nconmax=192 if "stairs" in s.kinds else 96, njmax=800,
                      mujoco=MujocoCfg(timestep=P.SIM_DT, iterations=10, ls_iterations=20)),
    viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="pelvis",
                        distance=3.0, elevation=-15.0, azimuth=120.0),
  )


# --- The tasks ----------------------------------------------------------------------

def _sides(share: float, prone: bool = False) -> dict[str, float]:
  return {k + ("_prone" if prone else ""): share / 2 for k in SIDES}


_SUPINE = Recipe(postures={"supine": 1.0})
_PRONE = Recipe(postures={"prone": 1.0})
_SIT_WALL = Recipe(postures={"sit": 1.0}, drop=0.45, pose=P.RESET_POSE_WALL, joint_scale=P.INIT_JOINT_SCALE_WIDE,
                   jitter=(0.025, 0.0), unactuated=P.UNACTUATED_STEPS_WALL, pushback=P.WALL_PUSHBACK_N)
_PLATFORM = Recipe(postures={"supine": 1.0}, pose=P.RESET_POSE_PLATFORM, joint_scale=P.INIT_JOINT_SCALE_WIDE,
                   jitter=(0.05, 0.0))
_SLOPE = Recipe(postures={"supine": 1.0}, joint_scale=P.INIT_JOINT_SCALE_WIDE, jitter=(0.5, 0.0))
_ANY_YAW = (0.0, PI / 2, PI, -PI / 2)
_ON_STAIRS = Recipe(postures={"supine": 0.6, **_sides(0.4)}, yaws=_ANY_YAW, yaw_noise=0.2, drop=0.6,
                    unactuated=P.UNACTUATED_STEPS_OURS)

_PLATFORM_TERM = {
  "rolled_off": TerminationTermCfg(func=mdp.pelvis_below, params={"height": 0.2}),
  "climbed_on": TerminationTermCfg(func=mdp.feet_above, params={"height": 0.35}),
}
_WALL_TERM = {
  "slid_down": TerminationTermCfg(func=mdp.head_below, params={"height": 0.2}),
  "climbed_on": TerminationTermCfg(func=mdp.feet_above, params={"height": 0.35}),
}

_KP_KD_PRONE = (P.KP_KD_SCALE_PRONE, P.KP_KD_SCALE_PRONE)
_MIX_KINDS = ("ground", "platform", "wall", "ramp", "stairs")


def _mixed_world(kinds: tuple[str, ...]) -> TerrainGeneratorCfg:
  make = {"ground": terrains.FlatTerrainCfg, "platform": terrains.PlatformTerrainCfg, "wall": terrains.WallTerrainCfg,
          "ramp": partial(terrains.SlopeTerrainCfg, max_deg=P.RAMP_MAX_DEG), "stairs": terrains.StairsTerrainCfg}
  return terrains.world({k: make[k]() for k in kinds})


SPECS: dict[str, Spec] = {
  "HoST-Ground": Spec(kinds=("ground",), recipes=(_SUPINE,)),
  "HoST-Prone": Spec(kinds=("ground",), recipes=(_PRONE,), prone=True, group_weights=P.GROUP_WEIGHTS_PRONE,
                     pull_needs_upright=False, kp_kd_scale=_KP_KD_PRONE),
  "HoST-Platform": Spec(kinds=("platform",), recipes=(_PLATFORM,),
                        world=terrains.world({"platform": terrains.PlatformTerrainCfg()}),
                        extra_terminations=_PLATFORM_TERM),
  "HoST-Wall": Spec(kinds=("wall",), recipes=(_SIT_WALL,), world=terrains.world({"wall": terrains.WallTerrainCfg()}),
                    shoulder=(0.0, 0.0), l2c2_value=P.L2C2_VALUE_WALL, extra_terminations=_WALL_TERM),
  "HoST-Slope": Spec(kinds=("slope",), recipes=(_SLOPE,), world=terrains.world({"slope": terrains.SlopeTerrainCfg()}),
                     ref="feet", stages=P.SLOPE_STAGES, shoulder=(0.15, -0.15),
                     extra_terminations={"below_feet": TerminationTermCfg(func=mdp.pelvis_below_feet,
                                                                          params={"height": -0.1})}),
  "HoST-Stairs": Spec(kinds=("stairs",), recipes=(_ON_STAIRS,), ref="ground",
                      world=terrains.world({"stairs": terrains.StairsTerrainCfg()})),
  "HoST-Supine-Mixed": Spec(
    kinds=_MIX_KINDS,
    recipes=(
      Recipe(postures={"supine": 0.6, **_sides(0.4)}, yaw_noise=PI),
      _PLATFORM,
      _SIT_WALL,
      Recipe(postures={"supine": 0.7, **_sides(0.3)}, yaws=(0.0, PI), yaw_noise=0.3,
             joint_scale=P.INIT_JOINT_SCALE_WIDE, jitter=(0.5, 0.0)),
      _ON_STAIRS,
    ),
    world=_mixed_world(_MIX_KINDS),
    ref="ground",
  ),
  "HoST-Prone-Mixed": Spec(
    kinds=("ground", "ramp", "stairs"),
    recipes=(
      Recipe(postures={"prone": 0.6, **_sides(0.4, prone=True)}, yaw_noise=PI),
      Recipe(postures={"prone": 0.7, **_sides(0.3, prone=True)}, yaws=(0.0, PI), yaw_noise=0.3, jitter=(0.5, 0.0)),
      Recipe(postures={"prone": 0.6, **_sides(0.4, prone=True)}, yaws=_ANY_YAW, yaw_noise=0.2, drop=0.6,
             unactuated=P.UNACTUATED_STEPS_OURS),
    ),
    world=_mixed_world(("ground", "ramp", "stairs")),
    ref="ground", prone=True, group_weights=P.GROUP_WEIGHTS_PRONE, pull_needs_upright=False, kp_kd_scale=_KP_KD_PRONE,
  ),
  "HoST-All": Spec(
    kinds=_MIX_KINDS,
    recipes=(
      Recipe(postures={"supine": 0.3, "prone": 0.3, "random": 0.2, **_sides(0.2)}, yaw_noise=PI,
             unactuated=P.UNACTUATED_STEPS_OURS),
      _PLATFORM,
      _SIT_WALL,
      Recipe(postures={"supine": 0.4, "prone": 0.4, **_sides(0.2)}, yaws=(0.0, PI), yaw_noise=0.3, jitter=(0.5, 0.0)),
      Recipe(postures={"supine": 0.3, "prone": 0.3, "random": 0.2, **_sides(0.2)}, yaws=_ANY_YAW, yaw_noise=0.2,
             drop=0.6, unactuated=P.UNACTUATED_STEPS_OURS),
    ),
    world=_mixed_world(_MIX_KINDS),
    ref="ground", pull_needs_upright=False,
  ),
}


def experiment_name(task_id: str) -> str:
  return task_id.lower().replace("-", "_")


# Per-experiment settings the runner needs that rsl_rl's config has no field for.
HOST_ALGO: dict[str, dict] = {
  experiment_name(t): {"group_weights": s.group_weights, "l2c2_value": s.l2c2_value, "task": t}
  for t, s in SPECS.items()
}


def runner_cfg(task_id: str) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=P.ACTOR_HIDDEN, activation="elu", obs_normalization=False,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": P.INIT_NOISE_STD,
                                          "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=P.CRITIC_HIDDEN, activation="elu", obs_normalization=False),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=P.VALUE_LOSS_COEF, use_clipped_value_loss=True, clip_param=P.CLIP_PARAM,
      entropy_coef=P.ENTROPY_COEF, num_learning_epochs=P.NUM_EPOCHS, num_mini_batches=P.NUM_MINI_BATCHES,
      learning_rate=P.LEARNING_RATE, schedule=P.SCHEDULE, gamma=P.GAMMA, lam=P.LAM, desired_kl=P.DESIRED_KL,
      max_grad_norm=P.MAX_GRAD_NORM,
    ),
    experiment_name=experiment_name(task_id),
    num_steps_per_env=P.STEPS_PER_ENV,
    max_iterations=P.ITERATIONS[task_id],
    save_interval=100,
    logger="tensorboard",
    upload_model=False,
  )


TASKS = {t: partial(make_env_cfg, s) for t, s in SPECS.items()}
