"""CNN stair expert: existing RPL stair world, separate RPL-inspired rewards."""

from __future__ import annotations

from copy import deepcopy

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from . import rpl_cnn_reward as terms
from .rpl import rpl_env_cfg, rpl_runner_cfg
from .sensors import rpl_foot_safety_sensor_cfgs

TASK = "RPL-CNN-Ascend-Stairs"

# (name, function, weight, scaled by RPL penalty curriculum, parameters)
# Zero-weight contact_pattern can be enabled after checking natural double support.
REWARD_SPEC = (
  ("tracking_lin_vel_x", terms.track_axis, 6.0, False, {"index": 0}),
  ("tracking_lin_vel_y", terms.track_axis, 1.5, False, {"index": 1}),
  ("tracking_ang_vel_z", terms.track_axis, 4.0, False, {"index": 2, "angular": True}),
  ("penalty_stumble", terms.stumble, -50.0, False, {"ratio": 0.5, "minimum": 5.0}),
  ("penalty_foothold", terms.foothold, -10.0, True, {}),
  ("penalty_contact_no_vel", terms.contact_velocity, -1.0, False, {"feet": terms.FEET}),
  ("penalty_feet_ori", terms.foot_orientation, -1.0, True, {"feet": terms.FEET}),
  ("feet_air_time", terms.AirTime(), 1.0, False, {}),
  ("penalty_diff_feet_air_time", terms.air_time_difference, -2.5, False, {}),
  ("penalty_feet_contact_forces", terms.excess_foot_force, -0.01, True, {"max_contact_force": 350.0}),
  # RPL uses -4; leave disabled until double-support statistics are reviewed.
  ("penalty_contact_pattern", terms.contact_pattern, 0.0, False, {}),
  ("penalty_close_feet_xy", terms.close_feet, -10.0, False, {"sites": terms.FOOT_SITES}),
  ("penalty_close_knees_xy", terms.close_knees, -10.0, False, {"knees": terms.KNEES}),
  ("feet_heading_alignment", terms.foot_heading_alignment, -0.25, True, {"feet": terms.FEET}),
  ("base_height", terms.base_height, -25.0, False, {"target": 0.8}),
  ("penalty_orientation", terms.base_orientation, -1.5, False, {}),
  ("penalty_base_roll", terms.base_roll, -1.0, False, {}),
  ("penalty_base_pitch", terms.base_pitch, -1.0, False, {}),
  ("penalty_ang_vel_xy_base", terms.angular_velocity_xy, -0.05, False, {}),
  ("penalty_ang_vel_xy_torso", terms.angular_velocity_xy, -1.0, False,
   {"torso": True, "torso_cfg": terms.TORSO}),
  ("penalty_lin_vel_z", terms.vertical_velocity, -2.0, False, {}),
  ("penalty_lower_body_torques", terms.torque_l2, -4e-5, True, {"asset_cfg": terms.LOWER_ACT}),
  ("penalty_lower_body_dof_vel", terms.joint_velocity_l2, -1e-4, True, {"asset_cfg": terms.LOWER}),
  ("penalty_lower_body_dof_acc", terms.ControlStepAcceleration(), -2.5e-7, True,
   {"asset_cfg": terms.LOWER}),
  ("penalty_lower_body_action_rate", terms.action_rate, -0.5, True, {"asset_cfg": terms.LOWER}),
  ("limits_lower_body_dof_pos", terms.joint_limits, -5.0, True, {"asset_cfg": terms.LOWER}),
  ("penalty_hip_pos", terms.hip_position, -2.5, False, {"asset_cfg": terms.HIPS}),
  ("penalty_ankle_roll", terms.ankle_roll, -2.0, False, {"asset_cfg": terms.ANKLES}),
  ("penalty_negative_knee_joint", terms.negative_knee, -1.0, False,
   {"asset_cfg": terms.KNEE_JOINTS}),
  ("upper_body_neutral_pose", terms.neutral_arm_pose, -1.0, True, {"asset_cfg": terms.ARMS}),
  ("upper_body_dof_vel", terms.joint_velocity_l2, -0.001, True, {"asset_cfg": terms.ARMS}),
  ("upper_body_action_rate", terms.action_rate, -0.5, True, {"asset_cfg": terms.ARMS}),
  ("upper_body_torques", terms.torque_l2, -1e-5, True, {"asset_cfg": terms.ARM_ACT}),
  ("termination", terms.failure_termination, -250.0, False, {}),
)


def cnn_ascend_env_cfg(play: bool = False):
  cfg = rpl_env_cfg("ascend", play=play, stumble_enabled=False, foothold_enabled=False)
  cfg.scene.sensors = (cfg.scene.sensors or ()) + rpl_foot_safety_sensor_cfgs()
  # rpl_env_cfg is the shared world/commands/curriculum source; its original
  # reward set is replaced only on this new task.
  cfg.rewards = {}
  cfg.metrics = {"traversal_success": cfg.metrics["traversal_success"]}
  cfg.curriculum = dict(cfg.curriculum or {})
  if not play:
    cfg.curriculum["rpl_penalty_scale"] = CurriculumTermCfg(func=terms.PenaltyCurriculum(), params={
      "initial_scale": 0.1, "degree": 1e-5, "down": 40, "up": 210,
    })
  for name, func, weight, scaled, params in REWARD_SPEC:
    cfg.rewards[name] = RewardTermCfg(func=terms.RawReward(name, func, scaled), weight=weight,
                                      params=deepcopy(params))
    if weight:
      cfg.metrics[f"raw_{name}"] = MetricsTermCfg(func=terms.raw_metric, params={"name": name})
  cfg.metrics["penalty_scale"] = MetricsTermCfg(func=terms.penalty_scale_metric)
  cfg.metrics["forward_velocity_error"] = MetricsTermCfg(func=terms.forward_error)
  cfg.metrics["yaw_rate_error"] = MetricsTermCfg(func=terms.yaw_error)
  cfg.metrics["sliding_contact_fraction"] = MetricsTermCfg(func=terms.sliding_fraction,
      params={"feet": deepcopy(terms.FEET)})
  cfg.metrics["contact_pattern_rate"] = MetricsTermCfg(func=terms.contact_pattern)
  cfg.metrics["foot_air_time_left"] = MetricsTermCfg(func=terms.air_time_last, params={"side": 0})
  cfg.metrics["foot_air_time_right"] = MetricsTermCfg(func=terms.air_time_last, params={"side": 1})
  return cfg


def cnn_ascend_runner_cfg():
  cfg = rpl_runner_cfg(TASK)
  cfg.actor.class_name = "g1_comp.rpl_cnn_actor:RplCnnActor"
  cfg.actor.hidden_dims = (512, 256, 256, 128)
  # Critic retains the old 806-value MLP and does not share encoder weights.
  return cfg
