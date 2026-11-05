"""Our G1 tasks for the IEEE Humanoids 2026 Loco-Manipulation Challenge.

Importing this package registers:
  G1-Comp-Rough   height-aware walking on rough terrain with competition-height stairs
  G1-Comp-Course  the same policy setup on the competition course (sim/competition/)
  G1-Comp-Crouch-Test  quick check of crouching + arm override on flat ground (small GPUs)

All three share the same observations and actions, so a checkpoint from one plays on the others.
Needs Unitree's code on the path (training/unitree_rl_mjlab); the scripts in
training/scripts/ set that up.
"""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  g1_comp_course_env_cfg,
  g1_comp_crouch_test_env_cfg,
  g1_comp_ppo_runner_cfg,
  g1_comp_rough_env_cfg,
)

register_mjlab_task(
  task_id="G1-Comp-Rough",
  env_cfg=g1_comp_rough_env_cfg(),
  play_env_cfg=g1_comp_rough_env_cfg(play=True),
  rl_cfg=g1_comp_ppo_runner_cfg("g1_comp_rough"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="G1-Comp-Course",
  env_cfg=g1_comp_course_env_cfg(),
  play_env_cfg=g1_comp_course_env_cfg(play=True),
  rl_cfg=g1_comp_ppo_runner_cfg("g1_comp_course"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="G1-Comp-Crouch-Test",
  env_cfg=g1_comp_crouch_test_env_cfg(),
  play_env_cfg=g1_comp_crouch_test_env_cfg(play=True),
  rl_cfg=g1_comp_ppo_runner_cfg("g1_comp_crouch_test"),
  runner_cls=VelocityOnPolicyRunner,
)
