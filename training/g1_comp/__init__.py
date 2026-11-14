"""Our G1 tasks for the IEEE Humanoids 2026 Loco-Manipulation Challenge.

Importing this package registers:
  G1-Flat-HeightMap  Unitree's flat walking setup with the flattened competition scan
  G1-Rough-HeightMap  compatible rough-terrain walking, without crouch/arm override
  G1-Course-HeightMap  compatible policy evaluation on the competition CAD course
  G1-Comp-Rough   height-aware walking on rough terrain with competition-height stairs
  G1-Comp-Course  the same policy setup on the competition course (sim/competition/)
  G1-Comp-Crouch-Test  quick check of crouching + arm override on flat ground (small GPUs)

The three G1-Comp-* tasks share observations and actions, so their checkpoints
play on each other. G1-Flat-HeightMap and G1-Rough-HeightMap share a separate
observation layout and can continue from each other's checkpoints. They are
also distinct from Unitree-G1-Flat.
Needs Unitree's code on the path (training/unitree_rl_mjlab); the scripts in
training/scripts/ set that up.
"""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  g1_course_heightmap_env_cfg,
  g1_course_heightmap_ppo_runner_cfg,
  g1_comp_course_env_cfg,
  g1_comp_crouch_test_env_cfg,
  g1_comp_ppo_runner_cfg,
  g1_comp_rough_env_cfg,
  g1_flat_heightmap_env_cfg,
  g1_flat_heightmap_ppo_runner_cfg,
  g1_rough_heightmap_env_cfg,
  g1_rough_heightmap_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="G1-Flat-HeightMap",
  env_cfg=g1_flat_heightmap_env_cfg(),
  play_env_cfg=g1_flat_heightmap_env_cfg(play=True),
  rl_cfg=g1_flat_heightmap_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="G1-Rough-HeightMap",
  env_cfg=g1_rough_heightmap_env_cfg(),
  play_env_cfg=g1_rough_heightmap_env_cfg(play=True),
  rl_cfg=g1_rough_heightmap_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="G1-Course-HeightMap",
  env_cfg=g1_course_heightmap_env_cfg(),
  play_env_cfg=g1_course_heightmap_env_cfg(play=True),
  rl_cfg=g1_course_heightmap_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
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
