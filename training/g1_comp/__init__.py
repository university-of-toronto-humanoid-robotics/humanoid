"""Our G1 tasks for the IEEE Humanoids 2026 Loco-Manipulation Challenge.

Importing this package registers:
  G1-Flat-HeightMap  original flat walking with a flattened height map
  G1-Rough-HeightMap  compatible rough terrain for the flat checkpoint
  G1-Course-HeightMap  competition course for that checkpoint
  G1-Comp-Flat    stage 1: walking only, on flat ground (fine-tune it on G1-Comp-Rough)
  G1-Comp-Rough   height-aware walking on rough terrain with competition-height stairs
  G1-Comp-Course  the same policy setup on the competition course (sim/competition/)
  G1-Comp-Crouch-Test  quick check of crouching + arm override on flat ground (small GPUs)

Flat, Ramp, Stairs, Open, Rough, VeryRough, Competition and their -Crouch
variants share the new policy layout. G1-Flat-HeightMap, G1-Rough-HeightMap,
and G1-Course-HeightMap share the older height-map layout. G1-Comp-* is a
third layout. Checkpoints from different families are not interchangeable.
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
  g1_comp_flat_env_cfg,
  g1_comp_ppo_runner_cfg,
  g1_comp_rough_env_cfg,
  g1_flat_heightmap_env_cfg,
  g1_flat_heightmap_ppo_runner_cfg,
  g1_rough_heightmap_env_cfg,
  g1_rough_heightmap_ppo_runner_cfg,
)
from .tasks import TASKS, runner_cfg

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
  task_id="G1-Comp-Flat",
  env_cfg=g1_comp_flat_env_cfg(),
  play_env_cfg=g1_comp_flat_env_cfg(play=True),
  rl_cfg=g1_comp_ppo_runner_cfg("g1_comp_flat"),
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

# The new tasks (tasks.py): Flat, Ramp, Stairs, Open, Rough, VeryRough, Competition,
# and each again as <Task>-Crouch.
for _task_id, _make in TASKS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=_make(),
    play_env_cfg=_make(play=True),
    rl_cfg=runner_cfg(_task_id),
    runner_cls=VelocityOnPolicyRunner,
  )
