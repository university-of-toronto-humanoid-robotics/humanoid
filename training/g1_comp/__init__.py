"""Our G1 tasks for the IEEE Humanoids 2026 Loco-Manipulation Challenge.

Importing this package registers:
  G1-Comp-Flat    stage 1: walking only, on flat ground (fine-tune it on G1-Comp-Rough)
  G1-Comp-Rough   height-aware walking on rough terrain with competition-height stairs
  G1-Comp-Course  the same policy setup on the competition course (sim/competition/)
  G1-Comp-Crouch-Test  quick check of crouching + arm override on flat ground (small GPUs)

All of them share the same observations and actions, so a checkpoint from one plays on the others.
Needs Unitree's code on the path (training/unitree_rl_mjlab); the scripts in
training/scripts/ set that up.
"""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  g1_comp_course_env_cfg,
  g1_comp_crouch_test_env_cfg,
  g1_comp_flat_env_cfg,
  g1_comp_ppo_runner_cfg,
  g1_comp_rough_env_cfg,
)
from .tasks import TASKS, runner_cfg

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

# play.py --fpv / --ros: Unitree's own G1 tasks get the head camera too, so the
# operator has a first-person view with their walking models (models/g1_walk/).
from .camera import add_head_camera_to_unitree_tasks, fpv_enabled  # noqa: E402

if fpv_enabled():
  add_head_camera_to_unitree_tasks()
