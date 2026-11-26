"""Fall recovery for the G1: HoST ("Learning Humanoid Standing-up Control across
Diverse Postures", RSS 2025) ported to our mjlab stack.

Importing this package registers the HoST-* tasks (tasks.py) with HoST's runner
(runner.py). Everything for this policy lives in this folder; the walking tasks
(g1_comp) are untouched. Start with README.md.
"""

from mjlab.tasks.registry import register_mjlab_task

from .runner import HostRunner
from .tasks import TASKS, runner_cfg

for _task_id, _make in TASKS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=_make(),
    play_env_cfg=_make(play=True),
    rl_cfg=runner_cfg(_task_id),
    runner_cls=HostRunner,
  )
