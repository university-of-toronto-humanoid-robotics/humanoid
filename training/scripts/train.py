"""Train a task (ours or Unitree's), optionally starting from a saved model.

Runs Unitree's scripts/train.py with our tasks registered first, so it takes all
of its options. From training/ inside the container:

  python scripts/train.py G1-Comp-Rough --env.scene.num-envs 512
  python scripts/train.py G1-Comp-Rough --from models/g1_comp_crouch/model_799.pt

--from FILE continues from a model_<N>.pt with the same policy observation and
action layout (e.g. G1-Comp-Flat to G1-Comp-Rough, or Flat to Open). The file
is copied into this task's log folder and Unitree's resume options are filled
in for you. For the newer and RPL task families, the copy's observation
normalizer adapts quickly to the new task (NORMALIZER_MEMORY). The older
G1-*-HeightMap tasks keep their original normalizer behavior.

Saves to training/logs/rsl_rl/<experiment>/<date>_<run name>/. One GPU only:
with --gpu-ids for several GPUs, the worker processes would not see our tasks.
"""

import os
import runpy
import shutil
import sys
from datetime import datetime
from pathlib import Path

start_terrain_level = None
if "--start-terrain-level" in sys.argv:
  index = sys.argv.index("--start-terrain-level")
  if index + 1 >= len(sys.argv):
    raise SystemExit("--start-terrain-level needs an integer from 0 to 9")
  try:
    start_terrain_level = int(sys.argv[index + 1])
  except ValueError as exc:
    raise SystemExit("--start-terrain-level needs an integer from 0 to 9") from exc
  if not 0 <= start_terrain_level <= 9:
    raise SystemExit("--start-terrain-level needs an integer from 0 to 9")
  os.environ["G1_TRAIN_START_LEVEL"] = str(start_terrain_level)
  del sys.argv[index:index + 2]

import _setup
import src.tasks  # noqa: F401  (registers Unitree's tasks)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from g1_comp.tasks import TASKS
from g1_comp.rpl import RPL_TASKS
from g1_comp.rpl_cnn import TASK as RPL_CNN_ASCEND_TASK

if start_terrain_level is not None and "start_terrain_level" not in load_env_cfg(sys.argv[1]).events:
  raise SystemExit("--start-terrain-level is supported for the Open height-map and lane tasks")


# Samples the observation normalizer counts as "already seen" when continuing with
# --from (~100 iterations at 1024 robots). It averages over every sample since the
# start, so after thousands of flat-ground iterations it would take thousands more
# to learn what stairs look like in the height scan. Its averages are kept.
NORMALIZER_MEMORY = 2_500_000


def shorten_normalizer_memory(path: Path) -> None:
  import torch

  model = torch.load(path, map_location="cpu", weights_only=False)
  for part in ("actor_state_dict", "critic_state_dict"):
    count = model.get(part, {}).get("obs_normalizer.count")
    if count is not None and count > NORMALIZER_MEMORY:
      count.fill_(NORMALIZER_MEMORY)
  torch.save(model, path)


def use_from_option(argv: list[str]) -> list[str]:
  """Turn `--from FILE` into Unitree's --agent.resume / --agent.load-run options."""
  if "--from" not in argv:
    return argv
  i = argv.index("--from")
  source = Path(argv[i + 1]).resolve()
  if not source.is_file():
    raise SystemExit(f"--from: no such file: {source}")
  args = argv[:i] + argv[i + 2 :]
  task = args[1]
  experiment = load_rl_cfg(task).experiment_name
  run = f"from_{source.stem}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
  dest = Path("logs") / "rsl_rl" / experiment / run
  dest.mkdir(parents=True, exist_ok=True)
  shutil.copy2(source, dest / source.name)
  if task in TASKS or task in RPL_TASKS or task == RPL_CNN_ASCEND_TASK or task.startswith("G1-Comp-"):
    shorten_normalizer_memory(dest / source.name)
  print(f"[INFO] Continuing from {source} (copied to {dest})")
  return args + ["--agent.resume", "True", "--agent.load-run", run, "--agent.load-checkpoint", source.name]


sys.argv = use_from_option(sys.argv)
runpy.run_path(str(_setup.UNITREE / "scripts" / "train.py"), run_name="__main__")
