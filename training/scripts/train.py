"""Train a task (ours or Unitree's), optionally starting from a saved model.

Runs Unitree's scripts/train.py with our tasks registered first, so it takes all
of its options. From training/ inside the container:

  python scripts/train.py G1-Comp-Rough --env.scene.num-envs 512
  python scripts/train.py G1-Comp-Rough --from models/g1_comp_crouch/model_799.pt

--from FILE continues from any saved model_<N>.pt, also of another task (e.g. a
G1-Comp-Flat model on G1-Comp-Rough): the file is copied into this task's log
folder and Unitree's resume options are filled in for you. The copy's observation
normalizer is told to adapt quickly to the new task (NORMALIZER_MEMORY).

Saves to training/logs/rsl_rl/<experiment>/<date>_<run name>/. One GPU only:
with --gpu-ids for several GPUs, the worker processes would not see our tasks.
"""

import runpy
import shutil
import sys
from datetime import datetime
from pathlib import Path

import _setup
import src.tasks  # noqa: F401  (registers Unitree's tasks)
from mjlab.tasks.registry import load_rl_cfg


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
  shorten_normalizer_memory(dest / source.name)
  print(f"[INFO] Continuing from {source} (copied to {dest})")
  return args + ["--agent.resume", "True", "--agent.load-run", run, "--agent.load-checkpoint", source.name]


sys.argv = use_from_option(sys.argv)
runpy.run_path(str(_setup.UNITREE / "scripts" / "train.py"), run_name="__main__")
