"""Train a HoST task. Runs Unitree's scripts/train.py with the HoST tasks
registered, so it takes all of its options. From training/ inside the container:

  python g1_host/scripts/train.py HoST-Ground --env.scene.num-envs 4096
  python g1_host/scripts/train.py HoST-Supine-Mixed --from logs/rsl_rl/host_ground/<run>/model_<N>.pt

--from FILE starts from a saved model_<N>.pt (any HoST task: they all have the
same inputs and outputs). The file is copied into this task's log folder and
Unitree's resume options are filled in; the run continues its iteration count.

--continue NAME names the run NAME and makes the command safe to repeat (the
cluster jobs use it): it starts the task if no run NAME exists, continues it from
its newest checkpoint if it does, and does nothing if it already reached its
iterations (the task's own number, or --agent.max-iterations as the total).

Saves to training/logs/rsl_rl/<experiment>/<date>_<run name>/ (run from training/).
"""

import runpy
import shutil
import sys
from datetime import datetime
from pathlib import Path

import _setup
import src.tasks  # noqa: F401  (Unitree's tasks; its train.py expects them registered)
from _runs import latest_checkpoint
from mjlab.tasks.registry import load_rl_cfg


def _pop_option(argv: list[str], name: str) -> str | None:
  if name not in argv:
    return None
  i = argv.index(name)
  value = argv[i + 1]
  del argv[i:i + 2]
  return value


def use_continue_option(argv: list[str]) -> list[str]:
  args = list(argv)
  name = _pop_option(args, "--continue")
  if name is None:
    return argv
  cfg = load_rl_cfg(args[1])
  total = int(_pop_option(args, "--agent.max-iterations") or cfg.max_iterations)
  args += ["--agent.run-name", name]
  latest = latest_checkpoint(cfg.experiment_name, name)
  if latest is None:
    print(f"[INFO] {args[1]}: starting run '{name}', {total} iterations")
    return args + ["--agent.max-iterations", str(total)]
  ckpt, n = latest
  left = total - (n + 1)  # model_<n>.pt is saved after iteration n
  if left <= 0:
    print(f"[INFO] {args[1]}: run '{name}' already did its {total} iterations ({ckpt})")
    sys.exit(0)
  print(f"[INFO] {args[1]}: continuing run '{name}' from {ckpt}, {left} of {total} iterations left")
  return args + ["--agent.max-iterations", str(left), "--agent.resume", "True",
                 "--agent.load-run", ckpt.parent.name, "--agent.load-checkpoint", ckpt.name]


def use_from_option(argv: list[str]) -> list[str]:
  if "--from" not in argv:
    return argv
  i = argv.index("--from")
  source = Path(argv[i + 1]).resolve()
  if not source.is_file():
    raise SystemExit(f"--from: no such file: {source}")
  args = argv[:i] + argv[i + 2:]
  experiment = load_rl_cfg(args[1]).experiment_name
  run = f"from_{source.stem}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
  dest = Path("logs") / "rsl_rl" / experiment / run
  dest.mkdir(parents=True, exist_ok=True)
  shutil.copy2(source, dest / source.name)
  print(f"[INFO] Continuing from {source} (copied to {dest})")
  return args + ["--agent.resume", "True", "--agent.load-run", run, "--agent.load-checkpoint", source.name]


sys.argv = use_continue_option(use_from_option(sys.argv))
runpy.run_path(str(_setup.UNITREE / "scripts" / "train.py"), run_name="__main__")
