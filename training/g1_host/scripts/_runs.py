"""Find the newest checkpoint of a named run (train.py --continue, eval.py --run)."""

from __future__ import annotations

from pathlib import Path

LOGS = Path("logs") / "rsl_rl"  # relative to where the scripts run from, as Unitree's train.py


def latest_checkpoint(experiment: str, run_name: str) -> tuple[Path, int] | None:
  """(model_<N>.pt with the highest N, N) over all runs <date>_<run_name> of the
  experiment, or None if there is none yet."""
  best: tuple[Path, int] | None = None
  for run in (LOGS / experiment).glob(f"*_{run_name}"):
    for ckpt in run.glob("model_*.pt"):
      n = int(ckpt.stem.split("_")[1])
      if best is None or n > best[1]:
        best = (ckpt, n)
  return best
