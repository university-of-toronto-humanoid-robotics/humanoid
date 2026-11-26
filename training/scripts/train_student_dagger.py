"""Train the temporal CNN on student rollouts labeled by four RPL experts."""

import argparse
from pathlib import Path

import _setup  # noqa: F401
from g1_comp.student_trainer import StudentTrainer, add_common_args


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  add_common_args(parser)
  parser.add_argument("--init-mode", choices=("bc-with-demos", "bc-weights", "scratch"))
  parser.add_argument("--bc-checkpoint", type=Path)
  parser.add_argument("--bc-demo-checkpoint", type=Path,
                      help="BC demos_*.pt file; defaults to the file paired with --bc-checkpoint")
  StudentTrainer(parser.parse_args(), "dagger").train()


if __name__ == "__main__":
  main()
