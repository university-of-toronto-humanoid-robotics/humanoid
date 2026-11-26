"""Collect four RPL experts and warm up the temporal CNN student with BC."""

import argparse

import _setup  # noqa: F401
from g1_comp.student_trainer import StudentTrainer, add_common_args


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  add_common_args(parser)
  StudentTrainer(parser.parse_args(), "bc").train()


if __name__ == "__main__":
  main()
