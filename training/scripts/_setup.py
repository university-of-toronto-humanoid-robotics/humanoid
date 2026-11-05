"""Put Unitree's code and ours on the path, and register the G1-Comp-* tasks."""

import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
UNITREE = TRAINING / "unitree_rl_mjlab"
for path in (str(UNITREE), str(TRAINING)):
  if path not in sys.path:
    sys.path.insert(0, path)

import g1_comp  # noqa: E402,F401  (registers G1-Comp-Rough and G1-Comp-Course)
