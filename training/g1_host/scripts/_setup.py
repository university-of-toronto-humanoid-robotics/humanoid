"""Put Unitree's code and training/ on the path and register the HoST-* tasks only."""

import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[2]
UNITREE = TRAINING / "unitree_rl_mjlab"
for path in (str(UNITREE), str(TRAINING)):
  if path not in sys.path:
    sys.path.insert(0, path)

import g1_host  # noqa: E402,F401  (registers HoST-Ground, HoST-Prone, ...)
