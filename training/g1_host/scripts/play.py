"""Watch a HoST policy (or the robots falling, with --agent zero). Runs Unitree's
scripts/play.py with the HoST tasks registered. From training/ inside the container:

  python g1_host/scripts/play.py HoST-Ground --viewer viser --checkpoint-file logs/rsl_rl/host_ground/<run>/model_<N>.pt
  python g1_host/scripts/play.py HoST-Supine-Mixed --agent zero --viewer viser

Then open http://localhost:8080. Play uses the final action bound (0.25), no pull
force, no noise and no randomization; episodes are 10 s, then the robots fall again.
"""

import runpy

import _setup

runpy.run_path(str(_setup.UNITREE / "scripts" / "play.py"), run_name="__main__")
