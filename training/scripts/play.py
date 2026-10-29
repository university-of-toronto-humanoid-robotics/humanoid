"""Play a trained policy on any task, including ours (G1-Comp-Rough, G1-Comp-Course).

Runs Unitree's scripts/play.py unchanged, with our tasks registered first. From
training/ inside the container:

  python scripts/play.py G1-Comp-Course --viewer viser --checkpoint-file logs/rsl_rl/g1_comp_rough/<date>/model_<N>.pt

Then open http://localhost:8080 (Commands -> Twist -> Enable to steer).
"""

import runpy

import _setup

runpy.run_path(str(_setup.UNITREE / "scripts" / "play.py"), run_name="__main__")
