"""Play a trained policy on any task (ours or Unitree's).

Runs Unitree's scripts/play.py with our tasks registered first, so it takes all of
its options. From training/ inside the container:

  python scripts/play.py Open --viewer viser --checkpoint-file logs/rsl_rl/open/<date>/model_<N>.pt

Then open http://localhost:8080 (Commands -> Twist -> Enable to steer).

Two options of ours, for the new tasks (Flat, Ramp, Stairs, Open, ...):

  --fpv   first-person view: the robot's head camera, shown in the viewer under
          "Camera Feeds" (color and depth).
  --ros   the same, and the camera is published on ROS 2 while the robot follows
          /cmd_vel (g1_comp/camera.py lists the topics). In two more terminals:
            ros2 run rqt_image_view rqt_image_view /g1/head_camera/image_raw
            ros2 run teleop_twist_keyboard teleop_twist_keyboard
"""

import os
import runpy
import sys

for flag in ("--fpv", "--ros"):
  if flag in sys.argv:
    sys.argv.remove(flag)
    os.environ["G1_FPV"] = "1"  # read when our tasks are registered (import _setup)
    if flag == "--ros":
      os.environ["G1_ROS"] = "1"

import _setup  # noqa: E402

if os.environ.get("G1_ROS") == "1":
  from g1_comp.camera import attach_ros_bridge

  attach_ros_bridge()

runpy.run_path(str(_setup.UNITREE / "scripts" / "play.py"), run_name="__main__")
