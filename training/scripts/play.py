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
import secrets
import sys

_PLAY_ARGV = sys.argv.copy()

if "--terrain-level" in sys.argv:
  task = sys.argv[1].removesuffix("-Crouch") if len(sys.argv) > 1 else ""
  if task not in {"G1-Open-HeightMap", "Open", "Ramp", "Stairs"}:
    raise SystemExit("--terrain-level is supported for Open, Ramp, Stairs, and G1-Open-HeightMap")
  index = sys.argv.index("--terrain-level")
  if index + 1 >= len(sys.argv):
    raise SystemExit("--terrain-level needs a number from 0 to 9")
  try:
    level = int(sys.argv[index + 1])
  except ValueError as exc:
    raise SystemExit("--terrain-level needs a number from 0 to 9") from exc
  if not 0 <= level <= 9:
    raise SystemExit("--terrain-level needs a number from 0 to 9")
  os.environ["G1_PLAY_TERRAIN_LEVEL"] = str(level)
  os.environ.setdefault("G1_PLAY_TERRAIN_SEED", str(secrets.randbits(32)))
  print(f"[INFO] Level {level} lane seed: {os.environ['G1_PLAY_TERRAIN_SEED']}")
  del sys.argv[index:index + 2]

for flag in ("--fpv", "--ros"):
  if flag in sys.argv:
    sys.argv.remove(flag)
    os.environ["G1_FPV"] = "1"  # read when our tasks are registered (import _setup)
    if flag == "--ros":
      os.environ["G1_ROS"] = "1"

import _setup  # noqa: E402

if "G1_PLAY_TERRAIN_LEVEL" in os.environ:
  # The upstream player creates the environment before starting the viewer.
  # Reset it once after construction so the first rendered pose uses a lane
  # spawn patch rather than the default robot pose at the world origin.
  import mjlab.envs as _mjlab_envs

  class _ResetOnPlayEnv(_mjlab_envs.ManagerBasedRlEnv):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.reset()

  _mjlab_envs.ManagerBasedRlEnv = _ResetOnPlayEnv

  import mjlab.viewer as _mjlab_viewer
  from mjlab.viewer.base import ViewerAction

  class _RegeneratingViser(_mjlab_viewer.ViserPlayViewer):
    def setup(self):
      super().setup()
      with self._server.gui.add_folder("Terrain"):
        new_seed_button = self._server.gui.add_button("New Lane Seed")

        @new_seed_button.on_click
        def _(_):
          self.request_action("CUSTOM", "new_lane_seed")

    def _handle_custom_action(self, action, payload):
      if action != ViewerAction.CUSTOM or payload != "new_lane_seed":
        return super()._handle_custom_action(action, payload)
      old_seed = os.environ.get("G1_PLAY_TERRAIN_SEED")
      new_seed = str(secrets.randbits(32))
      while new_seed == old_seed:
        new_seed = str(secrets.randbits(32))
      new_env = os.environ.copy()
      new_env["G1_PLAY_TERRAIN_SEED"] = new_seed
      print(f"[INFO] Generating a new lane (seed {new_seed}); Viser will reconnect.", flush=True)
      self.close()
      self.env.close()
      os.execvpe(sys.executable, [sys.executable, *_PLAY_ARGV], new_env)
      return True

  _mjlab_viewer.ViserPlayViewer = _RegeneratingViser

if os.environ.get("G1_ROS") == "1":
  from g1_comp.camera import attach_ros_bridge

  attach_ros_bridge()

runpy.run_path(str(_setup.UNITREE / "scripts" / "play.py"), run_name="__main__")
