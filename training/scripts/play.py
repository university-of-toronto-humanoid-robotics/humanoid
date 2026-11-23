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

import numpy as np

_PLAY_ARGV = sys.argv.copy()

if "--terrain-level" in sys.argv:
  task = sys.argv[1].removesuffix("-Crouch") if len(sys.argv) > 1 else ""
  if task not in {"G1-Open-HeightMap", "G1-Open-Maze-HeightMap", "Open", "Ramp", "Stairs",
                  "RPL-Ascend-Stairs", "RPL-Descend-Stairs", "RPL-Ramps", "RPL-Obstacles",
                  "RPL-CNN-Ascend-Stairs"}:
    raise SystemExit("--terrain-level is supported for Open, Ramp, Stairs, G1-Open-*-HeightMap, and RPL tasks")
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
from g1_comp import lanes  # noqa: E402

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
      manager = self.env.unwrapped.command_manager
      term = manager.get_term("twist") if "twist" in manager.active_terms else None
      self._maze_term = term if isinstance(term, lanes.MazeVelocityCommand) else None
      if self._maze_term is not None:
        self._maze_row = None
        self._maze_origin_y = None
        self._maze_path = None
        self._maze_waypoints = []
        self._maze_target = self._server.scene.add_icosphere(
          "/fixed_bodies/maze_route/active_target", radius=0.10,
          color=(255, 225, 55), position=(0, 0, 0.35), visible=False,
        )
        self._maze_aim = self._server.scene.add_line_segments(
          "/fixed_bodies/maze_route/aim",
          np.zeros((1, 2, 3), dtype=np.float32), colors=(255, 225, 55),
          thickness=0.025, visible=False,
        )
        with self._server.gui.add_folder("Maze route"):
          self._maze_info = self._server.gui.add_html("Waiting for the first command...")
        self._refresh_maze_route()
      with self._server.gui.add_folder("Terrain"):
        new_seed_button = self._server.gui.add_button("New Lane Seed")

        @new_seed_button.on_click
        def _(_):
          self.request_action("CUSTOM", "new_lane_seed")

    def _draw_maze_route(self, row: int, origin_y: float) -> None:
      term = self._maze_term
      assert term is not None
      if self._maze_path is not None:
        self._maze_path.remove()
      for marker, label in self._maze_waypoints:
        marker.remove()
        label.remove()
      self._maze_waypoints = []
      points = []
      for tile in range(term.cfg.num_rows):
        level = term.cfg.fixed_level if term.cfg.fixed_level is not None else tile
        tile_x = tile * term.cfg.tile_length - term.cfg.num_rows * term.cfg.tile_length / 2
        points.extend((tile_x + x, y, 0.35) for x, y in lanes.maze_route_tile_points(level, origin_y))
      xyz = np.asarray(points)
      self._maze_path = self._server.scene.add_line_segments(
        "/fixed_bodies/maze_route/path", np.stack((xyz[:-1], xyz[1:]), axis=1),
        colors=(0, 220, 190), thickness=0.018,
      )
      for number, (x, y) in enumerate(lanes.maze_route_tile_points(
        term.cfg.fixed_level if term.cfg.fixed_level is not None else row, origin_y
      )):
        world_x = row * term.cfg.tile_length - term.cfg.num_rows * term.cfg.tile_length / 2 + x
        marker = self._server.scene.add_icosphere(
          f"/fixed_bodies/maze_route/point_{number}", radius=0.045,
          color=(0, 220, 190), position=(world_x, y, 0.35),
        )
        label = self._server.scene.add_label(
          f"/fixed_bodies/maze_route/label_{number}", str(number),
          position=(world_x, y, 0.52),
        )
        self._maze_waypoints.append((marker, label))
      self._maze_row, self._maze_origin_y = row, origin_y

    def _refresh_maze_route(self) -> None:
      term = self._maze_term
      if term is None:
        return
      env_idx = self._scene.env_idx
      terrain = self.env.unwrapped.scene.terrain
      assert terrain is not None and terrain.env_origins is not None
      active_index = int(term.active_target_index[env_idx].item())
      row = active_index // len(lanes.MAZE_ROUTE_X)
      origin_y = float(terrain.env_origins[env_idx, 1].item())
      if self._maze_row != row or self._maze_origin_y != origin_y:
        self._draw_maze_route(row, origin_y)
      pending = bool(term.route_pending[env_idx].item())
      self._maze_target.visible = not pending
      self._maze_aim.visible = not pending
      if pending:
        self._maze_info.content = "Waiting for the first command..."
        return
      target = term.active_target_w[env_idx].detach().cpu().numpy()
      robot = term.robot.data.root_link_pos_w[env_idx, :2].detach().cpu().numpy()
      visible = bool(term.target_visible[env_idx].item())
      aligning = bool(term.target_is_alignment[env_idx].item())
      self._maze_target.position = (float(target[0]), float(target[1]), 0.42)
      color = (70, 160, 255) if aligning else ((255, 225, 55) if visible else (255, 70, 70))
      self._maze_target.color = color
      self._maze_aim.points = np.array([[[robot[0], robot[1], 0.42],
                                         [target[0], target[1], 0.42]]], dtype=np.float32)
      self._maze_aim.colors = np.array([[color, color]], dtype=np.uint8)
      point = active_index % len(lanes.MAZE_ROUTE_X)
      command = term.command[env_idx].detach().cpu().numpy()
      self._maze_info.content = (
        f"<b>Next waypoint:</b> {point} (tile {row})<br>"
        f"<b>Robot:</b> ({robot[0]:.2f}, {robot[1]:.2f}) m<br>"
        f"<b>Target:</b> ({target[0]:.2f}, {target[1]:.2f}) m<br>"
        f"<b>Path:</b> {'lateral alignment' if aligning else ('clear' if visible else 'blocked — speed zero')}<br>"
        f"<b>Command:</b> {command[0]:.2f} m/s forward, {command[2]:.2f} rad/s yaw"
      )

    def sync_env_to_viewer(self):
      super().sync_env_to_viewer()
      self._refresh_maze_route()

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
