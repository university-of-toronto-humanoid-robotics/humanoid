"""First-person view: the G1's head camera in simulation, and a ROS 2 bridge for it.

In the competition the operator may not look at the robot while it walks, only
at what it sees. This puts a camera where the G1's RealSense D435i is (Unitree's
URDF, d435_joint: in the head, pitched 47.6 degrees down) and shows its image:

  --fpv   in the browser viewer, under "Camera Feeds" (color and depth)
  --ros   the same, plus ROS 2 topics, and the robot follows /cmd_vel (standing
          still until the first one, and 0.5 s after the last):
            /g1/head_camera/image_raw         sensor_msgs/Image, rgb8
            /g1/head_camera/depth/image_raw   sensor_msgs/Image, 32FC1 (meters)
            /g1/head_camera/camera_info       sensor_msgs/CameraInfo
            /cmd_vel                          geometry_msgs/Twist (linear x, y; angular z)
            /g1/velocity                      geometry_msgs/TwistStamped, measured, body frame
            /g1/tilt                          std_msgs/Float32, degrees from upright
          The operator drives with ros2_ws/src/teleop (its README has the steps).

Both are options of scripts/play.py. Only while playing: training never renders.
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mjlab.sensor import CameraSensorCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

CAMERA_NAME = "head_camera"
# The D435i in the torso frame (x forward, y left, z up), from Unitree's g1_29dof URDF.
CAMERA_POS = (0.0576235, 0.01753, 0.42987)
CAMERA_PITCH = float(os.environ.get("G1_FPV_PITCH", math.degrees(0.8307767)))  # degrees below level
CAMERA_FOVY = 42.0  # degrees, the D435i color camera (69 x 42)
CAMERA_SIZE = (int(os.environ.get("G1_FPV_WIDTH", 640)), int(os.environ.get("G1_FPV_HEIGHT", 360)))
CMD_VEL_TIMEOUT = 0.5  # s without a /cmd_vel message before the robot is told to stop


def fpv_enabled() -> bool:
  return os.environ.get("G1_FPV", "0") == "1"


def head_camera_cfg() -> CameraSensorCfg:
  """The head camera, looking forward and `CAMERA_PITCH` degrees down."""
  # MuJoCo cameras look along their -z axis, with +x right and +y up.
  pitch = math.radians(CAMERA_PITCH)
  right, up = (0.0, -1.0, 0.0), (math.sin(pitch), 0.0, math.cos(pitch))
  back = (-math.cos(pitch), 0.0, math.sin(pitch))
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, np.array([right, up, back]).T.flatten())
  return CameraSensorCfg(
    name=CAMERA_NAME,
    parent_body="robot/torso_link",
    pos=CAMERA_POS,
    quat=tuple(float(q) for q in quat),
    fovy=CAMERA_FOVY,
    width=CAMERA_SIZE[0],
    height=CAMERA_SIZE[1],
    data_types=("rgb", "depth"),
  )


UNITREE_G1_TASKS = ("Unitree-G1-Flat", "Unitree-G1-Rough")


def add_head_camera_to_unitree_tasks() -> None:
  """Add the head camera to the play configs of Unitree's G1 tasks.

  Only rendering: the camera isn't one of the policy's inputs, so their trained
  models run unchanged. mjlab has no public way to edit a registered task, so this
  edits the stored play config (training configs are left alone).
  """
  from mjlab.tasks.registry import _REGISTRY

  for task_id in UNITREE_G1_TASKS:
    if task_id in _REGISTRY:
      scene = _REGISTRY[task_id].play_env_cfg.scene
      if all(s.name != CAMERA_NAME for s in scene.sensors or ()):
        scene.sensors = (*(scene.sensors or ()), head_camera_cfg())


class RosBridge:
  """Publishes one robot's head camera and drives it from /cmd_vel.

  Needs ROS 2 on the Python path: `source /opt/ros/jazzy/setup.bash` before
  activating the mjlab environment (the container's shell already does).
  """

  def __init__(self, env: ManagerBasedRlEnv, env_idx: int = 0, rate_hz: float = 30.0):
    import rclpy
    from geometry_msgs.msg import Twist, TwistStamped
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Float32

    self.env, self.env_idx = env, env_idx
    self.period = 1.0 / rate_hz
    self.last_publish = 0.0
    self.cmd: tuple[float, float, float] | None = None
    self.cmd_time = 0.0
    self._Image, self._CameraInfo = Image, CameraInfo
    self._TwistStamped, self._Float32 = TwistStamped, Float32

    if not rclpy.ok():
      rclpy.init()
    self.rclpy = rclpy
    self.node = rclpy.create_node("g1_sim")
    self.image_pub = self.node.create_publisher(Image, "/g1/head_camera/image_raw", 2)
    self.depth_pub = self.node.create_publisher(Image, "/g1/head_camera/depth/image_raw", 2)
    self.info_pub = self.node.create_publisher(CameraInfo, "/g1/head_camera/camera_info", 2)
    self.node.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 2)
    # Robot state for the operator, who may not look at the robot.
    self.velocity_pub = self.node.create_publisher(TwistStamped, "/g1/velocity", 2)
    self.tilt_pub = self.node.create_publisher(Float32, "/g1/tilt", 2)

    # The robot follows /cmd_vel while messages keep coming (the same way the
    # viewer's Twist sliders work): set the command right after it is computed.
    # With no fresh /cmd_vel, including before the first one, it stands still
    # instead of following play.py's random commands.
    twist = env.command_manager.get_term("twist")
    compute = twist.compute

    def compute_with_cmd_vel(dt: float) -> None:
      compute(dt)
      fresh = self.cmd is not None and time.monotonic() - self.cmd_time < CMD_VEL_TIMEOUT
      for i, v in enumerate(self.cmd if fresh else (0.0, 0.0, 0.0)):
        twist.vel_command_b[self.env_idx, i] = v

    twist.compute = compute_with_cmd_vel
    print("[INFO] ROS 2 bridge: publishing /g1/head_camera/*, listening on /cmd_vel")

  def _on_cmd_vel(self, msg) -> None:
    self.cmd = (msg.linear.x, msg.linear.y, msg.angular.z)
    self.cmd_time = time.monotonic()

  def _image(self, array: np.ndarray, encoding: str, stamp):
    msg = self._Image()
    msg.header.stamp = stamp
    msg.header.frame_id = "d435_link"
    msg.height, msg.width = array.shape[:2]
    msg.encoding = encoding
    msg.step = array.strides[0]
    msg.data = array.tobytes()
    return msg

  def _publish_robot_state(self, stamp) -> None:
    robot = self.env.scene["robot"].data
    lin = robot.root_link_lin_vel_b[self.env_idx].tolist()
    ang = robot.root_link_ang_vel_b[self.env_idx].tolist()
    msg = self._TwistStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = "pelvis"
    msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = lin
    msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = ang
    self.velocity_pub.publish(msg)
    # Gravity in the body frame is (0, 0, -1) when upright.
    gz = float(robot.projected_gravity_b[self.env_idx, 2])
    self.tilt_pub.publish(self._Float32(data=math.degrees(math.acos(max(-1.0, min(1.0, -gz))))))

  def update(self) -> None:
    """Call after every environment step."""
    self.rclpy.spin_once(self.node, timeout_sec=0.0)
    now = time.monotonic()
    if now - self.last_publish < self.period:
      return
    self.last_publish = now
    stamp = self.node.get_clock().now().to_msg()
    self._publish_robot_state(stamp)
    try:
      data = self.env.scene[CAMERA_NAME].data
    except KeyError:  # a task without our head camera (e.g. Unitree-G1-Flat): drive and status only
      return
    if data.rgb is not None:
      rgb = np.ascontiguousarray(data.rgb[self.env_idx].cpu().numpy())
      self.image_pub.publish(self._image(rgb, "rgb8", stamp))
    if data.depth is not None:
      depth = np.ascontiguousarray(data.depth[self.env_idx, ..., 0].cpu().numpy().astype(np.float32))
      self.depth_pub.publish(self._image(depth, "32FC1", stamp))
    width, height = CAMERA_SIZE
    focal = height / 2 / math.tan(math.radians(CAMERA_FOVY) / 2)
    info = self._CameraInfo()
    info.header.stamp = stamp
    info.header.frame_id = "d435_link"
    info.width, info.height = width, height
    info.distortion_model = "plumb_bob"
    info.d = [0.0] * 5
    info.k = [focal, 0.0, width / 2, 0.0, focal, height / 2, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [focal, 0.0, width / 2, 0.0, 0.0, focal, height / 2, 0.0, 0.0, 0.0, 1.0, 0.0]
    self.info_pub.publish(info)


def attach_ros_bridge() -> None:
  """Make every environment created from now on publish through a RosBridge."""
  from mjlab.envs import ManagerBasedRlEnv

  step = ManagerBasedRlEnv.step

  def step_and_publish(self, action):
    out = step(self, action)
    if not hasattr(self, "_g1_ros_bridge"):
      self._g1_ros_bridge = RosBridge(self)
    self._g1_ros_bridge.update()
    return out

  ManagerBasedRlEnv.step = step_and_publish
