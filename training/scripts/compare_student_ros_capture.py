"""Replay the training student map on a recorded ROS room-stairs trajectory.

Run from training/ with the native mjlab environment:

  python scripts/compare_student_ros_capture.py ~/Downloads/20261125T220315Z_ros_mapping.tar.gz \
    --output-dir logs/ros_mapping_comparison

The archive comes from notebooks/collect_ros_data.ipynb. This script uses the
recorded joints and ground-truth LiDAR odometry to reconstruct the G1 pelvis,
raycasts the *same* MuJoCo scene as the ROS capture, and calls the actual
StudentHeightMap observation term. It never substitutes the recorded ROS map
for the student's simulated map. The expert policy is not rerun.

ROS Float32MultiArray scans have no header stamp, so they are matched by MCAP
record time. The report includes timing offsets and a reconstructed true-scan
check; inspect those before interpreting map differences.
"""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import asdict
import importlib.util
import io
import json
import html
from pathlib import Path
import struct
import sys
import tarfile
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

import _setup  # noqa: F401  (mjlab and training packages)
from g1_comp.student_mapping.observation import StudentHeightMap, StudentMapCfg


ROOT = Path(__file__).resolve().parents[2]
MAGIC = b"\x89MCAP0\r\n"
TOPICS = {
  "/height_scan", "/ground_truth/height_scan", "/ground_truth/odom",
  "/joint_states", "/height_scan/observed_share", "/height_scan/feet_share",
  "/elevation_mapping_node/elevation_map_filter",
  "/tf", "/tf_static",
}


def _records(stream):
  """Read MCAP records without a ROS install; this capture uses plain chunks."""
  while True:
    header = stream.read(9)
    if not header or header == MAGIC:
      return
    if len(header) != 9:
      raise ValueError("Truncated MCAP record header")
    opcode, length = struct.unpack("<BQ", header)
    body = stream.read(length)
    if len(body) != length:
      raise ValueError("Truncated MCAP record body")
    yield opcode, body


def _mcap_stream(path: Path):
  if path.suffix != ".gz":
    return path.open("rb"), None
  archive = tarfile.open(path, "r:gz")
  members = [m for m in archive.getmembers() if m.name.endswith(".mcap")]
  if len(members) != 1:
    archive.close()
    raise ValueError(f"Expected one MCAP in {path}; found {len(members)}")
  return archive.extractfile(members[0]), archive


def capture_manifest(path: Path):
  if path.suffix != ".gz":
    return None
  with tarfile.open(path, "r:gz") as archive:
    members = [m for m in archive.getmembers() if m.name.endswith("manifest.json")]
    if len(members) != 1:
      raise ValueError("Capture archive has no unique manifest.json")
    manifest = json.load(archive.extractfile(members[0]))
  if manifest.get("scenarios") != ["room_stairs_heightmap"]:
    raise ValueError("This replay currently supports only the room_stairs_heightmap capture")
  if manifest.get("policy_scan_source") != "true_terrain":
    raise ValueError("Replay needs the expert to have used true terrain, not the live ROS map")
  return manifest


class Cdr:
  def __init__(self, data: bytes):
    if data[:4] != b"\x00\x01\x00\x00":
      raise ValueError("Only little-endian CDR messages are supported")
    self.data = data
    self.pos = 4

  def read(self, fmt: str):
    size = struct.calcsize(fmt)
    # CDR alignment starts *after* the four-byte encapsulation header.
    self.pos = 4 + (self.pos - 4 + size - 1) // size * size
    result = struct.unpack_from("<" + fmt, self.data, self.pos)[0]
    self.pos += size
    return result

  def string(self):
    n = self.read("I")
    if n > 1_000_000 or self.pos + n > len(self.data):
      raise ValueError(f"Invalid CDR string length {n} at byte {self.pos}")
    raw = self.data[self.pos:self.pos + n]
    self.pos += n
    return raw.rstrip(b"\x00").decode("utf-8")

  def float_array(self):
    n = self.read("I")
    if n > 1_000_000:
      raise ValueError(f"Invalid CDR float array length {n}")
    self.pos = (self.pos + 3) & ~3
    result = np.frombuffer(self.data, dtype="<f4", count=n, offset=self.pos).copy()
    self.pos += n * 4
    return result

  def double_array(self):
    n = self.read("I")
    if n > 100_000:
      raise ValueError(f"Invalid CDR double array length {n}")
    self.pos = 4 + ((self.pos - 4 + 7) & ~7)
    result = np.frombuffer(self.data, dtype="<f8", count=n, offset=self.pos).copy()
    self.pos += n * 8
    return result

  def header(self):
    sec, nsec = self.read("i"), self.read("I")
    frame = self.string()
    return sec + nsec * 1e-9, frame

  def multiarray(self):
    dims = []
    count = self.read("I")
    if count > 16:
      raise ValueError(f"Invalid CDR array dimension count {count}")
    for _ in range(count):
      dims.append((self.string(), self.read("I"), self.read("I")))
    self.read("I")  # data_offset
    return dims, self.float_array()


def _pose(c: Cdr):
  pos = np.array([c.read("d") for _ in range(3)])
  xyzw = np.array([c.read("d") for _ in range(4)])
  return pos, np.roll(xyzw, 1)  # MuJoCo quaternion order is wxyz


def _decode(topic: str, data: bytes):
  c = Cdr(data)
  if topic in ("/height_scan", "/ground_truth/height_scan"):
    _, values = c.multiarray()
    if values.size != 693:
      raise ValueError(f"{topic} has {values.size} values, expected 693")
    return values
  if topic.endswith("_share"):
    return c.read("f")
  if topic == "/ground_truth/odom":
    stamp, _ = c.header()
    c.string()  # child_frame_id
    return stamp, *_pose(c)
  if topic == "/joint_states":
    stamp, _ = c.header()
    count = c.read("I")
    if count > 200:
      raise ValueError(f"Invalid joint count {count}")
    names = [c.string() for _ in range(count)]
    positions = c.double_array()
    return stamp, dict(zip(names, positions, strict=True))
  if topic in ("/tf", "/tf_static"):
    count = c.read("I")
    if count > 200:
      raise ValueError(f"Invalid TF transform count {count}")
    transforms = []
    for _ in range(count):
      _, parent = c.header()
      child = c.string()
      pos, quat = _pose(c)
      transforms.append((parent, child, pos, quat))
    return transforms
  if topic.endswith("elevation_map_filter"):
    stamp, frame = c.header()
    resolution, length_x, length_y = (c.read("d") for _ in range(3))
    center, _ = _pose(c)
    count = c.read("I")
    if count > 64:
      raise ValueError(f"Invalid GridMap layer count {count}")
    layers = [c.string() for _ in range(count)]
    count = c.read("I")
    if count > 64:
      raise ValueError(f"Invalid GridMap basic layer count {count}")
    basic = [c.string() for _ in range(count)]
    count = c.read("I")
    if count > 64:
      raise ValueError(f"Invalid GridMap data count {count}")
    arrays = [c.multiarray() for _ in range(count)]
    outer, inner = c.read("H"), c.read("H")
    return {"stamp": stamp, "frame": frame, "resolution": resolution,
            "length": (length_x, length_y), "center": center,
            "layers": dict(zip(layers, arrays, strict=True)),
            "basic_layers": basic, "outer": outer, "inner": inner}
  raise ValueError(f"Unsupported topic {topic}")


def load_capture(path: Path):
  topics = {name: [] for name in TOPICS}
  stream, archive = _mcap_stream(path)
  try:
    if stream.read(8) != MAGIC:
      raise ValueError("Not an MCAP file")
    channels = {}
    for opcode, body in _records(stream):
      if opcode != 6:  # Chunk
        continue
      pos = 28  # start/end time, uncompressed size, CRC
      name_length = struct.unpack_from("<I", body, pos)[0]
      pos += 4
      compression = body[pos:pos + name_length].decode()
      pos += name_length
      nbytes = struct.unpack_from("<Q", body, pos)[0]
      pos += 8
      if compression:
        raise ValueError(f"MCAP compression {compression!r} is unsupported; record uncompressed")
      for record_type, record in _records(io.BytesIO(body[pos:pos + nbytes])):
        if record_type == 4:  # Channel
          channel_id = struct.unpack_from("<H", record)[0]
          topic_len = struct.unpack_from("<I", record, 4)[0]
          channels[channel_id] = record[8:8 + topic_len].decode()
        elif record_type == 5:  # Message
          channel_id, _, log_time, _ = struct.unpack_from("<HIQQ", record)
          topic = channels.get(channel_id)
          if topic in topics:
            topics[topic].append((log_time * 1e-9, _decode(topic, record[22:])))
  finally:
    stream.close()
    if archive is not None:
      archive.close()
  for key in topics:
    if not topics[key]:
      raise ValueError(f"Required topic {key} missing from capture")
  return topics


def nearest(messages, time):
  times = [entry[0] for entry in messages]
  i = bisect.bisect_left(times, time)
  if i == 0:
    return messages[0]
  if i == len(times):
    return messages[-1]
  return messages[i - 1] if time - times[i - 1] < times[i] - time else messages[i]


def latest(messages, time):
  times = [entry[0] for entry in messages]
  i = bisect.bisect_right(times, time) - 1
  return messages[i] if i >= 0 else None


def _quat_mul(a, b):
  out = np.empty(4)
  mujoco.mju_mulQuat(out, a, b)
  return out


def _quat_rot(q, vec):
  out = np.empty(3)
  mujoco.mju_rotVecQuat(out, vec, q)
  return out


def _quat_inv(q):
  out = np.empty(4)
  mujoco.mju_negQuat(out, q)
  return out


class TfTree:
  """The ROS transform tree at each bag record time."""

  def __init__(self, topics):
    self.dynamic = topics["/tf"]
    self.next_dynamic = 0
    self.edges = {}
    for _, transforms in topics["/tf_static"]:
      for parent, child, pos, quat in transforms:
        self.edges[child] = (parent, pos, quat)

  def advance(self, time):
    while self.next_dynamic < len(self.dynamic) and self.dynamic[self.next_dynamic][0] <= time:
      for parent, child, pos, quat in self.dynamic[self.next_dynamic][1]:
        self.edges[child] = (parent, pos, quat)
      self.next_dynamic += 1

  def pose(self, frame, root="odom"):
    pos = np.zeros(3)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    current = frame
    visited = set()
    while current != root:
      if current in visited or current not in self.edges:
        raise ValueError(f"No TF chain from {root} to {frame}: stopped at {current}")
      visited.add(current)
      parent, edge_pos, edge_quat = self.edges[current]
      pos = edge_pos + _quat_rot(edge_quat, pos)
      quat = _quat_mul(edge_quat, quat)
      current = parent
    return pos, quat


def _yaw(q):
  w, x, y, z = q
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _ros_scan_module():
  path = ROOT / "ros2_ws/src/perception/height_scan/height_scan/scan.py"
  spec = importlib.util.spec_from_file_location("ros_height_scan", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class Trajectory:
  """World pelvis/head/feet poses from relative LiDAR odometry and joints."""

  def __init__(self, first_joint, scene: Path, description: Path):
    sys.path[:0] = [str(ROOT / "ros2_ws/src/bridge"), str(ROOT / "ros2_ws/src/policy")]
    from bridge import g1_model
    self.model = g1_model.build_model(str(scene), str(description))
    self.data = mujoco.MjData(self.model)
    g1_model.spawn(self.model, self.data, 3.6, 2.5, 0.0)
    self.joints = {self.model.joint(i).name: self.model.jnt_qposadr[i]
                   for i in range(self.model.njnt)
                   if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE}
    self._set_joints(first_joint)
    mujoco.mj_forward(self.model, self.data)
    self.sensor_id = self.model.site("mid360_imu").id
    self.pelvis_id = self.model.body("pelvis").id
    self.torso_id = self.model.body("torso_link").id
    self.feet_ids = [self.model.body(n).id for n in
                     ("left_ankle_roll_link", "right_ankle_roll_link")]
    self.initial_site_pos = self.data.site_xpos[self.sensor_id].copy()
    self.initial_site_quat = self._site_quat()
    self.terrain_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    self.geom_id = np.zeros(1, dtype=np.int32)
    self.ray_dir = np.array([0.0, 0.0, -1.0])

  def _site_quat(self):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, self.data.site_xmat[self.sensor_id])
    return q

  def _set_joints(self, joints):
    for name, adr in self.joints.items():
      if name not in joints:
        raise ValueError(f"Missing joint {name} from /joint_states")
      self.data.qpos[adr] = joints[name]

  def update(self, odom, joints):
    _, relative_pos, relative_quat = odom
    self._set_joints(joints[1])
    self.data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_forward(self.model, self.data)
    site_local_pos = self.data.site_xpos[self.sensor_id].copy()
    site_local_quat = self._site_quat()
    site_world_quat = _quat_mul(self.initial_site_quat, relative_quat)
    site_world_pos = self.initial_site_pos + _quat_rot(self.initial_site_quat, relative_pos)
    root_quat = _quat_mul(site_world_quat, _quat_inv(site_local_quat))
    root_pos = site_world_pos - _quat_rot(root_quat, site_local_pos)
    self.data.qpos[:3] = root_pos
    self.data.qpos[3:7] = root_quat
    mujoco.mj_forward(self.model, self.data)
    pelvis_pos = self.data.xpos[self.pelvis_id].copy()
    pelvis_quat = self.data.xquat[self.pelvis_id].copy()
    torso_pos = self.data.xpos[self.torso_id].copy()
    torso_quat = self.data.xquat[self.torso_id].copy()
    feet = self.data.xpos[self.feet_ids, 2].copy()
    return pelvis_pos, pelvis_quat, torso_pos, torso_quat, feet

  def ground(self, xy, top):
    """Same terrain-only vertical ray query used by the ROS true-scan bridge."""
    out = np.empty(len(xy), dtype=float)
    for i, (x, y) in enumerate(xy):
      distance = mujoco.mj_ray(self.model, self.data, np.array([x, y, top]),
                               self.ray_dir, self.terrain_group, 1, -1, self.geom_id)
      out[i] = np.nan if distance < 0 else top - distance
    return out


def _robot_env():
  names = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]
  data = SimpleNamespace(body_link_pose_w=torch.zeros((1, 4, 7)),
                         body_link_pos_w=torch.zeros((1, 4, 3)))
  robot = SimpleNamespace(body_names=names, data=data)
  sensor = SimpleNamespace(data=SimpleNamespace(hit_pos_w=None, distances=None))
  scene = {"robot": robot, "student_map_probe": sensor}
  return SimpleNamespace(num_envs=1, device="cpu", scene=scene,
                         step_dt=0.02, common_step_counter=0)


def _probe(trajectory: Trajectory, pelvis, yaw, cfg: StudentMapCfg):
  side = round(cfg.cells * cfg.resolution / cfg.probe_resolution) + 1
  line = np.linspace(-2.5, 2.5, side)
  x, y = np.meshgrid(line, line, indexing="xy")
  local = np.stack((x.ravel(), y.ravel()), axis=1)
  c, s = np.cos(yaw), np.sin(yaw)
  xy = pelvis[:2] + local @ np.array([[c, s], [-s, c]])
  top = pelvis[2] + 2.0
  ground = trajectory.ground(xy, top)
  valid = np.isfinite(ground)
  hits = np.column_stack((xy, ground))
  hits[~valid, 2] = 0
  return torch.tensor(hits[None], dtype=torch.float32), torch.tensor(
    np.where(valid, top - ground, -1)[None], dtype=torch.float32)


def _map_projection(ros, entry, pelvis, pelvis_quat, feet, sole_depth):
  m = entry[1]
  def layer(name):
    if name not in m["layers"]:
      return None
    dims, values = m["layers"][name]
    return ros.unpack_layer(values, dims[0][1], dims[1][1], m["outer"], m["inner"])
  observed = layer("elevation")
  if observed is None:
    raise ValueError("ROS GridMap has no elevation layer")
  top = (m["center"][0] + m["length"][0] / 2,
         m["center"][1] + m["length"][1] / 2)
  yaw = _yaw(pelvis_quat)
  scan = ros.compute_scan(observed, layer("inpaint"), top, m["resolution"],
                          pelvis, yaw, feet.min() - sole_depth)[0]
  observed_ground = ros.sample(observed, top, m["resolution"],
                               ros.scan_points(pelvis[:2], yaw))
  observed_mask = np.isfinite(observed_ground)
  observed_scan = np.clip(pelvis[2] - observed_ground, -1.0, 2.0)
  return scan, observed_mask, observed_scan


def _svg_plot(rows, path):
  """Write an offline plot without Matplotlib or a system font cache."""
  width, height = 1050, 710
  left, right = 88, 25
  xmin, xmax = rows[0]["bag_time_s"], rows[-1]["bag_time_s"]
  if xmax == xmin:
    xmax += 1
  panels = [
    ("Scan RMSE (cm)", [("ROS", "ros_rmse_m", "#1976d2"),
                         ("Student", "student_rmse_m", "#e64a19")], 100),
    ("Observed share", [("ROS", "ros_observed_share", "#1976d2"),
                         ("Student", "student_observed_share", "#e64a19")], 1),
    ("Comparison MAE (cm)", [("Student vs ROS", "student_vs_ros_mae_m", "#7b1fa2"),
                              ("Pose/scene check", "reconstructed_truth_mae_m", "#388e3c")], 100),
  ]
  parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font:14px sans-serif;fill:#222}.axis{stroke:#888;stroke-width:1}</style>']
  for panel, (title, lines, scale) in enumerate(panels):
    top = 28 + panel * 225
    bottom = top + 170
    ymax = max(max(r[key] * scale for r in rows) for _, key, _ in lines)
    ymax = max(ymax * 1.05, 0.01)
    parts += [f'<text x="{left}" y="{top - 8}">{html.escape(title)}</text>',
              f'<path class="axis" d="M {left} {top} V {bottom} H {width-right}" fill="none"/>',
              f'<text x="15" y="{top + 7}">{ymax:.1f}</text>',
              f'<text x="33" y="{bottom + 5}">0</text>']
    for j, (label, key, color) in enumerate(lines):
      pts = " ".join(f'{left + (r["bag_time_s"]-xmin)/(xmax-xmin)*(width-left-right):.1f},'
                     f'{bottom - r[key]*scale/ymax*(bottom-top):.1f}' for r in rows)
      parts += [f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>',
                f'<text x="{width - 250 + j*115}" y="{top - 8}" fill="{color}">{html.escape(label)}</text>']
  parts += [f'<text x="{left}" y="{height - 5}">Bag time {xmin:.0f}–{xmax:.0f} s since first true scan</text>', '</svg>']
  path.write_text("\n".join(parts) + "\n")


def _errors(a, truth):
  d = a - truth
  return float(np.mean(np.abs(d))), float(np.sqrt(np.mean(d * d))), float(np.max(np.abs(d))), float(np.mean(np.abs(d) > 0.2))


def run(path: Path, output_dir: Path, limit: int | None,
        map_cfg: StudentMapCfg | None = None, seed: int = 42):
  torch.manual_seed(seed)
  manifest = capture_manifest(path)
  topics = load_capture(path)
  ros = _ros_scan_module()
  scene = ROOT / "sim/g1_walk_scene.xml"
  description = ROOT / "description/g1/g1_29dof_rev_1_0.xml"
  trajectory = Trajectory(topics["/joint_states"][0][1][1], scene, description)
  tf = TfTree(topics)
  env = _robot_env()
  cfg = map_cfg or StudentMapCfg()
  student = StudentHeightMap(SimpleNamespace(params={"map_cfg": cfg}), env)
  rows = []
  live = topics["/height_scan"][:limit]
  truth_frames = topics["/ground_truth/height_scan"]
  truth_times = [entry[0] for entry in truth_frames]
  start_time = truth_times[0]
  pairs = {}
  for live_index, (time, ros_scan) in enumerate(live):
    j = bisect.bisect_left(truth_times, time)
    j = min(max(j, 1), len(truth_times) - 1)
    if abs(truth_times[j - 1] - time) < abs(truth_times[j] - time):
      j -= 1
    pairs.setdefault(j, []).append((live_index, time, ros_scan))
  last_truth_index = max(pairs)
  for i, (truth_time, truth) in enumerate(truth_frames[:last_truth_index + 1]):
    odom_time, odom = nearest(topics["/ground_truth/odom"], truth_time)
    joint_time, joints = nearest(topics["/joint_states"], truth_time)
    pelvis, pelvis_quat, torso, torso_quat, feet = trajectory.update(odom, joints)
    robot = env.scene["robot"].data
    robot.body_link_pose_w[0, 0] = torch.tensor(np.r_[pelvis, pelvis_quat], dtype=torch.float32)
    robot.body_link_pose_w[0, 1] = torch.tensor(np.r_[torso, torso_quat], dtype=torch.float32)
    robot.body_link_pos_w[0, 2:, 2] = torch.tensor(feet, dtype=torch.float32)
    env.common_step_counter = i
    if i == 0 or (i - student.last_step) > 0 and (i * env.step_dt - student.last_map_time.item() >= 1 / cfg.map_hz):
      hits, distances = _probe(trajectory, pelvis, _yaw(pelvis_quat), cfg)
      env.scene["student_map_probe"].data.hit_pos_w = hits
      env.scene["student_map_probe"].data.distances = distances
    simulated = student(env, "student_map_probe", cfg)[0].detach().numpy()
    exact_xy = ros.scan_points(pelvis[:2], _yaw(pelvis_quat))
    exact = ros.to_scan(pelvis[2], trajectory.ground(exact_xy, 5.0))
    for live_index, time, ros_scan in pairs.get(i, ()):
      map_entry = latest(topics["/elevation_mapping_node/elevation_map_filter"], time)
      if map_entry is None:
        continue
      tf.advance(time)
      ros_pelvis, ros_pelvis_quat = tf.pose("pelvis")
      ros_feet = np.array([tf.pose(name)[0][2] for name in
                           ("left_ankle_roll_link", "right_ankle_roll_link")])
      projected, ros_observed, ros_observed_scan = _map_projection(
        ros, map_entry, ros_pelvis, ros_pelvis_quat, ros_feet, cfg.sole_depth)
      source = student.last_source[0].numpy()
      student_observed = source == 0
      both_observed = student_observed & ros_observed
      ros_error = _errors(ros_scan, truth)
      student_error = _errors(simulated, truth)
      row = {
        "step": i, "bag_time_s": time - start_time,
        "truth_match_ms": abs(time - truth_time) * 1000,
        "odom_match_ms": abs(truth_time - odom_time) * 1000,
        "joints_match_ms": abs(truth_time - joint_time) * 1000,
        "map_age_wall_s": time - map_entry[0],
        "pelvis_x": pelvis[0], "pelvis_y": pelvis[1],
        "ros_mae_m": ros_error[0], "ros_rmse_m": ros_error[1],
        "ros_max_m": ros_error[2], "ros_over_20cm": ros_error[3],
        "student_mae_m": student_error[0], "student_rmse_m": student_error[1],
        "student_max_m": student_error[2], "student_over_20cm": student_error[3],
        "student_vs_ros_mae_m": _errors(simulated, ros_scan)[0],
        "reconstructed_truth_mae_m": _errors(exact, truth)[0],
        "ros_map_projection_mae_m": _errors(projected, ros_scan)[0],
        "observed_source_disagreement": float(np.mean(student_observed != ros_observed)),
        "both_observed_scan_mae_m": (float(np.mean(np.abs(simulated[both_observed] -
                                                           ros_observed_scan[both_observed])))
                                     if both_observed.any() else float("nan")),
        "ros_observed_share": nearest(topics["/height_scan/observed_share"], time)[1],
        "ros_feet_share": nearest(topics["/height_scan/feet_share"], time)[1],
        "student_observed_share": float(np.mean(student_observed)),
        "student_feet_share": float(np.mean(source == 1)),
        "student_inpaint_share": float(np.mean(source == 2)),
      }
      rows.append(row)
      if (live_index + 1) % 200 == 0:
        print(f"Compared {live_index + 1}/{len(live)} ROS scans", flush=True)
  if not rows:
    raise ValueError("No ROS scans had a preceding GridMap")
  turn = int(np.argmax([r["pelvis_x"] for r in rows]))
  for j, row in enumerate(rows):
    row["phase"] = "outbound" if j <= turn else "return"
    row["stair_zone"] = 4.7 <= row["pelvis_x"] <= 6.5
  output_dir.mkdir(parents=True, exist_ok=True)
  with (output_dir / "steps.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
  def summary(key):
    a = np.asarray([r[key] for r in rows], dtype=float)
    a = a[np.isfinite(a)]
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.quantile(a, 0.95))} if len(a) else None
  def segment_summary(selected):
    return {"steps": len(selected),
            "ros_rmse_m": float(np.mean([r["ros_rmse_m"] for r in selected])),
            "student_rmse_m": float(np.mean([r["student_rmse_m"] for r in selected])),
            "student_vs_ros_mae_m": float(np.mean([r["student_vs_ros_mae_m"] for r in selected]))}
  report = {"capture": str(path), "scene": str(scene), "steps": len(rows),
            "matching": "nearest MCAP record time; ROS scan messages are unstamped",
            "manifest": manifest,
            "student_map_cfg": asdict(cfg), "noise_seed": seed,
            "metrics": {k: summary(k) for k in rows[0] if k not in
                        ("step", "bag_time_s", "pelvis_x", "pelvis_y", "phase", "stair_zone")},
            "segments": {label: segment_summary(selected) for label, selected in (
              (label, [r for r in rows if r["stair_zone"] and r["phase"] == label])
              for label in ("outbound", "return")) if selected}}
  report["validation"] = {
    "scene_pose_reconstruction_under_1cm_mean_mae":
      report["metrics"]["reconstructed_truth_mae_m"]["mean"] < 0.01,
    "ros_gridmap_projection_under_2cm_mean_mae":
      report["metrics"]["ros_map_projection_mae_m"]["mean"] < 0.02,
  }
  (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
  _svg_plot(rows, output_dir / "comparison.svg")
  print(json.dumps({"steps": len(rows), "output": str(output_dir),
                    "ros_rmse_cm": report["metrics"]["ros_rmse_m"]["mean"] * 100,
                    "student_rmse_cm": report["metrics"]["student_rmse_m"]["mean"] * 100,
                    "student_vs_ros_mae_cm": report["metrics"]["student_vs_ros_mae_m"]["mean"] * 100,
                    "observed_source_disagreement_pct":
                      report["metrics"]["observed_source_disagreement"]["mean"] * 100,
                    "pose_scene_check_mae_cm": report["metrics"]["reconstructed_truth_mae_m"]["mean"] * 100,
                    "ros_map_projection_mae_cm": report["metrics"]["ros_map_projection_mae_m"]["mean"] * 100,
                    "validation": report["validation"]}, indent=2))
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("capture", type=Path, help="ROS capture .tar.gz or bag_0.mcap")
  parser.add_argument("--output-dir", type=Path, default=Path("logs/ros_mapping_comparison"))
  parser.add_argument("--limit", type=int, help="Replay only the first N scans (for a quick check)")
  parser.add_argument("--hit-keep-probability", type=float, default=StudentMapCfg.hit_keep_probability)
  parser.add_argument("--observation-delay-s", type=float, default=StudentMapCfg.observation_delay_s)
  parser.add_argument("--height-bias-std-m", type=float, default=StudentMapCfg.height_bias_std_m)
  parser.add_argument("--height-noise-std-m", type=float, default=StudentMapCfg.height_noise_std_m)
  parser.add_argument("--pose-noise-xy-std-m", type=float, default=StudentMapCfg.pose_noise_xy_std_m)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  if args.limit is not None and args.limit < 1:
    parser.error("--limit must be positive")
  cfg = StudentMapCfg(
    hit_keep_probability=args.hit_keep_probability,
    observation_delay_s=args.observation_delay_s,
    height_bias_std_m=args.height_bias_std_m,
    height_noise_std_m=args.height_noise_std_m,
    pose_noise_xy_std_m=args.pose_noise_xy_std_m)
  run(args.capture.expanduser(), args.output_dir, args.limit, cfg, args.seed)


if __name__ == "__main__":
  main()
