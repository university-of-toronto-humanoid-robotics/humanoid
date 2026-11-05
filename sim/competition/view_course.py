"""Open the competition course in the MuJoCo viewer with a G1 standing on it.

No policy runs here. The G1 holds its home pose with Unitree's FixStand gains
(what the real robot uses to stand without a policy), on mjlab's motor model:
armature and torque limits from g1_constants.py, implicitfast integrator, 5 ms
steps. Without a policy nothing balances the robot, so a soft "gantry" spring
holds the pelvis upright while the feet carry the weight (turn it off in the
viewer under Physics > Equality, or pass --no-gantry, to watch it tip over).

Usage:
  python sim/competition/view_course.py                     # G1 on the first stair
  python sim/competition/view_course.py --spawn landing
  python sim/competition/view_course.py --no-robot
  python sim/competition/view_course.py --spawn stairs --render stairs.png   # no window

Needs only mujoco + numpy.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
COURSE_XML = HERE / "course.xml"
HEIGHTMAP = HERE / "course_heightmap.npz"
G1_XML = REPO_ROOT / "training" / "unitree_rl_mjlab" / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
PREFIX = "robot/"

# name: (x, y, yaw) in the course frame. yaw = pi faces -x, toward the stairs end.
SPAWNS = {
  "start": (11.00, 4.21, math.pi),
  "stepover": (9.25, 4.21, math.pi),
  "ramps": (7.62, 3.60, math.pi),  # on a 15 deg slope, facing downhill
  "flat": (5.00, 4.20, math.pi),
  "stairs": (3.56, 4.96, math.pi),  # first step, facing up the stairs
  "landing": (1.80, 3.80, -math.pi / 2),  # top of the stairs, after turning left
  "button5": (2.20, 3.35, 0.0),  # at the landing edge, facing the drop
}
SPAWN_CLEARANCE = 0.01  # m; the feet start this far above the ground and settle onto it

# Home pose (HOME_KEYFRAME in g1_constants.py); unlisted joints are 0.
HOME_POSE = [
  (r".*_hip_pitch_joint", -0.1),
  (r".*_knee_joint", 0.3),
  (r".*_ankle_pitch_joint", -0.2),
  (r".*_shoulder_pitch_joint", 0.35),
  (r".*_elbow_joint", 0.87),
  (r".*left_shoulder_roll_joint", 0.18),
  (r".*right_shoulder_roll_joint", -0.18),
]


def reflected_inertia(rotor: tuple[float, float, float], gears: tuple[float, float, float]) -> float:
  """Two-stage planetary gearbox, as in mjlab.utils.actuator."""
  return rotor[0] * (gears[1] * gears[2]) ** 2 + rotor[1] * gears[2] ** 2 + rotor[2]


# Unitree motor specs, copied from g1_constants.py: (armature kg*m^2, torque limit N*m).
M5020 = (reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16)), 25.0)
M7520_14 = (reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22)), 88.0)
M7520_22 = (reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5)), 139.0)
M4010 = (reflected_inertia((0.068e-4, 0.0, 0.0), (1, 5, 5)), 5.0)
DOUBLE_5020 = (M5020[0] * 2, M5020[1] * 2)  # ankles and waist pitch/roll: two motors per joint
MOTORS = [  # (joint pattern, motor)
  (r".*_(hip_pitch|hip_yaw)_joint|.*waist_yaw_joint", M7520_14),
  (r".*_(hip_roll|knee)_joint", M7520_22),
  (r".*_(wrist_pitch|wrist_yaw)_joint", M4010),
  (r".*waist_(pitch|roll)_joint|.*_ankle_(pitch|roll)_joint", DOUBLE_5020),
  (r".*_(shoulder_pitch|shoulder_roll|shoulder_yaw|elbow|wrist_roll)_joint", M5020),
]
# (joint pattern, kp, kd): FixStand gains from deploy/robots/g1/config/config.yaml.
HOLD_GAINS = [
  (r".*_hip_.*", (100.0, 2.0)),
  (r".*_knee_joint", (150.0, 4.0)),
  (r".*_ankle_.*", (40.0, 2.0)),
  (r".*waist_.*", (200.0, 5.0)),
  (r".*_(shoulder|elbow|wrist)_.*", (40.0, 10.0)),
]


def lookup(name: str, table):
  for pattern, value in table:
    if re.fullmatch(pattern, name):
      return value
  return None


def ground_height(x: float, y: float, half: float = 0.03) -> float:
  """Highest point of the course in a small square around (x, y)."""
  d = np.load(HEIGHTMAP)
  h, x0, y0, res = d["heights"], float(d["x0"]), float(d["y0"]), float(d["res"])
  i0, i1 = int((x - half - x0) / res), int((x + half - x0) / res) + 1
  j0, j1 = int((y - half - y0) / res), int((y + half - y0) / res) + 1
  return float(h[i0:i1, j0:j1].max())


def footprint_height(x: float, y: float, yaw: float) -> float:
  """Highest point under both feet (about 22 cm long, 26 cm wide, turned with the robot)."""
  along, across = 0.11, 0.13
  if abs(math.cos(yaw)) > 0.7:
    hx, hy = along, across
  else:
    hx, hy = across, along
  d = np.load(HEIGHTMAP)
  h, x0, y0, res = d["heights"], float(d["x0"]), float(d["y0"]), float(d["res"])
  i0, i1 = int(round((x - hx - x0) / res)), int(round((x + hx - x0) / res)) + 1
  j0, j1 = int(round((y - hy - y0) / res)), int(round((y + hy - y0) / res)) + 1
  return float(h[i0:i1, j0:j1].max())


def _sole_height(g1: mujoco.MjSpec) -> float:
  """Lowest point of the G1's foot collision geoms in the home pose, in g1.xml's frame."""
  model = g1.compile()
  data = mujoco.MjData(model)
  for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
      data.qpos[model.jnt_qposadr[j]] = lookup(model.joint(j).name, HOME_POSE) or 0.0
  mujoco.mj_forward(model, data)
  lowest = np.inf
  for g in range(model.ngeom):
    name = model.geom(g).name
    if "foot" in name and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
      lowest = min(lowest, data.geom_xpos[g][2] - model.geom_size[g][0])
  return float(lowest)


def build(spawn: str | None, gantry: bool = True) -> tuple[mujoco.MjModel, mujoco.MjData]:
  spec = mujoco.MjSpec.from_file(str(COURSE_XML))
  spec.option.timestep = 0.005
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  if spawn is None:
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data

  g1 = mujoco.MjSpec.from_file(str(G1_XML))
  g1.meshdir = str(G1_XML.parent / "assets")  # absolute, so meshes resolve after attaching
  # g1.xml already places the pelvis 0.79 m above its origin, so put the origin
  # on the ground; then lift by the gap between the soles and the ground below.
  x, y, yaw = SPAWNS[spawn]
  z = footprint_height(x, y, yaw) + SPAWN_CLEARANCE - _sole_height(g1)
  frame = spec.worldbody.add_frame(pos=[x, y, z], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
  spec.attach(g1, frame=frame, prefix=PREFIX)

  # Contact settings from FULL_COLLISION_WITHOUT_SELF in g1_constants.py. (With a
  # frozen pose, self-collisions would just keep the hands pressed into the hips.)
  for geom in spec.geoms:
    if geom.name.startswith(PREFIX) and geom.name.endswith("_collision"):
      is_foot = re.fullmatch(rf"{PREFIX}(left|right)_foot[1-7]_collision", geom.name) is not None
      geom.contype, geom.conaffinity = 0, 1
      geom.condim = 3 if is_foot else 1
      if is_foot:
        geom.priority = 1
        geom.friction = [0.6, geom.friction[1], geom.friction[2]]

  if gantry:
    # Soft weld of the pelvis to a fixed anchor at its spawn pose, lowered by the
    # spawn gap plus a little sag so the feet carry the weight.
    pelvis_offset = spec.body(PREFIX + "pelvis").pos  # pelvis height in g1.xml (0.79 m)
    anchor = spec.worldbody.add_body(name="gantry_anchor", mocap=True,
                                     pos=[x, y, z + pelvis_offset[2] - SPAWN_CLEARANCE - 0.002],
                                     quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    weld = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                             name1=PREFIX + "pelvis", name2=anchor.name, name="gantry",
                             solref=[0.1, 1.0])
    # Weld data: anchor point (3), pelvis pose relative to the anchor (pos 3 +
    # quat 4), torque scale; newer MuJoCo versions append more unused slots.
    data = np.zeros(len(weld.data))
    data[6], data[10] = 1.0, 1.0
    weld.data = data

  for joint in spec.joints:
    if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    armature, effort = lookup(joint.name, MOTORS)
    kp, kd = lookup(joint.name, HOLD_GAINS)
    joint.armature = armature
    act = spec.add_actuator(name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.gainprm = [kp] + [0.0] * 9
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.biasprm = [0.0, -kp, -kd] + [0.0] * 7
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-effort, effort]

  # Compile once to find joint addresses, then store the home pose as a keyframe
  # (the viewer can reload it from Simulation > Key).
  model = spec.compile()
  qpos = model.qpos0.copy()
  ctrl = np.zeros(model.nu)
  for a in range(model.nu):
    jid = model.actuator_trnid[a, 0]
    angle = lookup(model.joint(jid).name, HOME_POSE) or 0.0
    qpos[model.jnt_qposadr[jid]] = angle
    ctrl[a] = angle
  spec.add_key(name="home", qpos=qpos.tolist(), ctrl=ctrl.tolist())
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return model, data


def settle(model, data, seconds: float, spawn: str) -> bool:
  """Run the sim with the robot holding its pose; report whether it stayed up."""
  x, y, _ = SPAWNS[spawn]
  ground = ground_height(x, y)
  pelvis = data.body(PREFIX + "pelvis")
  while data.time < seconds:
    mujoco.mj_step(model, data)
    if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
      print(f"simulation became unstable at t={data.time:.3f} s")
      return False
  height = pelvis.xpos[2] - ground
  tilt = math.degrees(math.acos(np.clip(pelvis.xmat[8], -1.0, 1.0)))
  upright = height > 0.6 and tilt < 15.0
  # How each sole sits on the course: gap between its lowest point and the
  # ground right below it (about 0 = resting on it, negative = sunk in).
  gaps = []
  for side in ("left", "right"):
    lows = []
    for g in range(model.ngeom):
      if re.fullmatch(rf"{PREFIX}{side}_foot[1-7]_collision", model.geom(g).name):
        gx, gy, gz = data.geom_xpos[g]
        lows.append((gz - model.geom_size[g][0], ground_height(gx, gy, half=0.0)))
    sole, under = min(lows)
    gaps.append(f"{side} {sole - under:+.3f} m")
  weight = data.efc_force[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
  print(f"after {seconds:.1f} s on '{spawn}': pelvis {height:.3f} m above the ground, "
        f"tilt {tilt:.1f} deg -> {'standing' if upright else 'FELL'}; sole gaps: {', '.join(gaps)}"
        + (f"; gantry force {np.abs(weight[:3]).sum():.0f} N" if weight.size else ""))
  return upright


def render(model, data, path: Path, spawn: str | None) -> None:
  renderer = mujoco.Renderer(model, height=900, width=1600)
  cam = mujoco.MjvCamera()
  if spawn is None:  # whole course, seen from the side
    cam.lookat[:] = [7.1, 4.2, 0.0]
    cam.distance, cam.azimuth, cam.elevation = 10.0, 90.0, -45.0
  else:  # from behind the robot, looking where it faces
    yaw = SPAWNS[spawn][2]
    ahead = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    cam.lookat[:] = data.body(PREFIX + "pelvis").xpos + 0.5 * ahead - [0, 0, 0.3]
    cam.distance, cam.azimuth, cam.elevation = 3.0, math.degrees(yaw) + 25.0, -25.0
  renderer.update_scene(data, camera=cam)
  write_png(path, renderer.render())
  print(f"saved {path}")


def write_png(path: Path, rgb: np.ndarray) -> None:
  """Minimal PNG writer (RGB, 8 bit), so no imaging library is needed."""
  import struct
  import zlib

  def chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

  h, w, _ = rgb.shape
  rows = b"".join(b"\x00" + rgb[r].astype(np.uint8).tobytes() for r in range(h))
  header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
  path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                   + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--spawn", choices=sorted(SPAWNS), default="stairs")
  parser.add_argument("--no-robot", action="store_true")
  parser.add_argument("--no-gantry", action="store_true", help="no spring holding the pelvis")
  parser.add_argument("--render", type=Path, help="save a picture instead of opening the viewer")
  parser.add_argument("--seconds", type=float, default=2.0, help="settle time before --render")
  args = parser.parse_args()

  spawn = None if args.no_robot else args.spawn
  model, data = build(spawn, gantry=not args.no_gantry)
  if args.render:
    if spawn:
      settle(model, data, args.seconds, spawn)
    render(model, data, args.render, spawn)
    return

  import mujoco.viewer

  mujoco.viewer.launch(model, data)


if __name__ == "__main__":
  main()
