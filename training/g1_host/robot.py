"""The G1 (Unitree's 29-DoF model) set up for standing up.

What differs from the walking tasks' robot, and why:

  - Every collision shape touches the world, with friction. The MJCF's collision
    class gives every shape priority 1 and Unitree's FULL_COLLISION sets condim 1
    (no friction) on everything but the feet; with priority the robot's condim wins,
    so the back, hands, knees and pelvis would slide over the ground and the robot
    could never push itself up. Here every shape has condim 3.
  - Extra boxes for the torso and pelvis. The stock torso is one 9 cm capsule and the
    pelvis a 7 cm sphere: lying on its back the robot would roll like a log. The
    boxes are fitted to the visual meshes (a little smaller than them).
  - MuJoCo position actuators with HoST's PD gains (params.KP/KD). Unitree's
    armature and torque limits are kept. Position actuators because MuJoCo's
    implicitfast integrator only treats their damping implicitly: a PD computed in
    torch would be unstable on the light arm joints at Kd = 4.
  - The 6 joints the paper's 23-DoF G1 doesn't have (waist roll/pitch, wrist
    pitch/yaw) are held at 0 by stiff PD, like that robot's locked joints.
"""

from __future__ import annotations

import dataclasses
import re

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from src.assets.robots.unitree_g1 import g1_constants as g1

from . import params as P

# The policy's joints, in the order of its actions and of the joint inputs. HoST's
# 23 (6 per leg, waist yaw, 5 per arm), in Unitree's MJCF order.
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
HOST23_JOINTS = (
  *(f"left_{j}_joint" for j in LEG),
  *(f"right_{j}_joint" for j in LEG),
  "waist_yaw_joint",
  *(f"left_{j}_joint" for j in ARM),
  *(f"right_{j}_joint" for j in ARM),
)
HELD_JOINTS = (
  "waist_roll_joint", "waist_pitch_joint",
  "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
ALL29_JOINTS = (
  *(f"left_{j}_joint" for j in LEG), *(f"right_{j}_joint" for j in LEG),
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  *(f"left_{j}_joint" for j in (*ARM, "wrist_pitch", "wrist_yaw")),
  *(f"right_{j}_joint" for j in (*ARM, "wrist_pitch", "wrist_yaw")),
)


def policy_joints() -> tuple[str, ...]:
  return HOST23_JOINTS if P.POLICY_JOINTS == "host23" else ALL29_JOINTS


def held_joints() -> tuple[str, ...]:
  return HELD_JOINTS if P.POLICY_JOINTS == "host23" else ()


# Ankle "keypoints" for the ankle-parallel reward: in ankle_roll_link's frame, as the
# auxiliary links of HoST's URDF (only the variance of their heights is used).
ANKLE_KEYPOINTS = ((0.0, 0.0, 0.0), (0.035, 0.1, 0.0), (0.035, -0.1, 0.0), (0.2, 0.0, 0.0), (-0.15, 0.0, 0.0))

# --- Collision shapes --------------------------------------------------------------


def _mesh_points_in_body(model: mujoco.MjModel, body: str) -> np.ndarray | None:
  """Vertices of a body's mesh geoms, in the body frame."""
  bid = model.body(body).id
  pts = []
  for g in range(model.ngeom):
    if model.geom_bodyid[g] != bid or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    m = model.geom_dataid[g]
    v = model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[g])
    pts.append(v @ rot.reshape(3, 3).T + model.geom_pos[g])
  return np.concatenate(pts) if pts else None


def _add_box(spec: mujoco.MjSpec, body: str, name: str, pts: np.ndarray | None,
             fallback: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
  if pts is None or len(pts) < 8:
    center, half = fallback
  else:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center, half = tuple(float(c) for c in (lo + hi) / 2), tuple(float(h) for h in (hi - lo) / 2 * BOX_SHRINK)
  spec.body(body).add_geom(
    name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=center,
    group=3, contype=1, conaffinity=1, condim=3, priority=1, rgba=(0.2, 0.6, 0.2, 0.3),
  )


# Torso box: the torso mesh below the head (the head keeps its sphere). Pelvis box: the
# pelvis mesh above the hips, so it doesn't overlap the hip and thigh capsules.
TORSO_BOX_Z = (0.0, 0.36)
PELVIS_BOX_ZMIN = -0.05
BOX_SHRINK = 0.85


def get_spec() -> mujoco.MjSpec:
  model = g1.get_spec().compile()  # measured on a separate copy: a compiled spec's name lookups go stale
  spec = g1.get_spec()
  torso = _mesh_points_in_body(model, "torso_link")
  if torso is not None:
    torso = torso[(torso[:, 2] > TORSO_BOX_Z[0]) & (torso[:, 2] < TORSO_BOX_Z[1])]
  _add_box(spec, "torso_link", "torso_box_collision", torso, ((0.0, 0.0, 0.15), (0.07, 0.11, 0.13)))
  pelvis = _mesh_points_in_body(model, "pelvis")
  if pelvis is not None:
    pelvis = pelvis[pelvis[:, 2] > PELVIS_BOX_ZMIN]
  _add_box(spec, "pelvis", "pelvis_box_collision", pelvis, ((0.0, 0.0, 0.0), (0.06, 0.10, 0.04)))
  return spec


# Every shape: world and self contact, friction (condim 3). Feet keep Unitree's 0.6.
FULL_BODY_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,), ".*_collision": (1.0,)},
)

# --- Actuators ----------------------------------------------------------------------


def _pd(names: tuple[str, ...], kp: float, kd: float, effort: float, armature: float) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=names, stiffness=kp, damping=kd, effort_limit=effort, armature=armature
  )


def actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
  a5020, a14, a22, a4010 = g1.ACTUATOR_5020, g1.ACTUATOR_7520_14, g1.ACTUATOR_7520_22, g1.ACTUATOR_4010
  kp, kd = P.KP, P.KD
  out = [
    _pd((".*_hip_pitch_joint", ".*_hip_yaw_joint"), kp["hip"], kd["hip"], a14.effort_limit, a14.reflected_inertia),
    _pd((".*_hip_roll_joint",), kp["hip"], kd["hip"], a22.effort_limit, a22.reflected_inertia),
    _pd((".*_knee_joint",), kp["knee"], kd["knee"], a22.effort_limit, a22.reflected_inertia),
    _pd((".*_ankle_pitch_joint", ".*_ankle_roll_joint"), kp["ankle"], kd["ankle"],
        a5020.effort_limit * 2, a5020.reflected_inertia * 2),
    _pd(("waist_yaw_joint",), kp["waist"], kd["waist"], a14.effort_limit, a14.reflected_inertia),
    _pd((".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint",
         ".*_wrist_roll_joint"), kp["shoulder"], kd["shoulder"], a5020.effort_limit, a5020.reflected_inertia),
  ]
  held = P.POLICY_JOINTS == "host23"
  wkp, wkd = (P.HELD_KP["waist"], P.HELD_KD["waist"]) if held else (kp["waist"], kd["waist"])
  hkp, hkd = (P.HELD_KP["wrist"], P.HELD_KD["wrist"]) if held else (kp["wrist"], kd["wrist"])
  out.append(_pd(("waist_pitch_joint", "waist_roll_joint"), wkp, wkd, a5020.effort_limit * 2, a5020.reflected_inertia * 2))
  out.append(_pd((".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), hkp, hkd, a4010.effort_limit, a4010.reflected_inertia))
  return tuple(out)


def gains(joint: str) -> tuple[float, float, float]:
  """(Kp, Kd, torque limit) of a joint, from the actuator groups above."""
  for a in actuators():
    if any(re.fullmatch(p, joint) for p in a.target_names_expr):
      assert a.effort_limit is not None
      return float(a.stiffness), float(a.damping), float(a.effort_limit)
  raise KeyError(joint)


def vel_limit(joint: str) -> float:
  for key, v in P.VEL_LIMIT.items():
    if key in joint:
      return v
  return 37.0


# Supine: lying on the back, head toward -x (rotation of -90 deg about y; (w, x, y, z)).
SUPINE_QUAT = (float(np.sqrt(0.5)), 0.0, -float(np.sqrt(0.5)), 0.0)


def robot_cfg(reset_pose: dict[str, float] | None = None) -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      # High above the world origin: before the first reset the simulator sizes its
      # contact buffers from this pose, and a robot inside the stairs overflows them.
      # Every reset places the robot itself (events.reset_posture).
      pos=(0.0, 0.0, 3.0),
      rot=SUPINE_QUAT,
      joint_pos={".*": 0.0, **(reset_pose or P.RESET_POSE)},
      joint_vel={".*": 0.0},
    ),
    collisions=(FULL_BODY_COLLISION,),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(actuators=actuators(), soft_joint_pos_limit_factor=1.0),
  )


def with_reset_pose(cfg: EntityCfg, reset_pose: dict[str, float]) -> EntityCfg:
  init = dataclasses.replace(cfg.init_state, joint_pos={".*": 0.0, **reset_pose})
  return dataclasses.replace(cfg, init_state=init)
