"""The G1 (29 joints) as the walking policies were trained on it: joint order, standing pose and motor gains.

No ROS and no MuJoCo in here, so the controller on the robot and the sim share one definition. The numbers come
from unitree_rl_mjlab's g1_constants.py and its velocity task, so update them together.
"""
import re

import numpy as np

# Unitree's motor order: motor i in /lowstate and /lowcmd is JOINTS[i]. The policies use the same order.
JOINTS = [f'{side}_{joint}_joint' for side in ('left', 'right')
          for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
JOINTS += ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
JOINTS += [f'{side}_{joint}_joint' for side in ('left', 'right')
           for joint in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch',
                         'wrist_yaw')]

# Gains: each joint's come from its motor's reflected inertia (armature), a 10 Hz natural frequency and a
# damping ratio of 2.
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def two_stage_planetary_armature(rotor, gears):
    return rotor[0] * (gears[1] * gears[2]) ** 2 + rotor[1] * gears[2] ** 2 + rotor[2]


ARMATURE_5020 = two_stage_planetary_armature((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
ARMATURE_7520_14 = two_stage_planetary_armature((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22))
ARMATURE_7520_22 = two_stage_planetary_armature((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
ARMATURE_4010 = two_stage_planetary_armature((0.068e-4, 0.0, 0.0), (1, 5, 5))
MOTORS = (  # (joint names, armature, effort limit in N m). Waist pitch/roll and ankles are two 5020s each.
    (r'.*_(elbow|shoulder_pitch|shoulder_roll|shoulder_yaw|wrist_roll)_joint', ARMATURE_5020, 25.0),
    (r'.*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint', ARMATURE_7520_14, 88.0),
    (r'.*_hip_roll_joint|.*_knee_joint', ARMATURE_7520_22, 139.0),
    (r'.*_wrist_(pitch|yaw)_joint', ARMATURE_4010, 5.0),
    (r'waist_(pitch|roll)_joint|.*_ankle_(pitch|roll)_joint', 2 * ARMATURE_5020, 50.0),
)
# The standing pose (HOME_KEYFRAME); other joints are 0.
DEFAULT_POSE = {
    r'.*_hip_pitch_joint': -0.1, r'.*_knee_joint': 0.3, r'.*_ankle_pitch_joint': -0.2,
    r'.*_shoulder_pitch_joint': 0.35, r'.*_elbow_joint': 0.87,
    r'left_shoulder_roll_joint': 0.18, r'right_shoulder_roll_joint': -0.18,
}


def motor(name):
    """(armature, effort limit) of a joint's motor."""
    hits = [value for pattern, *value in MOTORS if re.fullmatch(pattern, name)]
    if len(hits) != 1:
        raise ValueError(f'{name} matches {len(hits)} motors, expected 1')
    return hits[0]


def default_pose(names=JOINTS):
    return np.array([next((v for p, v in DEFAULT_POSE.items() if re.fullmatch(p, n)), 0.0) for n in names])


def armature(names=JOINTS):
    return np.array([motor(n)[0] for n in names])


def effort(names=JOINTS):
    """Each motor's torque limit, N m."""
    return np.array([motor(n)[1] for n in names])


def stiffness(names=JOINTS):
    """Position gain kp, N m/rad."""
    return armature(names) * NATURAL_FREQ**2


def damping(names=JOINTS):
    """Velocity gain kd, N m s/rad."""
    return 2.0 * DAMPING_RATIO * armature(names) * NATURAL_FREQ
