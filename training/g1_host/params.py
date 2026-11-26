"""Every number of the HoST port, in one place, with where it comes from.

HoST = "Learning Humanoid Standing-up Control across Diverse Postures" (Huang et al.,
RSS 2025, arXiv 2502.08378v2) and its code (github.com/OpenRobotLab/HoST). Where the
paper and the released code disagree, we use the code: it is what produced the
paper's numbers (the HoST README thanks a reimplementer for finding "typos in the
paper"). The paper's value is in the comment next to each such constant, so going
back to it is one edit. File names in the comments are HoST's:

  ground.py    legged_gym/envs/base/host_ground.py
  cfg_ground   legged_gym/envs/g1/g1_config_ground.py   (also _platform, _wall, _slope, _ground_prone)
  ppo.py       rsl_rl/rsl_rl/algorithms/ppo.py
  storage.py   rsl_rl/rsl_rl/storage/rollout_storage.py

PORTING.md lists what changed on the way to MuJoCo and why.
"""

from __future__ import annotations

import math

# --- Simulation and control --------------------------------------------------------
SIM_DT = 0.005  # 200 Hz PD (cfg_ground sim.dt)
DECIMATION = 4  # 50 Hz policy (cfg_ground control.decimation)
EPISODE_S = 10.0  # 500 policy steps (cfg_ground env.episode_length_s)
UNACTUATED_STEPS = 30  # first 0.6 s: no actions, zero observations; the robot falls and settles (cfg_ground env.unactuated_timesteps)
UNACTUATED_STEPS_WALL = 50  # wall: 1.0 s, while being pushed back against it (cfg_wall)
UNACTUATED_STEPS_OURS = 50  # our stairs / random-drop starts: 1.0 s to settle (they fall further)

# PD gains in simulation (paper App. B table; cfg_ground control.stiffness/damping).
# On the real G1 the paper raised hip/knee Kp to 200/275; that is for deployment, not here.
KP = {"hip": 150.0, "knee": 200.0, "ankle": 40.0, "shoulder": 100.0, "elbow": 100.0, "waist": 100.0, "wrist": 100.0}
KD = {"hip": 4.0, "knee": 6.0, "ankle": 2.0, "shoulder": 4.0, "elbow": 4.0, "waist": 4.0, "wrist": 4.0}
# The 6 joints the 23-DoF G1 doesn't have (waist roll/pitch, wrist pitch/yaw) are held
# at 0 like its locked joints. Stiffer than HoST's 100 on the waist: it carries the torso.
HELD_KP = {"waist": 200.0, "wrist": 20.0}
HELD_KD = {"waist": 5.0, "wrist": 0.5}

# "policy_step": the PD target q + beta*a is computed once per 20 ms policy step and
# held, as in the paper's Eq. 1 and on the real robot (Unitree's PD holds the target
# between commands). "substep": recomputed every 5 ms from the current q, as HoST's
# code does (ground.py:424-443), which makes it a constant torque Kp*beta*a.
TARGET_UPDATE = "policy_step"
# "host23": the policy moves HoST's 23 joints. "all29": all 29 (not the paper).
POLICY_JOINTS = "host23"

# --- Observations (ground.py:281-297, cfg_ground normalization/noise) ---------------
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0  # absolute joint angles, not relative to a default pose
OBS_SCALE_DOF_VEL = 0.05
HISTORY = 6  # current frame + the 5 before (cfg_ground env.num_actor_history)
# Uniform noise half-widths, in raw units (paper App. B table).
NOISE_ANG_VEL = 0.2
NOISE_GRAVITY = 0.05
NOISE_DOF_POS = 0.01
NOISE_DOF_VEL = 1.5
NOISE_BETA = 0.025  # the action-bound input, +-0.025 (ground.py:290)

# --- Curriculum: pulling force and action bound (paper App. B, ground.py:505-508) ---
PULL_FORCE_START = 200.0  # N. Code: 100 N on torso_link and 100 N on a point 0.2 m above it (cfg_ground curriculum.force)
PULL_FORCE_STEP = 20.0
PULL_POINT_OFFSET = 0.2  # m above torso_link origin, in its frame (HoST URDF keyframe_torso_joint)
FORCE_UPRIGHT_GZ = -0.8  # force only while pelvis projected gravity z < this (trunk near vertical; ground.py:113)
BETA_START = 1.0
BETA_STEP = 0.02
BETA_MIN = 0.25
CURRICULUM_HEAD_HEIGHT = 0.9  # m: mean head height of a reset batch above this -> step down (cfg_ground threshold_height)
WALL_PUSHBACK_N = 100.0  # wall: push toward the wall during the unactuated phase; 50 N on each of HoST's two torso bodies (host_wall.py:107-122)
BETA_EVAL = 0.25  # evaluation and play: the final bound, no force (scripts/eval/eval_ground.py)

# --- Heights, stages, targets ----------------------------------------------------
HEAD_OFFSET = (0.0, 0.0, 0.45)  # head point in torso_link frame (HoST URDF keyframe_head_joint)
H_STAGE1 = 0.45  # righting the body below, rising above (paper App. B)
H_STAGE2 = 0.65  # standing above
BASE_HEIGHT_TARGET = 0.75  # pelvis height once standing (cfg_ground rewards.base_height_target)
# Slope (heights measured from the feet there): 0.40 / 0.60 / 0.70 (cfg_slope).
SLOPE_STAGES = (0.40, 0.60, 0.70)
HEAD_HEIGHT_TARGET = 1.0  # tolerance bound for the head-height task reward
HEAD_MARGIN = 1.0
ORIENTATION_BOUND = 0.99  # tolerance bound on -gravity_z for the orientation task reward

# Pose the robot spawns in (cfg_ground init_state.default_joint_angles), then settles.
RESET_POSE = {
  ".*_hip_pitch_joint": -0.1, ".*_knee_joint": 0.3, ".*_ankle_pitch_joint": -0.2, ".*_elbow_joint": 0.8,
}
RESET_POSE_PLATFORM = {**RESET_POSE, ".*_knee_joint": 1.0, ".*_elbow_joint": 0.0}  # cfg_platform
RESET_POSE_WALL = {**RESET_POSE, ".*_hip_pitch_joint": -1.5, ".*_knee_joint": 0.0, ".*_elbow_joint": 0.0}  # cfg_wall (sitting)
# Upper-body pose to hold once standing (cfg_ground target_joint_angles; arms + waist yaw).
UPPER_TARGET = {"left_shoulder_roll_joint": 0.3, "right_shoulder_roll_joint": -0.3}

# --- Reward groups and weights (paper Tab. VI; cfg_ground rewards/constraints) ------
GROUPS = ("task", "regu", "style", "post")  # HoST: task, regu, style, target
GROUP_WEIGHTS = (2.5, 0.1, 1.0, 1.0)
GROUP_WEIGHTS_PRONE = (1.0, 0.1, 1.0, 1.0)  # cfg_ground_prone

W = {
  # regularization (all x dt by the runner, as HoST's constraints)
  "regu_dof_acc": -2.5e-7,
  "regu_action_rate": -1e-2,
  "regu_smoothness": -1e-2,
  "regu_torques": -2.5e-6,
  "regu_joint_power": -2.5e-5,
  "regu_dof_vel": -1e-3,  # paper: -1e-4
  "regu_tracking": -2.5e-4,  # paper: -2.5e-1
  "regu_dof_pos_limits": -100.0,
  "regu_dof_vel_limits": -1.0,
  # style
  "style_waist_deviation": -10.0,
  "style_hip_yaw_deviation": -10.0,
  "style_hip_roll_deviation": -10.0,
  "style_shoulder_roll_deviation": -2.5,
  "style_foot_displacement": 2.5,  # each foot
  "style_knee_deviation": -0.25,  # ground and prone; platform/slope/wall: -10
  "style_shank_orientation": 10.0,
  "style_ankle_parallel": 20.0,
  "style_feet_distance": -10.0,
  "style_feet_stumble": 0.0,  # ground; platform/slope/wall: -25
  "style_ang_vel_xy": 1.0,  # prone: 25
  "style_lower_body_deviation": 0.0,  # platform and wall: 1
  "style_hip_pitch_deviation": 0.0,  # prone: -10
  # post-task (after standing up)
  "post_ang_vel_xy": 10.0,
  "post_lin_vel_xy": 10.0,
  "post_feet_height_var": 2.5,
  "post_upper_pose": 10.0,
  "post_orientation": 10.0,
  "post_base_height": 10.0,
  "post_lower_body_deviation": 0.0,  # prone: 10
}
# Terrain-specific weights (cfg_platform, cfg_wall, cfg_slope, cfg_ground_prone).
W_PSW = {"style_knee_deviation": -10.0, "style_feet_stumble": -25.0}  # platform, slope, wall (and our stairs)
W_PLATFORM_WALL = {"style_lower_body_deviation": 1.0}
W_PRONE = {"style_hip_pitch_deviation": -10.0, "style_ang_vel_xy": 25.0, "post_lower_body_deviation": 10.0,
           "style_ankle_parallel": 0.0}

SOFT_DOF_POS_LIMIT = 0.9  # each bound x 0.9, as HoST (not around the centre)
SOFT_DOF_VEL_LIMIT = 0.9
# Joint speed limits (rad/s), Unitree's motor specs (g1_constants ElectricActuator).
VEL_LIMIT = {"hip_pitch": 32.0, "hip_yaw": 32.0, "waist_yaw": 32.0, "hip_roll": 20.0, "knee": 20.0,
             "ankle": 37.0, "shoulder": 37.0, "elbow": 37.0, "wrist_roll": 37.0}

# --- Terminations (ground.py:168-179; platform/wall/slope check_termination) -------
MAX_DOF_VEL = 300.0  # rad/s: simulation blew up
MAX_BASE_VEL = 20.0  # m/s

# --- Domain randomization (paper Tab. II; cfg_ground domain_rand) -----------------
PAYLOAD_KG = (-2.0, 5.0)  # added to torso_link
COM_OFFSET_XY = 0.12  # m, torso_link (code 0.03 x 4)
COM_OFFSET_Z = 0.06  # paper: 0.08 (code 0.03 x 2)
LINK_MASS_SCALE = (0.8, 1.2)
FRICTION = (0.1, 1.0)
# Restitution U(0,1): MuJoCo contacts have no restitution coefficient. Not ported.
KP_SCALE = (0.85, 1.15)
KD_SCALE = (0.85, 1.15)
KP_KD_SCALE_PRONE = (0.85, 1.25)
MOTOR_STRENGTH = (0.9, 1.1)
ACTUATION_OFFSET = 0.05  # x effort limit, per joint per episode ("torque RFI")
DELAY_SUBSTEPS = (0, 4)  # 0-20 ms in 5 ms substeps (code max_delay_timesteps=5). Paper: U(0, 100) ms
INIT_JOINT_SCALE = (0.9, 1.1)  # platform/wall/slope: (0.8, 1.2)
INIT_JOINT_SCALE_WIDE = (0.8, 1.2)
INIT_JOINT_OFFSET = 0.1

# --- Terrains (HoST terrain_single.py + configs; ours below) ------------------------
TILE = 8.0  # m, square tiles
PLATFORM_HEIGHT = (0.22, 0.46)  # m. Code: 0.10 + 0.4 d, d in [0.3, 0.9]. Paper text: 0.20-0.92
WALL_SLOPES = (0.25, 0.5, 0.75, 1.0, 2.0, 2.5, 5.0, 7.5, 10.0)  # rise/run: 14 to 84 deg (cfg_wall difficulty)
WALL_HEIGHT = 1.0  # m (HoST caps the incline at 1 m)
SLOPE_SLOPES = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.26)  # 1.1 to 14.6 deg (cfg_slope)
RAMP_MAX_DEG = 17.0  # ours: the course's ramps are 15 deg
# Course stairs (sim/competition/README.md): 5 risers of 20.3 cm on 24 cm treads, 1.22 m landing.
STAIR_RISER = (0.10, 0.215)  # m, easiest to hardest row (course: 0.203)
STAIR_TREAD = 0.24
STAIR_RISERS = 5
STAIR_LANDING = 1.22
STAIR_WIDTH = 3.0  # wide enough to lie across; the course's 0.77 m side panels are not modelled

# --- PPO (paper App. B; legged_robot_config.py PPO; cfg_ground G1CfgPPO) -----------
ACTOR_HIDDEN = (512, 256, 128)  # + tanh on the output
CRITIC_HIDDEN = (512, 256)
INIT_NOISE_STD = 0.8
LEARNING_RATE = 1e-3
SCHEDULE = "adaptive"
DESIRED_KL = 0.01
CLIP_PARAM = 0.2
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 1.0
NUM_EPOCHS = 5
NUM_MINI_BATCHES = 4
GAMMA = 0.99
LAM = 0.95
STEPS_PER_ENV = 50
MAX_GRAD_NORM = 1.0
# L2C2 (ppo.py:182-195). Paper: lambda_pi = 1, lambda_V = 0.1. The code's effective
# policy weight is upper*lower/(upper-lower) = 0.111 and the value weight 0.1 x that.
L2C2_UPPER = 1.0
L2C2_LOWER = 0.1
L2C2_VALUE = 0.1  # wall: 0.25
L2C2_VALUE_WALL = 0.25


def l2c2_coefs(value_coef: float = L2C2_VALUE) -> tuple[float, float]:
  policy = L2C2_UPPER * L2C2_LOWER / (L2C2_UPPER - L2C2_LOWER)
  return policy, value_coef * policy


# Training iterations per task (cfg_* max_iterations; the README's GIFs are at
# ground 10000, platform 12000, wall 4000, slope 8000).
ITERATIONS = {
  "HoST-Ground": 12000,
  "HoST-Prone": 12000,
  "HoST-Platform": 12000,
  "HoST-Wall": 7500,
  "HoST-Slope": 10000,
  "HoST-Stairs": 12000,
  "HoST-Supine-Mixed": 20000,
  "HoST-Prone-Mixed": 20000,
  "HoST-All": 25000,
}

DEG = math.pi / 180.0
