"""The G1 as mjlab builds it for training, in any scene that includes description/g1/g1_29dof_mjlab.xml.

The constants come from unitree_rl_mjlab's g1_constants.py and its velocity task, so update them together.
"""
import re

import mujoco
import numpy as np

PHYSICS_DT = 0.005          # s, as in training (and the Mid-360's 200 Hz IMU)
START_HEIGHT = 0.8          # m, pelvis height above the ground at spawn
TERRAIN = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # geom groups of the scene; the robot's are 2 (visual), 3

# Actuators: each joint is a <position> actuator whose gains come from its motor's reflected inertia
# (armature), a 10 Hz natural frequency and a damping ratio of 2.
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
FOOT_GEOM = r'(left|right)_foot[1-7]_collision'


def match_one(table, name):
    hits = [value for pattern, *value in table if re.fullmatch(pattern, name)]
    if len(hits) != 1:
        raise ValueError(f'{name} matches {len(hits)} entries, expected 1')
    return hits[0]


def hinge_names(model):
    return [model.joint(int(j)).name for j in np.nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]]


def default_pose(model):
    return np.array([next((v for p, v in DEFAULT_POSE.items() if re.fullmatch(p, n)), 0.0)
                     for n in hinge_names(model)])


def stiffness(model):
    return np.array([match_one(MOTORS, n)[0] for n in hinge_names(model)]) * NATURAL_FREQ**2


def effort(model):
    return np.array([match_one(MOTORS, n)[1] for n in hinge_names(model)])


def build_model(scene, description):
    """The scene's G1 set up the way mjlab sets it up for training, plus the Mid-360 from our description."""
    spec = mujoco.MjSpec.from_file(scene)
    robot = spec.body('pelvis')

    # Position actuators without control limits, force-limited to the motor's effort limit.
    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        armature, limit = match_one(MOTORS, joint.name)
        kp = armature * NATURAL_FREQ**2
        kd = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
        actuator = spec.add_actuator(name=joint.name, target=joint.name)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm[0] = kp
        actuator.biasprm[1] = -kp
        actuator.biasprm[2] = -kd
        actuator.inheritrange = 0.0
        actuator.ctrllimited = False
        actuator.forcelimited = True
        actuator.forcerange[:] = [-limit, limit]
        joint.armature = armature
        joint.frictionloss = 0.0

    # Collisions as in mjlab's FULL_COLLISION: only the *_collision capsules, self-collisions included.
    for geom in robot.find_all(mujoco.mjtObj.mjOBJ_GEOM):
        if re.match(r'.*_collision', geom.name):
            foot = re.match(FOOT_GEOM, geom.name) is not None
            geom.contype, geom.conaffinity = 1, 1
            geom.condim = 3 if foot else 1
            geom.priority = 1 if foot else 0
            if foot:
                geom.friction[0] = 0.6
        else:
            geom.contype, geom.conaffinity = 0, 0

    # The Mid-360 and its IMU, copied from our description so the two sims can't drift apart.
    mount = mujoco.MjSpec.from_file(description)
    torso = spec.body('torso_link')
    for name in ('mid360', 'mid360_imu'):
        site = mount.site(name)
        torso.add_site(name=name, pos=site.pos, quat=site.quat, size=site.size)
    for name in ('mid360-imu-angular-velocity', 'mid360-imu-linear-acceleration'):
        sensor = mount.sensor(name)
        spec.add_sensor(name=name, type=sensor.type, objtype=sensor.objtype, objname=sensor.objname,
                        noise=sensor.noise)

    model = spec.compile()
    hinges = np.nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
    assert (model.actuator_trnid[:, 0] == hinges).all(), 'actuators must follow joint order, like the policy'

    # Unitree-G1-Flat's solver settings, with mjlab's defaults for the rest.
    model.opt.timestep = PHYSICS_DT
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    model.opt.tolerance = 1e-8
    model.opt.ls_tolerance = 0.01
    model.opt.ccd_iterations = 50
    model.opt.impratio = 1.0
    return model


def ground_below(model, data, x, y, top=5.0):
    """Height of the scene (not the robot) below (x, y), or None if nothing is there."""
    geom = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, np.array([x, y, top]), np.array([0.0, 0.0, -1.0]), TERRAIN, 1, -1, geom)
    return None if dist < 0 else top - dist


def spawn(model, data, x, y, yaw):
    """Stand the robot in its default pose at (x, y), facing yaw, with its feet just above the ground.

    The ground is the highest point under the feet (a 22 x 26 cm patch turned with the robot), so a spawn on a
    ramp or next to a step doesn't put a foot inside it. Returns that ground height.
    """
    mujoco.mj_forward(model, data)
    c, s = np.cos(yaw), np.sin(yaw)
    heights = [ground_below(model, data, x + c * a - s * b, y + s * a + c * b)
               for a in np.linspace(-0.11, 0.11, 5) for b in np.linspace(-0.13, 0.13, 5)]
    ground = max((h for h in heights if h is not None), default=0.0)
    data.qpos[:3] = [x, y, ground + START_HEIGHT]
    data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    data.qpos[7:] = default_pose(model)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return ground


def sole_depth(model):
    """How far the soles are below the ankle_roll_link origins, standing in the default pose: robot geometry."""
    data = mujoco.MjData(model)
    data.qpos[7:] = default_pose(model)
    mujoco.mj_kinematics(model, data)
    ankle = data.body('left_ankle_roll_link').xpos[2]
    soles = [data.geom_xpos[g][2] - model.geom_size[g][0] for g in range(model.ngeom)
             if re.fullmatch(r'left_foot[1-7]_collision', model.geom(g).name)]
    return float(ankle - min(soles))
