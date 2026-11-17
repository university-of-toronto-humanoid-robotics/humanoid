"""The G1 as mjlab builds it for training, in any scene that includes description/g1/g1_29dof_mjlab.xml.

The robot's own numbers (joint order, standing pose, motor gains) are in the policy package's g1.py, shared with
the controller. The solver settings here come from unitree_rl_mjlab's velocity task, so update them together.
"""
import re

import mujoco
import numpy as np

from policy import g1

PHYSICS_DT = 0.005          # s, as in training (and the Mid-360's 200 Hz IMU)
START_HEIGHT = 0.8          # m, pelvis height above the ground at spawn
TERRAIN = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # geom groups of the scene; the robot's are 2 (visual), 3

FOOT_GEOM = r'(left|right)_foot[1-7]_collision'
GANTRY = 'gantry'


def hinge_names(model):
    return [model.joint(int(j)).name for j in np.nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]]


def default_pose(model):
    return g1.default_pose(hinge_names(model))


def build_model(scene, description):
    """The scene's G1 set up the way mjlab sets it up for training, plus the Mid-360 from our description."""
    spec = mujoco.MjSpec.from_file(scene)
    robot = spec.body('pelvis')

    # Position actuators without control limits, force-limited to the motor's effort limit. They start with
    # the training gains; set_motor_commands replaces them with whatever /lowcmd asks for.
    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        armature, limit = g1.motor(joint.name)
        kp = armature * g1.NATURAL_FREQ**2
        kd = 2.0 * g1.DAMPING_RATIO * armature * g1.NATURAL_FREQ
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

    # The gantry: the pelvis welded to the world, switched on and placed by hold_on_gantry.
    gantry = spec.add_equality(name=GANTRY, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                               name1='world', name2='pelvis')
    gantry.active = False

    model = spec.compile()
    hinges = np.nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
    assert (model.actuator_trnid[:, 0] == hinges).all(), 'actuators must follow joint order'
    assert hinge_names(model) == g1.JOINTS, 'joints must be in Unitree\'s motor order: actuator i is motor i'

    # Unitree-G1-Flat's solver settings, with mjlab's defaults for the rest. ccd_iterations is the rough-terrain
    # tasks' 500 (flat uses 50): it only matters where a collision is hard to resolve.
    model.opt.timestep = PHYSICS_DT
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    model.opt.tolerance = 1e-8
    model.opt.ls_tolerance = 0.01
    model.opt.ccd_iterations = 500
    model.opt.impratio = 1.0
    return model


def set_motor_commands(model, data, q, dq, tau, kp, kd):
    """Drive each motor as the real one obeys /lowcmd: torque = tau + kp (q - angle) + kd (dq - speed).

    The gains go into the position actuators rather than a torque computed here, so MuJoCo integrates the damping
    implicitly, as it did in training. With the training gains and dq = tau = 0 this is exactly training's motor.
    """
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 0] = tau + kd * dq
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -kd
    data.ctrl[:] = q


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


def hold_on_gantry(model, data):
    """Fix the pelvis where it is now, as the real gantry's strap does. The joints stay free to move."""
    eq = model.equality(GANTRY)
    eq.data[:] = 0.0
    eq.data[3:10] = data.qpos[:7]   # the pelvis's pose in the world; before it, the weld point: the pelvis origin
    eq.data[10] = 1.0               # torque scale
    data.eq_active[eq.id] = 1


def release_gantry(data):
    data.eq_active[data.model.equality(GANTRY).id] = 0


def sole_depth(model):
    """How far the soles are below the ankle_roll_link origins, standing in the default pose: robot geometry."""
    data = mujoco.MjData(model)
    data.qpos[7:] = default_pose(model)
    mujoco.mj_kinematics(model, data)
    ankle = data.body('left_ankle_roll_link').xpos[2]
    soles = [data.geom_xpos[g][2] - model.geom_size[g][0] for g in range(model.ngeom)
             if re.fullmatch(r'left_foot[1-7]_collision', model.geom(g).name)]
    return float(ankle - min(soles))
