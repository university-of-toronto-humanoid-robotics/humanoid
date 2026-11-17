from types import SimpleNamespace

import numpy as np
import pytest

from height_scan.scan import N_POINTS
from policy import g1, lowcmd
from policy.walk_policy import RslRlG1Flat, RslRlG1HeightMap, gravity_in_body

N = 29
UPRIGHT = (1.0, 0.0, 0.0, 0.0)


def actor(observations, rng):
    """A random network of the right shape, as load_actor would return it."""
    layers = [(rng.normal(size=(16, observations)), rng.normal(size=16)), (rng.normal(size=(N, 16)), rng.normal(size=N))]
    return rng.normal(size=observations), rng.uniform(0.5, 2.0, observations), layers


def fake_lowcmd(rng):
    motors = [SimpleNamespace(mode=1, q=float(q), dq=0.0, tau=0.0, kp=40.0, kd=2.5, reserve=0)
              for q in rng.normal(size=lowcmd.N_MOTORS)]
    return SimpleNamespace(mode_pr=0, mode_machine=5, motor_cmd=motors, reserve=[0, 0, 0, 0])


def unitree_crc(words):
    """Unitree's crc32_core, as written in unitree_ros2's motor_crc_hg.cpp."""
    crc = 0xFFFFFFFF
    for word in words:
        bit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ 0x04C11DB7
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if word & bit:
                crc ^= 0x04C11DB7
            bit >>= 1
    return crc


def test_joints_and_gains_match_unitrees_deploy_config():
    # training/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml
    assert len(g1.JOINTS) == N
    assert g1.JOINTS[3] == 'left_knee_joint' and g1.JOINTS[12] == 'waist_yaw_joint'
    assert g1.JOINTS[22] == 'right_shoulder_pitch_joint'
    assert g1.default_pose() == pytest.approx(
        [-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0,
         0.35, 0.18, 0, 0.87, 0, 0, 0, 0.35, -0.18, 0, 0.87, 0, 0, 0])
    assert g1.stiffness()[[0, 1, 4, 15, 20]] == pytest.approx([40.2, 99.1, 28.5, 14.3, 16.8], abs=0.05)
    assert g1.damping()[[0, 1, 4, 15, 20]] == pytest.approx([2.6, 6.3, 1.8, 0.9, 1.1], abs=0.05)


def test_crc_matches_unitrees_and_notices_a_change():
    msg = fake_lowcmd(np.random.default_rng(0))
    fields = [msg.mode_pr, msg.mode_machine]
    for m in msg.motor_cmd:
        fields += [m.mode, m.q, m.dq, m.tau, m.kp, m.kd, m.reserve]
    words = np.frombuffer(lowcmd._LAYOUT.pack(*fields, *msg.reserve), dtype='<u4')
    assert len(words) == 1 + 7 * 35 + 4     # the C struct, without its crc
    assert lowcmd.crc(msg) == unitree_crc(int(w) for w in words)
    before = lowcmd.crc(msg)
    msg.motor_cmd[7].q += 0.01
    assert lowcmd.crc(msg) != before


def test_gravity_in_body():
    assert gravity_in_body(UPRIGHT) == pytest.approx([0, 0, -1])
    half = np.sqrt(0.5)
    assert gravity_in_body((half, 0, half, 0)) == pytest.approx([1, 0, 0])    # pitched 90 deg nose down
    assert gravity_in_body((half, half, 0, 0)) == pytest.approx([0, -1, 0])   # rolled 90 deg onto its right side


def test_flat_observation_order_and_gait_clock():
    rng = np.random.default_rng(1)
    policy = RslRlG1Flat(*actor(98, rng))
    gyro, pos, vel = rng.normal(size=3), rng.normal(size=N), rng.normal(size=N)
    obs = policy.observation(gyro, UPRIGHT, pos, vel, np.array([0.5, 0.0, 0.1]))
    assert len(obs) == 98
    assert obs[:3] == pytest.approx(gyro)
    assert obs[3:6] == pytest.approx([0, 0, -1])
    assert obs[6:9] == pytest.approx([0.5, 0.0, 0.1])
    assert obs[9:11] == pytest.approx([0, 1])                   # start of the gait cycle
    assert obs[11:40] == pytest.approx(pos - g1.default_pose())
    assert obs[40:69] == pytest.approx(vel)
    assert obs[69:] == pytest.approx(np.zeros(N))               # no action yet
    standing = policy.observation(gyro, UPRIGHT, pos, vel, np.zeros(3))
    assert standing[9:11] == pytest.approx([0, 0])              # the clock stops with no command


def test_action_becomes_targets_and_is_remembered():
    rng = np.random.default_rng(2)
    policy = RslRlG1Flat(*actor(98, rng))
    args = (np.zeros(3), UPRIGHT, g1.default_pose(), np.zeros(N), np.array([0.5, 0, 0]))
    target = policy(*args)
    scale = 0.25 * g1.effort() / g1.stiffness()
    assert target == pytest.approx(g1.default_pose() + scale * policy.last_action)
    assert policy.observation(*args)[69:] == pytest.approx(policy.last_action)
    assert policy.steps == 1
    policy.reset()
    assert policy.steps == 0 and not policy.last_action.any()


def test_heightmap_policy_appends_the_scan():
    rng = np.random.default_rng(3)
    policy = RslRlG1HeightMap(*actor(98 + N_POINTS, rng))
    scan = rng.uniform(0.5, 1.0, N_POINTS)
    obs = policy.observation(np.zeros(3), UPRIGHT, np.zeros(N), np.zeros(N), np.zeros(3), scan)
    assert len(obs) == 791 and obs[98:] == pytest.approx(scan)


def test_wrong_size_checkpoint_is_refused():
    with pytest.raises(ValueError, match='793'):
        RslRlG1HeightMap(*actor(793, np.random.default_rng(4)))
