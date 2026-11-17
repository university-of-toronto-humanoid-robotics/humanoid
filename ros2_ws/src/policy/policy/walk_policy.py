"""Walking policies for walk_controller, chosen by its policy_type parameter.

A policy takes what the robot measures (the pelvis IMU and the joints), a velocity command (vx, vy, wz) and,
for some, the terrain scan, and returns joint position targets. No ROS and no MuJoCo in here. Add a new type by
writing a class with the same interface and registering it in POLICY_TYPES.
"""
import numpy as np

from height_scan.scan import N_POINTS
from policy import g1


def load_actor(path):
    """An rsl_rl checkpoint's actor: its observation normaliser (mean, std) and MLP layers [(weight, bias)]."""
    import torch  # only to read the checkpoint
    actor = torch.load(path, map_location='cpu', weights_only=True)['actor_state_dict']
    ids = sorted({int(k.split('.')[1]) for k in actor if k.startswith('mlp.')})
    layers = [(actor[f'mlp.{i}.weight'].numpy(), actor[f'mlp.{i}.bias'].numpy()) for i in ids]
    return actor['obs_normalizer._mean'].numpy()[0], actor['obs_normalizer._std'].numpy()[0], layers


def gravity_in_body(quat_wxyz):
    """The direction of gravity (a unit vector) in the frame of a body with this orientation."""
    w, x, y, z = quat_wxyz
    return -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


class RslRlG1Flat:
    """mjlab's Unitree-G1-Flat actor, from an rsl_rl checkpoint: its observation normaliser, then an ELU MLP.

    Returns the mean action, without sampling.
    """

    dt = 0.02            # s, 50 Hz, as in training
    GAIT_PERIOD = 0.6    # s, the policy's phase clock
    COMMAND_RANGE = ((-0.5, 1.0), (-0.5, 0.5), (-0.5, 0.5))   # vx, vy (m/s), wz (rad/s) it was trained on
    needs_scan = False
    extra_obs = 0        # observations after the 98 proprioceptive ones

    def __init__(self, mean, std, layers):
        self.mean, self.std, self.layers = mean, std, layers
        self.default = g1.default_pose()
        self.scale = 0.25 * g1.effort() / g1.stiffness()  # G1_ACTION_SCALE
        n = len(self.default)
        expected = 3 + 3 + 3 + 2 + 3 * n + self.extra_obs
        if layers[0][0].shape[1] != expected or layers[-1][0].shape[0] != n:
            raise ValueError(f'The checkpoint takes {layers[0][0].shape[1]} observations and gives '
                             f'{layers[-1][0].shape[0]} actions; this policy type expects {expected} and {n}')
        self.reset()

    def reset(self):
        self.last_action = np.zeros(len(self.default))
        self.steps = 0

    def observation(self, gyro, quat_wxyz, joint_pos, joint_vel, command, scan=None):
        """Unitree-G1-Flat's actor observations, in order."""
        phase = (self.steps * self.dt) % self.GAIT_PERIOD / self.GAIT_PERIOD
        clock = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)] if np.linalg.norm(command) >= 0.1 else [0, 0]
        return np.concatenate([
            gyro,  # pelvis angular velocity in the pelvis frame
            gravity_in_body(quat_wxyz), command, clock,
            joint_pos - self.default, joint_vel, self.last_action,
        ])

    def __call__(self, gyro, quat_wxyz, joint_pos, joint_vel, command, scan=None):
        """Joint position targets for the next policy step.

        gyro: pelvis angular velocity (rad/s), and quat_wxyz: pelvis orientation, both from the pelvis IMU.
        joint_pos, joint_vel: the 29 joints in g1.JOINTS order. scan: the 693-point terrain scan, if needs_scan.
        """
        x = self.observation(gyro, quat_wxyz, joint_pos, joint_vel, command, scan)
        x = (x - self.mean) / (self.std + 1e-2)  # rsl_rl's normaliser, eps 1e-2
        for i, (weight, bias) in enumerate(self.layers):
            x = weight @ x + bias
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))
        self.last_action = x
        self.steps += 1
        return self.default + self.scale * x


class RslRlG1HeightMap(RslRlG1Flat):
    """The team's height-map walking tasks, G1-Flat-HeightMap and G1-Rough-HeightMap (training/g1_comp): the flat
    actor's 98 observations followed by the 693-point terrain scan, 791 in all.

    The G1-Comp-<Rough/Course/Crouch-Test> tasks (793: they add a height command and an arm-override flag) are a
    different family, and won't load.
    """

    needs_scan = True
    extra_obs = N_POINTS

    def observation(self, gyro, quat_wxyz, joint_pos, joint_vel, command, scan=None):
        return np.concatenate([super().observation(gyro, quat_wxyz, joint_pos, joint_vel, command), scan])


POLICY_TYPES = {'rsl_rl_g1_flat': RslRlG1Flat, 'rsl_rl_g1_heightmap': RslRlG1HeightMap}


def make_policy(kind, path):
    if kind not in POLICY_TYPES:
        raise ValueError(f'Unknown policy type "{kind}"; known: {", ".join(POLICY_TYPES)}')
    return POLICY_TYPES[kind](*load_actor(path))
