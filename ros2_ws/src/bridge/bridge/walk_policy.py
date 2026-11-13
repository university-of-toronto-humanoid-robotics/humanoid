"""Walking policies for mujoco_g1_walk_sim, chosen by a scenario's policy.type.

A policy takes MuJoCo's state and a velocity command (vx, vy, wz) and returns joint position targets. Add a new
type by writing a class with the same interface (dt, __call__) and registering it in POLICY_TYPES. Types that read
the terrain set needs_scan and are given a scan source (bridge/height_scan.py).
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from bridge import g1_model
from bridge.height_scan import N_POINTS


class RslRlG1Flat:
    """mjlab's Unitree-G1-Flat actor, from an rsl_rl checkpoint: its observation normaliser, then an ELU MLP.

    Returns the mean action, without sampling. Command ranges it was trained on: vx -0.5 to 1.0, vy +-0.5,
    wz +-0.5.
    """

    dt = 0.02            # s, 50 Hz: 4 physics steps per action, as in training
    GAIT_PERIOD = 0.6    # s, the policy's phase clock
    needs_scan = False
    extra_obs = 0        # observations after the 98 proprioceptive ones
    model_options = {}   # MuJoCo solver settings the task trained with, where they differ from Unitree-G1-Flat's

    def __init__(self, path, model):
        import torch  # only to read the checkpoint
        actor = torch.load(path, map_location='cpu', weights_only=True)['actor_state_dict']
        self.mean = actor['obs_normalizer._mean'].numpy()[0]
        self.std = actor['obs_normalizer._std'].numpy()[0]
        ids = sorted({int(k.split('.')[1]) for k in actor if k.startswith('mlp.')})
        self.layers = [(actor[f'mlp.{i}.weight'].numpy(), actor[f'mlp.{i}.bias'].numpy()) for i in ids]

        self.default = g1_model.default_pose(model)
        self.scale = 0.25 * g1_model.effort(model) / g1_model.stiffness(model)  # G1_ACTION_SCALE
        self.last_action = np.zeros(len(self.default))
        self.steps = 0
        n = len(self.default)
        expected = 3 + 3 + 3 + 2 + 3 * n + self.extra_obs
        if self.layers[0][0].shape[1] != expected or self.layers[-1][0].shape[0] != n:
            raise ValueError(f'{path} takes {self.layers[0][0].shape[1]} observations and gives '
                             f'{self.layers[-1][0].shape[0]} actions; this policy type expects {expected} and {n}')

    def observation(self, data, command):
        """Unitree-G1-Flat's actor observations, in order."""
        w, x, y, z = data.qpos[3:7]
        gravity = Rotation.from_quat([x, y, z, w]).inv().apply([0.0, 0.0, -1.0])
        phase = (self.steps * self.dt) % self.GAIT_PERIOD / self.GAIT_PERIOD
        clock = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)] if np.linalg.norm(command) >= 0.1 else [0, 0]
        return np.concatenate([
            data.qvel[3:6],  # pelvis angular velocity in the pelvis frame, which the pelvis gyro measures
            gravity, command, clock,
            data.qpos[7:] - self.default, data.qvel[6:], self.last_action,
        ])

    def __call__(self, data, command):
        """Joint position targets for the next policy step."""
        x = (self.observation(data, command) - self.mean) / (self.std + 1e-2)  # rsl_rl's normaliser, eps 1e-2
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

    This will not work for the G1-Comp-<Rough/Course/Crouch-Test> tasks 
    (793: they add a height command and an arm-override flag) are a different family.
    """

    needs_scan = True
    extra_obs = N_POINTS
    model_options = {'ccd_iterations': 500}   # G1-Rough-HeightMap takes Unitree's rough-terrain solver settings

    def __init__(self, path, model, scan):
        self.scan = scan   # callable: MjData -> the 693 scan values
        super().__init__(path, model)

    def observation(self, data, command):
        return np.concatenate([super().observation(data, command), self.scan(data)])


POLICY_TYPES = {'rsl_rl_g1_flat': RslRlG1Flat, 'rsl_rl_g1_heightmap': RslRlG1HeightMap}


def make_policy(kind, path, model: mujoco.MjModel, scan=None):
    if kind not in POLICY_TYPES:
        raise ValueError(f'Unknown policy type "{kind}"; known: {", ".join(POLICY_TYPES)}')
    cls = POLICY_TYPES[kind]
    policy = cls(path, model, scan) if cls.needs_scan else cls(path, model)
    for name, value in cls.model_options.items():
        setattr(model.opt, name, value)
    return policy
