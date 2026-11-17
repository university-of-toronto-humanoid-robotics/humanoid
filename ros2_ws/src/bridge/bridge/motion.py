"""Motion scripts for mujoco_g1_walk_sim, chosen by a scenario's motion.type.

Each gives the policy a velocity command (vx, vy, wz) in the robot's heading frame, from the time since walking
started and the true pelvis pose. Add a new type by writing a class with command(), done and progress, and
registering it in MOTION_TYPES.
"""
import numpy as np


class Waypoints:
    """Turns toward the next (x, y) waypoint in the scene frame and walks when roughly facing it."""

    REACHED = 0.4   # m
    MAX_TURN = 0.5  # rad/s

    def __init__(self, points, speed=0.5, laps=1):
        self.points = [tuple(p) for p in points] * int(laps)
        self.speed = speed
        self.index = 0

    def command(self, t, pos, yaw):
        while self.index < len(self.points):
            dx, dy = np.array(self.points[self.index]) - pos[:2]
            if np.hypot(dx, dy) > self.REACHED:
                break
            self.index += 1
        if self.done:
            return np.zeros(3)
        error = np.arctan2(np.sin(np.arctan2(dy, dx) - yaw), np.cos(np.arctan2(dy, dx) - yaw))
        forward = self.speed * max(np.cos(error), 0.0) ** 2 if abs(error) < 0.8 else 0.0
        return np.array([forward, 0.0, float(np.clip(error, -self.MAX_TURN, self.MAX_TURN))])

    @property
    def done(self):
        return self.index == len(self.points)

    @property
    def progress(self):
        return f'{self.index} of {len(self.points)} waypoints'


class VelocityScript:
    """Open-loop commands: a list of segments, each held for its duration (s)."""

    def __init__(self, segments):
        self.segments = [(float(s['duration']), np.array([s.get('vx', 0.0), s.get('vy', 0.0), s.get('wz', 0.0)]))
                         for s in segments]
        self.ends = np.cumsum([d for d, _ in self.segments])
        self.t = 0.0

    def command(self, t, pos, yaw):
        self.t = t
        i = int(np.searchsorted(self.ends, t, side='right'))
        return self.segments[i][1] if i < len(self.segments) else np.zeros(3)

    @property
    def done(self):
        return self.t >= self.ends[-1]

    @property
    def progress(self):
        return f'{int(np.searchsorted(self.ends, self.t, side="right"))} of {len(self.segments)} segments'


MOTION_TYPES = {'waypoints': Waypoints, 'velocity': VelocityScript}


def make_motion(config, speed=None):
    """A motion from a scenario's motion section. speed, if given, overrides a waypoint route's speed."""
    config = dict(config)
    kind = config.pop('type')
    if kind not in MOTION_TYPES:
        raise ValueError(f'Unknown motion type "{kind}"; known: {", ".join(MOTION_TYPES)}')
    if kind == 'waypoints' and speed:
        config['speed'] = speed
    return MOTION_TYPES[kind](**config)
