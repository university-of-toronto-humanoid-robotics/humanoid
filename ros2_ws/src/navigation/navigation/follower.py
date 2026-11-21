"""Turns a route and the robot's pose into a velocity command (vx, vy, wz), the way walk_controller takes /cmd_vel.

Two ways of moving:
  following  between waypoints. Pure pursuit: aim at a point LOOKAHEAD metres further along the route, turn toward
             it, and walk toward it: forward for the part ahead, sideways for the part off to the side. If it is
             behind the robot, turn on the spot first. The route is the straight lines from the
             start through each waypoint. A walk-through waypoint is passed without stopping: the aim point moves on
             to the next line before the robot gets there, so it cuts the corner smoothly.
  aligning   within APPROACH of a stop. The robot steps straight to the spot (sideways too: a humanoid can),
             then turns on the spot to the stop's heading. Once inside the tolerance for SETTLE seconds the stop is
             reached; it waits there for the stop's wait time, then goes on. If it can't get inside within ALIGN_TIMEOUT,
             twice the tolerance will do, so one fussy stop can't stall the whole run.

No ROS in here, so it can be tested on its own.
"""
import numpy as np

LOOKAHEAD = 0.5      # m ahead along the route that pure pursuit aims at
PASSED = 0.3         # m from a walk-through waypoint that counts as passing it
APPROACH = 0.4       # m from a stop at which the robot stops walking and lines up
TURN_GAIN = 1.5      # rad/s of turn per rad of heading error
MAX_TURN = 0.5       # rad/s: the policies were trained up to this
FACING = np.pi / 2   # rad: with the aim point further round than this (behind), the robot turns on the spot first
MAX_SIDE = 0.3       # m/s of sidestep while following
SLOW_GAIN = 0.8      # m/s of speed per m from the next stop, so it slows down coming in
MIN_SPEED = 0.15     # m/s: walking policies barely move below about this
ALIGN_GAIN = 1.0     # m/s per m of position error while lining up
ALIGN_SPEED = 0.2    # m/s, the most while lining up
# The policies treat a command under about 0.1 (|vx| + |vy| + |wz|) as "stand still" (standing_threshold in
# training/g1_comp), so while still outside the tolerance the robot is asked for at least these.
ALIGN_MIN = 0.15     # m/s
TURN_MIN = 0.2       # rad/s
SETTLE = 0.5         # s inside the tolerance before a stop counts as reached
STUCK_TIME = 3.0     # s without PROGRESS after which lining up asks for more speed
PROGRESS = 0.02      # m (or 0.07 rad of heading) that counts as getting closer
MAX_BOOST = 3.0      # the most the lining-up commands are scaled up by. Overshooting the spot scales back down.
MAX_ALIGN_SPEED = 0.45  # m/s, even boosted: the flat policy barely steps forward below about 0.3
ALIGN_TIMEOUT = 20.0  # s lining up after which twice the tolerance is good enough
MAX_ACCEL = np.array([0.5, 0.5, 1.0])   # m/s^2, m/s^2, rad/s^2: how fast the command may change


def wrap(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class Follower:

    def __init__(self, route, start_xy):
        self.route = route
        self.index = 0                                   # the waypoint being walked to
        self.segment_start = np.asarray(start_xy, dtype=float)
        self.state = 'following'
        self.settled_since = self.wait_until = self.align_since = None
        self.loose = False   # whether the last stop was only reached within twice its tolerance (ALIGN_TIMEOUT)
        self.last_command, self.last_t = np.zeros(3), None

    @property
    def target(self):
        return self.route.waypoints[self.index]

    @property
    def done(self):
        return self.state == 'arrived'

    @property
    def progress(self):
        return f'waypoint {min(self.index + 1, len(self.route.waypoints))} of {len(self.route.waypoints)}'

    def update(self, t, xy, yaw):
        """The command at time t (s), for the pelvis at xy (m, course frame) facing yaw (rad)."""
        xy = np.asarray(xy, dtype=float)
        if self.state == 'arrived':
            command = np.zeros(3)
        elif self.state == 'waiting':
            command = np.zeros(3)
            if t >= self.wait_until:
                self.advance(xy)
        else:
            command = self.follow(xy, yaw) if self.state == 'following' else self.align(t, xy, yaw)
        return self.limit(t, command)

    def advance(self, xy):
        """On to the next waypoint, or arrived after the last."""
        self.segment_start = self.target.xy
        if self.index + 1 == len(self.route.waypoints):
            self.state = 'arrived'
            return
        self.index += 1
        self.state = 'following'

    def follow(self, xy, yaw):
        # Pass walk-through waypoints that are close, or already behind the robot along the line.
        while not self.target.stop:
            a, b = self.segment_start, self.target.xy
            along = np.dot(xy - a, b - a) / max(np.linalg.norm(b - a), 1e-6)
            if np.linalg.norm(b - xy) > PASSED and along < np.linalg.norm(b - a):
                break
            self.advance(xy)
        to_stop = self.distance_to_next_stop(xy)
        if self.target.stop and np.linalg.norm(self.target.xy - xy) < APPROACH:
            self.state, self.align_since = 'aligning', None
            return np.zeros(3)

        aim = self.aim_point(xy)
        error = wrap(np.arctan2(*(aim - xy)[::-1]) - yaw)
        turn = float(np.clip(TURN_GAIN * error, -MAX_TURN, MAX_TURN))
        if abs(error) > FACING:
            return np.array([0.0, 0.0, turn])
        # Walk toward the aim point, sidestepping for the part of it that is off to the side while turning to face it.
        speed = min(self.route.speed, max(MIN_SPEED, SLOW_GAIN * to_stop))
        return np.array([speed * np.cos(error), float(np.clip(speed * np.sin(error), -MAX_SIDE, MAX_SIDE)), turn])

    def aim_point(self, xy):
        """The point LOOKAHEAD further along the route than the robot, running on past walk-through waypoints."""
        a, i = self.segment_start, self.index
        left = LOOKAHEAD
        b = self.route.waypoints[i].xy
        length = np.linalg.norm(b - a)
        along = np.clip(np.dot(xy - a, b - a) / max(length, 1e-6), 0.0, length)
        left += along
        while True:
            if left <= length or self.route.waypoints[i].stop or i + 1 == len(self.route.waypoints):
                return a + (b - a) * min(left / max(length, 1e-6), 1.0)
            left -= length
            a, i = b, i + 1
            b = self.route.waypoints[i].xy
            length = np.linalg.norm(b - a)

    def distance_to_next_stop(self, xy):
        """m along the route to the next stop."""
        distance, here = 0.0, xy
        for w in self.route.waypoints[self.index:]:
            distance += np.linalg.norm(w.xy - here)
            here = w.xy
            if w.stop:
                break
        return distance

    def align(self, t, xy, yaw):
        """Lines up on a stop: first steps to the spot without turning, then turns on the spot to the heading.

        One thing at a time: the policies handle a plain sidestep or a plain turn far better than a slow mix of
        all three. Each part has some hysteresis, so it doesn't flick on and off at the edge of the tolerance. If
        the robot makes no progress for STUCK_TIME (a command too gentle for the policy), it asks for more, and
        eases off again if it overshoots.
        """
        stop = self.target
        error = stop.xy - xy
        distance = float(np.linalg.norm(error))
        if distance > 2 * APPROACH:          # pushed away: walk back
            self.state = 'following'
            return np.zeros(3)
        yaw_error = 0.0 if stop.yaw is None else wrap(stop.yaw - yaw)
        if self.align_since is None:
            self.start_aligning(t)
        loose = 2.0 if t - self.align_since > ALIGN_TIMEOUT else 1.0
        tolerance, yaw_tolerance = loose * stop.tolerance, loose * stop.yaw_tolerance

        if distance > tolerance:
            self.positioning = True
        elif distance < 0.6 * tolerance:
            self.positioning = False
        if abs(yaw_error) > yaw_tolerance:
            self.turning = True
        elif abs(yaw_error) < 0.5 * yaw_tolerance:
            self.turning = False

        if not self.positioning and not self.turning:
            self.loose = loose > 1.0
            if self.settled_since is None:
                self.settled_since = t
            if t - self.settled_since >= SETTLE:
                self.state, self.wait_until, self.align_since = 'waiting', t + stop.wait, None
            return np.zeros(3)
        self.settled_since = None

        # Overshot (the error changed sign): ease off.
        if self.last_error is not None:
            overshot = (np.dot(error, self.last_error[0]) < 0 if self.positioning
                        else yaw_error * self.last_error[1] < 0)
            if overshot:
                self.boost = max(1.0, self.boost * 0.7)
        self.last_error = (error, yaw_error)

        # Progress is distance while stepping, and heading (at 0.3 m per rad) while turning.
        measure = distance if self.positioning else 0.3 * abs(yaw_error)
        if measure < self.best - PROGRESS:
            self.best, self.best_t = measure, t
        elif t - self.best_t > STUCK_TIME:
            self.boost = min(self.boost * 1.4, MAX_BOOST)
            self.best, self.best_t = measure, t

        if self.positioning:
            # The error in the robot's own frame: forward and to its left.
            c, s = np.cos(yaw), np.sin(yaw)
            body = np.array([c * error[0] + s * error[1], -s * error[0] + c * error[1]])
            speed = min(float(np.clip(ALIGN_GAIN * distance, ALIGN_MIN, ALIGN_SPEED)) * self.boost, MAX_ALIGN_SPEED)
            velocity = body / max(distance, 1e-6) * speed
            return np.array([velocity[0], velocity[1], 0.0])
        rate = min(float(np.clip(TURN_GAIN * abs(yaw_error), TURN_MIN, MAX_TURN)) * self.boost, MAX_TURN)
        return np.array([0.0, 0.0, float(np.sign(yaw_error)) * rate])

    def start_aligning(self, t):
        self.align_since, self.settled_since = t, None
        self.positioning = self.turning = False
        self.boost, self.best, self.best_t = 1.0, np.inf, t
        self.last_error = None

    def limit(self, t, command):
        """Keeps the command from jumping: the policy walks more steadily on smooth commands."""
        if self.last_t is not None:
            step = MAX_ACCEL * max(t - self.last_t, 0.0)
            command = np.clip(command, self.last_command - step, self.last_command + step)
        self.last_command, self.last_t = command, t
        return command
