"""Checks for the route geometry and navigation state machine."""

import unittest

import numpy as np

from evaluation.route import CourseMap, RouteFollower, build_paths, load_route, start_before_waypoint


class RouteTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.course = CourseMap.load()
    cls.route = load_route()
    cls.outbound, cls.returning, cls.smoothed = build_paths(cls.route, cls.course)

  def test_smoothing_preserves_targets_and_clearance(self) -> None:
    self.assertEqual(self.smoothed, ["P6", "P7"])
    self.assertGreaterEqual(float(self.course.clearance_at(self.outbound.xy).min()),
                            self.route["wall_clearance_m"])
    np.testing.assert_allclose(self.returning.xy, self.outbound.xy[::-1])
    for p in self.route["waypoints"]:
      if p["kind"] in {"goal", "start", "end"}:
        self.assertLess(np.linalg.norm(self.outbound.point_at(self.outbound.goal_s[p["id"]])
                                       - [p["x"], p["y"]]), 1e-6)

  def test_goal_order_and_return(self) -> None:
    follower = RouteFollower(self.route, self.outbound, self.returning)
    self.assertEqual(follower.update(np.array([11.0, 4.21]), np.pi).next_goal, "A1")
    self.assertEqual(follower.update(np.array([6.45, 3.45]), np.pi).next_goal, "A1")
    outbound = [p for p in self.route["waypoints"] if p["kind"] in {"goal", "end"}]
    for p in outbound:
      follower.update(np.array([p["x"], p["y"]]), 0.0)
    self.assertEqual(follower.phase, "return")
    returning = [p for p in reversed(self.route["waypoints"]) if p["kind"] in {"goal", "start"}]
    for p in returning:
      follower.update(np.array([p["x"], p["y"]]), 0.0)
    self.assertEqual(follower.phase, "complete")

  def test_turn_in_place_then_drive(self) -> None:
    follower = RouteFollower(self.route, self.outbound, self.returning)
    start = np.array([11.0, 4.21])
    turning = follower.update(start, np.pi)
    self.assertEqual((turning.vx, turning.vy), (0.0, 0.0))
    self.assertLessEqual(abs(turning.yaw_rate), 1.0)
    aligned = follower.update(start, 2.0)
    self.assertGreater(aligned.vx, 0.0)

  def test_speed_feedback_raises_command_when_stalled(self) -> None:
    follower = RouteFollower(self.route, self.outbound, self.returning)
    start = np.array([11.0, 4.21])
    turning = follower.update(start, np.pi, measured_vx=0.0, dt=0.1)
    self.assertEqual(turning.vx, 0.0)
    follower.update(start, 2.0, measured_vx=0.0, dt=0.1)
    for _ in range(10):
      command = follower.update(start, 2.0, measured_vx=0.0, dt=0.1)
    self.assertGreater(command.vx, follower.target_vx)
    self.assertLessEqual(command.vx, 0.8)

  def test_speed_feedback_reduces_command_when_faster_than_target(self) -> None:
    follower = RouteFollower(self.route, self.outbound, self.returning)
    start = np.array([11.0, 4.21])
    follower.update(start, 2.0, measured_vx=1.0, dt=0.1)
    for _ in range(10):
      command = follower.update(start, 2.0, measured_vx=1.0, dt=0.1)
    self.assertLess(command.vx, follower.target_vx)
    self.assertGreaterEqual(command.vx, 0.0)

  def test_start_before_stairs_skips_earlier_goals(self) -> None:
    start = start_before_waypoint(self.route, self.course, self.outbound, "P8")
    target = next(p for p in self.route["waypoints"] if p["id"] == "P8")
    target_index = np.argmin(np.linalg.norm(
      self.outbound.xy - [target["x"], target["y"]], axis=1))
    self.assertLess(start.progress_s, self.outbound.s[target_index])
    follower = RouteFollower(self.route, self.outbound, self.returning)
    follower.reset(start.progress_s)
    self.assertEqual(follower.update(start.xy, start.yaw).next_goal, "D1")
    self.assertGreater(follower.progress_s, self.outbound.goal_s["C2"])


if __name__ == "__main__":
  unittest.main()
