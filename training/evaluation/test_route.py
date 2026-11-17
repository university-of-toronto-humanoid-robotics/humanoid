"""Checks for the route geometry and navigation state machine."""

import unittest

import numpy as np

from evaluation.route import CourseMap, RouteFollower, build_paths, load_route


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


if __name__ == "__main__":
  unittest.main()
