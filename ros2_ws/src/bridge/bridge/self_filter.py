"""Removes the robot's own body from FAST-LIO's scan before elevation mapping.

The Mid-360 on the G1's head sees the robot's legs and arms. Elevation mapping has no self-filter, so it adds them to
the terrain, and while walking they leave a trail of "obstacles" at about pelvis height where the robot has been.

This drops every point within `radius` of the vertical line through the Mid-360's IMU: the z axis of FAST-LIO's
body frame, which on the G1 runs down through the torso and pelvis. It needs no TF or joint states, so it works on
the robot as it is now. The ground right under the robot is dropped too, but that was mapped on the way there.
radius 0 passes everything through (for upright-LiDAR bags, which have no robot in them).
"""
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


class SelfFilter(Node):

    def __init__(self):
        super().__init__('self_filter')
        # m. The upside-down Mid-360 can't see ground within about 0.94 m anyway (its view stops 52 deg below
        # horizontal), and at 0.5 m a swinging foot or hand still got through.
        self.radius = self.declare_parameter('radius', 0.6).value
        self.pub = self.create_publisher(PointCloud2, '/cloud_self_filtered', 10)
        self.create_subscription(PointCloud2, '/cloud_registered_body', self.on_cloud, 10)

    def on_cloud(self, msg):
        if self.radius > 0 and msg.width * msg.height > 0:
            offset = {f.name: f.offset for f in msg.fields}
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
            x = raw[:, offset['x']:offset['x'] + 4].copy().view(np.float32)[:, 0]
            y = raw[:, offset['y']:offset['y'] + 4].copy().view(np.float32)[:, 0]
            keep = x * x + y * y >= self.radius ** 2 # only keep points outside this radius: x^2 + y^2 > radius
            msg.data = raw[keep].tobytes()
            msg.height, msg.width = 1, int(keep.sum())
            msg.row_step = msg.width * msg.point_step
        self.pub.publish(msg)


def main():
    rclpy.init()
    try:
        rclpy.spin(SelfFilter())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # ros2 launch can send a second SIGINT, so don't start any cleanup that it could interrupt
