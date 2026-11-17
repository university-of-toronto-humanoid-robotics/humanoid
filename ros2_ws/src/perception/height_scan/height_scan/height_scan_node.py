"""ROS node: elevation map in, the height-map policy's terrain scan out (see scan.py for the format and the rules
for ground the LiDAR hasn't seen).

The map updates a few times a second, but the robot moves between updates, so the scan is recomputed from the
latest map and the latest pelvis and foot poses (TF) at the policy's rate. The same node runs on the robot and in
bridge's walk sim, which reads /height_scan like the robot's controller will.

Subscribes:  /elevation_mapping_node/elevation_map_filter  grid_map_msgs/GridMap
             TF: <map frame> -> pelvis, and -> each foot frame (robot_state_publisher, from /joint_states)
Publishes:   /height_scan                 Float32MultiArray, the 693 values for the policy
             /height_scan/observed_share  Float32, 0-1: share of the points the map has really observed
             /height_scan/feet_share      Float32, 0-1: share taken from the ground under the feet
             /height_scan/points          PointCloud2, the scan as points, for RViz (only if someone listens)

Nothing is published on /height_scan until there is a map and a pelvis pose: the controller must not walk on a
scan that doesn't exist yet.
"""
from grid_map_msgs.msg import GridMap
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32, Float32MultiArray, Header, MultiArrayDimension
from tf2_ros import Buffer, TransformException, TransformListener

from height_scan.scan import N_ACROSS, N_ALONG, STANDING_HEIGHT, compute_scan, scan_points, unpack_layer


def yaw_of(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def grid_layer(msg, name):
    """A GridMap layer as a (rows, cols) array (see unpack_layer)."""
    d = msg.data[list(msg.layers).index(name)]
    return unpack_layer(d.data, d.layout.dim[0].size, d.layout.dim[1].size,
                        msg.outer_start_index, msg.inner_start_index)


class HeightScanNode(Node):

    def __init__(self):
        super().__init__('height_scan_node')
        p = lambda name, default: self.declare_parameter(name, default).value
        map_topic = p('map_topic', '/elevation_mapping_node/elevation_map_filter')
        self.observed_layer = p('observed_layer', 'elevation')
        self.filled_layer = p('filled_layer', 'inpaint')
        self.pelvis_frame = p('pelvis_frame', 'pelvis')
        self.foot_frames = p('foot_frames', ['left_ankle_roll_link', 'right_ankle_roll_link'])
        self.sole_depth = p('sole_depth', 0.035)     # m, sole below the ankle_roll_link origin (G1 geometry)
        self.max_map_age = Duration(seconds=p('max_map_age_s', 1.0))
        rate = p('rate_hz', 50.0)                    # the policy's rate

        self.map = None
        self.map_stamp = None
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.create_subscription(GridMap, map_topic, self.on_map, 1)
        self.scan_pub = self.create_publisher(Float32MultiArray, '/height_scan', qos_profile_sensor_data)
        self.observed_pub = self.create_publisher(Float32, '/height_scan/observed_share', 10)
        self.feet_pub = self.create_publisher(Float32, '/height_scan/feet_share', 10)
        self.points_pub = self.create_publisher(PointCloud2, '/height_scan/points', 1)
        self.create_timer(1.0 / rate, self.tick)

    def on_map(self, msg):
        if self.observed_layer not in msg.layers:
            self.get_logger().error(f'The map has no "{self.observed_layer}" layer', throttle_duration_sec=5.0)
            return
        info = msg.info
        self.map = dict(
            frame=msg.header.frame_id,
            observed=grid_layer(msg, self.observed_layer),
            filled=grid_layer(msg, self.filled_layer) if self.filled_layer in msg.layers else None,
            top=(info.pose.position.x + info.length_x / 2, info.pose.position.y + info.length_y / 2),
            resolution=info.resolution,
        )
        self.map_stamp = self.get_clock().now()

    def lookup(self, frame):
        try:
            return self.tf.lookup_transform(self.map['frame'], frame, Time())
        except TransformException:
            return None

    def tick(self):
        if self.map is None:
            self.get_logger().info('Waiting for the elevation map', throttle_duration_sec=5.0)
            return
        pelvis = self.lookup(self.pelvis_frame)
        if pelvis is None:
            self.get_logger().warn(f'No {self.map["frame"]} -> {self.pelvis_frame} transform yet',
                                   throttle_duration_sec=5.0)
            return
        if self.get_clock().now() - self.map_stamp > self.max_map_age:
            self.get_logger().warn('The elevation map has stopped updating; scanning the last one',
                                   throttle_duration_sec=2.0)

        t = pelvis.transform.translation
        pos = np.array([t.x, t.y, t.z])
        yaw = yaw_of(pelvis.transform.rotation)
        feet = [f.transform.translation.z - self.sole_depth for f in map(self.lookup, self.foot_frames) if f]
        if not feet:
            self.get_logger().warn('No foot transforms; taking the ground under the robot as standing height',
                                   throttle_duration_sec=5.0)
        feet_ground = min(feet) if feet else pos[2] - STANDING_HEIGHT

        scan, observed, from_feet = compute_scan(self.map['observed'], self.map['filled'], self.map['top'],
                                                 self.map['resolution'], pos, yaw, feet_ground)
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='across', size=N_ACROSS, stride=N_ACROSS * N_ALONG),
            MultiArrayDimension(label='along', size=N_ALONG, stride=N_ALONG),
        ]
        msg.data = scan.astype(np.float32).tolist()
        self.scan_pub.publish(msg)
        self.observed_pub.publish(Float32(data=observed))
        self.feet_pub.publish(Float32(data=from_feet))

        if self.points_pub.get_subscription_count():
            points = np.column_stack([scan_points(pos[:2], yaw), pos[2] - scan])
            header = Header(stamp=pelvis.header.stamp, frame_id=self.map['frame'])
            self.points_pub.publish(point_cloud2.create_cloud_xyz32(header, points.tolist()))


def main():
    rclpy.init()
    node = HeightScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
