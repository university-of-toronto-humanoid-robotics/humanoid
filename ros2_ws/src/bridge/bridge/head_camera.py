"""The G1's head camera (the D435i's color camera) for mujoco_g1_walk_sim: the operator's first-person view.

Out: /g1/head_camera/color/image_raw    sensor_msgs/Image, rgb8
     /g1/head_camera/color/camera_info  sensor_msgs/CameraInfo, a pinhole model of the same image
     TF d435_link -> head_camera_link -> head_camera_color_optical_frame (static); the image's frame is the
        optical one: z forward, x right, y down

Topics and frames are the ones Intel's driver publishes on the robot, run as
  ros2 launch realsense2_camera rs_launch.py camera_namespace:=g1 camera_name:=head_camera
so the operator side is the same in sim and on the robot. There, the driver publishes head_camera_link ->
head_camera_color_optical_frame itself, and only d435_link -> head_camera_link has to be added. The camera sits
where the URDF's d435_link is (description/g1/g1_29dof_rev_1_0.urdf: d435_joint), so anything placed with TF (the
map, a path) lines up with the image. bringup's g1.launch.py adds the compressed stream that the driver would
(image_raw/compressed), for an operator station on Wi-Fi. Rendering uses MUJOCO_GL's backend: egl in the Linux
container (the GPU, no window), glfw elsewhere. If it can't start (no display, no GPU, CI), the sim runs on
without a camera.
"""
import math

import mujoco
import numpy as np

NAME = 'head_camera'
POS = (0.0576235, 0.01753, 0.42987)   # in torso_link, from the URDF's d435_joint
PITCH = 0.8307767239493009            # rad below level, from the same joint
FOVY = 42.0                           # degrees: the D435i color camera is 69 x 42
WIDTH, HEIGHT = 640, 360
LINK = 'head_camera_link'
FRAME = 'head_camera_color_optical_frame'
TOPIC = '/g1/head_camera/color'


def add_to_spec(spec):
    """Add the camera to torso_link. MuJoCo cameras look along their -z, with +x right and +y up."""
    right, up = (0.0, -1.0, 0.0), (math.sin(PITCH), 0.0, math.cos(PITCH))
    back = (-math.cos(PITCH), 0.0, math.sin(PITCH))
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.array([right, up, back]).T.flatten())
    spec.body('torso_link').add_camera(name=NAME, pos=POS, quat=quat, fovy=FOVY)


def intrinsics(width=WIDTH, height=HEIGHT, fovy=FOVY):
    """(fx, fy, cx, cy) of the rendered image: square pixels, centred."""
    f = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
    return f, f, width / 2.0, height / 2.0


class HeadCamera:
    """Renders the head camera and publishes it.

    Call start() and render() from the physics thread, and start() before the MuJoCo viewer opens: an EGL
    context can't be made once the viewer's GLFW one exists.
    """

    def __init__(self, node, model):
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import StaticTransformBroadcaster
        from geometry_msgs.msg import TransformStamped

        self.node, self.model = node, model
        self.Image = Image
        self.renderer = None
        self.image_pub = node.create_publisher(Image, f'{TOPIC}/image_raw', 2)
        self.info_pub = node.create_publisher(CameraInfo, f'{TOPIC}/camera_info', 2)

        fx, fy, cx, cy = intrinsics()
        self.info = CameraInfo(width=WIDTH, height=HEIGHT, distortion_model='plumb_bob')
        self.info.header.frame_id = FRAME
        self.info.d = [0.0] * 5
        self.info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        # The camera's frames, as the driver names them. head_camera_link is the URDF's d435_link (x forward);
        # the optical frame is z forward, x right, y down.
        mount = TransformStamped()
        mount.header.frame_id, mount.child_frame_id = 'd435_link', LINK
        mount.transform.rotation.w = 1.0
        optical = TransformStamped()
        optical.header.frame_id, optical.child_frame_id = LINK, FRAME
        optical.transform.rotation.x, optical.transform.rotation.y = -0.5, 0.5
        optical.transform.rotation.z, optical.transform.rotation.w = -0.5, 0.5
        self.static_tf = StaticTransformBroadcaster(node)
        self.static_tf.sendTransform([mount, optical])

    def start(self):
        try:
            self.renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH)
        except Exception as error:   # no display or GPU: carry on without the camera
            self.node.get_logger().warn(f'No head camera: rendering failed to start ({error}).')

    def render(self, data, stamp):
        if self.renderer is None:
            return
        self.renderer.update_scene(data, camera=NAME)
        pixels = self.renderer.render()
        msg = self.Image(height=HEIGHT, width=WIDTH, encoding='rgb8', is_bigendian=0, step=WIDTH * 3)
        msg.header.stamp, msg.header.frame_id = stamp, FRAME
        msg.data = pixels.tobytes()
        self.info.header.stamp = stamp
        self.image_pub.publish(msg)
        self.info_pub.publish(self.info)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
