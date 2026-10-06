import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32, Polygon
from rclpy.node import Node
from sensor_msgs.msg import Image

from pitch_view.common import ViewTransformer, create_detector
from pitch_view.configs import (
    BasketballCourtConfiguration,
    League,
    SoccerPitchConfiguration,
)

CORNER_COLOR = (0, 0, 255)  # red (BGR)
BOUNDARY_COLOR = (0, 255, 0)  # green (BGR)


class CourtDetectorNode(Node):
    """
    Subscribes to a camera image, detects the pitch/court keypoints (soccer by
    default via the `football-field-detection-f07vi` model), and publishes:

      * ``~/annotated``    (sensor_msgs/Image) frame with keypoints drawn
      * ``~/pitch_map``    (sensor_msgs/Image) pitch warped to a top-down 2D map
      * ``~/keypoints``    (geometry_msgs/Polygon) the detected keypoints
    """

    def __init__(self):
        super().__init__('court_detector')

        self.bridge = CvBridge()

        # Tunable parameters (overridable via launch / YAML).
        # 中文说明见 config/court_detector.yaml。
        self.declare_parameter('sport', 'soccer')
        self.declare_parameter('scale', 0.05)
        self.declare_parameter('detector_type', 'inference')
        self.declare_parameter('model_path', 'models/pitch_keypoints.pt')
        self.declare_parameter('homography_file', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('api_key', '')
        self.declare_parameter('model_id', 'football-field-detection-f07vi/14')
        self.declare_parameter('api_url', 'https://serverless.roboflow.com')
        self.declare_parameter('min_confidence', 0.0)
        self.declare_parameter('canny_low', 50)
        self.declare_parameter('canny_high', 150)
        self.declare_parameter('hough_threshold', 80)
        self.declare_parameter('min_line_length', 100)
        self.declare_parameter('max_line_gap', 30)
        # 球员 / OCR / 特写 (供后续节点或同 YAML 的 GUI 使用)
        self.declare_parameter('player_model_path', 'models/players.pt')
        self.declare_parameter('player_conf', 0.25)
        self.declare_parameter('player_iou', 0.45)
        self.declare_parameter('player_agnostic_nms', True)
        self.declare_parameter('player_nms_iou', 0.45)
        self.declare_parameter('player_nms_contain', 0.7)
        self.declare_parameter('jersey_ocr_min_conf', 0.35)
        self.declare_parameter('jersey_torso_x_margin', 0.18)
        self.declare_parameter('jersey_torso_y0', 0.10)
        self.declare_parameter('jersey_torso_y1', 0.52)
        self.declare_parameter('closeup_pad_ratio', 0.18)
        self.declare_parameter('closeup_max_scale', 5.0)
        self.declare_parameter('closeup_track_iou', 0.3)

        sport = self.get_parameter('sport').value
        if sport == 'soccer':
            self.config = SoccerPitchConfiguration()
        elif sport == 'basketball':
            self.config = BasketballCourtConfiguration.from_league(League.FIBA)
        else:
            raise ValueError(f"Unknown sport '{sport}' (expected 'soccer' or 'basketball').")

        # Build the detector via the factory — the single switch point between
        # classical CV, a YOLO keypoint model, and a Roboflow model. ``detect``
        # returns an aligned (keypoints, target) pair.
        self.detector = create_detector(
            self.get_parameter('detector_type').value,
            model_path=self.get_parameter('model_path').value,
            homography_file=self.get_parameter('homography_file').value,
            api_key=self.get_parameter('api_key').value,
            model_id=self.get_parameter('model_id').value,
            api_url=self.get_parameter('api_url').value,
            config=self.config,
            device=self.get_parameter('device').value,
            min_confidence=self.get_parameter('min_confidence').value,
        )

        self._sub_image = self.create_subscription(
            Image, 'image', self._image_callback, 10)
        self._pub_annotated = self.create_publisher(Image, '~/annotated', 10)
        self._pub_map = self.create_publisher(Image, '~/pitch_map', 10)
        self._pub_keypoints = self.create_publisher(Polygon, '~/keypoints', 10)

    def _apply_params(self) -> None:
        # Classical-CV knobs only make sense for the classical detector.
        if not hasattr(self.detector, 'canny_low'):
            return
        self.detector.canny_low = self.get_parameter('canny_low').value
        self.detector.canny_high = self.get_parameter('canny_high').value
        self.detector.hough_threshold = self.get_parameter('hough_threshold').value
        self.detector.min_line_length = self.get_parameter('min_line_length').value
        self.detector.max_line_gap = self.get_parameter('max_line_gap').value

    def _image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self._apply_params()

        try:
            keypoints, target = self.detector.detect(frame)
        except RuntimeError as exc:
            self.get_logger().warn(
                f'pitch detection failed: {exc}', throttle_duration_sec=1.0)
            self._pub_annotated.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))
            return

        if len(keypoints) < 4:
            self.get_logger().warn(
                f'too few keypoints ({len(keypoints)}); skipping', throttle_duration_sec=1.0)
            self._pub_annotated.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))
            return

        scale = self.get_parameter('scale').value
        transformer = ViewTransformer(
            source=keypoints, target=target * scale, image_shape=frame.shape)

        # Annotated frame.
        annotated = frame.copy()
        pts = keypoints.astype(int)
        for point in pts:
            cv2.circle(annotated, tuple(point), 6, CORNER_COLOR, -1)

        # Top-down 2D map: warp the camera image onto the pitch plane.
        length_px = int(self.config.length * scale)
        width_px = int(self.config.width * scale)
        warped = transformer.transform_image(frame, (length_px, width_px))

        self._pub_annotated.publish(self.bridge.cv2_to_imgmsg(annotated, 'bgr8'))
        self._pub_map.publish(self.bridge.cv2_to_imgmsg(warped, 'bgr8'))
        self._pub_keypoints.publish(Polygon(points=[
            Point32(x=float(p[0]), y=float(p[1]), z=0.0) for p in keypoints
        ]))


def main(args=None):
    rclpy.init(args=args)
    node = CourtDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
