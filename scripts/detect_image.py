#!/usr/bin/env python3
"""
Quick non-ROS check of the court-detection pipeline on a single image.

Usage:
    python scripts/detect_image.py data/test_court.jpg data/out.jpg

This runs the same detection + homography logic as the ROS2 node, but reads a
file directly instead of subscribing to an `image` topic — handy for fast local
iteration before wiring up a camera / bag.
"""
import argparse
import os
import sys

# Make the package root importable regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from pitch_view.common import ViewTransformer, create_detector
from pitch_view.configs import SoccerPitchConfiguration

SCALE = 0.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Input image path.")
    parser.add_argument("target", help="Output image path.")
    parser.add_argument(
        "--detector", choices=["classical", "yolo", "inference", "roboflow"],
        default="classical",
        help="Detector implementation (default: classical).")
    parser.add_argument(
        "--model", default="",
        help="Model path when --detector yolo (default: weights/pitch_keypoints.pt).")
    args = parser.parse_args()

    config = SoccerPitchConfiguration()
    frame = cv2.imread(args.source)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.source}")

    model_path = args.model
    if args.detector == "yolo" and not model_path:
        from pitch_view.common.weights import pitch_keypoints_path
        model_path = pitch_keypoints_path()

    detector = create_detector(
        args.detector, model_path=model_path, config=config)

    try:
        keypoints, target = detector.detect(frame)
    except RuntimeError as exc:
        print(f"[warn] pitch detection failed: {exc}")
        return

    transformer = ViewTransformer(
        source=keypoints, target=target * SCALE, image_shape=frame.shape)

    annotated = frame.copy()
    for point in keypoints.astype(int):
        cv2.circle(annotated, tuple(point), 8, (0, 0, 255), -1)

    length_px = int(config.length * SCALE)
    width_px = int(config.width * SCALE)
    warped = transformer.transform_image(frame, (length_px, width_px))
    warped = cv2.resize(warped, (annotated.shape[1], annotated.shape[0]))

    cv2.imwrite(args.target, np.hstack([annotated, warped]))
    print(f"Wrote {args.target}")


if __name__ == "__main__":
    main()
