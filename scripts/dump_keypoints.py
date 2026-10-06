#!/usr/bin/env python3
"""
End-to-end check of the pitch detection + homography pipeline on a single image.

Usage (local Roboflow model, weights cached on disk — recommended):
    export ROBOFLOW_API_KEY=...
    pip install inference
    python scripts/dump_keypoints.py data/1.jpg --detector inference

Prints the number of detected keypoints and their pixel coordinates, then saves
two images next to the input: ``*_keypoints.jpg`` (annotated) and
``*_warped.jpg`` (the frame warped onto the top-down pitch plane).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from pitch_view.common import ViewTransformer, create_detector
from pitch_view.configs import BasketballCourtConfiguration, SoccerPitchConfiguration

DEFAULT_MODEL = "football-field-detection-f07vi/14"


def _config(sport):
    if sport == "soccer":
        return SoccerPitchConfiguration()
    if sport == "basketball":
        return BasketballCourtConfiguration()
    raise ValueError(f"Unknown sport: {sport}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Input image path.")
    parser.add_argument("--detector", choices=["inference", "roboflow"],
                        default="inference")
    parser.add_argument("--sport", choices=["soccer", "basketball"], default="soccer")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Drop keypoints below this confidence (0 = keep all).")
    parser.add_argument(
        "--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "Set ROBOFLOW_API_KEY or pass --api-key (register free at roboflow.com).")

    config = _config(args.sport)
    frame = cv2.imread(args.source)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.source}")

    detector = create_detector(
        args.detector, api_key=args.api_key, model_id=args.model, config=config,
        min_confidence=args.min_confidence)
    keypoints, target = detector.detect(frame)

    print(f"detector: {args.detector}  sport: {args.sport}  model: {args.model}")
    print(f"num keypoints: {len(keypoints)}")
    for i, (x, y) in enumerate(keypoints):
        print(f"  {i:2d}  ({x:7.1f}, {y:7.1f})")

    annotated = frame.copy()
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(annotated, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(annotated, str(i), (int(x) + 6, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    base = os.path.splitext(args.source)[0]

    if len(keypoints) >= 4:
        transformer = ViewTransformer(
            source=keypoints, target=target * args.scale, image_shape=frame.shape)
        length_px = int(config.length * args.scale)
        width_px = int(config.width * args.scale)
        warped = transformer.transform_image(frame, (length_px, width_px))
        cv2.imwrite(base + "_warped.jpg", warped)
        print(f"warped pitch: {base}_warped.jpg ({length_px}x{width_px})")
    else:
        print("too few keypoints to estimate a homography")

    cv2.imwrite(base + "_keypoints.jpg", annotated)
    print(f"annotated image: {base}_keypoints.jpg")


if __name__ == "__main__":
    main()
