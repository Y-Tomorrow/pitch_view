"""Detect the Gazebo block players by shirt colour.

The broadcast YOLO weights barely fire on these figures. Red and blue kits
are solid and saturated, so a colour mask finds them directly.
"""

from __future__ import annotations

import cv2
import numpy as np

# Same class index as weights/players.pt: 2 = player.
_PLAYER_CLS = 2


def _mask_boxes(mask: np.ndarray) -> list:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh < 10 or bw < 3 or bh < bw * 0.8 or cv2.contourArea(contour) < 30:
            continue
        top = max(0, y - int(0.22 * bh))
        boxes.append((x, top, x + bw, min(height - 1, y + bh)))
    boxes.sort(key=lambda item: item[0])
    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        prev = merged[-1]
        prev_cx = (prev[0] + prev[2]) / 2.0
        box_cx = (box[0] + box[2]) / 2.0
        gap = max(12.0, 0.6 * ((prev[2] - prev[0]) + (box[2] - box[0])))
        separated = box[1] > prev[3] or prev[1] > box[3] + 8
        if abs(prev_cx - box_cx) < gap and not separated:
            merged[-1] = (
                min(prev[0], box[0]),
                min(prev[1], box[1]),
                max(prev[2], box[2]),
                max(prev[3], box[3]),
            )
        else:
            merged.append(box)
    return merged


def detect_sim_players(frame: np.ndarray):
    """Return ``(class, score, xyxy)`` boxes for red and blue sim players."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    red = cv2.bitwise_or(red, cv2.inRange(hsv, (168, 80, 60), (180, 255, 255)))
    blue = cv2.inRange(hsv, (95, 80, 60), (135, 255, 255))
    boxes = []
    for mask in (red, blue):
        for x1, y1, x2, y2 in _mask_boxes(mask):
            boxes.append((
                _PLAYER_CLS,
                0.95,
                np.array([x1, y1, x2, y2], dtype=np.float32),
            ))
    return boxes
