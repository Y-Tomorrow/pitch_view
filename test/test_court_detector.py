import numpy as np
import cv2

from pitch_view.common import CourtDetector


def _synthetic_court():
    """Build a synthetic perspective court and return (image, ground-truth corners)."""
    h, w = 480, 720
    img = np.zeros((h, w, 3), np.uint8)
    dst = np.float32([[180, 80], [600, 120], [620, 420], [140, 360]])
    src = np.float32([[0, 0], [2800, 0], [2800, 1500], [0, 1500]])
    m = cv2.getPerspectiveTransform(src, dst)

    def line(a, b):
        p = np.float32([a, b]).reshape(-1, 1, 2)
        q = cv2.perspectiveTransform(p, m).reshape(-1, 2).astype(int)
        cv2.line(img, tuple(q[0]), tuple(q[1]), (255, 255, 255), 4)

    for a, b in [
        ([0, 0], [2800, 0]), ([2800, 0], [2800, 1500]),
        ([2800, 1500], [0, 1500]), ([0, 1500], [0, 0]),
        ([1400, 0], [1400, 1500]),
    ]:
        line(a, b)
    return img, dst


def test_detector_recovers_corners():
    img, gt = _synthetic_court()
    corners = CourtDetector().detect(img)
    assert corners.shape == (4, 2)
    # Each detected corner should be within a few pixels of ground truth.
    assert np.all(np.abs(corners - gt) < 8)
