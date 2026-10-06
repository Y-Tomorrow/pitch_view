from typing import List, Optional, Tuple

import cv2
import numpy as np


def _line_intersection(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> Optional[np.ndarray]:
    """
    Intersection point of two infinite lines, each defined by two points.

    Returns None when the lines are (nearly) parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return np.array([px, py])


def _segment_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """Angle of a segment in degrees, folded into [0, 180)."""
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0


def _order_corners(pts: List[np.ndarray]) -> np.ndarray:
    """Sort four corner points into [top-left, top-right, bottom-right, bottom-left]."""
    pts = sorted(pts, key=lambda p: p[1])  # by y
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


class CourtDetector:
    """
    Classical (non-learned) basketball court detector.

    Pipeline: Canny edge detection -> probabilistic Hough transform -> split the
    detected segments into near-horizontal and near-vertical groups -> pick the
    four extreme "boundary" lines (top / bottom / left / right) -> intersect them
    to recover the four court corners.

    The corners are ordered [top-left, top-right, bottom-right, bottom-left] and
    are ready to be passed to ``ViewTransformer`` for the pixel -> 2D court
    homography.

    Notes / limitations (first version):
      - Assumes a roughly straight-on broadcast view where the court boundary is
        the dominant quadrilateral in the frame.
      - The angle thresholds below are deliberately loose to tolerate perspective
        slant of the sidelines.
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 80,
        min_line_length: int = 100,
        max_line_gap: int = 30,
        horizontal_angle_threshold: float = 45.0,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.horizontal_angle_threshold = horizontal_angle_threshold

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect the four court corners in ``frame``.

        Args:
            frame: BGR image.

        Returns:
            Array of shape (4, 2) with corners ordered [tl, tr, br, bl].

        Raises:
            RuntimeError: If too few boundary lines are found to recover a
                quadrilateral.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        segments = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        if segments is None:
            raise RuntimeError("No lines detected in the frame.")

        horizontal: List[Tuple[float, float, float, float]] = []
        vertical: List[Tuple[float, float, float, float]] = []
        t = self.horizontal_angle_threshold
        for segment in segments:
            x1, y1, x2, y2 = segment[0]
            angle = _segment_angle(x1, y1, x2, y2)
            if angle < t or angle > 180.0 - t:
                horizontal.append((x1, y1, x2, y2))
            else:
                vertical.append((x1, y1, x2, y2))

        if len(horizontal) < 2 or len(vertical) < 2:
            raise RuntimeError(
                "Could not find two horizontal and two vertical boundary lines "
                f"(found {len(horizontal)} horizontal, {len(vertical)} vertical).")

        def avg(a: float, b: float) -> float:
            return (a + b) / 2

        top = min(horizontal, key=lambda s: avg(s[1], s[3]))
        bottom = max(horizontal, key=lambda s: avg(s[1], s[3]))
        left = min(vertical, key=lambda s: avg(s[0], s[2]))
        right = max(vertical, key=lambda s: avg(s[0], s[2]))

        corners = [
            _line_intersection(np.array(left[:2]), np.array(left[2:]),
                               np.array(top[:2]), np.array(top[2:])),
            _line_intersection(np.array(right[:2]), np.array(right[2:]),
                               np.array(top[:2]), np.array(top[2:])),
            _line_intersection(np.array(right[:2]), np.array(right[2:]),
                               np.array(bottom[:2]), np.array(bottom[2:])),
            _line_intersection(np.array(left[:2]), np.array(left[2:]),
                               np.array(bottom[:2]), np.array(bottom[2:])),
        ]
        if any(c is None for c in corners):
            raise RuntimeError("Boundary lines are (nearly) parallel; "
                               "could not recover court corners.")

        return _order_corners(corners)
