from typing import List, Optional, Tuple

import cv2
import numpy as np

from pitch_view.configs.basketball import BasketballCourtConfiguration

# BGR colour tuples.
COURT_BACKGROUND = (98, 160, 206)  # light tan / hardwood
LINE_WHITE = (255, 255, 255)
RED = (0, 0, 255)
BLACK = (0, 0, 0)


def _sx(x: float, scale: float, padding: int) -> int:
    return int(x * scale) + padding


def _sy(y: float, scale: float, padding: int) -> int:
    return int(y * scale) + padding


def draw_court(
    config: BasketballCourtConfiguration,
    background_color: Tuple[int, int, int] = COURT_BACKGROUND,
    line_color: Tuple[int, int, int] = LINE_WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    scale: float = 0.2,
) -> np.ndarray:
    """
    Draw a top-down 2D basketball court image.

    Args:
        config: Court dimensions (FIBA/NBA).
        background_color: Court surface colour (BGR).
        line_color: Court line colour (BGR).
        padding: Padding around the court in pixels.
        line_thickness: Line thickness in pixels.
        scale: Pixels per centimetre.

    Returns:
        BGR image of the court.
    """
    W = int(config.width * scale)
    L = int(config.length * scale)
    cw = config.width / 2  # centreline y
    cl = config.length / 2  # centreline x

    img = np.full((W + 2 * padding, L + 2 * padding, 3),
                  background_color, dtype=np.uint8)

    def pt(x: float, y: float) -> tuple:
        return (_sx(x, scale, padding), _sy(y, scale, padding))

    # Outer boundary.
    cv2.rectangle(img, pt(0, 0), pt(config.length, config.width),
                  line_color, line_thickness)

    # Centre line + centre circle.
    cv2.line(img, pt(cl, 0), pt(cl, config.width), line_color, line_thickness)
    cv2.circle(img, pt(cl, cw), int(config.centre_circle_radius * scale),
               line_color, line_thickness)

    for side in (0, config.length):  # left / right halves
        # Left half: features extend toward the centre (+x). Right half: toward -x.
        sign = 1 if side == 0 else -1

        # Lane (paint).
        lane_top = cw - config.lane_width / 2
        lane_bottom = cw + config.lane_width / 2
        ft_x = side + sign * config.free_throw_distance
        cv2.rectangle(
            img,
            pt(min(side, ft_x), lane_top),
            pt(max(side, ft_x), lane_bottom),
            line_color, line_thickness,
        )

        # Free-throw circle.
        cv2.circle(img, pt(ft_x, cw), int(config.free_throw_circle_radius * scale),
                   line_color, line_thickness)

        # Basket hoop (small filled circle).
        hoop_x = side + sign * config.basket_distance
        cv2.circle(img, pt(hoop_x, cw), max(2, int(4 * scale)),
                   line_color, -1)

        # Three-point line: straight segment along the sideline + arc.
        straight_offset = config.three_point_corner_distance
        corner_top = cw + straight_offset
        corner_bottom = cw - straight_offset
        # Where the straight segment meets the arc.
        r = config.three_point_distance
        dx = int(np.sqrt(max(0.0, r * r - straight_offset * straight_offset)))
        elbow_x = side + sign * dx

        cv2.line(img, pt(side, corner_top), pt(elbow_x, corner_top),
                 line_color, line_thickness)
        cv2.line(img, pt(side, corner_bottom), pt(elbow_x, corner_bottom),
                 line_color, line_thickness)

        # Arc of the three-point line (centred on the hoop). Left half faces the
        # centre (+x) -> angles -90..90; right half faces the centre (-x) -> 90..270.
        hoop = pt(hoop_x, cw)
        rr = int(r * scale)
        start_angle, end_angle = (-90, 90) if sign == 1 else (90, 270)
        cv2.ellipse(img, hoop, (rr, rr), 0, start_angle, end_angle,
                    line_color, line_thickness)

    return img


def draw_points_on_court(
    config: BasketballCourtConfiguration,
    xy: np.ndarray,
    face_color: Tuple[int, int, int] = RED,
    edge_color: Tuple[int, int, int] = BLACK,
    radius: int = 8,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.2,
    court: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw players (or arbitrary points) on a 2D court image.

    Args:
        config: Court dimensions.
        xy: Array of (x, y) court-coordinate points (in cm).
        face_color: Point fill colour (BGR).
        edge_color: Point border colour (BGR).
        radius: Point radius in pixels.
        thickness: Point border thickness in pixels.
        padding: Padding around the court in pixels.
        scale: Pixels per centimetre.
        court: Existing court image to draw on; a new one is created if None.

    Returns:
        Court image with points drawn.
    """
    if court is None:
        court = draw_court(config=config, padding=padding, scale=scale)

    for point in xy:
        centre = (_sx(point[0], scale, padding), _sy(point[1], scale, padding))
        cv2.circle(court, centre, radius, face_color, -1)
        cv2.circle(court, centre, radius, edge_color, thickness)

    return court


def draw_paths_on_court(
    config: BasketballCourtConfiguration,
    paths: List[np.ndarray],
    color: Tuple[int, int, int] = LINE_WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.2,
    court: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw movement trails on a 2D court image.

    Args:
        config: Court dimensions.
        paths: List of paths, each an array of (x, y) court-coordinate points.
        color: Trail colour (BGR).
        thickness: Trail thickness in pixels.
        padding: Padding around the court in pixels.
        scale: Pixels per centimetre.
        court: Existing court image to draw on; a new one is created if None.

    Returns:
        Court image with trails drawn.
    """
    if court is None:
        court = draw_court(config=config, padding=padding, scale=scale)

    for path in paths:
        pts = [
            (_sx(p[0], scale, padding), _sy(p[1], scale, padding))
            for p in path if p.size > 0
        ]
        if len(pts) < 2:
            continue
        cv2.polylines(court, [np.array(pts)], False, color, thickness)

    return court
