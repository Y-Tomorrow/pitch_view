from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pitch_view.configs.soccer import SoccerPitchConfiguration

# BGR colour tuples.
PITCH_BACKGROUND = (0, 0, 0)  # line-only schematic (no grass fill)
LINE_WHITE = (255, 255, 255)
KEYPOINT_EDGE = (20, 20, 20)
LABEL_BG = (0, 0, 0)

# High-contrast palette for keypoint dots (cycles).
_KP_PALETTE = (
    (0, 0, 255),      # red
    (0, 200, 255),    # amber
    (0, 255, 128),    # mint
    (255, 160, 0),    # sky
    (255, 0, 180),    # magenta
    (50, 220, 50),    # green
    (255, 255, 0),    # cyan
    (180, 100, 255),  # pink
)


def _pt(x: float, y: float, scale: float, padding: int) -> Tuple[int, int]:
    return (int(round(x * scale)) + padding, int(round(y * scale)) + padding)


def _draw_pitch_markings(
    img: np.ndarray,
    config: SoccerPitchConfiguration,
    *,
    scale: float,
    padding: int,
    line_color: Tuple[int, int, int] = LINE_WHITE,
    line_thickness: int = 3,
) -> None:
    """Paint boundary / box / circle markings onto ``img`` in-place."""
    def pt(x: float, y: float) -> Tuple[int, int]:
        return _pt(x, y, scale, padding)

    for a, b in config.edges:
        va, vb = config.vertices[a - 1], config.vertices[b - 1]
        cv2.line(img, pt(va[0], va[1]), pt(vb[0], vb[1]),
                 line_color, line_thickness, cv2.LINE_AA)

    centre = pt(config.length / 2, config.width / 2)
    cv2.circle(img, centre, int(config.centre_circle_radius * scale),
               line_color, line_thickness, cv2.LINE_AA)
    spot_r = max(2, int(4 * scale))
    cv2.circle(img, centre, spot_r, line_color, -1, cv2.LINE_AA)
    cv2.circle(img, pt(config.penalty_spot_distance, config.width / 2),
               spot_r, line_color, -1, cv2.LINE_AA)
    cv2.circle(img, pt(config.length - config.penalty_spot_distance, config.width / 2),
               spot_r, line_color, -1, cv2.LINE_AA)


def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: Tuple[int, int, int] = PITCH_BACKGROUND,
    line_color: Tuple[int, int, int] = LINE_WHITE,
    padding: int = 50,
    line_thickness: int = 3,
    scale: float = 0.05,
) -> np.ndarray:
    """
    Draw a top-down 2D football pitch image from the pitch configuration.

    Args:
        config: Pitch keypoint layout and dimensions.
        background_color: Pitch surface colour (BGR).
        line_color: Line colour (BGR).
        padding: Padding around the pitch in pixels.
        line_thickness: Line thickness in pixels.
        scale: Pixels per centimetre.

    Returns:
        BGR image of the pitch.
    """
    length_px = int(config.length * scale)
    width_px = int(config.width * scale)

    img = np.full((width_px + 2 * padding, length_px + 2 * padding, 3),
                  background_color, dtype=np.uint8)

    _draw_pitch_markings(
        img, config, scale=scale, padding=padding,
        line_color=line_color, line_thickness=line_thickness,
    )
    return img


def _match_vertex_indices(
    target: np.ndarray,
    vertices: Sequence[Tuple[float, float]],
    tol_cm: float = 25.0,
) -> List[Optional[int]]:
    """Map each target point to a config vertex index (or None if unmatched)."""
    verts = np.asarray(vertices, dtype=np.float32)
    out: List[Optional[int]] = []
    for pt in np.asarray(target, dtype=np.float32):
        d = np.linalg.norm(verts - pt[None, :], axis=1)
        i = int(np.argmin(d))
        out.append(i if d[i] <= tol_cm else None)
    return out


def annotate_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    *,
    target: Optional[np.ndarray] = None,
    config: Optional[SoccerPitchConfiguration] = None,
    labels: Optional[Sequence[str]] = None,
    draw_edges: bool = True,
    radius: int = 7,
) -> np.ndarray:
    """
    Draw keypoints on a camera frame with outlines, readable labels, and
    optional pitch-edge connections when ``target``/``config`` allow matching.
    """
    out = frame.copy()
    kps = np.asarray(keypoints, dtype=np.float32)
    if kps.size == 0:
        return out

    # Optional edges between detected keypoints that share a pitch marking.
    # Fall back to a convex hull when targets don't match known vertices
    # (e.g. a free 4-point calibration JSON).
    drew_edge = False
    if draw_edges and target is not None and config is not None:
        idx = _match_vertex_indices(target, config.vertices)
        pix_by_v = {v: (int(kps[i, 0]), int(kps[i, 1]))
                    for i, v in enumerate(idx) if v is not None}
        for a, b in config.edges:
            pa, pb = pix_by_v.get(a - 1), pix_by_v.get(b - 1)
            if pa is not None and pb is not None:
                cv2.line(out, pa, pb, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(out, pa, pb, (40, 180, 255), 1, cv2.LINE_AA)
                drew_edge = True
    if draw_edges and not drew_edge and len(kps) >= 3:
        hull = cv2.convexHull(kps.astype(np.float32))
        cv2.polylines(out, [hull.astype(np.int32)], True,
                      (255, 255, 255), 2, cv2.LINE_AA)
        cv2.polylines(out, [hull.astype(np.int32)], True,
                      (40, 180, 255), 1, cv2.LINE_AA)

    for i, (x, y) in enumerate(kps):
        centre = (int(round(x)), int(round(y)))
        color = _KP_PALETTE[i % len(_KP_PALETTE)]
        cv2.circle(out, centre, radius + 2, KEYPOINT_EDGE, -1, cv2.LINE_AA)
        cv2.circle(out, centre, radius, color, -1, cv2.LINE_AA)

        text = labels[i] if labels is not None and i < len(labels) else str(i)
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx, ty = centre[0] + radius + 4, centre[1] - 4
        cv2.rectangle(
            out,
            (tx - 2, ty - th - 2),
            (tx + tw + 2, ty + baseline + 2),
            LABEL_BG,
            -1,
        )
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: Optional[Tuple[int, int, int]] = None,
    edge_color: Tuple[int, int, int] = KEYPOINT_EDGE,
    radius: int = 7,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.05,
    pitch: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Draw points (court cm coordinates) onto a top-down pitch image."""
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)
    else:
        pitch = pitch.copy()

    for i, point in enumerate(np.asarray(xy, dtype=np.float32)):
        colour = face_color if face_color is not None else _KP_PALETTE[i % len(_KP_PALETTE)]
        centre = _pt(point[0], point[1], scale, padding)
        cv2.circle(pitch, centre, radius, colour, -1, cv2.LINE_AA)
        cv2.circle(pitch, centre, radius, edge_color, thickness, cv2.LINE_AA)
        if labels is not None and i < len(labels):
            cv2.putText(pitch, str(labels[i]), (centre[0] + 8, centre[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
    return pitch


def compose_topdown(
    config: SoccerPitchConfiguration,
    *,
    warped: Optional[np.ndarray] = None,
    keypoints_pitch: Optional[np.ndarray] = None,
    entities: Optional[Sequence] = None,
    scale: float = 0.1,
    padding: int = 40,
    warp_alpha: float = 1.0,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Build a readable top-down view: grass pitch template, optional camera warp
    blended into the field, crisp pitch lines on top, and keypoint markers.
    """
    pitch = draw_pitch(
        config, padding=padding, line_thickness=line_thickness, scale=scale)

    length_px = int(config.length * scale)
    width_px = int(config.width * scale)

    if warped is not None:
        if warped.shape[1] != length_px or warped.shape[0] != width_px:
            warped = cv2.resize(
                warped, (length_px, width_px), interpolation=cv2.INTER_LINEAR)

        roi = pitch[padding:padding + width_px, padding:padding + length_px]
        if warp_alpha >= 0.99:
            roi[:] = warped
        else:
            roi[:] = cv2.addWeighted(warped, warp_alpha, roi, 1.0 - warp_alpha, 0)

        _draw_pitch_markings(
            pitch, config, scale=scale, padding=padding,
            line_color=LINE_WHITE, line_thickness=line_thickness,
        )

    if keypoints_pitch is not None and len(keypoints_pitch) > 0:
        pitch = draw_points_on_pitch(
            config,
            keypoints_pitch,
            padding=padding,
            scale=scale,
            pitch=pitch,
            labels=[str(i) for i in range(len(keypoints_pitch))],
            radius=max(5, int(6 * scale / 0.1)),
        )

    if entities:
        r = max(8, int(8 * scale / 0.1))
        for xy, color, label in entities:
            pitch = draw_points_on_pitch(
                config,
                np.asarray([xy], dtype=np.float32),
                face_color=color,
                padding=padding,
                scale=scale,
                pitch=pitch,
                labels=[label],
                radius=r,
            )

    cv2.rectangle(
        pitch,
        (padding - 1, padding - 1),
        (padding + length_px, padding + width_px),
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return pitch
