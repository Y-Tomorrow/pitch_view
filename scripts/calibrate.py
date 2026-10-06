#!/usr/bin/env python3
"""
Standalone GUI calibration tool for a fixed camera.

Shows the actual camera frame (LEFT) beside the top-down 2D pitch map (RIGHT).
Click 4+ points on each image — point *i* on the left corresponds to point *i* on
the right — then press ``C`` to compute the homography and save it as JSON.

Controls
--------
    left-click  left half   -> add a point on the actual image
    left-click  right half  -> add a point on the 2D map
    C            compute + save + preview
    R            reset all points
    Q / ESC      quit

Notes
-----
  * The correspondence is by click ORDER, so click the SAME physical pitch point
    in both halves (e.g. a corner, penalty spot, …) in the same order on each.
  * You may use any visible pitch landmarks, not just the four corners — anything
    you can identify in both the real image and the map.
  * Needs a display (WSLg on Windows 11, or X11). For a headless fallback, pass
    ``--points`` with four pixel points (mapped to the four pitch corners).

Usage
-----
    python3 scripts/calibrate.py data/1.png

Writes ``<image>_calibration.json`` (used by the node) and
``<image>_calibration_preview.jpg`` (the warped result to eyeball).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from pitch_view.annotators.soccer import draw_pitch
from pitch_view.configs import SoccerPitchConfiguration

# Colours (BGR).
POINT_COLOR = (0, 0, 255)          # red
TEXT_COLOR = (0, 255, 0)           # green
BAR_BG = (30, 30, 30)
BAR_HEIGHT = 64                    # top status bar height in the window


def _parse_points(spec):
    """Parse "x1,y1;x2,y2;..." into a list of (x, y) tuples."""
    pts = []
    for tok in spec.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        x, y = tok.split(",")
        pts.append((float(x), float(y)))
    return pts


def _draw_bar(canvas, lines):
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], BAR_HEIGHT), BAR_BG, -1)
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (12, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)


def _mark(img, pts):
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (int(x), int(y)), 5, POINT_COLOR, -1)
        cv2.putText(img, str(i), (int(x) + 7, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)


def _save_calibration(image, left_pts, right_cm, config, out_json, preview_scale):
    src = np.asarray(left_pts, dtype=np.float32)
    tgt = np.asarray(right_cm, dtype=np.float32)
    homography, _ = cv2.findHomography(src, tgt)
    if homography is None:
        raise SystemExit("Could not compute a homography (points collinear or "
                         "too few?).")

    length_px = int(config.length * preview_scale)
    width_px = int(config.width * preview_scale)
    # ``homography`` maps source px -> centimetres; scale it so the warped
    # output actually lands inside the (length_px, width_px) preview canvas.
    s = preview_scale
    preview_h = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], np.float64) @ homography
    warped = cv2.warpPerspective(image, preview_h, (length_px, width_px))

    with open(out_json, "w") as f:
        json.dump({
            "sport": "soccer",
            "source": src.tolist(),
            "target": tgt.tolist(),
            "homography": homography.tolist(),
            "length": config.length,
            "width": config.width,
        }, f, indent=2)

    out_preview = os.path.splitext(out_json)[0] + "_preview.jpg"
    cv2.imwrite(out_preview, warped)
    return out_json, out_preview


def _gui(image, map_img, map_scale, map_padding, config, out_json):
    """Side-by-side click-to-calibrate window. Returns (left_pts, right_cm)."""
    lh, lw = image.shape[:2]
    mh, mw = map_img.shape[:2]
    canvas = np.zeros((BAR_HEIGHT + max(lh, mh), lw + mw, 3), dtype=np.uint8)

    left_pts, right_pts = [], []  # right_pts are in MAP PIXELS

    def render(status):
        canvas[:] = 0
        canvas[BAR_HEIGHT:BAR_HEIGHT + lh, :lw] = image
        canvas[BAR_HEIGHT:BAR_HEIGHT + mh, lw:lw + mw] = map_img
        # Draw a dividing line + labels.
        cv2.line(canvas, (lw, BAR_HEIGHT), (lw, canvas.shape[0]), (80, 80, 80), 1)
        left_view = canvas[BAR_HEIGHT:BAR_HEIGHT + lh, :lw]
        right_view = canvas[BAR_HEIGHT:BAR_HEIGHT + mh, lw:lw + mw]
        _mark(left_view, left_pts)
        _mark(right_view, right_pts)
        _draw_bar(canvas, [
            f"LEFT: real image ({len(left_pts)} pts)   RIGHT: 2D pitch ({len(right_pts)} pts)",
            status,
        ])
        cv2.imshow("pitch calibration", canvas)

    def on_click(event, x, y, flags, param):
        nonlocal left_pts, right_pts
        if event != cv2.EVENT_LBUTTONDOWN or y < BAR_HEIGHT:
            return
        yy = y - BAR_HEIGHT
        if x < lw and yy < lh:
            left_pts.append((float(x), float(yy)))
        elif x >= lw and yy < mh:
            # Store in map pixels; convert to cm at compute time.
            right_pts.append((float(x - lw), float(yy)))
        render("Click 4+ points on each image — same order. C=compute  R=reset  Q=quit")

    cv2.namedWindow("pitch calibration")
    cv2.setMouseCallback("pitch calibration", on_click)
    render("Click 4+ points on each image — same order. C=compute  R=reset  Q=quit")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyAllWindows()
            raise SystemExit("Aborted.")
        if key in (ord("r"), ord("R")):
            left_pts, right_pts = [], []
            render("Reset. Click 4+ points on each image — same order.")
        if key in (ord("c"), ord("C")):
            if len(left_pts) < 4 or len(left_pts) != len(right_pts):
                render(f"Need >=4 points AND equal counts "
                       f"(left {len(left_pts)}, right {len(right_pts)}). "
                       "Press R to reset.")
                continue
            # Convert map pixels -> centimetres.
            right_cm = [((px - map_padding) / map_scale,
                         (py - map_padding) / map_scale)
                        for (px, py) in right_pts]
            cv2.destroyAllWindows()
            return left_pts, right_cm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input frame from the fixed camera.")
    parser.add_argument("--map-scale", type=float, default=0.1,
                        help="Pixels per cm for the right-hand 2D map (default 0.1).")
    parser.add_argument("--map-padding", type=int, default=40,
                        help="Padding around the 2D map in pixels (default 40).")
    parser.add_argument("--preview-scale", type=float, default=0.05,
                        help="Pixels per cm for the warped preview (default 0.05).")
    parser.add_argument("--out", default="",
                        help="Output JSON path (default <image>_calibration.json).")
    parser.add_argument("--points", help="Headless fallback: four pixel points "
                        '"x1,y1;x2,y2;x3,y3;x4,y4" in order tl,tr,br,bl.')
    args = parser.parse_args()

    config = SoccerPitchConfiguration()
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    out_json = args.out or os.path.splitext(args.image)[0] + "_calibration.json"

    if args.points:
        # Headless: map the four corners directly to the four pitch corners.
        left_pts = _parse_points(args.points)
        right_cm = config.court_corners
        out_json, out_preview = _save_calibration(
            img, left_pts, right_cm, config, out_json, args.preview_scale)
    else:
        map_img = draw_pitch(config, scale=args.map_scale, padding=args.map_padding)
        left_pts, right_cm = _gui(
            img, map_img, args.map_scale, args.map_padding, config, out_json)
        out_json, out_preview = _save_calibration(
            img, left_pts, right_cm, config, out_json, args.preview_scale)

    print(f"calibration: {out_json}")
    print(f"preview:     {out_preview}")
    print("\nPoint config/court_detector.yaml at this file:")
    print(f'  detector_type: homography\n  homography_file: "{out_json}"')


if __name__ == "__main__":
    main()
