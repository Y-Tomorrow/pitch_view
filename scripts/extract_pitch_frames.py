#!/usr/bin/env python3
"""
Pull pitch views out of a broadcast video for keypoint labelling.

Close-ups and near-duplicate frames are dropped. What remains is a spread of
this video's own cameras, zooms, and lighting, ready to annotate as 32 pitch
keypoints (see docs/annotation.md). Images are not labelled.

Example:
    python3 scripts/extract_pitch_frames.py data/master.mp4
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np


def _foreground_frac(field):
    """Largest non-pitch object, ignoring the crowd band along the top."""
    mask = (~field).astype(np.uint8)
    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = 0
    for i in range(1, count):
        _x, y, w, h, area = (int(v) for v in stats[i])
        if w > 0.70 * width and (y + h) < 0.50 * height:
            continue
        if w > 0.65 * width and h < 0.18 * height:
            continue
        best = max(best, area)
    return best / float(mask.size)


def _pitch_stats(frame):
    """Field coverage, thin white lines, and how much one person fills the frame."""
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # This broadcast's grass sits at a yellow-green hue, including the shadows.
    field = (hue >= 26) & (hue <= 55) & (sat >= 40) & (val >= 25)
    white = (sat <= 55) & (val >= 175)
    near_field = cv2.dilate(
        field.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    thick = cv2.morphologyEx(
        white.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)
    ) > 0
    lines = white & near_field & ~thick
    height, width = field.shape
    y0, y1 = int(height * 0.30), int(height * 0.92)
    x0, x1 = int(width * 0.08), int(width * 0.92)
    center = field[y0:y1, x0:x1]
    glare = float((val >= 245).mean())
    return (float(field.mean()), float(center.mean()), float(lines.mean()),
            _foreground_frac(field), glare)


def _signature(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA)
    return tiny.astype(np.float32)


def _similar(left, right, threshold):
    a = left - left.mean()
    b = right - right.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return True
    return float((a * b).sum() / denom) >= threshold


def _is_pitch_view(green, center, lines, foreground, glare):
    """Wide or medium pitch view with visible markings, not a close-up."""
    if glare > 0.12 or foreground > 0.045:
        return False
    if green < 0.28 or center < 0.55:
        return False
    return lines >= 0.004


def _write_preview(paths, dest, cols=6):
    thumbs = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        thumbs.append(cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    rows = []
    blank = np.zeros_like(thumbs[0])
    for i in range(0, len(thumbs), cols):
        chunk = thumbs[i:i + cols]
        if len(chunk) < cols:
            chunk.extend([blank] * (cols - len(chunk)))
        rows.append(np.hstack(chunk))
    cv2.imwrite(dest, np.vstack(rows))


def extract(source, out_dir, sample_sec, min_gap_sec, refresh_sec, similar):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频 {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * sample_sec)))
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    for name in os.listdir(image_dir):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(os.path.join(image_dir, name))

    rows = []
    last_sig = None
    last_kept_t = -1e9
    index = 0
    scanned = 0
    while True:
        if step > 1 and not all(cap.grab() for _ in range(step - 1)):
            break
        ok, frame = cap.read()
        if not ok:
            break
        scanned += 1
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        green, center, lines, foreground, glare = _pitch_stats(frame)
        if scanned % 200 == 0:
            done = f"{timestamp:.0f}s"
            if frame_count:
                done = f"{timestamp:.0f}/{frame_count / fps:.0f}s"
            print(f"[extract] {done}  scanned {scanned}  kept {len(rows)}",
                  file=sys.stderr, flush=True)
        if not _is_pitch_view(green, center, lines, foreground, glare):
            continue
        sig = _signature(frame)
        gap = timestamp - last_kept_t
        if last_sig is not None and gap < refresh_sec and _similar(sig, last_sig, similar):
            continue
        if gap < min_gap_sec:
            continue
        index += 1
        name = f"frame_{index:05d}_t{timestamp:07.1f}.jpg"
        path = os.path.join(image_dir, name)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({
            "file": name,
            "time_sec": f"{timestamp:.2f}",
            "green": f"{green:.3f}",
            "center_green": f"{center:.3f}",
            "lines": f"{lines:.4f}",
            "foreground": f"{foreground:.3f}",
            "glare": f"{glare:.3f}",
        })
        last_sig = sig
        last_kept_t = timestamp
    cap.release()

    manifest = os.path.join(out_dir, "manifest.csv")
    with open(manifest, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[
                "file", "time_sec", "green", "center_green", "lines",
                "foreground", "glare"])
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        picks = [rows[int(i)]["file"] for i in np.linspace(0, len(rows) - 1, num=min(24, len(rows)))]
        _write_preview(
            [os.path.join(image_dir, name) for name in picks],
            os.path.join(out_dir, "preview.jpg"))
    print(f"[extract] kept {len(rows)} / scanned {scanned} -> {image_dir}",
          file=sys.stderr, flush=True)
    return len(rows)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="从转播视频抽取球场关键点标注帧")
    parser.add_argument("source", nargs="?", default=os.path.join(root, "data", "master.mp4"))
    parser.add_argument("--out-dir", default=os.path.join(root, "data", "pitch_dataset"))
    parser.add_argument("--sample-sec", type=float, default=0.8,
                        help="候选帧间隔（秒）。")
    parser.add_argument("--min-gap-sec", type=float, default=2.5,
                        help="两张入选图的最短间隔。")
    parser.add_argument("--refresh-sec", type=float, default=6.0,
                        help="同一机位超过这么久仍保留一张，用来覆盖遮挡和光线变化。")
    parser.add_argument("--similar", type=float, default=0.96,
                        help="画面相关度高于此值视为同一机位。")
    args = parser.parse_args()
    extract(
        args.source, args.out_dir, args.sample_sec,
        args.min_gap_sec, args.refresh_sec, args.similar)


if __name__ == "__main__":
    main()
