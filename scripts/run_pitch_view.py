#!/usr/bin/env python3
"""
Detect the pitch (fixed homography OR a keypoint model) and show the top-down
2D pitch view — no ROS2 required.

By default this opens a Qt window (no arguments needed). Pick the detection
method, its config, and the input source from the UI:

  * homography (default): fixed camera, needs a saved 4-point calibration JSON.
  * yolo: a local YOLO keypoint model (``weights/pitch_keypoints.pt``).
  * inference / roboflow: a Roboflow keypoint model (API key).

Weights live under ``weights/`` (train scripts publish there). For batch use,
pass ``--cli`` to process an image/video and write files to disk.

Usage:
    # Qt GUI (default)
    python3 scripts/run_pitch_view.py

    # batch: fixed homography (single image)
    python3 scripts/run_pitch_view.py --cli --calibration data/1_calibration.json data/1.png

    # batch: model inference (local .pt)
    python3 scripts/run_pitch_view.py --cli --detector yolo \\
        --model weights/pitch_keypoints.pt data/1.png
"""
import argparse
import multiprocessing as mp
import os
import sys
import threading
import time

# Set these BEFORE importing cv2 / torch / Qt: the three libraries can pull in
# conflicting OpenMP/threading runtimes (a common segfault source), and the xcb
# backend can crash on shared-memory blits under WSLg. These workarounds are
# harmless on machines that don't need them.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("QT_X11_NO_MITSHM", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from pitch_view.annotators.soccer import annotate_keypoints, compose_topdown
from pitch_view.common import ViewTransformer, create_detector
from pitch_view.common.settings import (
    default_config_path,
    load_settings,
    resolve_repo_path,
)
from pitch_view.common.weights import (
    weights_dir,
    paddle_ocr_dir,
    pitch_keypoints_path,
    players_path,
)
from pitch_view.configs import SoccerPitchConfiguration

# Filled in main() from config/court_detector.yaml (CLI can override a few).
SETTINGS = load_settings()

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_DEFAULT_MODEL = "football-field-detection-f07vi/14"

# Detection methods shown in the UI combo: (label, kind).
_METHODS = [
    ("固定机位标定 (homography)", "homography"),
    ("本地模型 (yolo)", "yolo"),
    ("Roboflow 本地推理 (inference)", "inference"),
    ("Roboflow 云端 (roboflow)", "roboflow"),
]


def _fix_qt_plugin_path():
    """opencv-python (non-headless) bundles its own Qt and overrides
    ``QT_QPA_PLATFORM_PLUGIN_PATH`` to a directory lacking the xcb plugin PyQt5
    needs, so the app crashes on startup. Point it back at the system Qt plugins."""
    for d in (
        "/usr/lib/x86_64-linux-gnu/qt5/plugins",
        "/usr/lib/x86_64-linux-gnu/qt6/plugins",
        "/usr/lib64/qt5/plugins",
    ):
        if os.path.isdir(os.path.join(d, "platforms")):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = d
            return
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


_fix_qt_plugin_path()

from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QPushButton, QSlider, QSplitter, QVBoxLayout,
    QWidget,
)


def _apply_cjk_font(app):
    """Set a CJK-capable application font so Chinese UI text renders."""
    for path in (
        os.path.expanduser("~/.local/share/fonts/NotoSansCJKsc-Regular.otf"),
        os.path.expanduser("~/.fonts/NotoSansCJKsc-Regular.otf"),
    ):
        if os.path.isfile(path):
            QFontDatabase.addApplicationFont(path)
    families = set(QFontDatabase().families())
    for name in ("Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei",
                 "Source Han Sans SC", "Microsoft YaHei", "PingFang SC"):
        if name in families:
            app.setFont(QFont(name, 10))
            app.setStyleSheet(f"QWidget {{ font-family: '{name}'; }}")
            return


# --------------------------------------------------------------------------- #
# Subprocess-isolated YOLO inference.
# --------------------------------------------------------------------------- #
def _yolo_worker(model_path, keypoint_names, keypoint_vertices, device, conn,
                 min_confidence=0.0):
    """Runs ``YoloKeyPointDetector`` in this process. torch/CUDA are imported and
    initialised here — never inside the Qt main thread, which segfaults under
    WSLg when the two share OpenMP/threading runtimes.

    Protocol: every ``recv()`` receives a frame (numpy array) or ``None``
    (shutdown). Every ``send()`` emits ``(ok, payload)`` — ``(True, (keypoints,
    target))`` on success, ``(False, error_str)`` on failure. The model is loaded
    before the loop, so the first reply arrives only once the (slow) load is done."""
    try:
        # Re-assert these in this process (a `spawn`ed child re-imports the module
        # fresh, but belt-and-suspenders). They must be set BEFORE torch/OpenMP
        # initialise, or the CUDA forward pass can abort on a duplicated OpenMP
        # runtime inherited from the GUI process.
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["MKL_NUM_THREADS"] = "1"

        from pitch_view.common.detector import YoloKeyPointDetector

        det = YoloKeyPointDetector(
            model_path, keypoint_names, keypoint_vertices, device,
            min_confidence=min_confidence)
        print("[yolo-worker] loading model…", file=sys.stderr, flush=True)
        det.load()
        print(f"[yolo-worker] model loaded on {det.device}"
              f"{' fp16' if det._half else ''}, ready.",
              file=sys.stderr, flush=True)
        while True:
            frame = conn.recv()
            if frame is None:
                break
            keypoints, target = det.detect(frame)
            conn.send((True, (keypoints, target)))
    except Exception as exc:  # noqa: BLE001 — surface load/infer errors to the GUI
        print(f"[yolo-worker] error: {exc}", file=sys.stderr, flush=True)
        try:
            conn.send((False, str(exc)))
        except Exception:
            pass


class SubprocessYoloDetector:
    """Same ``detect(frame) -> (keypoints, target)`` contract, but forwards each
    frame to a worker process and returns the keypoints. Keeps torch/CUDA out of
    the Qt process so model inference can't crash the GUI.

    ``__init__`` only spawns the worker (returns immediately); the model loads in
    the background, so the GUI is never blocked on the slow first-time load."""

    def __init__(self, model_path, keypoint_names, keypoint_vertices, device,
                 min_confidence=0.0):
        self._vertices = np.asarray(keypoint_vertices, dtype=np.float32)
        # Use the `spawn` start method (a fresh process), NOT the default `fork`.
        # The GUI process has already loaded OpenCV's OpenMP runtime and Qt; a
        # forked child inherits that state and aborts during torch's CUDA forward
        # pass. A spawned child re-imports torch cleanly with OMP_NUM_THREADS=1.
        ctx = mp.get_context("spawn")
        self._parent, child = ctx.Pipe()
        self._proc = ctx.Process(
            target=_yolo_worker,
            args=(model_path, keypoint_names, keypoint_vertices, device, child,
                  min_confidence),
            daemon=True,
        )
        self._proc.start()
        child.close()

    def detect(self, frame):
        self._parent.send(frame)
        ok, payload = self._parent.recv()
        if not ok:
            raise RuntimeError(payload)
        keypoints, target = payload
        return keypoints, target

    def close(self):
        try:
            self._parent.send(None)
        except Exception:
            pass
        self._proc.join(timeout=3)
        if self._proc.is_alive():
            self._proc.terminate()


_PLAYER_NAMES = ("ball", "GK", "player", "ref")
_PLAYER_NAMES_ZH = ("球", "门将", "球员", "裁判")
_PERSON_CLASSES = {1, 2, 3}  # GK / player / ref — NMS across these
# BGR
_PLAYER_COLORS = {
    0: (0, 255, 255),    # ball
    1: (0, 140, 255),    # goalkeeper
    2: (255, 180, 50),   # player
    3: (80, 220, 80),    # referee
}


def _xyxy_overlap(a, b):
    """Return (IoU, intersection / min(area)) for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0, 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    iou = inter / union if union > 0.0 else 0.0
    contain = inter / min(aa, ba) if min(aa, ba) > 0.0 else 0.0
    return iou, contain


def _nms_group(boxes, iou_thr=0.45, contain_thr=0.7):
    """Greedy NMS: keep highest-score box, drop overlapping duplicates."""
    if len(boxes) <= 1:
        return list(boxes)
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][1], reverse=True)
    keep = []
    suppressed = [False] * len(boxes)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(boxes[i])
        _, _, a = boxes[i]
        for j in order:
            if suppressed[j] or j == i:
                continue
            iou, contain = _xyxy_overlap(a, boxes[j][2])
            if iou >= iou_thr or contain >= contain_thr:
                suppressed[j] = True
    return keep


def _dedupe_player_boxes(boxes, iou_thr=0.45, contain_thr=0.7):
    """Drop stacked boxes on the same person (player+GK, nested player, …).

    YOLO's default NMS is class-aware, so one athlete often yields two boxes
    (e.g. ``player`` and ``GK``). Balls are NMS'd among themselves so a
    ball-in-feet box is not swallowed by the player box.
    """
    if not boxes:
        return boxes
    persons = [b for b in boxes if int(b[0]) in _PERSON_CLASSES]
    balls = [b for b in boxes if int(b[0]) not in _PERSON_CLASSES]
    return _nms_group(persons, iou_thr, contain_thr) + _nms_group(
        balls, iou_thr, contain_thr)


def _default_keypoint_model():
    """Packaged keypoint weights from YAML ``model_path`` or weights/."""
    path = resolve_repo_path(SETTINGS.get("model_path", "")) or pitch_keypoints_path()
    return path if os.path.isfile(path) else ""


def _default_player_model():
    """Packaged player weights from YAML ``player_model_path`` or weights/."""
    path = resolve_repo_path(SETTINGS.get("player_model_path", "")) or players_path()
    return path if os.path.isfile(path) else ""


def _player_worker(model_path, device, conn, conf, iou, agnostic_nms,
                   nms_iou, nms_contain):
    """YOLO detect worker in a spawn-ed process (keep torch out of Qt)."""
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["MKL_NUM_THREADS"] = "1"
        from ultralytics import YOLO

        print("[player-worker] loading model…", file=sys.stderr, flush=True)
        from pitch_view.common.detector import _cuda_or_cpu
        device, half = _cuda_or_cpu(device)
        model = YOLO(model_path).to(device)
        print(f"[player-worker] model loaded on {device}"
              f"{' fp16' if half else ''}, ready.",
              file=sys.stderr, flush=True)
        while True:
            frame = conn.recv()
            if frame is None:
                break
            result = model(
                frame, verbose=False, conf=conf, iou=iou,
                agnostic_nms=agnostic_nms, device=device, imgsz=640,
                **({"quantize": 16} if half else {}))[0]
            boxes = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                cls = result.boxes.cls.cpu().numpy().astype(np.int32)
                scores = result.boxes.conf.cpu().numpy().astype(np.float32)
                for i in range(len(xyxy)):
                    boxes.append((int(cls[i]), float(scores[i]), xyxy[i]))
            boxes = _dedupe_player_boxes(boxes, nms_iou, nms_contain)
            conn.send((True, boxes))
    except Exception as exc:  # noqa: BLE001
        print(f"[player-worker] error: {exc}", file=sys.stderr, flush=True)
        try:
            conn.send((False, str(exc)))
        except Exception:
            pass


class SubprocessPlayerDetector:
    def __init__(self, model_path, device, settings=None):
        s = settings or SETTINGS
        conf = float(s.get("player_conf", 0.25))
        iou = float(s.get("player_iou", 0.45))
        agnostic = bool(s.get("player_agnostic_nms", True))
        nms_iou = float(s.get("player_nms_iou", 0.45))
        nms_contain = float(s.get("player_nms_contain", 0.7))
        ctx = mp.get_context("spawn")
        self._parent, child = ctx.Pipe()
        self._proc = ctx.Process(
            target=_player_worker,
            args=(model_path, device, child, conf, iou, agnostic,
                  nms_iou, nms_contain),
            daemon=True,
        )
        self._proc.start()
        child.close()

    def detect(self, frame):
        self._parent.send(frame)
        ok, payload = self._parent.recv()
        if not ok:
            raise RuntimeError(payload)
        return payload

    def close(self):
        try:
            self._parent.send(None)
        except Exception:
            pass
        self._proc.join(timeout=3)
        if self._proc.is_alive():
            self._proc.terminate()


def _split_box(item):
    cls, conf, xyxy = item[0], item[1], item[2]
    number = item[3] if len(item) >= 4 else ""
    return int(cls), float(conf), xyxy, (number or "")


def _torso_kwargs(settings=None):
    s = settings or SETTINGS
    return dict(
        x_margin=float(s.get("jersey_torso_x_margin", 0.0)),
        y0=float(s.get("jersey_torso_y0", 0.05)),
        y1=float(s.get("jersey_torso_y1", 0.55)),
    )


def _clamp_xyxy(frame_shape, xyxy):
    height, width = int(frame_shape[0]), int(frame_shape[1])
    left, top, right, bottom = (int(round(float(v))) for v in xyxy)
    left = max(0, min(left, max(0, width - 1)))
    top = max(0, min(top, max(0, height - 1)))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def _player_box_crop(frame, xyxy):
    """Exact detection-box pixels for the clicked person."""
    left, top, right, bottom = _clamp_xyxy(frame.shape, xyxy)
    if right - left < 8 or bottom - top < 8:
        return None
    crop = frame[top:bottom, left:right].copy()
    return crop if crop.size else None


def _torso_bounds(frame_shape, xyxy, x_margin=0.0, y0=0.05, y1=0.55):
    """Upper body inside the player box, using the box's full width.

    ``y0`` / ``y1`` are fractions of the box height. They must not be named
    the same as the box's pixel coordinates, or the bottom fraction gets
    replaced by a pixel value and the crop lands off the player.
    """
    left, top, right, bottom = _clamp_xyxy(frame_shape, xyxy)
    box_w, box_h = right - left, bottom - top
    if box_w < 12 or box_h < 24:
        return None
    top_frac = min(max(0.0, float(y0)), 0.8)
    bot_frac = min(max(top_frac + 0.15, float(y1)), 1.0)
    inset = int(round(box_w * min(max(0.0, float(x_margin)), 0.15)))
    x1 = left + inset
    x2 = right - inset
    y_top = top + int(round(box_h * top_frac))
    y_bot = top + int(round(box_h * bot_frac))
    x1 = max(left, min(x1, right - 2))
    x2 = min(right, max(x1 + 1, x2))
    y_top = max(top, min(y_top, bottom - 2))
    y_bot = min(bottom, max(y_top + 1, y_bot))
    if x2 - x1 < 8 or y_bot - y_top < 8:
        return None
    return int(x1), int(y_top), int(x2), int(y_bot)


def _torso_crop(frame, xyxy, x_margin=0.0, y0=0.05, y1=0.55):
    """Upper-body crop taken from the player detection box."""
    bounds = _torso_bounds(
        frame.shape, xyxy, x_margin=x_margin, y0=y0, y1=y1)
    if bounds is None:
        return None
    x1, y_top, x2, y_bot = bounds
    crop = frame[y_top:y_bot, x1:x2]
    return crop if crop.size else None


def _upscale_min_height(image, min_height=260, max_side=720):
    height, width = image.shape[:2]
    scale = min_height / max(1, height) if height < min_height else 1.0
    if max(height, width) * scale > max_side:
        scale = max_side / max(height, width)
    if scale <= 1.01:
        return np.ascontiguousarray(image)
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))),
         max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_CUBIC)


def _enhance_digits(image):
    """Make a small jersey number large and high-contrast for the detector."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = _upscale_min_height(gray)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def _prepare_ocr_image(crop):
    """Color chest crop, enlarged so the debug window matches what we read."""
    return _upscale_min_height(np.ascontiguousarray(crop))


def _default_ocr_variants(chest):
    """EasyOCR still reads the high-contrast grayscale pair."""
    enhanced = _enhance_digits(chest)
    return (
        ("增强", enhanced),
        ("反色", cv2.bitwise_not(enhanced)),
    )


def _read_digits(reader, image):
    """EasyOCR with thresholds low enough to see a small jersey number."""
    kwargs = {
        "detail": 1,
        "paragraph": False,
        "allowlist": "0123456789",
        "min_size": 6,
        "text_threshold": 0.35,
        "low_text": 0.2,
        "link_threshold": 0.25,
        "mag_ratio": 1.8,
        "contrast_ths": 0.02,
        "adjust_contrast": 0.7,
    }
    optional = (
        "adjust_contrast", "contrast_ths", "mag_ratio", "link_threshold",
        "low_text", "text_threshold", "min_size", "allowlist",
    )
    while True:
        try:
            return reader.readtext(image, **kwargs)
        except TypeError:
            for key in optional:
                if key in kwargs:
                    kwargs.pop(key)
                    break
            else:
                return reader.readtext(image)


def _normalize_jersey_text(text):
    t = (text or "").strip().upper()
    t = (t.replace("O", "0").replace("I", "1").replace("L", "1")
           .replace("Z", "2").replace("S", "5").replace("B", "8"))
    digits = "".join(c for c in t if c.isdigit())
    if len(digits) in (1, 2):
        return digits
    return ""


def _ocr_debug(item):
    if item is not None and len(item) >= 5 and isinstance(item[4], dict):
        return item[4]
    return None


def _choose_jersey(candidates, min_conf):
    """Pick a jersey number, keeping a close two-digit read over one digit.

    A clear ``23`` is often detected as ``2`` and ``3``. The single box can
    score higher than the full number, so a two-digit candidate wins when it
    is within 0.25 of the best one-digit score.
    """
    valid = [c for c in candidates if c[0] and c[1] >= min_conf]
    ones = [c for c in valid if len(c[0]) == 1]
    twos = [c for c in valid if len(c[0]) == 2]
    best_one = max((c[1] for c in ones), default=-1.0)
    twos_ok = [c for c in twos if c[1] >= max(min_conf, best_one - 0.25)]
    pool = twos_ok or ones
    if not pool:
        return "", -1.0, ""
    return max(pool, key=lambda c: c[1])


def _describe_ocr_hits(ocr_items, min_conf=0.35):
    """Explain every OCR hit and pick the best 1–2 digit number."""
    rows = []
    for item in ocr_items or []:
        if not item or len(item) < 2:
            continue
        text = str(item[1])
        score = float(item[2]) if len(item) > 2 else 0.0
        number = _normalize_jersey_text(text)
        rows.append({"text": text, "score": score, "number": number})
    best_number, _best_score, best_i = _choose_jersey(
        [(row["number"], row["score"], i) for i, row in enumerate(rows)],
        min_conf)
    for i, row in enumerate(rows):
        if not row["number"]:
            row["verdict"] = "丢弃：不是 1～2 位数字"
        elif row["score"] < min_conf:
            row["verdict"] = f"丢弃：置信度 < {min_conf:.2f}"
        elif i == best_i:
            row["verdict"] = "采用"
        else:
            row["verdict"] = "丢弃：低于已采用的结果"
    return best_number, rows


def _empty_ocr_debug(status, reason, min_conf):
    return {
        "status": status,
        "reason": reason,
        "min_conf": float(min_conf),
        "number": "",
        "hits": [],
        "raw_crop": None,
        "crop": None,
    }


def _ocr_one_jersey(reader, frame, xyxy, settings=None):
    """Return ``(number, debug)``. ``debug`` keeps the crops OCR actually saw."""
    s = settings or SETTINGS
    min_conf = float(s.get("jersey_ocr_min_conf", 0.2))
    x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
    bw, bh = x2 - x1, y2 - y1
    player, chest = _jersey_views(frame, xyxy, s)
    if player is None or chest is None:
        return "", _empty_ocr_debug(
            "too_small",
            f"球员框过小（{max(0, bw)}×{max(0, bh)}），裁不出球员图像",
            min_conf)
    rows = []
    steps = []
    found = []
    variants = (reader.ocr_variants(chest)
                if hasattr(reader, "ocr_variants")
                else _default_ocr_variants(chest))
    for idx, (label, image) in enumerate(variants):
        image = np.ascontiguousarray(image)
        steps.append({"label": label, "image": image})
        hits = reader.read(image)
        number, part = _describe_ocr_hits(hits, min_conf=min_conf)
        for row in part:
            row["verdict"] = f"{label} · {row['verdict']}"
        rows.extend(part)
        if number:
            score = next(r["score"] for r in part if r["verdict"].endswith("采用"))
            found.append((number, score, label))
        best_number, best_score, best_label = _choose_jersey(found, min_conf)
        if not hasattr(reader, "ocr_variants"):
            if best_number:
                break
            continue
        # Color and its invert are enough when the read is already solid.
        # Enhancement runs only for the crops that both polarities missed.
        if best_number and best_score >= reader.confident_score and (
                len(best_number) == 2 or idx >= 1):
            break
        if idx >= 1 and best_number and best_score >= 0.55:
            break
    if not rows:
        reason = "未检出文字"
        status = "empty"
    elif not best_number:
        reason = "有文字检出，但都不是可用的 1～2 位号码"
        status = "rejected"
    else:
        engine = getattr(reader, "name", "OCR")
        reason = (
            f"{engine} 识别为 #{best_number}"
            f"（{best_label}，置信度 {best_score:.2f}）"
        )
        status = "ok"
    return best_number, {
        "status": status,
        "reason": reason,
        "min_conf": min_conf,
        "number": best_number,
        "hits": rows,
        "raw_crop": player,
        "crop": chest,
        "steps": steps,
    }


class _EasyDigitReader:
    """EasyOCR english_g2. General English text, then we keep 1–2 digits."""

    name = "EasyOCR"

    def __init__(self, device):
        import easyocr

        model_dir = weights_dir()
        need = ("craft_mlt_25k.pth", "english_g2.pth")
        missing = [n for n in need
                   if not os.path.isfile(os.path.join(model_dir, n))]
        print(f"[jersey-ocr] model dir: {model_dir}", file=sys.stderr, flush=True)
        if missing:
            print(f"[jersey-ocr] missing {missing}; will download into weights/…",
                  file=sys.stderr, flush=True)
        else:
            print("[jersey-ocr] loading EasyOCR from weights/…",
                  file=sys.stderr, flush=True)
        use_gpu = str(device).startswith("cuda")
        kwargs = dict(
            lang_list=["en"],
            gpu=use_gpu,
            model_storage_directory=model_dir,
            download_enabled=True,
            verbose=False,
        )
        try:
            self._reader = easyocr.Reader(**kwargs)
        except TypeError:
            kwargs.pop("verbose", None)
            try:
                self._reader = easyocr.Reader(**kwargs)
            except TypeError:
                self._reader = easyocr.Reader(
                    ["en"], gpu=use_gpu,
                    model_storage_directory=model_dir)

    def read(self, image):
        return _read_digits(self._reader, image)


def _poly_to_box(poly):
    """Axis-aligned box from a polygon or an ``x1, y1, x2, y2`` row."""
    try:
        arr = np.asarray(poly, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size < 4:
        return None
    if arr.ndim == 1:
        return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
    pts = arr.reshape(-1, 2)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def _digits_adjacent(a, b):
    """True when two digit boxes sit on one line, close enough to be one number."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ah = max(1.0, ay2 - ay1)
    bh = max(1.0, by2 - by1)
    if min(ah, bh) / max(ah, bh) < 0.55:
        return False
    if abs((ay1 + ay2) / 2.0 - (by1 + by2) / 2.0) > 0.45 * max(ah, bh):
        return False
    if ax1 <= bx1:
        gap = bx1 - ax2
    else:
        gap = ax1 - bx2
    return -0.25 * max(ah, bh) <= gap <= 0.85 * max(ah, bh)


def _merge_split_digits(hits):
    """Join a left/right pair of single digits into one two-digit number."""
    singles = []
    for hit in hits or []:
        if not hit or len(hit) < 2 or hit[0] is None:
            continue
        number = _normalize_jersey_text(str(hit[1]))
        rect = _poly_to_box(hit[0])
        if len(number) != 1 or rect is None:
            continue
        score = float(hit[2]) if len(hit) > 2 else 0.0
        singles.append((number, score, rect))
    neighbors = {i: [] for i in range(len(singles))}
    for i, left in enumerate(singles):
        for j in range(i + 1, len(singles)):
            if _digits_adjacent(left[2], singles[j][2]):
                neighbors[i].append(j)
                neighbors[j].append(i)
    extra = []
    used = set()
    for i, friends in neighbors.items():
        if i in used or len(friends) != 1:
            continue
        j = friends[0]
        if j in used or neighbors[j] != [i]:
            continue
        left, right = singles[i], singles[j]
        if left[2][0] > right[2][0]:
            left, right = right, left
        extra.append((None, left[0] + right[0], min(left[1], right[1])))
        used.add(i)
        used.add(j)
    if not extra:
        return list(hits or [])
    return list(hits) + extra


def _has_strong_two_digit(hits, floor=0.6):
    for hit in hits or []:
        if not hit or len(hit) < 2:
            continue
        score = float(hit[2]) if len(hit) > 2 else 0.0
        if len(_normalize_jersey_text(str(hit[1]))) == 2 and score >= floor:
            return True
    return False


def _paddle_lists(value):
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [value]
    return list(value)


def _paddle_page_hits(page):
    """Normalize one PaddleOCR 3.x page into EasyOCR-style hits.

    The detection pipeline returns ``rec_texts`` / ``rec_scores``. The
    whole-image recognizer returns the singular ``rec_text`` / ``rec_score``.
    """
    data = page
    if hasattr(page, "json"):
        raw = page.json
        if isinstance(raw, dict):
            data = raw.get("res", raw)
    texts = scores = polys = None
    if isinstance(data, dict):
        texts = data.get("rec_texts")
        scores = data.get("rec_scores")
        polys = data.get("rec_polys")
        if polys is None:
            polys = data.get("rec_boxes")
        if texts is None and data.get("rec_text") not in (None, ""):
            texts = [data.get("rec_text")]
            scores = [data.get("rec_score", 0.0)]
    if texts is None and hasattr(page, "rec_texts"):
        texts = page.rec_texts
        scores = getattr(page, "rec_scores", None)
        polys = getattr(page, "rec_polys", None)
    if texts is None and getattr(page, "rec_text", None) not in (None, ""):
        texts = [page.rec_text]
        scores = [getattr(page, "rec_score", 0.0)]
    texts = _paddle_lists(texts)
    if not texts:
        return []
    scores = _paddle_lists(scores) or []
    polys = _paddle_lists(polys) or []
    hits = []
    for i, text in enumerate(texts):
        score = float(scores[i]) if i < len(scores) else 0.0
        box = _poly_to_box(polys[i]) if i < len(polys) else None
        hits.append((box, str(text), score))
    return _merge_split_digits(hits)


def _call_accepting(ctor, kwargs):
    """Drop keyword arguments this PaddleOCR build does not take."""
    optional = (
        "text_rec_score_thresh",
        "text_det_thresh",
        "text_det_box_thresh",
        "text_det_unclip_ratio",
        "device",
        "enable_mkldnn",
        "model_name",
        "text_detection_model_name",
        "text_recognition_model_name",
        "use_doc_orientation_classify",
        "use_doc_unwarping",
        "use_textline_orientation",
    )
    while True:
        try:
            return ctor(**kwargs)
        except TypeError as exc:
            message = str(exc)
            removed = False
            for key in optional:
                if key in kwargs and key in message:
                    kwargs.pop(key)
                    removed = True
                    break
            if not removed:
                raise
        except ValueError as exc:
            message = str(exc)
            prefix = "Unknown argument:"
            if not message.startswith(prefix):
                raise
            key = message[len(prefix):].strip()
            if key not in kwargs:
                raise
            kwargs.pop(key)


def _pad_ocr(image, px=10):
    return cv2.copyMakeBorder(image, px, px, px, px, cv2.BORDER_REPLICATE)


class _PaddleDigitReader:
    """English PP-OCRv4 recognizer. Stronger on digits than EasyOCR english_g2."""

    name = "PaddleOCR"
    # A read this sure can skip the remaining image variants.
    confident_score = 0.85

    def __init__(self, device):
        cache = paddle_ocr_dir()
        os.environ["PADDLE_PDX_CACHE_HOME"] = cache
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        try:
            from paddleocr import PaddleOCR, TextRecognition
        except ImportError as exc:
            raise RuntimeError(
                "未安装 PaddleOCR。请在当前环境执行: "
                "python3 -m pip install --user paddleocr paddlepaddle"
            ) from exc
        try:
            import paddle
            gpu = (str(device).startswith("cuda")
                   and paddle.is_compiled_with_cuda())
        except Exception:
            gpu = False
        if str(device).startswith("cuda") and not gpu:
            print("[jersey-ocr] 当前 Paddle 是 CPU 版，号码识别用 CPU。",
                  file=sys.stderr, flush=True)
        print(f"[jersey-ocr] loading PaddleOCR en_PP-OCRv4 into {cache}…",
              file=sys.stderr, flush=True)
        device_name = "gpu:0" if gpu else "cpu"
        self._ocr = _call_accepting(PaddleOCR, dict(
            device=device_name,
            enable_mkldnn=False,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv4_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_thresh=0.15,
            text_det_box_thresh=0.2,
            text_det_unclip_ratio=2.4,
            text_rec_score_thresh=0.0,
        ))
        try:
            self._rec = _call_accepting(TextRecognition, dict(
                model_name="en_PP-OCRv4_mobile_rec",
                device=device_name,
                enable_mkldnn=False,
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"[jersey-ocr] whole-image recognizer unavailable: {exc}",
                  file=sys.stderr, flush=True)
            self._rec = None

    def ocr_variants(self, chest):
        """Color first. Grayscale enhancement is only a fallback."""
        color = _pad_ocr(np.ascontiguousarray(chest))
        enhanced = _pad_ocr(_enhance_digits(chest))
        return (
            ("彩色", color),
            ("彩色反色", cv2.bitwise_not(color)),
            ("增强", enhanced),
            ("增强反色", cv2.bitwise_not(enhanced)),
        )

    def read(self, image):
        image = np.ascontiguousarray(image)
        hits = []
        for page in self._ocr.predict(image) or []:
            hits.extend(_paddle_page_hits(page))
        # A partial detection used to block the whole-image recognizer, and
        # that recognizer's singular rec_text / rec_score was discarded.
        if self._rec is not None and not _has_strong_two_digit(hits):
            for page in self._rec.predict(image) or []:
                hits.extend(_paddle_page_hits(page))
        return hits


def _make_digit_reader(device, engine):
    engine = (engine or "easyocr").strip().lower()
    if engine == "paddle":
        return _PaddleDigitReader(device)
    if engine != "easyocr":
        raise RuntimeError(
            f"未知号码模型 {engine!r}（可选 easyocr 或 paddle）")
    return _EasyDigitReader(device)


def _jersey_worker(device, conn, settings=None):
    """OCR in a spawn-ed process (torch / paddle stay out of Qt)."""
    settings = dict(settings or SETTINGS)
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["MKL_NUM_THREADS"] = "1"
        engine = str(settings.get("jersey_ocr_engine", "easyocr"))
        print(f"[jersey-ocr] engine: {engine}", file=sys.stderr, flush=True)
        reader = _make_digit_reader(device, engine)
        print("[jersey-ocr] ready.", file=sys.stderr, flush=True)
        while True:
            msg = conn.recv()
            if msg is None:
                break
            frame, boxes = msg
            out = []
            for item in boxes:
                cls, conf, xyxy, _ = _split_box(item)
                number = ""
                if int(cls) in _PERSON_CLASSES:
                    number, debug = _ocr_one_jersey(reader, frame, xyxy, settings)
                else:
                    debug = _empty_ocr_debug(
                        "not_person", "球体不做号码识别",
                        float(settings.get("jersey_ocr_min_conf", 0.35)))
                out.append((cls, conf, xyxy, number, debug))
            n_hit = sum(1 for b in out if b[3])
            print(f"[jersey-ocr] {n_hit}/{len(out)} numbers",
                  file=sys.stderr, flush=True)
            conn.send((True, out))
    except Exception as exc:  # noqa: BLE001
        print(f"[jersey-ocr] error: {exc}", file=sys.stderr, flush=True)
        try:
            conn.send((False, str(exc)))
        except Exception:
            pass


class SubprocessJerseyOcr:
    def __init__(self, device, settings=None):
        ctx = mp.get_context("spawn")
        self._parent, child = ctx.Pipe()
        self._proc = ctx.Process(
            target=_jersey_worker,
            args=(device, child, dict(settings or SETTINGS)),
            daemon=True)
        self._proc.start()
        child.close()

    def recognize(self, frame, boxes):
        if not boxes:
            return []
        if self._parent.poll():
            ok, payload = self._parent.recv()
            if not ok:
                raise RuntimeError(payload)
        try:
            self._parent.send((frame, boxes))
            ok, payload = self._parent.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"号码识别进程中断: {exc}") from exc
        if not ok:
            raise RuntimeError(payload)
        return payload

    def close(self):
        try:
            self._parent.send(None)
        except Exception:
            pass
        self._proc.join(timeout=3)
        if self._proc.is_alive():
            self._proc.terminate()


def _draw_player_boxes(frame, boxes):
    out = frame.copy()
    for item in boxes:
        cls, conf, xyxy, number = _split_box(item)
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        color = _PLAYER_COLORS.get(cls, (200, 200, 200))
        name = _PLAYER_NAMES[cls] if 0 <= cls < len(_PLAYER_NAMES) else str(cls)
        label = f"{name} #{number} {conf:.2f}" if number else f"{name} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(out, label, (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def _player_entities(boxes, transformer, config, scale):
    """Map box feet (ball: centre) through H onto pitch centimetres."""
    if transformer is None or not boxes:
        return []
    out = []
    for item in boxes:
        cls, _conf, xyxy, number = _split_box(item)
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        if cls == 0:
            pt = np.array([[(x1 + x2) * 0.5, (y1 + y2) * 0.5]], np.float32)
        else:
            pt = np.array([[(x1 + x2) * 0.5, y2]], np.float32)
        mapped = transformer.transform_points(pt)[0] / scale
        if not (0.0 <= mapped[0] <= config.length and 0.0 <= mapped[1] <= config.width):
            continue
        color = _PLAYER_COLORS.get(cls, (200, 200, 200))
        name = _PLAYER_NAMES[cls] if 0 <= cls < len(_PLAYER_NAMES) else str(cls)
        if number:
            name = f"#{number}"
        out.append((mapped, color, name))
    return out


# --------------------------------------------------------------------------- #
# Headless (CLI) batch mode.
# --------------------------------------------------------------------------- #
def _build_detector(args, config):
    if args.detector == "homography" and not args.calibration:
        raise SystemExit("--detector homography 需要 --calibration <json>。")
    model_path = args.model
    if args.detector == "yolo" and not model_path:
        model_path = _default_keypoint_model()
        if not model_path:
            raise SystemExit(
                "--detector yolo 需要 --model，或先训练生成 "
                "runs/pose/train/weights/best.pt。")
    return create_detector(
        args.detector,
        model_path=model_path,
        homography_file=args.calibration,
        api_key=args.api_key,
        model_id=args.model or _DEFAULT_MODEL,
        config=config,
        device=args.device,
        min_confidence=args.min_confidence,
    )


def _warp(frame, detector, config, scale):
    """Return ``(warped_raw, keypoints, target)`` or ``(None, keypoints, target)``."""
    keypoints, target = detector.detect(frame)
    if len(keypoints) < 4:
        return None, keypoints, target
    try:
        transformer = ViewTransformer(
            source=keypoints, target=target * scale, image_shape=frame.shape)
    except ValueError:
        return None, keypoints, target
    length_px = int(config.length * scale)
    width_px = int(config.width * scale)
    warped = transformer.transform_image(frame, (length_px, width_px))
    return warped, keypoints, target


def _render_views(frame, keypoints, target, config, scale, players=None):
    """Annotated camera frame + composed top-down pitch map."""
    annotated = annotate_keypoints(
        frame, keypoints, target=target, config=config)
    if players:
        annotated = _draw_player_boxes(annotated, players)
    transformer = None
    warped = None
    if len(keypoints) >= 4:
        try:
            transformer = ViewTransformer(
                source=keypoints, target=target * scale, image_shape=frame.shape)
            length_px = int(config.length * scale)
            width_px = int(config.width * scale)
            warped = transformer.transform_image(frame, (length_px, width_px))
        except ValueError:
            transformer = None
            warped = None
    entities = _player_entities(players or [], transformer, config, scale)
    topdown = compose_topdown(
        config, warped=warped, keypoints_pitch=target, entities=entities,
        scale=scale)
    return annotated, topdown


def _run_cli(args):
    source = args.video or args.source
    if not source:
        raise SystemExit("--cli 需要图片路径或 --video <path>。")
    config = SoccerPitchConfiguration()
    detector = _build_detector(args, config)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {source!r}.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    is_image = (os.path.splitext(source)[1].lower() in _IMAGE_EXTS) \
        or frame_count == 1

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(source)) or "."
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(source))[0]

    writer = None
    wrote = 0
    topdown_size = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        warped, keypoints, target = _warp(frame, detector, config, args.scale)
        if warped is None:
            continue
        annotated = annotate_keypoints(
            frame, keypoints, target=target, config=config)
        topdown = compose_topdown(
            config, warped=warped, keypoints_pitch=target, scale=args.scale)
        if is_image:
            out_path = args.out or os.path.join(out_dir, base + "_warped.jpg")
            cv2.imwrite(out_path, topdown)
            cv2.imwrite(os.path.join(out_dir, base + "_annotated.jpg"), annotated)
            # Also keep the raw warp for downstream tooling.
            cv2.imwrite(os.path.join(out_dir, base + "_warp_raw.jpg"), warped)
            print(f"warped:    {out_path}  ({topdown.shape[1]}x{topdown.shape[0]})")
            print(f"annotated: {os.path.join(out_dir, base + '_annotated.jpg')}")
            wrote += 1
            break
        if writer is None:
            out_path = args.out or os.path.join(out_dir, base + "_warped.mp4")
            topdown_size = (topdown.shape[1], topdown.shape[0])
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, topdown_size)
        if (topdown.shape[1], topdown.shape[0]) != topdown_size:
            topdown = cv2.resize(topdown, topdown_size)
        writer.write(topdown)
        wrote += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"warped video: {out_path}  ({wrote} frames, "
              f"{topdown_size[0]}x{topdown_size[1]})")
    if wrote == 0:
        print("[warn] no frames were warped successfully (check the calibration JSON).")


# --------------------------------------------------------------------------- #
# Qt GUI.
# --------------------------------------------------------------------------- #
def _bgr_to_pixmap(frame_bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    height, width = rgb.shape[:2]
    row_bytes = int(rgb.strides[0])
    # RGB888 rows that are not 4-byte aligned get skewed by Qt, so a small
    # player crop no longer looks like the player.
    if row_bytes % 4 != 0:
        aligned = (row_bytes + 3) & ~3
        padded = np.zeros((height, aligned), np.uint8)
        padded[:, :row_bytes] = rgb.reshape(height, row_bytes)
        rgb = padded
        row_bytes = aligned
    owned = bytes(np.ascontiguousarray(rgb).tobytes())
    image = QImage(owned, width, height, row_bytes, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image)


def _player_info_lines(item, number_on=False, show_process=False):
    cls, conf, xyxy, number = _split_box(item)
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    en = _PLAYER_NAMES[cls] if 0 <= cls < len(_PLAYER_NAMES) else str(cls)
    zh = _PLAYER_NAMES_ZH[cls] if 0 <= cls < len(_PLAYER_NAMES_ZH) else en
    if number:
        num = f"#{number}"
    elif number_on:
        num = "未能识别号码"
    else:
        num = "未开启号码识别"
    lines = [
        f"类别  {zh} ({en})",
        f"检测置信度  {conf:.2f}",
        f"球衣号码  {num}",
        f"框  ({x1},{y1}) – ({x2},{y2})",
    ]
    if not show_process:
        return lines
    debug = _ocr_debug(item)
    if debug and debug.get("reason"):
        lines.append(f"识别情况  {debug['reason']}")
    if cls in _PERSON_CLASSES:
        lines.append("识别用图和逐条结果在「数字识别过程」窗口")
    return lines


def _padded_crop(frame, xyxy, pad_ratio=0.18):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = max(8.0, max(bw, bh) * pad_ratio)
    x1 = max(0, int(x1 - pad))
    y1 = max(0, int(y1 - pad))
    x2 = min(w, int(x2 + pad))
    y2 = min(h, int(y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def _zoom_crop(crop, max_w, max_h, max_scale=5.0):
    ch, cw = crop.shape[:2]
    if cw < 1 or ch < 1:
        return crop
    scale = min(max_w / cw, max_h / ch, max_scale)
    scale = max(scale, 0.05)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.01 else cv2.INTER_AREA
    return cv2.resize(crop, (nw, nh), interpolation=interp)


def _annotate_closeup(crop, item):
    """Redraw detection caption on the zoomed crop itself."""
    out = crop.copy()
    cls, conf, xyxy, number = _split_box(item)
    color = _PLAYER_COLORS.get(cls, (200, 200, 200))
    name = _PLAYER_NAMES[cls] if 0 <= cls < len(_PLAYER_NAMES) else str(cls)
    caption = f"{name} #{number} {conf:.2f}" if number else f"{name} {conf:.2f}"
    h, w = out.shape[:2]
    cv2.rectangle(out, (2, 2), (w - 3, h - 3), color, 3, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (6, 6), (12 + tw, 16 + th), (12, 14, 18), -1)
    cv2.putText(out, caption, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)
    return out


class CloseUpOverlay(QFrame):
    """Floating close-up on the camera canvas; click it to dismiss."""

    dismissed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self.img = QLabel()
        self.img.setAlignment(Qt.AlignCenter)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        lay.addWidget(self.img)
        lay.addWidget(self.info)
        self.setStyleSheet("""
            QFrame#closeup {
                background: rgba(16, 18, 22, 235);
                border: 2px solid #f0d060;
                border-radius: 6px;
            }
            QLabel { color: #f2f2f4; background: transparent; }
        """)
        self.setObjectName("closeup")

    def show_player(self, crop_bgr, lines):
        self.img.setPixmap(_bgr_to_pixmap(crop_bgr))
        self.info.setText("\n".join(lines) + "\n点击特写关闭")
        self.adjustSize()
        self.reposition()
        self.show()
        self.raise_()

    def reposition(self):
        p = self.parentWidget()
        if p is None:
            return
        m = 10
        x = max(m, p.width() - self.width() - m)
        self.move(x, m)

    def mousePressEvent(self, event):
        self.hide()
        self.dismissed.emit()
        event.accept()


def _jersey_views(frame, xyxy, settings=None):
    """Return ``(player box, upper-body slice of that same image)``.

    The slice uses the original torso fractions (``jersey_torso_y0/y1`` and
    side margin) so the recognition window stays the previously accepted crop.
    """
    player = _player_box_crop(frame, xyxy)
    if player is None:
        return None, None
    player = np.ascontiguousarray(player)
    kw = _torso_kwargs(settings)
    height, width = player.shape[:2]
    y0 = min(max(0.0, kw["y0"]), 0.8)
    y1 = min(max(y0 + 0.15, kw["y1"]), 1.0)
    margin = min(max(0.0, kw["x_margin"]), 0.2)
    top = int(round(height * y0))
    bot = int(round(height * y1))
    left = int(round(width * margin))
    right = width - left
    top = max(0, min(top, height - 2))
    bot = max(top + 1, min(bot, height))
    left = max(0, min(left, width - 2))
    right = min(width, max(left + 1, right))
    if bot - top < 8 or right - left < 8:
        chest = player
    else:
        chest = np.ascontiguousarray(player[top:bot, left:right])
    return player, _prepare_ocr_image(chest)


def _preview_jersey_crop(frame, xyxy, settings):
    """Player crops for the debug window when OCR has not run this frame."""
    s = settings or SETTINGS
    min_conf = float(s.get("jersey_ocr_min_conf", 0.35))
    raw, img = _jersey_views(frame, xyxy, s)
    if raw is None:
        x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
        return _empty_ocr_debug(
            "too_small",
            f"未开启号码识别。球员框过小（{max(0, x2 - x1)}×{max(0, y2 - y1)}），裁不出球员图像",
            min_conf)
    return {
        "status": "off",
        "reason": "未开启号码识别。左图是该球员，右图是按原比例从左图切出的上半身",
        "min_conf": min_conf,
        "number": "",
        "hits": [],
        "raw_crop": raw,
        "crop": img,
    }


def _ocr_debug_text(debug):
    if not debug:
        return "这一帧没有识别明细。"
    lines = [debug.get("reason") or "无识别说明"]
    min_conf = debug.get("min_conf")
    if min_conf is not None and debug.get("status") != "off":
        lines.append(f"置信度阈值  {float(min_conf):.2f}")
    hits = debug.get("hits") or []
    if hits:
        lines.append("原始结果：")
        for i, hit in enumerate(hits, 1):
            number = hit.get("number") or "—"
            lines.append(
                f"{i}. 「{hit.get('text', '')}」  置信度 {hit.get('score', 0):.2f}"
                f"  归一化 {number}  · {hit.get('verdict', '')}")
    elif debug.get("status") == "empty":
        lines.append("模型没有返回文字框。")
    raw = debug.get("raw_crop")
    prep = debug.get("crop")
    if raw is not None and getattr(raw, "size", 0):
        lines.append(f"球员框  {raw.shape[1]}×{raw.shape[0]} 像素")
    if prep is not None and getattr(prep, "size", 0):
        lines.append(f"送去识别的上半身  {prep.shape[1]}×{prep.shape[0]} 像素")
    return "\n".join(lines)


def _blow_up_crop(frame_bgr, max_side=280):
    """Enlarge a tiny OCR crop with nearest-neighbor so the pixels stay visible."""
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return None
    h, w = frame_bgr.shape[:2]
    scale = min(8.0, max(1.0, 220.0 / max(1, min(h, w))))
    if max(h, w) * scale > max_side:
        scale = max_side / max(h, w)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.01 else cv2.INTER_AREA
    return cv2.resize(frame_bgr, (nw, nh), interpolation=interp)


class JerseyOcrWindow(QWidget):
    """Separate window: color crop plus the images actually sent to OCR."""

    def __init__(self):
        super().__init__(None, Qt.Window)
        self.setWindowTitle("数字识别过程")
        self.user_closed = False
        self._slots = []
        pics = QHBoxLayout()
        pics.setSpacing(12)
        for title in (
                "彩色上半身",
                "送入识别 · 1",
                "送入识别 · 2",
                "送入识别 · 3",
                "送入识别 · 4"):
            caption = QLabel(title)
            caption.setAlignment(Qt.AlignCenter)
            image = QLabel("无图像")
            image.setAlignment(Qt.AlignCenter)
            image.setMinimumSize(160, 120)
            col = QVBoxLayout()
            col.addWidget(caption)
            col.addWidget(image, 1)
            pics.addLayout(col)
            self._slots.append((caption, image))
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        lay.addLayout(pics)
        lay.addWidget(self.detail)
        self.setStyleSheet("""
            QWidget { background: #16181c; color: #f2f2f4; }
            QLabel { background: transparent; }
        """)
        self.resize(760, 460)

    def present(self, debug):
        self._fill(debug)
        self.adjustSize()
        if not self.isVisible():
            self.show()
            self.raise_()

    def force_close(self):
        self._quitting = True
        self.close()

    def _fill(self, debug):
        debug = debug or _empty_ocr_debug("missing", "这一帧没有识别明细。", 0.2)
        self.detail.setText(_ocr_debug_text(debug))
        panels = [("彩色上半身", debug.get("crop"), "没有彩色上半身")]
        for step in debug.get("steps") or []:
            panels.append((
                f"送入识别 · {step.get('label', '')}",
                step.get("image"),
                "没有送入识别的图像",
            ))
        for idx, (caption, image) in enumerate(self._slots):
            if idx < len(panels):
                title, frame, empty = panels[idx]
                caption.setText(title)
                caption.setVisible(True)
                image.setVisible(True)
                self._set_crop(image, frame, empty)
            else:
                caption.setVisible(False)
                image.setVisible(False)

    def _set_crop(self, label, frame_bgr, empty):
        shown = _blow_up_crop(frame_bgr)
        if shown is None:
            label.setPixmap(QPixmap())
            label.setText(empty)
            label.setMinimumSize(160, 40)
            label.setMaximumSize(16777215, 16777215)
            return
        pix = _bgr_to_pixmap(shown)
        label.setText("")
        label.setPixmap(pix)
        label.setFixedSize(pix.size())

    def dismiss(self):
        self.hide()

    def closeEvent(self, event):
        if getattr(self, "_quitting", False):
            event.accept()
            return
        self.user_closed = True
        self.hide()
        event.ignore()


class DisplayCanvas(QWidget):
    """Aspect-fit image display; click maps back to original image pixels."""

    clicked_image = pyqtSignal(float, float)
    clicked_empty = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._orig_w = 0
        self._orig_h = 0
        self._scale = 1.0
        self._off_x = 0.0
        self._off_y = 0.0
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

    def set_image(self, frame_bgr):
        if frame_bgr is None:
            return
        h, w = frame_bgr.shape[:2]
        self._pixmap = _bgr_to_pixmap(frame_bgr)
        self._orig_w, self._orig_h = w, h
        self.update()

    def _recompute_fit(self):
        if self._pixmap is None:
            return
        s = min(self.width() / self._orig_w, self.height() / self._orig_h)
        self._scale = s
        self._off_x = (self.width() - self._orig_w * s) / 2.0
        self._off_y = (self.height() - self._orig_h * s) / 2.0

    def widget_to_image(self, pos):
        self._recompute_fit()
        if self._pixmap is None or self._scale <= 0:
            return None
        x = (pos.x() - self._off_x) / self._scale
        y = (pos.y() - self._off_y) / self._scale
        if 0 <= x < self._orig_w and 0 <= y < self._orig_h:
            return float(x), float(y)
        return None

    def clear(self):
        self._pixmap = None
        self._orig_w = self._orig_h = 0
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for child in self.findChildren(CloseUpOverlay):
            child.reposition()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        pt = self.widget_to_image(event.pos())
        if pt is None:
            self.clicked_empty.emit()
        else:
            self.clicked_image.emit(pt[0], pt[1])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor(28, 30, 34))
        if self._pixmap is None:
            painter.setPen(QColor(150, 154, 160))
            painter.drawText(self.rect(), Qt.AlignCenter, "无图像")
            return
        self._recompute_fit()
        target = QRect(int(self._off_x), int(self._off_y),
                       int(self._orig_w * self._scale),
                       int(self._orig_h * self._scale))
        painter.drawPixmap(target, self._pixmap, self._pixmap.rect())


class InferenceWindow(QMainWindow):
    def __init__(self, config, scale=0.1, device="cpu", min_confidence=0.0,
                 detector="homography", calibration="", model="", api_key="",
                 settings=None):
        super().__init__()
        self.settings = dict(settings or SETTINGS)
        self.config = config
        self.scale = scale
        self.min_confidence = min_confidence
        self.detector_obj = None
        self.player_obj = None
        self.player_on = False
        self.number_obj = None
        self.number_on = False
        self.ocr_detail_on = False
        self._last_frame = None
        self._last_kpts = None
        self._last_target = None
        self._last_players = None
        self._last_annotated = None
        self._selected_xyxy = None
        self._selected_item = None
        self.cap = None
        self.timer = None
        self.paused = False
        self._fps = 25.0
        self._frame_count = 0
        self._frame_i = 0
        self._scrubbing = False
        self._frame_consumed = True
        self._infer_epoch = 0
        self._pending_frame = None
        self._reader_stop = None
        self._reader_thread = None
        self._reader_eof = False
        self._latest = None
        self._latest_lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._seek_req = None
        self._shown_i = -1
        self._submitted_i = -1
        self._infer_thread = None
        self._infer_result = None
        self._first_infer_done = False
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_infer)
        self._build_ui()
        self._set_method(detector)
        idx = self.combo_device.findData(device)
        if idx >= 0:
            self.combo_device.setCurrentIndex(idx)
        self.edit_model.setText(calibration or model)
        if not self.edit_model.text().strip() and detector == "yolo":
            kpt = _default_keypoint_model()
            if kpt:
                self.edit_model.setText(kpt)
        self.edit_apikey.setText(api_key or os.environ.get("ROBOFLOW_API_KEY", ""))
        self._fit_to_screen()

    # -- UI ----------------------------------------------------------------- #
    def _build_ui(self):
        self.setWindowTitle("球场俯视图推理 (标定 / 模型)")

        self.canvas_src = DisplayCanvas()
        self.canvas_map = DisplayCanvas()
        self.closeup = CloseUpOverlay(self.canvas_src)
        self.ocr_window = JerseyOcrWindow()
        self.closeup.dismissed.connect(self._clear_closeup)
        self.canvas_src.clicked_image.connect(self._on_src_click)
        self.canvas_src.clicked_empty.connect(self._clear_closeup)

        left = self._panel("原图 · 点击球员看特写", self.canvas_src)
        right = self._panel("俯视图 · 球场线叠加", self.canvas_map)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([780, 560])

        # --- detector row ---
        self.combo_method = QComboBox()
        for label, kind in _METHODS:
            self.combo_method.addItem(label, kind)
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)

        self.combo_device = QComboBox()
        self.combo_device.addItem("GPU (cuda)", "cuda")
        self.combo_device.addItem("CPU", "cpu")

        self.lbl_model = QLabel("标定文件")
        self.edit_model = QLineEdit()
        self.edit_model.setMinimumWidth(220)
        self.btn_browse = QPushButton("浏览…")
        self.btn_browse.clicked.connect(self._browse)

        self.lbl_apikey = QLabel("API key")
        self.edit_apikey = QLineEdit()
        self.edit_apikey.setPlaceholderText("Roboflow API key(可选)")
        self.edit_apikey.setMinimumWidth(140)

        det_row = QHBoxLayout()
        det_row.addWidget(QLabel("方法:"))
        det_row.addWidget(self.combo_method)
        det_row.addWidget(self.lbl_model)
        det_row.addWidget(self.edit_model)
        det_row.addWidget(self.btn_browse)
        det_row.addWidget(self.lbl_apikey)
        det_row.addWidget(self.edit_apikey)
        det_row.addWidget(QLabel("设备:"))
        det_row.addWidget(self.combo_device)
        det_row.addStretch(1)

        # --- source row ---
        self.btn_file = QPushButton("打开文件…")
        self.btn_file.clicked.connect(self._open_file)
        self.btn_cam = QPushButton("打开相机")
        self.btn_cam.clicked.connect(self._open_camera)
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_pause.setEnabled(False)

        self.edit_rtsp = QLineEdit()
        self.edit_rtsp.setPlaceholderText("RTSP / 视频 URL")
        self.btn_rtsp = QPushButton("连接")
        self.btn_rtsp.clicked.connect(self._open_rtsp)

        src_row = QHBoxLayout()
        src_row.addWidget(self.btn_file)
        src_row.addWidget(self.btn_cam)
        src_row.addWidget(self.edit_rtsp)
        src_row.addWidget(self.btn_rtsp)
        src_row.addWidget(self.btn_pause)
        src_row.addStretch(1)

        self.edit_player = QLineEdit()
        self.edit_player.setMinimumWidth(220)
        self.edit_player.setPlaceholderText("球员检测 .pt (默认 weights/players.pt)")
        found = _default_player_model()
        if found:
            self.edit_player.setText(found)
        self.btn_browse_player = QPushButton("浏览球员权重…")
        self.btn_browse_player.clicked.connect(self._browse_player)
        self.btn_player = QPushButton("球员推理")
        self.btn_player.setCheckable(True)
        self.btn_player.clicked.connect(self._toggle_player)
        self.btn_player.setEnabled(False)
        self.btn_number = QPushButton("号码识别")
        self.btn_number.setCheckable(True)
        self.btn_number.clicked.connect(self._toggle_number)
        self.btn_number.setEnabled(False)
        self.btn_number.setToolTip("对已检出的球员/门将裁胸前区域做球衣数字 OCR")
        self.combo_ocr_engine = QComboBox()
        self.combo_ocr_engine.addItem("EasyOCR", "easyocr")
        self.combo_ocr_engine.addItem("PaddleOCR 数字", "paddle")
        engine = str(self.settings.get("jersey_ocr_engine", "easyocr"))
        engine_idx = self.combo_ocr_engine.findData(engine)
        if engine_idx >= 0:
            self.combo_ocr_engine.setCurrentIndex(engine_idx)
        self.combo_ocr_engine.setToolTip(
            "EasyOCR 是通用英文模型。PaddleOCR 数字使用英文识别模型 "
            "en_PP-OCRv4，对数字更稳。首次使用会把权重下载到 weights/paddle_ocr。"
        )
        self.combo_ocr_engine.currentIndexChanged.connect(self._on_ocr_engine)
        self.btn_ocr_detail = QPushButton("识别过程")
        self.btn_ocr_detail.setCheckable(True)
        self.btn_ocr_detail.clicked.connect(self._toggle_ocr_detail)
        self.btn_ocr_detail.setToolTip(
            "打开后，点选球员会同时显示实际送进数字识别的图像和逐条结果")

        ply_row = QHBoxLayout()
        ply_row.addWidget(QLabel("球员模型:"))
        ply_row.addWidget(self.edit_player)
        ply_row.addWidget(self.btn_browse_player)
        ply_row.addWidget(self.btn_player)
        ply_row.addWidget(self.btn_number)
        ply_row.addWidget(self.combo_ocr_engine)
        ply_row.addWidget(self.btn_ocr_detail)
        ply_row.addStretch(1)

        self.status = QLabel("先选方法并配置,再打开输入源。")
        self.status.setStyleSheet("color: #c8ccd2; padding: 2px 4px;")

        central = QWidget()
        central.setStyleSheet("""
            QWidget { background: #1b1d21; color: #e8e8ea; }
            QLineEdit, QComboBox {
                background: #2a2d34; color: #e8e8ea;
                border: 1px solid #4a4e57; border-radius: 3px;
                padding: 3px 6px; min-height: 22px;
            }
            QPushButton {
                background: #343842; color: #e8e8ea;
                border: 1px solid #4a4e57; border-radius: 3px;
                padding: 4px 10px;
            }
            QPushButton:hover { background: #3e4350; }
            QPushButton:checked { background: #2f5a40; }
            QPushButton:disabled { color: #777; }
            QSplitter::handle { background: #3a3d44; }
            QSlider::groove:horizontal {
                height: 6px; background: #3a3d44; border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #3d7a57; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #d0d4dc; width: 14px; margin: -5px 0;
                border-radius: 7px;
            }
            QSlider:disabled { }
            QSlider::handle:horizontal:disabled { background: #5a5e66; }
        """)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setMinimumWidth(128)
        self.lbl_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        seek_row = QHBoxLayout()
        seek_row.addWidget(self.slider, 1)
        seek_row.addWidget(self.lbl_time)

        layout.addLayout(det_row)
        layout.addLayout(src_row)
        layout.addLayout(ply_row)
        layout.addWidget(splitter, 1)
        layout.addLayout(seek_row)
        layout.addWidget(self.status)
        self.setCentralWidget(central)

        self._on_method_changed()

    def _fit_to_screen(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 760)
            return
        avail = screen.availableGeometry()
        self.resize(int(avail.width() * 0.95), int(avail.height() * 0.90))
        self.move(avail.topLeft())

    @staticmethod
    def _panel(title, canvas):
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        lbl = QLabel(title)
        lbl.setStyleSheet(
            "font-weight: 600; color: #e8e8ea; padding: 2px 4px;")
        v.addWidget(lbl)
        canvas.setStyleSheet(
            "border: 1px solid #3a3d44; border-radius: 4px;")
        v.addWidget(canvas, 1)
        return box

    # -- detector ----------------------------------------------------------- #
    def _set_method(self, kind):
        idx = self.combo_method.findData(kind)
        if idx >= 0:
            self.combo_method.setCurrentIndex(idx)

    def _on_method_changed(self, *_):
        method = self.combo_method.currentData()
        if method == "homography":
            self.lbl_model.setText("标定文件")
            self.edit_model.setPlaceholderText("标定 JSON 路径(必填)")
            self.btn_browse.setEnabled(True)
            self.edit_apikey.setEnabled(False)
        elif method == "yolo":
            self.lbl_model.setText("模型权重")
            self.edit_model.setPlaceholderText(
                "关键点 .pt (默认 runs/pose/train/weights/best.pt)")
            self.btn_browse.setEnabled(True)
            self.edit_apikey.setEnabled(False)
            if not self.edit_model.text().strip():
                kpt = _default_keypoint_model()
                if kpt:
                    self.edit_model.setText(kpt)
        else:
            self.lbl_model.setText("model_id")
            self.edit_model.setPlaceholderText("model_id(留空用公开模型)")
            self.btn_browse.setEnabled(False)
            self.edit_apikey.setEnabled(True)

    def _build_detector(self):
        method = self.combo_method.currentData()
        value = self.edit_model.text().strip()
        api_key = self.edit_apikey.text().strip() \
            or os.environ.get("ROBOFLOW_API_KEY", "")
        if method == "homography" and not value:
            self.status.setText("请先选择标定 JSON 文件。")
            return False
        if method == "yolo" and not value:
            self.status.setText("请先选择模型 .pt 文件。")
            return False
        if method in ("inference", "roboflow") and not api_key:
            self.status.setText("请填写 Roboflow API key。")
            return False
        self._close_detector()
        try:
            if method == "yolo":
                # Run torch/CUDA in a separate process so it can't crash Qt.
                self.detector_obj = SubprocessYoloDetector(
                    model_path=value,
                    keypoint_names=self.config.keypoint_labels,
                    keypoint_vertices=self.config.keypoint_vertices,
                    device=self.combo_device.currentData(),
                    min_confidence=self.min_confidence,
                )
            else:
                self.detector_obj = create_detector(
                    method,
                    model_path="",
                    homography_file=value if method == "homography" else "",
                    api_key=api_key,
                    model_id=(value or _DEFAULT_MODEL) if method in ("inference", "roboflow") else "",
                    config=self.config,
                    device=self.combo_device.currentData(),
                    min_confidence=self.min_confidence,
                )
            return True
        except Exception as exc:  # noqa: BLE001 — surface any init error to the user
            self.status.setText(f"检测器初始化失败: {exc}")
            return False

    def _close_detector(self):
        old = self.detector_obj
        self.detector_obj = None
        if isinstance(old, SubprocessYoloDetector):
            old.close()

    def _browse(self):
        method = self.combo_method.currentData()
        if method == "homography":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择标定 JSON", "", "JSON (*.json);;所有文件 (*)")
        elif method == "yolo":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择模型权重", "", "PyTorch (*.pt);;所有文件 (*)")
        else:
            return
        if path:
            self.edit_model.setText(path)

    # -- source ------------------------------------------------------------- #
    def _start_source(self, src):
        if not self._build_detector():
            return
        self._stop_source()
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.status.setText(f"无法打开输入源 {src!r}")
            return
        self.cap = cap
        self.paused = False
        self._first_infer_done = False
        self.btn_pause.setEnabled(True)
        self.btn_player.setEnabled(True)
        self.btn_number.setEnabled(True)

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        is_image = (isinstance(src, str)
                    and os.path.splitext(src)[1].lower() in _IMAGE_EXTS) \
            or frame_count == 1
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self._fps < 1.0:
            self._fps = 25.0
        self._frame_count = int(frame_count) if frame_count and frame_count > 1 else 0
        self._frame_i = 0
        self._scrubbing = False
        self._pending_frame = None
        seekable = self._frame_count > 1 and not isinstance(src, int)
        self.slider.blockSignals(True)
        if seekable:
            self.slider.setEnabled(True)
            self.slider.setRange(0, self._frame_count - 1)
            self.slider.setValue(0)
        else:
            self.slider.setEnabled(False)
            self.slider.setRange(0, 0)
            self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._update_time_label(0)
        self._shown_i = -1
        self._submitted_i = -1
        self._reader_eof = False
        self._latest = None
        self._seek_req = None

        # Set the pause state FIRST so its status text doesn't clobber the
        # "loading model" message we show right after.
        if is_image:
            ok, frame = cap.read()
            if not ok:
                self.status.setText("读不到帧。")
                return
            self._set_paused(True)
            self.canvas_src.set_image(frame)
            self._shown_i = 0
        else:
            self._reader_stop = threading.Event()
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._present_live)
            self.timer.start(15)
            self._set_paused(False)

        if getattr(self, "_auto_players", False) and self._start_player():
            self.btn_player.setChecked(True)
            self.btn_player.setText("关闭球员推理")
        self.status.setText("正在加载模型 / 推理(首次需 10~60 秒,请稍候)…")
        QApplication.processEvents()
        if is_image:
            self._process_frame(frame, 0)

    def _stop_source(self):
        if self._reader_stop is not None:
            self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        self._reader_stop = None
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
        with self._cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        self._infer_epoch += 1
        self._frame_count = 0
        self._frame_i = 0
        self._scrubbing = False
        self._pending_frame = None
        if hasattr(self, "slider"):
            self.slider.setEnabled(False)
            self.slider.setRange(0, 0)
            self.slider.setValue(0)
            self._update_time_label(0)

    def _fmt_time(self, frame_i):
        fps = self._fps if self._fps > 0 else 25.0
        def one(idx):
            sec = max(0, int(idx / fps))
            h, rem = divmod(sec, 3600)
            m, s = divmod(rem, 60)
            if h:
                return f"{h}:{m:02d}:{s:02d}"
            return f"{m:02d}:{s:02d}"
        total = self._frame_count if self._frame_count > 0 else 0
        return f"{one(frame_i)} / {one(max(0, total - 1))}"

    def _update_time_label(self, frame_i):
        self.lbl_time.setText(self._fmt_time(frame_i))

    def _sync_slider(self):
        if not self.slider.isEnabled() or self._scrubbing:
            return
        self.slider.blockSignals(True)
        self.slider.setValue(min(self._frame_i, self.slider.maximum()))
        self.slider.blockSignals(False)
        self._update_time_label(self._frame_i)

    def _on_slider_pressed(self):
        self._scrubbing = True

    def _on_slider_moved(self, value):
        self._update_time_label(int(value))
        self._queue_seek(int(value))

    def _on_slider_released(self):
        if not self.slider.isEnabled() or self.cap is None:
            self._scrubbing = False
            return
        self._scrubbing = False
        self._queue_seek(int(self.slider.value()))

    def _queue_seek(self, index):
        if self.cap is None or self._frame_count <= 1:
            return
        index = max(0, min(int(index), self._frame_count - 1))
        self._infer_epoch += 1
        self._submitted_i = -1
        with self._latest_lock:
            self._seek_req = index

    def _publish_frame(self, index, frame):
        frame = np.ascontiguousarray(frame).copy()
        with self._latest_lock:
            self._latest = (index, frame)

    def _reader_loop(self):
        """Decode on its own thread so the window keeps painting at video speed."""
        period = 1.0 / (self._fps if self._fps > 0 else 25.0)
        next_t = time.perf_counter()
        while self._reader_stop is not None and not self._reader_stop.is_set():
            with self._latest_lock:
                seek = self._seek_req
                self._seek_req = None
            if seek is not None:
                with self._cap_lock:
                    if self.cap is None:
                        return
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
                    ok, frame = self.cap.read()
                    pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if ok else 0
                if ok:
                    shown = pos - 1 if pos > 0 else seek
                    self._publish_frame(shown, frame)
                    self._frame_consumed = False
                next_t = time.perf_counter()
                continue
            if self.paused or self._scrubbing or not self._frame_consumed:
                if self._reader_stop.wait(0.01):
                    return
                next_t = time.perf_counter()
                continue
            with self._cap_lock:
                if self.cap is None:
                    return
                ok, frame = self.cap.read()
                pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if ok else 0
            if not ok:
                self._reader_eof = True
                return
            if pos > 0:
                index = pos - 1
            else:
                with self._latest_lock:
                    index = 0 if self._latest is None else self._latest[0] + 1
            self._publish_frame(index, frame)
            self._frame_consumed = False
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                if self._reader_stop.wait(delay):
                    return
            else:
                next_t = time.perf_counter()

    def _present_live(self):
        """Show the newest decoded frame. Inference runs beside this, not in front of it."""
        with self._latest_lock:
            item = self._latest
        if item is not None:
            index, frame = item
            if index != self._shown_i:
                self._shown_i = index
                self._frame_i = index
                if self._last_annotated is None or self._scrubbing:
                    self.canvas_src.set_image(frame)
            if self._scrubbing:
                return
            infer_busy = (self._infer_thread is not None
                          and self._infer_thread.is_alive())
            if not infer_busy and self._submitted_i != self._shown_i:
                self._process_frame(frame, self._shown_i)
        if self._reader_eof and item is not None and item[0] == self._shown_i:
            infer_busy = (self._infer_thread is not None
                          and self._infer_thread.is_alive())
            if not infer_busy and self.timer is not None:
                self.timer.stop()
                self.status.setText("视频流结束。")

    def _process_frame(self, frame, frame_i=None):
        """Kick off one inference in a background thread. The actual model run
        (which may block while the model loads on first use) happens off the Qt
        main thread; ``_poll_infer`` updates the canvases when it finishes."""
        if self.detector_obj is None:
            return
        if frame_i is None:
            frame_i = self._frame_i
        if self._infer_thread is not None and self._infer_thread.is_alive():
            return
        # cap.read() reuses its buffer. Keep a private copy so the player
        # crop cannot be taken from a later frame while the box stays put.
        frame = np.ascontiguousarray(frame).copy()
        epoch = self._infer_epoch
        self._submitted_i = frame_i
        self._infer_result = None
        self._infer_thread = threading.Thread(
            target=self._run_infer, args=(frame, epoch, frame_i), daemon=True)
        self._infer_thread.start()
        if not self._poll_timer.isActive():
            self._poll_timer.start(15)

    def _detect_parallel(self, frame):
        """Keypoints and player boxes are separate processes. Run them together."""
        boxes = {}
        errors = []

        def run_keypoints():
            try:
                boxes["k"] = self.detector_obj.detect(frame)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def run_players():
            try:
                boxes["p"] = self.player_obj.detect(frame)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        workers = [threading.Thread(target=run_keypoints, daemon=True)]
        if self.player_on and self.player_obj is not None:
            workers.append(threading.Thread(target=run_players, daemon=True))
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        if errors:
            raise errors[0]
        keypoints, target = boxes["k"]
        return keypoints, target, boxes.get("p")

    def _run_infer(self, frame, epoch, frame_i):
        try:
            keypoints, target, players = self._detect_parallel(frame)
            if players is not None:
                if self.number_on and self.number_obj is not None and players:
                    players = self.number_obj.recognize(frame, players)
            annotated, topdown = _render_views(
                frame, keypoints, target, self.config, self.scale, players=players)
            self._infer_result = (
                epoch, frame_i, frame, annotated, topdown,
                keypoints, target, players, None)
        except Exception as exc:  # noqa: BLE001 — e.g. model load / missing api key
            print(f"[infer] error: {exc}", file=sys.stderr, flush=True)
            self._infer_result = (
                epoch, frame_i, frame, None, None, None, None, None, exc)

    def _poll_infer(self):
        if self._infer_thread is None or self._infer_thread.is_alive():
            return
        self._poll_timer.stop()
        self._infer_thread = None
        (epoch, frame_i, frame, annotated, topdown,
         keypoints, target, players, err) = self._infer_result
        if epoch != self._infer_epoch:
            self._submitted_i = -1
            return
        if err is not None:
            self.status.setText(f"推理失败: {err}")
            if frame_i == self._shown_i:
                self.canvas_src.set_image(frame)
            self.canvas_map.clear()
            self._clear_closeup()
            self._frame_consumed = True
            return

        self._last_frame = np.ascontiguousarray(frame)
        self._last_kpts = keypoints
        self._last_target = target
        self._last_players = players
        self._last_annotated = annotated
        self._rematch_selection()
        self._frame_i = frame_i
        self._refresh_src_view()
        self.canvas_map.set_image(topdown)
        self._sync_slider()
        self._frame_consumed = True

        n = len(keypoints)
        msg = f"关键点 {n}"
        if n < 4:
            msg += "（不足 4 个，无法求单应）"
        if players is not None:
            msg += f" · 检测 {len(players)}"
            n_num = sum(1 for b in players if _split_box(b)[3])
            if self.number_on:
                msg += f" · 号码 {n_num}"
            msg += " · 点击球员看特写"
        if not self._first_infer_done:
            self._first_infer_done = True
            self.status.setText(f"推理完成 · {msg}")
        elif not self.paused:
            self.status.setText(f"直播中 · {msg}")
        else:
            self.status.setText(f"已暂停 · {msg}")

    # -- buttons ------------------------------------------------------------ #
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片或视频", "",
            "媒体文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.mp4 *.avi *.mov *.mkv);;所有文件 (*)")
        if path:
            self._start_source(path)

    def _open_camera(self):
        self._start_source(0)

    def _open_rtsp(self):
        url = self.edit_rtsp.text().strip()
        if url:
            self._start_source(url)

    def _toggle_pause(self):
        self._set_paused(not self.paused)

    def _set_paused(self, paused):
        self.paused = paused
        self.btn_pause.setText("继续" if paused else "暂停")
        self.status.setText("已暂停。" if paused else "直播中。")

    def _browse_player(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择球员检测权重", "", "PyTorch (*.pt);;所有文件 (*)")
        if path:
            self.edit_player.setText(path)
            if self.player_on:
                self._close_player()
                if self._start_player():
                    self._rerun_last()

    def _toggle_player(self):
        want = self.btn_player.isChecked()
        if want:
            if not self._start_player():
                self.btn_player.setChecked(False)
                return
            self.btn_player.setText("关闭球员推理")
            self.status.setText("球员模型加载中(首次需十几秒)…")
            QApplication.processEvents()
            self._rerun_last()
        else:
            self._close_number()
            self.btn_number.setChecked(False)
            self.btn_number.setText("号码识别")
            self._close_player()
            self.btn_player.setText("球员推理")
            self._clear_closeup()
            self._rerun_last()

    def _start_player(self):
        path = self.edit_player.text().strip() or _default_player_model()
        if not path:
            self.status.setText("请先选择球员权重（默认 weights/players.pt）。")
            return False
        if not os.path.isfile(path):
            self.status.setText(f"找不到球员权重: {path}")
            return False
        self._close_player()
        try:
            self.player_obj = SubprocessPlayerDetector(
                path, self.combo_device.currentData(), settings=self.settings)
            self.player_on = True
            self.edit_player.setText(path)
            return True
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"球员检测器启动失败: {exc}")
            self.player_on = False
            return False

    def _close_player(self):
        self.player_on = False
        old = self.player_obj
        self.player_obj = None
        if old is not None:
            old.close()

    def _toggle_number(self):
        want = self.btn_number.isChecked()
        if want:
            if not self.player_on:
                if not self._start_player():
                    self.btn_number.setChecked(False)
                    return
                self.btn_player.setChecked(True)
                self.btn_player.setText("关闭球员推理")
            if not self._start_number():
                self.btn_number.setChecked(False)
                return
            self.btn_number.setText("关闭号码识别")
            self.status.setText(
                f"{self.combo_ocr_engine.currentText()} 加载中"
                "（首次会下载模型，请稍候）…"
            )
            QApplication.processEvents()
            self._rerun_last()
        else:
            self._close_number()
            self.btn_number.setText("号码识别")
            self._rerun_last()

    def _toggle_ocr_detail(self):
        self.ocr_detail_on = self.btn_ocr_detail.isChecked()
        if self.ocr_detail_on:
            self.btn_ocr_detail.setText("关闭识别过程")
            self.ocr_window.user_closed = False
            self._refresh_src_view()
        else:
            self.btn_ocr_detail.setText("识别过程")
            self.ocr_window.dismiss()
            self._refresh_src_view()

    def _on_ocr_engine(self):
        self.settings["jersey_ocr_engine"] = self.combo_ocr_engine.currentData()
        if not self.number_on:
            return
        label = self.combo_ocr_engine.currentText()
        self.status.setText(f"正在切换到 {label}…")
        QApplication.processEvents()
        if self._start_number():
            self._rerun_last()
        else:
            self.btn_number.setChecked(False)
            self.btn_number.setText("号码识别")

    def _start_number(self):
        self._close_number()
        self.settings["jersey_ocr_engine"] = self.combo_ocr_engine.currentData()
        try:
            self.number_obj = SubprocessJerseyOcr(
                self.combo_device.currentData(), settings=self.settings)
            self.number_on = True
            return True
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"号码识别启动失败: {exc}")
            self.number_on = False
            return False

    def _close_number(self):
        self.number_on = False
        old = self.number_obj
        self.number_obj = None
        if old is not None:
            old.close()

    def _rerun_last(self):
        if self._last_frame is not None and self.detector_obj is not None:
            self._submitted_i = -1
            self._process_frame(self._last_frame, self._shown_i)

    def _hit_player(self, x, y):
        players = self._last_players
        if not players:
            return None
        best_i, best_area = None, 1e18
        for i, item in enumerate(players):
            _, _, xyxy, _ = _split_box(item)
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = max(1.0, (x2 - x1) * (y2 - y1))
                if area < best_area:
                    best_i, best_area = i, area
        return best_i

    def _on_src_click(self, x, y):
        idx = self._hit_player(x, y)
        if idx is None:
            self._clear_closeup()
            return
        item = self._last_players[idx]
        _, _, xyxy, _ = _split_box(item)
        self.ocr_window.user_closed = False
        self._selected_item = item
        self._selected_xyxy = np.asarray(xyxy, dtype=np.float32).copy()
        self._refresh_src_view()

    def _rematch_selection(self):
        if self._selected_xyxy is None or not self._last_players:
            if not self._last_players:
                self._selected_xyxy = None
                self._selected_item = None
            return
        best_item, best_iou = None, 0.0
        for item in self._last_players:
            _, _, xyxy, _ = _split_box(item)
            iou, _ = _xyxy_overlap(self._selected_xyxy, xyxy)
            if iou > best_iou:
                best_item, best_iou = item, iou
        if best_item is not None and best_iou >= float(
                self.settings.get("closeup_track_iou", 0.3)):
            _, _, xyxy, _ = _split_box(best_item)
            self._selected_item = best_item
            self._selected_xyxy = np.asarray(xyxy, dtype=np.float32).copy()
        else:
            self._selected_xyxy = None
            self._selected_item = None

    def _refresh_src_view(self):
        if self._last_annotated is None:
            return
        view = self._last_annotated
        item = self._selected_item
        if item is not None and self._last_frame is not None:
            view = self._last_annotated.copy()
            _, _, xyxy, _ = _split_box(item)
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 255), 3, cv2.LINE_AA)
            if self.ocr_detail_on:
                torso = _torso_bounds(
                    self._last_frame.shape, xyxy, **_torso_kwargs(self.settings))
                if torso is not None:
                    tx1, ty1, tx2, ty2 = torso
                    cv2.rectangle(view, (tx1, ty1), (tx2, ty2),
                                  (0, 255, 255), 2, cv2.LINE_AA)
            crop = _padded_crop(
                self._last_frame, xyxy,
                pad_ratio=float(self.settings.get("closeup_pad_ratio", 0.18)))
            if crop is not None:
                max_w = max(160, int(self.canvas_src.width() * 0.34))
                max_h = max(200, int(self.canvas_src.height() * 0.58))
                zoomed = _annotate_closeup(
                    _zoom_crop(
                        crop, max_w, max_h,
                        max_scale=float(self.settings.get("closeup_max_scale", 5.0))),
                    item)
                self.closeup.show_player(
                    zoomed, _player_info_lines(
                        item, number_on=self.number_on,
                        show_process=self.ocr_detail_on))
            else:
                self.closeup.hide()
            self._sync_ocr_window(item)
        else:
            self.closeup.hide()
            self._sync_ocr_window(None)
        self.canvas_src.set_image(view)

    def _sync_ocr_window(self, item):
        if not self.ocr_detail_on:
            self.ocr_window.dismiss()
            return
        if item is None or self._last_frame is None or self.ocr_window.user_closed:
            if item is None or self._last_frame is None:
                self.ocr_window.dismiss()
            return
        cls, _, xyxy, _ = _split_box(item)
        debug = _ocr_debug(item)
        if int(cls) not in _PERSON_CLASSES:
            debug = debug or _empty_ocr_debug(
                "not_person", "球体不做号码识别",
                float(self.settings.get("jersey_ocr_min_conf", 0.35)))
        elif debug is None:
            debug = _preview_jersey_crop(self._last_frame, xyxy, self.settings)
        else:
            raw, img = _jersey_views(self._last_frame, xyxy, self.settings)
            if raw is not None:
                debug = dict(debug)
                debug["raw_crop"] = raw
                debug["crop"] = img
        if not self.ocr_window.isVisible():
            geo = self.frameGeometry()
            self.ocr_window.move(geo.right() + 12, geo.top() + 40)
        self.ocr_window.present(debug)

    def _clear_closeup(self):
        self._selected_xyxy = None
        self._selected_item = None
        self.closeup.hide()
        self.ocr_window.dismiss()
        if self._last_annotated is not None:
            self.canvas_src.set_image(self._last_annotated)

    def closeEvent(self, event):
        self._stop_source()
        self._close_detector()
        self._close_number()
        self._close_player()
        self.ocr_window.force_close()
        super().closeEvent(event)


def _run_gui(args):
    config = SoccerPitchConfiguration()
    app = QApplication(sys.argv)
    _apply_cjk_font(app)
    win = InferenceWindow(
        config,
        scale=args.scale,
        device=args.device,
        min_confidence=args.min_confidence,
        detector=args.detector,
        calibration=args.calibration,
        model=args.model,
        api_key=args.api_key,
        settings=SETTINGS,
    )
    win.show()
    initial = args.video or args.source
    if initial:
        win._auto_players = True
        win._start_source(initial)
    sys.exit(app.exec_())


def main() -> None:
    global SETTINGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config_path(),
                        help="配置 YAML (默认 config/court_detector.yaml)。")
    parser.add_argument("--cli", action="store_true",
                        help="批量模式:处理图片/视频并写文件(默认打开 Qt 窗口)。")
    parser.add_argument("--detector", choices=["homography", "yolo", "inference", "roboflow"],
                        default=None,
                        help="检测方法(默认读 YAML detector_type；或 homography)。")
    parser.add_argument("--calibration", default="",
                        help="标定 JSON(detector=homography)。")
    parser.add_argument("--model", default="",
                        help="detector=yolo 时为本地 .pt "
                             "(默认 YAML model_path);"
                             "inference/roboflow 时为 model_id。")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""),
                        help="Roboflow API key(detector=inference/roboflow)。")
    parser.add_argument("--device", default=None,
                        help="推理设备(默认读 YAML device)。")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="关键点最低置信度(默认读 YAML min_confidence)。")
    parser.add_argument("source", nargs="?", default="",
                        help="输入图片/视频路径(CLI 模式;GUI 模式下作为初始源)。")
    parser.add_argument("--video", default="",
                        help="输入视频路径(CLI 模式;GUI 模式下作为初始源)。")
    parser.add_argument("--out-dir", default="",
                        help="输出目录(CLI 模式;默认在输入旁)。")
    parser.add_argument("--scale", type=float, default=None,
                        help="俯视图每厘米像素数(默认读 YAML scale)。")
    parser.add_argument("--out", default="",
                        help="显式输出路径(CLI 模式,单张图)。")
    args = parser.parse_args()

    SETTINGS = load_settings(args.config)
    if args.detector is None:
        # 脚本默认固定机位; ROS 节点仍读 YAML 里的 detector_type。
        args.detector = "homography"
    if args.device is None:
        args.device = SETTINGS.get("device") or "cpu"
    if args.min_confidence is None:
        args.min_confidence = float(SETTINGS.get("min_confidence", 0.0))
    if args.scale is None:
        args.scale = float(SETTINGS.get("scale", 0.05))
    if not args.model and args.detector == "yolo":
        args.model = resolve_repo_path(SETTINGS.get("model_path", ""))
    if not args.calibration and args.detector == "homography":
        args.calibration = resolve_repo_path(SETTINGS.get("homography_file", ""))
    if not args.api_key:
        args.api_key = SETTINGS.get("api_key") or os.environ.get(
            "ROBOFLOW_API_KEY", "")

    if args.cli:
        _run_cli(args)
    else:
        _run_gui(args)


if __name__ == "__main__":
    main()
