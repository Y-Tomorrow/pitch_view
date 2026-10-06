#!/usr/bin/env python3
"""
Detect the pitch (fixed homography OR a keypoint model) and show the top-down
2D pitch view — no ROS2 required.

By default this opens a Qt window (no arguments needed). Pick the detection
method, its config, and the input source from the UI:

  * homography (default): fixed camera, needs a saved 4-point calibration JSON.
  * yolo: a local YOLO keypoint model (``models/pitch_keypoints.pt``).
  * inference / roboflow: a Roboflow keypoint model (API key).

Weights live under ``models/`` (train scripts publish there). For batch use,
pass ``--cli`` to process an image/video and write files to disk.

Usage:
    # Qt GUI (default)
    python3 scripts/run_homography.py

    # batch: fixed homography (single image)
    python3 scripts/run_homography.py --cli --calibration data/1_calibration.json data/1.png

    # batch: model inference (local .pt)
    python3 scripts/run_homography.py --cli --detector yolo \\
        --model models/pitch_keypoints.pt data/1.png
"""
import argparse
import multiprocessing as mp
import os
import sys
import threading

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
    models_dir,
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
    QLineEdit, QMainWindow, QPushButton, QSplitter, QVBoxLayout, QWidget,
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
        print("[yolo-worker] model loaded, ready.", file=sys.stderr, flush=True)
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
    """Packaged keypoint weights from YAML ``model_path`` or models/."""
    path = resolve_repo_path(SETTINGS.get("model_path", "")) or pitch_keypoints_path()
    return path if os.path.isfile(path) else ""


def _default_player_model():
    """Packaged player weights from YAML ``player_model_path`` or models/."""
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
        model = YOLO(model_path)
        if device:
            model.to(device)
        print("[player-worker] model loaded, ready.", file=sys.stderr, flush=True)
        while True:
            frame = conn.recv()
            if frame is None:
                break
            result = model(
                frame, verbose=False, conf=conf, iou=iou,
                agnostic_nms=agnostic_nms)[0]
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


def _torso_crop(frame, xyxy, x_margin=0.18, y0=0.10, y1=0.52):
    """Chest ROI where a jersey number usually sits (upper-middle of the box)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 12 or bh < 24:
        return None
    tx1 = x1 + int(bw * x_margin)
    tx2 = x2 - int(bw * x_margin)
    ty1 = y1 + int(bh * y0)
    ty2 = y1 + int(bh * y1)
    if tx2 - tx1 < 8 or ty2 - ty1 < 8:
        crop = frame[y1:y2, x1:x2]
    else:
        crop = frame[ty1:ty2, tx1:tx2]
    return crop if crop.size else None


def _prepare_ocr_image(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ch, cw = gray.shape[:2]
    scale = max(1.0, 96.0 / max(1, min(ch, cw)))
    if scale > 1.01:
        gray = cv2.resize(
            gray, (int(cw * scale), int(ch * scale)),
            interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _normalize_jersey_text(text):
    t = (text or "").strip().upper()
    t = (t.replace("O", "0").replace("I", "1").replace("L", "1")
           .replace("Z", "2").replace("S", "5").replace("B", "8"))
    digits = "".join(c for c in t if c.isdigit())
    if len(digits) in (1, 2):
        return digits
    return ""


def _best_jersey_number(ocr_items, min_conf=0.35):
    """Pick the highest-confidence 1–2 digit string from EasyOCR hits."""
    best, best_s = "", -1.0
    for item in ocr_items or []:
        if not item or len(item) < 2:
            continue
        text = item[1]
        score = float(item[2]) if len(item) > 2 else 0.0
        number = _normalize_jersey_text(text)
        if number and score >= min_conf and score > best_s:
            best, best_s = number, score
    return best


def _ocr_one_jersey(reader, frame, xyxy, settings=None):
    s = settings or SETTINGS
    crop = _torso_crop(
        frame, xyxy,
        x_margin=float(s.get("jersey_torso_x_margin", 0.18)),
        y0=float(s.get("jersey_torso_y0", 0.10)),
        y1=float(s.get("jersey_torso_y1", 0.52)),
    )
    if crop is None:
        return ""
    img = _prepare_ocr_image(crop)
    try:
        hits = reader.readtext(
            img, detail=1, paragraph=False, allowlist="0123456789")
    except TypeError:
        hits = reader.readtext(img, detail=1, paragraph=False)
    return _best_jersey_number(
        hits, min_conf=float(s.get("jersey_ocr_min_conf", 0.35)))


def _jersey_worker(device, conn, settings=None):
    """EasyOCR in a spawn-ed process (torch stays out of Qt)."""
    settings = dict(settings or SETTINGS)
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["MKL_NUM_THREADS"] = "1"
        import easyocr

        model_dir = models_dir()
        need = ("craft_mlt_25k.pth", "english_g2.pth")
        missing = [n for n in need
                   if not os.path.isfile(os.path.join(model_dir, n))]
        print(f"[jersey-ocr] model dir: {model_dir}", file=sys.stderr, flush=True)
        if missing:
            print(f"[jersey-ocr] missing {missing}; will download into models/…",
                  file=sys.stderr, flush=True)
        else:
            print("[jersey-ocr] loading EasyOCR from models/…",
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
            reader = easyocr.Reader(**kwargs)
        except TypeError:
            # Older easyocr: fewer kwargs.
            kwargs.pop("verbose", None)
            try:
                reader = easyocr.Reader(**kwargs)
            except TypeError:
                reader = easyocr.Reader(
                    ["en"], gpu=use_gpu,
                    model_storage_directory=model_dir)
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
                    number = _ocr_one_jersey(reader, frame, xyxy, settings)
                out.append((cls, conf, xyxy, number))
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
        self._parent.send((frame, boxes))
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
                "--detector yolo 需要 --model，或先训练生成 models/pitch_keypoints.pt。")
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
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _player_info_lines(item, number_on=False):
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
    return [
        f"类别  {zh} ({en})",
        f"检测置信度  {conf:.2f}",
        f"球衣号码  {num}",
        f"框  ({x1},{y1}) – ({x2},{y2})",
    ]


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
        self.edit_player.setPlaceholderText("球员检测 .pt (默认 models/players.pt)")
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

        ply_row = QHBoxLayout()
        ply_row.addWidget(QLabel("球员模型:"))
        ply_row.addWidget(self.edit_player)
        ply_row.addWidget(self.btn_browse_player)
        ply_row.addWidget(self.btn_player)
        ply_row.addWidget(self.btn_number)
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
        """)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(det_row)
        layout.addLayout(src_row)
        layout.addLayout(ply_row)
        layout.addWidget(splitter, 1)
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
            self.edit_model.setPlaceholderText("关键点 .pt (默认 models/pitch_keypoints.pt)")
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

        ok, frame = cap.read()
        if not ok:
            self.status.setText("读不到帧。")
            return

        # Set the pause state FIRST so its status text doesn't clobber the
        # "loading model" message we show right after.
        if is_image:
            self._set_paused(True)
        else:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._grab)
            self.timer.start(33)
            self._set_paused(False)

        self.status.setText("正在加载模型 / 推理(首次需 10~60 秒,请稍候)…")
        QApplication.processEvents()
        self._process_frame(frame)

    def _stop_source(self):
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _grab(self):
        if self.paused or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.timer.stop()
            self.status.setText("视频流结束。")
            return
        self._process_frame(frame)

    def _process_frame(self, frame):
        """Kick off one inference in a background thread. The actual model run
        (which may block while the model loads on first use) happens off the Qt
        main thread; ``_poll_infer`` updates the canvases when it finishes."""
        if self.detector_obj is None:
            return
        if self._infer_thread is not None and self._infer_thread.is_alive():
            return  # still busy — drop this frame rather than queueing up
        self._infer_result = None
        self._infer_thread = threading.Thread(
            target=self._run_infer, args=(frame,), daemon=True)
        self._infer_thread.start()
        if not self._poll_timer.isActive():
            self._poll_timer.start(50)

    def _run_infer(self, frame):
        try:
            keypoints, target = self.detector_obj.detect(frame)
            print(f"[infer] {len(keypoints)} visible keypoints", file=sys.stderr, flush=True)
            players = None
            if self.player_on and self.player_obj is not None:
                players = self.player_obj.detect(frame)
                print(f"[infer] {len(players)} players", file=sys.stderr, flush=True)
                if self.number_on and self.number_obj is not None and players:
                    players = self.number_obj.recognize(frame, players)
            self._infer_result = (frame, keypoints, target, players, None)
        except Exception as exc:  # noqa: BLE001 — e.g. model load / missing api key
            print(f"[infer] error: {exc}", file=sys.stderr, flush=True)
            self._infer_result = (frame, None, None, None, exc)

    def _poll_infer(self):
        if self._infer_thread is None or self._infer_thread.is_alive():
            return
        self._poll_timer.stop()
        self._infer_thread = None
        frame, keypoints, target, players, err = self._infer_result
        if err is not None:
            self.status.setText(f"推理失败: {err}")
            self.canvas_src.set_image(frame)
            self.canvas_map.clear()
            self._clear_closeup()
            return

        self._last_frame = frame
        self._last_kpts = keypoints
        self._last_target = target
        self._last_players = players
        annotated, topdown = _render_views(
            frame, keypoints, target, self.config, self.scale, players=players)
        self._last_annotated = annotated
        self._rematch_selection()
        self._refresh_src_view()
        self.canvas_map.set_image(topdown)

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
            self.status.setText("请先选择球员权重（默认 models/players.pt）。")
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
            self.status.setText("号码 OCR 加载中(首次需下载英文模型,请稍候)…")
            QApplication.processEvents()
            self._rerun_last()
        else:
            self._close_number()
            self.btn_number.setText("号码识别")
            self._rerun_last()

    def _start_number(self):
        self._close_number()
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
            self._process_frame(self._last_frame)

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
                    zoomed, _player_info_lines(item, number_on=self.number_on))
            else:
                self.closeup.hide()
        else:
            self.closeup.hide()
        self.canvas_src.set_image(view)

    def _clear_closeup(self):
        self._selected_xyxy = None
        self._selected_item = None
        self.closeup.hide()
        if self._last_annotated is not None:
            self.canvas_src.set_image(self._last_annotated)

    def closeEvent(self, event):
        self._stop_source()
        self._close_detector()
        self._close_number()
        self._close_player()
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
