#!/usr/bin/env python3
"""
Label the 32 pitch keypoints on extracted frames.

Left: images in filename order. Click to place the current point.
Right: top-down pitch with the same 0–31 numbers, for reference.

Saves YOLO keypoint labels (class ``pitch``, 32×(x, y, visibility)) next to a
small JSON file so a half-finished image can be reopened.

    python3 scripts/annotate_keypoints.py
    python3 scripts/annotate_keypoints.py --images data/pitch_dataset/images

Keys: 左键可见  拖动已有点微调  右键遮挡  N 不在画面  W 用权重预标
      Z 撤销  A/D 上一张/下一张  空格播放
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QPointF, QRect
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pitch_view.annotators.soccer import draw_pitch
from pitch_view.common.weights import pitch_keypoints_path
from pitch_view.configs import SoccerPitchConfiguration

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
N_POINTS = 32

# Short names in vertex order (0 = top-left corner of the 2D pitch).
POINT_NAMES = (
    "左上角",
    "左边线·禁区上",
    "左边线·小禁区上",
    "左边线·小禁区下",
    "左边线·禁区下",
    "左下角",
    "左小禁区前缘上",
    "左小禁区前缘下",
    "左罚球点",
    "左禁区前缘上外",
    "左禁区前缘上内",
    "左禁区前缘下内",
    "左禁区前缘下外",
    "中线上点",
    "中圈上点",
    "中圈下点",
    "中线下点",
    "右禁区前缘上外",
    "右禁区前缘上内",
    "右禁区前缘下内",
    "右禁区前缘下外",
    "右罚球点",
    "右小禁区前缘上",
    "右小禁区前缘下",
    "右上角",
    "右边线·禁区上",
    "右边线·小禁区上",
    "右边线·小禁区下",
    "右边线·禁区下",
    "右下角",
    "中圈左点",
    "中圈右点",
)


def _fix_qt_plugin_path():
    for directory in (
        "/usr/lib/x86_64-linux-gnu/qt5/plugins",
        "/usr/lib/x86_64-linux-gnu/qt6/plugins",
        "/usr/lib64/qt5/plugins",
    ):
        if os.path.isdir(os.path.join(directory, "platforms")):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = directory
            return
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


_fix_qt_plugin_path()


def _apply_cjk_font(app):
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


def _pitch_px(x_cm, y_cm, scale, padding):
    return (int(round(x_cm * scale)) + padding,
            int(round(y_cm * scale)) + padding)


def _label_offset(x_cm, config):
    """Keep neighbouring numbers from landing on the same side of a point."""
    if x_cm < 80:
        return -28, 4
    if x_cm < config.goal_box_length + 120:
        return 8, -10
    if x_cm > config.length - 80:
        return 8, 4
    if x_cm > config.length - config.goal_box_length - 120:
        return -30, -10
    return 8, -12


def draw_numbered_pitch(config, cursor, points, scale=0.07, padding=36):
    """2D pitch with all 32 indices. ``cursor`` is drawn larger."""
    img = draw_pitch(
        config,
        background_color=(24, 92, 42),
        line_color=(245, 245, 245),
        padding=padding,
        line_thickness=2,
        scale=scale,
    )
    for i, (x_cm, y_cm) in enumerate(config.vertices):
        px, py = _pitch_px(x_cm, y_cm, scale, padding)
        slot = points[i] if points is not None and i < len(points) else None
        if i == cursor:
            color = (0, 255, 255)
            radius = 11
        elif slot is None:
            color = (255, 255, 255)
            radius = 6
        elif slot[2] == 0:
            color = (160, 160, 160)
            radius = 6
        elif slot[2] == 1:
            color = (0, 165, 255)
            radius = 7
        else:
            color = (80, 220, 80)
            radius = 7
        cv2.circle(img, (px, py), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), radius, color, -1, cv2.LINE_AA)
        ox, oy = _label_offset(x_cm, config)
        org = (px + ox, py + oy)
        text = str(i)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    return img, scale, padding


def _blank_points():
    return [None] * N_POINTS


def points_to_yolo(points, width, height):
    """One YOLO-pose line, or None when nothing has been marked."""
    if not any(p is not None for p in points):
        return None
    xs, ys, kpts = [], [], []
    for slot in points:
        if slot is None or slot[2] == 0:
            kpts.extend((0.0, 0.0, 0))
            continue
        x, y, vis = slot
        nx = min(1.0, max(0.0, float(x) / max(1, width)))
        ny = min(1.0, max(0.0, float(y) / max(1, height)))
        kpts.extend((nx, ny, int(vis)))
        xs.append(nx)
        ys.append(ny)
    if not xs:
        # Every marked point is "not in frame". Still a finished label.
        xs, ys = [0.5], [0.5]
        bw = bh = 1e-3
        cx = cy = 0.5
    else:
        pad = 0.01
        x0 = max(0.0, min(xs) - pad)
        y0 = max(0.0, min(ys) - pad)
        x1 = min(1.0, max(xs) + pad)
        y1 = min(1.0, max(ys) + pad)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        bw, bh = max(x1 - x0, 1e-4), max(y1 - y0, 1e-4)
    nums = [0, cx, cy, bw, bh, *kpts]
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in nums)


def points_from_json(payload):
    raw = payload.get("points") if isinstance(payload, dict) else None
    if not isinstance(raw, list) or len(raw) != N_POINTS:
        return None
    points = []
    for item in raw:
        if item is None:
            points.append(None)
            continue
        vis = int(item.get("v", 0))
        points.append((float(item.get("x", 0.0)), float(item.get("y", 0.0)), vis))
    return points


def load_annotation(json_path, txt_path, width, height):
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as handle:
            loaded = points_from_json(json.load(handle))
        if loaded is not None:
            return loaded
    if not os.path.isfile(txt_path):
        return _blank_points()
    line = open(txt_path, encoding="utf-8").read().strip().split()
    # class, bbox(4), then 32 * (x, y, v)
    if len(line) < 5 + N_POINTS * 3:
        return _blank_points()
    body = line[5:]
    points = []
    for i in range(N_POINTS):
        nx, ny, vis = float(body[i * 3]), float(body[i * 3 + 1]), int(float(body[i * 3 + 2]))
        if vis <= 0:
            points.append((0.0, 0.0, 0))
        else:
            points.append((nx * width, ny * height, vis))
    return points


def save_annotation(json_path, txt_path, points, width, height):
    if not any(slot is not None for slot in points):
        for path in (json_path, txt_path):
            if os.path.isfile(path):
                os.remove(path)
        return
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    payload = []
    for slot in points:
        if slot is None:
            payload.append(None)
        else:
            payload.append({"x": slot[0], "y": slot[1], "v": int(slot[2])})
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"points": payload}, handle)
    yolo = points_to_yolo(points, width, height)
    if yolo is None:
        if os.path.isfile(txt_path):
            os.remove(txt_path)
        return
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(yolo + "\n")


def _bgr_pixmap(frame):
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class FitCanvas(QWidget):
    """Aspect-fit image. Clicks are reported in the source image's pixels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._orig_w = 0
        self._orig_h = 0
        self._scale = 1.0
        self._off_x = 0.0
        self._off_y = 0.0
        self._on_click = None
        self._on_move = None
        self._on_release = None
        self.setMinimumSize(360, 280)

    def set_image(self, frame_bgr):
        if frame_bgr is None:
            self._pixmap = None
            self.update()
            return
        self._orig_h, self._orig_w = frame_bgr.shape[:2]
        self._pixmap = _bgr_pixmap(frame_bgr)
        self.update()

    def set_click_handler(self, handler):
        self._on_click = handler

    def set_drag_handlers(self, on_move, on_release):
        self._on_move = on_move
        self._on_release = on_release

    def _event_xy(self, event, clamp=False):
        if self._pixmap is None or self._orig_w <= 0:
            return None
        self._fit()
        ox = (event.pos().x() - self._off_x) / self._scale
        oy = (event.pos().y() - self._off_y) / self._scale
        if clamp:
            ox = min(max(ox, 0.0), self._orig_w - 1.0)
            oy = min(max(oy, 0.0), self._orig_h - 1.0)
            return ox, oy
        if 0.0 <= ox < self._orig_w and 0.0 <= oy < self._orig_h:
            return ox, oy
        return None

    def _fit(self):
        if self._pixmap is None or self._orig_w <= 0 or self._orig_h <= 0:
            return
        self._scale = min(self.width() / self._orig_w, self.height() / self._orig_h)
        self._off_x = (self.width() - self._orig_w * self._scale) / 2.0
        self._off_y = (self.height() - self._orig_h * self._scale) / 2.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 22, 26))
        if self._pixmap is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignCenter, "无图像")
            return
        self._fit()
        target = QRect(int(self._off_x), int(self._off_y),
                       int(self._orig_w * self._scale),
                       int(self._orig_h * self._scale))
        painter.drawPixmap(target, self._pixmap, self._pixmap.rect())

    def mousePressEvent(self, event):
        if self._on_click is None:
            return
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        xy = self._event_xy(event)
        if xy is not None:
            self._on_click(xy[0], xy[1], event.button())

    def mouseMoveEvent(self, event):
        if self._on_move is None or not (event.buttons() & Qt.LeftButton):
            return
        xy = self._event_xy(event, clamp=True)
        if xy is not None:
            self._on_move(xy[0], xy[1])

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self._on_release is None:
            return
        xy = self._event_xy(event, clamp=True)
        if xy is not None:
            self._on_release(xy[0], xy[1])


def _draw_photo(frame, points, cursor):
    out = frame.copy()
    for i, slot in enumerate(points):
        if slot is None or slot[2] == 0:
            continue
        center = (int(round(slot[0])), int(round(slot[1])))
        color = (0, 255, 255) if i == cursor else (
            (0, 165, 255) if slot[2] == 1 else (80, 220, 80))
        cv2.circle(out, center, 8, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, center, 6, color, -1, cv2.LINE_AA)
        cv2.putText(out, str(i), (center[0] + 8, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(i), (center[0] + 8, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


class AnnotateWindow(QMainWindow):
    def __init__(self, image_paths, label_dir, model_path):
        super().__init__()
        self.paths = image_paths
        self.label_dir = label_dir
        self.config = SoccerPitchConfiguration()
        self.index = 0
        self.points = _blank_points()
        self.cursor = 0
        self._undo = []
        self._frame = None
        self._map_scale = 0.07
        self._map_pad = 36
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._play_tick)
        self.model_path = model_path
        self.image_dir = os.path.dirname(image_paths[0]) if image_paths else ""
        self._prelabel = None
        self._prelabel_index = None
        self._prelabel_timer = QTimer(self)
        self._prelabel_timer.timeout.connect(self._poll_prelabel)
        self._drag = None
        self._drag_before = None
        self._drag_moved = False
        self._dirty = False
        self._build_ui()
        self.index = self._first_unlabeled()
        self._load_index(self.index)

    def _paths_for(self, image_path):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return (os.path.join(self.label_dir, stem + ".json"),
                os.path.join(self.label_dir, stem + ".txt"))

    def _first_unlabeled(self):
        for i, path in enumerate(self.paths):
            json_path, _txt = self._paths_for(path)
            if not os.path.isfile(json_path):
                return i
        return 0

    def _build_ui(self):
        self.setWindowTitle("球场关键点标定")
        self.photo = FitCanvas()
        self.photo.set_click_handler(self._on_photo)
        self.photo.set_drag_handlers(self._on_photo_move, self._on_photo_release)
        self.pitch = FitCanvas()
        self.pitch.set_click_handler(self._on_pitch)

        left = self._panel("待标定图片（按顺序）", self.photo)
        right = self._panel("二维球场 · 点编号 0–31", self.pitch)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.btn_prev = QPushButton("上一张 (A)")
        self.btn_next = QPushButton("下一张 (D)")
        self.btn_play = QPushButton("播放")
        self.btn_undo = QPushButton("撤销 (Z)")
        self.btn_miss = QPushButton("本点不在画面 (N)")
        self.btn_weight = QPushButton("权重预标 (W)")
        self.btn_weight.setToolTip("用上面选中的 .pt 给当前这张图填上关键点，再拖动或重标微调")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(1))
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_undo.clicked.connect(self._undo_point)
        self.btn_miss.clicked.connect(self._mark_missing)
        self.btn_weight.clicked.connect(self._prelabel_current)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, len(self.paths) - 1))
        self.slider.sliderReleased.connect(self._slider_released)

        self.edit_images = QLineEdit(self.image_dir)
        self.edit_images.setReadOnly(True)
        self.btn_images = QPushButton("打开图片…")
        self.btn_images.setToolTip("选一张图，从这一张开始；同一目录里的其余图片仍按文件名排在前后")
        self.btn_images.clicked.connect(self._choose_image_dir)
        self.edit_model = QLineEdit(self.model_path)
        self.edit_model.setReadOnly(True)
        self.btn_model = QPushButton("权重…")
        self.btn_model.clicked.connect(self._choose_model)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("图片"))
        path_row.addWidget(self.edit_images, 1)
        path_row.addWidget(self.btn_images)
        path_row.addWidget(QLabel("权重"))
        path_row.addWidget(self.edit_model, 1)
        path_row.addWidget(self.btn_model)

        row = QHBoxLayout()
        for widget in (self.btn_prev, self.btn_next, self.btn_play,
                       self.btn_weight, self.btn_undo, self.btn_miss, self.slider):
            row.addWidget(widget)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        central = QWidget()
        central.setStyleSheet("""
            QWidget { background: #1b1d21; color: #e8e8ea; }
            QLineEdit {
                background: #2a2d34; color: #e8e8ea;
                border: 1px solid #4a4e57; border-radius: 3px;
                padding: 3px 6px;
            }
            QPushButton {
                background: #343842; color: #e8e8ea;
                border: 1px solid #4a4e57; border-radius: 3px;
                padding: 4px 10px;
            }
            QPushButton:hover { background: #3e4350; }
            QLabel { color: #e8e8ea; }
        """)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(path_row)
        layout.addLayout(row)
        layout.addWidget(splitter, 1)
        layout.addLayout(self._point_button_grid())
        layout.addWidget(self.status)
        self.setCentralWidget(central)
        hint = QLabel(
            "黄=当前点  绿=可见  橙=遮挡  灰=不在画面。"
            " W 用权重预标，拖动绿点微调，左键标可见，右键标遮挡。"
            " 下方两行数字可跳到该点，之后仍按编号往后标。")
        layout.addWidget(hint)
        self._fit_to_screen()

    def _point_button_grid(self):
        grid = QGridLayout()
        grid.setSpacing(4)
        self.point_btns = []
        for i in range(N_POINTS):
            btn = QPushButton(str(i))
            btn.setFixedHeight(28)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setToolTip(f"{i} {POINT_NAMES[i]}")
            btn.clicked.connect(lambda _checked=False, index=i: self._jump_to_point(index))
            grid.addWidget(btn, i // 16, i % 16)
            self.point_btns.append(btn)
        for col in range(16):
            grid.setColumnStretch(col, 1)
        return grid

    def _jump_to_point(self, index):
        self._stop_play()
        self.cursor = int(index)
        self._refresh()

    def _style_point_buttons(self):
        for i, btn in enumerate(self.point_btns):
            slot = self.points[i] if i < len(self.points) else None
            if i == self.cursor:
                color = "#c9a227"
            elif slot is None:
                color = "#343842"
            elif slot[2] == 0:
                color = "#5c5c5c"
            elif slot[2] == 1:
                color = "#c47a20"
            else:
                color = "#2f7a45"
            btn.setStyleSheet(
                f"QPushButton {{ background: {color}; color: #f2f2f4;"
                " border: 1px solid #6a6e78; border-radius: 3px;"
                " padding: 0; font-weight: 600; }"
            )

    def _fit_to_screen(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 820)
            return
        avail = screen.availableGeometry()
        self.resize(int(avail.width() * 0.96), int(avail.height() * 0.92))
        self.move(avail.topLeft())

    @staticmethod
    def _panel(title, canvas):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        label = QLabel(title)
        label.setStyleSheet("font-weight: 600; padding: 2px 4px;")
        lay.addWidget(label)
        lay.addWidget(canvas, 1)
        return box

    def _choose_image_dir(self):
        start = self.image_dir or ""
        if self.paths and 0 <= self.index < len(self.paths):
            start = self.paths[self.index]
        path, _ = QFileDialog.getOpenFileName(
            self, "选择从哪张图开始", start,
            "图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;所有文件 (*)")
        if not path:
            return
        folder = _resolve_image_dir(os.path.dirname(path))
        paths = _list_images(folder)
        if not paths:
            self.status.setText(f"{folder} 里没有图片")
            return
        selected = os.path.abspath(path)
        start_at = 0
        for i, item in enumerate(paths):
            if os.path.abspath(item) == selected:
                start_at = i
                break
        self._stop_play()
        self._save_current()
        self._prelabel_index = None
        self.image_dir = folder
        self.paths = paths
        self.label_dir = os.path.join(os.path.dirname(folder), "labels")
        self.edit_images.setText(folder)
        self.slider.setMaximum(max(0, len(paths) - 1))
        self._frame = None
        self._load_index(start_at)

    def _choose_model(self):
        start = os.path.dirname(self.model_path) if self.model_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择关键点权重", start, "PyTorch (*.pt);;所有文件 (*)")
        if not path:
            return
        self._prelabel_timer.stop()
        if self._prelabel is not None:
            self._prelabel.close()
            self._prelabel = None
        self.btn_weight.setEnabled(True)
        self.model_path = path
        self.edit_model.setText(path)
        self.status.setText(f"预标权重已换成 {path}。按 W 对当前图重新预标。")

    def _load_index(self, index):
        if not self.paths:
            self.status.setText("没有找到图片。")
            return
        self._save_current()
        self.index = int(index) % len(self.paths)
        path = self.paths[self.index]
        frame = cv2.imread(path)
        if frame is None:
            self.status.setText(f"无法读取 {path}")
            return
        self._frame = frame
        json_path, txt_path = self._paths_for(path)
        height, width = frame.shape[:2]
        self.points = load_annotation(json_path, txt_path, width, height)
        self._undo = []
        self._dirty = False
        unset = self._first_unset()
        self.cursor = 0 if unset is None else unset
        self.slider.blockSignals(True)
        self.slider.setValue(self.index)
        self.slider.blockSignals(False)
        self._refresh()

    def _save_current(self):
        if not self._dirty or self._frame is None or not self.paths:
            return
        self._dirty = False
        json_path, txt_path = self._paths_for(self.paths[self.index])
        height, width = self._frame.shape[:2]
        save_annotation(json_path, txt_path, self.points, width, height)

    def _first_unset(self):
        for i, slot in enumerate(self.points):
            if slot is None:
                return i
        return None

    def _refresh(self):
        if self._frame is None:
            return
        self.photo.set_image(_draw_photo(self._frame, self.points, self.cursor))
        pitch, self._map_scale, self._map_pad = draw_numbered_pitch(
            self.config, self.cursor, self.points,
            scale=self._map_scale, padding=self._map_pad)
        self.pitch.set_image(pitch)
        done = sum(slot is not None for slot in self.points)
        name = POINT_NAMES[self.cursor]
        labeled_files = sum(
            1 for path in self.paths
            if os.path.isfile(self._paths_for(path)[0]))
        self.status.setText(
            f"{self.index + 1}/{len(self.paths)}  {os.path.basename(self.paths[self.index])}"
            f"    已完成文件 {labeled_files}"
            f"    本张 {done}/32"
            f"    当前 {self.cursor} {name}"
        )
        self._style_point_buttons()

    def _set_slot(self, index, value):
        self._undo.append((index, self.points[index]))
        self.points[index] = value
        self._dirty = True

    def _advance_cursor(self):
        if self.cursor < N_POINTS - 1:
            self.cursor += 1

    def _nearest_point(self, x, y, radius=36):
        best_i, best_d = None, radius * radius
        for i, slot in enumerate(self.points):
            if slot is None or slot[2] == 0:
                continue
            dist = (slot[0] - x) ** 2 + (slot[1] - y) ** 2
            if dist <= best_d:
                best_i, best_d = i, dist
        return best_i

    def _on_photo(self, x, y, button):
        self._stop_play()
        if button == Qt.LeftButton:
            near = self._nearest_point(x, y)
            if near is not None:
                self.cursor = near
                self._drag = near
                self._drag_before = self.points[near]
                self._drag_moved = False
                self._refresh()
                return
        vis = 2 if button == Qt.LeftButton else 1
        self._set_slot(self.cursor, (float(x), float(y), vis))
        self._advance_cursor()
        self._refresh()

    def _on_photo_move(self, x, y):
        if self._drag is None:
            return
        before = self._drag_before
        vis = 2 if before is None or before[2] == 0 else int(before[2])
        self._drag_moved = True
        self.points[self._drag] = (float(x), float(y), vis)
        self._refresh()

    def _on_photo_release(self, x, y):
        if self._drag is None:
            return
        index = self._drag
        before = self._drag_before
        moved = self._drag_moved
        self._drag = None
        if moved:
            vis = 2 if before is None or before[2] == 0 else int(before[2])
            self._undo.append((index, before))
            self.points[index] = (float(x), float(y), vis)
            self._dirty = True
        self._refresh()

    def _prelabel_current(self):
        self._stop_play()
        if self._frame is None:
            return
        if not self.model_path or not os.path.isfile(self.model_path):
            self.status.setText(f"找不到关键点权重: {self.model_path}")
            return
        if self._prelabel is not None and self._prelabel.busy:
            self.status.setText("权重还在跑上一张，请稍等。")
            return
        if self._prelabel is None:
            self._prelabel = _PrelabelProcess(self.model_path)
        self._prelabel_index = self.index
        self._prelabel.submit(self._frame)
        self.btn_weight.setEnabled(False)
        self.status.setText("正在用权重预标当前图片（首次加载模型要稍等）…")
        self._prelabel_timer.start(80)

    def _poll_prelabel(self):
        if self._prelabel is None:
            self._prelabel_timer.stop()
            return
        message = self._prelabel.poll()
        if message is None:
            if not self._prelabel._proc.is_alive():
                self._prelabel_timer.stop()
                self._prelabel.busy = False
                self.btn_weight.setEnabled(True)
                self.status.setText("权重进程已退出。")
            return
        self._prelabel_timer.stop()
        self.btn_weight.setEnabled(True)
        ok, xy, conf = message
        if self.index != self._prelabel_index or self._frame is None:
            self._refresh()
            return
        if not ok:
            self.status.setText(f"权重预标失败: {xy}")
            return
        height, width = self._frame.shape[:2]
        predicted = predictions_to_points(xy, conf, width, height)
        self._undo.append((None, list(self.points)))
        self.points = predicted
        self._dirty = True
        placed = sum(slot is not None for slot in predicted)
        unset = self._first_unset()
        self.cursor = 0 if unset is None else unset
        self._refresh()
        self.status.setText(
            self.status.text()
            + f"    权重给出 {placed} 个点，拖动可微调，空位请手标"
        )

    def _on_pitch(self, x, y, _button):
        self._stop_play()
        best_i, best_d = 0, 1e9
        for i, (x_cm, y_cm) in enumerate(self.config.vertices):
            px, py = _pitch_px(x_cm, y_cm, self._map_scale, self._map_pad)
            dist = (px - x) ** 2 + (py - y) ** 2
            if dist < best_d:
                best_i, best_d = i, dist
        if best_d <= 28 ** 2:
            self.cursor = best_i
            self._refresh()

    def _mark_missing(self):
        self._stop_play()
        self._set_slot(self.cursor, (0.0, 0.0, 0))
        self._advance_cursor()
        self._refresh()

    def _undo_point(self):
        self._stop_play()
        if not self._undo:
            return
        index, previous = self._undo.pop()
        if index is None:
            self.points = previous
            unset = self._first_unset()
            self.cursor = 0 if unset is None else unset
        else:
            self.points[index] = previous
            self.cursor = index
        self._dirty = True
        self._refresh()

    def _step(self, delta):
        self._stop_play()
        if not self.paths:
            return
        self._load_index(self.index + delta)

    def _slider_released(self):
        self._stop_play()
        self._load_index(self.slider.value())

    def _toggle_play(self):
        if self._playing:
            self._stop_play()
            return
        self._playing = True
        self.btn_play.setText("暂停")
        self._timer.start(800)

    def _stop_play(self):
        self._playing = False
        self._timer.stop()
        self.btn_play.setText("播放")

    def _play_tick(self):
        if not self.paths:
            self._stop_play()
            return
        if self.index >= len(self.paths) - 1:
            self._stop_play()
            self._save_current()
            return
        self._load_index(self.index + 1)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_A, Qt.Key_Left):
            self._step(-1)
        elif key in (Qt.Key_D, Qt.Key_Right):
            self._step(1)
        elif key in (Qt.Key_Z, Qt.Key_Backspace):
            self._undo_point()
        elif key in (Qt.Key_N, Qt.Key_0):
            self._mark_missing()
        elif key == Qt.Key_Space:
            self._toggle_play()
        elif key == Qt.Key_W:
            self._prelabel_current()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._prelabel_timer.stop()
        self._save_current()
        if self._prelabel is not None:
            self._prelabel.close()
        super().closeEvent(event)


def predictions_to_points(xy, conf, width, height, min_conf=0.25):
    """Turn one YOLO pose output into 32 slots. Low-confidence points stay unset."""
    points = _blank_points()
    if xy is None:
        return points
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        return points
    conf = None if conf is None else np.asarray(conf, dtype=np.float32).reshape(-1)
    margin = max(4, int(round(0.008 * min(width, height))))
    for i in range(min(N_POINTS, len(xy))):
        x, y = float(xy[i, 0]), float(xy[i, 1])
        score = 1.0 if conf is None or i >= len(conf) else float(conf[i])
        inside = margin <= x < width - margin and margin <= y < height - margin
        if score >= min_conf and inside and not (x == 0.0 and y == 0.0):
            points[i] = (x, y, 2)
    return points


def _prelabel_worker(model_path, conn):
    """YOLO pose in a spawned process so torch stays out of the Qt process."""
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["MKL_NUM_THREADS"] = "1"
        import torch
        from ultralytics import YOLO

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[prelabel] loading {model_path} on {device}…",
              file=sys.stderr, flush=True)
        model = YOLO(model_path)
        model.to(device)
        print("[prelabel] ready.", file=sys.stderr, flush=True)
        while True:
            frame = conn.recv()
            if frame is None:
                break
            result = model(frame, verbose=False)[0]
            keypoints = getattr(result, "keypoints", None)
            xy = conf = None
            if keypoints is not None and getattr(keypoints, "xy", None) is not None \
                    and len(keypoints.xy) > 0:
                xy_all = keypoints.xy.detach().cpu().numpy().astype(np.float32)
                conf_all = None
                kconf = getattr(keypoints, "conf", None)
                if kconf is not None and len(kconf) > 0:
                    conf_all = kconf.detach().cpu().numpy().astype(np.float32)
                best = 0
                boxes = getattr(result, "boxes", None)
                if xy_all.ndim == 3 and boxes is not None and len(boxes) == len(xy_all):
                    best = int(boxes.conf.argmax())
                xy = xy_all[best] if xy_all.ndim == 3 else xy_all
                if conf_all is not None:
                    conf = conf_all[best] if conf_all.ndim == 2 else conf_all
            conn.send((True, xy, conf))
    except Exception as exc:  # noqa: BLE001 — reported back to the window
        print(f"[prelabel] error: {exc}", file=sys.stderr, flush=True)
        try:
            conn.send((False, str(exc), None))
        except Exception:
            pass


class _PrelabelProcess:
    def __init__(self, model_path):
        ctx = mp.get_context("spawn")
        self._parent, child = ctx.Pipe()
        self._proc = ctx.Process(
            target=_prelabel_worker, args=(model_path, child), daemon=True)
        self._proc.start()
        child.close()
        self.busy = False

    def submit(self, frame):
        self._parent.send(np.ascontiguousarray(frame))
        self.busy = True

    def poll(self):
        if not self.busy or not self._parent.poll():
            return None
        self.busy = False
        return self._parent.recv()

    def close(self):
        try:
            self._parent.send(None)
        except Exception:
            pass
        self._proc.join(timeout=2)
        if self._proc.is_alive():
            self._proc.terminate()


def _list_images(folder):
    names = []
    for name in os.listdir(folder):
        if os.path.splitext(name)[1].lower() in _IMAGE_EXTS:
            names.append(name)
    names.sort()
    return [os.path.join(folder, name) for name in names]


def _resolve_image_dir(path):
    """Accept either an image folder or a YOLO split that contains ``images/``."""
    if _list_images(path):
        return path
    nested = os.path.join(path, "images")
    if os.path.isdir(nested) and _list_images(nested):
        return nested
    return path


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="标定 32 个球场关键点")
    parser.add_argument(
        "--images",
        default=os.path.join(root, "data", "pitch_dataset", "images"),
        help="待标定图片目录（按文件名顺序）。")
    parser.add_argument(
        "--labels",
        default="",
        help="标注输出目录（默认与 images 同级的 labels/）。")
    parser.add_argument(
        "--model",
        default=pitch_keypoints_path(),
        help="预标用的关键点权重（默认 weights/pitch_keypoints.pt）。")
    args = parser.parse_args()
    if not os.path.isdir(args.images):
        raise SystemExit(f"找不到图片目录 {args.images}")
    label_dir = args.labels or os.path.join(os.path.dirname(args.images), "labels")
    paths = _list_images(args.images)
    if not paths:
        raise SystemExit(f"{args.images} 里没有图片")
    app = QApplication(sys.argv)
    _apply_cjk_font(app)
    win = AnnotateWindow(paths, label_dir, args.model)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
