#!/usr/bin/env python3
"""
Qt-based calibration GUI for a fixed camera (video file / camera / RTSP).

Shows the live stream (TOP) above the top-down 2D pitch map (BOTTOM). Click
"Start Calibration" to pause the stream and freeze the current frame as the
calibration image, then click the SAME 4+ pitch points on both sides — point *i*
on the top corresponds to point *i* on the bottom. Click "Compute & Save" to fit
the homography and write the JSON consumed by ``detector_type: homography``.

Usage
-----
    python3 scripts/calibrate_gui.py --source 0                 # camera 0
    python3 scripts/calibrate_gui.py --source /path/match.mp4   # video file
    python3 scripts/calibrate_gui.py --source rtsp://...        # network stream

Writes ``<source>_calibration.json`` (used by the node) and a warped preview image.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QPointF, QRect, QRectF
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pitch_view.annotators.soccer import draw_pitch
from pitch_view.configs import SoccerPitchConfiguration

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _fix_qt_plugin_path():
    """opencv-python (non-headless) bundles its own Qt and overrides
    ``QT_QPA_PLATFORM_PLUGIN_PATH`` to a directory lacking the xcb plugin PyQt5
    needs, so the app crashes on startup. Point it back at the system Qt plugins
    (or clear it to fall back to Qt's compiled-in default)."""
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


def _apply_cjk_font(app):
    """Set a CJK-capable application font so Chinese UI text renders."""
    # Load the font file directly into Qt (robust against fontconfig cache issues).
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
            # Force the family app-wide so per-widget stylesheets don't drop it.
            app.setStyleSheet(f"QWidget {{ font-family: '{name}'; }}")
            return


# --------------------------------------------------------------------------- #
# Clickable, scalable image canvas with point overlays.
# --------------------------------------------------------------------------- #
class ImageCanvas(QWidget):
    """Displays an image fitted to the widget and lets the user click points.

    Clicked points are stored in ORIGINAL image coordinates (so they survive
    window resizes) and drawn back on the fitted image.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._orig_w = 0
        self._orig_h = 0
        self._points = []
        self._scale = 1.0
        self._off_x = 0.0
        self._off_y = 0.0
        self.setMinimumSize(320, 240)

    # -- public API -------------------------------------------------------- #
    def set_image(self, frame_bgr):
        if frame_bgr is None:
            return
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._orig_w, self._orig_h = w, h
        self.update()

    def points(self):
        return list(self._points)

    def clear_points(self):
        self._points = []
        self.update()

    # -- fitting ----------------------------------------------------------- #
    def _recompute_fit(self):
        if self._pixmap is None:
            return
        # Uniform aspect-fit: scale by the same factor on both axes so the image
        # is never distorted, and centre it. The leftover margin is filled with
        # the light canvas colour (not black), so there are no letterbox bars.
        s = min(self.width() / self._orig_w, self.height() / self._orig_h)
        self._scale = s
        self._off_x = (self.width() - self._orig_w * s) / 2.0
        self._off_y = (self.height() - self._orig_h * s) / 2.0

    def _to_display(self, ox, oy):
        return (self._off_x + ox * self._scale, self._off_y + oy * self._scale)

    def _to_original(self, dx, dy):
        return ((dx - self._off_x) / self._scale, (dy - self._off_y) / self._scale)

    # -- events ------------------------------------------------------------ #
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(245, 245, 245))
        if self._pixmap is None:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignCenter, "无图像")
            return

        self._recompute_fit()
        target = QRect(int(self._off_x), int(self._off_y),
                       int(self._orig_w * self._scale),
                       int(self._orig_h * self._scale))
        painter.drawPixmap(target, self._pixmap, self._pixmap.rect())

        for i, (ox, oy) in enumerate(self._points):
            dx, dy = self._to_display(ox, oy)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(QPointF(dx, dy), 5, 5)
            painter.setPen(QColor(0, 255, 0))
            painter.drawText(QPointF(dx + 7, dy), str(i))

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self._pixmap is None:
            return
        self._recompute_fit()
        ox, oy = self._to_original(event.pos().x(), event.pos().y())
        if 0.0 <= ox < self._orig_w and 0.0 <= oy < self._orig_h:
            self._points.append((ox, oy))
            self.update()


# --------------------------------------------------------------------------- #
# Main window.
# --------------------------------------------------------------------------- #
class CalibrationWindow(QMainWindow):
    def __init__(self, source, map_scale, map_padding, out_json, preview_scale):
        super().__init__()
        self.map_scale = map_scale
        self.map_padding = map_padding
        self.out_json = out_json
        self.preview_scale = preview_scale
        self.paused = False
        self.frame = None

        self.config = SoccerPitchConfiguration()

        # Open the video source (file path / RTSP URL / camera index).
        self.cap = _open_capture(source)

        self._build_ui()

        # Grab the first frame synchronously, then decide live vs. paused.
        ok, self.frame = self.cap.read()
        if not ok:
            raise SystemExit(f"Could not read a frame from source {source!r}.")
        self.canvas_src.set_image(self.frame)

        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        is_image = (isinstance(source, str)
                    and os.path.splitext(source)[1].lower() in _IMAGE_EXTS)
        if frame_count == 1 or is_image:  # a single still image -> start paused
            self._set_paused(True)
        else:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._grab_frame)
            self.timer.start(33)

    # -- UI ----------------------------------------------------------------- #
    def _build_ui(self):
        self.setWindowTitle("球场标定 (固定机位)")

        self.canvas_src = ImageCanvas()
        self.canvas_map = ImageCanvas()
        self.canvas_map.set_image(draw_pitch(
            self.config, scale=self.map_scale, padding=self.map_padding))

        top = self._panel("实拍图 (视频/相机)", self.canvas_src)
        bottom = self._panel("二维球场图", self.canvas_map)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(bottom)
        splitter.setSizes([520, 320])

        self.btn_toggle = QPushButton("开始标定 (暂停)")
        self.btn_toggle.clicked.connect(self._toggle_pause)
        self.btn_save = QPushButton("计算保存")
        self.btn_save.clicked.connect(self._compute_save)
        self.btn_reset = QPushButton("重置")
        self.btn_reset.clicked.connect(self._reset)

        controls = QHBoxLayout()
        controls.addWidget(self.btn_toggle)
        controls.addWidget(self.btn_save)
        controls.addWidget(self.btn_reset)
        controls.addStretch(1)

        self.status = QLabel("就绪。点「开始标定」暂停一帧后,在上下两图各点 4 点(顺序对应)。")
        self.status.setStyleSheet("color: #333333;")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addLayout(controls)
        layout.addWidget(splitter)
        layout.addWidget(self.status)
        self.setCentralWidget(central)
        self._fit_to_screen()

    def _fit_to_screen(self):
        """Size the window to the available screen area (fits small / HiDPI displays)."""
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
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: bold;")
        v.addWidget(lbl)
        v.addWidget(canvas)
        return box

    # -- video -------------------------------------------------------------- #
    def _grab_frame(self):
        if self.paused:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.timer.stop()
            self.status.setText("视频流结束。可点「重置」后重新标定当前帧。")
            return
        self.frame = frame
        self.canvas_src.set_image(frame)

    def _toggle_pause(self):
        self._set_paused(not self.paused)

    def _set_paused(self, paused):
        self.paused = paused
        if paused:
            self.btn_toggle.setText("继续播放")
            self.status.setText("已暂停当前帧。请在上下两图各点 4 点,序号顺序一一对应。")
        else:
            self.btn_toggle.setText("开始标定 (暂停)")
            self.status.setText("直播中。点「开始标定」暂停一帧用于标定。")

    def _reset(self):
        self.canvas_src.clear_points()
        self.canvas_map.clear_points()
        self.status.setText("已重置。请重新在上下两图各点 4 点。")

    # -- compute + save ----------------------------------------------------- #
    def _compute_save(self):
        src_pts = self.canvas_src.points()
        map_pts = self.canvas_map.points()  # in map-pixel coordinates
        if len(src_pts) < 4 or len(src_pts) != len(map_pts):
            QMessageBox.warning(self, "点数不足/不对应",
                                f"需要上下两图各点 >=4 个点且数量一致。\n"
                                f"当前上 {len(src_pts)} / 下 {len(map_pts)}。")
            return
        if self.frame is None:
            QMessageBox.warning(self, "无图像", "没有可用的标定帧。")
            return

        src = np.asarray(src_pts, dtype=np.float32)
        # Map pixels -> pitch centimetres.
        tgt = np.asarray([((px - self.map_padding) / self.map_scale,
                           (py - self.map_padding) / self.map_scale)
                          for (px, py) in map_pts], dtype=np.float32)

        homography, _ = cv2.findHomography(src, tgt)
        if homography is None:
            QMessageBox.warning(self, "无法计算", "点共线或数量不足,单应性计算失败。")
            return

        with open(self.out_json, "w") as f:
            json.dump({
                "sport": "soccer",
                "source": src.tolist(),
                "target": tgt.tolist(),
                "homography": homography.tolist(),
                "length": self.config.length,
                "width": self.config.width,
            }, f, indent=2)

        length_px = int(self.config.length * self.preview_scale)
        width_px = int(self.config.width * self.preview_scale)
        # ``homography`` maps source px -> centimetres; scale it so the warped
        # output actually lands inside the (length_px, width_px) preview canvas.
        s = self.preview_scale
        preview_h = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], np.float64) @ homography
        warped = cv2.warpPerspective(self.frame, preview_h, (length_px, width_px))
        out_preview = os.path.splitext(self.out_json)[0] + "_preview.jpg"
        cv2.imwrite(out_preview, warped)

        self.status.setText(f"已保存: {self.out_json}")
        _show_preview(self, warped, self.out_json)

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _open_capture(source):
    try:
        src = int(source)  # camera index
    except ValueError:
        src = source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {source!r} "
                         f"(pass a video file path, RTSP URL, or camera index).")
    return cap


def _show_preview(parent, warped_bgr, out_json):
    dialog = QDialog(parent)
    dialog.setWindowTitle("变换结果预览")
    h, w = warped_bgr.shape[:2]
    rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    lbl = QLabel()
    lbl.setPixmap(QPixmap.fromImage(qimg))
    lbl.setScaledContents(False)
    layout = QVBoxLayout(dialog)
    layout.addWidget(lbl)
    layout.addWidget(QLabel(f"单应性已保存到:\n{out_json}"))
    dialog.resize(min(600, w), min(500, h + 40))
    dialog.exec_()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                        help="Video file path, RTSP URL, or camera index (default 0).")
    parser.add_argument("--map-scale", type=float, default=0.1,
                        help="Pixels per cm for the 2D map (default 0.1).")
    parser.add_argument("--map-padding", type=int, default=40,
                        help="Padding around the 2D map in pixels (default 40).")
    parser.add_argument("--preview-scale", type=float, default=0.05,
                        help="Pixels per cm for the warped preview (default 0.05).")
    parser.add_argument("--out", default="",
                        help="Output JSON path (default <source>_calibration.json).")
    args = parser.parse_args()

    if not args.out:
        base = os.path.splitext(str(args.source))[0]
        args.out = (base if base not in ("", "0") else "calibration") + "_calibration.json"

    app = QApplication(sys.argv)
    _apply_cjk_font(app)
    win = CalibrationWindow(args.source, args.map_scale, args.map_padding,
                            args.out, args.preview_scale)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
