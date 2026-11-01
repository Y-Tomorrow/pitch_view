"""Live frames from a Gazebo Sim camera topic, with an OpenCV-like read()."""

from __future__ import annotations

import os
import threading

import cv2
import numpy as np

# System gz protobufs were generated for an older protobuf. Set this before
# the first import of gz.msgs.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_RGB = 3
_BGR = 8


class GzImageCapture:
    """Subscribe to a gz.msgs.Image topic. ``read`` waits for the next frame."""

    def __init__(self, topic: str = "/pitch_camera", fps: float = 20.0):
        import gz.msgs10.image_pb2 as image_pb2
        import gz.transport13 as transport

        self._topic = topic if topic.startswith("/") else f"/{topic}"
        self._fps = float(fps)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._latest = None
        self._closed = False
        self._node = transport.Node()
        self._node.subscribe(image_pb2.Image, self._topic, self._on_image)

    def _on_image(self, msg) -> None:
        width, height = int(msg.width), int(msg.height)
        if width <= 0 or height <= 0 or not msg.data:
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        fmt = int(msg.pixel_format_type)
        if fmt == _RGB:
            frame = buf.reshape(height, width, 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif fmt == _BGR:
            frame = buf.reshape(height, width, 3).copy()
        else:
            return
        with self._lock:
            self._latest = frame
            self._ready.set()

    def isOpened(self) -> bool:
        return not self._closed

    def read(self):
        while not self._closed:
            if not self._ready.wait(0.2):
                continue
            with self._lock:
                if self._closed:
                    return False, None
                frame = None if self._latest is None else self._latest.copy()
                self._ready.clear()
            if frame is not None:
                return True, frame
        return False, None

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def set(self, prop: int, value) -> bool:
        return False

    def release(self) -> None:
        self._closed = True
        self._ready.set()
        try:
            self._node.unsubscribe(self._topic)
        except Exception:
            pass


def open_gz_source(source: str) -> GzImageCapture:
    """``gz://pitch_camera`` or ``gz:///pitch_camera``."""
    topic = source[5:] if source.startswith("gz://") else source
    topic = topic.lstrip("/")
    return GzImageCapture("/" + topic)
