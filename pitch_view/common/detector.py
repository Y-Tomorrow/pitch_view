"""
Court / pitch keypoint detection interface.

A "detector" is anything with a ``detect(frame) -> (keypoints, target)`` method
returning a matched pair:

  * ``keypoints`` — detected keypoint pixel coordinates, shape (N, 2);
  * ``target``    — the 2D pitch/court vertices each keypoint corresponds to,
                    shape (N, 2), in centimetres, in the same order.

Because a model may only detect a *subset* of the landmarks (low-confidence
points are dropped), the detector aligns each detected keypoint to its vertex via
``class_id``, so ``keypoints`` and ``target`` always have the same length and the
homography is well-posed.

Implementations:

  * ``CourtDetector``            — classical CV (Canny + Hough), no model, 4 corners.
  * ``YoloKeyPointDetector``     — a local YOLO keypoint model (.pt).
  * ``InferenceKeyPointDetector``— a local Roboflow model (weights cached on disk).
  * ``RoboflowKeyPointDetector`` — a Roboflow model via the cloud inference-sdk.

``create_detector`` is the single switch point; the node picks an implementation
via the ``detector_type`` parameter.
"""
import json
import sys
from typing import List, Protocol, Tuple

import numpy as np


def _cuda_or_cpu(device: str):
    """Use CUDA FP16 when a GPU is available. Otherwise stay on CPU FP32."""
    device = str(device or "cpu")
    if not device.startswith("cuda"):
        return device, False
    try:
        import torch
        if torch.cuda.is_available():
            return device, True
    except Exception:
        pass
    print("[detector] CUDA 不可用，改用 CPU。", file=sys.stderr, flush=True)
    return "cpu", False


class KeyPointDetector(Protocol):
    """Interface implemented by all court/pitch keypoint detectors."""

    keypoint_names: List[str]
    keypoint_vertices: np.ndarray

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(keypoints, target)`` — aligned pixel points and 2D vertices."""
        ...


def _get(obj, name, default=None):
    """Read a field from a dict or a Pydantic/object, returning ``default`` if absent."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_roboflow_keypoints_full(result):
    """
    Extract keypoints (with metadata) from a Roboflow keypoint-detection result.

    Handles the ``inference-sdk`` HTTP result (a dict with a ``predictions`` key)
    and the local ``inference`` package result (which may be a Pydantic response
    object, or a list of such objects, with ``.predictions``).

    Returns a list of dicts ``{class_id, class_name, x, y, confidence}`` ordered
    by ``class_id``.
    """
    predictions = _get(result, "predictions")
    if not isinstance(predictions, (list, tuple)):
        predictions = result if isinstance(result, (list, tuple)) else []

    flat = []
    for p in predictions:
        inner = _get(p, "predictions")
        if isinstance(inner, (list, tuple)):
            flat.extend(inner)
        else:
            flat.append(p)
    if not flat:
        return []

    pred = max(flat, key=lambda p: float(_get(p, "confidence", 0.0) or 0.0))
    keypoints = _get(pred, "keypoints") or []

    out = []
    for k in keypoints:
        if isinstance(k, (list, tuple)) and len(k) >= 2:
            out.append({"class_id": -1, "class_name": "",
                        "x": float(k[0]), "y": float(k[1]), "confidence": None})
            continue
        out.append({
            "class_id": _get(k, "class_id", -1),
            "class_name": _get(k, "class_name", ""),
            "x": float(_get(k, "x")),
            "y": float(_get(k, "y")),
            "confidence": _get(k, "confidence"),
        })
    out.sort(key=lambda d: d["class_id"])
    return out


# Pose models often emit a dummy (0, 0) / clip off-screen landmarks onto the
# image border. Those points are not visible and will warp the homography.
_DEFAULT_VISIBILITY = 0.25
_BORDER_MARGIN_PX = 4


def _filter_aligned_keypoints(
    xy: np.ndarray,
    vertices: np.ndarray,
    frame_shape=None,
    conf=None,
    min_confidence: float = 0.0,
    border_margin: int = _BORDER_MARGIN_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only keypoints that are actually visible in the camera frame.

    Drops unused ``(0, 0)`` slots, off-screen / border-clamped points (the model
    guesses landmarks that lie outside the FOV and OpenCV/YOLO clips them to
    ``x=0/w`` or ``y=0/h``), and low-confidence points.
    """
    xy = np.asarray(xy, dtype=np.float32)
    vertices = np.asarray(vertices, dtype=np.float32)
    if xy.size == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty
    if len(xy) != len(vertices):
        n = min(len(xy), len(vertices))
        xy, vertices = xy[:n], vertices[:n]

    keep = np.ones(len(xy), dtype=bool)
    keep &= np.isfinite(xy).all(axis=1)
    # Unused YOLO slots are often exactly (0, 0).
    keep &= ~((xy[:, 0] == 0.0) & (xy[:, 1] == 0.0))

    if frame_shape is not None:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        m = max(int(border_margin), int(round(0.008 * min(h, w))))
        keep &= (xy[:, 0] >= m) & (xy[:, 0] < w - m)
        keep &= (xy[:, 1] >= m) & (xy[:, 1] < h - m)

    if conf is not None:
        conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        if len(conf) == len(xy):
            floor = min_confidence if min_confidence > 0 else _DEFAULT_VISIBILITY
            keep &= conf >= floor
    elif min_confidence > 0:
        keep[:] = False

    return xy[keep], vertices[keep]


def _align_keypoints(full, vertices, frame_shape=None, min_confidence=0.0) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Map detected keypoints to their corresponding 2D vertices by ``class_id``.

    ``full`` is the output of :func:`_parse_roboflow_keypoints_full`; ``vertices``
    is the full list of 2D pitch/court coordinates. Drops any keypoint whose
    ``class_id`` does not index a valid vertex, and — when ``frame_shape`` is given
    — keypoints predicted outside the frame (the model extrapolates landmarks that
    lie off-screen on zoomed/cropped views, and those are unreliable for the
    homography). Optionally also drops keypoints below ``min_confidence``.

    Returns ``(keypoints, target)``, both shape (N, 2) and aligned.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    valid = [d for d in full if 0 <= d["class_id"] < len(vertices)]
    if not valid:
        return (np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32))
    idx = [d["class_id"] for d in valid]
    xy = np.asarray([[d["x"], d["y"]] for d in valid], dtype=np.float32)
    conf = np.asarray(
        [(-1.0 if d["confidence"] is None else float(d["confidence"]))
         for d in valid],
        dtype=np.float32,
    )
    # Roboflow may omit per-point confidence; treat missing as "keep" unless
    # the caller set an explicit min_confidence.
    conf_arg = None if (conf < 0).all() else np.where(conf < 0, 1.0, conf)
    return _filter_aligned_keypoints(
        xy, vertices[idx], frame_shape=frame_shape, conf=conf_arg,
        min_confidence=min_confidence,
    )


class _ClassicalDetector:
    """
    Adapts ``CourtDetector`` (which returns the 4 corners) to the
    ``(keypoints, target)`` contract used by the rest of the pipeline.
    """

    def __init__(self, inner, target: np.ndarray) -> None:
        self._inner = inner
        self._target = np.asarray(target, dtype=np.float32)

    def __getattr__(self, name):
        # Delegate classical-CV tuning params (canny_low, etc.) to the inner object.
        return getattr(self._inner, name)

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._inner.detect(frame), self._target


class YoloKeyPointDetector:
    """
    YOLO keypoint detector (the sports-main pipeline).

    Reserved interface: give it the path to a trained keypoint model (``.pt``).
    ``detect`` lazily loads the model and returns keypoints in the model's own
    order — that order must match ``keypoint_vertices``.
    """

    def __init__(
        self,
        model_path: str,
        keypoint_names: List[str],
        keypoint_vertices: List[Tuple[int, int]],
        device: str = "cpu",
        min_confidence: float = 0.0,
    ) -> None:
        self.model_path = model_path
        self.keypoint_names = list(keypoint_names)
        self.keypoint_vertices = np.asarray(keypoint_vertices, dtype=np.float32)
        self.device = device
        self.min_confidence = min_confidence
        self._model = None
        self._half = False

    def load(self) -> None:
        if not self.model_path:
            raise FileNotFoundError(
                "No model_path set. Train a keypoint model "
                "(see docs/training.md) or download one, then set the path.")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YoloKeyPointDetector requires `ultralytics` "
                "(pip install ultralytics).") from exc
        self.device, self._half = _cuda_or_cpu(self.device)
        self._model = YOLO(self.model_path).to(self.device)

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            self.load()
        result = self._model(
            frame, verbose=False, device=self.device, imgsz=640,
            **({"quantize": 16} if self._half else {}))[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or getattr(keypoints, "xy", None) is None \
                or len(keypoints.xy) == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        xy = keypoints.xy[0].cpu().numpy().astype(np.float32)
        conf = None
        kconf = getattr(keypoints, "conf", None)
        if kconf is not None and len(kconf) > 0:
            conf = kconf[0].cpu().numpy().astype(np.float32)
        # YOLO always emits the full landmark set (fixed order == vertices).
        # Invisible / off-screen points are typically low-conf or clamped to
        # the image border — drop them so the homography uses only real hits.
        return _filter_aligned_keypoints(
            xy, self.keypoint_vertices, frame_shape=frame.shape, conf=conf,
            min_confidence=self.min_confidence,
        )


class InferenceKeyPointDetector:
    """
    Local (on-device) Roboflow model via the ``inference`` package.

    Unlike ``RoboflowKeyPointDetector`` (cloud), this runs inside your own process:
    weights are downloaded once by ``get_model`` and cached locally, then inference
    runs without any network round-trip. This is the right choice for a ROS2 stream.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str,
        keypoint_names: List[str],
        keypoint_vertices: List[Tuple[int, int]],
        device: str = "cpu",
        min_confidence: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.keypoint_names = list(keypoint_names)
        self.keypoint_vertices = np.asarray(keypoint_vertices, dtype=np.float32)
        self.device = device
        self.min_confidence = min_confidence
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            if not self.api_key:
                raise ValueError(
                    "InferenceKeyPointDetector needs an api_key (set the "
                    "ROBOFLOW_API_KEY env var or the api_key parameter).")
            try:
                from inference import get_model
            except ImportError as exc:
                raise ImportError(
                    "InferenceKeyPointDetector requires the Roboflow `inference` "
                    "package (pip install inference).") from exc
            self._model = get_model(model_id=self.model_id, api_key=self.api_key)
        return self._model

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        model = self._ensure_model()
        # The `inference` package accepts BGR numpy arrays directly.
        result = model.infer(frame)
        return _align_keypoints(
            _parse_roboflow_keypoints_full(result),
            self.keypoint_vertices,
            frame_shape=frame.shape,
            min_confidence=self.min_confidence,
        )


class RoboflowKeyPointDetector:
    """
    Roboflow hosted keypoint model via the cloud ``inference-sdk`` HTTP client.

    The image is sent to the cloud, so this is the zero-setup option, not a
    local/streaming runtime. Prefer ``InferenceKeyPointDetector`` for a ROS2 stream.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str,
        keypoint_names: List[str],
        keypoint_vertices: List[Tuple[int, int]],
        api_url: str = "https://serverless.roboflow.com",
        min_confidence: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.keypoint_names = list(keypoint_names)
        self.keypoint_vertices = np.asarray(keypoint_vertices, dtype=np.float32)
        self.api_url = api_url
        self.min_confidence = min_confidence
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "RoboflowKeyPointDetector needs an api_key (set the "
                    "ROBOFLOW_API_KEY env var or the api_key parameter).")
            try:
                from inference_sdk import (
                    InferenceConfiguration,
                    InferenceHTTPClient,
                )
            except ImportError as exc:
                raise ImportError(
                    "RoboflowKeyPointDetector requires `inference-sdk` "
                    "(pip install inference-sdk).") from exc
            self._client = InferenceHTTPClient(
                api_url=self.api_url, api_key=self.api_key).configure(
                    InferenceConfiguration(api_key_transport="header"))
        return self._client

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        client = self._ensure_client()
        # Roboflow inference expects RGB; our frames are BGR.
        rgb = frame[:, :, ::-1]
        result = client.infer(rgb, model_id=self.model_id)
        return _align_keypoints(
            _parse_roboflow_keypoints_full(result),
            self.keypoint_vertices,
            frame_shape=frame.shape,
            min_confidence=self.min_confidence,
        )


class FixedHomographyDetector:
    """
    Fixed-camera detector: a pre-calibrated point-to-pitch homography.

    Instead of detecting keypoints every frame, this loads the point pairs
    ``(source, target)`` saved once during calibration (``scripts/calibrate.py``)
    and returns them every frame. The node's ``ViewTransformer`` then applies a
    constant homography — correct only while the camera is fixed (position, zoom,
    and image resolution unchanged). Use ``detector_type: homography`` in
    ``config/court_detector.yaml``.
    """

    def __init__(self, calibration_file: str) -> None:
        with open(calibration_file, "r") as f:
            data = json.load(f)
        source = data.get("source")
        target = data.get("target")
        if (not source or not target or len(source) < 4
                or len(source) != len(target)):
            raise ValueError(
                f"Calibration file {calibration_file!r} needs matching "
                f"source/target lists of >=4 points.")
        self.source = np.asarray(source, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # A fixed homography: the same correspondences on every frame.
        return self.source.copy(), self.target


def create_detector(
    kind: str,
    *,
    model_path: str = "",
    homography_file: str = "",
    api_key: str = "",
    model_id: str = "football-field-detection-f07vi/14",
    api_url: str = "https://serverless.roboflow.com",
    config=None,
    device: str = "cpu",
    min_confidence: float = 0.0,
):
    """
    Build a detector whose ``detect(frame)`` returns an aligned
    ``(keypoints, target)`` pair for the given ``config``.
    """
    from pitch_view.configs import BasketballCourtConfiguration
    from pitch_view.common.court import CourtDetector

    config = config or BasketballCourtConfiguration()

    if kind == "classical":
        return _ClassicalDetector(
            CourtDetector(),
            np.asarray(config.court_corners, dtype=np.float32))
    if kind == "homography":
        if not homography_file:
            raise ValueError(
                "kind='homography' needs homography_file (the calibration JSON "
                "produced by scripts/calibrate.py).")
        return FixedHomographyDetector(calibration_file=homography_file)
    if kind == "yolo":
        return YoloKeyPointDetector(
            model_path=model_path,
            keypoint_names=config.keypoint_labels,
            keypoint_vertices=config.keypoint_vertices,
            device=device,
            min_confidence=min_confidence,
        )
    if kind == "inference":
        return InferenceKeyPointDetector(
            api_key=api_key,
            model_id=model_id,
            keypoint_names=config.keypoint_labels,
            keypoint_vertices=config.keypoint_vertices,
            device=device,
            min_confidence=min_confidence,
        )
    if kind == "roboflow":
        return RoboflowKeyPointDetector(
            api_key=api_key,
            model_id=model_id,
            keypoint_names=config.keypoint_labels,
            keypoint_vertices=config.keypoint_vertices,
            api_url=api_url,
            min_confidence=min_confidence,
        )
    raise ValueError(
        f"Unknown detector_type '{kind}' "
        f"(expected 'classical', 'homography', 'yolo', 'inference', or 'roboflow').")
