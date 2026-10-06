"""
Load ``config/court_detector.yaml`` for scripts and ROS alike.

ROS launch uses the nested ``court_detector.ros__parameters`` block.
Standalone scripts call :func:`load_settings` and get a flat dict with defaults.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(_ROOT, "config", "court_detector.yaml")

# Flat defaults — keep in sync with config/court_detector.yaml comments.
_DEFAULTS: Dict[str, Any] = {
    "sport": "soccer",
    "scale": 0.05,
    "device": "cpu",
    "detector_type": "inference",
    "model_path": "models/pitch_keypoints.pt",
    "homography_file": "",
    "api_key": "",
    "model_id": "football-field-detection-f07vi/14",
    "api_url": "https://serverless.roboflow.com",
    "min_confidence": 0.0,
    "canny_low": 50,
    "canny_high": 150,
    "hough_threshold": 80,
    "min_line_length": 100,
    "max_line_gap": 30,
    "player_model_path": "models/players.pt",
    "player_conf": 0.25,
    "player_iou": 0.45,
    "player_agnostic_nms": True,
    "player_nms_iou": 0.45,
    "player_nms_contain": 0.7,
    "jersey_ocr_min_conf": 0.35,
    "jersey_torso_x_margin": 0.18,
    "jersey_torso_y0": 0.10,
    "jersey_torso_y1": 0.52,
    "closeup_pad_ratio": 0.18,
    "closeup_max_scale": 5.0,
    "closeup_track_iou": 0.3,
}


def default_config_path() -> str:
    return DEFAULT_CONFIG_PATH


def resolve_repo_path(path: str) -> str:
    """Resolve a possibly relative path against the repo root."""
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_ROOT, path))


def load_settings(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML settings merged over defaults.

    Accepts either the ROS nested layout::

        court_detector:
          ros__parameters: { ... }

    or a flat mapping of the same keys.
    """
    cfg_path = path or DEFAULT_CONFIG_PATH
    out = dict(_DEFAULTS)
    if not cfg_path or not os.path.isfile(cfg_path):
        return out
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Reading config YAML needs PyYAML (pip install pyyaml).") from exc
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        return out
    params = raw
    nested = raw.get("court_detector")
    if isinstance(nested, dict):
        ros = nested.get("ros__parameters")
        if isinstance(ros, dict):
            params = ros
        else:
            params = nested
    for key, value in params.items():
        if key in _DEFAULTS or key in out:
            out[key] = value
        else:
            # Allow forward-compatible extra keys.
            out[key] = value
    return out
