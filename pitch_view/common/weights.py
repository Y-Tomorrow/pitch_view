"""
Canonical weight locations under ``<repo>/weights`` for train + inference.

Packaging: ship the whole ``weights/`` directory.
"""
import os
import shutil

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEIGHTS_DIR = os.path.join(_ROOT, "weights")
PITCH_KEYPOINTS_NAME = "pitch_keypoints.pt"
PLAYERS_NAME = "players.pt"
EASYOCR_DETECT_NAME = "craft_mlt_25k.pth"
EASYOCR_RECOG_NAME = "english_g2.pth"


def weights_dir() -> str:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    return WEIGHTS_DIR


def pitch_keypoints_path() -> str:
    return os.path.join(weights_dir(), PITCH_KEYPOINTS_NAME)


def players_path() -> str:
    return os.path.join(weights_dir(), PLAYERS_NAME)


def paddle_ocr_dir() -> str:
    """PaddleOCR weight cache. First use of the Paddle engine fills this."""
    path = os.path.join(weights_dir(), "paddle_ocr")
    os.makedirs(path, exist_ok=True)
    return path


def publish_weight(src_best: str, dest_path: str) -> str:
    """Copy a trained ``best.pt`` into ``weights/`` under the packaged name."""
    if not src_best or not os.path.isfile(src_best):
        raise FileNotFoundError(f"Trained weight not found: {src_best!r}")
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    shutil.copy2(src_best, dest_path)
    return dest_path


def resolve_trained_best(trainer) -> str:
    """Return absolute path to Ultralytics ``best.pt`` after ``model.train``."""
    best = getattr(trainer, "best", None)
    if best is not None and os.path.isfile(str(best)):
        return str(best)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir is not None:
        cand = os.path.join(str(save_dir), "weights", "best.pt")
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError("Could not locate Ultralytics best.pt after training.")
