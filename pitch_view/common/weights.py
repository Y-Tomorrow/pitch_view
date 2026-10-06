"""
Canonical weight locations under ``<repo>/models`` for train + inference.

Packaging: ship the whole ``models/`` directory.
"""
import os
import shutil

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(_ROOT, "models")
PITCH_KEYPOINTS_NAME = "pitch_keypoints.pt"
PLAYERS_NAME = "players.pt"
EASYOCR_DETECT_NAME = "craft_mlt_25k.pth"
EASYOCR_RECOG_NAME = "english_g2.pth"


def models_dir() -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    return MODELS_DIR


def pitch_keypoints_path() -> str:
    return os.path.join(models_dir(), PITCH_KEYPOINTS_NAME)


def players_path() -> str:
    return os.path.join(models_dir(), PLAYERS_NAME)


def publish_weight(src_best: str, dest_path: str) -> str:
    """Copy a trained ``best.pt`` into ``models/`` under the packaged name."""
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
