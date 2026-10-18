#!/usr/bin/env python3
"""
Train / fine-tune a football player detector (YOLOv8 object detection).

Public Roboflow dataset (same as sports-main):
  workspace: roboflow-jvuqo
  project:   football-players-detection-3zvbc
  classes:   ball, goalkeeper, player, referee

Examples:
    export ROBOFLOW_API_KEY=...
    # download + train
    python3 scripts/train_players.py --download --epochs 100 --model yolov8m.pt

    # already downloaded
    python3 scripts/train_players.py --data-dir football-players-detection-12 --epochs 100

Trained ``best.pt`` is published to ``weights/players.pt`` (same folder as
inference). Intermediate Ultralytics logs stay under ``runs/player/``.
Dataset downloads into the project root, next to football-field-detection-14.
"""
import argparse
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from pitch_view.common.weights import (
    players_path,
    publish_weight,
    resolve_trained_best,
)

DEFAULT_WORKSPACE = "roboflow-jvuqo"
DEFAULT_PROJECT = "football-players-detection-3zvbc"
# Dataset version used widely with the sports soccer demos; override with --version.
DEFAULT_VERSION = 12
DEFAULT_NAMES = ["ball", "goalkeeper", "player", "referee"]


def _download_dataset(api_key, workspace, project, version, location=""):
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download("yolov8")
    path = ds.location
    if location:
        location = os.path.abspath(location)
        if os.path.abspath(path) != location:
            if os.path.exists(location):
                shutil.rmtree(location)
            shutil.move(path, location)
            path = location
    return path


def _find_data_yaml(data_dir):
    """Prefer the Roboflow-exported data.yaml; else write a detect yaml."""
    data_dir = os.path.abspath(data_dir)
    for name in ("data.yaml", "data.yml"):
        cand = os.path.join(data_dir, name)
        if os.path.isfile(cand):
            return cand
    return None


def _write_detect_yaml(data_dir, names, out_path):
    data_dir = os.path.abspath(data_dir)
    train = os.path.join(data_dir, "train", "images")
    val = os.path.join(data_dir, "valid", "images")
    if not os.path.isdir(train):
        train = os.path.join(data_dir, "train")
    if not os.path.isdir(val):
        val = os.path.join(data_dir, "valid")
    if not os.path.isdir(train):
        raise SystemExit(f"No train images under {data_dir!r}.")
    if not os.path.isdir(val):
        raise SystemExit(f"No valid images under {data_dir!r}.")

    with open(out_path, "w") as f:
        f.write(f"train: {train}\n")
        f.write(f"val: {val}\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for i, name in enumerate(names):
            f.write(f"  {i}: {name}\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download / train football player YOLO detector.")
    parser.add_argument("--download", action="store_true",
                        help="Download the Roboflow players dataset first.")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION,
                        help=f"Roboflow dataset version (default {DEFAULT_VERSION}).")
    parser.add_argument("--data-dir", default="",
                        help="YOLO detect dataset dir (ignored with --download "
                             "unless --location is set).")
    parser.add_argument("--location", default="",
                        help="Where to place the download "
                             "(default: <repo>/football-players-detection-<version>).")
    parser.add_argument("--model", default="yolov8m.pt",
                        help="Pretrained detect checkpoint "
                             "(yolov8n/s/m/l/x.pt — NOT *-pose).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--names", nargs="+", default=DEFAULT_NAMES,
                        help="Class names if rewriting yaml "
                             f"(default: {' '.join(DEFAULT_NAMES)}).")
    parser.add_argument("--name", default="train",
                        help="Ultralytics run name under runs/player/.")
    args = parser.parse_args()

    if args.download:
        if not args.api_key:
            raise SystemExit("--download needs ROBOFLOW_API_KEY (or --api-key).")
        dest = args.location or args.data_dir or os.path.join(
            _ROOT, f"football-players-detection-{args.version}")
        data_dir = _download_dataset(
            args.api_key, args.workspace, args.project, args.version,
            location=dest)
        print(f"downloaded: {data_dir}")
    else:
        data_dir = args.data_dir
        if not data_dir or not os.path.isdir(data_dir):
            raise SystemExit(
                "Pass --data-dir <YOLO dataset dir> or --download to fetch one.")

    data_yaml = _find_data_yaml(data_dir)
    if data_yaml is None:
        data_yaml = _write_detect_yaml(
            data_dir, args.names,
            os.path.join(data_dir, "pitch_view_players.yaml"))
        print(f"wrote: {data_yaml}")
    else:
        print(f"using: {data_yaml}")

    if "pose" in os.path.basename(args.model).lower():
        raise SystemExit(
            f"--model {args.model!r} looks like a pose checkpoint. "
            "Player detection needs a detect model, e.g. yolov8m.pt.")

    from ultralytics import YOLO

    # Ultralytics logs → runs/player/<name>; packaged weight → weights/players.pt
    os.chdir(_ROOT)
    model = YOLO(args.model)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=os.path.join("runs", "player"),
        name=args.name,
        exist_ok=True,
    )
    best = resolve_trained_best(model.trainer)
    out = publish_weight(best, players_path())
    print(f"Training finished. Published: {out}")
    print(f"(Ultralytics run also kept at {model.trainer.save_dir})")


if __name__ == "__main__":
    main()
