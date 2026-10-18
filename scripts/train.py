#!/usr/bin/env python3
"""
Train / fine-tune a football pitch keypoint model (YOLOv8-pose).

Either downloads the public Roboflow dataset first (``--download``), or uses an
already-downloaded YOLO-format dataset directory (``--data-dir``).

Examples:
    export ROBOFLOW_API_KEY=...
    python scripts/train.py --download --epochs 100 --model yolov8m-pose.pt

    python scripts/train.py --data-dir /path/to/dataset --epochs 100

Trained ``best.pt`` is published to ``weights/pitch_keypoints.pt`` (same folder
as inference). Intermediate Ultralytics logs stay under ``runs/pose/``.

The 32 pitch keypoints must stay in the order defined by
``SoccerPitchConfiguration.vertices`` (the public dataset already matches).
"""
import argparse
import os
import shlex
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from pitch_view.common.weights import (
    pitch_keypoints_path,
    publish_weight,
    resolve_trained_best,
)

DEFAULT_WORKSPACE = "roboflow-jvuqo"
DEFAULT_PROJECT = "football-field-detection-f07vi"
DEFAULT_VERSION = 14


def _download_dataset(api_key, workspace, project, version):
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download("yolov8")
    return ds.location


def _write_data_yaml(data_dir, kpt_shape, names, out_path):
    # Ultralytics resolves yaml-relative paths against the yaml's own directory,
    # so use absolute paths to avoid double-prefixing when the yaml lives inside
    # the dataset dir.
    data_dir = os.path.abspath(data_dir)
    train = os.path.join(data_dir, "train", "images")
    val = os.path.join(data_dir, "valid", "images")
    if not os.path.isdir(train):
        train = os.path.join(data_dir, "train")
    if not os.path.isdir(val):
        val = os.path.join(data_dir, "valid")

    with open(out_path, "w") as f:
        f.write(f"train: {train}\n")
        f.write(f"val: {val}\n")
        f.write(f"kpt_shape: [{kpt_shape[0]}, {kpt_shape[1]}]\n")
        f.write("names:\n")
        for i, name in enumerate(names):
            f.write(f"  {i}: {name}\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download the public Roboflow dataset first.")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--data-dir", default="",
                        help="Path to a YOLO-format dataset (ignored with --download).")
    parser.add_argument("--model", default="yolov8m-pose.pt",
                        help="Pretrained pose checkpoint (n/s/m/l/x).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--run-dir", default=os.path.join(_ROOT, "runs", "pose"),
                        help="Ultralytics run directory (default: <repo>/runs/pose).")
    parser.add_argument("--name", default="train",
                        help="Run name under --run-dir (default: train).")
    parser.add_argument("--no-publish", action="store_true",
                        help="Keep best.pt under --run-dir and do not copy it to weights/.")
    parser.add_argument("--kpt-shape", nargs=2, type=int, default=[32, 3],
                        metavar=("N", "DIM"),
                        help="Keypoint shape [num_keypoints, dim] (default 32 3).")
    parser.add_argument("--names", nargs="+", default=["pitch"],
                        help="Class name(s) (default: pitch).")
    args = parser.parse_args()

    if args.download:
        if not args.api_key:
            raise SystemExit("--download needs ROBOFLOW_API_KEY (or --api-key).")
        data_dir = _download_dataset(
            args.api_key, args.workspace, args.project, args.version)
    else:
        data_dir = args.data_dir
        if not data_dir or not os.path.isdir(data_dir):
            raise SystemExit(
                "Pass --data-dir <YOLO dataset dir> or --download to fetch one.")

    data_yaml = _write_data_yaml(
        data_dir, args.kpt_shape, args.names,
        os.path.join(data_dir, "pitch_view_data.yaml"))
    print(f"data.yaml: {data_yaml}")

    run_dir = os.path.abspath(args.run_dir)
    save_dir = os.path.join(run_dir, args.name)
    os.makedirs(save_dir, exist_ok=True)
    command = shlex.join(sys.argv)
    command_path = os.path.join(save_dir, "command.txt")
    with open(command_path, "w") as f:
        f.write(command + "\n")
    print(f"command: {command}")
    print(f"command saved: {command_path}")

    from ultralytics import YOLO

    os.chdir(_ROOT)
    model = YOLO(args.model)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=run_dir,
        name=args.name,
        exist_ok=True,
    )
    best = resolve_trained_best(model.trainer)
    if args.no_publish:
        print(f"Training finished. Weights kept at {best}")
    else:
        out = publish_weight(best, pitch_keypoints_path())
        print(f"Training finished. Published: {out}")
    print(f"(Ultralytics run kept at {model.trainer.save_dir})")


if __name__ == "__main__":
    main()
