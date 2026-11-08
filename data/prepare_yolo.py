"""Convert FloodNet train/val semantic labels to a YOLO segmentation dataset.

Only train and val are written. The test split is never exported, so no
training or model-selection step can touch it.

Each connected region of a class becomes one polygon instance. FloodNet class
ids 1-9 map to YOLO ids 0-8 (background is not a class). Holes are dropped.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import yaml

from data.floodnet import CLASS_NAMES, EXPECTED_COUNTS, FloodNetDataset, read_label
from src.utils.segmentation import mask_to_polygons

POLY_MAX_SIDE = 1024  # polygons are extracted at this scale; coordinates are normalised


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except (OSError, NotImplementedError):  # e.g. Windows without developer mode
        shutil.copy2(src, dst)


def label_to_yolo_lines(label, min_area: int = 20) -> list:
    h, w = label.shape
    s = min(1.0, POLY_MAX_SIDE / max(h, w))
    if s < 1.0:
        label = cv2.resize(label, (round(w * s), round(h * s)), interpolation=cv2.INTER_NEAREST)
    lines = []
    for cid in range(1, len(CLASS_NAMES)):
        for poly in mask_to_polygons(label == cid, min_area=min_area):
            coords = " ".join(f"{v:.5f}" for v in poly.ravel())
            lines.append(f"{cid - 1} {coords}")
    return lines


def prepare(root, out_dir, expected=EXPECTED_COUNTS) -> Path:
    out_dir = Path(out_dir)
    for split in ("train", "val"):
        ds = FloodNetDataset(root, split, expected=expected)
        img_dir, lab_dir = out_dir / "images" / split, out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for s in ds.samples:
            _link_or_copy(s.image_path, img_dir / s.image_path.name)
            lines = label_to_yolo_lines(read_label(s.label_path))
            (lab_dir / f"{s.image_id}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"{split}: {len(ds)} images written")
    yaml_path = out_dir / "floodnet_seg.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i - 1: CLASS_NAMES[i] for i in range(1, len(CLASS_NAMES))},
    }, sort_keys=False), encoding="utf-8")
    return yaml_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="datasets/floodnet_yolo")
    a = ap.parse_args()
    print("dataset yaml:", prepare(a.root, a.out))
