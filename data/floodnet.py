"""FloodNet-Supervised loaders with a frozen 1,445 / 450 / 448 split.

Expected on-disk layout (the official release):

    <root>/train/train-org-img/<id>.jpg     <root>/train/train-label-img/<id>_lab.png
    <root>/val/val-org-img/<id>.jpg         <root>/val/val-label-img/<id>_lab.png
    <root>/test/test-org-img/<id>.jpg       <root>/test/test-label-img/<id>_lab.png

Label PNGs are single-channel class ids 0-9. The split is defined by the
official folders. This module checks the counts, checks that every image has
a label, and checks that no id appears in two splits. `--build-manifest`
freezes the file lists in data/splits/floodnet_split.json; commit that file.
Later loads verify against it, so the split cannot drift silently.

CLI:
    python -m data.floodnet --root /path/to/FloodNet-Supervised_v1.0 --build-manifest
    python -m data.floodnet --root /path/to/FloodNet-Supervised_v1.0 --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

EXPECTED_COUNTS: Dict[str, int] = {"train": 1445, "val": 450, "test": 448}
SPLITS = tuple(EXPECTED_COUNTS)
MANIFEST_PATH = Path(__file__).resolve().parent / "splits" / "floodnet_split.json"

CLASS_NAMES = [
    "background", "building-flooded", "building-non-flooded", "road-flooded",
    "road-non-flooded", "water", "tree", "vehicle", "pool", "grass",
]
NUM_CLASSES = len(CLASS_NAMES)

BUILDING_CLASSES = (1, 2)
ROAD_CLASSES = (3, 4)
ROAD_FLOODED = 3

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class SplitError(RuntimeError):
    """The dataset on disk does not match the frozen split."""


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    label_path: Path


def _split_dirs(root: Path, split: str) -> Tuple[Path, Path]:
    return root / split / f"{split}-org-img", root / split / f"{split}-label-img"


def _index_split(root: Path, split: str) -> List[Sample]:
    img_dir, lab_dir = _split_dirs(root, split)
    if not img_dir.is_dir() or not lab_dir.is_dir():
        raise SplitError(f"missing folder for '{split}': expected {img_dir} and {lab_dir}")
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    samples, missing = [], []
    for p in images:
        lab = lab_dir / f"{p.stem}_lab.png"
        if lab.is_file():
            samples.append(Sample(p.stem, p, lab))
        else:
            missing.append(p.name)
    if missing:
        raise SplitError(f"{len(missing)} '{split}' images have no label, e.g. {missing[:3]}")
    return samples


def index_all(root, expected: Optional[Dict[str, int]] = EXPECTED_COUNTS) -> Dict[str, List[Sample]]:
    """Index every split, then enforce counts and disjointness."""
    root = Path(root)
    out = {s: _index_split(root, s) for s in SPLITS}
    if expected is not None:
        for s in SPLITS:
            if len(out[s]) != expected[s]:
                raise SplitError(
                    f"'{s}' has {len(out[s])} images, expected {expected[s]}. "
                    "Check the dataset version before running any experiment.")
    seen: Dict[str, str] = {}
    for s in SPLITS:
        for smp in out[s]:
            if smp.image_id in seen:
                raise SplitError(f"id {smp.image_id} appears in both '{seen[smp.image_id]}' and '{s}'")
            seen[smp.image_id] = s
    return out


def _digest(ids: List[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def build_manifest(root, path: Path = MANIFEST_PATH,
                   expected: Optional[Dict[str, int]] = EXPECTED_COUNTS) -> dict:
    idx = index_all(root, expected)
    manifest = {
        "dataset": "FloodNet-Supervised v1.0",
        "counts": {s: len(v) for s, v in idx.items()},
        "sha256": {s: _digest([x.image_id for x in v]) for s, v in idx.items()},
        "ids": {s: [x.image_id for x in v] for s, v in idx.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def verify_manifest(idx: Dict[str, List[Sample]], path: Path = MANIFEST_PATH) -> bool:
    """Compare indexed splits with the frozen manifest. False if no manifest yet."""
    if not path.is_file():
        return False
    m = json.loads(path.read_text(encoding="utf-8"))
    for s in SPLITS:
        ids = [x.image_id for x in idx[s]]
        if _digest(ids) != m["sha256"][s]:
            raise SplitError(f"'{s}' file list differs from {path.name}; the split has changed")
    return True


def read_label(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        if arr.shape[2] > 1 and not np.all(arr[..., 0] == arr[..., 1]):
            raise ValueError(f"{path.name} looks colour-coded (RGB palette), not class ids; "
                             "decode it to ids 0-9 first")
        arr = arr[..., 0]
    arr = arr.astype(np.uint8)
    if arr.max() >= NUM_CLASSES:
        raise ValueError(f"{path.name} has class id {int(arr.max())}, expected 0-{NUM_CLASSES - 1}")
    return arr


class FloodNetDataset:
    """Indexable view of one split. Returns dicts with RGB image and id-map label."""

    def __init__(self, root, split: str, expected: Optional[Dict[str, int]] = EXPECTED_COUNTS,
                 manifest_path: Path = MANIFEST_PATH):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        idx = index_all(root, expected)
        self.manifest_verified = verify_manifest(idx, manifest_path) if expected is EXPECTED_COUNTS else False
        self.strict = expected is EXPECTED_COUNTS
        self.split = split
        self.samples = idx[split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        image = np.array(Image.open(s.image_path).convert("RGB"))
        label = read_label(s.label_path)
        if image.shape[:2] != label.shape:
            raise ValueError(f"{s.image_id}: image {image.shape[:2]} and label {label.shape} differ")
        return {"id": s.image_id, "image": image, "label": label}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def road_mask(label: np.ndarray) -> np.ndarray:
    return np.isin(label, ROAD_CLASSES)


def blocked_road_mask(label: np.ndarray) -> np.ndarray:
    return label == ROAD_FLOODED


def building_mask(label: np.ndarray) -> np.ndarray:
    return np.isin(label, BUILDING_CLASSES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--build-manifest", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.build_manifest:
        m = build_manifest(a.root)
        print("manifest written:", m["counts"])
    if a.verify or not a.build_manifest:
        idx = index_all(a.root)
        print("counts OK:", {s: len(v) for s, v in idx.items()})
        print("manifest verified" if verify_manifest(idx) else "no manifest yet (run --build-manifest)")


if __name__ == "__main__":
    main()
