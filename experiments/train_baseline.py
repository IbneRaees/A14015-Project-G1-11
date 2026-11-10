"""Fine-tune YOLOv8x-Seg on FloodNet train for E1 (validation split picks best.pt).

COCO weights have no road/building/flood classes, so evaluating them directly
would score ~0 on every FloodNet class. The E1 baseline is therefore the same
architecture fine-tuned on the 1,445 training images.

    python -m experiments.train_baseline --seed 42
"""
import argparse
from pathlib import Path

from data.floodnet import EXPECTED_COUNTS
from data.prepare_yolo import prepare
from experiments.common import load_config


def train_seed(seed: int, config: str | None = None, data_root: str | None = None,
               epochs: int | None = None, batch: int | None = None, imgsz: int | None = None,
               device: str | None = None, base_model: str | None = None,
               yolo_dir: str | None = None, project: str | None = None,
               smoke_counts: str | None = None) -> Path:
    """Fine-tune YOLO on FloodNet train for a specific seed. Reuses existing weights if found."""
    cfg = load_config(config, data_root=data_root, base_model=base_model, epochs=epochs,
                      batch=batch, imgsz=imgsz, device=device, yolo_dir=yolo_dir,
                      project=project)
    best_weights = Path(cfg["project"]) / f"seed{seed}" / "weights" / "best.pt"
    if best_weights.is_file():
        print(f"[+] Found existing fine-tuned weights at: {best_weights}")
        return best_weights

    expected = EXPECTED_COUNTS if smoke_counts is None else dict(
        zip(("train", "val", "test"), map(int, smoke_counts.split(","))))

    from ultralytics import YOLO  # imported late so --help works without it

    yaml_path = Path(cfg["yolo_dir"]) / "floodnet_seg.yaml"
    if not yaml_path.is_file():
        yaml_path = prepare(cfg["data_root"], cfg["yolo_dir"], expected)

    model = YOLO(cfg["base_model"])
    model.train(data=str(yaml_path), epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"],
                seed=seed, deterministic=True, device=cfg["device"],
                project=cfg["project"], name=f"seed{seed}", exist_ok=True, plots=False)
    print("best weights:", best_weights)
    return best_weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--yolo-dir", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--smoke-counts", default=None,
                    help="train,val,test counts of a tiny fake dataset; for pipeline smoke tests only")
    a = ap.parse_args()
    train_seed(seed=a.seed, config=a.config, data_root=a.data_root, base_model=a.base_model,
               epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, device=a.device,
               yolo_dir=a.yolo_dir, project=a.project, smoke_counts=a.smoke_counts)


if __name__ == "__main__":
    main()
