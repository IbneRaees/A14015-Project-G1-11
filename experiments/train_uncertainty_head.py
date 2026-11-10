"""Train the per-pixel uncertainty head for Gap 2 on frozen YOLOv8x-Seg neck features.

Two heads share the same body and backbone features:
  evidential : Dirichlet head trained with the EDL loss, KL weight annealed 0 -> 1
  softmax    : plain cross-entropy head, then post-hoc temperature scaling fitted on val

Only the train and val splits are read. Val picks the best epoch (by mIoU) and
fits the temperature; the test split is never opened here.

    python -m experiments.train_uncertainty_head --head evidential --seed 42
    python -m experiments.train_uncertainty_head --head softmax --seed 42

The backbone is the E1 fine-tuned YOLOv8x-Seg for the same seed (config:
`yolo_weights`). Its weights stay frozen; only the head is trained. Caveat: that
backbone was trained on the same 1,445 images, so train-set features are cleaner
than val/test ones. Both heads are affected equally, and all reported calibration
numbers come from val/test.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.floodnet import EXPECTED_COUNTS, NUM_CLASSES, FloodNetDataset
from experiments.common import git_commit, load_config
from src.components.calibration import (CalibrationAccumulator, RiskCoverageAccumulator, aurc,
                                        fit_temperature)
from src.components.evidential import edl_loss, kl_anneal
from src.components.heads import HEAD_KINDS, build_head, load_head, save_head
from src.components.yolo_features import (YoloNeckFeatures, downsample_labels_stride, fuse_features,
                                          fused_channels, resize_for_yolo, resize_label)
from src.utils.metrics import confusion_matrix, summarize


# ---------------------------------------------------------------- data
class FloodNetHeadDataset(Dataset):
    """Resized (image, label) pairs. Images are uint8 HWC, labels int64 HW."""

    def __init__(self, ds: FloodNetDataset, imgsz: int, augment: bool, max_items: Optional[int] = None):
        self.ds, self.imgsz, self.augment = ds, imgsz, augment
        self.n = len(ds) if max_items is None else min(max_items, len(ds))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        item = self.ds[i]
        img, hw = resize_for_yolo(item["image"], self.imgsz)
        lab = resize_label(item["label"], hw)
        if self.augment:  # aerial imagery has no preferred orientation
            if torch.rand(()) < 0.5:
                img, lab = img[:, ::-1], lab[:, ::-1]
            if torch.rand(()) < 0.5:
                img, lab = img[::-1], lab[::-1]
        return (torch.from_numpy(np.ascontiguousarray(img)),
                torch.from_numpy(np.ascontiguousarray(lab)).long())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- model glue
def head_forward(extractor: YoloNeckFeatures, head, x: torch.Tensor, levels, out_stride: int, amp: bool):
    """Frozen backbone -> fused neck features -> head output at input/out_stride resolution."""
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=amp and x.is_cuda):
            feats = extractor.extract(x)
        fused = fuse_features([f.float() for f in feats], levels)
    h, w = x.shape[-2:]
    return head(fused, (h // out_stride, w // out_stride))


def to_input(imgs_u8: torch.Tensor, device) -> torch.Tensor:
    return imgs_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0


def to_probs(kind: str, out: torch.Tensor, temperature: float = 1.0):
    """Class probabilities and (for the evidential head) vacuity."""
    if kind == "evidential":
        s = out.sum(1, keepdim=True)
        return out / s, out.shape[1] / s.squeeze(1)
    return F.softmax(out / temperature, dim=1), None


def normalized_entropy_t(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(1e-12))).sum(1) / math.log(p.shape[1])


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(extractor, head, loader, kind: str, cfg: dict, device, temperature: float = 1.0) -> dict:
    head.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    cal = CalibrationAccumulator(15)
    rc_ent = RiskCoverageAccumulator()
    rc_vac = RiskCoverageAccumulator() if kind == "evidential" else None
    loss_sum, n_img, conf_sum, n_pix = 0.0, 0, 0.0, 0
    for imgs, labels in loader:
        out = head_forward(extractor, head, to_input(imgs, device), cfg["levels"], cfg["out_stride"], cfg["amp"])
        lab = downsample_labels_stride(labels, cfg["out_stride"]).to(device)
        loss = edl_loss(out, lab, anneal=1.0) if kind == "evidential" else F.cross_entropy(out, lab)
        loss_sum += float(loss) * len(imgs)
        n_img += len(imgs)
        probs, vac = to_probs(kind, out, temperature)
        conf, pred = probs.max(1)
        wrong = (pred != lab).cpu().numpy()
        cm += confusion_matrix(lab.cpu().numpy().astype(np.uint8), pred.cpu().numpy().astype(np.uint8))
        c = conf.cpu().numpy()
        cal.update(c, ~wrong)
        rc_ent.update(normalized_entropy_t(probs).cpu().numpy(), wrong)
        if rc_vac is not None:
            rc_vac.update(vac.cpu().numpy(), wrong)
        conf_sum += float(c.sum())
        n_pix += c.size
    s = summarize(cm)
    return {
        "loss": loss_sum / max(n_img, 1),
        "pixel_accuracy": s["pixel_accuracy"], "miou_all": s["miou_all"],
        "miou_foreground": s["miou_foreground"],
        "ece": cal.ece(), "mean_confidence": conf_sum / n_pix,
        "aurc_entropy": aurc(*rc_ent.curve()),
        "aurc_vacuity": None if rc_vac is None else aurc(*rc_vac.curve()),
    }


@torch.no_grad()
def collect_logits(extractor, head, loader, cfg: dict, device, per_image: int, seed: int):
    """Random pixel sample of validation logits and labels, for fitting the temperature."""
    head.eval()
    rng = np.random.default_rng(seed)
    zs, ys = [], []
    for imgs, labels in loader:
        out = head_forward(extractor, head, to_input(imgs, device), cfg["levels"], cfg["out_stride"], cfg["amp"])
        lab = downsample_labels_stride(labels, cfg["out_stride"]).to(device)
        b, k, h, w = out.shape
        flat, flab = out.reshape(b, k, -1), lab.reshape(b, -1)
        for i in range(b):
            idx = torch.from_numpy(rng.choice(h * w, size=min(per_image, h * w), replace=False)).to(device)
            zs.append(flat[i][:, idx].T.float().cpu().numpy())
            ys.append(flab[i][idx].cpu().numpy())
    return np.concatenate(zs), np.concatenate(ys)


# ---------------------------------------------------------------- training
def train(cfg: dict, kind: str, seed: int, out_dir, model: Optional[torch.nn.Module] = None,
          expected: Optional[dict] = EXPECTED_COUNTS) -> dict:
    """Train one head. `model` is the YOLO torch module; if None it is loaded from cfg['yolo_weights']."""
    assert kind in HEAD_KINDS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = torch.device("cpu" if str(cfg["device"]) == "cpu" else f"cuda:{cfg['device']}")

    yolo_weights = cfg["yolo_weights"].format(seed=seed)
    if model is None:
        if not Path(yolo_weights).is_file():
            raise FileNotFoundError(
                f"\n[!] Required YOLO fine-tuned weights not found: '{yolo_weights}'\n"
                f"[!] Prerequisite missing: You must execute Stage 1 first to train the baseline model:\n"
                f"    python -m experiments.train_baseline --seed {seed} --epochs 5\n"
            )
        from ultralytics import YOLO
        model = YOLO(yolo_weights).model
    model = model.to(device).eval().requires_grad_(False)
    extractor = YoloNeckFeatures(model)
    channels = extractor.channels(device)
    in_ch = fused_channels(channels, cfg["levels"])

    head = build_head(kind, in_ch, NUM_CLASSES, cfg["hidden"]).to(device)
    train_ds = FloodNetHeadDataset(FloodNetDataset(cfg["data_root"], "train", expected), cfg["imgsz"], True)
    val_ds = FloodNetHeadDataset(FloodNetDataset(cfg["data_root"], "val", expected), cfg["imgsz"], False,
                                 cfg.get("val_max_images"))
    g = torch.Generator().manual_seed(seed)
    nw = int(cfg["num_workers"])
    loader_kw = dict(num_workers=nw, persistent_workers=nw > 0, pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, cfg["batch"], shuffle=True, generator=g, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, cfg["batch"], shuffle=False, **loader_kw)

    opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    meta = {"head": kind, "in_channels": in_ch, "num_classes": NUM_CLASSES, "hidden": cfg["hidden"],
            "levels": list(cfg["levels"]), "out_stride": cfg["out_stride"], "imgsz": cfg["imgsz"],
            "yolo_weights": yolo_weights, "seed": seed, "anneal_epochs": cfg["anneal_epochs"],
            "temperature": 1.0, "strict_split": expected is EXPECTED_COUNTS}

    history, best, t0 = [], -1.0, time.time()
    for epoch in range(cfg["epochs"]):
        head.train()
        run_loss, steps = 0.0, len(train_dl)
        for it, (imgs, labels) in enumerate(train_dl):
            out = head_forward(extractor, head, to_input(imgs, device), cfg["levels"], cfg["out_stride"], cfg["amp"])
            lab = downsample_labels_stride(labels, cfg["out_stride"]).to(device)
            if kind == "evidential":
                loss = edl_loss(out, lab, anneal=kl_anneal(epoch + it / steps, cfg["anneal_epochs"]))
            else:
                loss = F.cross_entropy(out, lab)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {it}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
            opt.step()
            run_loss += float(loss.detach())
        sched.step()
        val = evaluate(extractor, head, val_dl, kind, cfg, device)
        rec = {"epoch": epoch + 1, "train_loss": run_loss / max(steps, 1), "val": val,
               "kl_weight": kl_anneal(epoch + 1, cfg["anneal_epochs"]) if kind == "evidential" else None,
               "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        score = val["miou_all"] if val["miou_all"] is not None else -1.0
        if score > best:
            best = score
            save_head(out_dir / "best.pt", head, {**meta, "best_epoch": epoch + 1, "val_metrics": val})
        (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        print(f"[{kind} seed{seed}] epoch {epoch + 1}/{cfg['epochs']} loss {rec['train_loss']:.4f} "
              f"val mIoU {score:.4f} ECE {val['ece']:.4f}", flush=True)

    ck_head, ck_meta = load_head(out_dir / "best.pt", device)          # restore the best epoch
    ck_head = ck_head.to(device)
    summary = {"head": kind, "seed": seed, "best_epoch": ck_meta["best_epoch"],
               "val_metrics": ck_meta["val_metrics"], "git_commit": git_commit(),
               "strict_split": meta["strict_split"], "temperature": None}
    if kind == "softmax":
        z, y = collect_logits(extractor, ck_head, val_dl, cfg, device,
                              cfg["temperature_pixels_per_image"], seed)
        t = fit_temperature(z, y)
        after = evaluate(extractor, ck_head, val_dl, kind, cfg, device, temperature=t)
        summary.update({"temperature": t, "val_metrics_after_temperature": after})
        save_head(out_dir / "best.pt", ck_head, {**ck_meta, "temperature": t})
        print(f"temperature T={t:.3f}: val ECE {ck_meta['val_metrics']['ece']:.4f} -> {after['ece']:.4f} "
              "(fitted on val, so in-sample)", flush=True)
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs" / "head.yaml"))
    ap.add_argument("--head", choices=HEAD_KINDS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--yolo-weights", default=None, help="path, may contain {seed}")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--anneal-epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--val-max-images", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke-counts", default=None, help="train,val,test counts of a fake dataset; smoke tests only")
    a = ap.parse_args()
    cfg = load_config(a.config, yolo_weights=a.yolo_weights, data_root=a.data_root, epochs=a.epochs,
                      batch=a.batch, lr=a.lr, anneal_epochs=a.anneal_epochs, device=a.device,
                      num_workers=a.num_workers, val_max_images=a.val_max_images)
    expected = EXPECTED_COUNTS if a.smoke_counts is None else dict(
        zip(("train", "val", "test"), map(int, a.smoke_counts.split(","))))
    out_dir = a.out_dir or str(Path(cfg["out_root"]) / f"{a.head}_seed{a.seed}")
    train(cfg, a.head, a.seed, out_dir, expected=expected)


if __name__ == "__main__":
    main()
