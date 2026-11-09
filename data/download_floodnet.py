"""Download, extract, and prepare the official FloodNet-Supervised_v1.0 dataset.

Workflow:
1. Downloads the official 12.16 GB release (FloodNet.zip) from Dropbox (BinaLab official repository).
2. Extracts and reorganizes into standard layout:
       train/train-org-img, train/train-label-img
       val/val-org-img, val/val-label-img
       test/test-org-img, test/test-label-img
3. Automatically deletes the zip archive if requested (--clean-zip) or if in /tmp to preserve disk space.
4. Freezes the 1,445 / 450 / 448 manifest (data/splits/floodnet_split.json).
5. Converts train/val masks into YOLOv8 polygon instances (datasets/floodnet_yolo/).

Usage:
    python data/download_floodnet.py
    python data/download_floodnet.py --out-dir /tmp/FloodNet-Supervised_v1.0 --zip-path /tmp/FloodNet.zip --clean-zip
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# Official Dropbox link from BinaLab/FloodNet-Supervised_v1.0
DROPBOX_URL = (
    "https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc"
    "?rlkey=ao2493wzl1cltonowjdbrnp7f&dl=1"
)

DEFAULT_DEST = Path("data/FloodNet-Supervised_v1.0")
DEFAULT_ZIP = Path("data/FloodNet.zip")
DEFAULT_YOLO_DIR = Path("datasets/floodnet_yolo")


def download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download file with live transfer progress."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 10 * 1024 * 1024 * 1024:
        print(f"[+] Archive already exists ({output_path.stat().st_size / (1024**3):.2f} GB): {output_path}")
        return output_path

    print(f"[+] Connecting to download source: {url[:60]}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("content-length", 0))
        total_gb = total_size / (1024**3) if total_size else 0
        print(f"[+] Downloading FloodNet archive ({total_gb:.2f} GB)...")

        downloaded = 0
        start_time = time.time()
        last_log = start_time

        with open(output_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                now = time.time()
                if now - last_log >= 2.0 or downloaded == total_size:
                    elapsed = now - start_time
                    speed = (downloaded / (1024 * 1024)) / (elapsed if elapsed > 0 else 1e-5)
                    pct = (downloaded / total_size * 100) if total_size else 0
                    eta_sec = (total_size - downloaded) / (speed * 1024 * 1024) if speed > 0 and total_size else 0
                    print(
                        f"\r[-] {downloaded / (1024**3):.2f}/{total_gb:.2f} GB "
                        f"({pct:5.1f}%) | Speed: {speed:6.2f} MB/s | ETA: {eta_sec/60:4.1f} min",
                        end="",
                        flush=True,
                    )
                    last_log = now
    print("\n[+] Download complete!")
    return output_path


def extract_archive(zip_path: Path, target_dir: Path, clean_zip: bool = False) -> Path:
    """Extract zip archive and ensure standard FloodNet layout."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Extracting {zip_path.name} into {target_dir}...")
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_files = len(members)
        print(f"[+] Total files in archive: {total_files}")
        
        extracted = 0
        t0 = time.time()
        for idx, member in enumerate(members):
            zf.extract(member, target_dir)
            extracted += 1
            if (idx + 1) % 500 == 0 or idx + 1 == total_files:
                elapsed = time.time() - t0
                print(f"\r[-] Extracted {extracted}/{total_files} files ({extracted/total_files*100:.1f}%) in {elapsed:.1f}s", end="", flush=True)
    print("\n[+] Extraction finished!")
    
    if clean_zip:
        try:
            zip_path.unlink()
            print(f"[+] Removed archive {zip_path} to reclaim disk space.")
        except Exception as e:
            print(f"[!] Warning: Could not delete zip {zip_path}: {e}")

    return normalize_layout(target_dir)


def normalize_layout(target_dir: Path) -> Path:
    """Finds train-org-img and ensures target_dir contains train/, val/, test/ directly."""
    matches = list(target_dir.rglob("train-org-img"))
    if not matches:
        if (target_dir / "train" / "train-org-img").is_dir():
            print(f"[+] Layout verified at: {target_dir.resolve()}")
            return target_dir
        raise RuntimeError(f"Could not find 'train-org-img' inside {target_dir.resolve()}")

    actual_root = matches[0].parent.parent
    if actual_root.resolve() == target_dir.resolve():
        print(f"[+] Layout correctly aligned at: {target_dir.resolve()}")
        return target_dir

    print(f"[-] Moving contents from {actual_root} to {target_dir}...")
    for split in ("train", "val", "test"):
        src_split = actual_root / split
        dst_split = target_dir / split
        if src_split.is_dir() and not dst_split.is_dir():
            shutil.move(str(src_split), str(dst_split))
            print(f"[+] Moved {split}/ to {target_dir}")

    return target_dir


def prepare_supervised(root_dir: Path, yolo_dir: Path):
    """Executes Stage 0 manifest generation, verification, and YOLOv8 dataset preparation."""
    python_exe = sys.executable
    print("\n" + "=" * 60)
    print("Stage 0: Building & Verifying Split Manifest (1445 / 450 / 448)")
    print("=" * 60)
    
    cmd_build = [python_exe, "-m", "data.floodnet", "--root", str(root_dir), "--build-manifest"]
    print(f"Executing: {' '.join(cmd_build)}")
    res1 = subprocess.run(cmd_build)
    if res1.returncode != 0:
        raise RuntimeError("Failed to build split manifest.")

    cmd_verify = [python_exe, "-m", "data.floodnet", "--root", str(root_dir), "--verify"]
    print(f"Executing: {' '.join(cmd_verify)}")
    res2 = subprocess.run(cmd_verify)
    if res2.returncode != 0:
        raise RuntimeError("Split verification failed.")

    print("\n" + "=" * 60)
    print("Stage 0.5: Preparing Supervised YOLOv8 Dataset")
    print("=" * 60)
    cmd_yolo = [python_exe, "-m", "data.prepare_yolo", "--root", str(root_dir), "--out", str(yolo_dir)]
    print(f"Executing: {' '.join(cmd_yolo)}")
    res3 = subprocess.run(cmd_yolo)
    if res3.returncode != 0:
        raise RuntimeError("YOLO dataset preparation failed.")

    print("\n" + "=" * 60)
    print("[+] All Supervised Data Pipeline Preparations Succeeded!")
    print(f"[+] Manifest: data/splits/floodnet_split.json")
    print(f"[+] YOLO Dataset: {yolo_dir.resolve()}/floodnet_seg.yaml")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DROPBOX_URL, help="Download URL")
    parser.add_argument("--zip-path", default=str(DEFAULT_ZIP), help="Target local zip path")
    parser.add_argument("--out-dir", default=str(DEFAULT_DEST), help="Destination extraction directory")
    parser.add_argument("--yolo-dir", default=str(DEFAULT_YOLO_DIR), help="Destination YOLO dataset directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if archive already exists")
    parser.add_argument("--clean-zip", action="store_true", help="Delete zip archive after extraction to save space")
    parser.add_argument("--prep-only", action="store_true", help="Only run manifest check and YOLO preparation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    zip_path = Path(args.zip_path)
    yolo_dir = Path(args.yolo_dir)
    clean_zip = args.clean_zip or "tmp" in str(zip_path).lower()

    if not args.prep_only:
        if not args.skip_download and not (out_dir / "train" / "train-org-img").is_dir():
            download_file(args.url, zip_path)
        
        if not (out_dir / "train" / "train-org-img").is_dir():
            if not zip_path.exists():
                raise FileNotFoundError(f"Archive {zip_path} not found.")
            extract_archive(zip_path, out_dir, clean_zip=clean_zip)
        else:
            print(f"[+] Extracted dataset already present at: {out_dir.resolve()}")

    prepare_supervised(out_dir, yolo_dir)


if __name__ == "__main__":
    main()
