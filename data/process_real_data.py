import os
from pathlib import Path
from PIL import Image

def verify_real_uav_dataset(data_dir: str = "data/floodnet_sample"):
    """
    Scans data/floodnet_sample for real UAV aerial images, 
    verifies image integrity, and logs dataset statistics.
    """
    dataset_path = Path(data_dir)
    if not dataset_path.exists():
        dataset_path.mkdir(parents=True, exist_ok=True)
        print(f"[!] Created directory {dataset_path.resolve()}. Please place real images inside.")
        return []

    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".tif"}
    image_files = [f for f in dataset_path.iterdir() if f.suffix.lower() in valid_extensions]

    if not image_files:
        print(f"[!] No real images found in {dataset_path.resolve()}.")
        print("[!] Download real FloodNet/UAV images and save them in this folder.")
        return []

    print(f"\n================ FloodNet Dataset Inspection ================")
    print(f"Target Directory : {dataset_path.resolve()}")
    print(f"Total Images Found: {len(image_files)}\n")

    valid_images = []
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                format_type = img.format
                mode = img.mode
                print(f"[+] Loaded: {img_path.name} | Res: {width}x{height} | Mode: {mode} | Format: {format_type}")
                valid_images.append(img_path)
        except Exception as e:
            print(f"[!] Corrupted image file skipped: {img_path.name} ({e})")

    print(f"==============================================================")
    print(f"[+] Verification Complete. {len(valid_images)} valid image(s) ready for baseline run.\n")
    return valid_images

if __name__ == "__main__":
    verify_real_uav_dataset()