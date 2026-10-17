import os
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw

# Working fallback public sample image links (UAV / Aerial view style)
SAMPLE_IMAGE_URLS = {
    "sample_uav_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Post-Katrina_flooding_in_New_Orleans_2005.jpg/1280px-Post-Katrina_flooding_in_New_Orleans_2005.jpg",
    "sample_uav_2.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/UAV_imagery_flooded_building.jpg/1280px-UAV_imagery_flooded_building.jpg"
}

def create_synthetic_uav_tile(output_path: Path):
    """
    Generates a synthetic UAV aerial sample tile as a local fallback 
    to ensure the pipeline runs even without network connectivity.
    """
    img = Image.new('RGB', (1024, 1024), color=(110, 130, 90)) # Ground base
    draw = ImageDraw.Draw(img)
    
    # Draw flooded area / water body
    draw.polygon([(0, 300), (1024, 500), (1024, 1024), (0, 800)], fill=(60, 100, 140))
    
    # Draw building structures
    draw.rectangle([150, 150, 350, 350], fill=(180, 180, 180), outline=(50, 50, 50), width=4)
    draw.rectangle([600, 200, 850, 400], fill=(140, 70, 60), outline=(30, 30, 30), width=4)
    
    # Draw road network
    draw.line([(0, 100), (1024, 300)], fill=(80, 80, 80), width=40)
    
    img.save(output_path, quality=95)
    print(f"[+] Local fallback synthetic UAV tile generated: {output_path.name}")

def setup_floodnet_pipeline(data_dir: str = "data/floodnet_sample"):
    """
    Initializes local FloodNet dataset directory and retrieves sample UAV imagery.
    """
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    print(f"[+] Setting up FloodNet dataset in: {path.resolve()}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for filename, url in SAMPLE_IMAGE_URLS.items():
        output_path = path / filename
        if not output_path.exists():
            print(f"[-] Fetching sample image: {filename}...")
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"[+] Successfully downloaded: {filename}")
            except Exception as e:
                print(f"[!] Download failed for {filename} ({e}). Generating synthetic UAV sample...")
                create_synthetic_uav_tile(output_path)
        else:
            print(f"[+] File already present: {output_path}")

    # Ensure at least one image exists
    sample_files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
    if len(sample_files) == 0:
        fallback_file = path / "sample_uav_fallback.jpg"
        create_synthetic_uav_tile(fallback_file)
        sample_files = [fallback_file]

    print(f"[+] Verification complete. Total image tiles ready: {len(sample_files)}")
    return path

if __name__ == "__main__":
    setup_floodnet_pipeline()