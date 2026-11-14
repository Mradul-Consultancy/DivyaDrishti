import os
import sys
import json
import urllib.request
import urllib.error

TOKEN = os.environ.get("GITHUB_TOKEN", "")
UPLOAD_BASE = "https://uploads.github.com/repos/Mradul-Consultancy/DivyaDrishti/releases/445074181/assets"

FILES = [
    "models/checkpoints/resnet18_best.pth",
    "models/checkpoints/efficientnet_b3.ra2_in1k_best.pth",
    "models/checkpoints/convnext_tiny.fb_in22k_ft_in1k_best.pth",
]

def upload():
    for rel_path in FILES:
        if not os.path.exists(rel_path):
            print(f"Skipping {rel_path} (not found)")
            continue

        filename = os.path.basename(rel_path)
        size_mb = os.path.getsize(rel_path) / (1024 * 1024)
        print(f"Preparing {filename} ({size_mb:.1f} MB)...")
        upload_url = f"{UPLOAD_BASE}?name={filename}"

        with open(rel_path, "rb") as f:
            data = f.read()

        req = urllib.request.Request(upload_url, data=data, method="POST")
        req.add_header("Authorization", f"token {TOKEN}")
        req.add_header("Content-Type", "application/octet-stream")
        req.add_header("User-Agent", "DivyaDrishti-Uploader")

        try:
            with urllib.request.urlopen(req) as resp:
                res = json.loads(resp.read().decode())
                print(f"SUCCESS: {filename} -> {res.get('browser_download_url')}")
        except urllib.error.HTTPError as e:
            err = e.read().decode()
            print(f"FAILED {filename}: {e.code} - {err}")
        except Exception as e:
            print(f"FAILED {filename}: {e}")

if __name__ == "__main__":
    upload()

