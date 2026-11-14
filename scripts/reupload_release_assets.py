"""
Re-upload genuine local model checkpoints to GitHub Release v1.0.0.
Deletes the stale assets first, then streams the local weights via GitHub API.
"""

import os
import sys
import json
import subprocess
import urllib.request
import urllib.error

RELEASE_ID = 445074181
OWNER = "Mradul-Consultancy"
REPO = "DivyaDrishti"

FILES = [
    "models/checkpoints/resnet18_best.pth",
    "models/checkpoints/efficientnet_b3.ra2_in1k_best.pth",
    "models/checkpoints/convnext_tiny.fb_in22k_ft_in1k_best.pth",
]

def get_token():
    p = subprocess.Popen(
        ["git", "credential-wincred", "get"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, _ = p.communicate("protocol=https\nhost=github.com\n\n")
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    return os.environ.get("GITHUB_TOKEN", "")

def get_existing_assets(token):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/{RELEASE_ID}/assets"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "DivyaDrishti-Reuploader"
        }
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def delete_asset(asset_id, token):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/assets/{asset_id}"
    req = urllib.request.Request(
        url,
        method="DELETE",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "DivyaDrishti-Reuploader"
        }
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status

def upload_asset(rel_path, token):
    filename = os.path.basename(rel_path)
    size_bytes = os.path.getsize(rel_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Uploading {filename} ({size_mb:.2f} MB, {size_bytes} bytes)...")
    
    upload_url = f"https://uploads.github.com/repos/{OWNER}/{REPO}/releases/{RELEASE_ID}/assets?name={filename}"
    
    with open(rel_path, "rb") as f:
        data = f.read()

    req = urllib.request.Request(upload_url, data=data, method="POST")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Content-Type", "application/octet-stream")
    req.add_header("User-Agent", "DivyaDrishti-Reuploader")
    req.add_header("Content-Length", str(len(data)))

    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"  -> SUCCESS! Asset ID {res.get('id')} ({res.get('size')} bytes)")
        return res

def main():
    token = get_token()
    if not token:
        print("ERROR: GitHub token could not be retrieved from wincred or env.")
        sys.exit(1)

    print("Checking current assets in Release v1.0.0...")
    existing = get_existing_assets(token)
    asset_map = {a["name"]: a["id"] for a in existing}
    print(f"Found {len(existing)} assets: {list(asset_map.keys())}")

    for rel_path in FILES:
        filename = os.path.basename(rel_path)
        if not os.path.exists(rel_path):
            print(f"ERROR: Local file {rel_path} does not exist!")
            continue

        if filename in asset_map:
            old_id = asset_map[filename]
            print(f"Deleting stale remote asset {filename} (ID: {old_id})...")
            delete_asset(old_id, token)
            print(f"  -> Deleted {filename}")

        upload_asset(rel_path, token)

    print("\nVerifying re-uploaded assets...")
    new_assets = get_existing_assets(token)
    all_ok = True
    for a in new_assets:
        local_p = os.path.join("models", "checkpoints", a["name"])
        if os.path.exists(local_p):
            local_sz = os.path.getsize(local_p)
            match = (local_sz == a["size"])
            print(f"Verified {a['name']}: Remote={a['size']} vs Local={local_sz} | Match={match}")
            if not match:
                all_ok = False

    if all_ok:
        print("\nALL RELEASE ASSETS SUCCESSFULLY RE-UPLOADED AND VERIFIED BYTE-FOR-BYTE!")
    else:
        print("\nWARNING: Some sizes did not match.")

if __name__ == "__main__":
    main()
