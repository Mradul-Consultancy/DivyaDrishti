import json
import os
import subprocess
import urllib.request

def get_token():
    p = subprocess.Popen(["git", "credential-wincred", "get"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, _ = p.communicate("protocol=https\nhost=github.com\n\n")
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    return os.environ.get("GITHUB_TOKEN", "")

def inspect_assets():
    token = get_token()
    url = "https://api.github.com/repos/Mradul-Consultancy/DivyaDrishti/releases/tags/v1.0.0"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "DivyaDrishti-AssetManager"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print(f"Release ID: {data['id']}")
        for a in data.get("assets", []):
            size_mb = a['size'] / (1024 * 1024)
            print(f"Asset ID: {a['id']} | Name: {a['name']} | Size: {size_mb:.2f} MB | Created: {a['created_at']}")

if __name__ == "__main__":
    inspect_assets()
