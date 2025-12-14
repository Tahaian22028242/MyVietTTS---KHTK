"""
Download backbone and codec models from Hugging Face to local ./models/ folders.
Usage:
  1) Activate your venv: .\.venv\Scripts\Activate
  2) Ensure huggingface_hub is installed: python -m pip install --upgrade huggingface_hub
  3) Run: python download_models.py

If the repo is private, either run `python -c "from huggingface_hub import login; login()"` first
or set environment variable HUGGINGFACE_HUB_TOKEN.
"""

import sys
import os

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print("ERROR: huggingface_hub is not installed or failed to import.")
    print("Install it inside your active environment with:")
    print("  python -m pip install --upgrade huggingface_hub")
    sys.exit(1)

MODELS = [
    ("neuphonic/neucodec", "models/neucodec"),
]

os.makedirs("models", exist_ok=True)

for repo_id, local_dir in MODELS:
    print(f"\nDownloading {repo_id} -> {local_dir} ...")
    try:
        snapshot_download(repo_id, local_dir=local_dir, repo_type="model")
        print(f"Done: {local_dir}")
    except Exception as ex:
        print(f"Failed to download {repo_id}: {ex}")
        sys.exit(2)

print("\nAll downloads finished.")
