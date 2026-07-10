#!/usr/bin/env python3
"""One-time download of OpenCLIP weights into models/open_clip/ (project-local)."""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
PY_DIR = os.path.join(APP_DIR, "python")
sys.path.insert(0, PY_DIR)
sys.path.insert(0, APP_DIR)

from generate import OPENCLIP_HF_REPOS
from blobvision_paths import openclip_root

_OPENCLIP_ROOT = openclip_root()

def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(_OPENCLIP_ROOT, exist_ok=True)
    for (model, tag), repo_id in OPENCLIP_HF_REPOS.items():
        dest = os.path.join(_OPENCLIP_ROOT, "{}__{}".format(model, tag))
        print("Downloading", repo_id)
        print("  ->", dest)
        snapshot_download(repo_id=repo_id, local_dir=dest)
        print("OK")
    print()
    print("Done. OpenCLIP will load from models/open_clip/ (no Hugging Face at runtime).")

if __name__ == "__main__":
    main()