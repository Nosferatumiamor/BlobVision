#!/usr/bin/env python3
"""Download stabilityai/sdxl-turbo (fp16) into models/sdxl-turbo."""
import os
import sys

BLOBVISION_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(BLOBVISION_ROOT, "models", "sdxl-turbo")
HF_ID = "stabilityai/sdxl-turbo"


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    os.makedirs(out_dir, exist_ok=True)
    print("Downloading", HF_ID, "->", out_dir)
    print("(variant fp16, ~6.5 GB)")
    print()

    from huggingface_hub import snapshot_download

    snapshot_download(
        HF_ID,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.pdf", "*.png", "*.jpg", "*.webp"],
    )
    print()
    print("Done:", out_dir)
    print("Verify: launch BlobVision.bat and use Redux — SDXL Turbo loads from models/sdxl-turbo/")


if __name__ == "__main__":
    main()