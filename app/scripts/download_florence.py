#!/usr/bin/env python3
"""Download microsoft/Florence-2-base-ft into models/caption."""
import os
import sys

BLOBVISION_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(BLOBVISION_ROOT, "models", "caption")
HF_ID = "microsoft/Florence-2-base-ft"


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    os.makedirs(out_dir, exist_ok=True)
    print("Downloading", HF_ID, "->", out_dir)
    print("(~460 MB)")
    print()

    from huggingface_hub import snapshot_download

    snapshot_download(
        HF_ID,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.pdf"],
    )
    print()
    print("Done:", out_dir)
    print("Verify: launch BlobVision.bat and use a Style Transfer SDXL preset — captions load from models/caption/")


if __name__ == "__main__":
    main()
