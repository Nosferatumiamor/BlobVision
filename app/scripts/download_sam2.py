#!/usr/bin/env python3
"""Download the SAM2.1 (small) checkpoint into models/sam2/.

Only the checkpoint (.pt) needs fetching — the model config is a hydra YAML
bundled inside the installed `sam2` pip package itself, resolved by name at
load time (see blobvision_paths.SAM2_CONFIG_NAME / blobvision_sam.py).
"""
import os
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_DIR = os.path.join(APP_DIR, "python")
sys.path.insert(0, PY_DIR)
from blobvision_paths import SAM2_CHECKPOINT_NAME, SAM2_HF_REPO, sam2_checkpoint_path, sam2_model_dir


def main():
    out_dir = sam2_model_dir()
    ckpt_path = sam2_checkpoint_path()
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        print("Already present:", ckpt_path, "({:.1f} MB)".format(size_mb))
        return

    print("Downloading", SAM2_HF_REPO, "/", SAM2_CHECKPOINT_NAME, "->", out_dir)
    print("(~185 MB)")
    print()

    from huggingface_hub import hf_hub_download

    hf_hub_download(
        SAM2_HF_REPO,
        filename=SAM2_CHECKPOINT_NAME,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print()
    print("Done: {:.1f} MB".format(size_mb))


if __name__ == "__main__":
    main()
