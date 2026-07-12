#!/usr/bin/env python3
"""Download Real-ESRGAN x2plus weights into models/upscale/."""
import os
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_DIR = os.path.join(APP_DIR, "python")
sys.path.insert(0, PY_DIR)
from blobvision_upscale import UPSCALE_MODEL_URL, upscale_weights_status


def main():
    status = upscale_weights_status()
    if status["ready"]:
        size_mb = os.path.getsize(status["path"]) / (1024 * 1024)
        print("Already present:", status["path"], "({:.1f} MB)".format(size_mb))
        return

    print("Downloading Real-ESRGAN x2plus ->", status["path"])
    print("(~64 MB from", UPSCALE_MODEL_URL, ")")

    import torch

    os.makedirs(os.path.dirname(status["path"]), exist_ok=True)
    torch.hub.download_url_to_file(UPSCALE_MODEL_URL, status["path"], progress=True)
    size_mb = os.path.getsize(status["path"]) / (1024 * 1024)
    print("Done: {:.1f} MB".format(size_mb))


if __name__ == "__main__":
    main()
