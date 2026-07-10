#!/usr/bin/env python3
"""Download VGG19 ImageNet weights into models/style-transfer/."""
import os
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_DIR = os.path.join(APP_DIR, "python")
sys.path.insert(0, PY_DIR)
from blobvision_paths import STYLE_WEIGHTS_DIR, style_weights_path

WEIGHTS_DIR = STYLE_WEIGHTS_DIR
WEIGHTS_PATH = style_weights_path()


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if os.path.isfile(WEIGHTS_PATH):
        size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
        print("Already present:", WEIGHTS_PATH, "({:.1f} MB)".format(size_mb))
        return

    print("Downloading VGG19 ImageNet weights ->", WEIGHTS_PATH)
    print("(~548 MB via torchvision)")

    import torch
    from torchvision import models

    weights = models.VGG19_Weights.IMAGENET1K_V1
    vgg = models.vgg19(weights=weights)
    torch.save(vgg.state_dict(), WEIGHTS_PATH)
    size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
    print("Done: {:.1f} MB".format(size_mb))


if __name__ == "__main__":
    main()
