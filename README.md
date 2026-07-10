# BlobVision V1

Portable desktop app for AI image and video generation — three emulators in one UI:

- **VQGAN+CLIP** — Redux / Legacy / Video (SDXL Turbo sketch + VQGAN blob aesthetic)
- **DeepDream** — InceptionV3 dreaming (redux + img2img)
- **Style Transfer** — VGG19 Gatys optimization

## Quick start (Windows)

1. Clone or copy this folder anywhere on disk (e.g. `D:\BlobVision`).
2. Double-click **`BlobVision.bat`**.
3. Press **Start** in the UI when models are ready.

First launch may download Python dependencies into `venv/` and prompt for optional weights (VGG19, ffmpeg/RIFE).

## Layout

```
BlobVision/
  BlobVision.bat             # main launcher
  Setup Video Codecs.bat     # optional ffmpeg + RIFE
  app/                       # application code (python/, scripts/, CLIP/, generate.py)
  models/                    # weights (gitignored — local install)
    sdxl-turbo/
    vqgan/
    open_clip/
    style-transfer/
  outputs/                   # generations (gitignored)
    vqgan/
    deepdream/
    style-transfer/
  video-codecs/              # bundled ffmpeg/RIFE (gitignored, install via Setup .bat)
  vendor/taming-transformers/  # vendored VQGAN model code (trimmed, committed)
  venv/                      # Python env (gitignored)
```

Legacy paths under `app/checkpoints/` and `app/outputs/` are migrated automatically on first run.

## What is not in V1

ModelScope, AnimateDiff, and Disco Diffusion were removed from this release to keep the repo lean and stable.

## Git / distribution notes

- **Committed:** application code, vendored `app/CLIP/` and `vendor/taming-transformers/` (trimmed to the runtime package only), `video-codecs/README.txt`, small assets.
- **Gitignored (local install):** `venv/`, `models/`, `outputs/`, `video-codecs/ffmpeg/`, `video-codecs/rife/`, `__pycache__/`.
- Users install weights via in-app **Install** buttons or `app/scripts/` download helpers.
