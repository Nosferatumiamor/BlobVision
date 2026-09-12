# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

BlobVision is a portable Windows desktop app that emulates old/legacy generative-AI image and video
pipelines behind a single Gradio UI, wrapped in a native PyWebView window (no browser tab). Three
independent "families" of generation, switchable in the UI:

- **VQGAN+CLIP** (`app/generate.py` + `blobvision_engine.py`) — three modes: `legacy` (pure VQGAN+CLIP
  optimization loop, optional init image), `redux` (SDXL Turbo generates a sketch first, then VQGAN+CLIP
  "blobifies" it — the fast default), `video` (upload a clip, VQGAN blobifies keyframes, RIFE
  interpolates back up to full frame rate).
- **DeepDream** (`blobvision_deepdream.py`) — InceptionV3 layer-activation dreaming, with `redux`
  (SDXL sketch → DeepDream) and `img2img` variants.
- **Style Transfer** (`blobvision_style.py`) — classic VGG19 Gatys optimization (content + style image in).

The pitch is literally "old AI, made controllable": SDXL Turbo is used only to generate a clean sketch
that the legacy models (VQGAN+CLIP, InceptionV3, VGG19) then process, so the historical model's
aesthetic still dominates the output but composition is steerable.

## Running the app

No package.json/pyproject — everything runs through the `venv/` created on first launch.

```
BlobVision.bat                      # normal entry point: kills stale instances, launches native window
app\run.ps1                         # what the .bat calls; sets offline/proxy env vars
venv\Scripts\python.exe app\python\blobvision_app.py [--lazy] [--port 7860] [--debug]
```

- `blobvision_app.py` is the desktop launcher: builds the Gradio app from `blobvision_ui.py`, starts it
  in-process (`demo.launch(..., prevent_thread_lock=True)`), waits for the port, then opens a PyWebView
  native window pointed at `http://127.0.0.1:<port>`. Falls back to a real browser tab if PyWebView fails.
- `blobvision_ui.py` can also be run directly (`python app/python/blobvision_ui.py`) for a plain
  browser-based Gradio server — useful when iterating on UI without the native window.
- `blobvision_api.py` exposes the same generation engine over FastAPI/HTTP (Legacy/Redux/Corrupt) instead
  of Gradio, for programmatic/headless use.
- `--lazy` opens the UI without loading any models until the user presses **Start** (models normally
  warm up in a background thread on launch).
- There is no test suite, linter, or build step configured in this repo.

## Environment / dependencies

- `app/requirements-blobvision.txt` pins the ML stack: transformers, diffusers (SDXL Turbo), accelerate,
  safetensors, gradio, pywebview, imageio(-ffmpeg).
- `venv/`, `models/`, `outputs/`, `video-codecs/ffmpeg|rife/` are all gitignored — local install only.
  Weights are fetched on demand via in-app **Install** buttons or the scripts in `app/scripts/`
  (`download_sdxl_turbo.py`, `download_openclip_local.py`, `download_style_transfer.py`).
- `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are forced to `1` at launch — the app expects weights to
  already be present locally and will not silently hit the network mid-session.
- `Setup Video Codecs.bat` installs bundled ffmpeg + RIFE into `video-codecs/` for the video mode's
  frame interpolation; without it, video mode degrades gracefully (checked via
  `video_codecs_status()` / `_video_codecs_ready()`).

## Path & layout conventions

All filesystem layout is centralized in `app/python/blobvision_paths.py` — **always resolve paths through
it** rather than hardcoding relative paths, since the app must remain portable (works from any folder,
no absolute paths baked in):

- `BLOBVISION_ROOT` = repo root (derived from `__file__`, not cwd). `BLOBDREAM_ROOT` is a legacy alias
  kept for older modules — same value.
- `models/`, `outputs/`, `video-codecs/` all live under root; per-mode subdirs come from helpers like
  `sdxl_model_dir()`, `vqgan_output_dir()`, `deepdream_output_dir()`, `style_output_dir()`.
- `ensure_layout()` creates the expected directory tree and is called at import time by every engine
  module (`blobvision_engine.py`, `blobvision_deepdream.py`, `blobvision_style.py`, `generate.py`).
- `TAMING_REPO` resolves to `vendor/taming-transformers` (falls back to a legacy `taming-transformers/`
  root folder if present) — this is the vendored, trimmed copy of the VQGAN model code, committed to git.
- `gradio_allowed_paths()` whitelists the dirs Gradio is allowed to serve images/video from.
- Legacy paths under `app/checkpoints/` and `app/outputs/` are auto-migrated into the new `models/` /
  `outputs/` layout on first run — don't reintroduce writes to those old locations.

## Engine architecture

- **`blobvision_engine.py`** is the core VQGAN+CLIP engine (`BlobVisionEngine` class) plus all video
  pipeline logic (frame extraction/probing via ffmpeg, RIFE interpolation, frame reassembly/muxing).
  It owns `BlobVRAMCache`, which parks/unparks the SDXL sketch model and the VQGAN+CLIP stack so only
  one occupies GPU memory at a time (`PHASE_SKETCH` vs `PHASE_VQGAN`) — VRAM budget is tight enough that
  both models can't comfortably stay resident together.
- **`app/generate.py`** is the original standalone VQGAN+CLIP CLI script (Katherine Crowson-style
  optimization loop) that `blobvision_engine.py` wraps/drives; it imports `taming` from `TAMING_REPO`.
- **`blobvision_deepdream.py`** and **`blobvision_style.py`** are self-contained sibling engines (own
  dataclasses, own output dirs) — same shape as the VQGAN engine but far smaller, no VRAM-cache sharing.
- **`blobvision_cancel.py`** is a simple thread-safe cooperative cancel flag (`request()`/`is_requested()`
  → raises `AbortedError`) checked periodically inside long optimization loops; used for the UI's Abort
  button.
- **`blobvision_meta.py`** builds/reads a `blobdream` JSON blob embedded in output PNG metadata
  (prompt, seed, iterations, mode, etc.) so generated images are self-describing and reproducible.
- **`blobvision_log.py`** captures stdout/stderr into a bounded in-memory ring buffer for the UI's live
  console panel, filtering out known-noisy framework warnings (`_SKIP_FRAGMENTS`).

## UI architecture (`blobvision_ui.py`)

Single large Gradio `Blocks` app (`build_ui()`) built around a family switcher (VQGAN / DeepDream /
Style Transfer) rather than separate pages — swapping families updates visibility of shared/reused
components (`switch_family`, `_family_btn_updates`) instead of remounting the UI. Key patterns to know
before touching this file:

- Per-family mode dictionaries (`MODE_BLURBS`, `DD_MODE_BLURBS`, `ITER_SLIDER_MAX`, `DEFAULT_ITERATIONS`,
  `DEFAULT_DENOISE`) drive both UI copy and default slider values — when adding a mode, update these
  alongside the actual generation branch, not just one or the other.
- Model readiness is polled rather than pushed: `_sdxl_ready()`, `_vqgan_ready()`,
  `_style_weights_ready()`, `_video_codecs_ready()` gate which install/action buttons are active, and
  `poll_status()`/`poll_codecs_install_btn()`/`poll_style_install_btn()` re-check on a timer.
- `get_engine()` / `get_dd_engine()` / `get_st_engine()` lazily construct and cache the per-family engine
  singleton; `_dispose_engine()` / `_dispose_dd_engine()` / `_dispose_st_engine()` tear them down to free
  VRAM when switching families or on exit.
- `do_generate()` (VQGAN family) and `do_deepdream_generate()` / `do_style_generate()` are the three
  generate-button entry points; they're long because they branch heavily on mode (`legacy`/`redux`/
  `video`/`img2img`) — read the relevant mode branch only, don't assume shared logic across modes.
- This file is very large (~3200 lines) and has known instability/proportion bugs per the maintainer —
  when fixing UI bugs, prefer narrowly-scoped edits to the specific handler/component involved over
  broad refactors, unless a refactor is explicitly the task.

## Vendored code

- `vendor/taming-transformers/` — trimmed VQGAN model code (taming-transformers), committed to git,
  runtime package only (not the full upstream repo).
- `app/CLIP/clip/` — vendored OpenAI CLIP.
- Treat these as third-party code: prefer not to modify unless fixing an integration bug, and keep
  changes minimal/upstream-compatible.

## Removed scope

ModelScope, AnimateDiff, and Disco Diffusion were deliberately removed from V1 to keep the repo lean and
stable — don't reintroduce them without an explicit request.
