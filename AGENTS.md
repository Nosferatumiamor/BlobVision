# AGENTS.md

Guide for coding agents working in this repository. See [CLAUDE.md](CLAUDE.md) for the full
architecture writeup (engine internals, UI structure, path conventions) — this file is the shorter,
tool-agnostic entry point.

## Project

BlobVision: a portable Windows desktop app (Gradio UI in a PyWebView native window) that reproduces old
generative-AI pipelines — VQGAN+CLIP, DeepDream, VGG19 style transfer — optionally steered by an SDXL
Turbo sketch pass for composition control. No build system, no package.json; a single `venv/` created on
first launch drives everything.

## Setup / run

```
BlobVision.bat                                          # normal launch (Windows)
venv\Scripts\python.exe app\python\blobvision_ui.py      # Gradio only, plain browser tab, faster iteration
venv\Scripts\python.exe app\python\blobvision_app.py --lazy --debug   # native window, skip model warmup, devtools
venv\Scripts\python.exe app\python\blobvision_api.py      # FastAPI HTTP surface instead of Gradio
```

Dependencies: `app/requirements-blobvision.txt` into `venv/`. Model weights live in `models/` (gitignored,
fetched via in-app Install buttons or `app/scripts/download_*.py`) — don't assume they're present in a
fresh checkout or CI-like environment; code paths that touch models must degrade gracefully when the
weight files are missing (this is already how `*_ready()` checks in `blobvision_ui.py` work — follow that
pattern rather than assuming weights exist).

## No test suite

There is currently no automated test suite, linter, or type checker configured. Validate changes by
actually running the app (`blobvision_ui.py` in browser mode is the fastest loop) and exercising the
affected mode end-to-end — don't claim a fix works from reading code alone. If you add real tests, wire
them up here and in CLAUDE.md so future agents know how to run them.

## Conventions

- All filesystem paths go through `app/python/blobvision_paths.py` helpers — never hardcode paths
  relative to cwd; the app must stay portable (runnable from any folder).
- Keep `models/`, `outputs/`, `venv/`, `video-codecs/{ffmpeg,rife}/` out of git — the `.gitignore` at
  root and `app/.gitignore` already enforce this; don't add exceptions for weight files (`*.ckpt`,
  `*.safetensors`, `*.pth` are blocked at the root gitignore as a safety net).
- `vendor/taming-transformers/` and `app/CLIP/` are vendored third-party code — avoid modifying beyond
  targeted integration fixes.
- UI copy and in-app strings are French; keep new user-facing strings consistent with that.
- Comments/docstrings in this codebase are sparse and used only for non-obvious rationale — match that
  style rather than adding narrative comments.

## Current focus: stabilization + polish (local-only, not a web deployment)

This is a personal project being hardened for a wider release, but strictly as a **local single-user
desktop app** — there is no plan to host it online or support multiple concurrent users/GPUs. Don't
introduce multi-user/auth/remote-hosting concerns; they're out of scope.

- The Gradio UI (`blobvision_ui.py`, ~3200 lines) is the biggest pain point — proportions/layout break
  under some window sizes, and there are known intermittent UI bugs. This is largely why Gradio itself is
  under question: a full rewrite onto Tauri or Electron has been floated as the "real" fix, but it's
  acknowledged as a huge undertaking and is **not decided** — don't start migrating UI framework as a
  side effect of an unrelated fix. Until/unless that's explicitly greenlit, keep bug fixes here small and
  scoped rather than broad rewrites.
- `BlobVRAMCache` in `blobvision_engine.py` assumes exactly one GPU and one active generation at a time
  (`demo.queue(default_concurrency_limit=1)`) — this is a correct assumption for the local-only use case
  and should not be "fixed" toward concurrency.
- Known quality gap: DeepDream output (`blobvision_deepdream.py`) doesn't match the classic look —
  outputs read as a regular grid/texture overlay rather than the iconic eyes/dogs/birds hallucinations of
  the original. Root causes: no jitter (random pixel shift per step, which the original algorithm uses to
  avoid convolution-grid tiling artifacts) and torchvision's `Inception_V3` weights/architecture differ
  from the original Caffe GoogLeNet (Inception v1) DeepDream was built on, at the layer depths currently
  exposed (`Mixed_5b`/`Mixed_6a`/`Mixed_7c`). Worth revisiting if asked to improve DeepDream fidelity.

## Git / commits

- Never commit anything under `models/`, `outputs/`, `venv/`, or `video-codecs/{ffmpeg,rife}/` — check
  `git status` output carefully before staging if you've been downloading weights or generating outputs
  during a session, since large binaries here are exactly what the gitignore is designed to keep out.
- Follow the existing commit style (see `git log`): short, imperative, no trailing period.
