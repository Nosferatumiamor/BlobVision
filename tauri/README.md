# BlobVision — Tauri shell (WIP, replacing the pywebview + Gradio UI)

A native Tauri window and a hand-built frontend, talking to the existing
Python ML engine over HTTP instead of Gradio. `blobvision_api.py` (FastAPI)
is the shared backend for both UIs — this shell spawns it as a child process
and the frontend calls it directly at `http://127.0.0.1:8420`.

## What works right now

- **Launcher**: `BlobVision-Tauri.bat` at the repo root (double-click, or
  `npm run tauri dev` from this directory for development).
- **Rust shell** (`src-tauri/`): spawns
  `venv/Scripts/python.exe app/python/blobvision_api.py --port 8420 --no-warmup`
  with the same offline env vars as `app/run.ps1`, inherits its stdio (so
  Python's own logs show up in the same console), and kills it when the
  window closes. It does **not** proxy requests — the frontend hits the
  Python API directly (CORS is wide open on the FastAPI side). One custom
  command: `open_outputs_folder(family)`, opens that family's output folder
  in Explorer.
- **VQGAN+CLIP family**: full parity with the Gradio sidebar — mode
  (redux/legacy), prompt, negative prompt, aspect, steps, deslop, seed
  (random/fixed/reuse), img2img drag-and-drop with a square preview that
  shows either your dropped image or the auto-generated SDXL sketch,
  metadata readout, auto warmup on startup.
- **DeepDream family**: same img2img/seed/aspect plumbing, plus Inception
  layer (mixed5b/6a/7) and Intensity. Mode (txt2img/redux/img2img) is
  inferred from prompt/upload presence, matching `do_deepdream_generate`'s
  logic exactly. Lazily warms up GoogLeNet the first time you switch to the
  tab.
- **Style Transfer family**: single scrollable preset dropdown under the
  img2img block (39 SDXL presets + "None" = classic VGG19 Gatys), matching
  the Gradio redesign exactly. Preset mode reuses
  `generate_style_preset()` (caption-assist included, transparently).
  Classic mode needs a separate style-reference image (its own drop zone)
  plus style strength / content fidelity / steps sliders.
- **Output UX**: fixed square canvas regardless of image aspect ratio,
  click-to-zoom lightbox, one-click download button.
- Real branding (`app/assets/logo/blob-logo*.png`) instead of placeholder
  icons.

## What's NOT done yet

- **VQGAN video mode** — no HTTP endpoint for it.
- **Live console/progress streaming** — `/generate`,
  `/deepdream/generate`, and `/style/generate` are single blocking HTTP
  calls with no progress events; the UI just shows a static "Generating..."
  message. Would need SSE or WebSocket support added to `blobvision_api.py`.
- Path resolution in `src-tauri/src/main.rs` uses `CARGO_MANIFEST_DIR`
  (baked in at compile time) — fine for local dev, not portable to a
  distributed build. Needs to walk up from the running exe's own location
  instead (mirroring how `blobvision_paths.py` derives `BLOBVISION_ROOT`
  from `__file__`, not cwd) before ever packaging a release build.
- Gradio (`BlobVision.bat`) remains the only UI with full feature parity
  until video mode is ported over too.

## Running it

Requires Rust (`rustup`) and the existing BlobVision `venv/` already set up
(see the root `CLAUDE.md`).

```
cd tauri
npm install
npm run tauri dev
```

or just double-click `BlobVision-Tauri.bat` from the repo root.

## Next steps, in order

1. VQGAN video mode endpoint + frontend.
2. Progress streaming during generation.
3. Fix the portable path resolution before attempting a release build.

## Notes

- Native OS drag-and-drop into the window requires
  `app.windows[].dragDropEnabled: false` in `tauri.conf.json` — Tauri v2
  intercepts file drops for its own API by default, which otherwise
  silently prevents the webview's HTML5 `drop` events from ever firing.
- The Upscale x2 (Real-ESRGAN) checkbox is shared across all three
  families and lives at the bottom of the sidebar, outside the
  per-family field blocks.
