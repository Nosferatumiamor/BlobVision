#!/usr/bin/env python3
"""BlobVision UI — desktop-wide layout, sidebar + preview."""
import argparse
import atexit
import base64
import json
import os
import random
import subprocess
import sys
import threading
import time
import uuid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
LOGO_PATH = os.path.join(APP_DIR, "assets", "logo", "blob-logo2.png")
os.chdir(APP_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

print("BlobVision — chargement des modules...", flush=True)

_LOCAL_BYPASS = "127.0.0.1,localhost,<local>"
for _proxy_key in ("NO_PROXY", "no_proxy"):
    _cur = os.environ.get(_proxy_key, "")
    if "127.0.0.1" not in _cur:
        os.environ[_proxy_key] = f"{_cur},{_LOCAL_BYPASS}".strip(",")
os.environ.setdefault("GRADIO_SSR_MODE", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Pipelines loaded with `dtype=torch.float16` cannot run with `cpu` device.*",
)
warnings.filterwarnings("ignore", message=".*float16.*operations on this device.*")

import blobvision_log

blobvision_log.install()

import gradio as gr

print("BlobVision — modules OK, construction de l'interface...", flush=True)

from blobvision_engine import (
    DEFAULT_ASPECT,
    DEFAULT_DENOISE,
    DEFAULT_ITERATIONS,
    MAX_ITERATIONS,
    ASPECT_FORMATS,
    ASPECT_CUSTOM,
    BlobVisionEngine,
    enable_hub_downloads,
    guess_aspect_from_size,
    fit_size_preserving_aspect,
    resolve_aspect_size,
    VQ_VIDEO_FRAME_STEP_CHOICES,
    DEFAULT_VQ_VIDEO_FRAME_STEP,
    parse_vq_video_frame_step,
    STYLE_PRESETS,
    DEFAULT_STYLE_PRESET,
    STYLE_PRESET_STEPS,
)
# Family-agnostic video pipeline (see blobvision_video.py's module docstring).
from blobvision_video import (
    VIDEO_LONG_WARN_SECONDS,
    format_video_duration,
    probe_video_file,
    setup_bundled_video_codecs,
    video_codecs_status,
)

from blobvision_deepdream import (
    DeepDreamEngine,
    DEFAULT_LAYER as DD_DEFAULT_LAYER,
    INCEPTION_LAYERS as DD_LAYERS,
    intensity_to_params as dd_intensity_to_params,
)

DD_DEFAULT_INTENSITY = 85

from blobvision_style import (
    StyleTransferEngine,
    DEFAULT_STEPS as ST_DEFAULT_STEPS,
    DEFAULT_STYLE_STRENGTH as ST_DEFAULT_STYLE_STRENGTH,
    DEFAULT_CONTENT_WEIGHT as ST_DEFAULT_CONTENT_WEIGHT,
    download_style_transfer_weights,
    style_transfer_weights_status,
)

from blobvision_upscale import UpscaleEngine

MODE_BLURBS = {
    "redux": "SDXL sketch → VQGAN blob. Fast default.",
    "legacy": "VQGAN-only, 150-500 steps. Optional source image.",
    "video": "Upload a clip, VQGAN blobifies keyframes, RIFE fills.",
}

DD_MODE_BLURBS = {
    "redux": "Prompt → SDXL sketch → DeepDream. Or drop an image for img2img.",
    "img2img": "Upload a photo — classic DeepDream amplifies eyes, animals and patterns.",
}

ST_MODE_BLURB = "Drop a content photo + a style reference on the right, then DEGENERATE."
ST_SDXL_MODE_BLURB = "Drop a photo on the right, pick a preset (or write your own), then DEGENERATE."
NO_PRESET_VALUE = ""
NO_PRESET_LABEL = "— None (classic style transfer) —"

FAMILY_META = {
    "vqgan": {"label": "VQGAN+CLIP", "enabled": True, "tagline": "Retro AI Slop Emulator"},
    "deepdream": {"label": "DeepDream", "enabled": True, "tagline": "Retro AI Slop Emulator"},
    "style": {"label": "Style Transfer", "enabled": True, "tagline": "Retro AI Slop Emulator"},
}

FAMILY_THEMES = {
    "vqgan": {
        "accent": "#5b6ee1", "hover": "#7280e8", "text": "#ffffff",
        "soft": "rgba(91, 110, 225, 0.12)", "border": "rgba(91, 110, 225, 0.55)",
        "panel": "rgba(91, 110, 225, 0.05)",
        "body": "#0b0f19", "surface": "#111827", "surface2": "#1a2030",
        "label": "#ffffff",
    },
    "deepdream": {
        "accent": "#c9a227", "hover": "#d9b23a", "text": "#ffffff",
        "soft": "rgba(201, 162, 39, 0.14)", "border": "rgba(201, 162, 39, 0.55)",
        "panel": "rgba(201, 162, 39, 0.06)",
        "body": "#141108", "surface": "#1c1810", "surface2": "#252015",
        "label": "#ffffff",
    },
    "style": {
        "accent": "#4caf7a", "hover": "#5fc08f", "text": "#041408",
        "soft": "rgba(76, 175, 122, 0.12)", "border": "rgba(76, 175, 122, 0.50)",
        "panel": "rgba(76, 175, 122, 0.05)",
        "body": "#081410", "surface": "#101c14", "surface2": "#15261a",
        "label": "#ffffff",
    },
}

FAMILY_THEME_JS = {
    fid: (
        "() => {{ document.querySelector('.gradio-container')"
        "?.setAttribute('data-blob-family','{}'); }}".format(fid)
    )
    for fid in FAMILY_META
}

SIDEBAR_WIDTH = 380
PREVIEW_GAP = 12
PREVIEW_SIZE = 420
SOURCE_PREVIEW_SIZE = 380
ITER_SLIDER_MAX = {"redux": 100, "legacy": 500, "video": 100}
ASPECT_CHOICES = list(ASPECT_FORMATS.keys()) + [ASPECT_CUSTOM]
ASPECT_FORMAT_INFO = "Presets for txt2img. Custom keeps img2img ratio"


CONSOLE_SCROLL_JS = """
function() {
    if (window.__blobConsoleHooked) return;
    window.__blobConsoleHooked = true;
    const pick = () => document.querySelector("#blob-console-out textarea")
        || document.querySelector(".blob-console textarea");
    const scroll = () => {
        const el = pick();
        if (el) el.scrollTop = el.scrollHeight;
    };
    setInterval(scroll, 400);
    const obs = new MutationObserver(scroll);
    const boot = () => {
        const el = pick();
        if (!el) { setTimeout(boot, 200); return; }
        obs.observe(el, {characterData: true, childList: true, subtree: true});
        scroll();
    };
    boot();
    document.querySelector('.gradio-container')?.setAttribute('data-blob-family', 'vqgan');
}
"""

PROMPT_ENTER_JS = ""  # superseded by BLOB_JS_HOOK

BLOB_JS_HOOK = """
function() {
    if (window.__blobAllHooked) return;
    window.__blobAllHooked = true;
    document.querySelector('.gradio-container')?.setAttribute('data-blob-family','vqgan');

    // 0) Prevent browser from opening dragged files — let Gradio dropzones handle them
    window.addEventListener('dragover', function(e) { e.preventDefault(); });
    window.addEventListener('drop', function(e) { e.preventDefault(); });

    // 1) Enter key in prompt -> DEGENERATE
    // A capture-phase listener on document (not per-textarea) so it: (a) survives
    // Gradio re-rendering the textarea DOM node, and (b) runs before Gradio's own
    // keydown handling can swallow the event. Only the currently-visible DEGENERATE
    // button is clicked — all three families reuse that label, and the other two are
    // still present (just display:none) in the DOM at any given time.
    function findGenBtn() {
        var btns = document.querySelectorAll('button');
        for (var i = 0; i < btns.length; i++) {
            var b = btns[i];
            if (b.textContent.trim().toUpperCase() === 'DEGENERATE' && b.offsetParent !== null) {
                return b;
            }
        }
        return null;
    }
    document.addEventListener('keydown', function(e) {
        if (e.key !== 'Enter' || e.shiftKey) return;
        var target = e.target;
        if (!target || target.tagName !== 'TEXTAREA') return;
        if (!target.closest('#blob-sidebar-wrap')) return;
        var btn = findGenBtn();
        if (!btn) return;
        e.preventDefault();
        e.stopPropagation();
        btn.click();
    }, true);

    // 2) Loading banner is now inline in header — no overlay positioning needed

    // 3) Force-show all buttons on sketch images + fix drag-and-drop for all upload zones
    function fixSketchButtons() {
        var sketchIds = ['blob-sketch-image', 'blob-dd-sketch-image', 'blob-st-content-image', 'blob-st-style-image', 'blob-source-image'];
        for (var s = 0; s < sketchIds.length; s++) {
            var sketch = document.getElementById(sketchIds[s]);
            if (!sketch) continue;
            var els = sketch.querySelectorAll('button, [class*="button"], [class*="icon"], [class*="download"], [class*="tool"]');
            for (var i = 0; i < els.length; i++) {
                els[i].style.display = 'flex';
                els[i].style.visibility = 'visible';
                els[i].style.opacity = '1';
            }
        }
    }
    fixSketchButtons();
    setInterval(fixSketchButtons, 800);

    // 3b) Force-enable drag-and-drop on all Gradio upload zones
    function enableDropZones() {
        var zones = document.querySelectorAll('#blob-right-col [data-testid="dropzone"], #blob-right-col .upload-container, #blob-right-col .wrap.image-frame');
        for (var i = 0; i < zones.length; i++) {
            var z = zones[i];
            z.addEventListener('dragover', function(e) { e.preventDefault(); });
            z.addEventListener('dragenter', function(e) { e.preventDefault(); });
            z.addEventListener('drop', function(e) { e.preventDefault(); });
        }
    }
    enableDropZones();
    setInterval(enableDropZones, 1000);

    // 4) Console auto-scroll
    function scrollConsole() {
        var el = document.querySelector('#blob-console-out textarea') || document.querySelector('.blob-console textarea');
        if (el) el.scrollTop = el.scrollHeight;
    }
    setInterval(scrollConsole, 400);

    // 5) Family tab active highlight — match by button text, fallback to data-blob-family
    var FAMILY_MAP = {
        'VQGAN+CLIP': 'vqgan',
        'DEEPDREAM': 'deepdream',
        'STYLE TRANSFER': 'style'
    };
    function highlightActiveFamily() {
        var container = document.querySelector('.gradio-container');
        var activeFamily = container ? container.getAttribute('data-blob-family') : null;
        if (!activeFamily) activeFamily = 'vqgan';
        // Find all buttons in the header area
        var allBtns = document.querySelectorAll('#blob-header-toolbar button, #blob-header-wrap button');
        for (var i = 0; i < allBtns.length; i++) {
            var b = allBtns[i];
            var text = b.textContent.trim().toUpperCase().replace(/\s+/g, ' ');
            var fid = FAMILY_MAP[text];
            if (!fid) continue; // skip non-family buttons (Start, Disconnect, etc)
            if (fid === activeFamily) {
                b.style.setProperty('background', '#2563eb', 'important');
                b.style.setProperty('color', '#ffffff', 'important');
                b.style.setProperty('border-color', '#2563eb', 'important');
                b.style.setProperty('font-weight', '600', 'important');
                b.style.setProperty('opacity', '1', 'important');
            } else {
                b.style.setProperty('background', 'transparent', 'important');
                b.style.setProperty('opacity', '0.65', 'important');
                b.style.setProperty('font-weight', 'normal', 'important');
            }
        }
    }
    highlightActiveFamily();
    setInterval(highlightActiveFamily, 200);
}
"""


def _header_brand_html():
    logo = ""
    if os.path.isfile(LOGO_PATH):
        with open(LOGO_PATH, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        logo = '<img class="blob-logo" src="data:image/png;base64,' + b64 + '" alt="" />'
    return (
        '<div class="blob-brand">'
        + logo
        + '<div class="blob-title-wrap">'
        + '<div class="blob-title-main">BlobVision</div>'
        + '</div></div>'
    )


DISCLAIMER = (
    "BlobVision is unfiltered. Use at your own risk—you are solely responsible "
    "for any generated or shared images."
)

CREDIT = (
    "Vibe-coded by GROM with Cursor and Claude. Open source, inspired by "
    "RiversHaveWings' historic Colab notebook."
)


def _build_family_theme_css():
    """Family bar tab colors only — do not override Gradio input/block CSS variables."""
    vq = FAMILY_THEMES["vqgan"]
    lines = [
        ".gradio-container {",
        "  --blob-accent: " + vq["accent"] + ";",
        "  --blob-accent-hover: " + vq["hover"] + ";",
        "  --blob-accent-text: " + vq["text"] + ";",
        "  --blob-accent-soft: " + vq["soft"] + ";",
        "  --blob-accent-border: " + vq["border"] + ";",
        "  --blob-accent-panel: " + vq["panel"] + ";",
        "}",
    ]
    for fid, theme in FAMILY_THEMES.items():
        lines.append(
            '#blob-header-toolbar .blob-family-btn.blob-family-' + fid + ' button {'
        )
        lines.append("  border-color: " + theme["border"] + " !important;")
        lines.append("  color: " + theme["accent"] + " !important;")
        lines.append("  background: transparent !important;")
        lines.append("}")
        lines.append(
            '#blob-header-toolbar .blob-family-btn.blob-family-' + fid
            + '.blob-family-active button {'
        )
        lines.append("  background: " + theme["accent"] + " !important;")
        lines.append("  color: " + theme["text"] + " !important;")
        lines.append("  border-color: " + theme["accent"] + " !important;")
        lines.append("  font-weight: 600 !important;")
        lines.append("}")
    return "\n".join(lines)


CSS = """
.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 4px 4px 4px !important;
  box-sizing: border-box !important;
  overflow: hidden !important;
  height: 100vh !important;
  max-height: 100vh !important;
}
footer { display: none !important; }
.gradio-container > main,
.gradio-container > .main,
.gradio-container > form,
.gradio-container > div > .main {
  max-width: 100% !important;
  width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
  overflow: hidden !important;
}
.gradio-container .contain {
  max-width: 100% !important;
}
""" + _build_family_theme_css() + """
#blob-header-toolbar {
  display: flex !important;
  flex-wrap: wrap !important;
  align-items: center !important;
  gap: 5px !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
  box-sizing: border-box !important;
  flex: 1 1 auto !important;
  justify-content: flex-end !important;
  min-width: 0 !important;
}
#blob-header-toolbar .blob-status,
#blob-header-toolbar .blob-status p {
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  text-align: right !important;
  margin: 0 4px 0 0 !important;
  font-size: 0.68rem !important;
  line-height: 1.2 !important;
  max-width: 11rem !important;
}
#blob-header-toolbar .blob-family-btn {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
}
#blob-header-toolbar .blob-family-btn button {
  font-size: 0.54rem !important;
  padding: 2px 6px !important;
  border-radius: 4px !important;
  min-height: 20px !important;
  min-width: 4.2rem !important;
  line-height: 1.15 !important;
  background: transparent !important;
  border-width: 1px !important;
  opacity: 0.88 !important;
  white-space: nowrap !important;
  display: inline-flex !important;
}
#blob-header-toolbar .blob-family-btn.blob-family-active button {
  opacity: 1 !important;
  font-weight: 600 !important;
  background: var(--color-accent, #2563eb) !important;
  color: #fff !important;
  border-color: var(--color-accent, #2563eb) !important;
}
#blob-header-toolbar .blob-family-btn.blob-family-disabled button {
  opacity: 0.38 !important;
  cursor: not-allowed !important;
}
#generate-btn button, #dd-generate-btn button,
#generate-btn button span, #dd-generate-btn button span,
#blob-header-actions .primary button,
#blob-header-actions button.primary {
  background: var(--blob-accent) !important;
  color: #ffffff !important;
  border-color: var(--blob-accent-border) !important;
}
#generate-btn button:hover, #dd-generate-btn button:hover,
#blob-header-actions .primary button:hover,
#blob-header-actions button.primary:hover {
  background: var(--blob-accent-hover) !important;
  color: #ffffff !important;
}
#blob-sidebar-wrap input[type="range"] {
  accent-color: var(--blob-accent) !important;
}
#blob-sidebar-wrap input[type="range"]::-webkit-slider-thumb {
  background: var(--blob-accent) !important;
}
#blob-header-wrap {
  position: relative !important;
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
  min-height: 0 !important;
  background: linear-gradient(90deg, var(--blob-accent-panel, transparent) 0%, transparent 55%) !important;
}
#blob-header-wrap > .block,
#blob-header-wrap > .form { padding: 0 !important; margin: 0 !important; gap: 0 !important; }
#blob-header-wrap .gradio-row { margin: 0 !important; padding: 0 !important; gap: 4px !important; }
#blob-header { align-items: center !important; margin-bottom: 0 !important; gap: 6px !important; flex-wrap: nowrap !important; padding: 0 !important; }
#blob-header-left { flex: 0 0 auto !important; min-width: 0 !important; padding-left: 8px !important; }
#blob-tagline-slot { margin: 0 !important; padding: 0 !important; }
#blob-tagline-slot .blob-tagline { font-size: 0.62rem !important; line-height: 1 !important; margin-top: 0 !important; }
.blob-brand { gap: 6px !important; }
.blob-brand .blob-logo { height: 1.65rem !important; max-width: 2.6rem !important; }
.blob-title-main { font-size: 0.92rem !important; line-height: 1 !important; }
.blob-tagline { font-size: 0.62rem !important; line-height: 1 !important; }
#blob-header-left { flex: 0 0 auto !important; min-width: 0 !important; }
#blob-header-left { padding-left: 12px !important; }
.blob-brand {
  display: flex !important;
  align-items: center !important;
  gap: 6px !important;
}
.blob-brand .blob-logo {
  height: 1.65rem !important;
  width: auto !important;
  max-width: 2.6rem !important;
  object-fit: contain !important;
  display: block !important;
  flex-shrink: 0 !important;
  opacity: 0.92 !important;
}
#blob-header .blob-title-wrap {
  display: flex !important;
  flex-direction: column !important;
  gap: 0 !important;
  min-width: 0 !important;
  margin: 0 !important;
}
.blob-title-main {
  margin: 0 !important;
  font-size: 0.92rem !important;
  font-weight: 700 !important;
  line-height: 1 !important;
}
.blob-tagline {
  margin: 0 !important;
  font-size: 0.62rem !important;
  font-weight: 400 !important;
  opacity: 0.55 !important;
  font-style: italic !important;
  line-height: 1 !important;
  white-space: nowrap !important;
}
.blob-credit p {
  font-size: 0.58rem !important;
  line-height: 1.3 !important;
  opacity: 0.36;
  margin: 0.35rem 0 0 0 !important;
  max-width: 820px;
}

#blob-header-left .block, #blob-header-left .form { padding: 0 !important; margin: 0 !important; gap: 2px !important; }
#blob-header .blob-title-wrap { flex: 0 0 auto !important; margin: 0 !important; }
#blob-header-right {
  flex: 0 0 200px !important;
  min-width: 200px !important;
  max-width: 240px !important;
  margin-left: auto !important;
  align-self: center !important;
}
#blob-header-right .blob-status,
#blob-header-right .blob-status p {
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  text-align: right !important;
  margin: 0 !important;
  font-size: 0.72rem !important;
  line-height: 1.2 !important;
}
#blob-header .blob-status {
  flex: 1 1 auto !important;
  min-width: 0 !important;
  padding-right: 8px !important;
}
#blob-header-actions {
  flex: 0 0 auto !important;
  flex-shrink: 0 !important;
  min-width: 0 !important;
  max-width: max-content !important;
  align-self: center !important;
  gap: 4px !important;
  margin: 0 !important;
  margin-left: auto !important;
}
#blob-install-codecs-btn button,
#blob-install-style-btn button,
#st-install-models-btn button {
  max-width: 11rem !important;
  background: #f9a825 !important;
  color: #1a1200 !important;
  border: 1px solid #f57f17 !important;
  font-weight: 600 !important;
}
#blob-install-codecs-btn button:hover,
#blob-install-style-btn button:hover,
#st-install-models-btn button:hover {
  background: #ffb300 !important;
}
#st-install-models-btn {
  width: 100% !important;
  margin: 0 0 4px 0 !important;
}
#blob-header-actions .gr-button,
#blob-header-actions button {
  width: auto !important;
  min-width: 0 !important;
  max-width: 6.5rem !important;
  padding: 3px 8px !important;
  white-space: nowrap !important;
  font-size: 0.68rem !important;
}
.blob-console textarea {
  font-family: Consolas, "Courier New", monospace !important;
  font-size: 10px !important;
  line-height: 1.25 !important;
  min-height: 72px !important;
  max-height: 88px !important;
  overflow-y: auto !important;
}
.blob-negative-prompt textarea {
  min-height: 1.75rem !important;
  max-height: 1.75rem !important;
  overflow-y: auto !important;
  resize: none !important;
  line-height: 1.2 !important;
  padding-top: 4px !important;
  padding-bottom: 4px !important;
  font-size: 0.55rem !important;
}
.blob-compact-slider {
  margin: 0 !important;
  padding: 0 !important;
  width: 100% !important;
  max-width: 100% !important;
  flex: 0 0 auto !important;
  height: auto !important;
  min-height: 0 !important;
  overflow: visible !important;
}
#blob-sidebar-wrap .blob-compact-slider.block,
#blob-sidebar-wrap .block.blob-compact-slider {
  width: 100% !important;
  max-width: 100% !important;
  flex: 0 0 auto !important;
  height: auto !important;
  padding: 0 !important;
  margin: 0 0 1px 0 !important;
}
#blob-sidebar-wrap .blob-compact-slider label.container,
#blob-sidebar-wrap .blob-compact-slider .info {
  width: 100% !important;
  max-width: 100% !important;
  display: inline !important;
  margin: 0 !important;
  padding: 0 !important;
}
#blob-sidebar-wrap .blob-compact-slider .info {
  line-height: 1.2 !important;
  font-size: 0.52rem !important;
  opacity: 0.55;
  margin-left: 4px !important;
  white-space: normal !important;
}
#blob-sidebar-wrap .blob-compact-slider .form {
  width: 100% !important;
  max-width: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 2px !important;
  height: auto !important;
  flex: 0 0 auto !important;
  overflow: visible !important;
  box-sizing: border-box !important;
}
#blob-sidebar-wrap .blob-compact-slider .wrap {
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  width: 100% !important;
  max-width: 100% !important;
  flex: 1 1 auto !important;
  gap: 8px !important;
  height: auto !important;
  min-height: 0 !important;
  overflow: visible !important;
  box-sizing: border-box !important;
}
#blob-sidebar-wrap .blob-compact-slider input[type="range"] {
  flex: 1 1 auto !important;
  width: auto !important;
  min-width: 0 !important;
  max-width: 100% !important;
  margin: 0 !important;
  display: block !important;
}
#blob-sidebar-wrap .blob-compact-slider input[type="number"] {
  flex: 0 0 3.6rem !important;
  width: 3.6rem !important;
  min-width: 3.6rem !important;
  max-width: 3.6rem !important;
  min-height: 1.9rem !important;
  height: 1.9rem !important;
  max-height: 1.9rem !important;
  padding: 4px 4px !important;
  font-size: 0.72rem !important;
  line-height: 1.35 !important;
  box-sizing: border-box !important;
  display: block !important;
  background: var(--input-background-fill, #262640) !important;
  color: var(--body-text-color, #f5f5f5) !important;
  border: 1px solid var(--border-color-primary, #555) !important;
  border-radius: 6px !important;
}
#blob-main-row {
  align-items: stretch !important;
  flex-wrap: nowrap !important;
  width: 100% !important;
  box-sizing: border-box !important;
  margin-top: 0 !important;
  padding-top: 0 !important;
  gap: 6px !important;
  height: calc(100vh - 52px) !important;
  max-height: calc(100vh - 52px) !important;
  overflow: hidden !important;
}
#blob-sidebar-controls {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  max-height: calc(100vh - 60px) !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  display: flex !important;
  flex-direction: column !important;
  scrollbar-width: thin !important;
}
#blob-sidebar-controls::-webkit-scrollbar { width: 6px; }
#blob-sidebar-controls::-webkit-scrollbar-thumb {
  background: var(--border-color-primary, #555);
  border-radius: 3px;
}
#blob-sidebar-controls::-webkit-scrollbar-track { background: transparent; }
#blob-sidebar-controls > .block,
#blob-sidebar-controls > .column,
#blob-sidebar-controls .group {
  flex: 0 0 auto !important;
  height: auto !important;
  min-height: 0 !important;
  width: 100% !important;
}
#blob-sidebar-controls [style*="display: none"],
#blob-sidebar-controls [style*="display:none"],
#blob-sidebar-controls .hidden,
#blob-sidebar-controls .form[style*="display: none"],
#blob-sidebar-controls .form[style*="display:none"] {
  height: 0 !important;
  min-height: 0 !important;
  max-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  overflow: hidden !important;
  flex: 0 0 auto !important;
}
#blob-sidebar-wrap {
  border-right: 1px solid var(--blob-accent-border, var(--border-color-primary));
  align-self: stretch !important;
  min-height: calc(100vh - 52px) !important;
  height: calc(100vh - 52px) !important;
  max-height: calc(100vh - 52px) !important;
  max-width: """ + str(SIDEBAR_WIDTH) + """px !important;
  flex: 0 0 """ + str(SIDEBAR_WIDTH) + """px !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: hidden !important;
  padding-right: 6px !important;
  font-size: 0.78rem !important;
  box-sizing: border-box !important;
}
#blob-sidebar-wrap > .column,
#blob-sidebar-wrap > .form,
#blob-sidebar-wrap > .block {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  height: 100% !important;
  width: 100% !important;
  box-sizing: border-box !important;
}
#blob-preview {
  padding-left: 6px !important;
  min-width: 0 !important;
  flex: 1 1 auto !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: hidden !important;
  min-height: 0 !important;
}
#blob-preview-row {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  max-height: 100% !important;
  gap: 6px !important;
  align-items: stretch !important;
  flex-wrap: nowrap !important;
  flex-direction: row !important;
  overflow: hidden !important;
  width: 100% !important;
  display: flex !important;
}
#blob-output-col-wrap {
  flex: 1 1 auto !important;
  min-width: 0 !important;
  max-width: calc(100% - 290px) !important;
  display: flex !important;
  flex-direction: column !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  position: relative !important;
}
#blob-output-col-wrap > .group {
  flex: 0 0 auto !important;
  min-height: 0 !important;
  display: flex !important;
  flex-direction: column !important;
}
#blob-output-col-wrap > .group > .block,
#blob-output-col-wrap > .group > .form {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  display: flex !important;
  flex-direction: column !important;
}
#blob-right-col {
  flex: 0 0 280px !important;
  min-width: 280px !important;
  max-width: 280px !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 4px !important;
  border-left: 1px solid var(--blob-accent-border, var(--border-color-primary));
  padding-left: 6px !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
}
#blob-right-col > .group {
  flex: 0 0 auto !important;
}
#blob-right-col .blob-preview-footer {
  flex: 1 1 auto !important;
  min-height: 0 !important;
}
#blob-console-slot {
  flex: 0 0 auto !important;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  border-top: none !important;
  padding: 0 !important;
  margin: 0 !important;
  min-height: 88px !important;
  max-height: 140px !important;
  overflow: hidden !important;
}
#blob-console-slot .block,
#blob-console-slot .form { padding: 0 !important; margin: 0 !important; gap: 2px !important; }
#blob-console-slot label span { font-size: 0.68rem !important; line-height: 1 !important; }
#blob-console-slot .blob-console textarea {
  min-height: 72px !important;
  max-height: 96px !important;
}
#blob-sidebar-wrap textarea,
#blob-sidebar-wrap input[type="text"],
#blob-sidebar-wrap input[type="range"],
#blob-sidebar-wrap .input-container {
  visibility: visible !important;
  opacity: 1 !important;
}
#blob-sidebar-wrap textarea,
#blob-sidebar-wrap input[type="text"],
#blob-sidebar-wrap input[type="number"] {
  display: block !important;
  width: 100% !important;
  min-height: 2.5rem !important;
  background: var(--input-background-fill, #262640) !important;
  color: var(--body-text-color, #f5f5f5) !important;
  border: 1px solid var(--border-color-primary, #555) !important;
  box-shadow: var(--input-shadow, inset 0 1px 2px rgba(0,0,0,.25)) !important;
  border-radius: 6px !important;
  box-sizing: border-box !important;
}
#blob-sidebar-wrap fieldset {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
  width: 100% !important;
  visibility: visible !important;
  opacity: 1 !important;
  overflow: visible !important;
}
#blob-sidebar-wrap fieldset > label {
  display: inline-flex !important;
  align-items: center !important;
  gap: 4px !important;
  visibility: visible !important;
  opacity: 1 !important;
}
#blob-sidebar-wrap .input-container textarea {
  min-height: 3.25rem !important;
  width: 100% !important;
}
#blob-sidebar-wrap label.container {
  display: block !important;
  width: 100% !important;
}
#blob-sidebar-wrap input[type="radio"] {
  visibility: visible !important;
  opacity: 1 !important;
  width: auto !important;
  height: auto !important;
}
#blob-sidebar-wrap .block.hide-container,
#blob-sidebar-wrap .block {
  min-height: unset !important;
}
#blob-sidebar-wrap::-webkit-scrollbar { width: 6px; }
#blob-sidebar-wrap::-webkit-scrollbar-thumb {
  background: var(--border-color-primary, #555);
  border-radius: 4px;
}
#blob-sidebar-wrap .block { padding: 1px 0 !important; margin: 0 !important; }
#blob-sidebar-wrap .form { gap: 3px !important; }
#blob-sidebar-wrap label span { font-size: 0.78rem !important; }
#blob-sidebar-wrap .info {
  font-size: 0.55rem !important;
  line-height: 1.2 !important;
  opacity: 0.6;
  display: inline !important;
  margin-left: 4px !important;
}
#blob-sidebar-wrap label > span,
#blob-sidebar-wrap .label-wrap > .label {
  display: inline !important;
}
#blob-sidebar-wrap .blob-compact-slider .info {
  display: inline !important;
  margin-left: 6px !important;
}

.blob-status { font-size: 0.85rem !important; text-align: right !important; margin: 0 !important; }
.blob-mode-row { align-items: center !important; gap: 8px !important; margin-bottom: 2px !important; }
.blob-mode-row > .column:first-child { flex: 0 0 auto !important; min-width: 0 !important; }
.blob-mode-below p {
  font-size: 0.55rem !important;
  line-height: 1.25 !important;
  margin: 2px 0 4px 0 !important;
  opacity: 0.82;
}
.blob-mode-below { margin-bottom: 4px !important; }
#blob-output-col-wrap { padding: 0 !important; margin: 0 !important; }
#blob-output-col-wrap > .form:has(.blob-use-init-btn),
#blob-output-col-wrap .blob-use-init-btn,
#blob-dd-output-image + .blob-use-init-btn,
#blob-output-col-wrap:has(#blob-dd-output-image) .blob-use-init-btn {
  position: absolute !important;
  top: 30px !important;
  bottom: auto !important;
  right: 10px !important;
  left: auto !important;
  z-index: 40 !important;
  margin: 0 !important;
  padding: 0 !important;
  width: auto !important;
  max-width: 5.2rem !important;
  min-height: 0 !important;
  height: auto !important;
  flex: none !important;
  pointer-events: auto !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
}
.blob-use-init-btn button {
  font-size: 0.5rem !important;
  padding: 3px 6px !important;
  line-height: 1.05 !important;
  white-space: normal !important;
  text-align: center !important;
  border-radius: 4px !important;
  opacity: 0.92 !important;
}
#blob-output-image .upload-container,
#blob-output-image .upload-text,
#blob-output-image .icon-buttons,
#blob-output-image .empty,
#blob-output-image .empty-image,
#blob-output-image [data-testid="dropzone"],
#blob-output-col-wrap .upload-container,
#blob-output-col-wrap .upload-text,
#blob-output-col-wrap .icon-buttons { display: none !important; }
#blob-output-image .image-container,
#blob-output-image .contain,
#blob-output-image .wrap,
#blob-output-image .image-frame,
#blob-dd-output-image .image-container,
#blob-dd-output-image .contain,
#blob-dd-output-image .wrap,
#blob-dd-output-image .image-frame {
  height: 100% !important;
  width: 100% !important;
  max-width: 100% !important;
  max-height: 100% !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
#blob-output-image img,
#blob-dd-output-image img,
#blob-st-output-image img {
  object-fit: contain !important;
  width: 100% !important;
  height: 100% !important;
  max-width: 100% !important;
  max-height: 100% !important;
}
#blob-output-image .image-container,
#blob-output-image .contain,
#blob-dd-output-image .image-container,
#blob-dd-output-image .contain {
  flex: 1 1 auto !important;
}
#blob-output-col-wrap #blob-video-output,
#blob-output-col-wrap #blob-video-output .block,
#blob-output-col-wrap #blob-video-output .wrap,
#blob-output-col-wrap #blob-video-output .video-container,
#blob-output-col-wrap #blob-video-output > .form {
  width: 100% !important;
  max-width: 100% !important;
  height: """ + str(PREVIEW_SIZE) + """px !important;
  max-height: """ + str(PREVIEW_SIZE) + """px !important;
  aspect-ratio: auto !important;
  box-sizing: border-box !important;
}
#blob-output-col-wrap #blob-video-output video {
  width: 100% !important;
  height: 100% !important;
  max-width: 100% !important;
  max-height: 100% !important;
  object-fit: contain !important;
  background: #0a0a0a !important;
}
#blob-right-col #blob-video-input,
#blob-right-col #blob-video-input .block,
#blob-right-col #blob-video-input .wrap,
#blob-right-col #blob-video-input .video-container,
#blob-right-col #blob-video-input > .form {
  width: 100% !important;
  max-width: 100% !important;
  aspect-ratio: 1 / 1 !important;
  box-sizing: border-box !important;
}
#blob-right-col #blob-video-input video {
  width: 100% !important;
  height: 100% !important;
  max-width: 100% !important;
  max-height: 100% !important;
  object-fit: contain !important;
  background: #0a0a0a !important;
}
#blob-right-col #blob-source-image .upload-container,
#blob-right-col #blob-source-image .upload-text {
  display: flex !important;
}
#blob-dd-sketch-image .upload-container,
#blob-dd-sketch-image .upload-text,
#blob-dd-sketch-image [data-testid="dropzone"] { display: flex !important; }
#blob-dd-sketch-image .icon-buttons,
#blob-dd-sketch-image .image-buttons,
#blob-dd-sketch-image .tool-buttons {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
}
#blob-sketch-image .icon-buttons,
#blob-sketch-image .image-buttons,
#blob-sketch-image .tool-buttons {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  z-index: 50 !important;
  gap: 4px !important;
  flex-direction: row !important;
  align-items: center !important;
  flex-wrap: nowrap !important;
}
#blob-sketch-image .icon-buttons button,
#blob-sketch-image .image-buttons button,
#blob-sketch-image .tool-buttons button,
#blob-sketch-image [class*="download"],
#blob-sketch-image [class*="clear"],
#blob-sketch-image [class*="button"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
}
#blob-sketch-image {
  position: relative !important;
  overflow: visible !important;
}
#blob-sketch-image .wrap,
#blob-sketch-image .image-container,
#blob-sketch-image .contain,
#blob-sketch-image .form,
#blob-sketch-image .block {
  overflow: visible !important;
}
#blob-sketch-image .label-wrap,
#blob-sketch-image .label {
  overflow: visible !important;
  white-space: nowrap !important;
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 4px !important;
}
#blob-right-col .image-container { margin-top: 0 !important; }
#blob-preview-row {
  gap: """ + str(PREVIEW_GAP) + """px !important;
  align-items: stretch !important;
}
#blob-output-col-wrap .image-container,
#blob-output-col-wrap .contain,
#blob-output-col-wrap .wrap {
  width: 100% !important;
  max-width: 100% !important;
  max-height: """ + str(PREVIEW_SIZE) + """px !important;
  object-fit: contain !important;
}
#blob-output-col-wrap #blob-output-image,
#blob-output-col-wrap #blob-dd-output-image,
#blob-output-col-wrap #blob-st-output-image {
  height: """ + str(PREVIEW_SIZE) + """px !important;
  max-height: """ + str(PREVIEW_SIZE) + """px !important;
}
#blob-output-col-wrap #blob-output-image .image-container,
#blob-output-col-wrap #blob-output-image .contain,
#blob-output-col-wrap #blob-dd-output-image .image-container,
#blob-output-col-wrap #blob-dd-output-image .contain,
#blob-output-col-wrap #blob-st-output-image .image-container,
#blob-output-col-wrap #blob-st-output-image .contain {
  height: """ + str(PREVIEW_SIZE) + """px !important;
  max-height: """ + str(PREVIEW_SIZE) + """px !important;
  aspect-ratio: auto !important;
}
#blob-right-col .image-container,
#blob-right-col .contain,
#blob-right-col .wrap,
#blob-right-col .upload-container {
  width: 100% !important;
  max-width: 100% !important;
  aspect-ratio: 1 / 1 !important;
}
#blob-right-col > .group {
  width: 100% !important;
  margin: 0 !important;
}
#blob-preview-row img { object-fit: contain !important; width: 100% !important; height: 100% !important; }
#blob-source-image {
  overflow: visible !important;
}
#blob-source-image .block,
#blob-source-image .wrap,
#blob-st-content-image .block,
#blob-st-content-image .wrap {
  overflow: visible !important;
}
#blob-source-image .upload-container,
#blob-source-image .image-container,
#blob-source-image .contain,
#blob-source-image .image-frame,
#blob-st-content-image .upload-container,
#blob-st-content-image .image-container,
#blob-st-content-image .contain,
#blob-st-content-image .image-frame {
  aspect-ratio: 1 / 1 !important;
  width: 100% !important;
  max-width: 100% !important;
  min-height: 0 !important;
}
#blob-source-image .icon-buttons,
#blob-st-content-image .icon-buttons {
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  gap: 10px !important;
  margin-top: 6px !important;
  padding: 4px 0 8px !important;
  min-height: 38px !important;
  opacity: 1 !important;
  position: relative !important;
  z-index: 6 !important;
}
#blob-source-image .icon-buttons button,
#blob-st-content-image .icon-buttons button {
  transform: scale(1.15) !important;
}
#blob-right-col #blob-source-image,
#blob-right-col #blob-st-content-image {
  padding-bottom: 4px !important;
}
#blob-st-output-image .upload-container,
#blob-st-output-image .upload-text,
#blob-st-output-image .icon-buttons,
#blob-st-output-image .empty,
#blob-st-output-image [data-testid="dropzone"] {
  display: none !important;
}
#blob-st-style-image .image-container,
#blob-st-style-image .contain,
#blob-st-style-image .wrap {
  width: 100% !important;
  max-width: 100% !important;
  min-height: 180px !important;
  max-height: 220px !important;
}
.blob-meta-row { align-items: stretch !important; gap: 8px !important; flex-wrap: nowrap !important; }
.blob-meta-row > .column { min-width: 0 !important; }
.blob-abort-btn { flex: 0 0 auto !important; align-self: stretch !important; margin: 0 !important; padding: 0 !important; }
.blob-abort-btn button {
  background: #b71c1c !important;
  color: #fff !important;
  border: 1px solid #ff5252 !important;
  border-radius: 4px !important;
  min-width: 58px !important;
  width: 58px !important;
  min-height: 32px !important;
  height: 100% !important;
  font-size: 0.62rem !important;
  line-height: 1.1 !important;
  padding: 4px 6px !important;
  font-weight: 700 !important;
}
.blob-abort-btn button:hover { background: #d32f2f !important; }
.blob-preview-footer {
  gap: """ + str(PREVIEW_GAP) + """px !important;
  margin-top: 2px !important;
  align-items: stretch !important;
  flex-direction: column !important;
  flex-wrap: nowrap !important;
}
.blob-preview-footer > .column {
  flex: 0 0 auto !important;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
}
.blob-output-footer {
  padding: 0 !important;
  margin: 0 !important;
  gap: 2px !important;
}
.blob-output-footer .block,
.blob-output-footer .form { padding: 0 !important; margin: 0 !important; gap: 2px !important; }
.blob-output-footer label span { font-size: 0.62rem !important; line-height: 1 !important; }
.blob-seed-compact-row {
  align-items: center !important;
  gap: 4px !important;
  flex-wrap: wrap !important;
  margin: 0 !important;
}
.blob-seed-compact-row .block { padding: 0 !important; margin: 0 !important; flex: 0 0 auto !important; }
.blob-seed-compact-row .blob-seed-out { flex: 0 0 7.5rem !important; max-width: 7.5rem !important; }
.blob-seed-compact-row .blob-seed-out input {
  font-size: 0.68rem !important;
  padding: 2px 4px !important;
  min-height: 1.5rem !important;
  height: 1.5rem !important;
  line-height: 1.2 !important;
}
.blob-seed-compact-row fieldset { gap: 4px !important; padding: 0 !important; margin: 0 !important; }
.blob-seed-compact-row fieldset > label { font-size: 0.62rem !important; gap: 2px !important; }
.blob-seed-compact-row .blob-abort-btn { flex: 0 0 auto !important; margin-left: auto !important; }
.blob-seed-compact-row .blob-abort-btn button {
  min-width: 42px !important;
  width: 42px !important;
  min-height: 1.5rem !important;
  height: 1.5rem !important;
  font-size: 0.58rem !important;
  padding: 2px 4px !important;
}
.blob-seed-bar { align-items: center !important; gap: 6px !important; flex-wrap: wrap !important; }
.blob-seed-bar .block { padding: 0 !important; }
.blob-seed-out input { font-size: 0.78rem !important; padding: 2px 6px !important; }
.blob-output-footer .blob-meta textarea {
  font-size: 8px !important;
  min-height: 1.35rem !important;
  max-height: 1.65rem !important;
  line-height: 1.15 !important;
  padding: 2px 4px !important;
}
.blob-meta textarea {
  font-size: 8px !important;
  min-height: 1.35rem !important;
  max-height: 1.65rem !important;
  line-height: 1.15 !important;
}
#generate-btn, #dd-generate-btn, #st-generate-btn, #ms-generate-btn {
  width: 100% !important;
  margin: 2px 0 4px 0 !important;
  font-size: 0.9rem !important;
}
.blob-generate-top {
  margin-top: 2px !important;
  margin-bottom: 8px !important;
}
#blob-sidebar-controls.blob-gen-busy {
  pointer-events: none;
  opacity: 0.94;
}
#blob-sidebar-controls.blob-gen-busy .blob-generate-top,
#blob-sidebar-controls.blob-gen-busy #generate-btn,
#blob-sidebar-controls.blob-gen-busy #dd-generate-btn,
#blob-sidebar-controls.blob-gen-busy #st-generate-btn {
  pointer-events: none !important;
}
#blob-long-video-overlay {
  position: fixed !important;
  inset: 0 !important;
  z-index: 9999 !important;
  background: rgba(0, 0, 0, 0.72) !important;
  padding: 24px !important;
  /* Do NOT force display:flex here — it overrides Gradio hidden and blocks the whole UI. */
  align-items: center !important;
  justify-content: center !important;
}
#blob-long-video-overlay:not(.hidden) {
  display: flex !important;
}
#blob-long-video-overlay .blob-long-video-dialog {
  max-width: 520px !important;
  width: 100% !important;
  background: #1e1e1e !important;
  border: 1px solid #666 !important;
  border-radius: 10px !important;
  padding: 20px 22px !important;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.55) !important;
}
#blob-long-video-overlay .blob-long-video-dialog p {
  font-size: 0.95rem !important;
  line-height: 1.45 !important;
}
#blob-long-video-overlay .blob-long-video-actions {
  gap: 10px !important;
  margin-top: 12px !important;
  justify-content: flex-end !important;
}
.blob-disclaimer p {
  font-size: 0.6rem !important;
  line-height: 1.25 !important;
  opacity: 0.42;
  margin: 4px 0 0 0 !important;
  max-width: 820px;
}
#blob-js-hook { display: none !important; }
#blob-start-btn, #blob-disconnect-btn { display: none !important; }
#blob-header-banner-slot {
  display: none !important;
}
#blob-gallery-btn {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 120px !important;
  margin: 2px 0 0 0 !important;
  max-width: 200px !important;
}
#blob-output-overlay { display: none !important; }
#blob-header-banner {
  flex: 0 0 auto !important;
  min-width: 80px !important;
  max-width: 260px !important;
}
#blob-header-banner .blob-load-banner {
  font-size: 0.5rem !important;
  padding: 3px 8px !important;
  background: rgba(255, 200, 0, 0.18) !important;
  border: 1px solid rgba(255, 200, 0, 0.7) !important;
  border-radius: 4px !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  color: #ffd700 !important;
  text-align: center !important;
  font-weight: 700 !important;
  animation: blob-blink 1.2s ease-in-out infinite alternate;
}
#blob-header-banner .blob-load-banner.blob-banner-stopped {
  background: rgba(220, 60, 60, 0.15) !important;
  border-color: rgba(220, 60, 60, 0.5) !important;
  color: #d08080 !important;
}
#blob-header-banner .blob-load-stage {
  font-size: 0.46rem !important;
  opacity: 0.7;
}
@keyframes blob-blink {
  from { opacity: 0.55; }
  to { opacity: 1; }
}
.blob-disclaimer p {
  font-size: 0.52rem !important;
  line-height: 1.2 !important;
  opacity: 0.35;
  margin: 2px 0 0 0 !important;
}
.blob-credit p {
  font-size: 0.5rem !important;
  line-height: 1.2 !important;
  opacity: 0.3;
  margin: 1px 0 0 0 !important;
}
.blob-credit-mini p {
  font-size: 0.45rem !important;
  opacity: 0.25;
  margin: 0 !important;
}
#blob-header-wrap > .block:first-child {
  position: static !important;
  width: 100% !important;
  max-width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
  z-index: auto !important;
  pointer-events: auto !important;
  overflow: visible !important;
}
#blob-header-actions {
  margin-top: 0 !important;
  gap: 4px !important;
  flex-wrap: nowrap !important;
}
#blob-header-banner-slot .wrap,
#blob-header-banner-slot .html-container {
  min-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  width: auto !important;
  flex: 0 0 auto !important;
}
.blob-load-banner {
  border: 2px solid #e53935;
  border-radius: 6px;
  background: rgba(30, 10, 10, 0.94);
  color: #ff8a80;
  padding: 8px 16px;
  text-align: center;
  font-size: 0.82rem;
  font-weight: 600;
  line-height: 1.3;
  box-shadow: 0 4px 20px rgba(229, 57, 53, 0.4);
  max-width: 80%;
  margin: 0 auto;
}
.blob-load-banner.blob-banner-stopped {
  border-color: #78909c;
  background: rgba(22, 28, 34, 0.94);
  color: #cfd8dc;
  box-shadow: 0 2px 10px rgba(120, 144, 156, 0.2);
}
.blob-load-banner.blob-banner-warmup {
  border-color: #f9a825;
  background: rgba(40, 30, 5, 0.94);
  color: #ffe082;
  box-shadow: 0 2px 10px rgba(249, 168, 37, 0.25);
}
.blob-load-banner .blob-load-stage {
  font-size: 0.65rem; font-weight: 400; opacity: 0.88; margin-top: 2px;
}
.blob-load-banner.blob-load-blink {
  animation: blob-blink 0.45s ease-in-out 4;
}
.blob-load-banner.blob-banner-warmup.blob-load-blink {
  animation: blob-blink-warn 0.45s ease-in-out 4;
}
@keyframes blob-blink {
  0%, 100% { background: rgba(30, 10, 10, 0.94); border-color: #e53935; }
  50% { background: rgba(180, 20, 20, 0.95); border-color: #ff5252; }
}
@keyframes blob-blink-warn {
  0%, 100% { background: rgba(40, 30, 5, 0.94); border-color: #f9a825; }
  50% { background: rgba(249, 168, 37, 0.45); border-color: #ffb300; }
}
"""

_engine = None
_dd_engine = None
_st_engine = None
_upscale_engine = None
_current_family = "vqgan"
_last_meta = {}
_load = {
    "ready": False,
    "vqgan_ready": False,
    "sdxl_ready": False,
    "error": None,
    "stage": "Starting...",
    "blink_until": 0.0,
    "stopped": False,
    "warmup_bg": False,
    "warmup_stage": "",
}
_warmup_running = False
_codecs_setup_running = False
_style_setup_running = False
_generating = False
_video_long_ok = {"path": None, "ok": False}


def _video_upload_path(video_input):
    if video_input is None:
        return None
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict):
        return video_input.get("path") or video_input.get("name")
    path = getattr(video_input, "path", None) or getattr(video_input, "name", None)
    return path if path else None


def _video_codecs_ready():
    try:
        return video_codecs_status().get("ready", False)
    except Exception:
        return False


def _bundled_codecs_ready():
    try:
        return video_codecs_status().get("bundled_ready", False)
    except Exception:
        return False


def _style_weights_ready():
    try:
        return style_transfer_weights_status().get("ready", False)
    except Exception:
        return False


def resolve_ui_output_size(aspect, source_image=None):
    """Preset aspect sizes, or custom from upload with pixel-budget scaling."""
    if aspect == ASPECT_CUSTOM and source_image is not None:
        size = getattr(source_image, "size", None)
        if size and len(size) >= 2:
            return fit_size_preserving_aspect(int(size[0]), int(size[1]))
    if aspect in ASPECT_FORMATS:
        return resolve_aspect_size(aspect=aspect)
    return resolve_aspect_size(aspect=DEFAULT_ASPECT)


def _log_custom_size(source_image):
    if source_image is None:
        return
    size = getattr(source_image, "size", None)
    if not size or len(size) < 2:
        return
    src_w, src_h = int(size[0]), int(size[1])
    out_w, out_h = fit_size_preserving_aspect(src_w, src_h)
    blobvision_log.append(
        "Format: custom ({}x{} → {}x{}, ratio preserved).".format(
            src_w, src_h, out_w, out_h,
        )
    )


def _vqgan_ready():
    if _load.get("vqgan_ready"):
        return True
    return _engine is not None and _engine.vqgan_ready()


def _sdxl_ready():
    if _load.get("sdxl_ready"):
        return True
    return _engine is not None and _engine.sdxl_ready()


def _aux_cpu_during_warmup():
    import torch
    return _warmup_running and torch.cuda.is_available()


def _notify_sdxl_loading():
    _load["blink_until"] = time.time() + 3.0
    blobvision_log.append("SDXL still loading — redux / sketch modes need SDXL Turbo.")
    try:
        gr.Warning("SDXL still loading — please wait for SDXL Turbo.")
    except Exception:
        pass


def _notify_vqgan_loading():
    _load["blink_until"] = time.time() + 3.0
    blobvision_log.append("VQGAN still loading — please wait.")
    try:
        gr.Warning("VQGAN still loading — please wait.")
    except Exception:
        pass


def _codecs_install_btn_update(mode=None):
    if _current_family != "vqgan" or _bundled_codecs_ready():
        return gr.update(visible=False)
    label = "Downloading…" if _codecs_setup_running else "Install codecs"
    return gr.update(
        visible=True,
        interactive=not _codecs_setup_running,
        value=label,
    )


def poll_codecs_install_btn():
    return _codecs_install_btn_update()


def _style_install_btn_update():
    if _current_family != "style" or _style_weights_ready():
        return gr.update(visible=False)
    label = "Downloading…" if _style_setup_running else "Install model"
    return gr.update(
        visible=True,
        interactive=not _style_setup_running,
        value=label,
    )


def _video_mode_ui_updates(mode):
    """Hide image output in video mode; show upload/output when codecs ready."""
    if _current_family != "vqgan":
        return tuple(gr.update() for _ in range(9))
    show_video = mode == "video" and _video_codecs_ready()
    hidden = gr.update(visible=False)
    enc = gr.update(visible=show_video)
    if mode != "video":
        return (
            gr.update(visible=True),
            hidden,
            hidden,
            _codecs_install_btn_update(mode),
            hidden,
            hidden,
            hidden,
            hidden,
            hidden,
        )
    return (
        gr.update(visible=False),
        enc,
        enc,
        _codecs_install_btn_update(mode),
        enc,
        enc,
        gr.update(visible=show_video, interactive=False),
        gr.update(visible=show_video, interactive=False),
        enc,
    )


def activate_video_mode():
    global _codecs_setup_running
    if _bundled_codecs_ready():
        blobvision_log.append("Codecs BlobVision déjà installés dans video-codecs/.")
        return _video_mode_ui_updates("video")
    if _codecs_setup_running:
        blobvision_log.append("Installation codecs déjà en cours…")
        return _video_mode_ui_updates("video")

    def _worker():
        global _codecs_setup_running
        _codecs_setup_running = True
        try:
            from blobvision_paths import VIDEO_CODECS_ROOT
            blobvision_log.append(
                "Video: téléchargement ffmpeg + RIFE → {} …".format(VIDEO_CODECS_ROOT),
            )
            setup_bundled_video_codecs()
            status = video_codecs_status()
            if status.get("bundled_ready"):
                rife_note = " + RIFE" if status.get("bundled_rife") else " (RIFE manquant)"
                blobvision_log.append("Codecs BlobVision installés dans video-codecs/" + rife_note + ".")
            else:
                blobvision_log.append("Installation terminée mais ffmpeg bundlé manquant — réessayer.")
        except Exception as exc:
            blobvision_log.append("Échec installation codecs vidéo: " + str(exc))
        finally:
            _codecs_setup_running = False

    threading.Thread(target=_worker, name="video-codecs-setup", daemon=True).start()
    return _video_mode_ui_updates("video")


def activate_style_models():
    global _style_setup_running
    if _style_weights_ready():
        blobvision_log.append("VGG19 déjà installé dans models/style-transfer/.")
        return _style_install_pair()
    if _style_setup_running:
        blobvision_log.append("Téléchargement du modèle Style Transfer déjà en cours…")
        return _style_install_pair()

    def _worker():
        global _style_setup_running
        _style_setup_running = True
        try:
            download_style_transfer_weights(on_progress=blobvision_log.append)
            if _style_weights_ready():
                blobvision_log.append("Modèle Style Transfer prêt (VGG19).")
            else:
                blobvision_log.append("Téléchargement terminé mais poids manquants — réessayer.")
        except Exception as exc:
            blobvision_log.append("Échec téléchargement Style Transfer: " + str(exc))
        finally:
            _style_setup_running = False

    threading.Thread(target=_worker, name="style-weights-setup", daemon=True).start()
    return _style_install_pair()


def poll_style_install_btn():
    upd = _style_install_btn_update()
    return upd, upd


def _style_install_pair():
    upd = _style_install_btn_update()
    return upd, upd


def _abort_btn_active():
    return gr.update(visible=True, interactive=True)


def _abort_btn_idle():
    return gr.update(visible=False, interactive=False)


def abort_generation():
    if _generating:
        import blobvision_cancel
        blobvision_cancel.request()
        blobvision_log.append("Abort requested...")
    return gr.update()


def _gen_start():
    global _generating
    _generating = True
    return _abort_btn_active()


def _gen_end():
    global _generating
    _generating = False
    return _abort_btn_idle()


SIDEBAR_BUSY_ON_JS = """
() => {
  document.getElementById('blob-sidebar-controls')?.classList.add('blob-gen-busy');
}
"""
SIDEBAR_BUSY_OFF_JS = """
() => {
  document.getElementById('blob-sidebar-controls')?.classList.remove('blob-gen-busy');
}
"""


def _banner_html(blink=False):
    if _load.get("error"):
        title, stage, css = "LOAD FAILED", _load["error"], ""
    elif _load.get("stopped") and not _warmup_running:
        title, stage, css = "DISCONNECTED", "Press Start", " blob-banner-stopped"
    elif _load.get("warmup_bg"):
        title, stage, css = "WARMUP", _load.get("warmup_stage", "Pipeline warmup..."), " blob-banner-warmup"
    elif _load["ready"]:
        return ""
    else:
        title, stage, css = "LOADING", _load.get("stage", "Starting..."), " blob-banner-warmup"
    blink_cls = ""
    if blink:
        blink_cls = " blob-load-blink"
    return (
        '<div class="blob-load-banner' + css + blink_cls + '">'
        + title
        + '<div class="blob-load-stage">' + stage + "</div>"
        "</div>"
    )


def _bg_sketch_warmup():
    global _load
    _load["warmup_bg"] = True
    _load["warmup_stage"] = "SDXL pipeline warmup..."
    stop_heartbeat = threading.Event()

    def _on_stage(msg):
        _load["warmup_stage"] = msg

    def _on_progress(msg):
        _load["warmup_stage"] = msg

    def _heartbeat():
        elapsed = 0
        while not stop_heartbeat.wait(4.0):
            elapsed += 4
            _load["warmup_stage"] = (
                "SDXL warmup in progress ({}s) — first GPU pass, please wait...".format(elapsed)
            )

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()
    try:
        if _engine is not None:
            _engine.run_sketch_warmup(on_stage=_on_stage, on_progress=_on_progress)
    except Exception as exc:
        blobvision_log.append("Background warmup failed: " + str(exc))
    finally:
        stop_heartbeat.set()
        _load["warmup_bg"] = False
        _load["warmup_stage"] = ""


def _dispose_engine():
    global _engine
    if _engine is None:
        return
    try:
        _engine.shutdown()
    except Exception as exc:
        blobvision_log.append("Dispose: " + str(exc))
    _engine = None


def _dispose_dd_engine():
    global _dd_engine
    if _dd_engine is None:
        return
    try:
        _dd_engine._model = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        blobvision_log.append("DeepDream dispose: " + str(exc))
    _dd_engine = None


def _dispose_st_engine():
    global _st_engine
    if _st_engine is None:
        return
    try:
        _st_engine._vgg = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        blobvision_log.append("Style transfer dispose: " + str(exc))
    _st_engine = None


def _dispose_upscale_engine():
    global _upscale_engine
    if _upscale_engine is None:
        return
    try:
        _upscale_engine._model = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        blobvision_log.append("Upscale dispose: " + str(exc))
    _upscale_engine = None


def get_upscale_engine():
    global _upscale_engine
    if _upscale_engine is None:
        device = "cpu" if _aux_cpu_during_warmup() else None
        _upscale_engine = UpscaleEngine(device=device)
    return _upscale_engine


def _maybe_upscale(image_path, enabled):
    """Applied automatically at the end of generation when the sidebar's Upscale
    checkbox is on — a failed upscale shouldn't lose an otherwise-good generation,
    so it logs and falls back to the un-upscaled result instead of raising."""
    if not enabled or not image_path or not os.path.isfile(image_path):
        return image_path
    blobvision_log.append("Upscale x2: " + os.path.basename(image_path))
    try:
        return get_upscale_engine().upscale(image_path, on_progress=blobvision_log.append)
    except Exception as exc:
        blobvision_log.append("Upscale skipped (failed): " + str(exc))
        return image_path


def get_dd_engine():
    global _dd_engine
    if _dd_engine is None:
        device = "cpu" if _aux_cpu_during_warmup() else None
        _dd_engine = DeepDreamEngine(device=device)
    return _dd_engine


def get_st_engine():
    global _st_engine
    if _st_engine is None:
        device = "cpu" if _aux_cpu_during_warmup() else None
        _st_engine = StyleTransferEngine(device=device)
    return _st_engine


def get_engine(require_sdxl=False, require_vqgan=False):
    global _engine
    if _engine is None:
        raise gr.Error("Disconnected — press Start to load models.")
    if require_vqgan and not _vqgan_ready():
        raise gr.Error("VQGAN still loading — please wait.")
    if require_sdxl and not _sdxl_ready():
        raise gr.Error("SDXL still loading — please wait.")
    return _engine


def _family_btn_updates(active_id):
    updates = []
    for fid in FAMILY_META:
        meta = FAMILY_META[fid]
        classes = ["blob-family-btn", "blob-family-" + fid]
        if not meta["enabled"]:
            classes.append("blob-family-disabled")
        elif fid == active_id:
            classes.append("blob-family-active")
        updates.append(gr.update(elem_classes=classes))
    return updates


def _tagline_html(family_id):
    tag = FAMILY_META.get(family_id, FAMILY_META["vqgan"])["tagline"]
    return '<div class="blob-tagline">' + tag + "</div>"


def switch_family(family_id):
    global _current_family
    switch_outputs = 13 + len(FAMILY_META)
    if not FAMILY_META.get(family_id, {}).get("enabled"):
        return (gr.update(),) * switch_outputs
    _current_family = family_id
    blobvision_log.append("Emulator: " + FAMILY_META[family_id]["label"])
    vqgan = family_id == "vqgan"
    dd = family_id == "deepdream"
    st = family_id == "style"
    btn_updates = _family_btn_updates(family_id)
    codecs_btn = _codecs_install_btn_update() if vqgan else gr.update(visible=False)
    style_install = _style_install_btn_update() if st else gr.update(visible=False)
    return (
        gr.update(visible=vqgan),
        gr.update(visible=dd),
        gr.update(visible=st),
        gr.update(visible=vqgan),
        gr.update(visible=dd),
        gr.update(visible=st),
        gr.update(visible=vqgan),
        gr.update(visible=dd),
        gr.update(visible=st),
        gr.update(value=_tagline_html(family_id)),
        codecs_btn,
        style_install,
        style_install,
    ) + tuple(btn_updates)


def on_dd_mode_change(mode):
    if _current_family != "deepdream":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            gr.update(visible=False),
        )
    show_source = mode == "img2img"
    show_sketch = mode == "redux"
    return (
        gr.update(visible=show_source),
        gr.update(visible=show_sketch),
        DD_MODE_BLURBS.get(mode, ""),
        gr.update(visible=show_source),
    )


def use_dd_output_as_init(output_path):
    if not output_path:
        raise gr.Error("Generate an image first.")
    path = output_path if isinstance(output_path, str) else getattr(output_path, "name", None)
    if not path or not os.path.isfile(path):
        raise gr.Error("Output file not found.")
    blobvision_log.append("Using DeepDream output as init: " + os.path.basename(path))
    return path


def _prompt_seed(prompt, seed):
    if not prompt or not str(prompt).strip():
        return int(seed)
    import hashlib
    digest = hashlib.md5(str(prompt).strip().encode("utf-8")).hexdigest()
    mix = int(digest[:8], 16)
    return (int(seed) ^ mix) & 0x7FFFFFFF


def do_deepdream_generate(
    dd_prompt, dd_negative_prompt, aspect, dd_intensity, dd_layer,
    dd_sketch_upload, random_seed, seed_value, reuse_last_seed, dd_upscale,
):
    global _last_meta
    empty_sketch = gr.update()
    if _load.get("stopped") and not _warmup_running:
        _load["blink_until"] = time.time() + 2.0
        blobvision_log.append("Engine disconnected — press Start.")
        raise gr.Error("Engine disconnected — press Start.")
    has_sketch_upload = isinstance(dd_sketch_upload, str) and os.path.isfile(dd_sketch_upload)
    needs_sdxl_sketch = not has_sketch_upload and dd_prompt and dd_prompt.strip()
    if needs_sdxl_sketch and not _sdxl_ready():
        _notify_sdxl_loading()
        raise gr.Error("SDXL still loading — wait for the banner to clear.")
    sketch_out = empty_sketch
    try:
        source_for_size = None
        if has_sketch_upload and aspect == ASPECT_CUSTOM:
            from PIL import Image
            source_for_size = Image.open(dd_sketch_upload)
        width, height = resolve_ui_output_size(aspect, source_for_size)
        seed = None
        if reuse_last_seed and _last_meta.get("seed") is not None:
            seed = int(_last_meta["seed"])
        elif not random_seed:
            seed = int(seed_value)
        else:
            seed = random.randint(0, 2**31 - 1)
        init_path = None
        engine_mode = "txt2img"
        if has_sketch_upload:
            init_path = os.path.abspath(dd_sketch_upload)
            engine_mode = "img2img"
            blobvision_log.append("DeepDream img2img: using dropped image.")
        elif dd_prompt and dd_prompt.strip():
            dd = get_dd_engine()
            upload_dir = os.path.join(dd.output_dir, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            blobvision_log.append("Redux sketch: SDXL Turbo...")
            sketch_path = get_engine(require_sdxl=True).generate_redux_sketch(
                dd_prompt.strip(),
                width,
                height,
                seed,
                negative_prompt=(dd_negative_prompt or "").strip() or None,
                output_dir=upload_dir,
            )
            init_path = sketch_path
            sketch_out = sketch_path
            engine_mode = "redux"
            blobvision_log.append("Sketch saved — DeepDream pass...")
        else:
            blobvision_log.append(
                "No prompt, no image — dreaming from random noise."
            )
        steps, octaves, preserve_from_intensity = dd_intensity_to_params(dd_intensity)
        preserve = preserve_from_intensity if engine_mode in ("redux", "img2img") else None
        result = get_dd_engine().generate(
            mode=engine_mode,
            width=width,
            height=height,
            steps=steps,
            octaves=octaves,
            layer=dd_layer,
            init_image_path=init_path,
            seed=seed,
            prompt=(dd_prompt or "").strip(),
            negative_prompt=(dd_negative_prompt or "").strip(),
            preserve=preserve,
            on_progress=blobvision_log.append,
        )
    except Exception as exc:
        blobvision_log.append("DeepDream error: " + str(exc))
        raise gr.Error("DeepDream failed: " + str(exc)) from exc
    _last_meta = dict(result.metadata)
    meta_text = json.dumps(result.metadata, ensure_ascii=False)
    output_path = _maybe_upscale(result.output_path, dd_upscale)
    return (
        poll_banner(),
        output_path,
        sketch_out if init_path and engine_mode == "redux" else empty_sketch,
        int(result.seed),
        meta_text,
        _gen_end(),
    )


def on_st_content_upload(content_image):
    if content_image is None:
        return gr.update()
    _log_custom_size(content_image)
    return gr.update(value=ASPECT_CUSTOM)


def on_st_preset_change(preset_key):
    is_classic = not preset_key
    if is_classic:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            ST_MODE_BLURB,
            gr.update(visible=False),
            gr.update(),
        )
    preset = STYLE_PRESETS.get(preset_key, STYLE_PRESETS[DEFAULT_STYLE_PRESET])
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        ST_SDXL_MODE_BLURB,
        gr.update(visible=preset_key == "custom"),
        gr.update(value=preset["strength"]),
    )


def do_style_generate(
    st_aspect, st_style_image, st_style_strength, st_content_weight, st_steps,
    st_preset, st_preset_custom_prompt, st_preset_strength,
    st_content_image, random_seed, seed_value, reuse_last_seed, st_upscale,
):
    global _last_meta
    if st_content_image is None:
        blobvision_log.append("Error: upload a content image on the right.")
        raise gr.Error("Upload a content image on the right.")
    seed = None
    if reuse_last_seed and _last_meta.get("seed") is not None:
        seed = int(_last_meta["seed"])
    elif not random_seed:
        seed = int(seed_value)
    else:
        seed = random.randint(0, 2**31 - 1)

    if st_preset:
        if not _sdxl_ready():
            _notify_sdxl_loading()
            raise gr.Error("SDXL still loading — wait for the banner to clear.")
        preset = STYLE_PRESETS.get(st_preset, STYLE_PRESETS[DEFAULT_STYLE_PRESET])
        prompt = (st_preset_custom_prompt or "").strip() if st_preset == "custom" else preset["prompt"]
        if not prompt:
            blobvision_log.append("Error: enter a custom prompt for the Custom preset.")
            raise gr.Error("Enter a custom prompt for the Custom preset.")
        try:
            width, height = resolve_ui_output_size(st_aspect, st_content_image)
            out_dir = os.path.join(get_st_engine().output_dir, "uploads")
            os.makedirs(out_dir, exist_ok=True)
            content_path = os.path.join(out_dir, uuid.uuid4().hex + "_content.png")
            st_content_image.save(content_path)
            engine = get_engine(require_sdxl=True)
            result = engine.generate_style_preset(
                image_path=content_path,
                prompt=prompt,
                strength=float(st_preset_strength),
                seed=seed,
                negative_prompt=preset.get("negative_prompt"),
                steps=preset.get("steps", STYLE_PRESET_STEPS),
                width=width,
                height=height,
                output_dir=get_st_engine().output_dir,
                on_progress=blobvision_log.append,
            )
        except gr.Error:
            raise
        except Exception as exc:
            blobvision_log.append("SDXL restyle error: " + str(exc))
            raise gr.Error("SDXL restyle failed: " + str(exc)) from exc
        _last_meta = dict(result.metadata)
        meta_text = json.dumps(result.metadata, ensure_ascii=False)
        output_path = _maybe_upscale(result.output_path, st_upscale)
        return (
            poll_banner(),
            output_path,
            int(result.seed),
            meta_text,
            _gen_end(),
        )

    if not _style_weights_ready():
        _load["blink_until"] = time.time() + 2.0
        blobvision_log.append("Modèle VGG19 manquant — clique « Install model » (header ou sidebar).")
        raise gr.Error("VGG19 model missing — click Install model.")
    if st_style_image is None:
        blobvision_log.append("Error: upload a style reference in the sidebar.")
        raise gr.Error("Upload a style reference in the sidebar.")
    try:
        width, height = resolve_ui_output_size(st_aspect, st_content_image)
        st = get_st_engine()
        upload_dir = os.path.join(st.output_dir, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        content_path = os.path.join(upload_dir, uuid.uuid4().hex + "_content.png")
        style_path = os.path.join(upload_dir, uuid.uuid4().hex + "_style.png")
        st_content_image.save(content_path)
        st_style_image.save(style_path)
        result = st.generate(
            content_image_path=content_path,
            style_image_path=style_path,
            width=width,
            height=height,
            steps=int(st_steps),
            style_strength=float(st_style_strength),
            content_weight=float(st_content_weight),
            seed=seed,
            on_progress=blobvision_log.append,
        )
    except Exception as exc:
        blobvision_log.append("Style transfer error: " + str(exc))
        raise gr.Error("Style transfer failed: " + str(exc)) from exc
    _last_meta = dict(result.metadata)
    meta_text = json.dumps(result.metadata, ensure_ascii=False)
    output_path = _maybe_upscale(result.output_path, st_upscale)
    return (
        poll_banner(),
        output_path,
        int(result.seed),
        meta_text,
        _gen_end(),
    )


def _warmup_worker():
    global _engine, _load, _warmup_running
    _warmup_running = True
    try:
        time.sleep(0.5)
        _load["stopped"] = False
        _load["error"] = None
        _load["warmup_bg"] = False

        def _on_stage(msg):
            _load["stage"] = msg

        _load["stage"] = "Starting engine..."
        _load["vqgan_ready"] = False
        _load["sdxl_ready"] = False
        _dispose_engine()
        _engine = BlobVisionEngine(keep_models=True)

        def _on_vqgan_ready():
            _load["vqgan_ready"] = True
            blobvision_log.append("VQGAN ready — legacy VQGAN / DeepDream / Style Transfer can use GPU soon.")

        def _on_sdxl_ready():
            _load["sdxl_ready"] = True
            blobvision_log.append("SDXL Turbo ready — redux sketch modes unlocked.")

        _engine.warmup_staged(
            on_stage=_on_stage,
            on_vqgan_ready=_on_vqgan_ready,
            on_sdxl_ready=_on_sdxl_ready,
            get_current_family=lambda: _current_family,
        )
        _load["ready"] = True
        _load["stage"] = "Ready"
        print("Models ready (SDXL loaded; VQGAN may still be finishing in background).", flush=True)
        if _engine is not None and not os.environ.get("BLOBVISION_SKIP_WARMUP", "").strip().lower() in ("1", "true", "yes"):
            threading.Thread(target=_bg_sketch_warmup, daemon=True).start()
    except Exception as exc:
        import traceback
        _load["error"] = str(exc)
        _load["stage"] = "Failed"
        blobvision_log.append("Load failed: " + str(exc))
        for line in traceback.format_exc().splitlines()[-6:]:
            blobvision_log.append(line)
    finally:
        _warmup_running = False


def poll_status():
    if _load.get("error"):
        return "**Error:** " + _load["error"]
    if _load["ready"]:
        if _load.get("warmup_bg"):
            return "**Ready** — warmup..."
        return "**Ready** — SDXL + VQGAN loaded."
    if _load.get("stopped") and not _warmup_running:
        return "**Disconnected** — press Start."
    stage = _load.get("stage") or "Starting..."
    if _load.get("vqgan_ready") and not _load.get("sdxl_ready"):
        return "**Loading SDXL…** — " + stage + " (VQGAN ready — legacy / DeepDream / Style OK)"
    return "**Loading…** — " + stage


def poll_start_btn():
    if _load["ready"]:
        return gr.update(value="Running", interactive=False, variant="secondary")
    if _warmup_running:
        return gr.update(value="Loading…", interactive=False, variant="secondary")
    if _load.get("stopped") or _load.get("error"):
        return gr.update(value="Start", interactive=True, variant="primary")
    return gr.update(value="Loading…", interactive=False, variant="secondary")


def start_system():
    global _warmup_running
    if _load["ready"]:
        blobvision_log.append("System already running.")
        return poll_status(), poll_start_btn(), poll_banner()
    if _warmup_running:
        blobvision_log.append("Already loading models…")
        return poll_status(), poll_start_btn(), poll_banner()
    blobvision_log.append("Start — loading models…")
    _load["ready"] = False
    _load["error"] = None
    _load["stopped"] = False
    _load["stage"] = "Starting..."
    threading.Thread(target=_warmup_worker, daemon=True).start()
    return poll_status(), poll_start_btn(), poll_banner()


def poll_banner():
    blink = time.time() < _load.get("blink_until", 0)
    if _load.get("error"):
        return gr.update(value=_banner_html(blink=blink), visible=True)
    if _load.get("stopped") and not _warmup_running:
        return gr.update(value=_banner_html(), visible=True)
    if not _load["ready"] and not _load.get("stopped"):
        return gr.update(value=_banner_html(blink=blink), visible=True)
    if _load.get("warmup_bg"):
        return gr.update(value=_banner_html(blink=False), visible=True)
    return gr.update(value="", visible=False)


def on_mode_change(mode):
    global _video_long_ok
    if _current_family != "vqgan":
        return (
            gr.update(), gr.update(),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False), "",
        )
    if mode != "video":
        _video_long_ok = {"path": None, "ok": False}
    it = DEFAULT_ITERATIONS.get(mode, DEFAULT_ITERATIONS.get("corrupt", 25))
    den = DEFAULT_DENOISE.get(mode, DEFAULT_DENOISE.get("corrupt", 0.3))
    cap = ITER_SLIDER_MAX.get(mode, 100)
    show_sketch = mode == "redux"
    show_source = mode == "legacy"
    video_ui = _video_mode_ui_updates(mode)
    return (
        gr.update(value=min(it, cap), maximum=cap),
        gr.update(value=den),
        gr.update(visible=show_source),
        gr.update(visible=show_sketch),
        gr.update(visible=show_source),
        video_ui[0],
        video_ui[1],
        video_ui[2],
        video_ui[3],
        video_ui[4],
        video_ui[5],
        video_ui[6],
        video_ui[7],
        video_ui[8],
        gr.update(visible=False),
        MODE_BLURBS.get(mode, ""),
    )


def on_encode_range_toggle(enabled):
    on = bool(enabled)
    return gr.update(interactive=on), gr.update(interactive=on)


def refresh_sketch_components(mode_val=None):
    """Force re-render of image upload components after family switch
    so Gradio re-initializes their drag-and-drop handlers."""
    vqgan = _current_family == "vqgan"
    dd = _current_family == "deepdream"
    st = _current_family == "style"
    show_sketch = vqgan and mode_val == "redux"
    show_source = vqgan and mode_val == "legacy"
    return (
        gr.update(visible=show_sketch),
        gr.update(visible=show_source),
        gr.update(visible=dd),
        gr.update(visible=st),
        gr.update(visible=st),
    )


def on_source_image_upload(init_image, mode):
    if mode != "legacy" or init_image is None:
        return gr.update()
    if isinstance(init_image, str) and os.path.isfile(init_image):
        from PIL import Image
        _log_custom_size(Image.open(init_image))
    return gr.update(value=ASPECT_CUSTOM)


def on_sketch_upload(sketch_img):
    if sketch_img is None:
        return gr.update()
    if isinstance(sketch_img, str) and os.path.isfile(sketch_img):
        from PIL import Image
        _log_custom_size(Image.open(sketch_img))
    return gr.update(value=ASPECT_CUSTOM)


def on_video_upload(video_input, mode):
    global _video_long_ok
    hide_overlay = gr.update(visible=False)
    no_aspect = gr.update()
    if mode != "video" or not video_input:
        _video_long_ok = {"path": None, "ok": False}
        return hide_overlay, gr.update(value=""), no_aspect, gr.update()
    path = _video_upload_path(video_input)
    if not path or not os.path.isfile(path):
        _video_long_ok = {"path": None, "ok": False}
        return hide_overlay, gr.update(value=""), no_aspect, gr.update()
    try:
        info = probe_video_file(path)
        duration = float(info.get("duration") or 0.0)
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
    except Exception as exc:
        blobvision_log.append("Video probe failed: " + str(exc))
        _video_long_ok = {"path": path, "ok": True}
        return hide_overlay, gr.update(value=""), no_aspect, gr.update()
    if width > 0 and height > 0:
        guessed = guess_aspect_from_size(width, height)
        blobvision_log.append(
            "Format auto-detected: {} ({}x{} from source video).".format(
                guessed, width, height,
            ),
        )
        aspect_up = gr.update(value=guessed)
    else:
        aspect_up = gr.update()
    encode_to_up = gr.update(value=max(0.1, duration))
    if duration <= VIDEO_LONG_WARN_SECONDS:
        _video_long_ok = {"path": path, "ok": True}
        return hide_overlay, gr.update(value=""), aspect_up, encode_to_up
    _video_long_ok = {"path": path, "ok": False}
    dur_text = format_video_duration(duration)
    msg = (
        "The video is **{}** long. Blobification may take a very long time. "
        "Are you sure you want to continue?".format(dur_text)
    )
    return gr.update(visible=True), gr.update(value=msg), aspect_up, encode_to_up


def on_long_video_yes():
    global _video_long_ok
    _video_long_ok["ok"] = True
    return gr.update(visible=False)


def on_long_video_no():
    global _video_long_ok
    _video_long_ok = {"path": None, "ok": False}
    return gr.update(visible=False), gr.update(value=None)


def on_dd_source_upload(init_image):
    if init_image is None:
        return gr.update()
    if isinstance(init_image, str) and os.path.isfile(init_image):
        from PIL import Image
        _log_custom_size(Image.open(init_image))
    return gr.update(value=ASPECT_CUSTOM)


def use_output_as_init(output_path, mode):
    if mode != "legacy":
        raise gr.Error("Switch to legacy mode first.")
    if not output_path:
        raise gr.Error("Generate an image first.")
    path = output_path if isinstance(output_path, str) else getattr(output_path, "name", None)
    if not path or not os.path.isfile(path):
        raise gr.Error("Output file not found.")
    blobvision_log.append("Using output as init image: " + os.path.basename(path))
    return path


def do_generate(
    mode, prompt, negative_prompt, init_image, sketch_upload, video_input, aspect,
    iterations_slider, denoise, random_seed, seed_value, reuse_last_seed,
    video_encode_range, video_encode_from, video_encode_to, video_frame_step,
    upscale_x2,
):
    global _last_meta, _generating
    blobvision_log.append("DEGENERATE — mode={}.".format(mode))
    if _load.get("stopped") and not _warmup_running:
        _load["blink_until"] = time.time() + 2.0
        blobvision_log.append("Engine disconnected — press Start.")
        raise gr.Error("Engine disconnected — press Start.")
    if mode == "legacy":
        if not _vqgan_ready():
            _notify_vqgan_loading()
            raise gr.Error("VQGAN still loading — wait for the banner to clear.")
    elif mode == "redux":
        if not _sdxl_ready():
            _notify_sdxl_loading()
            raise gr.Error("SDXL still loading — wait for the banner to clear.")
        if not _vqgan_ready():
            _notify_vqgan_loading()
            raise gr.Error("VQGAN still loading — wait for the banner to clear.")
    elif mode == "video":
        if not _vqgan_ready():
            _notify_vqgan_loading()
            raise gr.Error("VQGAN still loading — wait for the banner to clear.")
    has_img2img = (mode == "redux" and sketch_upload
                   and isinstance(sketch_upload, str) and os.path.isfile(sketch_upload))
    if not prompt or not prompt.strip():
        if has_img2img:
            prompt = ""
        else:
            raise gr.Error("Enter a prompt before generating.")
    if mode == "video":
        if not _video_codecs_ready():
            raise gr.Error(
                "Video codecs missing — click « Install codecs » in the header.",
            )
        video_path = _video_upload_path(video_input)
        if not video_path or not os.path.isfile(video_path):
            raise gr.Error("Video mode requires a source video upload.")
        try:
            duration = float(probe_video_file(video_path).get("duration") or 0.0)
        except Exception as exc:
            raise gr.Error("Could not read video duration: " + str(exc)) from exc
        if duration > VIDEO_LONG_WARN_SECONDS:
            confirmed = (
                _video_long_ok.get("path") == video_path and _video_long_ok.get("ok")
            )
            if not confirmed:
                raise gr.Error(
                    "Long video — confirm the warning after upload (Yes) before generating.",
                )
        if video_encode_range:
            encode_from = max(0.0, float(video_encode_from or 0.0))
            encode_to = float(video_encode_to or duration)
            if encode_to <= encode_from:
                raise gr.Error("End conversion time must be after encode from time.")
            if encode_from >= duration:
                raise gr.Error("Encode from time is beyond video duration.")
    try:
        if mode == "legacy":
            engine = get_engine(require_vqgan=True)
        elif mode == "redux":
            engine = get_engine(require_sdxl=True, require_vqgan=True)
        else:
            engine = get_engine(require_vqgan=True)
        seed = None
        if reuse_last_seed and _last_meta.get("seed") is not None:
            seed = int(_last_meta["seed"])
        elif not random_seed:
            seed = int(seed_value)
        import blobvision_cancel
        blobvision_cancel.clear()
        neg = (negative_prompt or "").strip() or None
        if mode == "video":
            video_path = _video_upload_path(video_input)
            use_range = bool(video_encode_range)
            encode_from = max(0.0, float(video_encode_from or 0.0))
            encode_to = float(video_encode_to or duration)
            if use_range:
                blobvision_log.append(
                    "Video encode range: {:.1f}s -> {:.1f}s.".format(encode_from, encode_to),
                )
            frame_step = parse_vq_video_frame_step(video_frame_step)
            if frame_step <= 1:
                blobvision_log.append(
                    "Video job started — thinning OFF (every frame will be blobified).".format(),
                )
            else:
                blobvision_log.append(
                    "Video job started — thinning 1 every {} frames (UI: {}).".format(
                        frame_step, video_frame_step,
                    ),
                )
            result = engine.generate_video(
                prompt=prompt.strip(),
                video_path=video_path,
                negative_prompt=neg,
                iterations=int(iterations_slider),
                denoise_fidelity=float(denoise),
                seed=seed,
                aspect=aspect,
                frame_step=frame_step,
                encode_from_sec=encode_from if use_range else 0.0,
                encode_to_sec=encode_to if use_range else None,
                use_encode_range=use_range,
                on_progress=blobvision_log.append,
            )
            _last_meta = dict(result.metadata)
            meta_text = json.dumps(result.metadata, ensure_ascii=False)
            return (
                poll_banner(),
                gr.update(value=None),
                gr.update(value=None),
                int(seed or 0),
                meta_text,
                result.output_path,
                _gen_end(),
            )
        init_path = None
        source_for_size = None
        if aspect == ASPECT_CUSTOM:
            if mode == "legacy" and init_image and isinstance(init_image, str) and os.path.isfile(init_image):
                from PIL import Image
                source_for_size = Image.open(init_image)
            elif mode == "redux" and sketch_upload and isinstance(sketch_upload, str) and os.path.isfile(sketch_upload):
                from PIL import Image
                source_for_size = Image.open(sketch_upload)
        out_w, out_h = resolve_ui_output_size(aspect, source_for_size)
        if init_image and isinstance(init_image, str) and os.path.isfile(init_image):
            init_path = os.path.abspath(init_image)
        # Redux img2img: if user dropped an image in the sketch panel, use it as init
        if mode == "redux" and sketch_upload and isinstance(sketch_upload, str) and os.path.isfile(sketch_upload):
            init_path = os.path.abspath(sketch_upload)
            blobvision_log.append("Redux img2img: using user image from sketch panel.")
        result = engine.generate(
            mode=mode,
            prompt=prompt.strip(),
            negative_prompt=neg,
            iterations=int(iterations_slider),
            denoise_fidelity=float(denoise),
            seed=seed,
            init_image_path=init_path,
            aspect=aspect,
            width=out_w,
            height=out_h,
        )
    except gr.Error:
        _generating = False
        raise
    except blobvision_cancel.AbortedError:
        blobvision_log.append("Generation aborted.")
        _generating = False
        return (
            poll_banner(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            _gen_end(),
        )
    except Exception as exc:
        blobvision_log.append("Generate error: " + str(exc))
        _generating = False
        raise gr.Error("Generation failed: " + str(exc)) from exc
    _last_meta = dict(result.metadata)
    meta_text = json.dumps(result.metadata, ensure_ascii=False)
    sketch_out = result.sketch_path if result.sketch_path and os.path.isfile(result.sketch_path) else None
    output_path = _maybe_upscale(result.output_path, upscale_x2)
    return (
        poll_banner(),
        output_path, sketch_out, int(result.seed), meta_text,
        gr.update(value=None),
        _gen_end(),
    )


def poll_console():
    return blobvision_log.tail(100)


def disconnect_server():
    global _engine, _warmup_running
    if _warmup_running:
        blobvision_log.append("Disconnect - wait for loading to finish.")
        return poll_status(), poll_start_btn(), poll_banner()
    blobvision_log.append("Disconnect - unloading models...")
    _dispose_engine()
    _dispose_dd_engine()
    _dispose_st_engine()
    _dispose_upscale_engine()
    _load["ready"] = False
    _load["vqgan_ready"] = False
    _load["sdxl_ready"] = False
    _load["stopped"] = True
    _load["error"] = None
    _load["warmup_bg"] = False
    _load["warmup_stage"] = ""
    _load["stage"] = "Disconnected"
    _load["blink_until"] = 0.0
    blobvision_log.append("Disconnected - press Start to reload models.")
    return poll_status(), poll_start_btn(), poll_banner()

def open_gallery():
    # All families now share one flat outputs/ folder (see this session's
    # outputs-restructuring work — filenames carry a V/D/S/M type tag
    # instead of a subfolder), so there's no per-family engine/dir to
    # resolve anymore.
    from blobvision_paths import outputs_dir
    out = os.path.abspath(outputs_dir())
    os.makedirs(out, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(out)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", out])
    else:
        subprocess.Popen(["xdg-open", out])


def build_ui():
    with gr.Blocks(title="BlobVision") as demo:
        with gr.Column(elem_id="blob-header-wrap"):
            with gr.Row(elem_id="blob-header"):
                with gr.Column(scale=0, min_width=180, elem_id="blob-header-left"):
                    gr.HTML(_header_brand_html())
                    tagline_html = gr.HTML(
                        _tagline_html("vqgan"),
                        elem_id="blob-tagline-slot",
                    )
                family_btns = {}
                with gr.Row(elem_id="blob-header-toolbar"):
                    for fid, meta in FAMILY_META.items():
                        cls = ["blob-family-btn", "blob-family-" + fid]
                        if fid == "vqgan":
                            cls.append("blob-family-active")
                        if not meta["enabled"]:
                            cls.append("blob-family-disabled")
                        family_btns[fid] = gr.Button(
                            meta["label"],
                            interactive=meta["enabled"],
                            elem_classes=cls,
                        )
                    load_status = gr.Markdown(poll_status(), elem_classes=["blob-status"])
                    load_banner = gr.HTML(
                        _banner_html(),
                        visible=not _load["ready"],
                        elem_id="blob-header-banner",
                        elem_classes=["blob-header-banner"],
                    )
                    with gr.Row(elem_id="blob-header-actions"):
                        start_btn = gr.Button("Start", size="sm", variant="primary", interactive=True, elem_id="blob-start-btn")
                        disconnect_btn = gr.Button("Disconnect", size="sm", variant="stop", elem_id="blob-disconnect-btn")
                        install_codecs_btn = gr.Button(
                            "Install codecs",
                            size="sm",
                            visible=False,
                            elem_id="blob-install-codecs-btn",
                        )
                        install_style_btn = gr.Button(
                            "Install model",
                            size="sm",
                            visible=False,
                            elem_id="blob-install-style-btn",
                        )

        with gr.Row(equal_height=False, elem_id="blob-main-row"):
            with gr.Column(scale=4, min_width=SIDEBAR_WIDTH, elem_id="blob-sidebar-wrap"):
                with gr.Column(elem_id="blob-sidebar-controls"):
                    with gr.Group(visible=True, elem_id="vqgan-sidebar-panel") as vqgan_sidebar_panel:
                        mode = gr.Radio(
                            ["redux", "legacy", "video"],
                            value="redux",
                            label="Mode",
                        )
                        mode_help = gr.Markdown(
                            MODE_BLURBS["redux"],
                            elem_classes=["blob-mode-below"],
                        )
                        generate_btn = gr.Button(
                            "DEGENERATE", variant="primary", elem_id="generate-btn",
                            elem_classes=["blob-generate-top"],
                        )

                        prompt = gr.Textbox(
                            label="Prompt",
                            lines=2,
                            placeholder="Describe SLOP",
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative prompt",
                            lines=1,
                            max_lines=1,
                            elem_classes=["blob-negative-prompt"],
                            placeholder="Separate slop with commas",
                        )
                        aspect = gr.Radio(
                            ASPECT_CHOICES,
                            value=DEFAULT_ASPECT,
                            label="Format",
                            info=ASPECT_FORMAT_INFO,
                        )

                        iterations = gr.Slider(
                            0, ITER_SLIDER_MAX["redux"], value=DEFAULT_ITERATIONS["redux"], step=1, precision=0,
                            label="Steps/Blob",
                            info="0-100 standard, 150-500 legacy.",
                            elem_classes=["blob-compact-slider"],
                        )
                        denoise = gr.Slider(
                            0.0, 1.0, value=DEFAULT_DENOISE["redux"], step=0.05, precision=2,
                            label="Deslop",
                            info="Closer to 0 = closer to init",
                            elem_classes=["blob-compact-slider"],
                        )
                        upscale_x2 = gr.Checkbox(
                            value=False,
                            label="Upscale output x2 (Real-ESRGAN)",
                        )
                        video_frame_step = gr.Radio(
                            VQ_VIDEO_FRAME_STEP_CHOICES,
                            value=DEFAULT_VQ_VIDEO_FRAME_STEP,
                            label="Keyframe thinning",
                            info="off = every frame blobified. 2/4 = fewer keyframes, RIFE/ffmpeg fills gaps.",
                            visible=False,
                        )
                        video_encode_range = gr.Checkbox(
                            value=False,
                            label="Limit encode range",
                            info="Encode only part of the source clip.",
                            visible=False,
                        )
                        with gr.Row(visible=False) as video_encode_row:
                            video_encode_from = gr.Number(
                                value=0,
                                minimum=0,
                                precision=1,
                                label="Encode from (seconds)",
                                interactive=False,
                            )
                            video_encode_to = gr.Number(
                                value=0,
                                minimum=0,
                                precision=1,
                                label="End conversion at (seconds)",
                                interactive=False,
                            )

                    with gr.Group(visible=False, elem_id="deepdream-sidebar-panel") as deepdream_sidebar_panel:
                        dd_mode_help = gr.Markdown(
                            DD_MODE_BLURBS["redux"],
                            elem_classes=["blob-mode-below"],
                        )
                        dd_generate_btn = gr.Button(
                            "DEGENERATE", variant="primary", elem_id="dd-generate-btn",
                            elem_classes=["blob-generate-top"],
                        )
                        dd_prompt = gr.Textbox(
                            label="Prompt",
                            lines=2,
                            placeholder="Describe SLOP",
                        )
                        dd_negative_prompt = gr.Textbox(
                            label="Negative prompt",
                            lines=1,
                            max_lines=1,
                            elem_classes=["blob-negative-prompt"],
                            placeholder="Separate slop with commas",
                        )
                        dd_aspect = gr.Radio(
                            ASPECT_CHOICES,
                            value=DEFAULT_ASPECT,
                            label="Format",
                            info=ASPECT_FORMAT_INFO,
                        )
                        dd_layer = gr.Radio(
                            choices=["mixed5b", "mixed6a", "mixed7"],
                            value="mixed6a",
                            label="Inception layer",
                            info="mixed5b = faces; mixed6a = best default; mixed7 = abstract.",
                        )
                        dd_intensity = gr.Slider(
                            0, 100, value=DD_DEFAULT_INTENSITY, step=5, precision=0,
                            label="Intensity",
                            info="How hard to dream. Low = your image, barely touched. High = fully hallucinated.",
                            elem_classes=["blob-compact-slider"],
                        )
                        dd_upscale = gr.Checkbox(
                            value=False,
                            label="Upscale output x2 (Real-ESRGAN)",
                        )

                    with gr.Group(visible=False, elem_id="style-sidebar-panel") as style_sidebar_panel:
                        st_mode_help = gr.Markdown(
                            ST_MODE_BLURB,
                            elem_classes=["blob-mode-below"],
                        )
                        st_generate_btn = gr.Button(
                            "DEGENERATE", variant="primary", elem_id="st-generate-btn",
                            elem_classes=["blob-generate-top"],
                        )
                        st_install_models_btn = gr.Button(
                            "Install VGG19 model (~548 MB)",
                            visible=False,
                            elem_id="st-install-models-btn",
                        )
                        st_aspect = gr.Radio(
                            ASPECT_CHOICES,
                            value=ASPECT_CUSTOM,
                            label="Format",
                            info=ASPECT_FORMAT_INFO,
                        )
                        with gr.Group(visible=True) as style_classic_controls:
                            st_style_strength = gr.Slider(
                                0.1, 2.0, value=ST_DEFAULT_STYLE_STRENGTH, step=0.05, precision=2,
                                label="Style strength",
                                info="Higher = more of the style reference texture and palette.",
                                elem_classes=["blob-compact-slider"],
                            )
                            st_content_weight = gr.Slider(
                                0.1, 2.0, value=ST_DEFAULT_CONTENT_WEIGHT, step=0.05, precision=2,
                                label="Content fidelity",
                                info="Higher = keeps more of the content photo structure.",
                                elem_classes=["blob-compact-slider"],
                            )
                            st_steps = gr.Slider(
                                50, 500, value=ST_DEFAULT_STEPS, step=10, precision=0,
                                label="Steps",
                                info="Optimization iterations (more = slower, often cleaner).",
                                elem_classes=["blob-compact-slider"],
                            )
                        with gr.Group(visible=False) as style_preset_controls:
                            st_preset_custom_prompt = gr.Textbox(
                                label="Custom prompt",
                                visible=False,
                                placeholder="Describe the style to restyle into...",
                            )
                            st_preset_strength = gr.Slider(
                                0.1, 0.9, value=STYLE_PRESETS[DEFAULT_STYLE_PRESET]["strength"],
                                step=0.05, precision=2,
                                label="Strength",
                                info="Higher = more transformed, lower = closer to your photo.",
                                elem_classes=["blob-compact-slider"],
                            )
                        st_upscale = gr.Checkbox(
                            value=False,
                            label="Upscale output x2 (Real-ESRGAN)",
                        )

            with gr.Column(scale=8, elem_id="blob-preview"):
                with gr.Row(elem_id="blob-preview-row", equal_height=False):
                    with gr.Column(scale=3, elem_id="blob-output-col-wrap"):
                        with gr.Group(visible=True, elem_id="vqgan-preview-panel") as vqgan_preview_panel:
                            output = gr.Image(
                                label="Output", type="filepath",
                                height=PREVIEW_SIZE, scale=1,
                                sources=[], interactive=False,
                                buttons=["download", "fullscreen"], elem_id="blob-output-image",
                                placeholder=None, show_label=True,
                            )
                            video_output = gr.Video(
                                label="Output video",
                                visible=False,
                                interactive=False,
                                elem_id="blob-video-output",
                            )
                            use_init_btn = gr.Button(
                                "USE AS\nINIT", size="sm", visible=False,
                                elem_classes=["blob-use-init-btn"],
                            )
                        with gr.Group(visible=False, elem_id="deepdream-preview-panel") as deepdream_preview_panel:
                            dd_output = gr.Image(
                                label="Output",
                                type="filepath",
                                height=PREVIEW_SIZE,
                                sources=[],
                                interactive=False,
                                buttons=["download", "fullscreen"],
                                elem_id="blob-dd-output-image",
                            )
                            dd_use_init_btn = gr.Button(
                                "USE AS\nINIT", size="sm", visible=False,
                                elem_classes=["blob-use-init-btn"],
                            )
                        with gr.Group(visible=False, elem_id="style-preview-panel") as style_preview_panel:
                            st_output = gr.Image(
                                label="Output",
                                type="filepath",
                                height=PREVIEW_SIZE,
                                sources=[],
                                interactive=False,
                                buttons=["download", "fullscreen"],
                                elem_id="blob-st-output-image",
                            )
                        gallery_btn = gr.Button("Open gallery", size="sm", elem_id="blob-gallery-btn")
                        gr.Markdown(DISCLAIMER, elem_classes=["blob-disclaimer"])
                        gr.Markdown(CREDIT, elem_classes=["blob-credit"])

                    with gr.Column(scale=2, elem_id="blob-right-col"):
                        with gr.Group(visible=True, elem_id="vqgan-sketch-panel") as vqgan_sketch_panel:
                            sketch = gr.Image(
                                label="img2img",
                                type="filepath", height=260,
                                scale=1, visible=True,
                                sources=["upload"], interactive=True,
                                buttons=["download", "clear", "fullscreen"],
                                placeholder="Empty = SDXL sketch. Drop image = img2img.",
                                elem_id="blob-sketch-image",
                            )
                            init_image = gr.Image(
                                label="img2img",
                                type="filepath",
                                height=260,
                                visible=False,
                                scale=1,
                                sources=["upload"],
                                interactive=True,
                                buttons=["download", "clear", "fullscreen"],
                                placeholder="Drop image = img2img.",
                                elem_id="blob-source-image",
                            )
                            video_input = gr.Video(
                                label="Source video",
                                visible=False,
                                sources=["upload"],
                                elem_id="blob-video-input",
                            )
                        with gr.Group(visible=False, elem_id="deepdream-sketch-panel") as deepdream_sketch_panel:
                            dd_sketch = gr.Image(
                                label="img2img",
                                type="filepath",
                                height=260,
                                visible=True,
                                sources=["upload"],
                                interactive=True,
                                buttons=["download", "clear", "fullscreen"],
                                placeholder="Empty = SDXL sketch. Drop image = img2img.",
                                elem_id="blob-dd-sketch-image",
                            )
                        with gr.Group(visible=False, elem_id="style-sketch-panel") as style_sketch_panel:
                            st_style_image = gr.Image(
                                label="Style reference",
                                type="pil",
                                height=260,
                                sources=["upload"],
                                interactive=True,
                                buttons=["download", "clear", "fullscreen"],
                                elem_id="blob-st-style-image",
                            )
                            st_content_image = gr.Image(
                                label="img2img",
                                type="pil",
                                height=260,
                                sources=["upload"],
                                interactive=True,
                                buttons=["download", "clear", "fullscreen"],
                                elem_id="blob-st-content-image",
                            )
                            st_preset = gr.Dropdown(
                                choices=[(NO_PRESET_LABEL, NO_PRESET_VALUE)]
                                + [(v["label"], k) for k, v in STYLE_PRESETS.items()],
                                value=NO_PRESET_VALUE,
                                label="SDXL preset",
                                info="Pick a preset to restyle with SDXL Turbo. Leave on "
                                     "\"" + NO_PRESET_LABEL + "\" for classic VGG19 style transfer.",
                                elem_id="blob-st-preset-dropdown",
                            )

                        with gr.Row(elem_classes=["blob-preview-footer"]):
                            with gr.Column(scale=1, elem_classes=["blob-output-footer"]):
                                with gr.Row(elem_classes=["blob-seed-compact-row"]):
                                    seed_out = gr.Number(
                                        label="Seed", interactive=False,
                                        elem_classes=["blob-seed-out"], scale=2,
                                    )
                                    random_seed = gr.Checkbox(value=True, label="Random seed", scale=1)
                                    reuse_last_seed = gr.Checkbox(value=False, label="Reuse seed", scale=1)
                                    abort_btn = gr.Button(
                                        "X", scale=0, min_width=42,
                                        visible=False, interactive=False,
                                        elem_classes=["blob-abort-btn"],
                                    )
                                meta_out = gr.Textbox(
                                    label="Metadata", lines=1, max_lines=1,
                                    elem_classes=["blob-meta"],
                                )
                                seed_value = gr.Number(value=42, label="Fixed seed", precision=0, visible=False)
                            with gr.Column(scale=1, elem_id="blob-console-slot"):
                                console_out = gr.Textbox(
                                    label="Console",
                                    lines=4,
                                    max_lines=8,
                                    interactive=False,
                                    elem_id="blob-console-out",
                                    elem_classes=["blob-console"],
                                )
                        gr.Markdown(
                            '<div style="font-size:0.5rem;opacity:0.4;text-align:center;padding:2px;">'
                            'made with gradio'
                            '</div>',
                            elem_classes=["blob-credit-mini"],
                        )

        with gr.Group(visible=False, elem_id="blob-long-video-overlay") as long_video_overlay:
            with gr.Column(elem_classes=["blob-long-video-dialog"]):
                gr.Markdown("### Long video warning")
                long_video_warn = gr.Markdown("")
                with gr.Row(elem_classes=["blob-long-video-actions"]):
                    long_video_yes = gr.Button("Yes, continue", variant="primary")
                    long_video_no = gr.Button("No, cancel")

        def toggle_seed_ui(is_random):
            return gr.update(visible=not is_random)

        mode.change(
            on_mode_change, mode,
            [
                iterations, denoise, init_image, sketch, use_init_btn,
                output, video_input, video_output, install_codecs_btn,
                video_encode_range, video_encode_row, video_encode_from, video_encode_to,
                video_frame_step, long_video_overlay, mode_help,
            ],
        )
        install_codecs_btn.click(
            activate_video_mode, None,
            [
                output, video_input, video_output, install_codecs_btn,
                video_encode_range, video_encode_row, video_encode_from, video_encode_to,
                video_frame_step,
            ],
        )
        video_input.change(
            on_video_upload, [video_input, mode],
            [long_video_overlay, long_video_warn, aspect, video_encode_to],
        )
        init_image.change(on_source_image_upload, [init_image, mode], aspect)
        sketch.change(on_sketch_upload, [sketch], aspect)
        video_encode_range.change(
            on_encode_range_toggle, video_encode_range,
            [video_encode_from, video_encode_to],
        )
        long_video_yes.click(on_long_video_yes, None, long_video_overlay)
        long_video_no.click(on_long_video_no, None, [long_video_overlay, video_input])
        random_seed.change(toggle_seed_ui, random_seed, seed_value)
        vqgan_gen_outputs = [
            load_banner, output, sketch, seed_out, meta_out, video_output, abort_btn,
        ]
        vqgan_gen = generate_btn.click(
            _gen_start, None, abort_btn, js=SIDEBAR_BUSY_ON_JS,
        ).then(
            do_generate,
            [
                mode, prompt, negative_prompt, init_image, sketch, video_input, aspect,
                iterations, denoise, random_seed, seed_value, reuse_last_seed,
                video_encode_range, video_encode_from, video_encode_to, video_frame_step,
                upscale_x2,
            ],
            vqgan_gen_outputs,
        )
        vqgan_gen.then(None, None, None, js=SIDEBAR_BUSY_OFF_JS)
        abort_btn.click(abort_generation, None, abort_btn)
        use_init_btn.click(use_output_as_init, [output, mode], init_image)
        dd_use_init_btn.click(use_dd_output_as_init, [dd_output], dd_sketch)
        dd_sketch.change(on_dd_source_upload, [dd_sketch], dd_aspect)
        dd_gen_outputs = [
            load_banner, dd_output, dd_sketch, seed_out, meta_out, abort_btn,
        ]
        dd_gen = dd_generate_btn.click(
            _gen_start, None, abort_btn, js=SIDEBAR_BUSY_ON_JS,
        ).then(
            do_deepdream_generate,
            [
                dd_prompt, dd_negative_prompt, dd_aspect,
                dd_intensity, dd_layer, dd_sketch,
                random_seed, seed_value, reuse_last_seed, dd_upscale,
            ],
            dd_gen_outputs,
        )
        dd_gen.then(None, None, None, js=SIDEBAR_BUSY_OFF_JS)
        st_content_image.change(on_st_content_upload, st_content_image, st_aspect)
        st_preset.change(
            on_st_preset_change, st_preset,
            [
                style_classic_controls, style_preset_controls, st_style_image, st_mode_help,
                st_preset_custom_prompt, st_preset_strength,
            ],
        )
        st_gen_outputs = [load_banner, st_output, seed_out, meta_out, abort_btn]
        st_gen = st_generate_btn.click(
            _gen_start, None, abort_btn, js=SIDEBAR_BUSY_ON_JS,
        ).then(
            do_style_generate,
            [
                st_aspect, st_style_image, st_style_strength, st_content_weight, st_steps,
                st_preset, st_preset_custom_prompt, st_preset_strength,
                st_content_image, random_seed, seed_value, reuse_last_seed, st_upscale,
            ],
            st_gen_outputs,
        )
        st_gen.then(None, None, None, js=SIDEBAR_BUSY_OFF_JS)
        family_switch_outputs = [
            vqgan_sidebar_panel, deepdream_sidebar_panel, style_sidebar_panel,
            vqgan_preview_panel, deepdream_preview_panel, style_preview_panel,
            vqgan_sketch_panel, deepdream_sketch_panel, style_sketch_panel,
            tagline_html, install_codecs_btn, install_style_btn, st_install_models_btn,
        ] + list(family_btns.values())
        vqgan_mode_outputs = [
            iterations, denoise, init_image, sketch, use_init_btn,
            output, video_input, video_output, install_codecs_btn,
            video_encode_range, video_encode_row, video_encode_from, video_encode_to,
            video_frame_step, long_video_overlay, mode_help,
        ]
        for family_id, family_btn in family_btns.items():
            if not FAMILY_META[family_id]["enabled"]:
                continue
            family_click = family_btn.click(
                lambda fid=family_id: switch_family(fid),
                None,
                family_switch_outputs,
                js=FAMILY_THEME_JS.get(family_id, FAMILY_THEME_JS["vqgan"]),
            )
            family_click.then(
                on_mode_change, mode, vqgan_mode_outputs,
            ).then(
                refresh_sketch_components, mode,
                [sketch, init_image, dd_sketch, st_content_image, st_style_image],
            )
        gallery_btn.click(open_gallery, None, None)
        install_style_btn.click(
            activate_style_models, None, [install_style_btn, st_install_models_btn],
        )
        st_install_models_btn.click(
            activate_style_models, None, [install_style_btn, st_install_models_btn],
        )
        start_btn.click(start_system, None, [load_status, start_btn, load_banner])
        disconnect_btn.click(disconnect_server, None, [load_status, start_btn, load_banner])

        timer = gr.Timer(1.0, active=True)
        timer.tick(poll_status, None, load_status)
        timer.tick(poll_banner, None, load_banner)
        timer.tick(poll_console, None, console_out, js=CONSOLE_SCROLL_JS)
        timer.tick(poll_start_btn, None, start_btn)
        timer.tick(poll_style_install_btn, None, [install_style_btn, st_install_models_btn])
        timer.tick(poll_codecs_install_btn, None, install_codecs_btn)
        timer.tick(
            lambda m: _video_mode_ui_updates(m),
            mode,
            [
                output, video_input, video_output, install_codecs_btn,
                video_encode_range, video_encode_row, video_encode_from, video_encode_to,
                video_frame_step,
            ],
        )

        demo.load(poll_style_install_btn, None, [install_style_btn, st_install_models_btn])
        demo.load(poll_codecs_install_btn, None, install_codecs_btn)
        demo.load(None, None, None, js=CONSOLE_SCROLL_JS)
        demo.load(None, None, None, js=BLOB_JS_HOOK)

    return demo


def _launch_demo(demo, host, port, share, open_browser=False):
    from blobvision_paths import gradio_allowed_paths

    launch_kwargs = dict(
        server_name=host,
        share=share,
        show_error=True,
        ssr_mode=False,
        theme=gr.themes.Soft(),
        css=CSS,
        allowed_paths=gradio_allowed_paths(),
    )

    def _try_launch(server_port):
        launch_kwargs["server_port"] = server_port
        launch_kwargs["inbrowser"] = open_browser
        url = "http://{}:{}".format(host, server_port)
        if open_browser:
            print("Navigateur a l'ouverture du serveur: {}".format(url), flush=True)
        print("Serveur Gradio sur {}".format(url), flush=True)
        demo.launch(**launch_kwargs)

    last_exc = None
    for candidate in range(port, port + 10):
        if candidate != port:
            print(
                "Port {} deja utilise (ancienne instance BlobVision ?) — essai sur {}...".format(
                    port, candidate,
                ),
                flush=True,
            )
            print("Open: http://{}:{}".format(host, candidate), flush=True)
        try:
            _try_launch(candidate)
            return
        except OSError as exc:
            msg = str(exc)
            if "Cannot find empty port" in msg or "10048" in msg:
                last_exc = exc
                continue
            raise
        except ValueError as exc:
            if "localhost is not accessible" not in str(exc):
                raise
            import gradio.networking as networking

            print(
                "Warning: Gradio localhost check failed (often proxy or heavy startup). "
                "Retrying without the check...",
                flush=True,
            )
            _real_url_ok = networking.url_ok
            networking.url_ok = lambda _url: True
            try:
                _try_launch(candidate)
                return
            finally:
                networking.url_ok = _real_url_ok

    print(
        "Aucun port libre entre {} et {}.".format(port, port + 9),
        flush=True,
    )
    print(
        "Ferme l'autre fenetre BlobVision ou un processus qui utilise le port {}, puis relance.".format(
            port,
        ),
        flush=True,
    )
    raise last_exc


def _cleanup_on_exit():
    try:
        _dispose_engine()
    except Exception:
        pass
    # This standalone Gradio entry point exits via a normal Python
    # interpreter shutdown (unlike the Tauri shell, which hard-kills the
    # process — see main.rs's own uploads sweep for that path), so a real
    # atexit hook reliably fires here.
    try:
        import shutil

        from blobvision_paths import uploads_dir

        for name in os.listdir(uploads_dir()):
            path = os.path.join(uploads_dir(), name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
    except Exception:
        pass


def main():
    atexit.register(_cleanup_on_exit)
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--lazy", action="store_true", help="Open UI without loading models until Start is pressed")
    parser.add_argument("--open-browser", action="store_true", help="Open the UI in the default browser once the server is ready")
    args = parser.parse_args()

    print("BlobVision UI — chargement SDXL + VQGAN au demarrage...", flush=True)

    demo = build_ui()
    print("Interface prete — demarrage du serveur...", flush=True)
    demo.queue(default_concurrency_limit=1)
    if args.lazy:
        _load["stopped"] = True
        _load["stage"] = "Disconnected"
    else:
        threading.Thread(target=_warmup_worker, daemon=True).start()
    _launch_demo(demo, args.host, args.port, args.share, args.open_browser)


if __name__ == "__main__":
    main()
