#!/usr/bin/env python3
"""BlobVision HTTP API   Legacy / Redux / Corrupt generation."""
import argparse
import ctypes
import ctypes.wintypes as wintypes
import mimetypes
import os
import shutil
import sys
import threading
import time
import uuid
from typing import List, Optional


class _ElapsedStdout:
    """Prefixes every console line with seconds-since-process-start so gaps
    between warmup/generation steps are visible directly in the log instead
    of cross-referencing isolated duration messages by eye. Installed as
    early as possible (before any local module import) so it covers every
    print() in the process, including ones firing during model-loading
    imports.

    Carriage-return-driven progress bars (tqdm, run_with_patience's
    heartbeat) are left alone past their first character -- only a real
    newline starts a new "logical line" that gets its own prefix, otherwise
    every redraw of the same progress bar would get re-prefixed."""

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._start = time.time()
        self._at_line_start = True

    def write(self, s):
        if not s:
            return 0
        out = []
        for ch in s:
            if self._at_line_start and ch not in ("\n", "\r"):
                out.append("[+{:>7.1f}s] ".format(time.time() - self._start))
                self._at_line_start = False
            out.append(ch)
            if ch == "\n":
                self._at_line_start = True
        return self._wrapped.write("".join(out))

    def flush(self):
        self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


sys.stdout = _ElapsedStdout(sys.stdout)

# Windows process priority classes (winbase.h) — duplicated here rather than
# depending on pywin32/psutil just for two constants.
_ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
_NORMAL_PRIORITY_CLASS = 0x00000020


def _set_process_priority(priority_class):
    """Best-effort, silent no-op off Windows or on any failure — this must
    never be able to crash the app over a scheduling nicety. ABOVE_NORMAL
    (not HIGH/REALTIME, which can starve the rest of the system and cause
    UI stutter) is the officially-sanctioned "this matters a bit more right
    now" tier, raisable by a normal (non-admin) process. Used to give
    BlobVision fairer CPU scheduling during the loading burst against
    memory-heavy background apps (e.g. a browser with many tabs causing
    system-wide paging) — see startup below and POST /priority/normal,
    which the frontend calls once its staged warmup sequence finishes to
    drop back to normal instead of staying elevated during idle/generation
    time. This targets CPU scheduling specifically; it won't necessarily
    speed up raw disk-I/O contention if that's the actual bottleneck."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import ctypes.wintypes as wintypes
        kernel32 = ctypes.windll.kernel32
        # GetCurrentProcess() returns a pseudo-handle that ctypes' default
        # (32-bit int) return type silently truncates on 64-bit Windows,
        # producing a bogus handle SetPriorityClass then rejects with
        # ERROR_INVALID_HANDLE — explicit HANDLE/DWORD/BOOL types are
        # required for this to actually do anything instead of failing
        # silently (confirmed: without these, SetPriorityClass returned 0 /
        # GetLastError() == 6 every time).
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), priority_class)
    except Exception:
        pass


# CPU scheduling priority (above) and memory priority (this) are separate
# Windows mechanisms — measured proof they matter for different things:
# after the CPU boost alone, a real run showed VQGAN's first-pass warmup
# get ~38% faster (24.6s -> 15.2s, CPU/dispatch-bound work) while SDXL's
# CPU-offload warmup stayed just as slow (~70-90s for a single step).
# SDXL's CPU-offload mode moves weights between system RAM and VRAM on
# every layer of every step — that's memory-bus/paging-bound, not CPU-
# scheduling-bound, so a CPU priority boost can't touch it. A headless
# backend process with no window of its own (this one — the Tauri/Rust
# process owns the actual window) is exactly the kind of process Windows'
# memory manager tends to quietly deprioritize for page trimming under
# system-wide memory pressure (e.g. a browser with many tabs). Asserting
# MEMORY_PRIORITY_NORMAL — the ceiling a normal process can request, not
# an unusual boost — keeps this process's pages from being an easy first
# target when something else is starving the system for RAM.
_PROCESS_MEMORY_PRIORITY = 0  # PROCESS_INFORMATION_CLASS.ProcessMemoryPriority
_MEMORY_PRIORITY_NORMAL = 5


class _MemoryPriorityInformation(ctypes.Structure):
    _fields_ = [("MemoryPriority", ctypes.c_ulong)]


def _set_process_memory_priority(priority):
    """Best-effort, silent no-op off Windows or on any failure (this API is
    also only available on Windows 8+; SetProcessInformation itself simply
    won't exist as an attribute on kernel32 on older systems, caught by the
    same except below). Set once at startup and left alone — unlike the CPU
    priority class, NORMAL memory priority isn't an aggressive stance that
    needs relaxing afterward, it's just "don't quietly throttle this
    process below where a normal foreground app would sit"."""
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        info = _MemoryPriorityInformation(priority)
        kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(),
            _PROCESS_MEMORY_PRIORITY,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
    except Exception:
        pass


_set_process_priority(_ABOVE_NORMAL_PRIORITY_CLASS)
_set_process_memory_priority(_MEMORY_PRIORITY_NORMAL)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from blobvision_paths import BLOBVISION_ROOT, VENV_PYTHON
BLOBDREAM_ROOT = BLOBVISION_ROOT

os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

import blobvision_cancel
from blobvision_engine import (
    ASPECT_FORMATS,
    DEFAULT_ASPECT,
    DEFAULT_DENOISE,
    DEFAULT_ITERATIONS,
    DEFAULT_STYLE_PRESET,
    STYLE_PRESET_STEPS,
    STYLE_PRESETS,
    BlobVisionEngine,
    BlobVRAMCache,
    _preload_hf_imports,
    download_sdxl_weights,
    download_vqgan_family,
    resolve_aspect_size,
    resolve_seed,
    sdxl_weights_status,
    vqgan_family_status,
)
from blobvision_deepdream import (
    DEFAULT_LAYER as DD_DEFAULT_LAYER,
    DeepDreamEngine,
    intensity_to_params,
)
from blobvision_meta import read_metadata_from_image, seed_from_metadata
from blobvision_paths import build_output_name, outputs_dir, sanitize_basename, sketch_dir, uploads_dir
from blobvision_sam import Sam2Engine, download_sam2_weights, sam2_weights_status
# Family-agnostic video pipeline (see blobvision_video.py's module docstring)
# — every family's video endpoint below shares these same codec/probe/setup
# helpers regardless of which engine actually processes the frames.
from blobvision_video import (
    VIDEO_LONG_WARN_SECONDS,
    format_video_duration,
    probe_video_file,
    setup_bundled_video_codecs,
    video_codecs_status,
)
from blobvision_upscale import UpscaleEngine
from blobvision_style import (
    DEFAULT_CONTENT_WEIGHT as ST_DEFAULT_CONTENT_WEIGHT,
    DEFAULT_STEPS as ST_DEFAULT_STEPS,
    DEFAULT_STYLE_STRENGTH as ST_DEFAULT_STYLE_STRENGTH,
    StyleTransferEngine,
    download_style_transfer_weights,
    style_transfer_weights_status,
)

app = FastAPI(
    title="BlobVision API",
    description="VQGAN+CLIP renderer   Legacy, Redux, Corrupt",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: Optional[BlobVisionEngine] = None
_dd_engine: Optional[DeepDreamEngine] = None
_st_engine: Optional[StyleTransferEngine] = None
_sam_engine: Optional[Sam2Engine] = None
_upscale_engines: dict = {}


class ModeInfo(BaseModel):
    name: str
    description: str
    default_iterations: int
    default_denoise_fidelity: Optional[float] = None
    needs_init_image: bool


class GenerateResponse(BaseModel):
    ok: bool = True
    mode: str
    prompt: str
    seed: int
    iterations: int
    denoise_fidelity: Optional[float] = None
    output_url: str
    sketch_url: Optional[str] = None
    metadata: dict


class DeepDreamResponse(BaseModel):
    ok: bool = True
    mode: str
    seed: int
    layer: str
    output_url: str
    sketch_url: Optional[str] = None
    metadata: dict


class StylePresetInfo(BaseModel):
    key: str
    label: str
    strength: float


class StyleResponse(BaseModel):
    ok: bool = True
    mode: str
    preset: Optional[str] = None
    seed: int
    output_url: str
    metadata: dict


def get_engine() -> BlobVisionEngine:
    global _engine
    if _engine is None:
        raise HTTPException(503, "Engine not started. Call POST /warmup first.")
    return _engine


def get_dd_engine() -> DeepDreamEngine:
    global _dd_engine
    if _dd_engine is None:
        _dd_engine = DeepDreamEngine()
    return _dd_engine


def get_st_engine() -> StyleTransferEngine:
    global _st_engine
    if _st_engine is None:
        _st_engine = StyleTransferEngine()
    return _st_engine


def get_sam_engine() -> Sam2Engine:
    global _sam_engine
    if _sam_engine is None:
        _sam_engine = Sam2Engine()
    return _sam_engine


def get_upscale_engine(scale: int = 2) -> UpscaleEngine:
    engine = _upscale_engines.get(scale)
    if engine is None:
        engine = UpscaleEngine(scale=scale)
        _upscale_engines[scale] = engine
    return engine


def _maybe_upscale(image_path, scale):
    """Mirrors blobvision_ui.py's _maybe_upscale: a failed upscale shouldn't
    lose an otherwise-good generation, so it falls back to the un-upscaled
    result instead of raising. scale: 0/None = no upscale, else 2 or 4."""
    if not scale or not image_path or not os.path.isfile(image_path):
        return image_path
    try:
        return get_upscale_engine(scale).upscale(image_path)
    except Exception:
        return image_path


def _composite_masked_output(output_path, original_path, mask_upload, invert):
    """SAM2 Phase 1 (image, VQGAN): blends the full generation back down to
    only the selected region — `final = mask*blobified + (1-mask)*original`
    (or the inverse if `invert`). The mask is whatever the client already
    refined (erosion/feather — see tauri/src/main.ts, done client-side so
    slider drags don't round-trip here); this only ever composites the one
    final choice. Resized to `output_path`'s actual pixel size, which can
    differ from the mask's own (captured pre-generation, pre-upscale) size.
    Overwrites output_path in place and returns it."""
    from PIL import Image

    blobified = Image.open(output_path).convert("RGB")
    original = Image.open(original_path).convert("RGB").resize(blobified.size, Image.LANCZOS)
    mask_img = Image.open(mask_upload.file).convert("L").resize(blobified.size, Image.LANCZOS)
    if invert:
        mask_img = Image.eval(mask_img, lambda v: 255 - v)
    Image.composite(blobified, original, mask_img).save(output_path)
    return output_path


def _apply_cut_mask(output_path, cut_mask_upload, cut_invert, cut_mode, cut_color):
    """The overlay's Cut/Keep buttons bake a SEPARATE snapshot mask (see
    tauri/src/main.ts's commitCutMask) that always wins over the
    blobify/original choice above — it's a deliberate "punch a hole here"
    action, not a generation choice, so it's applied last, after upscale,
    directly on the already-finished output. `cut_mode="alpha"` makes the
    zone transparent (PNG alpha channel); `cut_mode="color"` fills it with
    `cut_color` instead (e.g. a green-screen backdrop for later keying).
    `cut_invert` mirrors the same semantics as the blobify mask's own
    `invert`: Cut sends False (the selection itself is punched out), Keep
    sends True (everything EXCEPT the selection is punched out). Overwrites
    output_path in place and returns it."""
    from PIL import Image

    base = Image.open(output_path).convert("RGB")
    cut_img = Image.open(cut_mask_upload.file).convert("L").resize(base.size, Image.LANCZOS)
    if cut_invert:
        cut_img = Image.eval(cut_img, lambda v: 255 - v)
    if cut_mode == "color":
        fill = Image.new("RGB", base.size, cut_color)
        Image.composite(fill, base, cut_img).save(output_path)
    else:
        base_rgba = base.convert("RGBA")
        alpha = base_rgba.split()[-1]
        new_alpha = Image.composite(Image.new("L", base.size, 0), alpha, cut_img)
        base_rgba.putalpha(new_alpha)
        base_rgba.save(output_path)
    return output_path


MODE_HELP = {
    "legacy": "Classic VQGAN+CLIP text2img (500 it default). Optional init image for img2img.",
    "redux": "SDXL Turbo sketch then VQGAN — fast modern path.",
}


class VideoJob:
    """In-memory state for one background video-generation run, keyed by a
    job_id the client polls. Video jobs take minutes, far too long to hold
    an HTTP request open for (every other endpoint in this file blocks
    until done, which is fine for a single image but not for this) — so
    /video/generate starts one of these in a background thread and returns
    immediately, and /video/status/{job_id} reports back whatever's
    accumulated in `lines` so far via the engine's on_progress callback."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.state = "running"  # running | done | error | cancelled
        self.lines: List[str] = []
        self.output_url: Optional[str] = None
        self.metadata: Optional[dict] = None
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def append(self, msg: str):
        with self._lock:
            self.lines.append(msg)
            if len(self.lines) > 300:
                self.lines = self.lines[-300:]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "job_id": self.job_id,
                "state": self.state,
                "lines": list(self.lines),
                "output_url": self.output_url,
                "error": self.error,
                "metadata": self.metadata,
            }


_video_jobs: dict = {}
_video_jobs_lock = threading.Lock()
_video_job_running = False  # only one video job at a time (matches the app's single-engine, one-generation-at-a-time design)


@app.get("/")
def root():
    return {
        "name": "BlobVision API",
        "modes": list(MODE_HELP.keys()),
        "docs": "/docs",
    }




@app.get("/docs.")
def docs_trailing_dot():
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico")
def favicon():
    return JSONResponse(status_code=204, content=None)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/priority/normal")
def priority_normal():
    """Called once by the frontend after its staged warmup sequence fully
    completes (SDXL + whichever family's model — see startStagedWarmup() in
    tauri/src/main.ts), dropping the ABOVE_NORMAL priority boost set at
    process startup (see the top of this file) back to normal. The boost
    only makes sense for the one-time loading burst; staying elevated
    during ordinary idle/generation time would just make BlobVision a worse
    neighbor to whatever else the user has running."""
    _set_process_priority(_NORMAL_PRIORITY_CLASS)
    return {"ok": True}


@app.get("/modes")
def modes():
    items = []
    for name in ("legacy", "redux"):
        items.append(ModeInfo(
            name=name,
            description=MODE_HELP[name],
            default_iterations=DEFAULT_ITERATIONS[name],
            default_denoise_fidelity=DEFAULT_DENOISE.get(name),
            needs_init_image=(name == "legacy"),
        ))
    return items


@app.get("/status")
def status():
    if _engine is None:
        return {"started": False}
    s = _engine.status()
    s["started"] = True
    return s


# The big three model weight sets (SDXL Turbo, VQGAN+CLIP, Style Transfer's
# VGG19) aren't bundled with the app or fetched automatically — main.rs
# forces HF_HUB_OFFLINE for normal operation (see
# blobvision_engine.enable_hub_downloads), and nothing else in the request
# path hits the network. This is the fresh-machine install surface: check
# what's present, and fetch whichever of the three the user asks for.
_MODEL_DOWNLOADERS = {
    "sdxl": download_sdxl_weights,
    "vqgan": download_vqgan_family,
    "style": download_style_transfer_weights,
}


@app.get("/models/status")
def models_status():
    return {
        "sdxl": sdxl_weights_status(),
        "vqgan": vqgan_family_status(),
        "style": style_transfer_weights_status(),
    }


@app.post("/models/install")
def models_install(targets: List[str] = Body(..., embed=True)):
    """Kicks off one background thread per requested target that isn't
    already installed — same fire-and-forget shape as /video/codecs/install
    and /sam2/install; poll /models/status to see progress. Unknown target
    names are ignored rather than erroring, so the frontend can always pass
    its full checkbox selection without pre-filtering."""
    started = []
    for name in targets:
        downloader = _MODEL_DOWNLOADERS.get(name)
        if downloader is None:
            continue
        threading.Thread(target=downloader, daemon=True).start()
        started.append(name)
    return {"ok": True, "installing": started}


@app.post("/warmup")
def warmup(mode: str = "redux"):
    global _engine
    mode = mode.lower().strip()
    if mode not in MODE_HELP:
        raise HTTPException(400, "Unknown mode: {}".format(mode))
    if _engine is None:
        _engine = BlobVisionEngine(keep_models=True)
    _engine.warmup(mode=mode)
    return {"ok": True, "warmed": mode, "status": _engine.status()}


def _spawn_background_warmup(engine, name, warmup_fn):
    """Runs a dummy CUDA-kernel-priming pass (SDXL's run_sketch_warmup or
    VQGAN's run_vqgan_warmup) in a background thread instead of making the
    /warmup/* response wait for it. The point of these passes is to move a
    real cost off the user's *first generation* — but blocking the Generate
    button on them too just relocates the same wait one step earlier
    instead of actually hiding it. Both warmup methods take engine._lock for
    their whole duration; a real /generate call (also via engine._lock)
    that arrives before a background pass finishes just blocks on that same
    lock like any other contending caller — the cost is still paid exactly
    once, it's just no longer guaranteed to land before the user is allowed
    to type a prompt and click Generate. In the common case (a human takes
    at least a few seconds to type something) the background pass has
    already finished by the time Generate is actually clicked. Idempotent —
    a pass already running or already finished is never started twice."""
    started_attr = "_" + name + "_warmup_started"
    done_attr = "_" + name + "_warmed_once"
    if getattr(engine, done_attr, False) or getattr(engine, started_attr, False):
        return
    setattr(engine, started_attr, True)

    def run():
        try:
            warmup_fn()
        finally:
            setattr(engine, done_attr, True)

    threading.Thread(target=run, daemon=True, name=name + "-warmup").start()


@app.post("/warmup/sdxl")
def warmup_sdxl():
    """SDXL only, without VQGAN — the shared first stage of the frontend's
    family-aware staged preload (see tauri/src/main.ts): SDXL loads for
    everyone first since every family's sketch pass needs it, then whichever
    family the user is actually looking at gets its own model loaded next.
    Returns as soon as the real weights are loaded and placed — see
    _spawn_background_warmup for why the CUDA-kernel-priming dummy pass
    (run_sketch_warmup) doesn't block this response."""
    # Uvicorn's own access log only fires on RESPONSE completion, not on
    # request arrival — for a handler that can legitimately run for
    # minutes (SDXL disk read + GPU transfer), that hides exactly when the
    # request actually reached the server. This one line is the cheapest
    # way to tell a slow frontend/network hop (request sent late) apart
    # from a slow handler (request received promptly, something after
    # this line is what's slow) — see the "[+Xs]" prefix from
    # _ElapsedStdout above.
    print("POST /warmup/sdxl received", flush=True)
    global _engine
    if _engine is None:
        _engine = BlobVisionEngine(keep_models=True)
    with _engine._lock:
        if _engine._sketch_pipe is None:
            _engine._load_sketch_pipe()
            _engine._vram.note_sketch_loaded_on_gpu()
        _engine._vram.activate(BlobVRAMCache.PHASE_SKETCH)
    _spawn_background_warmup(_engine, "sketch", _engine.run_sketch_warmup)
    return {"ok": True, "status": _engine.status()}


@app.post("/warmup/vqgan")
def warmup_vqgan():
    """VQGAN weights only, WITHOUT switching the active VRAM phase away from
    SDXL. The plain `/warmup?mode=redux` (engine.warmup()) ends by calling
    activate(PHASE_VQGAN), which parks SDXL back to CPU — fine on its own,
    but wrong as a *preload* target for redux mode, since redux's actual
    first generate() step is always the SDXL sketch, not VQGAN. Calling
    engine.warmup() and then loading VQGAN like this in sequence would just
    silently undo the SDXL warmup and bring back the "first click loads
    models" delay the staged preload exists to avoid. Mirrors the exact
    pattern blobvision_engine.py's warmup_staged() already uses for its
    background VQGAN load: load the weights, then park them straight back
    off GPU instead of activating that phase, leaving SDXL resident.
    Returns as soon as the real weights are loaded — see
    _spawn_background_warmup for why run_vqgan_warmup doesn't block this."""
    global _engine
    if _engine is None:
        _engine = BlobVisionEngine(keep_models=True)
    with _engine._lock:
        from blobvision_engine import DEFAULT_ITERATIONS

        _engine._ensure_vqgan("redux", DEFAULT_ITERATIONS["redux"])
        _engine._vram.note_vqgan_loaded_on_gpu()
        _engine._vram._park_vqgan(None)
    _spawn_background_warmup(_engine, "vqgan", _engine.run_vqgan_warmup)
    return {"ok": True, "status": _engine.status()}


@app.post("/unload/vqgan")
def unload_vqgan():
    """Frees VQGAN+CLIP while leaving SDXL resident. Used when the frontend
    switches which family's model it keeps loaded alongside SDXL, so VQGAN,
    DeepDream, and Style Transfer's models are never all resident at once."""
    if _engine is None:
        return {"ok": True}
    with _engine._lock:
        _engine._unload_vqgan_clip()
    return {"ok": True}


@app.post("/deepdream/unload")
def deepdream_unload():
    global _dd_engine
    if _dd_engine is None:
        return {"ok": True}
    _dd_engine._model = None
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _dd_engine = None
    return {"ok": True}


@app.post("/style/unload")
def style_unload():
    global _st_engine
    if _st_engine is None:
        return {"ok": True}
    _st_engine._vgg = None
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _st_engine = None
    return {"ok": True}


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    mode: str = Form(...),
    prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    aspect: Optional[str] = Form(None),
    iterations: Optional[int] = Form(None),
    denoise_fidelity: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    sketch_steps: int = Form(4),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    upscale: int = Form(0),
    init_image: Optional[UploadFile] = File(None),
    reuse_seed_from_image: Optional[UploadFile] = File(None),
    mask: Optional[UploadFile] = File(None),
    invert: bool = Form(False),
    cut_mask: Optional[UploadFile] = File(None),
    cut_invert: bool = Form(False),
    cut_mode: str = Form("alpha"),
    cut_color: str = Form("#00ff00"),
):
    engine = get_engine()
    mode = mode.lower().strip()
    if mode not in MODE_HELP:
        raise HTTPException(400, "Unknown mode: {}".format(mode))
    if mask is not None and init_image is None:
        raise HTTPException(400, "A mask requires init_image (composites generation against it).")

    if reuse_seed_from_image is not None:
        tmp = os.path.join(uploads_dir(), uuid.uuid4().hex + ".png")
        with open(tmp, "wb") as fh:
            shutil.copyfileobj(reuse_seed_from_image.file, fh)
        try:
            inherited = seed_from_metadata(read_metadata_from_image(tmp))
            if inherited is not None:
                seed = inherited
        finally:
            if os.path.isfile(tmp):
                os.remove(tmp)

    init_path = None
    if init_image is not None:
        ext = os.path.splitext(init_image.filename or "upload.png")[1] or ".png"
        # <uuid>__<original-basename> lets the engine recover a readable
        # BASENAME for the final output name (see
        # blobvision_paths.basename_hint_from_upload) without a separate
        # upload<->name lookup.
        init_path = os.path.join(
            uploads_dir(),
            uuid.uuid4().hex + "__" + sanitize_basename(init_image.filename) + ext,
        )
        with open(init_path, "wb") as fh:
            shutil.copyfileobj(init_image.file, fh)

    result = None
    try:
        try:
            # run_in_threadpool, not a direct call: engine.generate() blocks
            # for the whole optimization loop, and this handler is `async
            # def` — a direct call would freeze the single asyncio event
            # loop for that entire time, starving every other request
            # (including POST /cancel) until generation finished, which
            # would make cancellation of a plain image generation
            # impossible in practice.
            result = await run_in_threadpool(
                engine.generate,
                mode=mode,
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect=aspect or DEFAULT_ASPECT,
                iterations=iterations,
                denoise_fidelity=denoise_fidelity,
                seed=seed,
                init_image_path=init_path,
                sketch_steps=sketch_steps,
                width=width,
                height=height,
            )
        except blobvision_cancel.AbortedError as exc:
            raise HTTPException(499, str(exc) or "Generation cancelled.") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

        output_path = _maybe_upscale(result.output_path, upscale)
        if mask is not None:
            output_path = _composite_masked_output(output_path, init_path, mask, invert)
        if cut_mask is not None:
            output_path = _apply_cut_mask(output_path, cut_mask, cut_invert, cut_mode, cut_color)
        base = "/outputs/"
        out_name = os.path.basename(output_path)
        sketch_url = None
        if result.sketch_path:
            sketch_url = "/outputs/" + os.path.basename(result.sketch_path)

        return GenerateResponse(
            mode=result.mode,
            prompt=result.prompt,
            seed=result.seed,
            iterations=result.iterations,
            denoise_fidelity=result.denoise_fidelity,
            output_url=base + out_name,
            sketch_url=sketch_url,
            metadata=result.metadata,
        )
    finally:
        # The uploaded init image is single-use (the frontend re-sends it
        # fresh on every request) — except VQGAN's redux mode, which reuses
        # a user-supplied init image directly as the sketch and surfaces it
        # to the client as sketch_url; that one is left for the app-close
        # sweep instead of being deleted out from under the response.
        if init_path and (result is None or result.sketch_path != init_path):
            try:
                os.remove(init_path)
            except OSError:
                pass


@app.get("/video/codecs/status")
def video_codecs_status_endpoint():
    return video_codecs_status()


@app.post("/video/codecs/install")
def video_codecs_install():
    """Kicks off the ~150MB ffmpeg+RIFE download in a background thread and
    returns immediately — same daemon-thread pattern blobvision_ui.py's
    activate_video_mode() used, so the request doesn't hang for however
    long the download takes. Poll /video/codecs/status to see when it's
    ready; a second install call while one's already running just no-ops
    against the already-in-flight download rather than starting a second."""
    status = video_codecs_status()
    if status["ready"]:
        return {"ok": True, "already_ready": True, "status": status}
    threading.Thread(target=setup_bundled_video_codecs, daemon=True).start()
    return {"ok": True, "installing": True}


@app.get("/video/probe")
def video_probe(path: str):
    """Lets the frontend show duration/fps/size right after upload, before
    committing to Generate — mirrors the old Gradio UI's on_video_upload
    auto-probe (aspect guess, long-clip warning, encode-range default).
    `path` is a filename previously returned by /video/upload, not an
    arbitrary filesystem path."""
    safe = os.path.basename(path)
    full_path = os.path.join(uploads_dir(), safe)
    if not os.path.isfile(full_path):
        raise HTTPException(404, "Uploaded video not found: " + safe)
    try:
        info = probe_video_file(full_path)
    except Exception as exc:
        raise HTTPException(400, "Could not read video: {}".format(exc)) from exc
    info["long_warning"] = info["duration"] > VIDEO_LONG_WARN_SECONDS
    info["duration_label"] = format_video_duration(info["duration"])
    return info


@app.post("/video/upload")
async def video_upload(video: UploadFile = File(...)):
    """Separate from /video/generate so the (potentially large) upload and
    the /video/probe duration/fps readout can both happen right after the
    user picks a file, before Generate is even clicked — matches the old
    UI's on_video_upload flow (probe + long-clip warning as soon as the
    file is chosen, not deferred until generation starts)."""
    ext = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    # <uuid>__<original-basename> lets the engine recover a readable
    # BASENAME for the final rendered clip's name (see
    # blobvision_paths.basename_hint_from_upload).
    name = uuid.uuid4().hex + "__" + sanitize_basename(video.filename) + ext
    path = os.path.join(uploads_dir(), name)
    with open(path, "wb") as fh:
        shutil.copyfileobj(video.file, fh)
    return {"ok": True, "path": name}


@app.post("/video/discard")
async def video_discard(path: str = Form(...)):
    """Deletes a previously-uploaded source clip from uploads_dir() — called
    by the frontend's × Clear button on the video drop zone (and when a new
    clip replaces one that was never explicitly cleared), since an uploaded
    video otherwise sits there indefinitely once /video/generate no longer
    needs it (unlike single-image uploads, which are cleaned up right after
    the one request that used them — see /generate, /style/generate, etc.)."""
    safe = os.path.basename(path)
    full_path = os.path.join(uploads_dir(), safe)
    if os.path.isfile(full_path):
        try:
            os.remove(full_path)
        except OSError:
            pass
    return {"ok": True}


@app.post("/video/generate")
async def video_generate(
    path: str = Form(...),
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),
    aspect: Optional[str] = Form(None),
    iterations: Optional[int] = Form(None),
    denoise_fidelity: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    frame_step: int = Form(4),
    encode_from: Optional[float] = Form(None),
    encode_to: Optional[float] = Form(None),
    use_encode_range: bool = Form(False),
):
    """Starts a video-generation job in a background thread and returns a
    job_id immediately — see VideoJob's docstring for why. `path` is a
    filename previously returned by /video/upload (not a raw file upload
    itself, so re-generating after tweaking a param doesn't need to
    re-upload the whole clip)."""
    global _video_job_running
    engine = get_engine()
    codecs = video_codecs_status()
    if not codecs["ready"]:
        raise HTTPException(503, "Video codecs not installed. POST /video/codecs/install first.")
    with _video_jobs_lock:
        if _video_job_running:
            raise HTTPException(409, "A video job is already running — wait for it to finish first.")
        _video_job_running = True

    safe = os.path.basename(path)
    video_path = os.path.join(uploads_dir(), safe)
    if not os.path.isfile(video_path):
        with _video_jobs_lock:
            _video_job_running = False
        raise HTTPException(404, "Uploaded video not found: " + safe)

    job_id = uuid.uuid4().hex
    job = VideoJob(job_id)
    with _video_jobs_lock:
        _video_jobs[job_id] = job
    blobvision_cancel.clear()

    def run():
        global _video_job_running
        try:
            result = engine.generate_video(
                prompt=prompt,
                video_path=video_path,
                iterations=iterations,
                denoise_fidelity=denoise_fidelity,
                seed=seed,
                negative_prompt=negative_prompt,
                aspect=aspect or DEFAULT_ASPECT,
                frame_step=frame_step,
                on_progress=job.append,
                encode_from_sec=encode_from or 0.0,
                encode_to_sec=encode_to,
                use_encode_range=use_encode_range,
            )
            job.output_url = "/outputs/" + os.path.basename(result.output_path)
            job.metadata = result.metadata
            job.state = "done"
        except blobvision_cancel.AbortedError:
            job.state = "cancelled"
        except Exception as exc:
            job.error = str(exc)
            job.state = "error"
        finally:
            with _video_jobs_lock:
                _video_job_running = False

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "job_id": job_id}


@app.get("/video/status/{job_id}")
def video_status(job_id: str):
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    return job.snapshot()


@app.post("/video/cancel/{job_id}")
def video_cancel(job_id: str):
    """The engine's cancel flag is a single global (blobvision_cancel), not
    per-job — fine since only one video job can run at a time (enforced by
    _video_job_running above), so there's never ambiguity about which job
    a cancel request means."""
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    if job.state == "running":
        blobvision_cancel.request()
    return {"ok": True}


@app.post("/cancel")
def cancel_generation():
    """Generic cancel for whatever is currently running — a single image
    generation (VQGAN/DeepDream/Style Transfer, all now routed through
    run_in_threadpool so this request isn't stuck behind the event loop) or
    a video job (equivalent to POST /video/cancel/{job_id} without needing
    to know the job_id). Only one generation of any kind runs at a time in
    this app, so the flag is never ambiguous. Safe to call when nothing is
    running — the next generate() call clears the flag before it starts."""
    blobvision_cancel.request()
    return {"ok": True}


@app.post("/deepdream/video/generate")
async def deepdream_video_generate(
    path: str = Form(...),
    aspect: Optional[str] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    intensity: float = Form(50),
    layer: str = Form(DD_DEFAULT_LAYER),
    seed: Optional[int] = Form(None),
    frame_step: int = Form(4),
    encode_from: Optional[float] = Form(None),
    encode_to: Optional[float] = Form(None),
    use_encode_range: bool = Form(False),
):
    """Same job/polling shape as /video/generate (VQGAN) — see VideoJob's
    docstring — just driving DeepDreamEngine.generate_video() instead, which
    runs its own img2img mode as the per-keyframe transform through the
    exact same shared pipeline. Reuses the same job store, /video/status,
    /video/codecs/*, and /video/cancel endpoints; no per-family duplication
    needed there since none of them are VQGAN-specific."""
    global _video_job_running
    dd = get_dd_engine()
    codecs = video_codecs_status()
    if not codecs["ready"]:
        raise HTTPException(503, "Video codecs not installed. POST /video/codecs/install first.")
    with _video_jobs_lock:
        if _video_job_running:
            raise HTTPException(409, "A video job is already running — wait for it to finish first.")
        _video_job_running = True

    safe = os.path.basename(path)
    video_path = os.path.join(uploads_dir(), safe)
    if not os.path.isfile(video_path):
        with _video_jobs_lock:
            _video_job_running = False
        raise HTTPException(404, "Uploaded video not found: " + safe)

    try:
        out_width, out_height = resolve_aspect_size(aspect=aspect or DEFAULT_ASPECT, width=width, height=height)
    except ValueError as exc:
        with _video_jobs_lock:
            _video_job_running = False
        raise HTTPException(400, str(exc)) from exc

    job_id = uuid.uuid4().hex
    job = VideoJob(job_id)
    with _video_jobs_lock:
        _video_jobs[job_id] = job
    blobvision_cancel.clear()

    def run():
        global _video_job_running
        try:
            result = dd.generate_video(
                video_path=video_path,
                width=out_width,
                height=out_height,
                intensity=intensity,
                layer=layer,
                seed=seed,
                frame_step=frame_step,
                on_progress=job.append,
                encode_from_sec=encode_from or 0.0,
                encode_to_sec=encode_to,
                use_encode_range=use_encode_range,
            )
            job.output_url = "/outputs/" + os.path.basename(result.output_path)
            job.metadata = result.metadata
            job.state = "done"
        except blobvision_cancel.AbortedError:
            job.state = "cancelled"
        except Exception as exc:
            job.error = str(exc)
            job.state = "error"
        finally:
            with _video_jobs_lock:
                _video_job_running = False

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "job_id": job_id}


@app.post("/style/video/generate")
async def style_video_generate(
    path: str = Form(...),
    aspect: Optional[str] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    preset: Optional[str] = Form(None),
    custom_prompt: Optional[str] = Form(None),
    strength: Optional[float] = Form(None),
    style_image: Optional[UploadFile] = File(None),
    style_strength: float = Form(ST_DEFAULT_STYLE_STRENGTH),
    content_weight: float = Form(ST_DEFAULT_CONTENT_WEIGHT),
    seed: Optional[int] = Form(None),
    frame_step: int = Form(4),
    encode_from: Optional[float] = Form(None),
    encode_to: Optional[float] = Form(None),
    use_encode_range: bool = Form(False),
):
    """Mirrors /style/generate's preset-vs-classic branching (see its
    docstring) extended to video: an active preset routes to the shared
    VQGAN engine's generate_style_preset_video (SDXL Turbo per keyframe,
    bypasses the classic engine entirely — same as /style/generate does for
    a single image); no preset runs StyleTransferEngine.generate_video(),
    the classic VGG19 optimizer per keyframe at its own reduced VIDEO_STEPS
    default regardless of what's dialed into the Steps slider for images."""
    global _video_job_running
    st = get_st_engine()
    codecs = video_codecs_status()
    if not codecs["ready"]:
        raise HTTPException(503, "Video codecs not installed. POST /video/codecs/install first.")
    with _video_jobs_lock:
        if _video_job_running:
            raise HTTPException(409, "A video job is already running — wait for it to finish first.")
        _video_job_running = True

    safe = os.path.basename(path)
    video_path = os.path.join(uploads_dir(), safe)
    if not os.path.isfile(video_path):
        with _video_jobs_lock:
            _video_job_running = False
        raise HTTPException(404, "Uploaded video not found: " + safe)

    try:
        out_width, out_height = resolve_aspect_size(aspect=aspect or DEFAULT_ASPECT, width=width, height=height)
    except ValueError as exc:
        with _video_jobs_lock:
            _video_job_running = False
        raise HTTPException(400, str(exc)) from exc

    preset = (preset or "").strip() or None
    prompt = None
    preset_strength = None
    style_path = None
    if preset:
        preset_data = STYLE_PRESETS.get(preset, STYLE_PRESETS[DEFAULT_STYLE_PRESET])
        prompt = (custom_prompt or "").strip() if preset == "custom" else preset_data["prompt"]
        if not prompt:
            with _video_jobs_lock:
                _video_job_running = False
            raise HTTPException(400, "Enter a custom prompt for the Custom preset.")
        preset_strength = strength if strength is not None else preset_data["strength"]
    else:
        if not style_transfer_weights_status()["ready"]:
            with _video_jobs_lock:
                _video_job_running = False
            raise HTTPException(503, "VGG19 weights not installed.")
        if style_image is None:
            with _video_jobs_lock:
                _video_job_running = False
            raise HTTPException(400, "style_image is required for classic style transfer video.")
        style_ext = os.path.splitext(style_image.filename or "style.png")[1] or ".png"
        style_path = os.path.join(uploads_dir(), uuid.uuid4().hex + style_ext)
        with open(style_path, "wb") as fh:
            shutil.copyfileobj(style_image.file, fh)

    job_id = uuid.uuid4().hex
    job = VideoJob(job_id)
    with _video_jobs_lock:
        _video_jobs[job_id] = job
    blobvision_cancel.clear()

    def run():
        global _video_job_running
        try:
            if preset:
                vqgan_engine = get_engine()
                result = vqgan_engine.generate_style_preset_video(
                    video_path=video_path,
                    prompt=prompt,
                    strength=preset_strength,
                    width=out_width,
                    height=out_height,
                    seed=seed,
                    frame_step=frame_step,
                    on_progress=job.append,
                    encode_from_sec=encode_from or 0.0,
                    encode_to_sec=encode_to,
                    use_encode_range=use_encode_range,
                )
            else:
                result = st.generate_video(
                    video_path=video_path,
                    style_image_path=style_path,
                    width=out_width,
                    height=out_height,
                    style_strength=style_strength,
                    content_weight=content_weight,
                    seed=seed,
                    frame_step=frame_step,
                    on_progress=job.append,
                    encode_from_sec=encode_from or 0.0,
                    encode_to_sec=encode_to,
                    use_encode_range=use_encode_range,
                )
            job.output_url = "/outputs/" + os.path.basename(result.output_path)
            job.metadata = result.metadata
            job.state = "done"
        except blobvision_cancel.AbortedError:
            job.state = "cancelled"
        except Exception as exc:
            job.error = str(exc)
            job.state = "error"
        finally:
            with _video_jobs_lock:
                _video_job_running = False
            # Single-use scratch — every keyframe already read it during the
            # job above (video's own source clip is discarded separately,
            # see POST /video/discard from the frontend's Clear button).
            if style_path:
                try:
                    os.remove(style_path)
                except OSError:
                    pass

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "job_id": job_id}


@app.post("/deepdream/warmup")
def deepdream_warmup():
    dd = get_dd_engine()
    dd._load_model()
    return {"ok": True, "device": str(dd.device)}


@app.post("/deepdream/generate", response_model=DeepDreamResponse)
async def deepdream_generate(
    prompt: Optional[str] = Form(""),
    negative_prompt: Optional[str] = Form(None),
    aspect: Optional[str] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    intensity: float = Form(50),
    layer: str = Form(DD_DEFAULT_LAYER),
    seed: Optional[int] = Form(None),
    upscale: int = Form(0),
    sketch: Optional[UploadFile] = File(None),
):
    """Mirrors blobvision_ui.py's do_deepdream_generate: an uploaded sketch is
    img2img; a prompt with no upload gets an SDXL sketch first (redux) via
    the shared VQGAN engine's generate_redux_sketch(); no prompt and no
    upload dreams from random noise (txt2img)."""
    dd = get_dd_engine()
    prompt = (prompt or "").strip()
    negative_prompt = (negative_prompt or "").strip() or None

    sketch_path = None
    if sketch is not None:
        ext = os.path.splitext(sketch.filename or "upload.png")[1] or ".png"
        # <uuid>__<original-basename> lets DeepDreamEngine.generate() recover
        # a readable BASENAME for img2img mode — see
        # blobvision_paths.basename_hint_from_upload.
        sketch_path = os.path.join(
            uploads_dir(), uuid.uuid4().hex + "__" + sanitize_basename(sketch.filename) + ext,
        )
        with open(sketch_path, "wb") as fh:
            shutil.copyfileobj(sketch.file, fh)

    source_for_size = None
    if sketch_path and (aspect or DEFAULT_ASPECT) == "custom":
        from PIL import Image
        source_for_size = Image.open(sketch_path)
    width, height = resolve_aspect_size(
        aspect=aspect or DEFAULT_ASPECT, width=width, height=height,
        source_width=source_for_size.width if source_for_size else None,
        source_height=source_for_size.height if source_for_size else None,
    )
    seed = resolve_seed(seed)

    init_path = None
    engine_mode = "txt2img"
    generated_sketch_url = None
    try:
        try:
            if sketch_path:
                init_path = sketch_path
                engine_mode = "img2img"
            elif prompt:
                vqgan_engine = get_engine()
                sketch_out_path = await run_in_threadpool(
                    vqgan_engine.generate_redux_sketch,
                    prompt, width, height, seed,
                    negative_prompt=negative_prompt,
                )
                init_path = sketch_out_path
                generated_sketch_url = "/outputs/" + os.path.basename(sketch_out_path)
                engine_mode = "redux"

            steps, octaves, preserve = intensity_to_params(intensity)
            # run_in_threadpool — see the comment in /generate above; same
            # event-loop-blocking concern applies to dd.generate()'s dream loop.
            result = await run_in_threadpool(
                dd.generate,
                mode=engine_mode,
                width=width,
                height=height,
                steps=steps,
                octaves=octaves,
                layer=layer,
                init_image_path=init_path,
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                preserve=preserve if engine_mode in ("redux", "img2img") else None,
            )
        except blobvision_cancel.AbortedError as exc:
            raise HTTPException(499, str(exc) or "Generation cancelled.") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

        output_path = _maybe_upscale(result.output_path, upscale)

        return DeepDreamResponse(
            mode=result.metadata["mode"],
            seed=result.seed,
            layer=layer,
            output_url="/outputs/" + os.path.basename(output_path),
            sketch_url=generated_sketch_url,
            metadata=result.metadata,
        )
    finally:
        # Only the user-uploaded sketch is scratch — the SDXL-generated one
        # (sketch_out_path, tagged "DSK" in sketch_dir()) is a real kept
        # artifact, already surfaced as sketch_url, and must not be touched.
        if sketch_path:
            try:
                os.remove(sketch_path)
            except OSError:
                pass


@app.get("/style/presets")
def style_presets():
    return [
        StylePresetInfo(key=key, label=data["label"], strength=data["strength"])
        for key, data in STYLE_PRESETS.items()
    ]


@app.get("/style/weights/status")
def style_weights_status():
    return style_transfer_weights_status()


@app.post("/style/warmup")
def style_warmup():
    st = get_st_engine()
    if not style_transfer_weights_status()["ready"]:
        return {"ok": False, "reason": "VGG19 weights not installed."}
    st._load_vgg()
    return {"ok": True}


@app.post("/style/generate", response_model=StyleResponse)
async def style_generate(
    content_image: UploadFile = File(...),
    style_image: Optional[UploadFile] = File(None),
    preset: Optional[str] = Form(None),
    custom_prompt: Optional[str] = Form(None),
    strength: Optional[float] = Form(None),
    style_strength: float = Form(ST_DEFAULT_STYLE_STRENGTH),
    content_weight: float = Form(ST_DEFAULT_CONTENT_WEIGHT),
    steps: Optional[int] = Form(None),
    aspect: Optional[str] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    upscale: int = Form(0),
):
    """Mirrors blobvision_ui.py's do_style_generate: a preset key routes to
    the shared VQGAN engine's SDXL Turbo restyle (generate_style_preset,
    caption-assist included); no preset ("" or omitted) is the classic
    VGG19 Gatys optimizer and additionally requires a style_image."""
    st = get_st_engine()

    content_ext = os.path.splitext(content_image.filename or "content.png")[1] or ".png"
    # <uuid>__<original-basename> lets StyleTransferEngine.generate() recover
    # a readable BASENAME (the content image is the "subject" of a classic
    # style transfer) — see blobvision_paths.basename_hint_from_upload.
    content_path = os.path.join(
        uploads_dir(), uuid.uuid4().hex + "__" + sanitize_basename(content_image.filename) + content_ext,
    )
    with open(content_path, "wb") as fh:
        shutil.copyfileobj(content_image.file, fh)

    source_for_size = None
    if (aspect or DEFAULT_ASPECT) == "custom":
        from PIL import Image
        source_for_size = Image.open(content_path)
    out_width, out_height = resolve_aspect_size(
        aspect=aspect or DEFAULT_ASPECT, width=width, height=height,
        source_width=source_for_size.width if source_for_size else None,
        source_height=source_for_size.height if source_for_size else None,
    )
    seed = resolve_seed(seed)
    preset = (preset or "").strip() or None
    style_path = None

    try:
        try:
            if preset:
                preset_data = STYLE_PRESETS.get(preset, STYLE_PRESETS[DEFAULT_STYLE_PRESET])
                prompt = (custom_prompt or "").strip() if preset == "custom" else preset_data["prompt"]
                if not prompt:
                    raise HTTPException(400, "Enter a custom prompt for the Custom preset.")
                vqgan_engine = get_engine()
                # run_in_threadpool — see the comment in /generate above.
                result = await run_in_threadpool(
                    vqgan_engine.generate_style_preset,
                    image_path=content_path,
                    prompt=prompt,
                    strength=strength if strength is not None else preset_data["strength"],
                    seed=seed,
                    negative_prompt=preset_data.get("negative_prompt"),
                    steps=preset_data.get("steps", STYLE_PRESET_STEPS),
                    width=out_width,
                    height=out_height,
                )
                mode = "sdxl"
            else:
                if not style_transfer_weights_status()["ready"]:
                    raise HTTPException(503, "VGG19 weights not installed.")
                if style_image is None:
                    raise HTTPException(400, "style_image is required for classic style transfer.")
                style_ext = os.path.splitext(style_image.filename or "style.png")[1] or ".png"
                style_path = os.path.join(uploads_dir(), uuid.uuid4().hex + style_ext)
                with open(style_path, "wb") as fh:
                    shutil.copyfileobj(style_image.file, fh)
                result = await run_in_threadpool(
                    st.generate,
                    content_image_path=content_path,
                    style_image_path=style_path,
                    width=out_width,
                    height=out_height,
                    steps=steps or ST_DEFAULT_STEPS,
                    style_strength=style_strength,
                    content_weight=content_weight,
                    seed=seed,
                )
                mode = "classic"
        except blobvision_cancel.AbortedError as exc:
            raise HTTPException(499, str(exc) or "Generation cancelled.") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

        output_path = _maybe_upscale(result.output_path, upscale)

        return StyleResponse(
            mode=mode,
            preset=preset,
            seed=result.seed,
            output_url="/outputs/" + os.path.basename(output_path),
            metadata=result.metadata,
        )
    finally:
        # Both are single-use scratch — the frontend re-sends them fresh on
        # every request, and neither is ever surfaced back as a URL.
        for tmp_path in (content_path, style_path):
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


@app.get("/sam2/status")
def sam2_status():
    return sam2_weights_status()


@app.post("/sam2/install")
def sam2_install():
    """Same background-thread-download shape as /video/codecs/install — a
    second call while one's already running just no-ops against the
    already-in-flight download."""
    status = sam2_weights_status()
    if status["ready"]:
        return {"ok": True, "already_ready": True, "status": status}
    threading.Thread(target=download_sam2_weights, daemon=True).start()
    return {"ok": True, "installing": True}


@app.post("/sam2/warmup")
def sam2_warmup():
    if not sam2_weights_status()["ready"]:
        return {"ok": False, "reason": "SAM2 weights not installed."}
    engine = get_sam_engine()
    engine._load()
    return {"ok": True, "device": str(engine.device)}


@app.post("/sam2/embed")
async def sam2_embed(image: UploadFile = File(...)):
    """Runs SAM2's (expensive) image encoder once on whatever image the user
    is about to select a segment on — the frontend re-sends the same bytes
    it already has locally, mirroring how /generate's own init_image works
    (no pre-upload/reference-by-path flow for images, unlike video). The
    temp copy is only needed long enough for set_image() to read it — the
    resulting embedding lives in GPU tensors, not tied to the file — so it's
    deleted immediately after, same as every other single-shot upload."""
    if not sam2_weights_status()["ready"]:
        raise HTTPException(503, "SAM2 weights not installed. POST /sam2/install first.")
    engine = get_sam_engine()
    ext = os.path.splitext(image.filename or "upload.png")[1] or ".png"
    tmp_path = os.path.join(uploads_dir(), uuid.uuid4().hex + ext)
    with open(tmp_path, "wb") as fh:
        shutil.copyfileobj(image.file, fh)
    try:
        info = await run_in_threadpool(engine.embed_image, tmp_path)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return info


class Sam2Point(BaseModel):
    x: float
    y: float
    label: int = 1  # 1 = union into the accumulated selection, 0 = subtract — see Sam2Engine.segment
    size: float = 0.0  # -1..1, picks small/medium/large among SAM2's per-point candidates


class Sam2SegmentRequest(BaseModel):
    token: str
    points: List[Sam2Point]


@app.post("/sam2/segment")
async def sam2_segment(req: Sam2SegmentRequest):
    """Returns the raw (un-refined) union mask as a single-channel PNG —
    erosion/dilation/feather happen client-side on this, not here (see
    tauri/src/main.ts), so this only needs to run once per point added, not
    once per slider tick."""
    engine = get_sam_engine()
    points = [(p.x, p.y, p.label, p.size) for p in req.points]
    try:
        mask = await run_in_threadpool(engine.segment, req.token, points)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.fromarray((mask.astype("uint8") * 255), mode="L").save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/meta/read")
async def meta_read(image: UploadFile = File(...)):
    tmp = os.path.join(uploads_dir(), uuid.uuid4().hex + ".png")
    with open(tmp, "wb") as fh:
        shutil.copyfileobj(image.file, fh)
    try:
        meta = read_metadata_from_image(tmp)
        if meta is None:
            raise HTTPException(404, "No BlobDream metadata in image")
        return {"metadata": meta, "seed": seed_from_metadata(meta)}
    finally:
        if os.path.isfile(tmp):
            os.remove(tmp)


@app.get("/outputs/recent")
def outputs_recent(limit: int = 60):
    """Every family's most recent output images, newest first — feeds the
    meme studio's thumbnail strip. Scans disk directly (not the in-memory
    engines) so it works even before a family's engine has been touched
    this session. Must stay registered before /outputs/{filename} — FastAPI
    matches path routes in registration order, so a literal /outputs/recent
    route after the {filename} catch-all would never be reached (it'd match
    {filename}="recent" first and 404 there instead). Only scans outputs_dir()
    itself (not sketch_dir()) — sketches are pre-blobify intermediates, not
    finished pieces worth surfacing in the studio strip."""
    d = outputs_dir()
    entries = []
    if os.path.isdir(d):
        for name in os.listdir(d):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(d, name)
            if not os.path.isfile(path):
                continue
            entries.append((os.path.getmtime(path), name))
    entries.sort(key=lambda e: e[0], reverse=True)
    return [
        {"filename": name, "output_url": "/outputs/" + name}
        for _mtime, name in entries[:limit]
    ]


@app.get("/outputs/{filename}")
def get_output(filename: str):
    safe = os.path.basename(filename)
    search_dirs = [outputs_dir(), sketch_dir(), uploads_dir()]
    for d in search_dirs:
        path = os.path.join(d, safe)
        if os.path.isfile(path):
            media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            return FileResponse(path, media_type=media_type)
    raise HTTPException(404, "Not found")


@app.post("/studio/save")
async def studio_save(image: UploadFile = File(...)):
    """Saves a meme composited client-side (base image + text baked in via
    canvas) into outputs/meme/ so it persists, shows up in the gallery
    folder, and appears in future /outputs/recent thumbnail strips."""
    # The composited canvas export has no meaningful original filename (it's
    # baked client-side, not sourced from a single upload), so BASENAME is
    # just the literal "meme" — matches the naming scheme's own example.
    name = build_output_name("M", "meme", ".png")
    path = os.path.join(outputs_dir(), name)
    with open(path, "wb") as fh:
        shutil.copyfileobj(image.file, fh)
    return {"ok": True, "output_url": "/outputs/" + name}


@app.post("/studio/upscale")
async def studio_upscale(image: UploadFile = File(...), scale: int = Form(2)):
    """Refines detail in the studio's current source image before it's
    cropped/baked — deliberately NOT for making the exported meme bigger
    (studioExportSize() on the frontend still caps that). Returns the
    upscaled image bytes directly rather than a saved /outputs URL, since
    this is a transient refinement step, not a gallery-worthy output; both
    the uploaded original and the upscaled result are deleted once the
    response has been sent."""
    if scale not in (2, 4):
        raise HTTPException(400, "scale must be 2 or 4")
    tmp_path = os.path.join(uploads_dir(), uuid.uuid4().hex + ".png")
    with open(tmp_path, "wb") as fh:
        shutil.copyfileobj(image.file, fh)
    try:
        out_path = get_upscale_engine(scale).upscale(tmp_path)
    finally:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)

    def cleanup():
        if os.path.isfile(out_path):
            os.remove(out_path)

    return FileResponse(out_path, media_type="image/png", background=BackgroundTask(cleanup))


def parse_args():
    p = argparse.ArgumentParser(description="BlobVision API server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--warmup", default="redux", choices=["legacy", "redux"])
    p.add_argument("--no-warmup", action="store_true")
    return p.parse_args()


def _sweep_uploads_on_exit():
    # Won't fire under the Tauri shell's hard TerminateProcess kill (see
    # main.rs's own uploads sweep for that path) — this covers running the
    # API standalone / any future graceful-shutdown path instead.
    try:
        for name in os.listdir(uploads_dir()):
            path = os.path.join(uploads_dir(), name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
    except Exception:
        pass


def main():
    import atexit
    import uvicorn

    atexit.register(_sweep_uploads_on_exit)
    cli = parse_args()
    global _engine
    _engine = BlobVisionEngine(keep_models=True)
    if not cli.no_warmup:
        print("Warming up mode:", cli.warmup)
        _engine.warmup(mode=cli.warmup)
        print("Ready:", _engine.status())
    else:
        # The Tauri shell runs with --no-warmup and instead lets the
        # frontend drive loading via /warmup/sdxl + /warmup/vqgan once the
        # window is up (see tauri/src/main.ts's startStagedWarmup). Those
        # fast-path endpoints call straight into _load_sketch_pipe() and
        # skip the _preload_hf_imports() step that the full engine.warmup()
        # above does via run_with_patience — meaning the first-time
        # diffusers/transformers import (observed up to ~90s cold) was
        # paying its cost silently inside _create_sketch_pipe_from_disk(),
        # AFTER the request already arrived, with no progress message and
        # no way to tell it apart from "reading 7 components" itself just
        # being unexplainably slow. Kicking it off here, in the background,
        # overlaps that cost with the ~30s the window/webview otherwise
        # spends starting up before /warmup/sdxl is even sent, instead of
        # the user waiting through it twice.
        #
        # Holds _engine._lock for the whole import — NOT optional. A first
        # (real, reproduced) attempt without the lock let this background
        # import race /warmup/sdxl's own request-thread import of the same
        # transformers/diffusers modules and crashed with "cannot import
        # name 'PreTrainedModel' from partially initialized module
        # 'transformers'": CPython does not make concurrent first-imports
        # of the same module graph from two threads safe, even though each
        # import on its own is fine. Sharing the engine's own lock means
        # /warmup/sdxl's `with _engine._lock:` simply waits out whatever
        # part of the preload hasn't finished yet instead of racing it —
        # worst case (request arrives before preload starts) is a wash,
        # best case (preload finishes first) the later import is a free
        # cache hit.
        # Same story for VQGAN's own dependency chain (app/generate.py —
        # taming, CLIP, open_clip, kornia, torch_optimizer, lion_pytorch):
        # /warmup/vqgan's _ensure_vqgan() does `import generate as eng`
        # lazily too, so that import cost was ALSO paid silently, inside
        # whichever request happened to trigger it first. Chained after
        # the diffusers/transformers preload above (same thread, same
        # lock held throughout) rather than run concurrently with it:
        # the two import largely disjoint library trees, but there's no
        # benefit to risking a second, harder-to-reproduce cross-import
        # race for a step that's a fair bit smaller anyway.
        def _background_preload():
            print("Background: preloading diffusers/transformers (offline)...", flush=True)
            with _engine._lock:
                _preload_hf_imports()
                print("Background: diffusers/transformers preload done.", flush=True)
                try:
                    print("Background: preloading VQGAN deps (taming/CLIP/kornia)...", flush=True)
                    import generate  # noqa: F401
                    print("Background: VQGAN deps preload done.", flush=True)
                except Exception as exc:
                    # Never let a background preload hiccup take down the
                    # whole process — worst case, /warmup/vqgan just pays
                    # this import cost itself later, same as before this
                    # change existed.
                    print("Background: VQGAN deps preload failed (non-fatal): {}: {}".format(type(exc).__name__, exc), flush=True)

        threading.Thread(target=_background_preload, daemon=True).start()
    print("BlobVision API ready:")
    print("  Swagger UI: http://{}:{}/docs".format(cli.host, cli.port))
    uvicorn.run(app, host=cli.host, port=cli.port, log_level="info")


if __name__ == "__main__":
    main()
