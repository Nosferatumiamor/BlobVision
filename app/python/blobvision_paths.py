"""Central path layout for BlobVision V1 (portable, repo-relative)."""
import os
import re
import shutil
import threading
from datetime import datetime

_PY_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(_PY_DIR)
BLOBVISION_ROOT = os.path.dirname(APP_DIR)
# Legacy alias used in older modules
BLOBDREAM_ROOT = BLOBVISION_ROOT

# Lives here rather than in app/generate.py (where it's used) because
# openclip_weights_status()/download_openclip_weights() (blobvision_engine.py)
# need it just to check/fetch weights on disk — importing it FROM generate.py
# instead would drag in generate.py's own module-level `from taming.models
# import cond_transformer, vqgan`, a genuinely heavy import (torch,
# pytorch_lightning, the whole taming-transformers chain). That import chain
# is also triggered by blobvision_api.py's own startup background-preload
# thread, and Python's import lock serializes concurrent first-imports of the
# same module — so on a fresh launch, /models/status (which gates showing the
# "download models" panel at all) would block for as long as that background
# preload thread was still mid-import, up to ~1 minute. Confirmed via a real
# report: the "download models" screen sat blank behind "Engine reachable —
# warming up..." for nearly a minute on first launch, exactly the width of
# that background import. Keeping this constant in this dependency-free
# module means the status check never touches that lock at all.
OPENCLIP_DEFAULT_PRETRAINED = {
    "ViT-L-14": "laion2b_s32b_b82k",
    "ViT-B-16": "laion2b_s34b_b88k",
    "ViT-B-32": "laion2b_s39b_b160k",
    "ViT-H-14": "laion2b_s32b_b79k",
}

OPENCLIP_HF_REPOS = {
    ("ViT-L-14", "laion2b_s32b_b82k"): "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    ("ViT-B-16", "laion2b_s34b_b88k"): "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
}

MODELS_ROOT = os.path.join(BLOBVISION_ROOT, "models")
OUTPUTS_ROOT = os.path.join(BLOBVISION_ROOT, "outputs")
# Ephemeral video-job scratch space (per-job src/processed/interpolated frame
# dirs) — a sibling of outputs/, not nested inside it, so outputs/ only ever
# holds finished renders + sketch/. Cleared per-job once the final video is
# muxed into outputs/ (see blobvision_video.py), with a startup sweep here in
# ensure_layout() as a backstop for jobs killed mid-run.
WORK_ROOT = os.path.join(BLOBVISION_ROOT, "work")
VIDEO_CODECS_ROOT = os.path.join(BLOBVISION_ROOT, "video-codecs")
VENV_PYTHON = os.path.join(BLOBVISION_ROOT, "venv", "Scripts", "python.exe")

TAMING_REPO = os.path.join(BLOBVISION_ROOT, "vendor", "taming-transformers")
if not os.path.isdir(TAMING_REPO):
    _legacy_taming = os.path.join(BLOBVISION_ROOT, "taming-transformers")
    if os.path.isdir(_legacy_taming):
        TAMING_REPO = _legacy_taming

# --- Models ---
SDXL_MODEL_DIR = os.path.join(MODELS_ROOT, "sdxl-turbo")
VQGAN_MODEL_DIR = os.path.join(MODELS_ROOT, "vqgan")
OPENCLIP_ROOT = os.path.join(MODELS_ROOT, "open_clip")
HF_CACHE_ROOT = os.path.join(MODELS_ROOT, "hf_cache")
STYLE_WEIGHTS_DIR = os.path.join(MODELS_ROOT, "style-transfer")
STYLE_WEIGHTS_PATH = os.path.join(STYLE_WEIGHTS_DIR, "vgg19_imagenet.pth")
UPSCALE_MODEL_DIR = os.path.join(MODELS_ROOT, "upscale")
UPSCALE_MODEL_PATHS = {
    2: os.path.join(UPSCALE_MODEL_DIR, "RealESRGAN_x2plus.pth"),
    4: os.path.join(UPSCALE_MODEL_DIR, "RealESRGAN_x4plus.pth"),
}
CAPTION_MODEL_DIR = os.path.join(MODELS_ROOT, "caption")
SAM2_MODEL_DIR = os.path.join(MODELS_ROOT, "sam2")
# facebook/sam2.1-hiera-small on Hugging Face — config ships inside the sam2
# pip package itself (hydra-resolved by name, not downloaded); only the
# checkpoint needs fetching. See app/scripts/download_sam2.py.
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT_NAME = "sam2.1_hiera_small.pt"
SAM2_HF_REPO = "facebook/sam2.1-hiera-small"

VQGAN_CONFIG_NAME = "vqgan_imagenet_f16_16384.yaml"
VQGAN_CHECKPOINT_NAME = "vqgan_imagenet_f16_16384.ckpt"

# Legacy model locations (pre-V1)
_LEGACY_VQGAN_DIR = os.path.join(APP_DIR, "checkpoints")
_LEGACY_OPENCLIP = os.path.join(_LEGACY_VQGAN_DIR, "open_clip")
_LEGACY_HF_CACHE = os.path.join(_LEGACY_VQGAN_DIR, "hf_cache")
_LEGACY_STYLE_WEIGHTS = os.path.join(APP_DIR, "style-transfer", "weights", "vgg19_imagenet.pth")
_LEGACY_SDXL = os.path.join(APP_DIR, "checkpoints", "sdxl-turbo")


def _pick_dir(preferred, *legacy):
    for path in (preferred,) + legacy:
        if os.path.isdir(path):
            return path
    return preferred


def _pick_file(preferred, *legacy):
    if os.path.isfile(preferred):
        return preferred
    for path in legacy:
        if os.path.isfile(path):
            return path
    return preferred


def ensure_layout():
    """Create V1 folders; migrate legacy weights/outputs when safe."""
    os.makedirs(MODELS_ROOT, exist_ok=True)
    os.makedirs(outputs_dir(), exist_ok=True)
    os.makedirs(sketch_dir(), exist_ok=True)
    os.makedirs(uploads_dir(), exist_ok=True)
    os.makedirs(work_dir(), exist_ok=True)
    os.makedirs(VQGAN_MODEL_DIR, exist_ok=True)
    os.makedirs(OPENCLIP_ROOT, exist_ok=True)
    os.makedirs(HF_CACHE_ROOT, exist_ok=True)
    os.makedirs(STYLE_WEIGHTS_DIR, exist_ok=True)
    os.makedirs(UPSCALE_MODEL_DIR, exist_ok=True)
    os.makedirs(CAPTION_MODEL_DIR, exist_ok=True)
    os.makedirs(SAM2_MODEL_DIR, exist_ok=True)

    # A video job normally clears its own work/job_<uuid> dir in a finally
    # block once its final render lands in outputs/ (see blobvision_video.py)
    # — this only leaves stale entries behind if the process was killed
    # mid-job. Sweep them on every startup so work/ never accumulates.
    for name in os.listdir(work_dir()):
        path = os.path.join(work_dir(), name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except OSError:
            pass

    _migrate_tree(_LEGACY_VQGAN_DIR, VQGAN_MODEL_DIR, (
        VQGAN_CONFIG_NAME,
        VQGAN_CHECKPOINT_NAME,
    ))
    _migrate_tree(_LEGACY_OPENCLIP, OPENCLIP_ROOT)
    _migrate_tree(_LEGACY_HF_CACHE, HF_CACHE_ROOT)
    if os.path.isfile(_LEGACY_STYLE_WEIGHTS) and not os.path.isfile(STYLE_WEIGHTS_PATH):
        try:
            shutil.copy2(_LEGACY_STYLE_WEIGHTS, STYLE_WEIGHTS_PATH)
        except OSError:
            pass

    # Pre-V1 legacy location (app/outputs/, not the per-family outputs/vqgan
    # etc. this module used until this session's flatten) — much older/rarer
    # than the current data, so a straight flatten into outputs_dir() (no
    # retroactive YYMMDD_TYPE_NNN renaming) is enough.
    _legacy_outputs = os.path.join(APP_DIR, "outputs")
    if os.path.isdir(_legacy_outputs):
        for name in os.listdir(_legacy_outputs):
            if name in ("modelscope",):
                continue
            src = os.path.join(_legacy_outputs, name)
            if os.path.isdir(src):
                _merge_tree(src, outputs_dir())
                continue
            dst = os.path.join(outputs_dir(), name)
            if os.path.exists(dst):
                continue
            try:
                shutil.move(src, dst)
            except OSError:
                pass


def _migrate_tree(src_dir, dst_dir, only_names=()):
    if not os.path.isdir(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    names = only_names or tuple(os.listdir(src_dir))
    for name in names:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if not os.path.exists(src) or os.path.exists(dst):
            continue
        try:
            shutil.move(src, dst)
        except OSError:
            pass


def _merge_tree(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for root, _dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        target_root = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        os.makedirs(target_root, exist_ok=True)
        for fname in files:
            s = os.path.join(root, fname)
            d = os.path.join(target_root, fname)
            if not os.path.exists(d):
                try:
                    shutil.move(s, d)
                except OSError:
                    pass


def vqgan_model_dir():
    return _pick_dir(VQGAN_MODEL_DIR, _LEGACY_VQGAN_DIR)


def openclip_root():
    return _pick_dir(OPENCLIP_ROOT, _LEGACY_OPENCLIP)


def hf_cache_root():
    return _pick_dir(HF_CACHE_ROOT, _LEGACY_HF_CACHE)


def sdxl_model_dir():
    return _pick_dir(SDXL_MODEL_DIR, _LEGACY_SDXL)


def style_weights_path():
    return _pick_file(STYLE_WEIGHTS_PATH, _LEGACY_STYLE_WEIGHTS)


def upscale_model_path(scale=2):
    return UPSCALE_MODEL_PATHS[scale]


def caption_model_dir():
    return CAPTION_MODEL_DIR


def sam2_model_dir():
    return SAM2_MODEL_DIR


def sam2_checkpoint_path():
    return os.path.join(SAM2_MODEL_DIR, SAM2_CHECKPOINT_NAME)


def outputs_dir():
    """Flat home for every finished render — VQGAN/DeepDream/Style/Meme,
    images and videos alike. Distinguished on disk only by the TYPE tag in
    build_output_name()'s filename, not by subfolder."""
    return OUTPUTS_ROOT


def sketch_dir():
    """Pre-blobify SDXL sketch renders, kept out of outputs_dir() so the
    flat gallery only shows finished pieces."""
    return os.path.join(OUTPUTS_ROOT, "sketch")


def uploads_dir():
    """Shared scratch space for uploaded init/content/style images and
    source videos, across all families."""
    return os.path.join(OUTPUTS_ROOT, "uploads")


def work_dir():
    return WORK_ROOT


_naming_lock = threading.Lock()


def sanitize_basename(text, max_len=40):
    """Strips any extension, lower-cases, collapses everything that isn't
    a-z0-9 into single underscores, and truncates — used both for the
    BASENAME segment of build_output_name() and for embedding a readable
    hint into upload scratch filenames (see basename_hint_from_upload)."""
    text = os.path.splitext(text or "")[0].strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return (text or "output")[:max_len]


def basename_hint_from_upload(stored_path):
    """Recovers the sanitized original filename embedded by the upload
    endpoints' `<uuid>__<original>` naming (see blobvision_api.py's
    /generate, /style/generate, /video/upload) — falls back to the stored
    file's own stem if the `__` marker isn't present (e.g. a sketch/init
    path that was never routed through an upload endpoint)."""
    stem = os.path.splitext(os.path.basename(stored_path))[0]
    return stem.split("__", 1)[1] if "__" in stem else stem


def build_output_name(type_tag, basename_source, ext, base_dir=None):
    """YYMMDD_TYPE_NNN_BASENAME.EXT — NNN is the next free 3-digit counter
    for (date, type_tag) in base_dir (default outputs_dir()), scanned off
    disk rather than kept in memory so it survives process restarts.
    Reserves the slot immediately (an empty placeholder file, O_CREAT |
    O_EXCL) under a lock so two callers sharing a type_tag from different
    engines/locks (e.g. classic-Gatys and SDXL-preset both writing "S")
    can't race each other onto the same NNN — the caller's real save
    (PIL .save() / ffmpeg -y) just overwrites the placeholder afterward."""
    base_dir = base_dir or OUTPUTS_ROOT
    date_str = datetime.now().strftime("%y%m%d")
    basename = sanitize_basename(basename_source)
    prefix = "{}_{}_".format(date_str, type_tag)
    with _naming_lock:
        os.makedirs(base_dir, exist_ok=True)
        existing = [n for n in os.listdir(base_dir) if n.startswith(prefix)]
        max_n = 0
        for name in existing:
            num = name[len(prefix):].split("_", 1)[0]
            if num.isdigit():
                max_n = max(max_n, int(num))
        filename = "{}{:03d}_{}{}".format(prefix, max_n + 1, basename, ext)
        path = os.path.join(base_dir, filename)
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            pass
    return filename


def vqgan_config_path():
    d = vqgan_model_dir()
    return os.path.join(d, VQGAN_CONFIG_NAME)


def vqgan_checkpoint_path():
    d = vqgan_model_dir()
    return os.path.join(d, VQGAN_CHECKPOINT_NAME)


def gradio_allowed_paths():
    """Paths Gradio may read when serving generated files (outputs moved outside app/)."""
    import tempfile

    candidates = (
        BLOBVISION_ROOT,
        APP_DIR,
        OUTPUTS_ROOT,
        MODELS_ROOT,
        VIDEO_CODECS_ROOT,
        tempfile.gettempdir(),
        outputs_dir(),
        sketch_dir(),
        uploads_dir(),
    )
    allowed = []
    seen = set()
    for path in candidates:
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        try:
            os.makedirs(abs_path, exist_ok=True)
        except OSError:
            if not os.path.isdir(abs_path):
                continue
        allowed.append(abs_path)
    return allowed
