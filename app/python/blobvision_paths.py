"""Central path layout for BlobVision V1 (portable, repo-relative)."""
import os
import shutil

_PY_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(_PY_DIR)
BLOBVISION_ROOT = os.path.dirname(APP_DIR)
# Legacy alias used in older modules
BLOBDREAM_ROOT = BLOBVISION_ROOT

MODELS_ROOT = os.path.join(BLOBVISION_ROOT, "models")
OUTPUTS_ROOT = os.path.join(BLOBVISION_ROOT, "outputs")
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
    os.makedirs(OUTPUTS_ROOT, exist_ok=True)
    os.makedirs(VQGAN_MODEL_DIR, exist_ok=True)
    os.makedirs(OPENCLIP_ROOT, exist_ok=True)
    os.makedirs(HF_CACHE_ROOT, exist_ok=True)
    os.makedirs(STYLE_WEIGHTS_DIR, exist_ok=True)
    for sub in ("vqgan", "deepdream", "style-transfer"):
        os.makedirs(os.path.join(OUTPUTS_ROOT, sub), exist_ok=True)

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

    _legacy_outputs = os.path.join(APP_DIR, "outputs")
    if os.path.isdir(_legacy_outputs):
        for name in os.listdir(_legacy_outputs):
            if name in ("modelscope",):
                continue
            src = os.path.join(_legacy_outputs, name)
            if name.endswith(".png") or name.endswith(".mp4") or name.startswith("video_"):
                dst = os.path.join(vqgan_output_dir(), name)
            elif name == "deepdream":
                dst = os.path.join(deepdream_output_dir(), name)
                if os.path.isdir(src):
                    _merge_tree(src, deepdream_output_dir())
                    continue
            elif name == "style-transfer":
                dst = os.path.join(style_output_dir(), name)
                if os.path.isdir(src):
                    _merge_tree(src, style_output_dir())
                    continue
            elif name == "uploads":
                dst = os.path.join(vqgan_output_dir(), name)
            else:
                dst = os.path.join(vqgan_output_dir(), name)
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


def vqgan_output_dir():
    return os.path.join(OUTPUTS_ROOT, "vqgan")


def deepdream_output_dir():
    return os.path.join(OUTPUTS_ROOT, "deepdream")


def style_output_dir():
    return os.path.join(OUTPUTS_ROOT, "style-transfer")


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
        vqgan_output_dir(),
        deepdream_output_dir(),
        style_output_dir(),
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
