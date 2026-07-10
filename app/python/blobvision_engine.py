"""BlobVision unified generation engine (Legacy / Redux / Corrupt)."""
import gc
import math
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from blobvision_paths import (
    APP_DIR,
    BLOBVISION_ROOT,
    BLOBDREAM_ROOT,
    TAMING_REPO,
    VIDEO_CODECS_ROOT,
    ensure_layout,
    sdxl_model_dir,
    vqgan_output_dir,
)

SCRIPT_DIR = APP_DIR
ensure_layout()
DEFAULT_SKETCH_MODEL = sdxl_model_dir()
DEFAULT_OUTPUT = vqgan_output_dir()
_PY_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    os.chdir(APP_DIR)
except OSError as exc:
    print("Warning: could not chdir to APP_DIR: {}".format(exc), flush=True)
for path in (APP_DIR, _PY_DIR, TAMING_REPO, BLOBDREAM_ROOT):
    if path not in sys.path and os.path.isdir(path):
        sys.path.insert(0, path)

from blobvision_meta import build_metadata, metadata_for_png, read_metadata_from_image, seed_from_metadata


class BlobVRAMCache:
    """Park / unpark SDXL sketch and VQGAN+CLIP so only one stack owns the GPU."""

    PHASE_SKETCH = "sketch"
    PHASE_VQGAN = "vqgan"

    def __init__(self, engine):
        self.engine = engine
        self.active = None
        self._sketch_on_gpu = False
        self._vqgan_on_gpu = False

    def reset(self):
        self.active = None
        self._sketch_on_gpu = False
        self._vqgan_on_gpu = False

    def activate(self, phase, log=None):
        if phase not in (self.PHASE_SKETCH, self.PHASE_VQGAN):
            raise ValueError("phase must be sketch or vqgan")
        if phase == self.active:
            return
        if phase == self.PHASE_SKETCH:
            self._park_vqgan(log)
            self._unpark_sketch(log)
        else:
            self._park_sketch(log)
            self._unpark_vqgan(log)
        self.active = phase
        self._log_vram(phase, log)

    def note_sketch_loaded_on_gpu(self):
        if self.engine._sketch_pipe is not None and self.engine.sketch_full_gpu:
            self._sketch_on_gpu = True

    def note_vqgan_loaded_on_gpu(self):
        if self.engine._vqgan_loaded:
            self._vqgan_on_gpu = True

    def _say(self, msg, log):
        print(msg, flush=True)
        if log is not None:
            log(msg)

    def _gc_gpu(self):
        import torch
        gc.collect()
        dev = self.engine.cuda_device
        if dev.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _log_vram(self, phase, log):
        import torch
        dev = self.engine.cuda_device
        if not dev.startswith("cuda") or not torch.cuda.is_available():
            self._say("VRAM cache: active phase={}".format(phase), log)
            return
        try:
            free, total = torch.cuda.mem_get_info()
            self._say(
                "VRAM cache: phase={} - {:.1f} / {:.1f} GB free".format(
                    phase, free / 1e9, total / 1e9,
                ),
                log,
            )
        except Exception:
            self._say("VRAM cache: active phase={}".format(phase), log)

    def _park_sketch(self, log):
        pipe = self.engine._sketch_pipe
        if pipe is None or not self._sketch_on_gpu:
            return
        if not self.engine.sketch_full_gpu:
            self._sketch_on_gpu = False
            return
        self._say("VRAM cache: parking SDXL on CPU...", log)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.to("cpu")
        except Exception as exc:
            self._say("VRAM cache: SDXL park note: " + str(exc), log)
        self._sketch_on_gpu = False
        self._gc_gpu()

    def _unpark_sketch(self, log):
        pipe = self.engine._sketch_pipe
        if pipe is None or not self.engine.sketch_full_gpu:
            return
        import torch
        if not self.engine.cuda_device.startswith("cuda") or not torch.cuda.is_available():
            return
        if self._sketch_on_gpu:
            return
        self._say(
            "VRAM cache: moving SDXL to GPU (~1-3 min, please be patient)...",
            log,
        )
        try:
            run_with_patience(
                lambda: pipe.to(self.engine.cuda_device),
                label="SDXL GPU transfer",
                on_stage=log,
                estimate_sec=150,
                interval=30,
            )
            self._sketch_on_gpu = True
        except Exception as exc:
            self._say("VRAM cache: SDXL unpark note: " + str(exc), log)
        self._gc_gpu()

    def _park_vqgan(self, log):
        if not self.engine._vqgan_loaded or not self._vqgan_on_gpu:
            return
        import generate as eng
        if eng.model is None:
            return
        self._say("VRAM cache: parking VQGAN+CLIP on CPU...", log)
        try:
            eng.model.to("cpu")
            if eng.perceptor is not None:
                eng.perceptor.to("cpu")
        except Exception as exc:
            self._say("VRAM cache: VQGAN park note: " + str(exc), log)
        self._vqgan_on_gpu = False
        self._gc_gpu()

    def _unpark_vqgan(self, log):
        if not self.engine._vqgan_loaded or self._vqgan_on_gpu:
            return
        import generate as eng
        if eng.model is None:
            return
        import torch
        if not self.engine.cuda_device.startswith("cuda") or not torch.cuda.is_available():
            return
        self._say("VRAM cache: moving VQGAN+CLIP to GPU...", log)
        try:
            eng.model.to(self.engine.cuda_device)
            if eng.perceptor is not None:
                eng.perceptor.to(self.engine.cuda_device)
            self._vqgan_on_gpu = True
        except Exception as exc:
            self._say("VRAM cache: VQGAN unpark note: " + str(exc), log)
        self._gc_gpu()

MAX_ITERATIONS = 2000

VQGAN_PROFILES = {
    "legacy": {
        "iterations": 150,
        "cutn": 32,
        "clip_model": "ViT-B/32",
        "clip_pretrained": None,
        "clip_backend": "openai",
        "optimiser": "Adam",
        "size": None,
        "needs_init": False,
    },
    "redux": {
        "iterations": 25,
        "cutn": 4,
        "clip_model": "ViT-B/16",
        "clip_pretrained": None,
        "clip_backend": "openai",
        "optimiser": "Adam",
        "size": None,
        "needs_init": True,
    },
    "corrupt": {
        "iterations": 20,
        "cutn": 4,
        "clip_model": "ViT-B/16",
        "clip_pretrained": None,
        "clip_backend": "openai",
        "optimiser": "Adam",
        "size": None,
        "needs_init": True,
    },
}

DEFAULT_ITERATIONS = {name: p["iterations"] for name, p in VQGAN_PROFILES.items()}
DEFAULT_ITERATIONS["video"] = 15
VIDEO_LONG_WARN_SECONDS = 30
DEFAULT_DENOISE = {"redux": 0.5, "corrupt": 0.3, "video": 0.3}
SDXL_DIM_ALIGN = 8
ASPECT_FORMATS = {
    "1:1": (384, 384),
    "16:9": (480, 272),
    "9:16": (272, 480),
}
ASPECT_CUSTOM = "custom"
DEFAULT_ASPECT = "1:1"
IMG2IMG_MAX_PIXELS = max(w * h for w, h in ASPECT_FORMATS.values())
IMG2IMG_MAX_SIDE = max(max(w, h) for w, h in ASPECT_FORMATS.values())
IMG2IMG_MIN_SIDE = 64
VIDEO_FRAME_STEP = 4
VQ_VIDEO_MAX_SOURCE_FPS = 30.0
# Default first — Gradio may send radio index 0 when the control was hidden at submit.
VQ_VIDEO_FRAME_STEP_CHOICES = ["4 frames", "2 frames", "off"]
VQ_VIDEO_FRAME_STEP_MAP = {"off": 1, "2 frames": 2, "4 frames": 4}
DEFAULT_VQ_VIDEO_FRAME_STEP = "4 frames"


def parse_vq_video_frame_step(label):
    """Map UI radio value to stride (1=all frames, 2/4=every Nth frame)."""
    if label is None or label == "":
        return VQ_VIDEO_FRAME_STEP_MAP[DEFAULT_VQ_VIDEO_FRAME_STEP]
    if isinstance(label, (int, float)):
        idx = int(label)
        if 0 <= idx < len(VQ_VIDEO_FRAME_STEP_CHOICES):
            return VQ_VIDEO_FRAME_STEP_MAP[VQ_VIDEO_FRAME_STEP_CHOICES[idx]]
        if idx in (1, 2, 4):
            return idx
        return VQ_VIDEO_FRAME_STEP_MAP[DEFAULT_VQ_VIDEO_FRAME_STEP]
    text = str(label).strip()
    if text in VQ_VIDEO_FRAME_STEP_MAP:
        return VQ_VIDEO_FRAME_STEP_MAP[text]
    if text.isdigit():
        return max(1, int(text))
    return VQ_VIDEO_FRAME_STEP_MAP[DEFAULT_VQ_VIDEO_FRAME_STEP]


def _subsample_frame_paths(frames, step):
    """Keep every Nth frame; always retain the last frame for timeline coverage."""
    step = max(1, int(step))
    frames = list(frames)
    if step == 1 or len(frames) <= 1:
        return frames
    picked = frames[::step]
    if picked[-1] != frames[-1]:
        picked.append(frames[-1])
    return picked


def _resolve_timeline_fps(probed_fps, decoded_fps=None):
    """Playback cadence for sync — collapses 96/72fps export artifacts to 24/30/etc."""
    rates = [float(r) for r in (decoded_fps, probed_fps) if r and float(r) > 0.0]
    rate = max(rates) if rates else 24.0
    if rate <= 48.0:
        return rate if rate >= 5.0 else 24.0
    # Prefer the smallest standard base (96 -> 24x4, not 50x2).
    for base in (24.0, 25.0, 30.0, 48.0, 50.0, 60.0):
        ratio = rate / base
        nearest = round(ratio)
        if nearest >= 2 and abs(ratio - nearest) < 0.05:
            return base
    return 30.0 if rate > 60.0 else rate


def _normalize_working_fps(source_fps):
    """Cap input cadence to 24/25/30 before VQGAN when the container runs faster."""
    fps = float(source_fps)
    if fps <= VQ_VIDEO_MAX_SOURCE_FPS:
        return fps
    if fps <= 60.0:
        return 30.0
    return _resolve_timeline_fps(fps)


def _normalize_video_clip(
    video_path, out_path, start_sec, duration_sec, target_fps, on_progress=None,
):
    """Re-encode a clip segment at a lower constant fps (silent proxy for frame work)."""
    import subprocess

    cmd = [_find_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error"]
    if start_sec and float(start_sec) > 0:
        cmd.extend(["-ss", str(float(start_sec))])
    cmd.extend(["-i", video_path])
    if duration_sec is not None and float(duration_sec) > 0:
        cmd.extend(["-t", str(float(duration_sec))])
    cmd.extend([
        "-vf", "fps={:.6f}".format(float(target_fps)),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        out_path,
    ])
    _video_log(
        "Video: normalizing clip to {:.1f} fps — {}".format(float(target_fps), " ".join(cmd)),
        on_progress,
    )
    subprocess.check_call(cmd)
    if not os.path.isfile(out_path):
        raise RuntimeError("Video normalization failed: " + out_path)
    return out_path


def _estimate_vq_video_keyframes(clip_duration, timeline_fps, frame_step):
    clip_duration = max(0.05, float(clip_duration))
    timeline_fps = max(1.0, float(timeline_fps))
    frame_step = max(1, int(frame_step))
    if frame_step <= 1:
        return max(2, int(round(clip_duration * timeline_fps)))
    return max(2, int(round(clip_duration * timeline_fps / frame_step)))


def _downsample_frame_paths_to_count(frames, target_count):
    frames = list(frames)
    target_count = max(1, int(target_count))
    if len(frames) <= target_count:
        return frames
    if target_count == 1:
        return [frames[0]]
    picked = []
    last_idx = -1
    for i in range(target_count):
        idx = int(round(i * (len(frames) - 1) / float(target_count - 1)))
        idx = max(last_idx + 1 if i else 0, min(idx, len(frames) - 1))
        picked.append(frames[idx])
        last_idx = idx
    return picked
VQGAN_NEGATIVE_WEIGHT = -1.0
SDXL_CFG_WITH_NEGATIVE = 1.0

SKETCH_MARKERS = [
    "model_index.json",
    os.path.join("unet", "diffusion_pytorch_model.fp16.safetensors"),
    os.path.join("vae", "diffusion_pytorch_model.fp16.safetensors"),
]


@dataclass
class GenerateResult:
    mode: str
    prompt: str
    seed: int
    iterations: int
    denoise_fidelity: Optional[float]
    output_path: str
    sketch_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def clamp_iterations(n):
    n = int(n)
    if n < 0:
        raise ValueError("iterations must be >= 0")
    if n > MAX_ITERATIONS:
        return MAX_ITERATIONS
    return n


def resolve_seed(seed):
    import torch

    if seed is None or int(seed) < 0:
        return int(torch.randint(0, 2**31 - 1, (1,)).item())
    return int(seed)


def build_vqgan_prompts(positive, negative=None, neg_weight=VQGAN_NEGATIVE_WEIGHT):
    """Positive prompts plus weighted negatives for CLIP guidance."""
    prompts = []
    for part in (positive or "").split("|"):
        part = part.strip()
        if part:
            prompts.append(part)
    if negative and negative.strip():
        neg_text = negative.replace("|", ",")
        for part in neg_text.split(","):
            part = part.strip()
            if part:
                prompts.append("{}:{}".format(part, neg_weight))
    return prompts if prompts else [positive.strip()]


def slugify(text, max_len=40):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return (text or "prompt")[:max_len]


def sketch_model_ready(model_dir):
    model_dir = os.path.abspath(model_dir)
    missing = [
        rel for rel in SKETCH_MARKERS
        if not os.path.isfile(os.path.join(model_dir, rel))
    ]
    return len(missing) == 0, missing


def _preload_hf_imports():
    """Import transformers/diffusers Python packages (venv). No model download."""
    from transformers import PreTrainedModel  # noqa: F401
    from diffusers import AutoPipelineForText2Image  # noqa: F401


def log_line(msg, on_stage=None):
    """Print once to console/UI; optional on_stage updates the loading banner only."""
    msg = (msg or "").strip()
    if not msg:
        return
    print(msg, flush=True)
    if on_stage is not None:
        on_stage(msg)


def guess_aspect_from_size(width, height):
    """Map source dimensions to 1:1, 16:9, or 9:16 with clear dead zones."""
    w = max(1, int(width or 1))
    h = max(1, int(height or 1))
    ratio = float(w) / float(h)
    if ratio >= 1.35:
        return "16:9"
    if ratio <= 0.74:
        return "9:16"
    return "1:1"


def fit_size_preserving_aspect(
    width,
    height,
    max_pixels=None,
    max_side=None,
    min_side=None,
):
    """Scale to fit a pixel budget while keeping the source aspect ratio."""
    w = max(1, int(width or 1))
    h = max(1, int(height or 1))
    budget = max_pixels if max_pixels is not None else IMG2IMG_MAX_PIXELS
    side_cap = max_side if max_side is not None else IMG2IMG_MAX_SIDE
    floor = min_side if min_side is not None else IMG2IMG_MIN_SIDE
    scale = 1.0
    if w * h > budget:
        scale = min(scale, (float(budget) / float(w * h)) ** 0.5)
    if max(w, h) > side_cap:
        scale = min(scale, float(side_cap) / float(max(w, h)))
    out_w = max(floor, int(round(w * scale)))
    out_h = max(floor, int(round(h * scale)))
    return snap_sdxl_dims(out_w, out_h, min_side=floor)


def snap_sdxl_dims(width, height, min_side=None):
    """SDXL Turbo / Lightning require width and height divisible by 8."""
    floor = min_side if min_side is not None else IMG2IMG_MIN_SIDE
    w = max(floor, int(width or 1))
    h = max(floor, int(height or 1))
    w -= w % SDXL_DIM_ALIGN
    h -= h % SDXL_DIM_ALIGN
    w = max(floor, w)
    h = max(floor, h)
    return w, h


def resolve_aspect_size(aspect=None, width=None, height=None, source_width=None, source_height=None):
    if width is not None and height is not None:
        return snap_sdxl_dims(width, height)
    key = (aspect or DEFAULT_ASPECT).strip()
    if key == ASPECT_CUSTOM:
        if source_width is not None and source_height is not None:
            return fit_size_preserving_aspect(source_width, source_height)
        raise ValueError("custom aspect requires source dimensions or explicit width/height")
    if key not in ASPECT_FORMATS:
        raise ValueError("Unknown aspect format: {}".format(key))
    return snap_sdxl_dims(*ASPECT_FORMATS[key])


def run_with_patience(fn, label, on_stage=None, estimate_sec=120, interval=30):
    """Run a blocking call while emitting elapsed-time heartbeats (no native progress bar)."""
    import time
    import threading

    stop = threading.Event()
    start = time.time()

    def heartbeat():
        while not stop.wait(interval):
            elapsed = int(time.time() - start)
            log_line(
                "{} — still working ({}s / ~{}s typical). Please be patient.".format(
                    label, elapsed, estimate_sec,
                ),
                on_stage,
            )

    worker = threading.Thread(target=heartbeat, name="patience-" + label, daemon=True)
    worker.start()
    try:
        return fn()
    finally:
        stop.set()
        worker.join(timeout=1.0)
        elapsed = int(time.time() - start)
        log_line("{} done ({}s).".format(label, elapsed), on_stage)


def purge_generate_globals():
    """Drop generate.py training globals so a new session cannot stack on old tensors."""
    import generate as gen_eng

    gen_eng.model = None
    gen_eng.perceptor = None
    gen_eng.make_cutouts = None
    gen_eng.clip_tokenizer = None
    gen_eng.device = None
    if hasattr(gen_eng, "z"):
        gen_eng.z = None
    if hasattr(gen_eng, "z_orig"):
        gen_eng.z_orig = None
    if hasattr(gen_eng, "opt"):
        gen_eng.opt = None


def enable_hub_downloads(log=None):
    """Re-enable HuggingFace/network downloads (run.ps1 forces offline for local inference)."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    try:
        import huggingface_hub.constants as hf_constants
        hf_constants.HF_HUB_OFFLINE = False
    except Exception:
        pass
    if log is not None:
        log("Réseau HuggingFace activé pour le téléchargement.")


def resolve_sketch_full_gpu(cuda_device="cuda:0", explicit=None):
    """Full GPU SDXL is much faster; auto-on when VRAM >= 11 GB."""
    if explicit is not None:
        return bool(explicit)
    env = os.environ.get("BLOBVISION_SKETCH_FULL_GPU", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    import torch
    if not cuda_device.startswith("cuda") or not torch.cuda.is_available():
        return False
    try:
        free, total = torch.cuda.mem_get_info()
        return total >= 11 * 1024 ** 3 and free >= 7 * 1024 ** 3
    except Exception:
        return False


def skip_sketch_warmup_pass():
    return os.environ.get("BLOBVISION_SKIP_WARMUP", "").strip().lower() in ("1", "true", "yes")



def profile_argv(profile, cuda_device, seed, lr, iterations):
    argv = [
        "-i", str(iterations),
        "-se", str(max(1, iterations)),
        "-cuts", str(profile["cutn"]),
        "-m", profile["clip_model"],
        "--clip-backend", profile["clip_backend"],
        "-opt", profile["optimiser"],
        "-cd", cuda_device,
        "-p", "placeholder",
    ]
    if profile.get("clip_pretrained"):
        argv.extend(["--clip-pretrained", profile["clip_pretrained"]])
    if profile.get("size"):
        argv.extend(["-s", str(profile["size"][0]), str(profile["size"][1])])
    if seed is not None:
        argv.extend(["-sd", str(seed)])
    if lr is not None:
        argv.extend(["-lr", str(lr)])
    return argv


class BlobVisionEngine:
    def __init__(
        self,
        cuda_device="cuda:0",
        output_dir=None,
        sketch_model_dir=None,
        learning_rate=0.1,
        sketch_full_gpu=None,
        keep_models=True,
    ):
        self.cuda_device = cuda_device
        self.output_dir = os.path.abspath(output_dir or DEFAULT_OUTPUT)
        self.sketch_model_dir = os.path.abspath(sketch_model_dir or DEFAULT_SKETCH_MODEL)
        self.learning_rate = learning_rate
        self.sketch_full_gpu = resolve_sketch_full_gpu(cuda_device, sketch_full_gpu)
        self.keep_models = keep_models
        mode = "full GPU" if self.sketch_full_gpu else "CPU offload (slower, saves VRAM)"
        print(
            "SDXL placement: "
            + mode
            + " — after disk load, GPU upload is slow (~1-3 min) with no progress bar.",
            flush=True,
        )

        self._lock = threading.Lock()
        self._sketch_pipe = None
        self._vqgan_loaded = False
        self._vqgan_clip_key = None
        self._counter = 1
        self._vram = BlobVRAMCache(self)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "uploads"), exist_ok=True)

    def status(self):
        ready, missing = sketch_model_ready(self.sketch_model_dir)
        return {
            "vqgan_loaded": self._vqgan_loaded,
            "sdxl_loaded": self._sketch_pipe is not None,
            "sdxl_checkpoint_ok": ready,
            "sdxl_missing": missing,
            "output_dir": self.output_dir,
            "modes": ["legacy", "redux", "corrupt", "video"],
            "defaults": {
                "iterations": DEFAULT_ITERATIONS,
                "denoise_fidelity": DEFAULT_DENOISE,
                "aspect_formats": list(ASPECT_FORMATS.keys()) + [ASPECT_CUSTOM],
                "default_aspect": DEFAULT_ASPECT,
                "redux_sketch_size": list(ASPECT_FORMATS[DEFAULT_ASPECT]),
                "redux_sketch_steps": 4,
            },
        }


    def vqgan_ready(self):
        return bool(self._vqgan_loaded)

    def sdxl_ready(self):
        return self._sketch_pipe is not None

    def warmup_staged(self, on_stage=None, on_vqgan_ready=None, on_sdxl_ready=None):
        """Load VQGAN then SDXL with GPU burst prep (VRAM flush + TF32)."""

        def stage(msg):
            log_line(msg, on_stage)

        if self._sketch_pipe is not None and self._vqgan_loaded:
            stage("Ready")
            if on_vqgan_ready:
                on_vqgan_ready()
            if on_sdxl_ready:
                on_sdxl_ready()
            return

        purge_generate_globals()
        self._vram.reset()

        # --- Burst prep: maximize free VRAM + enable TF32 ---
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(1.0, 0)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            free, total = torch.cuda.mem_get_info()
            stage(
                "GPU burst: {:.1f} / {:.1f} GB free — TF32 enabled.".format(
                    free / 1e9, total / 1e9,
                )
            )

        stage("Preparing Python libs (diffusers/transformers, offline)...")
        run_with_patience(
            _preload_hf_imports,
            label="Python libs import",
            on_stage=on_stage,
            estimate_sec=90,
            interval=15,
        )

        with self._lock:
            stage("Loading VQGAN + CLIP...")
            self._ensure_vqgan("redux", DEFAULT_ITERATIONS["redux"])
            self._vram.note_vqgan_loaded_on_gpu()
        if on_vqgan_ready:
            on_vqgan_ready()

        with self._lock:
            stage("Loading SDXL Turbo (local checkpoints)...")
            self._load_sketch_pipe(on_stage=on_stage)
            self._vram.note_sketch_loaded_on_gpu()
            self._vram.activate(BlobVRAMCache.PHASE_SKETCH)
        if on_sdxl_ready:
            on_sdxl_ready()
        stage("Ready")

    def warmup_full(self, on_stage=None):
        """Load VQGAN then SDXL (same as warmup_staged, no partial callbacks)."""
        self.warmup_staged(on_stage=on_stage)

    def park_all_for_aux(self, log=None):
        """Park SDXL and VQGAN on CPU so auxiliary engines can use the GPU."""
        with self._lock:
            self._vram._park_vqgan(log)
            self._vram._park_sketch(log)
            self._vram.active = None
            self._vram._gc_gpu()

    def shutdown(self):
        """Unload SDXL + VQGAN so Start can reload without restarting the UI."""
        import gc
        import torch

        with self._lock:
            had_sdxl = self._sketch_pipe is not None
            had_vqgan = self._vqgan_loaded
            if had_sdxl:
                print("Disconnect: unloading SDXL Turbo...", flush=True)
                self._unload_sketch_pipe()
                print("Disconnect: SDXL Turbo removed from memory.", flush=True)
            else:
                print("Disconnect: SDXL Turbo was not loaded.", flush=True)
            if had_vqgan:
                print("Disconnect: unloading VQGAN + CLIP...", flush=True)
                import generate as gen_eng

                gen_eng.model = None
                gen_eng.perceptor = None
                if hasattr(gen_eng, "opt"):
                    gen_eng.opt = None
                self._vqgan_loaded = False
                self._vqgan_clip_key = None
                purge_generate_globals()
                print("Disconnect: VQGAN + CLIP removed from memory.", flush=True)
            else:
                print("Disconnect: VQGAN + CLIP was not loaded.", flush=True)
            gc.collect()
            if self.cuda_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    free, total = torch.cuda.mem_get_info()
                    print(
                        "Disconnect: GPU cache cleared ({:.1f} / {:.1f} GB free).".format(
                            free / 1e9, total / 1e9,
                        ),
                        flush=True,
                    )
                except Exception:
                    print("Disconnect: GPU cache cleared.", flush=True)
            self._vram.reset()
            print("Disconnect complete — safe to press Start for a full reload.", flush=True)

    def run_sketch_warmup(self, on_stage=None, on_progress=None):
        if self._sketch_pipe is None:
            return

        def stage(msg):
            log_line(msg, on_stage)

        if skip_sketch_warmup_pass():
            stage("Skipping SDXL warmup pass (BLOBVISION_SKIP_WARMUP=1)")
            return

        stage("SDXL pipeline warmup (first GPU forward pass)...")

        with self._lock:
            if self.sketch_full_gpu:
                stage("SDXL warmup (full GPU, ~5-15s)...")
            else:
                stage("SDXL warmup (CPU offload moves weights each step, 30-90s)...")
            self._vram.activate(BlobVRAMCache.PHASE_SKETCH)
            self._warmup_sketch_pipe(on_progress=on_progress)
            self._vram.activate(BlobVRAMCache.PHASE_SKETCH)
            stage("Warmup complete")

    def warmup(self, mode="redux", on_stage=None):
        with self._lock:
            if mode in ("redux", "corrupt") and self._sketch_pipe is None:
                self._load_sketch_pipe(on_stage=on_stage)
                self._vram.note_sketch_loaded_on_gpu()
            profile = "legacy" if mode == "legacy" else "redux"
            self._ensure_vqgan(profile, DEFAULT_ITERATIONS.get(mode, 20))
            self._vram.note_vqgan_loaded_on_gpu()
            self._vram.activate(BlobVRAMCache.PHASE_VQGAN)

    def generate(
        self,
        mode,
        prompt,
        iterations=None,
        denoise_fidelity=None,
        seed=None,
        init_image_path=None,
        negative_prompt=None,
        sketch_steps=4,
        aspect=DEFAULT_ASPECT,
        width=None,
        height=None,
    ):
        mode = mode.lower().strip()
        if mode not in VQGAN_PROFILES and mode != "video":
            raise ValueError("Unknown mode: {}".format(mode))
        if not prompt or not prompt.strip():
            if not (init_image_path and os.path.isfile(init_image_path)):
                raise ValueError("prompt is required")
            prompt = "image"

        width, height = resolve_aspect_size(aspect=aspect, width=width, height=height)

        prompt = prompt.strip()
        negative_prompt = (negative_prompt or "").strip() or None
        actual_seed = resolve_seed(seed)
        iters = clamp_iterations(
            iterations if iterations is not None else DEFAULT_ITERATIONS[mode],
        )

        if mode == "legacy":
            if init_image_path and os.path.isfile(init_image_path):
                denoise = float(
                    denoise_fidelity if denoise_fidelity is not None
                    else DEFAULT_DENOISE.get("redux", 0.5),
                )
            else:
                denoise = None
        else:
            denoise = float(
                denoise_fidelity if denoise_fidelity is not None
                else DEFAULT_DENOISE.get(mode, 0.5),
            )

        with self._lock:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = slugify(prompt)
            tag = "{:04d}_{}_{}".format(self._counter, stamp, slug)
            output_path = os.path.join(self.output_dir, tag + ".png")
            sketch_path = None

            if mode == "redux":
                if init_image_path and os.path.isfile(init_image_path):
                    # User provided an image (img2img redux) — skip SDXL sketch
                    init_image_path = os.path.abspath(init_image_path)
                    sketch_path = init_image_path
                    self._vram.activate(BlobVRAMCache.PHASE_VQGAN)
                    print("Redux img2img: using provided image, skipping SDXL sketch.", flush=True)
                else:
                    sketch_path = os.path.join(self.output_dir, tag + "_sketch.png")
                    self._vram.activate(BlobVRAMCache.PHASE_SKETCH)
                    self._generate_sketch(
                        prompt, sketch_path, width, height, sketch_steps, actual_seed,
                        negative_prompt=negative_prompt,
                    )
                    self._vram.activate(BlobVRAMCache.PHASE_VQGAN)
                    init_image_path = sketch_path
            elif mode == "corrupt":
                if not init_image_path or not os.path.isfile(init_image_path):
                    raise ValueError("corrupt mode requires init_image")
                init_image_path = os.path.abspath(init_image_path)
            elif mode == "legacy":
                if init_image_path and os.path.isfile(init_image_path):
                    init_image_path = os.path.abspath(init_image_path)
                else:
                    init_image_path = None
            else:
                init_image_path = None

            vqgan_profile = "legacy" if mode == "legacy" else "redux"
            if mode != "redux":
                self._vram.activate(BlobVRAMCache.PHASE_VQGAN)
            self._run_vqgan(
                profile_name=vqgan_profile,
                prompt=prompt,
                negative_prompt=negative_prompt,
                output_path=output_path,
                iterations=iters,
                init_image=init_image_path,
                init_weight=float(denoise) if denoise is not None else 0.0,
                seed=actual_seed,
                mode=mode,
                sketch_path=sketch_path,
                sketch_steps=sketch_steps,
                sketch_size=(width, height),
                output_size=(width, height),
            )

            meta = build_metadata(
                mode=mode,
                prompt=prompt,
                seed=actual_seed,
                iterations=iters,
                denoise_fidelity=denoise,
                negative_prompt=negative_prompt,
                sketch_steps=sketch_steps if mode == "redux" else None,
                sketch_size=(width, height) if mode == "redux" else None,
                init_image=init_image_path if mode == "corrupt" else None,
                sketch_path=sketch_path,
            )

            self._counter += 1
            return GenerateResult(
                mode=mode,
                prompt=prompt,
                seed=actual_seed,
                iterations=iters,
                denoise_fidelity=denoise,
                output_path=output_path,
                sketch_path=sketch_path,
                metadata=meta,
            )

    def _pick_sketch_dtype(self):
        import torch
        return torch.float16 if torch.cuda.is_available() else torch.float32

    def _create_sketch_pipe_from_disk(self, on_stage=None):
        def tick(msg):
            log_line(msg, on_stage)

        ready, missing = sketch_model_ready(self.sketch_model_dir)
        if not ready:
            raise RuntimeError(
                "SDXL Turbo checkpoint incomplete: " + ", ".join(missing),
            )

        import torch
        from diffusers import AutoPipelineForText2Image

        dtype = self._pick_sketch_dtype()
        tick(
            "SDXL: reading 7 components from local disk (~1-2 min, progress bar below)..."
        )
        pipe = AutoPipelineForText2Image.from_pretrained(
            self.sketch_model_dir,
            torch_dtype=dtype,
            variant="fp16",
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        tick("SDXL: pipeline components loaded.")
        return pipe

    def _finalize_sketch_pipe_gpu(self, pipe, on_stage=None):
        if pipe is None:
            return self._sketch_pipe
        if self._sketch_pipe is not None:
            return self._sketch_pipe

        def tick(msg):
            log_line(msg, on_stage)

        import torch

        if self.cuda_device.startswith("cuda") and torch.cuda.is_available():
            gpu_id = int(self.cuda_device.split(":")[-1]) if ":" in self.cuda_device else 0
            if self.sketch_full_gpu:
                tick(
                    "SDXL: moving ~6.5 GB to GPU — please be patient "
                    "(usually 1-3 min, up to 5 min, no progress bar)..."
                )
                run_with_patience(
                    lambda: pipe.to(self.cuda_device),
                    label="SDXL GPU transfer",
                    on_stage=on_stage,
                    estimate_sec=150,
                    interval=30,
                )
            else:
                tick("SDXL: enabling CPU offload (slow inference, saves VRAM)...")
                run_with_patience(
                    lambda: pipe.enable_model_cpu_offload(gpu_id=gpu_id),
                    label="SDXL CPU offload setup",
                    on_stage=on_stage,
                    estimate_sec=60,
                    interval=20,
                )
        tick("SDXL: pipeline ready.")
        self._sketch_pipe = pipe
        return pipe

    def _load_sketch_pipe(self, on_stage=None):
        if self._sketch_pipe is not None:
            return self._sketch_pipe
        pipe = self._create_sketch_pipe_from_disk(on_stage=on_stage)
        return self._finalize_sketch_pipe_gpu(pipe, on_stage=on_stage)

    def _warmup_sketch_pipe(self, on_progress=None):
        """Cheap forward pass so the first real sketch avoids cold-start latency."""
        if self._sketch_pipe is None:
            return
        import torch

        def tick(msg):
            if on_progress is not None:
                on_progress(msg)
            print(msg, flush=True)

        pipe = self._sketch_pipe
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        warm = 128 if self.sketch_full_gpu else 192
        tick("Warmup: SDXL forward pass {}x{} (1 step)...".format(warm, warm))
        generator = torch.Generator(self.cuda_device).manual_seed(0)

        def _on_step_end(pipe_obj, step, timestep, callback_kwargs):
            tick("Warmup: inference step {}/1".format(step + 1))
            return callback_kwargs

        with torch.inference_mode():
            try:
                pipe(
                    prompt="warmup",
                    height=warm,
                    width=warm,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                    generator=generator,
                    callback_on_step_end=_on_step_end,
                )
            except TypeError:
                pipe(
                    prompt="warmup",
                    height=warm,
                    width=warm,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                    generator=generator,
                )
        tick("Warmup: SDXL forward pass finished.")

    def _unload_sketch_pipe(self):
        if self._sketch_pipe is None:
            return
        import torch
        del self._sketch_pipe
        self._sketch_pipe = None
        gc.collect()
        if self.cuda_device.startswith("cuda"):
            torch.cuda.empty_cache()

    def generate_redux_sketch(
        self,
        prompt,
        width,
        height,
        seed,
        negative_prompt=None,
        sketch_steps=4,
        output_dir=None,
    ):
        """SDXL Turbo sketch only (for DeepDream redux)."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required")
        with self._lock:
            self._load_sketch_pipe()
            self._vram.activate(BlobVRAMCache.PHASE_SKETCH)
            out_dir = os.path.abspath(output_dir or self.output_dir)
            os.makedirs(out_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sketch_path = os.path.join(
                out_dir,
                "sketch_{}_{:04d}.png".format(stamp, self._counter),
            )
            self._generate_sketch(
                prompt.strip(),
                sketch_path,
                int(width),
                int(height),
                sketch_steps,
                int(seed),
                negative_prompt=(negative_prompt or "").strip() or None,
            )
            return sketch_path

    def _generate_sketch(self, prompt, out_path, width, height, steps, seed, negative_prompt=None):
        import torch

        pipe = self._load_sketch_pipe()
        generator = torch.Generator(self.cuda_device).manual_seed(seed)
        steps = max(1, min(int(steps), 4))
        neg = (negative_prompt or "").strip() or None
        guidance = SDXL_CFG_WITH_NEGATIVE if neg else 0.0
        result = pipe(
            prompt=prompt,
            negative_prompt=neg,
            height=int(height),
            width=int(width),
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result.images[0].save(out_path)
        del result

    def _ensure_vqgan(self, profile_name, iterations):
        import generate as eng

        profile = VQGAN_PROFILES[profile_name]
        if not self._vqgan_loaded:
            eng.init_args(profile_argv(
                profile, self.cuda_device, None, self.learning_rate, iterations,
            ))
            eng.load_models()
            self._vqgan_loaded = True
            self._vqgan_clip_key = self._clip_key(profile_name)

    def _clip_key(self, profile_name):
        profile = VQGAN_PROFILES[profile_name]
        import generate as eng
        size = tuple(eng.args.size) if getattr(eng, "args", None) and eng.args.size else None
        return (
            profile["clip_model"],
            profile["cutn"],
            size,
            profile["clip_backend"],
            profile["clip_pretrained"],
        )

    def _apply_vqgan_profile(
        self, profile_name, iterations, init_image, init_weight, output_size=None,
    ):
        import generate as eng

        profile = VQGAN_PROFILES[profile_name]
        iters = clamp_iterations(iterations)
        eng.args.max_iterations = iters
        eng.args.display_freq = max(50, iters) if iters >= 50 else iters
        eng.args.cutn = profile["cutn"]
        eng.args.clip_model = profile["clip_model"]
        eng.args.clip_backend = profile.get("clip_backend", "openai")
        eng.args.clip_pretrained = profile.get("clip_pretrained")
        eng.args.optimiser = profile["optimiser"]
        if output_size:
            eng.args.size = list(output_size)
        elif profile.get("size"):
            eng.args.size = list(profile["size"])

        if init_image:
            eng.args.init_image = os.path.abspath(init_image)
            eng.args.init_weight = float(init_weight)
        else:
            eng.args.init_image = None
            eng.args.init_weight = 0.0

        key = self._clip_key(profile_name)
        if self._vqgan_clip_key != key:
            eng.reload_clip_and_cutouts()
            self._vqgan_clip_key = key

    def _run_vqgan(
        self, profile_name, prompt, output_path, iterations,
        init_image, init_weight, seed, mode, sketch_path,
        sketch_steps, sketch_size, negative_prompt=None, output_size=None,
    ):
        import generate as eng

        self._ensure_vqgan(profile_name, iterations)
        self._apply_vqgan_profile(
            profile_name, iterations, init_image, init_weight,
            output_size=output_size or sketch_size,
        )

        meta = build_metadata(
            mode=mode,
            prompt=prompt,
            seed=seed,
            iterations=iterations,
            denoise_fidelity=init_weight if mode != "legacy" else None,
            negative_prompt=negative_prompt,
            sketch_steps=sketch_steps if mode == "redux" else None,
            sketch_size=sketch_size if mode == "redux" else None,
            init_image=init_image if mode == "corrupt" else None,
            sketch_path=sketch_path,
        )
        eng.args.output = os.path.abspath(output_path)
        eng.args.png_metadata = metadata_for_png(meta)
        prompt_list = build_vqgan_prompts(prompt, negative_prompt)
        print(
            "VQGAN run: profile={}, cutn={}, iterations={}, device={}".format(
                profile_name, eng.args.cutn, iterations, eng.args.cuda_device,
            ),
            flush=True,
        )
        eng.setup_generation(text_prompt=prompt_list, seed=seed)
        import blobvision_cancel
        blobvision_cancel.clear()
        eng.run_training()
        if blobvision_cancel.is_requested():
            raise blobvision_cancel.AbortedError("Generation aborted")

        if not self.keep_models and mode != "redux":
            self._unload_sketch_pipe()

    def generate_video(
        self,
        prompt,
        video_path,
        iterations=None,
        denoise_fidelity=None,
        seed=None,
        negative_prompt=None,
        aspect=DEFAULT_ASPECT,
        frame_step=VIDEO_FRAME_STEP,
        on_progress=None,
        encode_from_sec=0.0,
        encode_to_sec=None,
        use_encode_range=False,
    ):
        width, height = resolve_aspect_size(aspect=aspect)
        iters = clamp_iterations(
            iterations if iterations is not None else DEFAULT_ITERATIONS["video"],
        )
        denoise = float(
            denoise_fidelity if denoise_fidelity is not None
            else DEFAULT_DENOISE.get("video", 0.3),
        )
        return _process_video(
            engine=self,
            video_path=video_path,
            prompt=prompt.strip(),
            width=width,
            height=height,
            iterations=iters,
            denoise_fidelity=denoise,
            seed=seed,
            negative_prompt=negative_prompt,
            frame_step=frame_step,
            on_progress=on_progress,
            encode_from_sec=float(encode_from_sec or 0.0),
            encode_to_sec=encode_to_sec,
            use_encode_range=bool(use_encode_range),
        )

    @staticmethod
    def read_image_metadata(path):
        return read_metadata_from_image(path)

    @staticmethod
    def seed_from_image(path):
        return seed_from_metadata(read_metadata_from_image(path))


@dataclass
class VideoResult:
    output_path: str
    work_dir: str
    processed_frames: int
    output_frames: int
    fps: float
    metadata: Dict[str, Any]


def _video_log(msg, on_progress=None):
    log_line(msg, on_progress)


def _bundled_codec_exe(subdir, names):
    base = os.path.join(VIDEO_CODECS_ROOT, subdir)
    for name in names:
        path = os.path.join(base, name)
        if os.path.isfile(path):
            return path
    return None


def _resolve_codec_tool(env_key, subdir, names, path_names):
    env = os.environ.get(env_key, "").strip()
    if env and os.path.isfile(env):
        return env
    bundled = _bundled_codec_exe(subdir, names)
    if bundled:
        return bundled
    import shutil
    for name in path_names:
        found = shutil.which(name)
        if found:
            return found
    return None


def video_codecs_status():
    ffmpeg = _resolve_codec_tool(
        "BLOBVISION_FFMPEG_BIN", "ffmpeg", ("ffmpeg.exe", "ffmpeg"), ("ffmpeg",),
    )
    ffprobe = _resolve_codec_tool(
        "BLOBVISION_FFPROBE_BIN", "ffmpeg", ("ffprobe.exe", "ffprobe"), ("ffprobe",),
    )
    rife = _resolve_codec_tool(
        "BLOBVISION_RIFE_BIN", "rife",
        ("rife-ncnn-vulkan.exe", "rife-ncnn-vulkan"),
        ("rife-ncnn-vulkan",),
    )
    bundled_ffmpeg = _bundled_codec_exe("ffmpeg", ("ffmpeg.exe", "ffmpeg"))
    bundled_ffprobe = _bundled_codec_exe("ffmpeg", ("ffprobe.exe", "ffprobe"))
    bundled_rife = _bundled_codec_exe("rife", ("rife-ncnn-vulkan.exe", "rife-ncnn-vulkan"))
    return {
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "rife": rife,
        "bundled_root": VIDEO_CODECS_ROOT,
        "ready": bool(ffmpeg and ffprobe),
        "bundled_ready": bool(bundled_ffmpeg and bundled_ffprobe),
        "bundled_rife": bool(bundled_rife),
    }


def _resolve_ffmpeg():
    path = _resolve_codec_tool(
        "BLOBVISION_FFMPEG_BIN", "ffmpeg", ("ffmpeg.exe", "ffmpeg"), ("ffmpeg",),
    )
    if not path:
        raise RuntimeError(
            "ffmpeg not found. Run Setup Video Codecs.bat in the repo root once "
            "(creates video-codecs/ffmpeg/), or install ffmpeg in PATH.",
        )
    return path


def _resolve_ffprobe():
    path = _resolve_codec_tool(
        "BLOBVISION_FFPROBE_BIN", "ffmpeg", ("ffprobe.exe", "ffprobe"), ("ffprobe",),
    )
    if not path:
        raise RuntimeError(
            "ffprobe not found. Run Setup Video Codecs.bat or install ffmpeg in PATH.",
        )
    return path


def _resolve_rife():
    rife = _resolve_codec_tool(
        "BLOBVISION_RIFE_BIN", "rife",
        ("rife-ncnn-vulkan.exe", "rife-ncnn-vulkan"),
        ("rife-ncnn-vulkan",),
    )
    if rife:
        _repair_rife_model_layout(rife)
    return rife


def setup_bundled_video_codecs(force=False):
    """Download ffmpeg + RIFE into video-codecs/ (Windows x64). Run once per machine."""
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    ffmpeg_dir = os.path.join(VIDEO_CODECS_ROOT, "ffmpeg")
    rife_dir = os.path.join(VIDEO_CODECS_ROOT, "rife")
    os.makedirs(ffmpeg_dir, exist_ok=True)
    os.makedirs(rife_dir, exist_ok=True)

    readme = os.path.join(VIDEO_CODECS_ROOT, "README.txt")
    if not os.path.isfile(readme):
        with open(readme, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(
                "BlobVision bundled video tools (auto-downloaded, not in git).\n"
                "ffmpeg/  — decode, encode, motion interpolation\n"
                "rife/    — optional AI frame interpolation (smoother than ffmpeg alone)\n"
                "Re-run Setup Video Codecs.bat to refresh.\n",
            )

    ffmpeg_ok = _bundled_codec_exe("ffmpeg", ("ffmpeg.exe", "ffmpeg"))
    ffprobe_ok = _bundled_codec_exe("ffmpeg", ("ffprobe.exe", "ffprobe"))
    rife_ok = _bundled_codec_exe("rife", ("rife-ncnn-vulkan.exe", "rife-ncnn-vulkan"))

    if sys.platform != "win32":
        print(
            "Setup Video Codecs: auto-download is Windows-only. "
            "Install ffmpeg + optional rife-ncnn-vulkan, or copy binaries into video-codecs/.",
            flush=True,
        )
        return video_codecs_status()

    FFMPEG_URL = (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
        "ffmpeg-master-latest-win64-gpl.zip"
    )
    RIFE_URL = (
        "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/"
        "rife-ncnn-vulkan-20221029-windows.zip"
    )

    def _download_zip(url, label):
        print("Downloading {}...".format(label), flush=True)
        tmp = tempfile.mkdtemp(prefix="blobcodecs_")
        zip_path = os.path.join(tmp, "pkg.zip")
        try:
            urllib.request.urlretrieve(url, zip_path)
            return zip_path, tmp
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    if force or not (ffmpeg_ok and ffprobe_ok):
        zip_path, tmp = _download_zip(FFMPEG_URL, "ffmpeg (~100 MB)")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    base = os.path.basename(member)
                    if base in ("ffmpeg.exe", "ffprobe.exe"):
                        dest = os.path.join(ffmpeg_dir, base)
                        with zf.open(member) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
            print("ffmpeg installed -> {}".format(ffmpeg_dir), flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("ffmpeg already present in video-codecs/ffmpeg/", flush=True)

    if force or not rife_ok:
        zip_path, tmp = _download_zip(RIFE_URL, "RIFE ncnn (~40 MB)")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    if member.endswith("/"):
                        continue
                    norm = member.replace("\\", "/")
                    parts = [p for p in norm.split("/") if p]
                    if not parts:
                        continue
                    if parts[0].lower().startswith("rife-ncnn") and len(parts) > 1:
                        rel_parts = parts[1:]
                    else:
                        rel_parts = parts
                    dest = os.path.join(rife_dir, *rel_parts)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
            rife_exe = os.path.join(rife_dir, "rife-ncnn-vulkan.exe")
            if os.path.isfile(rife_exe):
                _repair_rife_model_layout(rife_exe)
            print("RIFE installed -> {}".format(rife_dir), flush=True)
            model = _rife_pick_model(rife_exe)
            v4 = _rife_pick_v4_model(rife_exe)
            if model:
                print("RIFE model detected: {}".format(model), flush=True)
            else:
                print(
                    "WARNING: RIFE binary OK but no flownet.param in rife-v2.3/ etc. "
                    "Re-run Setup Video Codecs.bat with --force.",
                    flush=True,
                )
            if v4:
                print("RIFE v4 model available for 4x interpolation: {}".format(v4), flush=True)
            elif model:
                print("RIFE 4x will use chained 2x passes (no v4 model folder).", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("RIFE already present in video-codecs/rife/", flush=True)

    status = video_codecs_status()
    print("Video codecs ready: ffmpeg={}, rife={}".format(
        bool(status["ffmpeg"]), bool(status["rife"]),
    ), flush=True)
    return status


def _find_tool(name):
    if name == "ffmpeg":
        return _resolve_ffmpeg()
    if name == "ffprobe":
        return _resolve_ffprobe()
    import shutil
    path = shutil.which(name)
    if not path:
        raise RuntimeError("{} not found.".format(name))
    return path


def format_video_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return "{}h {:02d}m {:.1f}s".format(hours, minutes, secs)
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return "{}m {:.1f}s".format(minutes, secs)
    return "{:.1f}s".format(seconds)


def probe_video_file(video_path):
    return _probe_video(video_path)


def _probe_video(video_path):
    import json
    import subprocess
    ffprobe = _find_tool("ffprobe")
    raw = subprocess.check_output([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration", "-of", "json", video_path,
    ], text=True)
    data = json.loads(raw)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}

    def _rate(value):
        if not value or value == "0/0":
            return 0.0
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / (float(den) or 1.0)
        return float(value)

    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    fps = _rate(stream.get("avg_frame_rate"))
    if fps <= 0.0 or fps > 120.0:
        fps = _rate(stream.get("r_frame_rate"))
    nb_frames = stream.get("nb_frames")
    if nb_frames and duration > 0:
        computed = float(nb_frames) / duration
        if 5.0 < computed < 120.0:
            fps = computed
    if fps <= 0.0 or fps > 120.0:
        fps = 24.0
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        width, height = 1920, 1080
    frame_count = int(nb_frames) if nb_frames else 0
    return {
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def _extract_video_frames(
    video_path, out_dir, size, start_sec=0.0, duration_sec=None,
    output_fps=None, on_progress=None,
):
    """Extract frames from a clip; optional output_fps thins by time (not decode index)."""
    import subprocess

    os.makedirs(out_dir, exist_ok=True)
    for stale in os.listdir(out_dir):
        if stale.lower().endswith(".png"):
            os.remove(os.path.join(out_dir, stale))
    w, h = size
    pattern = os.path.join(out_dir, "src_%06d.png")
    scale_crop = "scale={}:{}:force_original_aspect_ratio=increase,crop={}:{}".format(w, h, w, h)
    if output_fps and float(output_fps) > 0.0:
        vf = "fps={:.6f},{}".format(float(output_fps), scale_crop)
        rate_note = " at {:.3f} fps".format(float(output_fps))
    else:
        vf = scale_crop
        rate_note = " (full decode)"
    cmd = [_find_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error"]
    if start_sec and float(start_sec) > 0:
        cmd.extend(["-ss", str(float(start_sec))])
    cmd.extend(["-i", video_path])
    if duration_sec is not None and float(duration_sec) > 0:
        cmd.extend(["-t", str(float(duration_sec))])
    cmd.extend(["-vf", vf, "-fps_mode", "vfr", pattern])
    _video_log("Video: extracting keyframes{} — {}".format(rate_note, " ".join(cmd)), on_progress)
    subprocess.check_call(cmd)
    frames = sorted(
        os.path.join(out_dir, n) for n in os.listdir(out_dir) if n.lower().endswith(".png")
    )
    if not frames:
        raise RuntimeError("No frames extracted from video.")
    return frames


def _rife_workdir(rife_exe):
    return os.path.dirname(os.path.abspath(rife_exe))


def _repair_rife_model_layout(rife_exe):
    """Older installs flattened *.param/*.bin next to the exe; ncnn expects rife-v2.3/."""
    import shutil

    model_dir = _rife_workdir(rife_exe)
    flat_param = os.path.join(model_dir, "flownet.param")
    target = os.path.join(model_dir, "rife-v2.3")
    if not os.path.isfile(flat_param):
        return False
    if os.path.isfile(os.path.join(target, "flownet.param")):
        return True
    os.makedirs(target, exist_ok=True)
    for name in os.listdir(model_dir):
        lower = name.lower()
        if not lower.endswith((".param", ".bin")):
            continue
        src = os.path.join(model_dir, name)
        dst = os.path.join(target, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.move(src, dst)
    return os.path.isfile(os.path.join(target, "flownet.param"))


def _rife_pick_model(rife_exe):
    model_dir = _rife_workdir(rife_exe)
    for name in ("rife-v4.6", "rife-v4", "rife-v2.3", "rife-v2"):
        sub = os.path.join(model_dir, name)
        if os.path.isfile(os.path.join(sub, "flownet.param")):
            return name
        if os.path.isfile(os.path.join(model_dir, name + ".param")):
            return name
    return None


def _rife_pick_v4_model(rife_exe):
    model_dir = _rife_workdir(rife_exe)
    for name in ("rife-v4.6", "rife-v4"):
        if os.path.isfile(os.path.join(model_dir, name, "flownet.param")):
            return name
    return None


def _list_rife_pngs(output_dir):
    frames = []
    for root, _dirs, files in os.walk(output_dir):
        for name in files:
            if name.lower().endswith(".png"):
                frames.append(os.path.join(root, name))
    return sorted(frames)


def _rife_target_frame_count(input_count, multiplier):
    n_in = max(2, int(input_count))
    mult = max(2, int(multiplier))
    n_between = mult - 1
    return n_in * mult - n_between


def _run_rife_pass(rife_exe, input_dir, output_dir, on_progress=None, model=None, target_frames=None):
    import subprocess
    import shutil

    os.makedirs(output_dir, exist_ok=True)
    for stale in os.listdir(output_dir):
        path = os.path.join(output_dir, stale)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif stale.lower().endswith(".png"):
            os.remove(path)
    cmd = [
        rife_exe,
        "-i", input_dir,
        "-o", output_dir,
        "-f", "out_%06d.png",
    ]
    if model is None:
        model = _rife_pick_model(rife_exe)
    if model:
        cmd.extend(["-m", model])
    if target_frames is not None:
        cmd.extend(["-n", str(int(target_frames))])
    _video_log("Video: RIFE running — " + " ".join(cmd), on_progress)
    proc = subprocess.run(
        cmd,
        cwd=_rife_workdir(rife_exe),
        capture_output=True,
        text=True,
    )
    tail = (proc.stderr or proc.stdout or "").strip()
    if tail:
        for line in tail.splitlines()[-3:]:
            _video_log("Video: RIFE: " + line.strip(), on_progress)
    frames = _list_rife_pngs(output_dir)
    if proc.returncode != 0 or not frames:
        return None
    _video_log("Video: RIFE produced {} frames.".format(len(frames)), on_progress)
    return frames


def _interpolate_video_frames_rife(input_dir, output_dir, multiplier, on_progress=None):
    import shutil
    import tempfile

    os.makedirs(output_dir, exist_ok=True)
    rife = _resolve_rife()
    if not rife:
        return None
    input_count = len([
        name for name in os.listdir(input_dir)
        if name.lower().endswith(".png")
    ])
    if input_count < 2:
        return None
    multiplier = max(2, int(multiplier))
    _video_log(
        "Video: RIFE AI interpolation x{} ({} keyframes in)...".format(
            multiplier, input_count,
        ),
        on_progress,
    )
    v4_model = _rife_pick_v4_model(rife)
    if v4_model and multiplier > 2:
        target = _rife_target_frame_count(input_count, multiplier)
        out = _run_rife_pass(
            rife, input_dir, output_dir, on_progress=on_progress,
            model=v4_model, target_frames=target,
        )
        if out:
            return out
        _video_log(
            "Video: RIFE v4 pass failed — trying chained 2x passes...",
            on_progress,
        )
    if multiplier <= 2:
        return _run_rife_pass(rife, input_dir, output_dir, on_progress=on_progress)
    passes = int(round(math.log2(multiplier))) if multiplier > 2 else 1
    passes = max(1, passes)
    current_in = input_dir
    temp_dirs = []
    result = None
    try:
        for pass_idx in range(passes):
            if pass_idx == passes - 1:
                out_dir = output_dir
            else:
                out_dir = tempfile.mkdtemp(prefix="rife_pass_", dir=os.path.dirname(output_dir))
                temp_dirs.append(out_dir)
            _video_log(
                "Video: RIFE 2x pass {}/{}...".format(pass_idx + 1, passes),
                on_progress,
            )
            result = _run_rife_pass(rife, current_in, out_dir, on_progress=on_progress)
            if not result:
                return None
            current_in = out_dir
        return result
    finally:
        for temp_dir in temp_dirs:
            if temp_dir != output_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)


def _assemble_video_interpolated(
    proc_dir, output_path, keyframe_count, target_frame_count, clip_duration, on_progress=None,
):
    import subprocess

    ffmpeg = _resolve_ffmpeg()
    clip_duration = max(0.05, float(clip_duration))
    keyframe_count = max(1, int(keyframe_count))
    target_frame_count = max(keyframe_count + 1, int(target_frame_count))
    sparse_fps = keyframe_count / clip_duration
    out_fps = target_frame_count / clip_duration
    pattern = os.path.join(proc_dir, "proc_%06d.png")
    vf = "minterpolate=fps={:.3f}:mi_mode=mci:mc_mode=aobmc".format(out_fps)
    _video_log(
        "Video: ffmpeg motion interpolation {} keyframes -> {} frames ({:.1f} -> {:.1f} fps)...".format(
            keyframe_count, target_frame_count, sparse_fps, out_fps,
        ),
        on_progress,
    )
    subprocess.check_call([
        ffmpeg, "-y",
        "-framerate", str(sparse_fps),
        "-i", pattern,
        "-vf", vf,
        "-frames:v", str(target_frame_count),
        "-pix_fmt", "yuv420p",
        output_path,
    ])


def _interpolate_to_target_count(input_dir, output_dir, target_frames, on_progress=None):
    import shutil

    existing = sorted(
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.lower().endswith(".png")
    )
    in_count = len(existing)
    target = max(2, int(target_frames))
    if in_count < 2:
        return None
    if in_count >= target:
        os.makedirs(output_dir, exist_ok=True)
        for idx, src in enumerate(existing[:target]):
            shutil.copy2(src, os.path.join(output_dir, "out_{:06d}.png".format(idx + 1)))
        return existing[:target]
    multiplier = max(2, int((target + in_count - 2) // max(1, in_count - 1)))
    _video_log(
        "Video: RIFE stretch {} keyframes -> {} frames...".format(in_count, target),
        on_progress,
    )
    rife_out = _interpolate_video_frames_rife(
        input_dir, output_dir, multiplier, on_progress=on_progress,
    )
    if not rife_out:
        return None
    trimmed = sorted(_list_rife_pngs(output_dir))[:target]
    os.makedirs(output_dir, exist_ok=True)
    for idx, src in enumerate(trimmed):
        dest = os.path.join(output_dir, "out_{:06d}.png".format(idx + 1))
        if os.path.abspath(src) != os.path.abspath(dest):
            shutil.copy2(src, dest)
    return trimmed


def _interpolate_video_frames(input_dir, output_dir, multiplier, fps, on_progress=None, target_frames=None):
    if target_frames is not None:
        rife_out = _interpolate_to_target_count(
            input_dir, output_dir, target_frames, on_progress=on_progress,
        )
        if rife_out:
            return rife_out
    else:
        rife_out = _interpolate_video_frames_rife(input_dir, output_dir, multiplier, on_progress)
        if rife_out:
            return rife_out
    _video_log(
        "Video: RIFE unavailable or failed — using ffmpeg minterpolate instead.",
        on_progress,
    )
    return None


def _assemble_video(frame_dir, output_path, fps, frame_glob="out_%06d.png"):
    import subprocess
    subprocess.check_call([
        _find_tool("ffmpeg"), "-y", "-framerate", str(float(fps)),
        "-i", os.path.join(frame_dir, frame_glob),
        "-pix_fmt", "yuv420p", output_path,
    ])


def _mux_video_audio(source_video, silent_video, output_path, clip_start=0.0, clip_duration=None):
    import subprocess
    cmd = [
        _find_tool("ffmpeg"), "-y",
        "-i", silent_video,
    ]
    if clip_start and float(clip_start) > 0:
        cmd.extend(["-ss", str(float(clip_start))])
    cmd.extend(["-i", source_video])
    if clip_duration is not None and float(clip_duration) > 0:
        cmd.extend(["-t", str(float(clip_duration))])
    cmd.extend([
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_path,
    ])
    subprocess.check_call(cmd)


def _process_video(
    engine, video_path, prompt, width, height, iterations,
    denoise_fidelity, seed=None, negative_prompt=None, frame_step=4, on_progress=None,
    encode_from_sec=0.0, encode_to_sec=None, use_encode_range=False,
):
    import blobvision_cancel
    import shutil

    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise RuntimeError("Video file not found: " + video_path)
    codecs = video_codecs_status()
    if not codecs["ready"]:
        raise RuntimeError(
            "Video codecs missing. Click Install codecs in BlobVision or run Setup Video Codecs.bat.",
        )
    info = _probe_video(video_path)
    duration = float(info["duration"])
    if use_encode_range:
        encode_from = max(0.0, float(encode_from_sec or 0.0))
        encode_to = float(encode_to_sec if encode_to_sec is not None else duration)
        encode_to = min(duration, max(encode_from + 0.05, encode_to))
    else:
        encode_from = 0.0
        encode_to = duration
    clip_duration = encode_to - encode_from
    if clip_duration <= 0:
        raise RuntimeError("Invalid encode range: end must be after start.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(engine.output_dir, "video_{}".format(stamp))
    src_dir = os.path.join(work_dir, "src")
    proc_dir = os.path.join(work_dir, "processed")
    interp_dir = os.path.join(work_dir, "interpolated")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    frame_step = max(1, int(frame_step))
    step_label = "all frames" if frame_step == 1 else "1 every {} frames".format(frame_step)
    source_fps = float(info["fps"])
    decoded_fps = None
    if info.get("frame_count") and clip_duration > 0:
        decoded_fps = float(info["frame_count"]) / clip_duration
    effective_source_fps = source_fps
    if decoded_fps and decoded_fps > source_fps + 1.0:
        effective_source_fps = decoded_fps
    working_fps = _normalize_working_fps(effective_source_fps)
    fps_normalized = working_fps < effective_source_fps - 0.5
    working_video = video_path
    extract_from = encode_from
    if fps_normalized:
        norm_path = os.path.join(work_dir, "normalized.mp4")
        _normalize_video_clip(
            video_path, norm_path, encode_from, clip_duration, working_fps,
            on_progress=on_progress,
        )
        working_video = norm_path
        extract_from = 0.0
        _video_log(
            "Video: input {:.1f} fps -> {:.1f} fps before extract (24/25/30 max).".format(
                effective_source_fps, working_fps,
            ),
            on_progress,
        )
    _video_log(
        "Video: {:.1f}s -> {:.1f}s, {} at {}x{} (stride={})...".format(
            encode_from, encode_to, step_label, width, height, frame_step,
        ),
        on_progress,
    )
    native_frames = _extract_video_frames(
        working_video, src_dir, (width, height),
        start_sec=extract_from, duration_sec=clip_duration, on_progress=on_progress,
    )
    native_count = len(native_frames)
    timeline_fps = native_count / clip_duration if clip_duration > 0 else float(info["fps"])
    target_output_frames = native_count
    keyframes = _subsample_frame_paths(native_frames, frame_step)
    _video_log(
        "Video: {} frames in clip -> {} to blobify (VQGAN), then RIFE/ffmpeg -> {} frames.".format(
            native_count, len(keyframes), target_output_frames,
        ),
        on_progress,
    )
    if frame_step > 1 and len(keyframes) >= native_count:
        raise RuntimeError(
            "Frame thinning failed: stride={} but all {} frames would be blobified.".format(
                frame_step, native_count,
            ),
        )
    for index, frame_path in enumerate(keyframes):
        if blobvision_cancel.is_requested():
            raise blobvision_cancel.AbortedError("Video processing aborted")
        _video_log("Video: blobify {}/{}...".format(index + 1, len(keyframes)), on_progress)
        out_path = os.path.join(proc_dir, "proc_{:06d}.png".format(index))
        result = engine.generate(
            mode="corrupt",
            prompt=prompt,
            negative_prompt=negative_prompt,
            iterations=iterations,
            denoise_fidelity=denoise_fidelity,
            seed=seed,
            init_image_path=frame_path,
            width=width,
            height=height,
        )
        shutil.copy2(result.output_path, out_path)
    silent_path = os.path.join(work_dir, "silent.mp4")
    if frame_step <= 1:
        _video_log(
            "Video: all frames blobified — assembling at {:.2f} fps (no interpolation).".format(
                timeline_fps,
            ),
            on_progress,
        )
        _assemble_video(proc_dir, silent_path, timeline_fps, frame_glob="proc_%06d.png")
        interp_frames = keyframes
    else:
        interp_frames = _interpolate_video_frames(
            proc_dir, interp_dir, frame_step, timeline_fps,
            on_progress=on_progress, target_frames=target_output_frames,
        )
        if interp_frames:
            _video_log(
                "Video: assembling {} interpolated frames at {:.2f} fps...".format(
                    len(interp_frames), timeline_fps,
                ),
                on_progress,
            )
            _assemble_video(interp_dir, silent_path, timeline_fps)
        else:
            _assemble_video_interpolated(
                proc_dir, silent_path, len(keyframes), target_output_frames, clip_duration,
                on_progress=on_progress,
            )
            interp_frames = []
    final_path = os.path.join(engine.output_dir, "video_{}.mp4".format(stamp))
    _mux_video_audio(
        video_path, silent_path, final_path,
        clip_start=encode_from, clip_duration=clip_duration,
    )
    meta = {
        "mode": "video",
        "prompt": prompt,
        "source_video": video_path,
        "output_video": final_path,
        "work_dir": work_dir,
        "fps": timeline_fps,
        "source_fps": source_fps,
        "working_fps": working_fps,
        "fps_normalized": fps_normalized,
        "native_frames": native_count,
        "duration": duration,
        "encode_from_sec": encode_from,
        "encode_to_sec": encode_to,
        "use_encode_range": bool(use_encode_range),
        "frame_step": frame_step,
        "processed_frames": len(keyframes),
        "output_frames": len(interp_frames) if interp_frames else target_output_frames,
        "size": [width, height],
        "iterations": iterations,
        "denoise_fidelity": denoise_fidelity,
    }
    _video_log("Video: done — " + final_path, on_progress)
    return VideoResult(
        output_path=final_path,
        work_dir=work_dir,
        processed_frames=len(keyframes),
        output_frames=len(interp_frames) if interp_frames else target_output_frames,
        fps=timeline_fps,
        metadata=meta,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("setup-codecs", "setup-video-codecs"):
        setup_bundled_video_codecs(force="--force" in sys.argv)
    else:
        print("BlobVision engine utility.", flush=True)
        print("  setup-codecs [--force]  Download ffmpeg + RIFE into video-codecs/", flush=True)
