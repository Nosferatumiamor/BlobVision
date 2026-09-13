"""Family-agnostic video pipeline: extract frames from a source clip, run
each keyframe through a caller-supplied per-frame transform, RIFE/ffmpeg-
interpolate the gaps back to full framerate, and mux the result with the
original audio. Originally lived inside blobvision_engine.py hardwired to
VQGAN's "corrupt" mode; extracted here once DeepDream's img2img mode needed
to drive the exact same pipeline (see BlobVisionEngine.generate_video() and
DeepDreamEngine.generate_video() for the two callers, each supplying their
own blobify_frame(src_path, dst_path) callback — that one callback is the
only family-specific piece anywhere in this file).
"""
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict

from blobvision_paths import VIDEO_CODECS_ROOT, build_output_name, outputs_dir, work_dir

# The engine runs under pythonw.exe (no console of its own — see main.rs's
# spawn_python_engine), so without this, Windows allocates and flashes a
# brand new visible console for every ffmpeg/ffprobe/RIFE child process
# spawned below — one per probe/extract/interpolate/encode call, several
# times per video job.
_NO_WINDOW_KWARGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

# How long a source clip can be before generate_video() callers should ask
# for confirmation before starting (mirrors the old Gradio UI's behavior) —
# generic across families, not a VQGAN-specific number.
VIDEO_LONG_WARN_SECONDS = 30
# Default keyframe stride (1 = every frame, higher = more thinning, RIFE/
# ffmpeg fills the gap) — a starting point any family's generate_video() can
# default to; not tied to VQGAN specifically despite living here first.
VIDEO_FRAME_STEP = 4
# Cap on the DECODED source framerate before extraction — clips faster than
# this get re-encoded down first (see _normalize_working_fps) so extraction
# doesn't have to deal with 60/96/120fps sources.
VIDEO_MAX_SOURCE_FPS = 30.0


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
    """Cap input cadence to 24/25/30 before processing when the container runs faster."""
    fps = float(source_fps)
    if fps <= VIDEO_MAX_SOURCE_FPS:
        return fps
    if fps <= 60.0:
        return 30.0
    return _resolve_timeline_fps(fps)


def _normalize_video_clip(
    video_path, out_path, start_sec, duration_sec, target_fps, on_progress=None,
):
    """Re-encode a clip segment at a lower constant fps (silent proxy for frame work)."""
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
    subprocess.check_call(cmd, **_NO_WINDOW_KWARGS)
    if not os.path.isfile(out_path):
        raise RuntimeError("Video normalization failed: " + out_path)
    return out_path


@dataclass
class VideoResult:
    output_path: str
    work_dir: str
    processed_frames: int
    output_frames: int
    fps: float
    metadata: Dict[str, Any]


def _video_log(msg, on_progress=None):
    """Print once to console/UI; mirrors blobvision_engine.py's log_line()
    (kept as an inline duplicate, not an import, so this module has no
    dependency back on blobvision_engine — it imports _process_video etc.
    FROM here, so the reverse would be circular)."""
    msg = (msg or "").strip()
    if not msg:
        return
    try:
        print(msg, flush=True)
    except OSError:
        pass
    if on_progress is not None:
        on_progress(msg)


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
    ffprobe = _find_tool("ffprobe")
    raw = subprocess.check_output([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration", "-of", "json", video_path,
    ], text=True, **_NO_WINDOW_KWARGS)
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
    subprocess.check_call(cmd, **_NO_WINDOW_KWARGS)
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
        **_NO_WINDOW_KWARGS,
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
    ], **_NO_WINDOW_KWARGS)


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
    subprocess.check_call([
        _find_tool("ffmpeg"), "-y", "-framerate", str(float(fps)),
        "-i", os.path.join(frame_dir, frame_glob),
        "-pix_fmt", "yuv420p", output_path,
    ], **_NO_WINDOW_KWARGS)


def _mux_video_audio(source_video, silent_video, output_path, clip_start=0.0, clip_duration=None):
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
    subprocess.check_call(cmd, **_NO_WINDOW_KWARGS)


def _process_video(
    video_path, blobify_frame, width, height, type_tag, basename_source,
    frame_step=4, on_progress=None,
    encode_from_sec=0.0, encode_to_sec=None, use_encode_range=False,
    extra_meta=None,
):
    """Family-agnostic video pipeline: extract frames, keep every Nth as a
    keyframe, run each through `blobify_frame(src_path, dst_path)` (the
    ONLY family-specific step — the caller decides what that does: VQGAN
    img2img today, DeepDream/Style Transfer img2img would work exactly the
    same way), then RIFE/ffmpeg-interpolate the gaps back to full framerate
    and mux the original audio back on. `extra_meta` lets the caller add
    its own family-specific fields (prompt, iterations, ...) to the result
    metadata without this function needing to know what they are.

    `type_tag`/`basename_source` name the final muxed file (see
    blobvision_paths.build_output_name) — e.g. "VV"/the source video's
    basename. All per-job scratch (extracted/processed/interpolated frames)
    lives under work_dir()/job_<uuid>/, entirely outside outputs/, and is
    removed once the final file is in place (or the job fails/is
    cancelled) — see the try/finally below.
    """
    import shutil
    import uuid

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

    job_dir = os.path.join(work_dir(), "job_" + uuid.uuid4().hex)
    src_dir = os.path.join(job_dir, "src")
    proc_dir = os.path.join(job_dir, "processed")
    interp_dir = os.path.join(job_dir, "interpolated")
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    try:
        return _run_video_job(
            video_path, blobify_frame, width, height, type_tag, basename_source,
            frame_step, on_progress, encode_from, encode_to, clip_duration,
            use_encode_range, info, job_dir, src_dir, proc_dir, interp_dir, extra_meta,
        )
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def _run_video_job(
    video_path, blobify_frame, width, height, type_tag, basename_source, frame_step,
    on_progress, encode_from, encode_to, clip_duration, use_encode_range, info,
    job_dir, src_dir, proc_dir, interp_dir, extra_meta,
):
    """The pipeline body, split out from _process_video() only so that
    function's try/finally cleanly wraps every exit path (success, error,
    or blobvision_cancel.AbortedError) around the job_dir cleanup."""
    import blobvision_cancel
    import shutil

    # Same fixed-filename live-preview trick as the still-image path (see
    # blobvision_engine.py's _generate_sketch/generate() and generate.py's
    # checkin()) — the frontend polls this by a known URL while the job
    # runs. Cleared up front so a poll landing before the first keyframe is
    # blobified gets a clean miss instead of the previous job's last frame.
    live_preview_path = os.path.join(outputs_dir(), "_live_preview_video.png")
    try:
        os.remove(live_preview_path)
    except OSError:
        pass

    duration = float(info["duration"])
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
        norm_path = os.path.join(job_dir, "normalized.mp4")
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
        "Video: {} frames in clip -> {} to blobify, then RIFE/ffmpeg -> {} frames.".format(
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
        blobify_frame(frame_path, out_path)
        # Best-effort, cosmetic only — must never be the reason a real
        # video job fails (e.g. a transient file-lock while the frontend's
        # own poll is mid-read of the same path).
        try:
            shutil.copyfile(out_path, live_preview_path)
        except OSError:
            pass
    silent_path = os.path.join(job_dir, "silent.mp4")
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
    final_path = os.path.join(outputs_dir(), build_output_name(type_tag, basename_source, ".mp4"))
    _mux_video_audio(
        video_path, silent_path, final_path,
        clip_start=encode_from, clip_duration=clip_duration,
    )
    meta = {
        "source_video": video_path,
        "output_video": final_path,
        "work_dir": job_dir,
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
    }
    meta.update(extra_meta or {})
    _video_log("Video: done — " + final_path, on_progress)
    return VideoResult(
        output_path=final_path,
        work_dir=job_dir,
        processed_frames=len(keyframes),
        output_frames=len(interp_frames) if interp_frames else target_output_frames,
        fps=timeline_fps,
        metadata=meta,
    )

