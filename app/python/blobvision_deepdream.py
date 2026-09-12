"""DeepDream generation engine for BlobVision (GoogLeNet / Inception v1).

Uses torchvision's GoogLeNet rather than Inception v3: it's architecturally the same
family (Inception v1) as the Caffe BVLC GoogLeNet the original Google DeepDream blog
post ran on, so the same layer depths reproduce the classic eyes/dogs/birds look —
Inception v3's deeper, factorized-convolution filters don't hallucinate the same
iconography even at "equivalent" layer names.
"""
import os
import random
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import blobvision_cancel
from blobvision_paths import basename_hint_from_upload, build_output_name, ensure_layout, outputs_dir

ensure_layout()
DEFAULT_OUTPUT = outputs_dir()

DEFAULT_STEPS = 40
DEFAULT_OCTAVES = 4
DEFAULT_LAYER = "mixed6a"
OCTAVE_SCALE = 1.4
STEP_SIZE = 0.025
STEP_SIZE_IMG2IMG = 0.03
STEP_SIZE_REDUX = 0.016
PRESERVE_IMG2IMG = 0.45
PRESERVE_REDUX = 0.45
GRAD_BLUR_RADIUS = 0

# Random pixel-shift applied before each gradient step and undone after (the classic
# DeepDream "jitter" trick). Without it, the conv-net's receptive-field grid stays
# perfectly aligned with the image on every step, which shows up as a regular tiled
# grid pattern instead of organic hallucinated detail. Sized relative to the current
# octave's resolution so it scales from small to large octaves.
JITTER_FRACTION = 0.06
JITTER_MIN_PX = 4
JITTER_MAX_PX = 32

# Floor for the smallest octave in the pyramid (see _run_octaves) — stops shrinking well
# before the image degenerates into a meaningless handful of pixels.
MIN_OCTAVE_PX = 200

# Keys are the stable identifiers the UI radio button uses (unchanged across the
# InceptionV3 -> GoogLeNet swap); values are GoogLeNet module names. inception4c is the
# single most iconic DeepDream layer (the classic dog-slug/eyes look from the original
# blog post); 3b/5a bracket it for a more textural vs. more abstract/whole-object feel.
INCEPTION_LAYERS = {
    "mixed5b": "inception3b",
    "mixed6a": "inception4c",
    "mixed7": "inception5a",
}

# Single-dial intensity control (0..100) for img2img/redux. Steps, octaves, and preserve
# all need to move together to go from "barely touched" to "fully hallucinated" — tuning
# them as 3 separate sliders makes it easy to land on combinations that look like noise
# (e.g. many octaves + low preserve) or do nothing (few steps + high preserve) without any
# visual cue why. These anchors are hand-picked from visual comparison, not derived:
# 50 is the validated "recognizable source + a few strong creatures" sweet spot.
INTENSITY_ANCHORS = {
    0.0: {"steps": 10, "octaves": 1, "preserve": 0.70},
    0.5: {"steps": 16, "octaves": 2, "preserve": 0.45},
    1.0: {"steps": 24, "octaves": 3, "preserve": 0.00},
}


def intensity_to_params(intensity):
    """Map a 0..100 'how hard to dream' dial to (steps, octaves, preserve)."""
    t = max(0.0, min(1.0, float(intensity) / 100.0))
    lo, hi = (0.0, 0.5) if t <= 0.5 else (0.5, 1.0)
    local = 0.0 if hi == lo else (t - lo) / (hi - lo)
    a, b = INTENSITY_ANCHORS[lo], INTENSITY_ANCHORS[hi]
    steps = round(a["steps"] + local * (b["steps"] - a["steps"]))
    preserve = round(a["preserve"] + local * (b["preserve"] - a["preserve"]), 2)
    octaves = max(1, min(3, round(1 + t * 2)))
    return int(steps), int(octaves), float(preserve)


@dataclass
class DeepDreamResult:
    output_path: str
    seed: int
    metadata: Dict[str, Any]


class DeepDreamEngine:
    def __init__(self, output_dir=None, device=None):
        import torch

        self.output_dir = os.path.abspath(output_dir or DEFAULT_OUTPUT)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._lock = threading.Lock()
        self._model = None
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self):
        if self._model is not None:
            return self._model
        from torchvision import models

        weights = models.GoogLeNet_Weights.IMAGENET1K_V1
        model = models.googlenet(weights=weights)
        model.aux_logits = False
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self._model = model.to(self.device)
        return self._model

    def _layer_module(self, layer_key):
        target = INCEPTION_LAYERS.get(layer_key, INCEPTION_LAYERS[DEFAULT_LAYER])
        model = self._load_model()
        for name, module in model.named_modules():
            if name.endswith(target):
                return module
        raise ValueError("Unknown Inception layer: " + str(layer_key))

    def _init_tensor(self, width, height, init_image_path, seed):
        import torch
        from PIL import Image

        if init_image_path and os.path.isfile(init_image_path):
            img = Image.open(init_image_path).convert("RGB").resize((width, height), Image.LANCZOS)
            arr = torch.tensor(list(img.getdata()), dtype=torch.float32, device=self.device)
            arr = arr.view(height, width, 3).permute(2, 0, 1).unsqueeze(0) / 255.0
            return arr
        torch.manual_seed(int(seed))
        return torch.rand(1, 3, height, width, device=self.device)

    def _preprocess(self, tensor):
        return (tensor - self._mean) / self._std

    def _tensor_to_pil(self, tensor):
        from PIL import Image

        arr = tensor.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        arr = (arr * 255.0).astype("uint8")
        return Image.fromarray(arr, mode="RGB")

    def _dream_loss(self, acts, use_mean_loss=False):
        if use_mean_loss:
            return acts.mean()
        return acts.norm()

    def _blur_grad(self, grad, radius=GRAD_BLUR_RADIUS):
        if radius <= 0:
            return grad
        import torch
        import torch.nn.functional as F

        channels = grad.shape[1]
        kernel_size = 2 * int(radius) + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=grad.device, dtype=grad.dtype)
        kernel = kernel / kernel.numel()
        kernel = kernel.repeat(channels, 1, 1, 1)
        return F.conv2d(grad, kernel, padding=int(radius), groups=channels)

    def _dream_once(
        self, image, layer_module, steps, on_progress=None, step_size=None,
        use_mean_loss=False, grad_blur=0, jitter=0,
    ):
        import torch

        model = self._load_model()
        img = image.detach().clone()
        captured = {}

        def hook(_module, _inputs, output):
            captured["acts"] = output

        handle = layer_module.register_forward_hook(hook)
        try:
            for step in range(int(steps)):
                if blobvision_cancel.is_requested():
                    raise blobvision_cancel.AbortedError("Generation aborted")
                shift_h = shift_w = 0
                if jitter > 0:
                    shift_h = random.randint(-int(jitter), int(jitter))
                    shift_w = random.randint(-int(jitter), int(jitter))
                    img = torch.roll(img, shifts=(shift_h, shift_w), dims=(2, 3))
                img = img.detach().requires_grad_(True)
                model.zero_grad(set_to_none=True)
                out = model(self._preprocess(img))
                if hasattr(out, "logits"):
                    _ = out.logits
                acts = captured.get("acts")
                if acts is None:
                    raise RuntimeError("Inception hook did not capture activations.")
                step_sz = step_size if step_size is not None else STEP_SIZE
                loss = self._dream_loss(acts, use_mean_loss=use_mean_loss)
                loss.backward()
                grad = img.grad.detach()
                grad = grad / (grad.std() + 1e-8)
                if grad_blur > 0:
                    grad = self._blur_grad(grad, radius=grad_blur)
                with torch.no_grad():
                    img = img + step_sz * grad
                    img.clamp_(0.0, 1.0)
                if shift_h or shift_w:
                    img = torch.roll(img, shifts=(-shift_h, -shift_w), dims=(2, 3))
                img = img.detach()
                if on_progress and (step == 0 or step == int(steps) - 1 or step % max(1, int(steps) // 4) == 0):
                    on_progress("DeepDream step {}/{}".format(step + 1, int(steps)))
        finally:
            handle.remove()
        return img.detach()

    def _run_octaves(
        self, image, layer_module, steps, octaves, on_progress=None, step_size=None,
        use_mean_loss=False, grad_blur=0, jitter=True,
    ):
        import torch
        import torch.nn.functional as F

        target_h, target_w = image.shape[2], image.shape[3]

        # Build octave pyramid going DOWN from the requested/target resolution — this is
        # what the original Google DeepDream actually does (shrink the input to get
        # smaller octaves, dream smallest-to-largest, finish at the original size).
        # Growing UP from the target instead (the previous approach here) means the
        # network's fixed-pixel receptive field covers a shrinking fraction of the canvas
        # as target resolution increases, so the same settings produce visibly weaker,
        # finer-grained hallucination at higher output sizes. Shrinking down keeps the
        # coarsest working scale in roughly the same absolute pixel range regardless of
        # the requested output size, so large features (muzzles, eyes) stay proportionally
        # consistent instead of diluting at high resolution.
        octave_bases = [image.clone()]
        for i in range(int(octaves) - 1):
            prev = octave_bases[-1]
            new_h = max(MIN_OCTAVE_PX, int(prev.shape[2] / OCTAVE_SCALE))
            new_w = max(MIN_OCTAVE_PX, int(prev.shape[3] / OCTAVE_SCALE))
            if new_h >= prev.shape[2] and new_w >= prev.shape[3]:
                break
            down = F.interpolate(
                prev, size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            )
            octave_bases.append(down)

        actual_octaves = len(octave_bases)

        # Start with zero detail at the smallest scale
        detail = torch.zeros_like(octave_bases[-1])

        # Process from SMALLEST to LARGEST (coarse features first, then refine)
        dreamed = None
        for octave in range(actual_octaves):
            idx = actual_octaves - 1 - octave
            base = octave_bases[idx]

            if octave > 0:
                detail = F.interpolate(
                    detail, size=base.shape[2:],
                    mode="bilinear", align_corners=False,
                )

            if on_progress:
                on_progress("Octave {}/{} ({}x{})".format(
                    octave + 1, actual_octaves, base.shape[3], base.shape[2]))

            octave_jitter = 0
            if jitter:
                octave_jitter = int(min(base.shape[2], base.shape[3]) * JITTER_FRACTION)
                octave_jitter = max(JITTER_MIN_PX, min(JITTER_MAX_PX, octave_jitter))

            dreamed = self._dream_once(
                base + detail, layer_module, steps, on_progress=on_progress,
                step_size=step_size, use_mean_loss=use_mean_loss, grad_blur=grad_blur,
                jitter=octave_jitter,
            )
            detail = dreamed - base

        # detail is now at target scale — result = original + accumulated multi-scale detail
        if detail.shape[2:] != (target_h, target_w):
            detail = F.interpolate(
                detail, size=(target_h, target_w),
                mode="bilinear", align_corners=False,
            )
        return image + detail

    def generate(
        self,
        mode="txt2img",
        width=512,
        height=512,
        steps=DEFAULT_STEPS,
        octaves=DEFAULT_OCTAVES,
        layer=DEFAULT_LAYER,
        init_image_path=None,
        seed=None,
        prompt="",
        negative_prompt="",
        preserve=None,
        on_progress=None,
    ):
        if mode not in ("txt2img", "img2img", "redux"):
            raise ValueError("mode must be txt2img, img2img, or redux")
        if mode in ("img2img", "redux") and not init_image_path:
            raise ValueError("img2img/redux requires init_image_path")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        if on_progress:
            on_progress("Loading GoogLeNet...")
        layer_module = self._layer_module(layer)
        if mode == "redux":
            base_step = STEP_SIZE_REDUX
            if preserve is None:
                preserve = PRESERVE_REDUX
            use_mean_loss = False
            grad_blur = 0
        elif mode == "img2img":
            base_step = STEP_SIZE_IMG2IMG
            if preserve is None:
                preserve = PRESERVE_IMG2IMG
            use_mean_loss = False
            grad_blur = GRAD_BLUR_RADIUS
        else:
            base_step = STEP_SIZE
            preserve = 0.0
            use_mean_loss = False
            grad_blur = GRAD_BLUR_RADIUS
        if mode == "img2img":
            step_size = base_step
        else:
            wild = max(0.0, 1.0 - float(preserve)) if preserve is not None else 1.0
            step_size = base_step * (1.0 + wild * 1.5)
        init_path = init_image_path if mode in ("img2img", "redux") else None
        image = self._init_tensor(int(width), int(height), init_path, int(seed))
        if on_progress:
            on_progress(
                "DeepDream {} {}x{} (layer={}, steps={}, octaves={})".format(
                    mode, int(width), int(height), layer, int(steps), int(octaves),
                )
            )
        blobvision_cancel.clear()
        with self._lock:
            result_tensor = self._run_octaves(
                image, layer_module, int(steps), int(octaves), on_progress=on_progress,
                step_size=step_size, use_mean_loss=use_mean_loss, grad_blur=grad_blur,
            )
            if init_path and preserve > 0:
                import torch.nn.functional as F
                original = self._init_tensor(int(width), int(height), init_path, int(seed))
                if original.shape[2:] != result_tensor.shape[2:]:
                    original = F.interpolate(
                        original, size=result_tensor.shape[2:], mode="bilinear", align_corners=False,
                    )
                result_tensor = result_tensor * (1.0 - preserve) + original * preserve
            # img2img's init image is a real user upload — name after it like
            # VQGAN's corrupt mode does. redux's init image is an internally
            # chained SDXL sketch (not an upload, no embedded basename hint
            # to recover), so it's named after the prompt instead, same as
            # txt2img.
            if mode == "img2img" and init_path:
                basename_source = basename_hint_from_upload(init_path)
            else:
                basename_source = prompt
            output_path = os.path.join(self.output_dir, build_output_name("D", basename_source, ".png"))
            self._tensor_to_pil(result_tensor).save(output_path)
        meta = {
            "family": "deepdream",
            "mode": mode,
            "preserve": float(preserve) if preserve is not None else 0.0,
            "step_size": round(float(step_size), 5),
            "prompt": prompt or "",
            "negative_prompt": negative_prompt or "",
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "octaves": int(octaves),
            "layer": layer,
            "seed": int(seed),
            "output": os.path.basename(output_path),
        }
        if on_progress:
            on_progress("Saved " + os.path.basename(output_path))
        return DeepDreamResult(output_path=output_path, seed=int(seed), metadata=meta)

    def generate_video(
        self,
        video_path,
        width,
        height,
        intensity=50,
        layer=DEFAULT_LAYER,
        seed=None,
        frame_step=None,
        on_progress=None,
        encode_from_sec=0.0,
        encode_to_sec=None,
        use_encode_range=False,
    ):
        """Drives the same family-agnostic pipeline VQGAN's generate_video()
        uses (see blobvision_video.py), just with DeepDream's own img2img
        mode as the per-keyframe transform instead of VQGAN's "corrupt"
        mode — everything else (extraction, RIFE, reassembly, muxing) is
        identical between the two."""
        import shutil

        from blobvision_video import VIDEO_FRAME_STEP, _process_video

        steps, octaves, preserve = intensity_to_params(intensity)
        if frame_step is None:
            frame_step = VIDEO_FRAME_STEP

        def blobify_frame(src_path, dst_path):
            result = self.generate(
                mode="img2img",
                init_image_path=src_path,
                width=width,
                height=height,
                steps=steps,
                octaves=octaves,
                layer=layer,
                seed=seed,
                preserve=preserve,
            )
            shutil.copy2(result.output_path, dst_path)
            # The per-keyframe still is a real, fully-named "D" file — don't
            # let it permanently litter outputs_dir(); only the muxed final
            # video (built by _process_video below) is meant to persist.
            os.remove(result.output_path)

        return _process_video(
            video_path=video_path,
            blobify_frame=blobify_frame,
            width=width,
            height=height,
            frame_step=frame_step,
            on_progress=on_progress,
            encode_from_sec=float(encode_from_sec or 0.0),
            encode_to_sec=encode_to_sec,
            use_encode_range=bool(use_encode_range),
            type_tag="DV",
            basename_source=basename_hint_from_upload(video_path),
            extra_meta={
                "mode": "video",
                "layer": layer,
                "steps": steps,
                "octaves": octaves,
                "intensity": intensity,
            },
        )
