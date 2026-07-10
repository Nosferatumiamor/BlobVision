"""DeepDream generation engine for BlobVision (InceptionV3)."""
import os
import random
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from blobvision_paths import deepdream_output_dir, ensure_layout

ensure_layout()
DEFAULT_OUTPUT = deepdream_output_dir()

DEFAULT_STEPS = 40
DEFAULT_OCTAVES = 4
DEFAULT_LAYER = "mixed6a"
OCTAVE_SCALE = 1.4
STEP_SIZE = 0.01
STEP_SIZE_IMG2IMG = 0.009
STEP_SIZE_REDUX = 0.012
PRESERVE_IMG2IMG = 0.45
PRESERVE_REDUX = 0.45
GRAD_BLUR_RADIUS = 0

INCEPTION_LAYERS = {
    "mixed5b": "Mixed_5b",
    "mixed6a": "Mixed_6a",
    "mixed7": "Mixed_7c",
}


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
        self._counter = 1
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "uploads"), exist_ok=True)

    def _load_model(self):
        if self._model is not None:
            return self._model
        from torchvision import models

        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=weights)
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
        use_mean_loss=False, grad_blur=0,
    ):
        import torch

        model = self._load_model()
        img = image.detach().clone().requires_grad_(True)
        captured = {}

        def hook(_module, _inputs, output):
            captured["acts"] = output

        handle = layer_module.register_forward_hook(hook)
        try:
            for step in range(int(steps)):
                model.zero_grad(set_to_none=True)
                if img.grad is not None:
                    img.grad.zero_()
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
                    img.add_(step_sz * grad)
                    img.clamp_(0.0, 1.0)
                if on_progress and (step == 0 or step == int(steps) - 1 or step % max(1, int(steps) // 4) == 0):
                    on_progress("DeepDream step {}/{}".format(step + 1, int(steps)))
        finally:
            handle.remove()
        return img.detach()

    def _run_octaves(
        self, image, layer_module, steps, octaves, on_progress=None, step_size=None,
        use_mean_loss=False, grad_blur=0,
    ):
        import torch
        import torch.nn.functional as F

        base_h, base_w = image.shape[2], image.shape[3]

        # Build octave pyramid going UP (larger scales) — matches original Google DeepDream
        # Going up produces large features (dogs, faces) at high octaves,
        # then refines with fine detail at lower octaves.
        MAX_OCTAVE_PX = 900
        octave_bases = [image.clone()]
        for i in range(int(octaves) - 1):
            new_h = min(int(base_h * (OCTAVE_SCALE ** (i + 1))), MAX_OCTAVE_PX)
            new_w = min(int(base_w * (OCTAVE_SCALE ** (i + 1))), MAX_OCTAVE_PX)
            if new_h <= octave_bases[-1].shape[2] and new_w <= octave_bases[-1].shape[3]:
                break
            up = F.interpolate(
                octave_bases[-1], size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            )
            octave_bases.append(up)

        actual_octaves = len(octave_bases)

        # Start with zero detail at the largest scale
        detail = torch.zeros_like(octave_bases[-1])

        # Process from LARGEST to SMALLEST
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

            dreamed = self._dream_once(
                base + detail, layer_module, steps, on_progress=on_progress,
                step_size=step_size, use_mean_loss=use_mean_loss, grad_blur=grad_blur,
            )
            detail = dreamed - base

        # detail is now at base scale — result = original + accumulated multi-scale detail
        if detail.shape[2:] != (base_h, base_w):
            detail = F.interpolate(
                detail, size=(base_h, base_w),
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
            on_progress("Loading InceptionV3...")
        layer_module = self._layer_module(layer)
        if mode == "redux":
            base_step = STEP_SIZE_REDUX
            if preserve is None:
                preserve = PRESERVE_REDUX
            use_mean_loss = True
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
            tag = datetime.now().strftime("deepdream_%Y%m%d_%H%M%S") + "_{:04d}".format(self._counter)
            self._counter += 1
            output_path = os.path.join(self.output_dir, tag + ".png")
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
