"""Neural style transfer engine for BlobVision (VGG19 Gatys optimization)."""
import os
import random
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from blobvision_paths import (
    STYLE_WEIGHTS_DIR,
    ensure_layout,
    style_output_dir,
    style_weights_path,
)

ensure_layout()
WEIGHTS_DIR = STYLE_WEIGHTS_DIR
WEIGHTS_PATH = style_weights_path()
DEFAULT_OUTPUT = style_output_dir()
MIN_WEIGHTS_BYTES = 1024 * 1024


def style_transfer_weights_status():
    ready = (
        os.path.isfile(WEIGHTS_PATH)
        and os.path.getsize(WEIGHTS_PATH) >= MIN_WEIGHTS_BYTES
    )
    return {"ready": ready, "path": WEIGHTS_PATH}


def download_style_transfer_weights(on_progress=None):
    status = style_transfer_weights_status()
    if status["ready"]:
        return status
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if on_progress:
        on_progress("Style Transfer: downloading VGG19 (~548 MB)...")
    import torch
    from torchvision import models

    weights = models.VGG19_Weights.IMAGENET1K_V1
    vgg = models.vgg19(weights=weights)
    torch.save(vgg.state_dict(), WEIGHTS_PATH)
    if on_progress:
        size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
        on_progress("VGG19 saved ({:.1f} MB)".format(size_mb))
    return style_transfer_weights_status()


DEFAULT_STEPS = 300
DEFAULT_STYLE_STRENGTH = 1.0
DEFAULT_CONTENT_WEIGHT = 1.0
BASE_STYLE_WEIGHT = 1e6

CONTENT_LAYERS = ["21"]
STYLE_LAYERS = ["1", "6", "11", "20", "28"]


@dataclass
class StyleTransferResult:
    output_path: str
    seed: int
    metadata: Dict[str, Any]


class StyleTransferEngine:
    def __init__(self, output_dir=None, device=None):
        import torch

        self.output_dir = os.path.abspath(output_dir or DEFAULT_OUTPUT)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._lock = threading.Lock()
        self._vgg = None
        self._counter = 1
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "uploads"), exist_ok=True)
        os.makedirs(WEIGHTS_DIR, exist_ok=True)

    def _load_vgg(self):
        if self._vgg is not None:
            return self._vgg
        import torch
        from torchvision import models

        if os.path.isfile(WEIGHTS_PATH):
            vgg = models.vgg19(weights=None)
            try:
                state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(WEIGHTS_PATH, map_location="cpu")
            vgg.load_state_dict(state)
        else:
            weights = models.VGG19_Weights.IMAGENET1K_V1
            vgg = models.vgg19(weights=weights)
            try:
                torch.save(vgg.state_dict(), WEIGHTS_PATH)
            except OSError:
                pass

        vgg = vgg.features.to(self.device).eval()
        for param in vgg.parameters():
            param.requires_grad_(False)
        self._vgg = vgg
        return self._vgg

    def _load_image_tensor(self, path, width, height):
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
        arr = torch.tensor(list(img.getdata()), dtype=torch.float32, device=self.device)
        arr = arr.view(height, width, 3).permute(2, 0, 1).unsqueeze(0) / 255.0
        return arr

    def _normalize(self, tensor):
        return (tensor - self._mean) / self._std

    def _gram_matrix(self, tensor):
        import torch

        b, c, h, w = tensor.size()
        features = tensor.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def _extract_features(self, image, layer_ids: List[str]):
        vgg = self._load_vgg()
        x = self._normalize(image)
        captured = {}
        for index, layer in enumerate(vgg):
            x = layer(x)
            key = str(index)
            if key in layer_ids:
                captured[key] = x
        return captured

    def generate(
        self,
        content_image_path,
        style_image_path,
        width=512,
        height=512,
        steps=DEFAULT_STEPS,
        style_strength=DEFAULT_STYLE_STRENGTH,
        content_weight=DEFAULT_CONTENT_WEIGHT,
        seed=None,
        on_progress: Optional[Callable[[str], None]] = None,
    ):
        if not content_image_path or not os.path.isfile(content_image_path):
            raise ValueError("content_image_path required")
        if not style_image_path or not os.path.isfile(style_image_path):
            raise ValueError("style_image_path required")

        import torch

        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        torch.manual_seed(int(seed))

        if on_progress:
            on_progress("Loading VGG19...")
        self._load_vgg()

        content = self._load_image_tensor(content_image_path, int(width), int(height))
        style = self._load_image_tensor(style_image_path, int(width), int(height))

        style_weight = BASE_STYLE_WEIGHT * float(style_strength)
        content_w = float(content_weight)

        content_features = self._extract_features(content, CONTENT_LAYERS)
        style_features = self._extract_features(style, STYLE_LAYERS)
        style_grams = {layer: self._gram_matrix(style_features[layer]) for layer in STYLE_LAYERS}

        target = content.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([target], lr=0.01)

        if on_progress:
            on_progress(
                "Style transfer {}x{} (steps={}, strength={:.2f})".format(
                    int(width), int(height), int(steps), float(style_strength),
                )
            )

        with self._lock:
            for step in range(int(steps)):
                optimizer.zero_grad(set_to_none=True)
                target_features = self._extract_features(target, CONTENT_LAYERS + STYLE_LAYERS)

                content_loss = torch.mean(
                    (target_features[CONTENT_LAYERS[0]] - content_features[CONTENT_LAYERS[0]]) ** 2
                )
                style_loss = sum(
                    torch.mean((self._gram_matrix(target_features[layer]) - style_grams[layer]) ** 2)
                    for layer in STYLE_LAYERS
                )
                loss = content_w * content_loss + style_weight * style_loss
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    target.clamp_(0.0, 1.0)

                if on_progress and (
                    step == 0 or step == int(steps) - 1 or step % max(1, int(steps) // 4) == 0
                ):
                    on_progress("Style step {}/{}".format(step + 1, int(steps)))

            from PIL import Image

            arr = target.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
            arr = (arr * 255.0).astype("uint8")
            tag = datetime.now().strftime("style_%Y%m%d_%H%M%S") + "_{:04d}".format(self._counter)
            self._counter += 1
            output_path = os.path.join(self.output_dir, tag + ".png")
            Image.fromarray(arr, mode="RGB").save(output_path)

        meta = {
            "family": "style",
            "mode": "img2img",
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "style_strength": float(style_strength),
            "content_weight": float(content_weight),
            "style_weight": float(style_weight),
            "seed": int(seed),
            "output": os.path.basename(output_path),
        }
        if on_progress:
            on_progress("Saved " + os.path.basename(output_path))
        return StyleTransferResult(output_path=output_path, seed=int(seed), metadata=meta)
