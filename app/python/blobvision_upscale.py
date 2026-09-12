"""Real-ESRGAN x2/x4 post-processing upscaler, shared across all BlobVision families.

Runs after generation, on request only — never as part of the generation loop itself.
Uses spandrel (pure PyTorch arch loader) instead of the official realesrgan/basicsr
packages: basicsr imports torchvision.transforms.functional_tensor, which newer
torchvision releases removed, so the official package fails to import outright.

x2 is the default/lighter option: plenty to make the "pourri" texture readable
without the output turning into an unwieldy multi-thousand-pixel file, and it's a
purpose-trained x2 checkpoint rather than a x4 model resized down after the fact.
x4 is offered alongside it (its own purpose-trained checkpoint, not x2 run twice)
for when a bigger output is actually wanted.
"""
import os
import threading

from blobvision_paths import ensure_layout, upscale_model_path

ensure_layout()

UPSCALE_MODEL_URLS = {
    2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}
MIN_WEIGHTS_BYTES = 1024 * 1024

# Caps peak VRAM by downscaling oversized inputs before upscaling rather than feeding
# them through at full size — comfortable on an 8GB card even without tiled inference.
MAX_INPUT_SIDE = 900


def upscale_weights_status(scale=2):
    path = upscale_model_path(scale)
    ready = os.path.isfile(path) and os.path.getsize(path) >= MIN_WEIGHTS_BYTES
    return {"ready": ready, "path": path}


def download_upscale_weights(scale=2, on_progress=None):
    status = upscale_weights_status(scale)
    if status["ready"]:
        return status["path"]
    path = status["path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if on_progress:
        on_progress("Downloading Real-ESRGAN x{} weights...".format(scale))
    import torch

    torch.hub.download_url_to_file(UPSCALE_MODEL_URLS[scale], path, progress=True)
    if on_progress:
        on_progress("Real-ESRGAN x{} weights installed.".format(scale))
    return path


class UpscaleEngine:
    def __init__(self, device=None, scale=2):
        import torch

        if scale not in UPSCALE_MODEL_URLS:
            raise ValueError("Unsupported upscale factor: {}".format(scale))
        self.scale = scale
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._lock = threading.Lock()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        path = download_upscale_weights(self.scale)
        from spandrel import ModelLoader

        model = ModelLoader().load_from_file(path)
        self._model = model.to(self.device).eval()
        return self._model

    def upscale(self, image_path, output_path=None, on_progress=None):
        import numpy as np
        import torch
        from PIL import Image

        if not image_path or not os.path.isfile(image_path):
            raise ValueError("No image to upscale.")

        with self._lock:
            model = self._load_model()
            img = Image.open(image_path).convert("RGB")
            if max(img.size) > MAX_INPUT_SIDE:
                scale = MAX_INPUT_SIDE / float(max(img.size))
                img = img.resize(
                    (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                    Image.LANCZOS,
                )
            if on_progress:
                on_progress(
                    "Upscaling {}x{} -> x{}...".format(img.width, img.height, self.scale)
                )
            arr = np.asarray(img).astype("float32") / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = model(tensor)
            out_arr = (out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            result = Image.fromarray(out_arr)
            if output_path is None:
                base, ext = os.path.splitext(image_path)
                output_path = base + "_upscaled" + (ext or ".png")
            result.save(output_path)
            if on_progress:
                on_progress("Saved " + os.path.basename(output_path))
            return output_path
