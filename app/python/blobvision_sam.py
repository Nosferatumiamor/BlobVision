"""SAM2 (Segment Anything 2) click-to-select segmentation for BlobVision.

Used interactively — the user clicks a point on the init image, gets back a
mask for whatever segment is under it, possibly repeats with Ctrl+click to
add more segments — rather than once per generation like the other engines.
That usage shape (load once, query many times, explicitly unloaded rather
than parked) is why this deliberately does NOT go through BlobVRAMCache
(blobvision_engine.py's PHASE_SKETCH/PHASE_VQGAN park/unpark dance is a
hardcoded binary mutex, not a general N-phase registry) — it follows the
same lazy-load/`.unload()` pattern already used for the caption model
(blobvision_caption.py's CaptionEngine), which is lighter-weight than
SDXL/VQGAN and used the same on-demand way.

Mask edge refinement (erosion/dilation/feather) deliberately does NOT live
here — that happens client-side (see tauri/src/main.ts) so slider drags
don't round-trip to the backend. This module only ever returns raw,
un-refined masks, built by replaying the user's clicks in order (union or
subtract each one into an accumulator — see Sam2Engine.segment). Each click
is still queried independently of the others against the cached image
embedding — never combined into one shared multi-point prompt with a PRIOR
click, since the user is selecting several *distinct* objects/carve-outs,
not describing one object with several hints. A single click's OWN query
can still be multi-point though: at negative "segment size" it's paired
with synthetic background points ringed around it (see _fine_prompt) to
squeeze SAM2 toward a tighter reading than its 3 stock per-point candidates
alone can offer.
"""
import os
import threading
import uuid

from blobvision_paths import SAM2_CONFIG_NAME, ensure_layout, sam2_checkpoint_path

ensure_layout()

MIN_CHECKPOINT_BYTES = 1024 * 1024


def sam2_weights_status():
    path = sam2_checkpoint_path()
    ready = os.path.isfile(path) and os.path.getsize(path) >= MIN_CHECKPOINT_BYTES
    return {"ready": ready, "path": path}


def download_sam2_weights(on_progress=None):
    status = sam2_weights_status()
    if status["ready"]:
        return status["path"]
    from blobvision_paths import SAM2_CHECKPOINT_NAME, SAM2_HF_REPO, sam2_model_dir

    # main.rs forces HF_HUB_OFFLINE=1 for the whole app (see
    # blobvision_engine.enable_hub_downloads' own docstring) so hf_hub_download
    # below would otherwise raise immediately on a machine where this
    # checkpoint isn't already cached, rather than actually downloading it.
    from blobvision_engine import enable_hub_downloads
    enable_hub_downloads()

    if on_progress:
        on_progress("Downloading SAM2 checkpoint (~185 MB)...")
    from huggingface_hub import hf_hub_download

    out_dir = sam2_model_dir()
    os.makedirs(out_dir, exist_ok=True)
    hf_hub_download(
        SAM2_HF_REPO,
        filename=SAM2_CHECKPOINT_NAME,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )
    if on_progress:
        on_progress("SAM2 checkpoint installed.")
    return status["path"]


class Sam2Engine:
    def __init__(self, device=None):
        import torch

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._lock = threading.Lock()
        self._predictor = None
        # Opaque token identifying whichever image is currently embedded —
        # guards segment() against a stale request racing a re-embed (e.g.
        # the user swaps the init image while a click request is in flight).
        self._embed_token = None
        self._embed_w = 0
        self._embed_h = 0

    def _load(self):
        if self._predictor is not None:
            return
        if not sam2_weights_status()["ready"]:
            raise RuntimeError("SAM2 weights are not installed.")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # SAM2 loads straight onto the GPU outside BlobVRAMCache's park/
        # unpark mutex (see the module docstring), so its very first load
        # of a session can race the app's own automatic SDXL/VQGAN startup
        # warmup, which is also actively moving large models on/off the GPU
        # around the same time. That race has been observed (real user
        # report) to surface as a PyTorch "Cannot copy out of meta tensor;
        # no data!" error out of build_sam2()'s own model.to(device) call —
        # a transient symptom of the contention (a clean retry a moment
        # later succeeds), not a real incompatibility. A couple of short
        # retries covers the actual startup window without needing to
        # plumb a "warmup finished" signal in from the caller.
        last_err = None
        for attempt in range(3):
            try:
                model = build_sam2(
                    config_file=SAM2_CONFIG_NAME,
                    ckpt_path=sam2_checkpoint_path(),
                    device=str(self.device),
                )
                self._predictor = SAM2ImagePredictor(model)
                return
            except RuntimeError as exc:
                if "meta tensor" not in str(exc):
                    raise
                last_err = exc
                if self.device.type == "cuda":
                    import torch

                    torch.cuda.empty_cache()
                if attempt < 2:
                    import time

                    time.sleep(2.0)
        raise last_err

    def unload(self):
        import gc

        import torch

        if self._predictor is None:
            return
        del self._predictor
        self._predictor = None
        self._embed_token = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def embed_image(self, image_path):
        """Runs the (expensive) image encoder once; subsequent segment()
        calls against the returned token are cheap point queries against
        this cached embedding."""
        from PIL import Image

        with self._lock:
            self._load()
            image = Image.open(image_path).convert("RGB")
            self._predictor.set_image(image)
            self._embed_token = uuid.uuid4().hex
            self._embed_w, self._embed_h = image.width, image.height
            return {"token": self._embed_token, "width": image.width, "height": image.height}

    def segment(self, token, points):
        """`points` is a list of (x, y, label, size) in pixel coordinates,
        applied in order to build up one accumulated mask:
          - label=1 unions this point's own segment into the accumulator,
            label=0 subtracts it — this is what lets the overlay's
            click-to-toggle UX work (clicking inside the current selection
            removes just that segment instead of adding a redundant one, or
            carves a smaller piece back out of a big one). Order matters:
            each point is replayed against whatever the accumulator already
            is, not OR'd in blind, so add-then-subtract-then-add-again over
            the same area does the right thing.
          - size in [-1, 1] picks which of SAM2's 3 per-point candidate
            masks (multimask_output=True) to use, ranked by pixel area —
            -1 the smallest/most specific reading, +1 the largest/most
            inclusive one. This is what gives the UI a "segment size"
            control for free: SAM2 already computes all 3 candidates for a
            single point, we just weren't picking among them before
            (always took the highest-scoring one, which tends to skew
            toward whichever candidate SAM itself is most confident about,
            not necessarily the size the user wants).
            For negative size, we additionally bias SAM2's OWN prompt with
            4 synthetic background points ringed around the click (see
            _fine_query below) — picking among the 3 stock candidates alone
            can't go finer than whatever SAM naturally offers at that exact
            pixel (e.g. "face" might be the smallest reading it has there),
            but telling it "not out here either" forces a tighter read. The
            ring radius is calibrated off SAM's own baseline reading at that
            point (not a blind fraction of the image), so it scales with
            whatever's actually under the click; it shrinks further as size
            approaches -1, so more Fine = a closer, more aggressive squeeze.
        `label` is about accumulator combination only — the query itself
        (aside from the size<0 background-point bias above) always treats
        the clicked point as a single positive prompt; Ctrl+click means "a
        distinct object", not "another hint about this one region", so
        points are never combined into one shared multi-point prompt with
        each other. Returns a bool HxW numpy array (H, W = the embedded
        image's own size)."""
        import numpy as np

        with self._lock:
            if not points:
                raise ValueError("points must be non-empty")
            if token != self._embed_token:
                raise ValueError("Stale or unknown embed token — re-embed the image.")
            acc = None
            for x, y, label, size in points:
                clamped_size = max(-1.0, min(1.0, size))
                if clamped_size < 0:
                    chosen = self._fine_query(x, y, clamped_size)
                else:
                    masks, _scores, _logits = self._predictor.predict(
                        point_coords=np.array([[x, y]], dtype=np.float32),
                        point_labels=np.array([1], dtype=np.int64),
                        multimask_output=True,
                    )
                    order = np.argsort([m.sum() for m in masks])
                    chosen = masks[order[int(round(clamped_size + 1.0))]].astype(bool)
                if acc is None:
                    acc = chosen if label else np.zeros_like(chosen)
                elif label:
                    acc = np.logical_or(acc, chosen)
                else:
                    acc = np.logical_and(acc, np.logical_not(chosen))
            return acc

    def _fine_query(self, x, y, clamped_size):
        """Handles clamped_size < 0. First runs the plain single-point query
        to see what SAM naturally offers at this pixel, and uses the
        smallest of those 3 candidates as a reference scale (its
        equivalent-circle radius) — rather than a blind fraction of the
        whole image, which either barely nudges a small feature or way
        overshoots a large one depending on what's actually under the
        click. Then re-queries with 4 background points ringed around the
        click at a fraction of that reference radius (0.9x at the mild end,
        0.4x at size=-1) — close enough to say "the object doesn't reach
        this far", forcing SAM toward a tighter read than picking-by-area
        alone can reach. Picks the smallest or 2nd-smallest of the new
        (also 3-way ambiguous) result depending on how far into negative
        territory `size` is."""
        import numpy as np

        fineness = -clamped_size  # (0, 1]
        base_masks, _scores, _logits = self._predictor.predict(
            point_coords=np.array([[x, y]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int64),
            multimask_output=True,
        )
        base_area = min(float(m.sum()) for m in base_masks)
        base_radius = max(4.0, (base_area / np.pi) ** 0.5)
        radius = base_radius * (0.9 - 0.5 * fineness)
        coords = [[x, y]]
        labels = [1]
        for dx, dy in ((radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self._embed_w and 0 <= ny < self._embed_h:
                coords.append([nx, ny])
                labels.append(0)
        if len(coords) == 1:
            return base_masks[int(np.argmin([m.sum() for m in base_masks]))].astype(bool)
        masks, _scores2, _logits2 = self._predictor.predict(
            point_coords=np.array(coords, dtype=np.float32),
            point_labels=np.array(labels, dtype=np.int64),
            multimask_output=True,
        )
        order = np.argsort([m.sum() for m in masks])
        bucket = int(round(clamped_size + 1.0))  # size in [-1, 0) -> bucket 0 or 1
        return masks[order[bucket]].astype(bool)
