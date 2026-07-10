"""PNG metadata helpers for BlobDream / BlobVision outputs."""
import json
from datetime import datetime, timezone

PNG_META_KEY = "blobdream"
PNG_LEGACY_COMMENT_KEY = "comment"


def build_metadata(
    mode,
    prompt,
    seed,
    iterations,
    denoise_fidelity=None,
    negative_prompt=None,
    sketch_steps=None,
    sketch_size=None,
    init_image=None,
    sketch_path=None,
    extra=None,
):
    meta = {
        "blobdream": "1.0",
        "mode": mode,
        "prompt": prompt,
        "seed": int(seed),
        "iterations": int(iterations),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if negative_prompt:
        meta["negative_prompt"] = negative_prompt
    if denoise_fidelity is not None:
        meta["denoise_fidelity"] = float(denoise_fidelity)
    if sketch_steps is not None:
        meta["sketch_steps"] = int(sketch_steps)
    if sketch_size is not None:
        meta["sketch_width"] = int(sketch_size[0])
        meta["sketch_height"] = int(sketch_size[1])
    if init_image:
        meta["init_image"] = init_image
    if sketch_path:
        meta["sketch_path"] = sketch_path
    if extra:
        meta.update(extra)
    return meta


def metadata_for_png(meta):
    """Return dict suitable for generate.args.png_metadata."""
    return {
        PNG_META_KEY: json.dumps(meta, ensure_ascii=False),
        PNG_LEGACY_COMMENT_KEY: meta.get("prompt", ""),
    }


def read_metadata_from_image(path):
    from PIL import Image

    with Image.open(path) as img:
        text = img.text or {}
    raw = text.get(PNG_META_KEY)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    comment = text.get(PNG_LEGACY_COMMENT_KEY)
    if comment:
        return {"prompt": comment, "legacy_comment_only": True}
    return None


def seed_from_metadata(meta):
    if not meta:
        return None
    seed = meta.get("seed")
    if seed is None:
        return None
    return int(seed)
