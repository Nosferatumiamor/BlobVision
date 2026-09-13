"""Florence-2 image captioning assist for the SDXL preset restyle path.

Runs once per generation to describe the source image's subject in a short caption,
merged into the preset prompt so quirky non-human details (animal ears, expressions,
accessories) survive the restyle instead of being silently erased by the preset's own
style-defining prompt/negative_prompt — a static preset prompt has no idea what's
actually in any given source image.

Florence-2-base-ft (232M params, ~460MB) rather than a general-purpose VLM: this is a
single fixed task (describe the subject), and a purpose-built captioning model is both
lighter and more reliable at it than repurposing a chat-tuned model for vision.
"""
import os
import re
import threading

from blobvision_paths import ensure_layout, caption_model_dir

ensure_layout()

CAPTION_MODEL_ID = "microsoft/Florence-2-base-ft"
# "<CAPTION>" is too terse to reliably catch secondary details (an accessory, an
# expression); "<DETAILED_CAPTION>" was tested and came back too generic/vague on an
# illustrated source. "<MORE_DETAILED_CAPTION>" reliably named both cat ears AND a
# tongue-out expression on the reference test image — the level of detail this feature
# exists for — at the cost of a longer, list-like caption that needs trimming below.
CAPTION_TASK = "<MORE_DETAILED_CAPTION>"
MIN_WEIGHTS_BYTES = 1024 * 1024

# Florence-2 will happily describe the source as "an anime drawing of..." — but the
# preset's own negative_prompt is usually already fighting to suppress exactly that
# style, so echoing it back in the positive prompt just creates tug-of-war. Strip
# style/medium words and keep only the subject description.
_STYLE_WORDS = re.compile(
    r"\b(anime|manga|cartoon|illustration|illustrated|drawing|drawn|painting|painted|"
    r"digital art|artwork|sketch|sketched|comic|animated|animation)\b",
    re.IGNORECASE,
)
_BOILERPLATE_PREFIX = re.compile(r"^(in this image[,]?\s*(we can see)?\s*)", re.IGNORECASE)
_DANGLING_OF = re.compile(r"\b(a|an)\s+of\s+(a|an)\b", re.IGNORECASE)
_MAX_CAPTION_WORDS = 40

# Sentences describing a facial expression/pose (open mouth, tongue out, smiling...) —
# rather than a static feature like an accessory or color — actively instruct the SDXL
# pass to *distort* toward that expression instead of just preserving it as a detail.
# Confirmed on the reference test image: with these sentences included, the caption's
# "the mouth is open. the tongue is sticking out." stretched the jaw and shrank the eyes
# on the photorealistic preset at strength=0.5; removing them restored normal
# proportions with the ears/accessories still intact. Drop expression/pose sentences,
# keep everything else (colors, accessories, static features).
_EXPRESSION_SENTENCE = re.compile(
    r"\b(mouth|tongue|teeth|smil\w*|grin\w*|expression|frown\w*|laugh\w*)\b",
    re.IGNORECASE,
)


def _clean_caption(caption):
    if not caption:
        return ""
    caption = _BOILERPLATE_PREFIX.sub("", caption)
    caption = _STYLE_WORDS.sub("", caption)
    caption = _DANGLING_OF.sub(r"\2", caption)
    sentences = [s.strip() for s in caption.split(".")]
    sentences = [s for s in sentences if s and not _EXPRESSION_SENTENCE.search(s)]
    caption = ". ".join(sentences)
    caption = re.sub(r"\s{2,}", " ", caption).strip(" ,.")
    words = caption.split()
    if len(words) > _MAX_CAPTION_WORDS:
        caption = " ".join(words[:_MAX_CAPTION_WORDS])
    return caption


def caption_weights_status():
    path = caption_model_dir()
    config_path = os.path.join(path, "config.json")
    ready = os.path.isfile(config_path) and os.path.getsize(config_path) > 0
    return {"ready": ready, "path": path}


def download_caption_weights(on_progress=None):
    status = caption_weights_status()
    if status["ready"]:
        return status["path"]
    from blobvision_engine import enable_hub_downloads

    enable_hub_downloads(on_progress)
    if on_progress:
        on_progress("Downloading Florence-2 caption model (~460 MB)...")
    from huggingface_hub import snapshot_download

    path = status["path"]
    os.makedirs(path, exist_ok=True)
    snapshot_download(
        CAPTION_MODEL_ID,
        local_dir=path,
        local_dir_use_symlinks=False,
        # The repo carries the same weights as both model.safetensors and
        # pytorch_model.bin (transformers prefers safetensors when both are
        # present) — skipping the .bin halves this download for no loss.
        ignore_patterns=["*.md", "*.pdf", "pytorch_model.bin"],
    )
    if on_progress:
        on_progress("Florence-2 caption model installed.")
    return path


class CaptionEngine:
    def __init__(self, device=None):
        import torch

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._lock = threading.Lock()
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if not caption_weights_status()["ready"]:
            raise RuntimeError("Florence-2 caption weights are not installed.")
        model_dir = caption_model_dir()
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        # Florence-2's remote modeling code predates transformers' newer attention-
        # implementation dispatch; its _supports_sdpa property crashes on current
        # transformers unless attn_implementation is given explicitly (skips the
        # auto-detection path that trips over it).
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                model_dir, trust_remote_code=True, torch_dtype=dtype, attn_implementation="eager"
            )
            .to(self.device)
            .eval()
        )
        self._processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    def unload(self):
        import gc

        import torch

        if self._model is None:
            return
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def describe(self, image_path, on_progress=None):
        """Best-effort short subject caption; returns "" on any failure so a missing
        or broken caption model never blocks generation."""
        try:
            import torch
            from PIL import Image

            with self._lock:
                self._load()
                if on_progress:
                    on_progress("Describing source image...")
                img = Image.open(image_path).convert("RGB")
                inputs = self._processor(text=CAPTION_TASK, images=img, return_tensors="pt")
                inputs = {k: v.to(self.device, self._model.dtype) if v.is_floating_point() else v.to(self.device)
                          for k, v in inputs.items()}
                with torch.no_grad():
                    # Florence-2's custom prepare_inputs_for_generation still expects the
                    # old tuple-based KV-cache format; current transformers' generation
                    # loop passes its newer Cache object instead, which crashes that code
                    # path (`past_key_values[0][0]` on a non-tuple). use_cache=False
                    # sidesteps it entirely — slower, but the caption is one short call.
                    generated_ids = self._model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=100,
                        num_beams=1,
                        use_cache=False,
                    )
                text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed = self._processor.post_process_generation(
                    text, task=CAPTION_TASK, image_size=(img.width, img.height)
                )
                caption = _clean_caption(parsed.get(CAPTION_TASK, ""))
                if on_progress and caption:
                    on_progress("Caption: " + caption)
                return caption
        except Exception as exc:
            if on_progress:
                on_progress("Caption skipped: " + str(exc))
            return ""
