"""BlobVision unified generation engine (Legacy / Redux / Corrupt)."""
import gc
import math
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from blobvision_paths import (
    APP_DIR,
    BLOBVISION_ROOT,
    BLOBDREAM_ROOT,
    TAMING_REPO,
    basename_hint_from_upload,
    build_output_name,
    ensure_layout,
    openclip_root,
    outputs_dir,
    sdxl_model_dir,
    sketch_dir,
    vqgan_checkpoint_path,
    vqgan_config_path,
    vqgan_model_dir,
)

SCRIPT_DIR = APP_DIR
ensure_layout()
DEFAULT_SKETCH_MODEL = sdxl_model_dir()
DEFAULT_OUTPUT = outputs_dir()
_PY_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    os.chdir(APP_DIR)
except OSError as exc:
    print("Warning: could not chdir to APP_DIR: {}".format(exc), flush=True)
for path in (APP_DIR, _PY_DIR, TAMING_REPO, BLOBDREAM_ROOT):
    if path not in sys.path and os.path.isdir(path):
        sys.path.insert(0, path)

from blobvision_meta import build_metadata, metadata_for_png, read_metadata_from_image, seed_from_metadata
# The generic ffmpeg/RIFE video pipeline — see blobvision_video.py's module
# docstring for why this lives in its own file (DeepDream needs the exact
# same pipeline, driven by its own per-frame callback, not VQGAN's).
from blobvision_video import VIDEO_FRAME_STEP, _process_video, setup_bundled_video_codecs


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
        import time
        t0 = time.time()
        if phase == self.PHASE_SKETCH:
            self._park_vqgan(log)
            self._unpark_sketch(log)
        else:
            self._park_sketch(log)
            self._unpark_vqgan(log)
        self.active = phase
        self._log_vram(phase, log)
        self._say("VRAM cache: phase switch -> {} took {:.1f}s total.".format(phase, time.time() - t0), log)

    def note_sketch_loaded_on_gpu(self):
        if self.engine._sketch_pipe is not None and self.engine.sketch_full_gpu:
            self._sketch_on_gpu = True

    def note_vqgan_loaded_on_gpu(self):
        if self.engine._vqgan_loaded:
            self._vqgan_on_gpu = True

    def _say(self, msg, log):
        try:
            print(msg, flush=True)
        except OSError:
            # stdout can be an unflushable handle (e.g. a Windows anonymous pipe
            # under some subprocess launchers, or no console at all in a
            # windowed build) — a broken debug print must never abort a
            # generation that's otherwise working fine.
            pass
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
            import logging
            import warnings
            # warnings.catch_warnings() only silences Python's `warnings`
            # module — it does nothing for diffusers' "Pipelines loaded with
            # dtype=torch.float16 cannot run with cpu device..." message,
            # which comes from its own logging.Logger("diffusers") instead.
            # That message is a false alarm here (we're relocating the pipe
            # to free VRAM, not about to run inference on it while parked),
            # but printed on every single park — i.e. every VQGAN
            # generation from now on — it reads as something breaking.
            # Raising the logger's own level is the only thing that
            # actually silences it.
            diffusers_logger = logging.getLogger("diffusers")
            prev_level = diffusers_logger.level
            diffusers_logger.setLevel(logging.ERROR)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    run_with_patience(
                        lambda: pipe.to("cpu"),
                        label="SDXL CPU park",
                        on_stage=log,
                        estimate_sec=60,
                        interval=15,
                    )
            finally:
                diffusers_logger.setLevel(prev_level)
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
        # Measured: this park<->unpark cycle (moving an already-loaded pipe
        # between CPU RAM and GPU VRAM) is fast — 1-2s in testing, every
        # time, including the very first one. The genuinely slow "~1-3 min,
        # up to 5 min" cost is a SEPARATE one-time thing: _load_sketch_pipe()
        # placing SDXL on GPU for the first time ever in this process (cold
        # CUDA context + first big allocation), not this recurring swap.
        # This message used to (wrongly) quote that same "~1-3 min" estimate
        # here too, which was misleading for every generation after the
        # first — see the "moving ~6.5 GB to GPU" message in
        # _load_sketch_pipe for the one that actually deserves it.
        self._say("VRAM cache: moving SDXL to GPU (usually a couple seconds once already loaded)...", log)
        try:
            run_with_patience(
                lambda: pipe.to(self.engine.cuda_device),
                label="SDXL GPU transfer",
                on_stage=log,
                estimate_sec=10,
                interval=5,
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

        def _do_park():
            eng.model.to("cpu")
            if eng.perceptor is not None:
                eng.perceptor.to("cpu")

        try:
            run_with_patience(_do_park, label="VQGAN+CLIP CPU park", on_stage=log, estimate_sec=15, interval=10)
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

        def _do_unpark():
            eng.model.to(self.engine.cuda_device)
            if eng.perceptor is not None:
                eng.perceptor.to(self.engine.cuda_device)

        try:
            run_with_patience(_do_unpark, label="VQGAN+CLIP GPU transfer", on_stage=log, estimate_sec=15, interval=10)
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


VQGAN_NEGATIVE_WEIGHT = -1.0
# diffusers only turns on classifier-free guidance when guidance_scale > 1 (strictly —
# see StableDiffusionXLImg2ImgPipeline.do_classifier_free_guidance:
# `self._guidance_scale > 1 and ...`). At exactly 1.0 this was a silent no-op: every
# negative_prompt passed anywhere SDXL Turbo is used (style presets, VQGAN/DeepDream
# redux sketches) had zero effect on the actual generation. 1.5 is a real, working CFG
# value — validated on the photorealistic preset fixing a structural face-warp/artifact
# issue that no amount of negative_prompt wording could touch while this was inert.
# (2.0 was tried first but produced visibly worse results on direct comparison.)
SDXL_CFG_WITH_NEGATIVE = 1.5

# SDXL Turbo img2img "restyle" presets for the Style Transfer family — an alternative to
# the classic VGG19 Gatys optimizer. strength = diffusers' img2img strength (0 = keep
# input, 1 = ignore it); higher-concept prompts (Gothic, medieval, cave painting) tend to
# reinterpret the whole scene rather than just restyle it, even at moderate strength, so
# their defaults sit lower than the pure-texture/medium styles.
#
# Two tested lessons baked into these prompts:
# 1. Painting/drawing styles need their TOOL named explicitly (oil on canvas, acrylic,
#    watercolor washes, engraving crosshatch...) — without it the result reads as a
#    generic filter rather than the actual medium.
# 2. Named-artist prompts are wildly inconsistent depending on how well-represented that
#    artist is in the base model's training data. Famous 20th-century fine artists
#    (Basquiat, Miro, Klee) restyle strongly and are genuinely recognizable. Niche comic
#    artists (tested: Kawajiri, Miura, Toriyama, Mark Schultz) came back almost unchanged
#    at these settings — those are deliberately left out. Naming a painter whose own
#    self-portraits dominate their tagged work (e.g. literally "Van Gogh") can also pull
#    toward THAT face rather than restyling the input — safer to describe the technique
#    (brushwork, palette) than to lean on the name for those cases.
# "Black and white" presets (engraving, silent-film, some photo styles) were only tested
# on an already-illustrated, already-colored source image and came back tinted rather than
# true monochrome there — a negative_prompt nudge is included to push harder toward mono,
# but this is untested against a real photo input, which is the actual intended use case.
# "Pointillism" as a literal dot technique never rendered visible dots at any tested
# strength/step count and was dropped — reframed into a broader Impressionism preset
# ("undefined suggested brushwork" rather than naming the dot technique) which worked well.
# Scene-hijacking styles (Gothic, Medieval, Bayeux Tapestry, Poussin) have a narrow or
# nonexistent middle ground on the tested source image: too low a strength barely touches
# it, and the jump to "actually stylized" tends to also fully reinvent the composition into
# a new scene rather than restyling the existing subject — there wasn't a strength value
# found that reliably restyles-without-reinventing for these specific prompts.
# Photorealistic/fashion/documentary/street/Newton/Larry-Clark photo presets did not
# meaningfully transform the (already-illustrated) test image even at 0.55 strength — an
# illustration input has a lot of "this is a drawing" prior to overcome. These are kept in
# because the concept is sound and a real photo input is the actual target use case, but
# they are the least field-tested presets in this set.
# The 80s cartoon preset originally named "He-Man Masters of the Universe" directly, which
# pulled the subject toward He-Man's own specific look (masculine jaw/build) rather than
# just applying the era's animation texture — same "named character dominates the result"
# effect as the Van Gogh case above. Rephrased around the toy-box-art/Filmation-era studio
# style instead of the character name, which keeps the subject intact.
STYLE_PRESETS = {
    "ghibli": {
        "label": "Studio Ghibli",
        "prompt": "Studio Ghibli anime style, soft painterly watercolor, hand-drawn, warm colors",
        "strength": 0.5,
    },
    "disney": {
        "label": "Disney Animation",
        "prompt": "classic Disney animated film style, clean bold outlines, warm expressive character design, soft cel shading, storybook illustration",
        "strength": 0.5,
    },
    "heman": {
        "label": "80s Cartoon (Filmation)",
        "prompt": "1980s toy action figure box art illustration, airbrushed painted cover art, bold heroic fantasy proportions, thick black outlines, flat saturated colors, vintage Saturday morning cartoon aesthetic",
        "strength": 0.5,
    },
    "albator": {
        "label": "80s Anime (Albator/Ulysse 31)",
        "prompt": "early 1980s French-Japanese anime style, in the style of Captain Harlock and Ulysses 31, retro sci-fi character design, simple bold linework, muted vintage color palette, cel animation texture",
        "strength": 0.5,
    },
    "plympton": {
        "label": "Rough Pencil (Bill Plympton)",
        "prompt": "hand-drawn rough pencil sketch animation style, in the style of Bill Plympton, wobbly uneven linework, grotesque exaggerated features, colored pencil texture, surreal morphing look",
        "strength": 0.5,
    },
    "ren_stimpy": {
        "label": "Gross-out Cartoon (Ren & Stimpy)",
        "prompt": "1990s Nickelodeon cartoon style, in the style of Ren and Stimpy, exaggerated grotesque close-up features, rubbery character design, bright saturated colors, wild bulging eyes",
        "strength": 0.55,
    },
    "spitting_image": {
        "label": "Latex Puppet (Spitting Image)",
        "prompt": "satirical latex puppet, in the style of Spitting Image, exaggerated grotesque caricature features, rubbery foam latex skin texture, oversized head proportions, studio puppet photography",
        # Same leftover-linework-as-microtexture issue as the photorealistic preset can
        # show up here too, but unlike that preset, wrinkles/creases are wanted (part of
        # the exaggerated latex caricature look) — only the mud-crack failure mode itself
        # is excluded, not wrinkles.
        "negative_prompt": "cracked skin, cracked texture, mud texture, reptile skin, dry skin",
        "strength": 0.5,
    },
    "muppet": {
        "label": "Felt Puppet (Muppet)",
        "prompt": "felt puppet character, in the style of the Muppets by Jim Henson, fuzzy colorful felt texture, big round googly eyes, soft plush stitched seams, warm stage lighting",
        "strength": 0.5,
    },
    "fumetti": {
        "label": "Fumetti (Dylan Dog/Diabolik)",
        "prompt": "Italian fumetti comic book art, in the style of Dylan Dog and Diabolik, high contrast black and white ink, moody noir chiaroscuro shading, atmospheric horror thriller illustration",
        "strength": 0.5,
        "negative_prompt": "color, colorful",
    },
    "russian_animation": {
        "label": "Soviet Animation",
        "prompt": "Soviet era Russian animation style, thick painterly hand-brushed oil texture, muted folk-art earthy color palette, atmospheric hazy soft lighting, grainy film texture, in the style of Soyuzmultfilm and Yuri Norstein",
        "strength": 0.6,
    },
    "bakshi": {
        "label": "Rotoscope (Bakshi)",
        "prompt": "1970s rotoscoped animation still, in the style of Ralph Bakshi, gritty painted psychedelic backgrounds, semi-realistic rough wobbly linework, muted grainy smoky color palette, underground adult animation look",
        "strength": 0.6,
    },
    "artstation": {
        "label": "ArtStation Epic",
        "prompt": "epic fantasy digital painting, dramatic lighting, trending on artstation, highly detailed illustration",
        "strength": 0.45,
    },
    "watercolor": {
        "label": "Watercolor",
        "prompt": "watercolor painting, translucent liquid washes, bleeding wet-on-wet edges, visible paper texture, soft pigment bloom",
        "strength": 0.45,
    },
    "impressionism": {
        "label": "Impressionism",
        "prompt": "impressionist painting, thick daubs of acrylic paint on canvas, undefined suggested brushwork, forms dissolving into color and light, soft blurred edges, visible textured strokes",
        "strength": 0.55,
    },
    "poussin": {
        "label": "Classical (Poussin)",
        "prompt": "French classical baroque oil painting, in the style of Nicolas Poussin, smooth idealized academic brushwork, warm earthy color palette, dramatic sculptural lighting",
        "strength": 0.5,
    },
    "cubism": {
        "label": "Cubism",
        "prompt": "cubist painting, geometric fragmented shapes, multiple perspectives, bold angular lines",
        "strength": 0.45,
    },
    "german_expressionism": {
        "label": "German Expressionism",
        "prompt": "German Expressionist painting, bold jagged brushstrokes, intense emotional colors, distorted forms",
        "strength": 0.5,
    },
    "basquiat": {
        "label": "Neo-Expressionist (Basquiat)",
        "prompt": "neo-expressionist painting in the style of Jean-Michel Basquiat, raw crude figures, scrawled text and symbols, bold color blocks, graffiti texture on canvas",
        "strength": 0.5,
    },
    "miro": {
        "label": "Surrealist (Miro)",
        "prompt": "surrealist painting in the style of Joan Miro, biomorphic shapes, bold primary colors, playful abstract symbols, flat color fields",
        "strength": 0.5,
    },
    "klee": {
        "label": "Abstract (Klee)",
        "prompt": "whimsical abstract painting in the style of Paul Klee, delicate geometric linework, muted color fields, childlike symbolic forms",
        "strength": 0.5,
    },
    "gothic": {
        "label": "Gothic Painting",
        "prompt": "Gothic medieval painting, flat golden background, elongated figures, religious icon style",
        "strength": 0.4,
    },
    "medieval": {
        "label": "Medieval Manuscript",
        "prompt": "medieval illuminated manuscript painting, flat perspective, decorative borders, tempera colors",
        "strength": 0.4,
    },
    "bayeux": {
        "label": "Tapestry (Bayeux)",
        "prompt": "medieval tapestry embroidery, coarse wool thread texture on linen fabric, flat naive stylized figures with simple outlines, narrow earthy color palette of red ochre and gold, decorative border, in the style of the Bayeux Tapestry",
        "strength": 0.4,
    },
    "cave": {
        "label": "Cave Painting",
        "prompt": "prehistoric cave painting, ochre and charcoal pigments, primitive rock art style",
        "strength": 0.4,
    },
    "dore_engraving": {
        "label": "19th c. Engraving (Dore)",
        "prompt": "19th century engraving illustration, fine crosshatch linework, dramatic black and white chiaroscuro, in the style of Gustave Dore",
        "strength": 0.5,
        "negative_prompt": "color, colorful, painting",
    },
    "german_expr_cinema": {
        "label": "German Expressionist Cinema",
        "prompt": "black and white German Expressionist silent film still, extreme high contrast, sharp angular shadows, in the style of Nosferatu and The Cabinet of Dr Caligari",
        "strength": 0.5,
        "negative_prompt": "color, colorful",
    },
    "manga": {
        "label": "Manga (B&W)",
        "prompt": "1990s shonen manga style, black and white ink, dense screentone halftone shading, dynamic action lines, retro manga print texture",
        "strength": 0.55,
        "negative_prompt": "color, colorful",
    },
    "tezuka": {
        "label": "Retro Anime (Tezuka)",
        "prompt": "in the style of Osamu Tezuka manga and anime, retro 1960s cartoon character design, simple bold linework, big expressive eyes, vintage anime look",
        "strength": 0.5,
    },
    "bd_francobelge": {
        "label": "Franco-Belgian BD",
        "prompt": "Franco-Belgian comic album style, ligne claire, flat bold colors, clean outlines",
        "strength": 0.5,
    },
    "comics_70s80s": {
        "label": "Comics (70s-80s)",
        "prompt": "1970s 1980s comic book illustration, bold ink outlines, Ben-Day dots, saturated flat colors, dynamic action pose, vintage newsprint texture",
        "strength": 0.5,
    },
    "pulp_50s": {
        "label": "Pulp Comic (50s)",
        "prompt": "1950s pulp comic book cover art, bold saturated colors, dramatic illustration, halftone print texture, vintage adventure magazine style",
        "strength": 0.5,
    },
    "western_comic": {
        "label": "Western Comic",
        "prompt": "western comic book style, bold ink outlines, halftone dots, vibrant primary colors",
        "strength": 0.55,
    },
    "photorealistic": {
        "label": "Photorealistic",
        "prompt": "photograph of a real human, natural human face proportions, single undistorted head, one pair of eyes, symmetrical face, correct human anatomy, smooth realistic skin, soft photographic lighting, DSLR photo, real hair strands",
        # "with pores" / "8k detail" in the prompt, combined with the source's anime
        # linework (nose lines, cheek shading strokes) surviving partial denoising,
        # made the model render those leftover edges as hyper-detailed micro-texture —
        # a cracked/leathery "dried mud" skin/blood-streak artifact. Dropping that
        # phrasing and banning it explicitly below fixed most of it, but the deeper fix
        # was discovered later: SDXL_CFG_WITH_NEGATIVE was 1.0, which diffusers silently
        # treats as "no CFG" (needs strictly > 1) — this negative_prompt had never
        # actually been applied to any generation. See SDXL_CFG_WITH_NEGATIVE.
        # Fresh-seed testing at strength=0.5/steps=14 (post-CFG-fix) surfaced a second,
        # separate failure mode from the mud-crack one: an elongated/stretched skull and,
        # on one seed, a full duplicate row of eyes — ordinary SDXL img2img anatomy
        # drift, unrelated to caption or CFG. Added explicit anti-duplication/
        # anti-elongation terms below (and mirrored as positive instructions above) now
        # that CFG is confirmed to actually apply.
        "negative_prompt": "anime, cartoon, illustration, drawing, painting, cel shading, big glossy eyes, flat colors, line art, cracked skin, cracked texture, wrinkles, veins, scars, dry skin, leathery skin, reptile skin, mud texture, blood, bloody, blood splatter, cuts, scratches, wounds, red marks, red streaks, extra eyes, duplicate eyes, second pair of eyes, three eyes, extra face, duplicate face, elongated head, stretched head, deformed skull, distorted proportions, disfigured, mutated anatomy, malformed",
        # strength=0.8/steps=12 got photorealistic skin but always erased non-human
        # source details (animal ears, expressions) entirely, no matter the prompt —
        # too much of the trajectory was unconditioned on the source. Caption-assist
        # (blobvision_caption.py, auto-injects a description of the actual source
        # subject into the prompt) plus a real working negative_prompt let much lower
        # strength reach the same photorealism while keeping those details. steps=8
        # (~4 effective denoising steps, matching SDXL Turbo's native distillation
        # regime) got the texture/style right but wasn't enough iteration to keep facial
        # geometry stable — elongated jaw, misaligned eyes on some seeds. steps=14 (~7
        # effective) fixed the geometry while keeping everything else.
        "strength": 0.5,
        "steps": 14,
    },
    "fashion_photo": {
        "label": "Fashion Photo",
        "prompt": "high fashion editorial photograph, studio lighting, glossy magazine quality, sharp focus, dramatic pose",
        "strength": 0.55,
    },
    "documentary_photo": {
        "label": "Documentary Photo",
        "prompt": "documentary photograph, natural available light, candid realism, photojournalistic style, film grain",
        "strength": 0.55,
    },
    "street_photo": {
        "label": "Street Photo (Cartier-Bresson)",
        "prompt": "black and white street photograph, candid decisive moment composition, high contrast film grain, 35mm photojournalism, in the style of Henri Cartier-Bresson",
        "strength": 0.55,
        "negative_prompt": "color, colorful, painting, illustration",
    },
    "newton_photo": {
        "label": "Studio Photo (Newton)",
        "prompt": "black and white fashion photograph, high contrast dramatic studio lighting, glamorous provocative pose, in the style of Helmut Newton",
        "strength": 0.55,
        "negative_prompt": "color, colorful, painting, illustration",
    },
    "larry_clark_photo": {
        "label": "Snapshot (Larry Clark)",
        "prompt": "raw gritty documentary snapshot photograph, harsh flash lighting, grainy film texture, youth subculture aesthetic, in the style of Larry Clark",
        "strength": 0.55,
        "negative_prompt": "painting, illustration",
    },
    "frazetta": {
        "label": "Heroic Fantasy (Frazetta)",
        "prompt": "heroic fantasy oil painting, in the style of Frank Frazetta, muscular dynamic figures, dramatic moody lighting, rich earthy color palette, painterly brushwork, epic sword and sorcery illustration",
        "strength": 0.5,
    },
    "vallejo": {
        "label": "Fantasy Pin-up (Boris Vallejo)",
        "prompt": "fantasy paperback cover art, in the style of Boris Vallejo, airbrushed oil painting, glossy idealized musculature, dramatic heroic pose, vivid saturated lighting",
        "strength": 0.5,
    },
    "giger": {
        "label": "Biomechanical (Giger)",
        "prompt": "biomechanical surrealist artwork, in the style of H.R. Giger, dark chrome and organic textures fused, alien xenomorph aesthetic, airbrushed monochrome greys, ominous atmospheric lighting",
        "negative_prompt": "bright colors, pastel, cheerful",
        "strength": 0.5,
    },
    "howe": {
        "label": "Epic Fantasy (John Howe)",
        "prompt": "epic fantasy illustration, in the style of John Howe, detailed atmospheric watercolor and ink linework, painterly Tolkien-inspired landscapes, dramatic scale and lighting",
        "strength": 0.5,
    },
    "custom": {
        "label": "Custom",
        "prompt": "",
        "strength": 0.5,
    },
}
DEFAULT_STYLE_PRESET = "ghibli"
STYLE_PRESET_STEPS = 8

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


@dataclass
class StylePresetResult:
    output_path: str
    seed: int
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
    try:
        print(msg, flush=True)
    except OSError:
        # See BlobVRAMCache._say — an unflushable stdout must never abort
        # generation just because a progress line couldn't be printed.
        pass
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


def sdxl_weights_status():
    # model_index.json is diffusers' own manifest, always present in a
    # complete snapshot_download and absent from a partial/failed one —
    # cheaper and more reliable than sizing every shard.
    marker = os.path.join(sdxl_model_dir(), "model_index.json")
    return {"ready": os.path.isfile(marker), "path": sdxl_model_dir()}


def download_sdxl_weights(on_progress=None):
    status = sdxl_weights_status()
    if status["ready"]:
        return status
    enable_hub_downloads()
    if on_progress:
        on_progress("Downloading SDXL Turbo (~6.5 GB)...")
    from huggingface_hub import snapshot_download

    out_dir = sdxl_model_dir()
    os.makedirs(out_dir, exist_ok=True)
    snapshot_download(
        "stabilityai/sdxl-turbo",
        local_dir=out_dir,
        ignore_patterns=["*.md", "*.pdf", "*.png", "*.jpg", "*.webp"],
    )
    if on_progress:
        on_progress("SDXL Turbo installed.")
    return sdxl_weights_status()


def openclip_weights_status():
    from generate import OPENCLIP_HF_REPOS

    root = openclip_root()
    for (model, tag) in OPENCLIP_HF_REPOS:
        dest = os.path.join(root, "{}__{}".format(model, tag))
        if not os.path.isdir(dest) or not os.listdir(dest):
            return {"ready": False, "path": root}
    return {"ready": True, "path": root}


def download_openclip_weights(on_progress=None):
    from generate import OPENCLIP_HF_REPOS

    status = openclip_weights_status()
    if status["ready"]:
        return status
    enable_hub_downloads()
    from huggingface_hub import snapshot_download

    root = openclip_root()
    os.makedirs(root, exist_ok=True)
    for (model, tag), repo_id in OPENCLIP_HF_REPOS.items():
        dest = os.path.join(root, "{}__{}".format(model, tag))
        if os.path.isdir(dest) and os.listdir(dest):
            continue
        if on_progress:
            on_progress("Downloading CLIP weights: {}...".format(repo_id))
        snapshot_download(repo_id=repo_id, local_dir=dest)
    if on_progress:
        on_progress("CLIP weights installed.")
    return openclip_weights_status()


# The classic vqgan_imagenet_f16_16384 checkpoint (~980 MB) has no HF Hub
# home of its own — every VQGAN+CLIP notebook lineage (including the
# RiversHaveWings one this app is credited to in its own UI footer) pulls it
# from CompVis's original Heidelberg university file share. That link is
# known to be slow/occasionally flaky under load (see
# github.com/CompVis/taming-transformers/issues/53), which is exactly why a
# second source is worth having: boris/vqgan_f16_16384 on the HF Hub mirrors
# the identical config+checkpoint (verified: same byte size, same VQGAN
# config) and, being HF Hub, downloads through the same resumable,
# already-proven-working snapshot_download machinery as SDXL/CLIP above
# instead of a hand-rolled HTTP GET.
_VQGAN_HEIBOX_CONFIG_URL = (
    "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1"
)
_VQGAN_HEIBOX_CKPT_URL = (
    "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1"
)
_VQGAN_HF_MIRROR_REPO = "boris/vqgan_f16_16384"
_VQGAN_CKPT_MIN_BYTES = 500 * 1024 * 1024  # real file is ~980MB; well short of that means a bad/partial download


def vqgan_checkpoint_status():
    cfg = vqgan_config_path()
    ckpt = vqgan_checkpoint_path()
    ready = (
        os.path.isfile(cfg)
        and os.path.isfile(ckpt)
        and os.path.getsize(ckpt) >= _VQGAN_CKPT_MIN_BYTES
    )
    return {"ready": ready, "path": ckpt}


def _stream_download(url, dest_path, on_progress=None, label=""):
    """Plain HTTP(S) GET streamed to disk, for the non-HF-Hub heibox source."""
    import requests

    tmp_path = dest_path + ".part"
    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        with open(tmp_path, "wb") as out:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                out.write(chunk)
                read += len(chunk)
                if on_progress and total:
                    on_progress("{}: {:.0f}%".format(label, 100 * read / total))
    os.replace(tmp_path, dest_path)


def _download_vqgan_from_heibox(cfg, ckpt, on_progress=None):
    _stream_download(_VQGAN_HEIBOX_CONFIG_URL, cfg, on_progress, "VQGAN config (heibox)")
    _stream_download(_VQGAN_HEIBOX_CKPT_URL, ckpt, on_progress, "VQGAN checkpoint (heibox)")
    if os.path.getsize(ckpt) < _VQGAN_CKPT_MIN_BYTES:
        raise IOError("Downloaded VQGAN checkpoint from heibox is too small — likely an error page, not the real file.")


def _download_vqgan_from_hf_mirror(cfg, ckpt, on_progress=None):
    enable_hub_downloads()
    from huggingface_hub import hf_hub_download

    out_dir = os.path.dirname(ckpt)
    os.makedirs(out_dir, exist_ok=True)
    if on_progress:
        on_progress("VQGAN checkpoint (Hugging Face mirror): config...")
    got_cfg = hf_hub_download(_VQGAN_HF_MIRROR_REPO, filename="config.yaml", local_dir=out_dir)
    if os.path.abspath(got_cfg) != os.path.abspath(cfg):
        os.replace(got_cfg, cfg)
    if on_progress:
        on_progress("VQGAN checkpoint (Hugging Face mirror): ~980 MB checkpoint...")
    got_ckpt = hf_hub_download(_VQGAN_HF_MIRROR_REPO, filename="model.ckpt", local_dir=out_dir)
    if os.path.abspath(got_ckpt) != os.path.abspath(ckpt):
        os.replace(got_ckpt, ckpt)


def download_vqgan_checkpoint(on_progress=None):
    status = vqgan_checkpoint_status()
    if status["ready"]:
        return status
    cfg = vqgan_config_path()
    ckpt = vqgan_checkpoint_path()
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    try:
        if on_progress:
            on_progress("Downloading VQGAN checkpoint from heibox.uni-heidelberg.de...")
        _download_vqgan_from_heibox(cfg, ckpt, on_progress)
    except Exception as exc:
        print(
            "VQGAN heibox download failed ({}: {}) — falling back to the Hugging Face mirror.".format(
                type(exc).__name__, exc,
            ),
            flush=True,
        )
        if on_progress:
            on_progress("Heibox source unavailable — trying Hugging Face mirror instead...")
        _download_vqgan_from_hf_mirror(cfg, ckpt, on_progress)
    if on_progress:
        on_progress("VQGAN checkpoint installed.")
    return vqgan_checkpoint_status()


def vqgan_family_status():
    """VQGAN+CLIP as one unit: both the generative checkpoint and the CLIP
    weights are required together, neither is useful alone."""
    checkpoint = vqgan_checkpoint_status()
    clip = openclip_weights_status()
    return {"ready": checkpoint["ready"] and clip["ready"], "checkpoint": checkpoint, "clip": clip}


def download_vqgan_family(on_progress=None):
    download_vqgan_checkpoint(on_progress=on_progress)
    download_openclip_weights(on_progress=on_progress)
    return vqgan_family_status()


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
        # Printed unconditionally (not just on the "chose CPU offload" branch)
        # so the decision is never a black box — a user seeing CPU-offload
        # mode despite a card that "should" qualify can check this line
        # instead of guessing whether something else was already holding
        # VRAM when BlobVision started.
        print(
            "SDXL placement check: {:.1f} / {:.1f} GB free (need >=11 GB total, >=7 GB free for full GPU).".format(
                free / 1e9, total / 1e9,
            ),
            flush=True,
        )
        return total >= 11 * 1024 ** 3 and free >= 7 * 1024 ** 3
    except Exception as exc:
        # This used to swallow the exception silently, which meant a user
        # could see "CPU offload" chosen with NO "SDXL placement check: ..."
        # line before it (the print above never running) and no way to
        # tell whether that was VRAM genuinely being short, or
        # mem_get_info() itself failing for some other reason (a transient
        # CUDA/driver hiccup, contention from another process's GPU
        # context, etc.) — those are very different problems requiring
        # different fixes, and this was indistinguishable from the outside.
        print("SDXL placement check failed: {}: {}".format(type(exc).__name__, exc), flush=True)
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
        self._sketch_img2img_pipe = None
        self._caption_engine = None
        self._vqgan_loaded = False
        self._vqgan_clip_key = None
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

    def warmup_staged(
        self, on_stage=None, on_vqgan_ready=None, on_sdxl_ready=None,
        get_current_family=None,
    ):
        """Load SDXL then VQGAN, with GPU burst prep (VRAM flush + TF32).

        SDXL goes first: it's the sketch pass every family can use (VQGAN redux,
        DeepDream redux/txt2img), so it unblocks the most paths soonest. Once SDXL is
        ready, `get_current_family()` (if given) is consulted — if the user is
        already looking at DeepDream or Style (neither ever needs VQGAN), VQGAN load
        is deferred to a background thread instead of blocking readiness; otherwise
        (VQGAN tab, or family unknown) it loads synchronously as before.
        """

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
            stage("Loading SDXL Turbo (local checkpoints)...")
            self._load_sketch_pipe(on_stage=on_stage)
            self._vram.note_sketch_loaded_on_gpu()
            self._vram.activate(BlobVRAMCache.PHASE_SKETCH)
        if on_sdxl_ready:
            on_sdxl_ready()

        family = get_current_family() if get_current_family else None
        if family in ("deepdream", "style"):
            stage("Ready (VQGAN loading in background)...")

            def _load_vqgan_background():
                try:
                    with self._lock:
                        log_line("Loading VQGAN + CLIP (background)...", on_stage)
                        self._ensure_vqgan("redux", DEFAULT_ITERATIONS["redux"])
                        self._vram.note_vqgan_loaded_on_gpu()
                        self._vram._park_vqgan(on_stage)
                    if on_vqgan_ready:
                        on_vqgan_ready()
                except Exception as exc:
                    log_line("Background VQGAN load failed: {}".format(exc), on_stage)

            threading.Thread(target=_load_vqgan_background, daemon=True).start()
            return

        with self._lock:
            stage("Loading VQGAN + CLIP...")
            self._ensure_vqgan("redux", DEFAULT_ITERATIONS["redux"])
            self._vram.note_vqgan_loaded_on_gpu()
            # SDXL (the active phase) is already on GPU — park VQGAN straight back off
            # instead of activate(PHASE_SKETCH), which would no-op since it's already
            # the active phase and silently leave both stacks resident.
            self._vram._park_vqgan(on_stage)
        if on_vqgan_ready:
            on_vqgan_ready()
        stage("Ready")

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
            if self._caption_engine is not None:
                self._caption_engine.unload()
                self._caption_engine = None
            if had_vqgan:
                self._unload_vqgan_clip()
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

    def _unload_vqgan_clip(self):
        """Unload VQGAN+CLIP only, leaving SDXL resident if it's loaded.
        Caller must hold self._lock. Used to swap the single non-SDXL model
        that's kept resident when the active family changes (VQGAN <->
        DeepDream/Style), so all three families' models are never loaded
        at once."""
        import gc
        import torch

        if not self._vqgan_loaded:
            return
        print("Unloading VQGAN + CLIP (keeping SDXL resident)...", flush=True)
        import generate as gen_eng

        gen_eng.model = None
        gen_eng.perceptor = None
        if hasattr(gen_eng, "opt"):
            gen_eng.opt = None
        self._vqgan_loaded = False
        self._vqgan_clip_key = None
        purge_generate_globals()
        gc.collect()
        if self.cuda_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._vram._vqgan_on_gpu = False
        if self._vram.active == self._vram.PHASE_VQGAN:
            self._vram.active = None
        print("VQGAN + CLIP unloaded.", flush=True)

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

        source_width = source_height = None
        if (
            aspect == ASPECT_CUSTOM
            and width is None
            and height is None
            and init_image_path
            and os.path.isfile(init_image_path)
        ):
            from PIL import Image
            with Image.open(init_image_path) as _src:
                source_width, source_height = _src.size
        width, height = resolve_aspect_size(
            aspect=aspect, width=width, height=height,
            source_width=source_width, source_height=source_height,
        )

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
            if init_image_path and os.path.isfile(init_image_path):
                basename_source = basename_hint_from_upload(init_image_path)
            else:
                basename_source = prompt
            out_name = build_output_name("V", basename_source, ".png")
            output_path = os.path.join(self.output_dir, out_name)
            sketch_path = None

            if mode == "redux":
                if init_image_path and os.path.isfile(init_image_path):
                    # User provided an image (img2img redux) — skip SDXL sketch
                    init_image_path = os.path.abspath(init_image_path)
                    sketch_path = init_image_path
                    self._vram.activate(BlobVRAMCache.PHASE_VQGAN)
                    print("Redux img2img: using provided image, skipping SDXL sketch.", flush=True)
                else:
                    # Same date/NNN/basename as the final output (out_name),
                    # kept in sketch_dir() so outputs_dir() only shows
                    # finished pieces — but with "V" swapped for "VSK" so the
                    # filename itself also marks it as a sketch, not just its
                    # folder (a sketch pulled out of sketch_dir() can no
                    # longer collide with/overwrite the real final render).
                    sketch_name = out_name.replace("_V_", "_VSK_", 1)
                    sketch_path = os.path.join(sketch_dir(), sketch_name)
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
        self._sketch_img2img_pipe = None
        gc.collect()
        if self.cuda_device.startswith("cuda"):
            torch.cuda.empty_cache()

    def _load_sketch_img2img_pipe(self):
        """img2img view of the SDXL Turbo pipeline (shares weights, no extra load)."""
        self._load_sketch_pipe()
        if self._sketch_img2img_pipe is None:
            from diffusers import AutoPipelineForImage2Image
            self._sketch_img2img_pipe = AutoPipelineForImage2Image.from_pipe(self._sketch_pipe)
        return self._sketch_img2img_pipe

    def _load_caption_engine(self):
        if self._caption_engine is None:
            from blobvision_caption import CaptionEngine

            self._caption_engine = CaptionEngine(device=self.cuda_device)
        return self._caption_engine

    def generate_style_preset(
        self,
        image_path,
        prompt,
        strength,
        seed=None,
        negative_prompt=None,
        steps=STYLE_PRESET_STEPS,
        width=None,
        height=None,
        on_progress=None,
    ):
        """SDXL Turbo img2img restyle — the preset-driven alternative to VGG19 Style
        Transfer. Reuses the already-loaded sketch pipeline's weights via from_pipe()
        rather than loading a second checkpoint."""
        import torch
        from PIL import Image

        if not image_path or not os.path.isfile(image_path):
            raise ValueError("image_path is required")
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required")
        actual_seed = resolve_seed(seed)

        # Auto-describe the source subject (animal ears, expression, accessories...) so
        # the static preset prompt doesn't silently erase details it has no way of
        # knowing about. Best-effort: falls back to the preset prompt alone if the
        # caption model isn't installed or captioning fails for any reason.
        from blobvision_caption import caption_weights_status

        final_prompt = prompt.strip()
        if caption_weights_status()["ready"]:
            caption = self._load_caption_engine().describe(image_path, on_progress=on_progress)
            if caption:
                final_prompt = "{}, {}".format(final_prompt, caption)

        with self._lock:
            pipe = self._load_sketch_img2img_pipe()
            # No note_sketch_loaded_on_gpu() here (unlike warmup()) — this method
            # can run after the sketch pipe has been parked to CPU by a prior
            # VQGAN-active phase, and that call would wrongly mark it as already
            # on GPU, making activate() skip the real transfer below and crash
            # later with "Cannot generate a cpu tensor from a generator of type
            # cuda". activate() alone (like generate_redux_sketch already does)
            # correctly detects and performs the GPU move itself.
            self._vram.activate(BlobVRAMCache.PHASE_SKETCH)
            img = Image.open(image_path).convert("RGB")
            if width and height:
                img = img.resize((int(width), int(height)), Image.LANCZOS)
            else:
                w, h = snap_sdxl_dims(img.width, img.height)
                img = img.resize((w, h), Image.LANCZOS)
            generator = torch.Generator(self.cuda_device).manual_seed(actual_seed)
            neg = (negative_prompt or "").strip() or None
            guidance = SDXL_CFG_WITH_NEGATIVE if neg else 0.0
            steps = max(1, min(int(steps), 20))
            if on_progress:
                on_progress("SDXL restyle ({}x{}, strength={})...".format(img.width, img.height, strength))
            result = pipe(
                prompt=final_prompt,
                negative_prompt=neg,
                image=img,
                strength=float(strength),
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
            out_name = build_output_name("S", final_prompt, ".png")
            output_path = os.path.join(outputs_dir(), out_name)
            result.images[0].save(output_path)
            del result
        if on_progress:
            on_progress("Saved " + os.path.basename(output_path))
        meta = {
            "family": "style-sdxl",
            "prompt": final_prompt,
            "negative_prompt": neg or "",
            "strength": float(strength),
            "steps": int(steps),
            "seed": int(actual_seed),
            "width": img.width,
            "height": img.height,
            "output": os.path.basename(output_path),
        }
        return StylePresetResult(output_path=output_path, seed=int(actual_seed), metadata=meta)

    def generate_style_preset_video(
        self, video_path, prompt, strength, width, height,
        negative_prompt=None, steps=STYLE_PRESET_STEPS, seed=None,
        frame_step=None, on_progress=None, encode_from_sec=0.0,
        encode_to_sec=None, use_encode_range=False,
    ):
        """Style Transfer's video mode when an SDXL preset is active — bypasses
        the classic VGG19 Gatys engine (blobvision_style.py's StyleTransferEngine)
        entirely, since a preset never touches it even for still images (see
        /style/generate's branching). Drives the same family-agnostic pipeline
        VQGAN's own generate_video() uses (see blobvision_video.py), with
        generate_style_preset() (SDXL Turbo img2img, a handful of steps) as the
        per-keyframe transform instead of VQGAN's "corrupt" mode."""
        import shutil
        if frame_step is None:
            frame_step = VIDEO_FRAME_STEP

        def blobify_frame(src_path, dst_path):
            result = self.generate_style_preset(
                image_path=src_path, prompt=prompt, strength=strength, seed=seed,
                negative_prompt=negative_prompt, steps=steps, width=width, height=height,
            )
            shutil.copy2(result.output_path, dst_path)
            # The per-keyframe still is a real, fully-named "S" file — don't
            # let it permanently litter outputs_dir(); only the muxed final
            # video (built by _process_video below) is meant to persist.
            os.remove(result.output_path)

        return _process_video(
            video_path=video_path, blobify_frame=blobify_frame, width=width, height=height,
            frame_step=frame_step, on_progress=on_progress,
            encode_from_sec=float(encode_from_sec or 0.0), encode_to_sec=encode_to_sec,
            use_encode_range=bool(use_encode_range),
            type_tag="SV", basename_source=basename_hint_from_upload(video_path),
            extra_meta={"mode": "video", "engine": "sdxl", "prompt": prompt, "strength": float(strength)},
        )

    def generate_redux_sketch(
        self,
        prompt,
        width,
        height,
        seed,
        negative_prompt=None,
        sketch_steps=4,
    ):
        """SDXL Turbo sketch only (for DeepDream redux)."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required")
        with self._lock:
            self._load_sketch_pipe()
            self._vram.activate(BlobVRAMCache.PHASE_SKETCH)
            sketch_path = os.path.join(
                sketch_dir(), build_output_name("DSK", prompt, ".png", base_dir=sketch_dir()),
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

    def run_vqgan_warmup(self, on_stage=None):
        """Cheap 1-iteration VQGAN+CLIP pass so the first real generation
        avoids CUDA kernel-selection/JIT cold-start latency — same rationale
        as run_sketch_warmup(), but for VQGAN's own training step (CLIP
        cutouts, VQGAN decode, backward, optimizer.step()) instead of SDXL's
        forward pass; SDXL's warmup never exercises this half at all.
        Measured cause: a real first VQGAN loop ran visibly slower than the
        second one (e.g. 13s vs 7s for the same 25 iterations) with no
        other explanation — VRAM park/unpark for VQGAN+CLIP is itself near-
        instant (that model is small), so the gap is CUDA warming up, not
        data movement. Output goes to a throwaway temp file, deleted right
        after — this must never appear in the user's real output history."""
        if not self._vqgan_loaded:
            return

        def stage(msg):
            log_line(msg, on_stage)

        stage("VQGAN warmup pass (first GPU forward+backward)...")
        tmp_path = os.path.join(self.output_dir, ".vqgan_warmup_tmp.png")
        with self._lock:
            try:
                self._vram.activate(BlobVRAMCache.PHASE_VQGAN)
                self._run_vqgan(
                    profile_name="redux",
                    prompt="warmup",
                    output_path=tmp_path,
                    iterations=1,
                    init_image=None,
                    init_weight=0.0,
                    seed=0,
                    mode="legacy",
                    sketch_path=None,
                    sketch_steps=0,
                    # No explicit sketch_size/output_size (unlike SDXL's own
                    # warmup, which deliberately uses a small fixed size):
                    # the redux profile's own "size" is None, so passing a
                    # size here would be the ONLY thing setting eng.args.size
                    # away from whatever /warmup/vqgan's earlier
                    # _ensure_vqgan() call already left it at — which
                    # changes _clip_key() and silently forces a real,
                    # unnecessary reload_clip_and_cutouts() (the duplicate
                    # "Loading CLIP OpenAI (ViT-B/16)..." line, ~3s wasted)
                    # right before the dummy pass even starts. Leaving both
                    # None keeps this warmup at whatever size a real
                    # generation will actually use, which is also more
                    # representative of the CUDA kernel shapes worth priming.
                    sketch_size=None,
                    output_size=None,
                )
            finally:
                # Leaves the phase back at SKETCH — matches /warmup/vqgan's
                # existing contract of leaving SDXL resident, not VQGAN.
                self._vram.activate(BlobVRAMCache.PHASE_SKETCH)
                if os.path.isfile(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        stage("VQGAN warmup complete")

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
        import shutil

        width, height = resolve_aspect_size(aspect=aspect)
        iters = clamp_iterations(
            iterations if iterations is not None else DEFAULT_ITERATIONS["video"],
        )
        denoise = float(
            denoise_fidelity if denoise_fidelity is not None
            else DEFAULT_DENOISE.get("video", 0.3),
        )
        prompt = prompt.strip()

        # The one VQGAN-specific step in an otherwise generic ffmpeg/RIFE
        # pipeline (_process_video) — runs each extracted keyframe through
        # the ordinary "corrupt" (img2img) mode and copies the result to
        # where _process_video expects the processed frame. Any other
        # family's engine (DeepDream img2img, Style Transfer's Gatys loop)
        # could drive the same pipeline by passing an equivalent closure
        # here instead — nothing else below is VQGAN-aware.
        def blobify_frame(src_path, dst_path):
            result = self.generate(
                mode="corrupt",
                prompt=prompt,
                negative_prompt=negative_prompt,
                iterations=iters,
                denoise_fidelity=denoise,
                seed=seed,
                init_image_path=src_path,
                width=width,
                height=height,
            )
            shutil.copy2(result.output_path, dst_path)
            # The per-keyframe still is a real, fully-named "V" file — don't
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
            type_tag="VV",
            basename_source=basename_hint_from_upload(video_path),
            extra_meta={
                "mode": "video",
                "prompt": prompt,
                "iterations": iters,
                "denoise_fidelity": denoise,
            },
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("setup-codecs", "setup-video-codecs"):
        setup_bundled_video_codecs(force="--force" in sys.argv)
    else:
        print("BlobVision engine utility.", flush=True)
        print("  setup-codecs [--force]  Download ffmpeg + RIFE into video-codecs/", flush=True)
