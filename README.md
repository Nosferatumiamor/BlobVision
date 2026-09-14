# BlobVision

**Old AI, made controllable.** BlobVision is a Windows desktop app that emulates
three generations of "legacy" generative-AI image techniques — the models people
were using around 2021–2022, back when a single GPU and an afternoon of patience
were the whole toolkit — wrapped in a single UI with a modern SDXL Turbo sketch
step in front of them to make composition steerable.

## What it does

BlobVision runs three independent AI "families," switchable in the UI:

- **VQGAN+CLIP** — the original text-to-image approach from before diffusion models
  took over: a VQGAN image decoder is nudged, one optimization step at a time,
  toward whatever a CLIP model says matches your prompt. Slow and often chaotic on
  its own, which is exactly its aesthetic.
- **DeepDream** — Google's 2015 technique: it amplifies whatever an InceptionV3
  classifier already thinks it half-recognizes in an image, layer by layer, until
  the picture is dreaming.
- **Style Transfer** — the classic Gatys VGG19 optimization: repaint a content
  image in the brushstrokes of a style reference, the technique that predates
  every "style transfer" app that came after it.

### REDUX vs. Legacy — the actual principle

Each family has two ways to run:

- **Legacy** mode is the real, historic pipeline: the old model works from
  scratch (or from your own uploaded image) with no help from anything modern.
  Authentic, but slow and hard to steer toward a specific composition.
- **REDUX** mode (the fast default) puts **SDXL Turbo** in front: it generates a
  clean sketch from your prompt in about a second, and *that* sketch is what the
  old model then processes. SDXL only decides composition — the old model still
  does all the actual "look," so the output still has the VQGAN/DeepDream/VGG19
  aesthetic, just steerable by a text prompt instead of pure chance.

That's the whole idea: modern model for control, historic model for the look.

## How to use it

Once weights are installed and the engine's ready (the badge in the top-right
turns green), the basic loop is: pick a family, write a prompt — or drop in an
image or video — and hit **DEGENERATE**.

- **Family tabs** (top center) switch between VQGAN+CLIP, DeepDream, and Style
  Transfer — each keeps its own settings. Only one family's model sits on the
  GPU at a time; switching tabs doesn't reload anything by itself, generating
  does.
- **Prompt / Negative prompt** — what to generate, and what to avoid. The ▾
  next to each opens the **Warlock's Grimoire**, a library of curated prompt
  snippets you can drop in instead of typing from scratch.
- **Format** picks the output aspect ratio (1:1, 16:9, 9:16, or custom — set
  automatically when you drop your own image or video).
- **img2img** (right side) — drop or paste an image *or a video* here to steer
  generation from it instead of, or alongside, a prompt. Check **reuse img**
  to keep feeding the same source back in across several generations instead
  of it being replaced by the next result.
- **Magic wand** — once an image is loaded in img2img, click a point on it to
  segment out an object (SAM2); Ctrl+click subtracts from the selection. Cut
  punches a hole where selected, Keep punches everywhere else — from there
  you can export the masked result directly or feed it back into generation.
- **Video mode** kicks in automatically once a video's in img2img: trim the
  range you want, pick how many frames actually get regenerated (the rest are
  filled back in by RIFE/ffmpeg interpolation — fewer regenerated frames is
  faster but softer motion), and the render pane shows each processed frame
  live as it comes in.
- **Steps/Blob** and **Deslop** (VQGAN) control how long the optimization
  loop runs and, in img2img, how much of the source survives. DeepDream has
  its own **Inception layer** / **Intensity** pair; Style Transfer has
  **Style strength** / **Content fidelity** / **Steps**, or an **SDXL
  preset** shortcut instead of the classic, much slower optimization.
- **Upscale x2 / x4** runs the result through Real-ESRGAN afterward.
- **Seed** — pin a value and uncheck Random seed to reproduce a result, or
  check Reuse seed to carry the last one into the next generation.
- **Gallery** browses everything generated this session; **Cancel** stops
  whatever's currently running, image or video.
- **Fast reboot** (top right) keeps the engine warm in the background for 30
  minutes after you close the window, so a quick re-open skips reloading
  every model from scratch.

Nothing here is destructive or hard to undo, so the fastest way to actually
learn it is to just start dropping prompts and images in.

## Hardware & requirements

- **Windows only.** There is no macOS or Linux build, and none is planned.
- **An NVIDIA GPU is effectively required.** There is no practical CPU-only path
  — every mode leans on CUDA for anything resembling reasonable speed.
- Developed and tested on an **NVIDIA RTX 3060 (12 GB VRAM)** with **32 GB of
  system RAM**. A GPU with at least 12 GB of VRAM is recommended for smooth
  results across all three families; less will work but speed and maximum output
  size will vary a lot depending on your own hardware.
- **BlobVision runs entirely offline** once its models are installed — nothing
  you generate is ever sent anywhere. Weights are only downloaded once each and
  cached locally.
- Not guaranteed to run well — or at all — on every machine. If your GPU is old,
  underpowered, or on the low end of VRAM, expect a rougher experience than the
  spec above.

## Installing

1. Download the installer from this repo's [Releases](../../releases) page and
   run it. It'll ask where to install and shows real progress as it goes,
   including a first-run download of a private Python environment and PyTorch
   (several GB — see the installer's own notice for details).
   Windows will likely show a **SmartScreen** warning ("Windows protected
   your PC") the first time, since the installer isn't code-signed — this is
   normal for a small, free tool distributed this way. Click **More info →
   Run anyway** to proceed.
2. First launch also lets you pick which model weights to fetch (SDXL Turbo,
   VQGAN+CLIP, Style Transfer) — check whichever you want, skip the rest for
   now, install more later from the same panel.
3. Generate. Everything lands in the app's `outputs/` folder, which nothing
   ever clears automatically — worth emptying it out yourself once in a while
   if you generate a lot, especially video.

## License

BlobVision's own code is released under the [MIT license](LICENSE).

It downloads and runs several third-party AI models and libraries at runtime,
each under its own license — some are as permissive as BlobVision's own, one
(SDXL Turbo) is not. See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)
before you rely on this for anything beyond personal, non-commercial use.

## Disclaimer

BlobVision doesn't filter what you generate — same as Stable Diffusion,
ComfyUI, Automatic1111, and most local generative-AI tooling. It downloads
and runs SDXL Turbo (a model BlobVision didn't create) as one component
among several; nothing in the app steers you toward any particular kind of
content. What you generate and do with it is entirely your own
responsibility.
