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

BlobVision is unfiltered. You are solely responsible for anything you generate
or share with it.
