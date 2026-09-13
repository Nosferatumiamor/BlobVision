# Third-Party Notices

BlobVision's own code is MIT-licensed (see [LICENSE](LICENSE)), but it vendors
some third-party code directly and downloads several AI models and Python
libraries at runtime. Each of those keeps its own license — BlobVision's MIT
license does not, and cannot, override any of them. This file summarizes what
was verified and where to check the current, authoritative text yourself
before relying on any of this for more than personal, non-commercial use —
license terms move, and the summaries below are current as of this writing,
not a guarantee of what a given project's license says today.

## Vendored code (ships inside this repo)

| Component | Path | License |
|---|---|---|
| VQGAN+CLIP CLI (`generate.py`) | `app/generate.py` | MIT — Katherine Crowson, Nerdy Rodent |
| taming-transformers (VQGAN model code) | `vendor/taming-transformers/` | MIT — Esser, Rombach, Ommer (CompVis) |
| OpenAI CLIP | `app/CLIP/` | MIT — OpenAI |

## AI models downloaded at runtime (not bundled — fetched on demand)

| Model | Used by | License | Notes |
|---|---|---|---|
| **SDXL Turbo** (`stabilityai/sdxl-turbo`) | REDUX sketch step, every family | [Stability AI Community License](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md) | **Not MIT-equivalent.** Free for orgs/individuals under $1M annual revenue; above that, a separate Stability AI license is required. If you ever redistribute the weights yourself (BlobVision doesn't — it points users at Stability AI's own download), the license requires including a copy of the agreement and displaying "Powered by Stability AI." Also prohibits using its outputs to train or improve a competing foundation model. |
| **VQGAN checkpoint** (`vqgan_imagenet_f16_16384`) | VQGAN+CLIP (both modes) | No separate license found | The taming-transformers *code* is MIT, but this specific ImageNet-trained checkpoint (originally hosted by CompVis/Heidelberg) was never issued its own explicit license — it's a long-standing research artifact from the original VQGAN+CLIP notebook era, used here the same way the whole community has used it since 2021. Treat it as research-use provenance, not a clearly-licensed asset. |
| **OpenCLIP weights** (LAION `ViT-L-14`/`ViT-B-16`) | VQGAN+CLIP's CLIP guidance | MIT (open_clip) | Trained on LAION's web-scraped image-text pairs; open_clip's own license is permissive, but see [mlfoundations/open_clip#503](https://github.com/mlfoundations/open_clip/issues/503) for the community's own discussion of the weights specifically (not just the training code). |
| **GoogLeNet** (InceptionV3-style pretrained weights) | DeepDream | Code: BSD-3-Clause (torchvision/PyTorch) | torchvision's license covers the *code*; it does not separately re-license the ImageNet-pretrained weights it ships, which is a known gray area shared by essentially every torchvision-pretrained model — the original ImageNet dataset's own terms were research-oriented, though torchvision's pretrained weights are near-universally used in commercial and open-source projects alike. |
| **VGG19** (pretrained weights) | Style Transfer | Same as GoogLeNet above | Same torchvision/ImageNet-weights caveat. |
| **SAM 2** (Segment Anything 2, `facebook/sam2.1-hiera-small`) | Magic wand / segment select | [Apache 2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE) — Meta | Code and checkpoints both Apache 2.0, no restrictions of note. |
| **Real-ESRGAN** (`RealESRGAN_x2plus`/`x4plus`) | Upscale x2/x4 | [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN) — Xintao Wang | No restrictions of note. |
| **Florence-2-base-ft** | Caption-assist (SDXL preset restyle path) | MIT — Microsoft | No restrictions of note. |

## Python dependencies

BlobVision's first-run bootstrap installs PyTorch plus roughly 150 other Python
packages (`app/requirements-frozen.txt`). These are overwhelmingly permissively
licensed (BSD, MIT, Apache-2.0) as is typical for the scientific Python
ecosystem, but this file doesn't enumerate all of them individually — run
`pip-licenses` inside the app's `venv/` if you need a full manifest for
compliance purposes.

## If you plan to distribute or monetize anything built on BlobVision

The one entry above that actually changes what you're allowed to do is **SDXL
Turbo** — read its license before doing anything beyond personal use, especially
if revenue is involved. Everything else here is either fully permissive
(MIT/BSD/Apache) or an unresolved-but-widely-used research artifact (the VQGAN
checkpoint, the ImageNet-pretrained weights) rather than something under
explicit restrictive terms.
