BlobVision - Style Transfer branch
====================================

Code:   app/python/blobvision_style.py
Outputs: app/outputs/style-transfer/
Weights: app/style-transfer/weights/vgg19_imagenet.pth

Classic Gatys VGG19 neural style transfer (content photo + style reference image).

First run:
  app\venv\Scripts\python.exe app\scripts\download_style_transfer.py

PyTorch downloads VGG19 ImageNet weights (~548 MB) into weights/ unless already cached.
Optional style samples: drop JPG/PNG files in app/style-transfer/samples/

Modes:
  img2img - content image (preview right) + style reference (sidebar)

Sliders:
  Style strength - how strongly the style reference dominates
  Content fidelity - how closely the output keeps the content layout
  Steps - optimization iterations (more = slower, often sharper)
