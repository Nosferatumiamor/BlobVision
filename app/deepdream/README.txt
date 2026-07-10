BlobVision — DeepDream branch
===========================

Code: app/python/blobvision_deepdream.py
Outputs: outputs/deepdream/

Uses torchvision InceptionV3 (ImageNet weights). On first run PyTorch downloads weights into its cache (~100MB) unless already present.

Modes:
  txt2img — dream from random noise
  img2img — amplify patterns in your source image
