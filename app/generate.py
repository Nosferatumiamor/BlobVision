# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

import argparse
import math
import random
# from email.policy import default
from urllib.request import urlopen
from tqdm import tqdm
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PY_DIR = os.path.join(_SCRIPT_DIR, 'python')
if _PY_DIR not in sys.path:
    sys.path.insert(0, _PY_DIR)
from blobvision_paths import (
    OPENCLIP_DEFAULT_PRETRAINED,
    OPENCLIP_HF_REPOS,
    TAMING_REPO,
    ensure_layout,
    hf_cache_root,
    openclip_root,
    vqgan_checkpoint_path,
    vqgan_config_path,
)
ensure_layout()
_OPENCLIP_ROOT = openclip_root()
_HF_CACHE = hf_cache_root()
os.makedirs(_HF_CACHE, exist_ok=True)
os.environ.setdefault('HF_HOME', _HF_CACHE)
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(_HF_CACHE, 'hub'))

if os.path.isdir(TAMING_REPO):
    sys.path.insert(0, TAMING_REPO)
else:
    sys.path.append('taming-transformers')

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
#import taming.modules 

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties
torch.backends.cudnn.benchmark = False		# NR: True is a bit faster, but can lead to OOM. False is more deterministic.
#torch.use_deterministic_algorithms(True)	# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation

from torch_optimizer import DiffGrad, AdamP
from lion_pytorch import Lion

from CLIP import clip
import open_clip
import kornia.augmentation as K
import numpy as np
import imageio

from PIL import ImageFile, Image, PngImagePlugin, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True

from subprocess import Popen, PIPE
import re

# Supress warnings
import warnings
warnings.filterwarnings('ignore')


# Check for GPU and reduce the default image size if low VRAM
default_image_size = 512  # >8GB VRAM
if not torch.cuda.is_available():
    default_image_size = 256  # no GPU found
elif get_device_properties(0).total_memory <= 2 ** 33:  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
    default_image_size = 304  # <8GB VRAM

# Create the parser
vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')

# Add the arguments
vq_parser.add_argument("-p",    "--prompts", type=str, help="Text prompts", default=None, dest='prompts')
vq_parser.add_argument("-ip",   "--image_prompts", type=str, help="Image prompts / target image", default=[], dest='image_prompts')
vq_parser.add_argument("-i",    "--iterations", type=int, help="Number of iterations", default=500, dest='max_iterations')
vq_parser.add_argument("-se",   "--save_every", type=int, help="Save image iterations", default=50, dest='display_freq')
vq_parser.add_argument("-s",    "--size", nargs=2, type=int, help="Image size (width height) (default: %(default)s)", default=[default_image_size,default_image_size], dest='size')
vq_parser.add_argument("-ii",   "--init_image", type=str, help="Initial image", default=None, dest='init_image')
vq_parser.add_argument("-in",   "--init_noise", type=str, help="Initial noise image (pixels or gradient)", default=None, dest='init_noise')
vq_parser.add_argument("-iw",   "--init_weight", type=float, help="Initial weight", default=0., dest='init_weight')
vq_parser.add_argument("-m",    "--clip_model", type=str,
                       help="CLIP/OpenCLIP model (OpenAI: ViT-B/32, ViT-B/16 | OpenCLIP: ViT-L-14, ...)",
                       default='ViT-B/32', dest='clip_model')
vq_parser.add_argument("--clip-backend", type=str, choices=['openai', 'open_clip'], default='openai',
                       dest='clip_backend', help="CLIP implementation (default: openai)")
vq_parser.add_argument("--clip-pretrained", type=str, default=None, dest='clip_pretrained',
                       help="OpenCLIP pretrained tag (e.g. laion2b_s32b_b82k for ViT-L-14)")
vq_parser.add_argument("-conf", "--vqgan_config", type=str, help="VQGAN config", default=vqgan_config_path(), dest='vqgan_config')
vq_parser.add_argument("-ckpt", "--vqgan_checkpoint", type=str, help="VQGAN checkpoint", default=vqgan_checkpoint_path(), dest='vqgan_checkpoint')
vq_parser.add_argument("-nps",  "--noise_prompt_seeds", nargs="*", type=int, help="Noise prompt seeds", default=[], dest='noise_prompt_seeds')
vq_parser.add_argument("-npw",  "--noise_prompt_weights", nargs="*", type=float, help="Noise prompt weights", default=[], dest='noise_prompt_weights')
vq_parser.add_argument("-lr",   "--learning_rate", type=float, help="Learning rate", default=0.1, dest='step_size')
vq_parser.add_argument("-cutm", "--cut_method", type=str, help="Cut method", choices=['original','updated','nrupdated','updatedpooling','latest'], default='latest', dest='cut_method')
vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=32, dest='cutn')
vq_parser.add_argument("-cutp", "--cut_power", type=float, help="Cut power", default=1., dest='cut_pow')
vq_parser.add_argument("-sd",   "--seed", type=int, help="Seed", default=None, dest='seed')
vq_parser.add_argument("-opt",  "--optimiser", type=str, help="Optimiser", choices=['Adam','AdamW','Adagrad','Adamax','DiffGrad','AdamP','RAdam','RMSprop','Lion'], default='Adam', dest='optimiser')
vq_parser.add_argument("-o",    "--output", type=str, help="Output image filename", default="output.png", dest='output')
vq_parser.add_argument("-vid",  "--video", action='store_true', help="Create video frames?", dest='make_video')
vq_parser.add_argument("-zvid", "--zoom_video", action='store_true', help="Create zoom video?", dest='make_zoom_video')
vq_parser.add_argument("-zs",   "--zoom_start", type=int, help="Zoom start iteration", default=0, dest='zoom_start')
vq_parser.add_argument("-zse",  "--zoom_save_every", type=int, help="Save zoom image iterations", default=10, dest='zoom_frequency')
vq_parser.add_argument("-zsc",  "--zoom_scale", type=float, help="Zoom scale %%", default=0.99, dest='zoom_scale')
vq_parser.add_argument("-zsx",  "--zoom_shift_x", type=int, help="Zoom shift x (left/right) amount in pixels", default=0, dest='zoom_shift_x')
vq_parser.add_argument("-zsy",  "--zoom_shift_y", type=int, help="Zoom shift y (up/down) amount in pixels", default=0, dest='zoom_shift_y')
vq_parser.add_argument("-cpe",  "--change_prompt_every", type=int, help="Prompt change frequency", default=0, dest='prompt_frequency')
vq_parser.add_argument("-vl",   "--video_length", type=float, help="Video length in seconds (not interpolated)", default=10, dest='video_length')
vq_parser.add_argument("-ofps", "--output_video_fps", type=float, help="Create an interpolated video (Nvidia GPU only) with this fps (min 10. best set to 30 or 60)", default=0, dest='output_video_fps')
vq_parser.add_argument("-ifps", "--input_video_fps", type=float, help="When creating an interpolated video, use this as the input fps to interpolate from (>0 & <ofps)", default=15, dest='input_video_fps')
vq_parser.add_argument("-d",    "--deterministic", action='store_true', help="Enable cudnn.deterministic?", dest='cudnn_determinism')
vq_parser.add_argument("-aug",  "--augments", nargs='+', action='append', type=str, choices=['Ji','Sh','Gn','Pe','Ro','Af','Et','Ts','Cr','Er','Re'], help="Enabled augments (latest vut method only)", default=[], dest='augments')
vq_parser.add_argument("-vsd",  "--video_style_dir", type=str, help="Directory with video frames to style", default=None, dest='video_style_dir')
vq_parser.add_argument("-cd",   "--cuda_device", type=str, help="Cuda device to use", default="cuda:0", dest='cuda_device')


# Globals (BlobVision keeps models loaded; each image resets z / prompts / opt)
args = None
all_phrases = []
video_frame_list = []
cwd = os.getcwd()
num_video_frames = 0
device = None
model = None
perceptor = None
clip_tokenizer = None
make_cutouts = None

# OPENCLIP_DEFAULT_PRETRAINED / OPENCLIP_HF_REPOS: imported above from
# blobvision_paths rather than defined here, so status/download-only callers
# don't have to pay this module's own heavy `from taming.models import
# cond_transformer, vqgan` import just to read a plain dict — see the
# comment on those two names in blobvision_paths.py.


def resolve_openclip_pretrained(model_name, pretrained_tag):
    """Prefer weights under checkpoints/open_clip/ (no network)."""
    folder = os.path.join(_OPENCLIP_ROOT, '{}__{}'.format(model_name, pretrained_tag))
    for fname in (
        'open_clip_pytorch_model.bin',
        'open_clip_model.safetensors',
        'model.safetensors',
        'pytorch_model.bin',
    ):
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            return path
    return pretrained_tag
normalize = None
z = None
z_orig = None
z_min = None
z_max = None
pMs = []
opt = None
i = 0
gumbel = False
sideX = sideY = 0
toksX = toksY = 0
e_dim = n_toks = 0
f = 2
this_video_frame = 0
make_styled_video = False


def init_args(argv=None):
    """Parse CLI args and apply preprocessing. Safe to call before import-side effects."""
    global args, all_phrases, video_frame_list, cwd, num_video_frames

    args = vq_parser.parse_args(argv)

    if not args.prompts and not args.image_prompts:
        args.prompts = "A cute, smiling, Nerdy Rodent"

    if args.cudnn_determinism:
        torch.backends.cudnn.deterministic = True

    if not args.augments:
        args.augments = [['Af', 'Pe', 'Ji', 'Er']]

    all_phrases = []
    if args.prompts:
        story_phrases = [phrase.strip() for phrase in args.prompts.split("^")]
        for phrase in story_phrases:
            all_phrases.append(phrase.split("|"))
        args.prompts = all_phrases[0]

    if args.image_prompts:
        args.image_prompts = args.image_prompts.split("|")
        args.image_prompts = [image.strip() for image in args.image_prompts]

    if args.make_video and args.make_zoom_video:
        print("Warning: Make video and make zoom video are mutually exclusive.")
        args.make_video = False

    if args.make_video or args.make_zoom_video:
        if not os.path.exists('steps'):
            os.mkdir('steps')

    if not args.cuda_device == 'cpu' and not torch.cuda.is_available():
        args.cuda_device = 'cpu'
        args.video_fps = 0
        print("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
        print("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")

    if args.video_style_dir:
        print("Locating video frames...")
        video_frame_list = []
        for entry in os.scandir(args.video_style_dir):
            if (entry.path.endswith(".jpg")
                    or entry.path.endswith(".png")) and entry.is_file():
                video_frame_list.append(entry.path)

        if not os.path.exists('steps'):
            os.mkdir('steps')

        args.init_image = video_frame_list[0]
        filename = os.path.basename(args.init_image)
        cwd = os.getcwd()
        args.output = os.path.join(cwd, "steps", filename)
        num_video_frames = len(video_frame_list)

    return args


# Various functions and classes
def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


# For zoom video
def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2, 
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


# NR: Testing with different intital images
def random_noise_image(w,h):
    random_image = Image.fromarray(np.random.randint(0,255,(w,h,3),dtype=np.dtype('uint8')))
    return random_image


# create initial gradient image
def gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)

    return result

    
def random_gradient_image(w,h):
    array = gradient_3d(w, h, (0, 0, np.random.randint(0,255)), (np.random.randint(1,255), np.random.randint(2,255), np.random.randint(3,128)), (True, False, False))
    random_image = Image.fromarray(np.uint8(array))
    return random_image


# Used in older MakeCutouts
def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


#NR: Split prompts and weights
def split_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow # not used with pooling
        
        # Pick your own augments & their order
        augment_list = []
        for item in args.augments[0]:
            if item == 'Ji':
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
            elif item == 'Sh':
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == 'Gn':
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
            elif item == 'Pe':
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == 'Ro':
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == 'Af':
                augment_list.append(K.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True)) # border, reflection, zeros
            elif item == 'Et':
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == 'Ts':
                augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
            elif item == 'Cr':
                augment_list.append(K.RandomCrop(size=(self.cut_size,self.cut_size), pad_if_needed=True, padding_mode='reflect', p=0.5))
            elif item == 'Er':
                augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7))
            elif item == 'Re':
                augment_list.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5))
                
        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = 0.1
        # self.noise_fac = False

        # Uncomment if you like seeing the list ;)
        # print(augment_list)
        
        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        cutouts = []
        
        for _ in range(self.cutn):            
            # Use Pooling
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
            
        batch = self.augs(torch.cat(cutouts, dim=0))
        
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An updated version with Kornia augments and pooling (where my version started):
class MakeCutoutsPoolingUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow # Not used with pooling

        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7,p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),            
        )
        
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        
        for _ in range(self.cutn):
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
            
        batch = self.augs(torch.cat(cutouts, dim=0))
        
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An Nerdy updated version with selectable Kornia augments, but no pooling:
class MakeCutoutsNRUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        
        # Pick your own augments & their order
        augment_list = []
        for item in args.augments[0]:
            if item == 'Ji':
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
            elif item == 'Sh':
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == 'Gn':
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
            elif item == 'Pe':
                augment_list.append(K.RandomPerspective(distortion_scale=0.5, p=0.7))
            elif item == 'Ro':
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == 'Af':
                augment_list.append(K.RandomAffine(degrees=30, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True)) # border, reflection, zeros
            elif item == 'Et':
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == 'Ts':
                augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
            elif item == 'Cr':
                augment_list.append(K.RandomCrop(size=(self.cut_size,self.cut_size), pad_if_needed=True, padding_mode='reflect', p=0.5))
            elif item == 'Er':
                augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7))
            elif item == 'Re':
                augment_list.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5))
                
        self.augs = nn.Sequential(*augment_list)


    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An updated version with Kornia augments, but no pooling:
class MakeCutoutsUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),)
        self.noise_fac = 0.1


    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# This is the original version (No pooling)
class MakeCutoutsOrig(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)


def load_vqgan_model(config_path, checkpoint_path):
    global gumbel
    gumbel = False
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


def _create_make_cutouts(cut_size):
    if args.cut_method == 'latest':
        return MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
    if args.cut_method == 'original':
        return MakeCutoutsOrig(cut_size, args.cutn, cut_pow=args.cut_pow)
    if args.cut_method == 'updated':
        return MakeCutoutsUpdate(cut_size, args.cutn, cut_pow=args.cut_pow)
    if args.cut_method == 'nrupdated':
        return MakeCutoutsNRUpdate(cut_size, args.cutn, cut_pow=args.cut_pow)
    return MakeCutoutsPoolingUpdate(cut_size, args.cutn, cut_pow=args.cut_pow)


def _update_latent_geometry():
    global toksX, toksY, sideX, sideY
    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f


def _visual_input_size(visual):
    size = getattr(visual, 'input_resolution', None)
    if size is None:
        size = getattr(visual, 'image_size', 224)
    if isinstance(size, (list, tuple)):
        return int(size[0])
    return int(size)


def _visual_output_dim(visual):
    return int(getattr(visual, 'output_dim', 512))


def clip_tokenize(texts):
    if isinstance(texts, str):
        texts = [texts]
    if args.clip_backend == 'openai':
        return clip.tokenize(texts).to(device)
    return clip_tokenizer(texts).to(device)


def encode_text_prompt(txt):
    return perceptor.encode_text(clip_tokenize(txt)).float()


def reload_clip_and_cutouts():
    """Reload CLIP and cutouts after clip_model, cutn, or size change. VQGAN stays loaded."""
    global perceptor, make_cutouts, clip_tokenizer

    if args is None or model is None:
        raise RuntimeError("load_models() must be called before reload_clip_and_cutouts()")

    if args.clip_backend == 'openai':
        jit = True if "1.7.1" in torch.__version__ else False
        print(f"Loading CLIP OpenAI ({args.clip_model})...")
        perceptor = clip.load(args.clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)
        clip_tokenizer = None
    else:
        pretrained_tag = args.clip_pretrained or OPENCLIP_DEFAULT_PRETRAINED.get(args.clip_model)
        if not pretrained_tag:
            raise ValueError(
                f"Unknown OpenCLIP model '{args.clip_model}'. "
                "Set --clip-pretrained to a valid tag (see open_clip.list_pretrained())."
            )
        pretrained = resolve_openclip_pretrained(args.clip_model, pretrained_tag)
        if os.path.isfile(pretrained):
            print(f"Loading OpenCLIP ({args.clip_model}) from local:")
            print(f"  {pretrained}")
        else:
            print(f"Loading OpenCLIP ({args.clip_model}, {pretrained})...")
            print(f"  (HF cache: {_HF_CACHE} — run download_openclip_local.py once for offline)")
        perceptor, _, _ = open_clip.create_model_and_transforms(
            args.clip_model, pretrained=pretrained, device=device)
        perceptor.eval().requires_grad_(False)
        clip_tokenizer = open_clip.get_tokenizer(args.clip_model)

    cut_size = _visual_input_size(perceptor.visual)
    make_cutouts = _create_make_cutouts(cut_size)
    _update_latent_geometry()


def load_models():
    """Load VQGAN, CLIP, and cutouts once. Call after init_args()."""
    global device, model, gumbel, f
    global e_dim, n_toks, z_min, z_max, normalize

    if args is None:
        raise RuntimeError("init_args() must be called before load_models()")

    device = torch.device(args.cuda_device)
    print("Loading VQGAN...")
    model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
    f = 2**(model.decoder.num_resolutions - 1)
    reload_clip_and_cutouts()

    if gumbel:
        e_dim = 256
        n_toks = model.quantize.n_embed
        z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
    else:
        e_dim = model.quantize.e_dim
        n_toks = model.quantize.n_e
        z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])


def _init_latent():
    """Create a fresh latent z for a new image."""
    global z, z_orig

    if args.init_image:
        if 'http' in args.init_image:
            img = Image.open(urlopen(args.init_image))
        else:
            img = Image.open(args.init_image)
        pil_image = img.convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    elif args.init_noise == 'pixels':
        img = random_noise_image(args.size[0], args.size[1])
        pil_image = img.convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    elif args.init_noise == 'gradient':
        img = random_gradient_image(args.size[0], args.size[1])
        pil_image = img.convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        if gumbel:
            z = one_hot @ model.quantize.embed.weight
        else:
            z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

    z_orig = z.clone()
    z.requires_grad_(True)


def _build_prompts():
    global pMs
    pMs = []

    if args.prompts:
        for prompt in args.prompts:
            txt, weight, stop = split_prompt(prompt)
            embed = encode_text_prompt(txt)
            pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.image_prompts:
        path, weight, stop = split_prompt(prompt)
        img = Image.open(path)
        pil_image = img.convert('RGB')
        img = resize_image(pil_image, (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, _visual_output_dim(perceptor.visual)]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))


def setup_generation(text_prompt=None, seed=None):
    """Reset z, prompts, and optimizer for one image. Models must already be loaded."""
    global opt, i

    if text_prompt is not None:
        if isinstance(text_prompt, str):
            args.prompts = [p.strip() for p in text_prompt.split("|") if p.strip()]
        else:
            args.prompts = list(text_prompt)

    if seed is None:
        seed = torch.seed() if args.seed is None else args.seed
    else:
        seed = int(seed)
    torch.manual_seed(seed)
    print('Using seed:', seed)

    _init_latent()
    _build_prompts()
    opt = get_opt(args.optimiser, args.step_size)
    i = 0


def get_opt(opt_name, opt_lr):
    if opt_name == "Adam":
        opt = optim.Adam([z], lr=opt_lr)	# LR=0.1 (Default)
    elif opt_name == "AdamW":
        opt = optim.AdamW([z], lr=opt_lr)	
    elif opt_name == "Adagrad":
        opt = optim.Adagrad([z], lr=opt_lr)	
    elif opt_name == "Adamax":
        opt = optim.Adamax([z], lr=opt_lr)	
    elif opt_name == "DiffGrad":
        opt = DiffGrad([z], lr=opt_lr, eps=1e-9, weight_decay=1e-9) # NR: Playing for reasons
    elif opt_name == "AdamP":
        opt = AdamP([z], lr=opt_lr)		    
    elif opt_name == "RAdam":
        opt = optim.RAdam([z], lr=opt_lr)		    
    elif opt_name == "RMSprop":
        opt = optim.RMSprop([z], lr=opt_lr)
    elif opt_name == "Lion":
        opt = Lion([z], lr=opt_lr)
    else:
        print("Unknown optimiser. Are choices broken?")
        opt = optim.Adam([z], lr=opt_lr)
    return opt


def _print_run_info():
    print('Using device:', device)
    print('Optimising using:', args.optimiser)
    if args.prompts:
        print('Using text prompts:', args.prompts)
    if args.image_prompts:
        print('Using image prompts:', args.image_prompts)
    if args.init_image:
        print('Using initial image:', args.init_image)
    if args.noise_prompt_weights:
        print('Noise prompt weights:', args.noise_prompt_weights)


# Vector quantize
def synth(z):
    if gumbel:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
    else:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


#@torch.no_grad()
@torch.inference_mode()
def checkin(i, losses):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
    out = synth(z)
    info = PngImagePlugin.PngInfo()
    png_meta = getattr(args, 'png_metadata', None) or {}
    comment = png_meta.get('comment', f'{args.prompts}')
    info.add_text('comment', comment)
    if png_meta.get('blobdream'):
        info.add_text('blobdream', png_meta['blobdream'])
    TF.to_pil_image(out[0].cpu()).save(args.output, pnginfo=info) 	


def ascend_txt():
    global i
    out = synth(z)
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
    
    result = []

    if args.init_weight:
        # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
        result.append(F.mse_loss(z, torch.zeros_like(z_orig)) * ((1/torch.tensor(i*2 + 1))*args.init_weight) / 2)

    for prompt in pMs:
        result.append(prompt(iii))
    
    if args.make_video:    
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
        img = np.transpose(img, (1, 2, 0))
        imageio.imwrite('./steps/' + str(i) + '.png', np.array(img))

    return result # return loss


def train(i):
    opt.zero_grad(set_to_none=True)
    lossAll = ascend_txt()
    
    if i % args.display_freq == 0:
        checkin(i, lossAll)
       
    loss = sum(lossAll)
    loss.backward()
    opt.step()
    
    #with torch.no_grad():
    with torch.inference_mode():
        z.copy_(z.maximum(z_min).minimum(z_max))



_last_zoom_frame = 0


def run_training():
    """Run the optimisation loop until max_iterations (or video modes)."""
    global i, z, z_orig, opt, pMs, this_video_frame, make_styled_video, _last_zoom_frame

    j = 0
    p = 1
    i = 0
    this_video_frame = 0
    make_styled_video = False

    try:
        with tqdm() as pbar:
            while True:
                if args.make_zoom_video:
                    if i % args.zoom_frequency == 0:
                        out = synth(z)
                        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
                        img = np.transpose(img, (1, 2, 0))
                        imageio.imwrite('./steps/' + str(j) + '.png', np.array(img))
                        if args.zoom_start <= i:
                            pil_image = Image.fromarray(np.array(img).astype('uint8'), 'RGB')
                            if args.zoom_scale != 1:
                                pil_image_zoom = zoom_at(pil_image, sideX / 2, sideY / 2, args.zoom_scale)
                            else:
                                pil_image_zoom = pil_image
                            if args.zoom_shift_x or args.zoom_shift_y:
                                pil_image_zoom = ImageChops.offset(pil_image_zoom, args.zoom_shift_x, args.zoom_shift_y)
                            pil_tensor = TF.to_tensor(pil_image_zoom)
                            z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
                            z_orig = z.clone()
                            z.requires_grad_(True)
                            opt = get_opt(args.optimiser, args.step_size)
                        j += 1

                if args.prompt_frequency > 0:
                    if i % args.prompt_frequency == 0 and i > 0:
                        if p >= len(all_phrases):
                            p = 0
                        pMs = []
                        args.prompts = all_phrases[p]
                        print(args.prompts)
                        for prompt in args.prompts:
                            txt, weight, stop = split_prompt(prompt)
                            embed = encode_text_prompt(txt)
                            pMs.append(Prompt(embed, weight, stop).to(device))
                        p += 1

                try:
                    import blobvision_cancel
                    if blobvision_cancel.is_requested():
                        tqdm.write("Generation aborted.")
                        if i > 0:
                            checkin(i, [torch.tensor(0.)])
                        break
                except ImportError:
                    pass

                train(i)

                if i == args.max_iterations:
                    if not args.video_style_dir:
                        break
                    if this_video_frame == (num_video_frames - 1):
                        make_styled_video = True
                        break
                    this_video_frame += 1
                    i = -1
                    pbar.reset()
                    args.init_image = video_frame_list[this_video_frame]
                    print("Next frame: ", args.init_image)
                    if args.seed is None:
                        seed = torch.seed()
                    else:
                        seed = args.seed
                    torch.manual_seed(seed)
                    print("Seed: ", seed)
                    filename = os.path.basename(args.init_image)
                    args.output = os.path.join(cwd, "steps", filename)
                    img = Image.open(args.init_image)
                    pil_image = img.convert('RGB')
                    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
                    pil_tensor = TF.to_tensor(pil_image)
                    z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
                    z_orig = z.clone()
                    z.requires_grad_(True)
                    opt = get_opt(args.optimiser, args.step_size)

                i += 1
                pbar.update()
    except KeyboardInterrupt:
        pass

    _last_zoom_frame = j


def make_video_output():
    global i
    if not (args.make_video or args.make_zoom_video):
        return
    init_frame = 1
    if args.make_zoom_video:
        last_frame = _last_zoom_frame
    else:
        last_frame = i  # This will raise an error if that number of frames does not exist.

    length = args.video_length # Desired time of the video in seconds

    min_fps = 10
    max_fps = 60

    total_frames = last_frame-init_frame

    frames = []
    tqdm.write('Generating video...')
    for i in range(init_frame,last_frame):
        temp = Image.open("./steps/"+ str(i) +'.png')
        keep = temp.copy()
        frames.append(keep)
        temp.close()
    
    if args.output_video_fps > 9:
        # Hardware encoding and video frame interpolation
        print("Creating interpolated frames...")
        ffmpeg_filter = f"minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps={args.output_video_fps}'"
        output_file = re.compile('\.png$').sub('.mp4', args.output)
        try:
            p = Popen(['ffmpeg',
                       '-y',
                       '-f', 'image2pipe',
                       '-vcodec', 'png',
                       '-r', str(args.input_video_fps),               
                       '-i',
                       '-',
                       '-b:v', '10M',
                       '-vcodec', 'h264_nvenc',
                       '-pix_fmt', 'yuv420p',
                       '-strict', '-2',
                       '-filter:v', f'{ffmpeg_filter}',
                       '-metadata', f'comment={args.prompts}',
                   output_file], stdin=PIPE)
        except FileNotFoundError:
            print("ffmpeg command failed - check your installation")
        for im in tqdm(frames):
            im.save(p.stdin, 'PNG')
        p.stdin.close()
        p.wait()
    else:
        # CPU
        fps = np.clip(total_frames/length,min_fps,max_fps)
        output_file = re.compile('\.png$').sub('.mp4', args.output)
        try:
            p = Popen(['ffmpeg',
                       '-y',
                       '-f', 'image2pipe',
                       '-vcodec', 'png',
                       '-r', str(fps),
                       '-i',
                       '-',
                       '-vcodec', 'libx264',
                       '-r', str(fps),
                       '-pix_fmt', 'yuv420p',
                       '-crf', '17',
                       '-preset', 'veryslow',
                       '-metadata', f'comment={args.prompts}',
                       output_file], stdin=PIPE)
        except FileNotFoundError:
            print("ffmpeg command failed - check your installation")        
        for im in tqdm(frames):
            im.save(p.stdin, 'PNG')
        p.stdin.close()
        p.wait()


def main(argv=None):
    init_args(argv)
    load_models()
    _print_run_info()
    setup_generation()
    run_training()
    make_video_output()


if __name__ == '__main__':
    main()
