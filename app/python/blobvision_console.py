#!/usr/bin/env python3
"""BlobVision - VQGAN+CLIP console (fast / corrupt img2img / legacy)."""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
BLOBVISION_ROOT = os.path.dirname(APP_DIR)
BLOBDREAM_ROOT = BLOBVISION_ROOT
VENV_PYTHON = os.path.join(BLOBVISION_ROOT, "venv", "Scripts", "python.exe")
MAX_ITERATIONS = 500

os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from blobvision_paths import TAMING_REPO

for path in (SCRIPT_DIR, TAMING_REPO, BLOBVISION_ROOT):
    if path not in sys.path and os.path.isdir(path):
        sys.path.insert(0, path)

PROFILES = {
    "corrupt": {
        "iterations": 20,
        "save_every": 20,
        "cutn": 4,
        "init_weight": 0.0,
        "clip_model": "ViT-B/16",
        "clip_pretrained": None,
        "clip_backend": "openai",
        "optimiser": "Adam",
        "size": None,
        "needs_init": True,
    },
    "fast": {
        "iterations": 100,
        "save_every": 100,
        "cutn": 4,
        "init_weight": 0.0,
        "clip_model": "ViT-B/16",
        "clip_pretrained": None,
        "clip_backend": "openai",
        "optimiser": "Adam",
        "size": None,
        "needs_init": False,
    },
    "legacy": {
        "iterations": 500,
        "save_every": 50,
        "cutn": 32,
        "init_weight": 0.0,
        "clip_model": "ViT-B/32",
        "clip_pretrained": None,
        "clip_backend": "openai",
        "optimiser": "Adam",
        "size": None,
        "needs_init": False,
    },
}


def _check_environment():
    exe = os.path.normcase(os.path.abspath(sys.executable))
    expected = os.path.normcase(os.path.abspath(VENV_PYTHON))
    if os.path.isfile(VENV_PYTHON) and exe != expected:
        print("BlobVision: mauvais interpreteur Python.")
        print("  Utilise :", exe)
        print("  Attendu :", expected)
        print(r"  ..\venv\Scripts\python.exe blobvision_console.py")
        sys.exit(1)
    try:
        import omegaconf  # noqa: F401
    except ImportError:
        print("omegaconf manquant:", sys.executable)
        sys.exit(1)


_check_environment()
import generate as eng


def profile_argv(profile_name, console_args, iterations=None):
    p = PROFILES[profile_name]
    it = iterations if iterations is not None else p["iterations"]
    argv = [
        "-i", str(it),
        "-se", str(max(1, it)),
        "-cuts", str(p["cutn"]),
        "-m", p["clip_model"],
        "--clip-backend", p.get("clip_backend", "openai"),
        "-opt", p["optimiser"],
        "-cd", console_args.cuda_device,
        "-p", "placeholder",
    ]
    if p.get("clip_pretrained"):
        argv.extend(["--clip-pretrained", p["clip_pretrained"]])
    if p.get("size"):
        argv.extend(["-s", str(p["size"][0]), str(p["size"][1])])
    if console_args.seed is not None:
        argv.extend(["-sd", str(console_args.seed)])
    if console_args.learning_rate is not None:
        argv.extend(["-lr", str(console_args.learning_rate)])
    return argv


def clamp_iterations(n):
    n = int(n)
    if n < 0:
        raise ValueError("iterations must be >= 0")
    if n > MAX_ITERATIONS:
        print("Warning: iterations capped at {}".format(MAX_ITERATIONS))
        return MAX_ITERATIONS
    return n


def apply_profile(profile_name, loaded_clip_model, iterations, init_image, init_weight):
    p = PROFILES[profile_name]
    it = clamp_iterations(iterations)
    eng.args.max_iterations = it
    eng.args.display_freq = max(1, it)
    eng.args.cutn = p["cutn"]
    eng.args.clip_model = p["clip_model"]
    eng.args.clip_backend = p.get("clip_backend", "openai")
    eng.args.clip_pretrained = p.get("clip_pretrained")
    eng.args.optimiser = p["optimiser"]
    if p.get("size"):
        eng.args.size = list(p["size"])

    if p.get("needs_init"):
        if not init_image:
            print("Mode corrupt requires -ii / --init-image (sketch SDXL, etc.)")
            sys.exit(1)
        eng.args.init_image = init_image
        eng.args.init_weight = float(init_weight)
    else:
        eng.args.init_image = init_image if init_image else None
        eng.args.init_weight = float(init_weight) if init_image else 0.0

    state_key = (
        p["clip_model"],
        p["cutn"],
        tuple(eng.args.size),
        eng.args.clip_backend,
        eng.args.clip_pretrained,
    )
    if loaded_clip_model[0] != state_key:
        eng.reload_clip_and_cutouts()
        loaded_clip_model[:] = [state_key]
    return profile_name, it


def describe_profile(name, iterations=None):
    p = PROFILES[name]
    it = iterations if iterations is not None else p["iterations"]
    size = p["size"] or "auto (VRAM)"
    extra = ", img2img iw={}".format(p["init_weight"]) if p.get("needs_init") else ""
    return "{} it, {} cuts, CLIP {}, {}{}, {}px".format(
        it, p["cutn"], p["clip_model"], p["optimiser"], extra, size
    )


def ask_profile(has_init):
    print("Mode [Entree=corrupt / 1=fast / 2=legacy] > ", end="", flush=True)
    choice = input().strip().lower()
    if choice == "2":
        return "legacy"
    if choice == "1":
        return "fast"
    return "corrupt" if has_init else "fast"


def ask_iterations(default_it):
    print("Iterations [{}] (0-{}) > ".format(default_it, MAX_ITERATIONS), end="", flush=True)
    raw = input().strip()
    if not raw:
        return default_it
    return clamp_iterations(raw)


def parse_console_args():
    parser = argparse.ArgumentParser(description="BlobVision - VQGAN+CLIP prompt loop")
    parser.add_argument("--legacy", action="store_true", help="Session legacy (500 it)")
    parser.add_argument("--fast", action="store_true", help="Session fast text2img (100 it)")
    parser.add_argument(
        "--corrupt", action="store_true",
        help="Session corrupt img2img (default 20 it, needs -ii)",
    )
    parser.add_argument("-ii", "--init-image", default=None, help="Sketch / init image for corrupt")
    parser.add_argument("-iw", "--init-weight", type=float, default=0.0)
    parser.add_argument(
        "-i", "--iterations", type=int, default=None,
        help="Override iterations (corrupt default: 20, fast: 100, legacy: 500, max 500)",
    )
    parser.add_argument("--no-ask-mode", action="store_true")
    parser.add_argument("--no-ask-iters", action="store_true", help="Do not ask iterations each image")
    parser.add_argument("-cd", "--cuda-device", type=str, default="cuda:0")
    parser.add_argument("-sd", "--seed", type=int, default=None)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.1)
    parser.add_argument("-o", "--output-dir", type=str, default=".")
    return parser.parse_args()


def resolve_default_profile(console_args):
    if console_args.legacy:
        return "legacy"
    if console_args.fast:
        return "fast"
    if console_args.corrupt or console_args.init_image:
        return "corrupt"
    return "fast"


def resolve_default_iterations(profile_name, console_args):
    if console_args.iterations is not None:
        return clamp_iterations(console_args.iterations)
    return PROFILES[profile_name]["iterations"]


def main():
    console_args = parse_console_args()
    init_image = os.path.abspath(console_args.init_image) if console_args.init_image else None
    if init_image and not os.path.isfile(init_image):
        print("Init image not found:", init_image)
        sys.exit(1)

    default_profile = resolve_default_profile(console_args)
    session_iterations = resolve_default_iterations(default_profile, console_args)

    eng.init_args(profile_argv(default_profile, console_args, session_iterations))
    print("Loading VQGAN...")
    eng.load_models()
    loaded = [(
        eng.args.clip_model,
        eng.args.cutn,
        tuple(eng.args.size),
        eng.args.clip_backend,
        eng.args.clip_pretrained,
    )]

    print("Ready.")
    print("  corrupt :", describe_profile("corrupt"))
    print("  fast    :", describe_profile("fast"))
    print("  legacy  :", describe_profile("legacy"))
    if init_image:
        print("  init    :", init_image)
    if console_args.no_ask_mode:
        print("  session : {} @ {} it".format(default_profile, session_iterations))
    print("  quit / exit / q pour sortir")
    print()

    counter = 1
    output_dir = os.path.abspath(console_args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    while True:
        try:
            prompt = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            break

        if console_args.no_ask_mode:
            profile = default_profile
        else:
            profile = ask_profile(init_image is not None)

        prof_default_it = PROFILES[profile]["iterations"]
        if console_args.iterations is not None:
            prof_default_it = session_iterations
        if console_args.no_ask_iters:
            iters = prof_default_it if profile == default_profile else PROFILES[profile]["iterations"]
            if console_args.iterations is not None:
                iters = session_iterations
        else:
            iters = ask_iterations(prof_default_it)

        profile, iters = apply_profile(
            profile, loaded, iters, init_image, console_args.init_weight,
        )
        print("Profil:", profile, "(" + describe_profile(profile, iters) + ")")

        output_path = os.path.join(output_dir, "output_{:04d}.png".format(counter))
        eng.args.output = output_path
        print("Generating...")
        eng.setup_generation(text_prompt=prompt, seed=console_args.seed)
        eng.run_training()
        print("Saved:", output_path)
        print()
        counter += 1


if __name__ == "__main__":
    main()