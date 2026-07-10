#!/usr/bin/env python3
"""BlobVision HTTP API   Legacy / Redux / Corrupt generation."""
import argparse
import os
import shutil
import sys
import uuid
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from blobvision_paths import BLOBVISION_ROOT, VENV_PYTHON
BLOBDREAM_ROOT = BLOBVISION_ROOT

os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from blobvision_engine import (
    DEFAULT_DENOISE,
    DEFAULT_ITERATIONS,
    BlobVisionEngine,
    resolve_seed,
)
from blobvision_meta import read_metadata_from_image, seed_from_metadata

app = FastAPI(
    title="BlobVision API",
    description="VQGAN+CLIP renderer   Legacy, Redux, Corrupt",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: Optional[BlobVisionEngine] = None


class ModeInfo(BaseModel):
    name: str
    description: str
    default_iterations: int
    default_denoise_fidelity: Optional[float] = None
    needs_init_image: bool


class GenerateResponse(BaseModel):
    ok: bool = True
    mode: str
    prompt: str
    seed: int
    iterations: int
    denoise_fidelity: Optional[float] = None
    output_url: str
    sketch_url: Optional[str] = None
    metadata: dict


def get_engine() -> BlobVisionEngine:
    global _engine
    if _engine is None:
        raise HTTPException(503, "Engine not started. Call POST /warmup first.")
    return _engine


MODE_HELP = {
    "legacy": "Classic VQGAN+CLIP text2img (500 it default). Optional init image for img2img.",
    "redux": "SDXL Turbo sketch then VQGAN — fast modern path.",
}


@app.get("/")
def root():
    return {
        "name": "BlobVision API",
        "modes": list(MODE_HELP.keys()),
        "docs": "/docs",
    }




@app.get("/docs.")
def docs_trailing_dot():
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico")
def favicon():
    return JSONResponse(status_code=204, content=None)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/modes")
def modes():
    items = []
    for name in ("legacy", "redux"):
        items.append(ModeInfo(
            name=name,
            description=MODE_HELP[name],
            default_iterations=DEFAULT_ITERATIONS[name],
            default_denoise_fidelity=DEFAULT_DENOISE.get(name),
            needs_init_image=(name == "legacy"),
        ))
    return items


@app.get("/status")
def status():
    if _engine is None:
        return {"started": False}
    s = _engine.status()
    s["started"] = True
    return s


@app.post("/warmup")
def warmup(mode: str = "redux"):
    global _engine
    mode = mode.lower().strip()
    if mode not in MODE_HELP:
        raise HTTPException(400, "Unknown mode: {}".format(mode))
    if _engine is None:
        _engine = BlobVisionEngine(keep_models=True)
    _engine.warmup(mode=mode)
    return {"ok": True, "warmed": mode, "status": _engine.status()}


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    mode: str = Form(...),
    prompt: str = Form(...),
    iterations: Optional[int] = Form(None),
    denoise_fidelity: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    sketch_steps: int = Form(4),
    width: int = Form(384),
    height: int = Form(384),
    init_image: Optional[UploadFile] = File(None),
    reuse_seed_from_image: Optional[UploadFile] = File(None),
):
    engine = get_engine()
    mode = mode.lower().strip()
    if mode not in MODE_HELP:
        raise HTTPException(400, "Unknown mode: {}".format(mode))

    if reuse_seed_from_image is not None:
        tmp = os.path.join(engine.output_dir, "uploads", uuid.uuid4().hex + ".png")
        with open(tmp, "wb") as fh:
            shutil.copyfileobj(reuse_seed_from_image.file, fh)
        try:
            inherited = seed_from_metadata(read_metadata_from_image(tmp))
            if inherited is not None:
                seed = inherited
        finally:
            if os.path.isfile(tmp):
                os.remove(tmp)

    init_path = None
    if init_image is not None:
        ext = os.path.splitext(init_image.filename or "upload.png")[1] or ".png"
        init_path = os.path.join(
            engine.output_dir, "uploads", uuid.uuid4().hex + ext,
        )
        with open(init_path, "wb") as fh:
            shutil.copyfileobj(init_image.file, fh)

    try:
        result = engine.generate(
            mode=mode,
            prompt=prompt,
            iterations=iterations,
            denoise_fidelity=denoise_fidelity,
            seed=seed,
            init_image_path=init_path,
            sketch_steps=sketch_steps,
            width=width,
            height=height,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    base = "/outputs/"
    out_name = os.path.basename(result.output_path)
    sketch_url = None
    if result.sketch_path:
        sketch_url = base + os.path.basename(result.sketch_path)

    return GenerateResponse(
        mode=result.mode,
        prompt=result.prompt,
        seed=result.seed,
        iterations=result.iterations,
        denoise_fidelity=result.denoise_fidelity,
        output_url=base + out_name,
        sketch_url=sketch_url,
        metadata=result.metadata,
    )


@app.post("/meta/read")
async def meta_read(image: UploadFile = File(...)):
    tmp = os.path.join(SCRIPT_DIR, "outputs", "uploads", uuid.uuid4().hex + ".png")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "wb") as fh:
        shutil.copyfileobj(image.file, fh)
    try:
        meta = read_metadata_from_image(tmp)
        if meta is None:
            raise HTTPException(404, "No BlobDream metadata in image")
        return {"metadata": meta, "seed": seed_from_metadata(meta)}
    finally:
        if os.path.isfile(tmp):
            os.remove(tmp)


@app.get("/outputs/{filename}")
def get_output(filename: str):
    if _engine is None:
        raise HTTPException(503, "Engine not started")
    safe = os.path.basename(filename)
    path = os.path.join(_engine.output_dir, safe)
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="image/png")


def parse_args():
    p = argparse.ArgumentParser(description="BlobVision API server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--warmup", default="redux", choices=["legacy", "redux"])
    p.add_argument("--no-warmup", action="store_true")
    return p.parse_args()


def main():
    import uvicorn

    cli = parse_args()
    global _engine
    _engine = BlobVisionEngine(keep_models=True)
    if not cli.no_warmup:
        print("Warming up mode:", cli.warmup)
        _engine.warmup(mode=cli.warmup)
        print("Ready:", _engine.status())
    print("BlobVision API ready:")
    print("  Swagger UI: http://{}:{}/docs".format(cli.host, cli.port))
    uvicorn.run(app, host=cli.host, port=cli.port, log_level="info")


if __name__ == "__main__":
    main()
