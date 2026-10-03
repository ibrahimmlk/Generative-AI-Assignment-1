"""FastAPI backend: validates uploads, applies runtime corruptions, runs the ONNX models."""
import os
import platform
import sys
import time
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from genai import corruptions as C

from .inference import (BRANCHES, CLASSES, Registry, decode_image, error_map, from_nchw, png_data_url, psnr, ssim,
                        to_nchw)

SAMPLE_DIR = os.environ.get("SAMPLE_DIR", os.path.join(os.path.dirname(__file__), "..", "samples"))
MAX_BYTES = 10 * 2**20
ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}
STARTED = time.time()

app = FastAPI(title="GenAI Restoration Lab API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
registry = Registry()
if os.path.isdir(SAMPLE_DIR):
    app.mount("/api/sample-files", StaticFiles(directory=SAMPLE_DIR), name="samples")


# ------------------------------------------------------------------ helpers
async def read_input(file: Optional[UploadFile], sample: Optional[str], square_crop=False) -> np.ndarray:
    if file is not None and file.filename:
        if file.content_type not in ALLOWED:
            raise HTTPException(415, f"Unsupported file type {file.content_type}. Use PNG, JPEG, WEBP or BMP.")
        raw = await file.read()
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, "File larger than 10 MB.")
        try:
            return decode_image(raw, square_crop)
        except Exception:
            raise HTTPException(400, "The uploaded file could not be decoded as an image.")
    if sample:
        path = os.path.realpath(os.path.join(SAMPLE_DIR, sample))
        if not path.startswith(os.path.realpath(SAMPLE_DIR)) or not os.path.isfile(path):
            raise HTTPException(404, "Unknown sample image.")
        with open(path, "rb") as f:
            return decode_image(f.read(), square_crop)
    raise HTTPException(400, "Provide an image file or choose a sample.")


def corruption_params(corruption: str, severity: str, seed: Optional[int]):
    if corruption not in ("none", "clean", *C.CLASSES):
        raise HTTPException(400, f"Unknown corruption '{corruption}'.")
    if corruption in ("none", "clean"):
        return None
    if severity not in ("low", "medium", "high"):
        raise HTTPException(400, "Severity must be low, medium or high.")
    rng = np.random.default_rng(seed)
    return C.fixed_params(corruption, severity, rng)


def require(*keys):
    try:
        registry.require(*keys)
    except KeyError as e:
        raise HTTPException(503, f"Model not available: {e}. Download the ONNX models (see README).")


def restoration_payload(original, corrupted, restored, params, inference_ms, reference_known):
    p = {"input": png_data_url(corrupted), "output": png_data_url(restored), "inference_ms": round(inference_ms, 2),
         "corruption": params or {"type": "none (uploaded image used as-is)"}, "size": 128}
    if reference_known:
        p["original"] = png_data_url(original)
        p["error_map"] = png_data_url(error_map(restored, original))
        p["metrics"] = {"psnr_input": _num(psnr(corrupted, original)), "ssim_input": round(ssim(corrupted, original), 4),
                        "psnr_output": _num(psnr(restored, original)), "ssim_output": round(ssim(restored, original), 4)}
    return p


def _num(v):
    return None if v == float("inf") else round(v, 2)


async def prepare(file, sample, corruption, severity, seed):
    img = await read_input(file, sample)
    params = corruption_params(corruption, severity, seed)
    corrupted = C.apply(img, params) if params else img
    # a reference exists when we corrupted a known image ourselves (or a clean image is passed through)
    return img, corrupted, params


# ------------------------------------------------------------------ endpoints
@app.get("/api/health")
def health():
    return {"status": "ok" if registry.sessions else "degraded", "uptime_s": round(time.time() - STARTED, 1),
            "models_loaded": sorted(registry.sessions), "models_missing": registry.errors,
            "onnxruntime": ort.__version__, "providers": ort.get_available_providers()}


@app.get("/api/system")
def system():
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "onnxruntime": ort.__version__, "models": registry.meta, "missing": registry.errors,
            "onnx_verification": registry.verification, "input_size": [128, 128, 3],
            "corruption_presets": C.TEST_SEVERITIES, "classes": CLASSES, "branches": BRANCHES}


@app.get("/api/samples")
def samples():
    out = {}
    for kind in ("pets", "faces"):
        d = os.path.join(SAMPLE_DIR, kind)
        out[kind] = [f"{kind}/{f}" for f in sorted(os.listdir(d))] if os.path.isdir(d) else []
    return out


@app.post("/api/restore/universal")
async def universal(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                    corruption: str = Form("none"), severity: str = Form("medium"), seed: Optional[int] = Form(None)):
    require("universal_ae")
    img, corrupted, params = await prepare(file, sample, corruption, severity, seed)
    (out,), ms = registry.run("universal_ae", {"input": to_nchw(corrupted)})
    restored = from_nchw(out)
    return {"mode": "universal", **restoration_payload(img, corrupted, restored, params, ms, params is not None)}


@app.post("/api/restore/hard")
async def hard(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
               corruption: str = Form("none"), severity: str = Form("medium"), seed: Optional[int] = Form(None)):
    require("classifier", "specialist_salt_pepper", "specialist_blur", "specialist_occlusion")
    img, corrupted, params = await prepare(file, sample, corruption, severity, seed)
    x = to_nchw(corrupted)
    (probs, _), ms_cls = registry.run("classifier", {"input": x})
    probs = probs[0]
    k = int(probs.argmax())
    if k == 0:                       # identity bypass: clean input is not processed by an expert
        restored, ms_exp, expert = corrupted.copy(), 0.0, "identity bypass"
    else:
        (out,), ms_exp = registry.run(f"specialist_{CLASSES[k]}", {"input": x})
        restored, expert = from_nchw(out), f"{CLASSES[k]} specialist"
    payload = restoration_payload(img, corrupted, restored, params, ms_cls + ms_exp, params is not None)
    payload.update({"mode": "hard", "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
                    "predicted": CLASSES[k], "expert": expert,
                    "timing_ms": {"classifier": round(ms_cls, 2), "expert": round(ms_exp, 2)},
                    "true_corruption": params["type"] if params else None})
    return payload


@app.post("/api/restore/moe")
async def moe(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
              corruption: str = Form("none"), severity: str = Form("medium"), seed: Optional[int] = Form(None)):
    require("soft_moe")
    img, corrupted, params = await prepare(file, sample, corruption, severity, seed)
    (out, w), ms = registry.run("soft_moe", {"input": to_nchw(corrupted)})
    w = w[0]
    restored = from_nchw(out)
    payload = restoration_payload(img, corrupted, restored, params, ms, params is not None)
    order = np.argsort(-w)
    payload.update({"mode": "moe", "weights": {b: round(float(v), 4) for b, v in zip(BRANCHES, w)},
                    "dominant": BRANCHES[int(order[0])], "ranking": [BRANCHES[int(i)] for i in order],
                    "entropy": round(float(-(w * np.log(w + 1e-9)).sum()), 4),
                    "true_corruption": params["type"] if params else None})
    return payload


@app.post("/api/sketch")
async def sketch(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None), style: int = Form(1)):
    require("sketch_generator")
    if style not in (1, 2, 3):
        raise HTTPException(400, "Style must be 1, 2 or 3.")
    img = await read_input(file, sample, square_crop=True)
    (out,), ms = registry.run("sketch_generator", {"photo": to_nchw(img), "style": np.array([style - 1], np.int64)})
    return {"mode": "sketch", "photo": png_data_url(img), "sketch": png_data_url(from_nchw(out)), "style": style,
            "inference_ms": round(ms, 2), "size": 128}
