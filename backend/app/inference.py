"""ONNX Runtime model registry and pre/post-processing."""
import base64
import io
import json
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

SIZE = 128
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
CLASSES = ["clean", "salt_pepper", "blur", "occlusion"]
BRANCHES = ["identity", "salt_pepper", "blur", "occlusion"]

MODEL_FILES = {
    "universal_ae": "universal_ae.onnx",
    "classifier": "classifier.onnx",
    "specialist_salt_pepper": "specialist_salt_pepper.onnx",
    "specialist_blur": "specialist_blur.onnx",
    "specialist_occlusion": "specialist_occlusion.onnx",
    "soft_moe": "soft_moe.onnx",
    "sketch_generator": "sketch_generator.onnx",
}


class Registry:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.sessions, self.errors, self.meta = {}, {}, {}
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for key, fn in MODEL_FILES.items():
            path = os.path.join(model_dir, fn)
            if not os.path.exists(path):
                self.errors[key] = f"missing file {fn}"
                continue
            try:
                t0 = time.perf_counter()
                s = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
                self.sessions[key] = s
                self.meta[key] = {"file": fn, "size_mb": round(os.path.getsize(path) / 2**20, 2),
                                  "load_ms": round((time.perf_counter() - t0) * 1000, 1),
                                  "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in s.get_inputs()],
                                  "outputs": [o.name for o in s.get_outputs()]}
            except Exception as e:   # corrupted / incompatible file
                self.errors[key] = str(e)
        vp = os.path.join(model_dir, "onnx_verification.json")
        self.verification = json.load(open(vp)) if os.path.exists(vp) else []

    def require(self, *keys):
        missing = [k for k in keys if k not in self.sessions]
        if missing:
            raise KeyError(", ".join(f"{k} ({self.errors.get(k, 'not loaded')})" for k in missing))

    def run(self, key, feeds):
        t0 = time.perf_counter()
        out = self.sessions[key].run(None, feeds)
        return out, (time.perf_counter() - t0) * 1000


# ------------------------------------------------------------------ image helpers
def decode_image(raw: bytes, square_crop=False) -> np.ndarray:
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if square_crop:
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    arr = np.asarray(img)
    return cv2.resize(arr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def to_nchw(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def from_nchw(x: np.ndarray) -> np.ndarray:
    return (np.clip(x[0].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)


def png_data_url(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def error_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    err = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(2)
    err = np.clip(err / 128.0 * 255, 0, 255).astype(np.uint8)          # 0.5 abs error -> full scale
    return cv2.cvtColor(cv2.applyColorMap(err, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def ssim(a, b):
    """Gaussian-window SSIM (11x11, sigma 1.5), averaged over channels."""
    a, b = a.astype(np.float64) / 255, b.astype(np.float64) / 255
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    saa, sbb, sab = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return float(m[5:-5, 5:-5].mean())
