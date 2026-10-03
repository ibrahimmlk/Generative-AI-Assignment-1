"""Download the trained ONNX models into ./models (standard library only, works on any OS).

    python scripts/download_models.py
"""
import json
import os
import sys
import urllib.request

RELEASE = "https://github.com/ibrahimmlk/Generative-AI-Assignment-1/releases/download/v1.0"
FILES = ["universal_ae.onnx", "classifier.onnx", "specialist_salt_pepper.onnx", "specialist_blur.onnx",
         "specialist_occlusion.onnx", "soft_moe.onnx", "sketch_generator.onnx", "onnx_verification.json"]
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _progress(name):
    def hook(blocks, block_size, total):
        done = blocks * block_size
        pct = f"{100 * done / total:5.1f}%" if total > 0 else ""
        print(f"\r[get ] {name}: {done / 2**20:6.1f} MB {pct}", end="", flush=True)
    return hook


def main():
    os.makedirs(DEST, exist_ok=True)
    for f in FILES:
        path = os.path.join(DEST, f)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"[skip] {f} already present")
            continue
        try:
            urllib.request.urlretrieve(f"{RELEASE}/{f}", path + ".part", reporthook=_progress(f))
            os.replace(path + ".part", path)
            print(f"\r[ ok ] {f}: {os.path.getsize(path) / 2**20:.1f} MB" + " " * 20)
        except Exception as e:
            print("FAILED:", e)
            sys.exit(1)
    ver = json.load(open(os.path.join(DEST, "onnx_verification.json")))
    print("ONNX verification (PyTorch vs ONNX Runtime):")
    for v in ver:
        print(f"  {v['file']:32s} max|diff| = {v['max_abs_diff']:.2e}  {'OK' if v['passed'] else 'CHECK'}")


if __name__ == "__main__":
    main()
