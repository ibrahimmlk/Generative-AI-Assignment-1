"""Runtime image corruptions used by Tasks 1-3.

All functions work on uint8 RGB numpy arrays of shape (H, W, 3).
A corruption is fully described by a small JSON-serialisable dict ("params"),
so the same code serves dynamic training, the deterministic manifests and the
web application.
"""
import cv2
import numpy as np

CLASSES = ["clean", "salt_pepper", "blur", "occlusion"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Fixed test severities required by the assignment.
TEST_SEVERITIES = {
    "salt_pepper": {"low": {"prob": 0.03}, "medium": {"prob": 0.08}, "high": {"prob": 0.15}},
    "blur": {"low": {"ksize": 3, "sigma": 0.7}, "medium": {"ksize": 5, "sigma": 1.5},
             "high": {"ksize": 7, "sigma": 2.5}},
    "occlusion": {"low": {"n_rects": 1, "coverage": 0.10}, "medium": {"n_rects": 2, "coverage": 0.20},
                  "high": {"n_rects": 3, "coverage": 0.35}},
}


# ---------------------------------------------------------------- primitives
def salt_pepper(img, prob, seed):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    hit = rng.random((h, w)) < prob
    white = rng.random((h, w)) < 0.5          # black or white with equal probability
    out = img.copy()
    out[hit & white] = 255
    out[hit & ~white] = 0
    return out


def gaussian_blur(img, ksize, sigma):
    return cv2.GaussianBlur(img, (int(ksize), int(ksize)), sigmaX=float(sigma), sigmaY=float(sigma))


def occlude(img, rects):
    out = img.copy()
    for x, y, w, h in rects:
        out[y:y + h, x:x + w] = 0
    return out


def _overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def sample_rects(rng, n_rects, coverage, size=128, tol=0.01, max_tries=500):
    """Sample n non-overlapping rectangles whose total area ~= coverage * size^2.

    Non-overlap makes the joint coverage equal to the sum of the areas, so the
    requested fraction is met exactly up to integer rounding (checked by tol).
    """
    total = coverage * size * size
    for _ in range(max_tries):
        shares = rng.dirichlet(np.ones(n_rects)) if n_rects > 1 else np.array([1.0])
        shares = 0.5 / n_rects + 0.5 * shares          # avoid tiny slivers
        rects = []
        ok = True
        for s in shares:
            area = s * total
            ar = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))   # aspect ratio w/h
            w = int(round(np.sqrt(area * ar)))
            h = int(round(area / max(w, 1)))
            w, h = min(max(w, 4), size), min(max(h, 4), size)
            placed = False
            for _ in range(100):
                x = int(rng.integers(0, size - w + 1))
                y = int(rng.integers(0, size - h + 1))
                r = (x, y, w, h)
                if not any(_overlaps(r, q) for q in rects):
                    rects.append(r)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            actual = sum(w * h for _, _, w, h in rects) / (size * size)
            if abs(actual - coverage) <= tol:
                return rects, actual
    raise RuntimeError("could not place occlusion rectangles")


# ---------------------------------------------------------------- params
def sample_train_params(kind, rng, size=128):
    """Sample a random severity following the training configuration table."""
    if kind == "clean":
        return {"type": "clean"}
    if kind == "salt_pepper":
        return {"type": kind, "prob": float(rng.uniform(0.02, 0.15)), "seed": int(rng.integers(2**31))}
    if kind == "blur":
        return {"type": kind, "ksize": int(rng.choice([3, 5, 7])), "sigma": float(rng.uniform(0.5, 2.5))}
    if kind == "occlusion":
        n = int(rng.integers(1, 4))
        target = float(rng.uniform(0.11, 0.34))   # +-0.01 tolerance keeps it inside [0.10, 0.35]
        rects, actual = sample_rects(rng, n, target, size)
        return {"type": kind, "n_rects": n, "coverage": actual, "rects": [list(r) for r in rects]}
    raise ValueError(kind)


def fixed_params(kind, severity, rng, size=128):
    """Fixed test-time severity (low / medium / high)."""
    if kind == "clean":
        return {"type": "clean", "severity": "none"}
    cfg = dict(TEST_SEVERITIES[kind][severity])
    p = {"type": kind, "severity": severity, **cfg}
    if kind == "salt_pepper":
        p["seed"] = int(rng.integers(2**31))
    elif kind == "occlusion":
        rects, actual = sample_rects(rng, cfg["n_rects"], cfg["coverage"], size)
        p["rects"] = [list(r) for r in rects]
        p["coverage"] = actual
    return p


def apply(img, params):
    t = params["type"]
    if t == "clean":
        return img.copy()
    if t == "salt_pepper":
        return salt_pepper(img, params["prob"], params["seed"])
    if t == "blur":
        return gaussian_blur(img, params["ksize"], params["sigma"])
    if t == "occlusion":
        return occlude(img, params["rects"])
    raise ValueError(t)


def severity_bucket(params):
    """Map any (possibly random) severity to low / medium / high for reporting."""
    t = params["type"]
    if t == "clean":
        return "none"
    if "severity" in params:
        return params["severity"]
    v = {"salt_pepper": params.get("prob"), "blur": params.get("sigma"), "occlusion": params.get("coverage")}[t]
    lo, hi = {"salt_pepper": (0.0567, 0.1033), "blur": (1.167, 1.833), "occlusion": (0.183, 0.267)}[t]
    return "low" if v < lo else ("medium" if v < hi else "high")
