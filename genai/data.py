"""Dataset loading, splits, deterministic manifests and PyTorch datasets."""
import glob
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from . import corruptions as C

SIZE = 128
SEED = 42


# ============================================================ Oxford-IIIT Pet
def _resize_rgb(path, size=SIZE):
    img = Image.open(path).convert("RGB")
    img = np.asarray(img)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_pets(data_root, split):
    """Return (uint8 array N x 128 x 128 x 3, list of names) for 'trainval' or 'test'.

    Images are downloaded with torchvision once and cached as a compressed .npz
    so that every later run (and every Optuna trial) loads in a second.
    """
    cache = os.path.join(data_root, "cache", f"pets_{split}_{SIZE}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["images"], list(z["names"])
    from torchvision.datasets import OxfordIIITPet
    OxfordIIITPet(data_root, split=split, download=True)
    base = os.path.join(data_root, "oxford-iiit-pet")
    with open(os.path.join(base, "annotations", f"{split}.txt")) as f:
        names = [line.split()[0] for line in f if line.strip()]
    images = np.stack([_resize_rgb(os.path.join(base, "images", n + ".jpg")) for n in names])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, images=images, names=np.array(names))
    return images, names


def split_trainval(n, seed=SEED, train_frac=0.8):
    perm = np.random.default_rng(seed).permutation(n)
    k = int(round(train_frac * n))
    return np.sort(perm[:k]), np.sort(perm[k:])


def build_manifests(trainval_names, test_names, val_idx, out_dir, seed=SEED):
    """Create the deterministic validation and test corruption manifests (once)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    # validation: one condition per image, classes balanced, severity sampled from the training ranges
    labels = np.resize(np.arange(4), len(val_idx))
    rng.shuffle(labels)
    val = []
    for j, (idx, lab) in enumerate(zip(val_idx, labels)):
        s = int(seed * 100_000 + j)
        p = C.sample_train_params(C.CLASSES[lab], np.random.default_rng(s))
        val.append({"index": int(idx), "name": trainval_names[idx], "seed": s, "label": int(lab),
                    "severity": C.severity_bucket(p), "params": p})
    # test: every image x {clean + 3 corruptions x 3 fixed severities}
    test = []
    for idx, name in enumerate(test_names):
        conds = [("clean", None)] + [(k, s) for k in ("salt_pepper", "blur", "occlusion")
                                     for s in ("low", "medium", "high")]
        for c, (k, sev) in enumerate(conds):
            s = int(seed * 1_000_000 + idx * 16 + c)
            p = C.fixed_params(k, sev, np.random.default_rng(s))
            test.append({"index": idx, "name": name, "seed": s, "label": C.CLASS_TO_ID[k],
                         "severity": p["severity"], "params": p})
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({"seed": seed, "train_frac": 0.8, "val_indices": [int(i) for i in val_idx]}, f)
    with open(os.path.join(out_dir, "val_manifest.json"), "w") as f:
        json.dump(val, f)
    with open(os.path.join(out_dir, "test_manifest.json"), "w") as f:
        json.dump(test, f)
    return val, test


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)


class RuntimeCorruptionDataset(Dataset):
    """Training dataset: a new corruption type and severity is sampled every time an image is loaded.

    Indexing with an int samples the condition uniformly from `classes`;
    indexing with (image_index, class_id) (used by BalancedBatchSampler) forces the class.
    """

    def __init__(self, images, classes=(0, 1, 2, 3), flip=True):
        self.images = images
        self.classes = list(classes)
        self.flip = flip

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        rng = np.random.default_rng()   # fresh OS entropy -> different in every worker / epoch
        if isinstance(item, (tuple, list)):
            idx, cls = item
        else:
            idx, cls = item, self.classes[int(rng.integers(len(self.classes)))]
        clean = self.images[idx]
        if self.flip and rng.random() < 0.5:
            clean = clean[:, ::-1]
        params = C.sample_train_params(C.CLASSES[cls], rng)
        corrupted = C.apply(np.ascontiguousarray(clean), params)
        return to_tensor(corrupted), to_tensor(clean), cls


class BalancedBatchSampler(Sampler):
    """Yields batches of (image_index, class_id) where every class appears equally often."""

    def __init__(self, n_images, batch_size, classes=(0, 1, 2, 3), drop_last=True):
        self.n, self.bs, self.classes, self.drop_last = n_images, min(batch_size, n_images), list(classes), drop_last

    def __iter__(self):
        perm = np.random.permutation(self.n)
        for s in range(0, self.n, self.bs):
            chunk = perm[s:s + self.bs]
            if len(chunk) < self.bs and self.drop_last:
                break
            labels = np.resize(self.classes, len(chunk))
            np.random.shuffle(labels)
            yield [(int(i), int(c)) for i, c in zip(chunk, labels)]

    def __len__(self):
        return self.n // self.bs if self.drop_last else -(-self.n // self.bs)


class ManifestDataset(Dataset):
    """Deterministic validation / test dataset driven by a stored manifest."""

    def __init__(self, images, entries, classes=None):
        self.images = images
        self.entries = [e for e in entries if classes is None or e["label"] in classes]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        clean = self.images[e["index"]]
        return to_tensor(C.apply(clean, e["params"])), to_tensor(clean), e["label"], i


# ============================================================ FS2K
def load_fs2k(fs2k_root, cache_dir):
    """Return dict split -> (photos, sketches, styles, names) using the official anno_*.json split."""
    cache = os.path.join(cache_dir, f"fs2k_{SIZE}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["data"].item()
    anno = glob.glob(os.path.join(fs2k_root, "**", "anno_train.json"), recursive=True)
    if not anno:
        raise FileNotFoundError(f"anno_train.json not found under {fs2k_root}")
    base = os.path.dirname(anno[0])

    def find(stem):
        for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
            if os.path.exists(stem + ext):
                return stem + ext
        raise FileNotFoundError(stem)

    data = {}
    for split in ("train", "test"):
        with open(os.path.join(base, f"anno_{split}.json")) as f:
            entries = json.load(f)
        photos, sketches, styles, names = [], [], [], []
        for e in entries:
            name = e["image_name"]                                   # e.g. photo1/image0110
            sk = name.replace("image", "sketch").replace("photo", "sketch")  # sketch1/sketch0110
            photos.append(_resize_rgb(find(os.path.join(base, "photo", name))))
            sketches.append(_resize_rgb(find(os.path.join(base, "sketch", sk))))
            styles.append(int(e["style"]))
            names.append(name)
        data[split] = (np.stack(photos), np.stack(sketches), np.array(styles), names)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache, data=np.array(data, dtype=object))
    return data


def split_fs2k_train(styles, seed=SEED, val_frac=0.15):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(styles))
    tr, va = train_test_split(idx, test_size=val_frac, random_state=seed, stratify=styles)
    return np.sort(tr), np.sort(va)


class PairedSketchDataset(Dataset):
    """Photo/sketch pairs in [-1, 1]. Spatial augmentation is applied identically to both images."""

    def __init__(self, photos, sketches, styles, augment=False, jitter=142):
        self.p, self.s, self.st, self.augment, self.jitter = photos, sketches, styles, augment, jitter

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        p, s = self.p[i], self.s[i]
        if self.augment:
            rng = np.random.default_rng()
            j = self.jitter
            p = cv2.resize(p, (j, j), interpolation=cv2.INTER_CUBIC)
            s = cv2.resize(s, (j, j), interpolation=cv2.INTER_CUBIC)
            x, y = int(rng.integers(0, j - SIZE + 1)), int(rng.integers(0, j - SIZE + 1))
            p, s = p[y:y + SIZE, x:x + SIZE], s[y:y + SIZE, x:x + SIZE]      # same crop for both
            if rng.random() < 0.5:
                p, s = p[:, ::-1], s[:, ::-1]                              # same flip for both
        return to_tensor(p) * 2 - 1, to_tensor(s) * 2 - 1, int(self.st[i])
