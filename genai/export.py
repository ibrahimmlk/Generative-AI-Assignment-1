"""ONNX export of every inference model + numerical verification against PyTorch."""
import json
import os

import numpy as np
import torch
import torch.nn as nn

from .checkpoints import EXPERTS, load_ae, load_classifier, load_generator, load_moe

OPSET = 17


class ClassifierWithProbs(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        logits = self.m(x)
        return logits.softmax(1), logits


class MoEExport(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out, w, _ = self.m(x)
        return out, w


def _export(model, args, path, input_names, output_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dyn = {n: {0: "batch"} for n in input_names + output_names}
    torch.onnx.export(model.eval(), args, path, input_names=input_names, output_names=output_names,
                      dynamic_axes=dyn, opset_version=OPSET, dynamo=False)


def _verify(model, args, path, input_names):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(*args)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    got = sess.run(None, {n: a.numpy() for n, a in zip(input_names, args)})
    diffs = [float(np.abs(r.numpy() - g).max()) for r, g in zip(ref, got)]
    return {"file": os.path.basename(path), "size_mb": round(os.path.getsize(path) / 2**20, 2),
            "max_abs_diff": max(diffs), "per_output_max_abs_diff": diffs, "passed": max(diffs) < 1e-3}


def export_restoration(ck, onnx_dir, sample):
    """sample: uint8 N x 128 x 128 x 3 real test images used for the consistency check."""
    x = torch.from_numpy(sample).permute(0, 3, 1, 2).float() / 255.0
    x = torch.cat([x, torch.rand(2, 3, 128, 128)])
    report = []
    jobs = [("universal_ae.onnx", load_ae(os.path.join(ck, "t1_universal_ae.pt")), ["input"], ["output"]),
            ("classifier.onnx", ClassifierWithProbs(load_classifier(os.path.join(ck, "t2_classifier.pt"))),
             ["input"], ["probs", "logits"])]
    jobs += [(f"specialist_{k}.onnx", load_ae(os.path.join(ck, f"t2_specialist_{k}.pt")), ["input"], ["output"])
             for k in EXPERTS]
    jobs.append(("soft_moe.onnx", MoEExport(load_moe(os.path.join(ck, "t3_soft_moe.pt"))), ["input"],
                 ["output", "weights"]))
    for name, model, ins, outs in jobs:
        model = model.eval().cpu()
        path = os.path.join(onnx_dir, name)
        _export(model, (x[:1],), path, ins, outs)
        report.append(_verify(model, (x,), path, ins))
        print("exported", report[-1], flush=True)
    _write_report(onnx_dir, report)
    return report


class GeneratorExport(nn.Module):
    """Takes the photo in [0, 1] and returns the sketch in [0, 1] so the app needs no extra maths."""

    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, photo, style):
        return (self.g(photo * 2 - 1, style) + 1) / 2


def export_generator(ckpt, onnx_dir, sample_photos, sample_styles):
    g = GeneratorExport(load_generator(ckpt)).eval().cpu()
    x = torch.from_numpy(sample_photos).permute(0, 3, 1, 2).float() / 255.0
    s = torch.as_tensor(sample_styles, dtype=torch.long)
    path = os.path.join(onnx_dir, "sketch_generator.onnx")
    _export(g, (x[:1], s[:1]), path, ["photo", "style"], ["sketch"])
    rep = _verify(g, (x, s), path, ["photo", "style"])
    print("exported", rep, flush=True)
    _write_report(onnx_dir, [rep])
    return rep


def _write_report(onnx_dir, entries):
    p = os.path.join(onnx_dir, "onnx_verification.json")
    old = json.load(open(p)) if os.path.exists(p) else []
    keep = {e["file"]: e for e in old}
    keep.update({e["file"]: e for e in entries})
    with open(p, "w") as f:
        json.dump(list(keep.values()), f, indent=2)
