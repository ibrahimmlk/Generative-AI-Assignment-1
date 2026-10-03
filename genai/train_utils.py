"""Shared training / evaluation loops, experiment tracking and Optuna helpers."""
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import balance_loss, psnr, restoration_loss, ssim, val_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", min(4, os.cpu_count() or 1) if DEVICE.type == "cuda" else 0))


# ------------------------------------------------------------------ tracking (Weights & Biases)
class Tracker:
    """Thin wrapper so that the code still runs (offline) when W&B is unavailable."""

    def __init__(self, project="genai-a1", enabled=True):
        self.project = project
        self.wandb = None
        if enabled:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("[tracker] wandb not installed - logging disabled")

    def start(self, name, group, config=None, job_type="train"):
        if self.wandb is None:
            return None
        # without an API key W&B would block waiting for a login, so fall back to offline logging
        mode = os.environ.get("WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "offline")
        return self.wandb.init(project=self.project, name=name, group=group, job_type=job_type,
                               config=config or {}, reinit=True, mode=mode)

    def log(self, run, data, step=None):
        if run is not None:
            run.log(data, step=step)

    def images(self, run, key, arrays, captions=None, step=None):
        if run is not None:
            imgs = [self.wandb.Image(a, caption=(captions[i] if captions else None)) for i, a in enumerate(arrays)]
            run.log({key: imgs}, step=step)

    def artifact(self, run, path, name, type_="model"):
        if run is not None and os.path.exists(path):
            art = self.wandb.Artifact(name, type=type_)
            art.add_file(path)
            run.log_artifact(art)

    def finish(self, run, summary=None):
        if run is not None:
            if summary:
                for k, v in summary.items():
                    run.summary[k] = v
            run.finish()


# ------------------------------------------------------------------ loaders
def make_loader(ds, batch_size=None, shuffle=False, batch_sampler=None):
    kw = dict(num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
              persistent_workers=NUM_WORKERS > 0)
    if batch_sampler is not None:
        return DataLoader(ds, batch_sampler=batch_sampler, **kw)
    return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle, drop_last=shuffle, **kw)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ autoencoders
@torch.no_grad()
def evaluate_ae(model, loader):
    model.eval()
    ps, ss, l1 = [], [], []
    for xt, x, *_ in loader:
        xt, x = xt.to(DEVICE), x.to(DEVICE)
        with torch.autocast(DEVICE.type, enabled=AMP):
            out = model(xt)
        out = out.float().clamp(0, 1)
        ps.append(psnr(out, x, "none").cpu())
        ss.append(ssim(out, x, "none").cpu())
        l1.append((out - x).abs().flatten(1).mean(1).cpu())
    p, s = torch.cat(ps).mean().item(), torch.cat(ss).mean().item()
    return {"psnr": p, "ssim": s, "l1": torch.cat(l1).mean().item(), "score": val_score(p, s)}


def train_ae(model, train_loader, val_loader, epochs, lr, alpha, tracker=None, run=None, trial=None,
             prefix="", weight_decay=1e-5):
    """Train an autoencoder with alpha*L1 + (1-alpha)*(1-SSIM); keeps the best-validation weights."""
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    best, best_state, history = -math.inf, None, []
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for xt, x, _ in train_loader:
            xt, x = xt.to(DEVICE, non_blocking=True), x.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE.type, enabled=AMP):
                out = model(xt)
            loss = restoration_loss(out, x, alpha)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
            n += len(x)
        sched.step()
        v = evaluate_ae(model, val_loader)
        rec = {"epoch": ep + 1, "train_loss": tot / max(n, 1), **{f"val_{k}": val for k, val in v.items()},
               "lr": sched.get_last_lr()[0], "epoch_time": time.time() - t0}
        history.append(rec)
        if tracker:
            tracker.log(run, {f"{prefix}{k}": val for k, val in rec.items()})
        print(f"  {prefix}ep {ep + 1}/{epochs} loss {rec['train_loss']:.4f} val psnr {v['psnr']:.2f} "
              f"ssim {v['ssim']:.4f} score {v['score']:.4f} ({rec['epoch_time']:.0f}s)", flush=True)
        if v["score"] > best:
            best, best_state = v["score"], copy.deepcopy(model.state_dict())
        if trial is not None:
            import optuna
            trial.report(v["score"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, history


# ------------------------------------------------------------------ classifier
@torch.no_grad()
def predict_classifier(model, loader):
    model.eval()
    probs, labels = [], []
    for xt, _, y, *_ in loader:
        with torch.autocast(DEVICE.type, enabled=AMP):
            logits = model(xt.to(DEVICE))
        probs.append(logits.float().softmax(1).cpu())
        labels.append(torch.as_tensor(y))
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def classification_metrics(probs, labels, class_names):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(labels, pred, labels=range(len(class_names)), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)
    cm = confusion_matrix(labels, pred, labels=range(len(class_names)), normalize="true")
    return {"accuracy": float(accuracy_score(labels, pred)), "macro_precision": float(mp),
            "macro_recall": float(mr), "macro_f1": float(mf),
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                              "support": int(s[i])} for i, c in enumerate(class_names)},
            "confusion_matrix_normalized": cm.tolist()}


def train_classifier(model, train_loader, val_loader, epochs, lr, weight_decay, class_names,
                     tracker=None, run=None, trial=None, prefix=""):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    best, best_state, history = -math.inf, None, []
    for ep in range(epochs):
        model.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for xt, _, y in train_loader:
            xt, y = xt.to(DEVICE, non_blocking=True), torch.as_tensor(y).to(DEVICE)
            with torch.autocast(DEVICE.type, enabled=AMP):
                logits = model(xt)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()
        probs, labels = predict_classifier(model, val_loader)
        m = classification_metrics(probs, labels, class_names)
        rec = {"epoch": ep + 1, "train_loss": tot / n, "train_acc": correct / n, "val_acc": m["accuracy"],
               "val_macro_f1": m["macro_f1"], "epoch_time": time.time() - t0}
        history.append(rec)
        if tracker:
            tracker.log(run, {f"{prefix}{k}": v for k, v in rec.items()})
        print(f"  {prefix}ep {ep + 1}/{epochs} loss {rec['train_loss']:.4f} acc {rec['train_acc']:.3f} "
              f"val acc {m['accuracy']:.3f} f1 {m['macro_f1']:.3f}", flush=True)
        if m["macro_f1"] > best:
            best, best_state = m["macro_f1"], copy.deepcopy(model.state_dict())
        if trial is not None:
            import optuna
            trial.report(m["macro_f1"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    model.load_state_dict(best_state)
    return best, history


# ------------------------------------------------------------------ soft mixture of experts
@torch.no_grad()
def evaluate_moe(moe, loader):
    moe.eval()
    ps, ss, ws, ys = [], [], [], []
    for xt, x, y, *_ in loader:
        xt, x = xt.to(DEVICE), x.to(DEVICE)
        with torch.autocast(DEVICE.type, enabled=AMP):
            out, w, _ = moe(xt)
        out = out.float().clamp(0, 1)
        ps.append(psnr(out, x, "none").cpu())
        ss.append(ssim(out, x, "none").cpu())
        ws.append(w.float().cpu())
        ys.append(torch.as_tensor(y))
    w, y = torch.cat(ws), torch.cat(ys)
    p, s = torch.cat(ps).mean().item(), torch.cat(ss).mean().item()
    return {"psnr": p, "ssim": s, "score": val_score(p, s), "mean_w": w.mean(0).tolist(),
            "gate_acc": (w.argmax(1) == y).float().mean().item()}


def set_experts_trainable(moe, flag):
    for e in moe.experts:
        for prm in e.parameters():
            prm.requires_grad = flag
        e.train(flag)


def train_moe(moe, train_loader, val_loader, warm_epochs, epochs, warm_lr, lr, lam, tracker=None, run=None,
              trial=None, prefix="", collapse_thresh=0.02):
    """Stage 1: experts frozen, only the gate is trained. Stage 2: joint fine-tuning with a smaller lr.
    lam = dict(l1, ssim, ce, bal)."""
    moe.to(DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    best, best_state, history, step = -math.inf, None, [], 0
    stages = [("warmup", warm_epochs, warm_lr, False), ("joint", epochs, lr, True)]
    for stage, n_ep, stage_lr, joint in stages:
        if n_ep <= 0:
            continue
        set_experts_trainable(moe, joint)
        params = [p for p in moe.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=stage_lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_ep)
        for ep in range(n_ep):
            moe.gate.train()
            set_experts_trainable(moe, joint)
            t0, agg, n = time.time(), {"loss": 0, "l1": 0, "ssim": 0, "ce": 0, "bal": 0}, 0
            for xt, x, y in train_loader:
                xt, x, y = xt.to(DEVICE), x.to(DEVICE), torch.as_tensor(y).to(DEVICE)
                with torch.autocast(DEVICE.type, enabled=AMP):
                    out, w, logits = moe(xt)
                out, w, logits = out.float(), w.float(), logits.float()
                l1 = F.l1_loss(out, x)
                s = 1 - ssim(out, x)
                ce = F.cross_entropy(logits, y)
                bal = balance_loss(w)
                loss = lam["l1"] * l1 + lam["ssim"] * s + lam["ce"] * ce + lam["bal"] * bal
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                b = len(x)
                for k, v in zip(agg, (loss, l1, s, ce, bal)):
                    agg[k] += v.item() * b
                n += b
            sched.step()
            v = evaluate_moe(moe, val_loader)
            step += 1
            rec = {"epoch": step, "stage": stage, **{f"train_{k}": a / n for k, a in agg.items()},
                   **{f"val_{k}": val for k, val in v.items() if k != "mean_w"},
                   **{f"val_w{k}": wk for k, wk in enumerate(v["mean_w"])}, "epoch_time": time.time() - t0}
            history.append(rec)
            if tracker:
                tracker.log(run, {f"{prefix}{k}": val for k, val in rec.items() if k != "stage"})
            print(f"  {prefix}{stage} ep {ep + 1}/{n_ep} loss {rec['train_loss']:.4f} val psnr {v['psnr']:.2f} "
                  f"ssim {v['ssim']:.4f} gate acc {v['gate_acc']:.3f} w {np.round(v['mean_w'], 3)}", flush=True)
            if v["score"] > best:
                best, best_state = v["score"], copy.deepcopy(moe.state_dict())
            if trial is not None:
                import optuna
                trial.report(v["score"], step - 1)
                if min(v["mean_w"]) < collapse_thresh:      # routing collapse: an expert became inactive
                    trial.set_user_attr("pruned_reason", "routing_collapse")
                    raise optuna.TrialPruned()
                if trial.should_prune():
                    raise optuna.TrialPruned()
    moe.load_state_dict(best_state)
    return best, history


# ------------------------------------------------------------------ optuna
def run_study(name, storage, objective, n_trials, direction="maximize", seed=42, startup=3):
    import optuna
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=startup, n_warmup_steps=1)
    study = optuna.create_study(study_name=name, storage=storage, direction=direction, sampler=sampler,
                                pruner=pruner, load_if_exists=True)
    done = len([t for t in study.trials if t.state.is_finished()])
    if n_trials - done > 0:
        study.optimize(objective, n_trials=n_trials - done, gc_after_trial=True)
    return study


def study_summary(study):
    import optuna
    trials = study.trials
    st = lambda s: sum(t.state == s for t in trials)
    return {"study": study.study_name, "n_trials": len(trials),
            "completed": st(optuna.trial.TrialState.COMPLETE), "pruned": st(optuna.trial.TrialState.PRUNED),
            "failed": st(optuna.trial.TrialState.FAIL), "best_trial": study.best_trial.number,
            "best_value": study.best_value, "best_params": study.best_params,
            "trials": [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params,
                        "user_attrs": t.user_attrs} for t in trials]}
