"""Tasks 1-3 end-to-end: data prep, Optuna searches, final training, test evaluation, figures, ONNX.

Every stage writes its results to --out and is skipped on re-run if its outputs exist,
so an interrupted Kaggle session can simply be restarted.

    python scripts/run_restoration.py --data data --out outputs --budget full
    python scripts/run_restoration.py --budget smoke          # quick CPU sanity check
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Subset

from genai import corruptions as C
from genai import data as D
from genai import train_utils as U
from genai.checkpoints import EXPERTS, load_ae, load_classifier
from genai.models import CorruptionClassifier, DenoisingAE, SoftMoE

BUDGETS = {
    "smoke": dict(subset=48, t1_trials=2, t1_trial_epochs=1, t1_epochs=1, cls_trials=2, cls_trial_epochs=1,
                  cls_epochs=1, sp_trials=2, sp_trial_epochs=1, sp_epochs=1, moe_trials=2, moe_warm=1,
                  moe_trial_epochs=1, moe_epochs=1, test_images=12),
    "full": dict(subset=None, t1_trials=14, t1_trial_epochs=6, t1_epochs=45, cls_trials=10, cls_trial_epochs=4,
                 cls_epochs=25, sp_trials=8, sp_trial_epochs=4, sp_epochs=35, moe_trials=8, moe_warm=2,
                 moe_trial_epochs=3, moe_epochs=12, test_images=None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--budget", default="full", choices=BUDGETS)
    ap.add_argument("--stages", default="prep,t1,cls,spec,moe,eval,export")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    B = BUDGETS[args.budget]
    stages = args.stages.split(",")
    out = args.out
    ck = os.path.join(out, "checkpoints")
    os.makedirs(ck, exist_ok=True)
    os.makedirs(os.path.join(out, "optuna"), exist_ok=True)
    storage = f"sqlite:///{os.path.abspath(os.path.join(out, 'optuna', 'studies.db'))}"
    tracker = U.Tracker(enabled=not args.no_wandb)
    torch.manual_seed(42)
    print("device:", U.DEVICE, "budget:", args.budget, flush=True)

    # ------------------------------------------------------------ data + split + manifests
    trainval, tv_names = D.load_pets(args.data, "trainval")
    test, test_names = D.load_pets(args.data, "test")
    tr_idx, va_idx = D.split_trainval(len(trainval))
    man_dir = os.path.join(out, "manifests")
    if not os.path.exists(os.path.join(man_dir, "test_manifest.json")):
        D.build_manifests(tv_names, test_names, va_idx, man_dir)
    val_man = U.load_json(os.path.join(man_dir, "val_manifest.json"))
    test_man = U.load_json(os.path.join(man_dir, "test_manifest.json"))
    train_imgs = trainval[tr_idx]
    if B["subset"]:
        train_imgs = train_imgs[:B["subset"]]
        val_man = val_man[:B["subset"]]
    if B["test_images"]:
        test_man = [e for e in test_man if e["index"] < B["test_images"]]
    print(f"train {len(train_imgs)}  val {len(val_man)}  test entries {len(test_man)}", flush=True)
    U.save_json({"train": len(tr_idx), "val": len(va_idx), "test": len(test),
                 "test_entries": len(test_man)}, os.path.join(out, "results", "data_summary.json"))

    val_all = D.ManifestDataset(trainval, val_man)
    val_loader_all = U.make_loader(val_all, 64)

    # ============================================================ Task 1
    t1_ckpt = os.path.join(ck, "t1_universal_ae.pt")
    if "t1" in stages and not os.path.exists(t1_ckpt):
        train_ds = D.RuntimeCorruptionDataset(train_imgs)

        def objective(trial):
            cfg = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                       batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
                       latent_ch=trial.suggest_categorical("latent_ch", [8, 16, 32]),
                       base=trial.suggest_categorical("base", [32, 48, 64]),
                       dropout=trial.suggest_float("dropout", 0.0, 0.3),
                       alpha=trial.suggest_float("alpha", 0.5, 0.95))
            model = DenoisingAE(cfg["base"], 3, cfg["latent_ch"], cfg["dropout"])
            trial.set_user_attr("latent_dim", model.latent_dim())
            run = tracker.start(f"t1-trial-{trial.number}", "t1-optuna", cfg, "optuna")
            try:
                best, _ = U.train_ae(model, U.make_loader(train_ds, cfg["batch_size"], shuffle=True),
                                     val_loader_all, B["t1_trial_epochs"], cfg["lr"], cfg["alpha"],
                                     tracker, run, trial)
            finally:
                tracker.finish(run)
            return best

        study = U.run_study("t1_universal_ae", storage, objective, B["t1_trials"])
        U.save_json(U.study_summary(study), os.path.join(out, "optuna", "t1_universal_ae.json"))
        p = study.best_params
        cfg = dict(base=p["base"], depth=3, latent_ch=p["latent_ch"], dropout=p["dropout"])
        model = DenoisingAE(**cfg)
        run = tracker.start("t1-final", "t1-final", {**p, "epochs": B["t1_epochs"]})
        best, hist = U.train_ae(model, U.make_loader(train_ds, p["batch_size"], shuffle=True), val_loader_all,
                                B["t1_epochs"], p["lr"], p["alpha"], tracker, run, prefix="")
        torch.save({"cfg": cfg, "state": model.state_dict(), "hparams": p}, t1_ckpt)
        U.save_json(hist, os.path.join(out, "history", "t1_final.json"))
        tracker.artifact(run, t1_ckpt, "t1_universal_ae")
        tracker.finish(run, {"best_val_score": best})

    # ============================================================ Task 2a: classifier
    cls_ckpt = os.path.join(ck, "t2_classifier.pt")
    if "cls" in stages and not os.path.exists(cls_ckpt):
        train_ds = D.RuntimeCorruptionDataset(train_imgs)

        def cls_loader(bs):
            return U.make_loader(train_ds, batch_sampler=D.BalancedBatchSampler(len(train_ds), bs))

        def objective(trial):
            cfg = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                       batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                       channels=trial.suggest_categorical("channels", ["small", "medium", "large"]),
                       dropout=trial.suggest_float("dropout", 0.0, 0.5),
                       weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
            model = CorruptionClassifier(cfg["channels"], cfg["dropout"])
            run = tracker.start(f"cls-trial-{trial.number}", "t2-classifier-optuna", cfg, "optuna")
            try:
                best, _ = U.train_classifier(model, cls_loader(cfg["batch_size"]), val_loader_all,
                                             B["cls_trial_epochs"], cfg["lr"], cfg["weight_decay"], C.CLASSES,
                                             tracker, run, trial)
            finally:
                tracker.finish(run)
            return best

        study = U.run_study("t2_classifier", storage, objective, B["cls_trials"])
        U.save_json(U.study_summary(study), os.path.join(out, "optuna", "t2_classifier.json"))
        p = study.best_params
        model = CorruptionClassifier(p["channels"], p["dropout"])
        run = tracker.start("cls-final", "t2-classifier-final", {**p, "epochs": B["cls_epochs"]})
        best, hist = U.train_classifier(model, cls_loader(p["batch_size"]), val_loader_all, B["cls_epochs"],
                                        p["lr"], p["weight_decay"], C.CLASSES, tracker, run)
        torch.save({"cfg": {"channels": p["channels"], "dropout": p["dropout"]}, "state": model.state_dict(),
                    "hparams": p}, cls_ckpt)
        U.save_json(hist, os.path.join(out, "history", "t2_classifier.json"))
        tracker.artifact(run, cls_ckpt, "t2_classifier")
        tracker.finish(run, {"best_val_macro_f1": best})

    # ============================================================ Task 2b: specialists
    spec_ckpts = {k: os.path.join(ck, f"t2_specialist_{k}.pt") for k in EXPERTS}
    if "spec" in stages and not all(os.path.exists(p) for p in spec_ckpts.values()):
        spec_train = {k: D.RuntimeCorruptionDataset(train_imgs, classes=[C.CLASS_TO_ID[k]]) for k in EXPERTS}
        spec_val = {k: U.make_loader(D.ManifestDataset(trainval, val_man, classes=[C.CLASS_TO_ID[k]]), 64)
                    for k in EXPERTS}

        def objective(trial):
            # one shared search for a common architecture; each trial briefly trains all three specialists
            cfg = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                       batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
                       latent_ch=trial.suggest_categorical("latent_ch", [8, 16, 32]),
                       base=trial.suggest_categorical("base", [32, 48, 64]),
                       alpha=trial.suggest_float("alpha", 0.5, 0.95))
            run = tracker.start(f"spec-trial-{trial.number}", "t2-specialists-optuna", cfg, "optuna")
            scores = []
            try:
                for k in EXPERTS:
                    model = DenoisingAE(cfg["base"], 3, cfg["latent_ch"], 0.0)
                    s, _ = U.train_ae(model, U.make_loader(spec_train[k], cfg["batch_size"], shuffle=True),
                                      spec_val[k], B["sp_trial_epochs"], cfg["lr"], cfg["alpha"], tracker, run,
                                      prefix=f"{k}/")
                    scores.append(s)
                    trial.set_user_attr(f"score_{k}", s)
                    trial.report(float(np.mean(scores)), len(scores) - 1)
            finally:
                tracker.finish(run)
            return float(np.mean(scores))

        study = U.run_study("t2_specialists", storage, objective, B["sp_trials"])
        U.save_json(U.study_summary(study), os.path.join(out, "optuna", "t2_specialists.json"))
        p = study.best_params
        cfg = dict(base=p["base"], depth=3, latent_ch=p["latent_ch"], dropout=0.0)
        for k in EXPERTS:
            if os.path.exists(spec_ckpts[k]):
                continue
            model = DenoisingAE(**cfg)        # same architecture, independently trained parameters
            run = tracker.start(f"spec-final-{k}", "t2-specialists-final", {**p, "expert": k,
                                                                           "epochs": B["sp_epochs"]})
            best, hist = U.train_ae(model, U.make_loader(spec_train[k], p["batch_size"], shuffle=True),
                                    spec_val[k], B["sp_epochs"], p["lr"], p["alpha"], tracker, run)
            torch.save({"cfg": cfg, "state": model.state_dict(), "hparams": p, "expert": k}, spec_ckpts[k])
            U.save_json(hist, os.path.join(out, "history", f"t2_specialist_{k}.json"))
            tracker.artifact(run, spec_ckpts[k], f"t2_specialist_{k}")
            tracker.finish(run, {"best_val_score": best})

    # ============================================================ Task 3: soft MoE
    moe_ckpt = os.path.join(ck, "t3_soft_moe.pt")
    if "moe" in stages and not os.path.exists(moe_ckpt):
        train_ds = D.RuntimeCorruptionDataset(train_imgs)
        moe_bs = 32
        train_loader = U.make_loader(train_ds, batch_sampler=D.BalancedBatchSampler(len(train_ds), moe_bs))

        def build(tau):
            return SoftMoE(load_classifier(cls_ckpt), [load_ae(spec_ckpts[k]) for k in EXPERTS], tau)

        def objective(trial):
            l1 = trial.suggest_float("lambda_l1", 0.5, 0.95)
            cfg = dict(lr=trial.suggest_float("lr", 1e-5, 3e-4, log=True),
                       tau=trial.suggest_float("tau", 0.5, 2.0),
                       lambda_ce=trial.suggest_float("lambda_ce", 0.01, 0.5, log=True),
                       lambda_bal=trial.suggest_float("lambda_bal", 1e-3, 0.1, log=True), lambda_l1=l1)
            lam = {"l1": l1, "ssim": 1 - l1, "ce": cfg["lambda_ce"], "bal": cfg["lambda_bal"]}
            run = tracker.start(f"moe-trial-{trial.number}", "t3-moe-optuna", cfg, "optuna")
            try:
                best, _ = U.train_moe(build(cfg["tau"]), train_loader, val_loader_all, B["moe_warm"],
                                      B["moe_trial_epochs"], 1e-4, cfg["lr"], lam, tracker, run, trial)
            finally:
                tracker.finish(run)
            return best

        study = U.run_study("t3_soft_moe", storage, objective, B["moe_trials"], startup=2)
        U.save_json(U.study_summary(study), os.path.join(out, "optuna", "t3_soft_moe.json"))
        p = study.best_params
        lam = {"l1": p["lambda_l1"], "ssim": 1 - p["lambda_l1"], "ce": p["lambda_ce"], "bal": p["lambda_bal"]}
        moe = build(p["tau"])
        run = tracker.start("moe-final", "t3-moe-final", {**p, "warmup": B["moe_warm"], "epochs": B["moe_epochs"]})
        best, hist = U.train_moe(moe, train_loader, val_loader_all, B["moe_warm"], B["moe_epochs"], 1e-4,
                                 p["lr"], lam, tracker, run)
        torch.save({"tau": p["tau"], "gate_cfg": torch.load(cls_ckpt, weights_only=False)["cfg"],
                    "expert_cfg": torch.load(spec_ckpts["blur"], weights_only=False)["cfg"], "state": moe.state_dict(),
                    "hparams": p}, moe_ckpt)
        U.save_json(hist, os.path.join(out, "history", "t3_soft_moe.json"))
        tracker.artifact(run, moe_ckpt, "t3_soft_moe")
        tracker.finish(run, {"best_val_score": best})

    # ============================================================ evaluation + figures
    if "eval" in stages:
        from genai import evaluate_restoration as E
        run = tracker.start("test-evaluation", "evaluation", {}, "eval")
        E.run_all(trainval, val_man, test, test_man, ck, out, tracker, run)
        tracker.finish(run)

    if "export" in stages:
        from genai import export
        export.export_restoration(ck, os.path.join(out, "onnx"), sample=test[:8])
        from PIL import Image
        sd = os.path.join(out, "samples", "pets")
        os.makedirs(sd, exist_ok=True)
        for k, i in enumerate(np.random.default_rng(7).choice(len(test), 10, replace=False)):
            Image.fromarray(test[i]).save(os.path.join(sd, f"pet_{k + 1}.png"))


if __name__ == "__main__":
    main()
