"""Task 4: style-conditioned face-to-sketch pix2pix on FS2K (Optuna search, final training, test, ONNX).

    python scripts/run_sketch.py --fs2k data/FS2K --out outputs --budget full
"""
import argparse
import copy
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from genai import data as D
from genai import train_utils as U
from genai.metrics import psnr, ssim, val_score
from genai.models import StylePatchDiscriminator, StyleUNetGenerator, init_gan_weights

FS2K_GDRIVE_ID = "1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp"
BUDGETS = {
    "smoke": dict(subset=24, trials=2, trial_epochs=1, epochs=2, log_every=1),
    "full": dict(subset=None, trials=12, trial_epochs=30, epochs=200, log_every=10),
}
DEV = U.DEVICE


def ensure_fs2k(root):
    import glob
    if glob.glob(os.path.join(root, "**", "anno_train.json"), recursive=True):
        return root
    os.makedirs(root, exist_ok=True)
    zpath = os.path.join(root, "FS2K.zip")
    if not os.path.exists(zpath):
        import gdown
        gdown.download(id=FS2K_GDRIVE_ID, output=zpath, quiet=False)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    return root


@torch.no_grad()
def evaluate_g(G, loader):
    G.eval()
    ps, ss, l1s, styles = [], [], [], []
    for x, y, s in loader:
        x, y, s = x.to(DEV), y.to(DEV), s.to(DEV)
        fake = (G(x, s) + 1) / 2
        y01 = (y + 1) / 2
        ps.append(psnr(fake, y01, "none").cpu())
        ss.append(ssim(fake, y01, "none").cpu())
        l1s.append((fake - y01).abs().flatten(1).mean(1).cpu())
        styles.append(s.cpu())
    p, sm = torch.cat(ps), torch.cat(ss)
    out = {"psnr": p.mean().item(), "ssim": sm.mean().item(), "l1": torch.cat(l1s).mean().item()}
    out["score"] = val_score(out["psnr"], out["ssim"])
    st = torch.cat(styles)
    for k in range(3):
        if (st == k).any():
            out[f"ssim_style{k + 1}"] = sm[st == k].mean().item()
            out[f"psnr_style{k + 1}"] = p[st == k].mean().item()
    return out


def sample_grid(G, photos, styles, path=None):
    """Rows = fixed validation photos; columns = photo, sketch in style 1/2/3."""
    G.eval()
    with torch.no_grad():
        x = photos.to(DEV)
        outs = [((G(x, torch.full((len(x),), k, device=DEV, dtype=torch.long)) + 1) / 2).cpu() for k in range(3)]
    n = len(x)
    fig, axes = plt.subplots(n, 4, figsize=(8, 2 * n))
    for i in range(n):
        ims = [(photos[i] + 1) / 2] + [o[i] for o in outs]
        titles = ["photo"] + [f"style {k + 1}" + (" (GT style)" if k == int(styles[i]) else "") for k in range(3)]
        for j, (im, t) in enumerate(zip(ims, titles)):
            axes[i, j].imshow(im.permute(1, 2, 0).clamp(0, 1).numpy())
            axes[i, j].set_title(t, fontsize=7)
            axes[i, j].axis("off")
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def train_gan(cfg, train_ds, val_loader, epochs, fixed, tracker=None, run=None, trial=None, out_dir=None,
              log_every=10):
    G = StyleUNetGenerator(cfg["base"], cfg["emb_dim"], cfg["dropout"]).to(DEV)
    Dn = StylePatchDiscriminator(cfg["base"], cfg["emb_dim"]).to(DEV)
    G.apply(init_gan_weights)
    Dn.apply(init_gan_weights)
    opt_g = torch.optim.Adam(G.parameters(), cfg["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(Dn.parameters(), cfg["lr_d"], betas=(0.5, 0.999))
    # pix2pix schedule: constant lr for the first half, then linear decay to zero
    decay = lambda ep: 1.0 - max(0, ep + 1 - epochs // 2) / float(epochs - epochs // 2 + 1)
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, decay)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, decay)
    loader = U.make_loader(train_ds, cfg["batch_size"], shuffle=True)
    bce = F.binary_cross_entropy_with_logits
    best, best_state, hist = -1e9, None, []
    for ep in range(epochs):
        G.train()
        Dn.train()
        t0 = time.time()
        agg = {"d_real": 0.0, "d_fake": 0.0, "g_adv": 0.0, "g_l1": 0.0}
        n = 0
        for x, y, s in loader:
            x, y, s = x.to(DEV), y.to(DEV), s.to(DEV)
            fake = G(x, s)
            # ---- discriminator: real pairs -> 1, generated pairs -> 0
            pr = Dn(x, y, s)
            pf = Dn(x, fake.detach(), s)
            d_real = bce(pr, torch.ones_like(pr))
            d_fake = bce(pf, torch.zeros_like(pf))
            opt_d.zero_grad(set_to_none=True)
            (0.5 * (d_real + d_fake)).backward()
            opt_d.step()
            # ---- generator: fool D + stay close to the paired sketch
            pg = Dn(x, fake, s)
            g_adv = bce(pg, torch.ones_like(pg))
            g_l1 = F.l1_loss(fake, y)
            opt_g.zero_grad(set_to_none=True)
            (g_adv + cfg["lambda_l1"] * g_l1).backward()
            opt_g.step()
            b = len(x)
            for k, v in zip(agg, (d_real, d_fake, g_adv, g_l1)):
                agg[k] += v.item() * b
            n += b
        sch_g.step()
        sch_d.step()
        v = evaluate_g(G, val_loader)
        rec = {"epoch": ep + 1, **{k: a / n for k, a in agg.items()}, **{f"val_{k}": x for k, x in v.items()},
               "epoch_time": time.time() - t0}
        hist.append(rec)
        if tracker:
            tracker.log(run, rec)
        print(f"  ep {ep + 1}/{epochs} D real {rec['d_real']:.3f} fake {rec['d_fake']:.3f} G adv {rec['g_adv']:.3f} "
              f"l1 {rec['g_l1']:.3f} | val ssim {v['ssim']:.4f} psnr {v['psnr']:.2f} ({rec['epoch_time']:.0f}s)",
              flush=True)
        if v["score"] > best:
            best, best_state = v["score"], copy.deepcopy(G.state_dict())
        if out_dir and ((ep + 1) % log_every == 0 or ep == 0 or ep + 1 == epochs):
            fig = sample_grid(G, *fixed, os.path.join(out_dir, "progress", f"epoch_{ep + 1:03d}.png"))
            if tracker and run is not None:
                tracker.log(run, {"val_samples": tracker.wandb.Image(fig)})
            plt.close(fig)
        if trial is not None:
            import optuna
            trial.report(v["score"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    G.load_state_dict(best_state)
    return G, best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs2k", default="data/FS2K")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--budget", default="full", choices=BUDGETS)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    B = BUDGETS[args.budget]
    out = args.out
    sk = os.path.join(out, "sketch")
    ck = os.path.join(out, "checkpoints", "t4_generator.pt")
    storage = f"sqlite:///{os.path.abspath(os.path.join(out, 'optuna', 'sketch_study.db'))}"
    os.makedirs(os.path.join(out, "optuna"), exist_ok=True)
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    tracker = U.Tracker(enabled=not args.no_wandb)
    torch.manual_seed(42)

    data = D.load_fs2k(ensure_fs2k(args.fs2k), os.path.join(args.data, "cache"))
    P, S, ST, names = data["train"]
    tr, va = D.split_fs2k_train(ST)
    if B["subset"]:
        tr, va = tr[:B["subset"]], va[:B["subset"] // 2]
    Pt, St, STt, test_names = data["test"]
    U.save_json({"train": len(tr), "val": len(va), "test": len(Pt),
                 "train_style_counts": np.bincount(ST[tr], minlength=3).tolist(),
                 "val_style_counts": np.bincount(ST[va], minlength=3).tolist(),
                 "test_style_counts": np.bincount(STt, minlength=3).tolist(),
                 "val_names": [names[i] for i in va]}, os.path.join(sk, "split.json"))
    train_ds = D.PairedSketchDataset(P[tr], S[tr], ST[tr], augment=True)
    val_ds = D.PairedSketchDataset(P[va], S[va], ST[va])
    val_loader = U.make_loader(val_ds, 32)
    fixed_idx = np.concatenate([np.where(ST[va] == k)[0][:2] for k in range(3)])     # same photos every time
    fixed = (torch.stack([val_ds[i][0] for i in fixed_idx]), ST[va][fixed_idx])

    if not os.path.exists(ck):
        def objective(trial):
            cfg = dict(lr_g=trial.suggest_float("lr_g", 5e-5, 5e-4, log=True),
                       lr_d=trial.suggest_float("lr_d", 2e-5, 5e-4, log=True),
                       batch_size=trial.suggest_categorical("batch_size", [4, 8, 16]),
                       base=trial.suggest_categorical("base", [32, 48, 64]),
                       dropout=trial.suggest_float("dropout", 0.0, 0.5),
                       emb_dim=trial.suggest_categorical("emb_dim", [4, 8, 16]),
                       lambda_l1=trial.suggest_float("lambda_l1", 10, 200, log=True))
            run = tracker.start(f"gan-trial-{trial.number}", "t4-gan-optuna", cfg, "optuna")
            try:
                _, best, _ = train_gan(cfg, train_ds, val_loader, B["trial_epochs"], fixed, tracker, run, trial)
            finally:
                tracker.finish(run)
            return best

        study = U.run_study("t4_sketch_gan", storage, objective, B["trials"], startup=3)
        U.save_json(U.study_summary(study), os.path.join(out, "optuna", "t4_sketch_gan.json"))
        cfg = study.best_params
        run = tracker.start("gan-final", "t4-gan-final", {**cfg, "epochs": B["epochs"]})
        G, best, hist = train_gan(cfg, train_ds, val_loader, B["epochs"], fixed, tracker, run, out_dir=sk,
                                  log_every=B["log_every"])
        torch.save({"cfg": {"base": cfg["base"], "emb_dim": cfg["emb_dim"], "dropout": cfg["dropout"]},
                    "state": G.state_dict(), "hparams": cfg}, ck)
        U.save_json(hist, os.path.join(out, "history", "t4_gan.json"))
        tracker.artifact(run, ck, "t4_generator")
        tracker.finish(run, {"best_val_score": best})

    # ---------------------------------------------------------------- test evaluation (official test set)
    from genai.checkpoints import load_generator
    G = load_generator(ck).to(DEV)
    if B["subset"]:
        Pt, St, STt = Pt[:B["subset"]], St[:B["subset"]], STt[:B["subset"]]
    test_ds = D.PairedSketchDataset(Pt, St, STt)
    res = evaluate_g(G, U.make_loader(test_ds, 32))
    # per-sample ssim for failure cases
    G.eval()
    per = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y, s = test_ds[i]
            f = (G(x[None].to(DEV), torch.tensor([s], device=DEV)) + 1) / 2
            per.append(ssim(f, ((y[None] + 1) / 2).to(DEV)).item())
    per = np.array(per)
    res["n_test"] = len(test_ds)
    U.save_json(res, os.path.join(out, "results", "t4_test_metrics.json"))
    print("test:", res)
    figs = os.path.join(out, "figures")
    os.makedirs(figs, exist_ok=True)

    def grid(ids, path, title):
        fig, axes = plt.subplots(len(ids), 3, figsize=(6, 2 * len(ids)))
        axes = np.atleast_2d(axes)
        for r, i in enumerate(ids):
            x, y, s = test_ds[i]
            with torch.no_grad():
                f = (G(x[None].to(DEV), torch.tensor([s], device=DEV)) + 1) / 2
            for c, (im, t) in enumerate([((x + 1) / 2, "photo"), (f[0].cpu(), f"generated (style {s + 1})"),
                                         ((y + 1) / 2, f"ground truth  SSIM {per[i]:.3f}")]):
                axes[r, c].imshow(im.permute(1, 2, 0).clamp(0, 1).numpy())
                axes[r, c].set_title(t, fontsize=7)
                axes[r, c].axis("off")
        fig.suptitle(title, fontsize=9)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    rng = np.random.default_rng(0)
    good = [int(rng.choice(np.where(STt == k)[0])) for k in range(3) for _ in range(2)]
    grid(good, os.path.join(figs, "t4_test_examples.png"), "FS2K test: random examples (2 per style)")
    grid(list(np.argsort(per)[:4]), os.path.join(figs, "t4_failures.png"), "FS2K test: lowest-SSIM cases")
    fig = sample_grid(G, torch.stack([test_ds[i][0] for i in good[::2]]), STt[good[::2]],
                      os.path.join(figs, "t4_style_swap.png"))
    plt.close(fig)
    hp = os.path.join(out, "history", "t4_gan.json")
    if os.path.exists(hp):
        h = U.load_json(hp)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3))
        e = [r["epoch"] for r in h]
        for k in ["d_real", "d_fake", "g_adv"]:
            axes[0].plot(e, [r[k] for r in h], label=k)
        axes[1].plot(e, [r["g_l1"] for r in h], label="g_l1 (train)")
        axes[1].plot(e, [r["val_l1"] * 2 for r in h], label="val L1 ([-1,1] scale)")
        axes[2].plot(e, [r["val_ssim"] for r in h], label="val SSIM")
        for a in axes:
            a.legend(fontsize=7)
            a.grid(alpha=0.3)
            a.set_xlabel("epoch")
        fig.savefig(os.path.join(figs, "t4_curves.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
    if not args.no_wandb:
        run = tracker.start("gan-test", "evaluation", {}, "eval")
        tracker.log(run, {f"t4_test_{k}": v for k, v in res.items()})
        for fn in ["t4_test_examples.png", "t4_failures.png", "t4_style_swap.png", "t4_curves.png"]:
            if os.path.exists(os.path.join(figs, fn)) and run is not None:
                tracker.log(run, {f"figures/{fn[:-4]}": tracker.wandb.Image(os.path.join(figs, fn))})
        tracker.finish(run)

    from genai import export
    export.export_generator(ck, os.path.join(out, "onnx"), Pt[:6], STt[:6])
    # sample photos for the web app
    os.makedirs(os.path.join(out, "samples", "faces"), exist_ok=True)
    from PIL import Image
    for k, i in enumerate(good):
        Image.fromarray(Pt[i]).save(os.path.join(out, "samples", "faces", f"face_{k + 1}.png"))


if __name__ == "__main__":
    main()
