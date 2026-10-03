"""Test-set evaluation and report figures for Tasks 1-3."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import corruptions as C
from . import data as D
from . import train_utils as U
from .checkpoints import EXPERTS, load_ae, load_classifier, load_moe
from .metrics import psnr, ssim

METHODS = {"input": "No restoration", "t1": "T1 universal AE", "oracle": "T2 hard (oracle)",
           "pred": "T2 hard (predicted)", "moe": "T3 soft MoE"}
BRANCHES = ["identity", "salt_pepper", "blur", "occlusion"]
SEV_ORDER = ["none", "low", "medium", "high"]


class Models:
    def __init__(self, ck):
        self.t1 = load_ae(os.path.join(ck, "t1_universal_ae.pt")).to(U.DEVICE)
        self.cls = load_classifier(os.path.join(ck, "t2_classifier.pt")).to(U.DEVICE)
        self.experts = [load_ae(os.path.join(ck, f"t2_specialist_{k}.pt")).to(U.DEVICE) for k in EXPERTS]
        self.moe = load_moe(os.path.join(ck, "t3_soft_moe.pt")).to(U.DEVICE)

    @torch.no_grad()
    def run(self, xt, label):
        """All systems on one batch. Returns dict of B x 3 x H x W outputs plus probs / weights."""
        o = {"input": xt, "t1": self.t1(xt).clamp(0, 1)}
        probs = self.cls(xt).softmax(1)
        exp = torch.stack([xt] + [e(xt).clamp(0, 1) for e in self.experts], 1)   # B x 4 x ...
        ar = torch.arange(len(xt), device=xt.device)
        pred = probs.argmax(1)
        o["oracle"] = exp[ar, label]          # label 0 -> identity bypass
        o["pred"] = exp[ar, pred]
        out, w, _ = self.moe(xt)
        o["moe"] = out.clamp(0, 1)
        return o, probs, pred, w


def evaluate_test(models, images, entries, bs=64):
    ds = D.ManifestDataset(images, entries)
    rows = []
    for xt, x, y, idx in U.make_loader(ds, bs):
        xt, x, y = xt.to(U.DEVICE), x.to(U.DEVICE), y.to(U.DEVICE)
        o, probs, pred, w = models.run(xt, y)
        r = {"i": idx.numpy(), "label": y.cpu().numpy(), "pred": pred.cpu().numpy()}
        for m, img in o.items():
            r[f"psnr_{m}"] = psnr(img, x, "none").cpu().numpy()
            r[f"ssim_{m}"] = ssim(img, x, "none").cpu().numpy()
        for k in range(4):
            r[f"p{k}"] = probs[:, k].cpu().numpy()
            r[f"w{k}"] = w[:, k].cpu().numpy()
        rows.append(pd.DataFrame(r))
    df = pd.concat(rows, ignore_index=True)
    df["type"] = [C.CLASSES[l] for l in df["label"]]
    df["severity"] = [ds.entries[i]["severity"] for i in df["i"]]
    df["name"] = [ds.entries[i]["name"] for i in df["i"]]
    return df


def summary_tables(df):
    agg = {}
    for m in METHODS:
        agg[f"psnr_{m}"] = "mean"
        agg[f"ssim_{m}"] = "mean"
    by_type = df.groupby("type").agg(agg).reindex(C.CLASSES)
    by_sev = df.groupby(["type", "severity"]).agg(agg)
    corrupted = df[df["type"] != "clean"].agg(agg)
    return by_type, by_sev, corrupted


def to_latex_table(tab, metric, caption, label):
    cols = [f"{metric}_{m}" for m in METHODS]
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Condition & " + " & ".join(["Input", "T1", "T2-Oracle", "T2-Pred", "T3-MoE"]) + r" \\", r"\midrule"]
    for idx, row in tab.iterrows():
        name = " / ".join(idx) if isinstance(idx, tuple) else idx
        vals = [row[c] for c in cols]
        best = max(vals[1:])
        cells = [("\\textbf{%.3f}" if (v == best) else "%.3f") % v if metric == "ssim" else
                 (("\\textbf{%.2f}" if v == best else "%.2f") % v if v < 99 else r"$\infty$") for v in vals]
        lines.append(name.replace("_", "-") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------ figures
def _np(t):
    return (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy())


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_grid(models, images, entries, ids, method, path, title_fn=None, ncol_groups=2):
    """Rows of (clean | corrupted | output | |error|) for the given test-manifest ids."""
    ds = D.ManifestDataset(images, entries)
    n = len(ids)
    rows = -(-n // ncol_groups)
    fig, axes = plt.subplots(rows, 4 * ncol_groups, figsize=(2.0 * 4 * ncol_groups, 2.1 * rows))
    axes = np.atleast_2d(axes)
    for a in axes.flat:
        a.axis("off")
    for j, i in enumerate(ids):
        xt, x, y, _ = ds[i]
        xt_, x_ = xt[None].to(U.DEVICE), x[None].to(U.DEVICE)
        o, probs, pred, w = models.run(xt_, torch.tensor([y], device=U.DEVICE))
        out = o[method][0]
        err = (out - x_[0]).abs().mean(0).cpu().numpy()
        r, c = divmod(j, ncol_groups)
        ax = axes[r, 4 * c:4 * c + 4]
        e = ds.entries[i]
        ax[0].imshow(_np(x))
        ax[1].imshow(_np(xt))
        ax[2].imshow(_np(out))
        ax[3].imshow(err, cmap="inferno", vmin=0, vmax=0.5)
        p_out = psnr(out[None], x_).item()
        ax[0].set_title("clean", fontsize=7)
        ax[1].set_title(f"{e['params']['type']} ({e['severity']})", fontsize=7)
        ax[2].set_title(title_fn(e, pred, w, p_out) if title_fn else f"{p_out:.1f} dB", fontsize=7)
        ax[3].set_title("|error|", fontsize=7)
    save(fig, path)


def plot_history(hist, keys, path, title):
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3))
    axes = np.atleast_1d(axes)
    for ax, group in zip(axes, keys):
        for k in group:
            if k in hist[0]:
                ax.plot([h["epoch"] for h in hist], [h[k] for h in hist], label=k)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    save(fig, path)


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), C.CLASSES, rotation=30, fontsize=8)
    ax.set_yticks(range(4), C.CLASSES, fontsize=8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i][j]:.2f}", ha="center", va="center",
                    color="white" if cm[i][j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, path)


def plot_optuna(out):
    try:
        import optuna
        from optuna.visualization import matplotlib as ov
    except ImportError:
        return
    import glob
    for db in glob.glob(os.path.join(out, "optuna", "*.db")):
        _plot_db(optuna, ov, out, f"sqlite:///{os.path.abspath(db)}")


def _plot_db(optuna, ov, out, storage):
    for s in optuna.get_all_study_summaries(storage):
        study = optuna.load_study(study_name=s.study_name, storage=storage)
        try:
            ax = ov.plot_optimization_history(study)
            save(ax.figure, os.path.join(out, "figures", f"optuna_{s.study_name}_history.png"))
            ax = ov.plot_param_importances(study)
            save(ax.figure, os.path.join(out, "figures", f"optuna_{s.study_name}_importance.png"))
        except Exception as ex:   # importance needs >1 completed trial
            print("optuna plot skipped:", s.study_name, ex)
            plt.close("all")


def run_all(trainval, val_man, test, test_man, ck, out, tracker=None, run=None):
    res, fig = os.path.join(out, "results"), os.path.join(out, "figures")
    os.makedirs(res, exist_ok=True)
    models = Models(ck)
    for m in [models.t1, models.cls, models.moe, *models.experts]:
        m.eval()

    # ---- full test-set evaluation
    df = evaluate_test(models, test, test_man)
    df.to_csv(os.path.join(res, "test_per_sample.csv"), index=False)
    by_type, by_sev, corrupted = summary_tables(df)
    by_type.to_csv(os.path.join(res, "test_by_type.csv"))
    by_sev.to_csv(os.path.join(res, "test_by_type_severity.csv"))
    U.save_json(corrupted.to_dict(), os.path.join(res, "test_corrupted_mean.json"))
    with open(os.path.join(res, "tables.tex"), "w") as f:
        f.write(to_latex_table(by_type, "psnr", "Test PSNR (dB) by input condition", "tab:psnr_type") + "\n\n")
        f.write(to_latex_table(by_type, "ssim", "Test SSIM by input condition", "tab:ssim_type") + "\n\n")
        f.write(to_latex_table(by_sev, "psnr", "Test PSNR (dB) by corruption and severity", "tab:psnr_sev") + "\n\n")
        f.write(to_latex_table(by_sev, "ssim", "Test SSIM by corruption and severity", "tab:ssim_sev") + "\n")
    print(by_type.round(3).to_string())
    print(by_sev.round(3).to_string())

    # ---- classifier metrics (val + test)
    val_loader = U.make_loader(D.ManifestDataset(trainval, val_man), 64)
    pv, lv = U.predict_classifier(models.cls, val_loader)
    cls_val = U.classification_metrics(pv, lv, C.CLASSES)
    probs = df[["p0", "p1", "p2", "p3"]].to_numpy()
    cls_test = U.classification_metrics(probs, df["label"].to_numpy(), C.CLASSES)
    cls_test_sev = {sev: U.classification_metrics(probs[(df["severity"] == sev).to_numpy()],
                                                  df["label"][df["severity"] == sev].to_numpy(), C.CLASSES)["accuracy"]
                    for sev in SEV_ORDER if (df["severity"] == sev).any()}
    U.save_json({"val": cls_val, "test": cls_test, "test_accuracy_by_severity": cls_test_sev},
                os.path.join(res, "classifier_metrics.json"))
    plot_confusion(cls_test["confusion_matrix_normalized"], os.path.join(fig, "t2_confusion_test.png"),
                   "Classifier - test (normalised)")
    plot_confusion(cls_val["confusion_matrix_normalized"], os.path.join(fig, "t2_confusion_val.png"),
                   "Classifier - validation (normalised)")

    # ---- routing analysis (Task 3)
    wcols = ["w0", "w1", "w2", "w3"]
    heat = df.groupby(["type", "severity"])[wcols].mean()
    heat = heat.reindex([(t, s) for t in C.CLASSES for s in SEV_ORDER if (t, s) in heat.index])
    heat.to_csv(os.path.join(res, "moe_mean_weights_by_type_severity.csv"))
    W = df[wcols].to_numpy()
    routing = {"mean_weight_overall": W.mean(0).tolist(),
               "argmax_share": np.bincount(W.argmax(1), minlength=4).astype(float).tolist(),
               "inactive_experts(<0.05 overall)": [BRANCHES[k] for k in range(4) if W.mean(0)[k] < 0.05],
               "mean_weight_on_unrelated_inputs": {BRANCHES[k]: float(W[df["label"] != k, k].mean())
                                                   for k in range(4)},
               "gate_argmax_accuracy": float((W.argmax(1) == df["label"]).mean()),
               "dominant_share(max_w>0.9)": float((W.max(1) > 0.9).mean()),
               "mean_entropy": float((-(W * np.log(W + 1e-9)).sum(1)).mean())}
    routing["argmax_share"] = [c / len(W) for c in routing["argmax_share"]]
    U.save_json(routing, os.path.join(res, "moe_routing_analysis.json"))
    f_, ax = plt.subplots(figsize=(5, 6))
    im = ax.imshow(heat.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), BRANCHES, rotation=30, fontsize=8)
    ax.set_yticks(range(len(heat)), [f"{t} / {s}" for t, s in heat.index], fontsize=8)
    for i in range(len(heat)):
        for j in range(4):
            v = heat.to_numpy()[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.5 else "black", fontsize=7)
    ax.set_title("Mean gate weight by true condition", fontsize=9)
    f_.colorbar(im, ax=ax, fraction=0.046)
    save(f_, os.path.join(fig, "t3_routing_heatmap.png"))
    f_, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
    for k, ax in enumerate(axes):
        ax.boxplot([df[df["type"] == t][f"w{k}"] for t in C.CLASSES], showfliers=False)
        ax.set_xticks(range(1, 5), C.CLASSES, rotation=30, fontsize=7)
        ax.set_title(f"weight of {BRANCHES[k]} branch", fontsize=8)
    save(f_, os.path.join(fig, "t3_weight_distribution.png"))

    # ---- method comparison chart
    f_, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, met in zip(axes, ["psnr", "ssim"]):
        x = np.arange(4)
        for j, m in enumerate(METHODS):
            vals = by_type[f"{met}_{m}"].to_numpy().copy()
            vals[vals > 99] = np.nan
            ax.bar(x + (j - 2) * 0.16, vals, 0.16, label=METHODS[m])
        ax.set_xticks(x, C.CLASSES)
        ax.set_ylabel(met.upper())
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    save(f_, os.path.join(fig, "method_comparison.png"))

    # ---- example grids
    rng = np.random.default_rng(0)
    pick = []
    for t in C.CLASSES:
        sub = df[df["type"] == t]
        sevs = ["none"] if t == "clean" else ["low", "medium", "high"]
        for s in sevs:
            cand = sub[sub["severity"] == s]["i"].to_numpy()
            pick += list(rng.choice(cand, size=min(3 if t == "clean" else 1, len(cand)), replace=False))
    pick = pick[:12]
    example_grid(models, test, test_man, pick, "t1", os.path.join(fig, "t1_examples.png"))
    corr = df[df["type"] != "clean"]
    gain = corr["psnr_t1"] - corr["psnr_input"]
    worst = []
    for t in EXPERTS:     # most meaningful failure per corruption + overall worst SSIM
        worst.append(int(corr.loc[gain[corr["type"] == t].idxmin(), "i"]))
    worst.append(int(corr.loc[corr["ssim_t1"].idxmin(), "i"]))
    example_grid(models, test, test_man, worst, "t1", os.path.join(fig, "t1_failures.png"))
    clean_df = df[df["type"] == "clean"]
    clean_worst = clean_df.nsmallest(2, "psnr_t1")["i"].tolist()
    example_grid(models, test, test_man, clean_worst, "t1", os.path.join(fig, "t1_clean_degradation.png"))

    mis = df[(df["pred"] != df["label"])].copy()
    mis["drop"] = mis["psnr_oracle"].clip(upper=60) - mis["psnr_pred"]
    mis_ids = mis.nlargest(6, "drop")["i"].tolist()
    U.save_json({"n_misrouted": int(len(mis)), "misroute_rate": float(len(mis) / len(df)),
                 "misroutes_by_pair": {f"{t}->{C.CLASSES[p]}": int(n)
                                       for (t, p), n in mis.groupby(["type", "pred"]).size().items()},
                 "misroutes_by_severity": {s: int(n) for s, n in mis.groupby("severity").size().items()},
                 "mean_psnr_drop_when_misrouted": float(mis["drop"].mean()) if len(mis) else 0.0},
                os.path.join(res, "hard_routing_failures.json"))
    if mis_ids:
        example_grid(models, test, test_man, mis_ids, "pred", os.path.join(fig, "t2_misrouted.png"),
                     title_fn=lambda e, pred, w, p: f"routed->{C.CLASSES[pred.item()]} {p:.1f}dB")
    example_grid(models, test, test_man, pick[3:9], "oracle", os.path.join(fig, "t2_oracle_examples.png"))

    W_corr = df[df["type"] != "clean"]
    dom = W_corr[W_corr[wcols].max(1) > 0.9]
    dist = W_corr.assign(ent=-(W_corr[wcols] * np.log(W_corr[wcols] + 1e-9)).sum(1)).nlargest(4, "ent")
    wt = lambda e, pred, w, p: "w=" + ",".join(f"{v:.2f}" for v in w[0].tolist())
    if len(dom):
        example_grid(models, test, test_man, dom.sample(min(4, len(dom)), random_state=0)["i"].tolist(), "moe",
                     os.path.join(fig, "t3_dominant_examples.png"), title_fn=wt)
    example_grid(models, test, test_man, dist["i"].tolist(), "moe", os.path.join(fig, "t3_distributed_examples.png"),
                 title_fn=wt)

    # ---- mixed corruption stress test (blur + salt-and-pepper), not seen in training
    mixed = []
    for idx in range(min(300, len(test))):
        mixed.append({"index": idx, "name": "", "seed": 0, "label": 2, "severity": "mixed",
                      "params": {"type": "blur", "ksize": 5, "sigma": 1.5}})
    ds = D.ManifestDataset(test, mixed)
    rows = []
    for k in range(0, len(ds), 64):
        batch = [ds[i] for i in range(k, min(k + 64, len(ds)))]
        xt = torch.stack([b[0] for b in batch])
        xt = torch.stack([D.to_tensor(C.salt_pepper((_np(t) * 255).round().astype(np.uint8), 0.08, 1000 + k + j))
                          for j, t in enumerate(xt)]).to(U.DEVICE)
        x = torch.stack([b[1] for b in batch]).to(U.DEVICE)
        o, probs, pred, w = models.run(xt, torch.full((len(x),), 2, device=U.DEVICE))
        rows.append({**{m: psnr(o[m], x, "none").cpu().numpy() for m in ["input", "t1", "pred", "moe"]},
                     **{f"ssim_{m}": ssim(o[m], x, "none").cpu().numpy() for m in ["input", "t1", "pred", "moe"]},
                     "w": w.cpu().numpy(), "pred": pred.cpu().numpy()})
    mixed_res = {f"psnr_{m}": float(np.concatenate([r[m] for r in rows]).mean()) for m in ["input", "t1", "pred", "moe"]}
    mixed_res.update({f"ssim_{m}": float(np.concatenate([r[f"ssim_{m}"] for r in rows]).mean())
                      for m in ["input", "t1", "pred", "moe"]})
    mixed_res["moe_mean_weights"] = np.concatenate([r["w"] for r in rows]).mean(0).tolist()
    mixed_res["classifier_pred_share"] = (np.bincount(np.concatenate([r["pred"] for r in rows]), minlength=4)
                                          / sum(len(r["pred"]) for r in rows)).tolist()
    U.save_json(mixed_res, os.path.join(res, "mixed_corruption_test.json"))

    # ---- training curves + optuna
    hd = os.path.join(out, "history")
    if os.path.exists(os.path.join(hd, "t1_final.json")):
        plot_history(U.load_json(os.path.join(hd, "t1_final.json")),
                     [["train_loss"], ["val_psnr"], ["val_ssim"]], os.path.join(fig, "t1_curves.png"),
                     "Task 1 universal AE")
    if os.path.exists(os.path.join(hd, "t2_classifier.json")):
        plot_history(U.load_json(os.path.join(hd, "t2_classifier.json")),
                     [["train_loss"], ["train_acc", "val_acc", "val_macro_f1"]], os.path.join(fig, "t2_cls_curves.png"),
                     "Task 2 classifier")
    for k in EXPERTS:
        p = os.path.join(hd, f"t2_specialist_{k}.json")
        if os.path.exists(p):
            plot_history(U.load_json(p), [["train_loss"], ["val_psnr"], ["val_ssim"]],
                         os.path.join(fig, f"t2_spec_{k}_curves.png"), f"Specialist: {k}")
    if os.path.exists(os.path.join(hd, "t3_soft_moe.json")):
        plot_history(U.load_json(os.path.join(hd, "t3_soft_moe.json")),
                     [["train_loss", "train_l1", "train_ssim"], ["train_ce", "train_bal"], ["val_psnr"],
                      ["val_w0", "val_w1", "val_w2", "val_w3"]], os.path.join(fig, "t3_moe_curves.png"),
                     "Task 3 soft MoE (warm-up then joint)")
    plot_optuna(out)

    if tracker is not None and run is not None:
        tracker.wandb.log({"test_by_type": tracker.wandb.Table(dataframe=by_type.reset_index()),
                           "test_by_type_severity": tracker.wandb.Table(dataframe=by_sev.reset_index()),
                           "classifier_test_accuracy": cls_test["accuracy"],
                           "classifier_test_macro_f1": cls_test["macro_f1"]})
        for fn in sorted(os.listdir(fig)):
            tracker.wandb.log({f"figures/{fn[:-4]}": tracker.wandb.Image(os.path.join(fig, fn))})
    print("evaluation done ->", res)
