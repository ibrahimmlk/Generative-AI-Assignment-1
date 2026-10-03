"""Collect numbers + figures from the training outputs into report/ (results_macros.tex, figures/, tables).

    python scripts/make_report_assets.py --out outputs --sketch outputs_sketch
"""
import argparse
import glob
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "report")


def load(path, default=None):
    if not os.path.exists(path):
        print("missing", path)
        return default
    with open(path) as f:
        return json.load(f)


def fmt(v, d=2):
    if isinstance(v, float):
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.1e}"
        return f"{v:.{d}f}"
    return str(v)


def params_str(p):
    keymap = {"lr": "lr", "lr_g": "lr$_G$", "lr_d": "lr$_D$", "batch_size": "batch", "latent_ch": "$c_z$", "base": "$b$",
              "dropout": "dropout", "alpha": r"$\alpha$", "channels": "channels", "weight_decay": "wd", "tau": r"$\tau$",
              "lambda_ce": r"$\lambda_c$", "lambda_bal": r"$\lambda_b$", "lambda_l1": r"$\lambda_1$", "emb_dim": "$d$"}
    out = []
    for k, v in p.items():
        name = keymap.get(k, k.replace("_", r"\_"))
        if k == "lambda_l1" and "lr_g" in p:
            name = r"$\lambda_{L1}$"
        out.append(f"{name}={fmt(v, 3) if isinstance(v, float) else v}")
    return ", ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--sketch", default="outputs_sketch")
    a = ap.parse_args()
    o, s = a.out, a.sketch
    M = {}
    meta = load(os.path.join(REPORT, "meta.json"), {})
    M["AUTHORNAME"] = meta.get("author", "Author Name")
    M["AUTHORAFFIL"] = meta.get("affiliation", "Department, University")
    M["VIDEOURL"] = meta.get("video", "https://youtube.com/")
    M["WANDBURL"] = meta.get("wandb", "https://wandb.ai/")

    ds = load(os.path.join(o, "results", "data_summary.json"), {})
    M.update(NTRAIN=ds.get("train"), NVAL=ds.get("val"), NTEST=ds.get("test"), NTESTENTRIES=ds.get("test_entries"))
    M["NTRAINVAL"] = (ds.get("train") or 0) + (ds.get("val") or 0)

    t1 = load(os.path.join(o, "optuna", "t1_universal_ae.json"), {})
    if t1:
        bp = t1["best_params"]
        M.update(TONETRIALS=t1["n_trials"], TONEPRUNED=t1["pruned"], TONEBEST=t1["best_trial"],
                 TONEBESTVAL=fmt(t1["best_value"], 4), TONELR=fmt(bp["lr"], 5), TONEBS=bp["batch_size"],
                 TONELATCH=bp["latent_ch"], TONEBASE=bp["base"], TONEDROP=fmt(bp["dropout"], 3),
                 TONEALPHA=fmt(bp["alpha"], 3), TONELATDIM=f"{16 * 16 * bp['latent_ch']:,}".replace(",", "{,}"),
                 TONECOMPRESS=fmt(49152 / (16 * 16 * bp["latent_ch"]), 0))
    h = load(os.path.join(o, "history", "t1_final.json"), [])
    M["TONEEPOCHS"] = len(h)
    M["TONETRIALEP"] = meta.get("t1_trial_epochs", 6)

    cls = load(os.path.join(o, "optuna", "t2_classifier.json"), {})
    if cls:
        M.update(CLSTRIALS=cls["n_trials"], CLSBESTPARAMS=params_str(cls["best_params"]))
    sp = load(os.path.join(o, "optuna", "t2_specialists.json"), {})
    if sp:
        M.update(SPTRIALS=sp["n_trials"], SPBESTPARAMS=params_str(sp["best_params"]))
    hs = load(os.path.join(o, "history", "t2_specialist_blur.json"), [])
    M["SPEPOCHS"] = len(hs)
    moe = load(os.path.join(o, "optuna", "t3_soft_moe_v2.json"), {})
    if moe:
        M.update(MOETRIALS=moe["n_trials"], MOEBESTPARAMS=params_str(moe["best_params"]),
                 MOECOLLAPSED=sum(1 for t in moe["trials"] if t["user_attrs"].get("pruned_reason") == "routing_collapse"))
    hm = load(os.path.join(o, "history", "t3_soft_moe.json"), [])
    M["MOEWARM"] = sum(1 for r in hm if r["stage"] == "warmup")
    M["MOEEPOCHS"] = sum(1 for r in hm if r["stage"] == "joint")

    corr = load(os.path.join(o, "results", "test_corrupted_mean.json"), {})
    for key, name in [("t1", "TONE"), ("pred", "PRED"), ("oracle", "ORACLE"), ("moe", "MOE"), ("input", "INPUT")]:
        if corr:
            M[f"{name}PSNRCORR"] = fmt(corr[f"psnr_{key}"], 2)
            M[f"{name}SSIMCORR"] = fmt(corr[f"ssim_{key}"], 3)

    ver = load(os.path.join(o, "onnx", "onnx_verification.json"), []) + load(os.path.join(s, "onnx", "onnx_verification.json"), [])
    ver = list({v["file"]: v for v in ver}.values())
    M["ONNXMAXDIFF"] = f"{max(v['max_abs_diff'] for v in ver):.1e}" if ver else "?"

    with open(os.path.join(REPORT, "onnx_table.tex"), "w") as f:
        rows = [r"\begin{table}[t]", r"\centering",
                r"\caption{ONNX export: size and maximum absolute deviation from PyTorch}", r"\label{tab:onnx}",
                r"\begin{tabular}{lrr}", r"\toprule", r"Model & Size (MB) & max $|\Delta|$ \\", r"\midrule"]
        rows += [v["file"].replace("_", r"\_") + f" & {v['size_mb']:.1f} & {v['max_abs_diff']:.1e} " + r"\\" for v in ver]
        rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        f.write("\n".join(rows) + "\n")

    fs = load(os.path.join(s, "sketch", "split.json"), {})
    if fs:
        M.update(FSTRAIN=fs["train"], FSVAL=fs["val"], FSTEST=fs["test"], FSTRAINFULL=fs["train"] + fs["val"],
                 FSVALSTYLES="/".join(map(str, fs["val_style_counts"])))
    g = load(os.path.join(s, "optuna", "t4_sketch_gan.json"), {})
    if g:
        M.update(FSTRIALS=g["n_trials"], FSBESTPARAMS=params_str(g["best_params"]))
    hg = load(os.path.join(s, "history", "t4_gan.json"), [])
    M["FSEPOCHS"] = len(hg)
    M["FSTRIALEP"] = meta.get("t4_trial_epochs", 30)

    with open(os.path.join(REPORT, "results_macros.tex"), "w", encoding="utf8") as f:
        f.write("% generated by scripts/make_report_assets.py\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v if v is not None else '?'}}}\n")

    # figures + tables
    fig_dir = os.path.join(REPORT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for src in glob.glob(os.path.join(o, "figures", "*.png")) + glob.glob(os.path.join(s, "figures", "*.png")):
        shutil.copy(src, fig_dir)
    prog = sorted(glob.glob(os.path.join(s, "sketch", "progress", "*.png")))
    for p in prog:
        shutil.copy(p, os.path.join(fig_dir, "t4_progress_" + os.path.basename(p)))
    if os.path.exists(os.path.join(o, "results", "tables.tex")):
        shutil.copy(os.path.join(o, "results", "tables.tex"), os.path.join(REPORT, "tables_generated.tex"))
    # results that should live in the repository (small files)
    res_dir = os.path.join(ROOT, "results")
    for sub in ["results", "optuna", "manifests", "history"]:
        for base in (o, s):
            src = os.path.join(base, sub)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(res_dir, sub), dirs_exist_ok=True)
    for base in (o, s):
        if os.path.isdir(os.path.join(base, "figures")):
            shutil.copytree(os.path.join(base, "figures"), os.path.join(res_dir, "figures"), dirs_exist_ok=True)
    print(json.dumps(M, indent=1))


if __name__ == "__main__":
    main()
