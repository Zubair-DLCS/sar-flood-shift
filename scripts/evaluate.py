"""Evaluate single model / MC dropout / deep ensemble (+ temperature scaling) on
Sen1Floods11 test (in-distribution countries) and Bolivia (held-out country).

    python scripts/evaluate.py --data data/sen1floods11 --runs runs/ensemble --out results

Writes results/metrics.json, results/table.md, results/per_chip.csv and figures.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodshift.data import chip_country, normalise, read_label, read_s1_db, read_split, IGNORE  # noqa: E402
from floodshift.metrics import Accumulator  # noqa: E402
from floodshift.model import load_checkpoint  # noqa: E402
from floodshift.uncertainty import fit_temperature, member_logits, summarise  # noqa: E402

METHODS = ["single", "mc_dropout", "ensemble", "ensemble_ts"]
LABELS = {"single": "Single U-Net", "mc_dropout": "MC dropout", "ensemble": "Deep ensemble",
          "ensemble_ts": "Ensemble + temp. scaling"}
SPLIT_NAMES = {"test": "Test (seen countries)", "bolivia": "Bolivia (unseen country)"}


def load_chip(s1, lab, stats, norm_mode):
    x_db = read_s1_db(s1)
    x, valid = normalise(x_db, stats, norm_mode)
    y = read_label(lab)
    y[~valid] = IGNORE
    return x_db, x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sen1floods11")
    ap.add_argument("--runs", default="runs/ensemble")
    ap.add_argument("--out", default="results")
    ap.add_argument("--mc-samples", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)

    ckpts = sorted(Path(a.runs).glob("member_*.pt"))
    assert ckpts, f"no member_*.pt in {a.runs}"
    loaded = [load_checkpoint(c, device) for c in ckpts]
    models = [m for m, _, _ in loaded]
    stats = loaded[0][1]
    norm_mode = loaded[0][2].get("norm_mode", "global")
    print(f"{len(models)} members, norm_mode={norm_mode}, device={device}")

    def split(name):
        p = read_split(a.data, name)
        return p[:a.limit] if a.limit else p

    # ---- temperature on the validation split ----
    vl, vy = [], []
    for s1, lab in split("valid"):
        _, x, y = load_chip(s1, lab, stats, norm_mode)
        lg = member_logits(models, torch.from_numpy(x)[None].to(device))[:, 0].cpu().numpy()
        vl.append(lg); vy.append(y)
    T = fit_temperature(vl, vy)
    print(f"fitted temperature T={T:.2f}")
    del vl, vy

    results, rows, accs, examples = {}, [], {}, []
    for sp in ["test", "bolivia"]:
        accs[sp] = {m: Accumulator(seed=1) for m in METHODS}
        for s1, lab in split(sp):
            x_db, x, y = load_chip(s1, lab, stats, norm_mode)
            xt = torch.from_numpy(x)[None].to(device)
            ens = member_logits(models, xt)[:, 0]
            outs = {
                "single": summarise(ens[:1]),
                "mc_dropout": summarise(member_logits(models[:1], xt, a.mc_samples)[:, 0]),
                "ensemble": summarise(ens),
                "ensemble_ts": summarise(ens, T),
            }
            name = Path(s1).name.replace("_S1Hand.tif", "")
            for m, o in outs.items():
                accs[sp][m].update(o["prob"], o["entropy"], y, name)
            c = accs[sp]["ensemble"].per_chip[-1] if accs[sp]["ensemble"].per_chip else None
            if c is not None and c["name"] == name:
                rows.append({"split": sp, "country": chip_country(s1), **c,
                             "mean_mutual_info": float(outs["ensemble"]["mutual_info"][y != -1].mean())
                             if (y != -1).any() else np.nan,
                             "single_iou": accs[sp]["single"].per_chip[-1]["iou"]})
                examples.append((c["accuracy"], sp, name, x_db, y, outs["ensemble"]))
                examples.sort(key=lambda e: e[0])
                examples = examples[:4]  # keep the 4 worst chips only
        results[sp] = {m: accs[sp][m].summary() for m in METHODS}
        print(f"\n== {SPLIT_NAMES[sp]} ==")
        for m in METHODS:
            r = results[sp][m]
            print(f"  {LABELS[m]:26s} IoU {r['iou']:.3f}  ECE {r['ece']:.4f}  "
                  f"AUROC(err) {r['auroc_error_detection']:.3f}  mean H {r['mean_uncertainty']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_chip.csv", index=False)

    # ---- does chip-level uncertainty track chip-level error? ----
    rho, pval = spearmanr(df["mean_unc"], 1 - df["accuracy"])
    wet = df[df["water_frac"] >= 0.01]
    rho_iou, p_iou = spearmanr(wet["mean_unc"], wet["iou"]) if len(wet) > 2 else (np.nan, np.nan)
    results["temperature"] = T
    results["chip_level"] = {"spearman_uncertainty_vs_error": float(rho), "p": float(pval),
                             "spearman_uncertainty_vs_iou_wet_chips": float(rho_iou), "p_iou": float(p_iou),
                             "n_chips": int(len(df)), "n_wet_chips": int(len(wet))}
    results["per_country"] = (df.groupby("country")
                              .agg(n=("name", "size"), iou=("iou", "mean"), error=("accuracy", lambda s: 1 - s.mean()),
                                   mean_unc=("mean_unc", "mean"))
                              .reset_index().to_dict(orient="records"))
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=float))

    # ---- markdown table ----
    lines = ["| Split | Method | IoU | F1 | ECE | Error-detection AUROC | Risk @ 80% coverage | Mean entropy |",
             "|---|---|---|---|---|---|---|---|"]
    for sp in ["test", "bolivia"]:
        for m in METHODS:
            r = results[sp][m]
            lines.append(f"| {SPLIT_NAMES[sp]} | {LABELS[m]} | {r['iou']:.3f} | {r['f1']:.3f} | {r['ece']:.4f} | "
                         f"{r['auroc_error_detection']:.3f} | {r['risk_at_80pct_coverage']:.4f} | {r['mean_uncertainty']:.3f} |")
    lines.append(f"\nTemperature: {T:.2f}.  Chip-level Spearman(mean entropy, error rate) = {rho:.2f} "
                 f"(p={pval:.1e}, n={len(df)}); Spearman(mean entropy, IoU) on chips with >=1% water = {rho_iou:.2f}.")
    (out / "table.md").write_text("\n".join(lines) + "\n")

    # ---- figures ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, sp in zip(axs, ["test", "bolivia"]):
        for m, st in [("single", "--"), ("ensemble", "-"), ("ensemble_ts", ":")]:
            conf, acc, _ = accs[sp][m].reliability()
            ax.plot(conf, acc, st, marker="o", ms=3, label=f"{LABELS[m]} (ECE {results[sp][m]['ece']:.3f})")
        ax.plot([0.5, 1], [0.5, 1], color="grey", lw=0.8)
        ax.set(title=SPLIT_NAMES[sp], xlabel="Confidence", ylabel="Accuracy", xlim=(0.5, 1), ylim=(0.5, 1))
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "reliability.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for sp, col in [("test", "C0"), ("bolivia", "C3")]:
        for m, st in [("single", "--"), ("ensemble", "-")]:
            cov, risk = accs[sp][m].curve()
            ax.plot(cov, risk, st, color=col, label=f"{SPLIT_NAMES[sp]}: {LABELS[m]}")
    ax.set(xlabel="Coverage (most-certain pixels kept)", ylabel="Pixel error rate", title="Risk-coverage")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "figures" / "risk_coverage.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for sp, col in [("test", "C0"), ("bolivia", "C3")]:
        d = df[df.split == sp]
        ax.scatter(d["mean_unc"], 1 - d["accuracy"], s=14, alpha=0.7, color=col, label=SPLIT_NAMES[sp])
    ax.set(xlabel="Chip mean predictive entropy (ensemble)", ylabel="Chip pixel error rate",
           title=f"Does uncertainty track error?  Spearman ρ = {rho:.2f}")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "figures" / "uncertainty_vs_error.png", dpi=160); plt.close(fig)

    pc = pd.DataFrame(results["per_country"]).sort_values("iou")
    fig, ax1 = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(pc))
    ax1.bar(xs - 0.2, pc["iou"], 0.4, label="IoU", color="C0")
    ax2 = ax1.twinx(); ax2.bar(xs + 0.2, pc["mean_unc"], 0.4, label="Mean entropy", color="C1")
    ax1.set_xticks(xs); ax1.set_xticklabels(pc["country"], rotation=45, ha="right")
    ax1.set_ylabel("IoU (mean over chips)"); ax2.set_ylabel("Mean predictive entropy")
    ax1.set_title("Per-country performance vs uncertainty (ensemble)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(loc="lower center", ncol=2, fontsize=8, frameon=False); fig.savefig(out / "figures" / "per_country.png", dpi=160); plt.close(fig)

    if examples:
        fig, axs = plt.subplots(len(examples), 4, figsize=(11, 2.8 * len(examples)), squeeze=False)
        for r, (_, sp, name, x_db, y, o) in enumerate(examples):
            yy = np.where(y == -1, np.nan, y)
            for c, (img, t, kw) in enumerate([
                (np.clip(x_db[0], -30, 0), f"{name} VV (dB)", {"cmap": "gray"}),
                (yy, "Label", {"cmap": "Blues", "vmin": 0, "vmax": 1}),
                (o["prob"], "P(water)", {"cmap": "Blues", "vmin": 0, "vmax": 1}),
                (o["entropy"], "Entropy", {"cmap": "magma", "vmin": 0, "vmax": 1})]):
                axs[r, c].imshow(img, **kw); axs[r, c].set_title(t, fontsize=8); axs[r, c].axis("off")
        fig.suptitle("Worst chips (ensemble): does the entropy map light up where it is wrong?", fontsize=10)
        fig.tight_layout(); fig.savefig(out / "figures" / "worst_chips.png", dpi=130); plt.close(fig)

    print(f"\nwrote {out}/metrics.json, table.md, per_chip.csv, figures/")


if __name__ == "__main__":
    main()
