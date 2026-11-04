"""Segmentation, calibration and uncertainty-quality metrics.

All accumulators ignore pixels whose label is -1.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)  # bits, in [0, 1]


def seg_scores(tp, fp, fn, tn) -> dict:
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, EPS)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec, "accuracy": acc}


class Accumulator:
    """Streams over chips; keeps exact confusion counts, a calibration histogram,
    and a random pixel subsample for AUROC / risk-coverage."""

    def __init__(self, n_bins: int = 15, sample_per_chip: int = 20000, seed: int = 0):
        self.tp = self.fp = self.fn = self.tn = 0
        self.n_bins = n_bins
        self.bin_count = np.zeros(n_bins)
        self.bin_conf = np.zeros(n_bins)
        self.bin_correct = np.zeros(n_bins)
        self.sample_per_chip = sample_per_chip
        self.rng = np.random.default_rng(seed)
        self.s_unc, self.s_err = [], []
        self.per_chip = []

    def update(self, prob: np.ndarray, unc: np.ndarray, label: np.ndarray, name: str = ""):
        m = label != -1
        if not m.any():
            return
        p, u, y = prob[m], unc[m], label[m].astype(bool)
        pred = p >= 0.5
        tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
        self.tp += tp; self.fp += fp; self.fn += fn; self.tn += tn

        conf = np.where(pred, p, 1 - p)
        correct = pred == y
        idx = np.minimum(((conf - 0.5) / 0.5 * self.n_bins).astype(int), self.n_bins - 1)
        self.bin_count += np.bincount(idx, minlength=self.n_bins)
        self.bin_conf += np.bincount(idx, weights=conf, minlength=self.n_bins)
        self.bin_correct += np.bincount(idx, weights=correct, minlength=self.n_bins)

        k = min(self.sample_per_chip, p.size)
        sel = self.rng.choice(p.size, size=k, replace=False)
        self.s_unc.append(u[sel]); self.s_err.append(~correct[sel])

        chip = seg_scores(tp, fp, fn, tn)
        chip.update({"name": name, "water_frac": float(y.mean()),
                     "mean_unc": float(u.mean()), "n_pixels": int(m.sum())})
        self.per_chip.append(chip)

    def summary(self) -> dict:
        out = seg_scores(self.tp, self.fp, self.fn, self.tn)
        nz = self.bin_count > 0
        acc_b = self.bin_correct[nz] / self.bin_count[nz]
        conf_b = self.bin_conf[nz] / self.bin_count[nz]
        w = self.bin_count[nz] / self.bin_count.sum()
        out["ece"] = float(np.sum(w * np.abs(acc_b - conf_b)))
        unc = np.concatenate(self.s_unc); err = np.concatenate(self.s_err)
        out["error_rate"] = float(err.mean())
        out["mean_uncertainty"] = float(unc.mean())
        out["auroc_error_detection"] = (float(roc_auc_score(err, unc))
                                        if 0 < err.sum() < err.size else float("nan"))
        cov, risk = risk_coverage(unc, err)
        out["aurc"] = float(np.trapezoid(risk, cov))
        out["risk_at_80pct_coverage"] = float(np.interp(0.8, cov, risk))
        return out

    def reliability(self):
        nz = self.bin_count > 0
        return (self.bin_conf[nz] / self.bin_count[nz],
                self.bin_correct[nz] / self.bin_count[nz],
                self.bin_count[nz])

    def curve(self):
        return risk_coverage(np.concatenate(self.s_unc), np.concatenate(self.s_err))


def risk_coverage(unc: np.ndarray, err: np.ndarray, n_points: int = 101):
    """Keep the most-certain fraction `coverage` of pixels; risk = error rate among them."""
    order = np.argsort(unc, kind="stable")
    cum_err = np.cumsum(err[order])
    n = len(order)
    cov = np.linspace(0.01, 1.0, n_points)
    k = np.maximum((cov * n).astype(int), 1)
    risk = cum_err[k - 1] / k
    return cov, risk
