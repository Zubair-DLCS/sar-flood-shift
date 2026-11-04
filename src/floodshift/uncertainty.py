"""Deep-ensemble / MC-dropout prediction, temperature scaling and tiled inference."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .metrics import binary_entropy


def enable_mc_dropout(model: nn.Module):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def member_logits(models, x: torch.Tensor, mc_samples: int = 0) -> torch.Tensor:
    """Stack of logits, shape (S, B, H, W). S = n_models (ensemble) or
    n_models * mc_samples (MC dropout inside every member)."""
    outs = []
    for m in models:
        if mc_samples > 0:
            enable_mc_dropout(m)
            outs += [m(x) for _ in range(mc_samples)]
            m.eval()
        else:
            m.eval()
            outs.append(m(x))
    return torch.stack(outs)


def summarise(logits: torch.Tensor, temperature: float = 1.0) -> dict:
    """logits (S, ...) -> mean prob, total / aleatoric entropy, mutual information."""
    p = torch.sigmoid(logits / temperature).cpu().numpy()
    pm = p.mean(0)
    total = binary_entropy(pm)
    aleatoric = binary_entropy(p).mean(0)
    return {"prob": pm, "entropy": total, "mutual_info": np.clip(total - aleatoric, 0, None),
            "std": p.std(0)}


def fit_temperature(logit_list: list[np.ndarray], label_list: list[np.ndarray],
                    grid=np.geomspace(0.1, 10.0, 121), max_pixels: int = 2_000_000, seed: int = 0) -> float:
    """Single temperature for the ensemble, minimising NLL of the averaged probability.
    logit_list[i]: (S, H, W) member logits for chip i; label_list[i]: (H, W)."""
    rng = np.random.default_rng(seed)
    L, Y = [], []
    for lg, y in zip(logit_list, label_list):
        m = y != -1
        L.append(lg[:, m]); Y.append(y[m])
    L = np.concatenate(L, axis=1); Y = np.concatenate(Y).astype(np.float64)
    if Y.size > max_pixels:
        sel = rng.choice(Y.size, max_pixels, replace=False)
        L, Y = L[:, sel], Y[sel]
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = np.clip((1 / (1 + np.exp(-L / t))).mean(0), 1e-7, 1 - 1e-7)
        nll = -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def _window(tile: int) -> np.ndarray:
    w = np.hanning(tile + 2)[1:-1]
    return np.outer(w, w).astype(np.float32) + 1e-3


@torch.no_grad()
def tiled_predict(models, x: np.ndarray, tile: int = 512, overlap: int = 128,
                  temperature: float = 1.0, mc_samples: int = 0, device="cpu",
                  batch: int = 4) -> dict:
    """Sliding-window inference over a large normalised (2, H, W) array.
    Blends the per-member *probabilities* with a Hann window, then summarises."""
    C, H, W = x.shape
    ph, pw = max(tile - H, 0), max(tile - W, 0)
    xp = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect" if (ph < H and pw < W) else "constant")
    Hp, Wp = xp.shape[1:]
    stride = tile - overlap
    rows = list(range(0, Hp - tile + 1, stride)) or [0]
    cols = list(range(0, Wp - tile + 1, stride)) or [0]
    if rows[-1] != Hp - tile: rows.append(Hp - tile)
    if cols[-1] != Wp - tile: cols.append(Wp - tile)
    win = _window(tile)
    acc = None
    wsum = np.zeros((Hp, Wp), np.float32)
    coords = [(r, c) for r in rows for c in cols]
    for i in range(0, len(coords), batch):
        cb = coords[i:i + batch]
        xb = torch.from_numpy(np.stack([xp[:, r:r + tile, c:c + tile] for r, c in cb])).to(device)
        p = torch.sigmoid(member_logits(models, xb, mc_samples) / temperature).cpu().numpy()  # S,B,t,t
        if acc is None:
            acc = np.zeros((p.shape[0], Hp, Wp), np.float32)
        for j, (r, c) in enumerate(cb):
            acc[:, r:r + tile, c:c + tile] += p[:, j] * win
            wsum[r:r + tile, c:c + tile] += win
    probs = (acc / wsum)[:, :H, :W]
    pm = probs.mean(0)
    total = binary_entropy(pm)
    alea = binary_entropy(probs).mean(0)
    return {"prob": pm, "entropy": total, "mutual_info": np.clip(total - alea, 0, None),
            "std": probs.std(0)}
