"""Sen1Floods11 loading, normalisation and augmentation.

Sen1Floods11 hand-labelled chips (Bonafilia et al., CVPRW 2020):
  S1Hand/<Country>_<id>_S1Hand.tif       2 bands (VV, VH), sigma0 in dB, 512x512, 10 m
  LabelHand/<Country>_<id>_LabelHand.tif 1 band: -1 = no data, 0 = not water, 1 = water
Official splits: train / valid / test (11 countries) + Bolivia (held-out country).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

IGNORE = -1
DB_CLIP = (-40.0, 5.0)  # sigma0 dB range kept; outside is clipped

SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


def read_split(root: str | Path, split: str) -> list[tuple[Path, Path]]:
    """Return (s1_path, label_path) pairs for an official split."""
    root = Path(root)
    pairs = []
    with open(root / "splits" / SPLIT_FILES[split]) as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            s1, lab = row[0].strip(), row[1].strip()
            pairs.append((root / "S1Hand" / s1, root / "LabelHand" / lab))
    return pairs


def chip_country(path: Path) -> str:
    return Path(path).name.split("_")[0]


def read_s1_db(path: str | Path) -> np.ndarray:
    """Read a 2-band (VV, VH) dB raster as float32 (2, H, W), NaN where invalid."""
    with rasterio.open(path) as src:
        x = src.read([1, 2]).astype(np.float32)
        if src.nodata is not None:
            x[x == src.nodata] = np.nan
    x[~np.isfinite(x)] = np.nan
    return x


def read_label(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as src:
        y = src.read(1).astype(np.int64)
    y[(y != 0) & (y != 1)] = IGNORE
    return y


@dataclass
class NormStats:
    mean: tuple[float, float]
    std: tuple[float, float]

    def to_dict(self):
        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(d["mean"]), tuple(d["std"]))


def compute_stats(pairs: list[tuple[Path, Path]]) -> NormStats:
    s = np.zeros(2)
    ss = np.zeros(2)
    n = np.zeros(2)
    for s1, _ in pairs:
        x = np.clip(read_s1_db(s1), *DB_CLIP)
        for b in range(2):
            v = x[b][np.isfinite(x[b])].astype(np.float64)
            s[b] += v.sum()
            ss[b] += (v**2).sum()
            n[b] += v.size
    mean = s / n
    std = np.sqrt(ss / n - mean**2)
    return NormStats(tuple(mean.tolist()), tuple(std.tolist()))


def normalise(x_db: np.ndarray, stats: NormStats, mode: str = "global") -> tuple[np.ndarray, np.ndarray]:
    """dB -> standardised input. Returns (x, valid_mask).

    mode="global": standardise with training-set statistics (the usual choice).
    mode="per_image": standardise each image with its own statistics; an
        ablation that removes absolute-backscatter shift between sensors/products.
    """
    valid = np.all(np.isfinite(x_db), axis=0)
    x = np.clip(x_db, *DB_CLIP)
    out = np.zeros_like(x, dtype=np.float32)
    for b in range(2):
        if mode == "per_image" and valid.any():
            m, sd = float(x[b][valid].mean()), float(x[b][valid].std() + 1e-6)
        else:
            m, sd = stats.mean[b], stats.std[b]
        out[b] = np.where(valid, (x[b] - m) / sd, 0.0)
    return out.astype(np.float32), valid


class Sen1Floods11(Dataset):
    def __init__(self, pairs, stats: NormStats, train: bool = False, crop: int = 256,
                 norm_mode: str = "global"):
        self.pairs = list(pairs)
        self.stats = stats
        self.train = train
        self.crop = crop
        self.norm_mode = norm_mode
        # chips are small (2 MB each); cache them in memory
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self):
        return len(self.pairs)

    def _load(self, i):
        if i not in self._cache:
            s1, lab = self.pairs[i]
            x, valid = normalise(read_s1_db(s1), self.stats, self.norm_mode)
            y = read_label(lab)
            y[~valid] = IGNORE
            self._cache[i] = (x, y)
        return self._cache[i]

    def __getitem__(self, i):
        x, y = self._load(i)
        if self.train:
            # torch seeds each DataLoader worker differently, numpy copies would not
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
            H, W = y.shape
            c = min(self.crop, H, W)
            r0 = rng.integers(0, H - c + 1)
            c0 = rng.integers(0, W - c + 1)
            x = x[:, r0:r0 + c, c0:c0 + c]
            y = y[r0:r0 + c, c0:c0 + c]
            k = int(rng.integers(0, 4))
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
