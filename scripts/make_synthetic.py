"""Synthetic stand-in data with the exact Sen1Floods11 / Pakistan file layout.
Only for testing that the pipeline runs end to end (CI, smoke tests) — not for results.

    python scripts/make_synthetic.py --out data/synthetic --size 128
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter


def scene(rng, h, w, water_db=(-22, -27), land_db=(-9, -15), wind=0.0, frac=0.3):
    field = gaussian_filter(rng.standard_normal((h, w)), sigma=max(h, w) / 12)
    water = field > np.quantile(field, 1 - frac)
    speckle = 10 * np.log10(rng.gamma(4.4, 1 / 4.4, size=(2, h, w)))
    x = np.empty((2, h, w), np.float32)
    for b in range(2):
        x[b] = np.where(water, water_db[b] + wind, land_db[b]) + speckle[b]
    return x, water


def write(path, arr, transform, crs, nodata=None):
    if arr.ndim == 2:
        arr = arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[-2], width=arr.shape[-1],
                       count=arr.shape[0], dtype=arr.dtype, crs=crs, transform=transform, nodata=nodata) as d:
        d.write(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.out) / "sen1floods11"
    for d in ["S1Hand", "LabelHand", "splits"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    countries = ["Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA"]
    plan = {"train": 40, "valid": 10, "test": 15, "bolivia": 10}
    cid = 100000
    for split, n in plan.items():
        rows = []
        for _ in range(n):
            c = "Bolivia" if split == "bolivia" else countries[rng.integers(len(countries))]
            wind = rng.uniform(3, 7) if c == "Bolivia" else rng.uniform(0, 2)  # harder, shifted water
            x, wat = scene(rng, a.size, a.size, wind=wind, frac=rng.uniform(0.05, 0.5))
            y = wat.astype(np.int16)
            y[rng.random(y.shape) < 0.02] = -1
            name = f"{c}_{cid}"
            cid += 1
            t = from_origin(rng.uniform(-80, 80), rng.uniform(-30, 30), 1e-4, 1e-4)
            write(root / "S1Hand" / f"{name}_S1Hand.tif", x, t, "EPSG:4326")
            write(root / "LabelHand" / f"{name}_LabelHand.tif", y[None], t, "EPSG:4326")
            rows.append(f"{name}_S1Hand.tif,{name}_LabelHand.tif")
        (root / "splits" / f"flood_{split}_data.csv").write_text("\n".join(rows) + "\n")

    pk = Path(a.out) / "pakistan"
    pk.mkdir(parents=True, exist_ok=True)
    H, W = 3 * a.size + 37, 4 * a.size + 11  # deliberately not a multiple of the tile
    t = from_origin(650000, 2960000, 10, 10)
    pre, _ = scene(rng, H, W, frac=0.05)
    flood, _ = scene(rng, H, W, wind=2.5, frac=0.35)
    flood[:, :20, :30] = np.nan
    write(pk / "pre_s1_db.tif", pre, t, "EPSG:32642", nodata=np.nan)
    write(pk / "flood_s1_db.tif", flood, t, "EPSG:32642", nodata=np.nan)
    print(f"synthetic data in {a.out}")


if __name__ == "__main__":
    main()
