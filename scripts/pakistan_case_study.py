"""Case study: the 2022 Sindh floods (Pakistan) — an event, year and SAR product the
model never saw in training. (Sen1Floods11 does contain a few Pakistan chips from an
earlier event, so this is a new-event / new-product shift, not a new-country one.)

Step 1 — fetch Sentinel-1 RTC (pre-event + flood) from Microsoft Planetary Computer:
    python scripts/pakistan_case_study.py fetch --out data/pakistan

Step 2 — run the ensemble, compare against an Otsu-threshold baseline and (optionally)
an official flood-extent polygon (e.g. UNOSAT), write maps, metrics and figures:
    python scripts/pakistan_case_study.py predict --runs runs/ensemble --data data/pakistan \
        --out results/pakistan [--reference unosat_flood.shp] [--sen1floods11-metrics results/metrics.json]

Note on the shift: Sen1Floods11 is GEE sigma0 (GRD, dB); Planetary Computer gives
radiometrically terrain-corrected gamma0. Converting to dB keeps the units the same,
but the radiometry differs by a few dB on top of the geographic shift. That is part of
the experiment, not a bug: a real deployment would face exactly this.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Default AOI: Dadu / Johi / Lake Manchar, Sindh — among the worst-hit districts in 2022.
DEFAULT_BBOX = (67.45, 26.25, 67.95, 26.75)  # lon_min, lat_min, lon_max, lat_max
FLOOD_DATES = "2022-08-25/2022-09-08"
PRE_DATES = "2022-05-15/2022-06-20"


# ----------------------------------------------------------------------------- fetch
def cmd_fetch(a):
    import planetary_computer
    import pystac_client
    import rioxarray  # noqa: F401
    import xarray as xr
    from shapely.geometry import box, shape

    bbox = tuple(a.bbox)
    aoi = box(*bbox)
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)

    def pick(dates, orbit=None):
        items = list(cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=dates).items())
        if orbit is not None:
            items = [i for i in items if i.properties.get("sat:relative_orbit") == orbit]
        if not items:
            raise SystemExit(f"no Sentinel-1 RTC items for {dates} (orbit={orbit})")
        cov = [shape(i.geometry).intersection(aoi).area / aoi.area for i in items]
        best = max(range(len(items)), key=lambda k: (round(cov[k], 2), -items[k].datetime.timestamp()))
        it = items[best]
        print(f"  {it.id}  coverage={cov[best]:.2f}  orbit={it.properties.get('sat:relative_orbit')}")
        return it

    def read(item):
        bands = []
        for b in ["vv", "vh"]:
            da = rioxarray.open_rasterio(item.assets[b].href, masked=True).squeeze("band", drop=True)
            da = da.rio.clip_box(*bbox, crs="EPSG:4326")
            bands.append(da)
        da = xr.concat(bands, dim="band").assign_coords(band=[1, 2])
        lin = da.where(da > 0)
        db = 10 * np.log10(lin)
        return db.astype("float32").rio.write_nodata(np.nan, encoded=False)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("flood image:")
    fi = pick(a.flood_dates)
    flood = read(fi)
    print("pre-event image (same relative orbit):")
    pi = pick(a.pre_dates, fi.properties.get("sat:relative_orbit"))
    pre = read(pi).rio.reproject_match(flood)
    flood.rio.to_raster(out / "flood_s1_db.tif", compress="deflate")
    pre.rio.to_raster(out / "pre_s1_db.tif", compress="deflate")
    meta = {"bbox": bbox, "flood_item": fi.id, "flood_datetime": str(fi.datetime),
            "pre_item": pi.id, "pre_datetime": str(pi.datetime),
            "product": "sentinel-1-rtc (gamma0, converted to dB)", "shape": list(flood.shape)}
    (out / "scenes.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- predict
def otsu_threshold(v: np.ndarray, bins: int = 256) -> float:
    v = v[np.isfinite(v)]
    v = np.clip(v, -35, 5)
    hist, edges = np.histogram(v, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(mids[np.argmax(between)])


def iou(a_, b_, valid):
    a_, b_ = a_ & valid, b_ & valid
    return float((a_ & b_).sum() / max((a_ | b_).sum(), 1))


def write_tif(path, arr, profile, dtype="float32"):
    import rasterio
    p = profile.copy(); p.update(count=1, dtype=dtype, compress="deflate", nodata=None)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)


def cmd_predict(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    import torch
    from sklearn.metrics import roc_auc_score

    from floodshift.data import normalise, read_s1_db
    from floodshift.model import load_checkpoint
    from floodshift.uncertainty import tiled_predict

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    ckpts = sorted(Path(a.runs).glob("member_*.pt"))
    assert ckpts, f"no member_*.pt in {a.runs}"
    loaded = [load_checkpoint(c, device) for c in ckpts]
    models, stats = [m for m, _, _ in loaded], loaded[0][1]
    norm_mode = a.norm_mode or loaded[0][2].get("norm_mode", "global")

    s1f_metrics = None
    if a.sen1floods11_metrics and Path(a.sen1floods11_metrics).exists():
        s1f_metrics = json.loads(Path(a.sen1floods11_metrics).read_text())
    T = a.temperature if a.temperature else (s1f_metrics or {}).get("temperature", 1.0)

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(data / "flood_s1_db.tif") as src:
        profile, transform, crs = src.profile, src.transform, src.crs
    px_area_km2 = abs(transform.a * transform.e) / 1e6 if crs and crs.is_projected else None

    res = {}
    for tag in ["flood", "pre"]:
        f = data / f"{tag}_s1_db.tif"
        if not f.exists():
            continue
        x_db = read_s1_db(f)
        x, valid = normalise(x_db, stats, norm_mode)
        print(f"{tag}: {x.shape[1]}x{x.shape[2]} px, running {len(models)}-member ensemble (T={T:.2f}) ...")
        o = tiled_predict(models, x, tile=a.tile, overlap=a.overlap, temperature=T, device=device)
        o["valid"], o["x_db"] = valid, x_db
        thr = otsu_threshold(x_db[0][valid])
        o["otsu_thr"], o["otsu_water"] = thr, (x_db[0] < thr) & valid
        res[tag] = o

    F = res["flood"]
    valid = F["valid"]
    water = (F["prob"] >= 0.5) & valid
    uncertain = (F["entropy"] >= a.uncertain_thr) & valid
    disagree = water ^ F["otsu_water"]
    m = {"temperature": T, "norm_mode": norm_mode, "n_valid_pixels": int(valid.sum()),
         "otsu_threshold_vv_db": F["otsu_thr"],
         "water_fraction_model": float(water[valid].mean()),
         "water_fraction_otsu": float(F["otsu_water"][valid].mean()),
         "iou_model_vs_otsu": iou(water, F["otsu_water"], valid),
         "mean_entropy": float(F["entropy"][valid].mean()),
         "uncertain_fraction": float(uncertain[valid].mean()),
         "mean_entropy_where_disagree_with_otsu": float(F["entropy"][disagree & valid].mean()) if (disagree & valid).any() else None,
         "mean_entropy_where_agree_with_otsu": float(F["entropy"][~disagree & valid].mean()),
         }
    sel = np.flatnonzero(valid.ravel())
    if sel.size > 2_000_000:
        sel = np.random.default_rng(0).choice(sel, 2_000_000, replace=False)
    d = disagree.ravel()[sel]
    if 0 < d.sum() < d.size:
        m["auroc_entropy_flags_disagreement"] = float(roc_auc_score(d, F["entropy"].ravel()[sel]))

    if "pre" in res:
        P = res["pre"]
        both = valid & P["valid"]
        new_model = water & ~((P["prob"] >= 0.5)) & both
        new_otsu = F["otsu_water"] & ~P["otsu_water"] & both
        m["iou_newflood_model_vs_otsu_change"] = iou(new_model, new_otsu, both)
        m["newly_flooded_fraction_model"] = float(new_model[both].mean())
        m["mean_entropy_pre_event"] = float(P["entropy"][P["valid"]].mean())
        if px_area_km2:
            m["newly_flooded_km2_model"] = float(new_model.sum() * px_area_km2)
            m["newly_flooded_km2_otsu"] = float(new_otsu.sum() * px_area_km2)
        write_tif(out / "newly_flooded_model.tif", new_model, profile, "uint8")
    else:
        new_model = None

    if a.reference:
        import geopandas as gpd
        from rasterio.features import rasterize
        ref = gpd.read_file(a.reference).to_crs(crs)
        ref_mask = rasterize(((g, 1) for g in ref.geometry if g is not None), out_shape=water.shape,
                             transform=transform, fill=0, dtype="uint8").astype(bool)
        m["reference_file"] = str(a.reference)
        m["iou_model_water_vs_reference"] = iou(water, ref_mask, valid)
        m["iou_otsu_vs_reference"] = iou(F["otsu_water"], ref_mask, valid)
        if new_model is not None:
            m["iou_model_newflood_vs_reference"] = iou(new_model, ref_mask, valid)
        err = (water ^ ref_mask).ravel()[sel]
        if 0 < err.sum() < err.size:
            m["auroc_entropy_flags_error_vs_reference"] = float(roc_auc_score(err, F["entropy"].ravel()[sel]))

    if s1f_metrics:
        m["comparison_mean_entropy"] = {
            "sen1floods11_test": s1f_metrics["test"]["ensemble_ts"]["mean_uncertainty"],
            "bolivia": s1f_metrics["bolivia"]["ensemble_ts"]["mean_uncertainty"],
            "pakistan_2022": m["mean_entropy"]}
        m["comparison_iou"] = {"sen1floods11_test": s1f_metrics["test"]["ensemble_ts"]["iou"],
                               "bolivia": s1f_metrics["bolivia"]["ensemble_ts"]["iou"]}

    write_tif(out / "p_water.tif", F["prob"], profile)
    write_tif(out / "entropy.tif", F["entropy"], profile)
    write_tif(out / "water_model.tif", water, profile, "uint8")
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))

    # ---------------- figures (downsampled for display) ----------------
    step = max(1, max(water.shape) // 1500)
    ds = lambda z: z[::step, ::step]  # noqa: E731
    panels = []
    if "pre" in res:
        panels.append((np.clip(res["pre"]["x_db"][0], -30, 0), "VV before (dB)", {"cmap": "gray"}))
    panels += [(np.clip(F["x_db"][0], -30, 0), "VV during flood (dB)", {"cmap": "gray"}),
               (F["prob"], "Ensemble P(water)", {"cmap": "Blues", "vmin": 0, "vmax": 1}),
               (F["entropy"], "Predictive entropy", {"cmap": "magma", "vmin": 0, "vmax": 1})]
    if new_model is not None:
        panels.append((new_model.astype(float), "Newly flooded (model)", {"cmap": "Reds", "vmin": 0, "vmax": 1}))
    panels.append((disagree.astype(float), "Disagrees with Otsu", {"cmap": "Oranges", "vmin": 0, "vmax": 1}))
    n = len(panels)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    for k, ax in enumerate(axs.ravel()):
        ax.axis("off")
        if k < n:
            img, t, kw = panels[k]
            ax.imshow(ds(img), **kw); ax.set_title(t, fontsize=9)
    fig.suptitle("Sindh, Pakistan — 2022 floods (Sentinel-1)", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "pakistan_maps.png", dpi=150); plt.close(fig)

    if s1f_metrics:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        names = ["Test\n(seen)", "Bolivia\n(unseen)", "Sindh 2022\n(new event + new product)"]
        H = list(m["comparison_mean_entropy"].values())
        axs[0].bar(names, H, color=["C0", "C3", "C2"]); axs[0].set_ylabel("Mean predictive entropy")
        axs[0].set_title("Does the model know it is somewhere new?")
        ious = [m["comparison_iou"]["sen1floods11_test"], m["comparison_iou"]["bolivia"],
                m.get("iou_model_water_vs_reference", m["iou_model_vs_otsu"])]
        axs[1].bar(names, ious, color=["C0", "C3", "C2"])
        axs[1].set_ylabel("IoU" + ("" if a.reference else " (Pakistan: vs Otsu, not ground truth)"))
        axs[1].set_title("How well does it map water?")
        fig.tight_layout(); fig.savefig(out / "shift_summary.png", dpi=150); plt.close(fig)
    print(f"wrote {out}/")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", default="data/pakistan")
    f.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
    f.add_argument("--flood-dates", default=FLOOD_DATES)
    f.add_argument("--pre-dates", default=PRE_DATES)
    p = sub.add_parser("predict")
    p.add_argument("--runs", default="runs/ensemble")
    p.add_argument("--data", default="data/pakistan")
    p.add_argument("--out", default="results/pakistan")
    p.add_argument("--reference", default=None, help="optional flood-extent vector (shp/geojson/gpkg)")
    p.add_argument("--sen1floods11-metrics", default="results/metrics.json")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = take it from metrics.json")
    p.add_argument("--norm-mode", default=None, choices=[None, "global", "per_image"])
    p.add_argument("--uncertain-thr", type=float, default=0.5)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--device", default="auto")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
