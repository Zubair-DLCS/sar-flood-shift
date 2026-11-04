"""Download the hand-labelled Sen1Floods11 subset (~1 GB) from its public GCS bucket.

    python scripts/download_sen1floods11.py --out data/sen1floods11

Layout produced:
    data/sen1floods11/splits/flood_{train,valid,test,bolivia}_data.csv
    data/sen1floods11/S1Hand/*.tif
    data/sen1floods11/LabelHand/*.tif

If HTTP is blocked where you run this, the same files can be pulled with
    gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/{S1Hand,LabelHand} data/sen1floods11/
    gsutil -m cp gs://sen1floods11/v1.1/splits/flood_handlabeled/*.csv data/sen1floods11/splits/
"""
import argparse
import csv
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLITS = ["flood_train_data.csv", "flood_valid_data.csv", "flood_test_data.csv", "flood_bolivia_data.csv"]


def fetch(url: str, dst: Path, retries: int = 3):
    if dst.exists() and dst.stat().st_size > 0:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                f.write(r.read())
            tmp.rename(dst)
            return "ok"
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"failed {url}: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sen1floods11")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)

    for s in SPLITS:
        fetch(f"{BASE}/splits/flood_handlabeled/{s}", out / "splits" / s)

    jobs = []
    for s in SPLITS:
        with open(out / "splits" / s) as f:
            for row in csv.reader(f):
                if not row or not row[0].strip():
                    continue
                s1, lab = row[0].strip(), row[1].strip()
                jobs.append((f"{BASE}/data/flood_events/HandLabeled/S1Hand/{s1}", out / "S1Hand" / s1))
                jobs.append((f"{BASE}/data/flood_events/HandLabeled/LabelHand/{lab}", out / "LabelHand" / lab))

    print(f"{len(jobs) // 2} chips to fetch -> {out}")
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for _ in ex.map(lambda j: fetch(*j), jobs):
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} files", flush=True)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
