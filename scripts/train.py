"""Train a deep ensemble of U-Nets on Sen1Floods11 (hand-labelled).

    python scripts/train.py --data data/sen1floods11 --out runs/ensemble --members 5

Each member differs only in random seed (init of the decoder, crops, order).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodshift.data import Sen1Floods11, NormStats, compute_stats, read_split  # noqa: E402
from floodshift.model import FloodNet, save_checkpoint  # noqa: E402


def masked_loss(logits, y, pos_weight: float):
    m = y != -1
    if not m.any():
        return logits.sum() * 0
    lg, t = logits[m], y[m].float()
    bce = F.binary_cross_entropy_with_logits(lg, t, pos_weight=torch.tensor(pos_weight, device=lg.device))
    p = torch.sigmoid(lg)
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return bce + dice


@torch.no_grad()
def val_iou(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x) > 0
        m = y != -1
        t = y == 1
        tp += (pred & t & m).sum().item(); fp += (pred & ~t & m).sum().item(); fn += (~pred & t & m).sum().item()
    return tp / max(tp + fp + fn, 1)


def train_member(seed, a, train_pairs, val_pairs, stats, device, out: Path):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = Sen1Floods11(train_pairs, stats, train=True, crop=a.crop, norm_mode=a.norm_mode)
    va = Sen1Floods11(val_pairs, stats, train=False, norm_mode=a.norm_mode)
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                    generator=g, persistent_workers=a.workers > 0)
    vl = DataLoader(va, batch_size=4, shuffle=False, num_workers=0)

    model = FloodNet(a.encoder, None if a.weights == "none" else a.weights, a.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * max(len(tl), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, log = -1.0, []
    ckpt = out / f"member_{seed}.pt"
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(x)
            loss = masked_loss(logits.float(), y, a.pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); n += 1
        if ep % a.val_every == 0 or ep == a.epochs:
            iou = val_iou(model, vl, device)
            log.append({"epoch": ep, "loss": tot / max(n, 1), "val_iou": iou})
            flag = ""
            if iou > best:
                best = iou
                save_checkpoint(ckpt, model, stats, {"seed": seed, "epoch": ep, "val_iou": iou,
                                                     "norm_mode": a.norm_mode})
                flag = " *"
            print(f"[seed {seed}] ep {ep:3d} loss {tot / max(n, 1):.4f} val IoU {iou:.4f}"
                  f" ({time.time() - t0:.0f}s){flag}", flush=True)
    return {"seed": seed, "best_val_iou": best, "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sen1floods11")
    ap.add_argument("--out", default="runs/ensemble")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--norm-mode", default="global", choices=["global", "per_image"])
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N chips per split")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train_pairs, val_pairs = read_split(a.data, "train"), read_split(a.data, "valid")
    if a.limit:
        train_pairs, val_pairs = train_pairs[:a.limit], val_pairs[:a.limit]
    print(f"device={device}  train={len(train_pairs)}  valid={len(val_pairs)}")

    sp = out / "stats.json"
    if sp.exists():
        stats = NormStats.from_dict(json.loads(sp.read_text()))
    else:
        stats = compute_stats(train_pairs)
        sp.write_text(json.dumps(stats.to_dict(), indent=2))
    print("norm stats (dB):", stats)

    summary = []
    for s in range(a.first_seed, a.first_seed + a.members):
        summary.append(train_member(s, a, train_pairs, val_pairs, stats, device, out))
        (out / "train_log.json").write_text(json.dumps({"args": vars(a), "members": summary}, indent=2))
    print("best val IoU per member:", [round(m["best_val_iou"], 4) for m in summary])


if __name__ == "__main__":
    main()
