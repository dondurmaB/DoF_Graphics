"""Smoke test of the training path on the cluster (data -> loader -> model -> loss -> checkpoint).

    python ai/smoke_train.py bench   ROOT                     # loader samples/s for several worker counts
    python ai/smoke_train.py overfit ROOT --n 16 --steps 1500  # must drive the loss towards zero
    python ai/smoke_train.py train   ROOT --steps 2000         # must beat the identity baseline on held-out poses

Held-out poses: pose_id hash % 50 == 0 (about 2%), never seen in training. `--exclude FILE` drops lens dirs
listed one per line (samples the protocol check marks BAD). Metrics are in model space (log(1 + L e)).
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D      # noqa: E402
from model import UNet  # noqa: E402


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in ("x", "y", "settings")}


def held_out(sample) -> bool:
    return int(hashlib.md5(sample.pose_dir.name.encode()).hexdigest(), 16) % 50 == 0


def datasets(args):
    full = D.DofDataset(args.root, split="train", crop=args.crop, seed=args.seed)
    drop = set(Path(args.exclude).read_text().split()) if args.exclude else set()
    keep = [i for i, s in enumerate(full.samples) if str(s.lens_dir) not in drop]
    tr = [i for i in keep if not held_out(full.samples[i])]
    va = [i for i in keep if held_out(full.samples[i])]
    print(f"samples {len(full.samples)}, excluded {len(full.samples) - len(keep)}, train {len(tr)}, held-out {len(va)}",
          flush=True)
    return full, tr, va


def loader(ds, idx, args, shuffle):
    return DataLoader(Subset(ds, idx), batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                      collate_fn=collate, pin_memory=True, drop_last=shuffle, persistent_workers=args.workers > 0)


def psnr(a, b):
    return (10 * torch.log10(1.0 / torch.mean((a - b) ** 2).clamp_min(1e-10))).item()


@torch.no_grad()
def evaluate(model, dl, dev, max_batches=40):
    model.eval()
    l1, l1_id, ps, ps_id, n = 0.0, 0.0, 0.0, 0.0, 0
    for k, b in enumerate(dl):
        if k >= max_batches:
            break
        x, y = b["x"].to(dev), b["y"].to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            yh = model(x).float()
        l1 += (yh - y).abs().mean().item()
        l1_id += (x[:, :3] - y).abs().mean().item()
        ps += psnr(yh, y)
        ps_id += psnr(x[:, :3], y)
        n += 1
    model.train()
    return {"l1": l1 / n, "l1_identity": l1_id / n, "psnr": ps / n, "psnr_identity": ps_id / n}


def bench(args):
    full, tr, _ = datasets(args)
    for w in args.bench_workers:
        args.workers = w
        dl = loader(full, tr, args, True)
        it, t0, seen = iter(dl), None, 0
        for k in range(args.bench_batches + 2):
            next(it)
            if k == 1:
                t0 = time.time()           # skip worker start-up
            elif k > 1:
                seen += args.batch
        dt = time.time() - t0
        print(f"workers {w:3d}: {seen / dt:6.1f} samples/s  ({dt / args.bench_batches:.2f} s per batch of {args.batch})",
              flush=True)
        del dl, it


def train(args, overfit=False):
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    full, tr, va = datasets(args)
    if overfit:
        # a fixed set of crops, loaded once, no augmentation: the model should memorise them
        rng = np.random.default_rng(0)
        pick = [tr[i] for i in rng.choice(len(tr), args.n, replace=False)]
        exs = [D.load_example(full.samples[i], args.crop, np.random.default_rng(i), augment=False) for i in pick]
        X = torch.from_numpy(np.stack([e["x"] for e in exs])).to(dev)
        Y = torch.from_numpy(np.stack([e["y"] for e in exs])).to(dev)
        batches = lambda: iter([{"x": X, "y": Y}])    # noqa: E731
    else:
        dl_tr, dl_va = loader(full, tr, args, True), loader(full, va, args, False)

        def batches():
            while True:
                yield from dl_tr

    model = UNet(width=args.width).to(dev).to(memory_format=torch.channels_last)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")
    step, t0, it = 0, time.time(), batches()
    while step < args.steps:
        try:
            b = next(it)
        except StopIteration:
            it = batches()
            continue
        x = b["x"].to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        y = b["y"].to(dev, non_blocking=True)
        with torch.autocast("cuda", torch.bfloat16):
            loss = (model(x).float() - y).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            rec = {"step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                   "s_per_step": (time.time() - t0) / step}
            if overfit:
                with torch.no_grad():
                    rec["l1_identity"] = (X[:, :3] - Y).abs().mean().item()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
        if not overfit and (step % args.eval_every == 0 or step == args.steps):
            ev = {"step": step, **evaluate(model, dl_va, dev)}
            print("EVAL " + json.dumps(ev), flush=True)
            log.write(json.dumps({"eval": ev}) + "\n")
            log.flush()
    torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, out / "last.pt")
    print(f"done {step} steps in {(time.time() - t0) / 60:.1f} min -> {out / 'last.pt'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "overfit", "train"])
    ap.add_argument("root")
    ap.add_argument("--out", default="runs/smoke")
    ap.add_argument("--exclude")
    ap.add_argument("--crop", type=int, default=D.CROP)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--n", type=int, default=16, help="overfit: number of fixed crops")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench-workers", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--bench-batches", type=int, default=20)
    args = ap.parse_args()
    if args.mode == "bench":
        bench(args)
    else:
        train(args, overfit=args.mode == "overfit")


if __name__ == "__main__":
    main()
