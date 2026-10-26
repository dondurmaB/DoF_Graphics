"""Smoke test of the training path on the cluster (data -> loader -> model -> loss -> checkpoint).

    python ai/smoke_train.py bench   ROOT                     # loader samples/s for several worker counts
    python ai/smoke_train.py overfit ROOT --n 16 --steps 1500  # must drive the loss towards zero
    python ai/smoke_train.py train   ROOT --steps 2000         # must beat the identity baseline on held-out poses
    python ai/smoke_train.py train   ROOT --alpha 1 --beta 5   # CoC-weighted loss (ai/LOSS_PLAN.md)

Held-out poses: pose_id hash % 50 == 0 (about 2%), never seen in training, evaluated on fixed crops so every
run sees the same pixels. `--exclude FILE` drops lens dirs listed one per line (samples the protocol check
marks BAD). Losses and metrics are in model space (log(1 + L e)); the CoC map is a loss weight and a metric
region only, never an input (DATA_PROTOCOL.md v1.1).
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D      # noqa: E402
from model import UNet  # noqa: E402

MARGIN = 48                                   # CoC context around a crop: blur discs up to 96 px reach in
RADII = (1, 2, 4, 8, 16, 24, 32, 48)          # dilation steps (disc radius in px)
BOUNDARY_D = 4.0                              # an in-focus pixel under a neighbour's disc of >= 4 px
BINS = ((1, 4), (4, 16), (16, 64), (64, float("inf")))


def collate(batch):
    keys = ["x", "y", "settings"] + (["coc"] if "coc" in batch[0] else [])
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def one_thread(_):
    torch.set_num_threads(1)                  # 30 workers x N threads each oversubscribed the CPUs


def held_out(sample) -> bool:
    return int(hashlib.md5(sample.pose_dir.name.encode()).hexdigest(), 16) % 50 == 0


def regions(coc):
    """coc [B, H+2M, W+2M] signed -> own |CoC| and dilated CoC D, both [B, 1, H, W], and the boundary mask.

    D(p) is the largest |coc(q)| whose disc (radius |coc(q)| / 2) covers p, so the weight follows the blur
    onto the sharp pixels it spills over (square windows, LOSS_PLAN.md section 4)."""
    c = coc.abs().unsqueeze(1)
    d = c.clone()
    for r in RADII:
        d = torch.maximum(d, F.max_pool2d(torch.where(c >= 2 * r, c, 0.0), 2 * r + 1, 1, r))
    own, d = c[..., MARGIN:-MARGIN, MARGIN:-MARGIN], d[..., MARGIN:-MARGIN, MARGIN:-MARGIN]
    return own, d, (own < 1) & (d >= BOUNDARY_D)


def weights(coc, args):
    _, d, boundary = regions(coc)
    return 1 + args.alpha * (d / args.c0).clamp(max=args.cap) + args.beta * boundary.float()


def datasets(args):
    full = D.DofDataset(args.root, split="train", crop=args.crop, seed=args.seed, coc_margin=MARGIN)
    evalset = D.DofDataset(args.root, split="train", crop=args.crop, seed=12345, coc_margin=MARGIN,
                           augment=False, deterministic=True)
    drop = set(Path(args.exclude).read_text().split()) if args.exclude else set()
    keep = [i for i, s in enumerate(full.samples) if str(s.lens_dir) not in drop]
    tr = [i for i in keep if not held_out(full.samples[i])]
    va = [i for i in keep if held_out(full.samples[i])]
    print(f"samples {len(full.samples)}, excluded {len(full.samples) - len(keep)}, train {len(tr)}, held-out {len(va)}",
          flush=True)
    return full, evalset, tr, va


def loader(ds, idx, args, shuffle):
    return DataLoader(Subset(ds, idx), batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                      collate_fn=collate, pin_memory=True, drop_last=shuffle, persistent_workers=args.workers > 0,
                      worker_init_fn=one_thread)


@torch.no_grad()
def evaluate(model, dl, dev):
    """L1 of model and identity per region over all held-out crops (pixel-weighted sums, not means of means)."""
    model.eval()
    acc = {}

    def add(name, mask, e, e_id):
        m = mask.float()
        a = acc.setdefault(name, [0.0, 0.0, 0.0])
        a[0] += (e * m).sum().item()
        a[1] += (e_id * m).sum().item()
        a[2] += m.sum().item()

    for b in dl:
        x, y = b["x"].to(dev), b["y"].to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            yh = model(x).float()
        e = (yh - y).abs().mean(1, keepdim=True)
        e_id = (x[:, :3] - y).abs().mean(1, keepdim=True)
        own, d, boundary = regions(b["coc"].to(dev))
        add("all", torch.ones_like(own, dtype=torch.bool), e, e_id)
        add("in_focus", (own < 1) & ~boundary, e, e_id)
        add("boundary", boundary, e, e_id)
        add("defocused", own >= 1, e, e_id)
        for lo, hi in BINS:
            add(f"coc_{lo}_{hi if hi < float('inf') else 'up'}", (own >= lo) & (own < hi), e, e_id)
    model.train()
    total = acc["all"][2]
    return {k: {"l1": s / n, "identity": si / n, "ratio": s / si, "frac": n / total}
            for k, (s, si, n) in acc.items() if n > 0}


def bench(args):
    full, _, tr, _ = datasets(args)
    for w in args.bench_workers:
        args.workers = w
        dl = loader(full, tr, args, True)
        it, t0, seen = iter(dl), None, 0
        for k in range(2 * w + args.bench_batches):   # time only after the prefetch queue has drained
            next(it)
            if k == 2 * w:
                t0 = time.time()
            elif k > 2 * w:
                seen += args.batch
        dt = time.time() - t0
        print(f"workers {w:3d}: {seen / dt:6.1f} samples/s  ({dt / (args.bench_batches - 1):.2f} s per batch of "
              f"{args.batch})", flush=True)
        del dl, it


def train(args, overfit=False):
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    full, evalset, tr, va = datasets(args)
    if overfit:
        # a fixed set of crops, loaded once, no augmentation: the model should memorise them
        rng = np.random.default_rng(0)
        pick = [tr[i] for i in rng.choice(len(tr), args.n, replace=False)]
        exs = [D.load_example(full.samples[i], args.crop, np.random.default_rng(i), augment=False,
                              coc_margin=MARGIN) for i in pick]
        fixed = {k: torch.from_numpy(np.stack([e[k] for e in exs])).to(dev) for k in ("x", "y", "coc")}
        batches = lambda: iter([fixed])    # noqa: E731
    else:
        dl_tr, dl_va = loader(full, tr, args, True), loader(evalset, va, args, False)

        def batches():
            while True:
                yield from dl_tr

    model = UNet(width=args.width).to(dev).to(memory_format=torch.channels_last)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M | loss alpha {args.alpha} "
          f"beta {args.beta} c0 {args.c0} cap {args.cap}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
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
        w = weights(b["coc"].to(dev, non_blocking=True), args)
        with torch.autocast("cuda", torch.bfloat16):
            yh = model(x).float()
        err = (yh - y).abs().mean(1, keepdim=True)
        loss = (w * err).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            rec = {"step": step, "loss": loss.item(), "l1": err.mean().item(), "lr": sched.get_last_lr()[0],
                   "s_per_step": (time.time() - t0) / step}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
        if not overfit and (step % args.eval_every == 0 or step == args.steps):
            ev = evaluate(model, dl_va, dev)
            print(f"EVAL step {step}", flush=True)
            for k, v in ev.items():
                print(f"  {k:12s} frac {v['frac']:.4f}  l1 {v['l1']:.5f}  identity {v['identity']:.5f}  "
                      f"ratio {v['ratio']:.3f}", flush=True)
            log.write(json.dumps({"eval": {"step": step, **ev}}) + "\n")
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
    ap.add_argument("--alpha", type=float, default=0.0, help="extra weight for large dilated CoC")
    ap.add_argument("--beta", type=float, default=0.0, help="extra weight for boundary pixels")
    ap.add_argument("--c0", type=float, default=8.0, help="CoC (px) at which the alpha term reaches alpha")
    ap.add_argument("--cap", type=float, default=4.0, help="cap on D / c0")
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
