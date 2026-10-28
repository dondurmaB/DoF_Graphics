"""Smoke test of the training path on the cluster (data -> loader -> model -> loss -> checkpoint).

    python ai/smoke_train.py bench   ROOT                     # loader samples/s for several worker counts
    python ai/smoke_train.py overfit ROOT --n 16 --steps 1500  # must drive the loss towards zero
    python ai/smoke_train.py train   ROOT --steps 2000         # must beat the identity baseline on held-out poses
    python ai/smoke_train.py train   ROOT --alpha 1 --beta 5   # CoC-weighted loss (ai/LOSS_PLAN.md)
    python ai/smoke_train.py train   ROOT --arch dilated --film --gamma 20 --crops-per-image 4   # sweep variant

Held-out poses: pose_id hash % 50 == 0 (about 2%), never seen in training, evaluated on fixed crops so every
run sees the same pixels. `--exclude FILE` drops lens dirs listed one per line (samples the protocol check
marks BAD). Losses and metrics are in model space (log(1 + L e)); the CoC map is a loss weight and a metric
region only, never an input (DATA_PROTOCOL.md v1.1). Regions are occlusion-aware (LOSS_PLAN.md section 8).
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
from model import ARCHS, build  # noqa: E402

MARGIN = 48                                   # CoC context around a crop: blur discs up to 96 px reach in
RADII = (1, 2, 4, 8, 16, 24, 32, 48)          # dilation steps (disc radius in px)
BOUNDARY_D = 4.0                              # an in-focus pixel under a neighbour's disc of >= 4 px
BINS = ((1, 4), (4, 16), (16, 64), (64, float("inf")))
HARD_COC, HARD_MIN_PX = 32.0, 20000           # hard crops: windows full of blur >= 32 px (>= 64 px is ~1% of samples)


def collate(batch):
    """Stack every tensor field; items holding K crops each ([K, ...], crops_per_image > 1) are flattened."""
    keys = [k for k, v in batch[0].items() if torch.is_tensor(v)]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    if isinstance(batch[0]["meta"], list):
        out = {k: v.flatten(0, 1) for k, v in out.items()}
    return out


def one_thread(_):
    torch.set_num_threads(1)                  # 30 workers x N threads each oversubscribed the CPUs


def held_out(sample) -> bool:
    return int(hashlib.md5(sample.pose_dir.name.encode()).hexdigest(), 16) % 50 == 0


def dilate(c):
    """c [B, 1, H+2M, W+2M] >= 0 -> the largest c(q) whose disc (radius c(q) / 2) covers each pixel, cropped
    to the window (square max-pools, LOSS_PLAN.md section 4)."""
    d = torch.zeros_like(c)
    for r in RADII:
        d = torch.maximum(d, F.max_pool2d(torch.where(c >= 2 * r, c, 0.0), 2 * r + 1, 1, r))
    return d[..., MARGIN:-MARGIN, MARGIN:-MARGIN]


def regions(coc):
    """Occlusion-aware regions from the signed CoC (negative = in front of the focus plane), [B, 1, H, W].

    A blurred disc in front of the focus plane spreads over the in-focus pixels behind it (fg_spill: the
    target changes there). A blurred disc behind the focus plane cannot spread over an in-focus object in
    front of it (bg_edge: the target stays sharp). LOSS_PLAN.md section 8."""
    c = coc.unsqueeze(1)
    own = c.abs()[..., MARGIN:-MARGIN, MARGIN:-MARGIN]
    d_fg = dilate((-c).clamp(min=0))                    # spill from nearer, in-front discs only
    d_bg = dilate(c.clamp(min=0))
    sharp = own < 1
    fg_spill = sharp & (d_fg >= BOUNDARY_D)
    bg_edge = sharp & (d_bg >= BOUNDARY_D) & ~fg_spill
    return {"own": own, "blur": torch.maximum(own, d_fg), "in_focus": sharp & ~fg_spill & ~bg_edge,
            "fg_spill": fg_spill, "bg_edge": bg_edge, "defocused": ~sharp}


def weights(coc, args):
    """1 + alpha * blur reaching the pixel (own or spilled from in front) + boosts for both kinds of edge."""
    g = regions(coc)
    return (1 + args.alpha * (g["blur"] / args.c0).clamp(max=args.cap)
            + args.beta * g["fg_spill"].float() + args.gamma * g["bg_edge"].float())


def datasets(args):
    full = D.DofDataset(args.root, split="train", crop=args.crop, seed=args.seed, coc_margin=MARGIN,
                        crops_per_image=getattr(args, "crops_per_image", 1))
    evalset = D.DofDataset(args.root, split="train", crop=args.crop, seed=12345, coc_margin=MARGIN,
                           augment=False, deterministic=True)
    hardset = D.DofDataset(args.root, split="train", crop=args.crop, seed=12345, coc_margin=MARGIN,
                           augment=False, deterministic=True, hard_coc=HARD_COC)
    drop = set(Path(args.exclude).read_text().split()) if args.exclude else set()
    keep = [i for i, s in enumerate(full.samples) if str(s.lens_dir) not in drop]
    tr = [i for i in keep if not held_out(full.samples[i])]
    va = [i for i in keep if held_out(full.samples[i])]
    print(f"samples {len(full.samples)}, excluded {len(full.samples) - len(keep)}, train {len(tr)}, held-out {len(va)}",
          flush=True)
    hard = [i for i in va if D.hard_crop_at(np.load(full.samples[i].lens_dir / "coc.npy"), args.crop, HARD_COC)[2]
            >= HARD_MIN_PX]
    print(f"hard held-out crops (>= {HARD_MIN_PX} px with CoC >= {HARD_COC:.0f}): {len(hard)} of {len(va)}", flush=True)
    return full, evalset, hardset, tr, va, hard


def loader(ds, idx, args, shuffle, batch=None):
    return DataLoader(Subset(ds, idx), batch_size=batch or args.batch, shuffle=shuffle, num_workers=args.workers,
                      collate_fn=collate, pin_memory=True, drop_last=shuffle, persistent_workers=args.workers > 0,
                      worker_init_fn=one_thread)


@torch.no_grad()
def evaluate(model, dl, dev, pred=None):
    """L1 of model and identity per region over all held-out crops (pixel-weighted sums, not means of means).
    pred(batch) -> y_hat replaces the model (non-learned baselines, ai/classic_eval.py)."""
    if model is not None:
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
        if pred is not None:
            yh = pred(b).to(dev)
        else:
            with torch.autocast("cuda", torch.bfloat16):
                yh = model(x, b["settings"].to(dev)).float()
        e = (yh - y).abs().mean(1, keepdim=True)
        e_id = (x[:, :3] - y).abs().mean(1, keepdim=True)
        g = regions(b["coc"].to(dev))
        own = g["own"]
        add("all", torch.ones_like(own, dtype=torch.bool), e, e_id)
        for k in ("in_focus", "fg_spill", "bg_edge", "defocused"):
            add(k, g[k], e, e_id)
        for lo, hi in BINS:
            add(f"coc_{lo}_{hi if hi < float('inf') else 'up'}", (own >= lo) & (own < hi), e, e_id)
        add("coc_32_up", own >= 32, e, e_id)
    if model is not None:
        model.train()
    total = acc["all"][2]
    return {k: {"l1": s / n, "identity": si / n, "ratio": s / si, "frac": n / total}
            for k, (s, si, n) in acc.items() if n > 0}


def bench(args):
    full, _, _, tr, _, _ = datasets(args)
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
    full, evalset, hardset, tr, va, hard = datasets(args)
    if overfit:
        # a fixed set of crops, loaded once, no augmentation: the model should memorise them
        rng = np.random.default_rng(0)
        pick = [tr[i] for i in rng.choice(len(tr), args.n, replace=False)]
        exs = [D.load_example(full.samples[i], args.crop, np.random.default_rng(i), augment=False,
                              coc_margin=MARGIN) for i in pick]
        fixed = {k: torch.from_numpy(np.stack([e[k] for e in exs])).to(dev) for k in ("x", "y", "settings", "coc")}
        batches = lambda: iter([fixed])    # noqa: E731
    else:
        k = args.crops_per_image
        dl_tr = loader(full, tr, args, True, batch=max(1, args.batch // k))
        dl_va = loader(evalset, va, args, False)
        dl_hard = loader(hardset, hard, args, False) if hard else None
        mix = np.random.default_rng(args.seed + 1)

        def batches():
            if k == 1:
                while True:
                    yield from dl_tr
            # K crops per decoded frame: pool them and draw each batch at random from the pool, so a batch
            # mixes about args.mix / K frames instead of holding K crops of the same view
            pool = []
            while True:
                for b in dl_tr:
                    pool += [{f: v[j].clone() for f, v in b.items()} for j in range(len(b["x"]))]
                    while len(pool) >= args.mix:
                        picked = [pool.pop(j) for j in sorted(mix.choice(len(pool), args.batch, replace=False),
                                                              reverse=True)]
                        yield {f: torch.stack([p[f] for p in picked]) for f in picked[0]}

    model = build(args.arch, args.width, args.film).to(dev).to(memory_format=torch.channels_last)
    print(f"arch {args.arch} | params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M | film {args.film} | "
          f"crops/frame {args.crops_per_image} | loss alpha "
          f"{args.alpha} beta {args.beta} gamma {args.gamma} c0 {args.c0} cap {args.cap}", flush=True)
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
            yh = model(x, b["settings"].to(dev, non_blocking=True)).float()
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
                   "s_per_step": (time.time() - t0) / step,
                   "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
        if not overfit and (step % args.eval_every == 0 or step == args.steps):
            ev = evaluate(model, dl_va, dev)
            if dl_hard is not None:       # separate fixed crops full of blur >= 32 px; keys prefixed "hard_"
                ev.update({"hard_" + k: v for k, v in evaluate(model, dl_hard, dev).items()})
            print(f"EVAL step {step}", flush=True)
            for k, v in ev.items():
                print(f"  {k:18s} frac {v['frac']:.4f}  l1 {v['l1']:.5f}  identity {v['identity']:.5f}  "
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
    ap.add_argument("--arch", choices=ARCHS, default="unet")
    ap.add_argument("--width", type=int, default=None, help="base channels; default per arch (model.build)")
    ap.add_argument("--crops-per-image", type=int, default=1, help="crops cut from each decoded frame")
    ap.add_argument("--mix", type=int, default=64, help="pool of crops each batch is drawn from (crops-per-image > 1)")
    ap.add_argument("--alpha", type=float, default=0.0, help="extra weight for large dilated CoC")
    ap.add_argument("--beta", type=float, default=0.0, help="extra weight for in-focus pixels under a nearer blur")
    ap.add_argument("--gamma", type=float, default=0.0, help="extra weight for in-focus edges over a blurred background")
    ap.add_argument("--film", action="store_true", help="condition every U-Net level on the camera settings")
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
