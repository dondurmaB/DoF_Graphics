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
ENERGY_K = (4, 16)                            # box sizes (px) of the light-energy loss (experiment_plan.md T2)
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


def energy_map(t, k):
    """Local light energy: linear radiance box-averaged over k x k px, back in model space. Blur spreads light
    linearly, so this is what a lens image conserves and what log-space L1 misjudges (experiment_plan.md E0)."""
    return torch.log1p(F.avg_pool2d(torch.expm1(t.float()).clamp(min=0), k, ceil_mode=True))


def aux_losses(model, coc, y, args):
    """Per-arm losses for twoarm2 / pyramid (experiment_plan.md M1), label-side only. Arm B (or each pyramid
    level) on pixels whose blur it should handle; arm A on the rest; the blur share m towards a soft step at aux_t
    (slope aux_m_s px); and m pushed to 0 on sharp pixels that nothing in front blurs over (in_focus and bg_edge,
    bg_edge weighted 1 + gamma like the main loss), where any blur arm leaks a halo."""
    aux = getattr(model, "aux", None)
    if aux is None or not (args.aux_b or args.aux_a or args.aux_m or args.aux_m_halo):
        return 0.0
    g = regions(coc)
    blur = g["blur"]
    big = (blur >= args.aux_t).float()
    out = 0.0
    if args.aux_b and "B_levels" in aux:      # each level where blur is at least 4x its pixel size
        for b, sc in zip(aux["B_levels"], aux["scales"]):
            msk = (blur >= 4 * sc).float()
            out = out + args.aux_b / len(aux["scales"]) * ((b - y).abs().mean(1, keepdim=True) * msk).sum() / (msk.sum() + 1)
    elif args.aux_b:
        out = out + args.aux_b * ((aux["B"] - y).abs().mean(1, keepdim=True) * big).sum() / (big.sum() + 1)
    if args.aux_a:
        out = out + args.aux_a * ((aux["A"] - y).abs().mean(1, keepdim=True) * (1 - big)).sum() / ((1 - big).sum() + 1)
    if args.aux_m:
        tm = torch.sigmoid((blur - args.aux_t) / args.aux_m_s)
        out = out + args.aux_m * F.binary_cross_entropy(aux["m"].clamp(1e-4, 1 - 1e-4), tm)
    if args.aux_m_halo:
        wgt = (g["in_focus"] | g["bg_edge"]).float() * (1 + args.gamma * g["bg_edge"].float())
        out = out + args.aux_m_halo * (aux["m"] * wgt).sum() / (wgt.sum() + 1)
    return out


def datasets(args):
    top = not getattr(args, "no_topup", False)
    full = D.DofDataset(args.root, split="train", crop=args.crop, seed=args.seed, coc_margin=MARGIN,
                        crops_per_image=getattr(args, "crops_per_image", 1),
                        big_blur_prob=getattr(args, "big_blur_prob", 0.0), topups=top)
    evalset = D.DofDataset(args.root, split="train", crop=args.crop, seed=12345, coc_margin=MARGIN,
                           augment=False, deterministic=True, topups=top)
    hardset = D.DofDataset(args.root, split="train", crop=args.crop, seed=12345, coc_margin=MARGIN,
                           augment=False, deterministic=True, hard_coc=HARD_COC, topups=top)
    drop = set(Path(args.exclude).read_text().split()) if args.exclude else set()
    keep = [i for i, s in enumerate(full.samples) if str(s.lens_dir) not in drop]
    tr = [i for i in keep if not held_out(full.samples[i])]
    va = [i for i in keep if held_out(full.samples[i]) and not full.samples[i].entry.get("topup")]
    # held-out top-up lenses (experiment_plan.md D2) are scored separately ("top_" keys), so the regular and hard
    # sets stay exactly the ones every earlier run was scored on
    args.va_topup = [i for i in keep if held_out(full.samples[i]) and full.samples[i].entry.get("topup")]
    print(f"samples {len(full.samples)}, excluded {len(full.samples) - len(keep)}, train {len(tr)}, held-out {len(va)}"
          f" + {len(args.va_topup)} top-up", flush=True)
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
        # bokeh: the brightest 1% of target pixels in each crop that are blurred by >= 16 px
        lum = y.mean(1, keepdim=True)
        q = torch.quantile(lum.flatten(1), 0.99, dim=1)[:, None, None, None]
        add("highlight_blur16", (lum >= q) & (own >= 16), e, e_id)
        # light energy at 8 px (linear box average): too dark / too bright over an area, not fine detail
        ey = energy_map(y, 8)
        en = F.interpolate((energy_map(yh, 8) - ey).abs().mean(1, keepdim=True), size=own.shape[-2:], mode="nearest")
        en_id = F.interpolate((energy_map(x[:, :3], 8) - ey).abs().mean(1, keepdim=True), size=own.shape[-2:],
                              mode="nearest")
        add("energy_all", torch.ones_like(own, dtype=torch.bool), en, en_id)
        add("energy_coc_32_up", own >= 32, en, en_id)
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
        dl_top = loader(evalset, args.va_topup, args, False) if args.va_topup else None
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
          f"crops/frame {args.crops_per_image} | big-blur crops {args.big_blur_prob} | energy {args.energy} | aux "
          f"b {args.aux_b} a {args.aux_a} m {args.aux_m} t {args.aux_t} s {args.aux_m_s} halo {args.aux_m_halo} | loss alpha "
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
        coc = b["coc"].to(dev, non_blocking=True)
        w = weights(coc, args)
        with torch.autocast("cuda", torch.bfloat16):
            yh = model(x, b["settings"].to(dev, non_blocking=True)).float()
        err = (yh - y).abs().mean(1, keepdim=True)
        loss = (w * err).sum() / w.sum()
        if args.energy:
            loss = loss + args.energy * sum((energy_map(yh, k) - energy_map(y, k)).abs().mean()
                                            for k in ENERGY_K) / len(ENERGY_K)
        loss = loss + aux_losses(model, coc, y, args)
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
            if dl_top is not None:        # held-out top-up lenses (wide open / stopped down), random fixed crops
                ev.update({"top_" + k: v for k, v in evaluate(model, dl_top, dev).items()})
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
    ap.add_argument("--big-blur-prob", type=float, default=0.0,
                    help="share of training crops centred on large blur (data.load_examples)")
    ap.add_argument("--no-topup", action="store_true", help="leave out top-up lenses (runs before 2026-10-29)")
    ap.add_argument("--mix", type=int, default=64, help="pool of crops each batch is drawn from (crops-per-image > 1)")
    ap.add_argument("--alpha", type=float, default=0.0, help="extra weight for large dilated CoC")
    ap.add_argument("--beta", type=float, default=0.0, help="extra weight for in-focus pixels under a nearer blur")
    ap.add_argument("--gamma", type=float, default=0.0, help="extra weight for in-focus edges over a blurred background")
    ap.add_argument("--energy", type=float, default=0.0, help="weight of the light-energy loss (ENERGY_K boxes)")
    ap.add_argument("--aux-b", type=float, default=0.0, help="twoarm2: loss on the large-blur arm where blur >= aux-t")
    ap.add_argument("--aux-a", type=float, default=0.0, help="twoarm2: loss on the U-Net arm where blur < aux-t")
    ap.add_argument("--aux-m", type=float, default=0.0, help="twoarm2: mask towards a soft step at aux-t")
    ap.add_argument("--aux-t", type=float, default=24.0, help="twoarm2: blur (px) that splits the two arms")
    ap.add_argument("--aux-m-s", type=float, default=4.0, help="slope (px) of the mask target's soft step")
    ap.add_argument("--aux-m-halo", type=float, default=0.0,
                    help="weight of the penalty on blur-arm share at sharp pixels (bg_edge weighted 1 + gamma)")
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
