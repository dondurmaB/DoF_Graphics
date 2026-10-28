"""Time each model and the classic gather on one full 1080p frame, on the same GPU.

    python ai/speed.py ROOT --exclude runs/bad_samples.txt

Models run in bf16 at 1920 x 1088 (padded to a multiple of 16), batch 1, eager PyTorch, channels-last. The classic
method is `gather_torch`: a PyTorch port of renderer/mitsuba/traditional_dof.py's 64-tap depth-aware gather (the
same taps and weights as the DoFScene shader), run in fp32 on the GPU. It is checked against the numpy original
on a real frame before timing. Times are the median of `--reps` runs after warm-up, with CUDA synchronised.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "renderer" / "mitsuba"))
import data as D                     # noqa: E402
import smoke_train as S              # noqa: E402
import traditional_dof as T          # noqa: E402
from model import build              # noqa: E402

DISK = torch.tensor(T.DISK, dtype=torch.float32)


def gather_torch(rgb, depth, focal_m, sensor_m, f_number, focus_m, max_radius=T.MAX_RADIUS_PX):
    """rgb [3, H, W] linear, depth [H, W] metres (inf = sky), on the GPU -> [3, H, W]. Mirrors T.gather_dof."""
    _, h, w = rgb.shape
    dev = rgb.device

    def ss(e0, e1, x):
        t = ((x - e0) / (e1 - e0)).clamp(0, 1)
        return t * t * (3 - 2 * t)

    depth = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, T.SKY_DEPTH_M))
    aperture = focal_m / f_number
    radius = 0.5 * aperture * focal_m * (depth - focus_m) / (depth * (focus_m - focal_m)) / sensor_m * h
    blur = radius.abs().clamp(max=min(max_radius, 32.0))
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    packed = torch.cat([rgb, depth[None], radius[None]])[None]
    centre_fg = radius < 0
    total, wsum = rgb.clone(), torch.ones_like(depth)
    for (dx, dy), dl in zip(DISK.tolist(), T.DISK_LEN.tolist()):
        sx, sy = xs + dx * blur, ys + dy * blur
        grid = torch.stack([sx / (w - 1) * 2 - 1, sy / (h - 1) * 2 - 1], -1)[None]
        s = F.grid_sample(packed, grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
        s_rgb, s_depth, s_radius = s[:3], s[3], s[4]
        delta = s_depth - depth
        similar = 1 - ss(0.05, 1.4, delta.abs())
        keep = torch.where(centre_fg, 1 - ss(-0.05, 0.25, delta), torch.ones_like(delta))
        big_fg = torch.where(s_radius < -0.5, ss(0.2, 1.2, s_radius.abs()), torch.zeros_like(delta))
        reach = ss(dl * blur - 0.75, dl * blur + 0.75, s_radius.abs())
        wgt = (0.18 + 0.82 * similar) * torch.maximum(keep, big_fg * 0.85) * torch.maximum(reach, similar * 0.65)
        wgt = torch.where(blur >= 0.5, wgt, torch.zeros_like(wgt))
        total += s_rgb * wgt
        wsum += wgt
    return total / wsum


def timeit(fn, reps, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


VARIANTS = [("unet", 8), ("unet", 16), ("unet", 32), ("unet", 64), ("unet_noskip", 32), ("twoarm", 16),
            ("twoarm", 32), ("plain", 64), ("dilated", 64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--exclude")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--crop", type=int, default=D.CROP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/speed.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    full, _, _, _, va, _ = S.datasets(args)
    s = full.samples[va[0]]
    ex = D.load_example(s, crop=None, augment=False)
    m = ex["meta"]
    sharp = D.read_exr(s.pose_dir / "sharp.exr")
    depth = np.load(s.pose_dir / "depth.npy").astype(np.float32)
    lens = (m["focal_mm"] / 1000, m["sensor_mm"] / 1000, m["f_number"], m["focus_m"])
    rgb_t = torch.from_numpy(sharp.transpose(2, 0, 1).copy()).to(dev)
    depth_t = torch.from_numpy(depth).to(dev)

    ref = T.gather_dof(sharp, depth, *lens)
    got = gather_torch(rgb_t, depth_t, *lens).cpu().numpy().transpose(1, 2, 0)
    rel = float(np.abs(got - ref).mean() / np.abs(ref).mean())
    print(f"gather_torch vs numpy original: mean abs diff / mean = {rel:.2e}", flush=True)

    rows = [{"name": "classic gather (fp32)", "params_m": 0.0, "ms": 1000 * timeit(
        lambda: gather_torch(rgb_t, depth_t, *lens), args.reps), "check_rel_diff": rel}]
    h, w = ex["x"].shape[-2:]
    x = F.pad(torch.from_numpy(ex["x"])[None].to(dev), (0, -w % 16, 0, -h % 16), mode="reflect")
    x = x.to(memory_format=torch.channels_last)
    for arch, width in VARIANTS:
        model = build(arch, width, film=True).to(dev).eval().to(memory_format=torch.channels_last)

        @torch.no_grad()
        def run():
            with torch.autocast("cuda", torch.bfloat16):
                return model(x, x[:, 4:, 0, 0])
        torch.cuda.reset_peak_memory_stats()
        ms = 1000 * timeit(run, args.reps)
        rows.append({"name": f"{arch} w{width}", "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
                     "ms": ms, "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
        del model
        torch.cuda.empty_cache()
    print(f"{'method':24s} {'params':>8s} {'ms @1080p':>10s}", flush=True)
    for r in rows:
        print(f"{r['name']:24s} {r['params_m']:7.2f}M {r['ms']:10.1f}", flush=True)
    Path(args.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "frame": [h, w], "rows": rows},
                                         indent=1))


if __name__ == "__main__":
    main()
