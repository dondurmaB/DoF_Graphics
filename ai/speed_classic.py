"""Time the classic gather three ways on the same 1080p frames and GPU, and check each against the numpy original.

    python ai/speed_classic.py ROOT --exclude runs/bad_samples.txt --out runs/speed_classic.json

  PyTorch eager       ai/speed.py gather_torch: 64 whole-image passes
  PyTorch compiled    the same under torch.compile(mode="max-autotune")
  hand-written kernel ai/classic_triton.py: one fused Triton kernel, every pixel's 64 taps in registers
Accuracy: mean |out - numpy| / mean |numpy - sharp| on several held-out frames (fraction of the blur the method adds).
Times: median of --reps runs after warm-up, CUDA synchronised, batch 1, full 1920 x 1080 frame.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "renderer" / "mitsuba"))
import data as D              # noqa: E402
import smoke_train as S       # noqa: E402
import traditional_dof as T   # noqa: E402
from classic_triton import gather_triton  # noqa: E402
from speed import gather_torch, timeit     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--exclude")
    ap.add_argument("--frames", type=int, default=3, help="held-out frames for the accuracy check")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--crop", type=int, default=D.CROP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/speed_classic.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    full, _, _, _, va, _ = S.datasets(args)
    compiled = torch.compile(gather_torch, mode="max-autotune")
    methods = {"PyTorch eager": gather_torch, "PyTorch compiled": compiled, "hand-written kernel": gather_triton}
    errs = {k: [] for k in methods}
    times = {k: [] for k in methods}
    for i in np.linspace(0, len(va) - 1, args.frames).astype(int):
        s = full.samples[va[i]]
        meta = D.load_example(s, crop=None, augment=False)["meta"]
        lens = (meta["focal_mm"] / 1000, meta["sensor_mm"] / 1000, meta["f_number"], meta["focus_m"])
        sharp = D.read_exr(s.pose_dir / "sharp.exr")
        depth = np.load(s.pose_dir / "depth.npy").astype(np.float32)
        ref = T.gather_dof(sharp, depth, *lens)
        rgb_t = torch.from_numpy(sharp.transpose(2, 0, 1).copy()).to(dev)
        depth_t = torch.from_numpy(depth).to(dev)
        added = np.abs(ref - sharp).mean()
        for name, fn in methods.items():
            t0 = time.time()
            out = fn(rgb_t, depth_t, *lens)
            torch.cuda.synchronize()
            print(f"  {name}: first call {time.time() - t0:.1f} s", flush=True)
            errs[name].append(float(np.abs(out.cpu().numpy().transpose(1, 2, 0) - ref).mean() / added))
            times[name].append(1000 * timeit(lambda: fn(rgb_t, depth_t, *lens), args.reps))
        print(f"frame {s.lens_dir.name}: " + "  ".join(f"{n} {times[n][-1]:.2f} ms err {errs[n][-1]:.1e}"
                                                        for n in methods), flush=True)
    rows = [{"method": n, "ms_median": float(np.median(times[n])), "ms_per_frame": times[n],
             "rel_err_max": max(errs[n])} for n in methods]
    print(f"\n{'classic gather':22s} {'ms / 1080p frame':>17s} {'max rel. error vs numpy':>25s}")
    for r in rows:
        print(f"{r['method']:22s} {r['ms_median']:17.2f} {r['rel_err_max']:25.1e}")
    Path(args.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
