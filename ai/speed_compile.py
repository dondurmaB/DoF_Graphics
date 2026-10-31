"""How fast can each model and the classic gather get on one 1080p frame? Same GPU, several execution modes.

    python ai/speed_compile.py ROOT --exclude runs/bad_samples.txt --out runs/speed_compile.json

Modes (each timed as the median of --reps runs after warm-up, CUDA synchronised, batch 1, 1920 x 1088):
  eager-bf16        what ai/speed.py measures: autocast bf16, eager PyTorch
  compile-fp16      torch.compile (inductor, default mode), autocast fp16
  max-fp16          torch.compile mode="max-autotune" (kernel autotuning + CUDA graphs), autocast fp16
  max-half          as max-fp16, with the weights and input cast to fp16 (no autocast casts)
The classic gather (speed.gather_torch, fp32) runs eager and compiled (default and max-autotune); fp16 is not used for
it because it works on linear HDR radiance.

Trained checkpoints are loaded where they exist, and each mode's output is compared with the same model in fp32
eager: `rel_err` = mean |y_mode - y_fp32| / mean |y_fp32 - x_sharp| (the change the model makes), so a mode that
changes the result by more than a few percent of that would show.
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
import data as D          # noqa: E402
import smoke_train as S   # noqa: E402
from model import build   # noqa: E402
from speed import gather_torch, timeit  # noqa: E402

# (label, arch, width, checkpoint or None)
MODELS = [
    ("U-Net w16", "unet", 16, "runs/smoke-train-349873/last.pt"),
    ("U-Net w24 (E3)", "unet", 24, "runs/smoke-train-352592/last.pt"),
    ("U-Net w32", "unet", 32, "runs/smoke-train-349585/last.pt"),
    ("two-arm w16 (E4a)", "twoarm2", 16, "runs/smoke-train-352593/last.pt"),
    ("two-arm w24 (E5c)", "twoarm2", 24, "runs/smoke-train-354643-0/last.pt"),
    ("pyramid w16 (E5f)", "pyramid", 16, "runs/smoke-train-354644-1/last.pt"),
    ("pyramid w24 (E5e)", "pyramid", 24, "runs/smoke-train-354644-0/last.pt"),
]
MODES = ("eager-bf16", "compile-fp16", "max-fp16", "max-half")


def runner(model, x, mode):
    """A no-argument callable running `model` on `x` in the given mode."""
    if mode == "eager-bf16":
        fn, dtype, xin = model, torch.bfloat16, x
    elif mode == "compile-fp16":
        fn, dtype, xin = torch.compile(model), torch.float16, x
    elif mode == "max-fp16":
        fn, dtype, xin = torch.compile(model, mode="max-autotune"), torch.float16, x
    else:                                       # max-half: weights and input in fp16, no autocast
        model = model.half()
        fn, dtype, xin = torch.compile(model, mode="max-autotune"), None, x.half()

    @torch.no_grad()
    def run():
        if dtype is None:
            return fn(xin, xin[:, 4:, 0, 0]).float()
        with torch.autocast("cuda", dtype):
            return fn(xin, xin[:, 4:, 0, 0]).float()
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--exclude")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--crop", type=int, default=D.CROP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/speed_compile.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    full, _, _, _, va, _ = S.datasets(args)
    s = full.samples[va[0]]
    ex = D.load_example(s, crop=None, augment=False)
    m = ex["meta"]
    lens = (m["focal_mm"] / 1000, m["sensor_mm"] / 1000, m["f_number"], m["focus_m"])
    rgb = torch.from_numpy(D.read_exr(s.pose_dir / "sharp.exr").transpose(2, 0, 1).copy()).to(dev)
    depth = torch.from_numpy(np.load(s.pose_dir / "depth.npy").astype(np.float32)).to(dev)
    h, w = ex["x"].shape[-2:]
    x = F.pad(torch.from_numpy(ex["x"])[None].to(dev), (0, -w % 16, 0, -h % 16), mode="reflect")
    x = x.contiguous(memory_format=torch.channels_last)
    rows = []

    ref = gather_torch(rgb, depth, *lens)
    for mode, fn in (("eager-fp32", gather_torch), ("compile-fp32", torch.compile(gather_torch)),
                     ("max-fp32", torch.compile(gather_torch, mode="max-autotune"))):
        t0 = time.time()
        try:
            out = fn(rgb, depth, *lens)
            err = float((out - ref).abs().mean() / (ref - rgb).abs().mean())
            ms = 1000 * timeit(lambda: fn(rgb, depth, *lens), args.reps)
            rows.append({"name": "classic gather", "mode": mode, "ms": ms, "rel_err": err,
                         "compile_s": round(time.time() - t0, 1)})
        except Exception as e:                  # report and keep going: one mode failing must not hide the rest
            rows.append({"name": "classic gather", "mode": mode, "error": repr(e)[:300]})
        print(json.dumps(rows[-1]), flush=True)

    for label, arch, width, ck in MODELS:
        def fresh():
            mdl = build(arch, width, film=True)
            if ck and Path(ck).exists():
                mdl.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
            return mdl.to(dev).eval().to(memory_format=torch.channels_last)
        with torch.no_grad():
            base = fresh()
            y32 = base(x, x[:, 4:, 0, 0]).float()
            change = (y32 - x[:, :3]).abs().mean()
            del base
        for mode in MODES:
            torch._dynamo.reset()
            t0 = time.time()
            try:
                run = runner(fresh(), x, mode)
                y = run()
                err = float((y - y32).abs().mean() / change)
                ms = 1000 * timeit(run, args.reps)
                rows.append({"name": label, "mode": mode, "ms": ms, "rel_err": err,
                             "compile_s": round(time.time() - t0, 1), "trained": bool(ck and Path(ck).exists())})
            except Exception as e:
                rows.append({"name": label, "mode": mode, "error": repr(e)[:300]})
            print(json.dumps(rows[-1]), flush=True)
            torch.cuda.empty_cache()
        Path(args.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                                              "frame": [h, w], "rows": rows}, indent=1))

    names = list(dict.fromkeys(r["name"] for r in rows))
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    print(f"\n{'ms per 1080p frame':28s}" + "".join(f"{md:>14s}" for md in modes))
    for n in names:
        cells = {r["mode"]: r for r in rows if r["name"] == n}
        print(f"{n:28s}" + "".join(f"{cells[md]['ms']:14.1f}" if md in cells and "ms" in cells[md] else
                                   f"{'-':>14s}" for md in modes))
    print("\nrel_err (change vs fp32 eager, as a share of what the model changes):")
    for r in rows:
        if "rel_err" in r:
            print(f"  {r['name']:28s} {r['mode']:14s} {r['rel_err']:.4f}")


if __name__ == "__main__":
    main()
