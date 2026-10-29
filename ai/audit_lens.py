"""E1 (experiment_plan.md): what camera settings and blur sizes does the dataset actually cover?

    python ai/audit_lens.py ROOT --out runs/audit

Reads every sample's metadata.json and coc.npy (all splits) and writes:
  samples.csv   one row per sample: scene, env, focal length, f-number, aperture, focus distance and how it was
                chosen, blur at infinity, blur percentiles, share of pixels at or above 4 / 16 / 32 / 64 px
  summary.txt   per-scene and overall tables, plus histograms of f-number, focal length, focus distance and
                max blur, in text
"""

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402

THRESH = (4, 16, 32, 64)


def one(s):
    m = json.loads((s.lens_dir / "metadata.json").read_text())
    cam, draw = m["camera"], m.get("lens_draw", {})
    c = np.abs(np.load(s.lens_dir / "coc.npy"))
    c = c[np.isfinite(c)]
    row = {"scene": s.scene, "env": s.env, "lens_dir": str(s.lens_dir),
           "focal_mm": cam["focal_length_mm"], "f_number": cam["f_number"],
           "aperture_mm": cam["focal_length_mm"] / cam["f_number"], "focus_m": cam["focus_distance_m"],
           "focus_source": draw.get("focus_source", ""), "stratum": draw.get("stratum", -1),
           "coc_at_inf_px": cam.get("coc_diameter_px_at_infinity", float("nan"))}
    for q in (50, 95, 99):
        row[f"coc_p{q}"] = float(np.percentile(c, q))
    row["coc_max"] = float(c.max())
    for t in THRESH:
        row[f"frac_ge{t}"] = float((c >= t).mean())
    return row


def hist(vals, edges, label, unit=""):
    h, _ = np.histogram(vals, edges)
    out = [f"{label}"]
    for lo, hi, n in zip(edges[:-1], edges[1:], h):
        out.append(f"  {lo:8.3g} - {hi:<8.3g}{unit:3s} {n:6d}  {'#' * int(60 * n / max(h.max(), 1))}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="runs/audit")
    ap.add_argument("--workers", type=int, default=16)
    args, _ = ap.parse_known_args()
    samples = D.find_samples(args.root, split=None)
    with Pool(args.workers) as p:
        rows = p.map(one, samples, chunksize=16)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    col = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("scene", "env", "lens_dir", "focus_source")}
    scenes = sorted({r["scene"] for r in rows})
    lines = [f"{len(rows)} samples, {len(scenes)} scenes", ""]
    hdr = (f"{'scene':14s} {'n':>5s} {'f med':>6s} {'N med':>6s} {'N<2.8':>6s} {'N>=8':>6s} {'focus med':>9s} "
           f"{'p95 med':>8s} {'max med':>8s} {'max>=64':>8s} {'px>=32':>8s} {'px>=64':>8s}")
    lines.append(hdr)
    for sc in scenes + ["ALL"]:
        sel = np.array([sc == "ALL" or r["scene"] == sc for r in rows])
        g = {k: v[sel] for k, v in col.items()}
        lines.append(f"{sc:14s} {sel.sum():5d} {np.median(g['focal_mm']):6.1f} {np.median(g['f_number']):6.2f} "
                     f"{(g['f_number'] < 2.8).mean():6.1%} {(g['f_number'] >= 8).mean():6.1%} "
                     f"{np.median(g['focus_m']):9.2f} {np.median(g['coc_p95']):8.1f} {np.median(g['coc_max']):8.1f} "
                     f"{(g['coc_max'] >= 64).mean():8.1%} {g['frac_ge32'].mean():8.2%} {g['frac_ge64'].mean():8.2%}")
    lines += ["", "N<2.8 / N>=8: share of samples; max>=64: share of samples whose largest blur reaches 64 px;",
              "px>=32 / px>=64: share of all pixels with blur at or above 32 / 64 px.", ""]
    src = {}
    for r in rows:
        k = r["focus_source"].split(":")[0]
        src[k] = src.get(k, 0) + 1
    lines.append("focus chosen by: " + ", ".join(f"{k} {v}" for k, v in sorted(src.items())))
    lines.append("")
    lines.append(hist(col["f_number"], [1.2, 1.4, 2, 2.8, 4, 5.6, 8, 11, 16.01], "f-number"))
    lines.append(hist(col["focal_mm"], [20, 24, 35, 50, 70, 85, 105, 135, 140], "focal length", "mm"))
    lines.append(hist(col["focus_m"], [0, 0.5, 1, 2, 3, 5, 10, 20, 1e9], "focus distance", "m"))
    lines.append(hist(col["coc_p95"], [0, 4, 8, 16, 32, 43, 64, 128, 1e9], "p95 blur per sample", "px"))
    lines.append(hist(col["coc_max"], [0, 8, 16, 32, 64, 96, 128, 256, 1e9], "max blur per sample", "px"))
    lines.append(hist(col["coc_at_inf_px"], [0, 4, 8, 16, 32, 64, 128, 256, 1e9], "blur of the far background", "px"))
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
