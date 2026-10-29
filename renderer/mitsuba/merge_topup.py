#!/usr/bin/env python3
"""Merge a top-up render (sampler.py --topup, experiment_plan.md D2) into the dataset it extends.

  python merge_topup.py DATASET --topup TOPUP_ROOT --name topup1 [--dry-run]

For every lens in TOPUP_ROOT's manifests: the pose it belongs to must exist in DATASET (same pose_id, `.done`),
otherwise it is reported and skipped. The lens's coc.npy is recomputed from the dataset's stored depth.npy (the
model's input), so CoC and input always agree; the top-up's own re-rendered depth is compared with the stored one
(md5 in topup.json) and the largest CoC difference is reported. The lens directory is then moved into the pose
directory, and its manifest lines go to DATASET/<scene>/s<seed>/<env>/manifest.<name>.<chunk>.jsonl, which
ai/data.py find_samples() picks up like any other manifest. Safe to rerun: lenses already merged are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import render as R  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--topup", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    stats = {"merged": 0, "already": 0, "no_pose": 0, "depth_same": 0, "depth_diff": 0}
    worst = 0.0
    kinds = {}
    for manifest in sorted(args.topup.glob("*/s*/*/manifest.*.jsonl")):
        env_rel = manifest.parent.relative_to(args.topup)
        dest_env = args.dataset / env_rel
        chunk = manifest.name[len("manifest."):-len(".jsonl")]
        keep = []
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            pose_id, lens_id = entry["path"].split("/")
            src, dst_pose = manifest.parent / entry["path"], dest_env / pose_id
            if not (dst_pose / ".done").exists():
                stats["no_pose"] += 1
                print(f"NO POSE {env_rel}/{pose_id}: not in the dataset, skipped", flush=True)
                continue
            dst = dst_pose / lens_id
            if dst.exists():
                stats["already"] += 1
                keep.append(entry)
                continue
            depth = np.load(dst_pose / "depth.npy")
            info = json.loads((manifest.parent / pose_id / "topup.json").read_text())
            same = hashlib.md5(np.ascontiguousarray(depth).tobytes()).hexdigest() == info["depth_md5"]
            stats["depth_same" if same else "depth_diff"] += 1
            cam = json.loads((src / "metadata.json").read_text())["camera"]
            lens = R.Lens(cam["origin"], cam["target"], cam["focal_length_mm"], cam["sensor_height_mm"],
                          cam["f_number"], cam["focus_distance_m"], cam["roll_deg"])
            coc = lens.coc_map(depth, depth.shape[0]).astype(np.float32)
            old = np.load(src / "coc.npy")
            fin = np.isfinite(coc) & np.isfinite(old)
            worst = max(worst, float(np.abs(coc[fin] - old[fin]).max()) if fin.any() else 0.0)
            kinds[entry.get("topup_kind")] = kinds.get(entry.get("topup_kind"), 0) + 1
            if not args.dry_run:
                np.save(src / "coc.npy", coc)
                shutil.move(str(src), str(dst))
            stats["merged"] += 1
            keep.append(entry)
        if keep and not args.dry_run:
            (dest_env / f"manifest.{args.name}.{chunk}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in keep))
    print(f"{'dry run: ' if args.dry_run else ''}{stats} | kinds {kinds} | largest CoC change from re-rendered to stored "
          f"depth {worst:.4f} px", flush=True)
    return 1 if stats["no_pose"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
