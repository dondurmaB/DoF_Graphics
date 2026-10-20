#!/usr/bin/env python3
"""Bulk dataset generation: a plan of chunks and workers that pull them from a shared queue.

  python generate.py plan   --out DATASET --samples 10000        # write DATASET/_queue/plan.json
  python generate.py work   --out DATASET                        # one worker per GPU, until the queue is empty
  python generate.py status --out DATASET
  python generate.py release --out DATASET                       # requeue chunks claimed by dead workers
  python generate.py sheet  --out DATASET --per-scene 8          # PNG per scene: sharp | DoF pairs to review

The plan splits `--samples` evenly over the train scenes (scenes/splits.json) and, inside a scene, evenly over
the environments it declares; a sample is one (pose, lens) pair, so poses = samples / lenses-per-pose. Each
chunk is `--chunk` consecutive pose indices of one (scene, env), rendered by one `sampler.py` process.

Workers claim a chunk with an atomic mkdir in DATASET/_queue/claimed/ (safe on NFS, across nodes), write its
manifest to its own `manifest.<chunk>.jsonl` (parallel appends to one file over NFS can interleave), and mark
it done/ or failed/. A rerun of a chunk skips poses already finished (`.done`), so a requeued chunk only
renders what is missing. Samples are pure functions of their indices, so the result does not depend on how
many workers ran or in which order.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scene_api as S  # noqa: E402


def queue_dir(out: Path) -> Path:
    return Path(out) / "_queue"


def make_plan(args) -> dict:
    splits = json.loads((HERE / "scenes" / "splits.json").read_text())
    scenes = args.scenes or splits["train"]
    poses_total = args.samples // args.lenses_per_pose
    chunks = []
    for i, scene in enumerate(scenes):
        envs = list(S.load_scene(scene).envs) if args.envs == "all" else ["clear"]
        scene_poses = poses_total // len(scenes) + (i < poses_total % len(scenes))
        for j, env in enumerate(envs):
            n = scene_poses // len(envs) + (j < scene_poses % len(envs))
            for start in range(0, n, args.chunk):
                count = min(args.chunk, n - start)
                chunks.append({"id": f"{scene}.{env}.{start:06d}", "scene": scene, "env": env,
                               "start": start, "count": count})
    return {"samples": args.samples, "lenses_per_pose": args.lenses_per_pose, "envs": args.envs,
            "sampler_args": args.sampler_args, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "chunks": chunks}


def cmd_plan(args) -> int:
    q = queue_dir(args.out)
    if (q / "plan.json").exists() and not args.force:
        print(f"{q / 'plan.json'} exists; --force replaces it (claimed and done chunks keep their markers)")
        return 1
    for sub in ("claimed", "done", "failed", "logs"):
        (q / sub).mkdir(parents=True, exist_ok=True)
    plan = make_plan(args)
    (q / "plan.json").write_text(json.dumps(plan, indent=1))
    per = {}
    for c in plan["chunks"]:
        per.setdefault(c["scene"], {}).setdefault(c["env"], 0)
        per[c["scene"]][c["env"]] += c["count"]
    for scene, envs in per.items():
        print(f"{scene:14s} " + "  ".join(f"{e} {n} poses" for e, n in envs.items()))
    print(f"{len(plan['chunks'])} chunks, {sum(c['count'] for c in plan['chunks'])} poses, "
          f"{sum(c['count'] for c in plan['chunks']) * args.lenses_per_pose} samples")
    return 0


def claim(q: Path, chunk_id: str) -> bool:
    try:
        os.mkdir(q / "claimed" / chunk_id)
    except FileExistsError:
        return False
    owner = {"host": socket.gethostname(), "pid": os.getpid(), "job": os.environ.get("SLURM_JOB_ID"),
             "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (q / "claimed" / chunk_id / "owner.json").write_text(json.dumps(owner))
    return True


def cmd_work(args) -> int:
    q = queue_dir(args.out)
    plan = json.loads((q / "plan.json").read_text())
    done_here = 0
    for c in plan["chunks"]:
        if (q / "done" / c["id"]).exists() or not claim(q, c["id"]):
            continue
        cmd = [sys.executable, "-u", str(HERE / "sampler.py"), "--scene", c["scene"], "--env", c["env"],
               "--start", str(c["start"]), "--count", str(c["count"]),
               "--lenses-per-pose", str(plan["lenses_per_pose"]), "--out", str(args.out),
               "--manifest-name", f"manifest.{c['id']}.jsonl", *plan["sampler_args"]]
        log = q / "logs" / f"{c['id']}.log"
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {c['id']}: {c['count']} poses ...", flush=True)
        with open(log, "a") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        tail = log.read_text().splitlines()[-4:]
        finished = any(" poses in " in line for line in tail)   # the sampler's summary line: it ran to the end
        status = {"rc": rc, "seconds": round(time.time() - t0, 1), "host": socket.gethostname(), "tail": tail}
        (q / ("done" if finished else "failed") / c["id"]).write_text(json.dumps(status))
        print(f"[{time.strftime('%H:%M:%S')}] {c['id']}: {'done' if finished else 'FAILED'} rc {rc} "
              f"in {status['seconds']:.0f}s", flush=True)
        done_here += finished
        if args.max_chunks and done_here >= args.max_chunks:
            break
    print(f"worker finished: {done_here} chunks", flush=True)
    return 0


def cmd_status(args) -> int:
    q = queue_dir(args.out)
    plan = json.loads((q / "plan.json").read_text())
    ids = [c["id"] for c in plan["chunks"]]
    poses = {c["id"]: c["count"] for c in plan["chunks"]}
    done = {p.name for p in (q / "done").iterdir()} & set(ids)
    failed = {p.name for p in (q / "failed").iterdir()} & set(ids) - done
    claimed = {p.name for p in (q / "claimed").iterdir()} & set(ids) - done - failed
    secs = [json.loads((q / "done" / i).read_text())["seconds"] for i in done]
    print(f"chunks: {len(done)} done, {len(claimed)} running, {len(failed)} failed, "
          f"{len(ids) - len(done) - len(claimed) - len(failed)} waiting of {len(ids)}")
    print(f"poses done: {sum(poses[i] for i in done)} of {sum(poses.values())}")
    if secs:
        per_pose = sum(secs) / sum(poses[i] for i in done)
        left = sum(poses.values()) - sum(poses[i] for i in done)
        print(f"GPU time per pose: {per_pose:.1f}s | remaining work: {left * per_pose / 3600:.1f} GPU-hours")
    for i in sorted(failed):
        print(f"FAILED {i}: {json.loads((q / 'failed' / i).read_text())['tail'][-1:]}")
    return 0


def cmd_release(args) -> int:
    """Requeue chunks that are claimed or failed but not done. Run only when no worker is active."""
    q = queue_dir(args.out)
    n = 0
    for p in (q / "claimed").iterdir():
        if not (q / "done" / p.name).exists():
            for f in p.iterdir():
                f.unlink()
            p.rmdir()
            (q / "failed" / p.name).unlink(missing_ok=True)
            n += 1
    print(f"released {n} chunks")
    return 0


def cmd_sheet(args) -> int:
    """One PNG per scene of tone-mapped sharp | DoF pairs, labelled with env and lens, for review by eye."""
    import numpy as np
    from PIL import Image, ImageDraw

    sys.path.insert(0, str(HERE.parent.parent / "ai"))
    import data as D
    import render as R

    samples = D.find_samples(args.out, split=None)
    by_scene = {}
    for smp in samples:
        by_scene.setdefault(smp.scene, []).append(smp)
    sheet_dir = Path(args.out) / "_sheets"
    sheet_dir.mkdir(exist_ok=True)
    tw, th = 480, 270
    for scene, items in sorted(by_scene.items()):
        envs = sorted({x.env for x in items})
        pick = []           # round-robin over envs so each env shows up
        while len(pick) < min(args.per_scene, len(items)):
            for env in envs:
                rest = [x for x in items if x.env == env and x not in pick]
                if rest and len(pick) < args.per_scene:
                    pick.append(rest[0])
        sheet = Image.new("RGB", (2 * tw + 12, len(pick) * (th + 6) + 6), (20, 20, 22))
        for i, smp in enumerate(pick):
            scale = float(smp.entry.get("auto_exposure_scale", 1.0))
            meta = json.loads((smp.lens_dir / "metadata.json").read_text())["camera"]
            for j, path in enumerate((smp.pose_dir / "sharp.exr", smp.lens_dir / "dof.exr")):
                rgb = D.read_exr(path) * scale
                tile = Image.fromarray((np.clip(R.tonemap(rgb, 0.0), 0, 1) * 255).astype(np.uint8)).resize((tw, th))
                sheet.paste(tile, (6 + j * (tw + 0), 6 + i * (th + 6)))
            label = (f"{smp.env}  {meta['focal_length_mm']:.0f}mm f/{meta['f_number']:.1f} "
                     f"focus {meta['focus_distance_m']:.2f}m  roll {meta['roll_deg']:+.0f}")
            d = ImageDraw.Draw(sheet)
            d.rectangle((6, 6 + i * (th + 6), 6 + 7 * len(label), 20 + i * (th + 6)), fill=(0, 0, 0))
            d.text((9, 7 + i * (th + 6)), label, fill=(240, 240, 240))
        sheet.save(sheet_dir / f"{scene}.png")
        print(f"{sheet_dir / (scene + '.png')}: {len(pick)} pairs, envs {envs}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "work", "status", "release", "sheet"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True, help="dataset root")
        if name == "plan":
            p.add_argument("--samples", type=int, required=True, help="(pose, lens) samples in total")
            p.add_argument("--lenses-per-pose", type=int, default=2)
            p.add_argument("--chunk", type=int, default=50, help="poses per chunk")
            p.add_argument("--envs", choices=("all", "clear"), default="all")
            p.add_argument("--scenes", nargs="*", default=None, help="default: the train split")
            p.add_argument("--force", action="store_true")
            p.add_argument("sampler_args", nargs=argparse.REMAINDER,
                           help="after --: extra sampler.py arguments, e.g. -- --res 960x540 --spp 256")
        if name == "work":
            p.add_argument("--max-chunks", type=int, default=0, help="stop after this many (0: until empty)")
        if name == "sheet":
            p.add_argument("--per-scene", type=int, default=8)
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        args.sampler_args = [a for a in args.sampler_args if a != "--"]
    return {"plan": cmd_plan, "work": cmd_work, "status": cmd_status, "release": cmd_release,
            "sheet": cmd_sheet}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
