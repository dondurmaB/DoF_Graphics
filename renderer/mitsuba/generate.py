#!/usr/bin/env python3
"""Bulk dataset generation: a plan of chunks and workers that pull them from a shared queue.

  python generate.py plan   --out DATASET --samples 10000        # write DATASET/_queue/plan.json
  python generate.py work   --out DATASET                        # one worker per GPU, until the queue is empty
  python generate.py status --out DATASET
  python generate.py release --out DATASET                       # requeue chunks claimed by dead workers
  python generate.py sheet  --out DATASET --per-scene 8          # PNG per scene: sharp | DoF pairs to review
  python generate.py ship   --out DATASET --to HOST:/path        # copy finished poses there, keep only markers
  python generate.py plan-topup --out TOPUP --from DATASET/_queue/plan.json --name topup1
                                                                 # one extra realistic lens per existing pose

With `work --ship-to HOST:/path` each worker ships after every chunk, so the render machine only holds the
chunks in progress (the student-lab cluster renders, CIAI's /nfs-stor stores). A shipped pose keeps its empty
directory and `.done` locally, so it is never rendered again; the copy at the destination is complete.

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
import shutil
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
        declared = list(S.load_scene(scene).envs)
        envs = declared if args.envs == "all" else [e for e in declared if e in args.envs.split(",")]
        if not envs:
            continue
        scene_poses = poses_total // len(scenes) + (i < poses_total % len(scenes))
        for j, env in enumerate(envs):
            n = scene_poses // len(envs) + (j < scene_poses % len(envs))
            for start in range(0, n, args.chunk):
                count = min(args.chunk, n - start)
                chunks.append({"id": f"{scene}.{env}.{start:06d}", "scene": scene, "env": env,
                               "start": start, "count": count})
    return {"samples": args.samples, "lenses_per_pose": args.lenses_per_pose, "envs": args.envs,
            "sampler_args": args.sampler_args, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "chunks": chunks}


def cmd_plan_topup(args) -> int:
    """A top-up plan: the same chunks and sampler arguments as an existing plan, so every pose is redrawn
    exactly, plus the top-up name the workers pass to sampler.py (experiment_plan.md D2)."""
    q = queue_dir(args.out)
    if (q / "plan.json").exists() and not args.force:
        print(f"{q / 'plan.json'} exists; --force replaces it")
        return 1
    for sub in ("claimed", "done", "failed", "logs"):
        (q / sub).mkdir(parents=True, exist_ok=True)
    src = json.loads(Path(args.source).read_text())
    plan = {**src, "topup": args.name, "topup_of": str(args.source), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (q / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"top-up {args.name}: {len(plan['chunks'])} chunks, {sum(c['count'] for c in plan['chunks'])} poses, "
          f"one new lens each")
    return 0


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


def ship(out: Path, dest: str) -> int:
    """Copy every finished, not yet shipped pose of a done chunk to `dest`, then empty it locally.

    Order makes a partial transfer harmless: pose data first, then the `.done` markers, manifests and queue
    records, so the destination never has a `.done` without its data; local files are removed only after both
    copies succeeded. Exit code 24 (a file vanished because another worker pruned it) counts as success."""
    import tempfile

    out = Path(out)
    q = queue_dir(out)
    plan = {c["id"]: c for c in json.loads((q / "plan.json").read_text())["chunks"]}
    poses, extra = [], []
    for marker in sorted((q / "done").iterdir()):
        c = plan.get(marker.name)
        if c is None:
            continue
        for env_dir in out.glob(f"{c['scene']}/s*/{c['env']}"):
            for i in range(c["start"], c["start"] + c["count"]):
                poses += [p for p in env_dir.glob(f"p{i:06d}_*") if (p / ".done").exists()
                          and ((p / "sharp.exr").exists() or (p / "topup.json").exists())]
            extra += [m for m in env_dir.glob(f"manifest.{c['id']}.jsonl")]
        extra += [marker] + [p for p in (q / "logs" / f"{c['id']}.log",) if p.exists()]
    if not poses:
        return 0
    rel = lambda paths: "".join(str(p.relative_to(out)) + "\n" for p in paths)
    with tempfile.TemporaryDirectory() as tmp:
        data, marks = Path(tmp) / "data", Path(tmp) / "marks"
        data.write_text(rel(poses))
        marks.write_text(rel([p / ".done" for p in poses] + extra + [q / "plan.json"]))
        ssh = "ssh -o BatchMode=yes -o ConnectTimeout=30"
        for files, extra_args in ((data, ["-r", "--exclude", ".done"]), (marks, [])):
            rc = subprocess.run(["rsync", "-a", "--partial", "-e", ssh, *extra_args, f"--files-from={files}",
                                 f"{out}/", f"{dest.rstrip('/')}/"]).returncode
            if rc not in (0, 24):
                print(f"ship to {dest} failed (rsync {rc}); will retry after the next chunk", flush=True)
                return 0
    for p in poses:
        for child in p.iterdir():
            if child.name != ".done":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    print(f"shipped {len(poses)} poses to {dest}", flush=True)
    return len(poses)


def cmd_ship(args) -> int:
    ship(args.out, args.to)
    return 0


def resumable(q: Path) -> set:
    """Claimed, unfinished chunks this worker should take over: its own from before a requeue (same SLURM job
    and GPU), and orphans whose owning job is no longer running (node failure, a job that ended). An orphan is
    adopted with an atomic mkdir, so two workers starting together never take the same one."""
    job, gpu = os.environ.get("SLURM_JOB_ID"), os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        running = set(subprocess.run(["squeue", "-h", "-o", "%A"], capture_output=True, text=True,
                                     check=True).stdout.split())
    except (OSError, subprocess.CalledProcessError):
        return set()            # no scheduler to ask: never guess that a claim is dead
    take = set()
    for p in (q / "claimed").iterdir():
        if (q / "done" / p.name).exists():
            continue
        try:
            owner = json.loads((p / "owner.json").read_text())
        except (OSError, ValueError):
            continue
        if owner.get("job") == job and owner.get("gpu") == gpu:
            take.add(p.name)
        elif owner.get("job") and owner["job"] not in running:
            try:
                os.mkdir(p / f"adopted-from-{owner['job']}-{owner.get('gpu')}")
            except FileExistsError:
                continue
            (p / "owner.json").write_text(json.dumps({**owner, "job": job, "gpu": gpu, "host": socket.gethostname(),
                                                      "adopted_from": owner["job"]}))
            take.add(p.name)
    return take


def cmd_work(args) -> int:
    q = queue_dir(args.out)
    plan = json.loads((q / "plan.json").read_text())
    done_here = 0
    resume = resumable(q)
    if resume:
        print(f"resuming unfinished chunks (own before a requeue, or orphaned): {sorted(resume)}", flush=True)
    for c in sorted(plan["chunks"], key=lambda c: c["id"] not in resume):
        if (q / "done" / c["id"]).exists() or not (c["id"] in resume or claim(q, c["id"])):
            continue
        cmd = [sys.executable, "-u", str(HERE / "sampler.py"), "--scene", c["scene"], "--env", c["env"],
               "--start", str(c["start"]), "--count", str(c["count"]),
               "--lenses-per-pose", str(plan["lenses_per_pose"]), "--out", str(args.out),
               "--manifest-name", f"manifest.{c['id']}.jsonl", *plan["sampler_args"],
               *(["--topup", plan["topup"]] if plan.get("topup") else [])]
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
        if finished and args.ship_to:
            ship(args.out, args.ship_to)
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
            shutil.rmtree(p)
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
    for name in ("plan", "plan-topup", "work", "status", "release", "sheet", "ship"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True, help="dataset root")
        if name == "plan-topup":
            p.add_argument("--from", dest="source", required=True, help="the original dataset's _queue/plan.json")
            p.add_argument("--name", required=True, help="top-up name, e.g. topup1")
            p.add_argument("--force", action="store_true")
        if name == "plan":
            p.add_argument("--samples", type=int, required=True, help="(pose, lens) samples in total")
            p.add_argument("--lenses-per-pose", type=int, default=2)
            p.add_argument("--chunk", type=int, default=50, help="poses per chunk")
            p.add_argument("--envs", default="all", help="'all' (each scene's declared envs) or a comma list, "
                           "e.g. clear,overcast: a scene gets those of them it declares")
            p.add_argument("--scenes", nargs="*", default=None, help="default: the train split")
            p.add_argument("--force", action="store_true")
            p.add_argument("sampler_args", nargs=argparse.REMAINDER,
                           help="after --: extra sampler.py arguments, e.g. -- --res 960x540 --spp 256")
        if name == "work":
            p.add_argument("--max-chunks", type=int, default=0, help="stop after this many (0: until empty)")
            p.add_argument("--ship-to", default=None, help="HOST:/path to copy each finished chunk to (see ship)")
        if name == "ship":
            p.add_argument("--to", required=True, help="HOST:/path, e.g. ciai-copy:/nfs-stor/rui.gao/Researches/dof/dof_dataset/train10k")
        if name == "sheet":
            p.add_argument("--per-scene", type=int, default=8)
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        args.sampler_args = [a for a in args.sampler_args if a != "--"]
    return {"plan": cmd_plan, "plan-topup": cmd_plan_topup, "work": cmd_work, "status": cmd_status, "release": cmd_release,
            "sheet": cmd_sheet, "ship": cmd_ship}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
