#!/usr/bin/env python3
"""Render the Mitsuba cafe: all-in-focus, thin-lens depth of field, and a G-buffer.

Every pass uses the same camera pose and field of view, so pixels correspond
exactly across outputs:

  sharp.exr / .png   pinhole camera, path traced (the all-in-focus input)
  dof.exr / .png     thin-lens camera, path traced (ground-truth depth of field)
  depth.exr / .npy   planar z-depth in meters along the optical axis (inf = sky)
  gbuffer.exr        position, shading normal, albedo, shape index
  metadata.json      lens, focus, intrinsics, extrinsics, spp, timings

Examples:
  python render.py --preview                              # quick local check
  python render.py --res 1920x1080 --spp 1024 --out output/mitsuba/cafe
  python render.py --focus-target books --f-number 1.4    # focus the midground
  python render.py --scene cafe --roll 12                 # camera tilted 12 degrees about its axis
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent.parent

import scene_api as S  # noqa: E402

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="cafe", help=f"scene id; available: {', '.join(S.list_scenes())}")
    ap.add_argument("--scene-seed", type=int, default=None, help="seed for the scene's procedural layout (default: the scene's own)")
    ap.add_argument("--env", default="clear", help="environment preset the scene implements")
    ap.add_argument("--roll", type=float, default=None, help="camera roll about the optical axis, degrees (default: the view's)")
    ap.add_argument("--variant", default="auto",
                    help="Mitsuba variant, or 'auto' for cuda > metal > llvm > scalar (rgb)")
    ap.add_argument("--res", default="1280x720", help="WIDTHxHEIGHT")
    ap.add_argument("--spp", type=int, default=256, help="samples per pixel for each beauty pass")
    ap.add_argument("--dof-spp", type=int, default=None, help="override spp for the DoF pass (default: --spp)")
    ap.add_argument("--spp-per-pass", type=int, default=64,
                    help="split rendering into chunks of this many spp to bound GPU memory")
    ap.add_argument("--max-depth", type=int, default=None, help="default: the scene's own")
    ap.add_argument("--passes", default="sharp,dof,gbuffer", help="comma list of sharp,dof,gbuffer")
    ap.add_argument("--view", default=None, help="a view declared by the scene (default: its default_view)")
    ap.add_argument("--lens", type=float, default=50.0, help="focal length, mm")
    ap.add_argument("--sensor-height", type=float, default=S.SENSOR_HEIGHT_MM, help="sensor height, mm")
    ap.add_argument("--f-number", type=float, default=1.8)
    ap.add_argument("--focus", type=float, default=None, help="focus distance, meters (planar)")
    ap.add_argument("--focus-target", default=None, help="named point to focus on (see --list-targets; default: the view's)")
    ap.add_argument("--list-targets", action="store_true")
    ap.add_argument("--exposure", type=float, default=0.0, help="EV applied to PNG previews only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: output/mitsuba/<scene>")
    ap.add_argument("--assets", default=str(HERE / "generated"), help="procedural asset cache root")
    ap.add_argument("--rebuild-assets", action="store_true")
    ap.add_argument("--preview", action="store_true", help="640x360, 32 spp")
    args = ap.parse_args(argv)
    if args.preview:
        args.res, args.spp = "640x360", min(args.spp, 32)
    args.width, args.height = (int(v) for v in args.res.lower().split("x"))
    args.passes = [p.strip() for p in args.passes.split(",") if p.strip()]
    return args


def pick_variant(requested: str) -> str:
    import mitsuba as mi
    import drjit as dr

    if requested != "auto":
        mi.set_variant(requested)
        return requested
    backends = [("cuda_ad_rgb", "CUDA"), ("metal_ad_rgb", "Metal"), ("llvm_ad_rgb", "LLVM")]
    for variant, backend in backends:
        jb = getattr(dr.JitBackend, backend, None)
        if variant in mi.variants() and jb is not None and dr.has_backend(jb):
            mi.set_variant(variant)
            return variant
    mi.set_variant("scalar_rgb")
    return "scalar_rgb"


def rolled_up(forward: np.ndarray, roll_deg: float) -> np.ndarray:
    """World-up projected perpendicular to `forward`, rotated about `forward` by roll_deg."""
    up = np.array([0.0, 1.0, 0.0])
    up = up - np.dot(up, forward) * forward
    norm = np.linalg.norm(up)
    if norm < 1e-6:
        raise ValueError("camera looks straight up or down; roll is undefined")
    up /= norm
    a = math.radians(roll_deg)
    return up * math.cos(a) + np.cross(forward, up) * math.sin(a)


class Lens:
    """Thin-lens camera matching the DoFScene/DoFApproaches convention.

    Vertical FOV comes from focal length over sensor height; the aperture
    diameter is focal length over f-number. Mitsuba's `thinlens` focuses on a
    plane at `focus_distance` along the optical axis, so the focus distance
    here is planar z-depth, the same quantity written to depth.exr.
    """

    def __init__(self, origin, target, lens_mm, sensor_mm, f_number, focus_m, roll_deg=0.0, far_clip=1000.0):
        self.origin = np.asarray(origin, float)
        self.target = np.asarray(target, float)
        self.forward = (self.target - self.origin) / np.linalg.norm(self.target - self.origin)
        self.roll_deg = float(roll_deg)
        self.far_clip = float(far_clip)
        self.up = rolled_up(self.forward, self.roll_deg)
        self.lens_m = lens_mm / 1000.0
        self.sensor_m = sensor_mm / 1000.0
        self.f_number = f_number
        self.focus_m = focus_m
        self.fov_y = math.degrees(2.0 * math.atan(0.5 * sensor_mm / lens_mm))
        self.aperture_radius = 0.5 * self.lens_m / f_number

    def depth_of(self, point) -> float:
        return float(np.dot(np.asarray(point, float) - self.origin, self.forward))

    def coc_pixels(self, depth_m: float, height_px: int) -> float:
        """Circle-of-confusion diameter in pixels for a point at planar depth.

        Mitsuba's `thinlens` keeps the field of view fixed by the focal length
        (image distance = f), so the diameter is H f^2 |z - s| / (N z s sensor).
        The textbook form divides by (s - f) instead of s; it overestimates
        by s / (s - f), measured 16-29% at 135 mm.
        """
        f, s = self.lens_m, self.focus_m
        return height_px * f * f * abs(depth_m - s) / (self.f_number * depth_m * s * self.sensor_m)

    def coc_map(self, depth: np.ndarray, height_px: int) -> np.ndarray:
        """Signed CoC diameter in pixels per pixel: positive behind the focus plane, negative in front.

        Pixels with no surface (depth = inf) get the at-infinity value, positive.
        """
        f, s = self.lens_m, self.focus_m
        scale = height_px * f * f / (self.f_number * s * self.sensor_m)
        with np.errstate(invalid="ignore", divide="ignore"):
            signed = np.where(np.isfinite(depth), scale * (depth - s) / depth, scale)
        return signed.astype(np.float32)

    def sensor(self, width, height, spp, thin_lens: bool, rfilter: str = "gaussian") -> dict:
        import mitsuba as mi

        spec = {
            "type": "thinlens" if thin_lens else "perspective",
            "fov": self.fov_y,
            "fov_axis": "y",
            "near_clip": 0.01,
            "far_clip": self.far_clip,
            "to_world": mi.ScalarTransform4f().look_at(origin=self.origin.tolist(),
                                                       target=self.target.tolist(), up=self.up.tolist()),
            "sampler": {"type": "independent", "sample_count": spp},
            "film": {"type": "hdrfilm", "width": width, "height": height, "pixel_format": "rgb",
                     "rfilter": {"type": rfilter}},
        }
        if thin_lens:
            spec["aperture_radius"] = self.aperture_radius
            spec["focus_distance"] = self.focus_m
        return spec

    def metadata(self, width, height) -> dict:
        fy = self.lens_m / self.sensor_m * height
        return {
            "origin": self.origin.tolist(), "target": self.target.tolist(), "up": self.up.tolist(),
            "roll_deg": self.roll_deg,
            "aperture": {"shape": "disc", "rotation_deg": 0.0},
            "forward": self.forward.tolist(),
            "focal_length_mm": self.lens_m * 1000, "sensor_height_mm": self.sensor_m * 1000,
            "f_number": self.f_number, "aperture_radius_m": self.aperture_radius,
            "focus_distance_m": self.focus_m, "fov_y_deg": self.fov_y,
            "intrinsics_px": {"fx": fy, "fy": fy, "cx": width / 2, "cy": height / 2},
            "coc_diameter_px_at_infinity": self.lens_m ** 2 / (self.f_number * self.focus_m) / self.sensor_m * height,
        }


def tonemap(rgb: np.ndarray, exposure_ev: float) -> np.ndarray:
    """ACES filmic fit (Narkowicz 2015) then sRGB encoding, for previews."""
    x = np.maximum(rgb, 0.0) * (2.0 ** exposure_ev) * 0.8
    x = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def save_png(path: Path, rgb01: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray((np.clip(rgb01, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def render_beauty(scene, sensor_dict, spp, chunk, seed, label):
    import mitsuba as mi

    sensor = mi.load_dict(sensor_dict)
    total = np.zeros((sensor_dict["film"]["height"], sensor_dict["film"]["width"], 3), np.float64)
    done, k, t0 = 0, 0, time.time()
    while done < spp:
        n = min(chunk, spp - done)
        img = mi.render(scene, sensor=sensor, spp=n, seed=seed * 100003 + k)
        total += np.asarray(img, dtype=np.float64)[..., :3] * n
        done += n
        k += 1
        print(f"  [{label}] {done}/{spp} spp  {time.time() - t0:6.1f}s", flush=True)
    return (total / spp).astype(np.float32), time.time() - t0


def render_gbuffer(scene, sensor_dict, lens: Lens, seed):
    """One jittered sample per pixel: every value belongs to a single surface.

    Averaging many samples would blend foreground and background depth along
    silhouettes into depths that exist nowhere in the scene, which is the
    wrong ground truth for a depth-of-field study.
    """
    import mitsuba as mi

    integrator = mi.load_dict({"type": "aov", "aovs": "pos:position,nn:sh_normal,alb:albedo,idx:shape_index,t:depth"})
    sensor = mi.load_dict(sensor_dict)
    t0 = time.time()
    img = np.asarray(mi.render(scene, sensor=sensor, integrator=integrator, spp=1, seed=seed), np.float32)
    names = [n.split(".")[0] for n in integrator.aov_names()]
    layers = {n: img[..., [i for i, m in enumerate(names) if m == n]] for n in dict.fromkeys(names)}
    pos, hit = layers["pos"], layers["t"][..., 0] > 0
    depth = np.einsum("hwc,c->hw", pos - lens.origin.astype(np.float32), lens.forward.astype(np.float32))
    depth = np.where(hit, depth, np.inf).astype(np.float32)
    return depth, layers, time.time() - t0


def write_gbuffer(path: Path, layers: dict) -> None:
    import mitsuba as mi

    gb = np.concatenate([layers["pos"], layers["nn"], layers["alb"], layers["idx"][..., :1]], -1)
    mi.Bitmap(gb, channel_names=["P.X", "P.Y", "P.Z", "N.X", "N.Y", "N.Z", "albedo.R", "albedo.G",
                                 "albedo.B", "shape_index.Y"]).write(str(path))


def _ply_faces(path) -> int:
    with open(path, "rb") as f:
        for line in f:
            line = line.decode("ascii", "ignore").strip()
            if line.startswith("element face"):
                return int(line.split()[-1])
            if line == "end_header":
                return 0
    return 0


def unique_triangles(scene, scene_dict: dict) -> int:
    """Triangles stored in the scene: loaded meshes plus each shapegroup's geometry once.

    Instances report face_count 0 (measured), so instanced web assets would otherwise vanish
    from the budget; repeated instances cost no extra geometry, so they are counted once."""
    tris = sum(int(s.face_count()) for s in scene.shapes() if hasattr(s, "face_count"))
    for v in scene_dict.values():
        if isinstance(v, dict) and v.get("type") == "shapegroup":
            tris += sum(_ply_faces(c["filename"]) for c in v.values()
                        if isinstance(c, dict) and c.get("type") == "ply")
    return tris


def git_state() -> dict:
    try:
        run = lambda *c: subprocess.run(["git", *c], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except Exception:
        return {"commit": None, "dirty": None}


def main(argv=None) -> int:
    args = parse_args(argv)
    variant = pick_variant(args.variant)
    import mitsuba as mi

    t0 = time.time()
    scene_def = S.load_scene(args.scene)
    view_name = args.view or scene_def.default_view
    if view_name not in scene_def.views:
        raise SystemExit(f"scene {args.scene!r} has no view {view_name!r}; choose from {sorted(scene_def.views)}")
    view = scene_def.views[view_name]
    ctx = S.make_context(scene_def, Path(args.assets), args.scene_seed, args.env, args.rebuild_assets)
    bundle = S.build_checked(scene_def, ctx)
    focus_points = bundle.focus_points
    if args.list_targets:
        for k, v in focus_points.items():
            print(f"{k:10s} {v}")
        return 0
    roll = view.roll_deg if args.roll is None else args.roll
    target_name = args.focus_target or view.focus or next(iter(focus_points))
    probe = Lens(view.origin, view.target, args.lens, args.sensor_height, args.f_number, 1.0, roll, scene_def.far_clip)
    if args.focus is None:
        args.focus = probe.depth_of(focus_points[target_name][0])
    lens = Lens(view.origin, view.target, args.lens, args.sensor_height, args.f_number, args.focus, roll, scene_def.far_clip)

    max_depth = args.max_depth or scene_def.max_depth
    scene_dict = dict(bundle.scene)
    scene_dict["integrator"] = {"type": "path", "max_depth": max_depth, "rr_depth": scene_def.rr_depth}
    scene_dict["sensor"] = lens.sensor(args.width, args.height, 1, thin_lens=False)
    scene = mi.load_dict(scene_dict)
    shapes = len(scene.shapes())
    tris = unique_triangles(scene, scene_dict)
    print(f"variant {variant} | {shapes} shapes, {tris:,} triangles | load {time.time() - t0:.1f}s")
    print(f"lens {args.lens:.0f} mm f/{args.f_number:g} | focus {lens.focus_m:.3f} m ({target_name}) | roll {roll:g} deg | "
          f"CoC at infinity {lens.metadata(args.width, args.height)['coc_diameter_px_at_infinity']:.1f} px")

    out = Path(args.out or REPO / "output" / "mitsuba" / args.scene)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"contract_version": S.CONTRACT_VERSION,
            "scene": {"id": scene_def.id, "group": scene_def.group, "seed": ctx.seed, "env": ctx.env,
                      "fingerprint": S.fingerprint(bundle.scene, [ctx.shared_dir, ctx.assets_dir]),
                      "asset_version": scene_def.asset_version, **bundle.meta},
            "git": git_state(), "variant": variant, "mitsuba": mi.__version__,
            "resolution": [args.width, args.height], "view": view_name, "focus_target": target_name,
            "render_seed": args.seed, "exposure_ev_png_only": args.exposure,
            "camera": lens.metadata(args.width, args.height),
            "integrator": {"type": "path", "max_depth": max_depth, "rr_depth": scene_def.rr_depth},
            "coc_formula": "H f^2 |z - s| / (N z s sensor_h), planar z, matches Mitsuba thinlens",
            "shapes": shapes, "triangles": tris, "passes": {}}

    if "sharp" in args.passes:
        rgb, sec = render_beauty(scene, lens.sensor(args.width, args.height, args.spp, False),
                                 args.spp, args.spp_per_pass, args.seed, "sharp")
        mi.Bitmap(rgb).write(str(out / "sharp.exr"))
        save_png(out / "sharp.png", tonemap(rgb, args.exposure))
        meta["passes"]["sharp"] = {"spp": args.spp, "seconds": round(sec, 2)}
    if "dof" in args.passes:
        spp = args.dof_spp or args.spp
        rgb, sec = render_beauty(scene, lens.sensor(args.width, args.height, spp, True),
                                 spp, args.spp_per_pass, args.seed + 1, "dof")
        mi.Bitmap(rgb).write(str(out / "dof.exr"))
        save_png(out / "dof.png", tonemap(rgb, args.exposure))
        meta["passes"]["dof"] = {"spp": spp, "seconds": round(sec, 2)}
    if "gbuffer" in args.passes:
        depth, layers, sec = render_gbuffer(scene, lens.sensor(args.width, args.height, 1, False, "box"),
                                            lens, (args.seed + 2) * 100003 + 50000)
        np.save(out / "depth.npy", depth)
        mi.Bitmap(np.where(np.isfinite(depth), depth, 0.0)[..., None]).write(str(out / "depth.exr"))
        finite = depth[np.isfinite(depth)]
        near, far = float(finite.min()), float(np.percentile(finite, 99.5))
        vis = 1.0 - np.clip((np.log(np.where(np.isfinite(depth), depth, far)) - math.log(near))
                            / (math.log(far) - math.log(near)), 0, 1)
        save_png(out / "depth.png", np.repeat(vis[..., None], 3, -1))
        write_gbuffer(out / "gbuffer.exr", layers)
        save_png(out / "normal.png", 0.5 + 0.5 * layers["nn"])
        meta["depth_range_m"] = [near, float(finite.max())]
        meta["sky_fraction"] = float(1.0 - finite.size / depth.size)
        meta["coc_px_percentiles"] = {str(p): float(v) for p, v in zip(
            (5, 50, 95), np.percentile(np.abs(lens.coc_map(depth[::8, ::8], args.height)), (5, 50, 95)))}
        meta["passes"]["gbuffer"] = {"spp": 1, "seconds": round(sec, 2)}

    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}  (total {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
