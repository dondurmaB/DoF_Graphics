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
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent.parent

VIEWS = {
    # Seated at the hero table, looking down the room toward the counter.
    "home": {"origin": (0.34, 1.22, 1.85), "target": (-0.4, 0.88, -3.0)},
    # Lower and closer: strong foreground, the counter far behind.
    "close": {"origin": (0.12, 0.98, 0.95), "target": (-0.25, 0.86, -2.0)},
    # From the front corner, taking in the windows and most of the room.
    "wide": {"origin": (2.6, 1.55, 2.9), "target": (-1.2, 1.0, -3.0)},
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="auto",
                    help="Mitsuba variant, or 'auto' for cuda > metal > llvm > scalar (rgb)")
    ap.add_argument("--res", default="1280x720", help="WIDTHxHEIGHT")
    ap.add_argument("--spp", type=int, default=256, help="samples per pixel for each beauty pass")
    ap.add_argument("--dof-spp", type=int, default=None, help="override spp for the DoF pass (default: --spp)")
    ap.add_argument("--spp-per-pass", type=int, default=64,
                    help="split rendering into chunks of this many spp to bound GPU memory")
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--passes", default="sharp,dof,gbuffer", help="comma list of sharp,dof,gbuffer")
    ap.add_argument("--view", choices=sorted(VIEWS), default="home")
    ap.add_argument("--lens", type=float, default=50.0, help="focal length, mm")
    ap.add_argument("--sensor-height", type=float, default=24.0, help="sensor height, mm")
    ap.add_argument("--f-number", type=float, default=1.8)
    ap.add_argument("--focus", type=float, default=None, help="focus distance, meters (planar)")
    ap.add_argument("--focus-target", default="teapot", help="named point to focus on (see --list-targets)")
    ap.add_argument("--list-targets", action="store_true")
    ap.add_argument("--exposure", type=float, default=0.0, help="EV applied to PNG previews only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "output" / "mitsuba" / "cafe"))
    ap.add_argument("--assets", default=str(HERE / "generated"), help="procedural asset cache")
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


class Lens:
    """Thin-lens camera matching the DoFScene/DoFApproaches convention.

    Vertical FOV comes from focal length over sensor height; the aperture
    diameter is focal length over f-number. Mitsuba's `thinlens` focuses on a
    plane at `focus_distance` along the optical axis, so the focus distance
    here is planar z-depth, the same quantity written to depth.exr.
    """

    def __init__(self, origin, target, lens_mm, sensor_mm, f_number, focus_m):
        self.origin = np.asarray(origin, float)
        self.target = np.asarray(target, float)
        self.forward = (self.target - self.origin) / np.linalg.norm(self.target - self.origin)
        self.lens_m = lens_mm / 1000.0
        self.sensor_m = sensor_mm / 1000.0
        self.f_number = f_number
        self.focus_m = focus_m
        self.fov_y = math.degrees(2.0 * math.atan(0.5 * sensor_mm / lens_mm))
        self.aperture_radius = 0.5 * self.lens_m / f_number

    def depth_of(self, point) -> float:
        return float(np.dot(np.asarray(point, float) - self.origin, self.forward))

    def coc_pixels(self, depth_m: float, height_px: int) -> float:
        """Thin-lens circle-of-confusion diameter in pixels for a point at depth."""
        f, s, a = self.lens_m, self.focus_m, 2 * self.aperture_radius
        coc_sensor = abs(a * f * (depth_m - s) / (depth_m * s))
        return coc_sensor / self.sensor_m * height_px

    def sensor(self, width, height, spp, thin_lens: bool, rfilter: str = "gaussian", sampler: str = "independent") -> dict:
        import mitsuba as mi

        spec = {
            "type": "thinlens" if thin_lens else "perspective",
            "fov": self.fov_y,
            "fov_axis": "y",
            "near_clip": 0.01,
            "far_clip": 1000.0,
            "to_world": mi.ScalarTransform4f().look_at(origin=self.origin.tolist(),
                                                       target=self.target.tolist(), up=[0, 1, 0]),
            "sampler": {"type": sampler, "sample_count": spp},
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
            "origin": self.origin.tolist(), "target": self.target.tolist(), "up": [0, 1, 0],
            "forward": self.forward.tolist(),
            "focal_length_mm": self.lens_m * 1000, "sensor_height_mm": self.sensor_m * 1000,
            "f_number": self.f_number, "aperture_radius_m": self.aperture_radius,
            "focus_distance_m": self.focus_m, "fov_y_deg": self.fov_y,
            "intrinsics_px": {"fx": fy, "fy": fy, "cx": width / 2, "cy": height / 2},
            "coc_diameter_px_at_infinity": self.lens_m ** 2 / (self.f_number * self.focus_m)
            / self.sensor_m * height,
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


def main(argv=None) -> int:
    args = parse_args(argv)
    variant = pick_variant(args.variant)
    import mitsuba as mi

    from cafe_scene import build_scene

    t0 = time.time()
    scene_dict, focus_points = build_scene(Path(args.assets), args.rebuild_assets)
    if args.list_targets:
        for k, v in focus_points.items():
            print(f"{k:10s} {v}")
        return 0
    view = VIEWS[args.view]
    probe = Lens(view["origin"], view["target"], args.lens, args.sensor_height, args.f_number, 1.0)
    if args.focus is None:
        point = focus_points[args.focus_target]
        point = point[0] if isinstance(point[0], list) else point
        args.focus = probe.depth_of(point)
    lens = Lens(view["origin"], view["target"], args.lens, args.sensor_height, args.f_number, args.focus)

    scene_dict["integrator"] = {"type": "path", "max_depth": args.max_depth, "rr_depth": 6}
    scene_dict["sensor"] = lens.sensor(args.width, args.height, 1, thin_lens=False)
    scene = mi.load_dict(scene_dict)
    shapes = len(scene.shapes())
    tris = sum(int(s.face_count()) for s in scene.shapes() if hasattr(s, "face_count"))
    print(f"variant {variant} | {shapes} shapes, {tris:,} triangles | load {time.time() - t0:.1f}s")
    print(f"lens {args.lens:.0f} mm f/{args.f_number:g} | focus {lens.focus_m:.3f} m ({args.focus_target}) | "
          f"CoC at infinity {lens.metadata(args.width, args.height)['coc_diameter_px_at_infinity']:.1f} px")

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to mix passes with an existing render; choose a fresh --out directory")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"variant": variant, "mitsuba": mi.__version__, "resolution": [args.width, args.height],
            "view": args.view, "camera": lens.metadata(args.width, args.height),
            "integrator": {"type": "path", "max_depth": args.max_depth, "rr_depth": 6, "hide_emitters": False},
            "qualified_reference": False, "denoising": False, "seed": args.seed,
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
                                            lens, args.seed + 2)
        np.save(out / "depth.npy", depth)
        mi.Bitmap(np.where(np.isfinite(depth), depth, 0.0)[..., None]).write(str(out / "depth.exr"))
        finite = depth[np.isfinite(depth)]
        near, far = float(finite.min()), float(np.percentile(finite, 99.5))
        vis = 1.0 - np.clip((np.log(np.where(np.isfinite(depth), depth, far)) - math.log(near))
                            / (math.log(far) - math.log(near)), 0, 1)
        save_png(out / "depth.png", np.repeat(vis[..., None], 3, -1))
        gb = np.concatenate([layers["pos"], layers["nn"], layers["alb"], layers["idx"][..., :1]], -1)
        mi.Bitmap(gb, channel_names=["P.X", "P.Y", "P.Z", "N.X", "N.Y", "N.Z", "albedo.R", "albedo.G",
                                     "albedo.B", "shape_index.Y"]).write(str(out / "gbuffer.exr"))
        save_png(out / "normal.png", 0.5 + 0.5 * layers["nn"])
        coc = np.vectorize(lambda d: lens.coc_pixels(d, args.height) if np.isfinite(d) else
                           lens.metadata(args.width, args.height)["coc_diameter_px_at_infinity"])
        meta["depth_range_m"] = [near, float(finite.max())]
        meta["sky_fraction"] = float(1.0 - finite.size / depth.size)
        meta["coc_px_percentiles"] = {str(p): float(v) for p, v in zip(
            (5, 50, 95), np.percentile(coc(depth[::8, ::8]), (5, 50, 95)))}
        meta["passes"]["gbuffer"] = {"spp": 1, "seconds": round(sec, 2)}

    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}  (total {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
