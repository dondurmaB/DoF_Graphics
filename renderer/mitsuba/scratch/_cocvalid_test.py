"""Scratch: coc_valid inside a frozen AOV kernel, tested apart from the modules the pilot uses."""
import sys
from pathlib import Path

import numpy as np

import render as R
import scene_api as S
import lens_trace as LT

R.pick_variant("auto")
import drjit as dr  # noqa: E402
import mitsuba as mi  # noqa: E402


def kernel(scene, integrator, seed, cam, w, h, channels):
    sampler = mi.load_dict({"type": "independent"})
    sampler.seed(seed, w * h)
    pix, _, ray = LT._rays(sampler, cam, w, h, 1)
    _, _, aovs = integrator.sample(scene, sampler, ray)
    si = scene.ray_intersect(ray)
    flags = dr.select(si.is_valid(), si.bsdf(ray).flags(), mi.UInt32(0))
    F = mi.BSDFFlags
    specular = mi.has_flag(flags, F.Delta) | mi.has_flag(flags, F.Glossy)
    diffuse = mi.has_flag(flags, F.Diffuse)
    invisible = mi.has_flag(flags, F.Null)
    valid = dr.select((specular & ~diffuse) | invisible, mi.Float(0.0), mi.Float(1.0))
    n = channels + 1
    out = dr.zeros(mi.Float, w * h * n)
    for ch, v in enumerate(list(aovs[:channels]) + [valid]):
        dr.scatter(out, v, pix * n + ch)
    return mi.TensorXf(out, shape=(h, w, n))


out = Path("output/dev/cocvalid")
for name in sys.argv[1:]:
    sd = S.load_scene(name)
    b = S.build_checked(sd, S.make_context(sd, R.HERE / "generated", None, "clear"))
    d = dict(b.scene)
    d["integrator"] = {"type": "path", "max_depth": 4}
    integ = mi.load_dict({"type": "aov", "aovs": "pos:position,nn:sh_normal,alb:albedo,idx:shape_index,t:depth"})
    C = len(integ.aov_names())
    fn = dr.freeze(lambda scene, integrator, seed, cam: kernel(scene, integrator, seed, cam, 640, 360, C))
    for vname, view in sd.views.items():
        lens = R.Lens(view.origin, view.target, 35, 24, 4.0, 3.0, view.roll_deg, sd.far_clip)
        d["sensor"] = lens.sensor(64, 36, 1, False)
        scene = mi.load_dict(d) if vname == next(iter(sd.views)) else scene
        LT._warm(scene)
        img = np.asarray(fn(scene, integ, mi.UInt32(1), LT._camera(lens, 640, 360, False)))
        valid = img[..., -1]
        alb = np.clip(img[..., 6:9], 0, 1) ** (1 / 2.2)
        vis = alb * 0.6 + 0.4 * np.where(valid[..., None] < 0.5, [1.0, 0.1, 0.1], [0.5, 0.5, 0.5])
        out.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray((np.clip(vis, 0, 1) * 255).astype(np.uint8)).save(out / f"{name}_{vname}.png")
        print(f"{name} {vname}: invalid fraction {1 - valid.mean():.4f}", flush=True)
