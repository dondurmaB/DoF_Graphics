"""Thin-lens path tracing that Dr.Jit can freeze.

mi.render retraces the whole path tracer on every call: ~0.5 s on a small scene, ~10 s on one with a few
hundred materials (workshop), with the GPU idle meanwhile. dr.freeze records the call once and replays it
(0.04 s), but Mitsuba 3.9.1's `thinlens` sensor cannot be recorded (a member is not traversable: "variable
was created before recording ... not specified as an input"), even on the stock Cornell box.

So the camera lives here instead: rays are generated from plain arrays (pose, field of view, focus, aperture)
passed in as frozen-function inputs, and Mitsuba's integrator only shades them. The model is the one
`thinlens` uses, planar focus at `focus_m` along the optical axis and a uniform disc aperture, so `aperture`
is the place where other aperture shapes plug in later (SCENE_CONTRACT.md section 4). Pixels are box-filtered:
each sample lands in the pixel it was drawn in.
"""

from __future__ import annotations

import math

import numpy as np

_FROZEN: dict = {}


def _kernel(scene, integrator, seed, cam, w: int, h: int, spp: int):
    import drjit as dr
    import mitsuba as mi

    def c(i):  # symbolic read: indexing would pull the value to Python, which a recording forbids
        return dr.gather(mi.Float, cam, mi.UInt32(i))

    o, r, u, f = (mi.Vector3f(c(3 * i), c(3 * i + 1), c(3 * i + 2)) for i in range(4))
    tan_x, tan_y, focus, radius, far = (c(i) for i in range(12, 17))

    n = w * h * spp
    sampler = mi.load_dict({"type": "independent"})
    sampler.seed(seed, n)
    pix = dr.arange(mi.UInt32, n) // spp
    px = (mi.Float(pix % w) + sampler.next_1d()) / w
    py = (mi.Float(pix // w) + sampler.next_1d()) / h
    # Direction with unit component along the axis, so `o + d * focus` lies on the focus plane.
    d = f + (2 * px - 1) * tan_x * r - (2 * py - 1) * tan_y * u
    disk = mi.warp.square_to_uniform_disk_concentric(sampler.next_2d())
    origin = o + radius * (disk.x * r + disk.y * u)
    direction = dr.normalize(o + d * focus - origin)
    ray = mi.RayDifferential3f(origin, direction)
    ray.maxt = far / dr.dot(direction, f)
    L, valid, _ = integrator.sample(scene, sampler, ray)
    L = dr.select(valid, L, 0.0)

    out = dr.zeros(mi.Float, w * h * 3)
    for c in range(3):
        dr.scatter_reduce(dr.ReduceOp.Add, out, L[c] / spp, pix * 3 + c)
    return mi.TensorXf(out, shape=(h, w, 3))


def trace(scene, lens, w: int, h: int, spp: int, seed: int, integrator=None, aperture: bool = True) -> np.ndarray:
    """Linear RGB (h, w, 3) of `scene` through `lens` (render.Lens); aperture=False gives the pinhole image.

    One frozen recording per (size, spp, integrator); every later call with new pose, lens or seed replays it.
    """
    import drjit as dr
    import mitsuba as mi

    integrator = integrator or scene.integrator()
    right = np.cross(lens.forward, lens.up)
    right /= np.linalg.norm(right)
    up = np.cross(right, lens.forward)
    tan_y = math.tan(math.radians(lens.fov_y) / 2)
    cam = np.concatenate([lens.origin, right, up, lens.forward,
                          [tan_y * w / h, tan_y, lens.focus_m, lens.aperture_radius if aperture else 0.0,
                           lens.far_clip]]).astype(np.float32)
    key = (w, h, spp, id(integrator))
    fn = _FROZEN.get(key)
    if fn is None:
        fn = _FROZEN[key] = dr.freeze(lambda scene, integrator, seed, cam: _kernel(scene, integrator, seed, cam,
                                                                                    w, h, spp))
    img = fn(scene, integrator, mi.UInt32(seed % 2 ** 32), mi.Float(cam))
    return np.asarray(img, dtype=np.float32)
