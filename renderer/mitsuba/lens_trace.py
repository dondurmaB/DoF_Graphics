"""Thin-lens path tracing that Dr.Jit can freeze.

mi.render retraces the whole path tracer on every call: ~0.5 s on a small scene, ~10 s on one with a few
hundred materials (workshop), with the GPU idle meanwhile. dr.freeze records the call once and replays it
(0.04 s), but Mitsuba 3.9.1's `thinlens` sensor cannot be recorded (a member is not traversable: "variable
was created before recording ... not specified as an input"), even on the stock Cornell box.

So the camera lives here instead: rays are generated from plain arrays (pose, field of view, focus, aperture)
passed in as frozen-function inputs, and Mitsuba's integrator only shades them. The model is the one
`thinlens` uses, planar focus at `focus_m` along the optical axis and a uniform disc aperture, so `aperture`
is the place where other aperture shapes plug in later (SCENE_CONTRACT.md section 4).

Pixel filter: "gaussian" (the default, what the dataset uses) splats every sample into the pixels around it
with Mitsuba's own ImageBlock and gaussian filter, exactly as mi.render does with the sensor's default film,
so the images match it in blur and in how noise spreads between neighbours. "box" puts each sample only in
the pixel it was drawn in: about half a pixel sharper (see ai/DATA_PROTOCOL.md) and slightly faster.
`trace_aovs` gives one jittered sample per pixel of an AOV integrator, for the G-buffer.
"""

from __future__ import annotations

import math

import numpy as np

_FROZEN: dict = {}
_WARMED: set = set()
_FILTERS: dict = {}


def _warm(scene) -> None:
    """Sample every area emitter once outside the recording.

    A bitmap-textured emitter (desk's monitor, museum's exit sign) builds its sampling distribution lazily on
    first use; built inside a recording, that copy reads memory the recording does not know ("Tried to record
    a memcpy_async operation, but the source pointer ... is unknown").
    """
    import drjit as dr
    import mitsuba as mi

    if id(scene) in _WARMED:
        return
    for shape in scene.shapes():
        if shape.is_emitter():
            em = shape.emitter()
            ps, _ = em.sample_position(0.0, mi.Point2f(0.5, 0.5))
            si = dr.zeros(mi.SurfaceInteraction3f)
            si.p, si.n, si.uv = ps.p, ps.n, ps.uv
            ds, weight = em.sample_direction(si, mi.Point2f(0.5, 0.5))
            dr.eval(ps.p, ds.d, weight)
    _WARMED.add(id(scene))


def _rays(sampler, cam, w: int, h: int, spp: int):
    """Thin-lens rays for spp samples per pixel; returns (pixel index, film position in pixels, ray)."""
    import drjit as dr
    import mitsuba as mi

    def c(i):  # symbolic read: indexing would pull the value to Python, which a recording forbids
        return dr.gather(mi.Float, cam, mi.UInt32(i))

    o, r, u, f = (mi.Vector3f(c(3 * i), c(3 * i + 1), c(3 * i + 2)) for i in range(4))
    tan_x, tan_y, focus, radius, far = (c(i) for i in range(12, 17))

    pix = dr.arange(mi.UInt32, w * h * spp) // spp
    pos = mi.Point2f(mi.Float(pix % w), mi.Float(pix // w)) + sampler.next_2d()
    # Direction with unit component along the axis, so `o + d * focus` lies on the focus plane.
    d = f + (2 * pos.x / w - 1) * tan_x * r - (2 * pos.y / h - 1) * tan_y * u
    disk = mi.warp.square_to_uniform_disk_concentric(sampler.next_2d())
    origin = o + radius * (disk.x * r + disk.y * u)
    direction = dr.normalize(o + d * focus - origin)
    ray = mi.RayDifferential3f(origin, direction)
    ray.maxt = far / dr.dot(direction, f)
    return pix, pos, ray


def _kernel(scene, integrator, seed, cam, w: int, h: int, spp: int, rfilter: str):
    """Returns (h, w, 4): filter-weighted sums of R, G, B and of the weights themselves."""
    import drjit as dr
    import mitsuba as mi

    sampler = mi.load_dict({"type": "independent"})
    sampler.seed(seed, w * h * spp)
    pix, pos, ray = _rays(sampler, cam, w, h, spp)
    L, valid, _ = integrator.sample(scene, sampler, ray)
    L = dr.select(valid, L, 0.0)
    if rfilter == "box":
        out = dr.zeros(mi.Float, w * h * 4)
        for ch, v in enumerate((L[0], L[1], L[2], mi.Float(1.0))):
            dr.scatter_reduce(dr.ReduceOp.Add, out, v, pix * 4 + ch)
        return mi.TensorXf(out, shape=(h, w, 4))
    # As mi.render: the film's block has no border, normalised splats, and a weight channel to divide by.
    block = mi.ImageBlock(mi.ScalarVector2u(w, h), mi.ScalarPoint2i(0, 0), 4, _FILTERS[rfilter],
                          border=False, normalize=True)
    block.put(pos, [L[0], L[1], L[2], mi.Float(1.0)])
    return block.tensor()


def _aov_kernel(scene, integrator, seed, cam, w: int, h: int, channels: int):
    """Returns (h, w, channels): the AOVs of one jittered sample per pixel."""
    import drjit as dr
    import mitsuba as mi

    sampler = mi.load_dict({"type": "independent"})
    sampler.seed(seed, w * h)
    pix, _, ray = _rays(sampler, cam, w, h, 1)
    _, _, aovs = integrator.sample(scene, sampler, ray)
    out = dr.zeros(mi.Float, w * h * channels)
    for ch in range(channels):
        dr.scatter(out, aovs[ch], pix * channels + ch)
    return mi.TensorXf(out, shape=(h, w, channels))


def _camera(lens, w: int, h: int, aperture: bool):
    import mitsuba as mi

    right = np.cross(lens.forward, lens.up)
    right /= np.linalg.norm(right)
    up = np.cross(right, lens.forward)
    tan_y = math.tan(math.radians(lens.fov_y) / 2)
    cam = np.concatenate([lens.origin, right, up, lens.forward,
                          [tan_y * w / h, tan_y, lens.focus_m, lens.aperture_radius if aperture else 0.0,
                           lens.far_clip]]).astype(np.float32)
    return mi.Float(cam)


def trace_sums(scene, lens, w: int, h: int, spp: int, seed: int, integrator=None, aperture: bool = True,
               rfilter: str = "gaussian") -> np.ndarray:
    """(h, w, 4) float64 filter-weighted sums (RGB, weight). Add them over calls, then divide RGB by weight:
    that equals one mi.render call with all the samples."""
    import drjit as dr
    import mitsuba as mi

    integrator = integrator or scene.integrator()
    _warm(scene)
    if rfilter != "box" and rfilter not in _FILTERS:
        _FILTERS[rfilter] = mi.load_dict({"type": rfilter})
    key = ("rgb", w, h, spp, id(integrator), rfilter)
    fn = _FROZEN.get(key)
    if fn is None:
        fn = _FROZEN[key] = dr.freeze(lambda scene, integrator, seed, cam: _kernel(scene, integrator, seed, cam,
                                                                                    w, h, spp, rfilter))
    img = fn(scene, integrator, mi.UInt32(seed % 2 ** 32), _camera(lens, w, h, aperture))
    return np.asarray(img, dtype=np.float64)


def trace(scene, lens, w: int, h: int, spp: int, seed: int, integrator=None, aperture: bool = True,
          rfilter: str = "gaussian") -> np.ndarray:
    """Linear RGB (h, w, 3) of `scene` through `lens` (render.Lens); aperture=False gives the pinhole image.

    One frozen recording per (size, spp, integrator, filter); every later call with new pose, lens or seed
    replays it.
    """
    s = trace_sums(scene, lens, w, h, spp, seed, integrator, aperture, rfilter)
    return (s[..., :3] / np.maximum(s[..., 3:], 1e-12)).astype(np.float32)


def trace_aovs(scene, lens, w: int, h: int, seed: int, integrator) -> np.ndarray:
    """(h, w, C) AOVs of an `aov` integrator, one jittered pinhole sample per pixel, in aov_names() order."""
    import drjit as dr
    import mitsuba as mi

    _warm(scene)
    channels = len(integrator.aov_names())
    key = ("aov", w, h, id(integrator))
    fn = _FROZEN.get(key)
    if fn is None:
        fn = _FROZEN[key] = dr.freeze(lambda scene, integrator, seed, cam: _aov_kernel(scene, integrator, seed,
                                                                                        cam, w, h, channels))
    img = fn(scene, integrator, mi.UInt32(seed % 2 ** 32), _camera(lens, w, h, False))
    return np.asarray(img, dtype=np.float32)
