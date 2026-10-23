"""Scratch: frozen render_beauty / render_gbuffer vs mi.render, same lens, same filter."""
import sys
import time

import numpy as np

import render as R
import scene_api as S

R.pick_variant("auto")
import mitsuba as mi  # noqa: E402


def slow_beauty(scene, lens, w, h, thin, spp, chunk, seed):
    sensor = mi.load_dict(lens.sensor(w, h, spp, thin))
    acc = np.zeros((h, w, 3))
    for k in range(spp // chunk):
        acc += np.asarray(mi.render(scene, sensor=sensor, spp=chunk, seed=seed * 1000 + k), np.float64)[..., :3]
    return acc / (spp // chunk)


def slow_gbuffer(scene, lens, w, h, seed):
    integ = mi.load_dict({"type": "aov", "aovs": "pos:position,nn:sh_normal,alb:albedo,idx:shape_index,t:depth"})
    img = np.asarray(mi.render(scene, sensor=mi.load_dict(lens.sensor(w, h, 1, False, "box")), integrator=integ,
                               spp=1, seed=seed), np.float32)
    return img, integ.aov_names()


def compare(tag, A, B):
    a, b = (x.mean(-1) for x in A), (x.mean(-1) for x in B)
    a1, a2 = A[0].mean(-1), A[1].mean(-1)
    b1, b2 = B[0].mean(-1), B[1].mean(-1)
    noise_a = np.std(a1 - a2) / np.sqrt(2)
    noise_b = np.std(b1 - b2) / np.sqrt(2)
    diff = (a1 + a2) / 2 - (b1 + b2) / 2
    ratio = np.std(diff) / np.sqrt((noise_a ** 2 + noise_b ** 2) / 2)
    edge = np.zeros(diff.shape, bool)
    edge[:2], edge[-2:], edge[:, :2], edge[:, -2:] = True, True, True, True
    nedge = np.sqrt(((a1 - a2)[edge].var() + (b1 - b2)[edge].var()) / 4)
    ratio_edge = np.std(diff[edge]) / nedge
    # Blur mismatch shows up as a structured difference: compare against a 1-px shifted copy as a scale.
    print(f"{tag}: mean fast {np.mean(A[0]):.5f} slow {np.mean(B[0]):.5f} | noise fast {noise_a:.5f} slow {noise_b:.5f}"
          f" | diff/noise all {ratio:.3f} edges {ratio_edge:.3f}", flush=True)


def run_scene(name, scene, lens, w, h, spp, chunk):
    for thin in (False, True):
        t0 = time.time()
        A = [R.render_beauty(scene, lens, w, h, thin, spp, chunk, s, "fast")[0] for s in (1, 2)]
        tf = time.time() - t0
        t0 = time.time()
        B = [slow_beauty(scene, lens, w, h, thin, spp, chunk, s) for s in (3, 4)]
        ts = time.time() - t0
        compare(f"{name} {'dof' if thin else 'pinhole'} {w}x{h} {spp}spp (fast {tf:.1f}s, slow {ts:.1f}s)", A, B)
    t0 = time.time()
    d1, L1, _ = R.render_gbuffer(scene, lens, w, h, 5)
    tf = time.time() - t0
    t0 = time.time()
    img, names = slow_gbuffer(scene, lens, w, h, 6)
    ts = time.time() - t0
    print(f"  slow aov film channels {img.shape[-1]}, aov_names {len(names)}: {names[:4]}...", flush=True)
    pos = img[..., [i for i, n in enumerate(names) if n.startswith("pos.")]]
    t = img[..., [i for i, n in enumerate(names) if n.startswith("t.")]][..., 0]
    d2 = np.where(t > 0, np.einsum("hwc,c->hw", pos - lens.origin.astype(np.float32),
                                   lens.forward.astype(np.float32)), np.inf)
    both = np.isfinite(d1) & np.isfinite(d2)
    rel = np.abs(d1[both] - d2[both]) / d2[both]
    print(f"  gbuffer (fast {tf:.2f}s, slow {ts:.2f}s): sky agree {np.mean(np.isfinite(d1) == np.isfinite(d2)):.4f}, "
          f"depth rel diff median {np.median(rel):.4f} p90 {np.percentile(rel, 90):.4f}; "
          f"normals mean dot {np.mean(np.sum(L1['nn'][both] * img[..., [i for i, n in enumerate(names) if n.startswith('nn.')]][both], -1)):.4f}",
          flush=True)
    t0 = time.time()
    for k in range(3):
        R.render_beauty(scene, lens, w, h, True, chunk, chunk, 50 + k, "t")
    print(f"  per call: fast {(time.time() - t0) / 3:.3f}s", flush=True)


# Cornell box
d = mi.cornell_box()
d.pop("sensor")
cb = mi.load_dict(d)
lens = R.Lens([0, 0, 3.9], [0, 0, 0], 50, 24, 1.4, 3.0)
run_scene("cornell", cb, lens, 256, 256, 2048, 256)

for name in sys.argv[1:]:
    sd = S.load_scene(name)
    b = S.build_checked(sd, S.make_context(sd, R.HERE / "generated", None, "clear"))
    view = sd.views[sd.default_view]
    target = view.focus or next(iter(b.focus_points))
    probe = R.Lens(view.origin, view.target, 50, 24, 2.0, 1.0, view.roll_deg, sd.far_clip)
    lens = R.Lens(view.origin, view.target, 50, 24, 2.0, probe.depth_of(b.focus_points[target][0]),
                  view.roll_deg, sd.far_clip)
    dd = dict(b.scene)
    dd["integrator"] = {"type": "path", "max_depth": sd.max_depth, "rr_depth": sd.rr_depth}
    dd["sensor"] = lens.sensor(64, 36, 1, False)
    run_scene(name, mi.load_dict(dd), lens, 640, 360, 1024, 256)
