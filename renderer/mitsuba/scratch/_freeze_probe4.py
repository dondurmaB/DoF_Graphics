"""Scratch: which area emitters of a scene break lens_trace recording."""
import sys
from pathlib import Path
import render as R
import scene_api as S
import lens_trace as LT

R.pick_variant("auto")
import mitsuba as mi


def ok(d, lens):
    LT._FROZEN.clear()
    try:
        sc = mi.load_dict(d)
        import drjit as dr
        for sh in sc.shapes():
            if sh.is_emitter():
                ps, _ = sh.emitter().sample_position(0.0, mi.Point2f(0.5, 0.5))
                si = dr.zeros(mi.SurfaceInteraction3f); si.p = ps.p; si.n = ps.n; si.uv = ps.uv
                ds, w = sh.emitter().sample_direction(si, mi.Point2f(0.5, 0.5))
                dr.eval(ps.p, ds.d, w)
        for s in range(3):
            LT.trace(sc, lens, 32, 18, 1, s)
        return "OK"
    except Exception as e:
        c = e
        while c.__cause__ is not None:
            c = c.__cause__
        return "FAIL " + str(c).splitlines()[0][:160]


def short(v):
    if isinstance(v, dict):
        return {k: short(x) for k, x in v.items() if k not in ("to_world",)}
    return v if isinstance(v, (int, float, str, bool)) or v is None else type(v).__name__


for name in sys.argv[1:]:
    sd = S.load_scene(name)
    b = S.build_checked(sd, S.make_context(sd, R.HERE / "generated", None, "clear"))
    import viewer as V
    lens = V.ViewerState(sd.default_view, sd, b.focus_points).lens()
    base = dict(b.scene)
    base["integrator"] = {"type": "path", "max_depth": 6}
    area = [k for k, v in base.items() if isinstance(v, dict) and "emitter" in v]
    noarea = {k: v for k, v in base.items() if k not in area}
    print(name, "full:", ok(base, lens), "| no area:", ok(noarea, lens), flush=True)
    seen = set()
    for k in area:
        sig = repr(short({kk: vv for kk, vv in base[k].items() if kk != "filename"}))
        if sig in seen:
            continue
        seen.add(sig)
        print(f"  {k}: {ok({**noarea, k: base[k]}, lens)}  {sig[:400]}", flush=True)
