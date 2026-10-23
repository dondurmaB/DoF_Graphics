"""Scratch: which scenes can lens_trace record, and which scene parts break the recording."""
import sys
import traceback
from pathlib import Path

import render as R
import scene_api as S
import lens_trace as LT


def attempt(d, lens, tag):
    import mitsuba as mi
    LT._FROZEN.clear()
    try:
        sc = mi.load_dict(d)
        if WARM:
            import drjit as dr
            for sh in sc.shapes():
                if sh.is_emitter():
                    dr.eval(sh.sample_position(0.0, mi.Point2f(0.5, 0.5)).p)
        for s in range(3):
            LT.trace(sc, lens, 64, 36, 2, s)
        print(f"  {tag}: OK", flush=True)
        return True
    except Exception as e:
        msg = str(e).splitlines()
        print(f"  {tag}: FAIL {[l for l in msg if 'record' in l][:1] or msg[:1]}", flush=True)
        return False


WARM = False


def main():
    R.pick_variant("auto")
    for name in sys.argv[1:] or S.list_scenes():
        sd = S.load_scene(name)
        ctx = S.make_context(sd, Path(R.HERE / "generated") if hasattr(R, "HERE") else Path("generated"), None, "clear")
        b = S.build_checked(sd, ctx)
        import viewer as V
        lens = V.ViewerState(sd.default_view, sd, b.focus_points).lens()
        base = dict(b.scene)
        base["integrator"] = {"type": "path", "max_depth": 6}
        print(name, flush=True)
        attempt(base, lens, "full")
        continue
        envs = [k for k, v in base.items() if isinstance(v, dict) and v.get("type") in ("envmap", "constant", "sunsky", "directional", "point", "spot")]
        area = [k for k, v in base.items() if isinstance(v, dict) and "emitter" in v]
        print(f"  {len(envs)} non-shape emitters {[base[k]['type'] for k in envs]}, {len(area)} area-emitter shapes", flush=True)
        attempt({k: v for k, v in base.items() if k not in envs}, lens, "no env/point emitters")
        attempt({k: v for k, v in base.items() if k not in area}, lens, "no area emitters")
        nw = {k: ({**v, "emitter": {kk: vv for kk, vv in v["emitter"].items() if kk != "sampling_weight"}} if k in area else
                  {kk: vv for kk, vv in v.items() if kk != "sampling_weight"} if k in envs else v) for k, v in base.items()}
        attempt(nw, lens, "no sampling_weight")
        attempt({**base, "integrator": {"type": "direct"}}, lens, "direct integrator")


if __name__ == "__main__":
    main()
