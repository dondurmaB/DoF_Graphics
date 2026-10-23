"""Scratch: minimal reproduction of the area-emitter recording failure."""
import sys
import numpy as np
import render as R
import lens_trace as LT

R.pick_variant("auto")
import mitsuba as mi
import drjit as dr

lens = R.Lens([0, 0, 3.9], [0, 0, 0], 50, 24, 4.0, 3.9)


def run(tag, extra):
    d = mi.cornell_box()
    d.pop("sensor")
    d.update(extra)
    LT._FROZEN.clear()
    try:
        sc = mi.load_dict(d)
        for s in range(3):
            LT.trace(sc, lens, 32, 32, 2, s)
        print(tag, "OK", flush=True)
    except Exception as e:
        c = e
        while c.__cause__ is not None:
            c = c.__cause__
        print(tag, "FAIL", str(c).splitlines()[:2], flush=True)


em = {"emitter": {"type": "area", "radiance": {"type": "rgb", "value": [5, 5, 5]}}}
run("stock", {})
run("rect emitter", {"l2": {"type": "rectangle", "to_world": mi.ScalarTransform4f().translate([0.5, 0.5, 0]).scale(0.1), **em}})
run("sphere emitter", {"l2": {"type": "sphere", "radius": 0.1, "center": [0.5, 0.5, 0], **em}})
run("cube mesh emitter", {"l2": {"type": "cube", "to_world": mi.ScalarTransform4f().translate([0.5, 0.5, 0]).scale(0.1), **em}})
run("obj mesh emitter", {"l2": {"type": "obj", "filename": sys.argv[1], "to_world": mi.ScalarTransform4f().scale(0.01), **em}} if len(sys.argv) > 1 else {})
run("two cube emitters", {f"l{i}": {"type": "cube", "to_world": mi.ScalarTransform4f().translate([0.3 * i - 0.5, 0.5, 0]).scale(0.1), **em} for i in range(2)})
run("cube emitter weight", {"l2": {"type": "cube", "to_world": mi.ScalarTransform4f().translate([0.5, 0.5, 0]).scale(0.1),
                                   "emitter": {**em["emitter"], "sampling_weight": 3.0}}})
