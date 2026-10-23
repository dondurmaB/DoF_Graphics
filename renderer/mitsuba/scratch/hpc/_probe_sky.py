import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import numpy as np

def shot(sky, target, up=(0, 1, 0), fov=40):
    scene = mi.load_dict({"type": "scene", "integrator": {"type": "path", "max_depth": 2}, "sky": dict(sky),
        "sensor": {"type": "perspective", "fov": fov,
                   "to_world": mi.ScalarTransform4f().look_at(origin=[0, 0, 0], target=list(target), up=list(up)),
                   "film": {"type": "hdrfilm", "width": 48, "height": 48}, "sampler": {"type": "independent", "sample_count": 16}}})
    img = np.array(mi.render(scene))
    return img.mean(), img.max(), (img.sum(-1) == 0).mean()

ZUP_TO_YUP = mi.ScalarTransform4f().rotate(axis=[1, 0, 0], angle=-90)
sun = np.array([0.6, 0.3, -0.74]); sun /= np.linalg.norm(sun)   # sun toward -z: the side that was black
R = np.array(ZUP_TO_YUP.matrix)[:3, :3]
for name, extra in (("no to_world", {}), ("to_world z->y", {"to_world": ZUP_TO_YUP})):
    sky = {"type": "sunsky", "sun_direction": sun.tolist(), **extra}
    print(f"--- {name}")
    for label, tgt, up in (("horizon -z", (0, 0.15, -1), (0, 1, 0)), ("horizon +z", (0, 0.15, 1), (0, 1, 0)),
                           ("horizon +x", (1, 0.15, 0), (0, 1, 0)), ("zenith", (0, 1, 0), (0, 0, -1)),
                           ("nadir", (0, -1, 0), (0, 0, -1)), ("sun as given", tuple(sun), (0, 1, 0)),
                           ("sun rotated R*d", tuple(R @ sun), (0, 1, 0)), ("sun R^T*d", tuple(R.T @ sun), (0, 1, 0))):
        m, mx, z = shot(sky, tgt, up)
        print(f"  {label:16s} mean {m:8.4g}  max {mx:9.4g}  zero-frac {z:.2f}")
