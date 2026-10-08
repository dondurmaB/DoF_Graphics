"""A rooftop terrace bar at golden hour, 42 m above a city: the FAR-VISTA case.

Architecture in the near and mid field (decking, cable railing, olive trees, a pergola lounge, a bar,
parasols, festoon lights, a stair hut and HVAC) against an open city skyline, a river, and hills
3.4-10 km away, so the far-distance blur saturates while the foreground stays man-made.

Layout, meters, y up, the default camera looks toward -z (the vista):
  terrace      deck at y = 42 on our building, x in [-10, 10], z in [-8, 8]; cable railing on all edges
  front        bistro tables and two parasols along the railing (z ~ -6.3)
  left / back  corten troughs of lavender and grass along x = -9.4; a pergola lounge (sofa, armchairs) at
               x in [-8.8, -2.4], z in [1, 6.8]; olive trees in square planters
  right        a bar (counter at x ~ 5.7, back bar against the railing, stools); the stair hut with a timber
               water tank on its roof at the back-right corner; HVAC units behind a slatted screen
  city         a 110 m block grid (22 m streets) to ~3.3 km: mid-rise near, a high-rise cluster ~1 km out with
               a 310 m spired tower, a river with parks and bridges at z ~ -1380, a stepped stone tower
               beyond it, then hills from 3.4 to 10 km

AERIAL PERSPECTIVE without a participating medium (haze belongs to the shared side and the `path`
integrator ignores camera media): far geometry gets L = T(d) L_surface + (1 - T(d)) L_horizon with
T(d) = exp(-d / 3.5 km). The city is binned into distance tiers; each tier's materials are a
`blendbsdf` of the real BSDF (weight T) and black (weight 1 - T), and the shapes carry a constant
area emission (1 - T) L_horizon with a negligible light-sampling weight. The far ground and the hills
use the same idea with continuous gradient textures along a log-distance uv.

Environments: `clear` is a low golden-hour sun (10.5 deg) behind the camera's left, long shadows and warm
facades; `overcast` has no sun, a flat grey sky and the festoon lights reading brighter.
The seed picks the city (lots, heights, facade types, roof clutter, street trees, cars), the furniture
jitter, which festoon strands are lit, and every plant.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import env_kit
import web_assets as W
import procedural as G
import props as PR
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, bitmap, luminance, principled, rgb
from scenes import _rooftop_props as R

ASSET_VERSION = 8
TY = 42.0                                     # deck level
BX, BZ = 10.0, 8.0                            # half extents of our building / terrace
SUN = env_kit.default_sun_direction(10.5, -40.0)
GOLDEN_TURBIDITY = 5.0
SKY_SCALE = {"clear": 2.5, "overcast": 2.6}
# horizon radiance (per unit sky scale) that the far city fades toward
HAZE_L = {"clear": (0.105, 0.100, 0.105), "overcast": (env_kit.OVERCAST_RADIANCE,) * 3}
BULB = (26.0, 15.0, 5.5)
WINDOW_L = 0.55
SHOP_L = 0.45                                 # radiance of lit shop interiors and signs                               # radiance of a lit window at level 1 (warm interior light)
FLAME = (9.0, 4.2, 1.1)
CAR_COLOURS = [((0.62, 0.62, 0.60), 0.0), ((0.015, 0.015, 0.017), 0.0), ((0.30, 0.31, 0.32), 0.6),
               ((0.42, 0.03, 0.02), 0.0), ((0.03, 0.06, 0.20), 0.0), ((0.12, 0.12, 0.12), 0.3)]
BOTTLE_COLOURS = [(0.02, 0.07, 0.03), (0.13, 0.05, 0.01), (0.22, 0.24, 0.24), (0.01, 0.01, 0.01), (0.02, 0.04, 0.11)]

TABLES = [(-7.3, -6.25), (-3.6, -6.25), (0.1, -6.25), (3.7, -6.25)]
UMBRELLAS = [(-5.45, -5.75), (1.9, -5.75)]
OLIVES = [(-6.5, -2.4), (-1.3, 2.7), (6.9, -4.7)]
PERGOLA = (-8.8, -2.4, 1.0, 6.8)               # x0, x1, z0, z1
BAR_X, BAR_Z = (5.45, 6.05), (-2.6, 3.0)
HUT = (5.0, 9.6, 4.6, 7.85)
HVAC = (0.5, 4.8, 5.4, 7.7)
POLE_TOP = TY + 3.3
ANCHORS = {"fl": (-9.65, POLE_TOP, -7.65), "fm": (-0.2, POLE_TOP, -7.65), "fr": (9.65, POLE_TOP, -7.65),
           "ml": (-9.65, POLE_TOP, 0.2), "br": (9.65, POLE_TOP, -2.9), "mast": (5.35, TY + 4.3, 7.45)}
STRANDS = [("fl", "mast", 0.75), ("fr", "ml", 0.7), ("fm", "mast", 0.6), ("br", "ml", 0.7), ("fl", "fm", 0.35),
           ("fm", "fr", 0.35)]


def hazed(spec: dict, t: int) -> dict:
    T = 1.0 if t == 0 else float(R.transmittance(R.tier_distance(t)))
    if T > 0.999:
        return spec
    return {"type": "blendbsdf", "weight": 1.0 - T, "bsdf_0": spec,
            "bsdf_1": {"type": "diffuse", "reflectance": rgb(0.0, 0.0, 0.0)}}


def haze_emission(env: str, t: int):
    if t == 0:
        return None
    k = 1.0 - float(R.transmittance(R.tier_distance(t)))
    s = SKY_SCALE[env]
    return tuple(k * s * c for c in HAZE_L[env])


def thin(base, roughness, diff_trans, **kw) -> dict:
    return {"type": "principledthin", "base_color": base if isinstance(base, dict) else rgb(*base),
            "roughness": roughness, "diff_trans": diff_trans, **kw}


def albedo_tex(A: Assets, name: str, fn, uv_scale=None) -> dict:
    return bitmap(A.texture(name, lambda: R.srgb(fn())), uv_scale=uv_scale)


def data_tex(A: Assets, name: str, fn, uv_scale=None) -> dict:
    return bitmap(A.texture(name, fn, gray=True), raw=True, uv_scale=uv_scale)


def exr(A: Assets, name: str, fn) -> str:
    path = A.root / "textures" / f"{name}.exr"
    if A.rebuild or not path.exists():
        import mitsuba as mi

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{name}.tmp.exr")
        mi.Bitmap(np.ascontiguousarray(fn(), np.float32)).write(str(tmp))
        tmp.replace(path)
    return str(path)


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------
def facade_bsdf(As: Assets, kind: str) -> dict:
    spec = R.FACADES[kind]
    tid, tint, size = spec["tex"]
    wall = W.surface(tid, tint=tint, uv_scale=(1 / size[0], 1 / size[1]) if size else None)
    if spec["layout"] is None:
        return wall
    alb, rough, mask, _, (tw, th) = facade_maps(kind)
    sc = (1 / tw, 1 / th)
    win = principled(bitmap(As.texture(f"facade_{kind}", lambda: R.srgb(alb)), uv_scale=sc),
                     bitmap(As.texture(f"facade_{kind}_rough", lambda: rough, gray=True), raw=True, uv_scale=sc), specular=0.5)
    return {"type": "blendbsdf", "bsdf_0": wall, "bsdf_1": win,
            "weight": bitmap(As.texture(f"facade_{kind}_mask", lambda: mask, gray=True), raw=True, uv_scale=sc)}


def base_color_path(spec: dict) -> str:
    return (spec["bsdf"] if spec["type"] == "normalmap" else spec)["base_color"]["filename"]


def add_materials(b: SceneBuilder, As: Assets, Ad: Assets, env: str) -> None:
    for kind in R.FACADE_KINDS:
        base = facade_bsdf(As, kind)
        for t in range(5):
            b.material(f"walls_{kind}_t{t}", hazed(base, t))
    alb, rough, _, (tw, th) = shop_maps()
    sc = (1 / tw, 1 / th)
    shop = principled(bitmap(As.texture("shop", lambda: R.srgb(alb)), uv_scale=sc),
                      bitmap(As.texture("shop_rough", lambda: rough, gray=True), raw=True, uv_scale=sc), specular=0.5)
    roofs = {"roof": W.surface("bitumen", tint=(1.25, 1.22, 1.18)), "roofclay": W.surface("clay_roof_tiles_02"),
             "roofmetal": W.surface("box_profile_metal_sheet", tint=(0.55, 0.62, 0.66)),
             "parapet": W.surface("brushed_concrete"),
             "spire": principled((0.50, 0.51, 0.52), 0.35, metallic=0.5)}
    for t in range(5):
        b.material(f"shop_t{t}", hazed(shop, t))
        for name, spec in roofs.items():
            b.material(f"{name}_t{t}", hazed(spec, t))
    for i, c in enumerate([(0.05, 0.16, 0.10), (0.40, 0.05, 0.04), (0.04, 0.06, 0.16), (0.62, 0.58, 0.50)]):
        b.material(f"awning_{i}", thin(c, 0.8, 0.15))
    t_park = R.tier_of(1380.0)
    b.material("park", hazed(principled((0.06, 0.12, 0.035), 0.9), t_park))
    b.material("park_tree", hazed(principled((0.04, 0.085, 0.025), 0.85), t_park))
    b.material("water", hazed(principled((0.012, 0.025, 0.03), 0.04, specular=0.6), t_park))
    b.material("bridge", hazed(principled((0.45, 0.43, 0.40), 0.8), t_park))
    b.material("tank", W.surface("dark_planks", uv_scale=(5.0, 3.0), tint=(1.1, 0.95, 0.85)))
    b.material("steel", W.surface("rusty_metal_03", uv_scale=(2.0, 2.0)))
    b.material("canopy", principled((0.045, 0.10, 0.03), 0.85))
    b.material("trunk", principled((0.10, 0.08, 0.06), 0.9))
    for i, (c, met) in enumerate(CAR_COLOURS):
        b.material(f"car_{i}", principled(c, 0.3, metallic=met, clearcoat=0.6, clearcoat_gloss=0.8))
    b.material("car_glass", principled((0.01, 0.012, 0.014), 0.08, specular=0.5))
    b.material("tyre", principled((0.02, 0.02, 0.02), 0.8))
    pitch = (1 / R.STREET_PITCH, 1 / R.STREET_PITCH)
    b.material("street", {"type": "blendbsdf",
                          "bsdf_0": principled(albedo_tex(As, "street", R.street_tile, pitch), 0.85),
                          "bsdf_1": W.surface("asphalt_02"),
                          "weight": data_tex(As, "street_mask", R.street_mask, pitch)})
    # far ground and hills: continuous haze along the log-distance v coordinate
    weight = data_tex(As, "haze_weight", R.haze_weight)
    for name, fn in (("far_ground", R.far_ground_albedo), ("hills", R.hills_albedo)):
        b.material(name, {"type": "blendbsdf", "weight": weight,
                          "bsdf_0": principled(albedo_tex(As, name, fn), 0.9),
                          "bsdf_1": {"type": "diffuse", "reflectance": rgb(0.0, 0.0, 0.0)}})
    # terrace
    b.material("deck", W.surface("brown_planks_09", tint=(1.0, 0.98, 0.95)))
    b.material("pavers", W.surface("concrete_tiles"))
    b.material("upstand", W.surface("concrete_block_wall_02"))
    b.material("coping", W.surface("chipped_concrete"))
    b.material("hut", W.surface("concrete_block_wall"))
    b.material("hut_roof", W.surface("bitumen"))
    b.material("door", W.surface("painted_metal_shutter"))
    b.material("corten", W.surface("rusty_metal_02"))
    b.material("soil", W.surface("bark_brown_02"))
    b.material("olive_soil", W.surface("brown_mud_dry"))
    b.material("black_steel", principled((0.025, 0.025, 0.027), 0.55, metallic=0.6))
    b.material("cable", principled((0.62, 0.62, 0.64), 0.5, metallic=1.0))
    b.material("teak", principled((0.30, 0.16, 0.075), 0.6))
    b.material("teak_veneer", W.surface("teak_veneer"))
    b.material("backbar", W.surface("dark_wooden_planks"))
    b.material("kick", W.surface("metal_plate_02"))
    b.material("brass", W.surface("metal_plate", tint=(1.0, 0.78, 0.45)))
    b.material("pergola_wood", W.surface("weathered_brown_planks"))
    b.material("linen", W.surface("rough_linen", uv_scale=3.0))
    b.material("rug", W.surface("acg:Carpet012", uv_scale=(0.8, 0.8)))
    b.material("marble", W.surface("marble_01"))
    b.material("glass", {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0})
    b.material("wax", principled((0.75, 0.70, 0.60), 0.6))
    b.material("wick", principled((0.02, 0.02, 0.02), 0.9))
    hess = W.surface("hessian_380", uv_scale=(1 / 0.27, 1 / 0.27))
    cream = thin((0.80, 0.76, 0.66), 0.9, 0.3)
    b.material("canvas", {"type": "normalmap", "normalmap": hess["normalmap"], "bsdf": cream} if hess["type"] == "normalmap" else cream)
    b.material("umbrella_pole", W.surface("metal_plate", tint=(0.85, 0.84, 0.80)))
    b.material("umbrella_base", W.surface("brushed_concrete"))
    b.material("bulb", {"type": "diffuse", "reflectance": rgb(0.8, 0.78, 0.72)})
    b.material("leaf", thin(albedo_tex(As, "leaf_atlas", R.leaf_atlas), 0.55, 0.35))
    b.material("stem", principled((0.15, 0.17, 0.09), 0.7))
    b.material("lavender_flower", principled((0.22, 0.12, 0.40), 0.7))
    b.material("plume", thin((0.55, 0.48, 0.32), 0.9, 0.4))
    b.material("bark", W.surface("bark_willow_02", uv_scale=(1.0, 5.0), tint=(0.85, 0.85, 0.82)))


def ply_points(path: str) -> np.ndarray:
    """Vertex positions of a PLY written by procedural.Mesh.write_ply (8 float32 per vertex)."""
    with open(path, "rb") as fh:
        n = 0
        while True:
            line = fh.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        return np.frombuffer(fh.read(n * 32), "<f4").reshape(n, 8)[:, :3].astype(float)


def surface_y(m, matrix, x: float, z: float, r: float = 0.12) -> float:
    """Height of the highest surface of a placed web model within r of (x, z), e.g. a table top."""
    best = -np.inf
    for part in m.parts:
        p = ply_points(part.ply) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]
        sel = (np.abs(p[:, 0] - x) < r) & (np.abs(p[:, 2] - z) < r)
        if sel.any():
            best = max(best, float(p[sel, 1].max()))
    if not np.isfinite(best):                       # no vertex near the centre (e.g. a lid with only rim vertices)
        best = max(float((ply_points(part.ply) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3])[:, 1].max())
                   for part in m.parts)
    return best


def long_axis_yaw(m, along: str = "x") -> float:
    """Yaw that turns the model's longer footprint axis onto world `along`."""
    (x0, _, z0), (x1, _, z1) = m.bounds
    longer_x = (x1 - x0) >= (z1 - z0)
    return 0.0 if longer_x == (along == "x") else 90.0


_MAPS = {}
_SHOP = []


def shop_maps():
    if not _SHOP:
        _SHOP.append(R.shop_texture())
    return _SHOP[0]


def shop_emission(As: Assets, t: int, env: str):
    T = 1.0 if t == 0 else float(R.transmittance(R.tier_distance(t)))
    haze = np.zeros(3) if t == 0 else (1.0 - T) * SKY_SCALE[env] * np.asarray(HAZE_L[env])
    lit, (tw, th) = shop_maps()[2], shop_maps()[3]
    path = exr(As, f"shop_em_t{t}_{env}", lambda: haze[None, None, :] + T * SHOP_L * lit[::4, ::4])
    import mitsuba as mi

    return {"type": "bitmap", "filename": path, "raw": True, "filter_type": "bilinear",
            "to_uv": mi.ScalarTransform4f().scale([1 / tw, 1 / th, 1.0])}


def facade_maps(kind: str, ppm: float = 42.0):
    if (kind, ppm) not in _MAPS:
        _MAPS[(kind, ppm)] = R.facade_texture(kind, ppm)
    return _MAPS[(kind, ppm)]


def wall_emission(As: Assets, kind: str, t: int, env: str):
    """Lit windows (scaled by the haze transmittance, none from tier 3 on) plus the haze in-scatter, as one
    emission bitmap. Photographed facades derive their lit windows from their brightest warm pixels."""
    T = 1.0 if t == 0 else float(R.transmittance(R.tier_distance(t)))
    haze = np.zeros(3) if t == 0 else (1.0 - T) * SKY_SCALE[env] * np.asarray(HAZE_L[env])
    spec = R.FACADES[kind]
    lit_on = 0.0 if t >= 3 else 1.0
    import mitsuba as mi

    if spec["layout"] is None:
        if spec.get("lit") != "lum" or t >= 3:
            return tuple(haze) if t else None
        from PIL import Image

        size = spec["tex"][2]
        src = base_color_path(W.surface(spec["tex"][0]))

        def em():
            a = np.asarray(Image.open(src).convert("RGB").resize((256, 256)), np.float32) / 255.0
            lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            warm = (a[..., 0] > a[..., 2] * 1.15)
            k = np.clip((lum - 0.45) / 0.4, 0, 1) * warm
            return haze[None, None, :] + T * WINDOW_L * 1.4 * a * k[..., None]

        path = exr(As, f"facade_em_{kind}_t{t}_{env}", em)
        return {"type": "bitmap", "filename": path, "raw": True, "filter_type": "bilinear",
                "to_uv": mi.ScalarTransform4f().scale([1 / size[0], 1 / size[1], 1.0])}
    lit, (tw, th) = facade_maps(kind, 10.0)[3], facade_maps(kind, 10.0)[4]
    path = exr(As, f"facade_em_{kind}_t{t}_{env}", lambda: haze[None, None, :] + lit_on * T * WINDOW_L * lit)
    return {"type": "bitmap", "filename": path, "raw": True, "filter_type": "bilinear",
            "to_uv": mi.ScalarTransform4f().scale([1 / tw, 1 / th, 1.0])}


# --------------------------------------------------------------------------
# City
# --------------------------------------------------------------------------
NEIGHBOURS = [((12, 44, -44, -10), 27.0, "render_a", "podium"), ((12, 44, -8, 44), 33.0, "brick_b", "box"),
              ((-44, -12, -44, 4), 36.0, "render_b", "box"), ((-44, -12, 6, 44), 39.0, "concrete_a", "stepped"),
              ((-10, 10, 10, 44), 38.0, "stone", "box"), ((-10, 10, -44, -10), 14.0, "render_c", "prewar")]


def city_specs(seed: int) -> list:
    rng = np.random.default_rng([seed, 11])
    specs = R.city_layout(rng)
    for (bx, h, kind, profile) in NEIGHBOURS:
        cx, cz = (bx[0] + bx[1]) / 2, (bx[2] + bx[3]) / 2
        specs.append(R.massing(rng, {"box": bx, "h": h, "kind": kind, "d": math.hypot(cx, cz), "u0": rng.uniform(0, 20)},
                               profile))
    return specs


def add_city(b: SceneBuilder, As: Assets, Ad: Assets, seed: int, env: str, focus: dict) -> None:
    specs = city_specs(seed)
    parts = Ad.meshes("city", lambda: R.city_groups(np.random.default_rng([seed, 12]), specs).meshes())
    life = Ad.meshes("street", lambda: R.street_life(np.random.default_rng([seed, 12])).meshes())
    river = Ad.meshes("river", lambda: R.river_park(np.random.default_rng([seed, 13])).meshes())
    t_park = R.tier_of(1380.0)
    I = np.eye(4)
    for part in sorted(parts):
        tier = int(part.rsplit("_t", 1)[1]) if "_t" in part and part.rsplit("_t", 1)[1].isdigit() else None
        if part.startswith("walls_"):
            kind = part[len("walls_"):part.rindex("_t")]
            b.ply(parts[part], I, part, emission=wall_emission(As, kind, tier, env), prefix="city", power=1e-3)
        elif part.startswith("shop_t"):
            b.ply(parts[part], I, part, emission=shop_emission(As, tier, env), prefix="city", power=1e-3)
        elif tier is not None:
            b.ply(parts[part], I, part, emission=haze_emission(env, tier), prefix="city", power=1e-3)
        else:
            b.ply(parts[part], I, part, prefix="city")
    for part in sorted(life):
        b.ply(life[part], I, part, prefix="street")
    for part in sorted(river):
        b.ply(river[part], I, part, emission=haze_emission(env, t_park), prefix="river", power=1e-3)
    # scanned roof plant on flat roofs within 450 m (instanced); farther roofs keep procedural overruns
    models = {}
    for mid, (x, y, z), yaw in R.roof_clutter(specs, seed):
        m = models.setdefault(mid, W.model(mid))
        W.add(b, m, W.place(m, at=(x, y, z), yaw=yaw + long_axis_yaw(m, "x")), prefix=f"city_{mid}")
    # ground: near streets, far ground and hills with continuous haze
    near = As.mesh("near_ground", lambda: R.quad((-640, 0.03, -640), (640, 0.03, -640), (640, 0.03, 640), (-640, 0.03, 640),
                                                  (0, 1, 0), [(-585, -585), (695, -585), (695, 695), (-585, 695)]))
    b.ply(near, I, "street", prefix="ground")
    s = SKY_SCALE[env]
    em = {"type": "bitmap", "filename": exr(Ad, f"haze_em_{env}", lambda: R.haze_emission([s * c for c in HAZE_L[env]])),
          "raw": True, "filter_type": "bilinear"}
    b.ply(As.mesh("far_ground", R.far_ground_mesh), I, "far_ground", emission=em, prefix="ground", power=1e-3)
    b.ply(As.mesh("hills", R.hills_mesh), I, "hills", emission=dict(em), prefix="ground", power=1e-3)
    ax, az = R.LANDMARK_A
    focus["tower_spire"] = [ax, 262.0 + 9 + 30.0, az]
    focus["tower_a"] = [ax, 150.0, az + 21.0]
    bx, bz = R.LANDMARK_B
    focus["tower_b"] = [bx, 150.0, bz + 26.0]
    focus["bridge"] = [0.0, 9.0, R.RIVER[1]]
    focus["hills"] = [600.0, 260.0, -5600.0]
    focus["neighbour_roof"] = [-28.0, 36.5, -20.0]


# --------------------------------------------------------------------------
# Terrace
# --------------------------------------------------------------------------
def add_terrace(b: SceneBuilder, As: Assets, Ad: Assets, seed: int, focus: dict) -> None:
    rng = np.random.default_rng([seed, 31])
    I = np.eye(4)
    T = G.translate
    # our building below the deck, the deck, pavers, parapet upstand
    b.ply(As.mesh("our_walls", lambda: R.walls(-BX, BX, -BZ, BZ, 0.0, TY, 7.0)), I, "walls_brick_e_t0", prefix="ours")

    def deck_zone(x0, x1, z0, z1, rotate):
        uv = [(z0, x0), (z0, x1), (z1, x1), (z1, x0)] if rotate else [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
        return R.quad((x0, TY, z0), (x1, TY, z0), (x1, TY, z1), (x0, TY, z1), (0, 1, 0), uv)

    # three board-direction zones so the deck is not one tiled plane; boards turn 90 degrees under the lounge
    b.ply(As.mesh("deck", lambda: R.merge([deck_zone(-BX, BX, -BZ, -2.5, False), deck_zone(-2.4, BX, -2.5, 4.4, False)])),
          I, "deck", prefix="deck")
    b.ply(As.mesh("deck_lounge", lambda: deck_zone(-BX, -2.4, -2.5, 4.4, True)), I, "deck", prefix="deck")
    b.ply(As.mesh("pavers", lambda: R.quad((-BX, TY, 4.4), (BX, TY, 4.4), (BX, TY, BZ), (-BX, TY, BZ), (0, 1, 0),
                                            [(-BX, 4.4), (BX, 4.4), (BX, BZ), (-BX, BZ)])), I, "pavers", prefix="deck")
    def upstand(coping: bool):
        acc, h, w = R.Acc(), 0.3, 0.25
        for (sx, sz, cx, cz) in ((2 * BX, w, 0, -BZ + w / 2), (2 * BX, w, 0, BZ - w / 2),
                                 (w, 2 * BZ - 2 * w, -BX + w / 2, 0), (w, 2 * BZ - 2 * w, BX - w / 2, 0)):
            if coping:   # coping with a 3 cm overhang each side (the drip edge)
                acc.add(R.box((sx + (0.06 if sx > w else 0.06), 0.05, sz + 0.06), (cx, TY + h + 0.025, cz)))
            else:
                acc.add(R.box((sx, h, sz), (cx, TY + h / 2, cz)))
        return acc.mesh()

    b.ply(As.mesh("upstand", lambda: upstand(False)), I, "upstand", prefix="deck")
    b.ply(As.mesh("coping", lambda: upstand(True)), I, "coping", prefix="deck")
    # cable railing on the upstand (not along the hut)
    runs = [((-BX + 0.12, -BZ + 0.12), (BX - 0.12, -BZ + 0.12)), ((-BX + 0.12, -BZ + 0.12), (-BX + 0.12, BZ - 0.12)),
            ((BX - 0.12, -BZ + 0.12), (BX - 0.12, HUT[2])), ((-BX + 0.12, BZ - 0.12), (HUT[0], BZ - 0.12))]
    for i, (a, c) in enumerate(runs):
        parts = As.meshes(f"railing_{i}", lambda a=a, c=c: R.railing((a[0], 0, a[1]), (c[0], 0, c[1]), TY + 0.35))
        mats = {"post": "black_steel", "cable": "cable", "handrail": "teak"}
        for part in sorted(parts):
            b.ply(parts[part], I, mats[part], prefix="railing")
    focus["railing"] = [-1.8, TY + 1.2, -BZ + 0.12]

    # tables: scanned outdoor table-and-chairs sets, laid with glasses, a candle, plates and food
    wine = As.mesh("wine_glass", PR.wine_glass)
    cand = As.meshes("candle", PR.candle)
    tset, plate, crois = W.model("outdoor_table_chair_set_01"), W.model("carved_wooden_plate"), W.model("croissant")
    succ, lime = W.model("potted_plant_04"), W.model("food_lime_01")
    yaw0 = long_axis_yaw(tset, "x")
    for i, (x, z) in enumerate(TABLES):
        x, z = x + rng.uniform(-0.12, 0.12), z + rng.uniform(-0.08, 0.08)
        m = W.place(tset, at=(x, TY, z), yaw=yaw0 + rng.choice([0.0, 180.0]) + rng.uniform(-5, 5))
        W.add(b, tset, m, prefix="tableset")
        ty = surface_y(tset, m, x, z)
        g = []
        for k in range(int(rng.integers(1, 3))):
            gx, gz = x + rng.uniform(-0.17, 0.17), z + rng.uniform(-0.12, 0.12)
            b.ply(wine, T((gx, ty, gz)), "glass", prefix="glass")
            g.append([gx, ty + 0.13, gz])
        cm = T((x + rng.uniform(-0.05, 0.05), ty, z + rng.uniform(-0.05, 0.05)))
        b.ply(cand["holder"], cm, "glass", prefix="candle")
        b.ply(cand["wax"], cm, "wax", prefix="candle")
        b.ply(cand["wick"], cm, "wick", prefix="candle")
        b.ply(cand["flame"], cm, "bulb", emission=FLAME, prefix="flame", power=4 * math.pi * 0.006 ** 2 * luminance(FLAME))
        if rng.random() < 0.6:
            px, pz = x + rng.choice([-0.2, 0.2]), z + rng.uniform(-0.08, 0.08)
            W.add(b, plate, W.place(plate, at=(px, ty, pz), yaw=rng.uniform(0, 360)))
            W.add(b, crois, W.place(crois, at=(px, ty + 0.012, pz), yaw=rng.uniform(0, 360)))
        else:
            W.add(b, succ, W.place(succ, at=(x + rng.uniform(-0.15, 0.15), ty, z + rng.uniform(-0.1, 0.1)),
                                   yaw=rng.uniform(0, 360), height=0.17))
        focus[f"table_{i}"] = [x, ty, z]
        focus[f"glass_{i}"] = g[0]

    # parasols
    um = As.meshes("umbrella", R.umbrella)
    for i, (x, z) in enumerate(UMBRELLAS):
        m = G.compose(T((x, TY, z)), G.rotate((0, 1, 0), rng.uniform(0, 45)))
        for part, mat in (("canvas", "canvas"), ("valance", "canvas"), ("pole", "umbrella_pole"), ("base", "umbrella_base")):
            b.ply(um[part], m, mat, prefix="umbrella")
        focus[f"umbrella_{i}"] = [x + 1.0, TY + 2.2, z]

    # planters: corten troughs left and right, square planters for the olives
    troughs = [(-9.75, -9.05, -6.6, 3.2), (9.05, 9.75, -7.0, -3.4)]
    for i, (x0, x1, z0, z1) in enumerate(troughs):
        b.ply(As.mesh(f"trough_{i}", lambda x0=x0, x1=x1, z0=z0, z1=z1: R.merge([
            R.box((x1 - x0, 0.6, 0.02), ((x0 + x1) / 2, TY + 0.3, z0)), R.box((x1 - x0, 0.6, 0.02), ((x0 + x1) / 2, TY + 0.3, z1)),
            R.box((0.02, 0.6, z1 - z0), (x0, TY + 0.3, (z0 + z1) / 2)), R.box((0.02, 0.6, z1 - z0), (x1, TY + 0.3, (z0 + z1) / 2)),
            # 3 cm rolled lip round the top edge
            R.box((x1 - x0 + 0.03, 0.025, 0.05), ((x0 + x1) / 2, TY + 0.6, z0)), R.box((x1 - x0 + 0.03, 0.025, 0.05), ((x0 + x1) / 2, TY + 0.6, z1)),
            R.box((0.05, 0.025, z1 - z0), (x0, TY + 0.6, (z0 + z1) / 2)), R.box((0.05, 0.025, z1 - z0), (x1, TY + 0.6, (z0 + z1) / 2))])),
            I, "corten", prefix="planter")
        b.ply(As.mesh(f"trough_soil_{i}", lambda x0=x0, x1=x1, z0=z0, z1=z1: R.quad(
            (x0, TY + 0.55, z0), (x1, TY + 0.55, z0), (x1, TY + 0.55, z1), (x0, TY + 0.55, z1), (0, 1, 0),
            [(x0, z0), (x1, z0), (x1, z1), (x0, z1)])), I, "soil", prefix="planter")
    box3 = W.model("planter_box_03")
    for i, (x, z) in enumerate(OLIVES):
        m = W.place(box3, at=(x, TY, z), yaw=0.0, height=0.75)
        W.add(b, box3, m)
        (lx0, _, lz0), (lx1, _, lz1) = box3.bounds
        s = 0.75 / (box3.bounds[1][1] - box3.bounds[0][1])
        hw, hd = 0.5 * s * (lx1 - lx0) - 0.06, 0.5 * s * (lz1 - lz0) - 0.06
        b.ply(As.mesh("olive_soil", lambda hw=hw, hd=hd: R.quad((-hw, 0.68, -hd), (hw, 0.68, -hd), (hw, 0.68, hd), (-hw, 0.68, hd),
                                                                 (0, 1, 0), [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)])),
              T((x, TY, z)), "olive_soil", prefix="planter")

    # pergola: posts, beams, rafters; lounge under it
    px0, px1, pz0, pz1 = PERGOLA

    def pergola():
        acc = R.Acc()
        for x in (px0, (px0 + px1) / 2, px1):
            for z in (pz0, pz1):
                acc.add(R.box((0.14, 2.75, 0.14), (x, TY + 1.375, z)))
        for z in (pz0, pz1):
            acc.add(R.box((px1 - px0 + 0.5, 0.22, 0.08), ((px0 + px1) / 2, TY + 2.64, z)))
        for x in np.arange(px0 - 0.1, px1 + 0.15, 0.45):
            acc.add(R.box((0.05, 0.15, pz1 - pz0 + 0.5), (x, TY + 2.83, (pz0 + pz1) / 2)))
        return acc.mesh()

    b.ply(As.mesh("pergola", pergola), I, "pergola_wood", prefix="pergola")
    focus["pergola"] = [(px0 + px1) / 2, TY + 2.7, pz0]
    lx, lz = (px0 + px1) / 2, 4.3
    sofa = W.model("sofa_02")
    linen = b.d["linen"]
    W.add(b, sofa, W.place(sofa, at=(lx, TY, 6.15), yaw=180.0), overrides={p.name: linen for p in sofa.parts})
    pillows = W.model("throw_pillows_01")
    W.add(b, pillows, W.place(pillows, at=(lx, TY + 0.42, 6.02), yaw=180.0 + long_axis_yaw(pillows, "x")))
    arm = W.model("modern_arm_chair_01")
    fabric_parts = {p.name: linen for p in arm.parts if not any(w in p.name.lower() for w in ("leg", "wood", "metal", "frame"))}
    W.add(b, arm, W.place(arm, at=(px0 + 0.9, TY, lz), yaw=90 + rng.uniform(-8, 8)), prefix="armchair", overrides=fabric_parts)
    W.add(b, arm, W.place(arm, at=(px1 - 0.9, TY, lz), yaw=-90 + rng.uniform(-8, 8)), prefix="armchair", overrides=fabric_parts)
    side = W.model("side_table_tall_01")
    W.add(b, side, W.place(side, at=(px0 + 0.9, TY, lz + 0.85)))
    lantern = W.model("wooden_lantern_01")
    ctab = W.model("modern_coffee_table_01")
    mct = W.place(ctab, at=(lx, TY, lz), yaw=long_axis_yaw(ctab, "x"))
    W.add(b, ctab, mct)
    cty = surface_y(ctab, mct, lx, lz, 0.2)
    rug_h = 0.01
    b.ply(As.mesh("rug", lambda: R.box((4.0, rug_h, 2.8), (0, rug_h / 2, 0))), T((lx, TY, lz + 0.3)), "rug", prefix="lounge")
    W.add(b, lantern, W.place(lantern, at=(lx + 0.35, cty, lz), yaw=rng.uniform(0, 360), height=0.32))
    jug, mjug = W.model("jug_01"), W.model("metal_jug")
    W.add(b, jug, W.place(jug, at=(lx - 0.2, cty, lz + 0.05), yaw=rng.uniform(0, 360)))
    W.add(b, mjug, W.place(mjug, at=(lx - 0.42, cty, lz - 0.08), yaw=rng.uniform(0, 360)))
    pot = W.model("potted_plant_02")
    for (x, z) in ((px0 + 0.45, 6.35), (px1 - 0.45, 6.35)):
        W.add(b, pot, W.place(pot, at=(x, TY, z), yaw=rng.uniform(0, 360)))
    focus["sofa"] = [lx, TY + 0.55, 5.9]
    focus["armchair"] = [px0 + 0.85, TY + 0.5, lz]

    # bar: counter, back bar with bottles, stools
    def counter():
        acc = R.Acc()
        acc.add(R.box((BAR_X[1] - BAR_X[0], 1.05, BAR_Z[1] - BAR_Z[0]), ((BAR_X[0] + BAR_X[1]) / 2, TY + 0.525, (BAR_Z[0] + BAR_Z[1]) / 2)))
        for z in np.arange(BAR_Z[0] + 0.04, BAR_Z[1], 0.09):           # vertical slat cladding on the guest side
            acc.add(R.box((0.025, 1.0, 0.06), (BAR_X[0] - 0.012, TY + 0.5, z)))
        return acc.mesh()

    b.ply(As.mesh("bar_counter", counter), I, "teak_veneer", prefix="bar")
    b.ply(As.mesh("bar_kick", lambda: R.box((0.012, 0.15, BAR_Z[1] - BAR_Z[0]), (BAR_X[0] - 0.032, TY + 0.075, (BAR_Z[0] + BAR_Z[1]) / 2))),
          I, "kick", prefix="bar")
    b.ply(As.mesh("bar_top", lambda: R.box((0.85, 0.04, BAR_Z[1] - BAR_Z[0] + 0.1),
                                            ((BAR_X[0] + BAR_X[1]) / 2 - 0.08, TY + 1.07, (BAR_Z[0] + BAR_Z[1]) / 2))), I, "marble", prefix="bar")
    b.ply(As.mesh("foot_rail", lambda: R.tube([(BAR_X[0] - 0.18, TY + 0.22, BAR_Z[0]), (BAR_X[0] - 0.18, TY + 0.22, BAR_Z[1])], 0.022, 10)),
          I, "brass", prefix="bar")
    bbx = 9.3

    def back_bar():
        acc = R.Acc()
        acc.add(R.box((0.55, 0.9, 5.2), (bbx, TY + 0.45, 0.2)))
        for y in (1.25, 1.62, 1.99):
            acc.add(R.box((0.3, 0.03, 5.2), (bbx + 0.08, TY + y, 0.2)))
        acc.add(R.box((0.03, 1.5, 5.2), (bbx + 0.24, TY + 1.65, 0.2)))
        return acc.mesh()

    b.ply(As.mesh("back_bar", back_bar), I, "backbar", prefix="bar")
    focus["bar_bottles"] = [bbx + 0.08, TY + 1.75, 0.3]
    stool = W.model("bar_chair_round_01")
    for k, z in enumerate(np.linspace(BAR_Z[0] + 0.5, BAR_Z[1] - 0.5, 5)):
        W.add(b, stool, W.place(stool, at=(BAR_X[0] - 0.55 + rng.uniform(-0.06, 0.06), TY, z + rng.uniform(-0.08, 0.08)),
                                yaw=rng.uniform(0, 360)))
    bar_y = TY + 1.09
    bowl = W.model("wooden_bowl_01")
    W.add(b, bowl, W.place(bowl, at=(5.62, bar_y, -1.6)))
    for k in range(5):
        a_ = rng.uniform(0, 2 * math.pi)
        W.add(b, lime, W.place(lime, at=(5.62 + 0.06 * math.cos(a_), bar_y + 0.02 + 0.02 * (k > 2), -1.6 + 0.06 * math.sin(a_)),
                               yaw=rng.uniform(0, 360)))
    W.add(b, lantern, W.place(lantern, at=(5.6, bar_y, 1.3), yaw=rng.uniform(0, 360), height=0.4))
    W.add(b, succ, W.place(succ, at=(5.6, bar_y, 2.5), yaw=rng.uniform(0, 360), height=0.2))
    wb = W.model("wine_bottles_01")
    for ys in (TY + 1.265, TY + 1.635, TY + 2.005):                  # scanned bottle rows on every back-bar shelf
        for zc in (-1.7, -0.9, -0.1, 0.7, 1.5, 2.3):
            if rng.random() < 0.8:
                W.add(b, wb, W.place(wb, at=(bbx + 0.08, ys, zc + rng.uniform(-0.05, 0.05)), yaw=long_axis_yaw(wb, "z") + rng.choice([0, 180])))
    diya, basket, lemon = W.model("brass_diya_lantern"), W.model("wicker_basket_01"), W.model("lemon")
    W.add(b, diya, W.place(diya, at=(5.62, bar_y, -0.4), yaw=rng.uniform(0, 360)))
    W.add(b, basket, W.place(basket, at=(5.6, bar_y, 0.3), yaw=long_axis_yaw(basket, "z")))
    for k in range(4):
        W.add(b, lemon, W.place(lemon, at=(5.6 + rng.uniform(-0.06, 0.06), bar_y + 0.03, 0.3 + rng.uniform(-0.1, 0.1)), yaw=rng.uniform(0, 360)))
    W.add(b, W.model("propane_tank"), W.place(W.model("propane_tank"), at=(6.35, TY, 3.4)))
    chalk = W.model("standing_chalkboard_01")
    W.add(b, chalk, W.place(chalk, at=(4.4, TY, -3.4), yaw=120.0))
    barrel = W.model("wine_barrel_01")
    mbar = W.place(barrel, at=(3.4, TY, -0.8), yaw=rng.uniform(0, 360))
    W.add(b, barrel, mbar)
    by = surface_y(barrel, mbar, 3.4, -0.8, 0.15)
    for k in range(2):
        b.ply(wine, T((3.4 + rng.uniform(-0.15, 0.15), by, -0.8 + rng.uniform(-0.15, 0.15))), "glass", prefix="glass")
    focus["barrel"] = [3.4, by, -0.8]
    W.add(b, pot, W.place(pot, at=(BAR_X[0] - 0.2, TY, BAR_Z[1] + 0.55), yaw=rng.uniform(0, 360)))
    focus["stool"] = [BAR_X[0] - 0.55, TY + 0.78, BAR_Z[0] + 0.5]

    # stair hut with a timber water tank, HVAC behind a slatted screen
    hx0, hx1, hz0, hz1 = HUT

    def hut():
        acc = R.Acc()
        acc.add(R.walls(hx0, hx1, hz0, hz1, TY, TY + 2.9))
        acc.add(R.roof_cap(hx0, hx1, hz0, hz1, TY + 2.9))
        return acc.mesh()

    b.ply(As.mesh("hut", hut), I, "hut", prefix="hut")
    b.ply(As.mesh("hut_roof", lambda: R.box((hx1 - hx0 + 0.3, 0.18, hz1 - hz0 + 0.3), ((hx0 + hx1) / 2, TY + 2.99, (hz0 + hz1) / 2))),
          I, "hut_roof", prefix="hut")
    b.ply(As.mesh("hut_door", lambda: R.box((0.06, 2.1, 1.0), (hx0 - 0.02, TY + 1.05, 6.2))), I, "door", prefix="hut")
    tank, steel = R.water_tank(((hx0 + hx1) / 2 + 0.6, (hz0 + hz1) / 2), TY + 3.08, np.random.default_rng(5))
    b.ply(As.mesh("hut_tank", lambda: tank), I, "tank", prefix="hut")
    b.ply(As.mesh("hut_tank_steel", lambda: steel), I, "steel", prefix="hut")
    mx, my, mz = ANCHORS["mast"]
    b.ply(As.mesh("mast", lambda: R.tube([(mx, TY + 2.9, mz), (mx, my + 0.4, mz)], 0.04, 8)), I, "black_steel", prefix="pole")
    focus["water_tank"] = [(hx0 + hx1) / 2 + 0.6, TY + 5.5, (hz0 + hz1) / 2]
    focus["hut_door"] = [hx0, TY + 1.4, 6.2]

    def wall_item(mid, x, y, z, face):
        """Mount a scanned wall item (thin along its local z, front +z) on a hut face: 'w' (x = hx0) or 's' (z = hz0)."""
        m = W.model(mid)
        dz = m.bounds[1][2] - m.bounds[0][2]
        if face == "w":
            W.add(b, m, W.place(m, at=(hx0 - dz / 2, y, z), yaw=-90.0))
        else:
            W.add(b, m, W.place(m, at=(x, y, hz0 - dz / 2), yaw=180.0))

    wall_item("industrial_wall_lamp", 0, TY + 2.3, 6.2, "w")
    wall_item("fire_alarm", 5.55, TY + 1.45, 0, "s")
    wall_item("wall_clock", 6.5, TY + 2.15, 0, "s")
    wall_item("garden_hose_wall_mounted_01", 9.0, TY + 0.75, 0, "s")
    pipes = W.model("modular_industrial_pipes_01")
    W.add(b, pipes, W.place(pipes, at=(hx0 - 0.17, TY, 7.35), yaw=-90.0))
    cans, bag = W.model("metal_trash_can"), W.model("trashbag")
    W.add(b, cans, W.place(cans, at=(7.55, TY, hz0 - 0.32), yaw=long_axis_yaw(cans, "x")))
    W.add(b, bag, W.place(bag, at=(8.75, TY, hz0 - 0.35), yaw=rng.uniform(0, 360)))
    vx0, vx1, vz0, vz1 = HVAC

    ac, duct = W.model("exterior_aircon_unit"), W.model("modular_airduct_circular_01")
    for cx in (vx0 + 1.1, vx0 + 3.2):
        W.add(b, ac, W.place(ac, at=(cx, TY, 6.7), yaw=180.0 + long_axis_yaw(ac, "x")))
    W.add(b, duct, W.place(duct, at=(vx0 + 2.2, TY, 7.55), yaw=long_axis_yaw(duct, "x")))
    b.ply(As.mesh("screen", lambda: R.merge([R.box((0.07, 1.9, 0.035), (x, TY + 0.95, vz0)) for x in np.arange(vx0, vx1, 0.11)])),
          I, "teak_veneer", prefix="screen")
    sec = W.model("security_light")
    W.add(b, sec, W.place(sec, at=(2.65, TY + 1.55, vz0 - 0.24), yaw=180.0))
    shrub3, shrub4 = W.model("shrub_03"), W.model("shrub_04")
    for mid, x, sh in (("planter_box_02", 1.35, shrub3), ("planter_box_01", 3.35, shrub4)):
        pm = W.model(mid)
        mp = W.place(pm, at=(x, TY, vz0 - 0.33), yaw=long_axis_yaw(pm, "x"))
        W.add(b, pm, mp)
        top = surface_y(pm, mp, x, vz0 - 0.33, 0.1)
        W.add(b, sh, W.place(sh, at=(x, top - 0.03, vz0 - 0.33), yaw=long_axis_yaw(sh, "x")))
    # festoon poles and strands
    for name, (x, y, z) in ANCHORS.items():
        if name == "mast":
            continue
        b.ply(As.mesh(f"pole_{name}", lambda x=x, y=y, z=z: R.tube([(x, TY + 0.3, z), (x, y + 0.15, z)], 0.035, 8)), I, "black_steel", prefix="pole")
    lr = np.random.default_rng([seed, 41])
    lit = [lr.random() < 0.9 for _ in STRANDS]
    on_area, n_on, bulbs_all = 0.0, 0, []

    def strands():
        out = {"cable": R.Acc(), "bulbs_on": R.Acc(), "bulbs_off": R.Acc()}
        for (a, c, sag), on in zip(STRANDS, lit):
            cab, bl, cent = R.festoon(ANCHORS[a], ANCHORS[c], sag)
            out["cable"].add(cab)
            out["bulbs_on" if on else "bulbs_off"].add(bl)
        return {k: v.mesh() for k, v in out.items() if v.off}

    for (a, c, sag), on in zip(STRANDS, lit):
        cent = R.festoon(ANCHORS[a], ANCHORS[c], sag)[2]
        bulbs_all += cent
        n_on += len(cent) if on else 0
    st = Ad.meshes("festoon", strands)
    b.ply(st["cable"], I, "black_steel", prefix="festoon")
    if "bulbs_off" in st:
        b.ply(st["bulbs_off"], I, "bulb", prefix="festoon")
    if "bulbs_on" in st:
        area = n_on * 4 * math.pi * 0.032 ** 2 * 1.1
        b.ply(st["bulbs_on"], I, "bulb", emission=BULB, prefix="bulbs", power=area * luminance(BULB))
    focus["bulbs"] = bulbs_all[::7]
    focus["bulb_near"] = bulbs_all[len(bulbs_all) // 8]


TROUGHS = [(-9.75, -9.05, -6.6, 3.2), (9.05, 9.75, -7.0, -3.4)]


def trough_slots(seed: int) -> list:
    """(kind, position) for every planting slot along the troughs: lavender, procedural grass, or a scanned shrub."""
    rng = np.random.default_rng([seed, 22])
    out = []
    for (x0, x1, z0, z1) in TROUGHS:
        for z in np.arange(z0 + 0.25, z1 - 0.2, 0.42):
            p = ((x0 + x1) / 2 + rng.uniform(-0.15, 0.15), TY + 0.55, z + rng.uniform(-0.08, 0.08))
            r = rng.random()
            out.append(("lavender" if r < 0.55 else "grass" if r < 0.78 else "shrub", p))
    return out


def add_plants(b: SceneBuilder, Ad: Assets, seed: int, focus: dict) -> None:
    slots = trough_slots(seed)

    def plants():
        rng = np.random.default_rng([seed, 21])
        grp = R.Groups()
        for (x, z) in OLIVES:
            R.olive_tree(grp, rng, (x + rng.uniform(-0.08, 0.08), TY + 0.7, z + rng.uniform(-0.08, 0.08)), rng.uniform(2.3, 2.9))
        for kind, p in slots:
            if kind == "lavender":
                R.lavender(grp, rng, p)
            elif kind == "grass":
                R.grass(grp, rng, p, length=rng.uniform(0.55, 0.8))
        # vines up the pergola posts and along the rafters, a few trailing ends
        px0, px1, pz0, pz1 = PERGOLA
        for x in (px0, (px0 + px1) / 2, px1):
            for z in (pz0, pz1):
                if rng.random() < 0.7:
                    path = np.column_stack([np.full(30, x) + 0.09 * np.sin(np.linspace(0, 9, 30)),
                                            np.linspace(TY + 0.05, TY + 2.85, 30),
                                            np.full(30, z) + 0.09 * np.cos(np.linspace(0, 9, 30))])
                    R.vine(grp, rng, path)
        for x in np.arange(px0 - 0.1, px1 + 0.15, 0.45):
            if rng.random() < 0.65:
                zz = np.linspace(pz0 - 0.2, pz1 + 0.2, 40)
                path = np.column_stack([np.full(40, x) + 0.06 * np.sin(zz * 3 + rng.uniform(0, 6)),
                                        np.full(40, TY + 2.93) + 0.03 * np.sin(zz * 5), zz])
                R.vine(grp, rng, path, every=0.03)
                for _ in range(2):
                    z = rng.uniform(pz0, pz1)
                    drop = rng.uniform(0.3, 0.9)
                    R.vine(grp, rng, np.array([[x, TY + 2.9, z], [x + 0.05, TY + 2.9 - drop * 0.5, z + 0.04],
                                               [x + 0.02, TY + 2.9 - drop, z + 0.08]]), every=0.05)
        return grp.meshes()

    parts = Ad.meshes("plants", plants)
    I = np.eye(4)
    srng = np.random.default_rng([seed, 23])
    shrubs = [W.model("shrub_04"), W.model("shrub_03")]
    for kind, (x, y, z) in slots:
        if kind == "shrub":
            m = shrubs[int(srng.integers(2))]
            W.add(b, m, W.place(m, at=(x, y - 0.02, z), yaw=long_axis_yaw(m, "z") + srng.uniform(-15, 15),
                                scale=srng.uniform(0.9, 1.2)))
    for part in sorted(parts):
        b.ply(parts[part], I, part, prefix="plant")
    focus["olive_0"] = [OLIVES[0][0], TY + 2.3, OLIVES[0][1]]
    focus["olive_2"] = [OLIVES[2][0], TY + 2.3, OLIVES[2][1]]
    focus["lavender"] = [-9.4, TY + 0.85, -4.0]


def build(ctx: BuildContext) -> SceneBundle:
    As, Ad = Assets(ctx.shared_dir, ctx.rebuild), Assets(ctx.assets_dir, ctx.rebuild)
    b = SceneBuilder(As)
    add_materials(b, As, Ad, ctx.env)
    focus: dict = {}
    add_city(b, As, Ad, ctx.seed, ctx.env, focus)
    add_terrace(b, As, Ad, ctx.seed, focus)
    add_plants(b, Ad, ctx.seed, focus)
    lamps = b.lamp_weight
    sky = env_kit.sky(ctx.env, sun_direction=SUN, scale=SKY_SCALE[ctx.env], sampling_weight=max(4.0 * lamps, 1.0))
    if sky["type"] == "sunsky":
        sky["turbidity"] = GOLDEN_TURBIDITY          # hazier air than env_kit's clear: a warmer low sun
    b.d["sky"] = sky
    return SceneBundle(b.d, focus, {"sun_direction": list(SUN), "sky_scale": SKY_SCALE[ctx.env],
                                    "haze": "aerial perspective by distance tiers, T = exp(-d / 3.5 km); see module docstring"})


CAMERA_BOX = ((-9.5, TY + 0.45, -7.55), (9.5, TY + 2.3, 7.6))
TARGET_BOX = ((-60.0, 20.0, -300.0), (60.0, 50.0, 8.0))


def _exclusions() -> tuple:
    ex = [((HUT[0] - 0.25, TY, HUT[2] - 0.25), (HUT[1] + 0.3, TY + 2.4, HUT[3] + 0.3)),
          ((HVAC[0] - 0.15, TY, HVAC[2] - 0.35), (HVAC[1] + 0.2, TY + 2.4, HVAC[3] + 0.2)),
          ((BAR_X[0] - 0.85, TY, BAR_Z[0] - 0.3), (BAR_X[1] + 0.2, TY + 1.3, BAR_Z[1] + 0.3)),
          ((8.6, TY, -2.7), (9.6, TY + 2.4, 3.1)),
          ((PERGOLA[0] - 0.3, TY, 2.6), (PERGOLA[1] + 0.3, TY + 1.25, PERGOLA[3] + 0.3)),
          ((-9.6, TY, -6.9), (-8.6, TY + 1.4, 3.5)), ((8.6, TY, -7.3), (9.6, TY + 1.4, -3.1))]
    ex += [((3.0, TY, -1.2), (3.8, TY + 1.15, -0.4)),            # barrel table
           ((4.0, TY, -3.8), (4.8, TY + 1.6, -3.0)),             # chalkboard
           ((6.4, TY, 3.0), (9.6, TY + 1.2, 4.6))]               # trash cans, propane
    for (x, z) in TABLES:
        ex.append(((x - 1.0, TY, z - 0.75), (x + 1.0, TY + 1.05, z + 0.75)))
    for (x, z) in OLIVES:
        ex.append(((x - 0.95, TY, z - 0.95), (x + 0.95, TY + 2.4, z + 0.95)))
    for (x, z) in UMBRELLAS:
        ex.append(((x - 1.6, TY + 1.95, z - 1.6), (x + 1.6, TY + 2.4, z + 1.6)))
    return tuple(ex)


SCENE = SceneDef(
    id="rooftop", group="hybrid", owner="rui",
    description="Rooftop terrace bar at golden hour 42 m above a city: decking, cable railing, olive trees, pergola "
                "lounge, bar, parasols and festoon lights against a skyline, a river and hazy hills up to 10 km away.",
    build=build,
    views={
        # From the back of the terrace across the tables and parasols to the skyline.
        "skyline": View((1.4, TY + 1.55, 0.6), (-4.0, TY - 1.2, -60.0), focus="table_2"),
        # Seated close to a table by the railing, the city beyond.
        "table": View((-2.45, TY + 0.95, -4.55), (-4.5, TY + 0.25, -8.6), focus="glass_1"),
        # Along the festoon strands toward the front-left corner.
        "lights": View((4.0, TY + 1.5, 4.1), (-9.0, TY + 2.2, -6.5), focus="bulb_near"),
        # From the front back toward the pergola lounge and the olive tree.
        "lounge": View((-0.8, TY + 1.4, -4.9), (-6.5, TY + 0.6, 5.0), focus="sofa"),
    },
    default_view="skyline",
    camera_box=CAMERA_BOX, target_box=TARGET_BOX,
    exclude_boxes=_exclusions(),
    envs=("clear", "overcast"),
    tags=("outdoor", "day", "vista", "architecture", "foliage", "bokeh", "artificial_light", "open"),
    default_seed=5, asset_version=ASSET_VERSION, max_depth=8, rr_depth=5, spp_hint=2048, far_clip=20000.0,
)
