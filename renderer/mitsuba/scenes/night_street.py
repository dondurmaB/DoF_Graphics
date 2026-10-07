"""A city street at night: the bokeh and high-dynamic-range stress case.

Layout, meters, y up, the street runs along z and the default camera looks toward -z:
  carriageway   x in [-4.5, 4.5], z in [-70, 10]: aggregate asphalt, patches, cracks, worn markings,
                a zebra crossing at z = -20 with drop kerbs, manholes and gullies; parking bays at |x| = 3.45
  pavements     |x| in [4.65, 7.5] behind granite kerbs: slabs, tree grates, lamps, signs, bins, bollards,
                parking meters, a bike rack, a bus shelter
  facades       seven buildings per side and one at each end (`_night_street_facade`): brick, stucco, stone
                and a glass office block, with recessed windows, lit rooms, shopfronts, signs, fire escapes
  lights        heritage sodium lanterns (left), LED cobra-heads (right), room and shop lights, neon,
                traffic signals, car lamps, the bus-shelter poster, a city-glow sky
  overhead      tram contact wires on span wires, utility cables

`clear` is dry asphalt; `wet` is rain-wet asphalt (rough enough that lights streak) with mirror puddles.
The seed picks which rooms are lit and how they are furnished, shop types and signs, car models, colours,
plates and bay occupancy, the signal phase and the puddles.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import mitsuba as mi
import procedural as G
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, luminance, principled, rgb
from scenes import _night_street_facade as F
from scenes import _night_street_props as P

ROAD_HALF, KERB_W, PAVE_OUT, Z_NEAR, Z_FAR = 4.5, 0.15, 7.5, 10.0, -68.0
CROSSING = (-21.5, -18.5)
LAMP_H, LAMP_REACH, LANTERN_POST = 8.0, 2.0, 4.2
LED, SODIUM = (0.84, 0.92, 1.0), (1.0, 0.56, 0.20)
LED_L, LANTERN_L = 650.0, 85.0
HEADLIGHT, TAILLIGHT = (1.0, 0.95, 0.86), (1.0, 0.04, 0.03)
HEADLIGHT_L, TAILLIGHT_L, SIGNAL_L = 420.0, 70.0, 70.0
SKY_WEIGHT = 0.05
# Light-sampling share of interior lights relative to their power. Rooms and shops are closed boxes, so from
# the street almost every shadow ray to them is blocked; at full weight they took ~85% of all light samples
# and starved the street lamps. Inside a room the lamp is still found, by BSDF sampling and the smaller share.
INTERIOR_WEIGHT = 0.08
INTERIOR = ("emit_room", "emit_shop", "emit_office", "emit_tv", "emit_hall", "emit_dormer", "emit_display", "emit_sheer", "emit_screen")

STYLE_TRIM = {"brick": "stone_trim", "stone": "stone_trim", "stucco": "stucco_trim", "glass": "frame_alu"}
# (width, storeys, style, wall, features) from z = +10 toward -68
BUILDINGS = {
    -1: [(12, 5, "brick", "brick_red", {"fire_escape": True}), (9, 4, "stucco", "stucco_cream", {"balconies": True}),
         (14, 6, "glass", "stone_grey", {}), (8, 5, "brick", "brick_brown", {}),
         (11, 5, "stone", "stone_beige", {"mansard": True}), (10, 4, "stucco", "stucco_pink", {"balconies": True, "tank": True}),
         (14, 5, "brick", "brick_buff", {})],
    +1: [(10, 5, "stucco", "stucco_sage", {"balconies": True}), (13, 6, "brick", "brick_red", {"tank": True}),
         (8, 4, "stone", "stone_grey", {"mansard": True}), (12, 5, "brick", "brick_brown", {"fire_escape": True}),
         (9, 6, "stucco", "stucco_ochre", {}), (14, 4, "brick", "brick_red", {}), (12, 5, "stone", "stone_beige", {})],
}
END_CAPS = [(Z_FAR, 7, "stone", "stone_grey", 0.0), (Z_NEAR, 6, "brick", "brick_brown", 180.0)]

RIGHT_BAYS, LEFT_BAYS = [-3.0, -8.7, -25.0, -30.7, -37.5, -44.5], [-1.5, -7.2, -12.9, -26.5, -32.2, -39.0, -46.0]
MOVING = [(+1.25, -30.0, 0.0, "head"), (-1.30, -15.0, 180.0, "tail"), (+1.25, -56.0, 0.0, "head")]
LANTERNS, COBRAS = [6.0, -10.0, -26.0, -42.0, -58.0], [0.0, -16.0, -32.0, -48.0, -64.0]
TREES = [(6.0, -6.0), (6.0, -35.5), (-6.0, -17.0), (-6.0, -50.0)]
BOLLARDS = [(-4.85, z) for z in (-3.0, -4.5, -6.0, -7.5, -9.0)] + [(4.85, z) for z in (-15.5, -16.5, -17.5)]
BINS = [(-5.6, -4.0), (-5.6, -29.0), (5.6, -22.5)]
SIGNS = [(-4.85, -13.5, "crossing", False), (-4.85, -24.5, "one_way", False), (4.85, -27.2, "speed", True),
         (4.85, -41.0, "no_parking", True)]
METERS = [(-4.85, -31.5), (-4.85, -37.0)]
RACK = (6.75, -24.6)
SHELTER = {"x": (5.2, 7.3), "z": (-14.0, -10.4)}
MANHOLES = [(-1.0, -6.0), (1.8, -28.0), (-2.2, -44.0), (0.4, -58.0)]
GULLIES = [(-4.3, -2.0), (4.3, -12.0), (-4.3, -34.0), (4.3, -46.0)]
PATCHES = [(-3.9, -1.6, -12.0, -9.6), (0.6, 2.7, -30.0, -27.4), (-1.3, 0.9, -50.0, -46.2), (1.9, 4.3, -3.6, -0.9),
           (-4.3, -2.9, -38.0, -33.0)]
SPANS = [4.0, -9.0, -22.0, -35.0, -48.0, -61.0]
HERO_PUDDLE = (-0.3, -4.0, 2.2, 2.8)

PAINTS = [(0.015, 0.015, 0.018), (0.80, 0.80, 0.78), (0.42, 0.43, 0.45), (0.20, 0.21, 0.23), (0.03, 0.07, 0.20),
          (0.40, 0.03, 0.03), (0.05, 0.16, 0.10), (0.45, 0.38, 0.26), (0.30, 0.45, 0.62)]
PAINT_P = np.array([0.20, 0.18, 0.17, 0.12, 0.10, 0.08, 0.05, 0.05, 0.05])
FASCIAS = [(0.03, 0.10, 0.06), (0.02, 0.02, 0.025), (0.30, 0.04, 0.05), (0.03, 0.06, 0.16), (0.55, 0.50, 0.42), (0.10, 0.20, 0.22)]
AWNINGS = [[(0.55, 0.06, 0.06), (0.85, 0.82, 0.74)], [(0.05, 0.25, 0.12), (0.85, 0.82, 0.74)],
           [(0.06, 0.10, 0.30), (0.80, 0.78, 0.70)], [(0.30, 0.20, 0.10), (0.75, 0.62, 0.40)]]


def tex(path: str, raw: bool = False, scale=None, wrap: str = "repeat", to_uv=None) -> dict:
    spec = {"type": "bitmap", "filename": path, "raw": raw, "filter_type": "bilinear", "wrap_mode": wrap}
    if scale is not None:
        spec["to_uv"] = mi.ScalarTransform4f().scale([scale[0], scale[1], 1.0])
    if to_uv is not None:
        spec["to_uv"] = to_uv
    return spec


def diffuse(c) -> dict:
    return {"type": "diffuse", "reflectance": c if isinstance(c, dict) else rgb(*c)}


def add_materials(b: SceneBuilder, As: Assets, Ad: Assets, wet: bool) -> None:
    A = As
    # ground
    b.material("asphalt", principled(                                         # wet asphalt is darker
        tex(A.texture(f"asphalt_{'wet' if wet else 'dry'}", lambda: P.asphalt(2048, 3)[0] * (0.55 if wet else 1.0)),
            scale=(1 / P.ASPHALT_TILE_M,) * 2, wrap="mirror"),
        tex(Ad.texture("road_rough", lambda: P.road_roughness(wet, 11, P.puddle_list(11, HERO_PUDDLE)), gray=True), raw=True,
            scale=(1 / 9.0, 1 / 80.0), wrap="clamp"), specular=0.5))
    b.material("asphalt_patch", principled(tex(A.texture(f"asphalt_patch_{'wet' if wet else 'dry'}",
                                                         lambda: P.asphalt(1024, 8, patch=True)[0] * (0.6 if wet else 1.0)), scale=(1 / P.ASPHALT_TILE_M,) * 2, wrap="mirror"),
                                           0.10 if wet else 0.55, specular=0.5))
    b.material("paving", principled(tex(A.texture(f"paving_{'wet' if wet else 'dry'}", lambda: P.paving(2048, 5)[0] * (0.7 if wet else 1.0)), scale=(1 / P.PAVING_TILE_M,) * 2), 0.22 if wet else 0.78,
                                    specular=0.5))
    b.material("granite", principled(tex(A.texture("granite", lambda: P.granite()), scale=(1 / 0.6, 1 / 0.6)), 0.25 if wet else 0.6))
    b.material("tactile", principled((0.55, 0.45, 0.20), 0.3 if wet else 0.7))
    b.material("paint", principled((0.70, 0.70, 0.66), 0.15 if wet else 0.55, specular=0.5))
    b.material("crack", principled((0.012, 0.012, 0.012), 0.9))
    b.material("manhole", principled(tex(A.texture("manhole", lambda: P.manhole()[0])), 0.25 if wet else 0.55, metallic=0.6))
    b.material("gully_hole", diffuse((0.0, 0.0, 0.0)))
    # facades
    for name, (kind, _c) in F.WALLS.items():
        tw, th = F.TILE[kind]
        b.material(f"wall_{name}", principled(tex(A.texture(f"wall_{name}", lambda name=name: F.wall_texture(name)), scale=(1 / tw, 1 / th)),
                                              0.8 if kind != "stone" else 0.7, specular=0.35))
    b.material("wall_glass_frame", principled((0.10, 0.11, 0.12), 0.35, metallic=0.5))
    b.material("stone_trim", principled((0.62, 0.58, 0.50), 0.7, specular=0.3))
    b.material("stucco_trim", principled((0.82, 0.80, 0.74), 0.6, specular=0.3))
    b.material("frame_white", principled((0.80, 0.80, 0.77), 0.4))
    b.material("frame_dark", principled((0.04, 0.05, 0.05), 0.4))
    b.material("frame_wood", principled((0.22, 0.12, 0.06), 0.45))
    b.material("frame_alu", principled((0.55, 0.56, 0.58), 0.3, metallic=0.9))
    # smooth (delta) plastic: crisp glass reflections, and no glossy-lobe noise from the many small lights
    b.material("glass_dark", {"type": "plastic", "diffuse_reflectance": rgb(0.004, 0.005, 0.006), "int_ior": 1.52})
    b.material("glass_curtain", {"type": "plastic", "diffuse_reflectance": rgb(0.005, 0.009, 0.013), "int_ior": 1.52})
    b.material("spandrel", principled((0.02, 0.03, 0.035), 0.1, specular=0.8))
    b.material("shop_glass", {"type": "thindielectric"})
    b.material("roof", diffuse((0.08, 0.08, 0.08)))
    b.material("slate", principled((0.08, 0.09, 0.10), 0.45))
    b.material("ac_unit", principled((0.62, 0.62, 0.60), 0.45))
    b.material("drain", principled((0.05, 0.05, 0.05), 0.4))
    b.material("iron", principled((0.025, 0.025, 0.028), 0.45, metallic=0.5))
    b.material("wood_tank", principled((0.25, 0.17, 0.10), 0.8))
    b.material("rim_brass", {"type": "roughconductor", "material": "Au", "alpha": 0.2})
    b.material("riser", principled((0.10, 0.12, 0.11), 0.3, specular=0.6))
    for i, c in enumerate(FASCIAS):
        b.material(f"fascia_{i}", principled(c, 0.35, specular=0.5))
    for i, cols in enumerate(AWNINGS):
        b.material(f"awning_{i}", {"type": "twosided", "bsdf": {"type": "diffuse", "reflectance": tex(
            A.texture(f"awning_{i}", lambda cols=cols, i=i: P.awning_stripes(cols, seed=i)), scale=(1 / 2.4, 1 / 1.4))}})
    b.material("letters_gold", {"type": "roughconductor", "material": "Au", "alpha": 0.15})
    b.material("letters_white", principled((0.85, 0.85, 0.82), 0.3))
    b.material("letters_dark", principled((0.02, 0.02, 0.02), 0.4))
    b.material("shutter", principled((0.40, 0.41, 0.42), 0.35, metallic=0.8))
    for i, c in enumerate([(0.30, 0.05, 0.05), (0.03, 0.08, 0.04), (0.05, 0.07, 0.18), (0.02, 0.02, 0.02)]):
        b.material(f"door_{i}", principled(c, 0.25, clearcoat=0.6))
    # interiors
    b.material("room_a", diffuse((0.70, 0.64, 0.55)))
    b.material("room_b", diffuse((0.62, 0.62, 0.63)))
    b.material("room_c", diffuse((0.55, 0.40, 0.30)))
    b.material("room_floor", principled((0.30, 0.18, 0.09), 0.4))
    b.material("room_ceiling", diffuse((0.80, 0.79, 0.76)))
    b.material("office_floor", diffuse((0.20, 0.21, 0.24)))
    b.material("furn_dark", diffuse((0.06, 0.05, 0.05)))
    b.material("furn_light", diffuse((0.55, 0.52, 0.48)))
    b.material("furn_red", diffuse((0.40, 0.08, 0.06)))
    b.material("furn_wood", principled((0.30, 0.17, 0.08), 0.4))
    b.material("shelf_books", diffuse(tex(A.texture("books", lambda: F.book_spines()), scale=(1 / 0.9, 1 / 1.9))))
    b.material("art", diffuse(tex(A.texture("art", lambda: G.abstract_art(256, 256, [(0.8, 0.3, 0.2), (0.2, 0.4, 0.7), (0.9, 0.8, 0.5), (0.1, 0.1, 0.1)], seed=4)),
                                  scale=(1 / 0.8, 1 / 0.6))))
    b.material("blind", diffuse((0.75, 0.72, 0.66)))
    for i, c in enumerate([(0.45, 0.06, 0.06), (0.75, 0.70, 0.58), (0.12, 0.18, 0.30), (0.30, 0.35, 0.20)]):
        b.material(f"curtain_{i}", {"type": "twosided", "bsdf": diffuse(c)})
    b.material("shop_wall_a", diffuse((0.75, 0.72, 0.65)))
    b.material("shop_wall_b", diffuse((0.30, 0.22, 0.16)))
    b.material("shop_wall_c", diffuse((0.20, 0.32, 0.30)))
    b.material("shop_wall_cool", diffuse((0.82, 0.84, 0.85)))
    b.material("shop_floor", principled((0.45, 0.44, 0.42), 0.25))
    b.material("leaf_plant", diffuse((0.05, 0.16, 0.05)))
    b.material("shelf", principled((0.70, 0.70, 0.70), 0.5))
    b.material("products", principled(tex(A.texture("products", lambda: F.product_palette())), 0.35, specular=0.5))
    # street furniture and lamps
    b.material("lamp_grey", principled((0.12, 0.13, 0.14), 0.35, metallic=0.8))
    b.material("lamp_black", principled((0.015, 0.016, 0.018), 0.35, metallic=0.4))
    b.material("bin_green", principled((0.03, 0.12, 0.06), 0.4, clearcoat=0.3))
    b.material("bollard_band", principled((0.85, 0.85, 0.82), 0.3))
    b.material("bark", diffuse(tex(A.texture("bark", lambda: P.bark()), scale=(1.0, 0.5))))
    b.material("leaf", {"type": "twosided", "bsdf": principled(tex(A.texture("leaf_atlas", lambda: P.leaf_atlas())), 0.5, specular=0.3)})
    b.material("soil", diffuse((0.04, 0.03, 0.02)))
    b.material("sign_back", principled((0.45, 0.46, 0.47), 0.4, metallic=0.7))
    for kind in ("no_parking", "speed", "one_way", "crossing"):
        b.material(f"sign_{kind}", principled(tex(A.texture(f"sign_{kind}", lambda kind=kind: np.flipud(P.road_sign(kind)).copy())), 0.3, specular=0.6))
    b.material("street_name", principled(tex(A.texture("street_name", lambda: np.flipud(P.street_name("Market Street")).copy())), 0.4))
    b.material("bike_frame", principled((0.05, 0.18, 0.35), 0.25, clearcoat=0.5))
    b.material("rubber", principled((0.02, 0.02, 0.02), 0.7))
    b.material("steel", {"type": "roughconductor", "material": "Cr", "alpha": 0.15})
    b.material("signal_lens_off", principled((0.02, 0.02, 0.02), 0.08, specular=0.9))
    b.material("shelter_metal", principled((0.15, 0.16, 0.17), 0.35, metallic=0.8))
    b.material("wood", principled((0.20, 0.10, 0.05), 0.5))
    b.material("cable", principled((0.02, 0.02, 0.02), 0.5))
    b.material("emitter_backing", diffuse((0.0, 0.0, 0.0)))
    # cars
    for i, c in enumerate(PAINTS):
        # clear coat over a diffuse base. Glossy, not a perfect mirror: a delta coat reflects lamps onto nearby
        # surfaces along paths only BSDF sampling can find (caustic fireflies); a glossy one light sampling handles.
        b.material(f"paint_{i}", {"type": "roughplastic", "diffuse_reflectance": rgb(*c), "int_ior": 1.5, "alpha": 0.06})
    b.material("car_glass", {"type": "plastic", "diffuse_reflectance": rgb(0.005, 0.006, 0.007), "int_ior": 1.52})
    b.material("car_trim", principled((0.02, 0.02, 0.022), 0.5))
    b.material("tyre", principled((0.022, 0.022, 0.024), 0.75))
    b.material("alloy", {"type": "roughconductor", "material": "Al", "alpha": 0.18})
    b.material("brake_disc", principled((0.18, 0.17, 0.16), 0.45, metallic=0.9))
    b.material("head_lens", {"type": "plastic", "diffuse_reflectance": tex(A.texture("lamp_head", lambda: P.lamp_texture("head"))), "int_ior": 1.5})
    b.material("tail_lens", {"type": "plastic", "diffuse_reflectance": tex(A.texture("lamp_tail", lambda: P.lamp_texture("tail"))), "int_ior": 1.5})
    b.material("reverse_lens", {"type": "plastic", "diffuse_reflectance": rgb(0.62, 0.62, 0.62), "int_ior": 1.5})
    b.material("bezel", {"type": "roughconductor", "material": "Cr", "alpha": 0.25})
    b.material("grille", principled((0.015, 0.015, 0.016), 0.35))
    b.material("seam", diffuse((0.0, 0.0, 0.0)))
    b.material("mirror", {"type": "conductor", "material": "Ag"})
    plates = As.texture("plates", lambda: np.flipud(P.plate_atlas()).copy())
    for k in range(P.PLATE_CELLS):
        uv = mi.ScalarTransform4f().translate([0.0, k / P.PLATE_CELLS, 0.0]).scale([1.0, 1.0 / P.PLATE_CELLS, 1.0])
        b.material(f"plate_{k}", principled(tex(plates, to_uv=uv), 0.35))


CAR_PARTS = {"body": None, "glass": "car_glass", "trim": "car_trim", "tyre": "tyre", "rim": "alloy", "disc": "brake_disc", "bezel": "bezel",
             "head_lens": "head_lens", "tail_lens": "tail_lens", "reverse_lens": "reverse_lens", "grille": "grille",
             "seam": "seam", "mirror": "mirror"}


def facade(b: SceneBuilder, Ad: Assets, idx: int, spec: dict, seed: int, M: np.ndarray) -> list:
    """Generate (or reuse) one building's meshes, register them, return its sign points in world space."""
    d = Path(Ad.root) / "facades" / f"b{idx:02d}"
    meta = d / "meta.json"
    if Ad.rebuild or not meta.exists():
        meshes, emit, signs = F.build_facade(spec, np.random.default_rng([seed, idx]))
        areas = {}
        for k, m in meshes.items():
            m.write_ply(d / f"{k}.ply")
            a, bb, c = (m.p[m.f[:, i]] for i in range(3))
            areas[k] = float(0.5 * np.linalg.norm(np.cross(bb - a, c - a), axis=1).sum())
        meta.write_text(json.dumps({"parts": sorted(meshes), "emit": emit, "area": areas, "signs": signs}))
    data = json.loads(meta.read_text())
    for k in data["parts"]:
        path = str(d / f"{k}.ply")
        if k.startswith("emit_"):
            L = data["emit"][k]
            share = INTERIOR_WEIGHT if k.startswith(INTERIOR) else 1.0
            b.ply(path, M, "emitter_backing", emission=tuple(L), prefix="facadelight", power=share * luminance(L) * data["area"][k])
        else:
            b.ply(path, M, k, prefix="facade")
    return [(M @ np.array([*p, 1.0]))[:3].tolist() for p in data["signs"]]


def build(ctx: BuildContext) -> SceneBundle:
    rng = np.random.default_rng(ctx.seed)
    wet = ctx.env == "wet"
    As, Ad = Assets(ctx.shared_dir, ctx.rebuild), Assets(ctx.assets_dir, ctx.rebuild)
    b = SceneBuilder(As)
    add_materials(b, As, Ad, wet)
    focus: dict[str, list] = {}

    # --- ground -----------------------------------------------------------------------------------
    b.box_mesh("road", (0.0, -0.1, -30.0), (2 * ROAD_HALF, 0.2, 80.0), "asphalt")
    drng = np.random.default_rng(1234)                                  # decals are seed-independent
    dec = P.Groups()
    for x0, x1, z0, z1 in PATCHES:
        dec["asphalt_patch"].add(P.quad((x0, 0.0015, z1), (x1, 0.0015, z1), (x1, 0.0015, z0), (x0, 0.0015, z0)))
    for _ in range(46):
        dec["crack"].add(P.crack(drng, (drng.uniform(-4.2, 4.2), 0.0, drng.uniform(-66, 8)), drng.uniform(0.6, 3.0)))
    for z0 in np.arange(Z_NEAR - 1.5, Z_FAR, -9.0):                     # centre dashes
        if not (CROSSING[0] - 3 < z0 - 1.5 < CROSSING[1] + 3):
            dec["paint"].add(P.worn_marking(-0.06, 0.06, z0 - 3.0, z0, 0.002, drng))
    for x in (-2.5, 2.5):                                               # bay lines
        for za, zb in ((CROSSING[1] + 2.5, Z_NEAR - 0.5), (Z_FAR + 1.0, CROSSING[0] - 2.5)):
            dec["paint"].add(P.worn_marking(x - 0.05, x + 0.05, za, zb, 0.002, drng, cell=0.025, wear=0.08))
    for x in np.arange(-3.75, 3.8, 1.0):                                # zebra
        dec["paint"].add(P.worn_marking(x - 0.25, x + 0.25, CROSSING[0], CROSSING[1], 0.0022, drng, wear=0.1))
    for zs in (CROSSING[1] + 1.2, CROSSING[0] - 1.2):                   # zig-zag-free give-way bars
        dec["paint"].add(P.worn_marking(-4.3, 4.3, zs - 0.1, zs + 0.1, 0.002, drng, cell=0.02, wear=0.08))
    for k, m in dec.meshes().items():
        path = As.mesh(f"decal_{k}", lambda m=m: m)
        b.ply(path, np.eye(4), k, prefix="decal")
    for x, z in MANHOLES:
        disc = As.mesh("manhole_disc", lambda: G.lathe([(0.0, 0.003), (0.33, 0.003), (0.35, 0.0)], 40))
        b.ply(disc, G.translate((x, 0.0, z)), "manhole", prefix="manhole")
    for x, z in GULLIES:
        b.rect(G.compose(G.translate((x, 0.0012, z)), G.rotate((1, 0, 0), -90.0), G.scale((0.22, 0.35, 1.0))), "gully_hole", prefix="gully")
        bars = As.mesh("gully_bars", lambda: P.join(*[P.box((0.40, 0.012, 0.02), (0.0, 0.006, zz)) for zz in np.arange(-0.32, 0.33, 0.06)]
                                                    + [P.box((0.44, 0.012, 0.03), (0.0, 0.006, s * 0.345)) for s in (-1, 1)]))
        b.ply(bars, G.translate((x, 0.0, z)), "iron", prefix="gully")

    # kerbs, pavements with drop kerbs at the crossing, tactile paving
    zz = np.array([CROSSING[0] - 1.5, CROSSING[0] - 1.4, CROSSING[0] - 0.2, CROSSING[1] + 0.2, CROSSING[1] + 1.4, CROSSING[1] + 1.5])
    drop = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    for s in (-1, 1):
        kerbs = P.Acc()
        for z0 in np.arange(Z_FAR, Z_NEAR, 0.92):
            hk = 0.15 - 0.12 * float(np.interp(z0 + 0.46, zz, drop))
            kerbs.add(P.box((KERB_W, hk, 0.91), (s * (ROAD_HALF + KERB_W / 2), hk / 2, z0 + 0.46)))
        b.ply(As.mesh(f"kerbs_{s}", lambda kerbs=kerbs: kerbs.mesh()), np.eye(4), "granite", prefix="kerb")
        xs = [ROAD_HALF + KERB_W, 5.9, 6.6, PAVE_OUT]
        for za, zb in ((Z_FAR, zz[0]), (zz[-1], Z_NEAR)):
            xm = 0.5 * (xs[0] + xs[-1])
            b.box_mesh(f"pave_{s}_{int(za)}", (s * xm, 0.075, 0.5 * (za + zb)), (xs[-1] - xs[0], 0.15, zb - za), "paving")
        ramp_x = np.array([1.0, 1.0, 0.0, 0.0])
        Pv = np.array([(s * x, 0.15 - 0.12 * dz * rx, z) for z, dz in zip(zz, drop) for x, rx in zip(xs, ramp_x)])
        f = G._grid_faces(len(zz), len(xs))
        mesh = G.Mesh(Pv, G.vertex_normals(Pv, f), np.column_stack([Pv[:, 0] * s, Pv[:, 2]]), f)
        if mesh.n[:, 1].mean() < 0:
            mesh = G.Mesh(Pv, -mesh.n, mesh.uv, f[:, ::-1])
        b.ply(As.mesh(f"pave_drop_{s}", lambda mesh=mesh: mesh), np.eye(4), "paving", prefix="pave")
        tx0, tx1 = sorted((s * (ROAD_HALF + KERB_W + 0.05), s * (ROAD_HALF + KERB_W + 0.85)))
        bumps = P.Acc()
        bumps.add(P.quad((tx0, 0.0305, CROSSING[1]), (tx1, 0.0305, CROSSING[1]), (tx1, 0.0305, CROSSING[0]), (tx0, 0.0305, CROSSING[0])))
        for x in np.arange(tx0 + 0.04, tx1, 0.066):
            for z in np.arange(CROSSING[0] + 0.05, CROSSING[1], 0.066):
                bumps.add(P.box((0.03, 0.006, 0.03), (x, 0.033, z)))
        b.ply(As.mesh(f"tactile_{s}", lambda bumps=bumps: bumps.mesh()), np.eye(4), "tactile", prefix="tactile")

    # --- buildings --------------------------------------------------------------------------------
    signs_by_side: dict[int, list] = {-1: [], 1: []}
    idx = 0
    for side, items in BUILDINGS.items():
        z = Z_NEAR
        for (w, storeys, style, wall, extra) in items:
            spec = dict(width=float(w), storeys=storeys, style=style, wall=wall, trim=STYLE_TRIM[style], seed=200 + idx,
                        u0=float(idx) * 3.7, **extra)
            zc = z - w / 2
            yaw = 90.0 if side < 0 else -90.0
            M = G.compose(G.translate((side * PAVE_OUT, 0.0, zc)), G.rotate((0, 1, 0), yaw))
            signs_by_side[side].append(facade(b, Ad, idx, spec, ctx.seed, M))
            b.cube((side * (PAVE_OUT + F.BODY_DEPTH + 6.0), F.layout(spec)["h"] / 2, zc), (12.0, F.layout(spec)["h"], w), "roof", prefix="body")
            z -= w
            idx += 1
    for zc, storeys, style, wall, yaw in END_CAPS:
        spec = dict(width=2 * PAVE_OUT, storeys=storeys, style=style, wall=wall, trim=STYLE_TRIM[style], seed=200 + idx, u0=float(idx) * 3.7)
        M = G.compose(G.translate((0.0, 0.0, zc)), G.rotate((0, 1, 0), yaw))
        facade(b, Ad, idx, spec, ctx.seed, M)
        dz = -1.0 if yaw == 0.0 else 1.0
        h = F.layout(spec)["h"]
        b.cube((0.0, h / 2, zc + dz * (F.BODY_DEPTH + 6.0)), (2 * PAVE_OUT, h, 12.0), "roof", prefix="body")
        idx += 1

    # --- lamps: heritage sodium lanterns on the left, LED cobra-heads on the right --------------------
    post, frame_m, panels = As.mesh("lantern_post", P.lantern_post), As.mesh("lantern_frame", P.lantern_frame), As.mesh("lantern_panels", P.lantern_panels)
    panel_area = 4 * (P.LANTERN_W - 0.024) * (P.LANTERN_H - 0.04)
    for z in LANTERNS:
        x = -(ROAD_HALF + 0.55)
        b.ply(post, G.translate((x, 0.0, z)), "lamp_black", prefix="lanternpost")
        b.ply(frame_m, G.translate((x, LANTERN_POST, z)), "lamp_black", prefix="lantern")
        L = tuple(LANTERN_L * c for c in SODIUM)
        b.ply(panels, G.translate((x, LANTERN_POST, z)), "emitter_backing", emission=L, prefix="lamp", power=luminance(L) * panel_area)
        focus[f"lamp_l{int(abs(z))}"] = [x, LANTERN_POST + 0.25, z + P.LANTERN_W / 2]
    cpost, chead = As.mesh("cobra_post", P.cobra_post), As.mesh("cobra_head", P.cobra_head)
    for z in COBRAS:
        x = ROAD_HALF + 0.6
        b.ply(cpost, G.compose(G.translate((x, 0.0, z)), G.rotate((0, 1, 0), 180.0)), "lamp_grey", prefix="cobrapost")
        hx = x - LAMP_REACH
        H = G.compose(G.translate((hx, LAMP_H + 0.14, z)), G.rotate((0, 1, 0), 180.0))
        b.ply(chead, H, "lamp_grey", prefix="cobrahead")
        lens = G.compose(H, G.translate((0.32, -0.004, 0.0)), G.rotate((1, 0, 0), 90.0), G.scale((0.30, 0.12, 1.0)))
        L = tuple(LED_L * c for c in LED)
        b.rect(lens, "emitter_backing", emission=L, prefix="lamp", power=luminance(L) * 0.6 * 0.24)
        focus[f"lamp_r{int(abs(z))}"] = [hx - 0.32, LAMP_H + 0.13, z]

    # --- trees with grates ---------------------------------------------------------------------------
    grate = As.mesh("tree_grate", P.tree_grate)
    for k, (x, z) in enumerate(TREES):
        parts = As.meshes(f"tree_{k}", lambda k=k: P.tree(k + 1))
        b.part_set(parts, G.compose(G.translate((x, 0.15, z)), G.rotate((0, 1, 0), 70.0 * k)), {"bark": "bark", "leaf": "leaf"})
        b.ply(grate, G.translate((x, 0.15, z)), "iron", prefix="grate")
        b.rect(G.compose(G.translate((x, 0.152, z)), G.rotate((1, 0, 0), -90.0), G.scale((0.58, 0.58, 1.0))), "soil", prefix="pit")
    focus["tree_r"] = [TREES[0][0], 2.0, TREES[0][1] + 0.17]

    # --- furniture -------------------------------------------------------------------------------------
    bol = As.meshes("bollard", P.bollard)
    for x, z in BOLLARDS:
        b.part_set(bol, G.translate((x, 0.15, z)), {"body": "lamp_black", "band": "bollard_band"})
    focus["bollard"] = [BOLLARDS[0][0], 1.0, BOLLARDS[0][1]]
    bins = As.meshes("bin", P.litter_bin)
    for x, z in BINS:
        b.part_set(bins, G.compose(G.translate((x, 0.15, z)), G.rotate((0, 1, 0), 90.0 if x < 0 else -90.0)), {"body": "bin_green", "lid": "lamp_black"})
    for x, z, kind, round_ in SIGNS:
        b.ply(As.mesh("sign_pole", lambda: P.pole(2.9)), G.translate((x, 0.15, z)), "lamp_grey", prefix="signpole")
        sp = As.meshes(f"plate_{'round' if round_ else 'square'}", lambda round_=round_: P.sign_plate(0.6, 0.6, round_))
        b.part_set(sp, G.translate((x, 2.75, z + 0.05)), {"face": f"sign_{kind}", "back": "sign_back"})
    for x, z in METERS:
        b.ply(As.mesh("meter", P.parking_meter), G.translate((x, 0.15, z)), "lamp_grey", prefix="meter")
    rx, rz = RACK
    b.ply(As.mesh("bike_rack", P.bike_rack), G.translate((rx, 0.15, rz)), "steel", prefix="rack")
    bike = As.meshes("bike", lambda: P.bicycle(1))
    b.part_set(bike, G.compose(G.translate((rx + 0.14, 0.15, rz + 0.9)), G.rotate((0, 1, 0), 4.0)),
               {"frame": "bike_frame", "tyre": "rubber", "metal": "steel", "saddle": "rubber"})
    focus["bike"] = [rx + 0.14, 0.85, rz + 0.9]

    # bus shelter (no glass panes: a delta-transmission pane would hide the lamps from light sampling)
    (x0, x1), (z0, z1) = SHELTER["x"], SHELTER["z"]
    zc = 0.5 * (z0 + z1)
    b.cube((0.5 * (x0 + x1), 2.62, zc), (x1 - x0, 0.08, z0 - z1), "shelter_metal", prefix="shelter")
    for xx in (x0 + 0.03, x1 - 0.03):
        for zz in (z0 - 0.03, z1 + 0.03):
            b.cube((xx, 1.38, zz), (0.06, 2.46, 0.06), "shelter_metal", prefix="shelter")
    for zz in (z0 - 0.02, z1 + 0.02):
        b.cube((0.5 * (x0 + x1), 2.45, zz), (x1 - x0, 0.12, 0.04), "shelter_metal", prefix="shelter")
    poster = str(Ad.root / "textures" / "poster.exr")
    if Ad.rebuild or not Path(poster).exists():
        art = G.abstract_art(512, 384, [(0.9, 0.3, 0.2), (0.2, 0.5, 0.9), (0.95, 0.85, 0.3), (0.2, 0.8, 0.5)], seed=ctx.seed + 31)
        P.save_exr(poster, art * 3.5)
    b.rect(G.compose(G.translate((x1 - 0.105, 1.35, zc)), G.rotate((0, 1, 0), -90.0), G.scale((1.0, 1.25, 1.0))), "emitter_backing",
           emission=tex(poster, raw=True), prefix="poster", power=3.5 * 0.3 * 2.5 * 2.0)
    b.cube((x1 - 0.06, 1.35, zc), (0.08, 2.6, 2.1), "shelter_metal", prefix="shelter")
    b.cube((x1 - 0.5, 0.45, zc), (0.5, 0.05, 2.4), "wood", prefix="bench")
    for zz in (z0 - 0.4, z1 + 0.4):
        b.cube((x1 - 0.5, 0.22, zz), (0.4, 0.44, 0.05), "shelter_metal", prefix="bench")
    focus["bus_shelter"] = [x1 - 0.11, 1.35, zc]

    # traffic signals
    head = As.meshes("signal_head", P.traffic_signal_head)
    green_left = bool(rng.random() < 0.5)
    for side in (-1, 1):
        sx = side * 4.85
        b.ply(As.mesh("signal_pole", lambda: P.pole(3.7, 0.055)), G.translate((sx, 0.15, -19.2)), "lamp_grey", prefix="signal")
        b.part_set(head, G.translate((sx, 3.2, -19.2)), {"body": "lamp_black", "hood": "lamp_black"})
        b.cube((sx, 1.1, -19.2 + 0.08), (0.12, 0.2, 0.1), "lamp_grey", prefix="pushbutton")
        go = (side < 0) == green_left
        for yy, name, colour in ((3.49, "red", (1.0, 0.05, 0.03)), (3.2, "amber", (1.0, 0.55, 0.05)), (2.91, "green", (0.1, 1.0, 0.45))):
            on = (name == "green") if go else (name == "red")
            b.sphere((sx, yy, -19.2 + 0.12), 0.1, "emitter_backing" if on else "signal_lens_off",
                     emission=tuple(SIGNAL_L * c for c in colour) if on else None, prefix="signal")
    focus["signal"] = [4.85, 3.49, -19.2 + 0.22]

    # street-name sign on the first right-hand building
    b.rect(G.compose(G.translate((PAVE_OUT - 0.02, 4.75, 7.2)), G.rotate((0, 1, 0), -90.0), G.scale((0.75, 0.16, 1.0))), "street_name", prefix="streetname")
    focus["street_sign"] = [PAVE_OUT - 0.03, 4.75, 7.2]

    # overhead: span wires between facades, tram contact wires, utility cables
    wires = P.Acc()
    for z in SPANS:
        wires.add(P.tube(P.catenary((-PAVE_OUT, 7.4, z), (PAVE_OUT, 7.6, z), 0.35), 0.008, 6, caps=False))
    for x in (-1.25, 1.25):
        pts = []
        for za, zb in zip(SPANS[:-1], SPANS[1:]):
            seg = P.catenary((x, 7.4 - 0.33, za), (x, 7.4 - 0.33, zb), 0.12, 16)
            pts.append(seg[:-1])
        wires.add(P.tube(np.vstack(pts + [[(x, 7.07, SPANS[-1])]]), 0.0065, 6, caps=False))
    crng = np.random.default_rng(77)
    for _ in range(5):
        za, zb = crng.uniform(-60, 8, 2)
        wires.add(P.tube(P.catenary((-PAVE_OUT, crng.uniform(8.5, 11.5), za), (PAVE_OUT, crng.uniform(8.5, 11.5), zb), crng.uniform(0.3, 0.9)), 0.011, 6, caps=False))
    b.ply(As.mesh("wires", lambda: wires.mesh()), np.eye(4), "cable", prefix="wires")
    focus["wire"] = [0.0, 7.5 - 0.35, SPANS[2]]

    # --- cars -------------------------------------------------------------------------------------------
    kinds = list(P.CAR_SPECS)
    models = {k: As.meshes(f"car_{k}", lambda k=k: P.car(k)) for k in kinds}
    anchors = {k: P.car_anchors(k) for k in kinds}

    def place(x, z, yaw, lit, kind=None):
        kind = kind or kinds[int(rng.choice(len(kinds), p=[0.42, 0.28, 0.22, 0.08]))]
        paint = f"paint_{int(rng.choice(len(PAINTS), p=PAINT_P / PAINT_P.sum()))}"
        M = G.compose(G.translate((x + rng.normal(0, 0.05), 0.0, z + rng.normal(0, 0.08))), G.rotate((0, 1, 0), yaw + rng.normal(0, 1.0)))
        b.part_set(models[kind], M, {k: (paint if v is None else v) for k, v in CAR_PARTS.items()})
        a = anchors[kind]
        for end, z_off, sgn in (("front", a["z_front"], 1.0), ("rear", a["z_rear"], -1.0)):
            px, py = a[f"plate_{end}"]
            hx, hy = a["plate_half"]
            pm = G.compose(M, G.translate((px, py, z_off + sgn * 0.012)), G.rotate((0, 1, 0), 0.0 if sgn > 0 else 180.0), G.scale((hx, hy, 1.0)))
            b.rect(pm, f"plate_{int(rng.integers(P.PLATE_CELLS))}", prefix="plate")
        if lit:
            # lit lamps follow the lens texture: two projector discs and a DRL strip, or six LED bars
            sgn, zo = (1.0, a["z_front"]) if lit == "head" else (-1.0, a["z_rear"])
            hx, hy = a["head_half"] if lit == "head" else a["tail_half"]
            pieces = []
            for (lx, ly) in (a["head"] if lit == "head" else a["tail"]):
                if lit == "head":                              # round projector bulbs, then the DRL strip
                    for s in (-1, 1):
                        c = (M @ np.array([lx + s * 0.4 * hx, ly - 0.1 * hy, zo - 0.012, 1.0]))[:3]
                        b.sphere(c, 0.03, "emitter_backing", emission=tuple(HEADLIGHT_L * 5.0 * k for k in HEADLIGHT), prefix="carlight")
                    pieces.append(((lx, ly + 0.76 * hy), (0.88 * hx, 0.07 * hy), (0.95, 0.97, 1.0), HEADLIGHT_L * 0.7))
                else:
                    pieces += [((lx + ((k + 0.5) / 6 - 0.5) * 2 * hx, ly), (0.06 * hx, 0.62 * hy), TAILLIGHT, TAILLIGHT_L) for k in range(6)]
            for (px, py), (qx, qy), colour, rad in pieces:
                lm = G.compose(M, G.translate((px, py, zo + sgn * 0.008)), G.rotate((0, 1, 0), 0.0 if sgn > 0 else 180.0), G.scale((qx, qy, 1.0)))
                Lc = tuple(rad * c for c in colour)
                b.rect(lm, "emitter_backing", emission=Lc, prefix="carlight", power=luminance(Lc) * 4 * qx * qy)
        return kind

    for x, bays, yaw in ((3.45, RIGHT_BAYS, 0.0), (-3.45, LEFT_BAYS, 180.0)):
        for z in bays:
            if rng.random() < 0.85 or z in (RIGHT_BAYS[1], LEFT_BAYS[2]):
                place(x, z, yaw, None)
    for x, z, yaw, light in MOVING:
        place(x, z, yaw, light)
    focus["car_r1"] = [3.45, 0.8, RIGHT_BAYS[1] + 1.9]                  # on the bonnet of the forced bay
    focus["car_l2"] = [-3.45, 0.8, LEFT_BAYS[2] - 1.9]

    # --- sky ----------------------------------------------------------------------------------------------
    sky = str(Ad.root / "textures" / "skyglow.exr")
    if Ad.rebuild or not Path(sky).exists():
        P.save_exr(sky, P.skyglow(ctx.env))
    b.d["sky"] = {"type": "envmap", "filename": sky, "scale": 1.0, "sampling_weight": max(SKY_WEIGHT * b.lamp_weight, 1e-3)}

    focus["far_facade"] = [0.0, 8.0, Z_FAR + 0.05]
    focus["crosswalk"] = [0.0, 0.0, -20.0]
    focus["shop_sign_r"] = signs_by_side[1][1][0]
    focus["shop_sign_l"] = signs_by_side[-1][2][0]
    return SceneBundle(b.d, focus, {"buildings": idx, "puddle_hero": list(HERO_PUDDLE) if wet else None})


def exclude_boxes() -> tuple:
    boxes = []
    for x, bays in ((3.45, RIGHT_BAYS), (-3.45, LEFT_BAYS)):
        for z in bays:
            boxes.append(((x - 1.4, 0.0, z - 3.0), (x + 1.4, 2.4, z + 3.0)))
    for x, zs in ((-(ROAD_HALF + 0.55), LANTERNS), (ROAD_HALF + 0.6, COBRAS)):
        for z in zs:
            boxes.append(((x - 0.5, 0.0, z - 0.5), (x + 0.5, 9.0, z + 0.5)))
    for x, z in TREES:
        boxes.append(((x - 0.8, 0.0, z - 0.8), (x + 0.8, 3.6, z + 0.8)))
    for x, z in BOLLARDS + BINS + METERS + [(s[0], s[1]) for s in SIGNS]:
        boxes.append(((x - 0.45, 0.0, z - 0.45), (x + 0.45, 3.0, z + 0.45)))
    boxes.append(((SHELTER["x"][0] - 0.3, 0.0, SHELTER["z"][0] - 0.4), (SHELTER["x"][1] + 0.3, 3.0, SHELTER["z"][1] + 0.4)))
    boxes.append(((RACK[0] - 0.6, 0.0, RACK[1] - 0.6), (RACK[0] + 0.8, 1.5, RACK[1] + 2.4)))
    for s in (-1, 1):
        boxes.append(((s * 4.85 - 0.5, 0.0, -19.9), (s * 4.85 + 0.5, 4.2, -18.5)))
    return tuple(boxes)


SCENE = SceneDef(
    id="night_street", group="artificial", owner="rui",
    description="City street at night: lit rooms behind real windows, shopfronts and neon, sodium lanterns and LED lamps, cars, wet asphalt.",
    build=build,
    views={
        # Down the street from the right-hand parking lane, lamps receding.
        "lamps": View((3.9, 1.5, 7.0), (-1.0, 3.2, -50.0), focus="lamp_r16"),
        # Low across the (puddled) road toward a parked car.
        "puddle_car": View((-2.2, 0.45, 3.5), (2.5, 1.0, -9.0), focus="car_r1"),
        # From the left kerb, across the road to the shopfronts and the bus shelter.
        "kerb_wide": View((-5.6, 1.15, 4.0), (3.0, 2.6, -16.0), focus="bus_shelter"),
        # From the middle of the road at eye height, onto the right-hand shopfronts.
        "shopfronts": View((-1.0, 1.6, -2.0), (7.5, 2.6, -14.0), focus="shop_sign_r"),
    },
    default_view="lamps",
    camera_box=((-6.2, 0.3, -8.0), (6.2, 3.5, 7.0)),
    target_box=((-7.6, 0.0, -68.5), (7.6, 14.0, 10.5)),
    exclude_boxes=exclude_boxes(),
    envs=("clear", "wet"),
    tags=("outdoor", "night", "bokeh", "specular", "architecture", "artificial_light"),
    default_seed=3, asset_version=11, max_depth=8, rr_depth=5, spp_hint=4096,
)
