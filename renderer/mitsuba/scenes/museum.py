"""A museum gallery for Mitsuba 3: free-standing glass vitrines (real dielectric) are the point.

Units are meters, y up, the visitor enters at +z and looks toward -z.

  gallery 1   x in [-4, 4], z in [-12, 2], 5 m high; polished limestone floor, framed paintings with
              wall labels, a laylight (frosted ceiling panel) and track spotlights; four vitrines (two
              tall cases with a lit hood, two flat table cases), three sculptures on plinths, a bench,
              a rope barrier in front of the big canvas, and an introductory wall text
  doorway     2.4 x 3.4 m opening in the back wall into gallery 2
  gallery 2   x in [-3.5, 3.5], z in [-20, -12.4], 4.2 m high; oak floor, darker walls, a sculpture

Light inside the glass: Mitsuba's path tracer cannot connect a shading point to a light through a
dielectric, so a case lit only from the room would be noise. Each tall case has a diffuse light panel
under its opaque hood, INSIDE the glass; each flat case sees the laylight through its glass top (a large,
dim emitter, which BSDF sampling finds easily). Small spots inside the cases were tried and removed: with
one light picked per bounce in proportion to power, a delta light gets so few samples it sparkles.
The glass itself stays a real dielectric slab (8 mm, ior 1.5): its reflections and refraction are what
this scene is for.

Seed: which paintings hang where (from a seed-independent library), wall colours, case finishes and
cloth, which artefacts go into which case, which sculpture stands on which plinth, small jitter.
"""

from __future__ import annotations

import math

import numpy as np

import mitsuba as mi
import procedural as G
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, bitmap, luminance, principled, rgb
from scenes import _museum_props as M

X1, Z_BACK, Z_FRONT, H1 = 4.0, -12.0, 2.0, 5.0         # gallery 1 (x in [-X1, X1])
WALL_T = 0.4
X2, Z2_BACK, H2 = 3.5, -20.0, 4.2                        # gallery 2 (x in [-X2, X2])
Z2_FRONT = Z_BACK - WALL_T
DOOR_W, DOOR_H = 2.4, 3.4
TRACK_X1, TRACK_Y1 = 2.4, H1 - 0.03       # 1.6 m off the walls: beams meet the paintings at about 30 degrees from vertical
TRACK_X2, TRACK_Y2 = 1.9, H2 - 0.03
LAYLIGHT = 3.0                       # radiance of the frosted ceiling panels
HOOD_LIGHT = 2.6                     # radiance of the light panel under each tall case hood
SPOT_E = 1.5                         # irradiance a track spot puts on its painting
# Mitsuba picks one light per bounce in proportion to sampling_weight. Each spot lights only its own pool,
# so at pure power share (~2% each) the pools are grainy; spots and case hoods get this multiple of it.
SPOT_WEIGHT_BOOST = 2.5
WARM = np.array([1.0, 0.93, 0.8])

# Painting library, by aspect class: (width px, height px) and the pictures in it. Seed-independent.
LIBRARY = {
    "P": ((788, 1024), ["portrait", "portrait", "still_life", "colour_field", "geometric", "gestural", "portrait", "blobs"]),
    "L": ((1024, 770), ["landscape", "seascape", "still_life", "blobs", "gestural", "landscape", "geometric"]),
    "W": ((1024, 724), ["landscape", "seascape", "colour_field", "gestural"]),
}
# Hanging slots: name, wall, position along the wall, centre height, canvas w, h, aspect class.
SLOTS = [
    ("L1", "L", -1.2, 1.65, 0.9, 1.15, "P"), ("L2", "L", -5.5, 2.05, 1.5, 1.12, "L"),
    ("L3", "L", -8.6, 1.65, 1.0, 1.28, "P"), ("L4", "L", -10.9, 1.6, 0.62, 0.8, "P"),
    ("R1", "R", -1.5, 1.65, 1.0, 0.76, "L"), ("R2", "R", -5.5, 2.05, 1.25, 0.94, "L"),
    ("R3", "R", -9.5, 1.75, 2.2, 1.55, "W"), ("B1", "B", 2.6, 1.75, 1.1, 1.4, "P"),
    ("E1", "E", -2.0, 1.65, 1.2, 0.9, "L"), ("E2", "E", 2.0, 1.65, 0.9, 1.18, "P"),
    ("G2B", "GB", 0.0, 1.85, 2.4, 1.7, "W"), ("G2L1", "GL", -14.8, 1.65, 1.0, 1.3, "P"),
    ("G2L2", "GL", -18.0, 1.65, 1.3, 1.0, "L"), ("G2R1", "GR", -14.8, 1.65, 1.3, 1.0, "L"),
    ("G2R2", "GR", -18.0, 1.65, 1.0, 1.3, "P"),
]
VITRINES = {  # name: (kind, centre x, centre z, width x, depth z)
    "v1": ("tall", -1.7, -3.0, 1.0, 0.6), "v2": ("tall", 1.7, -3.0, 1.0, 0.6),
    "v3": ("flat", -1.7, -8.0, 1.2, 0.7), "v4": ("flat", 1.7, -8.0, 1.2, 0.7),
}
TALL_BASE, TALL_GLASS, TALL_HOOD = 0.9, 0.55, 0.14
FLAT_BASE, FLAT_GLASS = 0.85, 0.28
PANE = 0.008
PLINTHS = {"p1": (-3.0, -5.5), "p2": (3.0, -5.5), "p3": (-3.0, -10.4), "g2": (0.0, -16.5)}
# Colours below are sRGB display values: procedural textures are written as 8-bit PNG and Mitsuba's bitmap
# (raw=False) decodes them as sRGB, so a "linear" 0.25 would come out as albedo 0.05 (near black).
WALL_PALETTES = {
    "g1": [("warm_white", (0.88, 0.86, 0.81)), ("pale_grey", (0.78, 0.78, 0.76)), ("sage", (0.68, 0.72, 0.64))],
    "feature": [("teal", (0.12, 0.38, 0.40)), ("oxblood", (0.46, 0.13, 0.13)), ("navy", (0.16, 0.22, 0.40))],
    "g2": [("crimson", (0.56, 0.16, 0.17)), ("forest", (0.26, 0.40, 0.31)), ("slate", (0.40, 0.46, 0.53))],
}
CLOTHS = [("blue", (0.13, 0.19, 0.40)), ("oxblood", (0.44, 0.09, 0.11)), ("linen", (0.70, 0.66, 0.58)), ("charcoal", (0.18, 0.18, 0.19))]


def rng_for(ctx: BuildContext, k: int) -> np.random.Generator:
    return np.random.default_rng([ctx.seed, k])


def place(wall: str, along: float) -> np.ndarray:
    """Frame on a wall: local x to the viewer's right, y up, z out of the wall (wall surface at z = 0)."""
    if wall == "L":
        return G.compose(G.translate((-X1, 0, along)), G.rotate((0, 1, 0), 90))
    if wall == "R":
        return G.compose(G.translate((X1, 0, along)), G.rotate((0, 1, 0), -90))
    if wall == "B":
        return G.translate((along, 0, Z_BACK))
    if wall == "E":
        return G.compose(G.translate((along, 0, Z_FRONT)), G.rotate((0, 1, 0), 180))
    if wall == "GB":
        return G.translate((along, 0, Z2_BACK))
    if wall == "GL":
        return G.compose(G.translate((-X2, 0, along)), G.rotate((0, 1, 0), 90))
    if wall == "GR":
        return G.compose(G.translate((X2, 0, along)), G.rotate((0, 1, 0), -90))
    raise ValueError(wall)


def pt(m: np.ndarray, local) -> np.ndarray:
    return (m @ np.array([*local, 1.0]))[:3]


_MEMO: dict = {}


def memo(key, fn):
    if key not in _MEMO:
        _MEMO[key] = fn()
    return _MEMO[key]


# ----------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------

def add_materials(b: SceneBuilder, ctx: BuildContext) -> dict:
    A = b.assets
    rng = rng_for(ctx, 0)
    pick = lambda opts: opts[int(rng.integers(len(opts)))]
    pal = {k: pick(v) for k, v in WALL_PALETTES.items()}
    cloth = [CLOTHS[i] for i in rng.permutation(len(CLOTHS))]
    case_finish = "walnut" if rng.random() < 0.5 else "white"

    stone = lambda: memo("limestone", lambda: M.limestone(1920, 1920, 320))
    b.material("floor_stone", principled(bitmap(A.texture("limestone_albedo", lambda: stone()[0]), uv_scale=(2 * X1 / 4.8, (Z_FRONT - Z_BACK) / 4.8)),
                                         bitmap(A.texture("limestone_rough", lambda: stone()[1], gray=True), raw=True,
                                                uv_scale=(2 * X1 / 4.8, (Z_FRONT - Z_BACK) / 4.8)), specular=0.5))
    oak = lambda: memo("oak", lambda: G.wood_planks(2048, 2048, planks=32, board_len=0.25, seed=5))
    b.material("floor_oak", principled(bitmap(A.texture("oak_albedo", lambda: np.clip(oak()[0] * np.array([1.45, 1.32, 1.12]), 0, 1)),
                                              uv_scale=(2 * X2 / 4.0, (Z2_FRONT - Z2_BACK) / 4.0)),
                                       bitmap(A.texture("oak_rough", lambda: oak()[1], gray=True), raw=True,
                                              uv_scale=(2 * X2 / 4.0, (Z2_FRONT - Z2_BACK) / 4.0)), clearcoat=0.3, clearcoat_gloss=0.25))
    for key in ("g1", "feature", "g2"):
        name, col = pal[key]
        b.material(f"wall_{key}", {"type": "diffuse", "reflectance": bitmap(A.texture(
            f"plaster_{name}", lambda col=col, name=name: G.plaster(1024, 1024, col, seed=7 + len(name), strength=0.05)))})
    b.material("ceiling", {"type": "diffuse", "reflectance": rgb(0.86, 0.86, 0.84)})
    b.material("paint_white", principled((0.86, 0.86, 0.84), 0.55))
    b.material("shadow_gap", {"type": "diffuse", "reflectance": rgb(0.03, 0.03, 0.03)})
    b.material("skirting", principled((0.16, 0.16, 0.16), 0.5))
    b.material("trim_stone", principled((0.62, 0.59, 0.53), 0.45))
    b.material("emitter_backing", {"type": "diffuse", "reflectance": rgb(0.0, 0.0, 0.0)})
    b.material("track_black", principled((0.02, 0.02, 0.022), 0.4))
    b.material("spot_body", principled((0.03, 0.03, 0.032), 0.35, specular=0.5))
    b.material("spot_lens", principled((0.9, 0.88, 0.82), 0.08, specular=0.8))

    b.material("glass", {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0,
                         "specular_transmittance": rgb(0.97, 0.99, 0.98)})
    if case_finish == "walnut":
        b.material("case_body", principled((0.13, 0.075, 0.04), 0.4, clearcoat=0.25, clearcoat_gloss=0.25))
    else:
        b.material("case_body", principled((0.82, 0.82, 0.8), 0.4, clearcoat=0.2, clearcoat_gloss=0.25))
    b.material("case_metal", principled((0.55, 0.45, 0.32), 0.3, metallic=1.0))
    for i, (name, col) in enumerate(cloth[:4]):
        b.material(f"deck_{i}", {"type": "diffuse", "reflectance": bitmap(A.texture(f"cloth_{name}", lambda col=col: M.fabric(1024, 1024, col)))})
    b.material("velvet_red", principled((0.42, 0.03, 0.05), 0.85, sheen=1.0, sheen_tint=0.5))
    b.material("velvet_dark", principled((0.04, 0.04, 0.05), 0.85, sheen=0.8))
    b.material("brass", principled((0.8, 0.62, 0.34), 0.22, metallic=1.0))
    b.material("steel", principled((0.55, 0.55, 0.56), 0.3, metallic=1.0))
    b.material("leather", principled((0.04, 0.035, 0.03), 0.45, specular=0.5))

    b.material("gilt", principled((0.86, 0.66, 0.32), 0.3, metallic=1.0))
    b.material("frame_black", principled((0.02, 0.02, 0.02), 0.35, specular=0.5))
    b.material("frame_oak", principled((0.48, 0.33, 0.19), 0.5))
    b.material("canvas_side", {"type": "diffuse", "reflectance": rgb(0.72, 0.68, 0.6)})
    b.material("label_edge", principled((0.9, 0.9, 0.9), 0.3))

    b.material("blackfig", principled(bitmap(A.texture("blackfig", lambda: M.black_figure(1024, 1024))), 0.4, specular=0.5))
    b.material("jug_glaze", principled((0.18, 0.08, 0.04), 0.2, clearcoat=0.4, clearcoat_gloss=0.4))
    b.material("celadon", principled((0.48, 0.62, 0.52), 0.2, clearcoat=0.5, clearcoat_gloss=0.4))
    b.material("gold", {"type": "roughconductor", "material": "Au", "alpha": 0.12})
    for metal, tint in (("gold", (1.0, 0.78, 0.36)), ("silver", (0.92, 0.92, 0.9)), ("bronze", (0.72, 0.48, 0.28))):
        b.material(f"coin_{metal}", principled(bitmap(A.texture(f"coin_{metal}", lambda tint=tint: np.clip(
            M.coin_face(256)[..., None] * np.array(tint), 0, 1))), 0.28, metallic=1.0))
    b.material("gem_red", {"type": "dielectric", "int_ior": 1.77, "specular_transmittance": rgb(0.85, 0.12, 0.16)})
    b.material("gem_blue", {"type": "dielectric", "int_ior": 1.77, "specular_transmittance": rgb(0.12, 0.25, 0.85)})
    b.material("fossil", principled(bitmap(A.texture("fossil", lambda: M.fossil_stone(512, 512))), 0.6))
    b.material("slab", principled(bitmap(A.texture("slab", lambda: M.fossil_stone(512, 512, seed=43) * 0.8)), 0.75))
    b.material("acrylic", {"type": "thindielectric", "int_ior": 1.49})
    b.material("parchment", {"type": "diffuse", "reflectance": bitmap(A.texture(
        "manuscript", lambda: M.manuscript_spread(1100, 1600, 91)))})
    b.material("book_board", principled((0.22, 0.08, 0.05), 0.55))
    b.material("page_block", {"type": "diffuse", "reflectance": rgb(0.78, 0.7, 0.54)})

    b.material("bronze_polish", principled((0.86, 0.62, 0.36), 0.18, metallic=1.0))
    b.material("bronze_patina", {"type": "blendbsdf",
                                 "weight": bitmap(A.texture("patina", lambda: M.patina_mask(512, 512), gray=True), raw=True),
                                 "bsdf_0": principled(bitmap(A.texture("bronze", lambda: M.bronze_albedo(512, 512))), 0.3, metallic=1.0),
                                 "bsdf_1": principled((0.17, 0.27, 0.2), 0.75)})
    b.material("steel_polish", {"type": "roughconductor", "material": "Cr", "alpha": 0.1})
    b.material("marble", principled(bitmap(A.texture("marble", lambda: G.marble(1024, 1024, seed=5))), 0.24, specular=0.5))
    b.material("drum_stone", principled((0.4, 0.39, 0.37), 0.7))
    return {"case_finish": case_finish, "walls": {k: v[0] for k, v in pal.items()}, "cloth": [c[0] for c in cloth[:4]]}


# ----------------------------------------------------------------------------
# Architecture
# ----------------------------------------------------------------------------

def wall_rect(b: SceneBuilder, wall: str, along: float, y: float, half_w: float, half_h: float, mat: str, off: float = 0.0):
    b.rect(G.compose(place(wall, along), G.translate((0, y, off)), G.scale((half_w, half_h, 1))), mat, prefix="wall")


def add_laylight(b: SceneBuilder, x0, x1, z0, z1, y_ceil, depth, rx0, rx1, rz0, rz1, radiance: float, tag: str, mullion=0.9):
    """Ceiling over [rx0, rx1] x [rz0, rz1] at y_ceil with a recessed frosted panel over [x0, x1] x [z0, z1]."""
    down = G.rotate((1, 0, 0), 90)
    for (ax0, ax1, az0, az1) in ((rx0, x0, rz0, rz1), (x1, rx1, rz0, rz1), (x0, x1, z1, rz1), (x0, x1, rz0, z0)):
        b.rect(G.compose(G.translate(((ax0 + ax1) / 2, y_ceil, (az0 + az1) / 2)), down,
                         G.scale(((ax1 - ax0) / 2, (az1 - az0) / 2, 1))), "ceiling", prefix="ceiling")
    cx, cz, hx, hz = (x0 + x1) / 2, (z0 + z1) / 2, (x1 - x0) / 2, (z1 - z0) / 2
    em = tuple(radiance * c for c in (1.0, 0.985, 0.95))
    b.rect(G.compose(G.translate((cx, y_ceil + depth, cz)), down, G.scale((hx, hz, 1))), "emitter_backing",
           emission=em, prefix=f"laylight_{tag}", power=luminance(em) * 4 * hx * hz)
    ym = y_ceil + depth / 2
    b.rect(G.compose(G.translate((x0, ym, cz)), G.rotate((0, 1, 0), 90), G.scale((hz, depth / 2, 1))), "ceiling", prefix="recess")
    b.rect(G.compose(G.translate((x1, ym, cz)), G.rotate((0, 1, 0), -90), G.scale((hz, depth / 2, 1))), "ceiling", prefix="recess")
    b.rect(G.compose(G.translate((cx, ym, z0)), G.scale((hx, depth / 2, 1))), "ceiling", prefix="recess")
    b.rect(G.compose(G.translate((cx, ym, z1)), G.rotate((0, 1, 0), 180), G.scale((hx, depth / 2, 1))), "ceiling", prefix="recess")
    nx = max(1, int(round((x1 - x0) / mullion)))
    nz = max(1, int(round((z1 - z0) / (mullion * 1.25))))
    for i in range(1, nx):
        b.cube((x0 + i * (x1 - x0) / nx, y_ceil + 0.03, cz), (0.045, 0.06, z1 - z0), "paint_white")
    for j in range(1, nz):
        b.cube((cx, y_ceil + 0.03, z0 + j * (z1 - z0) / nz), (x1 - x0, 0.06, 0.045), "paint_white")


def add_architecture(b: SceneBuilder) -> None:
    up = G.rotate((1, 0, 0), -90)
    lz = Z_FRONT - Z_BACK
    b.rect(G.compose(G.translate((0, 0, (Z_FRONT + Z_BACK) / 2)), up, G.scale((X1, lz / 2, 1))), "floor_stone", prefix="floor")
    l2 = Z2_FRONT - Z2_BACK
    b.rect(G.compose(G.translate((0, 0, (Z2_FRONT + Z2_BACK) / 2)), up, G.scale((X2, l2 / 2, 1))), "floor_oak", prefix="floor")
    b.rect(G.compose(G.translate((0, 0.001, Z_BACK - WALL_T / 2)), up, G.scale((DOOR_W / 2, WALL_T / 2, 1))), "trim_stone", prefix="threshold")

    # Gallery 1 walls; the back wall is solid blocks around the doorway so its reveals are real.
    wall_rect(b, "L", (Z_FRONT + Z_BACK) / 2, H1 / 2, lz / 2, H1 / 2, "wall_g1")
    wall_rect(b, "R", (Z_FRONT + Z_BACK) / 2, H1 / 2, lz / 2, H1 / 2, "wall_g1")
    wall_rect(b, "E", 0.0, H1 / 2, X1, H1 / 2, "wall_g1")
    seg = X1 - DOOR_W / 2
    zc = Z_BACK - WALL_T / 2
    for sx in (-1, 1):
        b.cube((sx * (DOOR_W / 2 + seg / 2), H1 / 2, zc), (seg, H1, WALL_T), "wall_feature")
    b.cube((0, (DOOR_H + H1) / 2, zc), (DOOR_W, H1 - DOOR_H, WALL_T), "wall_feature")
    # Architrave round the doorway (gallery 1 side).
    for sx in (-1, 1):
        b.cube((sx * (DOOR_W / 2 + 0.07), DOOR_H / 2 + 0.05, Z_BACK + 0.02), (0.14, DOOR_H + 0.1, 0.04), "trim_stone")
    b.cube((0, DOOR_H + 0.11, Z_BACK + 0.02), (DOOR_W + 0.28, 0.14, 0.04), "trim_stone")

    # Gallery 2: walls (its side of the dividing wall gets its own colour) and the room's far end.
    l2c = (Z2_FRONT + Z2_BACK) / 2
    wall_rect(b, "GL", l2c, H2 / 2, l2 / 2, H2 / 2, "wall_g2")
    wall_rect(b, "GR", l2c, H2 / 2, l2 / 2, H2 / 2, "wall_g2")
    wall_rect(b, "GB", 0.0, H2 / 2, X2, H2 / 2, "wall_g2")
    face_back = G.rotate((0, 1, 0), 180)
    seg2 = X2 - DOOR_W / 2
    for sx in (-1, 1):
        b.rect(G.compose(G.translate((sx * (DOOR_W / 2 + seg2 / 2), H2 / 2, Z2_FRONT - 0.001)), face_back, G.scale((seg2 / 2, H2 / 2, 1))),
               "wall_g2", prefix="wall")
    b.rect(G.compose(G.translate((0, (DOOR_H + H2) / 2, Z2_FRONT - 0.001)), face_back, G.scale((DOOR_W / 2, (H2 - DOOR_H) / 2, 1))),
           "wall_g2", prefix="wall")

    # Skirting and cornices.
    runs = [("L", (Z_FRONT + Z_BACK) / 2, lz, H1), ("R", (Z_FRONT + Z_BACK) / 2, lz, H1), ("E", 0.0, 2 * X1, H1),
            ("GL", l2c, l2, H2), ("GR", l2c, l2, H2), ("GB", 0.0, 2 * X2, H2)]
    for wall, along, length, h in runs:
        m = place(wall, along)
        b.ply(b.assets.mesh(f"skirt_{length:.2f}", lambda length=length: G.box((length, 0.11, 0.016), (0, 0.055, 0.008))), m, "skirting", prefix="skirt")
        b.ply(b.assets.mesh(f"cornice_{length:.2f}_{h}", lambda length=length, h=h: M.join(
            G.box((length, 0.12, 0.05), (0, h - 0.06, 0.025)), G.box((length, 0.05, 0.09), (0, h - 0.145, 0.045)))), m, "paint_white", prefix="cornice")
    for sx in (-1, 1):
        m = place("B", sx * (DOOR_W / 2 + seg / 2))
        b.ply(b.assets.mesh(f"skirt_b_{seg:.2f}", lambda: G.box((seg, 0.11, 0.016), (0, 0.055, 0.008))), m, "skirting", prefix="skirt")

    # Ceilings with laylights.
    add_laylight(b, -1.8, 1.8, -10.6, 0.6, H1, 0.4, -X1, X1, Z_BACK, Z_FRONT, LAYLIGHT, "g1")
    add_laylight(b, -1.5, 1.5, -18.8, -13.0, H2, 0.3, -X2, X2, Z2_BACK, Z2_FRONT, LAYLIGHT, "g2")   # reaches the dividing wall
    for sx in (-1, 1):
        b.cube((sx * TRACK_X1, TRACK_Y1 + 0.015, (Z_FRONT + Z_BACK) / 2), (0.035, 0.03, lz - 0.6), "track_black")
        b.cube((sx * TRACK_X2, TRACK_Y2 + 0.015, l2c), (0.035, 0.03, l2 - 0.6), "track_black")
    b.focus_points["doorway"] = [0.0, 1.7, Z_BACK - WALL_T / 2]


# ----------------------------------------------------------------------------
# Lights
# ----------------------------------------------------------------------------

class Spots:
    def __init__(self, b: SceneBuilder):
        self.b, self.n = b, 0
        self.parts = b.assets.meshes("spot_head", M.spot_head)

    def add(self, track_y: float, px: float, pz: float, target, e: float = SPOT_E, cutoff: float | None = None,
            reach: float = 0.6, lens: bool = True) -> list:
        b = self.b
        pivot = np.array([px, track_y - 0.16, pz])
        target = np.asarray(target, float)
        d = target - pivot
        dist = float(np.linalg.norm(d))
        d /= dist
        head = G.compose(G.translate(pivot), M.rot_between((0, -1, 0), d))
        b.ply(b.assets.mesh(f"spot_mount_{self.n}_{px:.2f}_{pz:.2f}_{track_y:.2f}", lambda: M.spot_mount(track_y, pivot)),
              np.eye(4), "spot_body", prefix="spot_mount")
        b.ply(self.parts["can"], head, "spot_body", prefix="spot_can")
        if lens:
            # Not an emitter: a small bright lens seen through glass or a glossy floor is a firefly factory.
            b.ply(self.parts["lens"], head, "spot_lens", prefix="spot_lens")
        cutoff = cutoff or min(40.0, math.degrees(math.atan(reach / dist)) + 5.0)
        intensity = e * dist ** 2
        origin = pivot + d * 0.21
        up = [0, 1, 0] if abs(d[1]) < 0.98 else [1, 0, 0]
        b.d[f"spot_{self.n:03d}"] = {
            "type": "spot", "to_world": mi.ScalarTransform4f().look_at(origin=origin.tolist(), target=target.tolist(), up=up),
            "intensity": rgb(*(intensity * WARM)), "cutoff_angle": float(cutoff), "beam_width": float(0.75 * cutoff),
            "sampling_weight": float(SPOT_WEIGHT_BOOST * 2 * intensity * (1 - math.cos(math.radians(cutoff))))}
        self.n += 1
        return (pivot + d * 0.2).tolist()


# ----------------------------------------------------------------------------
# Paintings, labels, wall text
# ----------------------------------------------------------------------------

def add_paintings(b: SceneBuilder, ctx: BuildContext, spots: Spots) -> None:
    A = b.assets
    rng = rng_for(ctx, 1)
    order = {cls: list(rng.permutation(len(items))) for cls, (_, items) in LIBRARY.items()}
    used = {cls: 0 for cls in LIBRARY}
    for name, wall, along, yc, w, h, cls in SLOTS:
        idx = int(order[cls][used[cls]])
        used[cls] += 1
        (tw, th), kinds = LIBRARY[cls]
        kind = kinds[idx]
        pid = f"{cls}{idx}"
        if f"art_{pid}" not in b.d:
            old = kind in M.OLD
            tex = A.texture(f"art_{pid}", lambda kind=kind, idx=idx, cls=cls: np.flipud(M.PAINTERS[kind](th, tw, 1000 + 100 * "PLW".index(cls) + idx)))
            b.material(f"art_{pid}", principled(bitmap(tex), 0.32 if old else 0.75, specular=0.45 if old else 0.25))
            b.material(f"label_{pid}", principled(bitmap(A.texture(f"label_{pid}", lambda idx=idx, cls=cls: np.flipud(
                M.label_texture(5000 + 100 * "PLW".index(cls) + idx, brass=False)))), 0.3))
        old = kind in M.OLD
        style = ("gilt" if rng.random() < 0.75 else "oak") if old else ("float" if kind == "colour_field" else ("black" if rng.random() < 0.6 else "oak"))
        fmat = {"gilt": "gilt", "oak": "frame_oak", "black": "frame_black", "float": "frame_black"}[style]
        m = G.compose(place(wall, along), G.translate((0, yc, 0.012)))
        cv = A.meshes(f"canvas_{round(w * 100)}x{round(h * 100)}", lambda w=w, h=h: M.canvas_box(w, h, 0.032))
        b.ply(cv["face"], m, f"art_{pid}", prefix="canvas")
        b.ply(cv["sides"], m, "canvas_side", prefix="canvas_side")
        b.ply(A.mesh(f"frame_{style}_{w:.2f}x{h:.2f}", lambda w=w, h=h, style=style: M.frame_moulding(w, h, style)),
              G.compose(m, G.translate((0, 0, 0.0 if style != "float" else -0.01))), fmat, prefix="frame")
        fo = M.FRAME_PROFILES[style][-1][0]
        lab = G.compose(place(wall, along), G.translate((w / 2 + fo + 0.2, 1.42 if h < 1.3 else 1.25, 0.004)))
        b.ply(A.mesh("label_block", lambda: G.box((0.16, 0.1, 0.005), (0, 0, -0.0005))), lab, "label_edge", prefix="label_block")
        b.rect(G.compose(lab, G.translate((0, 0, 0.0021)), G.scale((0.08, 0.05, 1))), f"label_{pid}", prefix="label")
        centre = pt(m, (0, 0, 0.04))
        track_x, track_y = (TRACK_X2, TRACK_Y2) if wall.startswith("G") else (TRACK_X1, TRACK_Y1)
        if wall in ("L", "GL"):
            sp = spots.add(track_y, -track_x, along + 0.35, centre, reach=0.6 * max(w, h) + 0.1)
        elif wall in ("R", "GR"):
            sp = spots.add(track_y, track_x, along + 0.35, centre, reach=0.6 * max(w, h) + 0.1)
        elif wall == "B":
            sp = spots.add(track_y, track_x if along > 0 else -track_x, Z_BACK + 1.6, centre, reach=0.6 * max(w, h) + 0.1)
        elif wall == "E":
            pass     # behind the usual viewpoints: the laylight alone, so no spot spends light samples here
        else:        # gallery 2 back wall: one wide spot
            sp = spots.add(track_y, 0.0, Z2_BACK + 2.0, centre, reach=0.55 * w + 0.15)
        if name == "L2":
            b.focus_points["painting_l2"] = centre.tolist()
        if name == "R3":
            b.focus_points["painting_r3"] = centre.tolist()
        if name == "G2B":
            b.focus_points["far_painting"] = centre.tolist()
        if name == "R1":
            b.focus_points["label_r1"] = pt(lab, (0, 0, 0.003)).tolist()
            b.focus_points["spot"] = sp

    # Introductory wall text on the left of the doorway.
    m = G.compose(place("B", -(DOOR_W / 2 + (X1 - DOOR_W / 2) / 2)), G.translate((0, 2.0, 0.006)))
    b.material("wall_text", {"type": "diffuse", "reflectance": bitmap(A.texture("wall_text", lambda: np.flipud(M.text_panel(77))))})
    b.ply(A.mesh("text_board", lambda: G.box((1.5, 2.1, 0.012), (0, 0, -0.006))), m, "paint_white", prefix="text_board")
    b.rect(G.compose(m, G.translate((0, 0, 0.0005)), G.scale((0.75, 1.05, 1))), "wall_text", prefix="wall_text")
    b.focus_points["wall_text"] = pt(m, (0, 0.5, 0)).tolist()
    spots.add(TRACK_Y1, -TRACK_X1, Z_BACK + 1.6, pt(m, (0, 0, 0)), e=1.6, reach=1.3)


# ----------------------------------------------------------------------------
# Vitrines and their contents
# ----------------------------------------------------------------------------

def add_case_shell(b: SceneBuilder, name: str, kind: str, cx: float, cz: float, w: float, d: float, deck: str) -> float:
    """Cabinet, deck, glass and (for tall cases) the lit hood. Returns the deck height."""
    base = TALL_BASE if kind == "tall" else FLAT_BASE
    gh = TALL_GLASS if kind == "tall" else FLAT_GLASS
    b.cube((cx, 0.015, cz), (w - 0.04, 0.03, d - 0.04), "shadow_gap")
    b.cube((cx, 0.03 + (base - 0.03) / 2, cz), (w, base - 0.03, d), "case_body")
    b.cube((cx, base + 0.005, cz), (w - 2 * PANE - 0.004, 0.01, d - 2 * PANE - 0.004), deck)
    deck_y = base + 0.01
    yc = base + gh / 2
    for sz in (-1, 1):
        b.cube((cx, yc, cz + sz * (d / 2 - PANE / 2)), (w, gh, PANE), "glass", prefix="glass")
    for sx in (-1, 1):
        b.cube((cx + sx * (w / 2 - PANE / 2), yc, cz), (PANE, gh, d - 2 * PANE), "glass", prefix="glass")
    for sx in (-1, 1):
        for sz in (-1, 1):
            b.cube((cx + sx * (w / 2 + 0.005), yc, cz + sz * (d / 2 + 0.005)), (0.01, gh, 0.01), "case_metal")
    if kind == "tall":
        hy = base + gh + TALL_HOOD / 2
        b.cube((cx, hy, cz), (w + 0.02, TALL_HOOD, d + 0.02), "case_body")
        em = tuple(HOOD_LIGHT * c for c in (1.0, 0.97, 0.92))
        b.rect(G.compose(G.translate((cx, base + gh - 0.001, cz)), G.rotate((1, 0, 0), 90), G.scale((w / 2 - 0.06, d / 2 - 0.06, 1))),
               "emitter_backing", emission=em, prefix=f"hood_{name}", power=12.0)
    else:
        b.cube((cx, base + gh + PANE / 2, cz), (w, PANE, d), "glass", prefix="glass")
        for sz in (-1, 1):
            b.cube((cx, base + gh + PANE / 2, cz + sz * (d / 2 + 0.006)), (w + 0.024, PANE + 0.004, 0.012), "case_metal")
        for sx in (-1, 1):
            b.cube((cx + sx * (w / 2 + 0.006), base + gh + PANE / 2, cz), (0.012, PANE + 0.004, d), "case_metal")
    return deck_y


def fill_pottery(b, ctx, name, cx, cz, y) -> list:
    A = b.assets
    rng = rng_for(ctx, 10)
    amph = A.meshes("amphora", lambda: M.amphora(0.44))
    pos = np.array([cx - 0.22 + rng.uniform(-0.02, 0.02), y, cz - 0.06])
    b.part_set(amph, G.compose(G.translate(pos), G.rotate((0, 1, 0), float(rng.uniform(0, 360)))), {"body": "blackfig", "handles": "blackfig"})
    jg = A.meshes("jug", lambda: M.jug(0.24))
    b.part_set(jg, G.compose(G.translate((cx + 0.2, y, cz - 0.1)), G.rotate((0, 1, 0), float(rng.uniform(150, 240)))),
               {"body": "jug_glaze", "handles": "jug_glaze"})
    b.ply(A.mesh("bowl", M.bowl), G.translate((cx + 0.04, y, cz + 0.14)), "celadon", prefix="bowl")
    b.focus_points["amphora"] = (pos + np.array([0, 0.22, 0])).tolist()
    return (pos + np.array([0, 0.22, 0])).tolist()


def fill_jewels(b, ctx, name, cx, cz, y) -> list:
    A = b.assets
    rng = rng_for(ctx, 11)
    stand = G.compose(G.translate((cx - 0.05, y, cz - 0.04)), G.rotate((0, 1, 0), float(rng.uniform(-10, 10))))
    b.ply(A.mesh("neck_stand", M.neck_stand), stand, "velvet_dark", prefix="neck_stand")
    nk = A.meshes("necklace", M.necklace)
    b.part_set(nk, stand, {"gold": "gold", "gem": "gem_red" if rng.random() < 0.5 else "gem_blue"})
    ring_parts = A.meshes("ring", M.ring)
    for k, (dx, dz) in enumerate(((0.27, 0.1), (0.36, -0.06), (-0.33, 0.1))):
        base = np.array([cx + dx, y, cz + dz])
        b.ply(A.mesh("ring_cone", M.ring_cone), G.translate(base), "velvet_dark", prefix="ring_cone")
        b.part_set(ring_parts, G.compose(G.translate(base + np.array([0, 0.052 + 0.0095 - 0.004, 0])), G.rotate((0, 1, 0), float(rng.uniform(-25, 25)))),
                   {"gold": "gold", "gem": ["gem_red", "gem_blue", "gem_red"][k]})
    pendant = pt(stand, (0, 0.113, 0.123))
    b.focus_points["necklace"] = pendant.tolist()
    return pendant.tolist()


def fill_manuscript(b, ctx, name, cx, cz, y) -> list:
    A = b.assets
    rng = rng_for(ctx, 12)
    mm = A.meshes("manuscript", M.manuscript)
    m = G.compose(G.translate((cx, y + 0.021, cz)), G.rotate((0, 1, 0), float(rng.uniform(-4, 4))))
    b.part_set(mm, m, {"pages": "parchment", "block": "page_block", "boards": "book_board", "cradle": "velvet_dark"})
    p = pt(m, (-0.09, 0.06, 0.0))
    b.focus_points["manuscript"] = p.tolist()
    return p.tolist()


def fill_fossils(b, ctx, name, cx, cz, y) -> list:
    A = b.assets
    rng = rng_for(ctx, 13)
    flat = A.mesh("ammonite_big", lambda: M.rest_on_floor(M.ammonite(0.2), G.rotate((1, 0, 0), -90)))
    slab = A.mesh("fossil_slab", lambda: M.stone_slab(0.32, 0.26, 0.03, np.random.default_rng(3)))
    sx = cx - 0.3
    b.ply(slab, G.translate((sx, y, cz)), "slab", prefix="slab")
    lie = G.compose(G.translate((sx, y + 0.03, cz)), G.rotate((0, 1, 0), float(rng.uniform(0, 360))))
    b.ply(flat, lie, "fossil", prefix="ammonite")
    small = A.mesh("ammonite_small", lambda: M.rest_on_floor(M.ammonite(0.13)))
    stand = np.array([cx + 0.02, y, cz - 0.08])
    b.ply(A.mesh("acrylic_stand", M.acrylic_stand), G.translate(stand), "acrylic", prefix="acrylic")
    up = G.compose(G.translate(stand + np.array([0, 0.004, 0.012])), G.rotate((0, 1, 0), float(rng.uniform(-15, 15))))
    b.ply(small, up, "fossil", prefix="ammonite")
    tray = A.meshes("coin_tray", lambda: M.coin_tray(np.random.default_rng(17)))
    tpos = np.array([cx + 0.33, y, cz + 0.06])
    b.cube(tpos + np.array([0, 0.006, 0]), (0.36, 0.012, 0.18), "velvet_dark")
    b.part_set(tray, G.compose(G.translate(tpos + np.array([0, 0.012, 0])), G.rotate((0, 1, 0), float(rng.uniform(-4, 4)))),
               {"gold": "coin_gold", "silver": "coin_silver", "bronze": "coin_bronze"})
    b.focus_points["ammonite"] = (stand + np.array([0, 0.07, 0])).tolist()
    b.focus_points["coins"] = (tpos + np.array([0, 0.015, 0])).tolist()
    return [sx, y + 0.06, cz]


def add_vitrines(b: SceneBuilder, ctx: BuildContext) -> None:
    rng = rng_for(ctx, 2)
    tall = ["pottery", "jewels"] if rng.random() < 0.5 else ["jewels", "pottery"]
    flat = ["manuscript", "fossils"] if rng.random() < 0.5 else ["fossils", "manuscript"]
    fills = {"pottery": fill_pottery, "jewels": fill_jewels, "manuscript": fill_manuscript, "fossils": fill_fossils}
    order = {"v1": tall[0], "v2": tall[1], "v3": flat[0], "v4": flat[1]}
    for i, (name, (kind, cx, cz, w, d)) in enumerate(VITRINES.items()):
        y = add_case_shell(b, name, kind, cx, cz, w, d, f"deck_{i}")
        focus = fills[order[name]](b, ctx, name, cx, cz, y)
        b.focus_points[f"case_{name}"] = focus
        # No small spots inside the cases: a delta light gets a few percent of the light samples, so it
        # would sparkle across the deck; the hood panels are big enough for BSDF sampling to find.


# ----------------------------------------------------------------------------
# Sculpture, bench, rope barrier
# ----------------------------------------------------------------------------

def add_sculptures(b: SceneBuilder, ctx: BuildContext, spots: Spots) -> None:
    A = b.assets
    rng = rng_for(ctx, 3)
    kinds = [str(k) for k in rng.permutation(["bird", "knot", "muse", "concretion", "urn"])][:4]
    for (slot, (x, z)), kind in zip(PLINTHS.items(), kinds):
        if kind == "bird":
            pw, ph = 0.5, 0.22
        elif kind == "concretion":
            pw, ph = 0.62, 0.78
        elif kind == "urn":
            pw, ph = 0.56, 0.86
        else:
            pw, ph = 0.46, 1.08
        pl = A.meshes(f"plinth_{round(pw * 100)}_{round(ph * 100)}", lambda pw=pw, ph=ph: M.plinth(pw, ph, pw))
        b.part_set(pl, G.translate((x, 0, z)), {"body": "paint_white", "gap": "shadow_gap"})
        yaw = float(rng.uniform(0, 360))
        top = G.compose(G.translate((x, ph, z)), G.rotate((0, 1, 0), yaw))
        if kind == "bird":
            parts = A.meshes("bird", M.bird)
            b.part_set(parts, top, {"bronze": "bronze_polish", "stone": "drum_stone"})
            centre = pt(top, (0, 1.2, 0))
        elif kind == "knot":
            b.ply(A.mesh("trefoil", M.trefoil), top, "bronze_patina" if rng.random() < 0.5 else "steel_polish", prefix="knot")
            centre = pt(top, (0, 0.24, 0))
        elif kind == "muse":
            b.cube(pt(top, (0, 0.03, 0)), (0.3, 0.06, 0.24), "marble", yaw)
            b.ply(A.mesh("muse", M.muse), G.compose(top, G.translate((0, 0.06, 0))), "bronze_polish" if rng.random() < 0.5 else "marble", prefix="muse")
            centre = pt(top, (0, 0.17, 0))
        elif kind == "concretion":
            b.ply(A.mesh("concretion", M.concretion), top, "marble", prefix="concretion")
            centre = pt(top, (0, 0.22, 0))
        else:
            b.ply(A.meshes("urn", M.urn)["body"], top, "marble", prefix="urn")
            centre = pt(top, (0, 0.4, 0))
        b.focus_points[f"sculpture_{slot}"] = centre.tolist()
        if slot == "g2":
            spots.add(TRACK_Y2, -TRACK_X2, z + 1.2, centre, e=2.6, reach=0.5)
        else:
            spots.add(TRACK_Y1, math.copysign(TRACK_X1, x), z + 1.3, centre, e=2.6, reach=0.5)


def add_furniture(b: SceneBuilder, ctx: BuildContext) -> None:
    A = b.assets
    rng = rng_for(ctx, 4)
    bz = -5.5 + float(rng.uniform(-0.05, 0.05))
    b.part_set(A.meshes("bench", M.bench), G.translate((0, 0, bz)), {"cushion": "leather", "steel": "steel"})
    b.focus_points["bench"] = [0.0, 0.5, bz]
    zs = (-10.7, -9.5, -8.3)
    post = A.mesh("stanchion", M.stanchion)
    for z in zs:
        b.ply(post, G.translate((3.25, 0, z)), "brass", prefix="stanchion")
    for z0, z1 in zip(zs[:-1], zs[1:]):
        b.ply(A.mesh(f"rope_{z0}", lambda z0=z0, z1=z1: M.rope((3.25, 0.93, z0 + 0.03), (3.25, 0.93, z1 - 0.03), 0.13)),
              np.eye(4), "velvet_red", prefix="rope")
    b.focus_points["stanchion"] = [3.25, 0.95, -9.5]


def build(ctx: BuildContext) -> SceneBundle:
    b = SceneBuilder(Assets(ctx.shared_dir, ctx.rebuild))
    choices = add_materials(b, ctx)
    add_architecture(b)
    spots = Spots(b)
    add_paintings(b, ctx, spots)
    add_vitrines(b, ctx)
    add_sculptures(b, ctx, spots)
    add_furniture(b, ctx)
    return SceneBundle(b.d, b.focus_points, {"layout": choices, "spots": spots.n})


SCENE = SceneDef(
    id="museum", group="artificial", owner="rui",
    description="Museum gallery: glass vitrines with pottery, jewellery, a manuscript and fossils; framed paintings with labels; "
                "sculptures; laylight and track spots; a doorway into a second gallery.",
    build=build,
    views={
        # Close at the first tall case, looking through its front glass.
        "case": View((-1.35, 1.42, -1.72), (-1.78, 1.12, -3.0), focus="case_v1"),
        # Down the gallery: cases, bench, the doorway and the far painting.
        "gallery": View((1.05, 1.62, 1.4), (-0.35, 1.45, -16.0), focus="bench"),
        # A sculpture on its plinth with the paintings of the left wall behind it.
        "sculpture": View((0.55, 1.5, -3.75), (-3.0, 1.12, -5.6), focus="sculpture_p1"),
        # A painting's wall label up close, the frame edge beside it.
        "label": View((3.3, 1.5, -0.32), (4.0, 1.42, -0.86), focus="label_r1"),
        # From the far gallery looking back through the doorway into gallery 1.
        "look_back": View((-1.0, 1.5, -17.6), (0.3, 1.6, -8.0), focus="doorway"),
        # Looking down into the first flat table case (manuscript or fossils, by seed).
        "table_case": View((-1.45, 1.6, -6.75), (-1.72, 0.92, -8.0), focus="case_v3"),
    },
    default_view="gallery",
    camera_box=((-3.3, 0.6, -19.3), (3.3, 2.2, 1.6)),
    target_box=((-4.0, 0.2, -20.0), (4.0, 4.2, 2.0)),
    exclude_boxes=(
        ((-4.0, 0.0, -12.65), (-0.95, 5.0, -11.75)), ((0.95, 0.0, -12.65), (4.0, 5.0, -11.75)),   # dividing wall
        ((-2.5, 0.0, -3.6), (-0.9, 3.0, -2.4)), ((0.9, 0.0, -3.6), (2.5, 3.0, -2.4)),            # tall cases
        ((-2.6, 0.0, -8.65), (-0.8, 3.0, -7.35)), ((0.8, 0.0, -8.65), (2.6, 3.0, -7.35)),        # flat cases
        ((-0.55, 0.0, -6.6), (0.55, 0.95, -4.4)),                                                  # bench
        ((-3.35, 0.0, -6.05), (-2.45, 4.0, -4.95)), ((2.45, 0.0, -6.05), (3.35, 4.0, -4.95)),    # plinths
        ((-3.35, 0.0, -10.95), (-2.45, 4.0, -9.85)), ((-0.6, 0.0, -17.1), (0.6, 4.0, -15.9)),
        ((2.9, 0.0, -11.0), (3.6, 1.2, -8.0)),                                                     # rope barrier
    ),
    tags=("indoor", "artificial_light", "glass", "specular", "architecture", "metal"),
    default_seed=11,
    asset_version=4,
    spp_hint=4096,
)
