"""A physically based cafe interior for Mitsuba 3.

`SCENE.build(ctx)` returns a Mitsuba scene dictionary (no sensor; the renderer adds
it) plus named points of interest for focusing. Meshes and textures are generated
procedurally on first use and cached as PLY/PNG in `ctx.shared_dir`.

Layout, meters, y up, camera looking toward -z:
  room          x in [-3.8, 3.8], z in [-7.0, 3.5], ceiling at 3.2
  left wall     two tall windows; late-afternoon sun rakes across the floor
  foreground    round bistro table with teapot, cup, wine glass, croissant, vase
  midground     three more tables with chairs, laptop, books, candles
  background    marble counter, espresso machine, cake dome, jars, shelves of
                glass bottles, chalkboard menu, string lights
  lights        sun + sky, enamel pendants, bare globe bulbs, string lights,
                candles, a laptop screen
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import mitsuba as mi
import procedural as G
import env_kit
import props as P
import web_assets as W
from scenes._cafe_props import NULL, add_street, box, put
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, bitmap, luminance, principled, rgb, xf

ROOM_X, ROOM_Z0, ROOM_Z1, ROOM_H = 3.8, -7.0, 3.5, 3.2
TABLE_H = 0.75
WINDOWS = [(-5.3, -3.5), (-2.5, -0.7)]  # z extents of the left-wall openings
WINDOW_Y = (0.85, 2.65)

# Sun above the left-hand street, low enough to throw window patches onto the
# floor and midground tables. Points toward the sun.
SUN_DIRECTION = (-0.80, 0.42, 0.30)
# Mitsuba's sunsky puts sunlit plaster near radiance 0.1, three orders below a
# bare bulb. Scaling sun and sky together keeps their ratio physical while
# putting a sunlit patch a few stops under the bulbs, as in a daytime interior.
DAYLIGHT_SCALE = 60.0


def add_materials(b: SceneBuilder) -> None:
    A = b.assets
    # Big surfaces: scanned CC0 textures (Poly Haven, ambientCG) at real-world scale on uv-in-meters boxes.
    b.material("floor", W.surface("old_wood_floor"))
    b.material("wall", W.surface("painted_plaster_wall", tint=(1.45, 1.28, 1.02)))
    b.material("wall_green", W.surface("green_rough_planks", tint=(1.25, 1.4, 1.3)))
    b.material("ceiling", W.surface("white_stucco", tint=(0.98, 0.95, 0.88)))
    b.material("dark_wood", W.surface("dark_wood"))
    b.material("shelf_wood", W.surface("oak_wood_planks"))
    b.material("marble_counter", W.surface("marble_01"))
    # Counter front stays the procedural green subway tile: long_white_tiles read as a dull grey grid.
    tile_rgb, tile_rough = G.subway_tiles(512, 1024, 1.2, 0.6, seed=9)
    b.material("tiles", principled(bitmap(A.texture("tiles", lambda: tile_rgb), uv_scale=(1 / 1.2, 1 / 0.6)),
                                   bitmap(A.texture("tiles_rough", lambda: tile_rough, gray=True), raw=True,
                                          uv_scale=(1 / 1.2, 1 / 0.6)), specular=0.6))
    b.material("window_frame", W.surface("acg:PaintedWood007A", tint=(0.32, 0.5, 0.42)))
    # Disc table tops have planar uv spanning the diameter (0.72-0.8 m), so scale to the texture width.
    b.material("table_wood", W.surface("mocha_oak_veneer", uv_scale=0.8 / 1.0))
    b.material("marble", W.surface("marble_01", uv_scale=0.72 / 1.5))

    b.material("iron", principled((0.025, 0.025, 0.025), 0.4, specular=0.4))
    b.material("bentwood", principled((0.13, 0.065, 0.03), 0.32, clearcoat=0.5))
    b.material("cane", principled((0.55, 0.40, 0.22), 0.6))
    b.material("ceramic_white", principled((0.90, 0.88, 0.84), 0.15, specular=0.6, clearcoat=0.5, clearcoat_gloss=0.5))
    b.material("ceramic_blue", principled((0.07, 0.19, 0.36), 0.15, specular=0.6, clearcoat=0.5, clearcoat_gloss=0.5))
    b.material("ceramic_sage", principled((0.45, 0.55, 0.47), 0.15, specular=0.6, clearcoat=0.5, clearcoat_gloss=0.5))
    b.material("terracotta", principled((0.56, 0.27, 0.15), 0.8))
    b.material("coffee", principled((0.03, 0.012, 0.005), 0.1, specular=0.5))
    b.material("steel", {"type": "roughconductor", "material": "Cr", "alpha": 0.12})
    b.material("chrome", {"type": "roughconductor", "material": "Cr", "alpha": 0.06})
    b.material("brass", {"type": "roughconductor", "material": "Au", "alpha": 0.2})
    b.material("copper", {"type": "roughconductor", "material": "Cu", "alpha": 0.15})
    b.material("aluminium", {"type": "roughconductor", "material": "Al", "alpha": 0.25})
    b.material("mirror", {"type": "conductor", "material": "Ag"})
    b.material("glass", {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0})
    b.material("glass_green", {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0,
                               "specular_transmittance": rgb(0.30, 0.62, 0.34)})
    b.material("glass_amber", {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0,
                               "specular_transmittance": rgb(0.78, 0.45, 0.14)})
    b.material("glass_frosted", {"type": "roughdielectric", "int_ior": 1.5, "ext_ior": 1.0, "alpha": 0.15})
    b.material("leaf", {"type": "twosided", "bsdf": principled((0.07, 0.24, 0.07), 0.45)})
    b.material("leaf_dark", {"type": "twosided", "bsdf": principled((0.04, 0.15, 0.06), 0.4)})
    b.material("petal", {"type": "twosided", "bsdf": principled((0.85, 0.32, 0.22), 0.5)})
    b.material("flower_heart", principled((0.85, 0.62, 0.1), 0.7))
    b.material("stem", principled((0.12, 0.3, 0.08), 0.6))
    b.material("soil", {"type": "diffuse", "reflectance": rgb(0.06, 0.04, 0.03)})
    b.material("pastry", principled((0.70, 0.40, 0.13), 0.5))
    b.material("cake", principled((0.93, 0.84, 0.70), 0.7))
    b.material("orange", principled((0.88, 0.36, 0.04), 0.3))
    b.material("wax", principled((0.92, 0.88, 0.80), 0.5))
    b.material("black", principled((0.012, 0.012, 0.012), 0.35))
    b.material("enamel_green", principled((0.05, 0.17, 0.13), 0.25, clearcoat=0.4, clearcoat_gloss=0.5))
    b.material("enamel_cream", principled((0.85, 0.80, 0.68), 0.25, clearcoat=0.4, clearcoat_gloss=0.5))
    b.material("shade_inner", {"type": "diffuse", "reflectance": rgb(0.88, 0.86, 0.82)})
    b.material("window_frame", principled((0.07, 0.12, 0.10), 0.35))
    b.material("paper", {"type": "diffuse", "reflectance": rgb(0.82, 0.78, 0.68)})
    for i, c in enumerate([(0.45, 0.08, 0.06), (0.08, 0.2, 0.3), (0.55, 0.42, 0.12), (0.12, 0.25, 0.14),
                           (0.3, 0.12, 0.25), (0.75, 0.7, 0.6)]):
        b.material(f"book_{i}", principled(c, 0.55))
    b.material("chalkboard", {"type": "diffuse", "reflectance": bitmap(A.texture(
        "chalkboard", lambda: G.chalkboard(768, 1152, seed=11)))})
    b.material("art_a", {"type": "diffuse", "reflectance": bitmap(A.texture(
        "art_a", lambda: G.abstract_art(768, 576, [(0.85, 0.78, 0.62), (0.78, 0.35, 0.18),
                                                    (0.14, 0.26, 0.36), (0.9, 0.62, 0.2)], seed=13)))})
    b.material("art_b", {"type": "diffuse", "reflectance": bitmap(A.texture(
        "art_b", lambda: G.abstract_art(576, 768, [(0.2, 0.3, 0.28), (0.85, 0.82, 0.7),
                                                    (0.55, 0.2, 0.2), (0.3, 0.45, 0.6)], seed=17)))})
    b.material("emitter_backing", {"type": "diffuse", "reflectance": rgb(0.0, 0.0, 0.0)})


# ----------------------------------------------------------------------------
# Lights
# ----------------------------------------------------------------------------

BULB = (38.0, 22.0, 9.0)        # warm filament globe, radiance
PENDANT = (60.0, 38.0, 17.0)    # bulb hidden in a shade
STRING = (55.0, 30.0, 11.0)
FLAME = (60.0, 26.0, 6.0)


def add_sky(b: SceneBuilder) -> None:
    d = np.asarray(SUN_DIRECTION, dtype=float)
    d /= np.linalg.norm(d)
    b.d["sky"] = {"type": "sunsky", "to_world": env_kit.sunsky_to_world(), "sun_direction": d.tolist(), "turbidity": 3.0,
                  "sun_scale": DAYLIGHT_SCALE, "sky_scale": DAYLIGHT_SCALE}


def pendant(b: SceneBuilder, x: float, z: float, drop: float, enamel: str) -> None:
    """Enamel dome shade hanging `drop` meters below the ceiling."""
    shade = b.assets.meshes("dome_shade", P.dome_shade)
    sock = b.assets.mesh("socket", P.socket)
    cord = b.assets.mesh("cord", lambda: G.sweep([(0, 0, 0), (0, 1, 0)], 0.003, 8))
    top = ROOM_H - drop
    b.ply(cord, G.compose(G.translate((x, top + 0.05, z)), G.scale((1, drop - 0.05, 1))), "black")
    b.part_set(shade, G.translate((x, top, z)), {"outer": enamel, "inner": "shade_inner"})
    b.ply(sock, G.compose(G.translate((x, top - 0.02, z)), G.rotate((1, 0, 0), 180)), "brass")
    b.sphere((x, top - 0.1, z), 0.032, "emitter_backing", emission=PENDANT)
    b.focus_points.setdefault("pendants", []).append([x, top - 0.1, z])


def globe_bulb(b: SceneBuilder, x: float, z: float, drop: float) -> None:
    sock = b.assets.mesh("socket", P.socket)
    cord = b.assets.mesh("cord", lambda: G.sweep([(0, 0, 0), (0, 1, 0)], 0.003, 8))
    top = ROOM_H - drop
    b.ply(cord, G.compose(G.translate((x, top, z)), G.scale((1, drop, 1))), "black")
    b.ply(sock, G.compose(G.translate((x, top + 0.002, z)), G.rotate((1, 0, 0), 180)), "brass")
    b.sphere((x, top - 0.1, z), 0.055, "emitter_backing", emission=BULB)


def string_lights(b: SceneBuilder, z: float, y_top: float, sag: float, x0: float, x1: float, swags: int) -> None:
    xs = np.linspace(x0, x1, swags * 40 + 1)
    phase = (xs - x0) / (x1 - x0) * swags
    ys = y_top - sag * np.sin(np.pi * (phase % 1.0))
    wire = np.stack([xs, ys, np.full_like(xs, z)], 1)
    path = b.assets.mesh(f"string_wire_{abs(int(z * 100))}", lambda: G.sweep(wire, 0.0015, 6))
    b.ply(path, np.eye(4), "black")
    for x in np.arange(x0 + 0.12, x1, 0.26):
        ph = (x - x0) / (x1 - x0) * swags
        y = y_top - sag * np.sin(np.pi * (ph % 1.0))
        b.sphere((x, y - 0.03, z), 0.014, "emitter_backing", emission=STRING)


# ----------------------------------------------------------------------------
# Room shell
# ----------------------------------------------------------------------------

def add_room(b: SceneBuilder) -> None:
    cz, dz = 0.5 * (ROOM_Z0 + ROOM_Z1), ROOM_Z1 - ROOM_Z0
    W2 = 2 * ROOM_X
    box(b, (0, -0.01, cz), (W2, 0.02, dz), "floor")
    box(b, (0, ROOM_H + 0.01, cz), (W2, 0.02, dz), "ceiling")
    # Back and front walls: green plank wainscot below a dado rail, warm plaster above.
    for zw, sgn in ((ROOM_Z0, 1), (ROOM_Z1, -1)):
        box(b, (0, 0.55, zw + sgn * 0.01), (W2, 1.1, 0.02), "wall_green")
        box(b, (0, 0.5 * (ROOM_H + 1.1), zw - sgn * 0.01), (W2, ROOM_H - 1.1, 0.02), "wall")
        box(b, (0, 0.06, zw + sgn * 0.03), (W2, 0.12, 0.02), "dark_wood")
        box(b, (0, 1.1, zw + sgn * 0.032), (W2, 0.04, 0.025), "dark_wood")
    box(b, (ROOM_X + 0.01, 0.5 * ROOM_H, cz), (0.02, ROOM_H, dz), "wall")
    box(b, (ROOM_X - 0.01, 0.06, cz), (0.02, 0.12, dz), "dark_wood")
    box(b, (ROOM_X - 0.012, 1.1, cz), (0.025, 0.04, dz), "dark_wood")

    # Left wall with two window openings, built from thick blocks so the
    # openings have real reveals that shape the sun patches.
    t = 0.25
    wx = -ROOM_X - t / 2
    y0, y1 = WINDOW_Y
    edges = [ROOM_Z1] + [z for w in reversed(WINDOWS) for z in (w[1], w[0])] + [ROOM_Z0]
    for i in range(0, len(edges), 2):
        z_hi, z_lo = edges[i], edges[i + 1]
        box(b, (wx, 0.5 * ROOM_H, 0.5 * (z_hi + z_lo)), (t, ROOM_H, z_hi - z_lo), "wall")
        box(b, (-ROOM_X + 0.01, 0.06, 0.5 * (z_hi + z_lo)), (0.02, 0.12, z_hi - z_lo), "dark_wood")
    for z_lo, z_hi in WINDOWS:
        zc, zw = 0.5 * (z_lo + z_hi), z_hi - z_lo
        box(b, (wx, 0.5 * y0, zc), (t, y0, zw), "wall")
        box(b, (wx, 0.5 * (y1 + ROOM_H), zc), (t, ROOM_H - y1, zw), "wall")
        # Frame, mullion, transom, and a deep sill.
        f = 0.05
        for z in (z_lo + f / 2, z_hi - f / 2, zc):
            box(b, (wx, 0.5 * (y0 + y1), z), (0.07, y1 - y0, f if z != zc else 0.035), "window_frame")
        for y in (y0 + f / 2, y1 - f / 2, y1 - 0.45):
            box(b, (wx, y, zc), (0.07, f if y != y1 - 0.45 else 0.035, zw), "window_frame")
        box(b, (-ROOM_X + 0.08, y0 - 0.02, zc), (0.36, 0.04, zw + 0.12), "shelf_wood")

    # Ceiling beams.
    for z in np.arange(ROOM_Z1 - 0.8, ROOM_Z0, -1.9):
        box(b, (0, ROOM_H - 0.09, z), (W2, 0.18, 0.16), "dark_wood")


def add_entrance(b: SceneBuilder) -> None:
    """Front wall (z = ROOM_Z1, facing the room): panelled door, coat stand, radiator, clock, art."""
    zf = ROOM_Z1
    dx, dw, dh = 2.3, 1.0, 2.2
    box(b, (dx, dh / 2, zf - 0.03), (dw, dh, 0.05), "dark_wood")                     # door leaf
    for py in (0.55, 1.5):                                                         # raised panels
        box(b, (dx, py, zf - 0.06), (dw - 0.3, 0.7, 0.02), "dark_wood")
    for sx, w in ((-1, 0.08), (1, 0.08)):                                          # architrave
        box(b, (dx + sx * (dw / 2 + w / 2), (dh + 0.08) / 2, zf - 0.035), (w, dh + 0.08, 0.06), "window_frame")
    box(b, (dx, dh + 0.08, zf - 0.035), (dw + 0.16, 0.08, 0.06), "window_frame")
    b.sphere((dx - 0.4, 1.02, zf - 0.09), 0.025, "brass")                           # knob
    box(b, (dx, 0.006, zf - 0.55), (1.0, 0.012, 0.6), "dark_wood")                   # door mat
    # Coat stand by the door: pole, base, hooks.
    cs = (1.35, zf - 0.35)
    box(b, (cs[0], 0.9, cs[1]), (0.045, 1.8, 0.045), "dark_wood")
    box(b, (cs[0], 0.02, cs[1]), (0.4, 0.04, 0.4), "dark_wood", yaw=45)
    for k in range(4):
        box(b, (cs[0], 1.72, cs[1]), (0.34, 0.03, 0.03), "brass", yaw=45 * k)
    # Cast-iron radiator under the left half of the front wall.
    rx = -1.6
    for k in range(14):
        box(b, (rx - 0.65 + 0.1 * k, 0.45, zf - 0.09), (0.06, 0.62, 0.12), "iron")
    box(b, (rx, 0.12, zf - 0.09), (1.42, 0.05, 0.05), "iron")
    put(b, "vintage_telephone_wall_clock", at=(0.0, 2.15, zf - 0.02), yaw=180)
    put(b, "hanging_picture_frame_03", at=(-1.6, 1.45, zf - 0.02), yaw=180, scale=1.4)
    b.focus_points["door"] = [dx, 1.1, zf - 0.05]


# ----------------------------------------------------------------------------
# Furniture groups
# ----------------------------------------------------------------------------

def table(b: SceneBuilder, x: float, z: float, top_material: str, radius: float = 0.36) -> None:
    top = b.assets.mesh(f"table_top_{int(radius * 100)}", lambda: P.slab_disc(radius, 0.03, 0.009))
    base = b.assets.mesh("table_base", lambda: P.table_base(TABLE_H - 0.03))
    b.ply(base, G.translate((x, 0, z)), "iron")
    b.ply(top, G.translate((x, TABLE_H - 0.03, z)), top_material)


def chair(b: SceneBuilder, x: float, z: float, yaw: float) -> None:
    # Kept procedural: the scanned dining chairs (dining_chair_02 and kin) are upholstered dining-room
    # chairs and lose the bentwood bistro look in renders.
    parts = b.assets.meshes("bistro_chair", P.bistro_chair)
    b.part_set(parts, G.compose(G.translate((x, 0, z)), G.rotate((0, 1, 0), yaw)),
               {"frame": "bentwood", "seat": "cane"})


def cup_on_saucer(b: SceneBuilder, x: float, z: float, yaw: float, material: str = "ceramic_white",
                  with_spoon: bool = True, surface: float = TABLE_H) -> None:
    cup = b.assets.meshes("cup", P.cup)
    sau = b.assets.mesh("saucer", P.saucer)
    base = G.compose(G.translate((x, surface, z)), G.rotate((0, 1, 0), yaw))
    b.ply(sau, base, material)
    b.part_set(cup, G.compose(base, G.translate((0, 0.0075, 0))), {"cup": material, "coffee": "coffee"})
    if with_spoon:
        spoon = b.assets.mesh("spoon", P.spoon)
        b.ply(spoon, G.compose(base, G.translate((0.012, 0.0125, 0.055)), G.rotate((0, 1, 0), 18),
                               G.rotate((0, 0, 1), -4)), "steel")


def book_stack(b: SceneBuilder, x: float, y: float, z: float, yaw: float, rng) -> None:
    h = y
    for i in range(int(rng.integers(2, 5))):
        w, d, t = rng.uniform(0.15, 0.22), rng.uniform(0.21, 0.28), rng.uniform(0.018, 0.04)
        a = yaw + rng.uniform(-12, 12)
        cover = f"book_{int(rng.integers(0, 6))}"
        b.cube((x + rng.uniform(-0.01, 0.01), h + t / 2, z + rng.uniform(-0.01, 0.01)), (w, t, d), cover, a)
        b.cube((x + 0.004, h + t / 2, z), (w - 0.01, t - 0.006, d - 0.004), "paper", a)
        h += t


def plant(b: SceneBuilder, name: str, recipe, matrix, pot: str = "terracotta", leaf: str = "leaf") -> None:
    parts = b.assets.meshes(name, recipe)
    b.part_set(parts, matrix, {"pot": pot, "soil": "soil", "stems": "stem", "leaves": leaf})


def candle(b: SceneBuilder, x: float, z: float, surface: float = TABLE_H) -> None:
    parts = b.assets.meshes("candle", P.candle)
    b.part_set(parts, G.translate((x, surface, z)),
               {"holder": "glass_frosted", "wax": "wax", "wick": "black", "flame": "emitter_backing"},
               emission={"flame": FLAME}, power=4.5e-4 * luminance(FLAME))


def laptop(b: SceneBuilder, x: float, z: float, yaw: float) -> None:
    base = G.compose(G.translate((x, TABLE_H, z)), G.rotate((0, 1, 0), yaw))
    m_base = G.compose(base, G.translate((0, 0.008, 0)), G.scale((0.16, 0.008, 0.11)))
    b._shape("laptop", {"type": "cube", "to_world": xf(m_base)}, "aluminium")
    hinge = G.compose(base, G.translate((0, 0.016, -0.11)), G.rotate((1, 0, 0), -12))
    lid = G.compose(hinge, G.translate((0, 0.11, -0.004)), G.scale((0.16, 0.11, 0.004)))
    b._shape("laptop", {"type": "cube", "to_world": xf(lid)}, "aluminium")
    screen_tex = b.assets.texture("laptop_screen", lambda: G.laptop_screen(256, 384))
    screen = G.compose(hinge, G.translate((0, 0.112, 0.0005)), G.scale((0.148, 0.098, 1)))
    b.rect(screen, "emitter_backing", emission={"type": "bitmap", "filename": screen_tex, "raw": False},
           power=0.058 * 0.8)


def add_hero_table(b: SceneBuilder) -> None:
    """The foreground table the camera sits at."""
    x, z = 0.0, 0.1
    table(b, x, z, "table_wood", radius=0.40)
    chair(b, -0.62, -0.28, 55)

    teapot = b.assets.meshes("teapot", P.teapot)
    tp = (x - 0.03, TABLE_H, z - 0.08)
    b.part_set(teapot, G.compose(G.translate(tp), G.rotate((0, 1, 0), 200)), {"body": "ceramic_blue", "lid": "ceramic_blue"})
    b.focus_points["teapot"] = [tp[0], tp[1] + 0.08, tp[2]]

    cup_on_saucer(b, x + 0.2, z + 0.2, -30)
    b.focus_points["cup"] = [x + 0.2, TABLE_H + 0.05, z + 0.2]

    glass = b.assets.mesh("wine_glass", P.wine_glass)
    b.ply(glass, G.translate((x - 0.22, TABLE_H, z + 0.17)), "glass")
    b.focus_points["glass"] = [x - 0.22, TABLE_H + 0.12, z + 0.17]

    pl = b.assets.mesh("plate", lambda: P.plate(0.1))
    b.ply(pl, G.translate((x + 0.18, TABLE_H, z - 0.2)), "ceramic_white")
    put(b, "croissant", at=(x + 0.18, TABLE_H + 0.0085, z - 0.2), yaw=35, scale=0.8)

    vase = dict(b.assets.meshes("vase", P.vase_with_flowers))
    vase.pop("vase")
    b.part_set(vase, G.translate((x - 0.2, TABLE_H, z - 0.24)),
               {"stems": "stem", "petals": "petal", "hearts": "flower_heart", "leaves": "leaf"})
    put(b, "ceramic_vase_02", at=(x - 0.2, TABLE_H, z - 0.24), height=0.215)
    b.focus_points["flowers"] = [x - 0.2, TABLE_H + 0.38, z - 0.24]


def add_midground(b: SceneBuilder, rng) -> None:
    # Table 2: left, catching the sun from the second window.
    table(b, -1.45, -1.55, "marble")
    chair(b, -1.45, -0.95, 180)
    chair(b, -2.0, -1.95, 60)
    laptop(b, -1.5, -1.55, 25)
    cup_on_saucer(b, -1.25, -1.35, 60, "ceramic_sage")
    b.focus_points["laptop"] = [-1.5, TABLE_H + 0.1, -1.6]

    # Table 3: right, books and a candle.
    table(b, 1.3, -2.8, "table_wood")
    chair(b, 1.3, -2.2, 190)
    chair(b, 1.85, -3.2, -60)
    book_stack(b, 1.38, TABLE_H, -2.9, 20, rng)
    candle(b, 1.12, -2.68)
    b.ply(b.assets.mesh("tumbler", P.tumbler), G.translate((1.5, TABLE_H, -2.62)), "glass")
    b.focus_points["books"] = [1.38, TABLE_H + 0.06, -2.9]

    # Table 4: far left, under a window.
    table(b, -1.0, -4.35, "marble")
    chair(b, -0.5, -4.05, -120)
    chair(b, -1.55, -4.75, 45)
    cup_on_saucer(b, -1.1, -4.3, 10)
    cup_on_saucer(b, -0.85, -4.5, 140, "ceramic_blue", with_spoon=False)
    candle(b, -0.95, -4.15)

    # Plants: a big floor plant, one by the right wall, and sill plants.
    put(b, "potted_plant_01", at=(-3.2, 0.0, -6.4), yaw=float(rng.uniform(0, 360)))
    put(b, "potted_plant_02", at=(3.3, 0.0, -1.3), yaw=float(rng.uniform(0, 360)))
    put(b, "potted_plant_01", at=(3.25, 0.0, 2.9), yaw=float(rng.uniform(0, 360)), scale=0.85)
    for z_lo, z_hi in WINDOWS:
        zc = 0.5 * (z_lo + z_hi)
        put(b, "potted_plant_04", at=(-ROOM_X + 0.1, WINDOW_Y[0], zc + 0.35), yaw=float(rng.uniform(0, 360)))
    put(b, "standing_picture_frame_01", at=(-ROOM_X + 0.1, WINDOW_Y[0], -4.9), yaw=90)


def add_counter(b: SceneBuilder, rng) -> None:
    x0, x1, z_front, z_back, h = -0.9, 3.5, -5.55, -6.25, 1.0
    cx, cz = 0.5 * (x0 + x1), 0.5 * (z_front + z_back)
    b.box_mesh("counter_body", (cx, h / 2 - 0.02, cz), (x1 - x0, h - 0.04, z_front - z_back), "tiles")
    b.box_mesh("counter_top", (cx, h - 0.02, cz + 0.02), (x1 - x0 + 0.06, 0.04, z_front - z_back + 0.08), "marble_counter")
    b.cube((cx, 0.05, z_front + 0.03), (x1 - x0, 0.1, 0.02), "black")
    top = h

    esp = b.assets.meshes("espresso", P.espresso_machine)
    b.part_set(esp, G.translate((0.3, top, -5.95)), {"chrome": "chrome", "panel": "black", "black": "black"})
    for i, dx in enumerate((-0.2, 0.0, 0.2)):
        cup = b.assets.meshes("cup", P.cup)
        b.part_set(cup, G.compose(G.translate((0.3 + dx, top + 0.53, -6.0)), G.scale(0.85)),
                   {"cup": "ceramic_white", "coffee": "ceramic_white"})
    b.focus_points["espresso"] = [0.3, top + 0.3, -5.75]

    dome = b.assets.meshes("cake_dome", P.cake_dome)
    dome = dict(dome)
    dome.pop("cake")
    b.part_set(dome, G.translate((1.35, top, -5.8)), {"dome": "glass", "stand": "ceramic_white"})
    put(b, "strawberry_chocolate_cake", at=(1.35, top + 0.092, -5.8), yaw=float(rng.uniform(0, 360)), scale=0.85)
    b.focus_points["cake"] = [1.35, top + 0.15, -5.8]
    jar = b.assets.meshes("jar", P.jar)
    for i, x in enumerate((1.9, 2.1)):
        b.part_set(jar, G.translate((x, top, -5.85 - 0.08 * i)), {"glass": "glass", "lid": "shelf_wood"})
        for k in range(5):
            c = (x + rng.uniform(-0.03, 0.03), top + 0.015 + 0.012 * k, -5.85 - 0.08 * i + rng.uniform(-0.03, 0.03))
            b.cube(c, (0.05, 0.01, 0.05), "pastry", rng.uniform(0, 90))
    put(b, "wooden_bowl_01", at=(2.75, top, -5.8), scale=0.8)
    fruit = ["lemon", "food_apple_01", "food_pears_asian_01", "lemon", "food_apple_01"]
    for k, fid in enumerate(fruit):
        ang = 2 * np.pi * k / 5 + rng.uniform(-0.2, 0.2)
        put(b, fid, at=(2.75 + 0.06 * np.cos(ang), top + 0.025, -5.8 + 0.06 * np.sin(ang)),
            yaw=float(rng.uniform(0, 360)), scale=0.85)
    put(b, "lemon", at=(2.75, top + 0.07, -5.8), yaw=float(rng.uniform(0, 360)))
    put(b, "CashRegister_01", at=(3.15, top, -5.95), yaw=180, scale=0.55)
    put(b, "vintage_electric_kettle", at=(0.95, top, -6.05), yaw=150)
    put(b, "potted_plant_04", at=(-0.75, top, -5.75), yaw=float(rng.uniform(0, 360)))
    for x in (1.9, 2.55, 3.2):
        put(b, "bar_chair_round_01", at=(x, 0.0, -5.15), yaw=float(rng.uniform(0, 360)))
    b.focus_points["register"] = [3.15, top + 0.2, -5.95]
    for k in range(4):
        b.ply(b.assets.mesh("saucer", P.saucer), G.translate((-0.45, top + 0.009 * k, -5.8)), "ceramic_white")
    b.ply(b.assets.mesh("tumbler", P.tumbler), G.translate((-0.2, top, -5.7)), "glass")

    # Shelves on the back wall with bottles, jars and stacked cups.
    bottle = b.assets.mesh("bottle", P.bottle)
    glass_kinds = ["glass_green", "glass_amber", "glass", "glass_green", "glass_amber"]
    for level, y in enumerate((1.45, 1.85, 2.25)):
        box(b, (1.3, y - 0.02, -6.85), (4.2, 0.04, 0.26), "shelf_wood")
        for bx in (-0.6, 1.3, 3.2):
            b.cube((bx, y - 0.1, -6.95), (0.03, 0.16, 0.06), "brass")
        x = -0.72
        while x < 2.6:
            kind = rng.random()
            if kind < 0.6:
                b.ply(bottle, G.compose(G.translate((x, y, -6.85 + rng.uniform(-0.04, 0.04))),
                                        G.scale(rng.uniform(0.8, 1.05))),
                      glass_kinds[int(rng.integers(0, len(glass_kinds)))])
                x += rng.uniform(0.09, 0.14)
            elif kind < 0.8:
                b.part_set(jar, G.compose(G.translate((x + 0.04, y, -6.85)), G.scale(0.7)),
                           {"glass": "glass", "lid": "shelf_wood"})
                x += 0.16
            else:
                for k in range(3):
                    b.ply(b.assets.mesh("tumbler", P.tumbler), G.translate((x + 0.03, y + 0.1 * k, -6.85)),
                          "ceramic_white" if level != 1 else "copper")
                x += 0.12
            x += rng.uniform(0.0, 0.05)

    put(b, "book_encyclopedia_set_01", at=(3.05, 1.45, -6.85), yaw=0, scale=0.9)
    put(b, "jug_01", at=(3.05, 1.85, -6.85), yaw=30)
    put(b, "brass_vase_02", at=(3.05, 2.25, -6.85), height=0.36)

    # Chalkboard menu and the back-wall string lights.
    board_c, board_w, board_h = (-2.35, 1.95, ROOM_Z0 + 0.02), 1.5, 1.0
    b.rect(G.compose(G.translate(board_c), G.scale((board_w / 2, board_h / 2, 1))), "chalkboard")
    for dx, dy, w, hh in ((0, board_h / 2, board_w + 0.06, 0.05), (0, -board_h / 2, board_w + 0.06, 0.05),
                          (board_w / 2, 0, 0.05, board_h), (-board_w / 2, 0, 0.05, board_h)):
        box(b, (board_c[0] + dx, board_c[1] + dy, board_c[2] + 0.015), (w, hh, 0.03), "dark_wood")
    b.focus_points["menu"] = list(board_c)
    string_lights(b, ROOM_Z0 + 0.08, 2.75, 0.22, -3.6, 3.6, 4)


def add_walls_decor(b: SceneBuilder) -> None:
    """Scanned framed mirror, framed paintings and a wall clock on the right wall (facing -x)."""
    put(b, "ornate_mirror_01", at=(ROOM_X - 0.02, 1.3, -3.6), yaw=-90, scale=1.2)
    put(b, "hanging_picture_frame_01", at=(ROOM_X - 0.02, 1.3, -0.6), yaw=-90)
    put(b, "hanging_picture_frame_02", at=(ROOM_X - 0.02, 1.45, -5.9), yaw=-90)
    put(b, "wall_clock", at=(ROOM_X - 0.02, 2.3, -2.05), yaw=-90)
    b.focus_points["mirror"] = [ROOM_X - 0.03, 1.75, -3.6]


def add_lights(b: SceneBuilder) -> None:
    add_sky(b)
    pendant(b, 0.0, 0.05, 1.3, "enamel_green")        # over the hero table, out of frame
    pendant(b, -1.45, -1.55, 1.35, "enamel_cream")
    pendant(b, 1.3, -2.8, 1.35, "enamel_green")
    pendant(b, -1.0, -4.35, 1.3, "enamel_cream")
    for x in (-0.3, 0.7, 1.7, 2.7):
        globe_bulb(b, x, -5.85, 1.0)
    # The entrance end gets its own light so the front wall reads in random poses.
    for x, z, drop in ((-1.6, 3.15, 1.45), (0.6, 3.15, 1.45), (2.3, 2.95, 0.9)):
        globe_bulb(b, x, z, drop)


def build(ctx: BuildContext) -> SceneBundle:
    rng = np.random.default_rng(ctx.seed)
    b = SceneBuilder(Assets(ctx.shared_dir, ctx.rebuild))
    add_materials(b)
    add_room(b)
    add_hero_table(b)
    add_midground(b, rng)
    add_counter(b, rng)
    add_walls_decor(b)
    add_entrance(b)
    add_street(b, rng)
    add_lights(b)
    # Daylight gets as many light samples as all the lamps together: it lights
    # most of the room, but lamp-lit corners it cannot reach still converge.
    b.d["sky"]["sampling_weight"] = b.lamp_weight
    return SceneBundle(b.d, b.focus_points, {"sun_direction": list(SUN_DIRECTION), "daylight_scale": DAYLIGHT_SCALE})


SCENE = SceneDef(
    id="cafe", group="artificial", owner="rui",
    description="Daytime cafe interior: window sun, pendant and string lights, glass, metal, ceramic, plants.",
    build=build,
    views={
        # Seated at the hero table, looking down the room toward the counter.
        "home": View((0.34, 1.22, 1.85), (-0.4, 0.88, -3.0), focus="teapot"),
        # Lower and closer: strong foreground, the counter far behind.
        "close": View((0.12, 0.98, 0.95), (-0.25, 0.86, -2.0), focus="teapot"),
        # From the front corner, taking in the windows and most of the room.
        "wide": View((2.6, 1.55, 2.9), (-1.2, 1.0, -3.0), focus="teapot"),
        # Through the second window onto the street and the buildings across the road.
        "window": View((-1.0, 1.4, -0.4), (-3.75, 1.65, -1.45), focus="facade"),
        # Back toward the entrance wall: door, coat stand, radiator, clock.
        "entrance": View((-0.4, 1.3, 1.1), (2.0, 1.05, 3.45), focus="door"),
    },
    default_view="home",
    # Cameras stay 0.6 m off the walls and in front of the counter, at seated-to-standing height;
    # targets sit at table-to-eye level, so random poses frame the room rather than beams or blank wall.
    camera_box=((-3.2, 0.6, -5.3), (3.2, 2.2, 2.95)),
    target_box=((-3.8, 0.4, -7.0), (3.8, 2.0, 3.5)),
    tags=("indoor", "day", "artificial_light", "clutter", "glass", "metal", "bokeh"),
    default_seed=7,
    asset_version=2,
)
