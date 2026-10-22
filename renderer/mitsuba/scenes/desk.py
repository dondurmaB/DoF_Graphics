"""A study desk close-up for Mitsuba 3: fine text and thin edges over a shallow-to-moderate depth range.

Units are meters, y up, the reader sits at +z looking toward -z. A 1.8 m desk stands in a 4.8 x 5.7 m
study, facing a bookcase on the far wall (about 4 m from the chair) with a window in the left wall.

  desk top (y = 0.75)  desk mat, keyboard, mouse, monitor with a code editor, open ruled notebook with
                       handwriting and a diagram, three printed sheets (report, table, marked-up),
                       sticky notes, pen cup with pens and pencils, mug, headphones, book stack, plant
  lighting             sun and sky through the window, a warm desk lamp over the notebook, the monitor
  room                 oak floor and rug, bookcase of ~300 spined books, pinboard, framed art, sideboard

All lettering comes from stroke templates in `_desk_text.py`; mesh recipes are in `_desk_props.py`.
Seed-independent meshes and textures live in `ctx.shared_dir`; layout, colours, which props appear and
every page of text depend on `ctx.seed` and live in `ctx.assets_dir`.
"""

from __future__ import annotations

import numpy as np

import env_kit
import web_assets as W
import procedural as G
from procedural import Mesh
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, bitmap, luminance, principled, rgb
from scenes import _desk_props as D
from scenes import _desk_text as T

ROOM_X0, ROOM_X1, ROOM_Z0, ROOM_Z1, ROOM_H = -2.4, 2.4, -4.2, 1.5, 2.7
DESK_Y = 0.75
WINDOW_Z, WINDOW_Y = (-0.3, 1.25), (0.9, 2.3)
SUN_ELEVATION, SUN_AZIMUTH = 28.0, -55.0
# Sunsky at scale 1 puts a sunlit wall near radiance 0.1; the room is lit through one window, so scale up
# until the median of the sharp pass sits in the contract's band. Not the cafe's 60: its interior is darker.
SKY_SCALE = 28.0
LAMP = (120.0, 76.0, 32.0)          # warm bulb radiance; about 0.4 m above the notebook
CEILING = (5.8, 5.1, 4.3)           # ceiling light level: daylight alone leaves the far end of the room dark
# The two pendants' frosted shades are the emitters (about 0.42 m^2 each, where the old panels were 0.72 m^2).
PENDANT = tuple(1.7 * c for c in CEILING)
OUTER_X = ROOM_X0 - 0.25            # outside face of the window wall
STRIP = (8.0, 6.8, 5.0)             # LED strip along the top of the bookcase, washing the spines
MONITOR_POS = (-0.20, DESK_Y, -0.60)
NOTEBOOK_POS = (0.52, DESK_Y, -0.30)
LAMP_BASE = (0.82, DESK_Y, -0.62)
CUP_POS = (-0.64, DESK_Y, -0.52)


def slab(b: SceneBuilder, centre, size, material: str, yaw: float = 0.0) -> None:
    """A box whose uv is in meters, so the scanned textures land at their real-world scale."""
    b.box_mesh("x".join(f"{s:.4f}" for s in size), centre, size, material, yaw)


def web(b: SceneBuilder, aid: str, at=(0.0, 0.0, 0.0), yaw: float = 0.0, scale: float = 1.0, height=None, prefix=None,
        emit=None, skip=("glass",), overrides=None):
    """Place a scanned model. Parts named in `emit` become emitters {part: (radiance, power)}; glass parts are skipped
    (their transmission is not converted, and a pane would hide what is behind it from light sampling)."""
    m = W.model(aid)
    M = W.place(m, at=at, yaw=yaw, height=height, scale=scale)
    prefix = prefix or f"web_{aid}"
    for part in m.parts:
        if emit and part.name in emit:
            rad, power = emit[part.name]
            b.ply(part.ply, M, "emitter_backing", emission=rad, prefix=prefix, power=power)
            continue
        if any(s in part.name.lower() for s in skip):
            continue
        mid = f"{prefix}__{part.name}"
        if mid not in b.d:
            b.material(mid, (overrides or {}).get(part.name, part.bsdf))
        b.ply(part.ply, M, mid, prefix=prefix)
    return M


def world(M, p) -> list:
    return (np.asarray(M) @ np.array([*p, 1.0]))[:3].tolist()


def rng_for(ctx: BuildContext, k: int) -> np.random.Generator:
    """One stream per component, so a warm cache never changes what later components draw."""
    return np.random.default_rng([ctx.seed, k])


# ----------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------

def paper(path: str) -> dict:
    return {"type": "twosided", "bsdf": {"type": "diffuse", "reflectance": bitmap(path)}}


def add_materials(b: SceneBuilder, local: Assets, ctx: BuildContext) -> None:
    A = b.assets
    rng = rng_for(ctx, 0)

    wall_tint = [(1.16, 1.10, 0.97), (1.12, 1.10, 1.02), (1.14, 1.05, 0.92)][ctx.seed % 3]
    b.material("floor", W.surface("herringbone_parquet"))
    b.material("rug", {"type": "diffuse", "reflectance": bitmap(A.texture("rug", lambda: T.rug(512, 768, 8)))})
    b.material("wall", W.surface("painted_plaster_wall", tint=wall_tint))
    b.material("wall_back", W.surface("painted_plaster_wall", tint=(0.34, 0.47, 0.40)))
    b.material("ceiling", W.surface("white_plaster_02", tint=(1.5, 1.5, 1.55)))
    b.material("ground", env_kit.ground("clear"))
    b.material("desk_wood", W.surface("american_walnut_veneer", tint=(1.15, 0.78, 0.58)))
    b.material("desk_dark", W.surface("black_walnut_veneer_01", tint=(0.50, 0.38, 0.30)))
    b.material("case_wood", W.surface("oak_veneer_01"))
    b.material("leather", W.surface("brown_leather"))
    b.material("white_paint", W.surface("acg:PaintedWood009C"))
    b.material("curtain", {"type": "twosided", "bsdf": W.surface("acg:Fabric045")})
    # outside
    b.material("grass", W.surface("leafy_grass"))
    b.material("pavement", W.surface("brick_pavement"))
    b.material("asphalt", W.surface("asphalt_02"))
    b.material("kerb", principled((0.55, 0.54, 0.52), 0.8))
    b.material("garden_brick", W.surface("brick_wall_006"))
    for i, tex in enumerate(("brick_wall_006", "white_stucco", "plastered_stone_wall")):
        b.material(f"facade_{i}", W.surface(tex))
    b.material("window_glass", principled((0.015, 0.018, 0.022), 0.04, specular=0.6))
    b.material("roof_slate", principled((0.10, 0.10, 0.11), 0.6))
    b.material("front_door", principled((0.08, 0.16, 0.12), 0.35, clearcoat=0.4))
    b.material("aluminium", {"type": "roughconductor", "material": "Al", "alpha": 0.2})
    b.material("chrome", {"type": "roughconductor", "material": "Cr", "alpha": 0.06})
    b.material("brass", {"type": "roughconductor", "material": "Au", "alpha": 0.2})
    b.material("black", principled((0.012, 0.012, 0.012), 0.35))
    b.material("monitor_body", principled((0.03, 0.03, 0.035), 0.35))
    b.material("mouse_plastic", principled((0.82, 0.82, 0.84), 0.3, specular=0.5))
    b.material("keycaps", principled(bitmap(A.texture("keycaps", lambda: D.keycap_atlas())), 0.45))
    b.material("lamp_metal", principled((0.025, 0.04, 0.03), 0.3, clearcoat=0.4, clearcoat_gloss=0.6))
    b.material("shade_inner", {"type": "diffuse", "reflectance": rgb(0.88, 0.86, 0.82)})
    b.material("emitter_backing", {"type": "diffuse", "reflectance": rgb(0.0, 0.0, 0.0)})
    b.material("cover_black", principled((0.03, 0.03, 0.035), 0.75))
    b.material("page_edge", {"type": "diffuse", "reflectance": rgb(0.80, 0.77, 0.66)})
    b.material("paper_plain", {"type": "diffuse", "reflectance": rgb(0.82, 0.78, 0.68)})
    b.material("spines", principled(bitmap(A.texture("spines", lambda: T.spine_atlas(5))), 0.55, specular=0.3))
    b.material("headphone_plastic", principled((0.03, 0.03, 0.035), 0.4, specular=0.5))
    b.material("headphone_pad", principled((0.02, 0.02, 0.022), 0.6))
    b.material("chair_fabric", principled((0.10, 0.11, 0.13), 0.95))
    b.material("pen_cup", principled((0.03, 0.03, 0.03), 0.5))
    b.material("coffee", principled((0.03, 0.012, 0.005), 0.1, specular=0.5))
    b.material("terracotta", principled((0.56, 0.27, 0.15), 0.8))
    b.material("soil", {"type": "diffuse", "reflectance": rgb(0.06, 0.04, 0.03)})
    b.material("leaf", {"type": "twosided", "bsdf": principled((0.07, 0.24, 0.07), 0.45)})
    b.material("leaf_dark", {"type": "twosided", "bsdf": principled((0.04, 0.15, 0.06), 0.4)})
    b.material("stem", principled((0.12, 0.3, 0.08), 0.6))
    b.material("wood_pencil", principled((0.75, 0.58, 0.36), 0.6))
    b.material("graphite", principled((0.08, 0.08, 0.09), 0.3))
    b.material("eraser", principled((0.80, 0.45, 0.45), 0.7))
    for i, c in enumerate([(0.55, 0.08, 0.06), (0.08, 0.2, 0.5), (0.03, 0.03, 0.04), (0.1, 0.45, 0.2), (0.85, 0.85, 0.82),
                           (0.85, 0.5, 0.05)]):
        b.material(f"pen_{i}", principled(c, 0.25, specular=0.6, clearcoat=0.4, clearcoat_gloss=0.7))
    for i, c in enumerate([(0.9, 0.75, 0.1), (0.1, 0.3, 0.6), (0.7, 0.12, 0.1), (0.15, 0.5, 0.25)]):
        b.material(f"pencil_{i}", principled(c, 0.35, clearcoat=0.3))
    for i, c in enumerate([(0.9, 0.88, 0.84), (0.07, 0.19, 0.36), (0.45, 0.55, 0.47), (0.75, 0.32, 0.2)]):
        b.material(f"mug_{i}", principled(c, 0.15, specular=0.6, clearcoat=0.5, clearcoat_gloss=0.5))
    # Desk book stack: cloth covers (tinted) and two in leather. The cubes' uv spans 0-1 per face, about 0.25 m.
    for i, c in enumerate([(0.45, 0.08, 0.06), (0.08, 0.2, 0.3), (0.55, 0.42, 0.12), (0.12, 0.25, 0.14), (0.3, 0.12, 0.25),
                           (0.75, 0.7, 0.6)]):
        if i in (0, 4):
            b.material(f"book_{i}", W.surface("acg:Leather026", uv_scale=0.25, tint=tuple(min(2.5, 2.2 * v) for v in c)))
        else:
            b.material(f"book_{i}", W.surface("acg:Fabric030", uv_scale=0.25, tint=tuple(min(2.5, 2.2 * v) for v in c)))

    # Seed-dependent pages: the text itself changes with the seed.
    seed = ctx.seed
    b.material("notebook_pages", {"type": "diffuse", "reflectance": bitmap(
        local.texture("notebook", lambda: T.notebook_spread(2000, 2800, seed)))})
    kinds = [str(k) for k in rng.permutation(["report", "table", "marked"])]
    for i, kind in enumerate(kinds):
        b.material(f"page_{i}", paper(local.texture(f"page_{i}", lambda i=i, kind=kind: T.printed_page(1400, 990, seed * 10 + i, kind))))
    sticky = [(0.98, 0.88, 0.35), (0.95, 0.55, 0.65), (0.55, 0.85, 0.65), (0.55, 0.75, 0.95)]
    for i, c in enumerate(sticky):
        b.material(f"sticky_{i}", {"type": "diffuse", "reflectance": bitmap(local.texture(
            f"sticky_{i}", lambda i=i, c=c: np.flipud(T.sticky_note(512, seed * 7 + i, c))))})
    b.material("pinboard", {"type": "diffuse", "reflectance": bitmap(A.texture("pinboard", lambda: np.flipud(T.pinboard(768, 1024, 7))))})


# ----------------------------------------------------------------------------
# Room, window, outside
# ----------------------------------------------------------------------------

def add_room(b: SceneBuilder) -> None:
    cx, cz = 0.5 * (ROOM_X0 + ROOM_X1), 0.5 * (ROOM_Z0 + ROOM_Z1)
    wx, wz = ROOM_X1 - ROOM_X0, ROOM_Z1 - ROOM_Z0
    slab(b, (cx, -0.02, cz), (wx, 0.04, wz), "floor")
    b.rect(G.compose(G.translate((0.0, 0.004, -0.35)), G.rotate((1, 0, 0), -90), G.scale((1.3, 0.95, 1))), "rug")
    slab(b, (cx, ROOM_H + 0.02, cz), (wx, 0.04, wz), "ceiling")
    slab(b, (cx, ROOM_H / 2, ROOM_Z0 - 0.02), (wx, ROOM_H, 0.04), "wall_back")
    slab(b, (cx, ROOM_H / 2, ROOM_Z1 + 0.02), (wx, ROOM_H, 0.04), "wall")
    slab(b, (ROOM_X1 + 0.02, ROOM_H / 2, cz), (0.04, ROOM_H, wz), "wall")
    # skirting, and a cornice where the walls meet the ceiling
    for c, s in (((cx, 0.05, ROOM_Z1 - 0.01), (wx, 0.1, 0.02)), ((ROOM_X1 - 0.01, 0.05, cz), (0.02, 0.1, wz)),
                 ((ROOM_X0 + 0.01, 0.05, cz), (0.02, 0.1, wz)), ((cx, 0.05, ROOM_Z0 + 0.01), (wx, 0.1, 0.02)),
                 ((cx, ROOM_H - 0.04, ROOM_Z1 - 0.03), (wx, 0.08, 0.06)), ((ROOM_X1 - 0.03, ROOM_H - 0.04, cz), (0.06, 0.08, wz)),
                 ((ROOM_X0 + 0.03, ROOM_H - 0.04, cz), (0.06, 0.08, wz)), ((cx, ROOM_H - 0.04, ROOM_Z0 + 0.03), (wx, 0.08, 0.06))):
        slab(b, c, s, "white_paint")

    # Left wall with the window opening; thick blocks give the opening real reveals.
    t = 0.25
    wxc = ROOM_X0 - t / 2
    z0, z1 = WINDOW_Z
    y0, y1 = WINDOW_Y
    slab(b, (wxc, ROOM_H / 2, 0.5 * (ROOM_Z0 + z0)), (t, ROOM_H, z0 - ROOM_Z0), "wall")
    slab(b, (wxc, ROOM_H / 2, 0.5 * (z1 + ROOM_Z1)), (t, ROOM_H, ROOM_Z1 - z1), "wall")
    slab(b, (wxc, y0 / 2, 0.5 * (z0 + z1)), (t, y0, z1 - z0), "wall")
    slab(b, (wxc, 0.5 * (y1 + ROOM_H), 0.5 * (z0 + z1)), (t, ROOM_H - y1, z1 - z0), "wall")
    zc = 0.5 * (z0 + z1)
    f = 0.05
    for z in (z0 + f / 2, z1 - f / 2, zc):
        slab(b, (wxc + 0.06, 0.5 * (y0 + y1), z), (0.06, y1 - y0, f if z != zc else 0.035), "white_paint")
    for y in (y0 + f / 2, y1 - f / 2, y0 + 0.62):
        slab(b, (wxc + 0.06, y, zc), (0.06, f if y != y0 + 0.62 else 0.035, z1 - z0), "white_paint")
    slab(b, (ROOM_X0 + 0.07, y0 - 0.02, zc), (0.30, 0.04, z1 - z0 + 0.1), "white_paint")
    # Curtains gathered beside the opening (never across it) on a brass rod.
    for i, (za, zb) in enumerate(((z0 - 0.62, z0 - 0.04), (z1 + 0.03, ROOM_Z1 - 0.03))):
        cur = b.assets.mesh(f"curtain_{i}", lambda za=za, zb=zb, i=i: D.curtain(zb - za, 2.38, seed=i))
        b.ply(cur, G.translate((ROOM_X0 + 0.07, 0.04, za)), "curtain", prefix="curtain")
    rod = b.assets.mesh("curtain_rod", lambda: D.tube((0, 0, z0 - 0.7), (0, 0, ROOM_Z1 - 0.02), 0.012, 12))
    b.ply(rod, G.translate((ROOM_X0 + 0.07, 2.44, 0)), "brass", prefix="curtain_rod")

    # Front wall: a four-panel door with casing, a light switch.
    door = b.assets.meshes("panel_door", D.panel_door)
    dm = G.compose(G.translate((1.6, 0.0, ROOM_Z1 - 0.03)), G.rotate((0, 1, 0), 180))
    b.part_set(door, dm, {"slab": "case_wood", "mould": "case_wood", "handle": "brass"})
    for c, s in (((1.6 - 0.48, 1.07, ROOM_Z1 - 0.015), (0.08, 2.14, 0.03)), ((1.6 + 0.48, 1.07, ROOM_Z1 - 0.015), (0.08, 2.14, 0.03)),
                 ((1.6, 2.1, ROOM_Z1 - 0.015), (1.04, 0.08, 0.03))):
        slab(b, c, s, "white_paint")
    slab(b, (0.95, 1.2, ROOM_Z1 - 0.006), (0.08, 0.12, 0.012), "white_paint")
    b.focus_points["door"] = [1.6, 1.2, ROOM_Z1 - 0.06]


def add_bookcase(b: SceneBuilder, local: Assets, ctx: BuildContext) -> None:
    w, depth, h = 3.0, 0.34, 2.28
    zc = ROOM_Z0 + depth / 2
    levels = [0.10, 0.54, 0.98, 1.42, 1.86]
    slab(b, (-w / 2 + 0.015, h / 2, zc), (0.03, h, depth), "case_wood")
    slab(b, (w / 2 - 0.015, h / 2, zc), (0.03, h, depth), "case_wood")
    slab(b, (0, h - 0.015, zc), (w, 0.03, depth), "case_wood")
    slab(b, (0, 0.04, zc), (w - 0.06, 0.08, depth), "case_wood")
    for y in levels:
        slab(b, (0, y - 0.0125, zc), (w - 0.06, 0.025, depth), "case_wood")
    z_front = ROOM_Z0 + depth - 0.03
    rng = rng_for(ctx, 5)
    ends = {0: (-1.46, 1.08), 1: (-1.46, 1.02), 2: (-0.66, 1.46), 3: (-1.46, 1.05), 4: (-1.46, 1.0)}   # gaps hold ornaments

    def build() -> Mesh:
        out = Mesh()
        for i, y in enumerate(levels):
            out += D.shelf_row(rng, *ends[i], y, z_front, 0.37)
        return out

    b.ply(local.mesh("books_v2", build), np.eye(4), "spines", prefix="books")
    orng = rng_for(ctx, 6)
    zo = z_front - 0.12
    web(b, "book_encyclopedia_set_01", (-1.17, levels[2], z_front - 0.09))
    web(b, "cassette_player", (-0.8, levels[2], zo), yaw=float(orng.uniform(-20, 20)))
    for i in (1, 3):
        web(b, "potted_plant_04", (1.3, levels[i], zo), yaw=float(orng.uniform(0, 360)), prefix="plant_shelf")
    web(b, "brass_vase_03", (1.12, levels[1], zo))
    web(b, "alarm_clock_01", (1.14, levels[3], zo), yaw=float(orng.uniform(-25, 10)))
    web(b, "Camera_01", (1.14, levels[4], zo), yaw=float(orng.uniform(-30, 30)))
    web(b, "standing_picture_frame_02", (1.34, levels[4], zo), yaw=-90 + float(orng.uniform(-15, 15)))
    web(b, "ceramic_vase_04", (1.2, levels[0], zo))
    web(b, "brass_vase_04", (1.4, levels[0], zo), yaw=float(orng.uniform(0, 360)))
    web(b, "cardboard_box_01", (-0.95, h, ROOM_Z0 + 0.19), yaw=float(orng.uniform(-8, 8)), scale=0.7)
    web(b, "cardboard_box_01", (0.55, h, ROOM_Z0 + 0.19), yaw=90 + float(orng.uniform(-8, 8)), scale=0.55, prefix="box_small")
    b.focus_points["shelf_books"] = [-0.3, levels[2] + 0.2, z_front]


def add_walls_decor(b: SceneBuilder, ctx: BuildContext) -> None:
    rng = rng_for(ctx, 7)
    face_left = G.rotate((0, 1, 0), -90)
    b.rect(G.compose(G.translate((ROOM_X1 - 0.012, 1.45, -1.4)), face_left, G.scale((0.62, 0.42, 1))), "pinboard")
    for dz, dy, ww, hh in ((0, 0.42, 1.28, 0.04), (0, -0.42, 1.28, 0.04), (0.62, 0, 0.04, 0.88), (-0.62, 0, 0.04, 0.88)):
        slab(b, (ROOM_X1 - 0.02, 1.45 + dy, -1.4 + dz), (0.03, hh, ww), "desk_dark")
    b.focus_points["pinboard"] = [ROOM_X1 - 0.02, 1.5, -1.4]
    # Scanned frames with their own art on the front wall, hung by their backs (local +z faces the room).
    web(b, "hanging_picture_frame_01", (-1.2, 1.6 - 0.42, ROOM_Z1 - 0.009), yaw=180)
    web(b, "hanging_picture_frame_02", (0.5, 1.6 - 0.28, ROOM_Z1 - 0.019), yaw=180)
    web(b, "vintage_telephone_wall_clock", (ROOM_X1 - 0.097, 1.55, -3.3), yaw=-90)

    # Sideboard against the right wall, with things on it.
    sz = -2.0
    slab(b, (ROOM_X1 - 0.24, 0.4, sz), (0.46, 0.8, 1.5), "desk_dark")
    for k in range(3):
        slab(b, (ROOM_X1 - 0.475, 0.18 + 0.27 * k, sz), (0.012, 0.22, 1.42), "case_wood")
    top = 0.8
    web(b, "potted_plant_02", (ROOM_X1 - 0.26, top, sz - 0.5), yaw=float(rng.uniform(0, 360)), scale=0.75)
    web(b, "wicker_basket_01", (ROOM_X1 - 0.25, top, sz - 0.02), yaw=90 + float(rng.uniform(-10, 10)))
    web(b, "binder_notebook", (ROOM_X1 - 0.25, top, sz + 0.42), yaw=90 + float(rng.uniform(-6, 6)))
    web(b, "clipboard", (ROOM_X1 - 0.27, top + 0.025, sz + 0.47), yaw=95 + float(rng.uniform(-10, 10)))
    web(b, "mantel_clock_01", (ROOM_X1 - 0.24, top, sz + 0.12), yaw=-90)
    # Reading corner at the back right: armchair, cushions, side table with a vase.
    web(b, "ArmChair_01", (1.98, 0.0, -3.3), yaw=-90 + float(rng.uniform(-12, 12)))
    web(b, "throw_pillows_01", (2.02, 0.43, -3.3), yaw=-90, scale=0.55)
    web(b, "side_table_01", (2.1, 0.0, -3.96))
    web(b, "antique_ceramic_vase_01", (2.1, 0.551, -3.96), yaw=float(rng.uniform(0, 360)), scale=0.8)
    # Plants and a floor vase on the window side.
    web(b, "potted_plant_01", (-2.0, 0.0, -2.6), yaw=float(rng.uniform(0, 360)))
    web(b, "ceramic_vase_01", (-2.05, 0.0, -3.9), yaw=float(rng.uniform(0, 360)))
    b.focus_points["armchair"] = [1.98, 0.6, -3.3]


def add_outside(b: SceneBuilder, ctx: BuildContext) -> None:
    """The street seen through the window, all real geometry at real distances (the window wall is at x = -2.65):
    front garden 0-3.5 m, a low brick wall, pavement to 6 m, the road to 13 m, the far pavement, then a terrace of
    houses about 15.5 m away. Nothing taller than the sun's line through the window stands in its way."""
    rng = rng_for(ctx, 8)
    span = 80.0
    x_wall, x_kerb, x_road, x_far = OUTER_X - 3.5, OUTER_X - 6.0, OUTER_X - 13.0, OUTER_X - 15.5
    slab(b, ((OUTER_X + x_wall) / 2, -0.02, 0), (OUTER_X - x_wall, 0.04, span), "grass")
    slab(b, (x_wall - 0.125, 0.33, 0), (0.25, 0.7, span), "garden_brick")
    slab(b, (x_wall - 0.125, 0.70, 0), (0.32, 0.05, span), "kerb")
    slab(b, ((x_wall - 0.25 + x_kerb) / 2, -0.02, 0), (x_wall - 0.25 - x_kerb, 0.04, span), "pavement")
    for xk in (x_kerb, x_road):
        slab(b, (xk, -0.07, 0), (0.15, 0.14, span), "kerb")
    slab(b, ((x_kerb + x_road) / 2, -0.15, 0), (x_kerb - x_road, 0.04, span), "asphalt")
    slab(b, ((x_road + x_far) / 2, -0.02, 0), (x_road - x_far, 0.04, span), "pavement")
    b.rect(G.compose(G.translate((0, -0.3, 0)), G.rotate((1, 0, 0), -90), G.scale((400.0, 400.0, 1))), "ground")
    # Terraced houses across the road, front faces toward the window.
    z = -30.0
    k = 0
    while z < 30.0:
        width = float(rng.uniform(7.5, 10.0))
        height = float(rng.uniform(7.0, 8.6))
        parts = b.assets.meshes(f"facade_{k % 5}", lambda w=width, h=height, k=k: D.facade(w, h, 2 if h < 7.8 else 3, 3, 40 + k))
        b.part_set(parts, G.translate((x_far, 0.0, z)), {"wall": f"facade_{k % 3}", "glass": "window_glass",
                                                          "frame": "white_paint", "roof": "roof_slate", "door": "front_door"})
        z += width + 0.02
        k += 1
    # Garden, pavement and street furniture.
    web(b, "planter_box_02", (OUTER_X - 0.5, 0.0, 0.45), yaw=90)
    for zz, yaw in ((-4.8, 90.0), (6.8, 270.0)):
        web(b, "shrub_02", (x_wall + 0.9, 0.0, zz), yaw=yaw, scale=0.7)
    web(b, "tree_small_02", (x_kerb + 0.9, 0.0, -3.6), yaw=float(rng.uniform(0, 360)), scale=1.25)
    web(b, "street_lamp_01", (x_kerb + 0.35, 0.0, 3.6), yaw=90)
    web(b, "painted_wooden_bench", (x_kerb + 1.2, 0.0, 1.4), yaw=90)
    web(b, "fire_hydrant", (x_kerb + 0.5, 0.0, -1.2), yaw=float(rng.uniform(0, 360)), scale=0.6)
    web(b, "utility_box_01", (x_wall - 0.6, 0.0, -7.2), yaw=90)
    web(b, "covered_car", (x_kerb - 1.3, -0.13, float(rng.uniform(-1.5, 0.5))), yaw=float(rng.choice([0.0, 180.0])))
    b.focus_points["street"] = [x_kerb - 1.3, 0.8, -0.5]
    b.focus_points["houses"] = [x_far, 3.5, 0.5]


def add_desk(b: SceneBuilder, local: Assets, ctx: BuildContext) -> None:
    A = b.assets
    rng = rng_for(ctx, 1)
    mx, my, mz = MONITOR_POS
    b.box_mesh("desk_top", (0, DESK_Y - 0.0175, -0.4), (1.8, 0.035, 0.8), "desk_wood")
    slab(b, (-0.865, 0.3575, -0.4), (0.03, 0.715, 0.7), "desk_dark")                 # left side panel
    slab(b, (0.65, 0.3575, -0.42), (0.46, 0.715, 0.68), "desk_dark")                 # drawer pedestal
    for k in range(3):
        slab(b, (0.65, 0.12 + 0.22 * k, -0.077), (0.42, 0.2, 0.012), "desk_wood")
        b.cube((0.65, 0.17 + 0.22 * k, -0.066), (0.16, 0.012, 0.012), "brass")
    slab(b, (0.0, 0.55, -0.76), (1.7, 0.3, 0.02), "desk_dark")                       # modesty panel
    web(b, "wicker_basket_02", (-0.45, 0.0, -0.42), yaw=float(rng.uniform(0, 360)))   # wastepaper basket
    b.box_mesh("mat", (-0.12, DESK_Y + 0.0015, -0.22), (0.78, 0.003, 0.36), "leather")

    # Keyboard and mouse
    kb = A.meshes("keyboard", lambda: D.keyboard(3))
    kb_m = G.compose(G.translate((-0.20, DESK_Y + 0.0045, -0.20)), G.rotate((0, 1, 0), float(rng.uniform(-3, 3))), G.rotate((1, 0, 0), 3))
    b.part_set(kb, kb_m, {"body": "aluminium", "keys": "keycaps"})
    b.focus_points["keyboard"] = [-0.20, DESK_Y + 0.02, -0.20]
    ms = A.meshes("mouse", D.mouse)
    b.part_set(ms, G.compose(G.translate((0.17, DESK_Y + 0.0035, -0.20)), G.rotate((0, 1, 0), float(rng.uniform(-20, 5)))),
               {"shell": "mouse_plastic", "dark": "black"})
    b.focus_points["mouse"] = [0.17, DESK_Y + 0.03, -0.20]

    # Monitor with a lit screen
    mon = A.meshes("monitor", D.monitor)
    b.part_set(mon, G.translate(MONITOR_POS), {"body": "monitor_body", "stand": "aluminium", "logo": "chrome"})
    cy = D.MONITOR_CENTRE_Y
    panel = G.compose(G.translate(MONITOR_POS), G.translate((0, cy, 0)), G.rotate((1, 0, 0), -5))
    screen_tex = A.texture("screen", lambda: np.flipud(T.code_screen(1152, 2048, 6)))
    b.rect(G.compose(panel, G.translate((0, 0, 0.0064)), G.scale((D.MONITOR_W / 2, D.MONITOR_H / 2, 1))), "emitter_backing",
           emission={"type": "bitmap", "filename": screen_tex, "raw": False}, power=D.MONITOR_W * D.MONITOR_H * 0.045)
    b.focus_points["monitor"] = [mx, my + cy, mz + 0.01]
    # Sticky notes on the bezel and flat on the desk
    notes = [(0.255, -0.115, 8), (0.272, -0.035, -12), (-0.275, 0.10, 5)]
    for i, (nx, ny, yaw) in enumerate(notes):
        b.rect(G.compose(panel, G.translate((nx, ny, 0.0069 + 0.0001 * i)), G.rotate((0, 0, 1), yaw), G.scale((0.038, 0.038, 1))),
               f"sticky_{(i + ctx.seed) % 4}")
    for i, (px, pz, yaw) in enumerate([(-0.52, -0.40, 20), (-0.46, -0.38, -8)]):
        b.rect(G.compose(G.translate((px, DESK_Y + 0.0007 + 0.0006 * i, pz)), G.rotate((0, 1, 0), yaw), G.rotate((1, 0, 0), -90),
                         G.scale((0.038, 0.038, 1))), f"sticky_{(i + 2 + ctx.seed) % 4}")
    b.focus_points["sticky_notes"] = [mx + 0.255, my + cy - 0.115, mz + 0.01]

    # Open notebook with a pen across it
    nb = A.meshes("notebook", D.notebook_pages)
    nb_yaw = 10.0 + float(rng.uniform(-4, 4))
    nb_m = G.compose(G.translate(NOTEBOOK_POS), G.rotate((0, 1, 0), nb_yaw))
    b.part_set(nb, nb_m, {"pages": "notebook_pages", "edges": "page_edge", "cover": "cover_black"})
    pen = A.meshes("ballpoint", D.ballpoint)
    pen_m = G.compose(nb_m, G.translate((0.06, 0.0147 + 0.0052, 0.06)), G.rotate((0, 1, 0), 28), G.rotate((0, 0, 1), -90))
    b.part_set(pen, pen_m, {"body": "pen_1", "metal": "chrome"})
    tip = (pen_m @ np.array([0, 0, 0, 1.0]))[:3]
    b.focus_points["pen_tip"] = tip.tolist()
    b.focus_points["notebook_text"] = (nb_m @ np.array([-0.10, 0.0165, 0.02, 1.0]))[:3].tolist()
    # Loose printed sheets behind the notebook
    for i, (x, z, yaw) in enumerate([(0.30, -0.60, -14), (0.34, -0.61, 6), (0.27, -0.58, 21)]):
        sheet = local.mesh(f"sheet_{i}", lambda i=i: D.paper_sheet(0.21, 0.297, 0.004, ctx.seed * 3 + i))
        b.ply(sheet, G.compose(G.translate((x, DESK_Y + 0.0012 + 0.0011 * i, z)), G.rotate((0, 1, 0), yaw)), f"page_{i}", prefix="sheet")
    b.focus_points["printed_page"] = [0.27, DESK_Y + 0.006, -0.58]

    # Pen cup with pens and pencils
    cx, cy0, cz = CUP_POS
    b.ply(A.mesh("pen_cup", D.pen_cup), G.translate(CUP_POS), "pen_cup")
    prng = rng_for(ctx, 2)
    bp, pc = A.meshes("ballpoint", D.ballpoint), A.meshes("pencil", D.pencil)
    for k in range(9):
        az = float(prng.uniform(0, 360))
        r0 = float(prng.uniform(0.0, 0.012))
        tilt = float(prng.uniform(3, 14))
        off = r0 * np.array([np.cos(np.radians(az)), 0.0, -np.sin(np.radians(az))])
        m = G.compose(G.translate((cx + off[0], cy0 + 0.012, cz + off[2])), G.rotate((0, 1, 0), az), G.rotate((0, 0, 1), -tilt))
        if prng.random() < 0.5:
            b.part_set(bp, m, {"body": f"pen_{int(prng.integers(0, 6))}", "metal": "chrome"})
        else:
            b.part_set(pc, m, {"paint": f"pencil_{int(prng.integers(0, 4))}", "wood": "wood_pencil", "lead": "graphite",
                               "metal": "chrome", "eraser": "eraser"})
    b.focus_points["pen_cup"] = [cx, cy0 + 0.13, cz]

    # Mug, book stack, headphones, plant
    mug = A.meshes("mug", D.mug)
    b.part_set(mug, G.compose(G.translate((-0.60, DESK_Y, -0.17)), G.rotate((0, 1, 0), float(rng.uniform(0, 360)))),
               {"mug": f"mug_{ctx.seed % 4}", "coffee": "coffee"})
    b.focus_points["mug"] = [-0.60, DESK_Y + 0.07, -0.17]
    h = DESK_Y
    brng = rng_for(ctx, 3)
    for i in range(int(brng.integers(3, 5))):
        w_, d_, t_ = brng.uniform(0.15, 0.2), brng.uniform(0.21, 0.26), brng.uniform(0.02, 0.04)
        a = 10 + brng.uniform(-12, 12)
        cover = f"book_{int(brng.integers(0, 6))}"
        b.cube((-0.78 + brng.uniform(-0.01, 0.01), h + t_ / 2, -0.33 + brng.uniform(-0.01, 0.01)), (w_, t_, d_), cover, a)
        b.cube((-0.776, h + t_ / 2, -0.33), (w_ - 0.01, t_ - 0.006, d_ - 0.004), "paper_plain", a)
        h += t_
    b.focus_points["desk_books"] = [-0.78, h, -0.33]
    if rng.random() < 0.8:
        hp = A.meshes("headphones", D.headphones)
        b.part_set(hp, G.compose(G.translate((0.77, DESK_Y, -0.06)), G.rotate((0, 1, 0), float(rng.uniform(-12, 12)))),
                   {"plastic": "headphone_plastic", "cushion": "headphone_pad", "metal": "chrome"})
        b.focus_points["headphones"] = [0.77, DESK_Y + 0.1, -0.06]
    web(b, "potted_plant_04", (-0.80, DESK_Y, -0.68), yaw=float(rng.uniform(0, 360)), prefix="plant_desk")
    b.focus_points["plant"] = [-0.80, DESK_Y + 0.2, -0.68]
    web(b, "standing_picture_frame_01", (-0.46, DESK_Y, -0.74), yaw=-100 + float(rng.uniform(-10, 10)))


def add_lamp(b: SceneBuilder) -> None:
    """Scanned architect lamp; its bulb is the emitter. The head (local -z of the model) hangs over the notebook."""
    to_nb = np.array([NOTEBOOK_POS[0] - LAMP_BASE[0], NOTEBOOK_POS[2] - LAMP_BASE[2]])
    yaw = float(np.degrees(np.arctan2(-to_nb[0], -to_nb[1])))
    at = (LAMP_BASE[0] + 0.55 * to_nb[0], DESK_Y, LAMP_BASE[2] + 0.55 * to_nb[1])
    M = web(b, "desk_lamp_arm_01", at, yaw=yaw,
            emit={"desk_lamp_arm_01_light": (LAMP, 0.015 * float(np.dot(LAMP, (0.2126, 0.7152, 0.0722))))})
    b.focus_points["lamp"] = world(M, (-0.005, 0.695, -0.176))


def add_chair(b: SceneBuilder, ctx: BuildContext) -> None:
    rng = rng_for(ctx, 4)
    web(b, "gallinera_chair", (float(rng.uniform(-0.1, 0.1)), 0.0, 0.5), yaw=180 + float(rng.uniform(-18, 18)))
    b.focus_points["chair"] = [0.0, 0.85, 0.30]


def add_room_lights(b: SceneBuilder) -> None:
    for z in (-1.2, -2.9):
        web(b, "modern_ceiling_lamp_01", (0.2, ROOM_H - 0.952, z), prefix="pendant",
            emit={"modern_ceiling_lamp_01_glass": (PENDANT, 0.42 * luminance(PENDANT))})
    web(b, "ceiling_fan", (0.0, ROOM_H - 0.516, 0.45))
    b.rect(G.compose(G.translate((0.0, 2.23, ROOM_Z0 + 0.36)), G.rotate((1, 0, 0), 60), G.scale((1.4, 0.015, 1))),
           "emitter_backing", emission=STRIP, prefix="case_strip", power=2.8 * 0.03 * luminance(STRIP))


def build(ctx: BuildContext) -> SceneBundle:
    local = Assets(ctx.assets_dir, ctx.rebuild)
    b = SceneBuilder(Assets(ctx.shared_dir, ctx.rebuild))
    add_materials(b, local, ctx)
    add_room(b)
    add_bookcase(b, local, ctx)
    add_walls_decor(b, ctx)
    add_desk(b, local, ctx)
    add_lamp(b)
    add_room_lights(b)
    add_chair(b, ctx)
    add_outside(b, ctx)
    sun = env_kit.default_sun_direction(SUN_ELEVATION, SUN_AZIMUTH)
    b.d["sky"] = env_kit.sky("clear", sun_direction=sun, scale=SKY_SCALE, sampling_weight=2.0 * b.lamp_weight)
    b.focus_points["window"] = [ROOM_X0 - 0.1, 1.6, 0.5 * sum(WINDOW_Z)]
    return SceneBundle(b.d, b.focus_points, {"sun_direction": sun, "sky_scale": SKY_SCALE})


SCENE = SceneDef(
    id="desk", group="artificial", owner="rui",
    description="Study desk close-up: notebook handwriting, printed pages, code on a monitor, keyboard keys, pens, lamp, bookcase behind.",
    build=build,
    views={
        # High angle over the shoulder, close to the open notebook.
        "close": View((0.30, 1.22, 0.42), (0.50, 0.77, -0.30), focus="notebook_text"),
        # Desk level, looking along the desk toward the monitor, the bookcase far behind.
        "along": View((-0.55, 0.98, 0.40), (0.10, 0.92, -2.5), focus="mug"),
        # Wider, from the front right: desk, lamp, chair edge, the whole bookcase behind.
        "wide": View((1.35, 1.55, 1.15), (-0.15, 0.85, -2.2), focus="monitor"),
        # From the chair side toward the window: curtains, the garden and the houses across the street.
        "window": View((0.9, 1.35, 0.6), (-2.2, 1.3, 0.45), focus="street"),
    },
    default_view="close",
    camera_box=((-1.6, 0.85, -0.4), (1.7, 1.9, 1.2)),
    target_box=((-2.3, 0.5, -4.0), (2.3, 1.65, 1.4)),   # desk-to-shelf height, not the ceiling
    lens_mm=(24.0, 70.0),   # a 4.6 m study: longer lenses mostly frame a patch of wall or floor
    exclude_boxes=(
        ((-1.0, 0.0, -0.9), (1.0, 0.80, 0.1)),        # desk body
        ((-0.52, 0.75, -0.72), (0.12, 1.2, -0.46)),   # monitor
        ((0.45, 0.75, -0.82), (0.98, 1.70, -0.22)),   # architect lamp
        ((-0.40, 0.0, 0.10), (0.40, 1.15, 0.80)),     # chair
        ((-0.9, 0.75, -0.8), (-0.55, 0.95, -0.45)),   # pen cup and plant
        ((-2.4, 0.0, -3.0), (-1.5, 1.9, -2.0)),       # floor plant
    ),
    tags=("indoor", "day", "artificial_light", "closeup", "clutter", "metal"),
    default_seed=3,
    asset_version=4,
    spp_hint=2048,
)
