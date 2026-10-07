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
import procedural as G
import props as P
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
CEILING = (5.8, 5.1, 4.3)           # two warm ceiling panels: daylight alone leaves the far end of the room dark
STRIP = (8.0, 6.8, 5.0)             # LED strip along the top of the bookcase, washing the spines
MONITOR_POS = (-0.20, DESK_Y, -0.60)
NOTEBOOK_POS = (0.52, DESK_Y, -0.30)
LAMP_BASE = (0.82, DESK_Y, -0.62)
CUP_POS = (-0.64, DESK_Y, -0.52)


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

    b.material("floor", principled(
        bitmap(A.texture("floor_albedo", lambda: G.wood_planks(2048, 1536, planks=36, board_len=0.2, seed=2)[0])),
        bitmap(A.texture("floor_rough", lambda: G.wood_planks(2048, 1536, planks=36, board_len=0.2, seed=2)[1], gray=True), raw=True),
        specular=0.5, clearcoat=0.1, clearcoat_gloss=0.3))
    b.material("rug", {"type": "diffuse", "reflectance": bitmap(A.texture("rug", lambda: T.rug(512, 768, 8)))})
    b.material("wall", {"type": "diffuse", "reflectance": bitmap(A.texture(
        "plaster_cream", lambda: G.plaster(1024, 2048, (0.80, 0.76, 0.68), seed=7)))})
    b.material("wall_back", {"type": "diffuse", "reflectance": bitmap(A.texture(
        "plaster_blue", lambda: G.plaster(1024, 2048, (0.30, 0.37, 0.40), seed=8)))})
    b.material("ceiling", {"type": "diffuse", "reflectance": rgb(0.85, 0.84, 0.80)})
    b.material("ground", env_kit.ground("clear"))
    b.material("hedge", {"type": "diffuse", "reflectance": bitmap(A.texture(
        "hedge", lambda: np.clip(np.stack([0.10 + 0.12 * (n := G.fbm(256, 256, 5, 8, seed=11)), 0.22 + 0.2 * n,
                                           0.08 + 0.08 * n], -1), 0, 1)))})

    b.material("desk_wood", principled(bitmap(A.texture("desk_wood", lambda: T.wood_top(1024, 2048, 3)),
                                              uv_scale=(1 / 2.0, 1 / 1.0)), 0.38, specular=0.5, clearcoat=0.25, clearcoat_gloss=0.4))
    b.material("desk_dark", principled((0.16, 0.10, 0.06), 0.45, clearcoat=0.2))
    b.material("case_wood", principled((0.62, 0.52, 0.38), 0.5))
    b.material("leather", principled(bitmap(A.texture("leather", lambda: T.leather_mat(512, 1024)), uv_scale=(1 / 1.0, 1 / 0.5)),
                                     0.5, specular=0.4))
    b.material("white_paint", principled((0.86, 0.86, 0.83), 0.4))
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
    for i, c in enumerate([(0.45, 0.08, 0.06), (0.08, 0.2, 0.3), (0.55, 0.42, 0.12), (0.12, 0.25, 0.14), (0.3, 0.12, 0.25),
                           (0.75, 0.7, 0.6)]):
        b.material(f"book_{i}", principled(c, 0.55))

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
    b.material("art_a", {"type": "diffuse", "reflectance": bitmap(A.texture(
        "art_a", lambda: np.flipud(G.abstract_art(768, 576, [(0.85, 0.78, 0.62), (0.78, 0.35, 0.18), (0.14, 0.26, 0.36), (0.9, 0.62, 0.2)], seed=13))))})
    b.material("art_b", {"type": "diffuse", "reflectance": bitmap(A.texture(
        "art_b", lambda: np.flipud(G.abstract_art(576, 768, [(0.2, 0.3, 0.28), (0.85, 0.82, 0.7), (0.55, 0.2, 0.2), (0.3, 0.45, 0.6)], seed=17))))})


# ----------------------------------------------------------------------------
# Room, window, outside
# ----------------------------------------------------------------------------

def add_room(b: SceneBuilder) -> None:
    cx, cz = 0.5 * (ROOM_X0 + ROOM_X1), 0.5 * (ROOM_Z0 + ROOM_Z1)
    hx, hz = 0.5 * (ROOM_X1 - ROOM_X0), 0.5 * (ROOM_Z1 - ROOM_Z0)
    b.rect(G.compose(G.translate((cx, 0, cz)), G.rotate((1, 0, 0), -90), G.scale((hx, hz, 1))), "floor")
    b.rect(G.compose(G.translate((0.0, 0.004, -0.35)), G.rotate((1, 0, 0), -90), G.scale((1.3, 0.95, 1))), "rug")
    b.rect(G.compose(G.translate((cx, ROOM_H, cz)), G.rotate((1, 0, 0), 90), G.scale((hx, hz, 1))), "ceiling")
    b.rect(G.compose(G.translate((cx, 0.5 * ROOM_H, ROOM_Z0)), G.scale((hx, 0.5 * ROOM_H, 1))), "wall_back")
    b.rect(G.compose(G.translate((cx, 0.5 * ROOM_H, ROOM_Z1)), G.rotate((0, 1, 0), 180), G.scale((hx, 0.5 * ROOM_H, 1))), "wall")
    b.rect(G.compose(G.translate((ROOM_X1, 0.5 * ROOM_H, cz)), G.rotate((0, 1, 0), -90), G.scale((hz, 0.5 * ROOM_H, 1))), "wall")
    for z in (ROOM_Z1 - 0.01,):
        b.cube((cx, 0.05, z), (2 * hx, 0.1, 0.02), "white_paint")
    b.cube((ROOM_X1 - 0.01, 0.05, cz), (0.02, 0.1, 2 * hz), "white_paint")
    b.cube((ROOM_X0 + 0.01, 0.05, cz), (0.02, 0.1, 2 * hz), "white_paint")
    b.cube((cx, 0.05, ROOM_Z0 + 0.01), (2 * hx, 0.1, 0.02), "white_paint")

    # Left wall with the window opening; thick blocks give the opening real reveals.
    t = 0.25
    wx = ROOM_X0 - t / 2
    z0, z1 = WINDOW_Z
    y0, y1 = WINDOW_Y
    b.cube((wx, ROOM_H / 2, 0.5 * (ROOM_Z0 + z0)), (t, ROOM_H, z0 - ROOM_Z0), "wall")
    b.cube((wx, ROOM_H / 2, 0.5 * (z1 + ROOM_Z1)), (t, ROOM_H, ROOM_Z1 - z1), "wall")
    b.cube((wx, y0 / 2, 0.5 * (z0 + z1)), (t, y0, z1 - z0), "wall")
    b.cube((wx, 0.5 * (y1 + ROOM_H), 0.5 * (z0 + z1)), (t, ROOM_H - y1, z1 - z0), "wall")
    zc = 0.5 * (z0 + z1)
    f = 0.05
    for z in (z0 + f / 2, z1 - f / 2, zc):
        b.cube((wx + 0.06, 0.5 * (y0 + y1), z), (0.06, y1 - y0, f if z != zc else 0.035), "white_paint")
    for y in (y0 + f / 2, y1 - f / 2, y0 + 0.62):
        b.cube((wx + 0.06, y, zc), (0.06, f if y != y0 + 0.62 else 0.035, z1 - z0), "white_paint")
    b.cube((ROOM_X0 + 0.07, y0 - 0.02, zc), (0.30, 0.04, z1 - z0 + 0.1), "white_paint")
    # Outside: ground and a hedge. sunsky is black below the horizon, so the ground must reach it.
    b.rect(G.compose(G.translate((0, -0.02, 0)), G.rotate((1, 0, 0), -90), G.scale((400.0, 400.0, 1))), "ground")
    b.cube((-8.0, 0.9, 0.0), (1.6, 1.8, 60.0), "hedge")


def add_bookcase(b: SceneBuilder, local: Assets, ctx: BuildContext) -> None:
    w, depth, h = 3.0, 0.34, 2.28
    zc = ROOM_Z0 + depth / 2
    levels = [0.10, 0.54, 0.98, 1.42, 1.86]
    b.cube((-w / 2 + 0.015, h / 2, zc), (0.03, h, depth), "case_wood")
    b.cube((w / 2 - 0.015, h / 2, zc), (0.03, h, depth), "case_wood")
    b.cube((0, h - 0.015, zc), (w, 0.03, depth), "case_wood")
    b.cube((0, 0.04, zc), (w - 0.06, 0.08, depth), "case_wood")
    for y in levels:
        b.cube((0, y - 0.0125, zc), (w - 0.06, 0.025, depth), "case_wood")
    z_front = ROOM_Z0 + depth - 0.03
    rng = rng_for(ctx, 5)
    gap_levels = {1: 1.02, 3: 1.05}        # shelves that end early, leaving room for ornaments

    def build() -> Mesh:
        out = Mesh()
        for i, y in enumerate(levels):
            out += D.shelf_row(rng, -1.46, gap_levels.get(i, 1.46), y, z_front, 0.37)
        return out

    b.ply(local.mesh("books", build), np.eye(4), "spines", prefix="books")
    small = lambda: P.potted_plant(52, stems=6, height=0.25, leaf_len=0.09, leaf_w=0.045, leaves_per_stem=6,
                                   pot_radius=0.06, pot_height=0.1, spread=0.12)
    for i, x in ((1, 1.25), (3, 1.28)):
        parts = b.assets.meshes("plant_shelf", small)
        b.part_set(parts, G.translate((x, levels[i], z_front - 0.1)), {"pot": "terracotta", "soil": "soil", "stems": "stem", "leaves": "leaf"})
    jar = b.assets.meshes("jar", P.jar)
    b.part_set(jar, G.compose(G.translate((1.28, levels[1], z_front - 0.1)), G.scale(0.8)), {"glass": "mug_0", "lid": "case_wood"})
    b.focus_points["shelf_books"] = [-0.3, levels[2] + 0.2, z_front]


def add_walls_decor(b: SceneBuilder) -> None:
    face_left = G.rotate((0, 1, 0), -90)
    b.rect(G.compose(G.translate((ROOM_X1 - 0.012, 1.45, -1.4)), face_left, G.scale((0.62, 0.42, 1))), "pinboard")
    for dz, dy, ww, hh in ((0, 0.42, 1.28, 0.04), (0, -0.42, 1.28, 0.04), (0.62, 0, 0.04, 0.88), (-0.62, 0, 0.04, 0.88)):
        b.cube((ROOM_X1 - 0.02, 1.45 + dy, -1.4 + dz), (0.03, hh, ww), "desk_dark")
    b.focus_points["pinboard"] = [ROOM_X1 - 0.02, 1.5, -1.4]
    back = G.rotate((0, 1, 0), 180)
    for art, x, w, h in (("art_a", -1.2, 0.6, 0.8), ("art_b", 0.5, 0.75, 0.56)):
        c = (x, 1.6, ROOM_Z1 - 0.02)
        b.rect(G.compose(G.translate(c), back, G.scale((w / 2, h / 2, 1))), art)
        for dx, dy, ww, hh in ((0, h / 2, w + 0.05, 0.035), (0, -h / 2, w + 0.05, 0.035), (w / 2, 0, 0.035, h), (-w / 2, 0, 0.035, h)):
            b.cube((c[0] + dx, c[1] + dy, c[2] - 0.005), (ww, hh, 0.03), "black")
    # Sideboard against the right wall with a plant.
    b.cube((ROOM_X1 - 0.24, 0.4, -2.4), (0.46, 0.8, 1.5), "desk_dark")
    for k in range(3):
        b.cube((ROOM_X1 - 0.47, 0.18 + 0.27 * k, -2.4), (0.012, 0.22, 1.42), "case_wood")
    big = lambda: P.potted_plant(31, stems=8, height=0.7, leaf_len=0.2, leaf_w=0.1, leaves_per_stem=8, pot_radius=0.12,
                                 pot_height=0.2, spread=0.3)
    b.part_set(b.assets.meshes("plant_side", big), G.translate((ROOM_X1 - 0.24, 0.8, -2.9)),
               {"pot": "mug_0", "soil": "soil", "stems": "stem", "leaves": "leaf_dark"})
    floor_plant = lambda: P.potted_plant(37, stems=9, height=1.2, leaf_len=0.26, leaf_w=0.12, leaves_per_stem=9,
                                         pot_radius=0.2, pot_height=0.38, spread=0.5)
    b.part_set(b.assets.meshes("plant_floor", floor_plant), G.translate((-2.0, 0, -2.6)),
               {"pot": "mug_0", "soil": "soil", "stems": "stem", "leaves": "leaf_dark"})


# ----------------------------------------------------------------------------
# The desk and everything on it
# ----------------------------------------------------------------------------

def add_desk(b: SceneBuilder, local: Assets, ctx: BuildContext) -> None:
    A = b.assets
    rng = rng_for(ctx, 1)
    mx, my, mz = MONITOR_POS
    b.box_mesh("desk_top", (0, DESK_Y - 0.0175, -0.4), (1.8, 0.035, 0.8), "desk_wood")
    b.cube((-0.865, 0.3575, -0.4), (0.03, 0.715, 0.7), "desk_dark")                 # left side panel
    b.cube((0.65, 0.3575, -0.42), (0.46, 0.715, 0.68), "desk_dark")                 # drawer pedestal
    for k in range(3):
        b.cube((0.65, 0.12 + 0.22 * k, -0.077), (0.42, 0.2, 0.012), "desk_wood")
        b.cube((0.65, 0.17 + 0.22 * k, -0.066), (0.16, 0.012, 0.012), "brass")
    b.cube((0.0, 0.55, -0.76), (1.7, 0.3, 0.02), "desk_dark")                       # modesty panel
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
    variant = ctx.seed % 2
    plant = lambda: P.potted_plant(41 + variant, stems=6 + variant, height=0.2 + 0.05 * variant, leaf_len=0.075, leaf_w=0.038,
                                   leaves_per_stem=6, pot_radius=0.06, pot_height=0.09, spread=0.1)
    b.part_set(A.meshes(f"plant_desk_{variant}", plant), G.translate((-0.80, DESK_Y, -0.68)),
               {"pot": "terracotta", "soil": "soil", "stems": "stem", "leaves": "leaf"})
    b.focus_points["plant"] = [-0.80, DESK_Y + 0.2, -0.68]


def add_lamp(b: SceneBuilder) -> None:
    """Articulated desk lamp: parallel rods, joints and a shade aimed at the notebook; the warm bulb is the emitter."""
    base = np.array(LAMP_BASE)
    A_, B_, C_ = (base + np.array(v) for v in ((0.0, 0.07, 0.0), (-0.07, 0.60, 0.04), (-0.30, 0.46, 0.24)))

    def rods(p0, p1, off):
        d = np.asarray(p1) - np.asarray(p0)
        side = np.cross(d, (0, 1, 0))
        side = side / np.linalg.norm(side) * off
        return D.join(D.tube(p0 + side, p1 + side, 0.0042, 8), D.tube(p0 - side, p1 - side, 0.0042, 8))

    arms = D.join(rods(A_, B_, 0.011), rods(B_, C_, 0.011))
    for p, r in ((A_, 0.015), (B_, 0.015), (C_, 0.014)):
        arms += G.ellipsoid((r, r, r), 10, 16).transformed(G.translate(p))
    b.ply(b.assets.mesh("lamp_base", lambda: G.lathe([(0, 0), (0.075, 0), (0.078, 0.004), (0.07, 0.018), (0.03, 0.026), (0, 0.026)], 48)),
          G.translate(base), "lamp_metal")
    b.ply(b.assets.mesh("lamp_arms", lambda: arms), np.eye(4), "lamp_metal", prefix="lamp_arms")
    aim = np.array(NOTEBOOK_POS) + np.array([0.0, 0.02, 0.0]) - C_
    aim = aim / np.linalg.norm(aim)
    head = G.compose(G.translate(C_), D.rot_between((0, -1, 0), aim))
    outer = [(0.078, -0.13), (0.060, -0.095), (0.040, -0.055), (0.024, -0.02), (0.016, 0.0), (0.016, 0.03), (0.010, 0.034)]
    inner = [(0.0, -0.004), (0.020, -0.022), (0.037, -0.058), (0.056, -0.098), (0.075, -0.13)]
    b.ply(b.assets.mesh("lamp_shade_out", lambda: G.lathe(outer, 56)), head, "lamp_metal")
    # The interior reflects about as much light as the bulb gives directly. A diffuse interior would make every
    # lit pixel noisy (the bulb is reached only through two bounces), so it is an explicit emitter instead:
    # albedo 0.88 x irradiance at ~0.07 m from the bulb / pi, which is about 8.5 for the 120 bulb.
    b.ply(b.assets.mesh("lamp_shade_in", lambda: G.lathe(inner, 56)), head, "emitter_backing",
          emission=tuple(0.071 * c for c in LAMP), prefix="shade_glow", power=0.1)
    bulb = C_ + aim * 0.055
    b.sphere(bulb, 0.02, "emitter_backing", emission=LAMP, prefix="bulb")
    b.focus_points["lamp"] = bulb.tolist()


def add_chair(b: SceneBuilder, ctx: BuildContext) -> None:
    rng = rng_for(ctx, 4)
    parts = b.assets.meshes("office_chair", D.office_chair)
    b.part_set(parts, G.compose(G.translate((float(rng.uniform(-0.1, 0.1)), 0, 0.52)),
                                G.rotate((0, 1, 0), 180 + float(rng.uniform(-18, 18)))),
               {"fabric": "chair_fabric", "plastic": "black", "chrome": "chrome"})
    b.focus_points["chair"] = [0.0, 0.85, 0.30]


def add_room_lights(b: SceneBuilder) -> None:
    for z in (-1.2, -2.9):
        b.rect(G.compose(G.translate((0.2, ROOM_H - 0.01, z)), G.rotate((1, 0, 0), 90), G.scale((0.6, 0.3, 1))),
               "emitter_backing", emission=CEILING, prefix="ceiling_panel", power=1.2 * 0.6 * luminance(CEILING))
    b.rect(G.compose(G.translate((0.0, 2.23, ROOM_Z0 + 0.36)), G.rotate((1, 0, 0), 60), G.scale((1.4, 0.015, 1))),
           "emitter_backing", emission=STRIP, prefix="case_strip", power=2.8 * 0.03 * luminance(STRIP))


def build(ctx: BuildContext) -> SceneBundle:
    local = Assets(ctx.assets_dir, ctx.rebuild)
    b = SceneBuilder(Assets(ctx.shared_dir, ctx.rebuild))
    add_materials(b, local, ctx)
    add_room(b)
    add_bookcase(b, local, ctx)
    add_walls_decor(b)
    add_desk(b, local, ctx)
    add_lamp(b)
    add_room_lights(b)
    add_chair(b, ctx)
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
    },
    default_view="close",
    camera_box=((-1.6, 0.85, -0.4), (1.7, 1.9, 1.2)),
    target_box=((-1.2, 0.55, -4.0), (1.2, 1.9, -0.1)),   # the desk and the bookcase, not the side walls
    exclude_boxes=(
        ((-1.0, 0.0, -0.9), (1.0, 0.80, 0.1)),        # desk body
        ((-0.52, 0.75, -0.72), (0.12, 1.2, -0.46)),   # monitor
        ((0.68, 0.75, -0.78), (0.98, 1.45, -0.50)),   # lamp base and lower arm
        ((0.38, 1.05, -0.58), (0.82, 1.40, -0.20)),   # lamp upper arm and head
        ((-0.40, 0.0, 0.10), (0.40, 1.15, 0.80)),     # chair
        ((-0.9, 0.75, -0.8), (-0.55, 0.95, -0.45)),   # pen cup and plant
        ((-2.4, 0.0, -3.0), (-1.5, 1.9, -2.0)),       # floor plant
    ),
    tags=("indoor", "day", "artificial_light", "closeup", "clutter", "metal"),
    default_seed=3,
    asset_version=3,
    spp_hint=2048,
)
