"""An industrial workshop / garage for Mitsuba 3: a lived-in shop of scanned tools and clutter in harsh fluorescent light.

Units are meters, y up, the entrance (a half-raised roller door) is in the front wall at +z, the workbench and
pegboard against the back wall at -z. The room is 8 x 9 x 3.6 m (x in [-4, 4], z in [-6, 3]).

  back wall   workbench with a scanned vise and drill press, a parts cabinet, a can of screwdrivers and a seed-chosen
              spread of scanned tools, cans and sprays; above it a pegboard of ~100 procedural hand tools; two
              hanging industrial lamps; sledgehammer, bolt cutters, cement and compost bags, a generator in the corner
  middle      a car under a dust cover, a caged trouble-light over it, oil stains and tyre marks on the concrete
  right wall  steel shelving filled with scanned crates, boxes, cans and jerrycans; barrels, a tool cart, a tool
              chest, a broom, a fire extinguisher
  left wall   three clerestory windows (sun rakes across the floor), a ladder, stacked tyres and a rim, an old drill
              press on a crate, a wall-mounted hose reel, a storage cart and a welding cart, a chain from the roof beam
  ceiling     steel I-beams under a profiled metal roof, ten fluorescent battens, drooping cables

Scanned models and PBR surfaces are CC0 Poly Haven assets fetched through `web_assets` (pinned in
scenes/assets/web_manifest.json); the pegboard tools, shelving frames, beams, battens, roller door, chain and cables are
procedural (`_workshop_props.py`, `_workshop_tex.py`). Light: the battens (rectangular emitters, power-weighted), warm
bulbs in the two hanging lamps and the trouble-light, and sun and sky through the open windows and door. Windows are
open frames: Mitsuba's path tracer cannot do next-event estimation through glass.

Seed-dependent (ctx.seed): the bench clutter, shelf contents, pegboard layout, barrel and jerrycan choice, stool and
broom placement, and the oil stains. Everything else is fixed.
"""

from __future__ import annotations

import numpy as np

import env_kit
import procedural as G
import web_assets as WA
from procedural import Mesh
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, bitmap, luminance, principled, rgb
from scenes import _workshop_props as W
from scenes import _workshop_tex as T

ROOM_X0, ROOM_X1, ROOM_Z0, ROOM_Z1, ROOM_H = -4.0, 4.0, -6.0, 3.0, 3.6
BENCH_Y = 0.92
BENCH_X0, BENCH_X1 = -3.7, -0.7
BENCH_Z0, BENCH_Z1 = ROOM_Z0 + 0.02, ROOM_Z0 + 0.77
PEG_Y0, PEG_Y1 = 1.25, 2.55
WIN_Y = (2.25, 3.15)
WIN_Z = [(-5.1, -4.3), (-3.1, -2.3), (-1.1, -0.3)]
DOOR_X = (0.0, 3.0)
DOOR_TOP, DOOR_SLATS_BOTTOM = 2.7, 1.75
SUN_ELEVATION, SUN_AZIMUTH = 28.0, -80.0
SKY_SCALE = 7.0
PANEL = (15.0, 15.5, 17.0)          # cool-white radiance of each 4-tube panel's diffuse emitter
LAMP = (40.0, 26.0, 12.0)           # warm bulbs in the hanging lamps; big and dim, not small and hot (fireflies)
WORKLIGHT = (40.0, 28.0, 14.5)           # big and dim (r 0.045), not small and hot: fireflies
DOOR_BULB = (22.0, 15.0, 7.4)
CAR_AT = (1.4, -2.55)
FLAT = G.rotate((1, 0, 0), -90)    # lays an upright Poly Haven tool (thin along z) flat on its back


def rng_for(ctx: BuildContext, k: int) -> np.random.Generator:
    """One stream per component, so a warm cache never changes what later components draw."""
    return np.random.default_rng([ctx.seed, k])


# ----------------------------------------------------------------------------
# Scanned models: placement by rotated bounds
# ----------------------------------------------------------------------------

def footprint(m, pre=None) -> np.ndarray:
    """(x, y, z) size of the model's bounds after the pre-rotation."""
    lo, hi = np.asarray(m.bounds[0]), np.asarray(m.bounds[1])
    c = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    c = (c @ (np.eye(4) if pre is None else pre).T)[:, :3]
    return c.max(0) - c.min(0)


def wplace(m, at, yaw=0.0, pre=None, scale=1.0) -> np.ndarray:
    """Put the model's footprint centre and lowest point at `at` after `pre` (any rotation) and a turn of `yaw` about y."""
    lo, hi = np.asarray(m.bounds[0]), np.asarray(m.bounds[1])
    R = G.compose(np.eye(4) if pre is None else pre, G.scale(scale))
    c = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    c = (c @ R.T)[:, :3]
    anchor = np.array([(c[:, 0].min() + c[:, 0].max()) / 2, c[:, 1].min(), (c[:, 2].min() + c[:, 2].max()) / 2])
    return G.compose(G.translate(at), G.rotate((0, 1, 0), yaw), G.translate(-anchor), R)


# steel_frame_shelves_01's own glTF stores positions at 10x its catalog size (21.4 m tall vs 2.14 m); scale it back.
ASSET_SCALE = {"steel_frame_shelves_01": 0.1}


def put(b: SceneBuilder, aid: str, at, yaw=0.0, pre=None, scale=1.0, height=None, prefix=None, overrides=None):
    m = WA.model(aid)
    scale *= ASSET_SCALE.get(aid, 1.0)
    if height is not None:
        scale = height / footprint(m, pre)[1]
    M = wplace(m, at, yaw, pre, scale)
    WA.add(b, m, M, prefix=prefix, overrides=overrides)
    return m, M


def top_of(m, M) -> np.ndarray:
    lo, hi = np.asarray(m.bounds[0]), np.asarray(m.bounds[1])
    c = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ M.T
    return np.array([c[:, 0].mean(), c[:, 1].max(), c[:, 2].mean()])


# ----------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------

def add_materials(b: SceneBuilder, local: Assets, ctx: BuildContext) -> None:
    A = b.assets
    # Floor: scanned concrete blended with a dark oil film where the seed-dependent stain mask says so.
    mask = local.texture("oil_mask", lambda: T.oil_mask(1152, 1024, 8.0, 9.0, ctx.seed), gray=True)
    oil = principled((0.025, 0.022, 0.02), 0.12, specular=0.6)
    b.material("floor", {"type": "blendbsdf", "weight": bitmap(mask, raw=True, uv_scale=(1 / 8.0, 1 / 9.0)),
                         "bsdf_0": WA.surface("garage_floor", res="2k"), "bsdf_1": oil})
    b.material("wall_back", WA.surface("whitewashed_brick", res="2k"))
    b.material("wall_left", WA.surface("concrete_wall_007", res="2k"))
    b.material("wall_right", WA.surface("concrete_layers_02", res="2k"))
    b.material("ceiling", WA.surface("corrugated_iron_03", res="2k"))
    b.material("plaster", WA.surface("concrete_wall_007", res="2k"))
    b.material("bench_top", WA.surface("brown_planks_04", res="2k"))
    b.material("bench_frame", WA.surface("blue_metal_plate", res="2k"))
    b.material("shelf_post", WA.surface("blue_metal_plate", res="2k", tint=(0.8, 0.85, 1.0)))
    b.material("shelf_plate", WA.surface("rusty_painted_metal", res="2k"))
    b.material("door_slats", {"type": "twosided", "bsdf": WA.surface("painted_metal_shutter", res="2k", uv_scale=(2.94 / 2.0, 0.95 / 2.0))})
    b.material("brick_far", WA.surface("brick_wall_02", res="2k", uv_scale=(24.0 / 2.0, 8.0 / 2.0)))
    b.material("yard", WA.surface("asphalt_floor", res="2k", uv_scale=(24.0 / 2.35, 11.0 / 2.35), tint=(0.6, 0.6, 0.6)))
    b.material("ground", env_kit.ground("clear"))
    b.material("emitter_backing", {"type": "diffuse", "reflectance": rgb(0.0, 0.0, 0.0)})
    # Near-black and diffuse on purpose: a glossy or bright housing a few millimetres from a tube is lit mostly by that
    # one tube, which next-event estimation picks 1 time in 20, and every pixel that sees it turns to salt noise
    # (measured: mean |pixel - 5x5 blur| / mean went from 0.34 to 0.19 at 64 spp).
    b.material("fixture", {"type": "diffuse", "reflectance": rgb(0.02, 0.02, 0.02)})
    # Roughness 0.5, not 0.3: a peaked metal lobe can only find the one tube that mirrors into it, and the
    # pegboard tools came out grainy even at 1536 spp.
    b.material("tool_steel", principled((0.74, 0.75, 0.78), 0.5, metallic=1.0))
    b.material("steel_plain", principled((0.62, 0.62, 0.64), 0.4, metallic=1.0))
    b.material("chrome", {"type": "roughconductor", "material": "Cr", "alpha": 0.12})
    b.material("paint_grey", principled((0.30, 0.31, 0.33), 0.5, metallic=0.3))
    b.material("rust_paint", WA.surface("rusty_painted_metal", res="2k"))
    b.material("rust_frame", WA.surface("rusty_metal_02", res="2k"))
    b.material("purlin", WA.surface("rusty_painted_metal", res="2k", tint=(0.55, 0.6, 0.65)))
    b.material("cabinet_frame", WA.surface("blue_metal_plate", res="2k"))
    b.material("paint_black", principled((0.04, 0.04, 0.045), 0.5))
    b.material("paint_yellow", principled((0.75, 0.55, 0.05), 0.45))
    b.material("black_matt", principled((0.02, 0.02, 0.02), 0.7))
    b.material("cable_yellow", principled((0.8, 0.65, 0.05), 0.5))
    b.material("wood_handle", principled((0.46, 0.30, 0.16), 0.5, clearcoat=0.2, clearcoat_gloss=0.4))
    b.material("peg", principled(bitmap(A.texture("peg_a", lambda: T.pegboard_painted(128)), uv_scale=(1 / 0.025, 1 / 0.025)), 0.6))
    for nm, c in (("grip_red", (0.55, 0.04, 0.03)), ("grip_blue", (0.04, 0.10, 0.45)), ("grip_yellow", (0.8, 0.6, 0.04)),
                  ("grip_green", (0.05, 0.32, 0.12)), ("grip_orange", (0.8, 0.28, 0.03)), ("grip_black", (0.03, 0.03, 0.035))):
        b.material(nm, principled(c, 0.55, specular=0.4))
    b.material("bin_yellow", principled((0.8, 0.62, 0.05), 0.4, specular=0.5))
    b.material("can_tin", principled((0.60, 0.60, 0.62), 0.3, metallic=0.9))
    b.material("rag_red", {"type": "twosided", "bsdf": WA.surface("acg:Fabric030", uv_scale=0.3, tint=(1.3, 0.45, 0.35))})
    b.material("rag_grey", {"type": "twosided", "bsdf": WA.surface("acg:Fabric030", uv_scale=0.3)})
    b.material("mat_rubber", WA.surface("acg:Rubber004", uv_scale=(2.2 / 0.75, 0.8 / 0.75)))
    for nm, c in (("bin_red", (0.55, 0.05, 0.04)), ("bin_blue", (0.05, 0.12, 0.5)), ("bin_grey", (0.35, 0.36, 0.38))):
        b.material(nm, principled(c, 0.4, specular=0.5))
    b.material("label", {"type": "diffuse", "reflectance": rgb(0.84, 0.82, 0.76)})


# ----------------------------------------------------------------------------
# Room: quads with uv in meters, so the scanned textures keep real scale and line up across pieces
# ----------------------------------------------------------------------------

def quad(p0, u, v, w: float, h: float, uv0) -> Mesh:
    """Rectangle from p0 spanning w along unit u and h along unit v; uv = uv0 + (a, b) in meters; normal u x v."""
    p0, u, v = (np.asarray(x, float) for x in (p0, u, v))
    p = np.array([p0, p0 + w * u, p0 + w * u + h * v, p0 + h * v])
    n = np.tile(np.cross(u, v), (4, 1))
    uv = np.asarray(uv0, float) + np.array([[0, 0], [w, 0], [w, h], [0, h]])
    return Mesh(p, n, uv, np.array([[0, 1, 2], [0, 2, 3]])).oriented()


def wall_piece(b: SceneBuilder, face: str, h0: float, h1: float, y0: float, y1: float) -> None:
    """Wall from h0 to h1 along the wall (x for back/front, z for left/right) and y0 to y1 up, facing into the room."""
    w, h = h1 - h0, y1 - y0
    if face == "back":
        q, mat = quad((h0, y0, ROOM_Z0), (1, 0, 0), (0, 1, 0), w, h, (h0 - ROOM_X0, y0)), "wall_back"
    elif face == "front":
        q, mat = quad((h1, y0, ROOM_Z1), (-1, 0, 0), (0, 1, 0), w, h, (ROOM_X1 - h1, y0)), "wall_back"
    elif face == "left":
        q, mat = quad((ROOM_X0, y0, h1), (0, 0, -1), (0, 1, 0), w, h, (ROOM_Z1 - h1, y0)), "wall_left"
    else:
        q, mat = quad((ROOM_X1, y0, h0), (0, 0, 1), (0, 1, 0), w, h, (h0 - ROOM_Z0, y0)), "wall_right"
    key = f"wall_{face}_{h0:+.2f}_{h1:+.2f}_{y0:.2f}_{y1:.2f}".replace(".", "p").replace("-", "m").replace("+", "")
    b.ply(b.assets.mesh(key, lambda: q), np.eye(4), mat, prefix="wall")


def add_room(b: SceneBuilder) -> None:
    A = b.assets
    floor = A.mesh("floor_quad", lambda: quad((ROOM_X0, 0, ROOM_Z1), (1, 0, 0), (0, 0, -1), 8.0, 9.0, (0, 0)))
    b.ply(floor, np.eye(4), "floor", prefix="floor")
    ceil = A.mesh("ceiling_quad", lambda: quad((ROOM_X0, ROOM_H, ROOM_Z0), (1, 0, 0), (0, 0, 1), 8.0, 9.0, (0, 0)))
    b.ply(ceil, np.eye(4), "ceiling", prefix="ceiling")
    wall_piece(b, "back", ROOM_X0, ROOM_X1, 0.0, ROOM_H)
    wall_piece(b, "right", ROOM_Z0, ROOM_Z1, 0.0, ROOM_H)
    wall_piece(b, "front", ROOM_X0, DOOR_X[0], 0.0, ROOM_H)
    wall_piece(b, "front", DOOR_X[1], ROOM_X1, 0.0, ROOM_H)
    wall_piece(b, "front", DOOR_X[0], DOOR_X[1], DOOR_TOP, ROOM_H)
    wall_piece(b, "left", ROOM_Z0, ROOM_Z1, 0.0, WIN_Y[0])
    wall_piece(b, "left", ROOM_Z0, ROOM_Z1, WIN_Y[1], ROOM_H)
    edges = [ROOM_Z0] + [z for w in WIN_Z for z in w] + [ROOM_Z1]
    for a, c in zip(edges[0::2], edges[1::2]):
        wall_piece(b, "left", a, c, WIN_Y[0], WIN_Y[1])
    t = 0.25
    for z0, z1 in WIN_Z:                               # reveals and a steel frame with mullions (open, no glass)
        zc = 0.5 * (z0 + z1)
        b.box_mesh(f"rev_h_{z1 - z0:.2f}".replace(".", "p"), (ROOM_X0 - t / 2, WIN_Y[0] - 0.02, zc), (t, 0.04, z1 - z0), "plaster")
        b.box_mesh(f"rev_h_{z1 - z0:.2f}".replace(".", "p"), (ROOM_X0 - t / 2, WIN_Y[1] + 0.02, zc), (t, 0.04, z1 - z0), "plaster")
        for z in (z0 - 0.02, z1 + 0.02):
            b.box_mesh("rev_v", (ROOM_X0 - t / 2, 0.5 * sum(WIN_Y), z), (t, WIN_Y[1] - WIN_Y[0], 0.04), "plaster")
        f = 0.035
        for z in (z0 + f / 2, z1 - f / 2, zc):
            b.box_mesh("win_mullion", (ROOM_X0 - 0.06, 0.5 * sum(WIN_Y), z), (0.05, WIN_Y[1] - WIN_Y[0], f), "rust_frame")
        for y in (WIN_Y[0] + f / 2, WIN_Y[1] - f / 2, 0.5 * sum(WIN_Y)):
            b.box_mesh(f"win_transom_{z1 - z0:.2f}".replace(".", "p"), (ROOM_X0 - 0.06, y, zc), (0.05, f, z1 - z0), "rust_frame")
    # Door jambs and the roller door: corrugated slats raised to DOOR_SLATS_BOTTOM, a roll housing above.
    d0, d1 = DOOR_X
    b.box_mesh("jamb", (d0 - 0.02, 0.5 * DOOR_TOP, ROOM_Z1 - 0.1), (0.14, DOOR_TOP, 0.2), "plaster")
    b.box_mesh("jamb", (d1 + 0.02, 0.5 * DOOR_TOP, ROOM_Z1 - 0.1), (0.14, DOOR_TOP, 0.2), "plaster")
    b.box_mesh("lintel", (0.5 * (d0 + d1), DOOR_TOP + 0.01, ROOM_Z1 - 0.1), (d1 - d0 + 0.2, 0.04, 0.2), "plaster")
    for x in (d0 + 0.02, d1 - 0.02):
        b.box_mesh("door_guide", (x, 0.5 * (DOOR_SLATS_BOTTOM + DOOR_TOP), ROOM_Z1 - 0.14), (0.05, DOOR_TOP - DOOR_SLATS_BOTTOM, 0.08), "rust_frame")
    slats = A.mesh("door_slats", lambda: corrugated(d1 - d0 - 0.06, DOOR_TOP - DOOR_SLATS_BOTTOM))
    b.ply(slats, G.translate((0.5 * (d0 + d1), DOOR_SLATS_BOTTOM, ROOM_Z1 - 0.15)), "door_slats", prefix="slats")
    b.box_mesh("door_hood", (0.5 * (d0 + d1), DOOR_TOP - 0.12, ROOM_Z1 - 0.2), (d1 - d0 - 0.06, 0.2, 0.2), "rust_frame")
    b.box_mesh("door_bottom", (0.5 * (d0 + d1), DOOR_SLATS_BOTTOM - 0.015, ROOM_Z1 - 0.15), (d1 - d0 - 0.06, 0.03, 0.06), "rust_frame")
    # Outside: yard, the facing building with an aircon unit, and ground to the horizon (sunsky needs it).
    b.rect(G.compose(G.translate((0, -0.02, 0)), G.rotate((1, 0, 0), -90), G.scale((400.0, 400.0, 1))), "ground")
    b.rect(G.compose(G.translate((0, 0.005, 0.5 * (ROOM_Z1 + 14.0))), G.rotate((1, 0, 0), -90), G.scale((12.0, 0.5 * (14.0 - ROOM_Z1), 1))), "yard")
    b.rect(G.compose(G.translate((0, 4.0, 14.0)), G.rotate((0, 1, 0), 180), G.scale((12.0, 4.0, 1))), "brick_far")
    put(b, "exterior_aircon_unit", (2.2, 0.0, 13.6), yaw=180.0)        # outside on the facing building only
    put(b, "Barrel_02", (-0.6, 0.0, 12.8))
    # Yard dressing, so the view out of the door is not a flat bright field.
    put(b, "modular_chainlink_fence", (0.5, 0.0, 10.5))
    put(b, "utility_box_02", (-2.6, 0.0, 9.2), yaw=180.0)
    put(b, "street_lamp_02", (4.2, 0.0, 8.0), yaw=-90.0)
    put(b, "rusted_wheel_rim_02", (0.9, 0.0, 6.2), pre=G.rotate((1, 0, 0), 90), yaw=40.0)
    put(b, "wooden_barrels_01", (-4.5, 0.0, 12.0), yaw=10.0)
    put(b, "metal_trash_can", (-1.9, 0.0, 4.0))


# ----------------------------------------------------------------------------
# Workbench: procedural frame with scanned planks, scanned vise and drill press, a seed-chosen spread of scanned clutter
# ----------------------------------------------------------------------------

BENCH_TALL = ["Drill_01", "lubricant_spray", "spray_paint_bottles_02", "oil_tin", "cleaner_tin_01", "can_rusted", "small_oil_can_01",
              "brass_blowtorch", "metal_toolbox", "hand_plane_no4", "spray_paint_bottles", "measuring_tape_01",
              "vintage_hand_drill", "propane_torch_02", "ammo_box", "retro_multimeter"]
# (asset, pre-rotation) for tools lying on the bench; Poly Haven models them upright (thin along z) or already flat.
BENCH_FLAT = [("adjustable_wrench", FLAT), ("cross_pein_hammer", FLAT), ("flathead_screwdriver", FLAT), ("pliers", FLAT),
              ("ratchet_wrench", FLAT), ("pipe_wrench", FLAT), ("combination_wrench", None), ("tongue_groove_pliers", None),
              ("screwdrivers_02", None), ("rusted_hacksaw", FLAT)]


def add_bench(b: SceneBuilder, local: Assets, ctx: BuildContext) -> None:
    A = b.assets
    bz0, bz1 = BENCH_Z0, BENCH_Z1
    bw = BENCH_X1 - BENCH_X0
    bx = 0.5 * (BENCH_X0 + BENCH_X1)
    bzc = 0.5 * (bz0 + bz1)
    b.box_mesh("bench_top_3x75", (bx, BENCH_Y - 0.0375, bzc), (bw, 0.075, bz1 - bz0), "bench_top")
    for sx in (BENCH_X0 + 0.05, BENCH_X1 - 0.05, bx):
        for sz in (bz0 + 0.05, bz1 - 0.05):
            b.box_mesh("bench_leg", (sx, 0.4325, sz), (0.06, 0.845, 0.06), "bench_frame")
    b.box_mesh("bench_rail", (bx, 0.80, bz1 - 0.05), (bw, 0.08, 0.04), "bench_frame")
    b.box_mesh("bench_rail", (bx, 0.80, bz0 + 0.05), (bw, 0.08, 0.04), "bench_frame")
    b.box_mesh("bench_lower_29x70", (bx, 0.27, bzc), (bw - 0.1, 0.03, bz1 - bz0 - 0.08), "bench_top")

    # Fixed pieces: vise at the left front corner, drill press at the right end, parts cabinet, a can of screwdrivers.
    m, M = put(b, "bench_vice_01", (BENCH_X0 + 0.32, BENCH_Y, bz1 - 0.16), yaw=180.0)
    b.focus_points["vise"] = top_of(m, M).tolist()
    m, M = put(b, "drill_press_01", (BENCH_X1 - 0.28, BENCH_Y, bz0 + 0.33), yaw=-20.0)
    b.focus_points["bench_drill"] = (top_of(m, M) - np.array([0, 0.3, 0])).tolist()
    pc = A.meshes("parts_cabinet", W.parts_cabinet)
    b.part_set(pc, G.translate((BENCH_X0 + 0.75, BENCH_Y, bz0 + 0.12)), {"frame": "cabinet_frame", "fronts": "bin_yellow", "pulls": "steel_plain"})
    b.focus_points["parts_cabinet"] = [BENCH_X0 + 0.75, BENCH_Y + 0.18, bz0 + 0.2]
    # A reused cleaner tin full of scanned screwdrivers, standing at random tilts.
    cx, cz = BENCH_X0 + 1.55, bz0 + 0.3
    m, M = put(b, "cleaner_tin_01", (cx, BENCH_Y, cz))
    tin_top = top_of(m, M)[1]
    r3 = rng_for(ctx, 12)
    for k in range(5):
        aid = ("screwdriver", "flathead_screwdriver")[k % 2]
        a, tilt = float(r3.uniform(0, 360)), float(r3.uniform(4, 16))
        off = float(r3.uniform(0.0, 0.018))
        put(b, aid, (cx + off * np.cos(np.radians(a)), tin_top - 0.1, cz + off * np.sin(np.radians(a))),
            pre=G.compose(G.rotate((0, 1, 0), a), G.rotate((0, 0, 1), tilt)), scale=float(r3.uniform(0.85, 1.1)))

    # Seed-chosen clutter: tall things along the back, tools lying in two rows at the front.
    r = rng_for(ctx, 13)
    back_x = [BENCH_X0 + 1.2, BENCH_X0 + 1.85, BENCH_X0 + 2.15]
    tall = [BENCH_TALL[int(i)] for i in r.permutation(len(BENCH_TALL))]
    for x, aid in zip(back_x, tall):
        put(b, aid, (x + float(r.uniform(-0.04, 0.04)), BENCH_Y, bz0 + 0.2 + float(r.uniform(-0.04, 0.04))), yaw=float(r.uniform(-50, 50)))
    flat = [BENCH_FLAT[int(i)] for i in r.permutation(len(BENCH_FLAT))]
    spots = [(BENCH_X0 + 0.75, bz1 - 0.2), (BENCH_X0 + 1.1, bz1 - 0.14), (BENCH_X0 + 1.45, bz1 - 0.22), (BENCH_X0 + 1.8, bz1 - 0.13),
             (BENCH_X0 + 2.1, bz1 - 0.24), (BENCH_X0 + 1.25, bz1 - 0.33), (BENCH_X0 + 1.95, bz1 - 0.38)]
    for k, ((x, z), (aid, pre)) in enumerate(zip(spots, flat)):
        m, M = put(b, aid, (x, BENCH_Y, z), yaw=float(r.uniform(-75, 75)), pre=pre)
        if k == 0:
            b.focus_points["wrench"] = top_of(m, M).tolist()
    rg = local.mesh("rag", lambda: W.rag(0.3, 0.22, ctx.seed))
    b.ply(rg, G.compose(G.translate((BENCH_X0 + 2.45, BENCH_Y, bz1 - 0.3)), G.rotate((0, 1, 0), float(r.uniform(0, 360)))),
          "rag_red" if ctx.seed % 2 else "rag_grey", prefix="rag")
    b.rect(G.compose(G.translate((bx, 0.003, bz1 + 0.5)), G.rotate((1, 0, 0), -90), G.scale((1.1, 0.4, 1))), "mat_rubber")

    # Under the bench: boxes, crates, cans and a jerrycan, packed along the lower shelf.
    r2 = rng_for(ctx, 11)
    under = ["cardboard_box_01", "plastic_crate_02", "metal_jerrycan", "can_rusted", "plastic_crate_03", "multi_cleaner_5_litre", "oil_tin"]
    x = BENCH_X0 + 0.12
    for aid in [under[int(i)] for i in r2.permutation(len(under))]:
        m = WA.model(aid)
        fp = footprint(m)
        yaw = 90.0 if fp[2] > fp[0] and fp[2] > 0.66 else 0.0
        wx = fp[2] if yaw else fp[0]
        if x + wx > BENCH_X1 - 0.1 or fp[1] > 0.52:
            continue
        put(b, aid, (x + wx / 2, 0.285, bzc + float(r2.uniform(-0.05, 0.05))), yaw=yaw + float(r2.uniform(-8, 8)))
        x += wx + 0.05

    # Two hanging industrial lamps over the bench with warm bulbs.
    for lx in (BENCH_X0 + 0.8, BENCH_X0 + 2.2):
        m = WA.model("hanging_industrial_lamp")
        drop = ROOM_H - footprint(m)[1]
        M = wplace(m, (lx, drop, bz0 + 0.42))
        WA.add(b, m, M, prefix="web_lamp_hung")
        b.sphere((lx, drop + 0.09, bz0 + 0.42), 0.05, "emitter_backing", emission=LAMP, prefix="lamp_bulb")
    b.focus_points["bench_lamp"] = [BENCH_X0 + 0.8, ROOM_H - 1.2, bz0 + 0.42]


# ----------------------------------------------------------------------------
# Middle: the covered car and its trouble-light
# ----------------------------------------------------------------------------

def add_car(b: SceneBuilder, ctx: BuildContext) -> None:
    A = b.assets
    cx, cz = CAR_AT
    # Converter v4 splits the car into two parts; the 4k-triangle one renders as a flat orange slab over the body, and
    # before v4 it was silently overwritten by the cover, which is the look that was approved. Keep only the cover.
    m, M = put(b, "covered_car", (cx, 0.0, cz), yaw=180.0, overrides={"covered_car": {"type": "null"}})
    t = top_of(m, M)
    b.focus_points["car"] = [cx, t[1] - 0.1, cz + 0.4]
    b.focus_points["car_rear"] = [cx, 0.8, cz - 2.1]
    put(b, "tire_pump", (cx - 1.25, 0.0, cz + 1.9), yaw=30.0)
    lx, lz = cx + 0.05, cz + 1.3
    b.focus_points["work_light"] = hanging_bulb(b, lx, lz, 2.5, WORKLIGHT)
    m, M = put(b, "ceiling_fan", (cx, ROOM_H - 0.53, cz - 0.6))
    b.focus_points["ceiling_fan"] = top_of(m, M).tolist()


# ----------------------------------------------------------------------------
# Left wall: ladder, tyres, old drill press, hose reel, carts, chain
# ----------------------------------------------------------------------------

def add_left_side(b: SceneBuilder, ctx: BuildContext) -> None:
    A = b.assets
    r = rng_for(ctx, 4)
    lie = G.rotate((1, 0, 0), 90)
    for (z, n) in ((-4.25, 4), (-4.95, 3)):
        y = 0.0
        for k in range(n):
            m, M = put(b, "old_tyre", (ROOM_X0 + 0.42 + float(r.uniform(-0.03, 0.03)), y, z), yaw=float(r.uniform(0, 360)), pre=lie)
            y = top_of(m, M)[1] - 0.005
    put(b, "rusted_wheel_rim_01", (ROOM_X0 + 0.42, y, -4.95), pre=lie, yaw=float(r.uniform(0, 360)))
    b.focus_points["tyres"] = [ROOM_X0 + 0.45, 0.5, -4.25]
    lean = G.compose(G.rotate((0, 0, 1), 13.0), G.rotate((0, 1, 0), 90.0))
    put(b, "ladder_sectioned_01", (ROOM_X0 + 0.3, 0.0, -3.3), pre=lean)
    b.focus_points["ladder"] = [ROOM_X0 + 0.25, 1.2, -3.3]
    m, M = put(b, "wooden_crate_01", (ROOM_X0 + 0.32, 0.0, -2.0), yaw=90.0)
    m, M = put(b, "old_drill_press", (ROOM_X0 + 0.35, top_of(m, M)[1], -2.0), yaw=90.0)
    b.focus_points["drill_press"] = (top_of(m, M) - np.array([0, 0.35, 0])).tolist()
    put(b, "industrial_wall_lamp", (ROOM_X0 + 0.08, 2.0, -2.0), yaw=90.0)
    put(b, "garden_hose_wall_mounted_01", (ROOM_X0 + 0.14, 0.9, -0.7), yaw=90.0)
    b.ply(A.mesh("chain_long", lambda: W.chain(1.5)), G.translate((ROOM_X0 + 1.2, ROOM_H - 0.15, -1.0)), "black_matt", prefix="chain")
    b.ply(A.mesh("chain_hook", lambda: W.sweep([(0, 0.0, 0), (0, -0.08, 0), (0.05, -0.14, 0), (0.04, -0.19, 0)], 0.007, 8)),
          G.translate((ROOM_X0 + 1.2, ROOM_H - 1.65, -1.0)), "steel_plain", prefix="hook")
    b.focus_points["chain"] = [ROOM_X0 + 1.2, 2.4, -1.0]
    put(b, "industrial_storage_cart", (ROOM_X0 + 0.62, 0.0, 0.95), yaw=90.0)
    b.focus_points["storage_cart"] = [ROOM_X0 + 0.6, 1.0, 0.95]
    put(b, "portable_welding_cart", (ROOM_X0 + 0.5, 0.0, 2.4), yaw=90.0 + float(r.uniform(-10, 10)))
    put(b, "propane_tank", (ROOM_X0 + 1.15, 0.0, 2.55))
    put(b, "small_lpg_tank", (ROOM_X0 + 1.4, 0.0, 1.95), yaw=float(r.uniform(0, 360)))
    put(b, "rubber_boots", (ROOM_X0 + 2.0, 0.0, 1.7), yaw=float(r.uniform(-40, 40)))
    put(b, "old_gas_mask", (ROOM_X0 + 0.08, 1.45, 0.45), yaw=90.0)
    put(b, "utility_box_01", (ROOM_X0 + 0.22, 0.0, -0.15), yaw=90.0)
    put(b, "wooden_ladder_02", (ROOM_X0 + 0.45, 0.0, -1.25), yaw=90.0)
    put(b, "plastic_broom", (ROOM_X0 + 0.12, 0.0, ROOM_Z1 - 0.1), pre=G.rotate((0, 0, 1), 9.0), yaw=45.0)
    put(b, "dustpan", (ROOM_X0 + 1.1, 0.0, ROOM_Z1 - 0.2), yaw=float(r.uniform(0, 360)))
    m, M = put(b, "worn_metal_rack", (-2.2, 0.0, ROOM_Z1 - 0.32), yaw=180.0)
    for top, gap in shelf_levels(m, M):
        fill_line(b, rng_for(ctx, 41), "kit" if top > 1.0 else "crates", top, -2.62, -1.78, ROOM_Z1 - 0.32, axis="x",
                  max_h=min(gap - 0.04, 0.4), max_d=0.55)
    b.focus_points["front_rack"] = [-2.2, 1.0, ROOM_Z1 - 0.32]


# ----------------------------------------------------------------------------
# Right wall: shelving filled with scanned stock, barrels, carts, chest
# ----------------------------------------------------------------------------

SHELF_STOCK = {
    "crates": ["plastic_crate_01", "plastic_crate_02", "plastic_crate_03"],
    "boxes": ["cardboard_box_01"],
    "cans": ["can_rusted", "oil_tin", "cleaner_tin_01", "multi_cleaner_5_litre", "lubricant_spray", "spray_paint_bottles", "plastic_jerrycan"],
    "kit": ["metal_toolbox", "can_rusted", "oil_tin", "spray_paint_bottles_02"],
}


def fill_line(b: SceneBuilder, r, kind: str, top: float, a0: float, a1: float, fixed: float, axis: str = "z",
              max_h: float = 0.4, max_d: float = 0.44) -> None:
    """Pack scanned stock along one shelf from a0 to a1 (along z, or x for a rack on the front wall); `fixed` is
    the shelf's centre on the other horizontal axis, `max_d` its usable depth."""
    a = a0 + 0.03
    pool = SHELF_STOCK[kind]
    tries = 0
    while a < a1 and tries < 12:
        tries += 1
        aid = pool[int(r.integers(0, len(pool)))]
        m = WA.model(aid)
        fp = footprint(m)
        if axis == "z":
            yaw = 90.0 if fp[0] > fp[2] else 0.0
            depth, run = (fp[2], fp[0]) if yaw else (fp[0], fp[2])
        else:
            yaw = 90.0 if fp[2] > fp[0] else 0.0
            depth, run = (fp[0], fp[2]) if yaw else (fp[2], fp[0])
        if depth > max_d or fp[1] > max_h or a + run > a1:
            continue
        jitter = float(r.uniform(-0.03, 0.03))
        at = (fixed + jitter, top, a + run / 2) if axis == "z" else (a + run / 2, top, fixed + jitter)
        put(b, aid, at, yaw=yaw + float(r.choice([0.0, 180.0])) + float(r.uniform(-8, 8)))
        a += run + 0.03


def read_ply_xyz(path: str) -> np.ndarray:
    raw = open(path, "rb").read()
    head, _, body = raw.partition(b"end_header\n")
    n = int(head.split(b"element vertex ")[1].split()[0])
    return np.frombuffer(body, dtype="<f4", count=n * 8).reshape(n, 8)[:, :3].astype(float)


def shelf_levels(m, M) -> list[tuple[float, float]]:
    """(world height of each shelf's top surface, clear height above it) for a scanned shelf unit placed by M.

    A shelf is a 1 cm band of vertices spanning most of the unit's width and depth."""
    p = np.concatenate([read_ply_xyz(part.ply) for part in m.parts])
    lo, hi = p.min(0), p.max(0)
    span = hi - lo
    ys = np.round((p[:, 1] - lo[1]) / 0.01).astype(int)
    tops = []
    for k in np.unique(ys):
        q = p[ys == k]
        if (np.ptp(q[:, 0]) > 0.7 * span[0]) and (np.ptp(q[:, 2]) > 0.7 * span[2]) and len(q) > 8:
            tops.append(float(q[:, 1].max()))
    levels = []
    for y in sorted(tops):
        if levels and y - levels[-1] < 0.12:
            levels[-1] = y
        else:
            levels.append(y)
    world = [float((M @ np.array([0.0, y, 0.0, 1.0]))[1] - (M @ np.array([0.0, lo[1], 0.0, 1.0]))[1]) for y in levels]
    world = [w for w in world if w > 0.05]
    gaps = [b_ - a for a, b_ in zip(world, world[1:])] + [0.45]
    return [(w + 0.004, g) for w, g in zip(world, gaps) if g > 0.15]


def hanging_bulb(b: SceneBuilder, x: float, z: float, y: float, radiance) -> list:
    """A scanned pull-chain socket on a cable from the roof, a scanned bulb under it, and a sphere emitter at the bulb.

    The bulb's glass parts are made `null`: glass blocks next-event estimation, and the emitter sphere is the filament."""
    s = WA.model("pull_chain_light_socket")
    WA.add(b, s, wplace(s, (x, y, z)))
    b.ply(b.assets.mesh(f"bulb_cable_{ROOM_H - y:.2f}".replace(".", "p"), lambda: W.drooping_cable((0, 0, 0), (0.0, -(ROOM_H - 0.08 - y - footprint(s)[1]), 0.0), 0.0, 0.004, 6)),
          G.translate((x, ROOM_H - 0.08, z)), "black_matt", prefix="bulb_cable")
    bulb = WA.model("lightbulb_01")
    glass = {p.name: {"type": "null"} for p in bulb.parts
             if "glass" in p.name.lower() or p.bsdf.get("type", "") in ("dielectric", "thindielectric", "roughdielectric")}
    h = footprint(bulb)[1]
    WA.add(b, bulb, wplace(bulb, (x, y - h + 0.01, z), pre=G.rotate((1, 0, 0), 180.0)), prefix="web_bulb", overrides=glass)
    c = (x, y - h * 0.62, z)
    b.sphere(c, 0.045, "emitter_backing", emission=radiance, prefix="bulb_glow")
    return list(c)


def add_right_side(b: SceneBuilder, ctx: BuildContext) -> None:
    A = b.assets
    r = rng_for(ctx, 5)
    r2 = rng_for(ctx, 51)
    kinds = ["crates", "boxes", "cans", "kit"]
    units = [("steel_frame_shelves_01", -5.35), ("steel_frame_shelves_01", -4.2), ("worn_metal_rack", -3.1)]
    for aid, z in units:
        m = WA.model(aid)
        s = ASSET_SCALE.get(aid, 1.0)
        dx = footprint(m)[2] * s                  # depth after the quarter turn
        x = ROOM_X1 - 0.03 - dx / 2
        M = wplace(m, (x, 0.0, z), yaw=90.0, scale=s)
        WA.add(b, m, M)
        length = footprint(m)[0] * s
        levels = shelf_levels(m, M)
        for li, (top, gap) in enumerate(levels):
            kind = "boxes" if li == len(levels) - 1 else kinds[int(r2.integers(0, len(kinds)))]
            fill_line(b, r2, kind, top, z - length / 2 + 0.04, z + length / 2 - 0.04, x, axis="z", max_h=min(gap - 0.04, 0.4), max_d=dx - 0.04)
    b.focus_points["shelving"] = [ROOM_X1 - 0.45, 1.06, -4.2]
    put(b, "wooden_crate_02", (ROOM_X1 - 1.0, 0.0, -2.0), yaw=90.0)
    m = WA.model("wooden_military_crate")
    m2 = WA.model("wooden_crate_02")
    WA.add(b, m, wplace(m, (ROOM_X1 - 1.0, footprint(m2)[1], -2.0), yaw=95.0))
    barrels = ["Barrel_01", "Barrel_02", "barrel_03"]
    for k, (x, z) in enumerate(((3.45, -1.05), (3.45, -0.4), (2.85, -0.75))):
        put(b, barrels[(ctx.seed + k) % 3], (x + float(r.uniform(-0.03, 0.03)), 0.0, z), yaw=float(r.uniform(0, 360)))
    b.focus_points["drum"] = [3.3, 0.6, -0.75]
    for aid, (x, z) in zip(["metal_jerrycan", "metal_jerrycan_green"] if ctx.seed % 2 else ["metal_jerrycan_green", "metal_jerrycan"],
                           ((2.75, -0.1), (2.5, -0.15))):
        put(b, aid, (x, 0.0, z), yaw=float(r.uniform(-30, 30)))
    put(b, "wooden_broom", (ROOM_X1 - 0.12, 0.0, -1.75), pre=G.rotate((0, 0, 1), 8.0), yaw=90.0)
    m, M = put(b, "tool_cart", (ROOM_X1 - 0.43, 0.0, 0.75), yaw=90.0)
    b.focus_points["tool_cart"] = top_of(m, M).tolist()
    put(b, "small_oil_can_01", (ROOM_X1 - 0.45, top_of(m, M)[1], 0.45), yaw=float(r.uniform(0, 360)))
    put(b, "retro_multimeter", (ROOM_X1 - 0.45, top_of(m, M)[1], 1.0), yaw=-90.0)
    m, M = put(b, "metal_tool_chest", (ROOM_X1 - 0.4, 0.0, 1.95), yaw=-90.0)
    put(b, "boombox", (ROOM_X1 - 0.4, top_of(m, M)[1], 1.95), yaw=-90.0)
    put(b, "dartboard", (ROOM_X1 - 0.03, 1.55, 0.75), yaw=-90.0)
    put(b, "modular_electric_cables", (ROOM_X1 - 0.12, ROOM_H - 0.85, 0.9), yaw=-90.0)
    lean_out = G.rotate((1, 0, 0), 10.0)
    put(b, "rusted_spade_01", (3.15, 0.0, ROOM_Z1 - 0.12), pre=lean_out)
    put(b, "picke_dirty_01", (3.45, 0.0, ROOM_Z1 - 0.12), pre=lean_out)
    b.focus_points["tool_chest"] = [ROOM_X1 - 0.5, 0.5, 1.95]
    put(b, "korean_fire_extinguisher_01", (ROOM_X1 - 0.22, 0.0, 2.72), yaw=-90.0)


# ----------------------------------------------------------------------------
# Back-right corner and the floor: bags, generator, leaning tools, stool, hand truck
# ----------------------------------------------------------------------------

def add_floor_props(b: SceneBuilder, ctx: BuildContext) -> None:
    r = rng_for(ctx, 7)
    lean_back = G.rotate((1, 0, 0), -10.0)
    put(b, "sledgehammer_01", (-0.45, 0.0, ROOM_Z0 + 0.14), pre=lean_back)
    put(b, "bolt_cutters_01", (-0.15, 0.0, ROOM_Z0 + 0.14), pre=lean_back)
    put(b, "crowbar_01", (0.05, 0.0, ROOM_Z0 + 0.1), pre=lean_back)
    put(b, "power_box_01", (0.55, 1.25, ROOM_Z0 + 0.2))
    put(b, "trashbag", (0.6, 0.0, ROOM_Z0 + 0.45), yaw=float(r.uniform(0, 360)))
    y = 0.0
    for k in range(3):
        m, M = put(b, "cement_bag", (1.45 + float(r.uniform(-0.04, 0.04)), y, ROOM_Z0 + 0.45), yaw=90.0 + float(r.uniform(-12, 12)))
        y = top_of(m, M)[1] - 0.01
    put(b, "compost_bag_02", (2.1, 0.0, -5.0), yaw=float(r.uniform(-20, 20)))
    m, M = put(b, "old_military_compressor", (2.6, 0.0, ROOM_Z0 + 0.42), yaw=90.0)
    b.focus_points["compressor"] = top_of(m, M).tolist()
    m, M = put(b, "portable_generator", (2.95, 0.0, -4.7), yaw=80.0)
    b.focus_points["generator"] = top_of(m, M).tolist()
    put(b, "metal_stool_02", (-2.0 + float(r.uniform(-0.25, 0.25)), 0.0, -4.7), yaw=float(r.uniform(0, 360)))
    put(b, "hand_truck", (-0.55, 0.0, ROOM_Z1 - 0.35), pre=G.rotate((1, 0, 0), 8.0), yaw=180.0)
    put(b, "plastic_crate_02", (-1.4, 0.0, ROOM_Z1 - 0.35), yaw=float(r.uniform(-10, 10)))
    # Around the car: a spare wheel lying flat, a drip tray, a crate of parts, a jerrycan by the bench end.
    lie = G.rotate((1, 0, 0), 90)
    m, M = put(b, "old_tyre", (-0.75, 0.0, -1.35), pre=lie, yaw=float(r.uniform(0, 360)))
    put(b, "rusted_wheel_rim_01", (-0.75, top_of(m, M)[1] - 0.12, -1.35), pre=lie)
    b.focus_points["spare_wheel"] = [-0.75, 0.2, -1.35]
    b.box_mesh("drip_tray", (CAR_AT[0] - 0.2, 0.012, CAR_AT[1] + 1.6), (0.8, 0.024, 0.5), "rust_paint")
    put(b, "wall_clock", (1.5, 2.95, ROOM_Z1 - 0.03), yaw=180.0)
    put(b, "fire_alarm", (3.35, 1.45, ROOM_Z1 - 0.02), yaw=180.0)
    hanging_bulb(b, 1.5, ROOM_Z1 - 1.0, 2.75, DOOR_BULB)
    put(b, "plastic_crate_03", (-0.55, 0.0, -0.45), yaw=float(r.uniform(-30, 30)))
    put(b, "plastic_jerrycan", (BENCH_X1 + 0.25, 0.0, BENCH_Z1 - 0.2), yaw=float(r.uniform(0, 360)))


def corrugated(width: float, height: float, pitch: float = 0.075, amp: float = 0.011) -> Mesh:
    """Horizontal corrugation: a sheet in the xy plane, bent in z along y (ribs run along x)."""
    cols, rows = 2, int(height / pitch * 8)
    x = np.linspace(-width / 2, width / 2, cols)
    y = np.linspace(0.0, height, rows + 1)
    z = amp * np.sin(2 * np.pi * y / pitch)
    p = np.stack([np.broadcast_to(x[None, :], (rows + 1, cols)), np.broadcast_to(y[:, None], (rows + 1, cols)),
                  np.broadcast_to(z[:, None], (rows + 1, cols))], -1).reshape(-1, 3)
    dz = amp * 2 * np.pi / pitch * np.cos(2 * np.pi * y / pitch)
    n = np.stack([np.zeros_like(y), -dz, np.ones_like(y)], -1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    nn = np.repeat(n[:, None, :], cols, 1).reshape(-1, 3)
    uv = np.stack(np.broadcast_arrays(np.linspace(0, 1, cols)[None, :], np.linspace(0, 1, rows + 1)[:, None]), -1).reshape(-1, 2)
    m = Mesh(p, nn, uv, G._grid_faces(rows + 1, cols))
    return G.Mesh.__iadd__(m, m.transformed(G.compose(G.scale((1.0, 1.0, -1.0)), G.translate((0, 0, 0))))) if False else m


def add_beams_and_lights(b: SceneBuilder, ctx: BuildContext) -> None:
    """Roof structure: rusty I-beams across, steel purlins along, under the corrugated sheets; ducts, pipes and cables;
    scanned 4-tube fluorescent panels bolted to the beams, each with one rectangular emitter just under it."""
    A = b.assets
    beam = A.mesh("ibeam8", lambda: W.i_beam(8.0))
    for z in (-5.0, -3.0, -1.0, 1.0, 2.7):
        b.ply(beam, G.translate((0.0, ROOM_H, z)), "rust_paint", prefix="beam")
    for x in np.arange(-3.5, 3.51, 1.0):
        b.box_mesh("purlin_9m", (float(x), ROOM_H - 0.05, 0.5 * (ROOM_Z0 + ROOM_Z1)), (0.05, 0.1, 9.0), "purlin")
    panel = WA.model("mounted_fluorescent_lights")
    th = footprint(panel)[1]
    w_, d_ = footprint(panel)[0], footprint(panel)[2]
    for z in (-5.0, -3.0, -1.0, 1.0, 2.7):
        for x in (-2.1, 2.1):
            y = ROOM_H - 0.22 - th
            WA.add(b, panel, wplace(panel, (x, y, z)))
            # Emitter 2 mm under the panel, standing in for its tubes behind a diffuser; same power as the old battens.
            b.rect(G.compose(G.translate((x, y - 0.002, z)), G.rotate((1, 0, 0), 90), G.scale((0.46 * w_, 0.44 * d_, 1))),
                   "emitter_backing", emission=PANEL, prefix="tube", power=0.92 * w_ * 0.88 * d_ * luminance(PANEL))
            # Real fittings spill some light up through their vents; without it the roof reads black from below.
            b.rect(G.compose(G.translate((x, y + th + 0.003, z)), G.rotate((1, 0, 0), -90), G.scale((0.46 * w_, 0.44 * d_, 1))),
                   "emitter_backing", emission=tuple(0.1 * c for c in PANEL), prefix="uplight",
                   power=0.1 * 0.92 * w_ * 0.88 * d_ * luminance(PANEL))
    # Drooping cables tied along the beams, with a yellow extension lead.
    cab = A.mesh("cable_a", lambda: W.drooping_cable((0, 0, 0), (3.2, 0.0, 0.0), 0.35, 0.006))
    b.ply(cab, G.translate((-3.9, 3.38, -3.9)), "black_matt", prefix="cable")
    cab2 = A.mesh("cable_b", lambda: W.drooping_cable((0, 0, 0), (2.6, 0.0, 0.0), 0.28, 0.007))
    b.ply(cab2, G.translate((1.3, 3.38, -1.5)), "cable_yellow", prefix="cable")
    b.ply(A.mesh("cable_c", lambda: W.drooping_cable((0, 0, 0), (0.0, -0.55, 1.6), 0.15, 0.005)), G.translate((-3.0, 3.38, -0.2)), "black_matt", prefix="cable")
    # Round duct along the left side under the roof, pipe run high on the back wall, a caged pendant over the left bay.
    duct = WA.model("modular_airduct_circular_01")
    WA.add(b, duct, wplace(duct, (ROOM_X0 + 0.95, ROOM_H - 0.12 - footprint(duct)[1], -2.6), yaw=90.0))
    pipes = WA.model("modular_industrial_pipes_01")
    WA.add(b, pipes, wplace(pipes, (2.0, ROOM_H - 0.1 - footprint(pipes)[1], ROOM_Z0 + 0.03 + footprint(pipes)[2] / 2)))
    cage = WA.model("caged_hanging_light")
    WA.add(b, cage, wplace(cage, (-2.4, ROOM_H - footprint(cage)[1], 0.7)))


# (asset, pre-rotation, scales): every pegboard tool is scanned; pre turns it to hang long axis up, thin axis off the wall.
PEG_TOOLS = [
    ("combination_wrench", G.rotate((1, 0, 0), -90), (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)),
    ("adjustable_wrench", None, (0.6, 0.8, 1.0)),
    ("pliers", None, (0.9, 1.0)),
    ("tongue_groove_pliers", G.rotate((1, 0, 0), -90), (0.85, 1.0)),
    ("cross_pein_hammer", None, (0.9, 1.0)),
    ("wooden_hammer_01", None, (1.0,)),
    ("handsaw_wood", G.compose(G.rotate((0, 1, 0), 90), G.rotate((1, 0, 0), -90)), (0.75, 0.9)),
    ("rusted_hacksaw", G.rotate((0, 0, 1), 90), (1.0,)),
    ("screwdriver", None, (0.6, 0.75, 0.9, 1.05, 1.2, 1.3)),
    ("flathead_screwdriver", None, (0.6, 0.75, 0.9, 1.05)),
]


def add_pegboard(b: SceneBuilder, ctx: BuildContext) -> None:
    A = b.assets
    zs = ROOM_Z0 + 0.012            # board front surface
    w = BENCH_X1 - BENCH_X0 + 0.1
    h = PEG_Y1 - PEG_Y0
    b.box_mesh("pegboard_31x130", (0.5 * (BENCH_X0 + BENCH_X1), 0.5 * (PEG_Y0 + PEG_Y1), ROOM_Z0 + 0.006), (w, h, 0.012), "peg")
    r = rng_for(ctx, 2)
    pool = [(aid, pre, s) for aid, pre, scales in PEG_TOOLS for s in scales]
    order = [pool[int(i)] for i in r.permutation(len(pool))]
    slots = 28                                  # 0.1 m wide; tools wider than WIDE take two neighbouring slots
    WIDE = 0.075
    x_slot = lambda k: BENCH_X0 + 0.1 + k * 0.1
    cursor = [PEG_Y1 - 0.06] * slots
    placed = 0
    for aid, pre, s in order * 4:
        m = WA.model(aid)
        fp = footprint(m, pre) * s
        ext, wide = fp[1], fp[0] > WIDE
        free = lambda k: min(cursor[k], cursor[k + 1]) if wide else cursor[k]
        cand = [k for k in range(slots - (1 if wide else 0)) if free(k) - ext > PEG_Y0 + 0.05]
        if not cand:
            continue
        k = max(cand, key=free)
        top = free(k)
        cx = x_slot(k) + (0.05 if wide else 0.0) + float(r.uniform(-0.008, 0.008))
        tilt = G.compose(G.rotate((0, 0, 1), float(r.uniform(-2.5, 2.5))), pre if pre is not None else np.eye(4))
        WA.add(b, m, wplace(m, (cx, top - ext, zs + fp[2] / 2 + 0.004), pre=tilt, scale=s))
        b.ply(A.mesh("hook_pin", W.hook_pin), G.translate((cx, top - 0.012, zs)), "steel_plain", prefix="hook")
        for q in ((k, k + 1) if wide else (k,)):
            cursor[q] = top - ext - 0.03
        placed += 1
        if placed == 12:
            b.focus_points["pegboard_tools"] = [cx, top - ext / 2, zs + 0.02]


def build(ctx: BuildContext) -> SceneBundle:
    local = Assets(ctx.assets_dir, ctx.rebuild)
    b = SceneBuilder(Assets(ctx.shared_dir, ctx.rebuild))
    add_materials(b, local, ctx)
    add_room(b)
    add_beams_and_lights(b, ctx)
    add_bench(b, local, ctx)
    add_pegboard(b, ctx)
    add_car(b, ctx)
    add_left_side(b, ctx)
    add_right_side(b, ctx)
    add_floor_props(b, ctx)
    sun = env_kit.default_sun_direction(SUN_ELEVATION, SUN_AZIMUTH)
    b.d["sky"] = env_kit.sky("clear", sun_direction=sun, scale=SKY_SCALE, sampling_weight=0.5 * b.lamp_weight)
    b.focus_points["door"] = [1.5, 1.0, ROOM_Z1 - 0.2]
    return SceneBundle(b.d, b.focus_points, {"sun_direction": sun, "sky_scale": SKY_SCALE})


SCENE = SceneDef(
    id="workshop", group="artificial", owner="rui",
    description="Lived-in workshop: scanned tools and clutter on a bench under a pegboard of ~100 tools, a covered car, stocked shelving, barrels, carts, fluorescent light.",
    build=build,
    views={
        # Close on the bench and the lower half of the pegboard.
        "bench": View((-2.85, 1.42, -3.2), (-2.85, 1.3, -5.9), focus="vise"),
        # Along the shop floor past the covered car, shelving on the right, the bench far behind.
        "along": View((-1.5, 1.5, 2.4), (-0.4, 0.9, -3.6), focus="car"),
        # Back toward the roller door from beside the car.
        "door": View((-1.9, 1.6, -4.3), (1.2, 1.15, 3.0), focus="work_light"),
        # Across to the stocked shelving and barrels on the right wall.
        "shelves": View((2.4, 1.6, 0.9), (3.6, 0.95, -4.2), focus="shelving"),
        # Across to the left wall: ladder, tyres, old drill press, carts.
        "left": View((-1.1, 1.6, 1.4), (-3.7, 0.9, -3.6), focus="drill_press"),
    },
    default_view="bench",
    camera_box=((-3.0, 0.9, -4.9), (3.0, 2.1, 2.7)),
    target_box=((-3.8, 0.0, -5.9), (3.9, 2.4, 3.0)),
    exclude_boxes=(
        ((0.35, 0.0, -4.95), (2.45, 1.7, -0.15)),    # covered car
        ((-2.75, 0.0, 2.2), (-1.65, 2.0, 3.0)),       # rack on the front wall
        ((-3.0, 0.0, 0.1), (-2.6, 1.6, 2.7)),        # storage and welding carts
        ((-0.95, 0.0, 2.2), (-0.1, 1.6, 2.7)),       # hand truck
    ),
    # The room is 8 x 9 m: past ~90 mm most random poses frame a single stretch of wall.
    lens_mm=(24.0, 90.0),
    tags=("indoor", "artificial_light", "clutter", "metal", "specular"),
    default_seed=5,
    asset_version=4,
    spp_hint=2048,
)
