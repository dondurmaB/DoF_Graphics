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
TUBE = (55.0, 57.0, 62.0)          # cool-white fluorescent radiance
LAMP = (40.0, 26.0, 12.0)           # warm bulbs in the hanging lamps; big and dim, not small and hot (fireflies)
WORKLIGHT = (170.0, 120.0, 62.0)
CAR_AT = (1.0, -2.55)
SHELF_X = ROOM_X1 - 0.23
SHELF_Z = [-5.45, -4.45, -3.45, -2.45]
SHELF_LEVELS = (0.18, 0.62, 1.06, 1.5, 1.94)
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


def put(b: SceneBuilder, aid: str, at, yaw=0.0, pre=None, scale=1.0, height=None, prefix=None, overrides=None):
    m = WA.model(aid)
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
    b.material("yard", WA.surface("asphalt_floor", res="2k", uv_scale=(24.0 / 2.35, 11.0 / 2.35)))
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
    b.material("rag_red", {"type": "diffuse", "reflectance": rgb(0.55, 0.12, 0.08)})
    b.material("rag_grey", {"type": "diffuse", "reflectance": rgb(0.45, 0.43, 0.40)})
    b.material("mat_rubber", principled((0.025, 0.025, 0.027), 0.8))
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
            b.cube((ROOM_X0 - 0.06, 0.5 * sum(WIN_Y), z), (0.05, WIN_Y[1] - WIN_Y[0], f), "paint_black")
        for y in (WIN_Y[0] + f / 2, WIN_Y[1] - f / 2, 0.5 * sum(WIN_Y)):
            b.cube((ROOM_X0 - 0.06, y, zc), (0.05, f, z1 - z0), "paint_black")
    # Door jambs and the roller door: corrugated slats raised to DOOR_SLATS_BOTTOM, a roll housing above.
    d0, d1 = DOOR_X
    b.box_mesh("jamb", (d0 - 0.02, 0.5 * DOOR_TOP, ROOM_Z1 - 0.1), (0.14, DOOR_TOP, 0.2), "plaster")
    b.box_mesh("jamb", (d1 + 0.02, 0.5 * DOOR_TOP, ROOM_Z1 - 0.1), (0.14, DOOR_TOP, 0.2), "plaster")
    b.box_mesh("lintel", (0.5 * (d0 + d1), DOOR_TOP + 0.01, ROOM_Z1 - 0.1), (d1 - d0 + 0.2, 0.04, 0.2), "plaster")
    for x in (d0 + 0.02, d1 - 0.02):
        b.cube((x, 0.5 * (DOOR_SLATS_BOTTOM + DOOR_TOP), ROOM_Z1 - 0.14), (0.05, DOOR_TOP - DOOR_SLATS_BOTTOM, 0.08), "paint_grey")
    slats = A.mesh("door_slats", lambda: corrugated(d1 - d0 - 0.06, DOOR_TOP - DOOR_SLATS_BOTTOM))
    b.ply(slats, G.translate((0.5 * (d0 + d1), DOOR_SLATS_BOTTOM, ROOM_Z1 - 0.15)), "door_slats", prefix="slats")
    b.cube((0.5 * (d0 + d1), DOOR_TOP - 0.12, ROOM_Z1 - 0.2), (d1 - d0 - 0.06, 0.2, 0.2), "paint_grey")
    b.cube((0.5 * (d0 + d1), DOOR_SLATS_BOTTOM - 0.015, ROOM_Z1 - 0.15), (d1 - d0 - 0.06, 0.03, 0.06), "paint_grey")
    # Outside: yard, the facing building with an aircon unit, and ground to the horizon (sunsky needs it).
    b.rect(G.compose(G.translate((0, -0.02, 0)), G.rotate((1, 0, 0), -90), G.scale((400.0, 400.0, 1))), "ground")
    b.rect(G.compose(G.translate((0, 0.005, 0.5 * (ROOM_Z1 + 14.0))), G.rotate((1, 0, 0), -90), G.scale((12.0, 0.5 * (14.0 - ROOM_Z1), 1))), "yard")
    b.rect(G.compose(G.translate((0, 4.0, 14.0)), G.rotate((0, 1, 0), 180), G.scale((12.0, 4.0, 1))), "brick_far")
    put(b, "exterior_aircon_unit", (2.2, 0.0, 13.6), yaw=180.0)
    put(b, "Barrel_02", (-0.6, 0.0, 12.8))


# ----------------------------------------------------------------------------
# Workbench: procedural frame with scanned planks, scanned vise and drill press, a seed-chosen spread of scanned clutter
# ----------------------------------------------------------------------------

BENCH_TALL = ["Drill_01", "lubricant_spray", "spray_paint_bottles_02", "oil_tin", "cleaner_tin_01", "can_rusted", "small_oil_can_01",
              "brass_blowtorch", "metal_toolbox", "hand_plane_no4", "spray_paint_bottles", "measuring_tape_01"]
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
    b.part_set(pc, G.translate((BENCH_X0 + 0.75, BENCH_Y, bz0 + 0.12)), {"frame": "paint_grey", "fronts": "bin_yellow", "pulls": "steel_plain"})
    b.focus_points["parts_cabinet"] = [BENCH_X0 + 0.75, BENCH_Y + 0.18, bz0 + 0.2]
    can = A.meshes("coffee_can", lambda: W.paint_can(0.055, 0.15))
    cx, cz = BENCH_X0 + 1.55, bz0 + 0.3
    b.part_set(can, G.translate((cx, BENCH_Y, cz)), {"can": "can_tin", "handle": "steel_plain"})
    r3 = rng_for(ctx, 12)
    handles = ["grip_red", "grip_blue", "grip_yellow", "grip_green", "grip_orange", "grip_black"]
    for k in range(6):
        sd = A.meshes("sd_100", lambda: W.screwdriver(0.1, True))
        a = float(r3.uniform(0, 360))
        tilt = float(r3.uniform(4, 22))
        off = float(r3.uniform(0.0, 0.025))
        m_ = G.compose(G.translate((cx + off * np.cos(np.radians(a)), BENCH_Y + 0.025, cz + off * np.sin(np.radians(a)))), G.rotate((0, 1, 0), a),
                       G.rotate((0, 0, 1), -tilt), G.rotate((0, 0, 1), 90))
        b.part_set(sd, m_, {"steel": "tool_steel", "handle": handles[int(r3.integers(0, 6))]})

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
    m, M = put(b, "covered_car", (cx, 0.0, cz), yaw=180.0)
    t = top_of(m, M)
    b.focus_points["car"] = [cx, t[1] - 0.1, cz + 0.4]
    b.focus_points["car_rear"] = [cx, 0.8, cz - 2.1]
    put(b, "tire_pump", (cx - 1.25, 0.0, cz + 1.9), yaw=30.0)
    lx, lz = 1.05, -2.35
    b.ply(A.mesh("wl_cable", lambda: W.drooping_cable((0, 0, 0), (0.0, -0.75, 0.0), 0.0, 0.0045, 10)), G.translate((lx, 3.38, lz)), "black_matt", prefix="wl")
    wl = A.meshes("work_light", W.caged_work_light)
    b.part_set(wl, G.translate((lx, 2.55, lz)), {"cage": "steel_plain", "body": "paint_yellow"})
    b.sphere((lx, 2.46, lz), 0.026, "emitter_backing", emission=WORKLIGHT, prefix="wl_bulb")
    b.focus_points["work_light"] = [lx, 2.5, lz]


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


# ----------------------------------------------------------------------------
# Right wall: shelving filled with scanned stock, barrels, carts, chest
# ----------------------------------------------------------------------------

SHELF_STOCK = {
    "crates": ["plastic_crate_01", "plastic_crate_02", "plastic_crate_03"],
    "boxes": ["cardboard_box_01"],
    "cans": ["can_rusted", "oil_tin", "cleaner_tin_01", "multi_cleaner_5_litre", "lubricant_spray", "spray_paint_bottles", "plastic_jerrycan"],
    "kit": ["metal_toolbox", "can_rusted", "oil_tin", "spray_paint_bottles_02"],
}


def fill_shelf(b: SceneBuilder, r, kind: str, top: float, z0: float, z1: float) -> None:
    """Pack scanned stock along one shelf from z0 to z1; the shelf is 0.43 m deep along x."""
    z = z0 + 0.03
    pool = SHELF_STOCK[kind]
    tries = 0
    while z < z1 and tries < 12:
        tries += 1
        aid = pool[int(r.integers(0, len(pool)))]
        m = WA.model(aid)
        fp = footprint(m)
        yaw = 90.0 if fp[0] > fp[2] else 0.0
        dx, dz = (fp[2], fp[0]) if yaw else (fp[0], fp[2])
        if dx > 0.44 or fp[1] > 0.4 or z + dz > z1:
            continue
        put(b, aid, (SHELF_X + float(r.uniform(-0.03, 0.03)), top, z + dz / 2), yaw=yaw + float(r.choice([0.0, 180.0])) + float(r.uniform(-8, 8)))
        z += dz + 0.03


def add_right_side(b: SceneBuilder, ctx: BuildContext) -> None:
    A = b.assets
    r = rng_for(ctx, 5)
    bay = A.meshes("shelving_bay", W.shelving_bay)
    face = G.rotate((0, 1, 0), -90.0)
    for z in SHELF_Z:
        b.part_set(bay, G.compose(G.translate((SHELF_X, 0.0, z)), face), {"post": "shelf_post", "shelf": "shelf_plate"})
    r2 = rng_for(ctx, 51)
    kinds = ["crates", "boxes", "cans", "kit"]
    for z in SHELF_Z:
        for li, y in enumerate(SHELF_LEVELS):
            kind = "boxes" if li == len(SHELF_LEVELS) - 1 else kinds[int(r2.integers(0, len(kinds)))]
            fill_shelf(b, r2, kind, y + 0.008, z - 0.47, z + 0.47)
    b.focus_points["shelving"] = [SHELF_X - 0.2, 1.06, SHELF_Z[1]]
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
    put(b, "metal_tool_chest", (ROOM_X1 - 0.4, 0.0, 1.95), yaw=-90.0)
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
    put(b, "compost_bag_02", (2.2, 0.0, ROOM_Z0 + 0.4), yaw=float(r.uniform(-20, 20)))
    m, M = put(b, "portable_generator", (2.55, 0.0, -4.7), yaw=80.0)
    b.focus_points["generator"] = top_of(m, M).tolist()
    put(b, "metal_stool_02", (-2.0 + float(r.uniform(-0.25, 0.25)), 0.0, -4.7), yaw=float(r.uniform(0, 360)))
    put(b, "hand_truck", (-0.55, 0.0, ROOM_Z1 - 0.35), pre=G.rotate((1, 0, 0), 8.0), yaw=180.0)
    put(b, "plastic_crate_02", (-1.4, 0.0, ROOM_Z1 - 0.35), yaw=float(r.uniform(-10, 10)))
    # Around the car: a spare wheel lying flat, a drip tray, a crate of parts, a jerrycan by the bench end.
    lie = G.rotate((1, 0, 0), 90)
    m, M = put(b, "old_tyre", (-0.75, 0.0, -1.35), pre=lie, yaw=float(r.uniform(0, 360)))
    put(b, "rusted_wheel_rim_01", (-0.75, top_of(m, M)[1] - 0.12, -1.35), pre=lie)
    b.focus_points["spare_wheel"] = [-0.75, 0.2, -1.35]
    b.box_mesh("drip_tray", (CAR_AT[0] - 0.2, 0.012, CAR_AT[1] + 1.6), (0.8, 0.024, 0.5), "paint_grey")
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
    A = b.assets
    beam = A.mesh("ibeam8", lambda: W.i_beam(8.0))
    fix = A.meshes("batten", W.tube_light)
    rng = rng_for(ctx, 6)
    for z in (-5.0, -3.0, -1.0, 1.0, 2.7):
        b.ply(beam, G.translate((0.0, ROOM_H, z)), "paint_grey", prefix="beam")
    for z in (-5.0, -3.0, -1.0, 1.0, 2.7):
        for x in (-2.1, 2.1):
            y = ROOM_H - 0.3
            b.part_set(fix, G.translate((x, y, z)), {"housing": "fixture"})
            # One emitter per fixture standing in for its two tubes: 20 emitters instead of 40, so next-event estimation
            # finds the right one twice as often, which matters most for glossy metal that mirrors a light.
            b.rect(G.compose(G.translate((x, y - 0.003, z)), G.rotate((1, 0, 0), 90), G.scale((0.72, 0.045, 1))),
                   "emitter_backing", emission=TUBE, prefix="tube", power=1.44 * 0.09 * luminance(TUBE))
            for dx in (-0.6, 0.6):                                            # hanging chains to the beam
                b.ply(A.mesh("chain_short", lambda: W.chain(0.28, 0.03, 0.016, 0.0028)), G.translate((x + dx, ROOM_H - 0.2, z)), "black_matt", prefix="chain")
    # Drooping cables tied along the beams, with a yellow extension lead.
    cab = A.mesh("cable_a", lambda: W.drooping_cable((0, 0, 0), (3.2, 0.0, 0.0), 0.35, 0.006))
    b.ply(cab, G.translate((-3.9, 3.38, -3.9)), "black_matt", prefix="cable")
    cab2 = A.mesh("cable_b", lambda: W.drooping_cable((0, 0, 0), (2.6, 0.0, 0.0), 0.28, 0.007))
    b.ply(cab2, G.translate((1.3, 3.38, -1.5)), "cable_yellow", prefix="cable")
    b.ply(A.mesh("cable_c", lambda: W.drooping_cable((0, 0, 0), (0.0, -0.55, 1.6), 0.15, 0.005)), G.translate((-3.0, 3.38, -0.2)), "black_matt", prefix="cable")


def ply_extent(paths: list[str]) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) of binary little-endian PLYs written by procedural.Mesh.write_ply (8 float32 per vertex)."""
    lo, hi = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
    for path in paths:
        raw = open(path, "rb").read()
        head, _, body = raw.partition(b"end_header\n")
        n = int(head.split(b"element vertex ")[1].split()[0])
        xy = np.frombuffer(body, dtype="<f4", count=n * 8).reshape(n, 8)[:, :2]
        lo, hi = np.minimum(lo, xy.min(0)), np.maximum(hi, xy.max(0))
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def add_pegboard(b: SceneBuilder, ctx: BuildContext) -> None:
    A = b.assets
    zs = ROOM_Z0 + 0.012            # board front surface
    w = BENCH_X1 - BENCH_X0 + 0.1
    h = PEG_Y1 - PEG_Y0
    b.box_mesh("pegboard_31x130", (0.5 * (BENCH_X0 + BENCH_X1), 0.5 * (PEG_Y0 + PEG_Y1), ROOM_Z0 + 0.006), (w, h, 0.012), "peg")
    r = rng_for(ctx, 2)
    handles = ["grip_red", "grip_blue", "grip_yellow", "grip_green", "grip_orange", "grip_black"]
    pool = []
    for sz, ln in ((0.008, 0.12), (0.009, 0.125), (0.010, 0.14), (0.011, 0.15), (0.012, 0.16), (0.013, 0.17), (0.014, 0.185), (0.015, 0.19),
                   (0.016, 0.2), (0.017, 0.21), (0.019, 0.23), (0.021, 0.25)):
        pool.append((f"pc_{int(sz * 1000)}", lambda sz=sz, ln=ln: W.combination_wrench(sz, ln), -90.0))
    for sz, ln in ((0.009, 0.12), (0.012, 0.15), (0.014, 0.17), (0.016, 0.19)):
        pool.append((f"po_{int(sz * 1000)}", lambda sz=sz, ln=ln: W.open_end_wrench(sz, ln), -90.0))
    pool += [("adj_15", lambda: W.adjustable_spanner(0.15), -90.0), ("adj_25", lambda: W.adjustable_spanner(0.25), -90.0),
             ("pl_16", lambda: W.pliers(0.16), 90.0), ("pl_20", lambda: W.pliers(0.2), 90.0), ("pl_18", lambda: W.pliers(0.18), 90.0),
             ("ham_30", lambda: W.hammer(0.3), 90.0), ("ham_36", lambda: W.hammer(0.36), 90.0), ("ham_26", lambda: W.hammer(0.26), 90.0),
             ("saw_45", lambda: W.handsaw(0.45), -90.0), ("saw_55", lambda: W.handsaw(0.55), -90.0), ("hack", W.hacksaw, -90.0)]
    for i, sh in enumerate((0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.19, 0.21)):
        pool.append((f"sd_{int(sh * 1000)}", lambda sh=sh: W.screwdriver(sh, i % 3 != 0, 0.0095 + 0.0002 * i), 90.0))
    order = [pool[int(i)] for i in r.permutation(len(pool))]
    slots = 28                                  # 0.1 m wide; tools wider than WIDE take two neighbouring slots
    WIDE = 0.075
    x_slot = lambda k: BENCH_X0 + 0.1 + k * 0.1
    cursor = [PEG_Y1 - 0.06] * slots
    mats = {"steel": "tool_steel", "wood": "wood_handle"}
    placed = 0
    for name, build, rotz in order * 6:
        parts = A.meshes(f"pegtool_{name}", build)
        xmin, xmax, ymin, ymax = ply_extent(list(parts.values()))
        ext, wide = xmax - xmin, (ymax - ymin) > WIDE
        free = lambda s: min(cursor[s], cursor[s + 1]) if wide else cursor[s]
        cand = [s for s in range(slots - (1 if wide else 0)) if free(s) - ext > PEG_Y0 + 0.05]
        if not cand:
            continue
        k = max(cand, key=free)
        top = free(k)
        top_local = -xmin if rotz < 0 else xmax
        oy = top - top_local
        cx = x_slot(k) + (0.05 if wide else 0.0) + float(r.uniform(-0.008, 0.008))
        m = G.compose(G.translate((cx, oy, zs + 0.016)), G.rotate((0, 0, 1), rotz + float(r.uniform(-2.5, 2.5))))
        grip = handles[int(r.integers(0, 6))]
        b.part_set(parts, m, {**mats, "grip": grip, "handle": grip})
        b.ply(A.mesh("hook_pin", W.hook_pin), G.translate((cx, top - 0.012, zs)), "steel_plain", prefix="hook")
        for s in ((k, k + 1) if wide else (k,)):
            cursor[s] = top - ext - 0.03
        placed += 1
        if placed == 12:
            b.focus_points["pegboard_tools"] = [cx, top - 0.1, zs + 0.02]
    # Shadow-board style outlines would be next; a shelf under the pegboard holds nothing here.


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
        "along": View((-1.7, 1.5, 2.4), (-0.4, 0.9, -3.6), focus="car"),
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
        ((-0.05, 0.0, -4.95), (2.05, 1.7, -0.15)),   # covered car
        ((-3.0, 0.0, 0.1), (-2.6, 1.6, 2.7)),        # storage and welding carts
        ((-0.95, 0.0, 2.2), (-0.1, 1.6, 2.7)),       # hand truck
    ),
    # The room is 8 x 9 m: past ~90 mm most random poses frame a single stretch of wall.
    lens_mm=(24.0, 90.0),
    tags=("indoor", "artificial_light", "clutter", "metal", "specular"),
    default_seed=5,
    asset_version=3,
    spp_hint=2048,
)
