"""An indoor public swimming pool (natatorium): the WATER case of the set. A rippled, refracting water surface over
a tiled 25 m pool, lane ropes of toothed floats, starting blocks, backstroke flags, spectator stands, a glazed wall.

Layout, meters, y up, the default camera looks toward -z down the lanes:
  hall        x in [-9.5, 13], z in [6, -30], ceiling at 8 with glulam beams every 4 m and pendant high-bays
  pool        x in [-6, 6], z in [0, -25], water at y = -0.15; the floor slopes from 2.0 m deep at the near
              (starting) end to 1.2 m at the far end; 6 lanes of 2 m, lane ropes at x = -4, -2, 0, 2, 4
  decks       3.5 m on the left (glazed wall side), 3 m on the right, 6 m near, 5 m far (y = 0)
  stands      4 tiers of moulded seats along the right wall, x in [9, 13]
  left wall   floor-to-ceiling openings with mullions and transoms, looking out on a lawn, trees, a hedge
  far wall    a lit results board, a pace clock, signs and doors

THE WATER TRAP, and what this scene does about it. The water is a real `dielectric` (ior 1.333) on a height
field, so the camera sees genuine refraction (wobbling lane lines, a bent ladder) and Fresnel reflections of
the hall. The price: Mitsuba's path tracer cannot do next-event estimation through a dielectric, so the tiles
under water are lit only by paths that refract back out and hit an emitter by chance, or bounce once more off a
lit surface (the ceiling, walls, deck) that does get light samples. The scene is designed so those paths are
cheap and quiet:
  * The sun never enters the hall. It stands behind the stands side (+x); the glazed wall faces the other way.
    Indoor daylight is skylight through big openings, which an escaping path finds easily. Sunlight would
    make floor caustics, which a unidirectional path tracer cannot converge in any sane spp, and a sun disc
    seen through water is a firefly generator.
  * The artificial light is 24 large (0.8 m) pendant discs of moderate radiance, not small bright points.
    A path from the pool floor that happens to hit one carries a bounded contribution.
  * No caustics are attempted. The floor reads as evenly lit, gently shaded tiles seen through moving water,
    which is what an indoor pool under diffuse light looks like.
Measured noise and the spp this needs are recorded in SCENE.spp_hint.

Environments: `clear` has a sunlit exterior (lawn, trees, a far building) and blue skylight through the openings;
`overcast` has a uniform grey sky. The seed picks the ripple pattern, lane rope colour schemes, flag colours,
seat colours, which benches and toys are out and where, the results board, and the wall signs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

import mitsuba as mi
import env_kit
import procedural as G
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, luminance, principled, rgb, xf
from scenes import _pool_props as P
from scenes import _pool_tex as T

# ---- layout --------------------------------------------------------------------------------------------------
X_L, X_R, Z_N, Z_F, CEIL = -9.5, 13.0, 6.0, -30.0, 8.0
PX, PZ0, PZ1 = 6.0, 0.0, -25.0                     # pool half-width, near and far edges
WATER_Y = -0.15
Y_DEEP, Y_SHALLOW = -2.15, -1.35                   # floor at the near (deep) and far (shallow) ends
LANES = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
ROPES = [-4.0, -2.0, 0.0, 2.0, 4.0]
STAND_X0, TIERS, TIER_H = 9.0, 4, 0.45
STAND_Z = (4.5, -28.5)
Z_BEAMS = [4.0 - 4.0 * k for k in range(9)]         # 4 .. -28
PENDANT_Z = [z - 2.0 for z in Z_BEAMS[:-1]]          # 2 .. -26
PENDANT_X = [-4.5, 0.0, 4.5]
FLAG_Z = [-5.0, -20.0]
FLAG_POLE_X = 7.2
LADDERS = [(-1, -3.5), (-1, -21.5), (1, -3.5), (1, -21.5)]   # (side, z)
CHAIR = (-7.9, -12.5)
RACK = (-8.7, 4.4)
BIN = (-8.6, 2.8)

# ---- light and exposure ------------------------------------------------------------------------------------------
SUN_DIR = env_kit.default_sun_direction(48.0, 75.0)  # from +x (behind the stands): never enters the hall
SKY_SCALE = {"clear": 20.0, "overcast": 40.0}
SKY_SHARE = 1.0                                     # sky's light-sample share relative to all lamps together
PENDANT_L = 9.0                                     # the downlight disc under each pendant
UPLIGHT_L = 40.0                                    # an up-facing disc on top: lights the ceiling, which lights the water
UPLIGHT_SAMPLE_SHARE = 0.2
PENDANT_RGB = (1.0, 0.93, 0.82)
SCORE_LEVEL = 2.5
ASSET_VERSION = 4

FLOAT_COLS = [(0.62, 0.03, 0.03), (0.02, 0.10, 0.48), (0.85, 0.62, 0.03), (0.03, 0.42, 0.12), (0.86, 0.86, 0.84),
              (0.95, 0.35, 0.03)]
FLAG_COLS = [(0.70, 0.04, 0.04), (0.88, 0.88, 0.86), (0.03, 0.12, 0.55), (0.90, 0.70, 0.03), (0.03, 0.45, 0.15)]
SEAT_COLS = [(0.02, 0.13, 0.42), (0.55, 0.04, 0.05), (0.86, 0.86, 0.84), (0.03, 0.33, 0.52), (0.78, 0.56, 0.04)]
TOY_COLS = [(0.85, 0.15, 0.08), (0.05, 0.35, 0.80), (0.95, 0.80, 0.08), (0.10, 0.65, 0.25), (0.85, 0.30, 0.55),
            (0.95, 0.50, 0.05)]


# ---- helpers -------------------------------------------------------------------------------------------------------
def tex(path: str, raw: bool = False, scale=None, to_uv=None) -> dict:
    spec = {"type": "bitmap", "filename": path, "raw": raw, "filter_type": "bilinear", "wrap_mode": "repeat"}
    if scale is not None:
        spec["to_uv"] = mi.ScalarTransform4f().scale([scale[0], scale[1], 1.0])
    if to_uv is not None:
        spec["to_uv"] = to_uv
    return spec


def save_exr(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".exr")
    os.close(fd)
    mi.Bitmap(np.ascontiguousarray(arr, np.float32)).write(tmp)
    os.replace(tmp, path)


def png(A: Assets, name: str, fn, gray: bool = False) -> str:
    path = A.root / "textures" / f"{name}.png"
    if A.rebuild or not path.exists():
        a = fn()
        T.save_png8(path, np.repeat(a[..., None], 3, -1) if gray else a)
    return str(path)


def exr(A: Assets, name: str, fn) -> str:
    path = A.root / "textures" / f"{name}.exr"
    if A.rebuild or not path.exists():
        save_exr(path, fn())
    return str(path)


def flip(a: np.ndarray) -> np.ndarray:
    """Rectangle uv has v = 0 at the bottom while bitmap rows start at the top: store rectangle maps bottom-up."""
    return np.ascontiguousarray(a[::-1])


def rect_m(centre, right, up, w: float, h: float) -> np.ndarray:
    """A w x h rectangle at `centre`; `right`/`up` are the text directions seen by a viewer it faces."""
    r, u = np.asarray(right, float), np.asarray(up, float)
    n = np.cross(r, u)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r * w / 2, u * h / 2, n, centre
    return m


def textured_rect(b: SceneBuilder, name: str, path: str, centre, right, up, w, h, roughness=0.5) -> None:
    b.material(name, principled(tex(path), roughness))
    b.rect(rect_m(centre, right, up, w, h), name, prefix="decal")


def mesh_once(b: SceneBuilder, A: Assets, name: str, fn, matrix, material: str, prefix: str) -> None:
    b.ply(A.mesh(name, fn), matrix, material, prefix=prefix)


def deck_top() -> G.Mesh:
    """The hall floor around the pool as four quads, uv = position in the whole-floor deck texture."""
    W, D = STAND_X0 - X_L, Z_N - Z_F
    parts = []
    for x0, x1, z0, z1 in ((X_L, -PX, Z_F, Z_N), (PX, STAND_X0, Z_F, Z_N), (-PX, PX, PZ0, Z_N), (-PX, PX, Z_F, PZ1)):
        p = np.array([[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1]], float)
        uv = np.stack([(p[:, 0] - X_L) / W, (p[:, 2] - Z_F) / D], -1)
        parts.append(G.Mesh(p, np.tile([0.0, 1.0, 0.0], (4, 1)), uv, np.array([[0, 1, 2], [0, 2, 3]])).oriented())
    return P.merge(parts)


# ---- materials -------------------------------------------------------------------------------------------------------
def add_materials(b: SceneBuilder, As: Assets, Ad: Assets, ctx: BuildContext) -> None:
    pa = png(As, "mosaic_pool", lambda: T.mosaic(11, T.POOL_PALETTE, T.POOL_WEIGHTS)[0])
    pr = png(As, "mosaic_pool_r", lambda: T.mosaic(11, T.POOL_PALETTE, T.POOL_WEIGHTS)[1], gray=True)
    la = png(As, "mosaic_lane", lambda: T.mosaic(12, T.LANE_PALETTE, [0.5, 0.3, 0.2])[0])
    lr = png(As, "mosaic_lane_r", lambda: T.mosaic(12, T.LANE_PALETTE, [0.5, 0.3, 0.2])[1], gray=True)
    wa = png(As, "mosaic_white", lambda: T.mosaic(13, T.WHITE_PALETTE, [0.5, 0.3, 0.2], tile_m=0.125, ppm=768)[0])
    wr = png(As, "mosaic_white_r", lambda: T.mosaic(13, T.WHITE_PALETTE, [0.5, 0.3, 0.2], tile_m=0.125, ppm=768)[1], gray=True)
    b.material("water", {"type": "dielectric", "int_ior": 1.333, "ext_ior": 1.000277,
                         "specular_transmittance": rgb(0.88, 0.96, 0.97)})
    b.material("pool_tile", principled(tex(pa), tex(pr, raw=True), specular=0.5))
    b.material("lane_tile", principled(tex(la), tex(lr, raw=True), specular=0.5))
    b.material("white_tile", principled(tex(wa), tex(wr, raw=True), specular=0.5))
    W, D = STAND_X0 - X_L, Z_N - Z_F
    rect_pool = (-PX - X_L, PX - X_L, PZ1 - Z_F, PZ0 - Z_F)
    da = Ad.root / "textures" / "deck.png"
    dr = Ad.root / "textures" / "deck_r.png"
    if Ad.rebuild or not (da.exists() and dr.exists()):
        alb, rough = T.deck(ctx.seed * 7 + 1, W, D, pool_rect=rect_pool)
        T.save_png8(da, alb)
        T.save_png8(dr, np.repeat(rough[..., None], 3, -1))
    b.material("deck", principled(tex(str(da)), tex(str(dr), raw=True), specular=0.5))
    b.material("coping", principled((0.86, 0.87, 0.86), 0.16, specular=0.5))
    b.material("coping_edge", principled((0.03, 0.07, 0.24), 0.2, specular=0.5))
    b.material("steel", {"type": "roughconductor", "material": "Cr", "alpha": 0.10})
    b.material("block_body", principled((0.86, 0.87, 0.87), 0.3, specular=0.5))
    b.material("block_mat", principled((0.04, 0.04, 0.045), 0.85))
    b.material("white_paint", principled((0.86, 0.86, 0.84), 0.35, specular=0.5))
    b.material("chair_seat", principled((0.66, 0.06, 0.05), 0.45, specular=0.5))
    wp = png(As, "teak", lambda: G.straight_wood(512, 512, seed=21, base=(0.62, 0.42, 0.24), dark=(0.42, 0.26, 0.13))[0])
    b.material("bench_wood", principled(tex(wp, scale=(1.0, 4.0)), 0.45))
    gp = png(As, "glulam", lambda: G.straight_wood(512, 512, seed=22, base=(0.78, 0.60, 0.40), dark=(0.62, 0.45, 0.28))[0])
    b.material("glulam", principled(tex(gp, scale=(0.25, 1.0)), 0.55))
    cp = png(As, "concrete", lambda: G.plaster(1024, 1024, (0.60, 0.59, 0.57), seed=23, strength=0.12))
    b.material("concrete", principled(tex(cp, scale=(1 / 3.0, 1 / 3.0)), 0.85))
    pp = png(As, "plaster", lambda: G.plaster(1024, 1024, (0.84, 0.82, 0.77), seed=24))
    b.material("plaster", principled(tex(pp, scale=(1 / 4.0, 1 / 4.0)), 0.8))
    b.material("ceiling", {"type": "diffuse", "reflectance": rgb(0.80, 0.80, 0.78)})
    b.material("alu", principled((0.13, 0.14, 0.15), 0.4, metallic=0.6))
    b.material("duct", {"type": "roughconductor", "material": "Al", "alpha": 0.28})
    b.material("backing", {"type": "diffuse", "reflectance": rgb(0.02, 0.02, 0.02)})
    b.material("seat_post", principled((0.20, 0.20, 0.21), 0.5, metallic=0.5))
    b.material("door", principled((0.16, 0.19, 0.22), 0.5, specular=0.5))
    b.material("rubber", principled((0.03, 0.03, 0.03), 0.8))
    b.material("cord", principled((0.85, 0.85, 0.82), 0.7))
    gr = png(As, "grass", lambda: T.grass(31))
    b.material("grass", principled(tex(gr, scale=(1 / 8.0, 1 / 8.0)), 0.92))
    b.material("path", principled((0.47, 0.46, 0.43), 0.85))
    b.material("bark", principled((0.12, 0.09, 0.06), 0.9))
    b.material("leaf", {"type": "twosided", "bsdf": principled((0.08, 0.24, 0.05), 0.5)})
    b.material("leaf2", {"type": "twosided", "bsdf": principled((0.14, 0.30, 0.06), 0.5)})
    b.material("hedge", {"type": "twosided", "bsdf": principled((0.04, 0.16, 0.04), 0.55)})
    fac = png(As, "facade", lambda: T.window_grid(41))
    b.material("facade", principled(tex(fac, scale=(1 / 12.0, 1 / 12.0)), 0.5, specular=0.5))
    for i, c in enumerate(FLOAT_COLS):
        b.material(f"float_{i}", principled(c, 0.35, specular=0.5))
    for i, c in enumerate(FLAG_COLS):
        b.material(f"flag_{i}", {"type": "twosided", "bsdf": principled(c, 0.6)})
    for i, c in enumerate(SEAT_COLS):
        b.material(f"seat_{i}", principled(c, 0.35, specular=0.5))
    for i, c in enumerate(TOY_COLS):
        b.material(f"toy_{i}", principled(c, 0.7))


# ---- the pool ---------------------------------------------------------------------------------------------------------
def add_pool(b: SceneBuilder, As: Assets, Ad: Assets, ctx: BuildContext, focus: dict) -> None:
    # tank: floor, four walls, lane lines and end-wall targets, the waterline band
    mesh_once(b, As, "pool_floor", lambda: P.pool_floor(-PX, PX, PZ0, PZ1, Y_DEEP, Y_SHALLOW), np.eye(4), "pool_tile", "floor")
    yb = Y_DEEP - 0.1
    walls = [((-PX, yb, PZ0), (-PX, yb, PZ1), (-PX, 0.0, PZ0), (1, 0, 0)),
             ((PX, yb, PZ1), (PX, yb, PZ0), (PX, 0.0, PZ1), (-1, 0, 0)),
             ((PX, yb, PZ0), (-PX, yb, PZ0), (PX, 0.0, PZ0), (0, 0, -1)),
             ((-PX, Y_SHALLOW - 0.1, PZ1), (PX, Y_SHALLOW - 0.1, PZ1), (-PX, 0.0, PZ1), (0, 0, 1))]
    for i, (p0, p1, p3, n) in enumerate(walls):
        mesh_once(b, As, f"pool_wall_{i}", lambda p0=p0, p1=p1, p3=p3, n=n: P.quad(p0, p1, p3, n), np.eye(4), "pool_tile", "wall")
        off = 0.003 * np.asarray(n, float)
        band = lambda p0=p0, p1=p1, n=n, off=off: P.quad(np.add((p0[0], -0.34, p0[2]), off), np.add((p1[0], -0.34, p1[2]), off),
                                                        np.add((p0[0], -0.06, p0[2]), off), n)
        mesh_once(b, As, f"pool_band_{i}", band, np.eye(4), "lane_tile", "band")

    def floor_y(z):
        return Y_DEEP + (Y_SHALLOW - Y_DEEP) * (z - PZ0) / (PZ1 - PZ0) + 0.004

    def lines() -> G.Mesh:
        parts = []
        for x in LANES:
            parts.append(P.pool_floor(x - 0.125, x + 0.125, -2.0, -23.0, floor_y(-2.0), floor_y(-23.0), step=0.25))
            for zc in (-2.0, -23.0):
                parts.append(P.pool_floor(x - 0.5, x + 0.5, zc + 0.125, zc - 0.125, floor_y(zc + 0.125), floor_y(zc - 0.125), step=0.25))
            for z_wall, sgn, ybot in ((PZ0, -1, Y_DEEP), (PZ1, 1, Y_SHALLOW)):
                zz = z_wall + sgn * 0.004
                n = (0, 0, sgn)
                parts.append(P.quad((x - 0.125, ybot + 0.3, zz), (x + 0.125, ybot + 0.3, zz), (x - 0.125, -0.42, zz), n))
                parts.append(P.quad((x - 0.25, -0.62, zz), (x + 0.25, -0.62, zz), (x - 0.25, -0.42, zz), n))
        return P.merge(parts)
    mesh_once(b, As, "lane_lines", lines, np.eye(4), "lane_tile", "laneline")

    # the water: a seed-dependent ripple field slightly wider than the tank so no seam shows at the walls
    water = Ad.mesh("water", lambda: P.water_surface(-PX - 0.01, PX + 0.01, PZ0 + 0.01, PZ1 - 0.01, WATER_Y, ctx.seed * 13 + 5))
    b.ply(water, np.eye(4), "water", prefix="water")

    # coping with a dark contrast nosing, around the pool edge
    for i, (cx, cz, sx, sz) in enumerate(((-PX - 0.11, (PZ0 + PZ1) / 2, 0.28, PZ0 - PZ1 + 0.56), (PX + 0.11, (PZ0 + PZ1) / 2, 0.28, PZ0 - PZ1 + 0.56),
                                          (0.0, PZ0 + 0.11, 2 * PX + 0.56, 0.28), (0.0, PZ1 - 0.11, 2 * PX + 0.56, 0.28))):
        b.box_mesh(f"coping_{i}", (cx, -0.02, cz), (sx, 0.07, sz), "white_tile")
    for (cx, cz, sx, sz) in ((-PX + 0.01, (PZ0 + PZ1) / 2, 0.06, PZ0 - PZ1), (PX - 0.01, (PZ0 + PZ1) / 2, 0.06, PZ0 - PZ1),
                             (0.0, PZ0 - 0.01, 2 * PX, 0.06), (0.0, PZ1 + 0.01, 2 * PX, 0.06)):
        b.cube((cx, 0.0155, cz), (sx, 0.002, sz), "coping_edge", prefix="nosing")

    # lane ropes: per-rope colour scheme (end colour, two alternating colours)
    rng = np.random.default_rng(ctx.seed * 3 + 1)
    schemes = [(0, int(a), int(b_)) for a, b_ in (rng.choice([1, 2, 3, 4, 5], 2, replace=False) for _ in ROPES)]
    for i, x in enumerate(ROPES):
        n = int((PZ0 - PZ1 - 0.1) / P.FLOAT_LEN)
        cols = P.lane_colours(n, schemes[i])
        parts = Ad.meshes(f"rope_{i}", lambda x=x, cols=cols, i=i: {
            f"c{c}": m for c, m in P.lane_rope(x, PZ0 - 0.05, PZ1 + 0.05, WATER_Y + 0.004, cols, np.random.default_rng(ctx.seed * 31 + i)).items()})
        for key, path in parts.items():
            b.ply(path, np.eye(4), f"float_{int(key[1:])}", prefix="float")
    focus["float_near"] = [-2.0, WATER_Y + 0.02, -3.5]
    focus["float_far"] = [2.0, WATER_Y + 0.02, -19.0]


def add_deck_furniture(b: SceneBuilder, As: Assets, Ad: Assets, ctx: BuildContext, focus: dict) -> None:
    rng = np.random.default_rng(ctx.seed * 5 + 2)
    # starting blocks at the near (deep) end, with lane numbers front and back
    blk = As.meshes("block", P.starting_block)
    for k, x in enumerate(LANES):
        M = G.translate((x, 0.015, PZ0))
        b.part_set(blk, M, {"body": "block_body", "mat": "block_mat", "steel": "steel", "handle": "steel"})
        b.cube((x, 0.53, -0.20), (0.48, 0.17, 0.02), "block_body", prefix="apron")
        num = png(As, f"lane_{k + 1}", lambda k=k: flip(T.lane_number(k + 1)))
        textured_rect(b, f"lanenum_{k}", num, (x, 0.53, -0.212), (-1, 0, 0), (0, 1, 0), 0.15, 0.15)
        b.cube((x, 0.30, 0.345), (0.34, 0.24, 0.02), "block_body", prefix="backplate")
        textured_rect(b, f"lanenumb_{k}", num, (x, 0.30, 0.356), (1, 0, 0), (0, 1, 0), 0.16, 0.16)
    focus["block_4"] = [LANES[3], 0.70, 0.06]

    # ladders on both side walls near each end
    lad = As.meshes("ladder", P.ladder)
    for side, z in LADDERS:
        M = G.compose(G.translate((side * PX, 0.015, z)), G.rotate((0, 1, 0), 90.0 * side))
        b.part_set(lad, M, {"steel": "steel"})
        x0 = side * PX
    focus["ladder_near"] = [-PX - 0.20, 0.84, -3.5 + 0.24]

    # backstroke flags across the pool, 5 m from each end
    for k, z in enumerate(FLAG_Z):
        for sx in (-1, 1):
            mesh_once(b, As, "flag_pole", lambda: P.merge([P.tube((0, 0, 0), (0, 2.05, 0), 0.03, 12),
                                                            G.lathe([(0, 0), (0.14, 0), (0.14, 0.02), (0, 0.02)], 16)]),
                      G.translate((sx * FLAG_POLE_X, 0.0, z)), "white_paint", "pole")
        pal = [int(c) for c in rng.choice(len(FLAG_COLS), 3, replace=False)]
        parts = Ad.meshes(f"flags_{k}", lambda z=z, pal=pal, k=k: {
            (key if key == "cord" else f"c{key}"): m for key, m in P.flag_line(-FLAG_POLE_X + 0.05, FLAG_POLE_X - 0.05, z, 1.97, 0.12, pal,
                                                                              np.random.default_rng(ctx.seed * 17 + k)).items()})
        for key, path in parts.items():
            b.ply(path, np.eye(4), "cord" if key == "cord" else f"flag_{int(key[1:])}", prefix="flags")
    focus["flags_near"] = [0.0, 1.80, FLAG_Z[0]]
    focus["flags_far"] = [0.0, 1.80, FLAG_Z[1]]

    # lifeguard chair facing the pool
    ch = As.meshes("chair", P.lifeguard_chair)
    b.part_set(ch, G.compose(G.translate((CHAIR[0], 0.0, CHAIR[1])), G.rotate((0, 1, 0), 90.0)), {"frame": "white_paint", "seat": "chair_seat"})
    focus["lifeguard_chair"] = [CHAIR[0], 1.6, CHAIR[1]]

    # benches: a seeded subset of the candidate spots
    ben = As.meshes("bench", lambda: P.bench(2.0))
    spots = [((-8.9, -6.5), 90.0), ((-8.9, -18.5), 90.0), ((-3.5, 5.4), 180.0), ((3.5, 5.4), 180.0), ((-8.9, -25.5), 90.0)]
    keep = rng.permutation(len(spots))[:3]
    for i in sorted(keep):
        (x, z), yaw = spots[i]
        x += rng.uniform(-0.05, 0.05)
        z += rng.uniform(-0.3, 0.3)
        b.part_set(ben, G.compose(G.translate((x, 0.0, z)), G.rotate((0, 1, 0), yaw)), {"wood": "bench_wood", "steel": "steel"})
        hx, hz = (0.35, 1.15) if yaw == 90.0 else (1.15, 0.35)
        focus.setdefault("bench", [x, 0.45, z])

    # kickboard rack and a bin of noodles in the near-left corner
    rk = As.mesh("rack", P.rack)
    b.ply(rk, G.compose(G.translate((RACK[0], 0.0, RACK[1])), G.rotate((0, 1, 0), 90.0)), "steel", prefix="rack")

    def toys():
        r = np.random.default_rng(ctx.seed * 19 + 3)
        out: dict[str, list] = {}
        kb = P.kickboard()
        for y in (0.15, 0.6, 1.05):
            for zc in (-0.3, 0.3):
                n = int(r.integers(5, 11))
                c = int(r.integers(len(TOY_COLS)))
                for j in range(n):
                    if r.random() < 0.25:
                        c = int(r.integers(len(TOY_COLS)))
                    m = G.compose(G.translate((r.normal(0, 0.01), y + 0.02 + j * 0.033, zc + r.normal(0, 0.01))),
                                  G.rotate((0, 1, 0), 90.0 + r.normal(0, 4.0)))
                    out.setdefault(f"t{c}", []).append(kb.transformed(m))
        for j in range(int(r.integers(7, 13))):
            c = int(r.integers(len(TOY_COLS)))
            a = r.uniform(0, 2 * np.pi)
            m = G.compose(G.translate((BIN[0] - RACK[0] + 0.12 * np.cos(a), 0.05, BIN[1] - RACK[1] + 0.12 * np.sin(a))),
                          G.rotate((np.cos(a + 1.5), 0, np.sin(a + 1.5)), r.uniform(4, 16)))
            out.setdefault(f"t{c}", []).append(P.noodle(r).transformed(m))
        return {k: P.merge(v) for k, v in out.items()}
    parts = Ad.meshes("toys", toys)
    for key, path in parts.items():
        b.ply(path, G.translate((RACK[0], 0.0, RACK[1])), f"toy_{int(key[1:])}", prefix="toy")
    mesh_once(b, As, "bin", lambda: G.lathe([(0, 0), (0.30, 0), (0.33, 0.6), (0.31, 0.6), (0.28, 0.02), (0, 0.02)], 32),
              G.translate((BIN[0], 0.0, BIN[1])), "toy_1", "bin")
    focus["kickboards"] = [RACK[0], 0.75, RACK[1]]

    # depth marks painted on the deck beside the coping, readable from the deck
    marks = [(-1.2, "2.0 M"), (-12.5, "1.6 M"), (-23.8, "1.2 M")]
    for k, (z, text) in enumerate(marks):
        p = png(As, f"mark_{k}", lambda text=text: flip(T.depth_mark(text)))
        textured_rect(b, f"markL_{k}", p, (-PX - 0.62, 0.003, z), (0, 0, 1), (1, 0, 0), 0.6, 0.2)
        textured_rect(b, f"markR_{k}", p, (PX + 0.62, 0.003, z), (0, 0, -1), (-1, 0, 0), 0.6, 0.2)
    nd = png(As, "mark_nodive", lambda: flip(T.sign(["NO DIVING"], (0.84, 0.86, 0.85), (0.6, 0.04, 0.04), 900, 200)))
    for x in (-3.0, 3.0):
        textured_rect(b, f"nodive_deck_{int(x)}", nd, (x, 0.003, PZ1 - 0.75), (-1, 0, 0), (0, 0, 1), 1.35, 0.3)


# ---- the hall ----------------------------------------------------------------------------------------------------------
def add_hall(b: SceneBuilder, As: Assets, Ad: Assets, ctx: BuildContext, focus: dict) -> None:
    rng = np.random.default_rng(ctx.seed * 11 + 4)
    b.ply(As.mesh("deck_top", deck_top), np.eye(4), "deck", prefix="deck")

    # walls: near and far ends, the right wall behind the stands
    t = 0.3
    for z, zc in ((Z_N, Z_N + t / 2), (Z_F, Z_F - t / 2)):
        b.box_mesh(f"endwall_{int(z)}", ((X_L + X_R) / 2, CEIL / 2, zc), (X_R - X_L + 2 * t, CEIL, t), "plaster")
        b.box_mesh(f"dado_{int(z)}", ((X_L + STAND_X0) / 2, 1.1, z - np.sign(z) * 0.01), (STAND_X0 - X_L, 2.2, 0.02), "white_tile")
    b.box_mesh("rightwall", (X_R + t / 2, CEIL / 2, (Z_N + Z_F) / 2), (t, CEIL, Z_N - Z_F), "plaster")
    b.box_mesh("ceiling", ((X_L + X_R) / 2, CEIL + 0.15, (Z_N + Z_F) / 2), (X_R - X_L + 2 * t, 0.3, Z_N - Z_F + 2 * t), "ceiling")

    # left wall: dado, head, pilasters at the beams, and open bays with mullions and transoms
    xw = X_L - t / 2
    b.box_mesh("lw_dado", (xw, 0.3, (Z_N + Z_F) / 2), (t, 0.6, Z_N - Z_F), "white_tile")
    b.box_mesh("lw_head", (xw, 7.2, (Z_N + Z_F) / 2), (t, 1.6, Z_N - Z_F), "plaster")
    for z in Z_BEAMS:
        b.cube((xw, 3.5, z), (t + 0.02, 5.8, 0.4), "plaster", prefix="pilaster")
    b.cube((xw, 3.5, (Z_N + 4.2) / 2), (t, 5.8, Z_N - 4.2), "plaster", prefix="lw_end")
    b.cube((xw, 3.5, (Z_F - 28.2) / 2), (t, 5.8, abs(Z_F + 28.2)), "plaster", prefix="lw_end")
    for z0, z1 in zip(Z_BEAMS[:-1], Z_BEAMS[1:]):
        za, zb = z0 - 0.2, z1 + 0.2
        zc, wz = (za + zb) / 2, za - zb
        for zz in (zc - wz / 6, zc + wz / 6):
            b.cube((xw, 3.5, zz), (0.12, 5.8, 0.07), "alu", prefix="mullion")
        for y in (0.63, 2.4, 4.4, 6.37):
            b.cube((xw, y, zc), (0.12, 0.07, wz), "alu", prefix="transom")
        for zz in (za - 0.03, zb + 0.03):
            b.cube((xw, 3.5, zz), (0.12, 5.8, 0.06), "alu", prefix="jamb")
    b.cube((X_L + 0.12, 0.62, (Z_N + Z_F) / 2), (0.25, 0.04, Z_N - Z_F), "white_tile", prefix="sill")
    focus["window_mullion"] = [xw, 3.0, (Z_BEAMS[3] + Z_BEAMS[4]) / 2 + (Z_BEAMS[3] - Z_BEAMS[4] - 0.4) / 6]

    # roof: glulam beams, pendants, a duct over the stands
    for z in Z_BEAMS:
        b.box_mesh(f"beam_{int(z)}", ((X_L + X_R) / 2, 7.55, z), (X_R - X_L, 0.9, 0.24), "glulam")
    housing = As.mesh("pendant", P.pendant)
    cable = As.mesh("pendant_cable", lambda: P.tube((0, 0, 0), (0, 1, 0), 0.006, 5))
    r = 0.40
    area = np.pi * r * r
    L = tuple(PENDANT_L * c for c in PENDANT_RGB)
    for z in PENDANT_Z:
        for x in PENDANT_X + [10.6]:
            y = 6.4 if x != 10.6 else 5.6
            b.ply(housing, G.translate((x, y, z)), "white_paint", prefix="pendant")
            b.ply(cable, G.compose(G.translate((x, y + 0.34, z)), G.scale((1.0, CEIL - y - 0.34, 1.0))), "rubber", prefix="cable")
            m = G.compose(G.translate((x, y + 0.012, z)), G.rotate((1, 0, 0), 90.0), G.scale((r, r, 1.0)))
            b._shape("lamp", {"type": "disk", "to_world": xf(m)}, "backing", emission=L, power=luminance(L) * area)
            ru = 0.32
            Lu = tuple(UPLIGHT_L * c for c in PENDANT_RGB)
            mu = G.compose(G.translate((x, y + 0.36, z)), G.rotate((1, 0, 0), -90.0), G.scale((ru, ru, 1.0)))
            # light samples follow power, but only the ceiling sees an uplight's face: weight it down so walls,
            # deck and stands spend their samples on the downlights and the sky
            b._shape("uplight", {"type": "disk", "to_world": xf(mu)}, "backing", emission=Lu,
                     power=UPLIGHT_SAMPLE_SHARE * luminance(Lu) * np.pi * ru * ru)
    focus["pendant"] = [0.0, 6.4, PENDANT_Z[3]]
    mesh_once(b, As, "duct", lambda: P.merge([G.sweep([(12.0, 6.6, Z_N - 0.2), (12.0, 6.6, Z_F + 0.2)], 0.4, 32, caps=False)] +
                                              [G.lathe([(0.40, -0.04), (0.43, -0.04), (0.43, 0.04), (0.40, 0.04)], 32).transformed(
                                                  G.compose(G.translate((12.0, 6.6, z)), G.rotate((1, 0, 0), 90.0))) for z in np.arange(4.0, -29.0, -3.0)]),
              np.eye(4), "duct", "duct")
    focus["duct"] = [12.0 - 0.4, 6.6, -12.0]

    # spectator stands: concrete tiers, moulded seats in seeded colour blocks, a front rail
    for k in range(TIERS):
        x0 = STAND_X0 + k
        top = TIER_H * (k + 1)
        b.box_mesh(f"tier_{k}", ((x0 + X_R) / 2, top / 2, (STAND_Z[0] + STAND_Z[1]) / 2), (X_R - x0, top, STAND_Z[0] - STAND_Z[1]), "concrete")

    def seats():
        r2 = np.random.default_rng(ctx.seed * 23 + 6)
        unit = P.stadium_seat()
        out: dict[str, list] = {"post": []}
        pal = [int(c) for c in r2.choice(len(SEAT_COLS), 2, replace=False)]
        block = int(r2.integers(6, 12))
        zs = np.arange(STAND_Z[0] - 0.75, STAND_Z[1] + 0.4, -0.52)
        for k in range(TIERS):
            x = STAND_X0 + k + 0.55
            for j, z in enumerate(zs):
                c = pal[(j // block + k // 2) % 2]
                if r2.random() < 0.04:
                    c = int(r2.integers(len(SEAT_COLS)))
                m = G.compose(G.translate((x, TIER_H * (k + 1), z)), G.rotate((0, 1, 0), -90.0))
                parts = unit
                out.setdefault(f"c{c}", []).append(parts["shell"].transformed(m))
                out["post"].append(parts["post"].transformed(m))
        return {k: P.merge(v) for k, v in out.items()}
    for key, path in Ad.meshes("seats", seats).items():
        b.ply(path, np.eye(4), "seat_post" if key == "post" else f"seat_{int(key[1:])}", prefix="seat")
    focus["stands_seat"] = [STAND_X0 + 1 + 0.55, TIER_H * 2 + 0.45, -10.2]

    def rail():
        parts = []
        zs = np.arange(STAND_Z[0], STAND_Z[1] - 0.01, -2.0)
        for z in zs:
            parts.append(P.tube((STAND_X0 - 0.05, 0.0, z), (STAND_X0 - 0.05, 1.05, z), 0.024, 8))
        for y in (0.55, 1.05):
            parts.append(P.tube((STAND_X0 - 0.05, y, STAND_Z[0]), (STAND_X0 - 0.05, y, STAND_Z[1]), 0.02, 8))
        return P.merge(parts)
    mesh_once(b, As, "stand_rail", rail, np.eye(4), "steel", "rail")

    # banners on the right wall, above the stands
    words = ["SWIM", "CLUB", "AQUA", "TEAM", "RELAY", "MASTERS", "DOLPHINS", "SHARKS"]
    for i, z in enumerate(PENDANT_Z[1::2]):
        w = words[int(rng.integers(len(words)))]
        bp = png(As, f"banner_{w}_{i % 3}", lambda w=w, i=i: flip(T.banner(50 + i % 3 * 7 + len(w), w)))
        textured_rect(b, f"banner_{i}", bp, (X_R - 0.02, 4.9, z), (0, 0, 1), (0, 1, 0), 0.9, 2.25, roughness=0.8)


def add_far_wall(b: SceneBuilder, As: Assets, Ad: Assets, ctx: BuildContext, focus: dict) -> None:
    rng = np.random.default_rng(ctx.seed * 29 + 7)
    zf = Z_F + 0.02
    # results board (emissive), in a frame
    alb, emi = T.scoreboard(ctx.seed * 37 + 9)
    ep = exr(Ad, "score_emi", lambda: flip(emi * SCORE_LEVEL / 3.0))
    w, h = 6.0, 2.0
    b.cube((0.0, 4.6, Z_F + 0.06), (w + 0.25, h + 0.25, 0.12), "door", prefix="scoreframe")
    b.rect(rect_m((0.0, 4.6, Z_F + 0.125), (1, 0, 0), (0, 1, 0), w, h), "backing",
           emission=tex(ep, raw=True), prefix="score", power=0.35 * SCORE_LEVEL * w * h)
    focus["scoreboard"] = [0.0, 4.6, Z_F + 0.13]
    # pace clock
    cp = png(As, "pace_clock", lambda: flip(T.pace_clock()))
    textured_rect(b, "clockface", cp, (-5.0, 2.7, zf + 0.04), (1, 0, 0), (0, 1, 0), 1.2, 1.2, roughness=0.3)
    mesh_once(b, As, "clock_bezel", lambda: G.lathe([(0.55, 0.0), (0.64, 0.0), (0.64, 0.08), (0.56, 0.08)], 48),
              G.compose(G.translate((-5.0, 2.7, zf)), G.rotate((1, 0, 0), 90.0)), "white_paint", "bezel")
    b.cube((-5.0, 2.7, zf + 0.01), (1.32, 1.32, 0.02), "backing", prefix="clockback")
    focus["pace_clock"] = [-5.0, 2.7, zf + 0.05]
    # doors and wall signs (the seed picks which sign hangs where)
    for x in (-8.2, 7.6):
        b.cube((x, 1.05, zf + 0.03), (1.0, 2.1, 0.06), "door", prefix="door")
        b.cube((x, 0.15, zf + 0.065), (0.96, 0.25, 0.01), "steel", prefix="kickplate")
        b.cube((x + 0.38, 1.05, zf + 0.09), (0.03, 0.3, 0.05), "steel", prefix="handle")
    signs = [(["CHANGING", "ROOMS"], (0.05, 0.25, 0.55), (0.92, 0.92, 0.9), ""),
             (["NO DIVING"], (0.92, 0.92, 0.9), (0.06, 0.06, 0.08), "nodive"),
             (["SHALLOW END", "1.2 M"], (0.05, 0.25, 0.55), (0.92, 0.92, 0.9), ""),
             (["LIFEGUARD", "ON DUTY"], (0.75, 0.08, 0.06), (0.95, 0.95, 0.92), ""),
             (["SHOWER BEFORE", "SWIMMING"], (0.92, 0.92, 0.9), (0.05, 0.20, 0.45), ""),
             (["NO RUNNING"], (0.92, 0.92, 0.9), (0.06, 0.06, 0.08), "")]
    order = rng.permutation(len(signs))
    slots = [((-8.2, 2.55), 1.0, 0.5), ((-2.3, 2.6), 1.5, 0.75), ((2.6, 2.6), 1.5, 0.75), ((7.6, 2.55), 1.0, 0.5)]
    for (pos, w_, h_), si in zip(slots, order):
        lines, bg, fg, icon = signs[int(si)]
        sp = png(As, f"sign_{int(si)}", lambda lines=lines, bg=bg, fg=fg, icon=icon: flip(T.sign(lines, bg, fg, 900, 450, icon, border=(0.1, 0.1, 0.1))))
        textured_rect(b, f"fsign_{int(si)}", sp, (pos[0], pos[1], zf + 0.02), (1, 0, 0), (0, 1, 0), w_, h_)
    focus["far_sign"] = [-2.3, 2.6, zf + 0.03]
    # near wall: doors and two signs
    zn = Z_N - 0.02
    for x in (-6.5, 6.5):
        b.cube((x, 1.05, zn - 0.03), (1.0, 2.1, 0.06), "door", prefix="door")
    for (x, si) in ((-3.8, 4), (3.8, 5)):
        lines, bg, fg, icon = signs[si]
        sp = png(As, f"sign_{si}", lambda lines=lines, bg=bg, fg=fg, icon=icon: flip(T.sign(lines, bg, fg, 900, 450, icon, border=(0.1, 0.1, 0.1))))
        textured_rect(b, f"nsign_{si}", sp, (x, 2.6, zn - 0.02), (-1, 0, 0), (0, 1, 0), 1.4, 0.7)


def add_outside(b: SceneBuilder, As: Assets, ctx: BuildContext, focus: dict) -> None:
    rng = np.random.default_rng(ctx.seed * 41 + 8)
    # `sunsky` is black below the horizon and render.py clips camera rays at 1000 m, so ground that only ends
    # at the horizon would leave a black strip under it. A ring of distant woodland at 900 m closes the view:
    # its canopy top (2 to 16 m) stands above the horizon for every camera height in camera_box.
    b.box_mesh("lawn", (-1510.0, -0.27, -12.0), (3000.0, 0.5, 6000.0), "grass")
    b.material("treeline", {"type": "twosided", "bsdf": principled((0.07, 0.11, 0.08), 0.9)})
    b.ply(As.mesh("treeline", lambda: P.treeline(900.0, 81)), np.eye(4), "treeline", prefix="treeline")
    b.box_mesh("path", (-11.2, -0.01, (Z_N + Z_F) / 2), (2.6, 0.04, Z_N - Z_F + 20.0), "path")
    for i in range(3):
        hp = As.mesh(f"hedge_{i}", lambda i=i: P.hedge(12.0, 60 + i))
        b.ply(hp, G.compose(G.translate((-14.0, 0.0, Z_N - 6.0 - 12.0 * i)), G.rotate((0, 1, 0), 90.0)), "hedge", prefix="hedge")
    trees = [As.meshes(f"tree_{i}", lambda i=i: P.tree(70 + i, 6.5 + i, 2.4 + 0.4 * i)) for i in range(3)]
    spots = []
    for j in range(7):
        x, z = rng.uniform(-36.0, -17.0), rng.uniform(-38.0, 12.0)
        spots.append((x, z))
        t = trees[j % 3]
        b.part_set(t, G.compose(G.translate((x, 0.0, z)), G.rotate((0, 1, 0), rng.uniform(0, 360)), G.scale(rng.uniform(0.85, 1.2))),
                   {"trunk": "bark", "leaf": "leaf" if j % 2 else "leaf2"})
    near = min(spots, key=lambda s: abs(s[1] + 12.0) + 0.3 * abs(s[0]))
    focus["tree_outside"] = [near[0], 4.5, near[1]]
    b.box_mesh("far_building", (-78.0, 11.0, -14.0), (24.0, 22.0, 70.0), "facade")


# ---- assembly ----------------------------------------------------------------------------------------------------------------
def build(ctx: BuildContext) -> SceneBundle:
    As, Ad = Assets(ctx.shared_dir, ctx.rebuild), Assets(ctx.assets_dir, ctx.rebuild)
    b = SceneBuilder(As)
    focus: dict[str, list] = {}
    add_materials(b, As, Ad, ctx)
    add_pool(b, As, Ad, ctx, focus)
    add_deck_furniture(b, As, Ad, ctx, focus)
    add_hall(b, As, Ad, ctx, focus)
    add_far_wall(b, As, Ad, ctx, focus)
    add_outside(b, As, ctx, focus)
    b.d["sky"] = env_kit.sky(ctx.env, sun_direction=SUN_DIR, scale=SKY_SCALE[ctx.env],
                             sampling_weight=max(SKY_SHARE * b.lamp_weight, 1e-3))
    focus["coping_edge"] = [-PX - 0.15, 0.015, -12.5]
    return SceneBundle(b.d, focus, {"sun_direction": list(SUN_DIR), "sky_scale": SKY_SCALE[ctx.env],
                                    "water_ior": 1.333, "water_level_m": WATER_Y})


def exclude_boxes() -> tuple:
    """Static carve-outs (the seed only moves benches and toys inside the boxes below)."""
    boxes = [((-5.4, 0.0, -0.35), (5.4, 0.95, 0.75)),                                      # starting blocks
             ((CHAIR[0] - 0.75, 0.0, CHAIR[1] - 0.75), (CHAIR[0] + 0.75, 2.5, CHAIR[1] + 0.75)),
             ((X_L, 0.0, BIN[1] - 0.6), (RACK[0] + 0.5, 1.6, RACK[1] + 0.8)),               # rack, noodle bin
             ((-9.3, 0.0, -7.8), (-8.4, 1.0, -5.2)), ((-9.3, 0.0, -19.8), (-8.4, 1.0, -17.2)),
             ((-9.3, 0.0, -26.8), (-8.4, 1.0, -24.2)), ((-4.8, 0.0, 4.9), (-2.2, 1.0, 5.9)),
             ((2.2, 0.0, 4.9), (4.8, 1.0, 5.9))]                                            # bench spots
    for side, z in LADDERS:
        x0 = side * PX
        boxes.append(((min(x0, x0 + side * 0.6) - 0.1, -0.2, z - 0.45), (max(x0, x0 + side * 0.6) + 0.1, 1.15, z + 0.45)))
    for z in FLAG_Z:
        for sx in (-1, 1):
            boxes.append(((sx * FLAG_POLE_X - 0.25, 0.0, z - 0.25), (sx * FLAG_POLE_X + 0.25, 2.2, z + 0.25)))
    for k in range(TIERS):
        boxes.append(((STAND_X0 + k - 0.05, 0.0, STAND_Z[1] - 0.1), (X_R, TIER_H * (k + 1) + 0.95, STAND_Z[0] + 0.1)))
    return tuple(boxes)


SCENE = SceneDef(
    id="pool", group="artificial", owner="rui",
    description="Indoor 25 m swimming pool hall: a refracting rippled water surface over blue mosaic tiles, lane ropes, "
                "starting blocks, backstroke flags, spectator stands, glulam roof with pendants, a glazed wall onto a lawn.",
    build=build,
    views={
        # Low behind the starting blocks, along the lanes to the far wall: water fills the lower half of the frame.
        "lanes": View((-2.45, 0.42, 1.0), (-1.4, 0.05, -25.0), focus="float_near"),
        # Close on a ladder and the pool edge: the rails bend where they enter the water.
        "ladder": View((-7.0, 0.95, -6.9), (-5.75, -0.45, -3.3), focus="ladder_near"),
        # High in the stands, across the pool toward the glazed wall and the far end.
        "stands": View((12.3, 3.9, 4.0), (-7.0, -0.3, -21.0), focus="lifeguard_chair"),
        # Over the water looking back at the starting blocks.
        "blocks": View((1.8, 0.55, -5.0), (-0.6, 0.35, 1.0), focus="block_4"),
    },
    default_view="lanes",
    camera_box=((-9.1, 0.3, -29.5), (12.7, 5.0, 5.6)),
    target_box=((-12.0, -2.1, -30.0), (13.0, 7.8, 6.0)),
    exclude_boxes=exclude_boxes(),
    envs=("clear", "overcast"),
    tags=("indoor", "day", "water", "specular", "architecture", "artificial_light"),
    default_seed=4, asset_version=ASSET_VERSION, max_depth=10, rr_depth=6, spp_hint=4096,
)
