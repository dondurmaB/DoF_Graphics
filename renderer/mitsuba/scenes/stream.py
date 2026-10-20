"""A woodland stream seen upstream: a wet boulder in the water, a fern bank, a log across, a riffle, trees over it.

Natural scene (owner beiba). Built for depth of field: subjects at ~3.6 m (the boulder), ~6 m (the fern bank),
~13 m (the fallen log) and ~20 m (a riffle of stones) from the hero cameras, and small bright highlights at every
depth: sun glints on the rippled water, wet clear-coated stones in the stream, the bright gap in the canopy
over the water, backlit leaves on both banks.

Layout, meters, y up, the stream runs along z and the hero cameras look upstream toward -z:
  water        a flat surface at y = -0.15 (`_stream_props.WATER_Y`) over a bed 0.45 m deeper, meandering
               about x = 0 (two bends, so it turns out of sight upstream); rippled like riffles (rms slope 0.12)
  banks        rise over 0.7 m from the water's edge, then valley sides climb gently to ~9 m at x = +-60;
               the terrain is cut along two contours (`_stream_props.terrain_parts`): mud bed from 8 cm under
               the water, a wet-mud shore up to 4 cm above it, litter or snow above. Cutting exactly along
               contours keeps the material edges smooth (a per-face split left saw teeth where snow met mud).
  foreground   a wet boulder in the stream at z = 1.5, a fern bank on the left bank at z = 3
  midground    a fallen log across the stream at z ~ -8, a riffle of stones at z ~ -15
  background   trees on both banks to ~55 m, a sparser belt beyond, then far woodland at 100 m (the same
               geometric skyline as forest_floor)
Light is only `env_kit`'s sun and sky. The sun is ahead of the hero cameras and high (36 degrees), so it
glints off the water facing them but stays out of their frames.

Environments: clear; overcast; snow (low winter sun, snow on the banks, trees bare, ferns and grass gone,
the stream open). Seed (`ctx.seed`) moves every tree away from the hero area, the bank plants, and the
stones in and beside the stream; the hero subjects keep their places so the views hold.

Cost: scattered plants and stones are merged per material (`_nature_kit.Batch`), as in forest_floor.
"""

from __future__ import annotations

import math

import numpy as np

import env_kit
import procedural as G
import web_assets as W
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, bitmap, principled
from scenes import _nature_kit as N
from scenes import _stream_props as P

ASSET_VERSION = 3                     # 2: opaque stones; 3: terrain cut along contours (smooth shoreline)
EYE = np.eye(4)
SUN = {"clear": env_kit.default_sun_direction(36.0, 160.0), "snow": env_kit.default_sun_direction(24.0, 165.0)}
# Calibrated on check_scene previews (6cc21fc): clear 14 gave medians 0.021-0.033, just over the 0.02 floor;
# snow 12 gave the bank view 0.886, over the 0.5 ceiling. Overcast 40 gave 0.047-0.066.
SKY_SCALE = {"clear": 26.0, "overcast": 40.0, "snow": 5.5}
LITTER_DARKEN = 0.55

TERRAIN = P.StreamTerrain(-60.0, 60.0, -90.0, 40.0, spacing=0.25,
                          waves=[(0.25, 23.0, 20.0, 0.3), (0.10, 9.0, 110.0, 1.1), (0.03, 3.1, 60.0, 2.0)])
WATER_Z = (-85.0, 35.0)


def cx(z) -> float:
    return float(P.centre_x(z))


def hw(z) -> float:
    return float(P.half_width(z))


ROCK = (cx(1.5) - 0.25, 1.5)                                   # wet boulder in the water
FERN_BANK = (cx(3.0) - hw(3.0) - 1.3, 3.0)                     # left bank
LOG = ((cx(-8.0) - hw(-8.0) - 1.6, -9.0), (cx(-7.0) + hw(-7.0) + 1.6, -7.0), 0.24)
RIFFLE_Z = (-17.0, -13.0)
NEAR_TREES = [(cx(7.0) + hw(7.0) + 3.2, 7.0), (cx(-1.0) - hw(-1.0) - 3.0, -1.0), (cx(-4.0) + hw(-4.0) + 2.8, -4.0),
              (cx(4.5) - hw(4.5) - 3.6, 4.5)]
HERO_AREA = ((-7.0, -11.0), (7.0, 11.0))                       # random trees stay out of this
BACKDROP = {"centre": (0.0, -25.0), "radius": 100.0, "y0": -6.0, "y_tex": 22.0, "tile": 30.0}
SKYLINE = N.Skyline(71, BACKDROP["radius"], base=14.0, spread=4.0)


def rng_for(ctx: BuildContext, k: int) -> np.random.Generator:
    return np.random.default_rng([ctx.seed, 7121, k])


def in_water(x, z, margin=0.0) -> bool:
    return abs(x - cx(z)) < hw(z) + margin


def stone(bsdf: dict, wet: bool = False) -> dict:
    """The rock scan as an opaque stone. web_assets classes rock_moss_set_01 as foliage (its name contains
    "moss", one of web_assets.FOLIAGE_WORDS) and converts it to a thin leaf BSDF (principledthin, 35% diffuse
    transmission): light would pass through the stones, and principledthin has no clear coat for the wet look.
    This keeps the scan's albedo, roughness and normal maps on an opaque principled BSDF; wet stones (in and at
    the edge of the stream) add a glossy clear coat. Scene-local: web_assets itself is left alone."""
    nm = bsdf if bsdf["type"] == "normalmap" else None
    inner = nm["bsdf"] if nm else bsdf
    new = {"type": "principled", "base_color": inner["base_color"], "roughness": inner["roughness"], "specular": 0.5}
    if wet:
        new.update(clearcoat=0.8, clearcoat_gloss=0.92)
    return {"type": "normalmap", "normalmap": nm["normalmap"], "bsdf": new} if nm else new


# --------------------------------------------------------------------------
# Materials and ground
# --------------------------------------------------------------------------
def add_materials(b: SceneBuilder, L: N.Library, ctx: BuildContext) -> None:
    snow = ctx.env == "snow"
    darken = lambda aid, f: W.surface(aid, tint=(f,) * 3)
    b.material("bank", env_kit.ground("snow") if snow else darken("brown_mud_leaves_01", LITTER_DARKEN))
    b.material("bed", darken("brown_mud_02", 0.7))
    shore = darken("brown_mud_02", 0.5)                                    # wet mud at the water's edge, every env
    (shore["bsdf"] if shore["type"] == "normalmap" else shore).update(clearcoat=0.6, clearcoat_gloss=0.9)
    b.material("shore", shore)
    b.material("bark", W.surface("bark_brown_02"))
    # Rough, not smooth: a smooth dielectric reflects the sun as caustics a path tracer finds only by chance;
    # alpha 0.06 keeps the glints small and bright while next-event estimation can still reach them.
    b.material("water", {"type": "roughdielectric", "int_ior": 1.33, "ext_ior": 1.0, "alpha": 0.06})
    alb = b.assets.texture("backdrop_albedo_v2", lambda: N.backdrop_albedo(71, BACKDROP["tile"], BACKDROP["y0"],
                                                                            BACKDROP["y_tex"]))
    b.material("backdrop", {"type": "twosided", "bsdf": principled(bitmap(alb), 0.9)})
    leaf = (0.035, 0.035, 0.03) if snow else (0.010, 0.018, 0.006)       # bare winter twigs vs leaf clusters
    b.material("backdrop_leaf", {"type": "twosided", "bsdf": principled(leaf, 0.85)})
    L.material_set("rock_moss_set_01", "_wet", lambda bsdf: stone(bsdf, wet=True))
    L.material_set("rock_moss_set_01", "_dry", lambda bsdf: stone(bsdf, wet=False))


def add_ground(b: SceneBuilder) -> None:
    parts = b.assets.meshes(f"stream_terrain_v{ASSET_VERSION}", lambda: P.terrain_parts(TERRAIN))
    b.ply(parts["bank"], EYE, "bank", prefix="ground")
    b.ply(parts["shore"], EYE, "shore", prefix="shore")
    b.ply(parts["bed"], EYE, "bed", prefix="bed")
    b.ply(b.assets.mesh(f"stream_water_v{ASSET_VERSION}", lambda: P.water(*WATER_Z, seed=29)), EYE, "water", prefix="water")
    c0, c1 = BACKDROP["centre"]
    b.ply(b.assets.mesh("backdrop_band", lambda: N.backdrop_band(c0, c1, SKYLINE, BACKDROP["y0"], BACKDROP["y_tex"],
                                                                 BACKDROP["tile"])), EYE, "backdrop", prefix="backdrop")
    b.ply(b.assets.mesh("backdrop_fringe", lambda: N.backdrop_fringe(np.random.default_rng(72), c0, c1, SKYLINE)), EYE,
          "backdrop_leaf", prefix="backdrop_leaf")


# --------------------------------------------------------------------------
# Trees
# --------------------------------------------------------------------------
def tree(L: N.Library, rng, x, z, snow, kind=None) -> None:
    kind = kind or ("tree_small_02" if rng.random() < 0.75 else "island_tree_02")
    h = rng.uniform(9.0, 13.0) if kind == "tree_small_02" else rng.uniform(5.5, 7.5)
    L.whole(kind, TERRAIN.at(x, z, -0.15), yaw=float(rng.uniform(0, 360)), height=h,
            drop_parts=("_leaves",) if snow else (), prefix=f"web_{kind}_bare" if snow else None)


def add_trees(L: N.Library, ctx: BuildContext, focus: dict) -> list:
    rng = rng_for(ctx, 1)
    snow = ctx.env == "snow"
    placed = []
    for k, (x, z) in enumerate(NEAR_TREES):
        tree(L, rng, x, z, snow, "tree_small_02")
        focus[f"bank_tree_{k}"] = TERRAIN.at(x, z, 1.4)
        placed.append((x, z))
    region = ((-55.0, -85.0), (55.0, 35.0))
    for x, z in N.scatter(rng, region, 300, 4.6, avoid=[HERO_AREA], taken=placed):
        if not in_water(x, z, P.BANK + 0.8):
            tree(L, rng, x, z, snow)
            placed.append((x, z))
    return placed


# --------------------------------------------------------------------------
# Hero subjects
# --------------------------------------------------------------------------
def add_hero(b: SceneBuilder, L: N.Library, ctx: BuildContext, focus: dict) -> None:
    rng = rng_for(ctx, 2)
    snow = ctx.env == "snow"
    # wet boulder in the stream: the most compact stone in the sheet (tallest for its width, so it does not dam
    # the 2.7 m channel), sunk into the bed, about 0.35 m out of the water
    rocks = L.variants("rock_moss_set_01")
    big = int(np.argmax([h / w for _, h, w in rocks]))
    base = float(TERRAIN.height(*ROCK)) - 0.15
    k, m, _ = L.matrix("rock_moss_set_01", big, [ROCK[0], base, ROCK[1]], yaw=float(rng.uniform(0, 360)),
                       height=P.WATER_Y + 0.35 - base)
    L.instance("rock_moss_set_01", k, m, tag="_wet")
    focus["boulder"] = L.top_point("rock_moss_set_01", k, m)

    # fern bank: a patch on the left bank, the tallest fern in the middle (gone in snow; the point stays on the bank)
    if not snow:
        tops = []
        for j in range(8):
            a, r = j * 2.4 + rng.uniform(-0.3, 0.3), 0.0 if j == 0 else rng.uniform(0.35, 1.0)
            x, z = FERN_BANK[0] + r * math.cos(a), FERN_BANK[1] + r * math.sin(a)
            if in_water(x, z, 0.2):
                continue
            kk, mm, _ = L.matrix("fern_02", int(rng.integers(4)), TERRAIN.at(x, z, -0.02), yaw=float(rng.uniform(0, 360)),
                                 height=(0.9 if j == 0 else rng.uniform(0.55, 0.8)))
            L.instance("fern_02", kk, mm)
            tops.append(L.top_point("fern_02", kk, mm))
        focus["fern_bank"] = tops
    else:
        focus["fern_bank"] = TERRAIN.at(*FERN_BANK, 0.0)

    # fallen log across the stream, resting on both banks
    (x0, z0), (x1, z1), r = LOG
    length, yaw = math.hypot(x1 - x0, z1 - z0), -math.degrees(math.atan2(z1 - z0, x1 - x0))
    y0 = max(TERRAIN.height(x0, z0), TERRAIN.height(x1, z1)) - 0.10
    M = G.compose(G.translate((x0, float(y0), z0)), G.rotate((0, 1, 0), yaw))
    b.ply(b.assets.mesh("stream_log", lambda: N.log(np.random.default_rng(23), length, r)), M, "bark", prefix="log")
    axis, radii = N.log_axis(np.random.default_rng(23), length, r)
    focus["log"] = [(M @ np.append(N.log_top(axis, radii, u), 1.0))[:3].tolist() for u in (0.35, 0.5, 0.65)]


# --------------------------------------------------------------------------
# Stones and bank plants (merged)
# --------------------------------------------------------------------------
def add_scatter(b: SceneBuilder, L: N.Library, ctx: BuildContext, trees: list, focus: dict) -> None:
    rng = rng_for(ctx, 3)
    snow = ctx.env == "snow"
    seeded = Assets(ctx.assets_dir, ctx.rebuild)
    batch = N.Batch(L)
    hero = [(ROCK[0], ROCK[1], 1.2), (FERN_BANK[0], FERN_BANK[1], 1.3)]
    trunks = [(x, z, 0.9) for x, z in trees]
    stones = []
    # riffle: a band of stones across the channel, tops just above the water
    for j in range(14):
        z = rng.uniform(*RIFFLE_Z)
        x = cx(z) + rng.uniform(-1.0, 1.0) * hw(z)
        base = float(TERRAIN.height(x, z)) - 0.08                  # tops 3-20 cm out of the water
        batch.plant("rock_moss_set_01", int(rng.integers(6)), [x, base, z], yaw=float(rng.uniform(0, 360)),
                    height=max(0.15, P.WATER_Y + rng.uniform(0.03, 0.2) - base), tag="_wet")
        stones.append(L.top_point(*batch.last))
    focus["riffle"] = stones[:]
    # stones scattered along the whole stream, in the water and at its edges (wet)
    for _ in range(90):
        z = rng.uniform(-70.0, 30.0)
        x = cx(z) + rng.uniform(-1.25, 1.25) * hw(z)
        if any((x - hx) ** 2 + (z - hz) ** 2 < hr ** 2 for hx, hz, hr in hero):
            continue
        base = float(TERRAIN.height(x, z)) - 0.12                  # tops from 15 cm under to 25 cm out of the water
        batch.plant("rock_moss_set_01", int(rng.integers(6)), [x, base, z], yaw=float(rng.uniform(0, 360)),
                    height=max(0.15, P.WATER_Y + rng.uniform(-0.15, 0.25) - base), tag="_wet")
        stones.append(L.top_point(*batch.last))
    focus["stones"] = stones
    # dry stones on the banks
    banks = lambda x, z: not in_water(x, z, P.BANK)
    for x, z in N.scatter(rng, ((-30.0, -70.0), (30.0, 30.0)), 50, 3.0, avoid=hero + trunks):
        if banks(x, z):
            batch.plant("rock_moss_set_01", int(rng.integers(6)), TERRAIN.at(x, z, -0.12), yaw=float(rng.uniform(0, 360)),
                        height=rng.uniform(0.3, 0.8), tag="_dry")
    if not snow:
        near_bank = lambda x, z: not in_water(x, z, P.BANK * 0.4) and in_water(x, z, 4.0)
        for x, z in N.scatter(rng, ((-12.0, -70.0), (12.0, 30.0)), 260, 1.4, avoid=hero + trunks):   # ferns by the water
            if near_bank(x, z):
                for _ in range(int(rng.integers(1, 4))):
                    dx, dz = rng.normal(0, 0.35, 2)
                    if not in_water(x + dx, z + dz, P.BANK * 0.4):
                        batch.plant("fern_02", int(rng.integers(4)), TERRAIN.at(x + dx, z + dz, -0.02),
                                    yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.4, 0.8))
        for x, z in N.scatter(rng, ((-12.0, -70.0), (12.0, 30.0)), 220, 1.0, avoid=hero + trunks):   # nettles, grass at the edge
            if not in_water(x, z, P.BANK * 0.3) and in_water(x, z, P.BANK + 1.5):
                aid = "nettle_plant" if rng.random() < 0.5 else "grass_medium_02"
                batch.plant(aid, int(rng.integers(6 if aid == "nettle_plant" else 2)), TERRAIN.at(x, z, -0.02),
                            yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.3, 0.7))
        for x, z in N.scatter(rng, ((-45.0, -80.0), (45.0, 30.0)), 160, 3.0, avoid=hero + trunks):   # woodland ferns
            if not in_water(x, z, P.BANK + 1.0):
                batch.plant("fern_02", int(rng.integers(4)), TERRAIN.at(x, z, -0.02), yaw=float(rng.uniform(0, 360)),
                            height=rng.uniform(0.45, 0.85))
        for x, z in N.scatter(rng, ((-45.0, -80.0), (45.0, 30.0)), 90, 3.5, avoid=hero + trunks):    # shrubs
            if not in_water(x, z, P.BANK + 1.5):
                batch.plant("shrub_02", int(rng.integers(4)), TERRAIN.at(x, z, -0.05), yaw=float(rng.uniform(0, 360)),
                            height=rng.uniform(1.0, 1.8))
    batch.flush(seeded, f"stream_scatter_s{ctx.seed}_{ctx.env == 'snow'}_v{ASSET_VERSION}")


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build(ctx: BuildContext) -> SceneBundle:
    b = SceneBuilder(Assets(ctx.shared_dir, ctx.rebuild))
    L = N.Library(b)
    focus: dict[str, list] = {}
    add_materials(b, L, ctx)
    add_ground(b)
    trees = add_trees(L, ctx, focus)
    add_hero(b, L, ctx, focus)
    add_scatter(b, L, ctx, trees, focus)
    focus["water"] = [[cx(z), P.WATER_Y, z] for z in (5.0, 0.0, -4.0, -10.0, -20.0, -30.0)]
    sun = SUN.get(ctx.env)
    b.d["sky"] = env_kit.sky(ctx.env, sun, scale=SKY_SCALE[ctx.env])
    return SceneBundle(b.d, focus, {"sun_direction": sun, "sky_scale": SKY_SCALE[ctx.env], "trees": len(trees)})


def _exclusions():
    ex = [((x - 0.9, -1.0, z - 0.9), (x + 0.9, 2.6, z + 0.9)) for x, z in NEAR_TREES]
    ex.append(((ROCK[0] - 1.1, -1.0, ROCK[1] - 1.1), (ROCK[0] + 1.1, 1.2, ROCK[1] + 1.1)))
    ex.append(((FERN_BANK[0] - 1.3, -1.0, FERN_BANK[1] - 1.3), (FERN_BANK[0] + 1.3, 1.3, FERN_BANK[1] + 1.3)))
    return tuple(ex)


SCENE = SceneDef(
    id="stream", group="natural", owner="beiba",
    description="Woodland stream seen upstream: a wet boulder in rippled water, a fern bank, a fallen log across the "
                "stream and a riffle of stones, trees over both banks, sun glinting on the water.",
    build=build,
    views={
        # Low beside the water, the wet boulder in front, the stream bending away upstream.
        "boulder": View((cx(5.0) + 1.2, 0.45, 5.0), (cx(-12.0), -0.1, -12.0), focus="boulder"),
        # Standing on the right bank, focused on the log across the stream.
        "log": View((cx(4.0) + hw(4.0) + 2.0, 1.5, 4.0), (cx(-9.0), 0.0, -9.0), focus="log"),
        # Low over the water, looking along it into the glints, focused on the water.
        "water": View((cx(8.0), 0.45, 8.0), (cx(-20.0), -0.15, -20.0), focus="water"),
        # From the left bank past the ferns to the riffle.
        "bank": View((cx(7.0) - hw(7.0) - 1.6, 1.0, 7.0), (cx(-15.0), 0.0, -15.0), focus="fern_bank"),
    },
    default_view="boulder",
    camera_box=((-6.0, 0.35, -6.0), (6.0, 1.9, 10.0)),
    target_box=((-25.0, -0.5, -60.0), (25.0, 3.0, 15.0)),
    exclude_boxes=_exclusions(),
    lens_mm=(24.0, 135.0), f_number=(1.4, 8.0),
    envs=("clear", "overcast", "snow"),
    tags=("outdoor", "day", "water", "foliage", "specular", "bokeh", "snow"),
    default_seed=1, asset_version=ASSET_VERSION, max_depth=10, rr_depth=5, spp_hint=2048, far_clip=2000.0,
)
