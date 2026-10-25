"""Summer meadow: wildflowers and tussocks in front, a pond, a stone and a lone tree in the middle, a hedgerow and a
tree line behind, wooded hills on the horizon.

Natural scene (owner beiba). Built for depth of field: subjects at about 2, 4, 9, 14, 21, 48 and 62 m from the hero
cameras, and small bright highlights at every depth: sunlit gazania petals and backlit grass blades in front, sun
and sky glints on the rippled pond and its wet shore in the middle, sky gaps through the lone tree's crown, and
sky between the crowns of the tree line far away.

Layout, meters, y up, the hero cameras look toward -z:
  meadow       a grass texture on gently rolling ground (two low hills behind), rising 2 cm per m toward -z
  camera area  x in [-8, 8], z in [-4, 8]: tussocks, short grass and flowers all around
  foreground   a flower patch at (-1.2, 0.8) and a weathered boulder with two stones at (2.4, -2.8)
  midground    an old stump at (-3.0, -6.5); a pond ~5 m across centred at (-6.5, -13.0) with a wet mud shore;
               a lone tree at (5.0, -20.0)
  background   a hedgerow of shrubs at z ~ -44 in front of a tree line from z -46 to -85 (landmark trunks at
               (-8, -48) and (12, -62)), then sparser trees to ~115 m
  backdrop     wooded hills at 120 m: an opaque band whose top follows a lumpy skyline frayed by leaf clusters,
               all geometry, so depth and image agree (as in forest_floor)
Light is only `env_kit`'s sun and sky. The sun stands 46 degrees up, ahead and to the right of the hero cameras
(azimuth 125): grass and leaves are partly backlit, and the sun stays above the frames the target box allows.

Cost: the ~1,100 scattered plants (tussocks, blades, flowers, weeds, shrubs, stones) are merged into one mesh per
material (`_nature_kit.Batch`); trees stay instanced. Stones use the opaque `_nature_kit.stone`, not the converter's
leaf BSDF (web_assets classes `rock_moss_set_01` as foliage). No alpha-masked geometry anywhere: the G-buffer takes
the first hit and ignores opacity masks.

Environments: clear (sun and sky), overcast, wet (overcast, darkened clear-coated ground, wet stones and shore).
Sky scales are provisional until the first check_scene on the cluster.
Seed (`ctx.seed`) moves the scattered plants, stones and every tree except the lone tree and the landmarks; the
hero subjects keep their places so the views hold.
"""

from __future__ import annotations

import math

import numpy as np

import env_kit
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, bitmap, principled
from scenes import _meadow_props as P
from scenes import _nature_kit as N
import web_assets as W

ASSET_VERSION = 1
EYE = np.eye(4)
SUN = env_kit.default_sun_direction(46.0, 125.0)
SKY_SCALE = {"clear": 12.0, "overcast": 30.0, "wet": 40.0}      # provisional: calibrate on the first check_scene
GRASS_DARKEN = 0.8
TERRAIN = P.TERRAIN

FLOWERS = (-1.2, 0.8)
FLOWERS_MID = (3.0, -8.0)
ROCK = (2.4, -2.8)
STUMP = (-3.0, -6.5)
LONE_TREE = (5.0, -20.0)
LANDMARKS = [(-8.0, -48.0), (12.0, -62.0)]
HEDGE_Z = (-47.0, -41.0)
TREE_LINE = ((-100.0, -85.0), (100.0, -46.0))
CAMERA_AREA = ((-8.0, -4.0), (8.0, 8.0))
BACKDROP = {"centre": (0.0, -30.0), "radius": 120.0, "y0": -6.0, "y_tex": 18.0, "tile": 30.0}
SKYLINE = N.Skyline(81, BACKDROP["radius"], base=9.0, spread=5.0)


def rng_for(ctx: BuildContext, k: int) -> np.random.Generator:
    return np.random.default_rng([ctx.seed, 7919, k])


def add_materials(b: SceneBuilder, L: N.Library, ctx: BuildContext) -> None:
    wet = ctx.env == "wet"

    def surf(aid, darken=1.0, coat=False):
        d = W.surface(aid, tint=(darken,) * 3 if darken != 1.0 else None)
        if coat:
            (d["bsdf"] if d["type"] == "normalmap" else d).update(clearcoat=0.6, clearcoat_gloss=0.9)
        return d

    b.material("meadow", surf("leafy_grass", GRASS_DARKEN * (0.7 if wet else 1.0), wet))
    shore = surf("brown_mud_02", 0.5, True)                      # wet mud at the water's edge, every env
    b.material("shore", shore)
    b.material("bed", surf("brown_mud_02", 0.7))
    # Rough, not smooth: alpha 0.06 keeps sun glints small and bright while next-event estimation reaches them.
    b.material("water", {"type": "roughdielectric", "int_ior": 1.33, "ext_ior": 1.0, "alpha": 0.06})
    alb = b.assets.texture("backdrop_albedo_meadow_v1", lambda: N.backdrop_albedo(81, BACKDROP["tile"], BACKDROP["y0"],
                                                                                   BACKDROP["y_tex"]))
    b.material("backdrop", {"type": "twosided", "bsdf": principled(bitmap(alb), 0.9)})
    b.material("backdrop_leaf", {"type": "twosided", "bsdf": principled((0.010, 0.018, 0.006), 0.85)})
    L.material_set("rock_moss_set_01", "_stone", lambda bsdf: N.stone(bsdf, wet=wet))


def add_ground(b: SceneBuilder, ctx: BuildContext) -> None:
    parts = b.assets.meshes(f"meadow_terrain_v{ASSET_VERSION}", P.terrain_parts)
    b.ply(parts["meadow"], EYE, "meadow", prefix="ground")
    b.ply(parts["shore"], EYE, "shore", prefix="shore")
    b.ply(parts["bed"], EYE, "bed", prefix="bed")
    b.ply(b.assets.mesh(f"pond_water_v{ASSET_VERSION}", lambda: P.water(91)), EYE, "water", prefix="water")
    cx, cz = BACKDROP["centre"]
    b.ply(b.assets.mesh("meadow_backdrop_band", lambda: N.backdrop_band(cx, cz, SKYLINE, BACKDROP["y0"], BACKDROP["y_tex"],
                                                                         BACKDROP["tile"])), EYE, "backdrop", prefix="backdrop")
    b.ply(b.assets.mesh("meadow_backdrop_fringe", lambda: N.backdrop_fringe(np.random.default_rng(82), cx, cz, SKYLINE)),
          EYE, "backdrop_leaf", prefix="backdrop_leaf")


def in_pond(x, z, margin=0.0) -> bool:
    return math.hypot(x - P.POND[0], z - P.POND[1]) < P.POND_R + margin


def tree(L: N.Library, rng, x, z, kind=None) -> None:
    kind = kind or ("tree_small_02" if rng.random() < 0.7 else "island_tree_02")
    h = rng.uniform(9.0, 14.0) if kind == "tree_small_02" else rng.uniform(5.5, 8.0)
    L.whole(kind, TERRAIN.at(x, z, -0.15), yaw=float(rng.uniform(0, 360)), height=h)


def add_trees(L: N.Library, ctx: BuildContext, focus: dict) -> list:
    rng = rng_for(ctx, 1)
    L.whole("tree_small_02", TERRAIN.at(*LONE_TREE, -0.15), yaw=float(rng.uniform(0, 360)), height=12.5)
    focus["lone_tree"] = TERRAIN.at(*LONE_TREE, 1.6)
    placed = [LONE_TREE]
    for k, (x, z) in enumerate(LANDMARKS):
        tree(L, rng, x, z, "tree_small_02")
        focus[f"trunk_{['mid', 'far'][k]}"] = TERRAIN.at(x, z, 1.6)
        placed.append((x, z))
    for x, z in N.scatter(rng, TREE_LINE, 150, 4.5, taken=placed):              # the tree line
        tree(L, rng, x, z)
        placed.append((x, z))
    open_meadow = ((-60.0, -46.0), (60.0, 40.0))
    for x, z in N.scatter(rng, ((-60.0, -40.0), (60.0, 40.0)), 8, 18.0,           # a few lone trees at the sides
                          avoid=[((-22.0, -40.0), (22.0, 40.0))], taken=placed):
        tree(L, rng, x, z)
        placed.append((x, z))
    for x, z in N.scatter(rng, ((-112.0, -142.0), (112.0, 82.0)), 170, 7.0,      # sparse outer belt
                          avoid=[open_meadow, TREE_LINE], taken=placed):
        if math.hypot(x - BACKDROP["centre"][0], z - BACKDROP["centre"][1]) < BACKDROP["radius"] - 6.0:
            tree(L, rng, x, z)
            placed.append((x, z))
    return placed


def add_hero(b: SceneBuilder, L: N.Library, batch: N.Batch, ctx: BuildContext, focus: dict) -> None:
    rng = rng_for(ctx, 2)
    # flower patch: gazania clumps around FLOWERS, the tallest in the middle; tussocks behind it
    for k in range(14):
        a, r = k * 2.4 + rng.uniform(-0.3, 0.3), 0.0 if k == 0 else rng.uniform(0.2, 1.1)
        batch.plant("flower_gazania", int(rng.integers(8)), TERRAIN.at(FLOWERS[0] + r * math.cos(a), FLOWERS[1] + r * math.sin(a), -0.01),
                    yaw=float(rng.uniform(0, 360)), height=0.3 if k == 0 else rng.uniform(0.16, 0.26))
        if k == 0:
            focus["flowers"] = L.top_point(*batch.last)
    for k in range(6):
        a = rng.uniform(0, math.tau)
        batch.plant("grass_medium_02", 1, TERRAIN.at(FLOWERS[0] + 1.3 * math.cos(a), FLOWERS[1] + 1.3 * math.sin(a), -0.02),
                    yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.45, 0.65))
    for k in range(10):                                                           # a second, mid-distance patch
        a, r = rng.uniform(0, math.tau), rng.uniform(0.0, 1.4)
        batch.plant("flower_gazania", int(rng.integers(8)), TERRAIN.at(FLOWERS_MID[0] + r * math.cos(a), FLOWERS_MID[1] + r * math.sin(a), -0.01),
                    yaw=float(rng.uniform(0, 360)), height=0.32 if k == 0 else rng.uniform(0.16, 0.28))
        if k == 0:
            focus["flowers_mid"] = L.top_point(*batch.last)

    # boulder: the largest stone in the sheet, weathered, with two smaller stones beside it
    rocks = L.variants("rock_moss_set_01")
    big = int(np.argmax([h * w for _, h, w in rocks]))
    k, m, _ = L.matrix("rock_moss_set_01", big, TERRAIN.at(*ROCK, -0.12), yaw=float(rng.uniform(0, 360)), height=0.75)
    L.instance("rock_moss_set_01", k, m, tag="_stone")
    focus["rock"] = L.top_point("rock_moss_set_01", k, m)
    for dx, dz in ((0.9, 0.5), (-0.7, 0.8)):
        batch.plant("rock_moss_set_01", int(rng.integers(6)), TERRAIN.at(ROCK[0] + dx, ROCK[1] + dz, -0.1),
                    yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.2, 0.35), tag="_stone")

    # old stump, grass round its foot
    L.whole("tree_stump_01", TERRAIN.at(*STUMP, -0.05), yaw=float(rng.uniform(0, 360)))
    focus["stump"] = TERRAIN.at(*STUMP, 0.5)
    for k in range(5):
        a = rng.uniform(0, math.tau)
        batch.plant("grass_medium_02", 1, TERRAIN.at(STUMP[0] + 0.9 * math.cos(a), STUMP[1] + 0.9 * math.sin(a), -0.02),
                    yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.4, 0.6))

    # pond: the water surface at its centre and the wet shore on the side facing the cameras
    focus["pond"] = [P.POND[0], P.WATER_Y, P.POND[1]]
    focus["pond_shore"] = P.shore_point(80.0)


def add_undergrowth(b: SceneBuilder, L: N.Library, batch: N.Batch, ctx: BuildContext, trees: list) -> None:
    rng = rng_for(ctx, 3)
    near = ((-12.0, -14.0), (12.0, 8.0))
    wide = ((-45.0, -44.0), (45.0, 25.0))
    hero = [(FLOWERS[0], FLOWERS[1], 1.2), (ROCK[0], ROCK[1], 1.4), (STUMP[0], STUMP[1], 1.0),
            (FLOWERS_MID[0], FLOWERS_MID[1], 1.4), (P.POND[0], P.POND[1], P.POND_R + 0.3)]
    trunks = [(x, z, 0.8) for x, z in trees]
    ok = lambda x, z: not in_pond(x, z, 0.3)
    for x, z in N.scatter(rng, near, 235, 0.9, avoid=hero + trunks):                  # tussocks round the cameras
        batch.plant("grass_medium_02", 1, TERRAIN.at(x, z, -0.02), yaw=float(rng.uniform(0, 360)),
                    height=rng.uniform(0.35, 0.65))
    for x, z in N.scatter(rng, near, 400, 0.45, avoid=hero + trunks):                 # short blades between them
        batch.plant("grass_medium_02", 0, TERRAIN.at(x, z, -0.01), yaw=float(rng.uniform(0, 360)),
                    height=rng.uniform(0.18, 0.32))
    for x, z in N.scatter(rng, wide, 220, 1.6, avoid=hero + trunks):                  # gazania clumps
        if ok(x, z):
            batch.plant("flower_gazania", int(rng.integers(8)), TERRAIN.at(x, z, -0.01), yaw=float(rng.uniform(0, 360)),
                        height=rng.uniform(0.15, 0.27))
    for x, z in N.scatter(rng, wide, 40, 4.0, avoid=hero + trunks):                   # periwinkle mats
        if ok(x, z):
            batch.plant("periwinkle_plant", int(rng.choice([0, 3])), TERRAIN.at(x, z, -0.02),
                        yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.2, 0.35))
    for x, z in N.scatter(rng, wide, 100, 2.5, avoid=hero + trunks):                  # weeds
        if ok(x, z):
            batch.plant("weed_plant_02", int(rng.integers(5)), TERRAIN.at(x, z, -0.01), yaw=float(rng.uniform(0, 360)),
                        height=rng.uniform(0.08, 0.16))
    for x, z in N.scatter(rng, wide, 18, 6.0, avoid=hero + trunks):                   # stones in the grass, mostly sunk
        if ok(x, z):
            batch.plant("rock_moss_set_01", int(rng.integers(6)), TERRAIN.at(x, z, -0.15), yaw=float(rng.uniform(0, 360)),
                        height=rng.uniform(0.25, 0.6), tag="_stone")
    for x, z in N.scatter(rng, ((-70.0, HEDGE_Z[0]), (70.0, HEDGE_Z[1])), 45, 2.6, taken=trees):   # hedgerow
        batch.plant("shrub_02", int(rng.integers(4)), TERRAIN.at(x, z, -0.05), yaw=float(rng.uniform(0, 360)),
                    height=rng.uniform(1.4, 2.4))
    for x, z in N.scatter(rng, wide, 40, 5.0, avoid=hero + trunks):                   # low bushes
        if ok(x, z):
            batch.plant("shrub_03", int(rng.integers(4)), TERRAIN.at(x, z, -0.03), yaw=float(rng.uniform(0, 360)),
                        height=rng.uniform(0.4, 0.8))


def build(ctx: BuildContext) -> SceneBundle:
    b = SceneBuilder(Assets(ctx.shared_dir, ctx.rebuild))
    seeded = Assets(ctx.assets_dir, ctx.rebuild)
    L = N.Library(b)
    focus: dict[str, list] = {}
    add_materials(b, L, ctx)
    add_ground(b, ctx)
    trees = add_trees(L, ctx, focus)
    batch = N.Batch(L)
    add_hero(b, L, batch, ctx, focus)
    add_undergrowth(b, L, batch, ctx, trees)
    batch.flush(seeded, f"meadow_plants_s{ctx.seed}_v{ASSET_VERSION}")
    sun = SUN if ctx.env == "clear" else None
    b.d["sky"] = env_kit.sky(ctx.env, sun, scale=SKY_SCALE[ctx.env])
    return SceneBundle(b.d, focus, {"sun_direction": sun, "sky_scale": SKY_SCALE[ctx.env], "trees": len(trees),
                                    "water_y": P.WATER_Y})


def _exclusions():
    ex = [((FLOWERS[0] - 1.2, -1.0, FLOWERS[1] - 1.2), (FLOWERS[0] + 1.2, 0.8, FLOWERS[1] + 1.2)),
          ((ROCK[0] - 1.4, -1.0, ROCK[1] - 1.4), (ROCK[0] + 1.4, 1.3, ROCK[1] + 1.4))]
    return tuple(ex)


SCENE = SceneDef(
    id="meadow", group="natural", owner="beiba",
    description="Summer meadow in sun: wildflowers and tussocks in front, a pond, a stone and a lone tree in the "
                "middle, a hedgerow and tree line at 45-85 m, wooded hills on the horizon.",
    build=build,
    views={
        # Low over the flower patch, the meadow and the lone tree behind it.
        "flowers": View((-0.3, 0.55, 3.4), (-1.8, 0.2, -6.0), focus="flowers"),
        # Past the boulder toward the pond.
        "rock": View((4.2, 1.0, 1.4), (0.2, 0.3, -11.0), focus="rock"),
        # Standing, focused on the pond's glints with the flowers soft in front.
        "pond": View((-2.5, 1.6, 3.0), (-6.5, -0.2, -13.0), focus="pond"),
        # The lone tree at ~24 m.
        "tree": View((1.0, 1.5, 4.0), (5.0, 3.0, -20.0), focus="lone_tree"),
        # Down the meadow to the tree line.
        "vista": View((0.5, 1.7, 6.5), (3.0, 3.5, -60.0), focus="trunk_far"),
    },
    default_view="flowers",
    camera_box=((-8.0, 0.45, -4.0), (8.0, 2.0, 8.0)),
    target_box=((-45.0, -0.5, -85.0), (45.0, 6.0, 4.0)),
    exclude_boxes=_exclusions(),
    lens_mm=(24.0, 135.0), f_number=(1.4, 8.0),
    envs=("clear", "overcast", "wet"),
    tags=("outdoor", "day", "foliage", "water", "specular", "bokeh", "open", "vista"),
    default_seed=1, asset_version=ASSET_VERSION, max_depth=8, rr_depth=5, spp_hint=2048, far_clip=2000.0,
)
