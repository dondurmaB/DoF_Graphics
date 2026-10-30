"""A dry riverbed in a rocky desert valley, seen up the channel: a varnished boulder on the gravel bed in front, a
bleached driftwood log and a dead stump in the middle, a gravel bar and a lone tree behind, a layered escarpment on
the horizon.

Natural scene (owner beiba). Built for depth of field: subjects at about 3, 5, 12, 15, 23 and 34 m from the hero
cameras, and small bright highlights at every depth: sun glints on varnished stones (desert varnish, a glossy clear
coat) all along the bed from 2 m to 90 m, the sunlit bleached log, sky gaps through the crowns of the trees on the
valley sides, and the escarpment's skyline against the bright sky.

Layout, meters, y up, the channel runs along z and the hero cameras look up it toward -z:
  bed          gravelly dry mud (`brown_mud_dry`) with low bars, 9-15 m wide, straight round the cameras (z -6..10),
               then one broad bend that hides it upstream (`_dry_riverbed_props`)
  banks        cut banks 0.9 m high over 1.6 m, then valley sides climbing ~10 m over 50 m; the terrain is cut along
               one contour (bed below 0.2 m, bank above), so the edge between them is smooth
  foreground   a varnished boulder on the bed at (1.2, 0), a bush on the left bank at z = 1
  midground    a bleached driftwood log lying diagonally on the bed at z -9..-5.5, a dead stump on the right bank at z = -11, a gravel
               bar across the channel at z -22..-17
  background   a lone tree on the left bank at z = -28, sparse trees and bushes on the valley sides to ~90 m
  backdrop     a layered rock escarpment at 110 m: an opaque band whose top follows a lumpy skyline, all geometry,
               so depth and image agree (as in forest_floor)
Light is only `env_kit`'s sun and sky. The sun stands 48 degrees up, ahead and to the right of the hero cameras
(azimuth 115): stones and bushes are side-lit with long shadows across the bed, and the sun stays above the frames
the target box allows.

Environments: clear; overcast; sand (a lower sun through dust, env_kit's turbidity 5 sky, wind-blown sand
lightening the bed and banks). Seed (`ctx.seed`) moves the scattered stones, bushes and every tree except the lone
tree; the hero subjects keep their places so the views hold.

Cost: the scattered stones and plants are merged per material (`_nature_kit.Batch`); trees stay instanced, and only
island_tree_02 is used (tree_small_02 alone is 4.6M triangles). Stones use the opaque `_nature_kit.stone` with the
scan's moss turned to bare rock (`_dry_riverbed_props.bare_rock`). No alpha-masked geometry anywhere: the G-buffer
takes the first hit and ignores opacity masks.
Sky scales are provisional until the first check_scene on the cluster.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import env_kit
import procedural as G
import web_assets as W
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, bitmap, principled
from scenes import _dry_riverbed_props as P
from scenes import _greenhouse_props as H
from scenes import _nature_kit as N

ASSET_VERSION = 1
EYE = np.eye(4)
SUN = {"clear": env_kit.default_sun_direction(48.0, 115.0), "sand": env_kit.default_sun_direction(34.0, 120.0)}
# check_scene previews (64 spp): clear 5 gave medians 0.17-0.23; overcast 20 gave 0.15-0.25, as bright as clear, so
# 15 (real overcast is darker than sun; ~0.11-0.18 expected); sand 4 gave 0.15-0.18.
SKY_SCALE = {"clear": 5.0, "overcast": 15.0, "sand": 4.0}
TINT = {"bed": {"sand": (1.15, 1.05, 0.85)}, "bank": {"sand": (0.95, 0.85, 0.7)}}
TINT_DEFAULT = {"bed": (0.95, 0.95, 0.95), "bank": (0.7, 0.66, 0.62)}
VARNISHED = 0.35                     # share of scattered stones with desert varnish (glossy)

TERRAIN = P.RiverbedTerrain(-60.0, 60.0, -100.0, 40.0, spacing=0.3,
                            waves=[(0.4, 31.0, 25.0, 0.6), (0.15, 11.0, 120.0, 1.9), (0.05, 3.7, 70.0, 0.3)])


def cx(z) -> float:
    return float(P.centre_x(z))


def hw(z) -> float:
    return float(P.half_width(z))


ROCK = (cx(0.0) + 1.2, 0.0)                                    # varnished boulder on the bed
BUSH = (cx(1.0) - hw(1.0) - 2.4, 1.0)                          # left bank, past the cut bank's top
LOG = ((cx(-5.5) - 2.4, -5.5), (cx(-9.0) + 0.8, -9.0), 0.16)
STUMP = (cx(-11.0) + hw(-11.0) + 2.6, -11.0)                   # right bank
BAR_Z = (-22.0, -17.0)
LONE_TREE = (cx(-28.0) - hw(-28.0) - 4.5, -28.0)               # left bank
HERO_AREA = ((-12.0, -14.0), (12.0, 12.0))                     # random trees stay out of this
RIDGE = {"centre": (0.0, -30.0), "radius": 110.0, "y0": -4.0, "y_tex": 30.0, "tile": 40.0}
SKYLINE = N.Skyline(93, RIDGE["radius"], base=15.0, spread=3.0)          # spread 7 read as castle teeth


def rng_for(ctx: BuildContext, k: int) -> np.random.Generator:
    return np.random.default_rng([ctx.seed, 8317, k])


def on_bed(x, z, margin=0.0) -> bool:
    return float(P.beyond_bed(x, z)) < -margin


# --------------------------------------------------------------------------
# Materials and ground
# --------------------------------------------------------------------------
def desert_stone(b: SceneBuilder, bsdf: dict, varnish: bool) -> dict:
    """`_nature_kit.stone` (opaque) on the scan's albedo turned to bare rock. Desert varnish, the dark glossy film
    on long-exposed desert stones, is the stream's wet clear coat: small, sharp sun glints."""
    s = N.stone(bsdf, wet=varnish)
    inner = s["bsdf"] if s["type"] == "normalmap" else s
    bc = inner["base_color"]
    if isinstance(bc, dict) and bc.get("type") == "bitmap":
        src = bc["filename"]
        inner["base_color"] = dict(bc, filename=b.assets.texture(f"bare_rock_{Path(src).stem}", lambda: P.bare_rock(src)))
    return s


def add_materials(b: SceneBuilder, L: N.Library, ctx: BuildContext) -> None:
    for name in ("bed", "bank"):
        b.material(name, W.surface("brown_mud_dry", tint=TINT[name].get(ctx.env, TINT_DEFAULT[name])))
    b.material("driftwood", W.surface("bark_brown_02", tint=(1.35, 1.3, 1.2)))         # sun-bleached
    alb = b.assets.texture(f"ridge_albedo_v{ASSET_VERSION}", lambda: P.ridge_albedo(91, RIDGE["tile"], RIDGE["y0"],
                                                                                     RIDGE["y_tex"]))
    b.material("ridge", {"type": "twosided", "bsdf": principled(bitmap(alb), 0.9)})
    L.material_set("rock_moss_set_01", "_rock", lambda bsdf: desert_stone(b, bsdf, varnish=False))
    L.material_set("rock_moss_set_01", "_varnish", lambda bsdf: desert_stone(b, bsdf, varnish=True))


def add_ground(b: SceneBuilder) -> None:
    parts = b.assets.meshes(f"riverbed_terrain_v{ASSET_VERSION}", lambda: P.terrain_parts(TERRAIN))
    b.ply(parts["bed"], EYE, "bed", prefix="bed")
    b.ply(parts["bank"], EYE, "bank", prefix="ground")
    c0, c1 = RIDGE["centre"]
    b.ply(b.assets.mesh("riverbed_ridge_band", lambda: N.backdrop_band(c0, c1, SKYLINE, RIDGE["y0"], RIDGE["y_tex"],
                                                                        RIDGE["tile"])), EYE, "ridge", prefix="backdrop")


# --------------------------------------------------------------------------
# Whole models with focus points on their real surfaces
# --------------------------------------------------------------------------
def place_model(b: SceneBuilder, aid: str, at, yaw: float, height: float | None = None):
    """W.place + W.add, returning the model and its matrix (for points on its surface)."""
    m = W.model(aid)
    M = W.place(m, at=at, yaw=yaw, height=height)
    W.add(b, m, M)
    return m, M


def surface_point(m, M, part: str, y: float | None = None) -> list[float]:
    """A vertex of `part` placed by M: the highest one (y None), or the one nearest the cameras (largest z) within
    15 cm of height y. A real surface point, so the sampler's visibility test can pass."""
    p = next(q for q in m.parts if q.name == part)
    v = H.read_ply(p.ply).p
    w = (np.c_[v, np.ones(len(v))] @ np.asarray(M).T)[:, :3]
    if y is None:
        return w[np.argmax(w[:, 1])].tolist()
    band = np.abs(w[:, 1] - y) < 0.15
    if not band.any():                                  # no vertex that close: the nearest in height
        band = np.abs(w[:, 1] - y) <= np.abs(w[:, 1] - y).min()
    return w[band][np.argmax(w[band][:, 2])].tolist()


# --------------------------------------------------------------------------
# Trees
# --------------------------------------------------------------------------
def tree(L: N.Library, rng, x, z) -> None:
    L.whole("island_tree_02", TERRAIN.at(x, z, -0.15), yaw=float(rng.uniform(0, 360)), height=float(rng.uniform(3.8, 6.5)))


def add_trees(b: SceneBuilder, L: N.Library, ctx: BuildContext, focus: dict) -> list:
    rng = rng_for(ctx, 1)
    at = TERRAIN.at(*LONE_TREE, -0.15)
    m, M = place_model(b, "island_tree_02", at, yaw=200.0, height=6.2)
    focus["lone_tree"] = surface_point(m, M, "island_tree_02", at[1] + 1.2)
    placed = [LONE_TREE]
    for x, z in N.scatter(rng, ((-55.0, -95.0), (55.0, 35.0)), 60, 9.0, avoid=[HERO_AREA], taken=placed):
        if float(P.beyond_bed(x, z)) > P.BANK + 1.0:
            tree(L, rng, x, z)
            placed.append((x, z))
    return placed


# --------------------------------------------------------------------------
# Hero subjects
# --------------------------------------------------------------------------
def add_hero(b: SceneBuilder, L: N.Library, batch: N.Batch, ctx: BuildContext, focus: dict) -> None:
    rng = rng_for(ctx, 2)
    # varnished boulder: the bulkiest stone in the sheet, sunk into the bed, 0.7 m out of it, two cobbles beside it
    rocks = L.variants("rock_moss_set_01")
    big = int(np.argmax([h * w for _, h, w in rocks]))
    k, m, _ = L.matrix("rock_moss_set_01", big, TERRAIN.at(*ROCK, -0.12), yaw=float(rng.uniform(0, 360)), height=0.82)
    L.instance("rock_moss_set_01", k, m, tag="_varnish")
    focus["boulder"] = L.top_point("rock_moss_set_01", k, m)
    for dx, dz in ((0.8, 0.6), (-0.6, 0.9)):
        batch.plant("rock_moss_set_01", int(rng.integers(6)), TERRAIN.at(ROCK[0] + dx, ROCK[1] + dz, -0.06),
                    yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.15, 0.28), tag="_rock")

    # bush on the left bank: a shrub_02 with low shrub_03 round it, small weeds at its foot
    batch.plant("shrub_02", 0, TERRAIN.at(*BUSH, -0.05), yaw=float(rng.uniform(0, 360)), height=1.3)
    focus["bush"] = L.top_point(*batch.last)
    for j in range(5):
        a, r = j * 2.4 + rng.uniform(-0.3, 0.3), rng.uniform(0.7, 1.1)
        batch.plant("shrub_03", int(rng.integers(4)), TERRAIN.at(BUSH[0] + r * math.cos(a), BUSH[1] + r * math.sin(a), -0.03),
                    yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.45, 0.75))
    for j in range(5):
        a = rng.uniform(0, math.tau)
        batch.plant("weed_plant_02", int(rng.integers(5)), TERRAIN.at(BUSH[0] + 1.1 * math.cos(a), BUSH[1] + 1.1 * math.sin(a), -0.01),
                    yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.08, 0.15))

    # bleached driftwood log lying across the bed, left by the last flood
    (x0, z0), (x1, z1), r = LOG
    length, yaw = math.hypot(x1 - x0, z1 - z0), -math.degrees(math.atan2(z1 - z0, x1 - x0))
    y0 = min(TERRAIN.height(x0, z0), TERRAIN.height(x1, z1)) - 0.04
    M = G.compose(G.translate((x0, float(y0), z0)), G.rotate((0, 1, 0), yaw))
    b.ply(b.assets.mesh("riverbed_log", lambda: N.log(np.random.default_rng(41), length, r)), M, "driftwood", prefix="log")
    axis, radii = N.log_axis(np.random.default_rng(41), length, r)
    focus["log"] = [(M @ np.append(N.log_top(axis, radii, u), 1.0))[:3].tolist() for u in (0.35, 0.5, 0.65)]

    # dead stump on the right bank
    m, M = place_model(b, "tree_stump_01", TERRAIN.at(*STUMP, -0.05), yaw=float(rng.uniform(0, 360)), height=0.75)
    focus["stump"] = surface_point(m, M, "tree_stump_01")


# --------------------------------------------------------------------------
# Stones and plants (merged)
# --------------------------------------------------------------------------
def add_scatter(L: N.Library, batch: N.Batch, ctx: BuildContext, trees: list, focus: dict) -> None:
    rng = rng_for(ctx, 3)
    hero = [(ROCK[0], ROCK[1], 1.3), (BUSH[0], BUSH[1], 1.4), (STUMP[0], STUMP[1], 1.1), (LONE_TREE[0], LONE_TREE[1], 1.5)]
    (lx0, lz0), (lx1, lz1), _ = LOG
    log_box = ((min(lx0, lx1) - 0.6, min(lz0, lz1) - 0.6), (max(lx0, lx1) + 0.6, max(lz0, lz1) + 0.6))
    trunks = [(x, z, 1.0) for x, z in trees]

    def stone(x, z, height, sink):
        batch.plant("rock_moss_set_01", int(rng.integers(6)), TERRAIN.at(x, z, -sink), yaw=float(rng.uniform(0, 360)),
                    height=height, tag="_varnish" if rng.random() < VARNISHED else "_rock")
        return L.top_point(*batch.last)

    # gravel bar across the channel: a dense band of cobbles
    bar = []
    for x, z in N.scatter(rng, ((-12.0, BAR_Z[0]), (8.0, BAR_Z[1])), 45, 0.7):
        if on_bed(x, z, 0.3):
            bar.append(stone(x, z, rng.uniform(0.12, 0.35), 0.05))
    focus["gravel_bar"] = bar
    # cobbles and small boulders over the whole bed
    stones = []
    for x, z in N.scatter(rng, ((-15.0, -95.0), (15.0, 35.0)), 220, 1.6, avoid=hero + [log_box]):
        if on_bed(x, z, 0.2) and not BAR_Z[0] <= z <= BAR_Z[1]:
            stones.append(stone(x, z, rng.uniform(0.1, 0.45), 0.06))
    focus["stones"] = stones
    # boulders fallen onto the banks and valley sides
    for x, z in N.scatter(rng, ((-45.0, -90.0), (45.0, 30.0)), 45, 4.0, avoid=hero + trunks):
        if not on_bed(x, z, -0.5):
            stone(x, z, rng.uniform(0.4, 1.3), 0.15)
    # bushes along the banks, bigger ones on the valley sides, sparse weeds on the bed
    for x, z in N.scatter(rng, ((-25.0, -90.0), (25.0, 30.0)), 160, 1.8, avoid=hero + trunks):
        if 0.6 < float(P.beyond_bed(x, z)) < 7.0:
            aid = "shrub_03" if rng.random() < 0.6 else "shrub_04"
            batch.plant(aid, int(rng.integers(4)), TERRAIN.at(x, z, -0.03), yaw=float(rng.uniform(0, 360)),
                        height=rng.uniform(0.35, 0.9))
    for x, z in N.scatter(rng, ((-50.0, -95.0), (50.0, 35.0)), 70, 4.5, avoid=hero + trunks + [HERO_AREA]):
        if float(P.beyond_bed(x, z)) > 4.0:
            batch.plant("shrub_02", int(rng.integers(4)), TERRAIN.at(x, z, -0.05), yaw=float(rng.uniform(0, 360)),
                        height=rng.uniform(0.9, 1.7))
    for x, z in N.scatter(rng, ((-15.0, -80.0), (15.0, 30.0)), 140, 1.5, avoid=hero + [log_box]):
        if on_bed(x, z, 0.5) or 0.0 < float(P.beyond_bed(x, z)) < 4.0:
            batch.plant("weed_plant_02", int(rng.integers(5)), TERRAIN.at(x, z, -0.01), yaw=float(rng.uniform(0, 360)),
                        height=rng.uniform(0.06, 0.16))


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build(ctx: BuildContext) -> SceneBundle:
    b = SceneBuilder(Assets(ctx.shared_dir, ctx.rebuild))
    seeded = Assets(ctx.assets_dir, ctx.rebuild)
    L = N.Library(b)
    focus: dict[str, list] = {}
    add_materials(b, L, ctx)
    add_ground(b)
    trees = add_trees(b, L, ctx, focus)
    batch = N.Batch(L)
    add_hero(b, L, batch, ctx, focus)
    add_scatter(L, batch, ctx, trees, focus)
    batch.flush(seeded, f"riverbed_scatter_s{ctx.seed}_v{ASSET_VERSION}")
    sun = SUN.get(ctx.env)
    b.d["sky"] = env_kit.sky(ctx.env, sun, scale=SKY_SCALE[ctx.env])
    return SceneBundle(b.d, focus, {"sun_direction": sun, "sky_scale": SKY_SCALE[ctx.env], "trees": len(trees)})


def _exclusions():
    ex = [((ROCK[0] - 1.2, -1.0, ROCK[1] - 1.2), (ROCK[0] + 1.2, 1.4, ROCK[1] + 1.2)),
          ((BUSH[0] - 1.4, -1.0, BUSH[1] - 1.4), (BUSH[0] + 1.4, 2.0, BUSH[1] + 1.4))]
    return tuple(ex)


SCENE = SceneDef(
    id="dry_riverbed", group="natural", owner="beiba",
    description="Dry riverbed in a rocky desert valley, seen up the channel: a varnished boulder on the gravel bed, a "
                "bleached driftwood log, a dead stump and a gravel bar, a lone tree on the bank, a layered "
                "escarpment on the horizon.",
    build=build,
    views={
        # Low beside the boulder, the bed running away up the valley.
        "boulder": View((cx(3.2) + 0.5, 0.45, 3.2), (cx(-15.0), 0.3, -15.0), focus="boulder"),
        # Standing on the bed, focused on the driftwood log.
        "log": View((cx(5.0) - 1.0, 1.6, 5.0), (cx(-7.5), 0.1, -7.5), focus="log"),
        # From the left edge of the bed, focused on the bush on the bank.
        "bank": View((cx(5.0) - 4.5, 0.9, 5.0), (BUSH[0] - 1.5, 1.2, BUSH[1] - 4.0), focus="bush"),
        # Standing, focused on the lone tree at ~35 m with the bed soft in front.
        "tree": View((cx(6.0) + 1.5, 1.7, 6.0), (LONE_TREE[0], 2.5, LONE_TREE[1]), focus="lone_tree"),
    },
    default_view="boulder",
    camera_box=((-5.0, 0.4, -4.0), (5.0, 1.9, 9.0)),
    target_box=((-30.0, -0.3, -85.0), (30.0, 6.0, 0.0)),
    exclude_boxes=_exclusions(),
    lens_mm=(24.0, 135.0), f_number=(1.4, 8.0),
    envs=("clear", "overcast", "sand"),
    tags=("outdoor", "day", "sand", "open", "foliage", "specular", "bokeh", "vista"),
    default_seed=1, asset_version=ASSET_VERSION, max_depth=8, rr_depth=5, spp_hint=2048, far_clip=2000.0,
)
