"""Broadleaf woodland floor: ferns and mossy rock in front, a stump and a fallen log behind, trunks receding.

Natural scene (owner beiba). Built for depth of field: subjects at 3.5, 4, 10 and 19 m from the hero
cameras, and small bright highlights at every depth (sunlit fern fronds, sun flecks on the leaf litter,
sky gaps between backlit crowns, and in `wet` glossy stone, bark and litter).

Layout, meters, y up, the hero cameras look toward -z:
  floor        dark leaf litter (scan darkened to humus albedo) on a gently rolling height field rising
               3.5 cm per m toward -z, so it recedes in view
  clearing     x in [-6, 6], z in [-7, 7]: where cameras go; five near trees stand at its edges with crowns over it
  foreground   a fern patch at (-1.4, 1.2) and a mossy boulder at (1.3, -1.0)
  midground    a stump at (-2.0, -6.0); ferns, shrubs and rocks scattered through the wood
  background   a fallen log from (-5.0, -12.0) to (3.5, -15.0), then scanned trees to ~55 m
  backdrop     a sparser belt of trees to ~95 m, then a dark woodland band at 100 m
Light is only `env_kit`'s sun and sky; the sun is high (55 degrees) and to the right of and behind the
hero cameras, so it never appears in their frames and comes through the crowns as flecks on the floor.

Cost: the ~1,050 scattered plants are merged into one mesh per material (`_nature_kit.Batch`) rather than
instanced; as instances they made a 1080p pass 3.8x slower (373 s vs 98 s per 1024 spp on an M4 Max).
Trees stay instanced: thinning the far ones saved little and opened empty sky behind the near trunks.

Environments: clear (sun and sky), overcast, wet (overcast, with darkened, clear-coated litter, stone and bark).
Seed (`ctx.seed`) moves every tree outside the clearing, the scattered undergrowth, rocks and extra logs,
and turns the near trees and the hero plants; the hero subjects keep their places so the views hold.
"""

from __future__ import annotations

import math

import numpy as np

import env_kit
import procedural as G
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, principled, rgb
from scenes import _nature_kit as N
from scenes import _pool_props as PP
import web_assets as W

ASSET_VERSION = 1
EYE = np.eye(4)
SUN = env_kit.default_sun_direction(55.0, 70.0)
SKY_SCALE = {"clear": 22.0, "overcast": 50.0, "wet": 70.0}
LITTER_DARKEN = 0.55

TERRAIN = N.Terrain(-90.0, 90.0, -110.0, 70.0, spacing=0.4,
                    waves=[(0.16, 23.0, 20.0, 0.3), (0.08, 11.0, 110.0, 1.1), (0.035, 4.3, 60.0, 2.0),
                           (0.015, 1.9, 150.0, 0.5)],
                    bumps=[(1.3, -1.0, 1.6, 0.12)], tilt=(0.0, -0.035))

FERN_A = (-1.4, 1.2)
ROCK = (1.3, -1.0)
STUMP = (-2.0, -6.0)
LOG = ((-5.0, -12.0), (3.5, -15.0), 0.33)                # start, end, radius
NEAR_TREES = [(-4.8, 4.6), (5.1, 2.2), (-5.3, -3.4), (4.6, -5.6), (-1.0, -9.6)]
LANDMARKS = [(-3.0, -24.0), (5.5, -36.0)]               # fixed mid and far trees, focus targets
CLEARING = ((-6.5, -10.5), (6.5, 7.5))                  # random trees stay out of this


def rng_for(ctx: BuildContext, k: int) -> np.random.Generator:
    return np.random.default_rng([ctx.seed, 5381, k])


def add_materials(b: SceneBuilder, ctx: BuildContext) -> None:
    wet = ctx.env == "wet"

    def surf(aid, darken=1.0, coat=False, uv_scale=None):
        d = W.surface(aid, uv_scale=uv_scale, tint=(darken,) * 3 if darken != 1.0 else None)
        if coat:
            (d["bsdf"] if d["type"] == "normalmap" else d).update(clearcoat=0.6, clearcoat_gloss=0.9)
        return d

    # The scan's mean albedo (0.096) read as pale sand in full sun; 0.55 brings it to dark humus (~0.053).
    b.material("litter", surf("brown_mud_leaves_01", LITTER_DARKEN * (0.65 if wet else 1.0), wet))
    b.material("bark", surf("bark_brown_02", 0.65 if wet else 1.0, wet))
    b.material("backdrop", {"type": "twosided", "bsdf": principled((0.035, 0.05, 0.03), 0.9)})


def add_ground(b: SceneBuilder) -> None:
    b.ply(b.assets.mesh(f"terrain_v{ASSET_VERSION}", TERRAIN.mesh), EYE, "litter", prefix="ground")
    ring = G.compose(G.translate((0.0, -5.0, -20.0)))
    b.ply(b.assets.mesh("backdrop_ring", lambda: PP.treeline(100.0, 61, base=19.0, var=4.0)), ring, "backdrop",
          prefix="backdrop")


def tree(L: N.Library, rng, x, z, kind=None) -> tuple[str, float]:
    kind = kind or ("tree_small_02" if rng.random() < 0.75 else "island_tree_02")
    h = rng.uniform(9.0, 13.0) if kind == "tree_small_02" else rng.uniform(5.5, 7.5)
    sink = 0.15                                           # roots below the litter on uneven ground
    L.whole(kind, TERRAIN.at(x, z, -sink), yaw=float(rng.uniform(0, 360)), height=h)
    return kind, h


def add_trees(L: N.Library, ctx: BuildContext, focus: dict) -> list:
    rng = rng_for(ctx, 1)
    placed = []
    for k, (x, z) in enumerate(NEAR_TREES):
        tree(L, rng, x, z, "tree_small_02")
        focus[f"near_tree_{k}"] = TERRAIN.at(x, z, 1.5)
        placed.append((x, z))
    for k, (x, z) in enumerate(LANDMARKS):
        tree(L, rng, x, z, "tree_small_02")
        focus[f"trunk_{['mid', 'far'][k]}"] = TERRAIN.at(x, z, 1.6)
        placed.append((x, z))
    avoid = [CLEARING, (LOG[0][0], LOG[0][1], 2.0), (LOG[1][0], LOG[1][1], 2.0), (-0.75, -13.5, 2.2)]
    for x, z in N.scatter(rng, ((-55.0, -75.0), (55.0, 40.0)), 260, 5.2, avoid=avoid, taken=placed):
        tree(L, rng, x, z)
        placed.append((x, z))
    inner = ((-55.0, -75.0), (55.0, 40.0))                 # a sparser outer belt hides the backdrop band
    for x, z in N.scatter(rng, ((-88.0, -104.0), (88.0, 64.0)), 320, 6.5, avoid=[inner], taken=placed):
        tree(L, rng, x, z)
        placed.append((x, z))
    return placed


def add_hero(b: SceneBuilder, L: N.Library, ctx: BuildContext, focus: dict) -> None:
    rng = rng_for(ctx, 2)
    # fern patch: seven fronds-clumps around FERN_A, the tallest in the middle
    for k in range(7):
        a, r = k * 2.4 + rng.uniform(-0.3, 0.3), 0.0 if k == 0 else rng.uniform(0.35, 0.85)
        x, z = FERN_A[0] + r * math.cos(a), FERN_A[1] + r * math.sin(a)
        L.plant("fern_02", int(rng.integers(4)), TERRAIN.at(x, z, -0.02), yaw=float(rng.uniform(0, 360)),
                height=(0.9 if k == 0 else rng.uniform(0.55, 0.8)))
    focus["fern_a"] = TERRAIN.at(FERN_A[0], FERN_A[1], 0.55)

    # boulder: the largest rock in the sheet, mossy
    rocks = L.variants("rock_moss_set_01")
    big = int(np.argmax([h * w for _, h, w in rocks]))
    rh = L.plant("rock_moss_set_01", big, TERRAIN.at(*ROCK, -0.12), yaw=float(rng.uniform(0, 360)), height=0.85)
    focus["rock"] = TERRAIN.at(*ROCK, rh - 0.1)
    for k in range(5):                                    # small ferns tucked against it
        a = rng.uniform(0, math.tau)
        L.plant("fern_02", int(rng.integers(4)), TERRAIN.at(ROCK[0] + 0.85 * math.cos(a), ROCK[1] + 0.85 * math.sin(a), -0.02),
                yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.4, 0.6))

    # stump
    L.whole("tree_stump_01", TERRAIN.at(*STUMP, -0.05), yaw=float(rng.uniform(0, 360)))
    focus["stump"] = TERRAIN.at(*STUMP, 0.5)
    for k in range(4):
        a = rng.uniform(0, math.tau)
        L.plant("fern_02", int(rng.integers(4)), TERRAIN.at(STUMP[0] + 1.0 * math.cos(a), STUMP[1] + 1.0 * math.sin(a), -0.02),
                yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.45, 0.7))

    # fallen log
    (x0, z0), (x1, z1), r = LOG
    length, yaw = math.hypot(x1 - x0, z1 - z0), -math.degrees(math.atan2(z1 - z0, x1 - x0))
    y0 = min(TERRAIN.height(x0, z0), TERRAIN.height(x1, z1)) - 0.08
    M = G.compose(G.translate((x0, float(y0), z0)), G.rotate((0, 1, 0), yaw))
    b.ply(b.assets.mesh("hero_log_v2", lambda: N.log(np.random.default_rng(17), length, r)), M, "bark", prefix="log")
    # No moss_01 clumps on it: that scan renders as dark upright spikes, not a carpet, at the close range the
    # sampler reaches. The focus point sits on the trunk's real (bent) top.
    axis, radii = N.log_axis(np.random.default_rng(17), length, r)          # the same bent centre line as the mesh
    focus["log"] = (M @ np.append(N.log_top(axis, radii, 0.5), 1.0))[:3].tolist()


def add_undergrowth(b: SceneBuilder, L: N.Library, ctx: BuildContext, trees: list) -> None:
    rng = rng_for(ctx, 3)
    region = ((-45.0, -65.0), (45.0, 30.0))
    hero = [CLEARING, (LOG[0][0], LOG[0][1], 1.5), (LOG[1][0], LOG[1][1], 1.5), (-0.75, -13.5, 1.8)]
    trunks = [(x, z, 1.0) for x, z in trees]
    seeded = Assets(ctx.assets_dir, ctx.rebuild)
    batch = N.Batch(L)                        # ~1,050 plants merged per material: 3.8x cheaper than instances
    for x, z in N.scatter(rng, region, 230, 2.0, avoid=hero + trunks):          # fern clumps
        for _ in range(int(rng.integers(2, 6))):
            dx, dz = rng.normal(0, 0.5, 2)
            batch.plant("fern_02", int(rng.integers(4)), TERRAIN.at(x + dx, z + dz, -0.02),
                        yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.45, 0.85))
    for x, z in N.scatter(rng, region, 110, 3.2, avoid=hero + trunks):          # shrubs
        batch.plant("shrub_02", int(rng.integers(4)), TERRAIN.at(x, z, -0.05), yaw=float(rng.uniform(0, 360)),
                    height=rng.uniform(1.0, 1.8))
    for x, z in N.scatter(rng, region, 60, 3.0, avoid=hero + trunks):           # rocks, mostly sunk
        batch.plant("rock_moss_set_01", int(rng.integers(6)), TERRAIN.at(x, z, -0.15), yaw=float(rng.uniform(0, 360)),
                    height=rng.uniform(0.3, 0.9))
    for x, z in N.scatter(rng, region, 50, 3.0, avoid=hero + trunks):           # grass tufts
        batch.plant("grass_medium_02", int(rng.integers(2)), TERRAIN.at(x, z, -0.02), yaw=float(rng.uniform(0, 360)),
                    height=rng.uniform(0.3, 0.5))
    batch.flush(seeded, f"undergrowth_s{ctx.seed}_v{ASSET_VERSION}")
    for x, z in N.scatter(rng, region, 6, 12.0, avoid=hero + trunks):           # old stumps
        L.whole("tree_stump_01", TERRAIN.at(x, z, -0.05), yaw=float(rng.uniform(0, 360)), scale=float(rng.uniform(0.7, 1.2)))
    for k, (x, z) in enumerate(N.scatter(rng, region, 5, 14.0, avoid=hero + trunks)):   # other fallen logs
        length, r, yaw = rng.uniform(4.0, 9.0), rng.uniform(0.15, 0.3), float(rng.uniform(0, 360))
        y = TERRAIN.height(x, z) - 0.06
        path = seeded.mesh(f"log_s{ctx.seed}_{k}", lambda length=length, r=r, k=k: N.log(rng_for(ctx, 40 + k), length, r))
        b.ply(path, G.compose(G.translate((x, float(y), z)), G.rotate((0, 1, 0), yaw)), "bark", prefix="log")


def build(ctx: BuildContext) -> SceneBundle:
    b = SceneBuilder(Assets(ctx.shared_dir, ctx.rebuild))
    L = N.Library(b)
    focus: dict[str, list] = {}
    add_materials(b, ctx)
    add_ground(b)
    trees = add_trees(L, ctx, focus)
    add_hero(b, L, ctx, focus)
    add_undergrowth(b, L, ctx, trees)
    sun = SUN if ctx.env == "clear" else None
    b.d["sky"] = env_kit.sky(ctx.env, sun, scale=SKY_SCALE[ctx.env])
    return SceneBundle(b.d, focus, {"sun_direction": sun, "sky_scale": SKY_SCALE[ctx.env], "trees": len(trees)})


def _exclusions():
    ex = [((x - 0.9, -1.0, z - 0.9), (x + 0.9, 2.6, z + 0.9)) for x, z in NEAR_TREES]
    ex.append(((FERN_A[0] - 1.2, -1.0, FERN_A[1] - 1.2), (FERN_A[0] + 1.2, 1.3, FERN_A[1] + 1.2)))
    ex.append(((ROCK[0] - 1.3, -1.0, ROCK[1] - 1.3), (ROCK[0] + 1.3, 1.4, ROCK[1] + 1.3)))
    ex.append(((STUMP[0] - 1.6, -1.0, STUMP[1] - 1.6), (STUMP[0] + 1.6, 1.2, STUMP[1] + 1.6)))
    return tuple(ex)


SCENE = SceneDef(
    id="forest_floor", group="natural", owner="beiba",
    description="Broadleaf woodland floor in dappled sun: a fern patch and a mossy boulder in front, a stump and a "
                "fallen log behind, scanned trees receding to 55 m.",
    build=build,
    views={
        # Low beside the fern patch, the wood receding behind it.
        "ferns": View((0.3, 1.0, 4.6), (-2.0, 0.6, -10.0), focus="fern_a"),
        # Past the boulder toward the stump and the log.
        "rock": View((2.6, 0.95, 3.0), (0.4, 0.5, -6.0), focus="rock"),
        # Standing, focused on the stump with ferns soft in front.
        "stump": View((-0.6, 1.45, 3.5), (-2.2, 0.4, -10.0), focus="stump"),
        # Down the wood to the fallen log, the near ferns well out of focus.
        "log": View((1.2, 1.5, 5.5), (-0.5, 0.9, -14.0), focus="log"),
    },
    default_view="ferns",
    camera_box=((-6.0, 0.7, -7.0), (6.0, 2.0, 7.0)),
    target_box=((-30.0, -0.5, -45.0), (30.0, 4.5, 20.0)),
    exclude_boxes=_exclusions(),
    lens_mm=(24.0, 135.0), f_number=(1.4, 8.0),
    envs=("clear", "overcast", "wet"),
    tags=("outdoor", "day", "foliage"),
    default_seed=1, asset_version=ASSET_VERSION, max_depth=8, rr_depth=5, spp_hint=2048, far_clip=2000.0,
)
