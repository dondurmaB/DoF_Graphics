"""Mossy fallen log at macro range: a moss-carpeted trunk with small mushrooms on it and at its foot, fallen
leaves, twigs and ferns around it, in a broadleaf wood.

Natural scene (owner beiba). Built for close focus: the cameras sit 0.12-1.0 m above the litter and 0.3-3.5 m
in front of the log, so focus lands mostly between 0.3 and 15 m (the sampler keeps 0.3 m of clearance), with
the wood behind for the large background blurs.

Layout, meters, y up, the hero cameras look toward -z:
  log          a 6 m trunk (radius 0.32 m) from (-3.0, -1.0) to (3.0, -1.45), resting on the litter, finely
               tessellated (2-3 cm facets), moss on its top and down its sides in patches
  moss         a separate shell 2-10 mm above the bark wherever the moss mask is on, with a procedural albedo at
               4 texels per mm (the scanned 1k textures give ~1 per mm, soft at this range)
  mushrooms    seven at the log's foot around (-0.75, -0.5), five on its top around x = 0.6, 3-9 cm tall
  ground       dark leaf litter on a gently rolling height field; in front of the log ~2,600 fallen leaves and
               ~140 twigs as real geometry, so near silhouettes are sharp where the texture would not be
  around       ferns at (2.0, -0.35) and (-2.3, 0.2), a small mossy rock at (1.15, -0.15), a stump at (-4.0, -2.8)
  wood         scanned trees outside the clearing to ~55 m, a sparser belt to ~95 m, the forest_floor backdrop
               band at 100 m
Light is only `env_kit`'s sun and sky; the sun is high and behind the cameras to the right.

Cost: leaves, twigs, mushrooms and the scattered undergrowth are merged into one mesh per material
(`_nature_kit.Batch` or here directly), trees stay instanced (forest_floor's measurements).

Environments: clear (sun and sky), overcast, wet (overcast, darkened and clear-coated litter, bark, moss,
leaves and mushrooms). Seed (`ctx.seed`) moves the trees outside the clearing, the undergrowth, the fallen
leaves and the twigs; the log, its moss and the mushrooms keep their places so the views hold.
"""

from __future__ import annotations

import math

import numpy as np

import env_kit
import procedural as G
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, bitmap, principled
from scenes import _nature_kit as N
import web_assets as W

ASSET_VERSION = 1
EYE = np.eye(4)
SUN = env_kit.default_sun_direction(52.0, 65.0)
# forest_floor's 22 / 50 / 70 left the along_log view under the 0.02 median-luminance floor (clear 0.012, overcast
# 0.017, wet 0.024; check_scene 269605 and 269625). x2.5 / x2 / x1.43 puts it at ~0.03; the brightest view
# (mushrooms, 0.077 in clear) goes to ~0.19, inside the 0.5 ceiling.
SKY_SCALE = {"clear": 55.0, "overcast": 100.0, "wet": 100.0}
LITTER_DARKEN = 0.55

TERRAIN = N.Terrain(-90.0, 90.0, -110.0, 70.0, spacing=0.4,
                    waves=[(0.16, 23.0, 20.0, 0.3), (0.08, 11.0, 110.0, 1.1), (0.035, 4.3, 60.0, 2.0),
                           (0.015, 1.9, 150.0, 0.5)],
                    tilt=(0.0, -0.03))

LOG = ((-3.0, -1.0), (3.0, -1.45), 0.32)                 # start, end, radius
LOG_SEED = 23
MOSS_TILE_M = 0.25                                       # moss albedo tile, 1024 px
BASE_CLUSTER = (-0.75, -0.5)
TOP_CLUSTER_U = 0.6                                      # fraction along the log
FERNS = [(2.0, -0.35), (-2.3, 0.2)]
ROCK = (1.15, -0.15)
STUMP = (-4.0, -2.8)
LEAF_ZONE = ((-3.4, -0.75), (3.4, 3.2))                  # fallen leaves and twigs, in front of the log
NEAR_TREES = [(-4.6, 3.8), (4.8, 2.6), (-5.2, -4.2), (4.4, -4.8), (0.8, -6.6)]
CLEARING = ((-5.8, -5.5), (5.8, 4.5))
BACKDROP = {"centre": (0.0, -20.0), "radius": 100.0, "y0": -6.0, "y_tex": 22.0, "tile": 30.0}
SKYLINE = N.Skyline(71, BACKDROP["radius"], base=13.0, spread=4.0)
LEAF_COLOURS = {"leaf_brown": (0.10, 0.045, 0.015), "leaf_tan": (0.17, 0.095, 0.03), "leaf_dark": (0.05, 0.028, 0.012)}


def rng_for(ctx: BuildContext, k: int) -> np.random.Generator:
    return np.random.default_rng([ctx.seed, 7919, k])


def log_frame():
    """(matrix, length, yaw) placing N.log's local trunk (along +x, resting on y = 0) between LOG's ends."""
    (x0, z0), (x1, z1), _ = LOG
    length, yaw = math.hypot(x1 - x0, z1 - z0), -math.degrees(math.atan2(z1 - z0, x1 - x0))
    y0 = min(TERRAIN.height(x0, z0), TERRAIN.height(x1, z1)) - 0.06
    return G.compose(G.translate((x0, float(y0), z0)), G.rotate((0, 1, 0), yaw)), length


def hero_log() -> G.Mesh:
    _, length = log_frame()
    return N.log(np.random.default_rng(LOG_SEED), length, LOG[2], sides=128, rings=240)


def moss_mask(mesh: G.Mesh) -> np.ndarray:
    """Per vertex: > 0 where moss grows. Upward-facing bark, broken into patches that creep down the sides."""
    u, v = mesh.uv[:, 0], mesh.uv[:, 1]
    noise = (0.5 * np.sin(u * 7.1 + v * 1.3 + 0.4) + 0.3 * np.sin(v * 3.7 - u * 4.2 + 2.1)
             + 0.2 * np.sin(u * 17.0 + v * 11.0 + 1.0))
    return mesh.n[:, 1] + 0.45 * noise - 0.15


def moss_shell() -> G.Mesh:
    """The moss carpet: the log's own surface pushed out 2-10 mm along its normals where the mask is on,
    thinning to ~1 mm at the patch edges so no step shows there."""
    log = hero_log()
    m = moss_mask(log)
    u, v = log.uv[:, 0], log.uv[:, 1]
    lump = 0.5 + 0.5 * np.sin(u * 97.0 + v * 61.0) * np.sin(u * 53.0 - v * 83.0)
    h = (0.002 + 0.008 * lump) * np.clip(m / 0.3, 0.12, 1.0)
    p = log.p + log.n * h[:, None]
    f = log.f[(m[log.f] > 0).all(1)]
    used = np.unique(f)                                  # drop the bark vertices the shell does not use
    remap = np.full(len(p), -1)
    remap[used] = np.arange(len(used))
    p, f = p[used], remap[f]
    return G.Mesh(p, G.vertex_normals(p, f), log.uv[used], f).oriented()


def moss_albedo(px: int = 1024, seed: int = 31) -> np.ndarray:
    """sRGB albedo of a moss cushion that tiles in both directions: olive to yellow-green tufts, dark gaps."""
    o = px // 8
    n = px + o
    tone = G.fbm(n, n, octaves=6, base=8, seed=seed)
    fine = G.fbm(n, n, octaves=3, base=96, seed=seed + 1)
    lo, hi = np.array([0.030, 0.055, 0.010]), np.array([0.095, 0.150, 0.030])
    lin = lo + (hi - lo) * (0.6 * tone + 0.4 * fine)[..., None]
    lin[fine < 0.22] = np.array([0.010, 0.014, 0.005])
    srgb = N._srgb(lin)
    a = N._periodic(srgb.transpose(2, 0, 1).reshape(3 * n, n), o).reshape(3, n, px)          # tile in u
    a = N._periodic(a.transpose(0, 2, 1).reshape(3 * px, n), o).reshape(3, px, px)          # tile in v
    return a.transpose(2, 1, 0).astype(np.float32)


def mushroom(h: float, cap_r: float, stem_r: float) -> dict[str, G.Mesh]:
    """{'cap', 'stem'}: a lathe-turned mushroom of height h standing on y = 0 (the cap's underside in 'stem')."""
    cap_h = 0.45 * cap_r
    stem = G.lathe([(0.0, -0.004), (stem_r * 1.15, -0.004), (stem_r * 1.1, 0.2 * h), (stem_r * 0.9, h - 0.3 * cap_h),
                    (cap_r * 0.92, h - 0.2 * cap_h), (0.0, h - 0.18 * cap_h)], segments=40)
    cap = G.lathe([(0.0, h + cap_h), (cap_r * 0.45, h + 0.9 * cap_h), (cap_r * 0.8, h + 0.55 * cap_h),
                   (cap_r, h + 0.1 * cap_h), (cap_r * 0.96, h - 0.2 * cap_h), (0.0, h - 0.18 * cap_h)], segments=48)
    return {"cap": cap, "stem": stem}


def mushrooms() -> tuple[dict[str, G.Mesh], dict[str, list]]:
    """The two fixed clusters, merged per material, and the top of every cap as focus points."""
    rng = np.random.default_rng(41)
    M, length = log_frame()
    axis, radii = N.log_axis(np.random.default_rng(LOG_SEED), length, LOG[2], rings=240)
    parts: dict[str, list] = {"cap": [], "stem": []}
    tops: dict[str, list] = {"mushrooms_base": [], "mushrooms_top": []}
    for k in range(7):                                   # at the foot of the log, in the litter
        a, r = rng.uniform(0, math.tau), rng.uniform(0.0, 0.22)
        x, z = BASE_CLUSTER[0] + r * math.cos(a), BASE_CLUSTER[1] + 0.5 * r * math.sin(a)
        h = rng.uniform(0.035, 0.09)
        cap_r = h * rng.uniform(0.35, 0.55)
        at = TERRAIN.at(x, z, -0.006)
        m = G.compose(G.translate(at), G.rotate((0, 1, 0), rng.uniform(0, 360)),
                      G.rotate((1, 0, 0), rng.uniform(-8, 8)))
        for name, mesh in mushroom(h, cap_r, cap_r * 0.18).items():
            parts[name].append(mesh.transformed(m))
        tops["mushrooms_base"].append((m @ np.array([0.0, h + 0.45 * cap_r, 0.0, 1.0]))[:3].tolist())
    for k in range(5):                                   # in the moss on top of the log
        u = TOP_CLUSTER_U + rng.uniform(-0.04, 0.04)
        x = u * axis[-1, 0]
        r = float(np.interp(x, axis[:, 0], radii))
        dz = rng.uniform(-0.35, 0.35) * r
        top = np.array([x, r + math.sqrt(r * r - dz * dz) + 0.002, float(np.interp(x, axis[:, 0], axis[:, 2])) + dz])
        h = rng.uniform(0.03, 0.06)
        cap_r = h * rng.uniform(0.35, 0.5)
        m = G.compose(M, G.translate(top), G.rotate((0, 1, 0), rng.uniform(0, 360)),
                      G.rotate((1, 0, 0), 0.6 * math.degrees(math.asin(dz / r))))    # lean with the trunk's curve
        for name, mesh in mushroom(h, cap_r, cap_r * 0.18).items():
            parts[name].append(mesh.transformed(m))
        tops["mushrooms_top"].append((m @ np.array([0.0, h + 0.45 * cap_r, 0.0, 1.0]))[:3].tolist())
    return {k: N._concat(v) for k, v in parts.items()}, tops


def litter_meshes(ctx: BuildContext) -> dict[str, G.Mesh]:
    """{'leaf_<colour>', 'twigs'}: fallen leaves and twigs lying on the terrain in front of the log."""
    rng = rng_for(ctx, 5)
    (xa, za), (xb, zb) = LEAF_ZONE
    acc: dict[str, list] = {k: [] for k in LEAF_COLOURS}
    names = list(LEAF_COLOURS)
    for _ in range(2600):
        x, z = rng.uniform(xa, xb), rng.uniform(za, zb)
        length = rng.uniform(0.05, 0.12)
        leaf = G.leaf(length, length * rng.uniform(0.4, 0.65), curl=-rng.uniform(0.02, 0.2),
                      fold=rng.uniform(0.1, 0.45), rows=8, cols=5)
        m = G.compose(G.translate(TERRAIN.at(x, z, 0.002 + rng.uniform(0, 0.004))),
                      G.rotate((0, 1, 0), rng.uniform(0, 360)), G.rotate((1, 0, 0), rng.uniform(-6, 6)))
        acc[names[int(rng.choice(3, p=[0.5, 0.3, 0.2]))]].append(leaf.transformed(m))
    twigs = []
    for _ in range(140):
        x, z = rng.uniform(xa, xb), rng.uniform(za, zb)
        length, r, yaw = rng.uniform(0.1, 0.6), rng.uniform(0.003, 0.011), rng.uniform(0, math.tau)
        s = np.linspace(0, 1, 6)
        side = np.cumsum(rng.normal(0, 0.03 * length, 6))
        px = x + s * length * math.cos(yaw) - side * math.sin(yaw)
        pz = z + s * length * math.sin(yaw) + side * math.cos(yaw)
        path = np.stack([px, TERRAIN.height(px, pz) + 0.8 * r, pz], -1)
        twigs.append(G.sweep(path, r * (1 - 0.4 * s), sides=8))
    out = {k: N._concat(v) for k, v in acc.items()}
    out["twigs"] = N._concat(twigs)
    return out


def add_materials(b: SceneBuilder, L: N.Library, ctx: BuildContext) -> None:
    wet = ctx.env == "wet"
    coat = {"clearcoat": 0.6, "clearcoat_gloss": 0.9} if wet else {}

    def surf(aid, darken=1.0):
        d = W.surface(aid, tint=(darken,) * 3 if darken != 1.0 else None)
        if wet:
            (d["bsdf"] if d["type"] == "normalmap" else d).update(coat)
        return d

    b.material("litter", surf("brown_mud_leaves_01", LITTER_DARKEN * (0.65 if wet else 1.0)))
    b.material("bark", surf("bark_brown_02", 0.65 if wet else 1.0))
    moss = b.assets.texture("moss_albedo_v1", moss_albedo)
    b.material("moss", {"type": "twosided", "bsdf": principled(bitmap(moss, uv_scale=(1 / MOSS_TILE_M,) * 2),
                                                               0.55 if wet else 0.85, sheen=0.4, **coat)})
    dk = 0.7 if wet else 1.0
    for name, c in LEAF_COLOURS.items():
        b.material(name, {"type": "twosided", "bsdf": principled(tuple(dk * t for t in c), 0.6, **coat)})
    b.material("mush_cap", principled((0.24 * dk, 0.11 * dk, 0.045 * dk), 0.35, **coat))
    b.material("mush_stem", principled((0.50, 0.45, 0.36), 0.6))
    alb = b.assets.texture("backdrop_albedo_v2", lambda: N.backdrop_albedo(71, BACKDROP["tile"], BACKDROP["y0"], BACKDROP["y_tex"]))
    b.material("backdrop", {"type": "twosided", "bsdf": principled(bitmap(alb), 0.9)})
    b.material("backdrop_leaf", {"type": "twosided", "bsdf": principled((0.010, 0.018, 0.006), 0.85)})
    L.material_set("rock_moss_set_01", "_stone", lambda bsdf: N.stone(bsdf, wet=wet))


def add_ground(b: SceneBuilder, ctx: BuildContext) -> None:
    b.ply(b.assets.mesh(f"terrain_v{ASSET_VERSION}", TERRAIN.mesh), EYE, "litter", prefix="ground")
    cx, cz = BACKDROP["centre"]
    b.ply(b.assets.mesh("backdrop_band", lambda: N.backdrop_band(cx, cz, SKYLINE, BACKDROP["y0"], BACKDROP["y_tex"],
                                                                  BACKDROP["tile"])), EYE, "backdrop", prefix="backdrop")
    b.ply(b.assets.mesh("backdrop_fringe", lambda: N.backdrop_fringe(np.random.default_rng(72), cx, cz, SKYLINE)), EYE,
          "backdrop_leaf", prefix="backdrop_leaf")
    seeded = Assets(ctx.assets_dir, ctx.rebuild)
    for name, path in sorted(seeded.meshes(f"litter_s{ctx.seed}_v{ASSET_VERSION}", lambda: litter_meshes(ctx)).items()):
        b.ply(path, EYE, "bark" if name == "twigs" else name, prefix="litter")


def add_log(b: SceneBuilder, focus: dict) -> None:
    M, length = log_frame()
    b.ply(b.assets.mesh(f"hero_log_v{ASSET_VERSION}", hero_log), M, "bark", prefix="log")
    b.ply(b.assets.mesh(f"moss_shell_v{ASSET_VERSION}", moss_shell), M, "moss", prefix="moss")
    axis, radii = N.log_axis(np.random.default_rng(LOG_SEED), length, LOG[2], rings=240)
    # on the moss (2-10 mm over the bark top), so the points sit on a real surface
    focus["log"] = [(M @ np.append(N.log_top(axis, radii, u) + [0.0, 0.004, 0.0], 1.0))[:3].tolist()
                    for u in (0.15, 0.35, 0.8, 0.92)]
    parts, tops = mushrooms()
    for name in ("cap", "stem"):
        b.ply(b.assets.mesh(f"mushroom_{name}_v{ASSET_VERSION}", lambda name=name: parts[name]), EYE,
              f"mush_{name}", prefix="mushroom")
    focus.update(tops)


def add_hero_plants(b: SceneBuilder, L: N.Library, focus: dict) -> None:
    rng = np.random.default_rng(43)
    focus["fern"] = []
    for fx, fz in FERNS:
        for k in range(5):
            a, r = k * 2.4 + rng.uniform(-0.3, 0.3), 0.0 if k == 0 else rng.uniform(0.25, 0.6)
            at = TERRAIN.at(fx + r * math.cos(a), fz + r * math.sin(a), -0.02)
            v, m, _ = L.matrix("fern_02", int(rng.integers(4)), at, yaw=float(rng.uniform(0, 360)),
                               height=(0.6 if k == 0 else rng.uniform(0.35, 0.55)))
            L.instance("fern_02", v, m)
            if k == 0:
                focus["fern"].append(L.top_point("fern_02", v, m))
    rocks = L.variants("rock_moss_set_01")
    small = int(np.argmin([h * w for _, h, w in rocks]))
    v, m, rh = L.matrix("rock_moss_set_01", small, TERRAIN.at(*ROCK, -0.04), yaw=float(rng.uniform(0, 360)), height=0.24)
    L.instance("rock_moss_set_01", v, m, tag="_stone")
    focus["rock"] = L.top_point("rock_moss_set_01", v, m)
    L.whole("tree_stump_01", TERRAIN.at(*STUMP, -0.05), yaw=float(rng.uniform(0, 360)))
    focus["stump"] = TERRAIN.at(*STUMP, 0.5)


def tree(L: N.Library, rng, x, z, kind=None) -> None:
    kind = kind or ("tree_small_02" if rng.random() < 0.75 else "island_tree_02")
    h = rng.uniform(9.0, 13.0) if kind == "tree_small_02" else rng.uniform(5.5, 7.5)
    L.whole(kind, TERRAIN.at(x, z, -0.15), yaw=float(rng.uniform(0, 360)), height=h)


def add_trees(L: N.Library, ctx: BuildContext, focus: dict) -> list:
    rng = rng_for(ctx, 1)
    placed = []
    for k, (x, z) in enumerate(NEAR_TREES):
        tree(L, rng, x, z, "tree_small_02")
        focus[f"near_tree_{k}"] = TERRAIN.at(x, z, 1.0)
        placed.append((x, z))
    for x, z in N.scatter(rng, ((-55.0, -75.0), (55.0, 40.0)), 240, 5.2, avoid=[CLEARING], taken=placed):
        tree(L, rng, x, z)
        placed.append((x, z))
    inner = ((-55.0, -75.0), (55.0, 40.0))
    for x, z in N.scatter(rng, ((-88.0, -104.0), (88.0, 64.0)), 300, 6.5, avoid=[inner], taken=placed):
        tree(L, rng, x, z)
        placed.append((x, z))
    return placed


def add_undergrowth(L: N.Library, ctx: BuildContext, trees: list) -> None:
    rng = rng_for(ctx, 3)
    region = ((-45.0, -65.0), (45.0, 30.0))
    hero = [CLEARING]
    trunks = [(x, z, 1.0) for x, z in trees]
    batch = N.Batch(L)
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
                    height=rng.uniform(0.3, 0.9), tag="_stone")
    for x, z in N.scatter(rng, ((-5.5, -5.0), (5.5, 4.2)), 18, 0.9,             # grass tufts near the log, off it
                          avoid=[((-3.4, -1.9), (3.4, -0.65))]):
        batch.plant("grass_medium_02", int(rng.integers(2)), TERRAIN.at(x, z, -0.02), yaw=float(rng.uniform(0, 360)),
                    height=rng.uniform(0.15, 0.3))
    batch.flush(Assets(ctx.assets_dir, ctx.rebuild), f"undergrowth_s{ctx.seed}_v{ASSET_VERSION}")


def build(ctx: BuildContext) -> SceneBundle:
    b = SceneBuilder(Assets(ctx.shared_dir, ctx.rebuild))
    L = N.Library(b)
    focus: dict[str, list] = {}
    add_materials(b, L, ctx)
    add_ground(b, ctx)
    add_log(b, focus)
    add_hero_plants(b, L, focus)
    trees = add_trees(L, ctx, focus)
    add_undergrowth(L, ctx, trees)
    sun = SUN if ctx.env == "clear" else None
    b.d["sky"] = env_kit.sky(ctx.env, sun, scale=SKY_SCALE[ctx.env])
    return SceneBundle(b.d, focus, {"sun_direction": sun, "sky_scale": SKY_SCALE[ctx.env], "trees": len(trees)})


def _exclusions():
    ex = [((-3.4, -0.5, -1.95), (3.4, 0.95, -0.55))]                         # the log
    ex += [((x - 0.9, -1.0, z - 0.9), (x + 0.9, 2.6, z + 0.9)) for x, z in NEAR_TREES]
    ex += [((x - 0.7, -1.0, z - 0.7), (x + 0.7, 0.8, z + 0.7)) for x, z in FERNS]
    ex.append(((ROCK[0] - 0.4, -1.0, ROCK[1] - 0.4), (ROCK[0] + 0.4, 0.5, ROCK[1] + 0.4)))
    return tuple(ex)


SCENE = SceneDef(
    id="mossy_log_macro", group="natural", owner="beiba",
    description="A moss-carpeted fallen log at macro range in a broadleaf wood: mushrooms on it and at its foot, "
                "fallen leaves, twigs and ferns in front, the wood receding behind.",
    build=build,
    views={
        # Low at the mushrooms by the log's foot, the moss and the wood soft behind.
        "mushrooms": View((-0.45, 0.22, 0.25), (-0.75, 0.08, -0.6), focus="mushrooms_base"),
        # Looking down at the mushrooms in the moss on top of the log.
        "log_moss": View((0.9, 0.9, 0.35), (0.45, 0.55, -1.2), focus="mushrooms_top"),
        # From the log's end, along the moss toward the far end.
        "along_log": View((3.1, 0.45, -0.35), (-2.0, 0.45, -1.15), focus="log"),
        # Leaves and twigs in front, the fern and the log behind.
        "fern": View((1.0, 0.45, 1.8), (2.0, 0.3, -0.35), focus="fern"),
    },
    default_view="mushrooms",
    camera_box=((-3.4, 0.12, -0.45), (3.4, 1.0, 3.5)),
    target_box=((-6.0, -0.3, -8.0), (6.0, 1.4, 1.0)),
    exclude_boxes=_exclusions(),
    lens_mm=(24.0, 135.0), f_number=(1.4, 16.0),
    envs=("clear", "overcast", "wet"),
    tags=("outdoor", "day", "foliage", "closeup", "bokeh"),
    default_seed=1, asset_version=ASSET_VERSION, max_depth=8, rr_depth=5, spp_hint=2048, far_clip=2000.0,
)
