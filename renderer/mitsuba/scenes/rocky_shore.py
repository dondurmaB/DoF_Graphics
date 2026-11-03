"""A rocky lake shore seen from the beach: pebbles and cobbles down to the water, a granite boulder at the
waterline, a bleached driftwood log, a wooded headland on the left and the far wooded shore across the lake.

Natural scene (owner beiba). Built for depth of field against open water and sky: pebbles from 0.3 m (the cameras
sit 0.35-1.8 m up on the beach), the boulder and log at 2-6 m, the headland at 20-45 m, the far shore at ~320 m,
and about a third of most frames sky over the lake. Small bright highlights sit behind the focus plane: sun
glitter on the rippled water ahead (the sun is ahead and to the right of the hero cameras), glints on the wet
stones at the waterline and sky through the headland's crowns.

Layout, meters, y up, the shoreline runs roughly along x, the lake is toward -z:
  shore        the waterline z = shore(x) wanders +-1.5 m round the cameras, swings out to z ~ -25 into a
               headland for x < -15, and curves back toward +z far to both sides, so the beach is a bay
  beach        coarse grey sand (`brown_mud_dry`, cooled) rising 7 cm per metre from the water, a low berm at ~7 m,
               then a grassy bank climbing ~9 m over 40 m (`leafy_grass`); the terrain is cut along two contours
               (lakebed below -6 cm, bank above 0.8 m) so the material edges are smooth lines
  lake         rippled water (roughdielectric) over a lakebed sloping to -2 m: a finely rippled sheet round the
               cameras, a flat one beyond it to the far shore; both stop exactly on one grid line, so they never
               overlap; a dark lake floor under all of it at -2.5 m
  stones       ~3,800 procedural pebbles (2-8 cm, four grey granite tones: ambientCG's Granite006B is red, so
               its luminance is used; wet-coated within 40 cm of the water)
               densest along the waterline, scanned cobbles and boulders recoloured to grey granite
  foreground   the granite boulder at the waterline (1.3, shore + 0.2), two boulders on the beach at x = -3.8, the
               driftwood log at x 2.5-5.2
  background   10 trees on the headland, more along the bank, the far shore a backdrop band at 320 m whose
               bottom dips under the water
Light is only `env_kit`'s sun and sky; the sun stands 42 degrees up, ahead and to the right (azimuth 145), above
any frame the target box allows.

Environments: clear, overcast, wet (overcast after rain: darker sand, every stone clear-coated). Seed (`ctx.seed`)
moves the pebbles, cobbles, grass, shrubs and trees except the headland's; the hero subjects keep their places.

Cost: pebbles are one merged mesh per material; scattered scans are merged with `_nature_kit.Batch`; trees stay
instanced. Stones use the opaque `_nature_kit.stone`. No alpha-masked geometry.
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
from scenes import _nature_kit as N
from scenes import _stream_props as SP

ASSET_VERSION = 1
EYE = np.eye(4)
SUN = env_kit.default_sun_direction(42.0, 145.0)
# Open scene like dry_riverbed (clear 5 gave frame medians 0.17-0.23 there); calibrate on check_scene.
SKY_SCALE = {"clear": 5.0, "overcast": 15.0, "wet": 15.0}
WATER_Y = 0.0
LAKEBED_TOP = -0.06                  # lakebed material below this height, beach above
BANK_FROM = 0.8                      # grassy bank above this height
NEAR_WATER = ((-16.0, -30.0), (16.0, 6.0))       # finely rippled sheet; its edges sit on FAR_WATER's 2 m grid
FAR_WATER = ((-340.0, -360.0), (340.0, 330.0), 2.0)
PEBBLE_TONES = {"grey": (0.95, 0.95, 0.97), "tan": (1.08, 1.0, 0.9), "dark": (0.5, 0.5, 0.52), "pale": (1.35, 1.33, 1.3)}

FAR = {"centre": (0.0, -20.0), "radius": 320.0, "y0": -1.5, "y_tex": 26.0, "tile": 40.0}
SKYLINE = N.Skyline(57, FAR["radius"], base=14.0, spread=3.0)                 # spread 10 read as black teeth


def shore(x):
    """z of the waterline at x: wandering round the cameras, a headland out to z ~ -25 for x < -15, a bay."""
    x = np.asarray(x, float)
    head = 1.0 / (1.0 + np.exp((x + 18.0) / 3.5))
    return -2.0 + 0.1 * x + 1.2 * np.sin(x / 7.0 + 0.3) + 0.4 * np.sin(x / 2.3) - 24.0 * head + 0.004 * x * x


class ShoreTerrain(N.Terrain):
    """Signed distance d = z - shore(x) (> 0 on land). Land: 7 cm per metre, a 25 cm berm at ~7 m, then a bank
    climbing ~9 m over 40 m with N.Terrain's undulation fading in. Lake: sloping 7 cm per metre at the waterline,
    levelling off toward -2 m."""

    def height(self, x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        d = z - shore(x)
        und = super().height(x, z) - self.base
        t = np.clip((d - 6.0) / 2.5, 0.0, 1.0)
        e = np.maximum(d - 8.0, 0.0)
        land = 0.07 * d + 0.25 * t * t * (3 - 2 * t) + 0.1 * e + 0.003 * e * e + und * np.clip(e / 6.0, 0.0, 1.0)
        lake = -2.0 * (1.0 - np.exp(0.035 * np.minimum(d, 0.0)))
        return self.base + np.where(d >= 0, land, lake)


TERRAIN = ShoreTerrain(-130.0, 130.0, -80.0, 95.0, spacing=0.35,
                       waves=[(0.8, 37.0, 30.0, 0.4), (0.3, 13.0, 110.0, 1.7), (0.08, 4.1, 65.0, 0.2)])

BOULDER = (1.3, float(shore(1.3)) + 0.2)                      # at the waterline
BEACH_ROCKS = (-3.8, float(shore(-3.8)) + 3.4)
LOG = ((2.5, float(shore(2.5)) + 4.6), (5.2, float(shore(5.2)) + 3.6), 0.13)
HEADLAND = ((-46.0, -30.0), (-20.0, -12.0))
HERO_AREA = ((-10.0, -4.0), (10.0, 10.0))


def rng_for(ctx: BuildContext, k: int) -> np.random.Generator:
    return np.random.default_rng([ctx.seed, 6271, k])


def land(x, z, margin=0.0) -> bool:
    return float(z - shore(x)) > margin


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def terrain_parts() -> dict[str, G.Mesh]:
    """{'lakebed', 'beach', 'bank'}: the terrain cut exactly along two contours (`_stream_props.cut_at_height`)."""
    bed, rest = SP.cut_at_height(TERRAIN.mesh(), LAKEBED_TOP)
    beach, bank = SP.cut_at_height(rest, BANK_FROM)
    return {"lakebed": bed, "beach": beach, "bank": bank}


def near_water() -> G.Mesh:
    """Rippled water round the cameras (a light breeze: slope_rms 0.05), clipped to where the terrain is under
    it. The ripples fade out over the last 2 m before the sheet's edge, so it meets the flat far sheet level."""
    (x0, z0), (x1, z1) = NEAR_WATER
    m = N.water_sheet(x0, x1, z0, z1, WATER_Y, seed=33, spacing=0.08, slope_rms=0.05, waves=40,
                      mask=lambda x, z: TERRAIN.height(x, z) < WATER_Y + 0.03)
    edge = np.minimum.reduce([m.p[:, 0] - x0, x1 - m.p[:, 0], m.p[:, 2] - z0, z1 - m.p[:, 2]])
    w = np.clip(edge / 2.0, 0.0, 1.0)
    p = m.p.copy()
    p[:, 1] = WATER_Y + (p[:, 1] - WATER_Y) * w
    n = m.n * w[:, None] + np.array([0.0, 1.0, 0.0]) * (1 - w[:, None])
    return G.Mesh(p, n / np.linalg.norm(n, axis=1, keepdims=True), m.uv, m.f)


def far_water() -> G.Mesh:
    """Flat water from NEAR_WATER's edge to past the far shore, on a grid NEAR_WATER's edges lie on."""
    (x0, z0), (x1, z1), s = FAR_WATER
    xs, zs = np.arange(x0, x1 + s / 2, s), np.arange(z0, z1 + s / 2, s)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    p = np.stack([X, np.full_like(X, WATER_Y), Z], -1).reshape(-1, 3)
    f = G._grid_faces(len(xs), len(zs))
    c = p[f].mean(1)
    (nx0, nz0), (nx1, nz1) = NEAR_WATER
    inside = (c[:, 0] > nx0) & (c[:, 0] < nx1) & (c[:, 2] > nz0) & (c[:, 2] < nz1)
    on_land = (c[:, 0] > TERRAIN.x0) & (c[:, 0] < TERRAIN.x1) & (c[:, 2] > TERRAIN.z0) & (c[:, 2] < TERRAIN.z1) \
        & (TERRAIN.height(c[:, 0], c[:, 2]) > WATER_Y + 0.03 + s * 0.2)
    f = f[~inside & ~on_land]
    used, inv = np.unique(f, return_inverse=True)
    q = p[used]
    return G.Mesh(q, np.tile([0.0, 1.0, 0.0], (len(q), 1)), q[:, [0, 2]], inv.reshape(-1, 3)).oriented()


def pebble(rng, size: float) -> G.Mesh:
    """A water-worn pebble about `size` across: a flattened, lopsided ellipsoid with low bumps, ~170 triangles."""
    t = np.linspace(-math.pi / 2, math.pi / 2, 7)
    m = G.lathe(np.stack([np.cos(t), np.sin(t)], 1) * 0.5, segments=14)
    a, b, c = size * rng.uniform(0.75, 1.0), size * rng.uniform(0.25, 0.5), size * rng.uniform(0.55, 0.85)
    ph = rng.uniform(0, math.tau, 3)
    p = m.p * [a, b, c]
    bump = 1.0 + 0.08 * np.sin(3 * m.p[:, 0] * 6 + ph[0]) * np.sin(3 * m.p[:, 2] * 6 + ph[1]) \
        + 0.06 * np.sin(5 * m.p[:, 1] * 6 + ph[2])
    p = p * bump[:, None]
    p[:, 1] -= p[:, 1].min()
    return G.Mesh(p, G.vertex_normals(p, m.f), m.uv, m.f).oriented()


def pebbles(ctx: BuildContext) -> tuple[dict[str, G.Mesh], list]:
    """{'pebble_<tone>[_wet]'}: pebbles on a 12 cm jittered grid over the beach band from 1.5 m under water to
    5 m up the beach, densest along the waterline, partly sunk; plus the top points of those nearest the cameras."""
    rng = rng_for(ctx, 4)
    cell = 0.12
    acc: dict[str, list] = {}
    tops = []
    tones = list(PEBBLE_TONES)
    for x in np.arange(-14.0, 14.0, cell):
        for d in np.arange(-1.5, 5.0, cell):
            density = 0.75 * math.exp(-((d - 0.3) / 1.4) ** 2) + 0.12
            if rng.random() > density:
                continue
            px, pd = x + rng.uniform(0, cell), d + rng.uniform(0, cell)
            pz = float(shore(px)) + pd
            if (px - BOULDER[0]) ** 2 + (pz - BOULDER[1]) ** 2 < 0.8 ** 2:
                continue
            size = float(np.exp(rng.uniform(math.log(0.02), math.log(0.08))))
            m = pebble(rng, size)
            h = float(m.p[:, 1].max())
            M = G.compose(G.translate(TERRAIN.at(px, pz, -0.4 * h)), G.rotate((0, 1, 0), rng.uniform(0, 360)),
                          G.rotate((1, 0, 0), rng.uniform(-12, 12)))
            tone = tones[int(rng.choice(4, p=[0.45, 0.25, 0.2, 0.1]))]
            key = f"pebble_{tone}" + ("_wet" if pd < 0.4 else "")
            acc.setdefault(key, []).append(m.transformed(M))
            if 0.3 < pd < 2.5 and abs(px) < 6 and size > 0.04 and rng.random() < 0.15:
                tops.append((M @ np.array([0.0, h, 0.0, 1.0]))[:3].tolist())
    return {k: N._concat(v) for k, v in sorted(acc.items())}, tops


def granite_rock(src: str) -> np.ndarray:
    """sRGB albedo for the scanned stones on a granite shore: the scan's albedo (green moss) to its luminance,
    slightly cool and lifted, so the texture detail stays and the moss reads as lichen-pale rock."""
    from PIL import Image

    a = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255.0
    lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.clip((0.08 + 0.95 * lum)[..., None] * np.array([1.0, 1.0, 1.04], np.float32), 0.0, 1.0)


def grey_granite(src: str, tint) -> np.ndarray:
    """sRGB albedo from ambientCG Granite006B (a red granite): its luminance times `tint`, so the grain stays
    and the stone reads grey."""
    from PIL import Image

    a = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255.0
    lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.clip(lum[..., None] * np.asarray(tint, np.float32), 0.0, 1.0)


def granite_surface(b: SceneBuilder, tint, uv_scale=None) -> dict:
    """W.surface of Granite006B (its roughness and normal maps) with the albedo turned grey by `grey_granite`."""
    d = W.surface("acg:Granite006B", uv_scale=uv_scale)
    inner = d["bsdf"] if d["type"] == "normalmap" else d
    src = inner["base_color"]["filename"]
    key = "_".join(f"{c:.2f}" for c in tint)
    inner["base_color"] = dict(inner["base_color"], filename=b.assets.texture(f"grey_granite_{key}",
                                                                              lambda: grey_granite(src, tint)))
    return d


# --------------------------------------------------------------------------
# Materials and ground
# --------------------------------------------------------------------------
def granite(b: SceneBuilder, bsdf: dict, wet: bool) -> dict:
    s = N.stone(bsdf, wet=wet)
    inner = s["bsdf"] if s["type"] == "normalmap" else s
    bc = inner["base_color"]
    if isinstance(bc, dict) and bc.get("type") == "bitmap":
        src = bc["filename"]
        inner["base_color"] = dict(bc, filename=b.assets.texture(f"granite_rock_{Path(src).stem}", lambda: granite_rock(src)))
    return s


def coat(d: dict) -> dict:
    (d["bsdf"] if d["type"] == "normalmap" else d).update(clearcoat=0.8, clearcoat_gloss=0.92)
    return d


def add_materials(b: SceneBuilder, L: N.Library, ctx: BuildContext) -> None:
    wet = ctx.env == "wet"
    beach = W.surface("brown_mud_dry", tint=(0.6, 0.6, 0.62) if wet else (0.95, 0.95, 1.0))
    b.material("beach", coat(beach) if wet else beach)
    b.material("lakebed", W.surface("brown_mud_02", tint=(0.55, 0.55, 0.5)))
    b.material("bank", W.surface("leafy_grass", tint=(0.8, 0.8, 0.8) if wet else None))
    b.material("lake_floor", principled((0.02, 0.022, 0.018), 0.9))
    b.material("driftwood", W.surface("bark_brown_02", tint=(1.35, 1.3, 1.2) if not wet else (0.9, 0.85, 0.8)))
    b.material("water", {"type": "roughdielectric", "int_ior": 1.33, "ext_ior": 1.0, "alpha": 0.06})
    alb = b.assets.texture("backdrop_albedo_v2", lambda: N.backdrop_albedo(71, FAR["tile"], FAR["y0"], FAR["y_tex"]))
    b.material("far_shore", {"type": "twosided", "bsdf": principled(bitmap(alb), 0.9)})
    for tone, tint in PEBBLE_TONES.items():
        dry = granite_surface(b, tint, uv_scale=(0.12, 0.12))
        b.material(f"pebble_{tone}", coat(dry) if wet else dry)
        b.material(f"pebble_{tone}_wet", coat(granite_surface(b, tuple(0.7 * t for t in tint), uv_scale=(0.12, 0.12))))
    L.material_set("rock_moss_set_01", "_granite", lambda bsdf: granite(b, bsdf, wet=wet))
    L.material_set("rock_moss_set_01", "_granite_wet", lambda bsdf: granite(b, bsdf, wet=True))


def add_ground(b: SceneBuilder, ctx: BuildContext, focus: dict) -> None:
    parts = b.assets.meshes(f"shore_terrain_v{ASSET_VERSION}", terrain_parts)
    for name in ("lakebed", "beach", "bank"):
        b.ply(parts[name], EYE, name, prefix=name)
    b.ply(b.assets.mesh(f"shore_near_water_v{ASSET_VERSION}", near_water), EYE, "water", prefix="water")
    b.ply(b.assets.mesh(f"shore_far_water_v{ASSET_VERSION}", far_water), EYE, "water", prefix="water")
    b.rect(G.compose(G.translate((0.0, -2.5, -20.0)), G.rotate((1, 0, 0), -90.0), G.scale(360.0)), "lake_floor",
           prefix="lake_floor")
    c0, c1 = FAR["centre"]
    b.ply(b.assets.mesh(f"shore_far_band_v{ASSET_VERSION}_s3", lambda: N.backdrop_band(c0, c1, SKYLINE, FAR["y0"],
                                                                                    FAR["y_tex"], FAR["tile"])),
          EYE, "far_shore", prefix="backdrop")
    seeded = Assets(ctx.assets_dir, ctx.rebuild)
    built = {}

    def build_pebbles():
        built["meshes"], built["tops"] = pebbles(ctx)
        return built["meshes"]

    for name, path in sorted(seeded.meshes(f"shore_pebbles_s{ctx.seed}_v{ASSET_VERSION}", build_pebbles).items()):
        b.ply(path, EYE, name, prefix="pebbles")
    if "tops" not in built:                             # warm cache: the placements again, without the meshes
        built["meshes"], built["tops"] = pebbles(ctx)
    focus["pebbles"] = built["tops"]


# --------------------------------------------------------------------------
# Hero subjects, stones, plants and trees
# --------------------------------------------------------------------------
def tree(L: N.Library, rng, x, z, kind=None) -> None:
    kind = kind or ("tree_small_02" if rng.random() < 0.6 else "island_tree_02")
    h = rng.uniform(8.0, 12.0) if kind == "tree_small_02" else rng.uniform(5.0, 7.5)
    L.whole(kind, TERRAIN.at(x, z, -0.15), yaw=float(rng.uniform(0, 360)), height=h)


def add_hero(b: SceneBuilder, L: N.Library, batch: N.Batch, ctx: BuildContext, focus: dict) -> None:
    rng = np.random.default_rng(47)
    rocks = L.variants("rock_moss_set_01")
    big = int(np.argmax([h * w for _, h, w in rocks]))
    k, m, _ = L.matrix("rock_moss_set_01", big, TERRAIN.at(*BOULDER, -0.15), yaw=35.0, height=1.05)
    L.instance("rock_moss_set_01", k, m, tag="_granite_wet")
    focus["boulder"] = L.top_point("rock_moss_set_01", k, m)
    focus["beach_rocks"] = []
    for dx, dz, h in ((0.0, 0.0, 0.62), (0.75, 0.45, 0.36)):
        k, m, _ = L.matrix("rock_moss_set_01", int(rng.integers(6)), TERRAIN.at(BEACH_ROCKS[0] + dx, BEACH_ROCKS[1] + dz, -0.08),
                           yaw=float(rng.uniform(0, 360)), height=h)
        L.instance("rock_moss_set_01", k, m, tag="_granite")
        focus["beach_rocks"].append(L.top_point("rock_moss_set_01", k, m))

    # bleached driftwood log lying along the beach above the waterline
    (x0, z0), (x1, z1), r = LOG
    length, yaw = math.hypot(x1 - x0, z1 - z0), -math.degrees(math.atan2(z1 - z0, x1 - x0))
    y0 = min(TERRAIN.height(x0, z0), TERRAIN.height(x1, z1)) - 0.03
    M = G.compose(G.translate((x0, float(y0), z0)), G.rotate((0, 1, 0), yaw))
    b.ply(b.assets.mesh("shore_log", lambda: N.log(np.random.default_rng(53), length, r, sides=48, rings=60)), M,
          "driftwood", prefix="log")
    axis, radii = N.log_axis(np.random.default_rng(53), length, r, rings=60)
    focus["log"] = [(M @ np.append(N.log_top(axis, radii, u), 1.0))[:3].tolist() for u in (0.3, 0.5, 0.7)]

    # trees on the headland, fixed
    focus["headland"] = []
    hrng = np.random.default_rng(59)
    for x, z in N.scatter(hrng, HEADLAND, 10, 4.0):
        if land(x, z, 3.0):
            kind = "tree_small_02" if hrng.random() < 0.6 else "island_tree_02"
            h = hrng.uniform(8.0, 12.0) if kind == "tree_small_02" else hrng.uniform(5.0, 7.5)
            at = TERRAIN.at(x, z, -0.15)
            L.whole(kind, at, yaw=float(hrng.uniform(0, 360)), height=h)
            focus["headland"].append([at[0], at[1] + 1.0, at[2]])          # on the trunk


def add_scatter(L: N.Library, batch: N.Batch, ctx: BuildContext, focus: dict) -> list:
    rng = rng_for(ctx, 3)
    hero = [(BOULDER[0], BOULDER[1], 1.2), (BEACH_ROCKS[0], BEACH_ROCKS[1], 1.2)]
    (lx0, lz0), (lx1, lz1), _ = LOG
    log_box = ((min(lx0, lx1) - 0.5, min(lz0, lz1) - 0.5), (max(lx0, lx1) + 0.5, max(lz0, lz1) + 0.5))
    wet = ctx.env == "wet"
    # cobbles along the waterline (wet-coated) and up the beach
    cobbles = []
    for x, z in N.scatter(rng, ((-30.0, -8.0), (30.0, 8.0)), 160, 0.9, avoid=hero + [log_box]):
        d = z - float(shore(x))
        if -1.2 < d < 5.5:
            batch.plant("rock_moss_set_01", int(rng.integers(6)), TERRAIN.at(x, z, -0.05), yaw=float(rng.uniform(0, 360)),
                        height=rng.uniform(0.1, 0.3), tag="_granite_wet" if (d < 0.5 or wet) else "_granite")
            if abs(x) < 8:
                cobbles.append(L.top_point(*batch.last))
    focus["cobbles"] = cobbles
    # boulders along the headland's shore and in the shallows
    for x, z in N.scatter(rng, ((-50.0, -32.0), (-12.0, -4.0)), 50, 2.5):
        d = z - float(shore(x))
        if -2.0 < d < 3.0:
            batch.plant("rock_moss_set_01", int(rng.integers(6)), TERRAIN.at(x, z, -0.15), yaw=float(rng.uniform(0, 360)),
                        height=rng.uniform(0.4, 1.4), tag="_granite_wet")
    # grass along the berm and on the bank, shrubs on the bank, weeds in the upper beach
    for x, z in N.scatter(rng, ((-40.0, -30.0), (40.0, 30.0)), 420, 0.9, avoid=hero + [log_box]):
        d = z - float(shore(x))
        if 5.5 < d < 25.0:
            batch.plant("grass_medium_02", int(rng.integers(2)), TERRAIN.at(x, z, -0.02), yaw=float(rng.uniform(0, 360)),
                        height=rng.uniform(0.25, 0.55))
    for x, z in N.scatter(rng, ((-45.0, -30.0), (45.0, 40.0)), 120, 2.5, avoid=hero):
        if float(z - shore(x)) > 8.0:
            batch.plant("shrub_02" if rng.random() < 0.5 else "shrub_04", int(rng.integers(4)), TERRAIN.at(x, z, -0.05),
                        yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.6, 1.6))
    for x, z in N.scatter(rng, ((-25.0, -8.0), (25.0, 10.0)), 60, 1.5, avoid=hero + [log_box]):
        if 3.0 < float(z - shore(x)) < 7.0:
            batch.plant("weed_plant_02", int(rng.integers(5)), TERRAIN.at(x, z, -0.01), yaw=float(rng.uniform(0, 360)),
                        height=rng.uniform(0.06, 0.16))
    # trees on the bank and the hills behind
    placed = []
    for x, z in N.scatter(rng, ((-120.0, -70.0), (120.0, 90.0)), 260, 6.0, avoid=[HEADLAND, HERO_AREA]):
        if float(z - shore(x)) > 14.0:
            tree(L, rng, x, z)
            placed.append((x, z))
    return placed


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build(ctx: BuildContext) -> SceneBundle:
    b = SceneBuilder(Assets(ctx.shared_dir, ctx.rebuild))
    L = N.Library(b)
    focus: dict[str, list] = {}
    add_materials(b, L, ctx)
    add_ground(b, ctx, focus)
    batch = N.Batch(L)
    add_hero(b, L, batch, ctx, focus)
    trees = add_scatter(L, batch, ctx, focus)
    batch.flush(Assets(ctx.assets_dir, ctx.rebuild), f"shore_scatter_s{ctx.seed}_v{ASSET_VERSION}")
    sun = SUN if ctx.env == "clear" else None
    b.d["sky"] = env_kit.sky(ctx.env, sun, scale=SKY_SCALE[ctx.env])
    return SceneBundle(b.d, focus, {"sun_direction": sun, "sky_scale": SKY_SCALE[ctx.env], "trees": len(trees)})


def _exclusions():
    (lx0, lz0), (lx1, lz1), _ = LOG
    return (((BOULDER[0] - 1.0, -1.0, BOULDER[1] - 1.0), (BOULDER[0] + 1.0, 1.6, BOULDER[1] + 1.0)),
            ((BEACH_ROCKS[0] - 0.8, -1.0, BEACH_ROCKS[1] - 0.8), (BEACH_ROCKS[0] + 1.5, 1.2, BEACH_ROCKS[1] + 1.3)),
            ((min(lx0, lx1) - 0.4, -1.0, min(lz0, lz1) - 0.4), (max(lx0, lx1) + 0.4, 0.9, max(lz0, lz1) + 0.4)))


SCENE = SceneDef(
    id="rocky_shore", group="natural", owner="beiba",
    description="A rocky lake shore seen from the beach: pebbles and cobbles down to rippled water, a granite "
                "boulder at the waterline, a bleached driftwood log, a wooded headland and the far shore across the lake.",
    build=build,
    views={
        # Low on the pebbles, the boulder at the waterline, the lake and far shore behind.
        "boulder": View((2.4, 0.55, float(shore(2.4)) + 3.2), (BOULDER[0], 0.4, BOULDER[1]), focus="boulder"),
        # Looking down at the pebbles at the water's edge.
        "pebbles": View((-1.2, 0.5, float(shore(-1.2)) + 2.0), (-1.0, 0.0, float(shore(-1.0)) + 0.5), focus="pebbles"),
        # Standing on the beach, focused on the headland's trees across the water.
        "headland": View((-1.5, 1.6, float(shore(-1.5)) + 4.5), (-30.0, 3.0, -22.0), focus="headland"),
        # From up the beach past the driftwood log to the waterline and the lake, the log in front.
        "log": View((4.2, 1.0, float(shore(4.2)) + 6.6), (3.0, 0.1, -0.6), focus="log"),
    },
    default_view="boulder",
    camera_box=((-8.0, 0.35, -0.5), (8.0, 1.8, 6.5)),
    target_box=((-45.0, -0.6, -120.0), (40.0, 4.0, -0.5)),
    exclude_boxes=_exclusions(),
    lens_mm=(24.0, 135.0), f_number=(1.4, 16.0),
    envs=("clear", "overcast", "wet"),
    tags=("outdoor", "day", "water", "open", "foliage", "specular", "bokeh", "vista"),
    default_seed=1, asset_version=ASSET_VERSION, max_depth=8, rr_depth=5, spp_hint=2048, far_clip=2000.0,
)
