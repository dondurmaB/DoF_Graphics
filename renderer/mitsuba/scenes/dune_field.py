"""A sand dune field in the late afternoon, seen from an interdune hollow: wind ripples at the cameras' feet,
dry grass tufts and a desert shrub, a bleached dead stump and branches, a dune crest ahead against the sky and
more ridges behind it to the horizon.

Natural scene (owner beiba). Built for low-texture depth of field: smooth sand at every depth with little texture
beyond the ripples, sharp dune crests against a large sky (the in-focus-silhouette-against-sky case), subjects at
0.3-3 m (grass, ripples, branches), 4-10 m (shrub, stump) and 25-60 m (the first crest). The sun is low (25
degrees) behind and to the right of the hero cameras, so the ripples and the dunes' lee faces are cross-lit with
long shadows, and it never enters a frame the target box allows.

Layout, meters, y up, the hero cameras look toward -z:
  dunes        transverse dunes 55 m apart, up to ~7 m high, crests running roughly along x, with a gentle windward
               slope and a steep lee face, crossed by smaller dunes 23 m apart; flattened round the cameras into
               an interdune hollow; one graded mesh (15 cm cells round the cameras, up to 4 m at 450 m) so near
               sand and far crests share one surface with no seams
  sand         a procedural albedo (warm quartz sand, slow tone drift, grain speckle) and a procedural normal map
               of wind ripples 9 cm apart (4 mm deep), so the ripples cost no triangles
  plants       a desert shrub at (1.6, -3.8), dry grass tufts and small weeds across the hollow and the lower
               slopes, a few shrubs further out
  dead wood    a bleached stump at (-2.8, -9.0), ~25 bleached branches half buried in the hollow
  horizon      a band of far dunes at 420 m (opaque, all geometry, so depth and image agree)
Light is only `env_kit`'s sun and sky.

Environments: clear; overcast; sand (a lower, dust-hazed sun: env_kit's turbidity 5 sky). Seed (`ctx.seed`) moves
the grass, weeds, branches and outer shrubs; the dunes, the hero shrub and the stump keep their places.

Cost: scattered plants and branches are merged per material (`_nature_kit.Batch` or here directly). No
alpha-masked geometry.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

import env_kit
import procedural as G
import web_assets as W
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, bitmap, principled
from scenes import _greenhouse_props as H
from scenes import _nature_kit as N

ASSET_VERSION = 1
EYE = np.eye(4)
SUN = {"clear": env_kit.default_sun_direction(25.0, 75.0), "sand": env_kit.default_sun_direction(19.0, 70.0)}
# Bright open sand; dry_riverbed's clear 5 / overcast 15 / sand 4 gave frame medians 0.15-0.25 on darker ground.
# Calibrate on check_scene.
SKY_SCALE = {"clear": 4.0, "overcast": 12.0, "sand": 3.5}
SAND_TILE_M = 3.0
RIPPLE_TILE_M = 2.0
SAND = (0.42, 0.29, 0.16)                                # linear albedo of the quartz sand

HOLLOW = (0.0, 2.0, 7.0, 20.0)                           # centre x, z; flat within 7 m, dunes full beyond 20 m
SHRUB = (1.6, -3.8)
STUMP = (-2.8, -9.0)
FAR = {"centre": (0.0, 0.0), "radius": 420.0, "y0": -4.0, "y_tex": 16.0, "tile": 60.0}
SKYLINE = N.Skyline(83, FAR["radius"], base=8.0, spread=6.0)


def rng_for(ctx: BuildContext, k: int) -> np.random.Generator:
    return np.random.default_rng([ctx.seed, 5179, k])


# --------------------------------------------------------------------------
# Dunes
# --------------------------------------------------------------------------
def _profile(phi, crest=0.72):
    """One dune across a wavelength, 0 in the trough, 1 at the crest: a long windward rise, a short steep lee."""
    phi = np.mod(phi, 1.0)
    up = (phi / crest) ** 1.6
    down = ((1.0 - phi) / (1.0 - crest)) ** 1.25
    return np.where(phi <= crest, up, down)


def height(x, z):
    x, z = np.asarray(x, float), np.asarray(z, float)
    warp = 4.0 * np.sin(x / 37.0 + 0.6) + 2.5 * np.sin(x / 13.0 - z / 41.0)
    a1 = 7.0 * (0.75 + 0.25 * np.sin(x / 83.0 + 1.3))
    h = a1 * _profile((-(z + 0.22 * x + warp)) / 55.0 + 0.18)
    h = h + 1.4 * _profile((-(0.6 * z - 0.8 * x + 0.6 * warp)) / 23.0 + 0.5, crest=0.65)
    h = h + 0.25 * np.sin(x / 9.0 + z / 17.0) + 0.12 * np.sin(z / 6.1 - x / 11.0)
    cx, cz, r0, r1 = HOLLOW
    t = np.clip((np.hypot(x - cx, z - cz) - r0) / (r1 - r0), 0.0, 1.0)
    return h * (0.12 + 0.88 * t * t * (3 - 2 * t))


def _axis(lo, hi, f_lo, f_hi, step, growth=1.04, cap=4.0):
    """Coordinates from lo to hi: `step` apart inside [f_lo, f_hi], growing by `growth` per cell outside, up to cap."""
    mid = list(np.arange(f_lo, f_hi + step / 2, step))
    out_hi, s, v = [], step, f_hi
    while v < hi:
        s = min(s * growth, cap)
        v += s
        out_hi.append(min(v, hi))
    out_lo, s, v = [], step, f_lo
    while v > lo:
        s = min(s * growth, cap)
        v -= s
        out_lo.append(max(v, lo))
    return np.array(out_lo[::-1] + mid + out_hi)


def dune_mesh() -> G.Mesh:
    xs = _axis(-450.0, 450.0, -15.0, 15.0, 0.15)
    zs = _axis(-450.0, 250.0, -22.0, 8.0, 0.15)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    p = np.stack([X, height(X, Z), Z], -1).reshape(-1, 3)
    f = G._grid_faces(len(xs), len(zs))
    uv = np.stack([X, Z], -1).reshape(-1, 2)                # metres
    return N._upward(G.Mesh(p, G.vertex_normals(p, f), uv, f))


def crest_points(xs=(-12.0, -6.0, 0.0, 6.0, 12.0)) -> list:
    """The first crest ahead of the hollow at each x: the highest terrain point for z in -70..-12."""
    zz = np.linspace(-70.0, -12.0, 1200)
    return [[float(x), float(height(x, zz[np.argmax(height(x, zz))])), float(zz[np.argmax(height(x, zz))])] for x in xs]


# --------------------------------------------------------------------------
# Textures
# --------------------------------------------------------------------------
def sand_albedo(px: int = 1024, seed: int = 11) -> np.ndarray:
    """sRGB sand albedo tiling every SAND_TILE_M: slow tone drift, darker heavy-mineral streaks, grain speckle."""
    o = px // 8
    n = px + o
    tone = G.fbm(n, n, octaves=5, base=3, seed=seed)
    streak = G.fbm(n, n, octaves=4, base=6, seed=seed + 1, aspect=4.0)
    grain = G.fbm(n, n, octaves=2, base=300, seed=seed + 2)
    lin = np.array(SAND) * (0.88 + 0.22 * tone[..., None]) * (1.0 - 0.18 * np.clip(streak - 0.6, 0, 1)[..., None] / 0.4)
    lin = lin * (0.9 + 0.2 * grain[..., None])
    a = N._periodic(N._srgb(lin).transpose(2, 0, 1).reshape(3 * n, n), o).reshape(3, n, px)
    a = N._periodic(a.transpose(0, 2, 1).reshape(3 * px, n), o).reshape(3, px, px)
    return a.transpose(2, 1, 0).astype(np.float32)


def ripple_normals(px: int = 1024, depth: float = 0.004) -> np.ndarray:
    """Tangent-space normal map (encoded 0..1) of wind ripples over one RIPPLE_TILE_M tile, periodic: crests
    about 9 cm apart, a gentle stoss side and a steeper lee, the crest lines meandering."""
    u, v = np.meshgrid(np.arange(px) / px, np.arange(px) / px, indexing="xy")
    warp = 0.6 * np.sin(math.tau * (2 * u + v)) + 0.35 * np.sin(math.tau * (3 * v - u) + 1.0) \
        + 0.2 * np.sin(math.tau * (7 * u + 4 * v) + 2.0)
    phase = 8 * u + 20 * v + warp                       # 8 x 20 cycles: ~9 cm apart over the 2 m tile
    h = depth * _profile(phase, crest=0.7)
    mod = 0.65 + 0.35 * np.sin(math.tau * (u - 2 * v) + 0.5) ** 2      # ripples fade and swell
    h = h * mod
    step = RIPPLE_TILE_M / px
    dhu = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * step)
    dhv = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * step)
    n = np.stack([-dhu, -dhv, np.ones_like(h)], -1)    # rows run with v (Mitsuba: v = 0 is the top row)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (0.5 * n + 0.5).astype(np.float32)


def bleached(src: str) -> np.ndarray:
    """sRGB albedo for sun-bleached dead wood: the scan's albedo to its luminance, lifted to a pale grey-beige."""
    from PIL import Image

    a = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255.0
    lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.clip((0.25 + 0.9 * lum)[..., None] * np.array([1.0, 0.95, 0.86], np.float32), 0.0, 1.0)


def straw(src: str) -> np.ndarray:
    """sRGB albedo for dry grass: the scan's green turned to its luminance and tinted straw."""
    from PIL import Image

    a = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255.0
    lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.clip((0.1 + 1.3 * lum)[..., None] * np.array([1.0, 0.86, 0.55], np.float32), 0.0, 1.0)


def far_albedo(seed: int, tile_w: float, y0: float, y_top: float, ppm: int = 16) -> np.ndarray:
    """sRGB albedo tile for the far dune band: sand, lit faces and shaded lee faces in broad soft patches."""
    W_, H_ = int(tile_w * ppm), int((y_top - y0) * ppm)
    O = W_ // 6
    shade = G.fbm(H_, W_ + O, octaves=4, base=3, seed=seed, aspect=(W_ + O) / H_)
    lin = np.array(SAND)[None, None, :] * (0.55 + 0.6 * shade[..., None])
    out = N._periodic(N._srgb(lin).transpose(2, 0, 1).reshape(3 * H_, W_ + O), O).reshape(3, H_, W_).transpose(1, 2, 0)
    return out.astype(np.float32)


def retexture(b: SceneBuilder, bsdf: dict, fn, name: str) -> dict:
    """A scan's BSDF with its albedo bitmap passed through fn (cached as <name>_<file stem>). Walks down the
    wrappers web_assets puts round it (mask, normalmap, twosided) to the BSDF that has the albedo."""
    inner = bsdf
    while "base_color" not in inner and isinstance(inner.get("bsdf"), dict):
        inner = inner["bsdf"]
    bc = inner.get("base_color")
    if isinstance(bc, dict) and bc.get("type") == "bitmap":
        src = bc["filename"]
        inner["base_color"] = dict(bc, filename=b.assets.texture(f"{name}_{Path(src).stem}", lambda: fn(src)))
    return bsdf


# --------------------------------------------------------------------------
# Materials and ground
# --------------------------------------------------------------------------
def add_materials(b: SceneBuilder, L: N.Library) -> None:
    alb = b.assets.texture(f"dune_sand_albedo_v{ASSET_VERSION}", sand_albedo)
    nrm = b.assets.texture(f"dune_ripples_v{ASSET_VERSION}", ripple_normals)
    sand = principled(bitmap(alb, uv_scale=(1 / SAND_TILE_M,) * 2), 0.95, specular=0.3)
    b.material("sand", {"type": "normalmap", "normalmap": bitmap(nrm, raw=True, uv_scale=(1 / RIPPLE_TILE_M,) * 2),
                        "bsdf": sand})
    far = b.assets.texture(f"dune_far_albedo_v{ASSET_VERSION}", lambda: far_albedo(29, FAR["tile"], FAR["y0"], FAR["y_tex"]))
    b.material("far_dunes", {"type": "twosided", "bsdf": principled(bitmap(far), 0.95)})
    b.material("deadwood", W.surface("bark_brown_02", tint=(1.45, 1.38, 1.25)))
    L.material_set("grass_medium_02", "_straw", lambda bsdf: retexture(b, bsdf, straw, "straw"))


def add_ground(b: SceneBuilder) -> None:
    b.ply(b.assets.mesh(f"dune_terrain_v{ASSET_VERSION}", dune_mesh), EYE, "sand", prefix="ground")
    c0, c1 = FAR["centre"]
    b.ply(b.assets.mesh(f"dune_far_band_v{ASSET_VERSION}", lambda: N.backdrop_band(c0, c1, SKYLINE, FAR["y0"],
                                                                                   FAR["y_tex"], FAR["tile"])),
          EYE, "far_dunes", prefix="backdrop")


def at(x, z, dy=0.0) -> list[float]:
    return [float(x), float(height(x, z)) + dy, float(z)]


# --------------------------------------------------------------------------
# Plants and dead wood
# --------------------------------------------------------------------------
def branches(ctx: BuildContext) -> G.Mesh:
    """~25 bleached branches lying half buried in the hollow: bent swept tubes with one side shoot each."""
    rng = rng_for(ctx, 5)
    out = []
    for x, z in N.scatter(rng, ((-9.0, -16.0), (9.0, 6.0)), 25, 1.6, avoid=[(SHRUB[0], SHRUB[1], 1.2),
                                                                           (STUMP[0], STUMP[1], 1.2)]):
        length, r, yaw = rng.uniform(0.3, 1.2), rng.uniform(0.008, 0.03), rng.uniform(0, math.tau)
        s = np.linspace(0, 1, 8)
        side = np.cumsum(rng.normal(0, 0.05 * length, 8))
        px = x + s * length * math.cos(yaw) - side * math.sin(yaw)
        pz = z + s * length * math.sin(yaw) + side * math.cos(yaw)
        path = np.stack([px, height(px, pz) + 0.3 * r, pz], -1)
        out.append(G.sweep(path, r * (1 - 0.6 * s), sides=10))
        k = int(rng.integers(2, 6))
        a = yaw + rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
        l2 = length * rng.uniform(0.25, 0.5)
        qx = px[k] + np.linspace(0, 1, 5) * l2 * math.cos(a)
        qz = pz[k] + np.linspace(0, 1, 5) * l2 * math.sin(a)
        out.append(G.sweep(np.stack([qx, height(qx, qz) + 0.3 * r, qz], -1), 0.6 * r * (1 - 0.7 * np.linspace(0, 1, 5)), sides=8))
    return N._concat(out)


def add_plants(b: SceneBuilder, L: N.Library, batch: N.Batch, ctx: BuildContext, focus: dict) -> None:
    # hero shrub and stump, fixed
    batch.plant("shrub_04", 0, at(*SHRUB, -0.04), yaw=40.0, height=0.85)
    focus["shrub"] = L.top_point(*batch.last)
    m = W.model("tree_stump_01")
    M = W.place(m, at=at(*STUMP, -0.12), yaw=110.0, height=0.7)
    W.add(b, m, M, prefix="web_tree_stump_01_bleached",
          overrides={p.name: retexture(b, copy.deepcopy(p.bsdf), bleached, "bleached") for p in m.parts})
    v = np.concatenate([H.read_ply(p.ply).p for p in m.parts])
    w = (np.c_[v, np.ones(len(v))] @ np.asarray(M).T)[:, :3]
    focus["stump"] = w[np.argmax(w[:, 1])].tolist()           # the highest vertex: a real surface point

    rng = rng_for(ctx, 3)
    hero = [(SHRUB[0], SHRUB[1], 1.3), (STUMP[0], STUMP[1], 1.3)]
    tufts = []
    for x, z in N.scatter(rng, ((-30.0, -40.0), (30.0, 12.0)), 260, 1.1, avoid=hero):
        near = math.hypot(x, z - 2.0) < 12.0
        batch.plant("grass_medium_02", int(rng.integers(2)), at(x, z, -0.02), yaw=float(rng.uniform(0, 360)),
                    height=rng.uniform(0.15, 0.45), tag="_straw")
        if near and -6.0 < z < 3.0 and abs(x) < 5.0:
            tufts.append(L.top_point(*batch.last))
    focus["grass"] = tufts
    for x, z in N.scatter(rng, ((-20.0, -25.0), (20.0, 10.0)), 70, 1.6, avoid=hero):
        batch.plant("weed_plant_02", int(rng.integers(5)), at(x, z, -0.01), yaw=float(rng.uniform(0, 360)),
                    height=rng.uniform(0.06, 0.18))
    for x, z in N.scatter(rng, ((-90.0, -150.0), (90.0, 40.0)), 140, 5.0, avoid=hero + [((-9.0, -12.0), (9.0, 8.0))]):
        if height(x, z) < 3.5:                           # hollows and lower slopes, not the crests
            batch.plant("shrub_04" if rng.random() < 0.6 else "shrub_03", int(rng.integers(4)), at(x, z, -0.05),
                        yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.4, 1.1))
    seeded = Assets(ctx.assets_dir, ctx.rebuild)
    b.ply(seeded.mesh(f"dune_branches_s{ctx.seed}_v{ASSET_VERSION}", lambda: branches(ctx)), EYE, "deadwood",
          prefix="branches")
    # sand in front of the hollow's cameras (real surface points for close focus on the ripples)
    focus["sand"] = [at(x, z) for x, z in ((-1.5, 0.5), (0.0, -0.5), (1.5, 0.0), (-0.5, -2.0), (0.8, -1.5))]
    focus["crest"] = crest_points()


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build(ctx: BuildContext) -> SceneBundle:
    b = SceneBuilder(Assets(ctx.shared_dir, ctx.rebuild))
    L = N.Library(b)
    focus: dict[str, list] = {}
    add_materials(b, L)
    add_ground(b)
    batch = N.Batch(L)
    add_plants(b, L, batch, ctx, focus)
    batch.flush(Assets(ctx.assets_dir, ctx.rebuild), f"dune_scatter_s{ctx.seed}_v{ASSET_VERSION}")
    sun = SUN.get(ctx.env)
    b.d["sky"] = env_kit.sky(ctx.env, sun, scale=SKY_SCALE[ctx.env])
    return SceneBundle(b.d, focus, {"sun_direction": sun, "sky_scale": SKY_SCALE[ctx.env]})


def _exclusions():
    return (((SHRUB[0] - 1.0, -1.0, SHRUB[1] - 1.0), (SHRUB[0] + 1.0, 1.4, SHRUB[1] + 1.0)),
            ((STUMP[0] - 0.9, -1.0, STUMP[1] - 0.9), (STUMP[0] + 0.9, 1.2, STUMP[1] + 0.9)))


SCENE = SceneDef(
    id="dune_field", group="natural", owner="beiba",
    description="A sand dune field in the late afternoon from an interdune hollow: wind ripples, dry grass and a "
                "desert shrub, a bleached stump and branches, dune crests against a large sky.",
    build=build,
    views={
        # Low over the ripples, dry grass in focus, the first crest soft behind.
        "ripples": View((0.4, 0.35, 3.5), (0.0, 0.0, 0.5), focus="sand"),
        # The shrub at ~6 m, the dune behind it.
        "shrub": View((-0.5, 1.0, 3.0), (SHRUB[0], 0.4, SHRUB[1]), focus="shrub"),
        # The bleached stump, crests behind.
        "stump": View((0.5, 0.7, 2.0), (STUMP[0], 0.35, STUMP[1]), focus="stump"),
        # Standing, focused on the first crest against the sky.
        "crest": View((0.0, 1.7, 4.0), (0.0, 4.0, -35.0), focus="crest"),
    },
    default_view="crest",
    camera_box=((-6.0, 0.3, -2.0), (6.0, 1.8, 8.0)),
    # at most ~52 degrees of yaw from -z: with a 42-degree half field the frame stays clear of the sun (105 degrees
    # round from -z, 25 up)
    target_box=((-12.0, -0.5, -200.0), (12.0, 8.0, -6.0)),
    exclude_boxes=_exclusions(),
    lens_mm=(24.0, 135.0), f_number=(1.4, 16.0),
    envs=("clear", "overcast", "sand"),
    tags=("outdoor", "day", "sand", "open", "bokeh", "vista"),
    default_seed=1, asset_version=ASSET_VERSION, max_depth=8, rr_depth=5, spp_hint=2048, far_clip=2000.0,
)
