"""An alpine cirque above the tree line, seen from a meadow bench: flowers and grass at the cameras' feet, a
lichen-grey boulder and a stone cairn on the bench, a talus cone of angular scree rising under a cliff band, the
ridge's peaks behind against a large sky.

Natural scene (owner beiba). Built for depth of field with no trees anywhere: subjects at 0.3-2 m (flowers,
grass, scree stones), 4-8 m (boulder, cairn), 25-60 m (the talus, its blocks against the sky where the cameras sit
low) and 130-450 m (the cliff band, the peaks). Rock texture at every depth and hard rock silhouettes against
sky, unlike the wooded scenes. The sun is behind and to the right of the hero cameras, so it never enters a
level frame the target box allows, and the talus is side-lit with sharp block shadows.

Layout, meters, y up, the hero cameras look up the cirque toward -z:
  bench        an alpine meadow (`leafy_grass`) rising 3 cm per metre, grass tufts, periwinkle and gazania
               clumps and small weeds
  talus        from ~22 m ahead the slope steepens into a talus cone at ~28 degrees up to ~20 m high at 55 m
               (`acg:Granite006B` turned grey: the texture is a red granite), angular procedural scree stones on
               its lower part and scanned blocks (opaque, recoloured to lichen-grey granite) all over it; valley
               sides rising to both sides
  cliff        a cliff band (49 degrees) above the talus up to the cirque's rim, ~35 m high at 68 m, ~25 degrees
               up from the cameras, then a gentle plateau (darker granite); the terrain is cut along two contours
               (meadow below 1.2 m, cliff above 21 m) so material edges are smooth lines
  peaks        a band of rock and snow peaks ~480 m away, 80-290 m high, the highest rising above the rim with
               sky above them (at 160-360 m they filled the sky: 7% sky in the talus view); all geometry, so depth
               and image agree
  subjects     the boulder at (-1.6, -4.2), the cairn at (2.4, -6.5), a flower patch at (0.6, -0.8)
Light is only `env_kit`'s sun and sky.

Environments: clear; overcast; snow (fresh snow on the bench and talus, a lower sun). Seed (`ctx.seed`) moves the
grass, flowers, weeds, scree and blocks; the boulder, the cairn and the terrain keep their places.

Cost: scree, blocks and plants are merged per material (`_nature_kit.Batch` or here directly). No alpha-masked
geometry, no trees.
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
SUN = {"clear": env_kit.default_sun_direction(42.0, 45.0), "snow": env_kit.default_sun_direction(30.0, 50.0)}
# Open and rocky, between dry_riverbed (clear 5, overcast 15) and stream's snow (5.5); calibrate on check_scene.
SKY_SCALE = {"clear": 6.0, "overcast": 18.0, "snow": 4.5}
MEADOW_TOP = 1.2                     # meadow below this height, talus above
CLIFF_FROM = 21.0                    # cliff rock above this height (the talus tops out at ~20 m, 55 m ahead)
GRANITE_TINT = {"talus": (0.85, 0.85, 0.88), "cliff": (0.55, 0.55, 0.58)}
SCREE_TONES = {"scree_grey": (0.9, 0.9, 0.93), "scree_dark": (0.55, 0.55, 0.58), "scree_warm": (1.05, 0.95, 0.85)}

BOULDER = (-1.6, -4.2)
CAIRN = (2.4, -6.5)
FLOWERS = (0.6, -0.8)
PEAKS = {"centre": (0.0, -60.0), "radius": 420.0, "y0": 30.0, "y_tex": 330.0, "tile": 120.0}


def rng_for(ctx: BuildContext, k: int) -> np.random.Generator:
    return np.random.default_rng([ctx.seed, 4421, k])


def _softplus(v, k):
    return k * np.logaddexp(0.0, np.asarray(v, float) / k)


def height(x, z):
    """Bench, then the talus cone, the cliff band and the rim, with valley sides rising to both sides."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    d = -z
    h = 0.03 * d + 0.12 * _softplus(d - 12.0, 3.0) + 0.38 * _softplus(d - 22.0, 4.0) - 0.38 * _softplus(d - 55.0, 3.0)
    h = h + 1.0 * _softplus(d - 55.0, 3.0) - 1.0 * _softplus(d - 68.0, 2.0)            # cliff band, then the rim
    h = h + 0.0035 * x * x + 0.35 * _softplus(np.abs(x) - 28.0, 4.0)
    h = h + 0.35 * np.sin(x / 7.0 + d / 13.0) * np.clip(d / 20.0, 0.0, 1.0) + 0.12 * np.sin(x / 2.9 - d / 4.1)
    return h


def at(x, z, dy=0.0) -> list[float]:
    return [float(x), float(height(x, z)) + dy, float(z)]


class CirqueTerrain(N.Terrain):
    def height(self, x, z):
        return height(x, z)


TERRAIN = CirqueTerrain(-160.0, 160.0, -170.0, 40.0, spacing=0.4)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def terrain_parts() -> dict[str, G.Mesh]:
    """{'meadow', 'talus', 'cliff'}: the terrain cut exactly along two contours (`_stream_props.cut_at_height`)."""
    meadow, rest = SP.cut_at_height(TERRAIN.mesh(), MEADOW_TOP)
    talus, cliff = SP.cut_at_height(rest, CLIFF_FROM)
    return {"meadow": meadow, "talus": talus, "cliff": cliff}


def scree_stone(rng, size: float) -> G.Mesh:
    """An angular scree block about `size` across: a coarse lathe (5-7 sides, 4 rings), each vertex jittered,
    flattened, flat-shaded (each face its own normal) so the facets read as fractured rock. ~50 triangles."""
    t = np.linspace(-math.pi / 2, math.pi / 2, 4)
    m = G.lathe(np.stack([np.cos(t), np.sin(t)], 1) * 0.5, segments=int(rng.integers(5, 8)))
    p = m.p * [rng.uniform(0.7, 1.0), rng.uniform(0.35, 0.7), rng.uniform(0.6, 0.95)] * size
    p = p + rng.normal(0.0, 0.08 * size, p.shape)
    p[:, 1] -= p[:, 1].min()
    tri = p[m.f].reshape(-1, 3)
    f = np.arange(len(tri)).reshape(-1, 3)
    n = np.cross(tri[1::3] - tri[0::3], tri[2::3] - tri[0::3])
    n = np.repeat(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12), 3, 0)
    return G.Mesh(tri, n, m.uv[m.f].reshape(-1, 2), f).oriented()


def scree(ctx: BuildContext) -> tuple[dict[str, G.Mesh], list]:
    """{'scree_<tone>'}: angular stones over the talus foot and the upper bench (denser and bigger up the slope),
    partly sunk; plus the top points of the larger ones near the hero cameras."""
    rng = rng_for(ctx, 4)
    acc: dict[str, list] = {}
    tops = []
    tones = list(SCREE_TONES)
    cell = 0.7
    for x, z in ((gx + rng.uniform(0, cell), gz + rng.uniform(0, cell))         # jittered grid, one stone a cell
                 for gx in np.arange(-30.0, 30.0, cell) for gz in np.arange(-60.0, -6.0, cell)):
        d = -z
        if rng.random() > np.clip((d - 8.0) / 20.0, 0.08, 1.0) * 0.8:
            continue
        size = float(np.exp(rng.uniform(math.log(0.12), math.log(0.3 + 0.6 * np.clip((d - 15) / 30, 0, 1)))))
        m = scree_stone(rng, size)
        h = float(m.p[:, 1].max())
        M = G.compose(G.translate(at(x, z, -0.3 * h)), G.rotate((0, 1, 0), rng.uniform(0, 360)),
                      G.rotate((1, 0, 0), rng.uniform(-15, 15)), G.rotate((0, 0, 1), rng.uniform(-15, 15)))
        acc.setdefault(tones[int(rng.choice(3, p=[0.55, 0.3, 0.15]))], []).append(m.transformed(M))
        if d < 26 and abs(x) < 10 and size > 0.25:
            tops.append((M @ np.array([0.0, h, 0.0, 1.0]))[:3].tolist())
    return {k: N._concat(v) for k, v in sorted(acc.items())}, tops


def cairn() -> tuple[G.Mesh, list]:
    """A stacked-stone cairn ~0.9 m high: eight flattened stones, each smaller and slightly off the one below."""
    rng = np.random.default_rng(67)
    parts, y = [], 0.0
    for k in range(8):
        size = 0.5 * (1.0 - 0.09 * k) * rng.uniform(0.9, 1.05)
        t = np.linspace(-math.pi / 2, math.pi / 2, 7)
        m = G.lathe(np.stack([np.cos(t), np.sin(t)], 1) * 0.5, segments=16)
        p = m.p * [size, size * rng.uniform(0.22, 0.3), size * rng.uniform(0.75, 0.95)]
        p = p + rng.normal(0.0, 0.012 * size, p.shape)
        p[:, 1] -= p[:, 1].min()
        stone = G.Mesh(p, G.vertex_normals(p, m.f), m.uv, m.f).oriented()
        off = rng.normal(0.0, 0.025, 2)
        parts.append(stone.transformed(G.compose(G.translate((off[0], y, off[1])), G.rotate((0, 1, 0), rng.uniform(0, 360)))))
        y += float(p[:, 1].max()) * 0.92
    return N._concat(parts), [0.0, y, 0.0]


class PeakSkyline:
    """Ridge line of the far peaks along a ring: long-wavelength rises plus ridged (|sin|-folded) terms, so the
    summits are sharp and the cols rounded. Whole cycles round the ring, so it closes."""

    def __init__(self, seed, radius, base=185.0):
        rng = np.random.default_rng(seed)
        self.radius, self.base = radius, base
        lam = np.array([900.0, 380.0, 160.0, 60.0, 22.0])
        self.amp = np.array([45.0, 35.0, 16.0, 6.0, 2.0])
        circ = math.tau * radius
        self.k = np.round(circ / lam) * math.tau / circ
        self.ph = rng.uniform(0, math.tau, len(lam))

    def top(self, theta):
        s = np.asarray(theta)[..., None] * self.radius
        ridged = 1.0 - 2.0 * np.abs(np.sin(0.5 * (self.k * s + self.ph)))
        return self.base + (self.amp * ridged).sum(-1)


def peaks_albedo(seed: int, tile_w: float, y0: float, y_top: float, ppm: int = 4) -> np.ndarray:
    """sRGB albedo tile for the peak band: grey rock with darker gullies, snow above a ragged snow line and in
    the gullies below it. Tiles in u."""
    W_, H_ = int(tile_w * ppm), int((y_top - y0) * ppm)
    O = W_ // 6
    aspect = (W_ + O) / H_
    y = y_top - (np.arange(H_)[:, None] + 0.5) / ppm
    rock = G.fbm(H_, W_ + O, octaves=5, base=4, seed=seed, aspect=aspect)
    gully = G.fbm(H_, W_ + O, octaves=3, base=10, seed=seed + 1, aspect=aspect * 0.25)
    lin = np.array([0.075, 0.074, 0.072])[None, None, :] * (0.7 + 0.6 * rock[..., None])
    lin = lin * (1.0 - 0.45 * (gully > 0.62)[..., None])
    line = 175.0 + 30.0 * (G.fbm(H_, W_ + O, octaves=4, base=6, seed=seed + 2, aspect=aspect) - 0.5)
    snow = (y > line) | ((gully > 0.66) & (y > line - 60.0))
    lin[snow] = np.array([0.62, 0.64, 0.68])
    out = N._periodic(N._srgb(lin).transpose(2, 0, 1).reshape(3 * H_, W_ + O), O).reshape(3, H_, W_).transpose(1, 2, 0)
    return out.astype(np.float32)


def lichen_granite(src: str) -> np.ndarray:
    """sRGB albedo for the scanned blocks: the scan's moss turned to its luminance, cool grey with a faint
    yellow-green lichen cast where the moss was brightest."""
    from PIL import Image

    a = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255.0
    lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    green = np.clip(a[..., 1] - 0.5 * (a[..., 0] + a[..., 2]), 0.0, 1.0)[..., None]
    grey = (0.1 + 0.9 * lum)[..., None] * np.array([1.0, 1.0, 1.03], np.float32)
    return np.clip(grey + 0.6 * green * np.array([0.25, 0.3, 0.0], np.float32), 0.0, 1.0)


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
def granite_block(b: SceneBuilder, bsdf: dict, snow: bool) -> dict:
    s = N.stone(bsdf, wet=False)
    inner = s["bsdf"] if s["type"] == "normalmap" else s
    bc = inner["base_color"]
    if isinstance(bc, dict) and bc.get("type") == "bitmap":
        src = bc["filename"]
        inner["base_color"] = dict(bc, filename=b.assets.texture(f"lichen_granite_{Path(src).stem}", lambda: lichen_granite(src)))
    return s


def add_materials(b: SceneBuilder, L: N.Library, ctx: BuildContext) -> None:
    snow = ctx.env == "snow"
    b.material("meadow", env_kit.ground("snow") if snow else W.surface("leafy_grass", tint=(0.85, 0.85, 0.75)))
    b.material("talus", env_kit.ground("snow") if snow else granite_surface(b, GRANITE_TINT["talus"]))
    b.material("cliff", granite_surface(b, GRANITE_TINT["cliff"]))
    for name, tint in SCREE_TONES.items():
        b.material(name, granite_surface(b, tint, uv_scale=(0.25, 0.25)))
    b.material("cairn", granite_surface(b, (0.8, 0.8, 0.82), uv_scale=(0.3, 0.3)))
    alb = b.assets.texture(f"peaks_albedo_v{ASSET_VERSION}_b185", lambda: peaks_albedo(37, PEAKS["tile"], PEAKS["y0"], PEAKS["y_tex"]))
    b.material("peaks", {"type": "twosided", "bsdf": principled(bitmap(alb), 0.9)})
    L.material_set("rock_moss_set_01", "_granite", lambda bsdf: granite_block(b, bsdf, snow))


def add_ground(b: SceneBuilder, ctx: BuildContext, focus: dict) -> None:
    parts = b.assets.meshes(f"cirque_terrain_v{ASSET_VERSION}", terrain_parts)
    for name in ("meadow", "talus", "cliff"):
        b.ply(parts[name], EYE, name, prefix=name)
    c0, c1 = PEAKS["centre"]
    sky = PeakSkyline(19, PEAKS["radius"])
    b.ply(b.assets.mesh(f"cirque_peaks_v{ASSET_VERSION}_b185", lambda: N.backdrop_band(c0, c1, sky, PEAKS["y0"], PEAKS["y_tex"],
                                                                                  PEAKS["tile"], step=0.5)),
          EYE, "peaks", prefix="backdrop")
    seeded = Assets(ctx.assets_dir, ctx.rebuild)
    meshes, tops = scree(ctx)
    for name, path in sorted(seeded.meshes(f"cirque_scree_s{ctx.seed}_v{ASSET_VERSION}", lambda: meshes).items()):
        b.ply(path, EYE, name, prefix="scree")
    focus["scree"] = tops
    # the rim of the talus where it meets the cliff, and the cliff top: real terrain points
    focus["talus"] = [at(x, z) for x, z in ((-8.0, -38.0), (0.0, -45.0), (9.0, -41.0))]
    focus["rim"] = [at(x, -68.0) for x in (-20.0, 0.0, 20.0)]


def add_subjects(b: SceneBuilder, L: N.Library, batch: N.Batch, ctx: BuildContext, focus: dict) -> None:
    rocks = L.variants("rock_moss_set_01")
    big = int(np.argmax([h * w for _, h, w in rocks]))
    k, m, _ = L.matrix("rock_moss_set_01", big, at(*BOULDER, -0.12), yaw=200.0, height=0.95)
    L.instance("rock_moss_set_01", k, m, tag="_granite")
    focus["boulder"] = L.top_point("rock_moss_set_01", k, m)
    mesh, top = cairn()
    base = at(*CAIRN, -0.04)
    b.ply(b.assets.mesh(f"cirque_cairn_v{ASSET_VERSION}", lambda: mesh), G.translate(base), "cairn", prefix="cairn")
    focus["cairn"] = [base[0] + top[0], base[1] + top[1] - 0.01, base[2] + top[2]]

    rng = rng_for(ctx, 3)
    hero = [(BOULDER[0], BOULDER[1], 1.2), (CAIRN[0], CAIRN[1], 1.0)]
    flowers = []
    for j in range(9):                                   # the flower patch in front of the cameras
        a, r = j * 2.4 + rng.uniform(-0.3, 0.3), 0.15 + 0.12 * j
        x, z = FLOWERS[0] + r * math.cos(a), FLOWERS[1] + 0.6 * r * math.sin(a)
        aid = "periwinkle_plant" if j % 3 else "flower_gazania"
        batch.plant(aid, int(rng.integers(4)), at(x, z, -0.01), yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.08, 0.16))
        flowers.append(L.top_point(*batch.last))
    focus["flowers"] = flowers
    for x, z in N.scatter(rng, ((-25.0, -22.0), (25.0, 10.0)), 120, 1.4, avoid=hero):
        if height(x, z) < MEADOW_TOP + 0.3:
            batch.plant("periwinkle_plant" if rng.random() < 0.7 else "flower_gazania", int(rng.integers(4)),
                        at(x, z, -0.01), yaw=float(rng.uniform(0, 360)), height=rng.uniform(0.07, 0.15))
    for x, z in N.scatter(rng, ((-30.0, -24.0), (30.0, 12.0)), 300, 1.0, avoid=hero):
        if height(x, z) < MEADOW_TOP + 0.5:
            batch.plant("grass_medium_02", int(rng.integers(2)), at(x, z, -0.02), yaw=float(rng.uniform(0, 360)),
                        height=rng.uniform(0.12, 0.3))
    for x, z in N.scatter(rng, ((-20.0, -20.0), (20.0, 10.0)), 80, 1.5, avoid=hero):
        batch.plant("weed_plant_02", int(rng.integers(5)), at(x, z, -0.01), yaw=float(rng.uniform(0, 360)),
                    height=rng.uniform(0.05, 0.12))
    # scanned blocks over the talus and the bench edge, bigger up the slope
    for x, z in N.scatter(rng, ((-45.0, -100.0), (45.0, -10.0)), 140, 3.0, avoid=hero):
        d = -z
        batch.plant("rock_moss_set_01", int(rng.integers(6)), at(x, z, -0.15), yaw=float(rng.uniform(0, 360)),
                    height=rng.uniform(0.4, 0.8 + 1.4 * np.clip((d - 15) / 50, 0, 1)), tag="_granite")


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
    add_subjects(b, L, batch, ctx, focus)
    batch.flush(Assets(ctx.assets_dir, ctx.rebuild), f"cirque_scatter_s{ctx.seed}_v{ASSET_VERSION}")
    sun = SUN.get(ctx.env)
    b.d["sky"] = env_kit.sky(ctx.env, sun, scale=SKY_SCALE[ctx.env])
    return SceneBundle(b.d, focus, {"sun_direction": sun, "sky_scale": SKY_SCALE[ctx.env]})


def _exclusions():
    return (((BOULDER[0] - 1.0, -1.0, BOULDER[1] - 1.0), (BOULDER[0] + 1.0, 1.5, BOULDER[1] + 1.0)),
            ((CAIRN[0] - 0.7, -1.0, CAIRN[1] - 0.7), (CAIRN[0] + 0.7, 1.5, CAIRN[1] + 0.7)))


SCENE = SceneDef(
    id="alpine_scree", group="natural", owner="beiba",
    description="An alpine cirque above the tree line from a meadow bench: flowers and grass, a lichen-grey boulder "
                "and a stone cairn, a talus cone of angular scree under a cliff band, rock and snow peaks behind.",
    build=build,
    views={
        # Low over the flower patch, the boulder and the talus soft behind.
        "flowers": View((0.4, 0.4, 1.4), (FLOWERS[0], 0.0, FLOWERS[1]), focus="flowers"),
        # The boulder at ~6 m, the talus behind it.
        "boulder": View((0.5, 0.9, 2.0), (BOULDER[0], 0.5, BOULDER[1]), focus="boulder"),
        # Low beside the cairn, its top against the talus and cliff.
        "cairn": View((3.2, 0.5, -2.0), (CAIRN[0], 0.6, CAIRN[1]), focus="cairn"),
        # Standing, focused on the talus with the cliff and peaks behind.
        "talus": View((0.0, 1.7, 4.0), (0.0, 14.0, -45.0), focus="talus"),
    },
    default_view="talus",
    camera_box=((-6.0, 0.3, -3.0), (6.0, 1.8, 8.0)),
    # at most ~75 degrees of yaw from -z: with a 42-degree half field a level frame stays clear of the sun (135
    # degrees round from -z)
    target_box=((-25.0, -0.5, -150.0), (25.0, 40.0, -0.5)),
    exclude_boxes=_exclusions(),
    lens_mm=(24.0, 135.0), f_number=(1.4, 16.0),
    envs=("clear", "overcast", "snow"),
    tags=("outdoor", "day", "open", "snow", "bokeh", "vista"),
    default_seed=1, asset_version=ASSET_VERSION, max_depth=8, rr_depth=5, spp_hint=2048, far_clip=2000.0,
)
