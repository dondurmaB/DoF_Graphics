"""An old stone courtyard overtaken by plants: the "building in nature" scene.

Layout, meters, y up (north = -z). The courtyard floor is x in [-7, 7], z in [-9, 7]:
  north     a 3-storey ashlar block (27 m long, 10.2 m to the cornice): arched ground-floor windows and door,
            a balcony, shutters, ivy over its west half
  east      2-storey ochre stucco, spalling to brick: door with a lantern, a bicycle against the wall,
            Virginia creeper on its south end
  west      a cloister: six round arches on piers over a vaulted walk (tiled floor, beamed ceiling,
            three doors), ivy climbing the north piers
  south     brick, with a 4.8 m vaulted carriage passage out to the street (the archway view)
  middle    flagstones with moss in the joints, a two-tier fountain, a mature plane tree in a raised
            bed, a bench, lemon trees and geraniums in pots, laundry across the north-east corner
  roofs     barrel-tile roofs with valleys and chimneys; the sky is open above

Walls are real geometry with deep reveals; every facade has its own texture (streaks under its sills,
rising damp, lichen). Light is only the sun and sky from `env_kit`, which is correct for a y-up scene.

Environments (geometry never changes, except thin snow layers on ledges in `snow`):
  clear     late-morning sun from the south-east; the south wing shades the near half of the court
  overcast  env_kit's constant overcast sky
  wet       overcast, rain-darkened stone and roofs, glossy flagstones with puddles in the hollows
  snow      low winter sun, snow on the paving, roofs, sills, ledges, fountain and bench; ice on the
            fountain; the creeper and the tree keep dry brown leaves (marcescent), the ivy stays green

Seed (`ctx.seed`) changes the climbers' coverage, the flagstone layout and wear, shutter colours and
open/closed states, the plants (tree shape, lemons, geranium colours), laundry and the bicycle colour.
"""

from __future__ import annotations

import math

import numpy as np

import env_kit
import web_assets as W
import procedural as G
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, bitmap, principled, rgb, xf
from scenes import _courtyard_props as P
from scenes import _courtyard_tex as T
from scenes._courtyard_props import Opening as O

ASSET_VERSION = 8
TREE_ASSET, TREE_HEIGHT = "tree_small_02", 7.0   # Poly Haven scan (4.56 m), scaled 1.53x to plane-tree size
EYE = np.eye(4)
FACADE_PPM = 150
SUN = {"clear": env_kit.default_sun_direction(52.0, 32.0), "snow": env_kit.default_sun_direction(30.0, 26.0)}
SKY_SCALE = {"clear": 8.0, "overcast": 30.0, "wet": 45.0, "snow": 10.5}
TAN_ROOF = math.tan(math.radians(28.0))

# --- walls: (matrix, length, height, thickness, openings, texture kind, ppm) ------------------------------
M_N = G.translate((-13.6, 0.0, -9.0))
M_E = G.compose(G.translate((7.0, 0.0, -9.6)), G.rotate((0, 1, 0), -90))
M_W = G.compose(G.translate((-7.0, 0.0, 7.6)), G.rotate((0, 1, 0), 90))
M_S = G.compose(G.translate((7.6, 0.0, 7.0)), G.rotate((0, 1, 0), 180))
M_P = G.translate((-15.0, 0.0, 13.0))
M_B = G.compose(G.translate((-10.5, 0.0, 7.0)), G.rotate((0, 1, 0), 90))


def _nx(x):            # north wall u from world x
    return x + 13.6


def _ez(z):            # east wall u from world z
    return z + 9.6


def _wz(z):            # west wall u from world z
    return 7.6 - z


def _sx(x):            # south wall u from world x
    return 7.6 - x


def _win(c, w, v0, v1, arch=False, kind="window"):
    return O(c - w / 2, c + w / 2, v0, v1, arch, kind)


NORTH_OPEN = (
    [_win(_nx(x), 1.2, 0.9, 2.3, True) for x in (-5.0, -2.4, 4.6)]
    + [_win(_nx(1.5), 1.5, 0.0, 2.25, True, "door")]
    + [_win(_nx(x), 1.05, 4.3, 6.0) for x in (-5.0, -2.4, 4.6)]
    + [_win(_nx(1.5), 1.2, 3.75, 6.05, False, "balcony")]
    + [_win(_nx(x), 0.95, 7.6, 9.05) for x in (-5.0, -2.4, 1.5, 4.6)]
)
EAST_OPEN = (
    [_win(_ez(z), 1.1, 0.9, 2.3) for z in (-6.0, -3.2, 3.2, 5.6)]
    + [_win(_ez(-0.4), 1.2, 0.0, 2.45, False, "door")]
    + [_win(_ez(z), 1.0, 4.0, 5.7) for z in (-6.0, -3.2, -0.4, 3.2, 5.6)]
)
ARCH_U0 = [1.4, 3.9, 6.4, 8.9, 11.4, 13.9]
WEST_OPEN = ([O(u, u + 1.9, 0.12, 2.3, True, "arcade") for u in ARCH_U0]
             + [_win(u + 0.95, 0.9, 4.4, 5.9) for u in ARCH_U0])
SOUTH_OPEN = (
    [O(_sx(1.4), _sx(-1.4), 0.0, 2.4, True, "passage")]
    + [_win(_sx(x), 1.0, 0.9, 2.2) for x in (-4.6, 4.6)]
    + [_win(_sx(x), 0.95, 4.4, 5.9) for x in (-5.0, -2.6, 0.0, 2.6, 5.0)]
)
STREET_OPEN = [O(13.6, 16.4, 0.0, 2.4, True, "passage")]
BACK_OPEN = [_win(u, 1.2, 0.12, 2.0, True, "door") for u in (3.5, 8.0, 12.5)]

WALLS = {
    "n": (M_N, 27.2, 10.2, 0.6, NORTH_OPEN, "ashlar", (3.55, 6.95)),
    "e": (M_E, 17.2, 7.4, 0.6, EAST_OPEN, "stucco", (3.6,)),
    "w": (M_W, 17.2, 7.4, 0.5, WEST_OPEN, "limestone", (3.62,)),
    "s": (M_S, 15.2, 7.4, 0.6, SOUTH_OPEN, "brick", (4.0,)),
}

FOUNTAIN = (-1.6, 0.2)
TREE = (3.2, -4.6)
BENCH = ((3.5, -2.35), -61.0)
CITRUS = [(-6.25, -6.9), (-6.25, -1.9), (-6.25, 3.1), (0.3, -8.25), (2.7, -8.25)]
BIKE = (6.52, 1.45)
LAUNDRY = ((5.3, 6.35, -8.95), (6.95, 6.35, -7.1))
PAINTS = [(0.32, 0.45, 0.36), (0.20, 0.36, 0.40), (0.12, 0.24, 0.16), (0.42, 0.52, 0.60), (0.66, 0.52, 0.25),
          (0.50, 0.50, 0.48), (0.42, 0.25, 0.16)]
BIKE_PAINTS = [(0.55, 0.06, 0.05), (0.06, 0.18, 0.40), (0.10, 0.30, 0.20), (0.75, 0.70, 0.60), (0.04, 0.04, 0.05)]
CLOTH = [(0.86, 0.86, 0.83), (0.25, 0.38, 0.62), (0.62, 0.12, 0.10), (0.85, 0.70, 0.25), (0.32, 0.46, 0.32)]
FLOWERS = {"flower_red": (0.75, 0.05, 0.06), "flower_pink": (0.90, 0.30, 0.50), "flower_salmon": (0.95, 0.45, 0.32),
           "flower_white": (0.90, 0.88, 0.84)}
PAVE_PATHS = [[(0.0, 7.0), (0.6, 0.0), (1.5, -8.6)], [(0.0, 6.0), (6.4, -0.4)], [(0.0, 6.0), (-6.8, 0.5)],
              [(-6.8, 0.5), (-1.6, -2.4), (3.0, -1.2), (6.4, -0.4)]]


def rng_for(ctx: BuildContext, k: int) -> np.random.Generator:
    return np.random.default_rng([ctx.seed, 7919, k])


FACADE_SCAN = {"n": ("white_sandstone_blocks_02", None), "e": ("yellow_plaster_02", "red_brick_plaster_patch_02"),
               "w": ("old_sandstone_02", None), "s": ("medieval_red_brick", None)}
FABRICS = ("acg:Fabric004", "acg:Fabric026", "acg:Fabric045")


def scan_files(aid: str, res: str = "1k") -> dict:
    """Paths and real width of a pinned scanned texture (fetched on first use)."""
    d = W.surface(aid, res)
    inner = d["bsdf"] if d["type"] == "normalmap" else d
    width = 1.0 / float(np.asarray(inner["base_color"]["to_uv"].matrix)[0, 0])
    return {"diff": inner["base_color"]["filename"], "rough": inner["roughness"]["filename"],
            "nor": d["normalmap"]["filename"] if d["type"] == "normalmap" else None, "width": width}


def mean_rgb(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB").resize((64, 64)), np.float32).reshape(-1, 3).mean(0) / 255.0


def scanned(aid: str, res: str = "1k", uv_scale=None, colour=None, darken: float = 1.0, coat: bool = False) -> dict:
    """A scanned PBR surface (albedo, roughness, normal), optionally recoloured so its mean albedo is `colour`
    (keeps the palette the scene was tuned with), darkened (rain), and clear-coated (wet)."""
    tint = None
    if colour is not None or darken != 1.0:
        base = np.ones(3) if colour is None else np.asarray(colour) / np.maximum(mean_rgb(scan_files(aid, res)["diff"]), 1e-3)
        tint = tuple(float(v) for v in np.clip(base * darken, 0.0, 6.0))
    d = W.surface(aid, res, uv_scale=uv_scale, tint=tint)
    if coat:
        (d["bsdf"] if d["type"] == "normalmap" else d).update(clearcoat=0.4, clearcoat_gloss=0.85)
    return d


def thin(base, roughness, diff_trans) -> dict:
    return {"type": "principledthin", "base_color": base if isinstance(base, dict) else rgb(*base),
            "roughness": float(roughness), "diff_trans": float(diff_trans)}


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------
def add_materials(b: SceneBuilder, A: Assets, ctx: BuildContext, paint_idx, bike_idx, pave_tex) -> None:
    env = ctx.env
    wet, snow = env == "wet", env == "snow"
    wk = "wet" if wet else "dry"
    dk = 0.62 if wet else 1.0                                   # rain darkens porous stone, wood and plaster
    import mitsuba as mi

    # Facades: one texture per wall (its openings, drips, damp, moss), scanned masonry as the base pattern,
    # scanned normal and roughness on top at real scale.
    for name, (_, L, H, _, ops, kind, _) in WALLS.items():
        main, brick = FACADE_SCAN[name]
        sm = scan_files(main)
        sb = scan_files(brick) if brick else None
        tex = A.texture(f"facade_{name}_{wk}_scan", lambda kind=kind, L=L, H=H, ops=ops, name=name, sm=sm, sb=sb: T.facade(
            kind, L, H, [o.tex() for o in ops], FACADE_PPM, 100 + ord(name), wet,
            T.load_scan(sm["diff"], sm["width"], FACADE_PPM), T.load_scan(sb["diff"], sb["width"], FACADE_PPM) if sb else None))
        to_uv = mi.ScalarTransform4f().scale([1 / sm["width"], 1 / sm["width"], 1.0])
        rough = 0.45 if wet else {"type": "bitmap", "filename": sm["rough"], "raw": True, "to_uv": to_uv}
        inner = principled(bitmap(tex, uv_scale=(1 / L, 1 / H)), rough, specular=0.35)
        b.material(f"facade_{name}", {"type": "normalmap", "bsdf": inner, "normalmap": {
            "type": "bitmap", "filename": sm["nor"], "raw": True, "to_uv": to_uv}} if sm["nor"] else inner)
        b.material(f"reveal_{name}", scanned(main, colour=mean_rgb(tex), coat=wet))
    stone = A.texture(f"stone_{wk}", lambda: T.wet_variant(T.stone_tile()) if wet else T.stone_tile())
    b.material("stone", principled(bitmap(stone, uv_scale=(0.5, 0.5)), 0.45 if wet else 0.8, specular=0.35))
    stone_mean = mean_rgb(stone)                                 # keep the stone colour the user approved
    b.material("trim_stone", scanned("old_sandstone_02", colour=stone_mean, coat=wet))
    b.material("kerb_stone", scanned("old_sandstone_02", uv_scale=(6.0, 0.6), colour=stone_mean, coat=wet))
    b.material("old_wall", scanned("old_stone_wall", darken=dk, coat=wet))
    b.material("chimney", scanned("red_brick_plaster_patch_02", darken=dk))
    plaster_mean = np.asarray(T.plaster(size=1.0, ppm=64)).reshape(-1, 3).mean(0)   # the lime plaster the scene was tuned with
    b.material("vault_plaster", scanned("worn_cracked_plaster", colour=plaster_mean))
    b.material("soffit", scanned("white_rough_plaster", colour=plaster_mean))
    b.material("walk_wall", scanned("worn_plaster_wall", colour=plaster_mean))
    snow_tex = A.texture("snow", T.snow)
    b.material("snow", principled(bitmap(snow_tex, uv_scale=(0.5, 0.5)), 0.85, specular=0.3))
    if snow:
        b.material("roof", principled(bitmap(snow_tex, uv_scale=(0.5, 0.5)), 0.85, specular=0.3))
        b.material("road", env_kit.ground("snow"))
        b.material("sidewalk", env_kit.ground("snow"))
    else:
        b.material("roof", scanned("clay_roof_tiles_02", darken=dk, coat=wet))
        b.material("road", scanned("cobblestone_pavement", darken=dk, coat=wet))
        b.material("sidewalk", scanned("concrete_pavement", darken=dk, coat=wet))
    b.material("cobbles", scanned("cobblestone_floor_08", darken=dk, coat=wet))
    b.material("floor_tiles", scanned("terracotta_floor_tiles"))
    joints = A.texture("joints", T.joints)
    b.material("joints", principled(bitmap(joints, uv_scale=(0.5, 0.5)), 0.4 if wet else 0.95))
    rgb_path, rough_path, pave_nor = pave_tex
    pave = principled(bitmap(rgb_path), bitmap(rough_path, raw=True), specular=0.4,
                      **({"clearcoat": 0.5, "clearcoat_gloss": 0.9} if wet else {}))
    if pave_nor is not None:                                     # stone relief, dry only: puddles and snow are flat
        nor_path, width = pave_nor
        pave = {"type": "normalmap", "bsdf": pave, "normalmap": {"type": "bitmap", "filename": nor_path, "raw": True,
                "to_uv": mi.ScalarTransform4f().scale([16.0 / width, 16.0 / width, 1.0])}}
    b.material("paving", pave)

    b.material("door_wood", scanned("rough_pine_door", darken=dk))
    for name, idx in paint_idx.items():
        b.material(f"paint_{name}", scanned("distressed_painted_planks", colour=np.asarray(PAINTS[idx]) * dk))
    b.material("frame_paint", scanned("acg:PaintedWood004", colour=(0.80, 0.78, 0.72)))
    # No clearcoat: a gloss-1.0 coat on sunlit panes throws pin-sharp sun glints across the court that a path
    # tracer resolves only as fireflies. Roughness 0.07 still reads as reflective old glass.
    b.material("glass", principled((0.02, 0.025, 0.03), 0.07, specular=0.6))
    b.material("iron", scanned("rusty_painted_metal", "2k", colour=(0.13, 0.12, 0.11)))
    b.material("gutter", scanned("rusty_painted_metal", "2k", uv_scale=(1.0, 8.0), colour=(0.30, 0.29, 0.27)))
    b.material("fascia", scanned("weathered_brown_planks", darken=dk))
    b.material("terracotta", scanned("acg:Plaster006", uv_scale=(2.1, 0.8), colour=np.array((0.66, 0.35, 0.21)) * dk))
    b.material("trough", scanned("acg:Plaster006", colour=np.array((0.62, 0.33, 0.20)) * dk))
    b.material("pool_lining", principled((0.035, 0.045, 0.035), 0.6))
    if snow:
        b.material("soil", principled(bitmap(snow_tex, uv_scale=(0.5, 0.5)), 0.85, specular=0.3))
    else:
        b.material("soil", scanned("farm_soil", uv_scale=(6.0, 1.0), darken=dk))
    b.material("fountain_stone", scanned("mossy_sandstone", uv_scale=(5.0, 1.6), darken=dk, coat=wet))
    # Slightly rough water: a smooth dielectric reflects the sun as caustics that a path tracer only finds
    # by chance (fireflies on every wall in view); a rough one gets the sun by next-event estimation.
    b.material("water", {"type": "roughdielectric", "int_ior": 1.31 if snow else 1.33, "ext_ior": 1.0,
                         "alpha": 0.12 if snow else 0.1})

    bark = A.texture("bark", T.bark)
    b.material("bark", principled(bitmap(bark), 0.85))
    b.material("stem", principled((0.24, 0.19, 0.14), 0.8))
    leaf_r = 0.6 if wet else 1.0                                  # wet leaves are glossier
    ivy = A.texture("ivy_atlas", lambda: T.leaf_atlas(T.IVY, 300, "palmate", vein_colour=(0.42, 0.48, 0.30)))
    b.material("ivy_leaf", thin(bitmap(ivy), 0.32 * leaf_r, 0.2))
    creeper = A.texture("creeper_winter" if snow else "creeper_atlas",
                        lambda: T.leaf_atlas(T.CREEPER_WINTER if snow else T.CREEPER, 310, "palmate"))
    b.material("creeper_leaf", thin(bitmap(creeper), (0.65 if snow else 0.45) * leaf_r, 0.25 if snow else 0.45))
    fallen = A.texture("tree_winter", lambda: T.leaf_atlas(T.TREE_WINTER, 320))
    b.material("fallen_leaf", thin(bitmap(fallen), 0.7 * leaf_r, 0.2))
    citrus = A.texture("citrus_atlas", lambda: T.leaf_atlas(T.CITRUS, 330))
    b.material("citrus_leaf", thin(bitmap(citrus), 0.3 * leaf_r, 0.2))
    ger = A.texture("geranium_atlas", lambda: T.leaf_atlas(T.GERANIUM, 340, "palmate", zonal=True))
    b.material("geranium_leaf", thin(bitmap(ger), 0.55 * leaf_r, 0.3))
    b.material("lemon", principled((0.85, 0.66, 0.06), 0.35, specular=0.5))
    for key, col in FLOWERS.items():
        b.material(key, thin((0.26, 0.11, 0.07) if snow else col, 0.5, 0.35))
    b.material("bike_paint", principled(BIKE_PAINTS[bike_idx], 0.3, clearcoat=0.5, clearcoat_gloss=0.6))
    b.material("chrome", principled((0.78, 0.78, 0.78), 0.18, metallic=1.0))
    b.material("rubber", principled((0.025, 0.025, 0.025), 0.7))
    b.material("leather", principled((0.30, 0.17, 0.08), 0.5))
    b.material("dark_metal", principled((0.06, 0.06, 0.06), 0.45, metallic=0.6))
    b.material("rope", principled((0.72, 0.70, 0.64), 0.8))
    for k, col in enumerate(CLOTH):                               # woven fabric, still thin and translucent
        d = scanned(FABRICS[k % len(FABRICS)], uv_scale=(1.5, 1.5), colour=col)
        b.material(f"cloth_{k}", thin((d["bsdf"] if d["type"] == "normalmap" else d)["base_color"], 0.8, 0.35))


def add_walls(b: SceneBuilder, A: Assets, ctx: BuildContext, shutter_rng, snow_acc: P.Acc, focus: dict):
    """The four courtyard walls with openings, windows, doors, shutters, sills, ledges and cornices.
    Returns the climber obstacles per wall (openings plus open shutters)."""
    obstacles = {}
    states = {}
    shutter_parts = P.Groups()
    for name, (M, L, H, Tk, ops, kind, ledges) in WALLS.items():
        parts = A.meshes(f"wall_{name}", lambda L=L, H=H, Tk=Tk, ops=ops: P.wall(L, H, Tk, ops))
        b.ply(parts["face"], M, f"facade_{name}", prefix=f"wall_{name}")
        b.ply(parts["back"], M, "stone", prefix=f"wallback_{name}")
        b.ply(parts["reveal"], M, f"reveal_{name}", prefix=f"reveal_{name}")
        b.ply(parts["cap"], M, f"reveal_{name}", prefix=f"wallcap_{name}")
        trim = A.meshes(f"trim_{name}", lambda L=L, H=H, ops=ops, ledges=ledges, name=name: _trim(L, H, ops, ledges, name))
        for part, path in trim.items():
            b.ply(path, M, {"stone": "trim_stone", "frame": "frame_paint", "glass": "glass",
                            "door": "door_wood", "iron": "iron"}[part], prefix=f"trim_{name}_{part}")
        snow_acc.add(_trim_snow(L, ops, ledges), M)
        obs = []
        for o in ops:
            obs.append((o.u0 - 0.05, o.u1 + 0.05, o.v0 - 0.12, o.apex + 0.06))
            if o.kind != "window":
                continue
            r = shutter_rng.random()
            state = "open" if r < 0.55 else ("ajar" if r < 0.8 else "closed")
            angle = 172.0 + shutter_rng.uniform(-4, 4) if state == "open" else shutter_rng.uniform(100, 140)
            mats, W = P.shutters(o, state, angle)
            leaf = P.shutter_leaf(W, o.v1 - o.v0)
            for m in mats:
                shutter_parts[name].add(leaf, G.compose(M, m))
            obs += P.shutter_footprint(o, state, angle)
            states[(name, round(o.c, 2), round(o.v0, 2))] = state
        obstacles[name] = obs
    s = ctx.seed
    paths = A_seeded_meshes(ctx, f"shutters_s{s}", lambda: shutter_parts.meshes())
    for name, path in paths.items():
        b.ply(path, EYE, f"paint_{name}", prefix=f"shutters_{name}")
    return obstacles, states


def A_seeded_meshes(ctx, key, build):
    return Assets(ctx.assets_dir, ctx.rebuild).meshes(key, build)


def _trim(L, H, ops, ledges, name) -> dict[str, G.Mesh]:
    stone, frame, glass, door, iron = P.Acc(), P.Acc(), P.Acc(), P.Acc(), P.Acc()
    for y in ledges:
        stone.add(P.ledge(L, y)[0])
    stone.add(P.cornice(L, H))
    for o in ops:
        if o.kind in ("window", "balcony"):
            if o.kind == "window":
                stone.add(P.sill(o)[0])
            w = P.window(o)
            frame.add(w["frame"]), glass.add(w["glass"])
        elif o.kind == "door":
            d = P.door(o)
            door.add(d["door"]), iron.add(d["iron"])
            stone.add(G.box((o.u1 - o.u0 + 0.5, 0.06, 0.45), (o.c, 0.03, 0.18)))        # threshold step
        elif o.kind == "arcade":
            stone.add(G.box((0.62, 0.2, 0.62), (o.u1 + 0.3, o.v1 - 0.1, -0.25)))        # impost blocks
            stone.add(G.box((0.62, 0.2, 0.62), (o.u0 - 0.3, o.v1 - 0.1, -0.25)))
            stone.add(G.box((0.7, 0.12, 0.6), (o.u1 + 0.3, 0.06, -0.25)))                  # pier bases
        if o.kind in ("window", "door", "passage", "balcony") and (name in ("e", "s") or o.arch):
            stone.add(P.surround(o, bw=0.18 if o.kind == "passage" else 0.15))
    if name == "w":
        stone.add(G.box((0.7, 0.12, 0.6), (ARCH_U0[0] - 0.3, 0.06, -0.25)))
    parts = {"stone": stone.mesh(), "frame": frame.mesh(), "glass": glass.mesh(), "door": door.mesh(), "iron": iron.mesh()}
    return {k: m for k, m in parts.items() if len(m.f)}                        # Mitsuba rejects empty meshes


def _trim_snow(L, ops, ledges) -> G.Mesh:
    acc = P.Acc()
    for y in ledges:
        size, centre = P.ledge(L, y)[1]
        acc.add(G.box(size, centre))
    for o in ops:
        if o.kind == "window":
            size, centre = P.sill(o)[1]
            acc.add(G.box(size, centre))
    return acc.mesh()


def add_roofs(b: SceneBuilder, A: Assets) -> None:
    roofs = {
        "n": (G.translate((0.0, 0.0, -8.55)), 13.6, 3.3, 10.25, False),
        "e": (G.compose(G.translate((6.55, 0.0, -1.0)), G.rotate((0, 1, 0), -90)), 7.55, 6.45, 7.45, True),
        "s": (G.compose(G.translate((0.0, 0.0, 6.55)), G.rotate((0, 1, 0), 180)), 6.55, 6.45, 7.45, True),
        "w": (G.compose(G.translate((-6.55, 0.0, -1.0)), G.rotate((0, 1, 0), 90)), 7.55, 6.45, 7.45, True),
    }
    for name, (M, a, d, y, widen) in roofs.items():
        b.ply(A.mesh(f"roof_{name}", lambda a=a, d=d, y=y, widen=widen: P.roof(a, d, y, widen=widen)), M, "roof",
              prefix=f"roof_{name}")
        trim = A.meshes(f"eave_{name}", lambda a=a, y=y: P.eave_trim(a, y))
        b.ply(trim["fascia"], M, "fascia", prefix="fascia")
        b.ply(trim["gutter"], M, "gutter", prefix="gutter")
    ch = A.mesh("chimney", P.chimney)
    for (x, z, base) in ((-4.0, 9.8, 8.9), (9.6, 2.5, 8.8), (-3.0, -10.6, 11.1), (5.5, -10.4, 11.0)):
        b.ply(ch, G.translate((x, base, z)), "chimney", prefix="chimney")
    b.ply(A.mesh("tv_aerial", P.tv_aerial), G.translate((-3.0, 13.42, -10.6)), "iron", prefix="aerial")
    pipe = A.mesh("downpipe", lambda: P.tube([(0, 0.0, 0), (0, 7.3, 0), (0, 7.38, 0.35)], 0.05, 10))
    for x, z, yaw in ((6.84, -8.84, 90.0), (-6.85, 6.84, -90.0), (-6.85, -8.84, -90.0), (6.84, 6.84, 90.0)):
        b.ply(pipe, G.compose(G.translate((x, 0.0, z)), G.rotate((0, 1, 0), yaw)), "gutter", prefix="downpipe")


def add_passage_and_walk(b: SceneBuilder, A: Assets) -> None:
    """Carriage passage through the south wing, the street beyond it, and the cloister walk."""
    street = A.meshes("street_wall", lambda: P.wall(30.0, 7.4, 0.6, STREET_OPEN))
    b.ply(street["face"], M_P, "old_wall", prefix="street_wall")
    b.ply(street["reveal"], M_P, "old_wall", prefix="street_reveal")
    b.ply(street["back"], M_P, "old_wall", prefix="street_wallback")
    passage = A.meshes("passage", _passage)
    b.ply(passage["wall"], EYE, "old_wall", prefix="passage_wall")
    b.ply(passage["vault"], EYE, "vault_plaster", prefix="passage_vault")
    b.ply(passage["floor"], EYE, "cobbles", prefix="passage_floor")
    street_parts = A.meshes("street_ground", _street)
    for part, path in street_parts.items():
        b.ply(path, EYE, {"road": "road", "walk": "sidewalk", "kerb": "trim_stone"}[part], prefix=f"street_{part}")
    back = A.meshes("walk_back", lambda: P.wall(16.0, 3.6, 0.4, BACK_OPEN))
    b.ply(back["face"], M_B, "walk_wall", prefix="walk_wall")
    b.ply(back["reveal"], M_B, "trim_stone", prefix="walk_reveal")
    walk = A.meshes("walk", _walk)
    b.ply(walk["floor"], EYE, "floor_tiles", prefix="walk_floor")
    b.ply(walk["plaster"], EYE, "soffit", prefix="walk_plaster")
    b.ply(walk["beams"], EYE, "fascia", prefix="walk_beams")
    doors = A.meshes("walk_doors", lambda: {"door": P._merge(*[P.door(o)["door"] for o in BACK_OPEN]),
                                            "iron": P._merge(*[P.door(o)["iron"] for o in BACK_OPEN])})
    b.ply(doors["door"], M_B, "door_wood", prefix="walk_door")
    b.ply(doors["iron"], M_B, "iron", prefix="walk_door_iron")


def _street() -> dict[str, G.Mesh]:
    """Pavement (0.12 m up) outside the gate, the road, and the far pavement under the opposite facade."""
    walk, kerb = P.Acc(), P.Acc()
    for z0, z1 in ((13.0, 14.8), (21.6, 24.2)):
        walk.add(P.floor_quad(-40.0, 40.0, z0, z1, 0.12))
    kerb.add(P.quad([40.0, 0.0, 14.8], [-40.0, 0.0, 14.8], [-40.0, 0.12, 14.8], [40.0, 0.12, 14.8]))
    kerb.add(P.quad([-40.0, 0.0, 21.6], [40.0, 0.0, 21.6], [40.0, 0.12, 21.6], [-40.0, 0.12, 21.6]))
    return {"road": P.floor_quad(-40.0, 40.0, 14.8, 21.6, 0.0), "walk": walk.mesh(), "kerb": kerb.mesh()}


def _passage() -> dict[str, G.Mesh]:
    wall, vault = P.Acc(), P.Acc()
    z0, z1, hw, spring = 7.6, 12.4, 1.4, 2.4
    wall.add(P.quad([-hw, 0, z1], [-hw, 0, z0], [-hw, spring, z0], [-hw, spring, z1]))
    wall.add(P.quad([hw, 0, z0], [hw, 0, z1], [hw, spring, z1], [hw, spring, z0]))
    ang = np.linspace(math.pi, 0.0, 33)
    for a0, a1 in zip(ang[:-1], ang[1:]):
        p0 = (hw * math.cos(a0), spring + hw * math.sin(a0))
        p1 = (hw * math.cos(a1), spring + hw * math.sin(a1))
        s0, s1 = (math.pi - a0) * hw, (math.pi - a1) * hw
        vault.add(P.quad([p0[0], p0[1], z1], [p1[0], p1[1], z1], [p1[0], p1[1], z0], [p0[0], p0[1], z0],
                         [(s0, z1), (s1, z1), (s1, z0), (s0, z0)]))
    return {"wall": wall.mesh(), "vault": vault.mesh(), "floor": P.floor_quad(-1.45, 1.45, 6.4, 13.6, 0.004)}


def _unused_gate():
    """Kept for reference: the procedural gate leaf is replaced by the scanned large_castle_door (add_web)."""


def _walk() -> dict[str, G.Mesh]:
    x0, x1, z0, z1, ceil = -10.5, -7.5, -9.0, 7.0, 3.6
    plaster, beams = P.Acc(), P.Acc()
    plaster.add(P.floor_quad(x0, x1, z0, z1, ceil, up=False))
    plaster.add(P.quad([x1, 0, z1], [x0, 0, z1], [x0, ceil, z1], [x1, ceil, z1]))
    plaster.add(P.quad([x0, 0, z0], [x1, 0, z0], [x1, ceil, z0], [x0, ceil, z0]))
    for z in np.arange(z0 + 0.6, z1 - 0.3, 1.25):
        beams.add(G.box((x1 - x0, 0.22, 0.16), ((x0 + x1) / 2, ceil - 0.11, z)))
    return {"floor": P.floor_quad(x0, x1 + 0.02, z0, z1, 0.12), "plaster": plaster.mesh(), "beams": beams.mesh()}


# --------------------------------------------------------------------------
# Courtyard furniture and plants
# --------------------------------------------------------------------------
def add_ground(b: SceneBuilder, A: Assets, seeded: Assets, ctx: BuildContext, snow_acc: P.Acc) -> list:
    rng = rng_for(ctx, 1)
    stones, mesh = P.flagstones(rng, -7.0, 7.0, -9.0, 7.0)
    b.ply(seeded.mesh(f"paving_s{ctx.seed}", lambda: mesh), EYE, "paving", prefix="paving")
    b.ply(A.mesh("ground", lambda: P.floor_quad(-40.0, 40.0, -40.0, 13.0, 0.0)), EYE, "joints", prefix="ground")
    if ctx.env == "snow":
        cover = P.quad([-7.0, 0.05, 7.0], [7.0, 0.05, 7.0], [7.0, 0.05, -9.0], [-7.0, 0.05, -9.0],
                       [(1 / 16, 1.0), (15 / 16, 1.0), (15 / 16, 0.0), (1 / 16, 0.0)])
        b.ply(seeded.mesh("snow_cover", lambda: cover), EYE, "paving", prefix="snow_cover")
    return stones


def add_fountain_tree_bench(b: SceneBuilder, A: Assets, snow_acc: P.Acc) -> None:
    f = A.meshes("fountain", P.fountain)
    M = G.translate((FOUNTAIN[0], 0.0, FOUNTAIN[1]))
    b.ply(f["stone"], M, "fountain_stone", prefix="fountain")
    b.ply(f["lining"], M, "pool_lining", prefix="fountain_lining")
    b.ply(f["water_low"], M, "water", prefix="water")
    b.ply(f["water_high"], M, "water", prefix="water")
    snow_acc.add(G.lathe([(1.575, 0.50), (1.56, 0.57), (1.48, 0.605), (1.40, 0.57), (1.37, 0.50)], 96), M)
    snow_acc.add(G.lathe([(0.62, 1.25), (0.61, 1.30), (0.56, 1.32), (0.51, 1.28)], 48), M)
    Mt = G.translate((TREE[0], 0.0, TREE[1]))
    b.ply(A.mesh("kerb", lambda: P.kerb_ring(1.25, 1.45, 0.42)), Mt, "kerb_stone", prefix="kerb")
    b.ply(A.mesh("bed_soil", lambda: P.disc(1.26, 0.33, 48)), Mt, "soil", prefix="bed_soil")
    snow_acc.add(G.lathe([(1.46, 0.36), (1.44, 0.45), (1.35, 0.47), (1.26, 0.45), (1.24, 0.36)], 96), Mt)
    (bx, bz), yaw = BENCH                                      # scanned bench (add_web places it)
    Mb = G.compose(G.translate((bx, 0.0, bz)), G.rotate((0, 1, 0), yaw))
    snow_acc.add(G.box((1.12, 0.025, 0.36), (0.0, 0.47, 0.02)), Mb)


def add_plants(b: SceneBuilder, A: Assets, seeded: Assets, ctx: BuildContext, obstacles, snow_acc: P.Acc,
               focus: dict) -> None:
    s = ctx.seed
    lib_rng = np.random.default_rng(4242)
    ivy_lib = P.LeafLibrary(lib_rng, lambda r, k: P.lobed_leaf(r, "ivy", 6, k, cup=0.12), 4, 6)
    creeper_lib = P.LeafLibrary(lib_rng, lambda r, k: P.lobed_leaf(r, "creeper", 6, k, cup=0.16), 4, 6)
    tree_lib = P.LeafLibrary(lib_rng, lambda r, k: P.blade_leaf(r, 6, k, 0.62), 4, 6)
    citrus_lib = P.LeafLibrary(lib_rng, lambda r, k: P.blade_leaf(r, 4, k, 0.48), 4, 4)
    ger_lib = P.LeafLibrary(lib_rng, lambda r, k: P.lobed_leaf(r, "round", 4, k, serrate=0.04, cup=0.18, subdiv=1), 3, 4)

    def build_climbers():
        grp = P.Groups()
        rng = rng_for(ctx, 2)
        specs = [
            ("n", "ivy", ivy_lib, (_nx(-6.75), _nx(-0.35)), P.smooth_top(rng, _nx(-7), _nx(0), 6.2, 2.2), 26, None,
             dict(size=(0.055, 0.09), every=0.04, budget=850.0)),
            ("e", "creeper", creeper_lib, (_ez(1.05), _ez(6.9)), P.smooth_top(rng, _ez(1), _ez(7), 4.2, 1.6), 10, None,
             dict(size=(0.08, 0.13), every=0.055, hang=0.8, budget=260.0)),
            ("w", "ivy", ivy_lib, (_wz(-8.95), _wz(-3.3)), P.smooth_top(rng, _wz(-9), _wz(-3), 5.6, 1.3), 0,
             [u + d for u in (11.1, 13.6, 16.25) for d in (-0.18, 0.0, 0.17)],
             dict(size=(0.055, 0.085), every=0.042, budget=380.0)),
            ("s", "creeper", creeper_lib, (_sx(6.9), _sx(2.4)), P.smooth_top(rng, _sx(7), _sx(2), 4.6, 1.2), 7, None,
             dict(size=(0.08, 0.12), every=0.055, hang=0.8, budget=180.0)),
        ]
        for wall_name, kind, lib, region, top, n, roots, kw in specs:
            M = WALLS[wall_name][0]
            out = P.climber(rng, region, top, obstacles[wall_name], n, lib, roots=roots, **kw)
            grp["stem"].add(out["stem"], M)
            grp[f"{kind}_leaf"].add(out["leaf"], M)
        return grp.meshes()

    climbers = seeded.meshes(f"climbers_s{s}_v{ASSET_VERSION}", build_climbers)
    for part, path in climbers.items():
        b.ply(path, EYE, part, prefix=f"climber_{part}")

    def build_plants():
        grp = P.Groups()
        rng = rng_for(ctx, 3)

        P.fallen_leaves(grp, rng, (TREE[0] - 0.8, 0.0, TREE[1] + 0.8), 3.6, 170, tree_lib)
        crown_pts = []
        for (x, z) in CITRUS:
            crown_pts.append(P.citrus(grp, rng, (x, 0.5, z), citrus_lib, height=rng.uniform(0.85, 1.1)))
        flower_keys = list(FLOWERS)
        ger_pos = [(6.55, 0.18, -1.4), (6.6, 0.18, -1.78), (6.5, 0.18, 0.55)]
        for k, p in enumerate(ger_pos):
            P.geranium(grp, rng, p, ger_lib, flower_keys[int(rng.integers(len(flower_keys)))])
        for x in (-2.4, 4.6):                                               # window boxes, north first floor
            key = flower_keys[int(rng.integers(len(flower_keys)))]
            for dx in np.linspace(-0.4, 0.4, 3):
                P.geranium(grp, rng, (x + dx, 4.48, -9.0 + 0.05), ger_lib, key, n_leaves=9, n_umbels=4, spread=0.14)
        for dx in (-0.6, 0.0, 0.6):                                          # balcony pots
            P.geranium(grp, rng, (1.5 + dx, 3.75 + 0.18, -9.0 + 0.45), ger_lib,
                       flower_keys[int(rng.integers(len(flower_keys)))], n_leaves=10, n_umbels=4)
        return grp.meshes(), crown_pts

    plants_key = f"plants_s{s}_v{ASSET_VERSION}"
    meshes = seeded.meshes(plants_key, lambda: build_plants()[0])
    for part, path in meshes.items():
        b.ply(path, EYE, part, prefix=f"plant_{part}")
    for k, (x, z) in enumerate(CITRUS):
        b.ply(A.mesh("pot_big", lambda: P.pot(0.34, 0.55)), G.translate((x, 0.0, z)), "terracotta", prefix="pot")
        b.ply(A.mesh("pot_big_soil", lambda: P.disc(0.33, 0.5)), G.translate((x, 0.0, z)), "soil", prefix="potsoil")
        focus[f"lemon_tree_{k}"] = [x, 1.55, z + 0.45]
    clay = W.model("planter_pot_clay")                                       # scanned small pots
    turn = rng_for(ctx, 9)
    for p in ((6.55, 0.0, -1.4), (6.6, 0.0, -1.78), (6.5, 0.0, 0.55), (0.9, 3.75, -8.55), (1.5, 3.75, -8.55),
              (2.1, 3.75, -8.55)):
        W.add(b, clay, W.place(clay, at=p, yaw=float(turn.uniform(0, 360))), prefix="clay_pot")
        b.ply(A.mesh("pot_small_soil", lambda: P.disc(0.112, 0.185)), G.translate(p), "soil", prefix="potsoil")
    for x in (-2.4, 4.6):
        b.ply(A.mesh("trough", lambda: P.trough(1.1)), G.translate((x, 4.3, -9.0 + 0.05)), "trough", prefix="trough")
        b.ply(A.mesh("trough_soil", lambda: G.box((1.05, 0.01, 0.15), (0, 0.17, 0))), G.translate((x, 4.3, -8.95)),
              "soil", prefix="trough_soil")


def add_tree(b: SceneBuilder, ctx: BuildContext) -> None:
    """The courtyard tree: a scanned CC0 broadleaf (web_assets) in the raised bed. The seed turns it.
    In `snow` it is bare: winter broadleaf trees have dropped their leaves (the fallen ones lie on the paving)."""
    m = W.model(TREE_ASSET)
    yaw = float(rng_for(ctx, 8).uniform(0.0, 360.0))
    M = W.place(m, at=(TREE[0], 0.33, TREE[1]), yaw=yaw, height=TREE_HEIGHT)
    if ctx.env == "snow":
        m = W.Model(m.id, m.res, [p for p in m.parts if not p.name.endswith("_leaves")], m.bounds, m.triangles)
    W.add(b, m, M, prefix="tree")


def add_props(b: SceneBuilder, A: Assets, seeded: Assets, ctx: BuildContext, snow_acc: P.Acc, focus: dict) -> None:
    bal = A.meshes("balcony", P.balcony)
    Mb = G.compose(M_N, G.translate((_nx(1.5), 3.75, 0.0)))
    b.ply(bal["stone"], Mb, "trim_stone", prefix="balcony")
    b.ply(bal["iron"], Mb, "iron", prefix="balcony_iron")
    snow_acc.add(G.box((1.86, 0.035, 0.7), (0.0, 0.0175, 0.37)), Mb)
    bike = A.meshes("bicycle", P.bicycle)
    Mk = G.compose(G.translate((BIKE[0], 0.0, BIKE[1])), G.rotate((0, 0, 1), -12.0))
    for part, mat in (("paint", "bike_paint"), ("chrome", "chrome"), ("rubber", "rubber"), ("leather", "leather"),
                      ("dark", "dark_metal")):
        b.ply(bike[part], Mk, mat, prefix=f"bike_{part}")
    focus["bicycle"] = (Mk @ np.array([0.0, 1.0, -0.26, 1.0]))[:3].tolist()
    a, c = np.array(LAUNDRY[0]), np.array(LAUNDRY[1])
    rng = rng_for(ctx, 4)
    lines = seeded.meshes(f"laundry_s{ctx.seed}", lambda: P.laundry(rng, a, c))
    for part, path in lines.items():
        b.ply(path, EYE, part, prefix=f"laundry_{part}")
    for p, yaw in ((a, 0.0), (c, -90.0)):
        b.ply(A.mesh("line_bracket", lambda: G.box((0.04, 0.04, 0.16), (0, 0, -0.08))),
              G.compose(G.translate(p), G.rotate((0, 1, 0), yaw)), "iron", prefix="line_bracket")


def on_wall(b, aid: str, wall_xz, yaw: float, y: float, scale: float = 1.0, prefix=None, overrides=None):
    """Mount a scanned model with its back (local -z extreme) against a wall point; yaw turns local +z to face out."""
    m = W.model(aid)
    lo, hi = np.asarray(m.bounds[0]), np.asarray(m.bounds[1])
    off = ((lo[2] + hi[2]) / 2 - lo[2]) * scale + 0.005
    a = math.radians(yaw)
    at = (wall_xz[0] + math.sin(a) * off, y, wall_xz[1] + math.cos(a) * off)
    W.add(b, m, W.place(m, at=at, yaw=yaw, scale=scale), prefix=prefix, overrides=overrides)


def put(b, aid: str, at, yaw: float = 0.0, scale: float = 1.0, height=None, prefix=None, tilt=None, overrides=None):
    m = W.model(aid)
    M = W.place(m, at=at, yaw=yaw, scale=scale, height=height)
    if tilt is not None:                                          # (axis, degrees) about the foot point
        M = G.compose(G.translate(at), G.rotate(tilt[0], tilt[1]), G.translate(-np.asarray(at, float)), M)
    W.add(b, m, M, prefix=prefix, overrides=overrides)


def lantern(b, A: Assets, M, scale: float = 1.3) -> None:
    """Scanned brass lantern hanging from the procedural scrolled bracket placed by M (local hook (0, -0.05, 0.55))."""
    b.ply(A.mesh("bracket_arm", P.bracket_arm), M, "iron", prefix="lantern_arm")
    m = W.model("Lantern_01")
    hook = (np.asarray(M) @ np.array([0.0, -0.05, 0.55, 1.0]))[:3]
    h = (m.bounds[1][1] - m.bounds[0][1]) * scale
    W.add(b, m, W.place(m, at=(hook[0], hook[1] - h, hook[2]), scale=scale))


def add_web(b: SceneBuilder, A: Assets, ctx: BuildContext, focus: dict) -> None:
    """Scanned CC0 substitutes and eye-level additions (scenes/substitutions/courtyard.md), so the court
    reads as a lived-in place from every angle the sampler picks."""
    rng = rng_for(ctx, 10)
    # --- substitutes ---------------------------------------------------------------------------------
    (bx, bz), byaw = BENCH
    put(b, "painted_wooden_bench", (bx, 0.0, bz), yaw=byaw)
    put(b, "large_castle_door", (-1.27, 0.0, 11.54), yaw=90.0, scale=0.81)          # the open gate leaf
    put(b, "modular_urban_apartments_facade", (0.0, -2.0, 27.34), yaw=180.0)      # street front opposite
    lantern(b, A, G.compose(G.translate((-1.4, 2.3, 10.0)), G.rotate((0, 1, 0), 90)))   # passage
    lantern(b, A, G.compose(M_E, G.translate((_ez(0.55), 2.75, 0.0))))                 # east door
    for x in (0.3, 2.7):                                                               # north door
        lantern(b, A, G.translate((x, 2.35, -9.0)))
    weeds = W.model("weed_plant_02")
    runs = [((-6.6, -8.93), 0.0, (-6.4, 0.2)), ((3.0, -8.93), 0.0, (2.6, 6.2)), ((6.93, -8.4), 90.0, (-8.4, -2.0)),
            ((6.93, 1.0), 90.0, (1.0, 6.4)), ((-6.6, 6.93), 0.0, (-6.4, -1.8)), ((2.0, 6.93), 0.0, (1.9, 6.4))]
    for (x0, z0), yaw, (lo, hi) in runs:
        s = lo + rng.uniform(0, 0.8)
        while s < hi - 1.0:
            x, z = (s + 1.0, z0) if yaw == 0.0 else (x0, s + 1.0)
            W.add(b, weeds, W.place(weeds, at=(x, 0.035, z), yaw=yaw + float(rng.choice([0.0, 180.0])),
                                    scale=float(rng.uniform(0.7, 1.0))))
            s += rng.uniform(2.2, 4.0)
    shrub = W.model("shrub_04")
    for k in range(6):                                                                 # ground cover in the tree bed
        a, r = rng.uniform(0, math.tau), rng.uniform(0.55, 1.0)
        W.add(b, shrub, W.place(shrub, at=(TREE[0] + r * math.cos(a), 0.33, TREE[1] + r * math.sin(a)),
                                yaw=float(rng.uniform(0, 360)), scale=float(rng.uniform(0.8, 1.2))))
    # --- the street beyond the arch -----------------------------------------------------------------------
    put(b, "street_lamp_01", (4.2, 0.12, 22.6), yaw=180.0)
    put(b, "water_manhole_cover", (0.9, 0.0, 18.0), yaw=float(rng.uniform(0, 90)))
    put(b, "fire_hydrant", (-2.5, 0.12, 14.3), yaw=180.0)
    put(b, "covered_car", (-0.6, 0.0, 18.2), yaw=78.0)
    for x in (2.3, 2.8):
        put(b, "trashbag", (x, 0.12, 14.35 + rng.uniform(0, 0.2)), yaw=float(rng.uniform(0, 360)))
    on_wall(b, "security_light", (0.0, 13.0), 0.0, 4.1)
    # --- inside the carriage passage -------------------------------------------------------------------
    on_wall(b, "power_box_01", (1.4, 9.4), -90.0, 1.25)
    on_wall(b, "utility_box_01", (1.4, 11.95), -90.0, 0.0)              # beside the arch-view camera, out of frame
    put(b, "wooden_broom", (-1.25, 0.0, 10.35), yaw=90.0, tilt=((1, 0, 0), -8.0))
    put(b, "plastic_crate_01", (1.22, 0.0, 11.75), yaw=0.0)
    put(b, "plastic_crate_01", (1.2, 0.26, 11.72), yaw=float(rng.uniform(-10, 10)))
    # --- corners where the wings meet ----------------------------------------------------------------------
    for cx, cz, yaw in ((6.45, -8.45, 0.0), (-6.45, -8.45, 0.0), (-6.45, 6.45, 180.0), (6.45, 6.45, 180.0)):
        sx, sz = (1 if cx > 0 else -1), (1 if cz > 0 else -1)
        put(b, "planter_pot_clay", (cx, 0.0, cz), yaw=float(rng.uniform(0, 360)), scale=1.25)
        put(b, "potted_plant_04", (cx - sx * 0.42, 0.0, cz + sz * 0.05), yaw=float(rng.uniform(0, 360)))
        put(b, "planter_pot_clay", (cx + sx * 0.05, 0.0, cz - sz * 0.42), yaw=float(rng.uniform(0, 360)))
        put(b, "periwinkle_plant", (cx - sx * 0.95, 0.0, cz + sz * 0.3), yaw=yaw, scale=0.8)
        for k in range(2):
            put(b, "moss_01", (cx + sx * 0.2 - sx * rng.uniform(0.6, 2.0), 0.035, cz + sz * 0.38),
                yaw=float(rng.uniform(0, 360)), scale=float(rng.uniform(1.5, 2.5)))
    # --- north wall at eye level ----------------------------------------------------------------------------
    put(b, "wicker_basket_02", (0.35, 0.0, -8.72), yaw=float(rng.uniform(-30, 30)))
    put(b, "wooden_bucket_01", (3.25, 0.0, -8.72), yaw=float(rng.uniform(0, 360)))
    put(b, "gallinera_table", (3.95, 0.0, -8.6), yaw=0.0)
    put(b, "painted_wooden_chair_01", (3.95, 0.0, -7.95), yaw=180.0 + float(rng.uniform(-15, 15)))
    # --- east wall ----------------------------------------------------------------------------------------------
    on_wall(b, "utility_box_02", (7.0, -4.6), -90.0, 0.0)
    on_wall(b, "garden_hose_wall_mounted_01", (7.0, 2.6), -90.0, 1.0)
    put(b, "watering_can_metal_01", (6.72, 0.0, 2.55), yaw=-90.0)
    put(b, "wooden_bucket_02", (6.62, 0.0, 4.35), yaw=float(rng.uniform(0, 360)))
    # --- west cloister walk -------------------------------------------------------------------------------
    put(b, "painted_wooden_bench", (-10.22, 0.12, 0.25), yaw=90.0, prefix="walk_bench")
    put(b, "planter_box_02", (-10.24, 0.12, -4.75), yaw=90.0)
    for dz in (-0.3, 0.25):
        put(b, "shrub_04", (-10.3, 0.57, -4.75 + dz), yaw=90.0 + float(rng.uniform(-20, 20)))
    put(b, "wooden_stool_01", (-9.9, 0.12, -1.1), yaw=float(rng.uniform(0, 360)))
    for z in (-1.5, 2.2):
        b.ply(A.mesh(f"walk_chain_{z}", lambda z=z: P.chain((-8.9, 3.6, z), 0.62)), EYE, "iron", prefix="walk_chain")
        m = W.model("Lantern_01")
        W.add(b, m, W.place(m, at=(-8.9, 2.98 - 0.29 * 1.3, z), scale=1.3))
    # --- south wing, court side -------------------------------------------------------------------------------
    on_wall(b, "exterior_aircon_unit", (-3.6, 7.0), 180.0, 3.0)
    for dx in (-0.6, 0.6):
        b.ply(A.mesh("aircon_bracket", lambda: G.box((0.04, 0.04, 0.4), (0.0, 0.0, -0.2))),
              G.translate((-3.6 + dx, 2.98, 7.0)), "iron", prefix="aircon_bracket")
    put(b, "metal_trash_can", (2.95, 0.0, 6.66), yaw=180.0)
    put(b, "trashbag", (4.25, 0.0, 6.62), yaw=float(rng.uniform(0, 360)))


def build(ctx: BuildContext) -> SceneBundle:
    shared = Assets(ctx.shared_dir, ctx.rebuild)
    seeded = Assets(ctx.assets_dir, ctx.rebuild)
    pick = rng_for(ctx, 0)
    paint_idx = {name: int(pick.integers(len(PAINTS))) for name in WALLS}
    bike_idx = int(pick.integers(len(BIKE_PAINTS)))
    b = SceneBuilder(shared)
    focus: dict[str, list] = {}

    pave_rng = rng_for(ctx, 1)
    stones, _ = P.flagstones(pave_rng, -7.0, 7.0, -9.0, 7.0)
    puddle_rng = rng_for(ctx, 5)
    puddles = []
    while len(puddles) < 18:
        x, z = puddle_rng.uniform(-6.5, 6.5), puddle_rng.uniform(-8.5, 6.5)
        if math.hypot(x - FOUNTAIN[0], z - FOUNTAIN[1]) < 1.8 or math.hypot(x - TREE[0], z - TREE[1]) < 1.6:
            continue
        puddles.append((x, z, puddle_rng.uniform(0.3, 1.0), puddle_rng.uniform(0.2, 0.7)))
    mode = {"wet": "wet", "snow": "snow"}.get(ctx.env, "dry")
    pave_cache = {}

    def pave():
        if mode not in pave_cache:
            pave_cache[mode] = T.paving(stones, 200, 500 + ctx.seed, mode, PAVE_PATHS, puddles,
                                        T.load_scan(ps["diff"], ps["width"], 200))
        return pave_cache[mode]

    ps = scan_files("monastery_stone_floor")
    pave_tex = (seeded.texture(f"paving_{mode}_scan_s{ctx.seed}", lambda: pave()[0]),
                seeded.texture(f"paving_rough_{mode}_scan_s{ctx.seed}", lambda: pave()[1], gray=True),
                (ps["nor"], ps["width"]) if mode == "dry" and ps["nor"] else None)
    add_materials(b, shared, ctx, paint_idx, bike_idx, pave_tex)

    snow_acc = P.Acc()
    obstacles, states = add_walls(b, shared, ctx, rng_for(ctx, 6), snow_acc, focus)
    add_roofs(b, shared)
    add_passage_and_walk(b, shared)
    add_ground(b, shared, seeded, ctx, snow_acc)
    add_fountain_tree_bench(b, shared, snow_acc)
    add_plants(b, shared, seeded, ctx, obstacles, snow_acc, focus)
    add_tree(b, ctx)
    add_props(b, shared, seeded, ctx, snow_acc, focus)
    add_web(b, shared, ctx, focus)
    if ctx.env == "snow":
        b.ply(seeded.mesh(f"snow_layers_s{ctx.seed}", snow_acc.mesh), EYE, "snow", prefix="snow_layer")

    sun = SUN.get(ctx.env)
    b.d["sky"] = env_kit.sky(ctx.env, sun, scale=SKY_SCALE[ctx.env]) if sun else env_kit.sky(ctx.env, scale=SKY_SCALE[ctx.env])

    focus.update({
        "passage_wall": [-1.4, 1.5, 9.0],
        "passage_lantern": [-0.85, 2.0, 10.0],
        "archway": [0.0, 3.8, 7.0],
        "fountain": [FOUNTAIN[0], 0.5, FOUNTAIN[1] + 1.5],
        "fountain_top": [FOUNTAIN[0], 1.45, FOUNTAIN[1] + 0.1],
        "tree": [TREE[0], 1.5, TREE[1] + 0.2],
        "canopy": [TREE[0] - 0.4, 4.7, TREE[1] + 0.6],
        "bench": [BENCH[0][0], 0.5, BENCH[0][1]],
        "north_door": [1.5, 1.2, -9.2],
        "window_w": [-5.0, 1.6, -9.17],
        "ivy": [-3.7, 1.0, -9.0],
        "balcony": [1.5, 4.7, -8.28],
        "laundry": [6.12, 5.8, -8.0],
        "geraniums": [6.55, 0.32, -1.55],
        "east_door": [7.2, 1.2, -0.4],
        "pier_1": [-7.0, 1.5, 7.6 - 6.1],
        "pier_2": [-7.0, 1.5, 7.6 - 8.6],
        "pier_4": [-7.0, 1.5, 7.6 - 13.6],
        "walk_door": [-10.3, 1.2, -1.0],
        "chimney": [-3.0, 12.2, -10.6],
        "street": [0.0, 3.0, 24.0],
        "street_car": [-0.6, 1.0, 18.2],
        "lantern_n": [0.3, 1.95, -8.45],
        "cloister_bench": [-10.2, 0.6, 0.25],
        "aircon": [-3.6, 3.3, 6.8],
    })
    meta = {"sun_direction": sun, "shutters": {f"{k[0]}@{k[1]}": v for k, v in states.items()},
            "paint": {k: PAINTS[v] for k, v in paint_idx.items()}}
    return SceneBundle(b.d, focus, meta)


def _exclusions():
    ex = [((-10.6, 0.0, 6.75), (-1.15, 3.6, 13.2)), ((1.15, 0.0, 6.75), (7.0, 3.6, 13.2)),
          ((-7.65, 0.0, -9.2), (-6.85, 3.6, 7.0)),
          ((FOUNTAIN[0] - 1.8, 0.0, FOUNTAIN[1] - 1.8), (FOUNTAIN[0] + 1.8, 2.0, FOUNTAIN[1] + 1.8)),
          ((TREE[0] - 1.5, 0.0, TREE[1] - 1.5), (TREE[0] + 1.5, 3.4, TREE[1] + 1.5)),
          ((BENCH[0][0] - 1.0, 0.0, BENCH[0][1] - 1.0), (BENCH[0][0] + 1.0, 1.2, BENCH[0][1] + 1.0)),
          ((5.9, 0.0, 0.6), (7.0, 1.3, 2.4)),
          ((6.0, 0.0, -2.2), (7.0, 0.7, 0.9)), ((-1.4, 1.5, 9.5), (-0.5, 2.6, 10.5))]
    ex += [((x - 0.75, 0.0, z - 0.75), (x + 0.75, 2.3, z + 0.75)) for x, z in CITRUS]
    ex += [((3.3, 0.0, -9.0), (4.6, 1.3, -7.4)),                       # table and chair under the north window
           ((-10.5, 0.0, -0.45), (-9.6, 1.2, 0.95)), ((-10.3, 0.0, -1.5), (-9.5, 0.8, -0.7)),   # cloister bench, stool
           ((-10.5, 0.0, -5.5), (-9.8, 1.1, -4.0)),                    # cloister planter
           ((0.85, 0.0, 9.0), (1.4, 1.9, 12.4)), ((-1.4, 0.0, 10.0), (-0.95, 2.5, 12.4)),   # passage boxes, gate, broom
           ((1.9, 0.0, 6.0), (4.6, 1.1, 7.0)), ((-4.5, 2.6, 6.2), (-2.7, 3.9, 7.0)),        # bins, aircon
           ((6.3, 0.0, -5.2), (7.0, 1.3, -4.0)), ((6.1, 0.0, 2.0), (7.0, 1.4, 4.8)),        # east wall boxes, can, bucket
           ((5.8, 0.0, -9.0), (7.0, 0.9, -7.8)), ((-7.0, 0.0, -9.0), (-5.8, 0.9, -7.8)),    # corner pot clusters
           ((-7.0, 0.0, 5.8), (-5.8, 0.9, 7.0)), ((5.8, 0.0, 5.8), (7.0, 0.9, 7.0))]
    return tuple(ex)


SCENE = SceneDef(
    id="courtyard", group="hybrid", owner="rui",
    description="Old stone courtyard overtaken by plants: ivy and creeper on weathered walls, a cloister, "
                "a fountain under a plane tree, flagstones with moss, seen through a vaulted carriage passage.",
    build=build,
    views={
        # From inside the carriage passage, the dark vault framing the bright courtyard.
        "arch": View((0.25, 1.55, 12.3), (-0.3, 4.4, -9.0), focus="fountain"),
        # Along the ivy-covered north wall at a grazing angle, to a shuttered arched window.
        "window": View((-1.6, 1.55, -7.4), (-5.0, 1.7, -9.0), focus="window_w"),
        # Looking up into the canopy from near the trunk, the ivy-covered north-east corner behind it.
        "tree": View((0.6, 1.5, -1.6), (3.4, 4.6, -4.9), focus="canopy"),
        # Down the cloister walk, piers receding, sunlit court through the arches.
        "arcade": View((-8.7, 1.6, 6.3), (-8.3, 1.45, -8.5), focus="pier_2"),
    },
    default_view="arch",
    camera_box=((-10.2, 0.5, -8.6), (6.6, 3.2, 12.9)),
    target_box=((-10.5, 0.0, -9.6), (7.0, 7.5, 13.0)),
    exclude_boxes=_exclusions(),
    envs=("clear", "overcast", "wet", "snow"),
    tags=("outdoor", "day", "foliage", "architecture", "snow", "water"),
    default_seed=11, asset_version=ASSET_VERSION, max_depth=8, rr_depth=5, spp_hint=2048,
)
