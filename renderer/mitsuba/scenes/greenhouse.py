"""A Victorian botanical conservatory: iron-ribbed nave, dense foliage, a pond, benches, hanging baskets.

Layout, meters, y up, the nave runs along z (camera at +z looks toward -z, like the cafe):
  nave          x in [-4, 4], z in [-16, 4]; 0.9 m brick dwarf wall, vertical glazing to 3.0 m, then a
                circular arch (radius 4.1) to a 6.2 m ridge; iron ribs every 1 m, 16 glazing bars
  floor         quarry tiles; a 2 m central path with four raised stone-edged beds either side
  rotunda       z in [-8.2, -3.3], no beds: a round pond with a stone urn, two benches, potted palms
  far end       a 2 m doorway onto a paved terrace, a lawn and garden trees (the bright bokeh backdrop)
  plants        palms, ferns, banana-like tropicals, hostas, flowering shrubs and grass in the beds;
                climbers on the ribs; hanging baskets; ~40k leaves in total

THE GLASS TRAP, AND HOW THIS SCENE AVOIDS IT
Mitsuba's `path` integrator cannot do next-event estimation through a dielectric (or through a `mask` /
`null` pass-through): a point under a glass roof could only see the sun by a BSDF-sampled bounce that
happens to pass the pane and hit the tiny sun disc, so real glass panes would light the interior with
rare, enormous contributions (fireflies) however many samples we take. So:

  * Clear glass is an open cell between the glazing bars. From inside, clean glass is nearly invisible;
    what reads as "glass" in a photo is the bar grid with sky and garden behind it, which is what we
    render, with NEE to sun and sky unobstructed.
  * Summer shading paint (whitewash brushed onto the upper roof in patchy runs, a real practice) is a
    `principledthin` pane with diffuse transmission: non-delta, so paths landing on it still do NEE on the
    far side. Measured on a test plane: as quiet as an opaque surface (std/mean 0.27 vs 0.26).
  * Leaves are `principledthin` too (scanned foliage gets diffuse translucency from web_assets), so
    backlit leaves glow without delta transmission. The one specular dielectric is the pond surface.

Seed (`ctx.seed`) changes which plant species go where, their sizes and shapes, the lily
pads, which roof panes carry shading paint, and the garden trees. The architecture is the same.
"""

from __future__ import annotations

import math

import numpy as np

import env_kit
import procedural as G
import web_assets as W
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, bitmap, principled, rgb, xf
from scenes import _greenhouse_props as H

Z_NEAR, Z_FAR = 4.0, -16.0
POND_C = (0.0, -5.8)
BEDS = [(-3.35, -1.15, -15.3, -9.2), (1.15, 3.35, -15.3, -9.2), (-3.35, -1.15, -3.3, 3.4), (1.15, 3.35, -3.3, 3.4)]
BED_Y = 0.24
ASSET_VERSION = 9                                  # bump on any recipe change; also keys the seeded caches
SUN_ELEVATION, SUN_AZIMUTH = 48.0, 25.0
SKY_SCALE = {"clear": 3.4, "overcast": 7.2}
EYE = np.eye(4)
POTS = [(-2.75, -3.7, "big"), (2.75, -3.7, "big"), (-2.75, -7.9, "big"), (2.75, -7.9, "big"),
        (-3.5, -11.0, "small"), (3.5, -12.6, "small"), (-3.5, 2.6, "small"), (3.5, 0.0, "small")]
# Near garden trees, scanned: (id, x, z, height).
NEAR_TREES = [("searsia_lucida", 12.0, -38.0, 6.2), ("othonna_cerarioides", -14.0, -40.0, 5.2),
              ("tree_small_02", 2.0, -52.0, 7.0), ("island_tree_02", -6.0, -58.0, 8.0)]
RING_TREES = ("island_tree_02", "tree_small_02", "searsia_lucida")      # horizon ring: instances of loaded trees
# Scanned garden trees: (id, x, z, height). One in the doorway sight line, one beyond each side wall.
WEB_TREES = [("island_tree_02", -2.5, -31.0, 6.5), ("tree_small_02", -11.0, -4.0, 6.0), ("island_tree_02", 12.0, -12.0, 7.0)]
TREE_FOCUS = (-2.5, 3.9, -31.0)                    # in the canopy of the doorway tree
POTTING = (2.85, POND_C[1])                        # potting table in the rotunda, long side along z
# Fixed so the named focus points are guaranteed to sit on something: hero plants are placed here.
HERO_FERN, HERO_PALM, HERO_ANTHURIUM = (-1.45, BED_Y, 1.2), (2.2, BED_Y, 0.6), (1.5, BED_Y, -13.0)
# Hanging baskets: (anchor x, anchor z, chain drop). The first three hang from the arch.
ARCH_BASKETS = [(-1.0, -1.5, 2.2), (1.0, -10.0, 2.2), (-0.9, -13.0, 2.2)]
BRACKETS = [(-1, -4.6), (1, -7.0)]                # (side, z): basket on a wall bracket, hanging 0.75 m


def arch_anchor(x, z):
    return np.array([x, float(H.roof_y(x)) - 0.05, z])


def bracket_hook(sx, z):
    return np.array([sx * 3.9 - sx * 0.7, 2.75, z])


BASKET_CENTRES = ([arch_anchor(x, z) - [0, d, 0] for x, z, d in ARCH_BASKETS]
                  + [bracket_hook(sx, z) - [0, 0.75, 0] for sx, z in BRACKETS])


def _exclusions():
    """Where a camera must not go: bed volumes (to the wall), the pond column, benches, big pots, bracket baskets."""
    ex = []
    for (x0, x1, z0, z1) in BEDS:
        lo, hi = (x0 - 0.55, x1 + 0.05) if x0 < 0 else (x0 - 0.05, x1 + 0.55)
        ex.append(((lo, 0.0, z0 - 0.1), (hi, 3.4, z1 + 0.1)))
    ex.append(((POND_C[0] - 1.95, 0.0, POND_C[1] - 1.95), (POND_C[0] + 1.95, 2.4, POND_C[1] + 1.95)))
    for sx in (-1, 1):
        ex.append(((sx * 2.75 - 0.55, 0.0, POND_C[1] - 1.0), (sx * 2.75 + 0.55, 1.3, POND_C[1] + 1.0)))
    for px, pz, kind in POTS:
        if kind == "big":
            ex.append(((px - 0.6, 0.0, pz - 0.6), (px + 0.6, 3.3, pz + 0.6)))
    for sx, z in BRACKETS:
        cx = bracket_hook(sx, z)[0]
        ex.append(((cx - 0.55, 1.2, z - 0.55), (cx + 0.55, 2.6, z + 0.55)))
    return tuple(ex)


class WebLib:
    """Scanned Poly Haven models: whole, or one plant cut out of a variant sheet (each sheet split once and cached)."""

    def __init__(self, b: SceneBuilder, shared: Assets):
        self.b, self.shared, self.sheets = b, shared, {}

    def _materials(self, aid, m):
        for p in m.parts:
            mid = f"web_{aid}__{p.name}"
            if mid not in self.b.d:
                self.b.material(mid, p.bsdf)

    def variants(self, aid) -> list:
        """[(parts {name: ply path}, height m)] for each plant in the sheet, in order along x then z."""
        if aid not in self.sheets:
            m = W.model(aid)
            self._materials(aid, m)

            def build():
                meshes = {p.name: H.read_ply(p.ply) for p in m.parts}
                return {f"v{k}--{name}": mesh for k, v in enumerate(H.split_variants(meshes)) for name, mesh in v.items()}

            found: dict = {}
            for key, path in self.shared.meshes(f"web_{aid}_variants", build).items():
                k, name = key.split("--", 1)
                found.setdefault(int(k[1:]), {})[name] = path
            self.sheets[aid] = [(found[k], max(float(H.read_ply(q).p[:, 1].max()) for q in found[k].values()))
                                for k in sorted(found)]
        return self.sheets[aid]

    def plant(self, aid, k, at, yaw, height):
        """One plant from a sheet, as an instance of a per-variant shapegroup (geometry stored once)."""
        vs = self.variants(aid)
        k %= len(vs)
        parts, h = vs[k]
        group = f"sg_var_{aid}_{k}"
        if group not in self.b.d:
            self.b.d[group] = {"type": "shapegroup", **{f"part_{i}": {"type": "ply", "filename": path,
                                                                      "bsdf": {"type": "ref", "id": f"web_{aid}__{name}"}}
                                                       for i, (name, path) in enumerate(parts.items())}}
        m = G.compose(G.translate(at), G.rotate((0, 1, 0), yaw), G.scale(height / h))
        self.b.d[self.b._key(f"inst_var_{aid}")] = {"type": "instance", "shapegroup": {"type": "ref", "id": group},
                                                    "to_world": xf(m)}

    def whole(self, aid, at, yaw=0.0, height=None, scale=1.0, tilt=0.0):
        """Whole model (instanced by web_assets). `tilt` leans it about its local x axis, in degrees."""
        m = W.model(aid)
        mat = W.place(m, at=at, yaw=yaw, height=height, scale=scale)
        if tilt:
            mat = G.compose(G.translate(at), G.rotate((0, 1, 0), yaw), G.rotate((1, 0, 0), tilt),
                            G.translate(-np.asarray(at, float)), mat)
        W.add(self.b, m, mat)


def thin(base, roughness, diff_trans, **kw) -> dict:
    """Thin translucent surface: diffuse reflection plus diffuse transmission (no delta lobes)."""
    return {"type": "principledthin", "base_color": base if isinstance(base, dict) else rgb(*base),
            "roughness": roughness, "diff_trans": diff_trans, **kw}


def add_materials(b: SceneBuilder) -> None:
    A = b.assets
    brick = A.texture("brick", H.brick_texture)
    tiles = A.texture("quarry", H.quarry_tiles)
    soil = A.texture("soil", H.soil_texture)
    stone = A.texture("stone", H.stone_texture)
    grime = A.texture("pane_grime", H.pane_grime)
    atlas = A.texture("leaf_atlas", H.leaf_atlas)
    wood_rgb, _ = G.wood_planks(1024, 512, planks=6, board_len=0.5, seed=5)
    wood = A.texture("bench_wood", lambda: wood_rgb * 0.8)

    bw, bh = H.BRICK_TILE_M
    b.material("iron_dark", principled((0.02, 0.022, 0.02), 0.45, metallic=0.5))
    b.material("pane", thin(bitmap(grime), 0.9, 0.9))   # shading paint with algae along the bars, run-off streaks
    # scanned CC0 surfaces (Poly Haven) at real-world scale on meshes whose uv is in meters
    b.material("brick", W.surface("brick_moss_001"))
    b.material("floor", W.surface("patterned_terracotta_tiling"))
    b.material("soil", W.surface("brown_mud_leaves_01"))
    b.material("stone", W.surface("mossy_sandstone"))
    # lathe meshes have uv = (turn, profile arc), so the scale is circumference and profile length over 1.53 m
    b.material("stone_rim", W.surface("mossy_sandstone", uv_scale=(10.1 / 1.89, 1.2 / 1.89)))
    b.material("stone_urn", W.surface("mossy_sandstone", uv_scale=(2.2 / 1.89, 2.0 / 1.89)))
    # sweeps have uv = (turn, run along the path): ~0.2 m round, ~15 m long ribs and bars, 1 m texture
    b.material("iron", W.surface("green_metal_rust", uv_scale=(0.2, 15.0)))
    b.material("trunk", W.surface("bark_brown_02", uv_scale=(0.7, 3.0)))
    b.material("puddle", principled((0.03, 0.03, 0.025), 0.04, specular=0.5))
    b.material("spill", W.surface("brown_mud_leaves_01"))
    b.material("label", principled((0.85, 0.85, 0.82), 0.4))
    b.material("hose", principled((0.08, 0.32, 0.12), 0.35, clearcoat=0.3))
    b.material("terrace", W.surface("stone_pathway_02", uv_scale=(9.0 / 1.95, 6.4 / 1.95)))
    b.material("lawn_scan", W.surface("leafy_grass", uv_scale=(1500.0, 1500.0), tint=(0.40, 0.58, 0.32)))   # mown, watered: greener than the scan
    b.material("terracotta", principled((0.55, 0.24, 0.12), 0.7))
    b.material("glaze_blue", principled((0.08, 0.20, 0.42), 0.12, clearcoat=0.6, clearcoat_gloss=0.9))
    b.material("wood", principled(bitmap(wood), 0.55))
    b.material("stem", principled((0.13, 0.17, 0.06), 0.65))
    b.material("chain", principled((0.03, 0.03, 0.03), 0.5, metallic=0.8))
    b.material("moss", principled((0.07, 0.17, 0.05), 0.95))
    b.material("hedge_core", W.surface("leafy_grass", tint=(0.32, 0.55, 0.26)))   # leafy mass between the shrubs
    b.material("basket", principled((0.04, 0.04, 0.035), 0.5, metallic=0.6))
    b.material("grass", thin((0.18, 0.42, 0.08), 0.6, 0.4))
    b.material("succulent", thin((0.23, 0.36, 0.28), 0.5, 0.2))
    b.material("pad", thin((0.10, 0.30, 0.10), 0.3, 0.4))
    b.material("leaf", thin(bitmap(atlas), 0.42, 0.55))
    for key, col in (("pink", (0.85, 0.16, 0.35)), ("orange", (0.95, 0.38, 0.06)), ("yellow", (0.95, 0.75, 0.08)),
                     ("white", (0.9, 0.88, 0.82)), ("red", (0.7, 0.04, 0.05))):
        b.material(f"flower_{key}", thin(col, 0.5, 0.35))
    # Dark, near-mirror water with no transmission: refracting water made the sun's caustics (light reaching the
    # pond bed only through a delta interface) show up as fireflies. A still garden pond reads like this anyway.
    b.material("water", principled((0.012, 0.025, 0.02), 0.08, specular=0.45))
    b.material("pond_bed", {"type": "diffuse", "reflectance": rgb(0.16, 0.20, 0.13)})   # mossy stone: light enough to see through the water


def build(ctx: BuildContext) -> SceneBundle:
    pane_rng = np.random.default_rng([ctx.seed, 1])        # which panes are open vents
    plant_rng = np.random.default_rng([ctx.seed, 2])       # every plant, pad and tree
    shared = Assets(ctx.shared_dir, ctx.rebuild)          # architecture: the same for every seed
    seeded = Assets(ctx.assets_dir, ctx.rebuild)          # plants, vents, pads, trees: depend on the seed
    b = SceneBuilder(shared)
    add_materials(b)
    s = ctx.seed
    focus: dict[str, list] = {}

    # --- architecture (seed-independent, cached once) -------------------------------------------------
    iron = shared.mesh("iron_frame", lambda: H.iron_frame(Z_NEAR, Z_FAR))
    b.ply(iron, EYE, "iron", prefix="iron")
    ie_far, pe_far, br_far = H.end_wall(Z_FAR, door=True)
    ie_near, pe_near, br_near = H.end_wall(Z_NEAR, door=False)
    ends = shared.meshes("end_walls", lambda: {"iron": _merge(ie_far, ie_near), "brick": _merge(br_far, br_near)})
    b.ply(ends["iron"], EYE, "iron", prefix="end_iron")
    b.ply(ends["brick"], EYE, "brick", prefix="end_brick")
    for side in (-1, 1):                                  # dwarf brick walls along the nave
        wall = shared.mesh(f"dwarf_{side}", lambda sd=side: H.brick_slab((sd * 3.85, Z_NEAR), (sd * 3.85, Z_FAR),
                                                                          0.0, H.Y_WALL, 0.3))
        b.ply(wall, EYE, "brick", prefix="dwarf")
    floor = shared.mesh("floor", lambda: G.box((8.0, 0.1, 20.0), (0.0, -0.05, -6.0)))
    b.ply(floor, EYE, "floor", prefix="floor")

    # --- roof panes: the seed picks which are open vents -----------------------------------------------
    pane_mesh, vents = H.roof_panes(pane_rng, Z_NEAR, Z_FAR)
    b.ply(seeded.mesh(f"roof_panes_s{s}_v{ASSET_VERSION}", lambda: pane_mesh), EYE, "pane", prefix="roof_pane")

    # --- beds ---------------------------------------------------------------------------------------------
    for (x0, x1, z0, z1) in BEDS:
        kerb = shared.mesh(f"kerb_{x0}_{z0}", lambda a=(x0, x1, z0, z1): H.stone_edge(*a))
        b.ply(kerb, EYE, "stone", prefix="kerb")
        top = shared.mesh(f"bed_soil_{x0}_{z0}", lambda a=(x0, x1, z0, z1): G.box(
            (a[1] - a[0] - 0.1, BED_Y, a[3] - a[2] - 0.1), ((a[0] + a[1]) / 2, BED_Y / 2, (a[2] + a[3]) / 2)))
        b.ply(top, EYE, "soil", prefix="bed")

    # --- rotunda: pond, urn, benches, pots -----------------------------------------------------------------
    pond = shared.meshes("pond", lambda: H.pond(POND_C))
    b.ply(pond["rim"], G.translate((POND_C[0], 0.0, POND_C[1])), "stone_rim", prefix="pond_rim")
    b.ply(shared.mesh("urn", H.urn), G.translate((POND_C[0], 0.04, POND_C[1])), "stone_urn", prefix="urn")
    b.d["pond_floor"] = {"type": "disk", "to_world": _xf(G.compose(G.translate((POND_C[0], 0.04, POND_C[1])),
                                                                 G.rotate((1, 0, 0), -90), G.scale(1.45))),
                       "bsdf": {"type": "ref", "id": "pond_bed"}}
    b.d["pond_surface"] = {"type": "disk", "to_world": _xf(G.compose(G.translate((POND_C[0], 0.36, POND_C[1])),
                                                                   G.rotate((1, 0, 0), -90), G.scale(1.45))),
                         "bsdf": {"type": "ref", "id": "water"}}
    focus["urn"] = [POND_C[0], 1.0, POND_C[1]]
    focus["bench_l"] = [-2.75, 0.45, POND_C[1]]           # scanned painted bench, placed in _scanned

    def build_plants():
        grp = H.Groups()
        _plants(grp, plant_rng)
        return grp.meshes()

    leaf_keys = {"leaf": "leaf", "stem": "stem", "trunk": "trunk", "grass": "grass", "succulent": "succulent",
                 "pad": "pad", "chain": "chain", "basket": "basket", "moss": "moss", "hedge_core": "hedge_core",
                 "label": "label", "hose": "hose", "puddle": "puddle", "spill": "spill",
                 **{k: k for k in H.FLOWER_KEYS}}
    plant_paths = seeded.meshes(f"plants_s{s}_v{ASSET_VERSION}", build_plants)
    for part, path in plant_paths.items():
        b.ply(path, EYE, leaf_keys[part], prefix=f"plant_{part}")

    # potted plants (pots are cheap meshes; the plants inside are part of the seeded group above)
    # --- scanned plants, potting bench, garden -------------------------------------------------------------
    _scanned(WebLib(b, shared), np.random.default_rng([ctx.seed, 3]))

    # --- exterior: terrace, lawn -------------------------------------------------------------------------
    b.d["terrace_slab"] = {"type": "rectangle", "to_world": _xf(G.compose(G.translate((0.0, 0.0, -19.2)),
                                                                     G.rotate((1, 0, 0), -90), G.scale((4.5, 3.2, 1)))),
                      "bsdf": {"type": "ref", "id": "terrace"}}
    b.d["lawn_plane"] = {"type": "rectangle", "to_world": _xf(G.compose(G.translate((0.0, -0.02, -20.0)),
                                                                  G.rotate((1, 0, 0), -90), G.scale(3000.0))),
                   "bsdf": {"type": "ref", "id": "lawn_scan"}}

    # --- light -----------------------------------------------------------------------------------------------
    sun = env_kit.default_sun_direction(SUN_ELEVATION, SUN_AZIMUTH)
    b.d["sky"] = env_kit.sky(ctx.env, sun, scale=SKY_SCALE[ctx.env])

    sx = POTS[-1]
    focus.update({
        "door": [0.0, 2.55, Z_FAR],
        "tree": list(TREE_FOCUS),
        "pond_rim": [POND_C[0] + 1.615, 0.52, POND_C[1]],
        "fern_near": [HERO_FERN[0], 0.62, HERO_FERN[2]],
        "palm_near": [HERO_PALM[0], 1.5, HERO_PALM[2]],
        "anthurium_far": [HERO_ANTHURIUM[0], 0.55, HERO_ANTHURIUM[2]],
        "potting": [POTTING[0], 0.86, POTTING[1]],
        "brick": [3.7, 0.5, -1.0],
        "terrace": [0.0, 0.0, -19.5],
        "potted_plant": [sx[0], 0.5, sx[1]],
    })
    for i, c in enumerate(BASKET_CENTRES):
        focus[f"basket_{i}"] = [float(c[0]), float(c[1]), float(c[2])]
    return SceneBundle(b.d, focus, {"sun_direction": sun, "clear_roof_panes": vents,
                                    "glass": "clear panes are open cells; shading-painted panes are principledthin; see module docstring"})


def _merge(*meshes):
    acc = H.Acc()
    for m in meshes:
        acc.add(m)
    return acc.mesh()


def _xf(m):
    import mitsuba as mi

    return mi.ScalarTransform4f(np.asarray(m, dtype=np.float32))




def _plants(grp, rng) -> None:
    """Every plant in the scene, appended to `grp` in world space."""
    # Poly Haven has no palm, so the hero palm and one palm per bed stay procedural (with scanned bark)
    H.palm(grp, rng, HERO_PALM, 2.6, 12, 1.45)
    for (x0, x1, z0, z1) in BEDS:
        sg = 1.0 if x0 > 0 else -1.0
        H.palm(grp, rng, (sg * rng.uniform(2.4, 2.9), BED_Y, rng.uniform(z0 + 1.0, z1 - 1.0)), rng.uniform(2.3, 3.0), 12, 1.45)
    # climbers on the ribs
    for z in rng.choice(np.arange(Z_NEAR - 1.0, Z_FAR + 0.5, -1.0), 9, replace=False):
        for s0 in (rng.uniform(0.5, 3.0), H.PROFILE_LEN - rng.uniform(5.0, 8.0)):
            H.climber_on_rib(grp, rng, float(z), s0, s0 + rng.uniform(4.5, 6.5), every=0.1)
    # hanging baskets: three on chains from the arch, one pair on wall brackets in the rotunda
    for x, z, drop in ARCH_BASKETS:
        H.hanging_basket(grp, rng, arch_anchor(x, z), drop=drop, radius=0.21, fill=False)
    for sx, z in BRACKETS:
        wall_pt, hook = np.array([sx * 3.9, 2.7, z]), bracket_hook(sx, z)
        grp["chain"].add(G.sweep(np.stack([wall_pt, (wall_pt + hook) / 2 + [0, 0.08, 0], hook]), 0.012, 6))
        H.hanging_basket(grp, rng, hook, drop=0.75, radius=0.19, trail=0.7, fill=False)
    # pond: lily pads and blooms
    for i in range(13):
        a, r = rng.uniform(0, 2 * math.pi), 0.55 + 0.75 * math.sqrt(rng.uniform(0, 1))
        H.lily_pad(grp, rng, (POND_C[0] + r * math.cos(a), 0.365, POND_C[1] + r * math.sin(a)), rng.uniform(0.1, 0.22))
    for _ in range(3):
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.7, 1.2)
        H.bloom(grp, rng, "flower_pink", (POND_C[0] + r * math.cos(a), 0.4, POND_C[1] + r * math.sin(a)), [0, 1, 0], 0.07)
    # hedge cores (the leaves are scanned shrubs, placed in _scanned)
    for sx in (-1, 1):
        grp["hedge_core"].add(G.box((0.7, 1.2, 30.0), (sx * 9.5, 0.6, -7.0)))
    grp["hedge_core"].add(G.box((26.0, 1.4, 0.8), (0.0, 0.7, -38.0)))
    # wear: puddles by the pond and under the watering cans, soil spilled at the bed edges, labels, a hose
    for (x, z, r) in ((1.3, -3.6, 0.35), (-1.1, -8.0, 0.5), (2.3, -6.6, 0.3), (0.4, -9.4, 0.25), (-0.5, 1.9, 0.3)):
        grp["puddle"].add(H.blob(rng, (x, 0, z), r))
    for (x0, x1, z0, z1) in BEDS:
        edge = x1 if x0 < 0 else x0
        for _ in range(3):
            grp["spill"].add(H.blob(rng, (edge + (0.12 if x0 < 0 else -0.12), 0, rng.uniform(z0 + 0.3, z1 - 0.3)),
                                    rng.uniform(0.08, 0.2), y=0.002))
        for _ in range(4):
            sg = 1.0 if x0 > 0 else -1.0
            H.label(grp, rng, (sg * rng.uniform(1.3, 1.8), BED_Y, rng.uniform(z0 + 0.3, z1 - 0.3)), rng.uniform(0, 360))
    tx, tz = POTTING
    for dz in (-0.48, -0.42, -0.36, -0.22, -0.14):
        H.label(grp, rng, (tx - 0.05 + rng.uniform(-0.06, 0.06), 0.86, tz + dz), rng.uniform(0, 360))
    H.coiled_hose(grp, (-3.3, 0.0, -9.0))


def _scanned(lib: WebLib, rng) -> None:
    """Scanned CC0 plants and props (Poly Haven), placed with `rng` (seeded, independent of the cache)."""
    U = rng.uniform
    yaw = lambda: float(U(0, 360))
    lib.plant("anthurium_botany_01", 0, HERO_ANTHURIUM, 30.0, 0.7)
    lib.plant("fern_02", 0, HERO_FERN, 20.0, 0.62)
    for (x0, x1, z0, z1) in BEDS:
        sg = 1.0 if x0 > 0 else -1.0
        zr = lambda m=0.4: float(U(z0 + m, z1 - m))
        for _ in range(2):          # one is the plan's replacement for a procedural bed palm
            lib.plant("pachira_aquatica_01", int(rng.choice([0, 2, 3])), (sg * U(2.1, 2.9), BED_Y, zr(0.8)), yaw(), U(2.0, 2.8))
        lib.plant("shrub_02", int(rng.integers(4)), (sg * U(2.6, 3.0), BED_Y, zr(0.7)), yaw(), U(1.0, 1.3))
        for _ in range(2):
            lib.plant("anthurium_botany_01", int(rng.integers(99)), (sg * U(1.5, 2.4), BED_Y, zr()), yaw(), U(0.55, 0.85))
            lib.plant("calathea_orbifolia_01", int(rng.choice([0, 2, 4])), (sg * U(1.4, 2.2), BED_Y, zr()), yaw(), U(0.35, 0.5))
        for _ in range(5):          # three plus the two procedural ferns they replace
            lib.plant("fern_02", int(rng.integers(99)), (sg * U(1.35, 2.3), BED_Y, zr()), yaw(), U(0.5, 0.7))
        for _ in range(4):
            lib.plant("periwinkle_plant", int(rng.choice([1, 2, 3])), (sg * U(1.3, 3.1), BED_Y, zr(0.2)), yaw(), U(0.28, 0.4))
            lib.plant("shrub_04", int(rng.integers(2)), (sg * U(1.3, 3.1), BED_Y, zr(0.2)), yaw(), U(0.2, 0.3))
    # pots: the four big-pot palms and the aisle pots are scanned potted plants (they bring their own pots)
    for i, (px, pz, kind) in enumerate(POTS):
        aid = "potted_plant_01" if i % 2 else "potted_plant_02"
        lib.whole(aid, (px, 0.0, pz), yaw(), height=U(1.5, 1.75) if kind == "big" else U(0.8, 1.2))
    # hanging baskets: moss-lined frames filled with trailing periwinkle, a tuft of shrub_04, gazania
    for c in BASKET_CENTRES:
        top = (float(c[0]), float(c[1]) + 0.05, float(c[2]))
        lib.plant("periwinkle_plant", int(rng.choice([1, 2, 3])), top, yaw(), 0.32)
        lib.plant("flower_gazania", int(rng.integers(99)), (top[0] + 0.07, top[1], top[2] - 0.05), yaw(), 0.18)
        lib.plant("shrub_04", int(rng.integers(2)), (top[0] - 0.08, top[1], top[2] + 0.06), yaw(), 0.16)
    # rotunda: painted bench, potting table with its clutter
    lib.whole("painted_wooden_bench", (-2.75, 0.0, POND_C[1]), 90.0)
    tx, tz = POTTING
    lib.whole("WoodenTable_03", (tx, 0.0, tz), -90.0)
    top = 0.83
    lib.whole("seeding_tray_01", (tx - 0.05, top, tz - 0.42), yaw())
    lib.whole("seeding_tray_01", (tx + 0.02, top, tz - 0.18), yaw())
    for dz in (0.12, 0.33):
        lib.whole("planter_pot_clay", (tx + U(-0.12, 0.1), top, tz + dz), yaw(), scale=U(0.7, 0.9))
    lib.whole("wicker_basket_01", (tx, top, tz + 0.55), 90.0)
    lib.whole("potted_plant_04", (tx + 0.12, top, tz - 0.55), yaw())
    lib.whole("trowel_01", (tx - 0.14, top, tz + 0.2), 75.0)
    lib.whole("garden_gloves_01", (tx + 0.1, top, tz - 0.02), yaw())
    lib.whole("watering_can_metal_01", (tx - 0.55, 0.0, tz - 0.75), 120.0)
    lib.whole("watering_can_metal_01", (1.25, 0.0, -3.55), 200.0)
    lib.whole("wooden_crate_01", (tx + 0.05, 0.0, tz + 0.25), 90.0)
    for s, (sx, sz) in enumerate(((tx + 0.05, tz - 0.35), (tx - 0.05, tz - 0.62))):     # nested stacks of clay pots
        for k in range(3):
            lib.whole("planter_pot_clay", (sx, 0.06 * k, sz), yaw(), scale=0.95)
    lib.whole("compost_bag_02", (3.25, 0.0, -4.75), yaw())
    lib.whole("wooden_stool_01", (tx - 0.6, 0.0, tz + 0.15), yaw())
    lib.whole("plastic_crate_01", (2.25, 0.0, -7.05), 15.0)
    lib.whole("seeding_tray_01", (2.25, 0.264, -7.05), 15.0)
    # doorway and inside
    lib.whole("garden_hose_wall_mounted_01", (-1.6, 0.25, Z_FAR + 0.28), 0.0)
    lib.whole("rubber_boots", (1.15, 0.0, Z_FAR + 0.5), yaw())
    lib.whole("rusted_spade_01", (3.5, 0.0, -2.2), 90.0, tilt=-14.0)
    lib.whole("wooden_bucket_01", (-1.35, 0.0, -3.7), yaw())
    for (x, y, z) in ((-1.25, 0.26, -12.0), (1.25, 0.26, 1.5), (1.4, 0.52, POND_C[1] + 0.6), (-1.0, 0.52, POND_C[1] - 1.2)):
        lib.whole("moss_01", (x, y, z), yaw(), scale=U(0.35, 0.5))
    # garden: scanned trees near and on the horizon, shrub hedges, terrace and lawn clutter
    for aid, x, z, h in WEB_TREES:
        lib.whole(aid, (x, 0.0, z), yaw(), height=h)
    for aid, x, z, h in NEAR_TREES:
        lib.whole(aid, (x, 0.0, z), yaw(), height=h * U(0.9, 1.1))
    for a in np.linspace(0, 2 * math.pi, 22, endpoint=False):
        r = U(45, 75)
        lib.whole(str(rng.choice(RING_TREES)), (r * math.sin(a + U(-0.1, 0.1)), 0.0, -6.0 + r * math.cos(a)), yaw(),
                  height=U(8.0, 13.0))
    for sx in (-1, 1):                                   # hedges: shrub_02 every 0.9 m on a dark core
        for z in np.arange(-21.5, 7.6, 0.9):
            lib.plant("shrub_02", int(rng.integers(4)), (sx * 9.5 + U(-0.15, 0.15), 0.0, float(z)), yaw(), U(1.3, 1.9))
    for x in np.arange(-12.5, 12.6, 0.9):
        lib.plant("shrub_02", int(rng.integers(4)), (float(x), 0.0, -38.0 + U(-0.15, 0.15)), yaw(), U(1.4, 2.0))
    for _ in range(5):
        lib.plant("shrub_02", int(rng.integers(4)), (U(-3.5, 3.5), 0.0, U(-25.0, -21.5)), yaw(), U(1.1, 1.7))
    for sx in (-1, 1):
        for z in np.arange(-14.5, 4.0, 3.4):             # long grass along the glasshouse base
            lib.whole("grass_medium_02", (sx * 4.6, 0.0, float(z)), 90.0 + U(-8, 8), scale=U(0.6, 0.8))
    for sx in (-1, 1):
        lib.whole("planter_box_02", (sx * 2.9, 0.0, -17.0), 0.0)
        for k in range(3):
            lib.plant("periwinkle_plant", int(rng.choice([1, 2, 3])), (sx * 2.9 + (k - 1) * 0.38, 0.40, -17.0), yaw(), U(0.3, 0.38))
        lib.whole("planter_box_01" if sx < 0 else "planter_box_03", (sx * 3.7, 0.0, -21.5), 90.0)
        lib.plant("periwinkle_plant", int(rng.choice([1, 2, 3])), (sx * 3.7, 0.42 if sx < 0 else 0.84, -21.5), yaw(), 0.34)
    for _ in range(26):                                  # scanned grass over the lawn the doorway and walls look onto
        lib.whole("grass_medium_02", (U(-8.5, 8.5), 0.0, U(-36.0, -21.5)), yaw(), scale=U(0.55, 0.85))
    for sx in (-1, 1):
        for _ in range(8):
            lib.whole("grass_medium_02", (sx * U(5.6, 8.8), 0.0, U(-20.0, 6.0)), yaw(), scale=U(0.55, 0.8))
    lib.whole("compost_bags", (-3.0, 0.0, -19.3), 90.0)
    lib.whole("garden_sprinkler_01", (1.6, 0.0, -23.0), yaw())
    lib.whole("tree_stump_01", (4.8, 0.0, -24.5), yaw())
    lib.whole("rock_moss_set_01", (-6.5, 0.0, -21.0), yaw(), scale=0.3)
    lib.whole("picke_dirty_01", (2.1, 0.0, -20.6), 60.0, tilt=-70.0)
    lib.whole("wooden_bucket_02", (-2.0, 0.0, -21.0), yaw())


CAMERA_BOX = ((-3.5, 0.55, -15.0), (3.5, 2.4, 3.5))
TARGET_BOX = ((-3.8, 0.0, -15.8), (3.8, 3.6, 3.5))   # glass roof: aiming higher mostly frames sky


SCENE = SceneDef(
    id="greenhouse", group="hybrid", owner="rui",
    description="Victorian conservatory: iron-ribbed glazed nave, dense foliage, a pond, benches and hanging "
                "baskets, with a bright garden seen through the far doorway.",
    build=build,
    views={
        # Down the central path toward the doorway, urn in mid-distance.
        "path": View((0.25, 1.55, 3.2), (0.0, 1.6, -12.0), focus="urn"),
        # Low and close from the path: the left bed, bench and pond edge.
        "close": View((0.35, 1.05, -2.2), (-2.3, 0.75, -5.4), focus="bench_l"),
        # Looking up into the ribs and hanging baskets.
        "roof": View((0.0, 1.3, -1.0), (0.0, 3.5, -9.0), focus="basket_0"),
        # The potting bench beside the pond.
        "potting": View((0.9, 1.35, -3.7), (2.8, 0.8, -6.0), focus="potting"),
        # Over the pond toward the doorway, lily pads in front.
        "pond": View((0.6, 1.0, -2.6), (0.0, 0.9, -9.5), focus="pond_rim"),
    },
    default_view="path",
    camera_box=CAMERA_BOX, target_box=TARGET_BOX,
    exclude_boxes=_exclusions(),
    envs=("clear", "overcast"),
    tags=("indoor", "day", "glass", "foliage", "architecture", "water"),
    default_seed=3, asset_version=ASSET_VERSION, max_depth=10, rr_depth=5, spp_hint=2048,
)
