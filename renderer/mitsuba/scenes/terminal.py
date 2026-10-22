"""A train-station / airport concourse: the largest indoor depth range of the set (about 1 m to 90 m inside, a
skyline 250 m beyond the open far end), with steel trusses, a polished reflective floor, signs and departure boards.

Layout, meters, y up, the hall runs along z and the default camera looks toward -z:
  nave         x in [-8.5, 8.5], z in [8, -82], under a barrel vault (arc radius 8.81, crown at y = 15.5) on 16 truss ribs
  colonnades   a round steel column under each rib, both sides (x = +-8.5)
  aisles       x in [8.5, 12] either side: shopfronts at ground level (y < 4.4), a gallery at y = 4.8 with posters and
               clerestory windows above, a flat soffit at y = 9
  nave contents  check-in counters and queue posts (left), seating, kiosk and luggage zones (right), a bridge across the
               nave at z = -43 joining the galleries, a departures board, clocks and hanging signs, potted trees
  ends         open truss frames at z = 8 and z = -82; outside is a forecourt and a skyline of office towers

THE GLAZING TRAP, and what this scene does about it: Mitsuba's `path` integrator cannot do next-event estimation
through glass (any surface blocks light-sampling shadow rays), and no BSDF, not even `null`, avoids that. A glass
roof would light the hall only through lucky BSDF-sampled paths: huge noise, and the sun disc would never be found.
So NOTHING in any light path is glazed. The skylight along the crown, the clerestory slots in the vault, the
clerestory windows in the gallery walls and the two end frames are OPEN apertures framed by ribs, glazing bars and
mullions; they read as glazing structure and let the sun and sky in directly. The cost is no glass reflections or
tint, which is acceptable for a depth-of-field dataset.

Environments: `clear` has a high sun (shafts and patches on the polished floor); `overcast` has no sun, only the
skylight, openings and artificial light. The seed picks floor tile colours, which shops and posters exist, the
contents of the departure boards, where luggage lies and stands, and which hanging signs go where.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import mitsuba as mi
import env_kit
import web_assets as W
import procedural as G
from scene_api import BuildContext, SceneBundle, SceneDef, View
from scene_kit import Assets, SceneBuilder, luminance, principled, rgb, xf
from scenes import _terminal_props as P
from scenes import _terminal_tex as T

NH, HH = 8.5, 12.0                                  # nave and hall half-widths
Z_BAYS = [8.0 - 6.0 * k for k in range(16)]          # rib and column stations, 8 .. -82
Z_NEAR, Z_FAR = Z_BAYS[0], Z_BAYS[-1]
BAY_C = [z - 3.0 for z in Z_BAYS[:-1]]               # centres of the 15 bays
R_IN, YC, TRUSS_D = 8.81, 6.69, 0.55
R_SH = R_IN + TRUSS_D + 0.03
HALF_DEG = 74.8
GAL_Y, GAL_T, SOFFIT = 4.8, 0.35, 9.0
SKYLIGHT_DEG, SLOT_DEG = 15.0, (48.0, 60.0)
BRIDGE_Z = -43.0
ESC_GAP = (-20.0, -10.8)                             # right-hand gallery slab opening for the escalators
ESC_BAYS = (3, 4)                                    # right-hand bays without shopfronts (the escalators stand there)
SEAT_PAIRS = [(-9.0 - 2.5 * i) for i in range(6)] + [(-33.0 - 2.5 * i) for i in range(6)] + [(-60.0 - 2.5 * i) for i in range(4)]
SEAT_X = 5.2
TREES = [(-6.4, -3.0), (-6.4, -47.0), (6.9, -26.0), (6.9, -52.0), (-6.4, -70.0), (6.9, -72.0), (6.6, 2.0), (-3.5, -76.0)]
TROLLEY_ZONE = ((-7.4, -4.6), (-4.2, 2.6))           # x range, z range
LUGGAGE_ZONES = [((6.0, 7.5), (-22.0, -8.0)), ((6.0, 7.5), (-46.0, -33.0)), ((6.0, 7.4), (-68.0, -59.0)),
                 ((-5.6, -3.4), (-34.0, -16.0))]
COUNTER_Z = [(-15.5 - 2.7 * i) for i in range(7)]    # near edge of each of the 7 check-in desks, running toward -z
KIOSK = (4.4, -50.0)
WET_SIGNS = [(-1.6, -21.0, 30.0), (2.2, -57.0, -20.0)]
LOUNGE = [(-6.2, -57.5, 70.0), (-6.4, -59.3, 110.0)]
INFO_DESK = (-4.8, -52.0)

# radiance units: see the exposure notes in the report; tuned so the median sharp luminance is inside [0.02, 0.5]
SUN_DIR = env_kit.default_sun_direction(58.0, 25.0)
SKY_SCALE = {"clear": 12.0, "overcast": 27.0}         # clear: sunlit views reached median 0.43 at 15, too close to 0.5
ASSET_VERSION = 3                                    # bump with every recipe change (make_context versions both caches)
SHOP_GAIN, POSTER_GAIN = 0.55, 0.6                   # emission multipliers: keep daylight the main light of the nave
SKY_SHARE = 2.0                                      # sky's light-sampling share relative to all artificial emitters
FIXTURE_L = {"warm": (1.0, 0.76, 0.50), "cool": (0.88, 0.94, 1.0)}
FIXTURE_RAD = 11.0
PAINTS = [(0.02, 0.05, 0.14), (0.01, 0.01, 0.012), (0.55, 0.05, 0.05), (0.60, 0.60, 0.62), (0.05, 0.25, 0.15),
          (0.70, 0.45, 0.05), (0.30, 0.10, 0.35), (0.12, 0.12, 0.14), (0.40, 0.15, 0.08)]


def tex(path: str, raw: bool = False, scale=None, to_uv=None) -> dict:
    spec = {"type": "bitmap", "filename": path, "raw": raw, "filter_type": "bilinear", "wrap_mode": "repeat"}
    if scale is not None:
        spec["to_uv"] = mi.ScalarTransform4f().scale([scale[0], scale[1], 1.0])
    if to_uv is not None:
        spec["to_uv"] = to_uv
    return spec


def exr(A: Assets, name: str, arr_fn) -> str:
    path = A.root / "textures" / f"{name}.exr"
    if A.rebuild or not path.exists():
        P.save_exr(path, arr_fn())
    return str(path)


def png(A: Assets, name: str, arr_fn, gray: bool = False) -> str:
    path = A.root / "textures" / f"{name}.png"
    if A.rebuild or not path.exists():
        arr = arr_fn()
        T.save_png8(path, arr if not gray else np.repeat(arr[..., None], 3, -1))
    return str(path)


def flip(a: np.ndarray) -> np.ndarray:
    """Rectangle uv has v = 0 at the bottom, bitmap rows start at the top: store rectangle maps bottom-up."""
    return np.ascontiguousarray(a[::-1])


def floor_paths(Ad: Assets, seed: int) -> tuple[str, str]:
    a, r = Ad.root / "textures" / "floor_albedo.png", Ad.root / "textures" / "floor_rough.png"
    if Ad.rebuild or not (a.exists() and r.exists()):
        alb, rough = T.floor_textures(seed)
        T.save_png8(a, alb)
        T.save_png8(r, np.repeat(rough[..., None], 3, -1))
    return str(a), str(r)


def add_materials(b: SceneBuilder, As: Assets, floor: tuple[str, str], env: str) -> None:
    b.material("floor", principled(tex(floor[0], scale=(1 / 12.0, 1 / 12.0)), tex(floor[1], raw=True, scale=(1 / 12.0, 1 / 12.0)),
                                   specular=0.5, clearcoat=0.25, clearcoat_gloss=0.95))
    b.material("ground_out", env_kit.ground(env))
    b.material("steel_white", principled((0.80, 0.80, 0.78), 0.45, metallic=0.05))
    b.material("steel_dark", principled((0.05, 0.05, 0.055), 0.38, metallic=0.6))
    b.material("vault", {"type": "diffuse", "reflectance": rgb(0.74, 0.72, 0.68)})
    plaster = png(As, "plaster", lambda: G.plaster(1024, 1024, (0.80, 0.76, 0.68), seed=4))
    b.material("plaster", principled(tex(plaster, scale=(1 / 6.0, 1 / 6.0)), 0.8))
    b.material("ceiling", {"type": "diffuse", "reflectance": rgb(0.80, 0.80, 0.78)})
    b.material("gallery_floor", principled((0.36, 0.34, 0.31), 0.3))
    b.material("stone_dark", principled((0.10, 0.10, 0.11), 0.12, clearcoat=0.3))
    b.material("laminate", principled((0.86, 0.86, 0.84), 0.35))
    b.material("rubber", principled((0.02, 0.02, 0.022), 0.8))
    b.material("seat", principled((0.05, 0.13, 0.26), 0.45))
    b.material("trim", principled((0.03, 0.03, 0.035), 0.5))
    b.material("chrome", {"type": "roughconductor", "material": "Cr", "alpha": 0.1})
    b.material("leaf", {"type": "twosided", "bsdf": principled((0.05, 0.20, 0.06), 0.45)})
    b.material("leaf2", {"type": "twosided", "bsdf": principled((0.10, 0.26, 0.05), 0.45)})
    b.material("pot", principled((0.45, 0.40, 0.35), 0.5))
    b.material("bark", principled((0.10, 0.07, 0.05), 0.85))
    b.material("tape_red", {"type": "diffuse", "reflectance": rgb(0.55, 0.03, 0.03)})
    b.material("cable", principled((0.08, 0.08, 0.08), 0.5, metallic=0.5))
    b.material("backing", {"type": "diffuse", "reflectance": rgb(0.02, 0.02, 0.02)})
    b.material("frame", principled((0.04, 0.04, 0.045), 0.4, metallic=0.5))
    b.material("escalator", principled(
        tex(png(As, "esc_steps", lambda: np.stack([_esc_steps()] * 3, -1)), scale=(1.0, 2.5)), 0.4, metallic=0.6))
    for i, c in enumerate(PAINTS):
        b.material(f"case_{i}", principled(c, 0.25, specular=0.6, clearcoat=0.6, clearcoat_gloss=0.8))
    for i in range(3):
        tw = png(As, f"tower_{i}", lambda i=i: T.window_grid(40 + i))
        b.material(f"tower_{i}", principled(tex(tw, scale=(1 / 12.0, 1 / 12.0)), 0.35, specular=0.5))


def _esc_steps() -> np.ndarray:
    a = np.full((32, 32), 0.45, np.float32)
    a[:4] = 0.06
    a[4:8] = 0.18
    return a


def surf(aid: str, uv=None, tint=None, **extra) -> dict:
    """W.surface with principled overrides (clearcoat, roughness, ...) applied to the inner BSDF."""
    # W.surface's tint multiplies the 8-bit sRGB albedo; the factors here are linear-albedo factors, so convert.
    s = W.surface(aid, uv_scale=uv, tint=None if tint is None else tuple(round(float(v) ** (1 / 2.2), 3) for v in tint))
    (s["bsdf"] if s.get("type") == "normalmap" else s).update(extra)
    return s


def retexture(b: SceneBuilder) -> None:
    """Scanned CC0 surfaces (Poly Haven, ambientCG) replace the flat procedural materials (substitution plan rows)."""
    # floor: terrazzo_tiles reverted (read as matte brown carpet); the procedural polished stone stays
    b.material("gallery_floor", surf("granite_tile_02", tint=(2.2, 2.4, 2.6), specular=0.5, clearcoat=0.15))
    b.material("plaster", surf("painted_plaster_wall", tint=(1.75, 1.75, 1.7)))
    b.material("vault", surf("painted_plaster_wall", tint=(1.75, 1.75, 1.72)))
    b.material("ceiling", surf("painted_plaster_wall", tint=(1.9, 1.9, 1.88)))
    b.material("steel_white", surf("painted_plaster_wall", uv=(2.0, 12.0), tint=(1.95, 1.95, 1.9), roughness=0.38, specular=0.5))
    b.material("steel_dark", surf("acg:Metal032", tint=(0.22, 0.22, 0.24)))
    b.material("frame", surf("acg:Metal032", tint=(0.15, 0.15, 0.16)))
    b.material("stone_dark", surf("acg:Granite006B", tint=(0.35, 0.45, 0.6), clearcoat=0.3))
    b.material("laminate", surf("acg:Plastic010", tint=(2.1, 2.05, 1.95)))
    b.material("rubber", surf("acg:Rubber004"))
    b.material("escalator", surf("acg:DiamondPlate008C"))
    # seat: Leather011 reverted (quilted purple read worse than the plain blue seat); keeps add_materials' seat
    b.material("trim", surf("acg:Plastic010", uv=2.0, tint=(0.1, 0.1, 0.1)))
    b.material("ground_out", surf("concrete_pavement_03"))
    b.material("road", surf("asphalt_02"))
    for i, c in enumerate(PAINTS):
        k = 1.0 / 0.39
        b.material(f"case_{i}", surf("acg:Plastic010", uv=2.0, tint=tuple(k * v for v in c), clearcoat=0.5, clearcoat_gloss=0.8))
        b.material(f"fab_{i}", surf("acg:Fabric030" if i % 2 else "denim_fabric_04", uv=3.0,
                                     tint=tuple(min(4.0, 0.5 + 6 * v) for v in c) if i % 2 else (1.0, 1.0, 1.0)))
    for k, aid in enumerate(OPAQUE_FACADES):
        b.material(f"fac_{k}", surf(aid))
    for k, aid in enumerate(GLASS_FACADES):
        b.material(f"glass_{k}", surf(aid, uv=1 / 24.0, roughness=0.08, specular=0.5))


OPAQUE_FACADES = ["acg:Facade018A", "acg:Facade018B", "acg:Facade019A", "acg:Facade020A", "acg:Facade020B"]
GLASS_FACADES = ["acg:Facade001", "acg:Facade005"]


def cut_variants(m, A: Assets) -> list[dict]:
    """Split a Poly Haven variant sheet (several objects side by side along x) into one {part: ply} per object."""
    def build():
        meshes = []
        for part in m.parts:
            sh = mi.load_dict({"type": "ply", "filename": part.ply})
            p = np.asarray(sh.vertex_positions_buffer(), float).reshape(-1, 3)
            n = np.asarray(sh.vertex_normals_buffer(), float).reshape(-1, 3) if sh.has_vertex_normals() else np.zeros_like(p)
            uv = np.asarray(sh.vertex_texcoords_buffer(), float).reshape(-1, 2) if sh.has_vertex_texcoords() else np.zeros((len(p), 2))
            f = np.asarray(sh.faces_buffer(), np.int64).reshape(-1, 3)
            meshes.append((part.name, p, n, uv, f))
        xs = np.concatenate([p[f].mean(1)[:, 0] for _, p, _, _, f in meshes])
        lo, hi = xs.min(), xs.max()
        occ = np.zeros(400, bool)
        occ[((xs - lo) / (hi - lo + 1e-9) * 399).astype(int)] = True
        edges, inside = [], False                                  # runs of occupied bins = objects
        for i, o in enumerate(occ):
            if o and not inside:
                start, inside = i, True
            if not o and inside:
                edges.append((start, i))
                inside = False
        if inside:
            edges.append((start, 400))
        out = {}
        for k, (a, bnd) in enumerate(edges):
            x0, x1 = lo + a / 399 * (hi - lo) - 1e-6, lo + bnd / 399 * (hi - lo) + 1e-6
            for name, p, n, uv, f in meshes:
                keep = f[(p[f].mean(1)[:, 0] >= x0) & (p[f].mean(1)[:, 0] <= x1)]
                if len(keep):
                    out[f"v{k}--{name}"] = G.Mesh(p, n, uv, keep)
        return out
    flat = A.meshes(f"web_{m.id}_variants", build)
    variants: dict[int, dict] = {}
    for key, path in flat.items():
        k, name = key.split("--", 1)
        variants.setdefault(int(k[1:]), {})[name] = path
    return [variants[k] for k in sorted(variants)]


def add_variant(b: SceneBuilder, m, parts: dict, tag: str, matrix) -> None:
    """Instance one cut variant (geometry stored once per variant), with the model's own materials."""
    group = f"sg_{tag}"
    if group not in b.d:
        bsdf = {p.name: p.bsdf for p in m.parts}
        sg = {"type": "shapegroup"}
        for i, (name, ply) in enumerate(sorted(parts.items())):
            mid = f"{tag}__{name}"
            b.material(mid, bsdf[name])
            sg[f"part_{i}"] = {"type": "ply", "filename": ply, "bsdf": {"type": "ref", "id": mid}}
        b.d[group] = sg
    b.d[b._key(f"inst_{tag}")] = {"type": "instance", "shapegroup": {"type": "ref", "id": group}, "to_world": xf(matrix)}


def variant_bounds(parts: dict) -> tuple:
    pts = np.concatenate([np.asarray(mi.load_dict({"type": "ply", "filename": f}).vertex_positions_buffer()).reshape(-1, 3)
                          for f in parts.values()])
    return pts.min(0), pts.max(0)


def place_bounds(lo, hi, at, yaw=0.0, height=None, scale=1.0):
    s = height / (hi[1] - lo[1]) if height is not None else scale
    anchor = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    return G.compose(G.translate(at), G.rotate((0, 1, 0), yaw), G.scale(s), G.translate(-anchor))


def light_rect(b: SceneBuilder, centre, size_xz, colour, radiance, facing_down=True) -> None:
    m = G.compose(G.translate(centre), G.rotate((1, 0, 0), 90.0), G.scale((size_xz[0] / 2, size_xz[1] / 2, 1.0)))
    L = tuple(radiance * c for c in colour)
    b.rect(m, "backing", emission=L, prefix="light", power=luminance(L) * size_xz[0] * size_xz[1])


def emissive_quad(b: SceneBuilder, centre, w, h, yaw, em_path, material, level_lum, uv=None, prefix="panel") -> None:
    """A w x h quad facing (sin yaw, 0, cos yaw) with an HDR emission bitmap."""
    m = G.compose(G.translate(centre), G.rotate((0, 1, 0), yaw), G.scale((w / 2, h / 2, 1.0)))
    b.rect(m, material, emission=tex(em_path, raw=True, to_uv=uv), prefix=prefix, power=level_lum * w * h)


def cables(b: SceneBuilder, As: Assets, x: float, z: float, y_bottom: float, y_top: float) -> None:
    """A vertical hanger from y_bottom up to y_top."""
    unit = As.mesh("cable_unit", lambda: P.tube((0, 0, 0), (0, 1, 0), 0.012, 5))
    b.ply(unit, G.compose(G.translate((x, y_bottom, z)), G.scale((1.0, y_top - y_bottom, 1.0))), "cable", prefix="cable")


def vault_y(x: float) -> float:
    """Height of the truss's inner chord above x (the lowest roof point over that x)."""
    return YC + np.sqrt(max(R_IN ** 2 - x ** 2, 0.0))


def build(ctx: BuildContext) -> SceneBundle:
    rng = np.random.default_rng(ctx.seed)
    As, Ad = Assets(ctx.shared_dir, ctx.rebuild), Assets(ctx.assets_dir, ctx.rebuild)
    b = SceneBuilder(As)
    add_materials(b, As, floor_paths(Ad, ctx.seed), ctx.env)
    retexture(b)

    def cube(centre, size, material, prefix="box"):
        """A box with uv in meters (scene_kit's `cube` maps 0..1 per face, which stretches real-scale textures)."""
        key = "x".join(f"{v:.3f}" for v in size)
        b.box_mesh(f"{prefix}_{key}", centre, size, material)
    focus: dict[str, list] = {}

    # ---- floor and ground --------------------------------------------------------------------------------------
    b.box_mesh("floor", (0.0, -0.1, (Z_NEAR + Z_FAR) / 2), (2 * HH, 0.2, Z_NEAR - Z_FAR), "floor")
    cube((0.0, -0.25, -100.0), (1400.0, 0.46, 1400.0), "ground_out", prefix="ground")
    for zc in (-98.0, 26.0):                                                   # streets across both open ends
        cube((0.0, -0.01, zc), (900.0, 0.02, 12.0), "road", prefix="road")
        for dz in (-7.5, 7.5):
            cube((0.0, 0.06, zc + dz), (900.0, 0.16, 3.0), "ground_out", prefix="kerb")

    # ---- vault: ribs, shell with a skylight and clerestory slots, glazing bars -------------------------------------
    rib = As.mesh("rib", lambda: P.rib(R_IN, YC, HALF_DEG, TRUSS_D, 36))
    for z in Z_BAYS:
        b.ply(rib, G.translate((0, 0, z)), "steel_white", prefix="rib")
        cube((0.0, 8.93, z), (2 * NH, 0.06, 0.06), "steel_white", prefix="tie")
    za, zb = Z_NEAR + 0.3, Z_FAR - 0.3
    for sgn in (-1, 1):
        for k, (lo, hi) in enumerate(((SKYLIGHT_DEG, SLOT_DEG[0]), (SLOT_DEG[1], HALF_DEG + 1.5))):
            sh = As.mesh(f"shell_{k}_{int(sgn > 0)}", lambda lo=lo, hi=hi, sgn=sgn: P.vault_shell(
                R_SH, YC, sgn * lo, sgn * hi, za, zb))
            b.ply(sh, np.eye(4), "vault", prefix="shell")
        for z in Z_BAYS:                                       # piers between the clerestory slots, one per rib
            sh = As.mesh(f"pier_{int(sgn > 0)}", lambda sgn=sgn: P.vault_shell(R_SH, YC, sgn * SLOT_DEG[0], sgn * SLOT_DEG[1], -0.6, 0.6))
            b.ply(sh, G.translate((0, 0, z)), "vault", prefix="pier")
        for ang in (SKYLIGHT_DEG, 10.0, 5.0):                  # skylight rim and glazing bars, along the hall
            x, y = R_SH * np.sin(np.radians(sgn * ang)), YC + R_SH * np.cos(np.radians(sgn * ang))
            b.ply(As.mesh(f"bar_{ang}_{int(sgn > 0)}", lambda x=x, y=y: P.tube((x, y, za), (x, y, zb), 0.05 if ang == SKYLIGHT_DEG else 0.03, 8)),
                  np.eye(4), "steel_white", prefix="bar")

    # ---- columns, aisle roof, end closures -------------------------------------------------------------------------
    col = As.mesh("column", lambda: P.column(SOFFIT))
    for z in Z_BAYS:
        for s in (-1, 1):
            b.ply(col, G.translate((s * NH, 0.0, z)), "steel_white", prefix="column")
    for s in (-1, 1):
        cube((s * (NH + HH + 0.4) / 2, SOFFIT + 0.15, (Z_NEAR + Z_FAR) / 2), (HH + 0.4 - NH, 0.3, Z_NEAR - Z_FAR + 0.8),
               "ceiling", prefix="aisleroof")
        for z in (Z_NEAR + 0.2, Z_FAR - 0.2):
            cube((s * (NH + HH) / 2, (SOFFIT + 0.3) / 2, z), (HH - NH + 0.4, SOFFIT + 0.3, 0.4), "plaster", prefix="aisleend")

    # ---- galleries, walls, windows ----------------------------------------------------------------------------------
    def slab_segments(side: int):
        if side < 0:
            return [(Z_NEAR + 0.2, Z_FAR - 0.2)]
        return [(Z_NEAR + 0.2, ESC_GAP[1]), (ESC_GAP[0], Z_FAR - 0.2)]

    for s in (-1, 1):
        for (z0, z1) in slab_segments(s):
            cube((s * (NH + HH) / 2, GAL_Y - GAL_T / 2, (z0 + z1) / 2), (HH - NH, GAL_T, z0 - z1), "gallery_floor", prefix="gallery")
        # balustrade bars along z at the nave edge of the gallery
        for (z0, z1) in slab_segments(s):
            length = z0 - z1 - 0.3
            bars = As.mesh(f"balu_{int(length * 10)}", lambda length=length: P.balustrade_bars(length))
            b.ply(bars, G.compose(G.translate((s * (NH + 0.02), GAL_Y, z0 - 0.15)), G.rotate((0, 1, 0), 90.0)), "steel_dark", prefix="balu")
            cube((s * (NH + 0.02), GAL_Y + 1.03, (z0 + z1) / 2), (0.07, 0.05, z0 - z1), "steel_dark", prefix="handrail")
            cube((s * (NH + 0.02), GAL_Y + 0.07, (z0 + z1) / 2), (0.07, 0.07, z0 - z1), "steel_dark", prefix="kick")
        # gallery wall per bay: solid below the clerestory, piers, a lintel, mullioned open windows
        for zc in BAY_C:
            xw = s * (HH - 0.2)
            cube((xw, (GAL_Y + 7.4) / 2, zc), (0.4, 7.4 - GAL_Y, 6.0), "plaster", prefix="walllow")
            cube((xw, 8.9, zc), (0.4, 0.2, 6.0), "plaster", prefix="lintel")
            for dz in (-2.4, 2.4):
                cube((xw, 8.1, zc + dz), (0.4, 1.4, 1.2), "plaster", prefix="pier")
            cube((xw - s * 0.05, 8.1, zc), (0.1, 1.5, 0.06), "frame", prefix="mullion")
            cube((xw - s * 0.05, 8.1, zc - 0.9), (0.1, 1.5, 0.06), "frame", prefix="mullion")
            cube((xw - s * 0.05, 8.1, zc + 0.9), (0.1, 1.5, 0.06), "frame", prefix="mullion")
        cube((s * (HH + 0.2), 2.2, (Z_NEAR + Z_FAR) / 2), (0.4, 4.4, Z_NEAR - Z_FAR + 0.8), "plaster", prefix="shopback")

    # ---- shopfronts (ground aisle) and posters (gallery) ----------------------------------------------------------------
    count = 6
    shops_total = 0
    for s in (-1, 1):
        atlas_seed = ctx.seed * 17 + (1 if s > 0 else 2)
        alb_f, emi_f = Ad.root / "textures" / f"shop_alb_{s}.png", Ad.root / "textures" / f"shop_emi_{s}.exr"
        if Ad.rebuild or not (alb_f.exists() and emi_f.exists()):
            alb_a, emi_a, _ = T.shop_atlas(atlas_seed, count)
            T.save_png8(alb_f, flip(alb_a))
            P.save_exr(emi_f, flip(emi_a * SHOP_GAIN))
        alb_p, emi_p = str(alb_f), str(emi_f)
        em_img = np.asarray(mi.Bitmap(emi_p))[..., :3]
        order = rng.permutation(count)
        for k, zc in enumerate(BAY_C):
            if s > 0 and k in ESC_BAYS:
                continue
            idx = int(order[k % count])
            uv = mi.ScalarTransform4f().translate([idx / count, 0.0, 0.0]).scale([1.0 / count, 1.0, 1.0])
            cell = em_img[:, int(idx * em_img.shape[1] / count): int((idx + 1) * em_img.shape[1] / count)]
            power = float(luminance(cell.reshape(-1, 3).mean(0))) * 6.0 * 4.4
            b.material(f"shop_{s}_{k}", principled(tex(alb_p, to_uv=uv), 0.45, specular=0.5))
            m = G.compose(G.translate((s * (HH - 0.02), 2.2, zc)), G.rotate((0, 1, 0), -90.0 * s), G.scale((3.0, 2.2, 1.0)))
            b.rect(m, f"shop_{s}_{k}", emission=tex(emi_p, raw=True, to_uv=uv), prefix="shop", power=power)
            shops_total += 1
            if k == 6 and s > 0:
                focus["shop_sign_r"] = [HH - 0.03, 3.75, zc]
            if k == 10 and s < 0:
                focus["shop_sign_l"] = [-HH + 0.03, 3.75, zc]
        # posters on the gallery wall
        pal_seed = ctx.seed * 29 + (3 if s > 0 else 4)
        slogans = ["VISIT|KYOTO", "FLY|HIGHER", "TASTE|ITALY", "NEW|YORK", "SEE|PARIS", "DISCOVER|OSLO"]

        def poster_atlas(sd=pal_seed):
            return POSTER_GAIN * np.concatenate([T.poster(sd + i, slogans[(i + sd) % len(slogans)])[1] for i in range(4)], 1)
        post_p = exr(Ad, f"poster_{s}", lambda: flip(poster_atlas()))
        pa_p = png(Ad, f"poster_alb_{s}", lambda sd=pal_seed: flip(np.concatenate([T.poster(sd + i, slogans[(i + sd) % len(slogans)])[0] for i in range(4)], 1)))
        for k, zc in enumerate(BAY_C):
            idx = int(rng.integers(4))
            uv = mi.ScalarTransform4f().translate([idx / 4, 0.0, 0.0]).scale([0.25, 1.0, 1.0])
            b.material(f"poster_{s}_{k}", principled(tex(pa_p, to_uv=uv), 0.4))
            m = G.compose(G.translate((s * (HH - 0.42), 6.1, zc)), G.rotate((0, 1, 0), -90.0 * s), G.scale((0.75, 1.1, 1.0)))
            b.rect(m, f"poster_{s}_{k}", emission=tex(post_p, raw=True, to_uv=uv), prefix="poster", power=3.0 * 0.4 * 1.5 * 2.2)

    # ---- aisle and gallery lighting -------------------------------------------------------------------------------
    for s in (-1, 1):
        for z in np.arange(Z_NEAR - 1.5, Z_FAR, -3.0):
            light_rect(b, (s * 10.2, GAL_Y - GAL_T - 0.01, z), (0.7, 0.7), (1.0, 0.82, 0.6), 7.0)
            light_rect(b, (s * 10.2, SOFFIT - 0.01, z), (0.6, 0.6), (1.0, 0.9, 0.78), 6.0)

    # ---- linear fixtures hung in the nave (alternating warm and cool) ---------------------------------------------------
    for i, zc in enumerate(BAY_C):
        for j, x in enumerate((-3.6, 3.6)):
            col_name = "warm" if (i + j) % 2 == 0 else "cool"
            y = 10.4
            light_rect(b, (x, y, zc), (0.18, 4.4), FIXTURE_L[col_name], FIXTURE_RAD)
            cube((x, y + 0.07, zc), (0.26, 0.10, 4.5), "steel_dark", prefix="fixture")
            for dz in (-1.8, 1.8):
                cables(b, As, x, zc + dz, y + 0.12, vault_y(x))

    # ---- escalators (right aisle) -----------------------------------------------------------------------------------
    esc = As.meshes("escalator", lambda: P.escalator(GAL_Y, 30.0, 1.1))
    for x in (9.35, 11.1):
        M = G.compose(G.translate((x, 0.0, ESC_GAP[1] - 0.45)), G.rotate((0, 1, 0), 180.0))
        b.part_set(esc, M, {"steps": "escalator", "metal": "steel_dark", "rubber": "rubber"})
    focus["escalator"] = [9.35, 2.6, ESC_GAP[1] - 0.45 - 4.2]

    # ---- bridge across the nave ----------------------------------------------------------------------------------------
    cube((0.0, GAL_Y - 0.5, BRIDGE_Z), (2 * NH, 1.0, 3.0), "frame", prefix="girder")
    cube((0.0, GAL_Y - 0.03, BRIDGE_Z), (2 * NH, 0.06, 3.0), "gallery_floor", prefix="bridgedeck")
    for dz in (-1.45, 1.45):
        bars = As.mesh("balu_bridge", lambda: P.balustrade_bars(2 * NH - 0.2))
        b.ply(bars, G.translate((-NH + 0.1, GAL_Y, BRIDGE_Z + dz)), "steel_dark", prefix="bbalu")
        cube((0.0, GAL_Y + 1.03, BRIDGE_Z + dz), (2 * NH, 0.05, 0.07), "steel_dark", prefix="brail")
    p = exr(As, "sign_bridge", lambda: flip(T.sign("GATES 1-30", (0.05, 0.28, 0.65), 1500, 270, arrow=">")[1]))
    emissive_quad(b, (0.0, GAL_Y - 0.5, BRIDGE_Z + 1.52), 5.0, 0.9, 0.0, p, "backing", 5.0 * 0.5, prefix="bridgesign")
    focus["bridge_sign"] = [0.0, GAL_Y - 0.5, BRIDGE_Z + 1.52]

    # ---- departures boards, clocks, hanging signs ---------------------------------------------------------------------------
    bz = -26.9
    for k, (w, h, y, seed_off, rows, name) in enumerate(((8.0, 3.4, 9.9, 0, 12, "board_main"), (5.5, 2.4, 9.4, 7, 9, "board_far"))):
        z = bz if k == 0 else -61.9
        e1 = exr(Ad, f"board_{k}", lambda: flip(T.board(ctx.seed * 5 + seed_off, int(w * 300), int(h * 300), rows)[1]))
        e2 = exr(Ad, f"board_{k}_back", lambda: flip(T.board(ctx.seed * 5 + seed_off + 100, int(w * 300), int(h * 300), rows)[1]))
        emissive_quad(b, (0.0, y, z + 0.01), w, h, 0.0, e1, "backing", 0.5 * 3.2, prefix="board")
        emissive_quad(b, (0.0, y, z - 0.26), w, h, 180.0, e2, "backing", 0.5 * 3.2, prefix="boardback")
        cube((0.0, y, z - 0.125), (w + 0.2, h + 0.2, 0.25), "frame", prefix="boardbox")
        for sx in (-w / 2 + 0.3, w / 2 - 0.3):
            cables(b, As, sx, z - 0.12, y + h / 2 + 0.1, vault_y(sx))
        focus[name] = [0.0, y, z + 0.01]
    # clocks
    clk_p = exr(As, "clock", lambda: flip(T.clock_face()[1]))
    ring = As.mesh("bezel", lambda: G.lathe([(0.7, 0.0), (1.08, 0.0), (1.08, 0.18), (0.74, 0.18), (0.70, 0.12)], 40))
    for z in (-8.0, -64.0):
        y = 11.6
        emissive_quad(b, (0.0, y, z), 1.4, 1.4, 0.0, clk_p, "backing", 2.6 * 0.4, prefix="clock")
        b.ply(ring, G.compose(G.translate((0.0, y, z + 0.0)), G.rotate((1, 0, 0), 90.0)), "steel_dark", prefix="bezel")
        cube((0.0, y, z - 0.12), (2.2, 2.2, 0.2), "frame", prefix="clockbox")
        cables(b, As, 0.0, z - 0.1, y + 1.1, vault_y(0.0))
    focus["clock"] = [0.0, 11.6, -8.0 + 0.01]
    # hanging directional signs
    kinds = [("GATES 1-30", (0.05, 0.28, 0.65), ">"), ("< TRAINS", (0.05, 0.45, 0.2), ""), ("TAXI", (0.75, 0.6, 0.05), "^"),
             ("EXIT", (0.05, 0.45, 0.2), ">"), ("CHECK-IN A-F", (0.05, 0.28, 0.65), "<"), ("INFORMATION", (0.05, 0.28, 0.65), ""),
             ("LOUNGE", (0.3, 0.1, 0.45), ">"), ("BAGGAGE", (0.05, 0.28, 0.65), "^")]
    slots = [(-12.0, -2.0), (-18.0, 3.0), (-38.0, 0.0), (-46.0, -3.5), (-52.0, 3.5), (-68.0, 0.0), (-74.0, -3.0), (-78.0, 3.0)]
    perm = rng.permutation(len(kinds))
    for (z, x), ki in zip(slots, perm):
        text, bg, arrow = kinds[int(ki)]
        sp = exr(As, f"sign_{int(ki)}", lambda text=text, bg=bg, arrow=arrow: flip(T.sign(text, bg, 900, 270, arrow=arrow)[1]))
        y = 8.3
        emissive_quad(b, (x, y, z + 0.01), 3.2, 0.96, 0.0, sp, "backing", 3.0, prefix="hsign")
        emissive_quad(b, (x, y, z - 0.13), 3.2, 0.96, 180.0, sp, "backing", 3.0, prefix="hsign")
        cube((x, y, z - 0.06), (3.3, 1.06, 0.12), "frame", prefix="hsignbox")
        for sx in (-1.4, 1.4):
            cables(b, As, x + sx, z - 0.06, y + 0.53, vault_y(x + sx))
        if ki == 0:
            focus["sign_gates"] = [x, y, z + 0.01]
    focus.setdefault("sign_gates", [slots[0][1], 8.3, slots[0][0] + 0.01])

    # ---- check-in counters, belts, queue posts, gantry signs (left of the nave) ------------------------------------------------
    for i, z0 in enumerate(COUNTER_Z):
        zc = z0 - 1.2
        cube((-6.6, 0.5, zc), (0.9, 1.0, 2.4), "laminate", prefix="counter")
        cube((-6.6, 1.03, zc), (1.0, 0.06, 2.5), "stone_dark", prefix="countertop")
        cube((-7.65, 0.22, zc), (1.0, 0.44, 2.4), "frame", prefix="belt")
        cube((-7.65, 0.46, zc), (0.9, 0.04, 2.3), "rubber", prefix="beltrubber")
        cube((-6.75, 1.25, zc - 0.4), (0.04, 0.38, 0.5), "frame", prefix="monitor")
        b.rect(G.compose(G.translate((-6.72, 1.27, zc - 0.4)), G.rotate((0, 1, 0), 90.0), G.scale((0.23, 0.15, 1.0))),
               "backing", emission=(0.25, 0.55, 1.0 if i % 2 else 0.7), prefix="screen", power=0.5 * 0.1)
        letter = "ABCDEFG"[i]
        sp = exr(As, f"sign_ci_{letter}", lambda letter=letter: flip(T.sign(f"CHECK-IN {letter}", (0.05, 0.28, 0.65), 900, 270)[1]))
        y = 5.8
        emissive_quad(b, (-6.6 + 0.006, y, zc), 2.2, 0.66, 90.0, sp, "backing", 3.0, prefix="cisign")
        emissive_quad(b, (-6.6 - 0.006, y, zc), 2.2, 0.66, -90.0, sp, "backing", 3.0, prefix="cisign2")
        for dz in (-0.9, 0.9):
            cables(b, As, -6.6, zc + dz, y + 0.33, vault_y(-6.6))
        if i == 3:
            focus["counter_sign"] = [-6.58, y, zc]
    post = As.mesh("stanchion", lambda: G.lathe([(0.0, 0.0), (0.18, 0.0), (0.18, 0.03), (0.04, 0.05), (0.03, 0.95), (0.05, 0.98),
                                                  (0.05, 1.02), (0.0, 1.02)], 14))
    pz = np.arange(-15.0, -35.0, -1.8)
    for z in pz:
        b.ply(post, G.translate((-4.7, 0.0, z)), "chrome", prefix="stanchion")
    for z0, z1 in zip(pz[:-1], pz[1:]):
        b.ply(As.mesh("tape", lambda: P.tube((-4.7, 0.88, 0.0), (-4.7, 0.88, -1.8), 0.013, 6)),
              G.translate((0.0, 0.0, z0)), "tape_red", prefix="tape")

    # ---- seating (right of the nave) ----------------------------------------------------------------------------------------
    bench_m = As.meshes("bench4", lambda: P.bench(4))
    seat_world = []
    for z in SEAT_PAIRS:
        for face, x in ((1, SEAT_X + 0.25), (-1, SEAT_X - 0.25)):
            M = G.compose(G.translate((x, 0.0, z)), G.rotate((0, 1, 0), 90.0 * face))
            b.part_set(bench_m, M, {"seat": "seat", "frame": "steel_dark"})
            for i in range(4):
                lx = (i + 0.5) * 0.55 - 1.1
                seat_world.append((x, 0.49, z - face * lx * 1.0 if face > 0 else z + lx))
    focus["bench_near"] = [SEAT_X + 0.45, 0.5, SEAT_PAIRS[1]]

    # ---- coffee kiosk and information desk ---------------------------------------------------------------------------------------
    kx, kz = KIOSK
    cube((kx, 0.55, kz), (3.0, 1.1, 2.0), "laminate", prefix="kiosk")
    cube((kx, 1.13, kz), (3.1, 0.06, 2.1), "stone_dark", prefix="kiosktop")
    cube((kx, 2.7, kz - 0.95), (3.1, 3.2, 0.1), "plaster", prefix="kioskback")
    cube((kx, 3.6, kz), (3.4, 0.14, 2.4), "steel_dark", prefix="kioskroof")
    sp = exr(As, "sign_coffee", lambda: flip(T.sign("COFFEE  TEA", (0.35, 0.18, 0.08), 900, 270)[1]))
    cube((kx, 3.1, kz + 1.12), (3.1, 1.0, 0.1), "frame", prefix="kiosksignbox")
    emissive_quad(b, (kx, 3.1, kz + 1.18), 3.0, 0.9, 0.0, sp, "backing", 3.0, prefix="kiosksign")
    cube((kx, 2.0, kz + 0.2), (2.6, 0.04, 0.5), "frame", prefix="shelf")
    bottles = W.model("wine_bottles_01")
    for dx in (-0.75, 0.75):
        W.add(b, bottles, W.place(bottles, at=(kx + dx, 2.02, kz + 0.2), yaw=0.0))
    focus["kiosk_sign"] = [kx, 3.1, kz + 1.19]
    ix, iz = INFO_DESK
    b.ply(As.mesh("info_desk", lambda: G.lathe([(0.0, 0.0), (1.25, 0.0), (1.25, 1.0), (1.35, 1.02), (1.35, 1.08), (0.0, 1.08)], 36)),
          G.translate((ix, 0.0, iz)), "laminate", prefix="infodesk")
    sp = exr(As, "sign_info", lambda: flip(T.sign("INFORMATION", (0.05, 0.28, 0.65), 900, 270)[1]))
    b.ply(As.mesh("info_pole", lambda: P.tube((0, 1.1, 0), (0, 3.4, 0), 0.05, 8)), G.translate((ix, 0.0, iz)), "steel_dark", prefix="infopole")
    for yaw in (0.0, 180.0):
        emissive_quad(b, (ix, 3.6, iz + (0.06 if yaw == 0 else -0.06)), 1.8, 0.54, yaw, sp, "backing", 3.0, prefix="infosign")
    cube((ix, 3.6, iz), (1.9, 0.64, 0.1), "frame", prefix="infosignbox")

    # ---- trees: two scanned trees in planters, scanned potted plants elsewhere ----------------------------------------
    tree = W.model("tree_small_02")
    planter = W.model("planter_box_03")
    pach = W.model("pachira_aquatica_01")
    pach_v = cut_variants(pach, As)
    pplant = W.model("potted_plant_01")
    for i, (x, z) in enumerate(TREES):
        yaw = 40.0 * i
        if i in (0, 3):
            W.add(b, planter, W.place(planter, at=(x, 0.0, z), yaw=yaw, scale=1.25))
            W.add(b, tree, W.place(tree, at=(x, 1.0, z), yaw=yaw + 70, height=3.3))
        elif i % 2 == 0 and pach_v:
            v = int(i // 2) % len(pach_v)
            lo, hi = variant_bounds(pach_v[v])
            add_variant(b, pach, pach_v[v], f"pachira_v{v}", place_bounds(lo, hi, (x, 0.0, z), yaw, height=2.1))
        else:
            W.add(b, pplant, W.place(pplant, at=(x, 0.0, z), yaw=yaw, height=1.6))
    focus["tree_near"] = [TREES[0][0], 2.4, TREES[0][1]]

    # ---- additions: bins, wet-floor signs, extinguishers, cameras, coffee cart, chalkboard, bar stools, planters,
    #      lounge chairs, a service corner ----------------------------------------------------------------------------
    can = W.model("metal_trash_can")
    can_v = cut_variants(can, As)
    lo_c, hi_c = variant_bounds(can_v[0])
    for k, z in enumerate(Z_BAYS[1:-1:2]):
        for s in (-1, 1):
            add_variant(b, can, can_v[0], "bin_v0", place_bounds(lo_c, hi_c, (s * 7.95, 0.0, z + 0.9), 90.0 * s, height=0.9))
    for z in (SEAT_PAIRS[0] + 1.6, SEAT_PAIRS[11] - 1.6):
        add_variant(b, can, can_v[0], "bin_v0", place_bounds(lo_c, hi_c, (SEAT_X, 0.0, z), 0.0, height=0.9))
    wet = W.model("WetFloorSign_01")
    for (x, z, yaw) in WET_SIGNS:
        W.add(b, wet, W.place(wet, at=(x, 0.0, z), yaw=yaw))
    ext = W.model("korean_fire_extinguisher_01")
    cam = W.model("security_camera_01")
    for k, z in enumerate(Z_BAYS[2:-1:3]):
        for s in (-1, 1):
            W.add(b, ext, W.place(ext, at=(s * (NH - 0.45), 0.0, z + 0.3), yaw=-90.0 * s))
            W.add(b, cam, W.place(cam, at=(s * (NH - 0.3), 4.0, z), yaw=-90.0 * s))
            W.add(b, cam, W.place(cam, at=(s * 10.2, GAL_Y - GAL_T - 0.32, z - 3.0), yaw=-90.0 * s))
    cart = W.model("CoffeeCart_01")
    W.add(b, cart, W.place(cart, at=(KIOSK[0] - 0.4, 0.0, KIOSK[1] + 3.2), yaw=180.0))
    chalk = W.model("standing_chalkboard_01")
    W.add(b, chalk, W.place(chalk, at=(KIOSK[0] - 2.0, 0.0, KIOSK[1] + 1.7), yaw=20.0))
    stool = W.model("bar_chair_round_01")
    for k in range(3):
        W.add(b, stool, W.place(stool, at=(KIOSK[0] - 0.9 + 0.9 * k, 0.0, KIOSK[1] + 1.45), yaw=30.0 * k))
    pbox, pp2 = W.model("planter_box_01"), W.model("potted_plant_02")
    for z in (-13.9, -34.5):
        W.add(b, pbox, W.place(pbox, at=(-4.7, 0.0, z), yaw=90.0))
        W.add(b, pp2, W.place(pp2, at=(-4.7, 0.42, z), yaw=15.0, height=0.7))
    W.add(b, pp2, W.place(pp2, at=(INFO_DESK[0] + 0.6, 1.08, INFO_DESK[1]), yaw=0.0, height=0.5))
    W.add(b, pp2, W.place(pp2, at=(-6.6, 1.06, COUNTER_Z[1] - 2.55), yaw=0.0, height=0.45))
    chair = W.model("modern_arm_chair_01")
    for (x, z, yaw) in LOUNGE:
        W.add(b, chair, W.place(chair, at=(x, 0.0, z), yaw=yaw))
    box_m, truck = W.model("cardboard_box_01"), W.model("hand_truck")
    for k, (dx, dz, yy) in enumerate(((0.0, 0.0, 0.0), (0.45, 0.1, 0.0), (0.2, 0.05, 0.34))):
        W.add(b, box_m, W.place(box_m, at=(KIOSK[0] + 0.8 + dx, yy, KIOSK[1] - 1.45 + dz), yaw=10.0 * k))
    W.add(b, truck, W.place(truck, at=(KIOSK[0] - 0.9, 0.0, KIOSK[1] - 1.45), yaw=200.0))
    broom = W.model("wooden_broom")
    W.add(b, broom, W.place(broom, at=(KIOSK[0] - 1.45, 0.0, KIOSK[1] - 1.1), yaw=0.0))

    # ---- trolleys and luggage ------------------------------------------------------------------------------------------------
    trolley = As.meshes("trolley", P.trolley)
    (tx0, tx1), (tz0, tz1) = TROLLEY_ZONE
    nt = 8
    for i in range(nt):
        x = tx0 + (i % 2) * 1.0 + rng.uniform(-0.05, 0.05)
        z = tz1 - (i // 2) * 1.55 + rng.uniform(-0.05, 0.05) - 0.5
        yaw = 90.0 + rng.normal(0, 2.0)
        b.part_set(trolley, G.compose(G.translate((x, 0.0, z)), G.rotate((0, 1, 0), yaw)), {"frame": "chrome", "wheel": "rubber"})
        if i == 1:
            focus["trolley"] = [x, 0.9, z]
        if rng.random() < 0.5:
            case = As.meshes("case_a", lambda: P.suitcase(0.48, 0.72, 0.28))
            b.part_set(case, G.compose(G.translate((x, 0.24, z - 0.1)), G.rotate((0, 1, 0), yaw)),
                       {"shell": f"case_{int(rng.integers(len(PAINTS)))}", "trim": "trim"})
    sizes = [("case_a", (0.48, 0.72, 0.28)), ("case_b", (0.40, 0.60, 0.24)), ("case_c", (0.55, 0.78, 0.32))]
    n_case = 0
    for (x0, x1), (z0, z1) in LUGGAGE_ZONES:
        for _ in range(int(rng.integers(6, 10))):
            x, z = rng.uniform(x0, x1), rng.uniform(z0, z1)
            yaw = rng.uniform(0, 360)
            r = rng.random()
            mat = f"case_{int(rng.integers(len(PAINTS)))}"
            if r < 0.55:
                nm, dims = sizes[int(rng.integers(len(sizes)))]
                parts = As.meshes(nm, lambda d=dims: P.suitcase(*d))
                b.part_set(parts, G.compose(G.translate((x, 0.0, z)), G.rotate((0, 1, 0), yaw)), {"shell": mat, "trim": "trim"})
            elif r < 0.8:
                parts = As.meshes("duffel", P.duffel)
                b.part_set(parts, G.compose(G.translate((x, 0.0, z)), G.rotate((0, 1, 0), yaw)), {"shell": mat.replace("case", "fab"), "trim": "trim"})
            else:
                parts = As.meshes("backpack", P.backpack)
                b.part_set(parts, G.compose(G.translate((x, 0.0, z)), G.rotate((0, 1, 0), yaw)), {"shell": mat.replace("case", "fab"), "trim": "trim"})
            n_case += 1
            if n_case == 3:
                focus["suitcase"] = [x, 0.4, z]
    for (sx, sy, sz) in rng.permutation(np.array(seat_world))[:10]:                 # bags left on seats
        parts = As.meshes("backpack", P.backpack)
        b.part_set(parts, G.compose(G.translate((sx, sy, sz)), G.rotate((0, 1, 0), rng.uniform(0, 360))),
                   {"shell": f"fab_{int(rng.integers(len(PAINTS)))}", "trim": "trim"})

    # ---- outside: towers with podiums, setbacks, crowns and (near ones) real fins; streets across both open ends ----
    trng = np.random.default_rng(99)
    towers = []
    for z in np.arange(-420.0, 160.0, 34.0):
        for side in (-1, 1):
            towers.append((side * trng.uniform(42.0, 60.0), z + trng.uniform(-4, 4), trng.uniform(22, 30), trng.uniform(20, 28), trng.uniform(30, 95)))
            towers.append((side * trng.uniform(85.0, 120.0), z + trng.uniform(-8, 8), trng.uniform(22, 36), trng.uniform(20, 32), trng.uniform(40, 130)))
    for x in np.arange(-160.0, 170.0, 32.0):
        for z in (-138.0, -200.0, -270.0, 70.0, 140.0):
            towers.append((x + trng.uniform(-3, 3), z + trng.uniform(-6, 6), trng.uniform(24, 30), trng.uniform(20, 30), trng.uniform(35, 120)))
    towers.append((0.0, -260.0, 40.0, 30.0, 70.0))
    aircon = W.model("exterior_aircon_unit")
    for i, (x, z, w, d, h) in enumerate(towers):
        near = abs(z - Z_FAR) < 150.0 or abs(z - Z_NEAR) < 150.0 or abs(x) < 70.0
        parts = As.meshes(f"tower_{i}", lambda i=i, w=w, d=d, h=h, near=near: P.tower(500 + i, w, d, h, near))
        glass = i % 5 == 1                      # dark mirror glass read black when it filled the frame: keep it a minority
        mats = {"podium": f"glass_{i % 2}" if glass else f"fac_{(i + 2) % len(OPAQUE_FACADES)}",
                "body": f"glass_{i % 2}" if glass else f"fac_{i % len(OPAQUE_FACADES)}",
                "crown": "plaster", "fins": "frame"}
        b.part_set(parts, G.translate((x, -0.02, z)), mats)
        if near and i % 3 == 0:
            W.add(b, aircon, W.place(aircon, at=(x + 0.18 * w, h + 0.0, z), yaw=90.0 * (i % 4)))
    focus["far_tower"] = [0.0, 20.0, -260.0 + 15.2]
    lamp, stree, hyd, seat_s = W.model("street_lamp_01"), W.model("searsia_lucida"), W.model("fire_hydrant"), W.model("modular_street_seating")
    for zc, yaw in ((-98.0, 0.0), (26.0, 180.0)):
        for x in np.arange(-100.0, 101.0, 25.0):
            W.add(b, lamp, W.place(lamp, at=(x, 0.14, zc - 7.5 * (1 if yaw == 0 else -1)), yaw=yaw))
            W.add(b, stree, W.place(stree, at=(x + 12.5, 0.14, zc + 7.5 * (1 if yaw == 0 else -1)), yaw=37.0 * x, height=4.5))
        W.add(b, hyd, W.place(hyd, at=(-16.0, 0.14, zc - 8.2 * (1 if yaw == 0 else -1)), yaw=yaw, height=0.8))
        for x in (-30.0, 30.0):
            W.add(b, seat_s, W.place(seat_s, at=(x, 0.14, zc + 8.0 * (1 if yaw == 0 else -1)), yaw=yaw))

    # ---- sky ----------------------------------------------------------------------------------------------------------------------------------
    b.d["sky"] = env_kit.sky(ctx.env, sun_direction=SUN_DIR, scale=SKY_SCALE[ctx.env],
                             sampling_weight=max(SKY_SHARE * b.lamp_weight, 1e-3))
    focus["column_near"] = [NH - 0.3, 1.5, Z_BAYS[2]]
    focus["floor_stripe"] = [0.0, 0.0, -9.0]
    return SceneBundle(b.d, focus, {"sun_direction": list(SUN_DIR), "sky_scale": SKY_SCALE[ctx.env], "towers": len(towers),
                                    "shops": shops_total})


def exclude_boxes() -> tuple:
    boxes = []
    boxes.append(((-8.4, 0.0, COUNTER_Z[-1] - 2.6), (-5.5, 3.2, COUNTER_Z[0] + 0.2)))      # counters, belts
    boxes.append(((-5.3, 0.0, -35.2), (-4.1, 3.2, -13.2)))                                    # queue posts and tape
    for lo, hi in ((SEAT_PAIRS[0], SEAT_PAIRS[5]), (SEAT_PAIRS[6], SEAT_PAIRS[11]), (SEAT_PAIRS[12], SEAT_PAIRS[15])):
        boxes.append(((SEAT_X - 0.9, 0.0, hi - 1.3), (SEAT_X + 1.0, 3.2, lo + 1.3)))
    for (x0, x1), (z0, z1) in LUGGAGE_ZONES + [TROLLEY_ZONE]:
        boxes.append(((x0 - 0.5, 0.0, z0 - 0.5), (x1 + 0.5, 3.2, z1 + 0.5)))
    for x, z in TREES:
        boxes.append(((x - 0.9, 0.0, z - 0.9), (x + 0.9, 3.2, z + 0.9)))
    boxes.append(((KIOSK[0] - 2.6, 0.0, KIOSK[1] - 2.2), (KIOSK[0] + 1.9, 3.2, KIOSK[1] + 4.4)))      # kiosk, cart, stools, chalkboard
    for x, z, _ in WET_SIGNS:
        boxes.append(((x - 0.6, 0.0, z - 0.6), (x + 0.6, 1.5, z + 0.6)))
    boxes.append(((-7.3, 0.0, -60.3), (-5.4, 1.6, -56.6)))                                              # lounge chairs
    for z in (SEAT_PAIRS[0] + 1.6, SEAT_PAIRS[11] - 1.6):
        boxes.append(((SEAT_X - 0.6, 0.0, z - 0.6), (SEAT_X + 0.6, 1.5, z + 0.6)))
    for z in Z_BAYS:
        for s in (-1, 1):
            boxes.append(((s * 7.6 - 0.5, 0.0, z - 0.2), (s * 7.6 + 0.5, 1.5, z + 1.5)))                   # bins, extinguishers
    boxes.append(((INFO_DESK[0] - 1.9, 0.0, INFO_DESK[1] - 1.9), (INFO_DESK[0] + 1.9, 3.2, INFO_DESK[1] + 1.9)))
    return tuple(boxes)


SCENE = SceneDef(
    id="terminal", group="artificial", owner="rui",
    description="Train station / airport concourse: 90 m barrel-vault hall with a skylight, polished reflective floor, trusses, "
                "departure boards, signs, shopfronts, seating and luggage, with a skyline beyond the open far end.",
    build=build,
    views={
        # Down the centre of the hall: the departures board, hanging signs and the far end.
        "hall": View((1.0, 1.65, 6.0), (-0.5, 6.9, -60.0), focus="floor_stripe"),
        # Close on a seating block, the hall receding behind it.
        "seating": View((3.4, 1.15, -7.0), (6.2, 1.0, -26.0), focus="bench_near"),
        # Oblique across the floor toward the right-hand shopfronts and the escalators.
        "shops": View((-2.0, 1.6, -6.0), (11.0, 2.8, -30.0), focus="shop_sign_r"),
        # Along the check-in desks.
        "counters": View((-3.4, 1.55, -10.0), (-7.2, 2.2, -36.0), focus="counter_sign"),
    },
    default_view="hall",
    camera_box=((-7.5, 0.5, -76.0), (7.5, 3.0, 7.0)),
    target_box=((-12.0, 0.2, -82.0), (12.0, 7.0, 8.0)),
    exclude_boxes=exclude_boxes(),
    envs=("clear", "overcast"),
    tags=("indoor", "day", "artificial_light", "corridor", "glass", "architecture", "specular"),
    default_seed=5, asset_version=ASSET_VERSION, max_depth=8, rr_depth=5, spp_hint=2048,
)
