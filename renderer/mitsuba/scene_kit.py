"""Building blocks shared by every scene: asset cache, scene-dict builder, material helpers.

Scene authors import from here instead of copying it. Changing this file
changes every scene, so treat it like `scene_api.py`: both owners agree first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import mitsuba as mi
import procedural as G


class Assets:
    """Generates each mesh/texture once and hands back its cached path.

    `root` must be a directory whose contents depend on nothing but the recipe
    (the scene's shared cache, see `scene_api.BuildContext`). Single files are
    written atomically by `procedural`; multi-part meshes get a `.ok` marker so
    a crash halfway never leaves a cache that looks complete.
    """

    def __init__(self, root: Path, rebuild: bool = False):
        self.root = Path(root)
        self.rebuild = rebuild

    def _path(self, sub: str, name: str, ext: str) -> Path:
        return self.root / sub / f"{name}.{ext}"

    def mesh(self, name: str, build) -> str:
        path = self._path("meshes", name, "ply")
        if self.rebuild or not path.exists():
            build().write_ply(path)
        return str(path)

    def meshes(self, name: str, build) -> dict[str, str]:
        """For recipes returning {part: Mesh}; builds all parts together."""
        marker = self._path("meshes", name, "ok")
        if not self.rebuild and marker.exists():
            return {p.stem.split(".", 1)[1]: str(p) for p in sorted((self.root / "meshes").glob(f"{name}.*.ply"))}
        out = {}
        for part, mesh in build().items():
            path = self._path("meshes", f"{name}.{part}", "ply")
            mesh.write_ply(path)
            out[part] = str(path)
        marker.write_text("\n".join(sorted(out)))
        return out

    def texture(self, name: str, build, gray: bool = False) -> str:
        path = self._path("textures", name, "png")
        if self.rebuild or not path.exists():
            (G.save_gray if gray else G.save_rgb)(path, build())
        return str(path)


def xf(matrix) -> mi.ScalarTransform4f:
    return mi.ScalarTransform4f(np.asarray(matrix, dtype=np.float32))


def rgb(*v):
    return {"type": "rgb", "value": list(v)}


def luminance(c) -> float:
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


class SceneBuilder:
    def __init__(self, assets: Assets):
        self.assets = assets
        self.d: dict = {"type": "scene"}
        self.counts: dict[str, int] = {}
        self.focus_points: dict[str, list[float]] = {}
        self.lamp_weight = 0.0

    # --- registration ---------------------------------------------------
    def material(self, name: str, spec: dict) -> None:
        self.d[name] = spec

    def _key(self, prefix: str) -> str:
        n = self.counts.get(prefix, 0)
        self.counts[prefix] = n + 1
        return f"{prefix}_{n:03d}"

    def _shape(self, prefix: str, spec: dict, material: str | None, emission=None, power: float = 1.0) -> None:
        """`power` (luminance x area) sets the emitter's share of light samples.

        Mitsuba otherwise picks emitters uniformly, so the sun would get one
        light sample in ~40 and every sunlit surface would sparkle.
        """
        if material is not None:
            spec["bsdf"] = {"type": "ref", "id": material}
        if emission is not None:
            spec["emitter"] = {"type": "area", "sampling_weight": float(power),
                               "radiance": emission if isinstance(emission, dict) else rgb(*emission)}
            self.lamp_weight += power
        self.d[self._key(prefix)] = spec

    def ply(self, path: str, matrix, material: str | None, emission=None, prefix: str = "mesh",
            power: float = 1.0) -> None:
        self._shape(prefix, {"type": "ply", "filename": path, "to_world": xf(matrix)}, material, emission, power)

    def cube(self, centre, size, material: str, yaw: float = 0.0, emission=None, prefix: str = "box") -> None:
        m = G.compose(G.translate(centre), G.rotate((0, 1, 0), yaw), G.scale(np.asarray(size) / 2))
        self._shape(prefix, {"type": "cube", "to_world": xf(m)}, material, emission)

    def box_mesh(self, name: str, centre, size, material: str, yaw: float = 0.0) -> None:
        """Box with uv in meters, for textures that tile at a physical scale."""
        path = self.assets.mesh(f"box_{name}", lambda: G.box(size))
        self.ply(path, G.compose(G.translate(centre), G.rotate((0, 1, 0), yaw)), material, prefix="boxuv")

    def rect(self, matrix, material: str | None, emission=None, prefix: str = "rect", power: float = 1.0) -> None:
        self._shape(prefix, {"type": "rectangle", "to_world": xf(matrix)}, material, emission, power)

    def sphere(self, centre, radius, material: str | None, emission=None, prefix: str = "sphere") -> None:
        power = 4 * np.pi * radius ** 2 * luminance(emission) if emission is not None else 1.0
        self._shape(prefix, {"type": "sphere", "center": list(map(float, centre)), "radius": float(radius)},
                    material, emission, power)

    def part_set(self, parts: dict[str, str], matrix, materials: dict[str, str], emission=None,
                 power: float = 1.0) -> None:
        for part, path in parts.items():
            emit = emission.get(part) if emission else None
            self.ply(path, matrix, materials.get(part), emit, prefix=part, power=power)


def principled(base, roughness, **kw) -> dict:
    spec = {"type": "principled", "base_color": base if isinstance(base, dict) else rgb(*base),
            "roughness": roughness if isinstance(roughness, dict) else float(roughness)}
    spec.update(kw)
    return spec


def bitmap(path: str, raw: bool = False, uv_scale=None) -> dict:
    spec = {"type": "bitmap", "filename": path, "raw": raw, "filter_type": "bilinear"}
    if uv_scale is not None:
        spec["to_uv"] = mi.ScalarTransform4f().scale([uv_scale[0], uv_scale[1], 1.0])
    return spec
