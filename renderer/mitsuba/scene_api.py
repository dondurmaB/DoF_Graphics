"""Scene contract shared by every scene author. Spec: SCENE_CONTRACT.md.

A scene lives in `scenes/<id>.py` and exports `SCENE: SceneDef`. The shared
renderer and sampler only ever talk to scenes through this file. Importing it
needs neither Mitsuba nor a GPU.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

CONTRACT_VERSION = 1
HERE = Path(__file__).resolve().parent
SCENES_DIR = HERE / "scenes"
SPLITS_FILE = SCENES_DIR / "splits.json"

SENSOR_HEIGHT_MM = 24.0   # every scene is rendered with this sensor
GROUPS = ("artificial", "natural", "hybrid")
SPLITS = ("train", "val", "test", "test_ood")
# Aperture shapes the renderer can produce. Only the circular disc exists today; see
# SCENE_CONTRACT.md, "Aperture shape", for how more shapes plug in.
APERTURES = ("disc",)
# Presets a scene implements itself. `haze` is applied by the shared renderer to
# any scene and must not be declared here.
ENVS = ("clear", "overcast", "snow", "sand", "wet")
SHARED_ENVS = ("haze",)
# Controlled vocabulary, so coverage can be counted. Extend it here, with both owners agreeing.
TAGS = ("indoor", "outdoor", "day", "night", "artificial_light", "clutter", "open", "corridor", "closeup",
        "glass", "water", "metal", "foliage", "bokeh", "specular", "snow", "sand", "architecture", "vista")
_EMITTER_PLUGINS = {"sunsky", "sun", "sky", "constant", "envmap", "point", "spot", "directional", "area",
                    "projector", "collimated", "directionalarea"}
_Box = tuple  # ((x0, y0, z0), (x1, y1, z1))


@dataclass(frozen=True)
class BuildContext:
    assets_dir: Path          # depends on seed/env: write caches that vary with them here
    shared_dir: Path          # depends on the recipe only: seed-independent meshes and textures go here
    seed: int                 # the only source of randomness a scene may use
    env: str = "clear"
    rebuild: bool = False


@dataclass(frozen=True)
class View:
    origin: tuple[float, float, float]
    target: tuple[float, float, float]
    focus: str | None = None  # name of a focus point; default: the scene's first one
    roll_deg: float = 0.0     # camera roll about the optical axis, right-hand rotation of `up` about forward


def _points(name: str, value) -> list[list[float]]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 3 or not np.isfinite(arr).all():
        raise ValueError(f"focus point {name!r} must be [x, y, z] or a non-empty list of them (finite)")
    return arr.tolist()


@dataclass
class SceneBundle:
    scene: dict                                   # Mitsuba scene dict: no sensor, no integrator
    focus_points: dict[str, list] = field(default_factory=dict)   # name -> [x,y,z] or list of them
    meta: dict = field(default_factory=dict)      # free-form JSON-able facts (sun direction, ...) copied to metadata

    def __post_init__(self):
        self.focus_points = {k: _points(k, v) for k, v in self.focus_points.items()}


@dataclass(frozen=True, eq=False)
class SceneDef:
    id: str
    group: str                                    # one of GROUPS
    owner: str
    description: str
    build: Callable[[BuildContext], SceneBundle]
    views: dict[str, View]                        # named hero cameras, all inside camera_box
    default_view: str
    camera_box: _Box                              # (min xyz, max xyz): cameras the sampler may place
    target_box: _Box                              # (min xyz, max xyz): points the camera may look at
    exclude_boxes: tuple[_Box, ...] = ()          # carve-outs of camera_box (furniture, walls)
    lens_mm: tuple[float, float] = (24.0, 135.0)  # focal-length range the scene is tuned for
    f_number: tuple[float, float] = (1.4, 8.0)
    envs: tuple[str, ...] = ("clear",)            # presets this scene implements
    tags: tuple[str, ...] = ()
    default_seed: int = 0
    asset_version: int = 1                        # bump when a recipe changes; invalidates shared_dir
    max_depth: int = 12                           # path-tracer bounces this scene needs
    rr_depth: int = 6
    spp_hint: int = 2048                          # spp this scene needs for a clean ground-truth DoF pass
    far_clip: float = 1000.0


def make_context(scene: SceneDef, root: Path, seed: int | None = None, env: str = "clear",
                 rebuild: bool = False) -> BuildContext:
    seed = scene.default_seed if seed is None else int(seed)
    base = Path(root) / scene.id
    return BuildContext(assets_dir=base / f"v{scene.asset_version}" / f"s{seed}" / env,
                        shared_dir=base / f"shared_v{scene.asset_version}",
                        seed=seed, env=env, rebuild=rebuild)


def build_checked(scene: SceneDef, ctx: BuildContext) -> SceneBundle:
    if ctx.env in SHARED_ENVS:
        raise ValueError(f"{ctx.env!r} is applied by the shared renderer; build the 'clear' scene")
    if ctx.env not in scene.envs:
        raise ValueError(f"scene {scene.id!r} does not implement env {ctx.env!r} (declares {scene.envs})")
    bundle = scene.build(ctx)
    errs = validate_bundle(scene, bundle)
    if errs:
        raise ValueError(f"scene {scene.id!r} violates the contract:\n  " + "\n  ".join(errs))
    return bundle


def list_scenes() -> list[str]:
    return sorted(p.stem for p in SCENES_DIR.glob("*.py") if not p.name.startswith("_"))


def load_scene(scene_id: str) -> SceneDef:
    path = SCENES_DIR / f"{scene_id}.py"
    if not path.is_file():
        raise ValueError(f"no scene {scene_id!r}; available: {list_scenes()}")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))      # scene files import procedural, scene_kit, ...
    name = f"dof_scene_{scene_id}"
    if name in sys.modules:
        module = sys.modules[name]
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    scene = getattr(module, "SCENE", None)
    if not isinstance(scene, SceneDef):
        raise ValueError(f"scenes/{scene_id}.py must export SCENE: SceneDef")
    if scene.id != scene_id:
        raise ValueError(f"scenes/{scene_id}.py declares id {scene.id!r}")
    return scene


def load_splits() -> dict[str, list[str]]:
    data = json.loads(SPLITS_FILE.read_text()) if SPLITS_FILE.exists() else {}
    return {k: list(data.get(k, [])) for k in SPLITS}


def validate_splits(scene_ids: list[str]) -> list[str]:
    """Every scene is in exactly one split; splits name no unknown scene."""
    splits, errs, seen = load_splits(), [], {}
    for split, ids in splits.items():
        for sid in ids:
            if sid in seen:
                errs.append(f"{sid!r} is in both {seen[sid]!r} and {split!r}")
            seen[sid] = split
    errs += [f"{sid!r} is in splits.json but has no scene file" for sid in seen if sid not in scene_ids]
    errs += [f"{sid!r} is not assigned to a split in splits.json" for sid in scene_ids if sid not in seen]
    return errs


def _in_box(p, box) -> bool:
    return all(lo <= v <= hi for v, lo, hi in zip(p, *box))


def _is_vec3(v) -> bool:
    return len(v) == 3 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)


def _is_box(b) -> bool:
    return len(b) == 2 and all(_is_vec3(v) for v in b) and all(lo < hi for lo, hi in zip(*b))


def _has_emitter(d: dict) -> bool:
    for v in d.values():
        if isinstance(v, dict) and ("emitter" in v or v.get("type") in _EMITTER_PLUGINS):
            return True
    return False


def validate_definition(s: SceneDef) -> list[str]:
    """Cheap static checks on the declaration; no Mitsuba needed."""
    import re

    errs = []
    if not re.fullmatch(r"[a-z][a-z0-9_]*", s.id):
        errs.append("id must be lowercase snake_case")
    if s.group not in GROUPS:
        errs.append(f"group must be one of {GROUPS}")
    if not callable(s.build):
        errs.append("build must be callable")
    if not s.description.strip():
        errs.append("description is empty")
    bad_tags = [t for t in s.tags if t not in TAGS]
    if not s.tags or bad_tags:
        errs.append(f"tags must be non-empty and from the controlled vocabulary; unknown: {bad_tags}")
    if "clear" not in s.envs or not set(s.envs) <= set(ENVS):
        errs.append(f"envs must include 'clear' and be a subset of {ENVS} (haze is applied by the renderer)")
    if not (_is_box(s.camera_box) and _is_box(s.target_box) and all(_is_box(b) for b in s.exclude_boxes)):
        errs.append("camera_box, target_box and exclude_boxes must be ((x0,y0,z0),(x1,y1,z1)) with min < max")
        return errs
    if s.default_view not in s.views:
        errs.append("default_view is not in views")
    for name, v in s.views.items():
        if not (_is_vec3(v.origin) and _is_vec3(v.target)):
            errs.append(f"view {name!r}: origin and target must be finite 3-vectors")
            continue
        if math.dist(v.origin, v.target) < 1e-6:
            errs.append(f"view {name!r}: origin and target coincide")
        if not _in_box(v.origin, s.camera_box):
            errs.append(f"view {name!r}: origin is outside camera_box")
        if any(_in_box(v.origin, b) for b in s.exclude_boxes):
            errs.append(f"view {name!r}: origin is inside an exclude_box")
        if not _in_box(v.target, s.target_box):
            errs.append(f"view {name!r}: target is outside target_box")
        if not math.isfinite(v.roll_deg):
            errs.append(f"view {name!r}: roll_deg must be finite")
    if not (0 < s.lens_mm[0] <= s.lens_mm[1] and 0 < s.f_number[0] <= s.f_number[1]):
        errs.append("lens_mm and f_number ranges must be positive (min, max)")
    if s.max_depth < 1 or s.spp_hint < 1 or s.far_clip <= 0:
        errs.append("max_depth, spp_hint and far_clip must be positive")
    return errs


def validate_bundle(s: SceneDef, b: SceneBundle) -> list[str]:
    """Cheap static checks on a built scene; no rendering needed."""
    errs = []
    d = b.scene
    if d.get("type") != "scene":
        errs.append("scene dict must have type 'scene'")
    for banned in ("sensor", "integrator"):
        if banned in d:
            errs.append(f"scene dict must not contain {banned!r}; the renderer owns it")
    if not _has_emitter(d):
        errs.append("scene has no emitter")
    if not b.focus_points:
        errs.append("scene needs at least one named focus point")
    for name, v in s.views.items():
        if v.focus is not None and v.focus not in b.focus_points:
            errs.append(f"view {name!r} focuses on unknown point {v.focus!r}")
    return errs


def fingerprint(scene_dict: dict, roots: list[Path] = ()) -> str:
    """Stable hash of a scene dict. Absolute cache paths are made relative and
    transforms become matrices, so two machines building the same scene agree."""
    web_cache = Path(os.environ.get("DOF_WEB_ASSETS", HERE / "web_assets"))   # same default as web_assets.CACHE
    prefixes = sorted((str(Path(r)) + "/" for r in [*roots, web_cache]), key=len, reverse=True)

    def norm(x):
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        if isinstance(x, str):
            for p in prefixes:
                if x.startswith(p):
                    return x[len(p):]
            return x
        if isinstance(x, (np.ndarray, np.generic)):
            return norm(x.tolist())
        if isinstance(x, float):
            return round(x, 6)
        if hasattr(x, "matrix"):                       # mi.ScalarTransform4f
            return norm(np.asarray(x.matrix, dtype=float).tolist())
        return x

    blob = json.dumps(norm(scene_dict), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
