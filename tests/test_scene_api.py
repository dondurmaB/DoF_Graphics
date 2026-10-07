"""Static checks for the scene contract. No Mitsuba or GPU needed.

    python tests/test_scene_api.py        (or pytest)
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "renderer" / "mitsuba"))

import numpy as np  # noqa: E402

import scene_api as S  # noqa: E402


def make_scene(**kw) -> S.SceneDef:
    base = dict(
        id="desk", group="artificial", owner="x", description="a desk",
        build=lambda c: S.SceneBundle({"type": "scene", "sky": {"type": "constant"}}, {"cup": [0, 1, 0]}),
        views={"home": S.View((0, 1, 2), (0, 1, 0), focus="cup")}, default_view="home",
        camera_box=((-1, 0, 0), (1, 2, 3)), target_box=((-1, 0, -3), (1, 2, 1)), tags=("indoor",))
    base.update(kw)
    return S.SceneDef(**base)


def test_good_scene_is_clean():
    s = make_scene()
    assert S.validate_definition(s) == []
    assert S.validate_bundle(s, s.build(None)) == []
    assert isinstance(hash(s), int)


def test_definition_errors():
    assert S.validate_definition(make_scene(id="Desk!"))
    assert S.validate_definition(make_scene(group="indoor"))
    assert S.validate_definition(make_scene(envs=("snow",)))
    assert S.validate_definition(make_scene(envs=("clear", "haze")))
    assert S.validate_definition(make_scene(tags=("cozy",)))
    assert S.validate_definition(make_scene(tags=()))
    assert S.validate_definition(make_scene(default_view="nope"))
    assert S.validate_definition(make_scene(views={"home": S.View((9, 9, 9), (0, 1, 0))}))
    assert S.validate_definition(make_scene(views={"home": S.View((0, 1, 2), (0, 1, 2))}))
    assert S.validate_definition(make_scene(views={"home": S.View((0, 1), (0, 1, 0))}))
    assert S.validate_definition(make_scene(exclude_boxes=(((-1, 0, 1), (1, 2, 3)),)))
    assert S.validate_definition(make_scene(camera_box=((1, 0, 0), (-1, 2, 3))))


def test_bundle_errors():
    s = make_scene()
    assert S.validate_bundle(s, S.SceneBundle({"type": "scene", "sensor": {}, "l": {"type": "constant"}}, {"a": [0, 0, 0]}))
    assert S.validate_bundle(s, S.SceneBundle({"type": "scene"}, {"a": [0, 0, 0]}))
    assert S.validate_bundle(s, S.SceneBundle({"type": "scene", "l": {"type": "constant"}}, {}))
    assert S.validate_bundle(replace(s, views={"home": S.View((0, 1, 2), (0, 1, 0), focus="ghost")}), s.build(None))


def test_focus_points_are_normalised_or_rejected():
    b = S.SceneBundle({}, {"a": (0, 1, 2), "b": np.zeros((3, 3)), "c": [[0, 0, 0], [1, 1, 1]]})
    assert b.focus_points["a"] == [[0.0, 1.0, 2.0]] and len(b.focus_points["b"]) == 3
    for bad in ([], [[]], [0, 1], [float("nan"), 0, 0]):
        try:
            S.SceneBundle({}, {"p": bad})
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_env_gate_and_context():
    s = make_scene()
    ctx = S.make_context(s, Path("/tmp/x"), seed=3, env="clear")
    assert ctx.assets_dir == Path("/tmp/x/desk/v1/s3/clear") and ctx.shared_dir == Path("/tmp/x/desk/shared_v1")
    for env in ("snow", "haze"):
        try:
            S.build_checked(s, S.make_context(s, Path("/tmp/x"), 3, env))
        except ValueError:
            continue
        raise AssertionError(env)
    assert S.build_checked(s, ctx).focus_points["cup"] == [[0.0, 1.0, 0.0]]


def test_fingerprint_ignores_cache_location_and_sees_changes():
    a = {"m": {"filename": "/home/a/gen/x.ply", "v": 1.0000001}}
    b = {"m": {"filename": "/data/b/x.ply", "v": 1.0}}
    assert S.fingerprint(a, [Path("/home/a/gen")]) == S.fingerprint({"m": {"filename": "x.ply", "v": 1.0}})
    assert S.fingerprint(b, [Path("/data/b")]) == S.fingerprint({"m": {"filename": "x.ply", "v": 1.0}})
    assert S.fingerprint(a) != S.fingerprint({"m": {"filename": "/home/a/gen/x.ply", "v": 2.0}})


def test_every_scene_file_validates_and_has_a_split():
    ids = S.list_scenes()
    assert ids, "no scenes found"
    for sid in ids:
        assert S.validate_definition(S.load_scene(sid)) == [], sid
    assert S.validate_splits(ids) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
