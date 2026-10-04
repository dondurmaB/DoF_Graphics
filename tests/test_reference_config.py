"""CPU-only checks; these do not claim Blender/Cycles runtime verification."""
import contextlib
import hashlib
import importlib.util
import io
import math
from pathlib import Path
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reference", ROOT / "tools/raytraced_reference/render_dof.py")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def transform(matrix, point):
    return tuple(sum(matrix[row][column] * (*point, 1)[column] for column in range(4)) for row in range(3))


class ReferenceConfigTests(unittest.TestCase):
    def test_geometry_only_import_preserves_bytes_and_cleans_up(self):
        geometry = (b"# keep mtllib in comments\r\no teapot\r\n"
                    b"v 1 2 3\r\nvt 0.25 0.5\r\nvn 0 1 0\r\n"
                    b"g handle\r\ns 1\r\nf 1/1/1 1/1/1 1/1/1\r\n")
        original = b"mtllib absent.mtl\r\n  usemtl old\r\n" + geometry
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scene.obj"
            source.write_bytes(original)
            for import_fails in (False, True):
                try:
                    with reference.geometry_only_obj(source) as sanitized:
                        self.assertNotEqual(sanitized, source)
                        self.assertEqual(sanitized.read_bytes(), geometry)
                        if import_fails:
                            raise RuntimeError("simulated importer failure")
                except RuntimeError:
                    if not import_fails:
                        raise
                self.assertFalse(sanitized.parent.exists())
                self.assertEqual(source.read_bytes(), original)

    def test_camera_and_non_square_sensor(self):
        plan = reference.make_plan(
            reference.parse_args(
                ["--scene", "scene/alley.scene", "--width", "1600", "--height", "900"]
            )
        )
        self.assertAlmostEqual(plan["vertical_fov_degrees"], 26.99146656, places=6)
        self.assertAlmostEqual(plan["effective_sensor_width_mm"], 24 * 16 / 9)
        self.assertEqual(plan["camera_position_blender"], (0, -5, 0))

        for actual, expected in zip(plan["camera_forward_blender"], (0, 1, 0)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(plan["camera_up_blender"], (0, 0, 1))
        self.assertLess(reference.vertical_fov_degrees(85, 24), plan["vertical_fov_degrees"])
        self.assertGreater(reference.vertical_fov_degrees(50, 36), plan["vertical_fov_degrees"])

    def test_world_mapping_preserves_distances_and_depth(self):
        for point in [(0, 0, 0), (-0.55, -0.35, 3), (1.6, 0.4, -10), (-3.8, 0.4, -20)]:
            converted = transform(reference.WORLD_CONVERSION, point)
            self.assertEqual(converted, reference.gl_to_blender(point))
            self.assertAlmostEqual(
                math.dist(point, (0, 0, 5)),
                math.dist(converted, reference.gl_to_blender((0, 0, 5))),
            )
            self.assertAlmostEqual(5 - point[2], converted[1] + 5)

    def test_import_rotation_and_scale_applied_once(self):
        model = reference.model_matrix((0, -0.75, 0), -90, 0, (0.1,) * 3, imported=True)
        world = reference.multiply(reference.WORLD_CONVERSION, model)
        for point in [(1, 2, 3), (-4.9784, -2.7938, -0.0347), (4.9596, 2.7938, 5.7429)]:
            expected = (point[0] * 0.1, point[1] * 0.1, point[2] * 0.1 - 0.75)
            for actual, target in zip(transform(world, point), expected):
                self.assertAlmostEqual(actual, target)
        # Nonzero rotations expose accidental reversal of the two source conventions.
        box = reference.model_matrix((0, 0, 0), 90, 90, (1, 1, 1))
        imported = reference.model_matrix((0, 0, 0), 90, 90, (1, 1, 1), imported=True)
        self.assertAlmostEqual(transform(box, (0, 1, 0))[0], 1)
        self.assertAlmostEqual(transform(imported, (0, 1, 0))[2], 1)

    def test_scene_drives_camera_capture_and_optional_import(self):
        loader, scene, geometry = reference.load_shared_scene()
        args = reference.parse_args([])
        plan = reference.make_plan(args)
        self.assertEqual((args.width, args.height), (scene.capture_width, scene.capture_height))
        self.assertEqual(plan["camera_position_gl"], scene.camera.position)
        self.assertEqual(
            plan["camera_forward_gl"],
            reference.camera_forward_gl(scene.camera.yaw_degrees, scene.camera.pitch_degrees),
        )
        self.assertEqual(args.lens, reference.DEFAULT_COMPARISON_LENS_MM)
        self.assertEqual(args.sensor_height, scene.camera.sensor_height_mm)
        self.assertEqual(args.focus, [scene.camera.focus_distance_m])
        self.assertIsNone(plan["asset"])
        self.assertEqual(plan["lighting_mode"], "matched_fill_direct")
        self.assertFalse(plan["ground_truth"])
        self.assertEqual(
            plan["scene_sha256"],
            hashlib.sha256((ROOT / reference.SCENE_FILE).read_bytes()).hexdigest(),
        )
        self.assertEqual(plan["scene_summary"]["triangles"], len(geometry.triangles))
        custom = "version 1\ncapture width 321 height 245\ncamera pos 1 2 3 yaw -37 pitch -11 lens 71 sensor 19 focus 4\nimported pos 2 3 4 rot 21 33 19 size .2 .3 .4\nbox\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.scene"
            path.write_text(custom)
            args = reference.parse_args(["--scene", str(path), "--import-mesh", "--full-gi"])
            plan = reference.make_plan(args)
            self.assertEqual(plan["resolution_pixels"], [321, 245])
            self.assertEqual(plan["focal_length_mm"], 71)
            self.assertEqual(plan["camera_position_gl"], (1, 2, 3))
            self.assertEqual(plan["lighting_mode"], "full_gi_not_comparable")
            self.assertIsNotNone(plan["asset"])
            self.assertEqual(transform(plan["import_matrix_gl"], (0, 0, 0)), (2, 3, 4))
            rotation = loader.rotation_matrix(21, 33, 19)
            expected = loader._apply3(rotation, (0.2, 0.6, 1.2))
            for a, b in zip(
                transform(plan["import_matrix_gl"], (1, 2, 3)),
                (expected[0] + 2, expected[1] + 3, expected[2] + 4),
            ):
                self.assertAlmostEqual(a, b)

    def test_controlled_jobs_and_invalid_arguments(self):
        args = reference.parse_args([])
        self.assertEqual(args.samples, 128)
        self.assertEqual(args.fstops, list(reference.DEFAULT_COMPARISON_F_STOPS))
        self.assertEqual(
            [job[0] for job in reference.render_jobs(args)],
            [f"rt_focus2.5m_f{f:g}_85mm.png" for f in reference.DEFAULT_COMPARISON_F_STOPS],
        )
        args = reference.parse_args(
            ["--preview", "--focus", "2", "5", "15", "--fstops", "1.4", "2.8", "8", "--sharp"]
        )
        self.assertEqual(args.samples, 32)
        self.assertEqual(len(reference.render_jobs(args)), 10)
        with self.assertRaises(ValueError):
            reference.render_jobs(reference.parse_args(["--focus", "5", "5", "--fstops", "1.4"]))
        for argv in [["--focus", "nan"], ["--focus", "0.01"], ["--fstops", "0"],
                     ["--samples", "0"], ["--width", "0"], ["--lens", "inf"], ["--sensor-height", "0"]]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                reference.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
