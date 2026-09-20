"""CPU-only checks; these do not claim Blender/Cycles runtime verification."""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reference", ROOT / "tools/raytraced_reference/render_dof.py")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def transform(matrix, point):
    return tuple(sum(matrix[row][column] * (*point, 1)[column] for column in range(4)) for row in range(3))


class ReferenceConfigTests(unittest.TestCase):
    def test_camera_and_non_square_sensor(self):
        plan = reference.make_plan(reference.parse_args(["--width", "1600", "--height", "900"]))
        self.assertAlmostEqual(plan["vertical_fov_degrees"], 26.99146656, places=6)
        self.assertAlmostEqual(plan["effective_sensor_width_mm"], 24 * 16 / 9)
        self.assertEqual(plan["camera_position_blender"], (0, -5, 0))
        self.assertEqual(plan["camera_forward_blender"], (0, 1, 0))
        self.assertEqual(plan["camera_up_blender"], (0, 0, 1))
        self.assertLess(reference.vertical_fov_degrees(85, 24), plan["vertical_fov_degrees"])
        self.assertGreater(reference.vertical_fov_degrees(50, 36), plan["vertical_fov_degrees"])

    def test_world_mapping_preserves_distances_and_depth(self):
        for point in [(0, 0, 0), (-0.55, -0.35, 3), (1.6, 0.4, -10), (-3.8, 0.4, -20)]:
            converted = transform(reference.WORLD_CONVERSION, point)
            self.assertEqual(converted, reference.gl_to_blender(point))
            self.assertAlmostEqual(math.dist(point, reference.CAMERA_POSITION_GL),
                                   math.dist(converted, reference.gl_to_blender(reference.CAMERA_POSITION_GL)))
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

    def test_settings_match_opengl_source(self):
        source = (ROOT / "src/main.cpp").read_text()
        scalar_pairs = {"focalLengthMillimeters": reference.FOCAL_LENGTH_MM,
                        "sensorHeightMillimeters": reference.SENSOR_HEIGHT_MM,
                        "importedSceneScale": reference.IMPORTED_SCENE_SCALE,
                        "nearPlane": reference.NEAR_M, "farPlane": reference.FAR_M}
        for name, expected in scalar_pairs.items():
            self.assertAlmostEqual(float(re.search(r"float " + name + r" = ([-\d.]+)f;", source)[1]), expected)
        for label, position, rx, ry, scale in reference.BOXES[:3]:
            letter = label[-1]
            match = re.search(r"cube" + letter + r"Position = glm::vec3\(([^)]+)\)", source)
            self.assertEqual(tuple(float(x.strip().rstrip('f')) for x in match[1].split(',')), position)
            for suffix, value in [("RotationXDegrees", rx), ("RotationYDegrees", ry), ("UniformScale", scale[0])]:
                self.assertEqual(float(re.search(r"float cube" + letter + suffix + r" = ([-\d.]+)f;", source)[1]), value)

    def test_controlled_jobs_and_invalid_arguments(self):
        args = reference.parse_args([])
        self.assertEqual(args.samples, 128)
        self.assertEqual([job[0] for job in reference.render_jobs(args)],
                         ["rt_focus5m_f1.4.png", "rt_focus5m_f2.8.png", "rt_focus5m_f8.png"])
        args = reference.parse_args(["--preview", "--focus", "2", "5", "15", "--sharp"])
        self.assertEqual(args.samples, 32)
        self.assertEqual(len(reference.render_jobs(args)), 10)
        with self.assertRaises(ValueError):
            reference.render_jobs(reference.parse_args(["--focus", "5", "5"]))
        for argv in [["--focus", "nan"], ["--focus", "0.01"], ["--fstops", "0"],
                     ["--samples", "0"], ["--width", "0"], ["--lens", "inf"], ["--sensor-height", "0"]]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                reference.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
