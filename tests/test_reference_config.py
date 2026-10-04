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
        vector = lambda name: tuple(
            float(value.strip().rstrip("f"))
            for value in re.search(name + r" = glm::vec3\(([^)]+)\)", source)[1].split(","))
        self.assertEqual(vector("importedScenePosition"), reference.IMPORTED_SCENE_POSITION_GL)
        self.assertEqual(vector("importedSceneAlbedo"), reference.IMPORTED_ALBEDO)
        # The seven hand-placed cubes this used to compare are gone; the
        # environment is checked against the shared scene file instead, below.

    def test_scene_file_agrees_with_both_renderers(self):
        """The one check that keeps the two renderers drawing the same scene.

        scene/alley.scene is loaded by src/SceneFile.cpp for the raster pass and
        by tools/scene/scene_loader.py for this Cycles script. The camera and
        lens are additionally duplicated as argparse defaults here and as
        fallback literals in src/main.cpp, so all three have to agree.
        """
        loader, scene, geometry = reference.load_shared_scene()
        source = (ROOT / "src/main.cpp").read_text()

        self.assertEqual(scene.camera.position, reference.CAMERA_POSITION_GL)
        forward = reference.camera_forward_gl(scene.camera.yaw_degrees, scene.camera.pitch_degrees)
        for actual, expected in zip(forward, reference.CAMERA_FORWARD_GL):
            self.assertAlmostEqual(actual, expected, places=9)
        self.assertEqual(scene.camera.focal_length_mm, reference.FOCAL_LENGTH_MM)
        self.assertEqual(scene.camera.sensor_height_mm, reference.SENSOR_HEIGHT_MM)
        self.assertIn(scene.camera.focus_distance_m, reference.FOCUS_DISTANCES_M)
        self.assertIn(scene.camera.f_number, reference.F_STOPS)

        # src/main.cpp keeps the same values as fallbacks for a missing file.
        def cpp_vector(name):
            match = re.search(name + r" = glm::vec3\(([^)]+)\)", source)
            self.assertIsNotNone(match, f"src/main.cpp no longer defines {name}")
            return tuple(float(value.strip().rstrip("f")) for value in match[1].split(","))

        def cpp_scalar(name):
            match = re.search(r"float " + name + r" = ([-\d.]+)f;", source)
            self.assertIsNotNone(match, f"src/main.cpp no longer defines {name}")
            return float(match[1])

        self.assertEqual(cpp_vector("lightDirection"), scene.sun.direction)
        self.assertEqual(cpp_vector("lightColor"), scene.sun.color)
        self.assertEqual(cpp_scalar("lightEnergy"), scene.sun.energy)
        self.assertEqual(cpp_vector("ambientColor"), scene.ambient.color)
        self.assertEqual(cpp_scalar("ambientStrength"), scene.ambient.strength)
        self.assertEqual(cpp_scalar("focusDistanceMeters"), scene.camera.focus_distance_m)
        self.assertEqual(cpp_scalar("fNumber"), scene.camera.f_number)
        self.assertIn(f'scenePath = "{reference.SCENE_FILE}"', source)

        # The digest in every sidecar JSON must be of the file actually loaded.
        plan = reference.make_plan(reference.parse_args([]))
        digest = hashlib.sha256((ROOT / reference.SCENE_FILE).read_bytes()).hexdigest()
        self.assertEqual(plan["scene_sha256"], digest)
        summary = loader.summary(scene, geometry)
        self.assertEqual(plan["scene_summary"]["triangles"], summary["triangles"])
        self.assertEqual(plan["scene_summary"]["sky_radiance"], list(scene.sky_radiance))
        # Emitters have to exist, or the alley has no practical lights at dusk
        # and the whole point of the HDR gather is lost.
        self.assertGreater(summary["emissive_triangles"], 0)

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
