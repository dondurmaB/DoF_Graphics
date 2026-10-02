"""CPU-only checks on the shared scene format.

The OpenGL pass and the Cycles reference are only comparable while they draw
the same scene, and they build it from `scene/alley.scene` with two separate
loaders: `tools/scene/scene_loader.py` and `src/SceneFile.cpp`. Nothing forces
those two to agree, so this file pins the Python side to a small set of numbers
and then checks that the same numbers are literally written into
`tests/scene_file.cpp`, which pins the C++ side. Changing one builder without
the other fails here with the exact number that moved.

Run `python3 tests/test_scene_file.py --print-expected` after regenerating the
scene to get the block of constants to paste into `tests/scene_file.cpp`.
"""
import importlib.util
import math
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("scene_loader", ROOT / "tools/scene/scene_loader.py")
scene_loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scene_loader)

SCENE_PATH = ROOT / "scene/cafe.scene"
CPP_TEST_PATH = ROOT / "tests/scene_file.cpp"
CPP_HEADER_PATH = ROOT / "include/SceneFile.h"

# The numbers both builders must agree on. Regenerate with --print-expected.
EXPECTED = {
    "primitives": 2692,
    "vertices": 135050,
    "triangles": 115864,
    "emissive_triangles": 5820,
    "smooth_triangles": 50996,
    "bounds_min": (-3.6, -1.32, -10.2),
    "bounds_max": (3.6, 1.795, 6.2),
    "position_sum": (78741.4137314, -24332.7662, -272601.4688),
    "area_sum": 1353.70549005,
}


def load_scene_under_test():
    scene = scene_loader.load_scene(SCENE_PATH)
    return scene, scene_loader.build_geometry(scene)


def cpp_constants():
    """The expected values as written in the C++ test, by name."""
    source = CPP_TEST_PATH.read_text()
    numbers = r"([-\d.eE+]+)"
    found = {}
    for name in ("kPrimitives", "kVertices", "kTriangles", "kEmissiveTriangles",
                 "kSmoothTriangles", "kAreaSum"):
        match = re.search(r"const (?:std::size_t|double) " + name + r" = " + numbers + ";", source)
        if match is None:
            raise AssertionError(f"tests/scene_file.cpp no longer defines {name}")
        found[name] = float(match[1])
    for name in ("kBoundsMin", "kBoundsMax", "kPositionSum"):
        match = re.search(r"const double " + name + r"\[3\] = \{([^}]+)\};", source)
        if match is None:
            raise AssertionError(f"tests/scene_file.cpp no longer defines {name}")
        found[name] = tuple(float(value) for value in match[1].split(","))
    return found


class SceneGrammarTests(unittest.TestCase):
    def rejects(self, text, fragment):
        with self.assertRaises(scene_loader.SceneError) as caught:
            scene_loader.parse_scene(text)
        self.assertIn(fragment, str(caught.exception))

    def test_bad_files_are_rejected_with_a_line_number(self):
        # Same list as tests/scene_file.cpp, so both loaders refuse the same files.
        self.rejects("box pos 0 0 0\n", "missing 'version'")
        self.rejects("version 2\nbox pos 0 0 0\n", "is not 1")
        self.rejects("version 1\nbox pos 0 0 0 wibble 1\n", "unknown key 'wibble'")
        self.rejects("version 1\nbox seg 8\n", "not valid here")
        self.rejects("version 1\nbox pos 0 0 0 pos 1 1 1\n", "appears twice")
        self.rejects("version 1\nbox pos 0 0\n", "needs 3 number(s)")
        self.rejects("version 1\nbox pos 0 0 x\n", "is not a number")
        self.rejects("version 1\nbox pos 0 0 nan\n", "must be finite")
        self.rejects("version 1\nbox size 1 0 1\n", "size must not be zero")
        self.rejects("version 1\ncyl seg 2\n", "between 3 and 128")
        self.rejects("version 1\nbox emit -1\n", "must not be negative")
        self.rejects("version 1\nshadow lo 0 0 0\n", "both 'lo' and 'hi'")
        self.rejects("version 1\nshadow lo 0 0 0 hi 0 1 1\n", "must exceed 'lo'")
        self.rejects("version 1\ncamera lens 0\n", "lens and sensor must be positive")
        self.rejects("version 1\ncamera focus 0.01\n", "focus must exceed")
        self.rejects("version 1\nwibble 1 2 3\n", "unknown entry 'wibble'")
        self.rejects("version 1\n\n\nbox pos 0 0\n", "line 4")

    def test_comments_and_blank_lines_are_ignored(self):
        scene = scene_loader.parse_scene("# header\nversion 1  # trailing\n\n   \nbox pos 1 2 3\n")
        self.assertEqual(len(scene.primitives), 1)
        self.assertEqual(scene.primitives[0].pos, (1.0, 2.0, 3.0))

    def test_defaults_match_the_cpp_header(self):
        # main.cpp falls back to these when no scene file is present, and the
        # existing reference tests assume the 50mm f/1.4 setup, so the two
        # loaders must not disagree about what "unset" means. Each struct is
        # read separately because `color` appears in two of them.
        scene = scene_loader.parse_scene("version 1\nbox pos 0 0 0\n")
        header = CPP_HEADER_PATH.read_text()

        def struct_body(name):
            match = re.search(r"struct " + name + r" \{(.*?)\n\};", header, re.S)
            self.assertIsNotNone(match, f"include/SceneFile.h no longer defines struct {name}")
            return match[1]

        def scalar(body, name):
            match = re.search(r"float " + name + r" = ([-\d.]+)f;", body)
            self.assertIsNotNone(match, f"include/SceneFile.h no longer defines {name}")
            return float(match[1])

        def vector(body, name):
            match = re.search(r"float " + name + r"\[3\]\{([^}]+)\}", body)
            self.assertIsNotNone(match, f"include/SceneFile.h no longer defines {name}")
            return tuple(float(value.strip().rstrip("f")) for value in match[1].split(","))

        camera = struct_body("SceneCamera")
        self.assertEqual(vector(camera, "position"), scene.camera.position)
        self.assertEqual(scalar(camera, "yawDegrees"), scene.camera.yaw_degrees)
        self.assertEqual(scalar(camera, "pitchDegrees"), scene.camera.pitch_degrees)
        self.assertEqual(scalar(camera, "focusDistanceMeters"), scene.camera.focus_distance_m)
        self.assertEqual(scalar(camera, "fNumber"), scene.camera.f_number)
        self.assertEqual(scalar(camera, "focalLengthMillimeters"), scene.camera.focal_length_mm)
        self.assertEqual(scalar(camera, "sensorHeightMillimeters"), scene.camera.sensor_height_mm)

        sun = struct_body("SceneSun")
        self.assertEqual(vector(sun, "direction"), scene.sun.direction)
        self.assertEqual(vector(sun, "color"), scene.sun.color)
        self.assertEqual(scalar(sun, "energy"), scene.sun.energy)
        self.assertEqual(scalar(sun, "angularDiameterDegrees"), scene.sun.angular_diameter_degrees)

        ambient = struct_body("SceneAmbient")
        self.assertEqual(vector(ambient, "color"), scene.ambient.color)
        self.assertEqual(scalar(ambient, "strength"), scene.ambient.strength)

    def test_unit_primitives_have_the_documented_extents(self):
        scene = scene_loader.parse_scene(
            "version 1\nbox pos 0 0 0\ncyl pos 10 0 0 seg 24\nsph pos 20 0 0 seg 16\n")
        for index, (center, name) in enumerate([(0.0, "box"), (10.0, "cyl"), (20.0, "sph")]):
            one = scene_loader.Scene()
            one.primitives = [scene.primitives[index]]
            low, high = scene_loader.build_geometry(one).bounds()
            for axis in range(3):
                offset = center if axis == 0 else 0.0
                self.assertAlmostEqual(low[axis], offset - 0.5, places=5,
                                       msg=f"{name} is not a unit shape on axis {axis}")
                self.assertAlmostEqual(high[axis], offset + 0.5, places=5,
                                       msg=f"{name} is not a unit shape on axis {axis}")

    def test_transform_order_is_ry_rx_rz(self):
        # Ry*Rx*Rz and Rz*Rx*Ry differ here, so a swapped order fails.
        scene = scene_loader.parse_scene("version 1\nbox pos 0 0 0 rot 90 90 0\n")
        rotation = scene_loader.rotation_matrix(*scene.primitives[0].rot)
        turned = scene_loader._apply3(rotation, (0.0, 1.0, 0.0))
        for actual, expected in zip(turned, (1.0, 0.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=6)
        # A non-uniform scale must not shear the normals of an axis-aligned box.
        stretched = scene_loader.parse_scene("version 1\nbox size 4 1 0.25 rot 0 90 0\n")
        geometry = scene_loader.build_geometry(stretched)
        for normal in geometry.normals:
            self.assertAlmostEqual(scene_loader._length(normal), 1.0, places=6)
            self.assertEqual(sum(1 for value in normal if abs(value) > 1e-6), 1)

    def test_taper_makes_a_frustum_without_changing_plain_cylinders(self):
        """`taper` scales a cylinder's top radius. Added for turned objects.

        Stacking cylinders along a profile always shows: each ring's cap leaves
        a small annular ledge, which reads as banding down the side of a cup. A
        frustum has no internal seams. The critical property is that taper 1.0
        is EXACTLY the old cylinder, so every existing scene file is unaffected.
        """
        plain = scene_loader.build_geometry(scene_loader.parse_scene(
            "version 1\ncyl pos 0 0 0 seg 24\n"))
        explicit = scene_loader.build_geometry(scene_loader.parse_scene(
            "version 1\ncyl pos 0 0 0 seg 24 taper 1\n"))
        self.assertEqual(plain.positions, explicit.positions)
        self.assertEqual(plain.normals, explicit.normals)
        self.assertEqual(plain.triangles, explicit.triangles)

        # A 45-degree cone: the side normal must tilt to match the slope, and
        # the degenerate top cap must be skipped rather than built as a disc of
        # zero radius.
        cone = scene_loader.build_geometry(scene_loader.parse_scene(
            "version 1\ncyl pos 0 0 0 size 1 1 1 seg 32 taper 0\n"))
        expected = scene_loader._normalize((1.0, 0.5, 0.0))
        # The bottom cap shares these positions, so skip axis-aligned normals.
        found = [n for n, p in zip(cone.normals, cone.positions)
                 if abs(p[1] + 0.5) < 1e-9 and abs(p[0] - 0.5) < 1e-9 and abs(n[1]) < 0.9]
        self.assertTrue(found)
        for actual, want in zip(found[0], expected):
            self.assertAlmostEqual(actual, want, places=9)
        self.assertLess(len(cone.triangles), len(plain.triangles) * 32 / 24)
        # Top radius really is zero.
        self.assertAlmostEqual(max(abs(p[0]) for p in cone.positions if abs(p[1] - 0.5) < 1e-9),
                               0.0, places=9)
        self.rejects("version 1\ncyl taper 9\n", "taper must be between 0 and 8")
        self.rejects("version 1\nbox taper 0.5\n", "not valid here")

    def test_emitters_are_split_out_with_premultiplied_radiance(self):
        scene = scene_loader.parse_scene(
            "version 1\nbox pos 0 0 0 rgb 0.5 0.25 0.125\nbox pos 2 0 0 rgb 1 0.5 0.25 emit 8\n")
        geometry = scene_loader.build_geometry(scene)
        shaded, emissive = scene_loader.split_by_emission(geometry)
        self.assertEqual(len(shaded["triangles"]), 12)
        self.assertEqual(len(emissive["triangles"]), 12)
        self.assertEqual(shaded["colors"][0], (0.5, 0.25, 0.125))
        # Cycles gets radiance in the colour attribute; OpenGL multiplies in the shader.
        self.assertEqual(emissive["colors"][0], (8.0, 4.0, 2.0))
        for part in (shaded, emissive):
            for triangle in part["triangles"]:
                for index in triangle:
                    self.assertLess(index, len(part["positions"]))


class CafeSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.geometry = load_scene_under_test()
        cls.summary = scene_loader.summary(cls.scene, cls.geometry)

    def test_counts_and_extents_match_the_expected_numbers(self):
        for name in ("primitives", "vertices", "triangles", "emissive_triangles", "smooth_triangles"):
            self.assertEqual(self.summary[name], EXPECTED[name], msg=name)
        for name in ("bounds_min", "bounds_max", "position_sum"):
            for axis, expected in enumerate(EXPECTED[name]):
                self.assertAlmostEqual(self.summary[name][axis], expected, places=3,
                                       msg=f"{name}[{axis}]")
        self.assertAlmostEqual(self.summary["area_sum"], EXPECTED["area_sum"], places=3)

    def test_cpp_test_pins_the_same_numbers(self):
        constants = cpp_constants()
        self.assertEqual(constants["kPrimitives"], EXPECTED["primitives"])
        self.assertEqual(constants["kVertices"], EXPECTED["vertices"])
        self.assertEqual(constants["kTriangles"], EXPECTED["triangles"])
        self.assertEqual(constants["kEmissiveTriangles"], EXPECTED["emissive_triangles"])
        self.assertEqual(constants["kSmoothTriangles"], EXPECTED["smooth_triangles"])
        self.assertAlmostEqual(constants["kAreaSum"], EXPECTED["area_sum"], places=6)
        for name, key in (("kBoundsMin", "bounds_min"), ("kBoundsMax", "bounds_max"),
                          ("kPositionSum", "position_sum")):
            for axis, expected in enumerate(EXPECTED[key]):
                self.assertAlmostEqual(constants[name][axis], expected, places=4,
                                       msg=f"{name}[{axis}]")

    def test_winding_agrees_with_the_shading_normals(self):
        # Cycles takes a face's front side from its winding, OpenGL from the
        # interpolated normal. Where they disagree the surface is lit in one
        # renderer and black in the other, which would look like a DoF result.
        disagreements = 0
        for (i0, i1, i2) in self.geometry.triangles:
            a, b, c = (self.geometry.positions[i] for i in (i0, i1, i2))
            u = tuple(b[k] - a[k] for k in range(3))
            v = tuple(c[k] - a[k] for k in range(3))
            cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            magnitude = scene_loader._length(cross)
            if magnitude < 1e-12:
                continue
            averaged = [sum(self.geometry.normals[i][k] for i in (i0, i1, i2)) / 3.0 for k in range(3)]
            if sum(cross[k] / magnitude * averaged[k] for k in range(3)) <= 0.0:
                disagreements += 1
        self.assertEqual(disagreements, 0)

    def test_normals_are_unit_length_and_finite(self):
        for index, normal in enumerate(self.geometry.normals):
            self.assertAlmostEqual(scene_loader._length(normal), 1.0, places=5,
                                   msg=f"vertex {index}")

    def test_lighting_is_physically_usable_by_both_renderers(self):
        sun = self.scene.sun
        self.assertAlmostEqual(scene_loader._length(scene_loader._normalize(sun.direction)), 1.0)
        self.assertGreater(sun.direction[1], 0.0, "The sun must be above the horizon")
        self.assertGreater(sun.energy, 0.0)
        # Albedo has to stay under 1 or a surface reflects more than it receives.
        for color in self.geometry.colors:
            for channel in color:
                self.assertGreaterEqual(channel, 0.0)
                self.assertLessEqual(channel, 1.0)
        for channel in self.scene.sky_radiance:
            self.assertGreater(channel, 0.0)
        self.assertEqual(self.scene.sky_radiance,
                         tuple(c * self.scene.ambient.strength for c in self.scene.ambient.color))

    def test_shadow_region_is_declared_and_covers_what_the_camera_sees(self):
        low, high = self.scene.shadow_low, self.scene.shadow_high
        self.assertIsNotNone(low, "Without a shadow region the map would cover the whole room")
        bounds_low, bounds_high = self.geometry.bounds()
        camera = self.scene.camera
        for axis in range(3):
            self.assertLess(low[axis], high[axis])
            # The region may sit slightly outside the geometry (walls are
            # modelled with thickness, so their outer faces are beyond what
            # matters), but it must not be wildly larger than the room or the
            # 4096 px map is wasted on empty space.
            span = high[axis] - low[axis]
            self.assertLess(span, (bounds_high[axis] - bounds_low[axis]) + 2.5,
                            msg=f"shadow region is far larger than the scene on axis {axis}")
        self.assertGreaterEqual(high[2], camera.position[2] - 1e-6,
                                "The shadow region must reach the camera plane")
        # The subject at the focus plane has to be inside it, or it casts no shadow.
        subject_z = camera.position[2] - camera.focus_distance_m
        self.assertLess(low[2], subject_z)
        self.assertGreater(high[2], subject_z)

    def test_the_scene_fills_the_frame_it_was_composed_for(self):
        # Vertical half-extent at distance d is 0.5 * sensor / lens * d. The alley
        # walls must clear it, otherwise the sky shows through the sides.
        camera = self.scene.camera
        half_angle = 0.5 * camera.sensor_height_mm / camera.focal_length_mm
        subject_z = camera.position[2] - camera.focus_distance_m
        half_height = half_angle * camera.focus_distance_m
        # The frame at the focus plane has to be big enough to hold the subject
        # (a 14 cm saucer) and tight enough to be a close framing rather than a
        # room shot, or the near-field blur has nothing to act on.
        self.assertGreater(half_height, 0.15)
        self.assertLess(half_height, 1.00)
        near, far, side = 0, 0, 0
        for position in self.geometry.positions:
            depth = camera.position[2] - position[2]
            if depth < 0.5:
                continue
            if depth < 2.0:
                near += 1
            if depth > 12.0:
                far += 1
            if abs(position[0]) > half_angle * depth:
                side += 1
        # Depth of field needs something sharp, something near and something far.
        self.assertGreater(near, 100, "Nothing within 2 m to blur in the foreground")
        self.assertGreater(far, 100, "Nothing past 12 m to blur in the background")
        self.assertGreater(side, 100, "Nothing outside the frame edges, so the walls must be too short")
        self.assertLess(subject_z, camera.position[2], "The focus plane must be in front of the camera")


def print_expected():
    """Emit the constant block for tests/scene_file.cpp."""
    scene, geometry = load_scene_under_test()
    values = scene_loader.summary(scene, geometry)
    print(f'const std::size_t kPrimitives = {values["primitives"]};')
    print(f'const std::size_t kVertices = {values["vertices"]};')
    print(f'const std::size_t kTriangles = {values["triangles"]};')
    print(f'const std::size_t kEmissiveTriangles = {values["emissive_triangles"]};')
    print(f'const std::size_t kSmoothTriangles = {values["smooth_triangles"]};')
    for name, key in (("kBoundsMin", "bounds_min"), ("kBoundsMax", "bounds_max"),
                      ("kPositionSum", "position_sum")):
        joined = ", ".join(f"{value:.12g}" for value in values[key])
        print(f"const double {name}[3] = {{{joined}}};")
    print(f'const double kAreaSum = {values["area_sum"]:.12g};')
    print()
    print("Paste the block above into tests/scene_file.cpp and copy the same")
    print("numbers into EXPECTED in this file.")


if __name__ == "__main__":
    if "--print-expected" in sys.argv:
        print_expected()
    else:
        unittest.main()
