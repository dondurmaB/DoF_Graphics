"""Parse `scene/*.scene` and build the triangle soup both renderers draw.

`src/SceneFile.cpp` is the C++ twin of this module. Both must produce identical
geometry from the same file, because the OpenGL raster pass and the Cycles
reference are only comparable while they draw the same scene. The equivalence is
pinned by `tests/test_scene_file.py` (this module) and `tests/scene_file.cpp`
(the C++ side) checking the same expected numbers.

Conventions, identical on both sides:
  * Right-handed OpenGL world space, 1 unit = 1 meter, +Y up, camera looks down -Z.
  * Primitives are authored as unit shapes and scaled by `size`:
      box: axis-aligned cube, corners at +/-0.5
      cyl: Y-axis cylinder, radius 0.5 in X/Z, height 1, closed with two caps
      sph: sphere of radius 0.5
  * world = pos + Ry(rot.y) * Rx(rot.x) * Rz(rot.z) * (size * local)
  * normals use the inverse transpose of that matrix, which for a diagonal
    scale means dividing the local normal by `size` before rotating.
  * `rgb` is LINEAR albedo (not sRGB). `emit` > 0 makes the surface a pure
    emitter of radiance rgb * emit; it is then not shaded.
"""

import math
from pathlib import Path

VERSION = 1

# Every key takes a fixed number of floats, so parsing needs no lookahead.
KEY_ARITY = {
    "pos": 3, "size": 3, "rot": 3, "rgb": 3, "emit": 1, "seg": 1, "smooth": 1,
    "yaw": 1, "pitch": 1, "focus": 1, "fnumber": 1, "lens": 1, "sensor": 1,
    "dir": 3, "color": 3, "energy": 1, "angle": 1, "strength": 1, "value": 1,
    "lo": 3, "hi": 3,
}
PRIMITIVE_KEYS = {
    "box": ("pos", "size", "rot", "rgb", "emit"),
    "cyl": ("pos", "size", "rot", "rgb", "emit", "seg", "smooth"),
    "sph": ("pos", "size", "rot", "rgb", "emit", "seg"),
}
DEFAULT_SEGMENTS = {"cyl": 16, "sph": 12}

# Unit cube: 6 faces x 4 corners, wound counter-clockwise seen from outside.
# Same face order and corner order as the hand-written cube this scene replaced.
BOX_FACES = (
    (((-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)), (0.0, 0.0, 1.0)),
    (((0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5)), (0.0, 0.0, -1.0)),
    (((-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)), (-1.0, 0.0, 0.0)),
    (((0.5, -0.5, 0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)), (1.0, 0.0, 0.0)),
    (((-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5)), (0.0, 1.0, 0.0)),
    (((-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)), (0.0, -1.0, 0.0)),
)


class SceneError(ValueError):
    """Raised with a line number so a bad scene file says where it is wrong."""


class Primitive:
    __slots__ = ("kind", "pos", "size", "rot", "rgb", "emit", "segments", "smooth", "line")

    def __init__(self, kind, line):
        self.kind = kind
        self.line = line
        self.pos = (0.0, 0.0, 0.0)
        self.size = (1.0, 1.0, 1.0)
        self.rot = (0.0, 0.0, 0.0)
        self.rgb = (0.8, 0.8, 0.8)
        self.emit = 0.0
        self.segments = DEFAULT_SEGMENTS.get(kind, 0)
        self.smooth = True


class Camera:
    def __init__(self):
        self.position = (0.0, 0.0, 5.0)
        self.yaw_degrees = -90.0
        self.pitch_degrees = 0.0
        self.focus_distance_m = 5.0
        self.f_number = 1.4
        self.focal_length_mm = 50.0
        self.sensor_height_mm = 24.0


class Sun:
    def __init__(self):
        self.direction = (-0.45, 0.78, 0.44)  # From a surface TOWARD the light.
        self.color = (1.0, 0.88, 0.72)
        self.energy = 4.2  # Irradiance in W/m^2, as Blender's sun strength.
        self.angular_diameter_degrees = 0.6


class Ambient:
    def __init__(self):
        self.color = (0.42, 0.52, 0.72)
        self.strength = 0.24  # Uniform sky radiance = color * strength.


class Scene:
    def __init__(self):
        self.version = VERSION
        self.camera = Camera()
        self.sun = Sun()
        self.ambient = Ambient()
        self.primitives = []
        # Region the shadow map must cover. None means "use the geometry bounds",
        # which is wrong when the scene also holds distant filler geometry.
        self.shadow_low = None
        self.shadow_high = None

    @property
    def sky_radiance(self):
        """What the camera sees where nothing is hit; also the ambient term."""
        return tuple(c * self.ambient.strength for c in self.ambient.color)


def _floats(tokens, index, count, line_number, key):
    if index + count > len(tokens):
        raise SceneError(f"line {line_number}: key '{key}' needs {count} number(s)")
    values = []
    for offset in range(count):
        token = tokens[index + offset]
        try:
            value = float(token)
        except ValueError:
            raise SceneError(f"line {line_number}: '{token}' after '{key}' is not a number") from None
        if not math.isfinite(value):
            raise SceneError(f"line {line_number}: '{key}' must be finite")
        values.append(value)
    return values


def _key_values(tokens, line_number, allowed):
    """Read `key v.. key v..` pairs; unknown or repeated keys are errors."""
    index, found = 0, {}
    while index < len(tokens):
        key = tokens[index]
        if key not in KEY_ARITY:
            raise SceneError(f"line {line_number}: unknown key '{key}'")
        if key not in allowed:
            raise SceneError(f"line {line_number}: key '{key}' is not valid here")
        if key in found:
            raise SceneError(f"line {line_number}: key '{key}' appears twice")
        arity = KEY_ARITY[key]
        found[key] = _floats(tokens, index + 1, arity, line_number, key)
        index += 1 + arity
    return found


def parse_scene(text):
    scene = Scene()
    seen_version = False
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        kind, rest = tokens[0], tokens[1:]
        if kind == "version":
            values = _floats(rest, 0, 1, line_number, "version")
            if int(values[0]) != VERSION:
                raise SceneError(f"line {line_number}: scene version {values[0]:g} is not {VERSION}")
            seen_version = True
        elif kind == "camera":
            found = _key_values(rest, line_number,
                                ("pos", "yaw", "pitch", "focus", "fnumber", "lens", "sensor"))
            camera = scene.camera
            if "pos" in found:
                camera.position = tuple(found["pos"])
            if "yaw" in found:
                camera.yaw_degrees = found["yaw"][0]
            if "pitch" in found:
                camera.pitch_degrees = found["pitch"][0]
            if "focus" in found:
                camera.focus_distance_m = found["focus"][0]
            if "fnumber" in found:
                camera.f_number = found["fnumber"][0]
            if "lens" in found:
                camera.focal_length_mm = found["lens"][0]
            if "sensor" in found:
                camera.sensor_height_mm = found["sensor"][0]
            if camera.focal_length_mm <= 0.0 or camera.sensor_height_mm <= 0.0:
                raise SceneError(f"line {line_number}: lens and sensor must be positive")
            if camera.f_number <= 0.0 or camera.focus_distance_m <= camera.focal_length_mm * 0.001:
                raise SceneError(f"line {line_number}: f-number must be positive and focus must exceed the focal length")
        elif kind == "sun":
            found = _key_values(rest, line_number, ("dir", "color", "energy", "angle"))
            if "dir" in found:
                direction = tuple(found["dir"])
                if _length(direction) < 1e-6:
                    raise SceneError(f"line {line_number}: sun direction must not be zero")
                scene.sun.direction = direction
            if "color" in found:
                scene.sun.color = tuple(found["color"])
            if "energy" in found:
                scene.sun.energy = found["energy"][0]
            if "angle" in found:
                scene.sun.angular_diameter_degrees = found["angle"][0]
        elif kind == "shadow":
            found = _key_values(rest, line_number, ("lo", "hi"))
            if ("lo" in found) != ("hi" in found):
                raise SceneError(f"line {line_number}: shadow needs both 'lo' and 'hi'")
            if "lo" in found:
                if any(found["hi"][axis] <= found["lo"][axis] for axis in range(3)):
                    raise SceneError(f"line {line_number}: shadow 'hi' must exceed 'lo' on every axis")
                scene.shadow_low = tuple(found["lo"])
                scene.shadow_high = tuple(found["hi"])
        elif kind == "ambient":
            found = _key_values(rest, line_number, ("color", "strength"))
            if "color" in found:
                scene.ambient.color = tuple(found["color"])
            if "strength" in found:
                scene.ambient.strength = found["strength"][0]
        elif kind in PRIMITIVE_KEYS:
            primitive = Primitive(kind, line_number)
            found = _key_values(rest, line_number, PRIMITIVE_KEYS[kind])
            if "pos" in found:
                primitive.pos = tuple(found["pos"])
            if "size" in found:
                primitive.size = tuple(found["size"])
            if "rot" in found:
                primitive.rot = tuple(found["rot"])
            if "rgb" in found:
                primitive.rgb = tuple(found["rgb"])
            if "emit" in found:
                primitive.emit = found["emit"][0]
            if "seg" in found:
                primitive.segments = int(found["seg"][0])
            if "smooth" in found:
                primitive.smooth = found["smooth"][0] != 0.0
            if any(abs(value) < 1e-9 for value in primitive.size):
                raise SceneError(f"line {line_number}: size must not be zero on any axis")
            if primitive.kind != "box" and not 3 <= primitive.segments <= 128:
                raise SceneError(f"line {line_number}: seg must be between 3 and 128")
            if primitive.emit < 0.0:
                raise SceneError(f"line {line_number}: emit must not be negative")
            scene.primitives.append(primitive)
        else:
            raise SceneError(f"line {line_number}: unknown entry '{kind}'")
    if not seen_version:
        raise SceneError("missing 'version' line; refusing to guess the scene format")
    return scene


def load_scene(path):
    return parse_scene(Path(path).read_text())


def _length(vector):
    return math.sqrt(sum(component * component for component in vector))


def _normalize(vector):
    length = _length(vector)
    if length < 1e-12:
        return (0.0, 1.0, 0.0)
    return tuple(component / length for component in vector)


def _multiply3(a, b):
    return tuple(tuple(sum(a[row][k] * b[k][column] for k in range(3)) for column in range(3))
                 for row in range(3))


def _apply3(matrix, vector):
    return tuple(sum(matrix[row][k] * vector[k] for k in range(3)) for row in range(3))


def rotation_matrix(rx_degrees, ry_degrees, rz_degrees):
    """R = Ry * Rx * Rz, the one order both builders use."""
    x, y, z = (math.radians(value) for value in (rx_degrees, ry_degrees, rz_degrees))
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rotate_x = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    rotate_y = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    rotate_z = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    return _multiply3(rotate_y, _multiply3(rotate_x, rotate_z))


class Geometry:
    """Triangle soup in world space: one vertex list, one index list."""

    def __init__(self):
        self.positions = []
        self.normals = []
        self.colors = []
        self.emissions = []
        self.triangles = []
        self.triangle_smooth = []

    def add_vertex(self, position, normal, color, emission):
        self.positions.append(position)
        self.normals.append(normal)
        self.colors.append(color)
        self.emissions.append(emission)
        return len(self.positions) - 1

    def add_triangle(self, i0, i1, i2, smooth):
        self.triangles.append((i0, i1, i2))
        self.triangle_smooth.append(smooth)

    def bounds(self):
        if not self.positions:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        low = tuple(min(p[axis] for p in self.positions) for axis in range(3))
        high = tuple(max(p[axis] for p in self.positions) for axis in range(3))
        return low, high

    def triangle_area_sum(self):
        total = 0.0
        for i0, i1, i2 in self.triangles:
            a, b, c = self.positions[i0], self.positions[i1], self.positions[i2]
            u = tuple(b[k] - a[k] for k in range(3))
            v = tuple(c[k] - a[k] for k in range(3))
            cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            total += 0.5 * _length(cross)
        return total


def _emit_primitive(geometry, primitive):
    rotation = rotation_matrix(*primitive.rot)
    size, position = primitive.size, primitive.pos

    def place(local_position, local_normal):
        scaled = tuple(local_position[k] * size[k] for k in range(3))
        rotated = _apply3(rotation, scaled)
        world = tuple(rotated[k] + position[k] for k in range(3))
        # Inverse transpose of R*S for a diagonal S: divide by the scale, then rotate.
        unscaled = tuple(local_normal[k] / size[k] for k in range(3))
        normal = _normalize(_apply3(rotation, unscaled))
        return geometry.add_vertex(world, normal, primitive.rgb, primitive.emit)

    if primitive.kind == "box":
        for corners, normal in BOX_FACES:
            base = [place(corner, normal) for corner in corners]
            geometry.add_triangle(base[0], base[1], base[2], False)
            geometry.add_triangle(base[2], base[3], base[0], False)
        return

    if primitive.kind == "cyl":
        segments = primitive.segments
        angles = [2.0 * math.pi * index / segments for index in range(segments)]
        if primitive.smooth:
            bottom, top = [], []
            for angle in angles:
                direction = (math.cos(angle), 0.0, math.sin(angle))
                bottom.append(place((0.5 * direction[0], -0.5, 0.5 * direction[2]), direction))
                top.append(place((0.5 * direction[0], 0.5, 0.5 * direction[2]), direction))
            for index in range(segments):
                nxt = (index + 1) % segments
                geometry.add_triangle(bottom[index], top[index], top[nxt], True)
                geometry.add_triangle(bottom[index], top[nxt], bottom[nxt], True)
        else:
            for index in range(segments):
                a, b = angles[index], angles[(index + 1) % segments]
                mid = 0.5 * (a + b) if index + 1 < segments else 0.5 * (a + b + 2.0 * math.pi)
                face_normal = (math.cos(mid), 0.0, math.sin(mid))
                quad = [
                    place((0.5 * math.cos(a), -0.5, 0.5 * math.sin(a)), face_normal),
                    place((0.5 * math.cos(a), 0.5, 0.5 * math.sin(a)), face_normal),
                    place((0.5 * math.cos(b), 0.5, 0.5 * math.sin(b)), face_normal),
                    place((0.5 * math.cos(b), -0.5, 0.5 * math.sin(b)), face_normal),
                ]
                geometry.add_triangle(quad[0], quad[1], quad[2], False)
                geometry.add_triangle(quad[0], quad[2], quad[3], False)
        for sign in (1.0, -1.0):
            normal = (0.0, sign, 0.0)
            center = place((0.0, 0.5 * sign, 0.0), normal)
            ring = [place((0.5 * math.cos(angle), 0.5 * sign, 0.5 * math.sin(angle)), normal)
                    for angle in angles]
            for index in range(segments):
                nxt = (index + 1) % segments
                if sign > 0.0:
                    geometry.add_triangle(center, ring[nxt], ring[index], False)
                else:
                    geometry.add_triangle(center, ring[index], ring[nxt], False)
        return

    # Sphere: UV grid with a duplicated seam column so both builders index alike.
    segments = primitive.segments
    rings = max(2, segments // 2)
    grid = []
    for ring in range(rings + 1):
        phi = -0.5 * math.pi + math.pi * ring / rings
        row = []
        for longitude in range(segments + 1):
            theta = 2.0 * math.pi * longitude / segments
            normal = (math.cos(phi) * math.cos(theta), math.sin(phi), math.cos(phi) * math.sin(theta))
            row.append(place((0.5 * normal[0], 0.5 * normal[1], 0.5 * normal[2]), normal))
        grid.append(row)
    for ring in range(rings):
        for longitude in range(segments):
            low_left, low_right = grid[ring][longitude], grid[ring][longitude + 1]
            high_left, high_right = grid[ring + 1][longitude], grid[ring + 1][longitude + 1]
            if ring > 0:  # Degenerate at the south pole, where the low row collapses.
                geometry.add_triangle(low_left, high_right, low_right, True)
            if ring + 1 < rings:  # Degenerate at the north pole.
                geometry.add_triangle(low_left, high_left, high_right, True)


def build_geometry(scene):
    geometry = Geometry()
    for primitive in scene.primitives:
        _emit_primitive(geometry, primitive)
    return geometry


def split_by_emission(geometry):
    """Two index/vertex sets: shaded surfaces and pure emitters.

    Cycles needs them as separate objects because one carries a Diffuse BSDF and
    the other an Emission shader. The OpenGL pass keeps a single mesh and
    branches on the per-vertex emission attribute instead.
    """
    parts = []
    for emissive in (False, True):
        remap = {}
        positions, normals, colors, triangles, smooth = [], [], [], [], []
        for index, (i0, i1, i2) in enumerate(geometry.triangles):
            is_emissive = geometry.emissions[i0] > 0.0
            if is_emissive != emissive:
                continue
            mapped = []
            for source in (i0, i1, i2):
                if source not in remap:
                    remap[source] = len(positions)
                    positions.append(geometry.positions[source])
                    normals.append(geometry.normals[source])
                    # Emitters carry radiance (albedo * emit); surfaces carry albedo.
                    scale = geometry.emissions[source] if emissive else 1.0
                    colors.append(tuple(channel * scale for channel in geometry.colors[source]))
                    mapped.append(remap[source])
                else:
                    mapped.append(remap[source])
            triangles.append(tuple(mapped))
            smooth.append(geometry.triangle_smooth[index])
        parts.append({"positions": positions, "normals": normals, "colors": colors,
                      "triangles": triangles, "smooth": smooth})
    return parts[0], parts[1]


def summary(scene, geometry):
    """Small set of numbers the C++ and Python builders must agree on."""
    low, high = geometry.bounds()
    return {
        "primitives": len(scene.primitives),
        "vertices": len(geometry.positions),
        "triangles": len(geometry.triangles),
        "emissive_triangles": sum(1 for tri in geometry.triangles if geometry.emissions[tri[0]] > 0.0),
        "smooth_triangles": sum(1 for flag in geometry.triangle_smooth if flag),
        "bounds_min": low,
        "bounds_max": high,
        "position_sum": tuple(sum(p[axis] for p in geometry.positions) for axis in range(3)),
        "area_sum": geometry.triangle_area_sum(),
    }
