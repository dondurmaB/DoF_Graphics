"""Shared toolkit for the scene builders.

Extracted from `build_alley.py` when a second scene (`build_cafe.py`) needed the
same machinery: the deterministic RNG, the text writer, the local-frame group
helper and the a-to-b box helper. Nothing here knows about any particular
scene, so a new one only has to describe its own props and layout.

The writer emits the format that `scene_loader.py` and `src/SceneFile.cpp` both
parse, so its output is the single description both renderers load.
"""
import math


class Rng:
    """Own LCG so a generated scene file does not depend on a Python version.

    `random` has changed its internals between releases; the checked-in scene
    file has to be byte-identical when regenerated on someone else's machine,
    or every cross-language test number and every reference render's digest
    would drift for no reason.
    """

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def unit(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296.0

    def range(self, low, high):
        return low + (high - low) * self.unit()

    def chance(self, probability):
        return self.unit() < probability

    def pick(self, items):
        return items[min(int(self.unit() * len(items)), len(items) - 1)]

    def jitter(self, color, amount=0.06, hue=False):
        """Nudge an albedo. Identical surfaces read as fake.

        Brightness only by default: one factor applied to all three channels, so
        the material gets lighter or darker without changing colour. The first
        version varied each channel independently, which shifts hue, and across
        a thousand floor tiles that reads as a patchwork of pink and green
        plastic rather than one batch of tiles with slight variation.

        Pass hue=True where real colour variety is wanted (bottle glass, leaves).
        """
        if hue:
            return tuple(max(0.0, min(1.0, channel * (1.0 + self.range(-amount, amount))))
                         for channel in color)
        factor = 1.0 + self.range(-amount, amount)
        return tuple(max(0.0, min(1.0, channel * factor)) for channel in color)


def number(value):
    """Trim a float for the scene file: 4 decimals, no trailing zeros, no '-0'."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


class SceneWriter:
    def __init__(self):
        self.lines = []
        self.count = 0

    def comment(self, text=""):
        self.lines.append(f"# {text}".rstrip())

    def section(self, title):
        """A `# --- title ---` line. tools/scene/scene_report.py groups on these."""
        self.lines.append(f"# --- {title} " + "-" * max(4, 74 - len(title)))

    def raw(self, line):
        self.lines.append(line)

    def _primitive(self, kind, pos, size, rot, rgb, emit, extra=""):
        parts = [kind, "pos", *(number(v) for v in pos), "size", *(number(v) for v in size)]
        # Defaults are omitted to keep the file readable; the loaders supply them.
        if any(abs(value) > 1e-9 for value in rot):
            parts += ["rot", *(number(v) for v in rot)]
        parts += ["rgb", *(number(v) for v in rgb)]
        if emit > 0.0:
            parts += ["emit", number(emit)]
        if extra:
            parts.append(extra)
        self.lines.append(" ".join(parts))
        self.count += 1

    def box(self, pos, size, rgb, rot=(0.0, 0.0, 0.0), emit=0.0):
        self._primitive("box", pos, size, rot, rgb, emit)

    def cyl(self, pos, size, rgb, rot=(0.0, 0.0, 0.0), emit=0.0, seg=16, smooth=True,
            taper=1.0):
        extra = f"seg {seg}" + ("" if smooth else " smooth 0")
        # Omitted at the default so existing scene files are unchanged.
        if abs(taper - 1.0) > 1e-9:
            extra += f" taper {number(taper)}"
        self._primitive("cyl", pos, size, rot, rgb, emit, extra)

    def sph(self, pos, size, rgb, rot=(0.0, 0.0, 0.0), emit=0.0, seg=12):
        self._primitive("sph", pos, size, rot, rgb, emit, f"seg {seg}")

    def text(self):
        return "\n".join(self.lines) + "\n"


def rotate_y(offset, degrees):
    angle = math.radians(degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x, y, z = offset
    return (cos_a * x + sin_a * z, y, -sin_a * x + cos_a * z)


class Group:
    """Places child primitives in a prop's local frame, rotated about Y only.

    Individual primitives can carry any rotation; groups are Y-only because the
    scene format gives each primitive one Euler triple, and composing a tilted
    group with a tilted child would need a matrix the file cannot express.
    """

    def __init__(self, writer, origin, yaw=0.0):
        self.writer = writer
        self.origin = origin
        self.yaw = yaw

    def _world(self, offset):
        turned = rotate_y(offset, self.yaw)
        return tuple(self.origin[k] + turned[k] for k in range(3))

    def box(self, offset, size, rgb, emit=0.0, extra_yaw=0.0):
        self.writer.box(self._world(offset), size, rgb,
                        rot=(0.0, self.yaw + extra_yaw, 0.0), emit=emit)

    def cyl(self, offset, size, rgb, emit=0.0, rot=None, seg=16, smooth=True, taper=1.0):
        self.writer.cyl(self._world(offset), size, rgb, rot=rot or (0.0, self.yaw, 0.0),
                        emit=emit, seg=seg, smooth=smooth, taper=taper)

    def sph(self, offset, size, rgb, emit=0.0, seg=12):
        self.writer.sph(self._world(offset), size, rgb, emit=emit, seg=seg)


def segment(writer, a, b, thickness, color, emit=0.0):
    """One box spanning a to b, length along its local +Z before rotation.

    With the scene format's R = Ry(yaw) * Rx(pitch) * Rz convention,
    R * (0,0,1) = (sin(yaw)cos(pitch), -sin(pitch), cos(yaw)cos(pitch)),
    so pitch takes a minus sign. Getting that backwards makes a sagging cable
    zigzag, which is exactly how this was caught in the CPU preview.
    """
    delta = tuple(b[k] - a[k] for k in range(3))
    length = math.sqrt(sum(component * component for component in delta))
    if length < 1e-6:
        return
    mid = tuple((a[k] + b[k]) * 0.5 for k in range(3))
    yaw = math.degrees(math.atan2(delta[0], delta[2]))
    pitch = -math.degrees(math.asin(max(-1.0, min(1.0, delta[1] / length))))
    writer.box(mid, (thickness, thickness, length), color, rot=(pitch, yaw, 0.0), emit=emit)


def tube(writer, a, b, diameter, color, seg=16, emit=0.0):
    """A cylinder spanning a to b: round stock, for legs, rails and stems.

    `segment` makes a BOX between two points, which is right for a cable or a
    square-section rail but wrong for a chair leg or a table rail, and square
    legs are a large part of what makes a render read as a game asset. The
    rotation is derived rather than guessed: with R = Ry(yaw) * Rx(pitch) * Rz,
    the cylinder's local +Y axis maps to

        (sin(yaw) sin(pitch), cos(pitch), cos(yaw) sin(pitch))

    so for a unit direction d, pitch = acos(d.y) and yaw = atan2(d.x, d.z).
    """
    delta = tuple(b[k] - a[k] for k in range(3))
    length = math.sqrt(sum(component * component for component in delta))
    if length < 1e-6:
        return
    direction = tuple(component / length for component in delta)
    mid = tuple((a[k] + b[k]) * 0.5 for k in range(3))
    pitch = math.degrees(math.acos(max(-1.0, min(1.0, direction[1]))))
    yaw = math.degrees(math.atan2(direction[0], direction[2]))
    writer.cyl(mid, (diameter, length, diameter), color,
               rot=(pitch, yaw, 0.0), emit=emit, seg=seg)


def dome(writer, center, radius, height, color, rings=14, seg=32, emit=0.0):
    """A hemisphere-ish dome sitting ON `center`, built from stacked rings.

    Used instead of a sphere for anything that rests on a surface. A full
    sphere placed at a surface sinks half of itself through it, which is how the
    pastry case ended up looking like a beach ball floating through the counter.
    """
    for ring in range(rings):
        lower = ring / rings
        upper = (ring + 1) / rings
        middle = 0.5 * (lower + upper)
        # Circular profile: diameter falls off as sqrt(1 - t^2).
        diameter = 2.0 * radius * math.sqrt(max(0.0, 1.0 - middle * middle))
        writer.cyl((center[0], center[1] + height * middle, center[2]),
                   (diameter, height / rings + 0.002, diameter), color,
                   emit=emit, seg=seg)


def leaf(writer, center, length, width, color, rot, seg=10):
    """A rounded leaf: a flattened elliptical disc rather than a flat box.

    Leaves cut from boxes read as folded paper, which is most of why the plants
    looked like origami.
    """
    writer.cyl(center, (width, 0.007, length), color, rot=rot, seg=seg)


def frame_half_height(focal_length_mm, sensor_height_mm, depth_m):
    """Visible half-height in meters at a given view depth.

    Every layout decision in a scene builder comes back to this: at 85 mm on a
    24 mm sensor the frame is only 0.14 * depth tall, so a prop 1.5 m away has
    about 42 cm of vertical room before it leaves the shot.
    """
    return 0.5 * sensor_height_mm / focal_length_mm * depth_m
