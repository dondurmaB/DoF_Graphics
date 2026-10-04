"""Write `scene/alley.scene`: the back-alley scene both renderers read.

Run it after editing anything below:

    python tools/scene/build_alley.py

Why a generator instead of hand-written geometry: the scene is ~1300 primitives
(brick relief, crates, railings, hanging bulbs), which is far past what is
readable as a literal list in `main.cpp`, but the OpenGL pass and the Cycles
reference must agree on every one of them. So the layout logic lives here once,
its output is the checked-in `scene/alley.scene`, and both renderers just load
that file. The file is plain text: small edits can be made there directly, and
the renderer reloads it with the `L` key without a rebuild.

Framing notes (why things sit where they do). The default camera is at
(0, 0, 5) looking down -Z with a 50 mm lens on a 24 mm sensor, so the vertical
half-angle is 13.5 degrees and the visible half-extent at view depth d is
0.24 * d meters:

    depth   1 m -> +/-0.24 m      depth  8 m -> +/-1.92 m
    depth   2 m -> +/-0.48 m      depth 13 m -> +/-3.12 m  (alley walls enter)
    depth   5 m -> +/-1.20 m      depth 35 m -> +/-8.40 m  (far facade fills)

Two consequences drive the composition: the ground at y = -1.55 only enters the
frame past 6.5 m of depth, and anything closer than ~4 m has to sit near eye
height (y ~ 0) to be seen at all. That is why the near field is hanging bulbs,
a cross pipe and tall crate stacks rather than props on the floor, and why the
brick detail is concentrated where the walls are actually visible.

The hanging bulb string is deliberate: it repeats one small bright object at
1 m, 2 m, 4 m, 7 m, 11 m and 17 m of depth, so a single frame shows the circle
of confusion growing with distance from the focus plane. Small bright emitters
are also the case where a CoC gather and a path-traced lens disagree most, which
is the comparison stage 2 exists to produce.
"""

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_loader  # noqa: E402  (path set above so this also runs from the repo root)

GROUND_Y = -1.55  # Floor surface; the camera at y = 0 is a 1.55 m eye height.
WALL_X = 3.2  # Inner faces of the two alley walls.
WALL_TOP = 5.6
Z_BACK = 6.5  # Behind the camera: only matters for the shadow map and wide lenses.
Z_END = -30.0  # Where the alley meets the far building.
FACADE_Z = -32.0

# Linear albedos, not sRGB. Dark values are intentional: asphalt really is
# ~0.04 and brick ~0.1, and the sun term below divides by pi like Cycles does.
ASPHALT = (0.036, 0.035, 0.037)
ASPHALT_WET = (0.017, 0.017, 0.020)
CONCRETE = (0.185, 0.180, 0.165)
CONCRETE_DARK = (0.105, 0.103, 0.097)
BRICK = (0.115, 0.045, 0.032)
BRICK_LIGHT = (0.150, 0.065, 0.044)
BRICK_DARK = (0.082, 0.034, 0.026)
MORTAR = (0.150, 0.142, 0.130)
PLASTER = (0.175, 0.165, 0.150)
WOOD = (0.105, 0.062, 0.030)
WOOD_PALE = (0.150, 0.098, 0.052)
RUST = (0.086, 0.032, 0.016)
STEEL = (0.082, 0.086, 0.094)
STEEL_DARK = (0.042, 0.044, 0.050)
PAINT_GREEN = (0.020, 0.055, 0.034)
PAINT_BLUE = (0.020, 0.034, 0.068)
PAINT_RED = (0.090, 0.021, 0.018)
GLASS = (0.010, 0.012, 0.016)
PAPER = (0.340, 0.330, 0.300)
LINE_PAINT = (0.260, 0.220, 0.095)
TEAPOT_CRATE = (0.125, 0.078, 0.040)

BULB = (1.0, 0.70, 0.40)
BULB_EMIT = 26.0
LAMP = (1.0, 0.82, 0.56)
LAMP_EMIT = 16.0
WINDOW_WARM = (1.0, 0.78, 0.50)
WINDOW_COOL = (0.62, 0.80, 1.0)
NEON_MAGENTA = (1.0, 0.24, 0.66)
NEON_CYAN = (0.28, 0.88, 1.0)


class Rng:
    """Own LCG so the generated file does not depend on a Python version."""

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


def number(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


class SceneWriter:
    def __init__(self):
        self.lines = []
        self.count = 0

    def comment(self, text=""):
        self.lines.append(f"# {text}".rstrip())

    def raw(self, line):
        self.lines.append(line)

    def _primitive(self, kind, pos, size, rot, rgb, emit, extra=""):
        parts = [kind, "pos", *(number(v) for v in pos), "size", *(number(v) for v in size)]
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

    def cyl(self, pos, size, rgb, rot=(0.0, 0.0, 0.0), emit=0.0, seg=16, smooth=True):
        extra = f"seg {seg}" + ("" if smooth else " smooth 0")
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
    """Places child boxes in a prop's local frame, rotated about Y only.

    Individual primitives can carry any rotation; groups are Y-only because the
    scene format gives each primitive one Euler triple, and composing a tilted
    group with a tilted child would need a matrix the file cannot express.
    """

    def __init__(self, writer, origin, yaw=0.0):
        self.writer = writer
        self.origin = origin
        self.yaw = yaw

    def box(self, offset, size, rgb, emit=0.0, extra_yaw=0.0):
        world = tuple(self.origin[k] + rotate_y(offset, self.yaw)[k] for k in range(3))
        self.writer.box(world, size, rgb, rot=(0.0, self.yaw + extra_yaw, 0.0), emit=emit)

    def cyl(self, offset, size, rgb, emit=0.0, rot=None, seg=16, smooth=True):
        world = tuple(self.origin[k] + rotate_y(offset, self.yaw)[k] for k in range(3))
        self.writer.cyl(world, size, rgb, rot=rot or (0.0, self.yaw, 0.0), emit=emit,
                        seg=seg, smooth=smooth)

    def sph(self, offset, size, rgb, emit=0.0, seg=12):
        world = tuple(self.origin[k] + rotate_y(offset, self.yaw)[k] for k in range(3))
        self.writer.sph(world, size, rgb, emit=emit, seg=seg)


# ----------------------------------------------------------------------------
# Props
# ----------------------------------------------------------------------------

def crate(writer, center, size, yaw, rng, color=WOOD):
    """Panelled wooden crate: 6 thin faces plus corner posts, so edges catch light."""
    width, height, depth = size
    plank = 0.035
    group = Group(writer, center, yaw)
    shade = rng.range(0.85, 1.15)
    body = tuple(channel * shade for channel in color)
    edge = tuple(channel * shade * 1.35 for channel in color)
    group.box((0.0, height * 0.5 - plank * 0.5, 0.0), (width, plank, depth), edge)
    group.box((0.0, -height * 0.5 + plank * 0.5, 0.0), (width, plank, depth), body)
    for sign in (-1.0, 1.0):
        group.box((sign * (width * 0.5 - plank * 0.5), 0.0, 0.0), (plank, height, depth), body)
        group.box((0.0, 0.0, sign * (depth * 0.5 - plank * 0.5)), (width, height, plank), body)
    for x_sign in (-1.0, 1.0):
        for z_sign in (-1.0, 1.0):
            group.box((x_sign * (width * 0.5 - 0.02), 0.0, z_sign * (depth * 0.5 - 0.02)),
                      (0.06, height * 1.01, 0.06), edge)
    # One horizontal batten per visible side; reads as a crate rather than a box.
    for sign in (-1.0, 1.0):
        group.box((0.0, height * 0.12, sign * (depth * 0.5 + 0.005)), (width * 0.98, 0.05, 0.02), edge)


def pallet(writer, center, yaw, rng):
    group = Group(writer, center, yaw)
    pale = tuple(channel * rng.range(0.9, 1.2) for channel in WOOD_PALE)
    for offset in (-0.45, 0.0, 0.45):
        group.box((offset, -0.06, 0.0), (0.1, 0.09, 1.1), pale)
    for index in range(6):
        group.box((0.0, 0.01, -0.5 + index * 0.2), (1.1, 0.03, 0.12), pale)


def barrel(writer, center, radius, height, color, rng, yaw=0.0):
    group = Group(writer, center, yaw)
    body = tuple(channel * rng.range(0.85, 1.15) for channel in color)
    group.cyl((0.0, 0.0, 0.0), (radius * 2, height, radius * 2), body, seg=20)
    for offset in (-height * 0.28, height * 0.28):
        group.cyl((0.0, offset, 0.0), (radius * 2.08, height * 0.06, radius * 2.08), RUST, seg=20)
    group.cyl((0.0, height * 0.5 + 0.01, 0.0), (radius * 1.9, 0.03, radius * 1.9), RUST, seg=20)


def tire(writer, center, radius, yaw, rng):
    Group(writer, center, yaw).cyl((0.0, 0.0, 0.0), (radius * 2, 0.22, radius * 2),
                                  tuple(c * rng.range(0.8, 1.1) for c in STEEL_DARK), seg=20)


def dumpster(writer, center, yaw, color):
    group = Group(writer, center, yaw)
    width, height, depth = 2.0, 1.15, 1.0
    group.box((0.0, 0.0, 0.0), (width, height, depth), color)
    group.box((0.0, height * 0.5 + 0.03, 0.0), (width * 1.04, 0.06, depth * 1.04),
              tuple(c * 1.2 for c in color))
    # Lids left half-open; the gap is a strong shadow-map test.
    group.box((-width * 0.26, height * 0.5 + 0.28, -0.12), (width * 0.46, 0.05, depth * 0.9),
              tuple(c * 1.1 for c in color))
    for sign in (-1.0, 1.0):
        group.box((sign * (width * 0.5 + 0.03), -height * 0.2, 0.0), (0.06, 0.5, depth * 0.8), STEEL_DARK)
    for x_sign in (-1.0, 1.0):
        for z_sign in (-1.0, 1.0):
            group.cyl((x_sign * (width * 0.42), -height * 0.5 - 0.09, z_sign * (depth * 0.34)),
                      (0.18, 0.09, 0.18), STEEL_DARK, seg=12)


def window(writer, wall_x, z, sill_y, rng, lit, cool=False):
    """Recessed window: dark reveal, frame, mullions, optional lit pane."""
    inward = -1.0 if wall_x > 0.0 else 1.0
    width, height = 0.9, 1.3
    face = wall_x + inward * 0.02
    writer.box((face + inward * 0.09, sill_y + height * 0.5, z), (0.2, height, width), CONCRETE_DARK)
    if lit:
        pane = WINDOW_COOL if cool else WINDOW_WARM
        writer.box((face + inward * 0.05, sill_y + height * 0.5, z), (0.03, height * 0.92, width * 0.92),
                   pane, emit=rng.range(2.5, 6.0))
    else:
        writer.box((face + inward * 0.05, sill_y + height * 0.5, z), (0.03, height * 0.92, width * 0.92), GLASS)
    frame = tuple(c * rng.range(0.8, 1.1) for c in PAINT_BLUE)
    writer.box((face, sill_y + height, z), (0.12, 0.09, width + 0.16), CONCRETE)
    writer.box((face, sill_y - 0.04, z), (0.16, 0.08, width + 0.2), CONCRETE)
    for sign in (-1.0, 1.0):
        writer.box((face, sill_y + height * 0.5, z + sign * width * 0.5), (0.08, height, 0.08), frame)
    writer.box((face, sill_y + height * 0.5, z), (0.06, height, 0.05), frame)
    writer.box((face, sill_y + height * 0.62, z), (0.06, 0.05, width), frame)


def steel_door(writer, wall_x, z, rng):
    inward = -1.0 if wall_x > 0.0 else 1.0
    face = wall_x + inward * 0.02
    writer.box((face + inward * 0.06, GROUND_Y + 1.05, z), (0.14, 2.1, 0.95), CONCRETE_DARK)
    writer.box((face + inward * 0.01, GROUND_Y + 1.05, z), (0.06, 2.05, 0.9),
               tuple(c * rng.range(0.9, 1.1) for c in PAINT_GREEN))
    writer.box((face - inward * 0.02, GROUND_Y + 1.0, z + 0.35), (0.05, 0.06, 0.16), STEEL)
    writer.box((face, GROUND_Y + 2.2, z), (0.3, 0.06, 1.3), STEEL_DARK)  # Lintel canopy.
    writer.box((face + inward * 0.1, GROUND_Y + 0.06, z), (0.5, 0.12, 1.1), CONCRETE)  # Step.


def railing(group, length, height, posts, color, bar=0.035):
    """Vertical bars plus two rails: thin geometry, on purpose."""
    group.box((0.0, height, 0.0), (length, bar * 1.6, bar * 1.6), color)
    group.box((0.0, height * 0.52, 0.0), (length, bar, bar), color)
    for index in range(posts):
        x = -length * 0.5 + length * index / max(1, posts - 1)
        group.box((x, height * 0.5, 0.0), (bar, height, bar), color)


def fire_escape(writer, wall_x, z_center, rng):
    inward = -1.0 if wall_x > 0.0 else 1.0
    for level, base_y in enumerate((GROUND_Y + 2.6, GROUND_Y + 5.0)):
        origin = (wall_x + inward * 0.85, base_y, z_center + level * 0.4)
        group = Group(writer, origin, 0.0)
        group.box((0.0, 0.0, 0.0), (1.7, 0.08, 2.8), STEEL_DARK)
        for index in range(9):  # Grating slats.
            group.box((0.0, 0.05, -1.3 + index * 0.325), (1.66, 0.03, 0.12), STEEL)
        outer = Group(writer, (origin[0] - inward * 0.85, origin[1], origin[2]), 90.0)
        railing(outer, 2.8, 0.95, 8, RUST)
        for z_sign in (-1.0, 1.0):
            side = Group(writer, (origin[0], origin[1], origin[2] + z_sign * 1.4), 0.0)
            railing(side, 1.7, 0.95, 5, RUST)
        for x_sign in (-1.0, 1.0):
            group.cyl((x_sign * 0.7, -1.2, -1.3), (0.07, 2.4, 0.07), RUST, seg=10)
    # Stair flight between the two platforms.
    for step in range(9):
        y = GROUND_Y + 2.75 + step * 0.26
        z = z_center + 1.5 + step * 0.26
        writer.box((wall_x + inward * 0.75, y, z), (1.3, 0.05, 0.24), STEEL)
    for x_sign in (-1.0, 1.0):
        x = wall_x + inward * 0.75 + x_sign * 0.66
        segment(writer, (x, GROUND_Y + 2.7, z_center + 1.4), (x, GROUND_Y + 5.0, z_center + 3.7),
                0.1, RUST)
        segment(writer, (x, GROUND_Y + 3.55, z_center + 1.4), (x, GROUND_Y + 5.85, z_center + 3.7),
                0.05, RUST)


def pipe_run(writer, wall_x, z, y_low, y_high, radius, rng):
    inward = -1.0 if wall_x > 0.0 else 1.0
    x = wall_x + inward * (radius + 0.06)
    writer.cyl((x, (y_low + y_high) * 0.5, z), (radius * 2, y_high - y_low, radius * 2),
               tuple(c * rng.range(0.8, 1.2) for c in RUST), seg=12)
    brackets = max(2, int((y_high - y_low) / 1.6))
    for index in range(brackets):
        y = y_low + (y_high - y_low) * (index + 0.5) / brackets
        writer.box((wall_x + inward * (radius * 0.5 + 0.03), y, z),
                   (radius + 0.1, 0.05, radius * 2.6), STEEL_DARK)


def wall(writer, wall_x, rng):
    """Brick wall: base slab, mortar courses, scattered protruding bricks.

    No image textures anywhere in the project, so surface detail has to be real
    geometry. That also keeps the Cycles reference exact: a protruding brick
    casts a real shadow in both renderers, where a normal map would not.
    """
    inward = -1.0 if wall_x > 0.0 else 1.0
    z_center = (Z_BACK + Z_END) * 0.5
    z_length = Z_BACK - Z_END
    writer.box((wall_x - inward * 0.35, GROUND_Y + WALL_TOP * 0.5 - 0.2, z_center),
               (0.7, WALL_TOP + 0.4, z_length), BRICK)
    # Coping and a plaster band, so the wall is not one flat colour.
    writer.box((wall_x - inward * 0.3, GROUND_Y + WALL_TOP + 0.05, z_center),
               (0.85, 0.18, z_length), CONCRETE)
    writer.box((wall_x + inward * 0.005, GROUND_Y + 0.55, z_center), (0.05, 1.1, z_length), CONCRETE_DARK)

    course_height = 0.088
    courses = int((WALL_TOP - 1.2) / course_height)
    for index in range(courses):
        y = GROUND_Y + 1.15 + index * course_height
        writer.box((wall_x + inward * 0.008, y, z_center), (0.02, 0.012, z_length), MORTAR)

    # Individual bricks only where the walls are actually in frame (past ~10 m).
    brick_length, brick_gap = 0.235, 0.012
    columns = int((Z_BACK - 4.0 - Z_END) / (brick_length + brick_gap))
    for row in range(courses):
        y = GROUND_Y + 1.15 + row * course_height + course_height * 0.5
        offset = 0.5 * (brick_length + brick_gap) if row % 2 else 0.0
        for column in range(columns):
            if not rng.chance(0.11):
                continue
            z = Z_BACK - 4.0 - offset - column * (brick_length + brick_gap)
            depth = rng.range(0.012, 0.03)
            writer.box((wall_x + inward * (0.01 - depth * 0.5), y, z),
                       (depth, course_height - 0.014, brick_length),
                       rng.pick((BRICK_LIGHT, BRICK_DARK, BRICK)))
    # Damp patches and graffiti-sized colour blocks: flat, cheap, breaks up tone.
    for _ in range(14):
        z = rng.range(Z_END + 1.0, Z_BACK - 3.0)
        y = rng.range(GROUND_Y + 0.3, GROUND_Y + 3.2)
        writer.box((wall_x + inward * 0.004, y, z),
                   (0.01, rng.range(0.3, 1.4), rng.range(0.4, 2.2)),
                   rng.pick((BRICK_DARK, ASPHALT_WET, PLASTER, CONCRETE_DARK)))


def ground(writer, rng):
    z_center = (Z_BACK + Z_END) * 0.5
    z_length = Z_BACK - Z_END + 6.0
    writer.box((0.0, GROUND_Y - 0.5, z_center), (2 * WALL_X + 2.0, 1.0, z_length), ASPHALT)
    for sign in (-1.0, 1.0):
        writer.box((sign * (WALL_X - 0.28), GROUND_Y + 0.05, z_center), (0.56, 0.1, z_length), CONCRETE_DARK)
        writer.box((sign * (WALL_X - 0.55), GROUND_Y + 0.035, z_center), (0.05, 0.07, z_length), CONCRETE)
    # Centre line, dashed: gives the eye a strong perspective cue down the alley.
    for index in range(14):
        z = 2.0 - index * 2.3
        writer.box((0.15, GROUND_Y + 0.006, z), (0.12, 0.012, 1.15), LINE_PAINT)
    # Wet patches; nearly black albedo reads as standing water under the sun term.
    for _ in range(9):
        writer.box((rng.range(-2.4, 2.4), GROUND_Y + 0.004, rng.range(Z_END + 2.0, 3.0)),
                   (rng.range(0.7, 2.6), 0.008, rng.range(0.8, 3.4)), ASPHALT_WET)
    writer.cyl((1.15, GROUND_Y + 0.012, -3.4), (0.68, 0.024, 0.68), STEEL_DARK, seg=20)
    writer.cyl((1.15, GROUND_Y + 0.02, -3.4), (0.5, 0.024, 0.5), STEEL, seg=20)
    # Drain grate.
    writer.box((-2.2, GROUND_Y + 0.02, -9.0), (0.5, 0.05, 0.8), STEEL_DARK)
    for index in range(6):
        writer.box((-2.2, GROUND_Y + 0.045, -9.35 + index * 0.14), (0.44, 0.02, 0.06), STEEL)
    # Loose debris: bricks, boards, paper.
    for _ in range(26):
        z = rng.range(Z_END + 1.0, 4.0)
        x = rng.range(-2.7, 2.7)
        kind = rng.unit()
        if kind < 0.45:
            writer.box((x, GROUND_Y + 0.035, z), (0.24, 0.07, 0.11),
                       rng.pick((BRICK, BRICK_DARK)), rot=(0.0, rng.range(0, 180), 0.0))
        elif kind < 0.75:
            writer.box((x, GROUND_Y + 0.02, z), (rng.range(0.4, 1.3), 0.03, 0.12), WOOD,
                       rot=(0.0, rng.range(0, 180), 0.0))
        else:
            writer.box((x, GROUND_Y + 0.008, z), (rng.range(0.1, 0.3), 0.004, rng.range(0.1, 0.3)),
                       PAPER, rot=(0.0, rng.range(0, 180), 0.0))


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


def festoon(writer, z, y_anchor, sag, bulbs, rng, tilt=0.0, x_shift=0.0):
    """Bulb string across the alley, wall to wall.

    Strung across rather than along the alley on purpose: at a 50 mm lens the
    visible half-width at depth d is only 0.24*d m, so a string running away
    from the camera piles all its bulbs into a vertical line near the vanishing
    point. Across, each string puts several bulbs at one known depth, spread
    over the frame, and the set of strings then samples depth from ~1.5 m to
    ~23 m. Same object, same size, many circles of confusion, one frame.
    """
    x_start, x_end = -WALL_X + 0.1 + x_shift, WALL_X - 0.1 + x_shift

    def cable_point(t):
        x = x_start + (x_end - x_start) * t
        # Quadratic sag, not a true catenary: close enough visually, and both
        # renderers read the same sampled points out of the scene file anyway.
        y = y_anchor + tilt * (t - 0.5) - sag * 4.0 * t * (1.0 - t)
        return (x, y, z + 0.12 * math.sin(t * 2.7))

    steps = max(8, bulbs * 2)
    for index in range(steps):
        segment(writer, cable_point(index / steps), cable_point((index + 1) / steps),
                0.018, STEEL_DARK)
    for sign in (0.0, 1.0):  # Wall brackets at both ends.
        anchor = cable_point(sign)
        inward = 1.0 if sign == 0.0 else -1.0
        writer.box((anchor[0] - inward * 0.12, anchor[1], anchor[2]), (0.3, 0.06, 0.06), STEEL_DARK)
    for index in range(bulbs):
        t = (index + 0.5) / bulbs
        anchor = cable_point(t)
        stem = rng.range(0.13, 0.24)
        writer.box((anchor[0], anchor[1] - stem * 0.5, anchor[2]), (0.012, stem, 0.012), STEEL_DARK)
        writer.box((anchor[0], anchor[1] - stem - 0.025, anchor[2]), (0.05, 0.06, 0.05), STEEL_DARK)
        writer.sph((anchor[0], anchor[1] - stem - 0.1, anchor[2]), (0.095, 0.115, 0.095),
                   BULB, emit=BULB_EMIT * rng.range(0.8, 1.2), seg=12)


def lamp_post(writer, x, z, rng):
    inward = -1.0 if x > 0.0 else 1.0
    writer.cyl((x, GROUND_Y + 1.7, z), (0.1, 3.4, 0.1), STEEL_DARK, seg=12)
    writer.cyl((x, GROUND_Y + 0.1, z), (0.28, 0.2, 0.28), CONCRETE_DARK, seg=12)
    writer.cyl((x + inward * 0.35, GROUND_Y + 3.35, z), (0.7, 0.07, 0.07), STEEL_DARK,
               rot=(0.0, 0.0, 90.0), seg=10)
    writer.box((x + inward * 0.68, GROUND_Y + 3.24, z), (0.42, 0.1, 0.3), STEEL_DARK)
    writer.box((x + inward * 0.68, GROUND_Y + 3.15, z), (0.34, 0.08, 0.24), LAMP, emit=LAMP_EMIT)


def facade(writer, rng):
    """Building closing the alley, ~37 m out: a grid of lit windows.

    Every window is a small bright rectangle at the same large depth, so the
    background bokeh is a repeated, measurable shape rather than a smooth wash.
    """
    writer.box((0.0, GROUND_Y + 6.0, FACADE_Z - 0.6), (26.0, 14.0, 1.2), BRICK_DARK)
    writer.box((0.0, GROUND_Y + 0.9, FACADE_Z - 0.02), (26.0, 1.8, 0.1), CONCRETE_DARK)
    for row in range(6):
        y = GROUND_Y + 2.4 + row * 1.75
        # Floor band: stops the window grid reading as a spreadsheet.
        writer.box((0.0, y - 0.85, FACADE_Z + 0.04), (26.0, 0.22, 0.14), CONCRETE_DARK)
        for column in range(9):
            x = -7.2 + column * 1.8
            writer.box((x, y, FACADE_Z + 0.06), (1.0, 1.25, 0.12), CONCRETE_DARK)
            if rng.chance(0.5):
                cool = rng.chance(0.35)
                # Kept dimmer than the bulbs: these are 37 m away and should
                # read as soft background bokeh, not as blown-out white tiles.
                writer.box((x, y, FACADE_Z + 0.13), (0.86, 1.1, 0.04),
                           WINDOW_COOL if cool else WINDOW_WARM, emit=rng.range(0.9, 3.2))
                if rng.chance(0.4):  # Blind or furniture blocking part of the pane.
                    writer.box((x, y + rng.range(0.1, 0.45), FACADE_Z + 0.15),
                               (0.86, rng.range(0.2, 0.5), 0.03), CONCRETE_DARK)
            else:
                writer.box((x, y, FACADE_Z + 0.13), (0.86, 1.1, 0.04), GLASS)
            writer.box((x, y, FACADE_Z + 0.16), (0.9, 0.05, 0.06), STEEL_DARK)
    # Silhouetted blocks further back, and a water tower outline.
    writer.box((-11.0, GROUND_Y + 9.0, FACADE_Z - 8.0), (9.0, 20.0, 8.0), CONCRETE_DARK)
    writer.box((12.0, GROUND_Y + 7.0, FACADE_Z - 11.0), (11.0, 16.0, 8.0), CONCRETE_DARK)
    writer.cyl((-9.5, GROUND_Y + 20.5, FACADE_Z - 6.0), (3.2, 3.4, 3.2), STEEL_DARK, seg=16)
    for index in range(4):
        writer.box((-10.6 + index * 0.75, GROUND_Y + 17.8, FACADE_Z - 6.0), (0.12, 2.0, 0.12), STEEL_DARK)
    for _ in range(26):
        x = rng.range(-15.0, 17.0)
        y = rng.range(GROUND_Y + 3.0, GROUND_Y + 16.0)
        z = FACADE_Z - rng.pick((7.9, 10.9))
        writer.box((x, y, z + 0.1), (0.5, 0.7, 0.04), WINDOW_WARM, emit=rng.range(1.0, 3.0))


def neon_sign(writer, wall_x, z, y):
    inward = -1.0 if wall_x > 0.0 else 1.0
    writer.box((wall_x + inward * 0.12, y, z), (0.24, 1.5, 2.4), STEEL_DARK)
    writer.box((wall_x + inward * 0.26, y + 0.35, z), (0.05, 0.42, 2.0), NEON_MAGENTA, emit=13.0)
    writer.box((wall_x + inward * 0.26, y - 0.3, z), (0.05, 0.22, 1.4), NEON_CYAN, emit=11.0)
    writer.cyl((wall_x + inward * 0.06, y, z), (0.08, 0.5, 0.08), STEEL_DARK, rot=(0.0, 0.0, 90.0), seg=8)


def hero_stack(writer, rng):
    """What the default 5 m focus distance is pointed at.

    The imported teapot (`assets/models/scene.obj`) sits at (0, -0.75, 0) with
    scale 0.1, wired in `src/main.cpp` and mirrored in `render_dof.py`. Its
    bottom is at y = -0.75, so this builds a 0.8 m platform for it to stand on.
    """
    # Just wide enough for the teapot (1.0 m across): a wider platform turns the
    # bottom quarter of the frame into one flat lit surface.
    crate(writer, (0.0, GROUND_Y + 0.4, 0.0), (1.05, 0.8, 0.9), 4.0, rng, TEAPOT_CRATE)
    pallet(writer, (0.0, GROUND_Y + 0.02, 0.05), 4.0, rng)
    # Curved silhouettes either side, tops just above the platform, so the
    # subject sits among objects instead of on a table.
    barrel(writer, (-1.18, GROUND_Y + 0.44, -0.45), 0.28, 0.88, RUST, rng, yaw=-25.0)
    barrel(writer, (1.22, GROUND_Y + 0.44, -0.7), 0.28, 0.88, PAINT_GREEN, rng, yaw=40.0)
    tire(writer, (-1.5, GROUND_Y + 0.12, 0.6), 0.33, 12.0, rng)
    # Lantern at the focus plane: a sharp bright core next to blurred ones.
    writer.box((0.82, GROUND_Y + 0.52, 0.35), (0.42, 0.42, 0.42), WOOD, rot=(0.0, -18.0, 0.0))
    writer.cyl((0.82, GROUND_Y + 0.86, 0.35), (0.16, 0.26, 0.16), STEEL_DARK, seg=12)
    writer.cyl((0.82, GROUND_Y + 0.86, 0.35), (0.13, 0.2, 0.13), LAMP, emit=9.0, seg=12)
    writer.cyl((0.82, GROUND_Y + 1.0, 0.35), (0.18, 0.05, 0.18), STEEL_DARK, seg=12)
    # A few sharp small props so "in focus" has visible high-frequency detail.
    writer.box((-0.78, GROUND_Y + 0.87, 0.28), (0.34, 0.16, 0.22), PAINT_RED, rot=(0.0, 22.0, 0.0))
    writer.box((-0.72, GROUND_Y + 0.98, 0.25), (0.2, 0.06, 0.14), STEEL, rot=(0.0, 14.0, 0.0))
    for index in range(3):
        writer.cyl((-0.3 + index * 0.16, GROUND_Y + 0.87, -0.35), (0.09, 0.16, 0.09),
                   rng.pick((PAINT_GREEN, PAINT_BLUE, RUST)), seg=12)


def build(seed=20260928):
    rng = Rng(seed)
    writer = SceneWriter()
    scene = scene_loader.Scene()

    writer.comment("dof-research alley scene, format version 1")
    writer.comment("GENERATED by tools/scene/build_alley.py -- regenerate instead of hand-editing")
    writer.comment("large edits, but single-line tweaks are fine (press L in the renderer to reload).")
    writer.comment()
    writer.comment("Read by src/SceneFile.cpp (OpenGL) and tools/raytraced_reference/render_dof.py")
    writer.comment("(Cycles). rgb is LINEAR albedo; emit > 0 means the surface emits rgb * emit")
    writer.comment("and is not shaded. 1 unit = 1 meter, +Y up, camera looks down -Z.")
    writer.raw("version 1")
    writer.comment()
    writer.comment("Camera and lens: the values main.cpp starts with and render_dof.py matches.")
    writer.raw("camera pos 0 0 5 yaw -90 pitch 0 focus 5 fnumber 1.4 lens 50 sensor 24")
    writer.comment()
    writer.comment("Dusk key light. energy is irradiance in W/m^2, as Blender's sun strength;")
    writer.comment("the raster pass divides by pi to match a Lambert diffuse BSDF.")
    writer.raw("sun dir -0.45 0.78 0.44 color 1 0.88 0.72 energy 4.2 angle 0.6")
    writer.comment("Uniform sky: also the OpenGL clear colour and Cycles' world background.")
    writer.raw("ambient color 0.42 0.52 0.72 strength 0.24")
    writer.comment()
    writer.comment("Region the shadow map has to cover. Without this the frustum would also")
    writer.comment("have to contain the distant blocks at z = -47, wasting most of its texels.")
    writer.raw("shadow lo -7 -2 -34 hi 7 8 9")

    writer.comment()
    writer.comment("--- ground -----------------------------------------------------------------")
    ground(writer, rng)

    writer.comment()
    writer.comment("--- alley walls ------------------------------------------------------------")
    for wall_x in (-WALL_X, WALL_X):
        wall(writer, wall_x, rng)

    writer.comment()
    writer.comment("--- wall fittings ----------------------------------------------------------")
    for z in (-11.0, -16.5, -22.0, -27.0):
        window(writer, -WALL_X, z, GROUND_Y + 2.3, rng, lit=rng.chance(0.6))
        window(writer, -WALL_X, z - 2.4, GROUND_Y + 4.1, rng, lit=rng.chance(0.45), cool=True)
    for z in (-13.0, -18.5, -24.5):
        window(writer, WALL_X, z, GROUND_Y + 2.5, rng, lit=rng.chance(0.7))
        window(writer, WALL_X, z - 2.6, GROUND_Y + 4.2, rng, lit=rng.chance(0.4))
    steel_door(writer, -WALL_X, -8.0, rng)
    steel_door(writer, WALL_X, -20.0, rng)
    fire_escape(writer, WALL_X, -15.0, rng)
    for x, z, low, high, radius in ((-WALL_X, -9.6, GROUND_Y, GROUND_Y + 5.2, 0.075),
                                    (-WALL_X, -19.0, GROUND_Y, GROUND_Y + 5.2, 0.05),
                                    (WALL_X, -11.5, GROUND_Y, GROUND_Y + 5.0, 0.065),
                                    (WALL_X, -25.5, GROUND_Y, GROUND_Y + 5.2, 0.09)):
        pipe_run(writer, x, z, low, high, radius, rng)
    # Cross pipes, both kept near the top of the frame so they frame the shot
    # instead of cutting it in half. The near one at 3.5 m is a thin hard edge
    # inside the near blur, which is where a gather bleeds most obviously.
    writer.cyl((0.0, 0.62, 1.5), (0.11, 6.6, 0.11), RUST, rot=(0.0, 0.0, 90.0), seg=12)
    writer.cyl((0.0, 2.55, -6.2), (0.16, 6.6, 0.16), RUST, rot=(0.0, 0.0, 90.0), seg=14)
    for sign in (-1.0, 1.0):
        writer.box((sign * (WALL_X - 0.12), 2.55, -6.2), (0.28, 0.24, 0.24), STEEL_DARK)
    neon_sign(writer, WALL_X, -21.0, GROUND_Y + 3.3)
    lamp_post(writer, -WALL_X + 0.55, -12.5, rng)
    lamp_post(writer, WALL_X - 0.55, -24.0, rng)

    writer.comment()
    writer.comment("--- hanging bulbs (main DoF subject matter) --------------------------------")
    # (z, cable height, sag, bulbs). Heights rise with distance so every string
    # sits near the top of the 27-degree frame rather than over the subject:
    # view depths are about 1.6, 5.8, 10, 16 and 23 m, against a 5 m focus.
    # (z, cable height, sag, bulbs, tilt, x shift). The nearest string is hung
    # high enough that its bulbs graze the top edge of the 27-degree frame: at
    # 2 m depth the visible half-height is only 0.48 m, so a bulb on the optical
    # axis there covers a tenth of the frame and sits directly over the subject.
    # Clipped at the top edge it still supplies the largest near-field circle of
    # confusion in the shot without hiding what is being focused on.
    for z, y_anchor, sag, bulbs, tilt, x_shift in ((3.0, 1.42, 0.52, 5, 0.10, -0.24),
                                                   (-0.8, 1.70, 0.75, 7, -0.12, 0.18),
                                                   (-5.0, 2.25, 0.80, 7, 0.14, -0.10),
                                                   (-11.0, 2.65, 0.85, 8, -0.10, 0.12),
                                                   (-18.0, 2.95, 0.90, 8, 0.08, 0.0)):
        festoon(writer, z, y_anchor, sag, bulbs, rng, tilt=tilt, x_shift=x_shift)

    writer.comment()
    writer.comment("--- focus plane, ~5 m ------------------------------------------------------")
    hero_stack(writer, rng)

    writer.comment()
    writer.comment("--- near field, 2.5-3.5 m (deliberately out of focus) ----------------------")
    # Placed so the inner faces sit just inside the frame edge: at 2.8 m the
    # visible half-width is only 0.67 m, so these intrude from the sides as
    # framing rather than blocking the subject. Anything further out than about
    # x = +/-1.2 here would be entirely outside a 50 mm frame.
    crate(writer, (-1.02, GROUND_Y + 0.42, 2.3), (0.86, 0.84, 0.8), -12.0, rng)
    crate(writer, (-0.98, GROUND_Y + 1.24, 2.22), (0.8, 0.8, 0.76), 9.0, rng)
    crate(writer, (1.06, GROUND_Y + 0.45, 2.45), (0.9, 0.9, 0.85), 16.0, rng)
    crate(writer, (1.12, GROUND_Y + 1.3, 2.38), (0.82, 0.8, 0.78), -6.0, rng)
    barrel(writer, (1.05, GROUND_Y + 2.16, 2.4), 0.26, 0.86, PAINT_BLUE, rng, yaw=12.0)
    # Wider-lens dressing: out of the 50 mm frame, still in the shadow map.
    barrel(writer, (-2.3, GROUND_Y + 0.44, 0.9), 0.28, 0.88, RUST, rng, yaw=20.0)
    pallet(writer, (-2.55, GROUND_Y + 0.55, 3.6), 78.0, rng)
    crate(writer, (2.3, GROUND_Y + 0.45, 3.4), (0.9, 0.9, 0.85), -20.0, rng)

    writer.comment()
    writer.comment("--- mid ground, 6-20 m -----------------------------------------------------")
    dumpster(writer, (-1.95, GROUND_Y + 0.6, -7.5), 6.0, PAINT_GREEN)
    dumpster(writer, (2.1, GROUND_Y + 0.6, -17.0), -4.0, PAINT_BLUE)
    crate(writer, (2.0, GROUND_Y + 0.45, -4.2), (0.9, 0.9, 0.85), -14.0, rng)
    crate(writer, (1.9, GROUND_Y + 1.3, -4.3), (0.85, 0.8, 0.8), 7.0, rng)
    crate(writer, (-2.3, GROUND_Y + 0.4, -11.0), (0.8, 0.8, 0.78), 24.0, rng)
    pallet(writer, (2.4, GROUND_Y + 0.55, -9.5), 95.0, rng)
    crate(writer, (-1.15, GROUND_Y + 0.42, -13.2), (0.84, 0.84, 0.8), 32.0, rng)
    crate(writer, (-1.05, GROUND_Y + 1.2, -13.4), (0.76, 0.72, 0.74), -9.0, rng)
    crate(writer, (1.35, GROUND_Y + 0.42, -15.5), (0.86, 0.84, 0.8), -24.0, rng)
    for index in range(3):
        pallet(writer, (0.9, GROUND_Y + 0.1 + index * 0.13, -11.0), 12.0 + index * 5.0, rng)
    pallet(writer, (-2.6, GROUND_Y + 0.06, -14.0), 8.0, rng)
    for index in range(4):
        tire(writer, (2.55, GROUND_Y + 0.12 + index * 0.2, -12.2), 0.33, index * 27.0, rng)
    for index in range(3):
        tire(writer, (-2.7, GROUND_Y + 0.12 + index * 0.2, -21.0), 0.33, index * 40.0, rng)
    for _ in range(7):
        writer.sph((rng.range(-2.6, -1.4), GROUND_Y + 0.3, rng.range(-9.5, -6.0)),
                   (rng.range(0.5, 0.75), rng.range(0.45, 0.6), rng.range(0.5, 0.75)),
                   STEEL_DARK, seg=12)
    barrel(writer, (2.45, GROUND_Y + 0.44, -6.0), 0.28, 0.88, RUST, rng, yaw=-35.0)
    barrel(writer, (-2.5, GROUND_Y + 0.44, -17.5), 0.28, 0.88, PAINT_RED, rng)
    # Hoarding with a gap: another hard silhouette across the middle distance.
    for index in range(7):
        writer.box((-2.9 + index * 0.42, GROUND_Y + 0.9, -19.5), (0.4, 1.8, 0.06),
                   tuple(c * rng.range(0.8, 1.2) for c in WOOD), rot=(0.0, rng.range(-3, 3), 0.0))
    writer.box((-0.1, GROUND_Y + 1.85, -19.5), (6.2, 0.12, 0.1), WOOD)

    writer.comment()
    writer.comment("--- far facade, ~37 m ------------------------------------------------------")
    facade(writer, rng)

    text = writer.text()
    scene = scene_loader.parse_scene(text)  # Fail here rather than in the renderer.
    return text, scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="scene/alley.scene")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--stdout", action="store_true", help="Print instead of writing the file")
    args = parser.parse_args(argv)

    text, scene = build(args.seed)
    geometry = scene_loader.build_geometry(scene)
    summary = scene_loader.summary(scene, geometry)
    if args.stdout:
        print(text, end="")
        return
    root = Path(__file__).resolve().parents[2]
    output = (root / args.output) if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(f"Wrote {output}")
    for key in ("primitives", "vertices", "triangles", "emissive_triangles", "smooth_triangles"):
        print(f"  {key}: {summary[key]}")
    print("  bounds: " + " to ".join(
        "(" + ", ".join(f"{value:.2f}" for value in corner) + ")"
        for corner in (summary["bounds_min"], summary["bounds_max"])))


if __name__ == "__main__":
    main()
