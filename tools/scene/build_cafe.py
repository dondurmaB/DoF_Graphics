"""Write `scene/cafe.scene`: the cafe interior both renderers read.

    python tools/scene/build_cafe.py

Replaces the alley as the stage-2 comparison scene. `build_alley.py` still
works and still produces `scene/alley.scene`; both are valid inputs to either
renderer, because the scene format is what they share, not the scene.

Why a cafe, and why this framing
--------------------------------
The point of the scene is to make the difference between a screen-space CoC
gather and a path-traced lens *obvious*, which needs three things at once: a
sharp subject at the focus plane, identical objects repeating away from it so
the circle of confusion can be seen growing, and small bright highlights at
many depths, because that is where the two methods disagree most.

The default camera is at (0, 0, 5) looking down -Z with a 50 mm lens on a 24 mm
sensor, focused at 1.4 m: a 27-degree vertical field of view, so the visible
half-extent at view depth d is 0.24 * d meters:

    depth  0.9 m -> +/-0.22 m      depth  5.6 m -> +/-1.34 m
    depth  1.4 m -> +/-0.34 m      depth  8.0 m -> +/-1.92 m  (bar becomes visible)
    depth  2.8 m -> +/-0.67 m      depth 15.0 m -> +/-3.60 m  (whole room fills)

The camera is pitched down 8 degrees. With no pitch at all, a surface has to sit
within about 20 cm of eye height to appear in the near field, which forces an
awkward chin-on-the-counter view; 8 degrees is what anyone photographing a cafe
table actually does, and it brings the floor into frame past about 3.2 m.

An 85 mm lens was tried first and rejected: at 1.6 m it gives a 45 cm frame, so
the cup fills the shot like a macro photograph and nothing off the optical axis
is in frame at all. 50 mm keeps a strong blur while letting the room open up
with depth, which is what makes this presentable as well as measurable.

The composition is still a long counter running *away* from the camera, because
that is what puts identical objects at known increasing distances in one frame:

    z = +4.1  glass and shaker, depth 0.9 m, deliberately blurred foreground
    z = +3.6  cup and saucer, depth 1.4 m, THE FOCUS SUBJECT
    z = +3.0  second cup, depth 2.0 m, first visible defocus
    z = +1.0  plant and cups along the counter, depth 4.0 m
    z = -5.0  end of the counter, depth 10.0 m
    z = -10.0 back wall, chalkboard and neon, depth 15 m

At 50 mm f/1.4 focused 1.4 m the CoC radius on a 1200 px frame is 0 px at the
cup, about 18 px on the foreground glass and about 30 px on the back wall. The
renderer's `K` preset (85 mm f/1.2 at 1.2 m) pushes the background to about
124 px for when the blur itself is the point.

The pendant lamps over the counter repeat one small bright object at 5.6, 8.0
and 10.8 m; the window panes on the left wall do the same at larger sizes. Small
bright emitters seen out of focus are the case where a gather and a path tracer
disagree most, so they are the measurement, not decoration.
"""

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_loader  # noqa: E402  (path set above so this also runs from the repo root)
from authoring import (Group, Rng, SceneWriter, dome, leaf, rotate_y, segment,  # noqa: E402
                       tube, SurfaceColor, recolor)

# ---------------------------------------------------------------------------
# Room dimensions, meters. The camera at y = 0 is a seated eye height of 1.2 m.
# ---------------------------------------------------------------------------
FLOOR_Y = -1.20
CEILING_Y = 1.60
WALL_LEFT_X = -2.90   # Window wall.
WALL_RIGHT_X = 2.90   # Service bar and tiled wall.
Z_NEAR = 6.00         # Behind the camera; only the shadow map sees it.
Z_BACK = -10.00       # Back wall, 15 m of depth away.

TABLE_TOP_Y = -0.42    # Cafe table, 0.78 m above the floor.
HERO_TABLE_Z = 3.50    # 1.5 m from the camera: the focus plane.
HERO_TABLE_X = 0.06

# ---------------------------------------------------------------------------
# Linear albedos, not sRGB, at roughly measured reflectances for a light cafe:
# painted plaster ~0.63, light oak ~0.24, glazed ceramic ~0.78, unglazed
# terracotta ~0.27. The first version of this palette was carried over from the
# dusk alley (oak at 0.128, a dark stain) and with a diffuse-only, direct-lit
# shading model that made every render murky regardless of the light level.
# ---------------------------------------------------------------------------
OAK = SurfaceColor((0.235, 0.140, 0.068), 0.38, 0.04)
OAK_PALE = SurfaceColor((0.340, 0.228, 0.120), 0.38, 0.04)
OAK_DARK = SurfaceColor((0.150, 0.088, 0.044), 0.42, 0.04)
WALNUT = SurfaceColor((0.105, 0.060, 0.036), 0.36, 0.04)
TILE_GREEN = SurfaceColor((0.085, 0.205, 0.162), 0.27, 0.04)
TILE_GREEN_2 = SurfaceColor((0.108, 0.245, 0.194), 0.27, 0.04)
TILE_CREAM = SurfaceColor((0.660, 0.618, 0.540), 0.24, 0.04)
GROUT = (0.300, 0.288, 0.265)
FLOOR_TILE_A = SurfaceColor((0.180, 0.172, 0.164), 0.32, 0.04)
FLOOR_TILE_B = SurfaceColor((0.420, 0.400, 0.368), 0.32, 0.04)
PLASTER = (0.630, 0.600, 0.545)
PLASTER_WARM = (0.675, 0.615, 0.525)
BRASS = SurfaceColor((0.430, 0.305, 0.125), 0.3, 0.55)
STEEL = SurfaceColor((0.300, 0.312, 0.330), 0.36, 0.6)
CHROME = SurfaceColor((0.530, 0.545, 0.562), 0.2, 0.7)
STEEL_DARK = SurfaceColor((0.105, 0.110, 0.122), 0.4, 0.045)
BLACKBOARD = (0.038, 0.044, 0.040)
CHALK = (0.730, 0.715, 0.670)
PAPER = (0.705, 0.685, 0.635)
CERAMIC = SurfaceColor((0.775, 0.760, 0.725), 0.16, 0.04)
CERAMIC_WARM = SurfaceColor((0.720, 0.640, 0.555), 0.2, 0.04)
COFFEE = (0.048, 0.024, 0.013)
CREMA = (0.245, 0.142, 0.068)
MILK = (0.790, 0.775, 0.745)
# Opaque frosted-glass approximation: matched reflection, without transmission.
# Clear refraction would require another verified term in both renderers.
GLASS_CLEAR = SurfaceColor((0.480, 0.510, 0.530), 0.34, 0.04)
GLASS_GREEN = SurfaceColor((0.115, 0.270, 0.180), 0.3, 0.04)
GLASS_AMBER = SurfaceColor((0.340, 0.190, 0.058), 0.3, 0.04)
LEAF = (0.068, 0.165, 0.058)
LEAF_PALE = (0.115, 0.250, 0.090)
TERRACOTTA = SurfaceColor((0.270, 0.122, 0.078), 0.85, 0.008)
LEATHER = SurfaceColor((0.125, 0.075, 0.050), 0.55, 0.025)
CAKE = (0.395, 0.280, 0.160)
CAKE_PINK = (0.440, 0.235, 0.240)

# Emitters. These are radiance multipliers on the albedo above, so a bulb is
# (1.0, 0.74, 0.44) * 30. Values well above 1 are the reason the renderer needs
# a float colour target: clipping them to 1 before the blur is what turns a
# bokeh highlight into a flat grey disc.
BULB = (1.00, 0.74, 0.44)
BULB_EMIT = 30.0
FILAMENT = (1.00, 0.82, 0.58)
FILAMENT_EMIT = 90.0      # Tiny and very bright: the hardest bokeh case there is.
WINDOW_DAY = (0.86, 0.92, 1.00)
WINDOW_EMIT = 11.0        # Overcast daylight seen from inside.
STRIP = (1.00, 0.88, 0.70)
STRIP_EMIT = 7.0
NEON_SIGN = (1.00, 0.30, 0.42)
NEON_EMIT = 22.0
DISPLAY = (0.55, 0.85, 1.00)
DISPLAY_EMIT = 5.0


# ---------------------------------------------------------------------------
# Small parts
# ---------------------------------------------------------------------------

def cup_and_saucer(writer, center, rng, yaw=0.0, filled=True, color=CERAMIC,
                   detail=48):
    """A 7.5 cm cup on a 14 cm saucer, built as a turned profile.

    The first version was a single cylinder with a flat disc of coffee on top,
    and it read as a bucket. Three things fix that, and they are the same three
    that make any turned object look real:

    * **Taper.** A cafe cup is wider at the rim than at the foot. Approximated
      by stacking six short cylinders along a slightly bellied profile.
    * **Wall thickness.** The rim is a visible ring, not an edge, so there is a
      proud rim band and an inner recess in a darker ceramic. Without a visible
      opening the eye reads a solid lump.
    * **Enough segments.** 48 around instead of 28: at the focus plane the cup
      is a fifth of the frame and 28 facets are individually visible.

    `detail` drops for cups on the far tables, which are a few pixels across and
    heavily defocused, so their facets can never be seen.
    """
    group = Group(writer, center, yaw)
    interior = recolor(color, (channel * 0.78 for channel in color))

    # --- Saucer: foot ring, concave well, raised rim.
    group.cyl((0.0, 0.004, 0.0), (0.082, 0.008, 0.082), color, seg=detail // 2)
    group.cyl((0.0, 0.011, 0.0), (0.138, 0.009, 0.138), color, seg=detail)
    group.cyl((0.0, 0.017, 0.0), (0.142, 0.006, 0.142), color, seg=detail)
    group.cyl((0.0, 0.0205, 0.0), (0.124, 0.005, 0.124), interior, seg=detail)
    group.cyl((0.0, 0.019, 0.0), (0.086, 0.004, 0.086), interior, seg=detail // 2)

    # --- Cup body: foot 5.2 cm, rim 7.8 cm, 7.2 cm tall, as ONE tapered
    # surface. Earlier versions stacked six and then fourteen cylinders along a
    # profile curve, and every stack shows: each ring's cap leaves a small
    # annular ledge, which reads as corduroy banding down the side of the cup.
    # A frustum has no internal seams at all and costs a third of the triangles.
    base_y, height = 0.022, 0.072
    group.cyl((0.0, base_y + height * 0.5, 0.0), (0.052, height, 0.052), color,
              seg=detail, taper=78.0 / 52.0)
    group.cyl((0.0, base_y + 0.005, 0.0), (0.056, 0.010, 0.056), color,
              seg=detail, taper=0.93)                       # Foot ring.
    group.cyl((0.0, base_y + height - 0.005, 0.0), (0.079, 0.011, 0.079), color,
              seg=detail, taper=0.99)                       # Rim band.
    group.cyl((0.0, base_y + height - 0.012, 0.0), (0.070, 0.012, 0.070), interior,
              seg=detail, taper=0.97)                       # Inner wall.
    if filled:
        group.cyl((0.0, base_y + height - 0.018, 0.0), (0.069, 0.006, 0.069),
                  rng.jitter(CREMA, 0.08), seg=detail)
        group.cyl((0.0, base_y + height - 0.015, 0.0), (0.043, 0.004, 0.043),
                  rng.jitter(COFFEE), seg=detail // 2)
    else:
        group.cyl((0.0, base_y + 0.026, 0.0), (0.056, 0.040, 0.056), interior,
                  seg=detail, taper=1.22)

    # --- Handle: seven short bars on an arc. Three read as a broken hook.
    bars = 7
    for index in range(bars):
        angle = -70.0 + index * (140.0 / (bars - 1))
        radians = math.radians(angle)
        # The bar's axis is its local +Y, and Rz(phi) sends +Y to
        # (-sin phi, cos phi, 0). The tangent to the handle arc at angle theta
        # is (-sin theta, cos theta), so phi must be theta. Using 90 + theta
        # points every bar radially outward instead, which made the handle read
        # as a fan rather than a loop.
        group.cyl((0.040 + math.cos(radians) * 0.030,
                   base_y + 0.040 + math.sin(radians) * 0.030, 0.0),
                  (0.010, 0.018, 0.010), color, rot=(0.0, yaw, angle), seg=8)


def tumbler(writer, center, height, rng, color=GLASS_CLEAR, water=True, detail=36):
    """Frosted tumbler: tapered, thick base, rim band, visible water line.

    Neither renderer does transparency (both are diffuse-only so they agree
    exactly), so this is frosted glass rather than clear. A plain cylinder read
    as a paper cup; the taper, heavy base and rim ring make it read as glass.
    """
    group = Group(writer, center, 0.0)
    # One frustum: 62 mm at the base, 76 mm at the rim.
    group.cyl((0.0, height * 0.5, 0.0), (0.062, height, 0.062), color,
              seg=detail, taper=76.0 / 62.0)
    group.cyl((0.0, 0.009, 0.0), (0.064, 0.018, 0.064), color, seg=detail, taper=1.01)
    group.cyl((0.0, height - 0.005, 0.0), (0.077, 0.010, 0.077), color, seg=detail, taper=0.995)
    group.cyl((0.0, height - 0.012, 0.0), (0.068, 0.012, 0.068),
              recolor(color, (channel * 0.82 for channel in color)), seg=detail, taper=0.99)
    if water:
        group.cyl((0.0, height * 0.30, 0.0), (0.064, height * 0.60, 0.064),
                  rng.jitter((0.260, 0.330, 0.360)), seg=detail)
        group.cyl((0.0, height * 0.60, 0.0), (0.065, 0.004, 0.065),
                  rng.jitter((0.420, 0.490, 0.520)), seg=detail)


def spoon(writer, center, yaw, rng):
    group = Group(writer, center, yaw)
    group.box((0.0, 0.003, 0.0), (0.011, 0.004, 0.082), rng.jitter(CHROME))
    group.cyl((0.0, 0.004, -0.055), (0.030, 0.006, 0.042), CHROME, seg=14)


def shaker(writer, center, rng, color=CERAMIC, lid=CHROME):
    group = Group(writer, center, 0.0)
    group.cyl((0.0, 0.038, 0.0), (0.048, 0.076, 0.048), color, seg=20)
    group.cyl((0.0, 0.082, 0.0), (0.042, 0.014, 0.042), lid, seg=20)
    group.cyl((0.0, 0.092, 0.0), (0.026, 0.008, 0.026), lid, seg=16)
    del rng


def bottle(writer, base, height, radius, color, rng, cap=None):
    """Body, shoulder, neck, cap. Bottles are the back bar's bokeh supply."""
    group = Group(writer, base, rng.range(0.0, 360.0))
    body = height * 0.62
    group.cyl((0.0, body * 0.5, 0.0), (radius * 2, body, radius * 2), color, seg=28)
    group.cyl((0.0, body + 0.018, 0.0), (radius * 2.0, 0.036, radius * 2.0), color, seg=24,
              taper=0.62 / 2.0)   # Shoulder: a cone from body to neck.
    neck = height - body - 0.036
    group.cyl((0.0, body + 0.036 + neck * 0.5, 0.0),
              (radius * 0.62, neck, radius * 0.62), color, seg=22)
    group.cyl((0.0, height - 0.008, 0.0), (radius * 0.70, 0.022, radius * 0.70),
              cap or rng.pick((BRASS, STEEL_DARK, PAPER)), seg=22)
    # Paper label, slightly proud of the glass so it catches the window light.
    group.cyl((0.0, body * 0.46, 0.0), (radius * 2.06, body * 0.44, radius * 2.06),
              rng.jitter(PAPER), seg=28)


def jar(writer, base, height, radius, contents, rng):
    group = Group(writer, base, 0.0)
    group.cyl((0.0, height * 0.5, 0.0), (radius * 2, height, radius * 2), GLASS_CLEAR, seg=28)
    group.cyl((0.0, height * 0.42, 0.0),
              (radius * 1.82, height * 0.72, radius * 1.82), rng.jitter(contents), seg=24)
    group.cyl((0.0, height + 0.010, 0.0), (radius * 2.05, 0.020, radius * 2.05), BRASS, seg=28)


def cup_stack(writer, base, count, rng, color=CERAMIC):
    """Nested cups. Repeating one shape upward gives a lot of edges cheaply."""
    for index in range(count):
        y = base[1] + index * 0.042
        writer.cyl((base[0], y + 0.030, base[2]), (0.086, 0.060, 0.086),
                   rng.jitter(color, 0.04), seg=30)


def plate_stack(writer, base, count, rng):
    for index in range(count):
        writer.cyl((base[0], base[1] + 0.009 + index * 0.017, base[2]),
                   (0.168, 0.014, 0.168), rng.jitter(CERAMIC, 0.03), seg=34)


def succulent(writer, base, rng, pot_radius=0.055):
    """Pot plus a rosette of short angled leaves."""
    writer.cyl((base[0], base[1] + 0.046, base[2]),
               (pot_radius * 1.74, 0.092, pot_radius * 1.74), TERRACOTTA, seg=30,
               taper=2.0 / 1.74)
    writer.cyl((base[0], base[1] + 0.094, base[2]),
               (pot_radius * 2.05, 0.014, pot_radius * 2.05), TERRACOTTA, seg=30)
    writer.cyl((base[0], base[1] + 0.098, base[2]),
               (pot_radius * 1.7, 0.012, pot_radius * 1.7), (0.045, 0.030, 0.020), seg=24)
    for index in range(11):
        angle = index * 33.0 + rng.range(-8.0, 8.0)
        reach = rng.range(0.032, 0.064)
        radians = math.radians(angle)
        leaf(writer, (base[0] + math.cos(radians) * reach * 0.6,
                      base[1] + 0.110 + rng.range(0.0, 0.030),
                      base[2] + math.sin(radians) * reach * 0.6),
             reach * 1.7, 0.026, rng.jitter(LEAF_PALE, 0.14, hue=True),
             rot=(rng.range(22.0, 46.0), angle, 0.0))


def potted_plant(writer, base, height, rng):
    """Floor plant: pot, stems, and leaves as thin angled boxes."""
    pot_radius = height * 0.115
    writer.cyl((base[0], base[1] + height * 0.14, base[2]),
               (pot_radius * 1.66, height * 0.28, pot_radius * 1.66), TERRACOTTA, seg=32,
               taper=2.0 / 1.66)
    writer.cyl((base[0], base[1] + height * 0.28, base[2]),
               (pot_radius * 2.1, height * 0.03, pot_radius * 2.1), TERRACOTTA, seg=32)
    writer.cyl((base[0], base[1] + height * 0.29, base[2]),
               (pot_radius * 1.7, height * 0.02, pot_radius * 1.7), (0.040, 0.026, 0.018), seg=16)
    for index in range(14):
        angle = rng.range(0.0, 360.0)
        radians = math.radians(angle)
        reach = rng.range(0.10, 0.34) * height
        top = base[1] + height * rng.range(0.36, 0.98)
        stem_end = (base[0] + math.cos(radians) * reach,
                    top,
                    base[2] + math.sin(radians) * reach)
        tube(writer, (base[0], base[1] + height * 0.30, base[2]), stem_end,
             0.012, rng.jitter((0.045, 0.078, 0.032), 0.15, hue=True), seg=10)
        blade = rng.range(0.10, 0.20) * height
        leaf(writer, stem_end, blade, blade * 0.66, rng.jitter(LEAF, 0.16, hue=True),
             rot=(rng.range(-35.0, 35.0), angle, rng.range(-20.0, 20.0)))
        del index


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def pendant_lamp(writer, x, z, cable_top, shade_y, rng, shade_radius=0.115, brass=True):
    """Cable, brass cup, dome shade, bulb and a tiny very bright filament.

    The filament is 12 mm across and 3x the bulb's radiance. Out of focus it
    becomes a clean bright disc, which is the single most revealing object in
    the frame: a gather smears it into a ring-edged blob, while a path tracer
    gives an even disc.
    """
    metal = BRASS if brass else STEEL_DARK
    segment(writer, (x, cable_top, z), (x, shade_y + 0.14, z), 0.008, STEEL_DARK)
    writer.cyl((x, cable_top - 0.024, z), (0.052, 0.048, 0.052), metal, seg=24)
    # Shade as one flared frustum. Stacking rings (five, then fourteen) always
    # left visible steps; a cone has none and needs far fewer triangles.
    writer.cyl((x, shade_y + 0.072, z), (shade_radius * 0.56, 0.150, shade_radius * 0.56),
               metal, seg=40, taper=shade_radius * 2.0 / (shade_radius * 0.56))
    writer.cyl((x, shade_y + 0.150, z), (shade_radius * 0.60, 0.016, shade_radius * 0.60),
               metal, seg=32)
    writer.cyl((x, shade_y - 0.004, z), (shade_radius * 1.62, 0.014, shade_radius * 1.62),
               rng.jitter(STRIP), emit=STRIP_EMIT * 0.5, seg=32)
    writer.sph((x, shade_y - 0.052, z), (0.062, 0.070, 0.062), BULB,
               emit=BULB_EMIT * rng.range(0.9, 1.1), seg=22)
    writer.sph((x, shade_y - 0.054, z), (0.012, 0.018, 0.012), FILAMENT,
               emit=FILAMENT_EMIT, seg=8)


def hero_table(writer, rng):
    """The foreground table, 1.5 m away, carrying the focus subject.

    A table that ENDS is the whole point. The first version of this scene had a
    bar counter running from 0.7 m to 10 m of depth, and because the surface sat
    just below eye level it filled roughly three quarters of the frame all the
    way to its vanishing point, whatever length it was given. A 0.88 m table
    stopping at 1.9 m occupies the bottom third and leaves the cafe visible.
    """
    radius = 0.44
    top_color = rng.jitter(OAK)
    writer.cyl((HERO_TABLE_X, TABLE_TOP_Y - 0.004, HERO_TABLE_Z),
               (radius * 2, 0.008, radius * 2), top_color, seg=64, taper=(radius-0.008)/radius)
    writer.cyl((HERO_TABLE_X, TABLE_TOP_Y - 0.022, HERO_TABLE_Z),
               (radius * 2, 0.028, radius * 2), top_color, seg=64)
    writer.cyl((HERO_TABLE_X, TABLE_TOP_Y - 0.038, HERO_TABLE_Z),
               ((radius-0.008)*2, 0.004, (radius-0.008)*2), OAK_DARK, seg=64, taper=radius/(radius-0.008))
    # Chamfered edge and a darker underside, so the rim is not one hard line.
    writer.cyl((HERO_TABLE_X, TABLE_TOP_Y - 0.048, HERO_TABLE_Z),
               (radius * 1.96, 0.024, radius * 1.96), OAK_DARK, seg=32)
    for index in range(7):  # Plank seams across the top.
        offset = (index + 0.5) / 7.0 - 0.5
        writer.box((HERO_TABLE_X + offset * radius * 1.9, TABLE_TOP_Y + 0.002, HERO_TABLE_Z),
                   (0.008, 0.010, radius * 1.7), OAK_DARK)
    writer.cyl((HERO_TABLE_X, TABLE_TOP_Y - 0.420, HERO_TABLE_Z),
               (0.104, 0.760, 0.104), STEEL_DARK, seg=32, taper=0.082 / 0.104)
    writer.cyl((HERO_TABLE_X, FLOOR_Y + 0.020, HERO_TABLE_Z),
               (0.480, 0.040, 0.480), STEEL_DARK, seg=26)
    bentwood_chair(writer, HERO_TABLE_X - 0.72, HERO_TABLE_Z + 0.16, 84.0, rng)
    bentwood_chair(writer, HERO_TABLE_X + 0.70, HERO_TABLE_Z - 0.08, -96.0, rng)


def hero_props(writer, rng):
    """What sits on the foreground table. This list IS the experiment.

    The cup at 1.5 m is the focus plane. The glass at 1.05 m is foreground
    defocus with a hard silhouette. Everything else in the scene is background
    defocus at a known, increasing distance, and the cups on the tables behind
    are the *same object* at 3.8 m, 6.5 m and 9.5 m, so the circle of confusion
    is the only thing changing between them.
    """
    top = TABLE_TOP_Y + 0.008
    x = HERO_TABLE_X

    # 1.05 m: deliberately clipped by the frame edge.
    tumbler(writer, (x - 0.300, top, 3.950), 0.128, rng)
    # 1.5 m: the subject.
    cup_and_saucer(writer, (x - 0.045, top, HERO_TABLE_Z), rng, yaw=-28.0)
    spoon(writer, (x + 0.170, top + 0.002, HERO_TABLE_Z - 0.030), -14.0, rng)
    writer.box((x + 0.175, top + 0.001, HERO_TABLE_Z + 0.135), (0.115, 0.004, 0.115),
               rng.jitter(PAPER, 0.04), rot=(0.0, 12.0, 0.0))
    # 1.75 m: a second cup, just past the focus plane.
    cup_and_saucer(writer, (x - 0.215, top, HERO_TABLE_Z - 0.240), rng, yaw=64.0,
                   color=CERAMIC_WARM, filled=False)
    shaker(writer, (x + 0.055, top, HERO_TABLE_Z - 0.290), rng)
    succulent(writer, (x + 0.250, top, HERO_TABLE_Z - 0.180), rng, pot_radius=0.052)
    writer.box((x - 0.060, top + 0.020, HERO_TABLE_Z - 0.115), (0.150, 0.040, 0.215),
               rng.jitter(OAK_PALE, 0.1), rot=(0.0, -18.0, 0.0))
    writer.box((x - 0.060, top + 0.042, HERO_TABLE_Z - 0.115), (0.142, 0.006, 0.205),
               rng.jitter(PAPER, 0.05), rot=(0.0, -18.0, 0.0))


def stool(writer, x, z, rng, yaw=0.0):
    group = Group(writer, (x, FLOOR_Y, z), yaw)
    seat_y = 0.74
    group.cyl((0.0, seat_y + 0.024, 0.0), (0.330, 0.048, 0.330), rng.jitter(LEATHER), seg=36)
    group.cyl((0.0, seat_y - 0.004, 0.0), (0.300, 0.028, 0.300), OAK_DARK, seg=32)
    for index in range(4):
        angle = 45.0 + index * 90.0
        radians = math.radians(angle + yaw)
        tube(writer, (x + math.cos(radians) * 0.055, FLOOR_Y + seat_y - 0.02,
                      z + math.sin(radians) * 0.055),
             (x + math.cos(radians) * 0.185, FLOOR_Y + 0.01,
              z + math.sin(radians) * 0.185), 0.032, STEEL, seg=14)
    group.cyl((0.0, 0.30, 0.0), (0.300, 0.026, 0.300), STEEL, seg=28)


def cafe_table(writer, x, z, rng, top_radius=0.34):
    writer.cyl((x, FLOOR_Y + 0.715, z), (top_radius * 2, 0.040, top_radius * 2),
               rng.jitter(OAK), seg=44)
    writer.cyl((x, FLOOR_Y + 0.692, z), (top_radius * 1.94, 0.010, top_radius * 1.94),
               OAK_DARK, seg=40)
    writer.cyl((x, FLOOR_Y + 0.345, z), (0.090, 0.700, 0.090), STEEL_DARK, seg=24,
               taper=0.072 / 0.090)
    writer.cyl((x, FLOOR_Y + 0.018, z), (0.400, 0.036, 0.400), STEEL_DARK, seg=36)


def bentwood_chair(writer, x, z, yaw, rng):
    """Cafe chair in round stock: four legs, a stretcher, and a bent back hoop.

    Everything that is a tube in the real object is a tube here. The first
    version used boxes for the back hoop and square-section sticks for the
    stool legs, and blocky legs are a large part of what makes a render read as
    a game asset rather than a photograph.
    """
    group = Group(writer, (x, FLOOR_Y, z), yaw)
    seat_y = 0.455
    group.cyl((0.0, seat_y, 0.0), (0.380, 0.036, 0.380), rng.jitter(OAK), seg=40)
    group.cyl((0.0, seat_y - 0.022, 0.0), (0.352, 0.014, 0.352), OAK_DARK, seg=36)
    for offset_x, offset_z in ((-0.145, -0.145), (0.145, -0.145), (-0.145, 0.145), (0.145, 0.145)):
        # Legs splay very slightly outward toward the floor, as bentwood does.
        top = rotate_y((offset_x, seat_y - 0.02, offset_z), yaw)
        foot = rotate_y((offset_x * 1.18, 0.012, offset_z * 1.18), yaw)
        tube(writer, (x + top[0], FLOOR_Y + top[1], z + top[2]),
             (x + foot[0], FLOOR_Y + foot[1], z + foot[2]), 0.030, OAK_DARK, seg=14)
    group.cyl((0.0, 0.20, 0.0), (0.330, 0.022, 0.330), OAK_DARK, seg=28)  # Stretcher hoop.

    # Back: two uprights and a hoop across the top, as a chain of short tubes
    # following an arc. Boxes here read as a bracket, not a bend.
    for side in (-1.0, 1.0):
        base = rotate_y((side * 0.160, seat_y, 0.150), yaw)
        top = rotate_y((side * 0.148, seat_y + 0.470, 0.128), yaw)
        tube(writer, (x + base[0], FLOOR_Y + base[1], z + base[2]),
             (x + top[0], FLOOR_Y + top[1], z + top[2]), 0.028, OAK_DARK, seg=14)
    hoop = 9
    previous = None
    for index in range(hoop + 1):
        fraction = index / hoop
        angle = math.pi * (1.0 - fraction)
        local = (math.cos(angle) * 0.148,
                 seat_y + 0.470 + math.sin(angle) * 0.052,
                 0.128 - math.sin(angle) * 0.020)
        turned = rotate_y(local, yaw)
        point = (x + turned[0], FLOOR_Y + turned[1], z + turned[2])
        if previous is not None:
            tube(writer, previous, point, 0.026, OAK_DARK, seg=12)
        previous = point
    # Splat across the middle of the back.
    group.box((0.0, seat_y + 0.260, 0.140), (0.290, 0.070, 0.020), rng.jitter(OAK_PALE, 0.08))
    group.cyl((0.0, seat_y + 0.150, 0.142), (0.280, 0.022, 0.022),
              OAK_DARK, rot=(0.0, yaw, 90.0), seg=12)


def espresso_machine(writer, x, y, z, rng):
    """Two-group machine: body, group heads, wands, portafilters, cup warmer."""
    group = Group(writer, (x, y, z), 180.0)
    group.box((0.0, 0.230, 0.0), (0.860, 0.460, 0.520), rng.jitter(STEEL, 0.04), bevel=0.025)
    group.box((0.0, 0.470, 0.0), (0.820, 0.030, 0.480), STEEL)
    # Painted steel front panel, plus one small lit indicator. The first version
    # made the whole panel a pink emitter at 2.2 radiance, which turned the
    # machine into a glowing slab and was the most obviously wrong thing in the
    # frame.
    group.box((0.0, 0.130, -0.270), (0.880, 0.120, 0.030), rng.jitter(STEEL_DARK, 0.08), bevel=0.006)
    group.box((0.0, 0.205, -0.286), (0.210, 0.026, 0.010), rng.jitter(BRASS, 0.05))
    group.cyl((-0.330, 0.130, -0.286), (0.026, 0.012, 0.026), (1.0, 0.42, 0.22),
              emit=3.0, rot=(90.0, 0.0, 0.0), seg=12)
    group.box((0.0, 0.055, -0.285), (0.560, 0.022, 0.040), STEEL_DARK)
    for index in range(9):  # Drip tray grate.
        group.box((-0.250 + index * 0.062, 0.070, -0.285), (0.020, 0.014, 0.046), STEEL)
    for side in (-1.0, 1.0):
        group.cyl((side * 0.215, 0.300, -0.345), (0.150, 0.130, 0.150), CHROME, seg=16)
        group.cyl((side * 0.215, 0.215, -0.345), (0.096, 0.070, 0.096), CHROME, seg=14)
        group.cyl((side * 0.215, 0.168, -0.345), (0.130, 0.032, 0.130), STEEL_DARK, seg=16)
        group.box((side * 0.215, 0.170, -0.435), (0.042, 0.036, 0.130), STEEL_DARK)
        group.cyl((side * 0.375, 0.250, -0.355), (0.026, 0.240, 0.026), CHROME,
                  rot=(24.0, 0.0, 0.0), seg=10)
        group.cyl((side * 0.310, 0.400, -0.240), (0.070, 0.026, 0.070), STEEL_DARK,
                  rot=(0.0, 0.0, 90.0), seg=12)
        group.box((side * 0.150, 0.500, 0.0), (0.220, 0.030, 0.300), STEEL)
    group.cyl((0.0, 0.520, 0.060), (0.096, 0.070, 0.096), BRASS, seg=16)  # Pressure gauge.
    cup_stack(writer, (x - 0.16, y + 0.49, z + 0.02), 3, rng)
    cup_stack(writer, (x + 0.16, y + 0.49, z - 0.06), 4, rng)


def grinder(writer, x, y, z, rng):
    group = Group(writer, (x, y, z), 180.0)
    group.cyl((0.0, 0.070, 0.0), (0.230, 0.140, 0.230), STEEL_DARK, seg=18)
    group.box((0.0, 0.230, 0.0), (0.180, 0.180, 0.200), rng.jitter(CHROME, 0.04))
    group.cyl((0.0, 0.410, 0.0), (0.200, 0.180, 0.200), GLASS_AMBER, seg=18)
    group.cyl((0.0, 0.400, 0.0), (0.170, 0.130, 0.170), rng.jitter(COFFEE, 0.2), seg=16)
    group.cyl((0.0, 0.510, 0.0), (0.150, 0.030, 0.150), STEEL_DARK, seg=16)
    group.box((0.0, 0.165, -0.110), (0.060, 0.090, 0.030), STEEL)


def pastry_case(writer, x, y, z, rng):
    """Cake stand under a glass dome, at mid depth.

    The dome used to be a full sphere centred 7.5 cm above the counter, which
    put 18 cm of it BELOW the counter top: in a wide shot it read as a beach
    ball floating through the bar. It is now a real dome that sits on the
    surface, built from stacked rings on a circular profile.
    """
    writer.cyl((x, y + 0.020, z), (0.520, 0.040, 0.520), OAK_DARK, seg=40)
    writer.cyl((x, y + 0.052, z), (0.440, 0.030, 0.440), rng.jitter(CERAMIC, 0.03), seg=36)
    for index in range(6):
        angle = math.radians(index * 60.0 + 12.0)
        writer.box((x + math.cos(angle) * 0.150, y + 0.092, z + math.sin(angle) * 0.150),
                   (0.110, 0.055, 0.110), rng.pick((CAKE, CAKE_PINK)),
                   rot=(0.0, index * 60.0, 0.0))
    dome(writer, (x, y + 0.068, z), 0.255, 0.300, GLASS_CLEAR, rings=16, seg=40)
    writer.cyl((x, y + 0.378, z), (0.055, 0.045, 0.055), BRASS, seg=20)   # Knob.


def back_bar(writer, rng):
    """Service run along the right wall: base cabinet, machine, shelves, bottles."""
    top_y = FLOOR_Y + 0.92
    writer.box((2.16, (FLOOR_Y + top_y) * 0.5, -1.40), (1.42, top_y - FLOOR_Y, 7.20),
               rng.jitter(TILE_GREEN, 0.05))
    writer.box((2.16, top_y + 0.025, -1.40), (1.49, 0.050, 7.28), WALNUT, bevel=0.009)
    writer.box((1.47, top_y - 0.10, -1.40), (0.030, 0.220, 7.20), OAK_DARK, bevel=0.006)
    # Cabinet doors, so the front is not one flat panel.
    for index in range(7):
        z = 1.55 - index * 0.96
        writer.box((1.46, FLOOR_Y + 0.42, z), (0.026, 0.640, 0.860),
                   rng.jitter(TILE_GREEN_2, 0.06), bevel=0.005)
        writer.cyl((1.44, FLOOR_Y + 0.42, z), (0.030, 0.140, 0.030), BRASS,
                   rot=(90.0, 0.0, 0.0), seg=10)

    # Original z=1.05 put the entire machine outside the fixed 50 mm frame.
    # Move along the same counter, without moving the research camera.
    espresso_machine(writer, 1.88, top_y + 0.05, -4.62, rng)
    grinder(writer, 2.05, top_y + 0.05, 0.05, rng)
    grinder(writer, 2.35, top_y + 0.05, -0.32, rng)
    # Keep the entire 0.52 m tray on the counter (front edge z=2.24).
    pastry_case(writer, 2.08, top_y + 0.05, 1.85, rng)
    plate_stack(writer, (2.50, top_y + 0.05, -1.20), 7, rng)
    cup_stack(writer, (1.95, top_y + 0.05, -1.05), 5, rng)
    cup_stack(writer, (2.18, top_y + 0.05, -1.35), 4, rng)
    for index in range(5):
        tumbler(writer, (1.92 + (index % 2) * 0.17, top_y + 0.05, -2.10 - index * 0.16),
                0.115, rng, water=False, detail=20)

    # Open shelving on the tiled wall: rows of bottles and jars, the main
    # background bokeh source on the right side of the frame.
    for level, shelf_y in enumerate((FLOOR_Y + 1.42, FLOOR_Y + 1.80, FLOOR_Y + 2.18)):
        # Leave an alcove above the machine instead of intersecting its body.
        shelf_z, shelf_length = (-0.90, 6.00) if level == 0 else (-1.40, 7.00)
        writer.box((2.46, shelf_y, shelf_z), (0.860, 0.042, shelf_length), rng.jitter(OAK, 0.06), bevel=0.006)
        for index in range(9):
            z = 1.70 - index * 0.80
            writer.box((2.84, shelf_y - 0.13, z), (0.040, 0.220, 0.040), STEEL_DARK)
        count = 26 - level * 4
        for index in range(count):
            z = 1.85 - index * (6.9 / count)
            if level == 0 and z < -3.8:
                continue  # Machine alcove, not bottles inside its casing.
            if rng.chance(0.30):
                jar(writer, (rng.range(2.24, 2.66), shelf_y + 0.021, z),
                    rng.range(0.16, 0.24), rng.range(0.050, 0.070),
                    rng.pick((COFFEE, CAKE, TILE_CREAM, (0.28, 0.16, 0.05))), rng)
            else:
                bottle(writer, (rng.range(2.20, 2.70), shelf_y + 0.021, z),
                       rng.range(0.22, 0.34), rng.range(0.036, 0.050),
                       rng.pick((GLASS_GREEN, GLASS_AMBER, GLASS_CLEAR,
                                 rng.jitter(PAPER, 0.2, hue=True))), rng)
        # Warm strip light under each shelf: a long thin emitter, the kind of
        # shape that shows a gather's square sampling footprint most clearly.
        writer.box((2.46, shelf_y - 0.030, shelf_z), (0.620, 0.016, shelf_length-0.4),
                   STRIP, emit=STRIP_EMIT * 0.45)


def window_wall(writer, rng):
    """Left wall: bright daylight panes between mullions, with a low bench.

    Panes are emitters rather than holes so that both renderers agree exactly:
    an actual opening would let Cycles' world background in with sky occlusion
    that the raster pass cannot reproduce.
    """
    writer.box((WALL_LEFT_X - 0.09, (FLOOR_Y + CEILING_Y) * 0.5, -2.00),
               (0.180, CEILING_Y - FLOOR_Y, 16.00), rng.jitter(PLASTER, 0.04))
    writer.box((WALL_LEFT_X + 0.02, FLOOR_Y + 0.44, -2.00), (0.060, 0.880, 15.60),
               rng.jitter(PLASTER_WARM, 0.04))  # Dado below the glazing.
    writer.box((WALL_LEFT_X + 0.05, FLOOR_Y + 0.90, -2.00), (0.130, 0.050, 15.60), OAK_DARK)

    panes = 9
    for index in range(panes):
        z = 4.40 - index * 1.66
        writer.box((WALL_LEFT_X + 0.03, FLOOR_Y + 1.92, z), (0.070, 1.960, 1.520),
                   rng.jitter(WINDOW_DAY, 0.05), emit=WINDOW_EMIT * rng.range(0.88, 1.12))
        # Frame and glazing bars.
        writer.box((WALL_LEFT_X + 0.07, FLOOR_Y + 1.92, z + 0.815), (0.120, 2.040, 0.110), OAK_DARK)
        writer.box((WALL_LEFT_X + 0.07, FLOOR_Y + 2.94, z), (0.120, 0.110, 1.740), OAK_DARK)
        writer.box((WALL_LEFT_X + 0.07, FLOOR_Y + 1.92, z), (0.110, 0.055, 1.520), OAK_DARK)
        for bar in range(2):
            writer.box((WALL_LEFT_X + 0.07, FLOOR_Y + 1.92, z - 0.51 + bar * 1.02),
                       (0.105, 1.960, 0.045), OAK_DARK)
    writer.box((WALL_LEFT_X + 0.07, FLOOR_Y + 0.92, -2.00), (0.120, 0.110, 15.60), OAK_DARK)

    # Window bench and plants along it.
    writer.box((WALL_LEFT_X + 0.42, FLOOR_Y + 0.44, -1.00), (0.620, 0.070, 11.00),
               rng.jitter(OAK, 0.06))
    for index in range(6):
        writer.cyl((WALL_LEFT_X + 0.62, FLOOR_Y + 0.20, 3.60 - index * 2.10),
                   (0.060, 0.410, 0.060), STEEL_DARK, seg=10)
    for index in range(5):
        succulent(writer, (WALL_LEFT_X + 0.36, FLOOR_Y + 0.475, 2.70 - index * 2.05), rng,
                  pot_radius=rng.range(0.050, 0.075))


def right_wall(writer, rng):
    """Offset ('subway') tile behind the shelving: 700+ small boxes of relief.

    Real geometry rather than a texture, so the grout lines self-shadow and the
    two renderers cannot disagree about filtering.
    """
    writer.box((WALL_RIGHT_X + 0.09, (FLOOR_Y + CEILING_Y) * 0.5, -2.00),
               (0.180, CEILING_Y - FLOOR_Y, 16.00), rng.jitter(GROUT, 0.03))
    rows = 13
    for row in range(rows):
        y = FLOOR_Y + 0.95 + row * 0.155
        if y > CEILING_Y - 0.10:
            break
        offset = 0.145 if row % 2 else 0.0
        for index in range(23):
            z = 2.20 - index * 0.290 - offset
            if -6.0 < z < 2.4:
                writer.box((WALL_RIGHT_X - 0.005, y, z), (0.030, 0.140, 0.272),
                           rng.jitter(TILE_CREAM, 0.07))


def back_wall(writer, rng):
    """Back wall at 15 m: chalkboard menu, clock, neon sign, shelf, doorway."""
    writer.box((0.0, (FLOOR_Y + CEILING_Y) * 0.5, Z_BACK - 0.10),
               (7.00, CEILING_Y - FLOOR_Y, 0.200), rng.jitter(PLASTER, 0.03))
    writer.box((0.0, FLOOR_Y + 0.06, Z_BACK + 0.02), (7.00, 0.120, 0.060), OAK_DARK)

    # Chalkboard menu: a dark panel with rows of pale marks. Fine high-contrast
    # detail at maximum depth, so it is the clearest read on background blur.
    board_x, board_y = -1.30, FLOOR_Y + 1.85
    writer.box((board_x, board_y, Z_BACK + 0.03), (2.300, 1.400, 0.050), BLACKBOARD)
    for side in (-1.0, 1.0):
        writer.box((board_x + side * 1.175, board_y, Z_BACK + 0.06), (0.070, 1.520, 0.060), OAK)
    for side in (-1.0, 1.0):
        writer.box((board_x, board_y + side * 0.735, Z_BACK + 0.06), (2.420, 0.070, 0.060), OAK)
    for row in range(9):
        y = board_y + 0.560 - row * 0.140
        width = rng.range(0.70, 1.85)
        writer.box((board_x - 0.98 + width * 0.5, y, Z_BACK + 0.07),
                   (width, 0.036, 0.020), rng.jitter(CHALK, 0.10))
        if rng.chance(0.75):  # Price column on the right.
            writer.box((board_x + 0.92, y, Z_BACK + 0.07), (0.220, 0.032, 0.020),
                       rng.jitter(CHALK, 0.10))

    writer.cyl((1.05, FLOOR_Y + 2.32, Z_BACK + 0.05), (0.420, 0.070, 0.420),
               rng.jitter(PAPER, 0.04), rot=(90.0, 0.0, 0.0), seg=26)
    writer.cyl((1.05, FLOOR_Y + 2.32, Z_BACK + 0.09), (0.360, 0.020, 0.360),
               CERAMIC, rot=(90.0, 0.0, 0.0), seg=24)
    for index, (length, thickness, angle) in enumerate(((0.145, 0.016, 30.0),
                                                         (0.105, 0.020, -70.0))):
        radians = math.radians(angle)
        writer.box((1.05 + math.sin(radians) * length * 0.5,
                    FLOOR_Y + 2.32 + math.cos(radians) * length * 0.5, Z_BACK + 0.11),
                   (thickness, length, 0.014), STEEL_DARK, rot=(0.0, 0.0, angle))
        del index

    # Neon sign: saturated, bright, and at the deepest point in the room.
    for index in range(7):
        writer.box((1.05 + (index - 3) * 0.145, FLOOR_Y + 1.52, Z_BACK + 0.07),
                   (0.038, 0.230, 0.030), NEON_SIGN, emit=NEON_EMIT)
    writer.box((1.05, FLOOR_Y + 1.30, Z_BACK + 0.07), (0.950, 0.036, 0.030),
               NEON_SIGN, emit=NEON_EMIT * 0.8)

    writer.box((1.10, FLOOR_Y + 0.98, Z_BACK + 0.18), (1.900, 0.045, 0.320), OAK)
    for index in range(7):
        bottle(writer, (0.35 + index * 0.24, FLOOR_Y + 1.003, Z_BACK + 0.18),
               rng.range(0.20, 0.30), rng.range(0.034, 0.046),
               rng.pick((GLASS_GREEN, GLASS_AMBER, PAPER)), rng)

    # Doorway to the back, with a dim service light behind it.
    writer.box((-2.85, FLOOR_Y + 1.02, Z_BACK + 0.02), (0.900, 2.040, 0.080), OAK_DARK)
    writer.box((-2.85, FLOOR_Y + 1.02, Z_BACK + 0.07), (0.760, 1.900, 0.040),
               rng.jitter((0.30, 0.26, 0.22), 0.05), emit=1.6)


def floor(writer, rng):
    """Checkerboard tile with grout gaps, plus a runner of boards under the counter."""
    writer.box((0.0, FLOOR_Y - 0.06, -2.00), (7.20, 0.120, 16.40), rng.jitter(GROUT, 0.03))
    tile = 0.30
    columns, rows = 22, 54
    for row in range(rows):
        z = 5.60 - row * tile
        if z < Z_BACK - 0.1:
            break
        for column in range(columns):
            x = -3.30 + column * tile
            # Skip tiles the bar cabinets hide completely; there is no point
            # tessellating floor nothing can see.
            if x > 1.40 and -5.1 < z < 2.3:
                continue
            base = FLOOR_TILE_A if (row + column) % 2 else FLOOR_TILE_B
            writer.box((x, FLOOR_Y - 0.008, z), (tile - 0.012, 0.018, tile - 0.012),
                       rng.jitter(base, 0.09))


def ceiling(writer, rng):
    writer.box((0.0, CEILING_Y + 0.08, -2.00), (7.20, 0.160, 16.40), rng.jitter(PLASTER, 0.02))
    for index in range(9):
        z = 4.00 - index * 1.70
        writer.box((0.0, CEILING_Y - 0.06, z), (6.90, 0.130, 0.200), rng.jitter(OAK_DARK, 0.06))
    # Conduit and a track of small downlights, more emitters at staggered depth.
    writer.cyl((-1.90, CEILING_Y - 0.05, -2.00), (0.044, 15.60, 0.044),
               STEEL_DARK, rot=(90.0, 0.0, 0.0), seg=10)
    for index in range(8):
        z = 3.40 - index * 1.80
        writer.cyl((1.55, CEILING_Y - 0.10, z), (0.110, 0.120, 0.110), STEEL_DARK, seg=14)
        writer.cyl((1.55, CEILING_Y - 0.165, z), (0.082, 0.020, 0.082),
                   STRIP, emit=STRIP_EMIT * 1.6, seg=12)


def counter_props(writer, rng):
    """What sits on the counter, ordered by depth from the camera.

    This list IS the experiment. The cup at 1.60 m is the focus plane; the glass
    at 1.15 m is foreground defocus; everything beyond is background defocus at
    a known, increasing distance, using objects of the same physical size so the
    circle of confusion is the only thing changing.
    """
    top = COUNTER_TOP_Y + 0.008
    x = COUNTER_CENTER_X

    # Foreground, 0.9 m: clipped by the frame edges on purpose, so the
    # foreground blur has a hard silhouette to get wrong.
    tumbler(writer, (x - 0.190, top, 4.100), 0.128, rng)
    shaker(writer, (x + 0.205, top, 4.140), rng)

    # Focus plane, 1.4 m.
    cup_and_saucer(writer, (x - 0.020, top, 3.600), rng, yaw=-28.0)
    spoon(writer, (x + 0.165, top + 0.002, 3.560), -14.0, rng)
    writer.box((x + 0.168, top + 0.001, 3.730), (0.115, 0.004, 0.115),
               rng.jitter(PAPER, 0.04), rot=(0.0, 12.0, 0.0))  # Napkin.

    # 2.0 m: the same cup again, the first clearly soft object.
    cup_and_saucer(writer, (x + 0.095, top, 3.000), rng, yaw=64.0, color=CERAMIC_WARM)
    tumbler(writer, (x - 0.205, top, 2.900), 0.115, rng)

    # 2.8 m and 4.0 m.
    cup_and_saucer(writer, (x - 0.130, top, 2.200), rng, yaw=8.0, filled=False)
    writer.box((x + 0.150, top + 0.022, 2.120), (0.150, 0.044, 0.215), rng.jitter(OAK_PALE, 0.1),
               rot=(0.0, -18.0, 0.0))  # Closed book.
    writer.box((x + 0.150, top + 0.046, 2.120), (0.142, 0.006, 0.205), rng.jitter(PAPER, 0.05),
               rot=(0.0, -18.0, 0.0))
    succulent(writer, (x + 0.140, top, 1.000), rng, pot_radius=0.062)
    tumbler(writer, (x - 0.175, top, 0.900), 0.120, rng, water=False)

    # 5.6 m onward: cups, jars and a till, thinning out down the counter.
    cup_stack(writer, (x - 0.150, top, -0.600), 4, rng)
    jar(writer, (x + 0.160, top, -0.680), 0.200, 0.062, CAKE, rng)
    cup_and_saucer(writer, (x - 0.020, top, -1.700), rng, yaw=140.0)
    succulent(writer, (x + 0.170, top, -2.600), rng, pot_radius=0.055)
    for index in range(5):
        tumbler(writer, (x - 0.200 + (index % 2) * 0.055, top, -3.300 - index * 0.190),
                0.112, rng, water=False)
    writer.box((x + 0.100, top + 0.090, -4.300), (0.300, 0.180, 0.240),
               rng.jitter(STEEL_DARK, 0.1))
    writer.box((x + 0.100, top + 0.186, -4.300), (0.250, 0.014, 0.170),
               DISPLAY, emit=DISPLAY_EMIT)
    plate_stack(writer, (x - 0.140, top, -4.900), 6, rng)


def seating(writer, rng):
    """Tables receding from 3.8 m to 9.5 m, each with the same cups on it.

    Depths are chosen so the same 14 cm saucer appears at 3.8, 6.5 and 9.5 m in
    one frame. Against the sharp one at 1.5 m that is four samples of the circle
    of confusion in a single image, which is what makes the blur measurable
    rather than merely visible.
    """
    layout = ((-0.62, 1.20, 0.36), (0.78, -1.50, 0.34), (-0.95, -4.50, 0.38),
              (0.60, -7.30, 0.34), (-1.55, -8.60, 0.32))
    for index, (x, z, radius) in enumerate(layout):
        cafe_table(writer, x, z, rng, top_radius=radius)
        bentwood_chair(writer, x - 0.66, z + 0.12, 80.0 + index * 11.0, rng)
        bentwood_chair(writer, x + 0.64, z - 0.16, -100.0 - index * 8.0, rng)
        if index < 3:
            bentwood_chair(writer, x + 0.10, z + 0.68, 172.0 + index * 6.0, rng)
        table_top = FLOOR_Y + 0.737
        cup_and_saucer(writer, (x + rng.range(-0.14, 0.14), table_top,
                                z + rng.range(-0.14, 0.14)), rng,
                       yaw=rng.range(0.0, 360.0),
                       color=rng.pick((CERAMIC, CERAMIC_WARM)),
                       detail=24 if index else 32)
        if rng.chance(0.75):
            tumbler(writer, (x + rng.range(-0.22, 0.22), table_top,
                             z + rng.range(-0.22, 0.22)), 0.115, rng, detail=20)
        if rng.chance(0.5):
            shaker(writer, (x + rng.range(-0.20, 0.20), table_top,
                            z + rng.range(-0.20, 0.20)), rng)
        if rng.chance(0.45):
            writer.box((x + rng.range(-0.16, 0.16), table_top + 0.020,
                        z + rng.range(-0.16, 0.16)), (0.150, 0.040, 0.215),
                       rng.jitter(OAK_PALE, 0.12), rot=(0.0, rng.range(0.0, 180.0), 0.0))
    potted_plant(writer, (-2.42, FLOOR_Y, -3.10), 1.35, rng)
    potted_plant(writer, (-2.30, FLOOR_Y, 2.40), 1.05, rng)
    potted_plant(writer, (1.85, FLOOR_Y, -9.10), 1.15, rng)
    for index in range(4):
        stool(writer, 1.05, 1.15 - index * 1.30, rng, yaw=rng.range(-22.0, 22.0))


def pendants(writer, rng):
    """Lamps over each table, at 1.5 m to 13.5 m of depth.

    One repeated small bright object at known increasing distances, and the
    single most revealing thing in the frame: each carries a 12 mm filament at
    three times the bulb's radiance, so out of focus it should become a clean
    even disc. A gather turns it into a ring-edged blob instead, which is
    exactly the error the learned stage has to fix.
    """
    for x, z, shade_y, radius in ((HERO_TABLE_X, HERO_TABLE_Z, 0.46, 0.125),
                                  (-0.62, 1.20, 0.50, 0.115),
                                  (0.78, -1.50, 0.46, 0.130),
                                  (-0.95, -4.50, 0.52, 0.115),
                                  (0.60, -7.30, 0.48, 0.130),
                                  (-1.55, -8.60, 0.44, 0.115)):
        pendant_lamp(writer, x + rng.range(-0.02, 0.02), z, CEILING_Y - 0.14,
                     shade_y, rng, shade_radius=radius)
    # Festoon along the top of the window wall: the same bulb repeated into
    # depth, spanning 1 m to 15 m in one continuous line.
    for index in range(12):
        z = 4.40 - index * 1.25
        sag = 0.055 * math.sin(index * 0.9 + 0.4)
        writer.sph((WALL_LEFT_X + 0.26, FLOOR_Y + 1.45 + sag, z),
                   (0.070, 0.082, 0.070), BULB, emit=BULB_EMIT * rng.range(0.8, 1.15), seg=12)
        if index:
            segment(writer, (WALL_LEFT_X + 0.26, FLOOR_Y + 1.51 + sag, z),
                    (WALL_LEFT_X + 0.26,
                     FLOOR_Y + 1.51 + 0.055 * math.sin((index - 1) * 0.9 + 0.4), z + 1.25),
                    0.010, STEEL_DARK)


# ---------------------------------------------------------------------------

def build(seed=20261001):
    rng = Rng(seed)
    writer = SceneWriter()
    writer.comment("dof-research cafe scene, format version 1")
    writer.comment("GENERATED by tools/scene/build_cafe.py -- regenerate instead of")
    writer.comment("hand-editing large changes. Single-line tweaks are fine: press L in the")
    writer.comment("renderer to reload without a rebuild.")
    writer.comment()
    writer.comment("Camera: 50 mm on a 24 mm sensor, f/1.4, focused 1.4 m on the cup.")
    writer.comment("CoC radius at 1200 px: 0 px on the cup, ~18 px on the foreground glass,")
    writer.comment("~30 px on the back wall. Press K in the renderer for ~124 px.")
    writer.comment()
    writer.raw("version 1")
    writer.raw("camera pos 0 0 5 yaw -90 pitch -10 focus 1.5 fnumber 1.4 lens 50 sensor 24")
    # Daylight through the window wall on the left. The wall itself blocks the
    # sun except at the panes, so both renderers get the same light shafts
    # across the counter rather than a uniformly lit room.
    # Shallow and mostly from the front, so the light actually enters: the room is
    # enclosed on the left, right, back, top and bottom, and the window panes are
    # opaque emitters, so a sun aimed through the window wall reaches nothing at
    # all. At 17 degrees above the horizon and angled back, the shaft comes in
    # through the open front of the room and crosses the floor from z = 6 to
    # about z = -1, which is exactly where the tables are.
    writer.raw("sun dir -0.55 0.30 0.78 color 1 0.94 0.84 energy 7 angle 1.4")
    # Interior bounce, brighter and cooler than the alley's dusk sky.
    # Interior fill. Both renderers apply this as an UNOCCLUDED term: in OpenGL it
    # is the ambient in basic.frag, and in Cycles render_dof.py adds the same
    # albedo * sky radiance as an emission on every surface, with diffuse bounces
    # set to zero so it is not double counted. Before that fix, Cycles took its
    # ambient from the world background, which cannot reach an enclosed room, and
    # the reference render came out black while OpenGL looked fine.
    writer.raw("ambient color 0.60 0.66 0.78 strength 0.52")
    # Everything the camera can see, and nothing more: the shadow map is 4096 px
    # over this box, about 4 mm per texel, which is what resolves the tile
    # relief and the counter's plank seams.
    writer.raw("shadow lo -3.6 -1.3 -10.6 hi 3.6 1.8 5.2")
    writer.comment()

    writer.section("floor")
    floor(writer, rng)
    writer.comment()
    writer.section("ceiling and downlights")
    ceiling(writer, rng)
    writer.comment()
    writer.section("window wall, left")
    window_wall(writer, rng)
    writer.comment()
    writer.section("tiled wall, right")
    right_wall(writer, rng)
    writer.comment()
    writer.section("back wall, 15 m")
    back_wall(writer, rng)
    writer.comment()
    writer.section("service bar and shelving, right")
    back_bar(writer, rng)
    writer.comment()
    writer.section("foreground table, 1.5 m")
    hero_table(writer, rng)
    writer.comment()
    writer.section("focus-plane props, 1.05 m to 1.8 m (the experiment)")
    hero_props(writer, rng)
    writer.comment()
    writer.section("seating and plants, left")
    seating(writer, rng)
    writer.comment()
    writer.section("pendant lamps and festoon")
    pendants(writer, rng)

    text = writer.text()
    return text, scene_loader.parse_scene(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="scene/cafe.scene")
    parser.add_argument("--seed", type=int, default=20261001)
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
