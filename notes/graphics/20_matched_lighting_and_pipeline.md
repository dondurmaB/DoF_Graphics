# Experiment 20: matched interior lighting, object realism, gather quality, one-command pipeline

Four problems, and the first three turned out to be closely related.

## 1. The Cycles render was almost black

**Cause.** `basic.frag` adds the ambient term as `albedo * sky_radiance` with **no occlusion** — every
surface gets it regardless of what is around it. Cycles got the same quantity from the *world
background*, which only reaches surfaces that can see the sky. The café is an enclosed room, and
the window panes are opaque emitters marked invisible to diffuse rays, so **no world light reached
anything**. Measured at a table top:

| | linear radiance | sRGB |
|---|---|---|
| OpenGL | 0.0956 | 87/255 |
| Cycles | 0.0000 | 0/255 |

In the outdoor alley this was a small residual, documented as "the raster ambient is slightly
flatter". Moving indoors turned it into the entire image. That is a design error on my part: I
carried an outdoor lighting model into an interior.

**Fix, in three parts.**

*Give Cycles the same unoccluded term.* Every surface material now gets an `Emission` of
`albedo * sky_radiance` added to its Diffuse BSDF (`add_ambient_fill` in `render_dof.py`), using a
`VectorMath MULTIPLY` node because `ShaderNodeMixRGB` was replaced in Blender 4.x.

*Stop it being counted twice.* An emission on every surface would also bounce around the room as
indirect light, which OpenGL has none of. So in matched mode `diffuse_bounces = 0`: the shading
model becomes exactly `albedo * (sky + sun * N·L / π)`, which is what `basic.frag` computes. The
world is gated behind a `Light Path → Is Camera Ray` node, so the sky is visible *through* an
opening (matching the GL clear colour) but lights nothing.

*Let the sun actually enter.* The old sun pointed through the window wall, which is solid, so it
reached nothing at all. It now comes in at 17° above the horizon through the open front of the
room and crosses the floor from z = 6 to about z = −1, which is where the tables are. Energy 3.6 →
7.0, fill strength 0.30 → 0.52.

This is a **deliberately non-physical fill in both renderers**, and that is the point: the
experiment measures the lens, so the shading model is pinned identical and the only remaining
difference is how depth of field was computed. It also removes the ambient-occlusion discrepancy
that was previously listed as a known limitation. `--full-gi` restores real global illumination and
is physically complete, but then a difference measures GI as well as defocus.

## 2. Objects looked like game props

Three causes, all geometric:

**The palette was an outdoor-dusk palette.** Oak at 0.128 linear is a dark stain. Real reflectances
for a light café are much higher, so everything was murky regardless of light level. The whole
palette is now at roughly measured values: plaster 0.63, light oak 0.235, glazed ceramic 0.775,
terracotta 0.27.

**The cups were solid cylinders.** A single cylinder with a flat disc of coffee on top reads as a
bucket. `cup_and_saucer` is now a turned profile: six stacked cylinders along a bellied curve
(cups are wider at the rim than the foot), a proud rim band with an inner recess in darker ceramic
so the opening is visible, a foot ring, and a saucer with a real profile — foot ring, concave well,
raised rim. The tumbler got the same treatment plus a thick base and a water line.

**Not enough segments, and a rotation bug in the handle.** The hero cup is 48 segments around
instead of 28; at the focus plane it is a fifth of the frame and 28 facets are individually
visible. Cups on the far tables stay at 20–24, since they are a few pixels across and fully
defocused. The handle was seven bars rotated by `90 + θ` about Z, which points each bar *radially*
— it rendered as a fan. `Rz(φ)` sends the bar's local +Y to `(−sin φ, cos φ, 0)`, and the tangent
to the arc at angle θ is `(−sin θ, cos θ)`, so φ must be θ.

Cost: 2519 → 2642 primitives, 74,416 → 85,796 triangles.

## 3. The strong blur looked bad

Two separate problems, and the visible halo was the worse one.

**Sharp objects bled into the blur.** The naive gather averages every neighbour inside the disc
equally, so a background pixel happily averages in a foreground pixel whose own circle of
confusion is a fraction of a pixel wide and could never physically have reached it. That is the
glow around the foreground glass.

The fix is to weight each sample by whether its *own* CoC reaches the pixel being written:

```glsl
weight = clamp(sampleRadius - distanceInPixels + 1.0, 0.0, 1.0);
```

A sharp sample has `sampleRadius ≈ 0`, so at any real distance its weight is zero and it cannot
bleed. A genuinely defocused sample contributes across its whole disc. This is a gather
approximating a scatter, at the cost of one extra depth fetch per tap.

**Structured undersampling.** The Vogel spiral's arms lined up across the whole frame, so a large
radius showed concentric rings and blocky structure. The spiral is now rotated by a per-pixel hash
of `gl_FragCoord` (deterministic, so screenshots are still reproducible), which turns that
structure into fine noise. Tap count also scales with the disc's *area* rather than being fixed,
up to the ceiling — a count that looks fine at 10 px is badly undersampled at 120 px.

**Both gathers are kept**, selected by `uGatherMode` / the panel / the `N` key, and the screenshot
filename records which one was used. The naive gather is the baseline the learned stage is meant to
repair; the weighted one is what you would actually ship. Measuring both against the same reference
shows how much the weighting buys, which is a stronger result than measuring either alone.

**Still not fixed, deliberately:** the gather searches only to the *centre* pixel's radius, so a
heavily blurred foreground does not spread onto a sharp background behind it (that needs a
near-field prepass). And neither mode can recover the surfaces a real lens sees *around* an
occluder, because the rasterizer never stored them. That is the irreducible part, and it is what
the comparison measures.

## 4. One command instead of three

`tools/compare/run_comparison.py` runs all three stages and stops at the first failure:

```sh
python3 tools/compare/run_comparison.py              # ground truth, both gathers
python3 tools/compare/run_comparison.py --quick      # 64 samples, ~minutes, pipeline check
python3 tools/compare/run_comparison.py --dry-run    # print the commands and expected files
```

The enabling change is **batch mode in the renderer**: `DepthResearch --batch --focus 1.5 --fstops
1.2 1.4 2.8 11 --gather 0 1` opens a hidden window, renders each combination through the normal
shaders and framebuffer, writes `output/gl_*.png`, and exits. The window is hidden rather than
absent because a GL context needs one on macOS, and reusing the real render path means a batch
capture is pixel-identical to what the UI shows.

The settings are now defined **once**, in the orchestrator, and passed to both renderers. Before,
each stage took its own flags and they had to be kept in step by hand — and getting that wrong does
not fail loudly, it produces a plausible difference image of two different camera setups. Three
places still build filenames (`render_jobs()`, the C++ batch loop, `expected_pairs()`), so
`tests/test_pipeline.py` pins them together; a disagreement now fails a test rather than silently
finding no pairs.

The orchestrator also verifies every expected file exists before running the metrics, refuses a
multi-lens sweep (one Cycles run has one field of view, so each lens needs its own reference run),
and prints the summary table at the end.

## Verify
```
cmake --build build && ctest --test-dir build
python3 tools/compare/run_comparison.py --dry-run
python3 tools/scene/build_cafe.py && python3 tests/test_scene_file.py --print-expected
python3 tools/scene/preview_scene.py --scene scene/cafe.scene --output output/cafe.png
```

---

## Follow-up: two modelling errors and the faceting

Found by rendering the scene from a wide viewport angle rather than the scene camera, which is
worth doing routinely — the fixed camera hides a lot.

### Two outright errors

**The pastry case was a floating beach ball.** It was a full sphere centred 7.5 cm above the
counter with a 0.56 m height, so **0.18 m of it hung below the counter top**. There is now a
`dome()` helper in `authoring.py` that builds a hemisphere from rings on a circular profile and
sits on the surface. Any object that rests on something should use it; a sphere placed on a surface
always sinks half of itself through it.

**The espresso machine had a glowing pink front.** A `0.88 x 0.23 m` panel in `NEON_SIGN`
(1.0, 0.30, 0.42) at `emit = 2.2`. The intent was a painted panel with a small lit indicator; what
it produced was a self-lit pink slab, the most obviously wrong thing in the frame. It is now
painted steel with a brass badge and one 26 mm warm indicator lamp.

### Why everything looked "cubic", and the fix

Four separate causes:

**Stacked cylinders.** The turned objects — cups, tumblers, pots, lampshades — were built by
stacking cylinders along a profile curve. That never works: each ring's cap leaves a small annular
ledge, which reads as corduroy banding down the side. Going from 6 rings to 14 made it finer, not
absent.

The fix is a new primitive capability: **`taper`**, a single float on `cyl` that scales the *top*
radius. The side becomes a frustum with correct smooth normals, derived rather than guessed:

```
P(theta, v) = (r(v) cos theta, v - 0.5, r(v) sin theta),  v in [0,1]
dP/dtheta x dP/dv = -r * (cos theta, -(r_top - r_bottom), sin theta)
```

so the outward normal is `normalize((cos t, r_bottom - r_top, sin t))` over a unit height. A cup
body is now **one** primitive with `taper = 78/52`, with no internal seams at all, and the scene
came *down* from 145,144 to 115,864 triangles.

`taper` defaults to 1.0, at which the middle term is zero and the maths reduces exactly to the old
cylinder. Both loaders are tested for byte-identical output at `taper 1.0`, so no existing scene
file changes and the format stays at version 1. `taper 0` gives a cone, and its degenerate top cap
is skipped rather than built as a zero-radius disc.

**Square legs.** `segment()` builds a *box* between two points, which is right for a cable but
wrong for a chair leg. A `tube()` helper now builds a cylinder between two points, with the
rotation derived from the same `Ry·Rx·Rz` convention: for a unit direction `d`,
`pitch = acos(d.y)` and `yaw = atan2(d.x, d.z)`. Chair legs, stool legs, the back hoop and plant
stems are all round stock now, and the bentwood back is a chain of short tubes following an arc
rather than five boxes.

**Low segment counts.** Table pedestals and seats at 16–26 around were visible polygons. Now 24–44
for anything a viewer sees up close; the hero cup is at 48 and far-table cups stay at 20–24, since
they are a few pixels across and fully defocused.

**Hue jitter.** `Rng.jitter` varied each channel independently, which shifts hue. Across a thousand
floor tiles that reads as a patchwork of pink and green plastic rather than one batch of tiles with
slight variation — clearly visible in the wide render. It is brightness-only by default now (one
factor on all three channels), with `hue=True` where real colour variety is wanted: bottle glass,
leaves, stems.

**Leaves** were flat boxes, which read as folded paper. A `leaf()` helper makes a flattened
elliptical disc instead.
