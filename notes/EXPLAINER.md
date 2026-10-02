# How this project works, and what changed

Written to be read start to finish before a meeting. It covers what depth of field is
physically, how each of the two renderers produces it, why they can be compared at all, and
what changed in experiment 18. Where a claim is checkable, the file or test that checks it is
named.

Current material model: [matched specular and café geometry](graphics/matched_specular_and_geometry.md). This adds verified direct GGX to both renderers; historical Lambert-only descriptions below precede that change. Matching the BRDF does not remove differences in shadow visibility or lens-dependent shading.

Companion documents:
- [`notes/graphics/19_cafe_inventory.md`](graphics/19_cafe_inventory.md) - object sizes and depths, generated from the scene
- [`notes/graphics/20_matched_lighting_and_pipeline.md`](graphics/20_matched_lighting_and_pipeline.md) - the change log for experiment 20
- [`notes/graphics/19_cafe_and_ground_truth.md`](graphics/19_cafe_and_ground_truth.md) - the change log for experiment 19
- [`notes/graphics/18_alley_scene_and_ui.md`](graphics/18_alley_scene_and_ui.md) - the change log for experiment 18
- [`ai/ARCHITECTURE.md`](../ai/ARCHITECTURE.md) - the machine-learning stage this is all feeding

---

## 1. Depth of field, physically

A pinhole camera has infinite depth of field: one ray per direction, everything sharp. A real
lens has an opening of finite width, so it collects a *cone* of rays from each point in the
scene. The lens focuses that cone back to a point only for objects at one particular distance,
the **focus distance**. Objects nearer or further converge in front of or behind the sensor, so
by the time they reach the sensor they have spread into a small disc.

That disc is the **circle of confusion** (CoC). Its diameter on the sensor is

```
CoC = (A * f * |d - s|) / (d * (s - f))          A = f / N
```

| Symbol | Meaning | Here |
|---|---|---|
| `f` | focal length | 0.050 m (50 mm) |
| `N` | f-number | 1.4 |
| `A` | aperture diameter | `f / N` = 35.7 mm |
| `s` | focus distance | 5 m |
| `d` | distance to the object | varies |

Converted to pixels by dividing by the sensor height and multiplying by the frame height in
pixels. Three things follow, and they are worth stating because they explain most of what the
renderer does:

1. **CoC is zero at `d = s` and grows in both directions.** Nearer than focus and further than
   focus are both blurred; the code keeps the sign so the two cases can be told apart.
2. **Background blur saturates.** As `d` grows, `|d - s| / d` tends to 1, so CoC tends to
   `A*f/(s-f)`, a finite limit. Foreground blur does not saturate: as `d` tends to 0 the CoC
   grows without bound. This is why a near object is the most expensive thing in the frame.
3. **Wide aperture, long lens, close focus all increase blur.** `A = f/N`, so CoC scales as
   `f^2 / N`. Doubling focal length quadruples the blur.

### Diameter against radius: the bug this project had

CoC as defined above is a **diameter**. `shaders/screen.frag` used to compute it and then pass
it straight in as the *radius* of its blur kernel, which made every OpenGL DoF image in
experiments 14-17 about twice as blurred as the Cycles reference at the same f-number. It is
now halved explicitly, at the one place the gather starts:

```glsl
float blurRadiusPixels = min(abs(cocDiameterPixels) * 0.5, uMaxBlurRadiusPixels);
```

If someone asks how a 2x error survived four experiments: nothing compared the two renderers
numerically, and a doubled blur still looks like plausible depth of field. That is the argument
for `tools/compare/compare_renders.py` existing.

### What "120 px" actually gets you

The blur ceiling is 120 px, as specified. It is a *ceiling*, and at the reference settings
nothing comes close to it. At 50 mm, f/1.4, focused 5 m, on a 1200 px frame:

| Lens | f-number | Focus | Background CoC radius |
|---|---|---|---|
| 50 mm | f/8 | 5 m | 1.6 px |
| 50 mm | f/2.8 | 5 m | 4.5 px |
| 50 mm | f/1.4 | 5 m | **9.0 px** |
| 85 mm | f/1.4 | 1.6 m | 85.2 px |

So at the settings that match the reference renders, the lens is the limit, not the 120 px
ceiling. The panel now prints the radius actually reached beside the ceiling and says
"lens-limited" when the ceiling is doing nothing, and `K` loads the last row for a visibly
strong blur. Both are worth showing: the first proves the gather works, the second is the one
that gets compared.

---

## 2. Method A: the OpenGL gather (fast, approximate)

A rasterizer cannot trace a cone of rays through a lens. It draws each triangle once, with one
sample per pixel, and gets a pinhole image. Depth of field is then *reconstructed* in screen
space, in three passes.

### Pass 0: shadow map
Render the scene depth-only from the sun's point of view into a 4096x4096 depth texture
(`shaders/shadow.vert`). In the main pass, a surface transforms its position into that same
light space and compares: if something else was closer to the light, this point is in shadow.

Two refinements, both needed because the cafe is full of shallow relief (tile grout lines,
plank seams, chair spindles, shelf brackets):
- **PCF** (percentage-closer filtering): average a 3x3 block of shadow-map texels instead of
  one, so edges are soft rather than staircased.
- **Normal-offset**: look the shadow map up from a point pushed about one texel out along the
  surface normal (`shaders/basic.vert`). Without it, a surface at a grazing angle to the sun
  shadows itself in stripes ("shadow acne"). A depth-only bias can fix that too, but only by
  detaching contact shadows so crates appear to float.

The map is only re-rendered when the sun direction or the visible geometry changes. The scene is
static, so redrawing 38,572 triangles every frame for an identical result is pure waste.

### Pass 1: the scene into a floating-point target
Draw everything into an off-screen framebuffer with an **RGBA16F** colour attachment and a
depth texture. The shading is one Lambert term plus a uniform sky term
(`shaders/basic.frag`, section 4 below).

The float target matters. The cafe contains emitters — window panes, pendant bulbs with 12 mm
filaments, strip lights under the shelves, a neon sign — whose
radiance is deliberately above 1.0. In the old `GL_RGB8` target they were clipped to white and
quantized to 256 levels *before* the blur ran, so an out-of-focus bulb averaged out to a flat
grey disc instead of a bright bokeh highlight. Blur must happen on real radiance.

### Pass 2: the defocus gather
Draw one full-screen quad. For each pixel (`shaders/screen.frag`):

1. Read the depth texture and convert it back to linear metres. Raw depth is stored
   non-linearly (most of its precision sits near the camera), so this is undone explicitly.
2. Evaluate the CoC formula for that depth → signed diameter in pixels.
3. Halve it to get a radius, clamp to the ceiling.
4. Sample the colour texture at 100 points spread over a disc of that radius, and average.

The sample pattern is a **Vogel spiral**: the *i*-th of *n* points sits at radius
`sqrt((i+0.5)/n)` and angle `i * 137.5°`. The square root keeps density even per unit area
rather than bunching samples in the middle, and the golden angle stops the spiral lining up
into visible spokes. Conceptually it stands in for sampling points across a real aperture,
which is exactly what the ray tracer does for real.

Finally, exposure is applied and the result is encoded with the sRGB transfer function. This is
the last step, deliberately, so that everything upstream is linear.

### What this method gets wrong, on purpose

These are not bugs. They are the error the machine-learning stage exists to fix, so they are
left in and documented.

- **No depth test between samples.** A pixel gathers its neighbours regardless of their depth,
  so a sharp foreground object bleeds into the blurred background behind it, and blurred
  background bleeds onto sharp foreground. Worst exactly at silhouettes.
- **No occluded information.** A real lens sees slightly *around* a foreground object, because
  different points on the aperture have different viewpoints. The rasterizer only ever stored
  the front-most surface, so that information does not exist to be recovered. This is the
  fundamental limit of the whole approach.
- **One CoC per pixel.** A pixel straddling a depth edge gets a single depth and therefore a
  single blur radius, when physically it should be a mix.
- **No ambient occlusion.** See section 4.

---

## 3. Method B: the Cycles reference (slow, correct)

`tools/raytraced_reference/render_dof.py` renders the same scene in Blender's Cycles path
tracer. There is no CoC formula anywhere in it. Instead, Cycles is told the physical lens —
focal length, sensor size, f-number, focus distance — and then, for every one of 128 samples
per pixel, it:

1. picks a random point on the aperture disc,
2. picks a point within the pixel,
3. traces the ray those two points define into the scene,
4. follows it as it bounces, accumulating light.

Depth of field emerges from step 1. A point at the focus distance maps to the same pixel from
every aperture point, so it stays sharp; a point at another distance maps to different pixels
and so spreads out. Because each ray is traced independently through the real geometry, all
four limitations listed above simply do not arise: occlusion is resolved per ray, and the lens
genuinely does see around near objects.

The cost is time. 4096 samples per pixel at 1200x1200 is about 5.9 billion primary rays, plus
bounces, and takes minutes to hours. The OpenGL pass runs in milliseconds. **That gap is the entire
motivation for the project**: learn to produce something close to the slow image at the speed of
the fast one.

### What makes a render ground truth, as opposed to merely path-traced

Using a path tracer is not sufficient. Three things disqualified the earlier references, and all
three are now controlled by `--ground-truth`:

1. **The denoiser.** OpenImageDenoise was on. A denoiser is a plausible-detail generator: it
   removes noise by inventing what it thinks should be there. Training a network against denoised
   output teaches it to match a guess, and large smooth out-of-focus discs are exactly the regions
   a denoiser rewrites most. Ground truth means the denoiser is **off** and the noise is beaten
   down with samples instead.
2. **8-bit PNG through a view transform.** Every value above 1.0 clipped, everything quantized to
   256 levels. The bright highlights whose bokeh *is* the measurement were the first thing lost.
   Ground truth writes **32-bit linear EXR** with no view transform.
3. **No error bar.** 128 samples with no statement of residual noise, so a measured difference
   between the two renderers could not be told apart from the reference's own noise.

| Setting | Default | `--ground-truth` | Why |
|---|---|---|---|
| Samples | 128 | 4096 | Noise floor about 5.7x lower (1/sqrt(N)) |
| Denoiser | OpenImageDenoise | **off** | Never train against invented detail |
| Output | PNG 8-bit sRGB | **EXR 32-bit linear** | No clipping, no quantization |
| Adaptive sampling | on | **off** | Early stopping makes the noise floor vary per pixel, so it is unknown |
| Sample clamping | Cycles default | **disabled** | Clamping removes energy from the bright highlights being measured |
| Max bounces | 12 | 32 | A reference should not lose energy to a limit it happens to hit |
| Seed | fixed | fixed, `--seed` | Reproducible |

**The error bar.** `--convergence` renders each frame a second time at half the samples with a
different seed and measures the difference. Two independent Monte Carlo estimates of the same
integral differ by about sqrt(2) times the error of the better one, so the full-sample image's own
error is roughly `mean_difference / sqrt(2)`. That figure goes into the sidecar JSON and then into
the comparison table, so a measured MAE can be read against the noise it sits on. If the two are
comparable in size, the difference being reported is the reference's noise and not the raster
pass's error. Without this, "the methods differ by 4 levels" is an unfalsifiable statement.

### Why this comparison is legitimate

Both renderers must describe the *same* scene, or any difference measured between them is
meaningless. Getting that right is most of the engineering in experiment 18.

**One file, two loaders.** The environment is described once in `scene/cafe.scene` — a plain
text list of primitives, camera, sun and sky — and is only ever *loaded*:

```
                    scene/cafe.scene
                     /             \
   src/SceneFile.cpp                tools/scene/scene_loader.py
   (the OpenGL pass)                 (this Blender script)
```

Before this, the scene was written out three separate times: `cubeA`..`cubeG` in `main.cpp`, a
`BOXES` tuple in the Blender script, and regex assertions in a test. Every change had to be made
in three places or the two renderers silently diverged.

**The two loaders are tested against each other.** They are the remaining duplication, so
`tests/scene_file.cpp` and `tests/test_scene_file.py` both assert the same numbers —
primitive/vertex/triangle counts, bounding box, summed vertex positions, summed surface area —
and the Python test additionally checks that those numbers are literally present in the C++
test. Change one builder without the other and exactly one test fails, naming the number that
moved.

**Winding and normals are checked on every triangle.** Cycles decides which side of a face is
the front from its vertex winding; OpenGL uses the interpolated normal. A triangle where the two
disagree is lit in one renderer and black in the other — and that would look like a depth-of-
field difference. Both tests walk all 38,572 triangles and require agreement.

**No image textures anywhere.** Every surface detail is real geometry: mortar courses, roughly
11% protruding bricks, crate panels, railing bars. No texture means no texture-filtering
differences to argue about, and every shadow is a real shadow.

**Each reference records what it was rendered from.** `render_dof.py` writes the scene file's
sha256 into a sidecar JSON beside every PNG. `tools/compare/compare_renders.py` refuses to
compare a reference whose digest does not match the scene file currently on disk, because a
stale reference will still produce plausible-looking numbers.

### Known remaining differences

Honesty here is more useful than a claim of perfection.

| Difference | Why | Size |
|---|---|---|
| Sky occlusion | **Fixed in experiment 20.** Cycles now gets the same unoccluded `albedo * sky` as an emission on every surface, with diffuse bounces off, so the two shading models are identical | None |
| Sun softness | Cycles samples a 0.6-degree sun disc, giving physically correct penumbrae; the shadow map gives a fixed 3x3 PCF softness | Small, at shadow edges |
| Sampling noise | Cycles is stochastic; OpenGL is deterministic | Measured per reference by `--convergence` and reported in the comparison table |
| Bokeh shape | Cycles traces a real circular aperture; the gather approximates it with 100 taps | Small at small radii, visible at large ones |

The ambient-occlusion difference that used to head this list is gone: in an enclosed interior it
was not a small residual but the whole image, because Cycles took its ambient from the world
background and a closed room sees no sky. Both renderers now apply the same unoccluded fill, and
Cycles runs direct-only so it is not double counted. The shading model is therefore identical by
construction and the residual is defocus, sampling noise and bokeh shape only. `--full-gi` swaps
in real global illumination when the question is what the raster pass is missing rather than how
good its depth of field is.

---

## 4. Lighting: making the two agree

This was quietly wrong before and is now physical. `shaders/basic.frag` computes

```glsl
radiance = albedo * (skyRadiance + sunColor * sunEnergy * max(dot(N, L), 0) * (1 - shadow) / PI);
```

- **The `1/PI`** is the Lambertian BRDF normalization. A perfectly diffuse surface scatters
  incoming light over a hemisphere, and integrating that correctly leaves a factor of `1/PI`.
  Its presence is what makes this agree with a Cycles Diffuse BSDF lit by a sun of the same
  strength. Without it the raster image is `PI` times too bright and no amount of tweaking the
  light intensity fixes it consistently across different surface angles.
- **`sunEnergy`** is irradiance in W/m², passed unchanged to Blender's sun strength.
- **`skyRadiance`** is `ambient.color * ambient.strength` from the scene file. That single
  expression is simultaneously the OpenGL clear colour, the OpenGL ambient term, and the Cycles
  world background — which is why the two images share a background colour exactly rather than
  approximately.
- **Emitters** render as `albedo * emit` in both. In Cycles they are additionally marked
  invisible to diffuse rays, so they illuminate nothing, matching the raster pass. Without that
  Cycles would bounce light off every window and bulb and come out far brighter.

All of these numbers live in the scene file, so there is one place to change them.

---

## 5. The scene

A cafe interior. A foreground table at 1.5 m with a cup, saucer, spoon, glass, napkin, book,
shaker and succulent; checkerboard tile floor; five tables with bentwood chairs receding to 9.5 m;
a service bar with a two-group espresso machine, grinders, a pastry dome and three shelves of
bottles and jars; a window wall with daylight panes and a bench of plants; a subway-tiled right
wall; and a back wall at 15 m with a chalkboard menu, clock, neon sign and bottle shelf.
2519 primitives, 74,416 triangles, 3864 of them emissive.

Full sizes and depths: [`notes/graphics/19_cafe_inventory.md`](graphics/19_cafe_inventory.md),
regenerate with `python3 tools/scene/scene_report.py --scene scene/cafe.scene --markdown`.

`scene/alley.scene` (a dusk back-alley, 2551 primitives) is still in the repository and still
loads in either renderer. It is just not the scene stage 2 compares on.

### Composition is driven by the field of view

A 50 mm lens on a 24 mm sensor height gives a **27-degree** vertical field of view. The visible
half-height at distance *d* is `0.24 * d` metres, and the camera is pitched 10 degrees down:

| Depth | Frame height |
|---|---|
| 0.9 m | 0.43 m |
| 1.5 m | 0.72 m |
| 5.6 m | 2.7 m |
| 15 m | 7.2 m |

Two framing attempts were rejected before this one, and both failures are informative:

- **85 mm at 1.6 m**, chosen for maximum blur, gives a 45 cm frame. The cup filled the shot like a
  macro photograph and the entire service bar rendered outside the frame.
- **A bar counter receding from 0.7 m to 10 m.** Because a surface just below eye level has its
  vanishing point at the pitch angle, it filled roughly three quarters of the frame no matter how
  long it was made. A table that *stops* at 1.9 m occupies the bottom third and leaves the room
  visible behind it.

The 10-degree pitch is necessary: with none at all, a surface must sit within about 20 cm of eye
height to be in the near field. It is also the largest pitch that keeps the chalkboard at 15 m
inside the frame.

### Why this scene makes the two methods differ visibly

The same 14 cm saucer appears at 1.5 m (sharp), 3.8 m, 6.5 m and 9.5 m in one frame — four samples
of the circle of confusion with physical size held constant, so the blur is the only variable.
Each pendant lamp carries a 12 mm filament at three times its bulb's radiance; out of focus that
should be a clean even disc, and a gather turns it into a ring-edged blob. The festoon along the
window wall repeats one bulb from 1 m to 15 m. Emitters are deliberately above 1.0 in radiance,
which is why both renderers need a floating-point target.

### Scene file format

```
version 1
camera pos 0 0 5 yaw -90 focus 5 fnumber 1.4 lens 50 sensor 24
sun dir -0.45 0.78 0.44 color 1 0.88 0.72 energy 4.2 angle 0.6
ambient color 0.42 0.52 0.72 strength 0.24
shadow lo -7 -2 -34 hi 7 8 9
box pos 0 -1.55 -12 size 8.4 1 42.5 rgb 0.055 0.055 0.06
cyl pos -2.1 -0.9 -3.2 size 0.56 0.88 0.56 rgb 0.18 0.12 0.08 seg 16
sph pos 1.2 0.7 -0.8 size 0.095 0.115 0.095 rgb 1 0.85 0.6 emit 12
```

Primitives are unit shapes spanning ±0.5 before scaling, so `size` *is* the object's dimensions
in metres. `rgb` is **linear** albedo. `emit` above zero makes a pure emitter. `cyl` also takes
`taper`, which scales the top radius: 1.0 (the default) is a plain cylinder, 0.0 a cone, 1.5 a
cup-shaped flare. That makes the side a frustum with correct smooth normals, which is the only way
to get a turned object — a cup, a pot, a lampshade — without the banding that stacking cylinders
along a profile always produces. Transform order is
`world = pos + Ry*Rx*Rz * (size * local)`, identical in both loaders. Unknown or repeated keys
are errors with a line number, because a silently ignored typo in a shared file is the worst
possible failure here.

---

## 6. How to run everything

```sh
# Build and test
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build

# Run the renderer
./build/DepthResearch

# Regenerate the scene, then refresh the pinned test numbers
python3 tools/scene/build_cafe.py
python3 tests/test_scene_file.py --print-expected

# Preview a scene edit without a GPU or a Cycles render
python3 tools/scene/preview_scene.py --output output/scene_preview.png

# Object sizes and depths
python3 tools/scene/scene_report.py
python3 tools/scene/scene_report.py --scene scene/cafe.scene --markdown \
    --output notes/graphics/19_cafe_inventory.md

# Cycles reference: check the plan first, then render.
# Without --ground-truth this is a quick visual check, NOT training data.
python3 tools/raytraced_reference/render_dof.py --ground-truth --convergence --dry-run
/Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 \
  --python tools/raytraced_reference/render_dof.py -- --ground-truth --convergence --sharp

# Compare
python3 tools/compare/compare_renders.py --write-images
```

### One command

```sh
python3 tools/compare/run_comparison.py              # ground truth, both gathers
python3 tools/compare/run_comparison.py --quick      # 64 samples, minutes, pipeline check
python3 tools/compare/run_comparison.py --dry-run    # print commands and expected filenames
```

That runs the Cycles references, then the renderer in batch mode (`--batch`, hidden window, same
shaders and framebuffer as the UI), then the metrics, stopping at the first failure. The settings
are defined once and passed to both, which matters because a mismatch there does not fail loudly —
it produces a plausible difference image of two different camera setups.

### Producing a comparison set by hand

Wide open and focused close is where the two methods disagree most, because the circle of
confusion is large and every error the gather makes scales with it. f/11 is the control: at a near
pinhole the two pipelines should agree to within the reference's own noise, and if they do not,
something other than depth of field is wrong.

1. Render the references with `--ground-truth --convergence --sharp`. They land in
   `reports/raytraced_dof/` as `rt_focus1.5m_f1.2.exr`, `rt_focus1.5m_f1.4.exr`,
   `rt_focus1.5m_f2.8.exr`, `rt_focus1.5m_f11.exr` and `rt_sharp.exr`, each with a sidecar JSON
   recording the scene digest, the settings and the measured error bar. For the strongest case add
   `--lens 85 --focus 1.2 --fstops 1.2`.
2. Run the renderer, press `T` (reference preset: 50 mm f/1.4 focused 1.5 m), then `6` for
   BasicDoF, then `P`. That writes `output/gl_focus1.5m_f1.4.png` — the name deliberately mirrors
   the reference name so the pair is found automatically.
3. Click f/1.2, f/2.8 and f/11 on the panel, pressing `P` after each. `K` for the strong-blur
   preset, `J` for the all-sharp control.
4. `python3 tools/compare/compare_renders.py --write-images --require-ground-truth` pairs them by
   name, refuses any reference not rendered in ground-truth mode or carrying a different scene
   digest, writes side-by-side and 4x difference images to `reports/comparison/`, and produces a
   metrics table with the reference's own noise beside each measurement.

Both images in a pair must be the same resolution; the references are 1200x1200 by default.

### Controls

Cursor is free on launch. The panel has a **mouse-look button** at the top that switches between
"pointer for the panel" and "camera follows the mouse"; `Tab` does the same from the keyboard.
While the camera has the mouse the panel cannot be clicked, so `Tab` is the way back — the panel
says so in that mode. Holding the right mouse button is still a quick look without toggling.

| Key | Action |
|---|---|
| `1`-`6` | Color, raw depth, linear depth, CoC magnitude, CoC signed, basic DoF |
| `0` | Split: sharp left of the divider, defocused right |
| `7` `8` `9` | Focus 2 / 5 / 15 m |
| `F` `B` | Toggle f/1.4 and f/8 / set f/2.8 |
| `T` | Reference preset, matching the Cycles jobs |
| `K` | Strong blur: 85 mm f/1.4 at 1.6 m |
| `L` | Reload the scene file |
| `R` | Reset camera | 
| `[` `]` `,` `.` | Focal length -/+5 mm, sensor height -/+2 mm |
| `G` | Cycle panel: compact → full → hidden |
| `P` | Screenshot (both `latest.png` and a settings-named file) |
| `Tab` | Free/capture the mouse |
| `H` | Print the key list |

---

## 7. Reading the diagnostic views

| Key | View | What to look for |
|---|---|---|
| `1` | Colour, no blur | The sharp input the gather and the network both start from |
| `2` | Raw depth | Almost all white past a few metres: shows *why* depth must be linearized |
| `3` | Linear depth | Even ramp in metres, scaled by the panel's depth max |
| `4` | CoC magnitude | Black at the focus plane, brightening both ways. The dark band *is* the depth of field |
| `5` | CoC signed | Red = foreground defocus, blue = background. Confirms the sign is right |
| `6` | Basic DoF | The result that gets compared to Cycles |
| `0` | Split | Sharp against blurred in one frame. The two halves must agree at the focus plane; if they do not, the blur is leaking where CoC is zero |

Views 2 to 5 are written without exposure or the sRGB curve, deliberately: they are data, so a
pixel value read off a screenshot still means the number it says it means.

---

## 8. Where this is going

The OpenGL pass is fast and wrong at silhouettes; Cycles is correct and slow. The plan is a
small network that takes the rasterizer's sharp colour, linear depth and analytic CoC — all
things a rasterizer gets for free — and predicts a residual on top of the baseline blur, trained
against matched Cycles frames.

Experiment 18 matters to that plan in three concrete ways: matched pairs are now provably of the
same environment (shared scene file, digest-checked), the inputs are linear HDR rather than
clipped 8-bit, and the baseline is no longer 2x over-blurred. Two decisions are open and written
up in `ai/ARCHITECTURE.md`: whether to train in linear (which needs EXR references instead of
8-bit PNG) and whether to add screen-space AO to the raster pass so the residual is defocus
only.
