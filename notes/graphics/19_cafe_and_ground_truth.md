# Experiment 19: café scene, ground-truth references, full aperture ladder

## What changed
- `scene/cafe.scene` replaces the alley as the stage-2 comparison scene. 2519 primitives,
  74,416 triangles, 3864 emissive. Built by `tools/scene/build_cafe.py`.
- `tools/scene/authoring.py`: the writer, RNG, group and segment helpers, extracted so both
  scene builders share one copy. The alley still regenerates byte-identically.
- `tools/raytraced_reference/render_dof.py --ground-truth`: denoiser off, 4096 samples,
  32-bit linear EXR, raised bounce limits, no clamping, no adaptive sampling.
  `--convergence` gives every reference a measured error bar.
- The imported teapot is off by default in both renderers. The OBJ path is kept.
- Aperture is now a full ladder, f/1.2 to f/16, with the blur ceiling raised from 120 to 320 px
  and the tap ceiling from 256 to 512.
- `tools/compare/compare_renders.py` reads EXR references, prefers them over PNG, and reports
  the reference's own noise beside the measured difference.

## On "the ray tracing is not good enough"

Worth separating two claims. Cycles **is** a physically based path tracer; it is not the wrong
tool, and replacing it would mean giving up a validated renderer for an unvalidated one. But the
way it was being *driven* did disqualify its output as ground truth, for three reasons:

1. **The denoiser was on.** A denoiser is a plausible-detail generator. Training against its
   output teaches a network to match a guess, and large smooth out-of-focus discs are exactly the
   regions a denoiser rewrites most aggressively.
2. **8-bit PNG through a view transform.** Every value above 1.0 clipped, everything quantized to
   256 levels. The bright highlights whose bokeh *is* the measurement were the first casualty.
3. **No convergence evidence.** 128 samples with no stated error, so a measured difference between
   the two renderers could not be distinguished from the reference's own noise.

"Absolute performance" is therefore a claim about convergence and provenance, not about the
algorithm. `--ground-truth` sets:

| Setting | Default | Ground truth | Why |
|---|---|---|---|
| Samples | 128 | 4096 | Noise floor roughly 5.7x lower (1/sqrt(N)) |
| Denoiser | OpenImageDenoise | **off** | Never train against invented detail |
| Output | PNG 8-bit sRGB | **EXR 32-bit linear** | No clipping, no quantization |
| Adaptive sampling | on | **off** | Early-stopping makes the noise floor vary per pixel, so it is unknown |
| Sample clamping | Cycles default | **disabled** | Clamping removes energy from exactly the bright highlights being measured |
| Max bounces | 12 | 32 | A reference should not lose energy to a limit it happens to hit |
| Seed | 16 | fixed, `--seed` | Reproducible |

`--convergence` renders each frame a second time at half the samples with a different seed and
reports the difference. Two independent Monte Carlo estimates of the same integral differ by about
sqrt(2) times the error of the better one, so the full-sample image's own error is roughly
`mean_difference / sqrt(2)`. That number lands in the sidecar JSON and then in the comparison
table, so **a measured MAE can be read against the noise it sits on**. If the two are comparable,
the difference being reported is the reference's noise, not the raster pass's error.

One honest caveat kept in place: emitters are still invisible to diffuse rays by default, so they
light nothing in either renderer. That is deliberate — it isolates the lens, which is the
experiment. `--full-gi` removes the restriction and is physically complete, but then a
raster-versus-Cycles difference measures global illumination as well as defocus and the two are no
longer separable. Use it to show what the raster pass is missing, not to produce DoF training
pairs.

## The café scene

A foreground table at 1.5 m with a cup, saucer, spoon, glass, napkin, book, shaker and succulent;
checkerboard tile floor; five tables with bentwood chairs receding to 9.5 m; a service bar with a
two-group espresso machine, grinders, a pastry dome and three shelves of bottles and jars; a window
wall with daylight panes and a bench of plants; a subway-tiled right wall; and a back wall at 15 m
with a chalkboard menu, clock, neon sign and bottle shelf. Sizes and depths per section:
[`19_cafe_inventory.md`](19_cafe_inventory.md).

### Two framing mistakes, and what they teach

**85 mm was wrong.** The first version used 85 mm focused at 1.6 m, for maximum blur. At that
framing the visible half-extent is 0.141 × depth, so the frame at the subject is 45 cm × 45 cm: the
cup filled the shot like a macro photograph, and the entire service bar was rendering outside the
frame. 50 mm gives 0.24 × depth, which keeps a strong blur and lets the room open up with depth.

**A receding counter cannot work.** The second version had a bar counter running from 0.7 m to 10 m
of depth. Because the surface sits just below eye level, its vanishing point is at the pitch angle,
so it filled roughly three quarters of the frame *regardless of how long it was made*. That is
geometry, not a tuning problem. A table that stops at 1.9 m occupies the bottom third.

The camera is pitched 10° down. With no pitch, a surface has to sit within about 20 cm of eye
height to appear in the near field at all, which forces a chin-on-the-counter view. 10° is the
largest pitch that still keeps the chalkboard at 15 m inside the frame.

### Why this scene makes the difference visible

The same 14 cm saucer appears at 1.5 m (sharp), 3.8 m, 6.5 m and 9.5 m in one frame: four samples
of the circle of confusion in a single image, with physical size held constant so the blur is the
only variable. Each pendant lamp carries a 12 mm filament at three times the bulb's radiance —
out of focus that should be a clean even disc, and a gather turns it into a ring-edged blob. The
festoon repeats one bulb from 1 m to 15 m along the window wall. Emitters above 1.0 are the reason
both renderers need a float target.

## Apertures and blur

| Lens | f-number | Focus | CoC radius at the back wall, 15 m (1200 px frame) |
|---|---|---|---|
| 50 mm | f/11 | 1.5 m | 3.5 px |
| 50 mm | f/2.8 | 1.5 m | 13.9 px |
| 50 mm | f/1.4 | 1.5 m | 27.7 px |
| 50 mm | f/1.2 | 1.5 m | 32.3 px |
| 85 mm | f/1.2 | 1.2 m | **124 px** |

The ladder is f/1.2, 1.4, 1.8, 2, 2.8, 4, 5.6, 8, 11, 16, as buttons rather than a slider: a
slider cannot be set to exactly f/1.4 twice in a row, and every Cycles reference renders at a
named stop. `=` opens up a stop, `-` stops down, `K` loads 85 mm f/1.2 at 1.2 m, and `J` loads
f/16 as the all-sharp control.

The ceiling went from 120 to 320 px because the wide end of the ladder genuinely exceeds 120, and
the old ceiling clipped it — which looks like the blur mysteriously "stopping" past a certain
depth. The tap ceiling went to 512 because a 300 px radius at 160 taps is visibly undersampled.
That undersampling is itself a real finding and is left visible rather than hidden.

## Which comparisons to run

Wide open and focused close is where the two methods disagree most, because the circle of
confusion is large and every error scales with it. f/11 is the control: at a near pinhole the two
pipelines should agree to within the reference's own noise, and if they do not, something other
than depth of field is wrong.

```sh
blender --background --python-exit-code 1 \
  --python tools/raytraced_reference/render_dof.py -- --ground-truth --convergence --sharp
# then, for the strongest case:
#   ... -- --ground-truth --convergence --lens 85 --focus 1.2 --fstops 1.2
python3 tools/compare/compare_renders.py --write-images --require-ground-truth
```

## Verify
```
cmake --build build && ctest --test-dir build
python3 tools/scene/build_cafe.py
python3 tests/test_scene_file.py --print-expected
python3 tools/scene/scene_report.py --scene scene/cafe.scene
python3 tools/scene/preview_scene.py --scene scene/cafe.scene --output output/cafe.png
python3 tools/raytraced_reference/render_dof.py --ground-truth --convergence --dry-run
```
