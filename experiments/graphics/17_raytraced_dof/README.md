# Experiment 17 — Lens-Sampled Ray Tracing

## Purpose

Experiment 16 evaluates the aperture integral at N fixed points, which leaves N discrete ghost copies. This experiment solves the same integral by sampling the aperture **stochastically, per pixel, per frame**, and establishes the reference the other two methods are measured against.

`raytrace.frag` traces one ray per pixel per frame from a random point on the aperture disc through that pixel's point on the focus plane, and adds the result to an accumulation buffer:

```glsl
float angle = 6.28318530718 * r1;
vec2 lens = vec2(cos(angle), sin(angle)) * (uApertureRadius * sqrt(r2));
vec3 focalPoint = uEye + uForward * uFocusDistance + uRight * (a * uFocusDistance) + uUp * (b * uFocusDistance);
vec3 origin    = uEye + uRight * lens.x + uUp * lens.y;
vec3 direction = normalize(focalPoint - origin);
```

The `sqrt(r2)` is what makes samples uniform over the disc's *area* rather than crowding the centre, and the frame index is mixed into the hash so successive passes do not land on correlated aperture positions — without it the accumulation reads as concentric banding instead of noise.

Every sample runs its own independent visibility solve, so hidden surfaces, correct occlusion ordering and true bokeh shape come out of the method rather than being approximated by it. The cost is noise until enough samples accumulate.

## Same Geometry, Two Solvers

The comparison is only meaningful because the rasterized and ray-traced paths describe the *same* scene. `scene.hpp` stores every solid as an analytic primitive; `scene.cpp` tessellates them for the rasterizer while `raytrace.frag` intersects them directly, and both shade through the same `shading.glsl`. The three methods therefore differ in exactly one respect: how visibility is resolved.

That equivalence is testable. Render both paths at a near-pinhole aperture, where depth of field should vanish and the two must agree:

```sh
BIN=./build/DoFApproaches.app/Contents/MacOS/DoFApproaches
$BIN --method 1 --f-number 22 --size 640x430 --capture /tmp/pinhole-raster.png
$BIN --method 3 --f-number 22 --samples 512 --size 640x430 --capture /tmp/pinhole-raytrace.png
```

They agree to a mean absolute error of **0.518 of 255**. That is the floor of this whole comparison, and it is not zero for two legitimate reasons: the rasterizer draws spheres and cylinders as 32-segment tessellations while the ray tracer intersects the exact surfaces, and neither path antialiases primary edges. Any measured difference below roughly 0.5 is geometry, not depth of field.

## Run It

```sh
BIN=./build/DoFApproaches.app/Contents/MacOS/DoFApproaches

$BIN --method 3 --samples 32  --size 1280x860 --capture output/dof3-m3-raytrace-32.png
$BIN --method 3 --samples 512 --size 1280x860 --capture output/dof3-m3-raytrace-512.png
```

`--samples` is capped at 512 here. Interactively the image refines progressively and the window title reports the accumulated count; any camera, focus or aperture change calls `resetAccumulation()` and starts over.

## Findings

Mean absolute error against the 512-sample reference:

| Method | Overall | Zone A (halo) | Zone B (near occluder) | Highlights |
|---|---|---|---|---|
| 1: gather, 64 taps | 4.946 | 2.925 | 12.641 | 4.014 |
| 2: multi-view, 64 views | 0.682 | 0.558 | 1.364 | 0.626 |
| 3: ray traced, 32 spp | 2.002 | 1.608 | 4.113 | 1.918 |

At 32 samples the ray tracer is *worse* than 64-view multi-view, because 32 stochastic samples are noisy while 64 deterministic views are merely banded. The advantage is not per-sample quality — it is that the error is noise rather than structure. Noise falls as 1/sqrt(N), is uncorrelated between neighbouring pixels, and reads to the eye as grain; banding is correlated, forms visible ghost copies of the geometry, and does not read as anything physical. The defocused highlights in `dof3-m3-raytrace-512.png` are clean round discs, where both other methods produce lumpy polygonal blobs from their discrete sample patterns.

## Two Bugs This Experiment Found

Building a reference is also a way of testing everything the reference is compared against. Two defects were invisible until a stochastic solver ran over the same scene.

**1. An ambiguous checker in shared shading.** The ground plane lies exactly on a checker cell boundary at y = 0. The rasterizer interpolates that y as exactly 0.0 and stays on one side of `floor(position / scale)`; the ray tracer computes it as `ro.y + rd.y*t` and lands on either side depending on rounding, so successive aperture samples disagreed about the parity and the ground averaged to a flat grey. Shifting cell boundaries off the world axes with `floor(position / scale + 0.5)` fixed it, and dropped pinhole disagreement from 6.35 to 0.65.

**2. An accumulation buffer that could not accumulate.** The accumulation target was `GL_RGBA16F`. Once the running total is large, fp16's 10 mantissa bits round each new sample's contribution away, so the average stops converging — and gets *worse* with more samples, which is the opposite of what a reference must do:

| Frames accumulated | Error, RGBA16F | Error, RGBA32F |
|---|---|---|
| 5 | 0.556 | 0.558 |
| 20 | 0.715 | 0.528 |
| 60 | 0.964 | 0.520 |
| 150 | 1.709 | 0.519 |
| 400 | 3.529 | **0.518** |

The rasterized path was bit-identical at 3 frames and 514 frames, which is what isolated the fault to accumulation rather than to input or camera drift. Method 2 accumulates into the same target and benefits equally.

## Cost and Limits

One visibility solve per sample per pixel, against all scene primitives; the ray tracer's uniform arrays cap the scene at 64 primitives (64 x 3 vec4 is 768 uniform components, under the 1024 GL 3.3 guarantees for the fragment stage). There is no acceleration structure — every ray tests every primitive — so this stays a reference for small analytic scenes, not a renderer.

It resolves the aperture exactly and primary visibility exactly. It is still direct lighting with one shading model: no reflection, refraction or indirect light. It is the reference *for depth of field*, not for light transport.

## Verification

| Check | Result |
|---|---|
| Build | Clean with `-Wall -Wextra -Wpedantic` |
| Raster/ray-trace equivalence | 0.518 mean absolute error at f/22, where depth of field vanishes |
| Convergence | Monotone improvement with accumulated samples after the RGBA32F fix; previously monotone degradation |
| Captures | `output/dof3-m3-raytrace-32.png`, `output/dof3-m3-raytrace-512.png` |

## Related Notes

[Experiment 17 notes](../../../notes/graphics/17_raytraced_dof.md)
