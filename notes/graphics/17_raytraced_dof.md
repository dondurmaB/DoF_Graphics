# Experiment 17 — Lens-Sampled Ray Tracing

Review time: approximately 4–5 minutes.

## The One Sentence

The same aperture integral as Experiment 16, but sampled randomly per pixel — which turns structured ghosting into noise, and noise is both better behaved and better looking.

## Same Integral, Different Sampling

Experiment 16 evaluates `I(p) = (1/A) ∫ L(p, u) du` at N *fixed* aperture points, shared by every pixel. This experiment picks the aperture point **randomly, per pixel, per frame**:

```glsl
float angle = 6.28318530718 * r1;
vec2 lens = vec2(cos(angle), sin(angle)) * (uApertureRadius * sqrt(r2));
vec3 origin    = uEye + uRight * lens.x + uUp * lens.y;
vec3 direction = normalize(focalPoint - origin);
```

Two details carry real weight. The `sqrt(r2)` makes samples uniform over the disc's *area*; without it they pile up at the centre and the bokeh is wrong. And the frame index is mixed into the hash, so successive passes do not reuse correlated aperture positions — skip that and the accumulation shows concentric banding instead of noise.

The geometry is the same as method 2's: `focalPoint` is where this pixel's ray must cross the focus plane, so all aperture positions agree there and the focus plane stays sharp. That is the shear's job done in ray form.

## Why Decorrelation Is the Point

At 32 samples the ray tracer measures *worse* than 64-view multi-view (2.002 against 0.682). That is not a contradiction — it is the whole distinction.

Multi-view's error is **correlated**: all pixels share the same 64 aperture points, so the error forms coherent ghost copies of the geometry. The eye is extremely good at spotting repeated structure, so those artifacts read as wrong immediately.

Ray tracing's error is **uncorrelated**: neighbouring pixels drew different aperture points, so the error looks like grain. It falls as 1/sqrt(N), and the eye tolerates it, because film grain is a thing the visual system already knows how to ignore. Same error magnitude, very different perceived quality.

It also samples the aperture *continuously*, so defocused highlights come out as clean round discs rather than the lumpy polygons both sampled methods produce from their discrete patterns.

## The Comparison Is Only Valid Because the Geometry Is Shared

`scene.hpp` stores every solid as an analytic primitive. `scene.cpp` tessellates them for the rasterizer; `raytrace.frag` intersects them directly; both shade through the same `shading.glsl`. The three methods therefore differ in exactly one respect — how visibility is resolved — which is what makes subtracting their images meaningful at all.

That is testable, not assumed. Render both paths at f/22, where depth of field should vanish: they agree to **0.518 of 255**. Not zero, for two honest reasons — the rasterizer's spheres are 32-segment tessellations while the ray tracer's are exact, and neither path antialiases edges. So 0.518 is the floor of the whole comparison, and any measured difference near it is geometry, not depth of field.

## Two Bugs a Stochastic Solver Found

Building a reference turns out to be a way of testing everything you compare against it. Both of these were invisible to the rasterizer.

**The checker had no defined answer on the ground.** The ground plane sits exactly on a checker cell boundary at y = 0. The rasterizer interpolates that y as exactly 0.0 and lands consistently on one side of `floor(position / scale)`. The ray tracer computes it as `ro.y + rd.y*t`, so rounding puts it on either side — successive aperture samples disagreed about the parity and the ground averaged to flat grey. Shifting boundaries off the world axes (`floor(position / scale + 0.5)`) fixed it and dropped pinhole disagreement from 6.35 to 0.65.

The general lesson: a procedural pattern evaluated exactly on its own discontinuity is undefined, and only a solver that computes the position twice will tell you.

**The accumulation buffer could not accumulate.** The target was `GL_RGBA16F`. Once the running sum is large, fp16's 10 mantissa bits round each new sample's contribution to nothing, so the average stops converging and actively *degrades*:

| Frames | RGBA16F | RGBA32F |
|---|---|---|
| 5 | 0.556 | 0.558 |
| 60 | 0.964 | 0.520 |
| 400 | 3.529 | **0.518** |

A reference that gets worse the longer you run it is worse than useless. What isolated it was that the rasterized path was bit-identical at 3 frames and at 514, which ruled out camera drift and input and left only the buffer the two paths share. Method 2 accumulates into the same target and improved with the fix.

Worth keeping: **half floats are for storing one frame, not for summing many.** The moment a buffer's job is accumulation, its precision requirement is set by the ratio of the running total to the smallest meaningful increment.

## Cost and Honest Limits

One visibility solve per sample per pixel, tested against every primitive — there is no acceleration structure, and the ray tracer's uniform arrays cap the scene at 64 primitives (64 x 3 vec4 = 768 uniform components, under the 1024 GL 3.3 guarantees for fragments). This is a reference for small analytic scenes, not a renderer.

And it is a reference **for depth of field only**. It resolves the aperture exactly and primary visibility exactly, but it is still direct lighting with one shading model: no reflection, no refraction, no indirect light. Calling it "ground truth" without that qualifier would be overclaiming.

## The Three Methods in One Table

| | Method 1: gather | Method 2: multi-view | Method 3: ray traced |
|---|---|---|---|
| Visibility solves | 1 | N | N per pixel |
| Blur comes from | a CoC formula + filter | parallax between views | parallax between rays |
| Sharp focus plane | by choosing radius ≈ 0 | structurally | structurally |
| Sees behind occluders | never | yes | yes |
| Error type | **bias** | banding (ghost copies) | noise |
| More samples | does not fix it | 1/N | 1/sqrt(N) |
| Error at 64 samples | 4.946 | 0.682 | — |
| Cost | 1 extra pass | N scene renders | N x primitives per pixel |

## Checkpoint

- [ ] Explain why 32 stochastic samples score worse than 64 fixed views, and why you might still prefer them.
- [ ] Explain what `sqrt(r2)` is for and what the bokeh looks like without it.
- [ ] Explain why 0.518 rather than 0 is the right floor for this comparison.
- [ ] Explain what "biased estimator" means for method 1, in terms of the table above.

## Related

[Experiment 17 code and findings](../../experiments/graphics/17_raytraced_dof/) · [Experiment 15 — Where It Breaks](15_screen_space_halo.md) · [Experiment 16 — Multi-View](16_multiview_dof.md)
