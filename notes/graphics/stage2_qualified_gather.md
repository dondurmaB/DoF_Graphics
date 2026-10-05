# Stage 2 — a qualified production-gather experiment

This experiment compares **Mitsuba pinhole RGB/depth → the actual OpenGL gather**
against **Mitsuba thin-lens RGB**. Both images originate from the same loaded
scene, geometry, materials, light and camera pose. No OpenGL café scene render
is used as an input or reference. The independently authored cafés have no
validated material match; the earlier approximately 24× local material-response
mismatch is exactly why subtracting their images would be invalid.

The [one-command guide](../../renderer/mitsuba/STAGE2.md) defines how to rerun it.
[Complete evidence](../../experiments/graphics/stage2_qualified_gather/), including
float32 EXRs, sidecars, paired seeds, difference images, masks and logs, is
preserved in the repository. Settings are specified in `stage2.json`.

## Scope: a controlled café, not a photorealism claim

The final scene retains the existing Mitsuba café geometry and transforms. It
uses opaque diffuse materials, preserving diffuse/base-colour textures where
available and assigning neutral diffuse reflectance to former glass/metals.
One fixed point light at `(0.34, 1.22, 1.85)` has RGB intensity `(15,15,15)`.
The environment and decorative lamps do not emit. The light stays fixed during
the wide-angle inspection. Path tracing uses maximum depth 2, roulette depth 6
(which is never reached), and `hide_emitters=false`. There are no indirect
bounces, glossy lobes, refraction, or denoising in this controlled experiment.

This restriction was necessary to obtain a defensible result, not to hide a
failed convergence check. At 8,192 spp, the original full-lighting café had
estimated reference noise **0.013847 linear RMS**, versus naive-gather MAE
**0.013690**. It failed qualification. Its images and failed report remain in
`output/stage2/qualified-8192`; diagnostic numbers are also preserved beside the
final evidence. An initial direct-sun trial and a 65,536-spp independent-sampler
trial also remain as failed qualification evidence. The final scene is therefore
not offered as a converged reference for the original full-GI café.

The benchmark uses 640×360 pixels, an 85 mm lens, a 24 mm sensor height, focus
at 1.887372 m, and f/1.2 versus f/22. A separate 28 mm view from the front corner
shows the room, table, chairs, lamps, shelves and floor. It was visually inspected,
but does not prove every surface is modelled correctly. The main view deliberately
crops the lower foreground; the overview makes that framing explicit.

## CoC and the actual shader

Mitsuba's fixed-FOV camera sends each aperture sample through a focus-plane
point. With focal length `f`, f-number `N`, focus depth `s`, object depth `z`,
sensor height `H`, and image height `h`, its signed pixel radius is:

```text
r = 0.5 × (f/N) × f × (z − s)/(z × s) × h/H
```

Lengths are metres; `z` is optical-axis depth, not range from the camera.
The factor 0.5 converts diameter to radius. The previous formula used `s−f`
instead of `s`, making its footprint too large by `s/(s−f)`: approximately 2.72%
in the audited 50 mm case. This was a focus/FOV convention mismatch, not the
old diameter/radius mistake. Regression tests now call Mitsuba `sample_ray`
and check the footprint at foreground, focus and background depths, including
the original 10 m case. Another test executes the production GLSL CoC function.

The experiment compiles `shaders/dof_scene/screen.frag` itself. It does not use
the former Python blur port. RGB is uploaded as linear RGB32F with bilinear
sampling; planar depth is R32F with nearest sampling. CoC is evaluated **after**
the depth fetch. Output is RGBA32F with linear readback and no display transform.
Array upload/readback preserve native OpenGL texture orientation.

Both variants use exactly **100 disk contributions** for a blurred pixel:
equal weights for `naive`, the existing depth/foreground/footprint heuristic
for `weighted`. Radii below half a pixel return the sharp centre. The cap is
**120 px**. At f/1.2 the largest visible radius is **19.0953 px**, and the
infinity limit is **23.9255 px**: **zero pixels are cap-clipped**. At f/22 the
largest visible radius is **1.04156 px**. That is a near-pinhole control, not a
claim that DoF vanishes. A separate GPU test exercises support beyond the former
32-pixel cap. These images do not establish image quality at a full 120-pixel
physical footprint.

The native café's unrelated GGX denominator floor was also corrected. At
roughness 0.15, GPU evaluation now agrees with `1/(πr⁴) ≈ 628.7603`, rather than
5.0625. Runtime roughness is at least 0.08, so the stable denominator is positive;
no flattening clamp is necessary. This change is not a Mitsuba material match.

## What qualifies the reference

Each beauty frame uses **131,072 spp per seed**, independently randomized
multijitter sampling, chunks of 256 spp, a box reconstruction filter, and linear
float32 EXR output. There are two sharp realizations reused at both stops and
two thin-lens realizations per stop. Depth is the first intersection along the
pixel-centre ray; it is not a filtered mixture of foreground/background depths.

Reference **A** is compared with gather outputs. B is an independent equal-N
replicate, not averaged into A. For independent equal-variance estimates,
`Var(A−B)=2 Var(A)`. The report therefore gives difference standard deviation
**divided by sqrt(2)**, and uses difference RMS divided by sqrt(2) as the more
conservative spatial/channel aggregate noise estimate. There is no N-versus-N/2
comparison and no sqrt(3) factor. These are empirical noise estimates, not 95%
confidence intervals, per-pixel bounds, or measurements of systematic bias.

Both sharp realizations are also passed through each gather. Their output
pair estimates propagated input noise. Qualification requires
`hypot(reference_noise, propagated_input_noise) <= 0.10 × measured_MAE`
for **every** variant and region. All eight final rows pass; the largest ratio
is **4.851%**, and the smallest is **0.254%**. A measured MAE is meaningful here
because it is well above the measured noise, not because 131,072 is a large
sample count.

Every image has an authenticated sidecar: file, scene, camera, context and
source hashes, integrator settings, sampling settings and seeds. Gather outputs
identify their variant and exact RGB/depth/shader inputs. Qualified reference
sidecars include their measured error estimate. Missing, stale, modified or
wrong-sized inputs fail; an empty comparison fails. Noise qualification is
recomputed on comparison. No result is published for an unqualified run.

## Results and observed failure

Errors below are **scene-linear RGB**, without exposure, clipping or tone mapping.
PSNR uses a fixed peak of **1 linear radiance unit**, including HDR data; it is
not per-image normalized. “Edges” means an adjacent depth ratio above 1.3,
marked on both sides and dilated by seven pixels: 16.6532% of the image.
Noise is estimated RMS for the reference A image; the full JSON also reports
centred standard deviation and propagated input noise.

| Stop | Gather | Region | MAE | RMSE | PSNR dB | Reference noise RMS |
|---|---|---|---:|---:|---:|---:|
| f/1.2 | Naive | Whole | 0.00449904 | 0.0196921 | 34.114 | 0.0000271941 |
| f/1.2 | Weighted | Whole | 0.00163810 | 0.00797282 | 41.968 | 0.0000271941 |
| f/1.2 | Naive | Edges | 0.0219508 | 0.0466005 | 26.632 | 0.0000555443 |
| f/1.2 | Weighted | Edges | 0.00634274 | 0.0184872 | 34.663 | 0.0000555443 |
| f/22 | Naive | Whole | 0.000412999 | 0.00293858 | 50.637 | 0.0000131815 |
| f/22 | Weighted | Whole | 0.000292105 | 0.00153705 | 56.266 | 0.0000131815 |
| f/22 | Naive | Edges | 0.00127950 | 0.00705493 | 43.030 | 0.0000242645 |
| f/22 | Weighted | Edges | 0.000716262 | 0.00354879 | 48.998 | 0.0000242645 |

The wide-stop naive image visibly spreads bright flower, stem and vessel colours
onto the background. Depth weighting reduces whole-frame MAE by about **63.6%**
and edge MAE by **71.1%**, but leaves hard, irregular background silhouettes and
incorrect transitions. The thin-lens reference resolves these boundaries more
smoothly. The f/22 comparison is substantially closer, with small remaining
edge/filter errors; it is not numerically identical to a pinhole image.

Neither weighting nor more disk taps can reconstruct a surface hidden in the
input's single colour/depth layer. Different aperture positions can see such a
surface, whereas the gather has no samples of it. That missing information is
the irreducible limitation motivating the learned stage. These aggregate
metrics do not separately quantify the hidden-surface contribution versus
kernel, support, sampling and depth-discontinuity errors.

## Verification and remaining limits

Build passed. CTest passed **5/5 suites**, covering 15 native checks and 15 Python
test methods. Tests include actual sensor rays, native-depth versus linear-depth
shader execution, a scalar 100-tap check, HDR/orientation, 120-pixel support,
GGX peak, provenance/pairing failures, insufficient noise qualification, and
real PNG plus separate rendering/gather-optics snapshot metadata. The scalar
sampling check uses a 0.0001 absolute tolerance on unit-range colours; it is not
bit-exact equivalence or a worst-pixel guarantee. No Python port supplies the
benchmark images. The legacy browser gather remains explicitly a preview.

The updated native renderer linked its shaders, completed its FBO and produced
a 30-frame capture. Its separate cursor-driven `--verify` failed at the HUD
mode-button assertion; it is **not** reported as a passed interaction check.
Browser live interaction and other GPU backends are **not verified**. The
qualified experiment is offscreen and does not depend on that HUD.

Still approximate: single centre-sampled depth versus box-filtered colour,
finite 100-tap quadrature, heuristic depth weights, edge clamping, and the
half-pixel early exit. The sensor omits diffraction, aberrations and physical
exposure variation. Full-GI/glass/glossy café convergence, worst-pixel bounds,
all-view modelling correctness, and any learned model are **not verified**.
No requested bug fix or gather variant was cut. The explicit scope reduction
is the controlled opaque/direct-light transport model; its results must not be
relabelled as full-lighting café ground truth.
