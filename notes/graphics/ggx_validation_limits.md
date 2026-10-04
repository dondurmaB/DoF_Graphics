# GGX validation, precision limit, and authoring floor

Measured 2026-10-04 against `bf10e9e`, with Blender 5.2.2 LTS and the local
macOS OpenGL implementation. Preserve this result separately from claims of
whole-scene equivalence. The [complete measurements](ggx_validation_2026_10_04.json)
contain both grids, every case, tolerances, and source SHA-256 fingerprints.

## Two different mappings

The working production pair is isotropic single-scatter GGX with
height-correlated Smith masking and half-vector Schlick Fresnel in GLSL,
against Cycles **Metallic BSDF / GGX / F82**, **white Edge Tint**, scalar **F0
as Base Color**, shared perceptual roughness (`alpha = roughness²`), and zero
anisotropy/thin-film thickness. White Edge Tint removes the F82 correction.
Both explicitly disable the specular lobe at F0=0 and use the same constant
`(1-F0)` diffuse/ambient allocation. See [the formulation](matched_specular_and_geometry.md).

The separate café checkpoint rejected **Glossy Color=F0** (GGX and MULTI_GGX)
against a **separable-Smith/Schlick** candidate. Its recorded 54-case grid had
up to **568.33%** relative radiance error. That rejects that mapping, not all
matched specular. The Glossy color must not be assumed to supply Schlick Fresnel,
and separable and height-correlated Smith are different masking functions.
A Fresnel/Layer Weight mix is also unsuitable: it uses N·V instead of V·H.

## Fresh cross-renderer measurements

The original 24-case production probe was rerun and reproduced its recorded
maximum relative patch error of **0.000161822%**. The expanded **384-case** grid
gave a maximum of **0.000393121%**, with **384/384** cross-renderer checks passing.

Method: actual production GLSL BRDF compiled into a GPU probe, read through
RGBA32F; actual production Cycles material rendered on a flat plane with a
unit zero-angle sun, black world, orthographic camera, 64 CPU samples, no
denoising/adaptive sampling, and 32-bit linear EXR. The Cycles measurement is
mean RGB over all 256 pixels of a **16×16 patch**. Python combines the measured
GPU specular value with analytic diffuse/fill to form expected RGB. This does
not execute the entire production GL material/attribute/FBO path.

For each case, relative error is the largest absolute RGB-channel difference
between patch means divided by the largest expected RGB component (floor 1e-6).
The reported value is the maximum across cases. It is **not a worst-pixel
bound**, not a per-channel-relative bound, not a confidence interval, and not
whole-scene image error. Combined RGB can hide relative errors in a tiny lobe.

The expanded grid crosses eight (light angle, view angle, relative azimuth)
triples in degrees: `(0,0,0)`, `(30,30,180)`, `(60,60,180)`, `(70,20,60)`,
`(25,55,100)`, `(80,75,170)`, `(85,85,180)`, `(89,89,180)`; roughness
`.05/.12/.16/.2/.35/.65/.85/1`; and F0 `0/.008/.04/.6/.7/1`.
This covers sampled N·L and N·V values down to cos(89°), not every continuous input.

## Independent oracle failures and diagnosis

The independent double-precision BRDF check **failed five of the 384 cases**:
roughness **0.05**, symmetric **30° light/view angles**, relative azimuth **180°**,
and F0 `.008/.04/.6/.7/1`. These failures were **not the grazing configurations**.
The largest GPU/oracle relative deviation was **3.70825%**. At F0=.04 the GPU
returned 654.5565186 versus the oracle's 679.7638310. Both renderers nevertheless
agreed closely: renderer agreement does not prove either matches the ideal equation.

Diagnosis: float32 normalization/rounding in `(N·H)²` is amplified through a
narrow GGX peak. A value one float32 step (2^-23) below 1 predicts 3.7082462%
loss in D at roughness=.05, essentially the measured loss. This is a
**numerical diagnosis, not a traced GPU intermediate**. It has not been fixed.

The oracle tolerance is `abs(GPU-oracle) <= 1e-6 + .003*abs(oracle)`.
Cross-renderer acceptance is `max_abs_RGB_error <= 2e-5 + .005*max(expected_RGB)`.
The expanded script exits with failure because of the oracle checks; it must
not be reported as an unconditional pass. Registered CTest and the original
24-case probe passed; the GPU/Cycles probe is not part of CTest.

## Practical authoring rule: roughness >= 0.12

Use **0.12 as the conservative minimum perceptual roughness for authored scene
materials**. Avoid anything shinier until the intermediates have been traced
and a precision change validated in both renderers. The parser still accepts
0.05; this is an authoring rule, not a newly enforced runtime clamp.

A follow-up 520-case sweep used the same eight angular configurations, five
nonzero F0 values, and thirteen roughness values. All 520 cross-renderer checks
passed. The worst sampled oracle error varied as follows:

| Roughness | Maximum relative oracle error | Oracle result |
|---|---:|---|
| 0.05 | 3.70825% | Fail |
| 0.08 | 0.579514% | Fail |
| 0.09 | 0.362377% | Fail |
| 0.0943 | 0.300826% | Fail |
| 0.0944 | 0.299580% | Pass |
| 0.10 | 0.237976% | Pass |
| 0.11 | 0.162612% | Pass |
| 0.12 | 0.114852% | Pass |

For this hardware, implementation, tolerance and sampled configuration set,
the observed crossing is **between 0.0943 and 0.0944**. This brackets the measured
boundary; it is not a proof of a universal continuous threshold. Choosing 0.12
provides margin: its observed worst error is about 2.6 times below the 0.3%
relative allowance. It also passed the wider 384-case grid. The boundary sweep
has 35 oracle failures across its deliberately included lower roughness values.

## Scope and remaining questions

This controlled model does not provide a full metallic workflow, spectral
conductor colors, layered dielectric energy compensation, refraction, indirect
light, or reflected environment detail. Full-GI mode is a different experiment.

Whole-scene equivalence remains unverified: GL PCF shadows differ from Cycles
ray visibility; shading-normal corrections near silhouettes may differ; and
Cycles varies the view vector across the aperture while raster DoF blurs
centrally shaded highlights. The probes use shading normal = geometric normal
and do not establish the production RGBA16F buffer's accuracy, pixel coverage,
attribute interpolation, convergence, or hidden-surface correctness.
