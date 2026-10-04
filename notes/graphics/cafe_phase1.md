# Café checkpoint: Phase 1, with matched-light validation

Status: stopping at the Phase 1 review point authorized in the brief. Scene/model work and the lighting needed to judge it are implemented and inspected. Phases 3–5 are **not complete**: there is no convergence-certified ground truth, improved gather, or complete reference/capture/comparison orchestrator. Some shared settings, capture, filename and comparison fixes needed for inspection are already in place.

Implementation is in `dof-research-stage2-work`, copied from the accepted `dof-research 3`. That baseline and all previous experiment archives remain unchanged. No rejected café folder was used. The separate repository clone is checked out cleanly at that commit in detached HEAD state; no branch base has been chosen. No push, merge or PR was made. Local commits await the requested confirmation of how to reconcile the baseline with the cloned repository's `origin/main` at `51b5e50f88e50946ff682c53228563e1a695c0fd`.

## Resolution finding

The baseline requested a **1200×1200 logical GLFW window**, not a fixed capture. Its saved OpenGL PNGs, Cycles PNGs/JSON, and Python defaults were **2400×1662 pixels**. Launching the unchanged application on this machine actually gave **2400×1654** because Retina scaling and the window manager affect the drawable. Neither rectangular height was a stable scene setting.

Both renderers now read `capture width 1200 height 1200` from `scene/cafe.scene`. Batch OpenGL allocates a presentation FBO in pixels, uses the normal scene/shadow/postprocess passes, queries its attachment dimensions, and fails on mismatch/incompleteness. Verified PNG headers and FBO logs: **1200×1200**, plus **321×245** for an odd non-square test. Interactive windows still follow the actual drawable size.

For 85 mm, f/1.2, focus 1.6 m, the infinity CoC radius is **99.3537 px at H=1200**, but **137.6049 px at H=1662**. The old 120 px cap would clip the latter. The new ceiling is derived from the selected lens/focus/resolution and the widest ladder stop (or a wider explicit CLI value). Since signed CoC is monotonic in `1/depth`, its maximum magnitude on `[near,far]` occurs at an endpoint. For the café at 1200 px, f/1.2 and focus 2.5 m, that conservative near/far bound is **510.2041 px**; it is **not** a claim that any café pixel blurs that much. The UI now says near/far bound.

## Framing and geometry

The complete arithmetic precedes placement in `tools/scene/build_cafe.py`'s docstring. At 50 mm / 24 mm, full frame height and square-frame width equal `0.48 × view depth`:

| Depth (m) | Full extent (m) |
|---:|---:|
| 0.9 | 0.432 |
| 1.5 | 0.720 |
| 2.0 | 0.960 |
| 2.5 | 1.200 |
| 4 | 1.920 |
| 6 | 2.880 |
| 9 | 4.320 |
| 12 | 5.760 |
| 15 | 7.200 |

With eye `(0,0,5)`, downward pitch `p`, and `dz=5-z`, view depth is `D=dz cos(p)-y sin(p)` and screen vertical is `V=dz sin(p)+y cos(p)`. The 78 cm table is 72 cm below the eye and ends after 88 cm. Its near edge stays in frame for pitch above about 6.4°; useful back-wall content at y=1.2 m stays in frame below about 8.9°. Chosen pitch: **8°**, deliberately cropping the highest ~16 cm of wall.

The 110 mm cup has a 70 mm base and 94.5 mm top. Five identical cups sit at measured view depths **2.5, 4, 6, 9, 12 m**. Each body is one frustum, with a separate saucer, coffee surface and rolled rim. Small emitters at additional depths have radiance up to **20**. Both final sharp linear buffers were measured to retain a peak of **20.0**, confirming that the HDR path did not clip them to 1. Foreground foliage is around 1.3 m. The service counter is lateral and distant; it is not a receding near-eye bar.

Near cup silhouettes use 64 segments (~0.057 px radial sagitta at 1200 px); the hero table uses 128 (~0.13 px). Round stock spans endpoints using the scene's actual `Ry*Rx*Rz` convention. Brightness jitter is scalar by default. Leaves are thick, smooth ellipsoids. Selected box edges have physical-metre chamfers. Bent stock is still a segmented approximation, not a manufactured CAD surface.

The tapered-cylinder normal follows the profile derivative: `normalize(cos(t), r_bottom-r_top, sin(t))` before inverse-transpose scaling. Flat polygonal frusta additionally project the radius difference by `cos(pi/segments)`; a perpendicularity test protects that distinction. Cone tips omit degenerate triangles. A default taper of 1 follows the old cylinder path numerically: explicit/implicit defaults are byte-compared in both test files. A separate direct check against untouched baseline loaders found identical C++ serialized vertex/index output and Python float64 buffers/topology. Original-cylinder dump SHA-256: `db59b4b550350ee4a95e6faefab052ced1694a239efbd22bbc6d93d643f42df0`.

The OBJ is opt-in (`--import-mesh`) and off by default. Its position, scale, rotation and albedo are read from the shared `imported` entry. OBJ source units are arbitrary; the declared transform maps them into **1 world unit = 1 metre**.

## Shading: what matches and what does not

For a non-emissive surface, both paths target

```
L = albedo * (A + E_sun * C_sun * max(N dot L_sun, 0) * visibility / pi)
A = ambient_color * ambient_strength
```

Emitters instead return `albedo * emission`; the background returns `A`. OpenGL inspection uses intensity/exposure 1. The café sun has zero angular diameter so the intended direct source is directional in both renderers.

Cycles matched mode uses Lambert diffuse plus **primary-camera-only unoccluded fill emission**, disables mesh-light sampling for that fill, uses a camera-visible-only world, and disables indirect diffuse/glossy/transmission continuation. The fill is intentionally non-physical. Simply setting diffuse bounces to zero and disabling mesh-light sampling was insufficient: ordinary fill emission still brightened the café by about 12–23% in sampled regions. Restricting its visibility to primary camera rays removed that error.

Reproduction: `python tools/verify/measure_lighting.py --render`. Radiance measurements use GL's pre-lens half-float buffer read as float32 and Cycles' sharp 32-bit linear EXR/PFM. Calibration uses 64² pixels, 32 samples, no denoiser; café checks use 1200², 128 samples, no denoiser. RGB means:

| Patch | OpenGL | Cycles | Largest channel relative difference |
|---|---|---|---:|
| Ambient only | .0999756, .0799561, .0599976 | .1000000, .0800000, .0600000 | 0.0549% |
| Unit sun only | .1590576, .1273193, .0954590 | .1591549, .1273239, .0954930 | 0.0611% |
| Both | .2590332, .2072754, .1553955 | .2591549, .2073240, .1554930 | 0.0627% |
| Sealed room, matched fill | .0999756, .0799561, .0599976 | .1000000, .0800000, .0600000 | 0.0549% |
| Café tabletop | .1445313, .0606079, .0259705 | .1445991, .0606179, .0259753 | 0.0469% |
| Café sunlit wall | .1878662, .1682129, .1348877 | .1879461, .1682340, .1349163 | 0.0425% |

The sealed room using physical world illumination instead returned **[0,0,0]** in Cycles while GL still returned its unoccluded fill. This reproduces the interior-lighting trap numerically. The café window is an actual open aperture, with no opaque emissive pane. Sunlit tables and wall patches in the rendered images confirm sunlight reaches the interior.

Agreement is established on these flat patches, within the GL half-float precision scale. It does **not** prove pixel-identical visibility: biased 3×3 PCF shadows at 4096² differ from Cycles ray intersections; raster coverage, sample filtering, floating-point storage, smooth-normal handling, and lens integration also differ. `--full-gi` restores world/indirect/emitter illumination, is labelled `full_gi_not_comparable`, and has finite bounce limits. It is not an equivalent benchmark.

## Specular investigation: rejected mapping

`tools/verify/measure_specular.py` rendered 54 isolated uniform patches: two Cycles Glossy distributions, `(N.L,N.V)` of `(1,1)`, `(.5,.7)`, `(.2,.3)`, roughness `.15/.4/.8`, and F0 `.04/.3/.8`. Unit zero-angle sun, orthographic camera, black world, 64 samples, seed 701, no denoising/adaptive/clamping; measurements are central 8×8 means from 16² linear EXRs.

Candidate: isotropic GGX with `alpha=roughness²`, separable Smith masking, and Schlick Fresnel. Tested Cycles mapping: Glossy `Color=F0`, using both GGX and MULTI_GGX. At grazing incidence, roughness .15, F0 .04, candidate radiance was **3.55219**, versus **0.531502** (GGX) and **0.531580** (MULTI_GGX): relative errors **568.33%** and **568.23%**. Even the diagnostic constant-F0 candidate differed by up to 20.71% / 39.68% over the grid. This mapping is nowhere near a 1% acceptance threshold, so **no specular term was added to either production renderer**.

This rejects the tested mapping; it does not prove an exact matched BRDF is impossible. The Glossy color parameter cannot simply be assumed to implement the candidate's Schlick Fresnel. Blender documents [Glossy distributions](https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/glossy.html) and changes to [multiple-scattering GGX](https://developer.blender.org/docs/release_notes/4.0/shading/). The café therefore remains visibly matte: no claim of physically faithful ceramic glaze, metal reflections or glass.

## Inspection images and observed limits

These are actual production renderer captures made with approved macOS services. The initial sandboxed Blender launch crashed; the completed images below were not produced by a mocked renderer. All files are local, generated and ignored by Git.

- [Wide front, 22 mm](../../output/verification/gl_wide_front.png)
- [Wide corner, 24 mm](../../output/verification/gl_wide_corner.png)
- [Espresso machine placement](../../output/verification/gl_machine.png)
- [Scene, sharp OpenGL](../../output/verification/gl_scene.png)
- [Scene, OpenGL f/1.2](../../output/verification/gl_f1.2.png)
- [Scene, OpenGL f/22](../../output/verification/gl_f22.png)
- [Cycles f/1.2](../../output/verification/cycles/rt_focus2.5m_f1.2.png)
- [Cycles f/22](../../output/verification/cycles/rt_focus2.5m_f22.png)
- [Cycles sharp](../../output/verification/cycles/rt_focus2.5m_f1.2_sharp.png)

Inspected several angles for support/placement: the machine feet meet the counter, cups meet the drip tray, pot meets its side table, and canisters meet shelves. No obvious floating props or appliance-crossing luminous panel was found. Intentional joins/intersections remain at assembled furniture and plant stems; this is not a watertight manufacturing model.

The focus cup remains sharp; repeated cups blur progressively at f/1.2 and are much sharper at f/22. The old **100-tap gather is unchanged**. Structured highlight sampling, foreground/background bleeding and hard/soft silhouette disagreement are visible in these wide-open captures. Shadow bias and raster aliasing are also visible. Cycles bokeh has visible Monte Carlo noise at 128 samples. These effects must not be conflated.

## Ground-truth status and cost

Current Cycles captures: 128 uniform samples, explicit seed 16, denoiser off, adaptive off, direct/indirect clamps 0, max bounces 12, matched diffuse/glossy/transmission continuation 0; 32-bit linear RGB EXR plus PNG preview/PFM companion/JSON. Sidecars explicitly record **`ground_truth: false`**. No independent second pass or measured error bar was produced. The corrected future estimator is `std(N − N/2) / sqrt(3)` because independent variances `c/N + c/(N/2) = 3c/N`; no numeric uncertainty is claimed here.

Final café: **1,415 primitives; 132,875 vertices; 131,636 triangles**, including 18,384 emissive and 79,696 smooth triangles. Bounds: `(-3.12,-1.59,-10.12)` to `(3.12,1.70,5.10)`. Scene SHA-256: `01b3320c4ea3c374c1ff5f028ffb3032243f06fcd8d9c235e7a53bc5de67348a`. Alley pins remain 2,551 / 66,802 / 38,572.

On this Apple M4 Max, 1200² OpenGL means over 20 warm frames after four warm-ups, including an explicit `glFinish` and CPU submission but excluding image I/O: **0.668 ms sharp**, **2.000 ms f/1.2**, **3.036 ms f/22**. The static shadow map is cached; its initial construction is excluded. These short measurements vary with scheduling and are not a sustained benchmark. Cycles/Metal inspection render-and-output times were **2.513 s f/1.2**, **1.558 s f/22**, **1.533 s sharp**, excluding initial process/scene setup. None are convergence-certified reference costs.

## Tests and file changes

Build passed with AppleClang 17 / C++17. CMake emitted existing cached-GLM minimum-version deprecation notices. **8/8 CTest suites passed**: mesh loading, physical camera, C++ scene grammar/geometry, comparison failure handling, executable filename contract, per-vertex cross-loader equivalence, reference configuration, Python scene grammar/geometry. Python suites contain 35 individual tests. Both scenes retain full winding/unit-normal/aggregate checks; existing tolerances were not relaxed. Blender 5.2.2 LTS / Metal was exercised; older-version fallbacks were not.

[Full unabridged CTest output](cafe_phase1_tests.txt). Scene regeneration was deterministic. OpenGL and Cycles dry runs and the Phase 1 inspection-runner dry run passed; all expected inspection outputs exist. The complete Phase 5 pipeline does not yet exist, so its dry run is **not** claimed.

| File | Change / reason |
|---|---|
| `.gitignore` | Ignore local ImGui layout. |
| `CMakeLists.txt` | Build dump/filename helpers; register new regression suites. |
| `include/SceneFile.h` | Shared capture and optional import settings. |
| `include/CaptureName.h` **new** | Tested settings-tag construction, including sharp and nondefault lens. |
| `src/SceneFile.cpp` | Tapered/conical cylinders, physical box chamfers, shared settings, strict version fix. |
| `tools/scene/scene_loader.py` | Matching Python geometry/settings; reject empty scenes, trailing version tokens and fractional versions. |
| `tools/scene/build_cafe.py` **new** | Framing derivation, deterministic café authoring, round stock and brightness-only jitter. |
| `scene/cafe.scene` **new** | Generated common scene data; alley retained byte-for-byte. |
| `src/main.cpp` | Selectable scene, opt-in import, shared transform, exact-pixel batch FBO, camera overrides, linear sharp readback, aperture buttons, honest radius bound/automatic ceiling, capture names. |
| `shaders/basic.frag` | Comments corrected to describe matched fill and visibility limits; production equation unchanged. |
| `tools/raytraced_reference/render_dof.py` | Scene-derived camera/size/import; matched camera-only fill, full-GI switch, inspection EXR/PFM and honest metadata/names. |
| `tools/scene/preview_scene.py` | Default CPU preview selects café. |
| `tools/compare/compare_renders.py` | Fail on no usable pairs; configurable scene hash; stale override report states actual verification; label PNG diagnostics honestly. |
| `tests/scene_file.cpp`, `tests/test_scene_file.py` | Keep alley pins; add café pins, grammar cases, taper identity, frustum/chamfer checks, framing and helper tests. |
| `tests/test_reference_config.py` | Test loaded camera/settings/import transform instead of duplicated constants. |
| `tests/scene_dump.cpp`, `tests/test_loader_equivalence.py` **new** | Actual C++/Python vertex attributes, indices, settings and rejection parity on a small multi-primitive fixture. |
| `tests/capture_name.cpp`, `tests/test_capture_contract.py` **new** | Exercise both filename implementations over lenses, focus values, apertures and sharp mode. |
| `tests/test_comparison.py` **new** | Empty/rejected/dimension-mismatched pairs fail; stale override cannot claim verification. |
| `tools/verify/check_cafe.py` **new** | Reproducible Phase 1 inspection views; fail on renderer failure/missing output. |
| `tools/verify/measure_lighting.py` **new** | Isolated and enclosed production-renderer calibration plus café ROI measurements. |
| `tools/verify/measure_specular.py` **new** | Reproducible rejected BRDF mapping measurements. |
| `README.md`, `tools/raytraced_reference/README.md`, `reports/raytraced_dof/README.md` | Current workflow, defaults and limitations; remove obsolete matching/ground-truth claims. |
| `notes/EXPLAINER.md` | Mark Experiment 18 explanation as historical and point to this checkpoint. |
| `notes/graphics/README.md` | Link this checkpoint without inventing a completed experiment archive. |
| This report and `cafe_phase1_tests.txt` **new** | Review evidence and complete test output. |

## Remaining work / approval gate

- Phase 3: independent convergence estimates, explicit acceptance thresholds, complete audit records and strict mode/hash rejection for quantitative comparison.
- Phase 4: keep naive gather and add sample-own-CoC weighting, area-scaled tap counts, deterministic pixel decorrelation, and gather tags. A gather still cannot recover hidden surfaces a real lens sees around an occluder.
- Phase 5: one settings definition driving the complete reference/capture/linear-comparison pipeline, complete output verification, fast mode and pairing checks. The current runner is inspection-only.
- Git: confirm whether the new branch should start from current `origin/main`, restoring the accepted baseline before the implementation commits, or from an earlier agreed commit. No repository was initialized over the existing history. No commits/push/merge/PR are claimed.
