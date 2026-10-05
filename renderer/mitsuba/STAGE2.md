# One qualified gather experiment

From the project root, using the existing environment with Mitsuba 3.9.1:

```sh
.venv/bin/python renderer/mitsuba/stage2_experiment.py --out output/stage2/my-run
```

The destination must not exist. Nothing from an earlier run is overwritten or
silently reused. This command builds the existing procedural asset cache if
needed, renders independent sharp and thin-lens pairs, computes centre-ray
depth, runs the **actual `shaders/dof_scene/screen.frag`** through macOS OpenGL,
authenticates all inputs, qualifies reference noise, and writes images, JSON
and a linear-domain `RESULTS.md` table. Any failure exits nonzero immediately;
partial expensive renders remain available, but are not qualified results.
macOS graphics services are required; no CPU/Python blur is substituted.

Settings are in [stage2.json](stage2.json). The experiment is deliberately an
**opaque diffuse, fixed-flash version of the Mitsuba café**, with its original
geometry and transforms. It is not the original glass/metal/full-GI lighting
configuration. One point light at `(0.34,1.22,1.85)` has RGB intensity `(15,15,15)`;
it remains fixed for the overview. Local lamps and the environment do not emit.
The `path` integrator uses `max_depth=2`, `rr_depth=6`, `hide_emitters=false`:
direct-light transport, no indirect bounces, no denoising. This restricted
transport model makes primary-visibility/DoF error affordable to resolve.

Each beauty frame is rendered twice at equal N with different seeds. The A
image is the reference, not the A/B average. Difference RMS divided by sqrt(2)
estimates A's spatial/channel-aggregate noise RMS. Centred difference standard
deviation divided by sqrt(2) is also reported. Independently randomized
multijitter realizations retain this equal-variance identity; samples within
one realization need not be independent. This is not a 95% confidence interval,
a worst-pixel bound, or a measurement of systematic light-transport bias.

Two independent sharp inputs are passed through both gathers as well. For
**every variant and region**, the quadrature sum of reference noise and
propagated sharp-input noise must be at most 10% of measured linear MAE. The
comparison refuses to publish results if this gate fails. A sample budget,
the viewer's “converged” label, and an attractive PNG do not qualify a reference.

## Files and data contract

Each `f<number>/` directory contains:

- `sharp-a/b.exr`: pinhole linear RGB, float32, H×W×3. Both stops reuse the same
  sharp realizations, with authenticated per-stop optics for the gather.
- `depth.npy`: float32 H×W, first intersection at the pixel centre, positive
  optical-axis metres, not Euclidean range. Misses are explicitly 10000 m.
- `gather-naive-a/b.exr` and `gather-weighted-a/b.exr`: production GLSL outputs.
  Exactly 100 disk contributions for a blurred pixel, radius cap 120 pixels;
  sub-half-pixel radii return the centre. Colour is bilinear RGB32F, depth is
  nearest-neighbour R32F, output is RGBA32F with linear readback.
- `reference-a/b.exr`: the same scene through Mitsuba's circular thin lens.
- Per-image `.exr.json` sidecars: file, scene, camera, context and source hashes;
  renderer/version, filter, sampler, all configured integrator parameters;
  seed, chunk schedule and spp for render inputs; exact input/depth/shader hashes
  and variant for each gather. Qualified references also carry measured noise.
- `qualification.json`, `results.json`, `footprint.json`, and `edge-mask.png`.
  `results.json` exists only after qualification succeeds.
- Linear absolute-difference EXRs and consistently scaled `linear-x4.png`
  previews. Ordinary PNG previews use the documented ACES-fit/sRGB transform
  and do not participate in metrics.

MAE and RMSE operate on unclipped scene-linear RGB. PSNR uses a fixed peak of
**1 linear radiance unit**, even for HDR values above one; it is not normalized
separately per image. Depth edges are adjacent depth ratios greater than 1.3,
marked on both sides and dilated by seven pixels. The exact mask is saved.

`wide-angle-inspection.exr/.png` is a 28 mm pinhole view of the same scene from
the front corner. It is explicitly an inspection image, **not a qualified
reference**. The overview has its own camera sidecar.

For a later learned stage, the input is one sharp linear RGB image, its planar
depth map, and the camera/gather parameters above. The target is qualified
`reference-a.exr` from the same context. The naive gather is the required
baseline. No depth network, train/test split or training result is supplied.
Predicted relative/inverse depth must be calibrated to these metric semantics
before it can replace `depth.npy`.

## Recheck and tests

```sh
.venv/bin/python renderer/mitsuba/stage2_experiment.py --compare-only --out output/stage2/my-run
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Recomparison verifies source identity as well as all file hashes and recomputes
the noise gate. Missing stops, pairs, sidecars, dimensions or qualification
inputs fail. After a code change, old results remain historical evidence, not
validation of the new code. `traditional_dof.py` now accepts only this qualified
contract for comparisons; its legacy Python gather remains a **viewer preview**,
not the benchmark implementation. The separate three-way PNG comparison is
outside this experiment and its old display-domain metrics are not mixed in.

CTest registers sensor rays, OpenGL adapter/100-tap equivalence, GGX peak,
120-pixel support, provenance/missing-pair/dimension/noise failure cases, and
PNG/optics snapshot tests, alongside the existing native math/mesh suite. The
GPU tests fail rather than skip when macOS graphics services are unavailable.
