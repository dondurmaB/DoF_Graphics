# Progress

## Current State

The repository is classified as ML/research. The active implementation work adds a separate `DoFScene` target for a procedural cafe tabletop depth-of-field demo while preserving the original `DepthResearch` renderer and archived experiments 1-14.

`DoFScene` currently includes:

- A native macOS GLFW app target built as `DoFScene.app`.
- A procedural cafe scene with 126 objects, 8 mesh shapes, smooth lathed teapot/cup/vase plus curved spout geometry, procedural material variation, shadow mapping, and no external asset dependencies.
- A dynamic camera with look, movement, vertical movement, fast movement, home reset, and focus picking.
- Focus controls through click-to-focus, center focus, scroll, keyboard, and HUD widgets.
- Aperture and focal-length controls.
- Five display modes: DoF, sharp color, depth, CoC, and split view.
- Screenshot output to `output/cafe-dof.png` with `P`, plus CLI capture support.
- CLI options for capture path, frame count, mode, focus distance, f-number, lens focal length, view preset, window size, HUD hiding, and runtime verification.
- A 900x560 minimum CLI window size to avoid HUD overlap.
- Source-change dependency tracking for copied DoFScene shaders.

`DoFApproaches` adds a three-way depth-of-field comparison on one shared analytic scene:

- Methods selected with `--method 1|2|3`: single-layer screen-space gather, multi-view aperture accumulation, lens-sampled ray tracing.
- `createHaloScene()` rebuilt to stage four failure modes in one frame: a sharp railing exactly on the focus plane against a bright wall, near posts thinner than their own defocus disc, emissive background bokeh, and a receding arcade depth ramp. 53 primitives, 14 of them emissive, laid out against the 50 mm frustum's visible half-height of 0.24 x depth.
- Ray tracer primitive cap raised from 48 to 64 (768 uniform components, under the 1024 GL 3.3 guarantees for the fragment stage).
- `tools/compare_dof_methods.py` measures each method against the 512-sample ray-traced reference and writes per-zone metrics plus amplified difference maps.

`renderer/mitsuba/` adds a physically based renderer path (Mitsuba 3.9.1) for realistic scenes, intended to run on the HPC:

- A procedural cafe interior (numpy meshes and textures, no downloaded assets, about 626k triangles). It has window openings lit by `sunsky`, pendants, globe bulbs, string lights, candles, glass with real wall thickness, metals and glazed ceramics.
- `render.py` writes pixel-aligned `sharp` (pinhole), `dof` (thin lens) and a 1-spp G-buffer with planar z-depth, plus `metadata.json` with the lens and intrinsics. It uses the same thin-lens convention as `DoFScene`.
- `hpc/render_cafe.slurm` is a SLURM template for `cuda_ad_rgb`. The partition and environment lines still need to be set for the cluster.

## Validation

Verified:

- `cmake --build build --target DoFScene DoFSceneTests` passed cleanly with `-Wall -Wextra -Wpedantic`; the only noted warning noise was an existing upstream GLM CMake deprecation message.
- `ctest --test-dir build --output-on-failure` passed all 15 CPU math and mesh checks.
- Native `DoFScene --verify` passed 14 frames, exercising resize, camera movement, center and click autofocus, HUD mode and focus buttons, mouse capture, all five display modes, aperture changes, reset behavior, and GL error checks.
- Direct framebuffer PNG capture worked even though OS-level screen capture failed.
- Validation captures were saved: `output/cafe-final.png` with HUD and no-HUD captures `output/cafe-sharp.png`, `output/cafe-wide-aperture.png`, `output/cafe-narrow-aperture.png`, `output/cafe-near-focus.png`, `output/cafe-far-focus.png`, `output/cafe-depth.png`, and `output/cafe-coc.png`.
- `output/cafe-validation.json` was saved.
- Image validation found nonblank captures, meaningful near-focus versus far-focus differences, and aperture-sensitive blur: mean absolute error versus sharp was 2.6422 for f/0.7 and 0.004412 for f/16 on a 0..255 pixel range.
- Clang static analysis of `src/dof_scene/main.cpp` and `src/dof_scene/scene.cpp` exited 0 with no findings.
- Invalid CLI inputs for size `640x480`, focus `-2`, f-number `0`, and mode `7` were rejected with exit 1.
- Final interactive GUI was verified running as PID 53648 with title `Cafe - Depth of Field | DOF | 4.37 m | f/1.2`.

`DoFApproaches` verified:

- `cmake --build build --target DoFApproaches` clean with `-Wall -Wextra -Wpedantic`.
- Camera consistency: railing bar centres identical across all six captures (1076, 1428, 1780, 2132, 2485), confirming the three methods differ only in how visibility is resolved.
- Raster/ray-trace equivalence at f/22, where depth of field vanishes: 0.518 mean absolute error on 0..255. Nonzero because the rasterizer tessellates spheres and cylinders the ray tracer intersects exactly, and neither path antialiases edges. This is the floor of the comparison.
- Mean absolute error against the 512-sample reference at 64 samples: gather 4.946 overall / 2.925 halo zone / 12.641 near-occluder zone; multi-view 0.682 / 0.558 / 1.364.
- Convergence behaviour over 8, 16, 32, 64 samples: gather 5.626, 5.238, 5.022, 4.946 (asymptotic, a biased estimator); multi-view 2.455, 1.487, 0.923, 0.682 (converging toward the 0.518 floor).
- Halo profile on the bright wall approaching a focus-plane railing bar: the gather darkens the wall by up to 35.1 of 255 at 2 px from the silhouette, while multi-view matches the reference to 0.1 at every distance.
- Captures saved: `output/dof3-reference-pinhole.png`, `dof3-m1-gather.png`, `dof3-m2-multiview-8.png`, `dof3-m2-multiview-64.png`, `dof3-m3-raytrace-32.png`, `dof3-m3-raytrace-512.png`, difference maps, and `output/dof3-comparison.json`.

Two defects were found and fixed while building the reference, both invisible to the rasterizer alone:

- **Ambiguous world checker.** The ground plane lies exactly on a checker cell boundary at y = 0. The rasterizer interpolates that y as exactly 0.0 and stays on one side of `floor(position / scale)`; the ray tracer computes it as `ro.y + rd.y*t`, so rounding put successive aperture samples on either side and the ground averaged to flat grey. Fixed with `floor(position / scale + 0.5)`; pinhole disagreement dropped from 6.35 to 0.65.
- **Accumulation buffer precision.** The accumulation target was `GL_RGBA16F`, whose 10 mantissa bits round away each new sample once the running total is large. The reference therefore degraded with more samples rather than converging: 0.556 at 5 frames to 3.529 at 400. Switching that target to `GL_RGBA32F` restored monotone convergence: 0.558 to 0.518 over the same range. Isolated by the rasterized path being bit-identical at 3 and 514 frames, which ruled out camera or input drift. Method 2 accumulates into the same target and improved equally.

Mitsuba cafe verified locally (`metal_ad_rgb`, M4 Max):

- Renders all three passes. At 960x540 and 1024 spp, each beauty pass takes 19 s.
- A shadow-ray probe confirmed that sunlight enters through the window openings. A two-storey facade across the street is kept low enough that it never shades them.
- Firefly root cause: uniform emitter selection gave the sun 1 light sample in about 40. Power-proportional `sampling_weight`s reduced firefly pixels from 2.1% to 0.15% at 256 spp (seed-to-seed relative difference 0.22 to 0.12). Roughening near-mirror materials alone had no measurable effect.
- Not yet run on the HPC.

## Decisions

- Keep `DoFScene` separate from `DepthResearch` to avoid disrupting the numbered teaching sequence.
- Keep the cafe scene procedural so the demo remains small, reproducible, and asset-free.
- Use a traditional screen-space DoF pass over color and depth rather than path tracing or multi-layer rendering.
- Correct the CoC footprint by computing thin-lens CoC diameter and using radius for the gather kernel.
- Derive projection FOV from focal length and 24 mm sensor height so lens controls affect both perspective and DoF.
- Keep `--verify` as a deterministic native runtime smoke exercise, not a claim of exhaustive physical keypress coverage.
- Use direct framebuffer PNG captures as the reliable validation path when OS screen capture is unavailable.
- Default the final validation camera to `(0.25, 1.55, 3.8)`, aimed 0.27 m above the focus point, with focus distance 4.37 m, a 50 mm lens, and f/1.2 aperture.
- Build `DoFApproaches` as a third target rather than extending `DoFScene`, so the cafe demo and its validation artifacts stay untouched.
- Make the comparison scene analytic, so the rasterized and ray-traced paths describe identical geometry and their difference images mean something.
- Share one `shading.glsl` between the rasterized and ray-traced paths, so the three methods differ in exactly one respect.
- Document Experiments 15-17 as configurations of the shared target rather than as three source snapshots, since all three are one program and three copies would drift apart.
- Treat the 512-sample ray tracer as the reference for depth of field only, not as ground truth for light transport.

- Use Mitsuba 3 for realistic scenes: it has a Python API, a thin-lens camera, AOVs for ground-truth depth, and CUDA plus Apple Metal backends from one pip wheel. Keep the OpenGL targets for real-time method comparison.
- Build the Mitsuba scene procedurally rather than from downloaded scenes, so depth layout, focus targets and licensing are under our control and HPC and laptop builds are identical.

## Known Risks

- Single-layer screen-space DoF cannot reveal hidden background behind foreground objects and may still show occlusion artifacts around depth discontinuities. Experiments 15-17 quantify this; `DoFScene` itself still uses the single-layer method.
- The ray-traced reference has no acceleration structure and is capped at 64 analytic primitives by its uniform arrays, so it does not scale to richer scenes.
- `output/` is git-ignored, so all comparison captures and metrics are local. They are reproduced by the commands in the Experiment 15-17 READMEs.
- macOS OpenGL support is deprecated by Apple, though it remains suitable for this educational/research demo.
- Runtime behavior depends on the current machine's OpenGL context and windowing environment; the current native smoke test passed on this machine, but future machines should rerun `--verify`.
- Glass caustics in the Mitsuba scene converge slowly under unidirectional path tracing; DoF-pass references need 2048+ spp.
