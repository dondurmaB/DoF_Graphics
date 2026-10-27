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

## 2026-10-17 — Mitsuba DoF dataset scenes (renderer/mitsuba)

- Work done: all ten artificial training scenes now use pinned CC0 web assets per `renderer/mitsuba/scenes/substitutions/<scene>.md` (statuses DONE/REVERTED per row). Committed today: workshop, museum, desk, pool, night_street (terminal, cafe, courtyard, greenhouse, rooftop earlier). `remote_run.sh` now detaches cluster steps so VPN drops do not kill renders; `pick_variant` refuses a silent LLVM fallback on GPU nodes; `web_assets` warns on mis-scaled assets.
- Verification: per scene on the cluster — `check_scene.py` (0 failures on all five), `audit_poses.py` contact sheets for sampler seeds 0 and 1, `sampler.py --dry-run` (0 failed samples); local `tests/test_scene_api.py` passes.
- State: pushed to `origin/ray-work`; viewers run as srun steps on the `dof-render` allocation (GPU memory fits ~8 viewers).
- Risks: per-`mi.render` tracing overhead is ~10 s on workshop (0.5–1 s elsewhere); `dr.freeze` cuts it to 0.04 s in isolation but failed to record through `render_beauty` (memcpy_async / uncaptured variable) and was reverted. Open GPU viewers can starve render jobs of memory.
- Next: fix frozen rendering before A100 bulk generation; val/test/test_ood scenes from the other author; haze, coc_valid, sharding.

## 2026-10-20 — Fast dataset rendering, data protocol (renderer/mitsuba, ai/)

- Work done: `lens_trace.py` now renders the dataset passes too. The gaussian pixel filter is splatted with Mitsuba's own ImageBlock inside a `dr.freeze` recording, and the G-buffer comes from a frozen AOV kernel. `render_beauty` / `render_gbuffer` take a `Lens` plus size, and `sampler.py`, `audit_poses.py` and `render.py` (so `check_scene.py`) use them. Bitmap-textured emitters are sampled once before recording (they broke recording on desk, museum, night_street, terminal and cafe). Viewer: trackpad pinch sets the focal length. `ai/DATA_PROTOCOL.md` v0: input `[B,4,512,512]` = sharp RGB + signed CoC, target DoF RGB, linear HDR colour (`log(1 + radiance * auto_exposure_scale)`), EXR/npy formats, G-buffer use, depth units, active learning.
- Verification (cluster GPU): all 11 scenes record; fast vs `mi.render` on Cornell, desk and workshop: method difference equals noise (ratio 0.98–1.04, edges included), same noise, means within 0.1%; G-buffer depth median relative difference under 0.1%. `sampler.py` end to end on workshop (3 poses) and night_street (2 poses), 0 failures. Workshop DoF pass 3 s vs 92 s; a pose (sharp 128 + 2 DoF 256 spp at 960x540) about 2.3 s after a one-time ~30 s of recording per process.
- State: committed and pushed to `origin/ray-work`. One viewer runs at a time (courtyard was last).
- Risks: the first calls of each new (size, spp) record for 20–50 s on heavy scenes, so chunks should keep one size; the protocol's `coc_valid`, loader and sharding are still todo; held-out scenes not built.
- Next: `coc_valid` mask, `ai/data.py` loader with its test, then a first generation run per scene on the cluster.

## 2026-10-21 — 10k training set generating (renderer/mitsuba, ai/)

- Work done: data protocol v1 (`ai/DATA_PROTOCOL.md`): x = sharp HDR RGB + 1/z + 6 broadcast camera settings (f, N, A, 1/s, sensor h, frame H), y = DoF RGB, colour log(1 + L * exposure), no masks; `ai/data.py` implements it. `generate.py` (plan / work / status / release / sheet) with an NFS-safe chunk queue, per-worker manifests, self-requeueing jobs (`hpc/ws_generate.slurm`) and orphan adoption. Blur floor raised to 0.5% of image height (5.4 px p95 at 1080p); poses that cannot reach it are redrawn.
- Verification: protocol check on 64 pilot samples (0 bad; blur computed from x matches coc.npy within 5e-6). Pilot: all 10 scenes, 80 poses, 0 failures, ~75 s/pose on an RTX 5000 Ada. CIAI A100 benchmark: ~11 min/pose on cafe (12x slower, no RT cores; GPU not shared), so rendering stays on the RTX cluster. Requeue tested on a 12-minute job.
- State: 10k run (5,000 poses, clear + overcast) started 2026-10-21 10:04 into `~/dof_dev/dataset/train10k` on the student-lab cluster: job 241177 plus a worker step in the interactive job 240004; a local watcher submits a second job when 240004 ends. ETA about 2.2 days.
- Risks: wet and snow presets look wrong (night_street wet renders daylight; courtyard snow/wet look like clear), so they are excluded until fixed. The 2-job QoS cap limits the run to 2 GPUs.
- Next: check sheets from the first finished chunks; fix the wet and snow presets; held-out scenes from the other author.

## 2026-10-22 — 10k run: renders shipped to CIAI /nfs-stor (renderer/mitsuba)

- Work done: the student-lab home quota (hidden; hit at ~52 GB) stopped both workers on 2026-10-21 at 12:36 after 322 poses. The data was copied to CIAI `/nfs-stor` over a restricted ssh key (`ciai-copy` on the student-lab cluster) and `generate.py work --ship-to` now copies each finished chunk there and empties it locally, so home holds only chunks in progress. Generation was paused 21:00–10:45 while `/nfs-stor/rui.gao` was reorganized into `Researches/<project>/`; jobs were resubmitted with the new destination, and the CIAI paths in `ciai_generate.slurm` follow the new layout.
- Verification: test write through the key to the new path; first automatic ship `shipped 38 poses`, CIAI then holds 360 finished poses (322 + 38, 49 GB) and the home staging dir 2.3 GB.
- State: jobs 242566 / 242567 on ws-l1-002 / ws-l5-013 shipping to `ciai-copy:/nfs-stor/rui.gao/Researches/dof/dof_dataset/train10k`; 7 of 100 chunks done, ~4,640 poses left, ETA about 36 h (Friday night).
- Risks: if shipping fails repeatedly, poses accumulate in home and the quota fills again. Do not use the student-lab `/l` (storage migration).
- Next: check sheets from shipped chunks on CIAI; fix wet/snow presets; decide whether to drop gbuffer/coc to save space.

## 2026-10-25 — 10k set finished and checked; smoke training passes (renderer/mitsuba, ai/)

- Work done: the 10k run finished at 01:15 (100/100 chunks, 5,000 poses, 9,999 unique samples, 634 GB on CIAI `/nfs-stor/rui.gao/Researches/dof/dof_dataset/train10k`). One pose (cafe p000281) has a single lens: both lens draws hit the same named focus point with a collapsed blur window, so they hashed to the same lens dir (sampler bug, fix before the next run). Render noise on glossy sunlit materials noted in `denoising.md` (deferred). Smoke training added: `ai/model.py` (8M-param residual U-Net, starts as identity), `ai/smoke_train.py` (bench / overfit / train, ~2% of poses held out by pose-id hash), `ai/smoke.slurm` (one A100 on cscc-gpu-p). PyTorch 2.5.1 (cu121) and OpenEXR 3.5.2 installed in the CIAI env.
- Verification: protocol check over all 9,999 samples: 10 bad (1 NaN in a museum DoF, 9 with DoF/sharp mean brightness >5% apart, 7 of them night_street), listed in `runs/bad_samples.txt` on CIAI and excluded. 20 samples inspected by eye (`inspect_data/`, git-ignored). Overfit on 16 crops: L1 0.0209 (identity) -> 0.0035. Short run (2,000 steps, batch 8): held-out L1 0.0166 vs identity 0.0215, PSNR 27.8 vs 25.4 dB, improving at every eval.
- State: dataset complete; no jobs running.
- Risks: training is loader-bound (0.49 s/step at batch 8, ~16 samples/s, vs ~62 samples/s the GPU can take); each sample decodes two 1080p EXRs for one 512 crop. The 9 brightness outliers may be real optics (lens seeing around occluders to bright lights), not render faults.
- Next: speed up the loader (one thread per worker, several crops per decoded frame, or a pre-cropped copy); pick the real architecture; decide on denoising and on Beibarys's scenes and evaluation sets.

## 2026-10-26/27 — Loss and conditioning experiments (ai/)

- Work done: occlusion-aware evaluation regions (in_focus, fg_spill, bg_edge, defocused) and CoC bins in `ai/smoke_train.py`; optional FiLM conditioning on the camera settings in `ai/model.py`; weighted L1 with alpha/beta/gamma terms; 18 held-out hard crops centred on CoC >= 32 px. Plan and all results in `ai/LOSS_PLAN.md` (sections 4-11). Protocol v1.1: the CoC map may be a loss weight or metric, never an input.
- Verification: three A/B rounds on CIAI (jobs 344947-344950, 345109-345112, 345647/345648, 347294). FiLM helps; the alpha (blur-size) weight hurts; a gamma 20 penalty on sharp edges in front of blurred background (bg_edge) cuts the halo. 10k steps, FiLM + gamma 20: held-out L1 at 0.549 of identity, bg_edge 1.03 (plain L1: 1.74), hard crops 0.444, CoC >= 32 px 0.365.
- State: no jobs running. Default recipe: FiLM + gamma 20.
- Risks: bg_edge never gets below identity, which points to the architecture; training is still loader-bound (~0.5 s/step at batch 8). Sampler lens-collision bug and denoising still open.
- Next: occlusion-aware architecture (sharpness gate or depth-ordered layers); loader speedup; sampler fix; data sharing with Beibarys.
