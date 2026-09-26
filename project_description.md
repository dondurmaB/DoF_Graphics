# Project Description

This repository is an ML/research-oriented C++17 / OpenGL 3.3 Core workspace for learning the graphics pipeline and building controlled depth-of-field experiments. The long-term research direction is to compare renderer ground-truth depth with monocular AI depth predictions, then drive depth-of-field post-processing from both depth sources.

The original `DepthResearch` target remains the numbered teaching renderer. Its archived experiments under `experiments/graphics/` and notes under `notes/graphics/` are the learning path for VAOs, VBOs, EBOs, uniforms, transforms, cameras, depth buffers, framebuffers, and basic DoF.

The newer `DoFScene` target is a separate native macOS GLFW app for a procedural cafe tabletop scene. It is intentionally isolated from `DepthResearch` so the original renderer and experiments 1-14 stay intact. The scene is generated in code, uses no external asset pipeline, and exists to make depth-of-field behavior visible through foreground, focus-plane, and background objects.

## Current DoFScene Behavior

`DoFScene` renders a cafe tabletop with a foreground mug, central teapot focus subject, books, fruit, plants, shelves, chairs, background tables, pendant bulbs, procedural materials, and shadow-mapped directional lighting. The current scene has 126 procedural objects and 8 mesh shapes, including smooth lathed teapot, cup, vase, and curved spout geometry. It is illustrative and procedural, not photoreal or ray traced.

The render path writes HDR color and depth to an offscreen framebuffer, then applies a screen-space depth-of-field pass. The post-process reconstructs linear depth, computes a signed thin-lens circle of confusion, converts diameter to blur radius, gathers 64 disk samples plus the center sample, applies depth-aware weighting, and filmic-tonemaps the result. The projection FOV is derived from a 50 mm lens over a 24 mm sensor height by default, with interactive focal-length changes from 24 mm to 100 mm. The depth diagnostic is scaled to a 15 m range.

The `DoFApproaches` target is a third, separate app for method comparison. It renders one shared scene through three depth-of-field methods - single-layer screen-space gather, multi-view aperture accumulation, and lens-sampled ray tracing - selected with `--method 1|2|3`. Its scene is built from analytic primitives so the rasterized and ray-traced paths describe identical geometry, and both shade through one shared GLSL file; the three methods therefore differ in exactly one respect, how visibility through the aperture is resolved. That equivalence is what makes their difference images and error metrics meaningful, and it is verified by rendering the rasterized and ray-traced paths at f/22, where they agree to 0.518 of 255. It is the code behind Experiments 15-17.

## Constraints

- macOS native OpenGL 3.3 Core Profile through GLFW.
- No new runtime dependencies beyond the existing CMake, GLFW, GLM, and bundled GL loader setup.
- No external assets for the cafe scene; geometry and materials are procedural.
- Preserve `DepthResearch`, the numbered experiment snapshots, and the teaching notes.
- Treat this as an ML/research repository; feature-level git commits are not required by default.
- `DoFScene` enforces a 900x560 minimum CLI window size to avoid HUD overlap.

## Known Rendering Limits

`DoFScene`'s DoF implementation is a traditional single-layer screen-space post-process. It uses the visible color and depth buffers, so it cannot reveal hidden background surfaces behind foreground objects. Its depth-aware blur reduces bleeding, but occlusion and disocclusion artifacts can still appear around strong foreground/background boundaries.

`DoFApproaches` exists to quantify that limit rather than restate it. Measured against its ray-traced reference, the single-layer gather carries a mean absolute error of 4.946 of 255 overall and 12.641 in the near-occluder region, and that error does not fall with more taps: 8 to 64 taps improves it by 12%, while the same increase in aperture views improves multi-view by a factor of 3.6. The screen-space error is structural bias, not sampling noise.

The ray tracer is a reference for depth of field only. It resolves the aperture and primary visibility exactly, but it is still direct lighting with one shading model, has no acceleration structure, and its uniform arrays cap the scene at 64 analytic primitives.
