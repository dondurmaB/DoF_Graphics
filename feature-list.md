# Feature List

## Shipped

| Feature | Status | User value | Notes |
| --- | --- | --- | --- |
| Numbered OpenGL experiments | Shipped | Teaches the graphics pipeline step by step. | Experiments 1-14 remain archived under `experiments/graphics/`. |
| Original `DepthResearch` renderer | Shipped | Provides the baseline interactive renderer and teaching target. | Uses the original shader path and cube/depth pipeline. |
| Separate `DoFScene` target | Shipped | Lets the cafe DoF demo evolve without disturbing the teaching snapshots. | Built as `DoFScene.app` on macOS. |
| Procedural cafe tabletop scene | Shipped | Gives foreground, focus-plane, and background depth cues for DoF learning. | Current scene has 126 procedural objects, 8 mesh shapes, smooth lathed teapot/cup/vase/spout geometry, and no external assets. |
| Interactive navigable camera | Shipped | Lets the learner inspect DoF from different distances and angles. | `RMB` look, `WASD`, `Q/E`, `Shift` speed, `0` home. |
| Click and center focus | Shipped | Makes focus distance intuitive by choosing visible scene depth. | `LMB` samples clicked depth; `F` samples screen center. |
| Lens and aperture controls | Shipped | Shows how focal length, focus distance, and f-number affect blur. | Scroll, `Shift`+scroll, `[ ]`, and `- / +` adjust values. |
| Five display modes | Shipped | Exposes the render pipeline and DoF diagnostics. | `1` DoF, `2` sharp, `3` depth, `4` CoC, `5` split. |
| Native HUD controls | Shipped | Allows GUI-style mode, focus, and aperture changes without external UI deps. | HUD has mode buttons and focus/aperture minus/plus widgets. |
| CLI capture and verification options | Shipped | Supports repeatable screenshots and runtime checks. | Includes `--capture`, `--frames`, `--mode`, `--focus`, `--f-number`, `--lens`, `--view`, `--size`, `--no-hud`, and `--verify`. |
| CPU math and mesh tests | Shipped | Protects camera, lens, CoC, mesh, and focus-target assumptions. | `DoFSceneTests` contains 15 tests. |
| Cafe DoF runtime validation | Shipped | Confirms the native window, framebuffer path, screenshot capture, GUI smoke controls, display modes, CLI validation, and static checks work on the current machine. | Native `--verify` passed 14 frames with no GL errors; invalid CLI inputs rejected; Clang static analysis had no findings; validation PNGs and `output/cafe-validation.json` were saved. |
| Three-way DoF comparison target | Shipped | Turns the known screen-space limitation from a caveat into a measured number. | `DoFApproaches` renders one analytic scene through a screen-space gather, multi-view accumulation, and lens-sampled ray tracing via `--method 1|2|3`. |
| Adversarial DoF comparison scene | Shipped | Exposes every screen-space failure mode in a single frame. | `createHaloScene()` stages a sharp railing on the focus plane, near posts thinner than their own defocus disc, emissive background bokeh, and a receding depth ramp. |
| Method comparison metrics and diff maps | Shipped | Makes method claims falsifiable instead of visual. | `tools/compare_dof_methods.py` writes per-zone mean absolute error and amplified difference maps to `output/dof3-comparison.json`. |
| Experiments 15-17 | Shipped | Extends the teaching sequence past basic DoF into why it fails and what fixes it. | Documentation and reproducible commands over the shared `DoFApproaches` target rather than three source snapshots. |
| Shader copy dependency tracking | Shipped | Keeps edited `shaders/dof_scene/*` files synchronized into the app bundle during rebuilds. | Source shader changes now trigger copy/link dependency updates. |

## Planned

| Feature | Status | User value | Notes |
| --- | --- | --- | --- |
| Dataset generation | Planned | Produces RGB/depth pairs and camera metadata for AI depth experiments. | Later stage of the roadmap. |
| Monocular depth integration | Planned | Compares predicted depth to renderer depth and drives DoF from both. | Later stage of the roadmap. |
| Stronger DoF artifact handling | Planned | Reduces single-layer screen-space limitations near occlusion edges. | Experiments 15-17 measured the gap; applying multi-view or layered depth to `DoFScene` itself is still open. |
| Accelerated ray-traced reference | Planned | Lets the reference scale past 64 analytic primitives. | The current ray tracer tests every ray against every primitive from uniform arrays, with no acceleration structure. |

## Deferred

| Feature | Status | Reason |
| --- | --- | --- |
| External cafe assets | Deferred | The current demo intentionally remains procedural and dependency-free. |
| Ray-traced/photoreal rendering | Deferred | The `DoFApproaches` ray tracer is a depth-of-field reference for small analytic scenes, not a path tracer: direct lighting only, no reflection, refraction, or indirect light. |
