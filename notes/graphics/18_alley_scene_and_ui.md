# Experiment 18: shared scene file, dusk alley, HDR gather, compact UI

## What changed
- `scene/alley.scene`: one text description of the whole environment, read by **both** renderers.
  2551 primitives, 66802 vertices, 38572 triangles, 5076 of them emissive.
- `tools/scene/scene_loader.py` + `src/SceneFile.cpp`: two loaders for that file, one per language,
  pinned to each other by `tests/test_scene_file.py` and `tests/scene_file.cpp`.
- `tools/scene/build_alley.py`: authors the file (deterministic, seed 20260928). Regenerate rather
  than hand-edit, except for single-line tweaks, which `L` reloads live.
- `tools/scene/preview_scene.py`: CPU rasterizer for the same file, for composing without a GPU.
- `src/main.cpp`: the seven hand-placed cubes are gone. The alley is one static VBO drawn in one
  call per pass; the shadow map is 4096^2, fitted to the file's `shadow` region, and only redrawn
  when the light or the visible geometry changes.
- `basic.vert/frag`: new vertex layout (0 position, 1 albedo, 2 normal, 4 emission), normal-offset
  shadow lookup, physically matched diffuse, and an emitter branch.
- `screen.frag`: gathers in linear HDR out of an `RGBA16F` target, then applies exposure and the
  sRGB curve; adds screen mode 6, a sharp|DoF wipe.
- `render_dof.py`: builds the Cycles scene from `scene/alley.scene` with per-vertex colour, custom
  split normals, and emitters that light nothing. Every sidecar JSON now records the scene sha256.

## Why one scene file
The environment used to be written out three times: `cubeA`..`cubeG` in `main.cpp`, the `BOXES`
tuple in `render_dof.py`, and regex assertions in `tests/test_reference_config.py`. Any change had
to be made in all three or the raster pass and the Cycles reference silently stopped drawing the
same scene, which makes every comparison between them meaningless. Now geometry is authored once
in Python, committed as data, and only ever *loaded*. The two loaders are the remaining duplication,
so they are tested against a shared set of numbers (counts, bounds, summed positions, summed area)
rather than trusted.

There are no image textures anywhere. All surface detail is real geometry: mortar courses, ~11%
protruding bricks, crate panels, railing bars, fire-escape slats. That keeps Cycles matching
exactly (no texture-filtering differences to argue about) and makes every shadow a real shadow.

## Two bugs this experiment fixes

**1. The gather used CoC diameter as its radius.** `screen.frag` computed the signed circle of
confusion — a *diameter*, by definition — and passed it straight in as the gather *radius*. Every
OpenGL DoF image in experiments 14 to 17 was therefore blurred about twice as hard as the Cycles
reference at the same f-number. `gatherDefocus` now multiplies by 0.5 explicitly.

**2. Linear radiance was stored in an 8-bit buffer.** The scene FBO was `GL_RGB8`, so linear values
were quantized and clipped at 1.0 before the gather ran, and nothing ever applied a transfer
function. Blender writes its PNGs through the "Standard" view transform, which *is* the sRGB curve,
so the OpenGL images came out visibly darker than the reference for reasons that had nothing to do
with depth of field. The scene target is now `RGBA16F` and `screen.frag` encodes at the end. The
practical difference: an out-of-focus bulb is a bright bokeh disc instead of a flat grey one,
because averaging happens on the real radiance.

## The 120 px number, honestly

The blur-radius ceiling is 120 px, as asked. It is a *ceiling*, and at the reference settings it is
never reached. Vertical CoC diameter in pixels is

    coc_px = (A * f * (d - s)) / (d * (s - f)) / sensor_height * frame_height

with `A = f / N`. At 50 mm, f/1.4, focused `s = 5 m`, over a 1200 px frame, the far background
(`d -> infinity`) gives a diameter of about 18.0 px, so a **radius of about 9.0 px**. The ceiling is
irrelevant; the lens is the limit.

| lens | f-number | focus | background CoC radius (1200 px frame) |
|---|---|---|---|
| 50 mm | f/8 | 5 m | 1.6 px |
| 50 mm | f/2.8 | 5 m | 4.5 px |
| 50 mm | f/1.4 | 5 m | 9.0 px |
| 85 mm | f/1.4 | 1.6 m | 85.2 px |

So the panel now reports the CoC radius actually reached next to the ceiling, and says
"lens-limited" when the ceiling is doing nothing. `K` sets 85 mm f/1.4 at 1.6 m, which produces an
unmistakable blur; `T` restores the 50 mm f/1.4 5 m reference setup. Both are worth showing: the
first proves the gather works, the second is the one that gets compared to Cycles.

## Physically matched lighting

`basic.frag` computes

    radiance = albedo * (sky + sunColor * energy * max(N.L, 0) * (1 - shadow) / pi)

The `1/pi` is the Lambertian BRDF normalization, which is what makes this agree with a Cycles
Diffuse BSDF lit by a sun of the same strength. `energy` is irradiance in W/m^2 and is passed
straight to Blender's sun strength; `sky` is `ambient.color * ambient.strength`, which is
simultaneously the GL clear colour, the ambient term, and the Cycles world background. Emitters
render as `albedo * emit` in both, and in Cycles they are marked invisible to diffuse rays with
`max_bounces = 0` so they illuminate nothing, matching the raster pass.

**Remaining known difference:** the sky term has no occlusion in the raster pass, while Cycles
darkens creases and undersides because the sky really is blocked there. The raster ambient is
therefore flatter. This is a genuine limitation of the baseline, not a bug, and is one of the things
the learned refinement stage has to account for.

## Known limits (still baseline, on purpose)
- The gather is not depth-aware, so sharp foreground and background silhouettes bleed into each
  other. This is exactly the error the ray-traced comparison and the AI stage exist to show and fix.
- Emitters do not bounce light in either renderer, so the alley has no practical illumination
  beyond the sun and the uniform sky.

## UI
Three sizes cycled with `G`: Compact (default, ~250 px, auto-height), Full, Hidden. Compact holds
only what a parameter sweep needs — aperture and focus as preset buttons, three sliders, the view
selector, and the CoC readout. Presets are buttons rather than sliders because a slider cannot be
set to exactly f/1.4 twice in a row and the Cycles jobs render at fixed stops. The cursor is free on
launch and the camera turns while the **right mouse button** is held, so changing a parameter no
longer costs two extra keystrokes; `Tab` still toggles a sticky captured cursor.

## Verify
```
cmake --build build && ctest --test-dir build        # scene_file, scene_file_python, reference_config, ...
python3 tests/test_scene_file.py --print-expected    # after regenerating the scene
python3 tools/scene/build_alley.py                   # rewrite scene/alley.scene
python3 tools/scene/preview_scene.py                 # CPU preview PNG, no GPU needed
python3 tools/raytraced_reference/render_dof.py --dry-run
```
