# Cycles DoF Reference

This separate Blender tool recreates the Experiment 16 reference scene and renders actual camera/lens DoF with **Cycles**. It does not consume OpenGL depth or add compositor blur. **Cycles is a quality reference; the future AI solution's ≤20 ms/frame target does not apply.**

## Availability and Evidence

Verified with installed **Blender 5.2.2 LTS**, hash `d13f752e3b9c`, built 2026-09-15. The exact `--preview --sharp` run completed four 1200×1200, 32-sample Cycles renders on **Metal: Apple M4 Max (GPU - 32 cores)** in 8.27 seconds total. All PNGs were inspected for basic scene sanity; paired OpenGL/Cycles comparison remains pending. Older Blender 4.2+ compatibility is retained with a small node-tree version check, but was not exercised in this verification.

## Run

From the repository root, generate the three final references at 1200×1200 and 128 samples:

```sh
blender --background --python-exit-code 1 --python tools/raytraced_reference/render_dof.py
```

On macOS with Blender installed in Applications:

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 \
  --python tools/raytraced_reference/render_dof.py
```

Arguments for this tool go **after `--`**. A 32-sample preview, including a sharp frame:

```sh
blender --background --python-exit-code 1 --python tools/raytraced_reference/render_dof.py -- \
  --preview --sharp --width 1200 --height 1200
```

Force CPU if Metal is unavailable or unreliable:

```sh
blender --background --python-exit-code 1 --python tools/raytraced_reference/render_dof.py -- --device cpu
```

For foreground/subject/background focus, at one aperture:

```sh
blender --background --python-exit-code 1 --python tools/raytraced_reference/render_dof.py -- \
  --focus 2 5 15 --fstops 1.4 --preview
```

Validate the plan without Blender or output files:

```sh
python3 tools/raytraced_reference/render_dof.py --dry-run
```

Controls include `--width`, `--height`, `--lens`, `--sensor-height`, `--focus`, `--fstops`, `--samples`, `--preview`, `--sharp`, `--device auto|cpu` and `--dry-run`. Default sample limits are `PREVIEW_SAMPLES = 32`, `FINAL_SAMPLES = 128`. At most 12 images per invocation are allowed. The tool resets the Blender scene, so run in a dedicated background process, not an unsaved interactive scene.

## Outputs and Device Reporting

Default output files, created only by a successful Blender run:

- `reports/raytraced_dof/rt_focus5m_f1.4.png`
- `reports/raytraced_dof/rt_focus5m_f2.8.png`
- `reports/raytraced_dof/rt_focus5m_f8.png`
- Optional `reports/raytraced_dof/rt_sharp.png`

Each completed render gets a JSON sidecar with the asset hash, transforms, physical settings, resolution, samples, Blender version, render device, elapsed time and DoF state. Re-running overwrites the same named outputs: keep preview/final runs separately if retaining both. Generated images and JSON are ignored by Git; the output README is retained.

`auto` attempts to enumerate Metal devices, selects those exposed by Cycles, otherwise uses CPU. A caught GPU render error retries on CPU and records the fallback. A failed CPU render exits with an error; no success record is written for that attempt. The terminal distinguishes selected devices from completed renders. OpenImageDenoise is enabled, with deterministic seed 16. The verified run completed all four images on Metal without CPU fallback. Each render logs index/count, output, focus, aperture, samples and device, then completion time; the final summary lists generated files and total rendering time.

## Exact Scene and Coordinate Matching

The source of truth is Experiment 15's scene retained in `src/main.cpp`. Script constants are an explicit reference snapshot, not a general scene exporter.

All OpenGL-world objects use one conversion `C = Rx(+90°)`:

```text
(x, y, z)_GL → (x, -z, y)_Blender
M_Blender = C * M_OpenGL
```

OBJ import uses `forward_axis='Y'`, `up_axis='Z'`, `global_scale=1`, and `clamp_size=0` to preserve authored coordinates. Then the teapot uses exactly `C * T(0,-0.75,0) * Rx(-90°) * Ry(0°) * S(0.1)`. The two X rotations cancel for this particular asset, leaving a 0.1-scaled Z-up teapot translated to Blender `(0,0,-0.75)`. This is the consequence of the common conversion, not a separate guessed orientation. Source units are arbitrary; the chosen transform maps them into meters. No auto-normalization or centering occurs.

| Geometry | OpenGL position | X/Y rotation | Scale |
|---|---|---|---|
| Foreground A | `(-0.55,-0.35,3)` | `50° / 70°` | `0.5` |
| Teapot | `(0,-0.75,0)` | `−90° / 0°` | `0.1` |
| Background D | `(1.6,0.4,-10)` | `35° / −20°` | `2` |
| Background E | `(-3.8,0.4,-20)` | `−15° / 60°` | `2` |
| Floor | `(0,-0.85,-7.5)` | `0° / 0°` | `(12,0.1,25)` |
| Wall | `(0,1.2,-22)` | `0° / 0°` | `(14,4,0.1)` |

Boxes use `T * Ry * Rx * S`, as in OpenGL. Fallback-only cubes B/C are omitted because the actual teapot is required. Its missing `.mtl` is intentionally bypassed: the installed OBJ operator has no material-loading switch, so a temporary binary OBJ copy strips only `mtllib` and `usemtl` lines before import. All other bytes are preserved, the original asset is unchanged, and the copy is deleted even if import raises an error. The script then assigns a warm diffuse material. Supporting boxes use the corresponding six face colors. One sun points along the converted OpenGL light direction; a modest world light supplies ambient illumination. Diffuse path-traced lighting, shadows, color management and indirect light are not pixel-identical to the OpenGL shading.

## Camera and Pixel Matching

| Parameter | OpenGL | Blender |
|---|---|---|
| Position | `(0,0,5)` m | `(0,-5,0)` m |
| Forward | `(0,0,-1)` | `(0,1,0)` |
| Up | `(0,1,0)` | `(0,0,1)` |
| Lens | 50 mm | 50 mm |
| Sensor height | 24 mm | 24 mm; `sensor_fit='VERTICAL'` |
| Vertical FOV | 26.9914666° | Same intended effective FOV, checked using `view_frame` at runtime |
| Focus | 5 m | 5 m; `focus_object=None`, direct `focus_distance` |
| Aperture | f/1.4, f/2.8, f/8 | Same f-stops; circular aperture |
| Near/far | 0.1/100 m | Same camera clip values |
| Resolution | Actual framebuffer pixels | Explicit width/height, 100%, square pixels |

The camera basis is constructed from the actual OpenGL forward/up vectors, with Blender local −Z looking forward, and then transformed by `C`. Sensor width is set to `sensor_height * width / height`, not assumed equal to height. It is 24 mm only for the default square image; at 1600×900 it is 42.6667 mm. Vertical fit keeps vertical FOV fixed as aspect changes. The formula is `FOV_y = 2 atan(sensor_height / (2 lens))`.

The 1200×1200 default matches the dimensions of the existing local OpenGL PNG, **not a newly verified Experiment 16 capture**. Use the dimensions printed in the current OpenGL title/`H` output or in the captured PNG, and pass those exact values to Blender. A logical 600×600 macOS window may have a 1200×1200 framebuffer; do not assume that on every display.

## OpenGL Capture Procedure

1. Build and run `./build/DepthResearch`. Press `T` for the reference preset: reset camera, 50 mm lens, 24 mm sensor height, physical perspective, 5 m focus, f/1.4, BasicDoF. It releases the cursor so comparisons stay still; framebuffer size is preserved.
2. Press `H` to record settings and pixel dimensions. Press `1` then `P` for the sharp Color frame; copy `output/latest.png` elsewhere if keeping it.
3. Press `6` then `P` for BasicDoF f/1.4. Press `B` for f/2.8 and capture; press `F` to reach f/8 and capture. Confirm the title before every capture. `P` overwrites `output/latest.png`.
4. Use `7/8/9` for focus at 2/5/15 m. `R` resets only the camera. `Tab` resumes/pauses navigation. `V` toggles physical/legacy projection; `[/]` change lens by 5 mm and `,/.` change sensor height by 2 mm. `T` restores the reference settings after experiments.
5. Render Cycles with matching resolution, lens, sensor, focus and aperture. Compare sharp frames first to check framing, then silhouettes, handle, spout and focus transitions in DoF pairs. No new visual comparison was possible here.

## Limits of This Comparison

BasicDoF retains the original CoC calculation, magnitude-as-radius convention, 12-pixel cap, 0.5-pixel early exit and 17 taps. The thin-lens CoC expression describes a diameter, while the baseline uses its magnitude directly as a radius; it is an approximate baseline, not a calibrated match to Cycles blur diameter. No correction is introduced in this experiment.

Cycles samples real geometry through the lens and can resolve visibility absent from one RGB/depth layer. It still has sampling noise and denoising bias. Imported polygon triangulation, shading and display transforms can differ. Match geometry/framing and physical inputs before drawing conclusions; do not attribute every pixel difference solely to DoF. No observed superiority or artifact comparison is claimed without rendered pairs.

Blender 5+ creates material and world node trees automatically; the script directly edits those trees and only sets `use_nodes` on older versions. `scene.render.use_compositing = False` bypasses compositor processing without the deprecated scene property. No warnings are suppressed. The verified render log had no missing-MTL errors or deprecation warnings. The preceding sandbox probe crashed and the version command emitted a USD cache-line warning; approved unsandboxed rendering completed cleanly. See the Experiment 16 notes/archive for per-image sizes and timings.

Implementation references: [Blender sensor-fit API](https://docs.blender.org/api/3.6/bpy.types.Camera.html), [Blender OBJ coordinate conversion](https://github.com/blender/blender/blob/v4.5.0/source/blender/io/wavefront_obj/importer/importer_mesh_utils.cc), [Cycles device enumeration](https://github.com/blender/blender/blob/v4.5.0/intern/cycles/blender/addon/properties.py).
