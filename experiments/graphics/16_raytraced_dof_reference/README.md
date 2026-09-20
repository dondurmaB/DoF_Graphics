# Experiment 16 — Physically Matched Camera and Ray-Traced DoF Reference

This snapshot adds physical raster projection and a separate Blender Cycles reference workflow. Experiment 15 was preserved at `42c46d2`, confirmed on local `main` and live `origin/main` before changes. The runtime fix began with a clean `main` and live `origin/main` at `63603ce3d79ae31ecb72c6145b9151b75e3e903d`, using the intended `dondurmaB/DoF_Graphics` origin. Blender Cycles previews now pass; the earlier macOS window-service restriction on OpenGL visual checks remains recorded.

## Implementation

The default projection uses `2 atan(sensorHeight / (2 focalLength))`: 50 mm lens, 24 mm sensor height, **26.9915° vertical FOV**. `usePhysicalCameraProjection` defaults to true; the legacy 45° setting remains available. The six display modes, original CoC calculation and 17-tap BasicDoF shaders are unchanged. Focus remains 5 m, aperture f/1.4, clipping 0.1–100 m, and maximum gather radius 12 pixels.

`T` restores the reference camera/settings and releases the cursor; `V` toggles projection. `[/]` adjust focal length, `,/.` adjust sensor height, `B` selects f/2.8, `F` toggles f/1.4/f/8, and `7/8/9` select 2/5/15 m focus. `H` reports settings and framebuffer pixels. Existing navigation, modes, resizing and screenshot controls remain implemented.

Cycles uses the same OBJ and explicit `0.1` import scale, with `T(0,-0.75,0) * Rx(-90°) * Ry(0°) * S(0.1)`. OBJ units are not assumed to be meters; this transform maps them into the project's meter convention. A common `(x,y,z) → (x,-z,y)` conversion applies to the model, supporting boxes, floor, wall, camera and lighting direction. The OpenGL camera `(0,0,5)`, forward `(0,0,-1)`, up `(0,1,0)` becomes Blender `(0,-5,0)`, forward `(0,1,0)`, up `(0,0,1)`.

The script configures Cycles lens sampling, vertical sensor fit, 5 m focus and f/1.4, f/2.8, f/8 outputs. Defaults are 1200×1200 framebuffer pixels, 32 preview or 128 final samples, and OpenImageDenoise. Device selection attempts Metal, with CPU fallback; the four verified previews used Apple M4 Max (GPU - 32 cores) via Metal without fallback. Optional controls support 2/15 m focus, other image dimensions and a sharp reference. Generated PNG/JSON files stay outside this archive and are ignored by Git.

## Verification Record

| Check | Result |
|---|---|
| `cmake --build build` | PASS during initial implementation; no C++ changes in this fix |
| `ctest --test-dir build --output-on-failure` | PASS, 3/3: mesh loading, physical camera, reference configuration |
| Actual application startup and OBJ loading | PASS: 246,528 GPU vertices, 369,792 indices, 123,264 triangles; normals and UVs present |
| Physical FOV | PASS in unit tests and startup output: 26.9915° |
| Lens/sensor response and non-square projection | PASS in mathematical/configuration tests; rendered response unverified |
| Scene conversion, import transform and reference jobs | PASS in Python configuration tests and dry run |
| Color, depth/CoC modes, BasicDoF, FBO completion | Experiment 16 visual checks blocked by macOS window-service restrictions |
| Focus/aperture response, navigation, resize, reference framing | Implementations preserved; new runtime visual verification pending |
| Installed Blender | 5.2.2 LTS, hash `d13f752e3b9c`, built 2026-09-15; macOS Applications executable |
| Cycles execution and OBJ import | PASS, 61,638 positions / 123,264 triangles; original asset hash unchanged |
| Cycles scene/camera sanity | All four PNGs inspected: visible upright teapot and intended scene direction; paired OpenGL framing still pending |
| Cycles f/1.4 versus f/8 | Stronger foreground/background blur visible at f/1.4 |
| Output PNGs / device | Four nonempty 1200×1200 previews, 32 samples, Apple M4 Max Metal |
| Paired BasicDoF/Cycles comparison | Not performed; no observed differences claimed |

Experiment 15's manual results remain valid for that experiment; they do not establish visual correctness of Experiment 16's narrower projection. An existing 1200×1200 OpenGL capture supplied the resolution default, but is not an Experiment 16 verification image.

## Blender Runtime Fix and Results

The missing `20900_Brown_Betty_Teapot_v1.mtl` is unnecessary because this reference assigns its own materials. Inspection of Blender 5.2.2's actual OBJ operator found no option to skip MTL loading. A temporary copy removes only `mtllib` and `usemtl` directives; every other byte (including positions, normals, UVs and faces) is preserved. The copy is removed after import, including error paths. Original `assets/models/scene.obj` stays unchanged, SHA-256 `1c96d356c5198b7dd45c4cbf4fec8cc01f1b7e42b3372b1ffc40cc270682b215`.

Blender 5+ creates material/world node trees automatically; deprecated `use_nodes` assignments are guarded for older versions only. `scene.render.use_compositing = False` disables the compositor pipeline without `scene.use_nodes`. Camera DoF, physical matching and sample defaults remain unchanged. Each image now logs its index, settings, device and elapsed time, followed by a generated-file summary.

The exact `--preview --sharp` command below completed successfully (exit 0):

| Output in `reports/raytraced_dof/` | Result | Bytes | Render time |
|---|---|---:|---:|
| `rt_focus5m_f1.4.png` | PASS | 1,216,935 | 2.43 s |
| `rt_focus5m_f2.8.png` | PASS | 1,210,066 | 2.01 s |
| `rt_focus5m_f8.png` | PASS | 1,201,341 | 1.97 s |
| `rt_sharp.png` | PASS | 1,194,873 | 1.86 s |

Total render loop: **8.27 s**, including saves/metadata and excluding scene setup. All used Cycles, Metal Apple M4 Max (GPU - 32 cores), 32 samples, 1200×1200 pixels and OpenImageDenoise. JSON sidecars record exact timings and settings. PNGs/JSON remain ignored, not archived or committed. All three CTest suites passed after the fix; the reference suite now also checks byte preservation and cleanup on success/failure.

No missing-MTL error, `use_nodes` deprecation or other warning appeared in the successful render log. A preceding sandboxed API probe crashed before Python execution, and the version command emitted a USD `ARCH_CACHE_LINE_SIZE` warning. Approved execution outside the sandbox completed cleanly. CPU fallback remains available but was not exercised by this run. The 128-sample final mode was not rendered.

## Run and Review Locally

From the repository root:

```sh
cmake --build build
ctest --test-dir build --output-on-failure
./build/DepthResearch
```

Press `T`, then `H`. Check all six modes, focus/aperture changes, lens/sensor FOV changes, camera movement and resizing. Capture sharp Color and BasicDoF with `P`, preserving each `output/latest.png` before the next capture. Record actual framebuffer dimensions.

With the installed Blender, run the matching reference (replace width/height if needed):

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 \
  --python tools/raytraced_reference/render_dof.py -- --preview --sharp --width 1200 --height 1200
```

Use `blender` instead of the absolute executable when it is on PATH. Omit `--preview` for 128 samples; `--device cpu` forces CPU. Confirm version/device, imported geometry, camera framing, focus, aperture response and generated images. The [archived tool guide](tools/raytraced_reference/README.md) contains all commands and settings.

## Limits and Snapshot Contents

Haloing, foreground/background bleeding, sparse sampling and missing hidden surfaces remain expected BasicDoF limitations; none are claimed as newly observed here. The inherited CoC expression describes a diameter but BasicDoF uses its magnitude as a capped radius. Cycles should improve lens-dependent visibility, while lighting, triangulation, color management, noise and denoising also affect comparisons.

**Cycles is a quality reference. The ≤20 ms/frame target applies to the future AI solution, not Cycles.** No AI stage is included.

The snapshot contains the application/mesh sources, physical camera helper, four shaders, CMake configuration, tests and separate reference tool. It is an archive, not a standalone checkout: building requires the repository's existing `external/` dependencies and original `assets/models/scene.obj`. No large model copy, generated renders, caches or temporary harnesses are included. Earlier archives are unchanged.

See [research notes](../../../notes/graphics/16_raytraced_dof_reference.md).
