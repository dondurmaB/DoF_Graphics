# Experiment 16 — Ray-Traced DoF Reference

Review time: approximately 3–5 minutes.

## Goal

Match the raster camera to physical lens settings and provide a reproducible Blender Cycles DoF reference using the same teapot, scale and scene layout. Use an existing renderer; no custom ray tracer is implemented. Four Cycles previews now render successfully in Blender 5.2.2 LTS on Apple M4 Max Metal. A paired comparison against the new OpenGL projection is still pending.

## Why a Ray-Traced Reference?

BasicDoF has only the final visible RGB and depth at each pixel. Its 17-tap color gather cannot discover surfaces hidden behind foreground objects or correctly model visibility from different parts of a lens. Cycles can sample the actual scene through a camera aperture, providing a higher-quality physical reference for later comparison. It is not a perfect or noise-free ground truth.

## Physical Camera Consistency

Previously the raster FOV was independent of the lens parameters used by CoC. With `usePhysicalCameraProjection = true`, projection now derives vertical FOV from:

```text
FOV_y = 2 atan(sensorHeight / (2 focalLength))
```

### Focal Length and Sensor Height

Both are in millimeters. At 50 mm focal length and 24 mm sensor height, vertical FOV is **26.9915°**. A longer lens narrows the view; a taller sensor widens it. Both parameters continue to affect the unchanged CoC calculation. `fieldOfViewDegrees = 45` remains available when physical projection is disabled.

### Focus Distance and F-number

The convention remains **1 world unit = 1 meter**. Focus is a view-depth plane, default 5 m. Lower f-numbers mean a larger aperture and stronger potential defocus. Defaults remain f/1.4, with f/2.8 and f/8 comparison settings. Near/far planes remain 0.1/100 m.

## OpenGL Pipeline

```text
Scene → raster color/depth → linear depth → physical CoC
      → 17-tap screen-space gather → BasicDoF
```

Color, RawDepth, LinearDepth, CoCMagnitude, CoCSigned and BasicDoF are preserved. The screen shaders are unchanged. The gather still caps radius at 12 pixels and exits early below 0.5 pixels.

## Cycles Pipeline

```text
Physical camera → lens samples / rays → actual scene geometry
                → accumulated path-traced DoF image
```

The separate script selects `CYCLES`, enables camera DoF, and uses direct focus distance and f-stop. It does not read OpenGL depth or blur an image in compositing. Defaults are 32 preview samples or 128 final samples, with OpenImageDenoise. Device selection attempts Metal and falls back to CPU; the verified preview used Apple M4 Max (GPU - 32 cores) through Metal without fallback.

## Coordinate-System Matching

OpenGL is Y-up; Blender is Z-up. A single `Rx(+90°)` conversion maps `(x,y,z)` to `(x,-z,y)` for all geometry, the camera and the light direction. Each Blender object receives `C * M_OpenGL`.

The teapot keeps scale `0.1` and model transform `T(0,-0.75,0) * Rx(-90°) * Ry(0°) * S(0.1)`. Blender's OBJ importer is configured to preserve authored coordinates before applying this matrix and the world conversion once. No automatic normalization occurs. OBJ source units are arbitrary; the explicit transform determines the meter-based scene size.

The foreground reference remains near 2 m, the teapot near 5 m, and background references near 15/25 m. The same floor and wall are recreated. The accepted teapot/floor spacing is preserved rather than changed for the reference.

## Camera Matching

The OpenGL reset camera is at `(0,0,5)`, forward `(0,0,-1)`, up `(0,1,0)`. In Blender these become `(0,-5,0)`, `(0,1,0)` and `(0,0,1)`. The camera basis respects Blender's local −Z viewing axis.

Blender uses **vertical sensor fit** with sensor height 24 mm. Effective width is `24 * imageWidth / imageHeight`: 24 mm for square images, not for arbitrary aspect ratios. The script checks Blender's effective camera frame against the intended vertical FOV when run.

## Reference Settings

| Setting | Default |
|---|---|
| Asset | `assets/models/scene.obj`, same teapot |
| Scale | `0.1`, explicit import rotation/translation |
| Lens / sensor height | 50 / 24 mm |
| Focus / f-stops | 5 m / 1.4, 2.8, 8 |
| Projection | Physical; 26.9915° vertical FOV |
| Reference resolution | 1200×1200 framebuffer pixels |
| Output | `reports/raytraced_dof/`, PNG + settings JSON |

Use `T` in OpenGL for the reference preset and a stationary camera, `H` to report settings, `1` for sharp Color, `6` for BasicDoF, `B` for f/2.8, and `F` to compare f/1.4/f/8. `7/8/9` select 2/5/15 m focus. `V` toggles projection, `[/]` adjusts lens, and `,/.` adjusts sensor height. Other camera, resize and screenshot controls remain available.

Resolution must match the actual PNG/framebuffer, not logical Retina window size. The default 1200×1200 comes from an existing local capture; verify the current title and use Blender's explicit width/height arguments. See the [tool instructions](../../tools/raytraced_reference/README.md) for exact commands and capture order.

## Expected Differences and Limitations

Haloing, foreground/background bleeding, incorrect silhouette blur, sparse-tap patterns and missing hidden surfaces are expected screen-space limitations, not newly observed results. No paired Experiment 16 images were available. Cycles should improve lens-dependent occlusion, but noise and denoising can affect fine details.

Matching physical inputs does not make the baseline fully physical: its inherited CoC magnitude is used as a radius although the underlying expression describes a diameter, and its radius is capped. Lighting, shadows, material response, polygon triangulation and color management also differ. Compare sharp framing first and avoid attributing all image differences to blur.

## Performance

**Cycles is a quality reference and is not expected to meet 20 ms/frame. The ≤20 ms/frame target applies to the future AI solution.**

## Verification and Remember

Build and all three CTest suites passed: mesh loading, physical camera projection, and Python reference configuration. The real application printed 26.9915° and loaded the teapot, but macOS window services blocked rendered-frame verification. The corrected Blender script completed four 1200×1200, 32-sample Cycles renders in Blender 5.2.2 LTS (`d13f752e3b9c`, built 2026-09-15), using Apple M4 Max Metal. Paired OpenGL/Cycles framing remains unverified. Previous Experiment 15 visual results do not verify the new projection.

The OBJ references absent `20900_Brown_Betty_Teapot_v1.mtl`. Blender 5.2.2's inspected importer API offers no material-loading switch, so a temporary binary copy omits only `mtllib`/`usemtl` lines and is deleted after import, even on failure. All other bytes and the original OBJ remain unchanged. Import reports 61,638 positions / 123,264 triangles. Controlled materials are assigned afterward. Blender 5+ supplies material/world node trees directly; only older versions set `use_nodes`. `render.use_compositing = False` bypasses compositing. No warnings are suppressed.

Preview results (all successful, under `reports/raytraced_dof/`):

| PNG | Bytes | Render time |
|---|---:|---:|
| `rt_focus5m_f1.4.png` | 1,216,935 | 2.43 s |
| `rt_focus5m_f2.8.png` | 1,210,066 | 2.01 s |
| `rt_focus5m_f8.png` | 1,201,341 | 1.97 s |
| `rt_sharp.png` | 1,194,873 | 1.86 s |

Total render loop: **8.27 s**, including saves and metadata, excluding scene setup. All four images were visually inspected: nonblack, upright visible teapot, intended scene direction; f/1.4 visibly blurs foreground/background more than f/8. These are Cycles-only observations, not a BasicDoF comparison. The successful render log contains no missing-MTL errors, deprecation warnings or other warnings. A prior sandboxed probe crashed and the version command emitted a USD cache-line warning; the approved unsandboxed render succeeded without these diagnostics. All three CTest suites passed again, including byte-preservation and temporary-file cleanup checks.

The workflow establishes explicit, reproducible comparison settings. It does not fix BasicDoF, implement ray tracing, or begin the AI stage.
