# Data protocol (v1)

The exact form of one training example: input, target, units, normalisation and the files they
come from. What is rendered and stored on disk is `renderer/mitsuba/SCENE_CONTRACT.md`, section 7.
Both authors' scenes go through this one protocol. `ai/data.py` implements it; when the two
disagree, this file is the specification and the code is the bug.

Decided on 2026-10-20:
- The input is the **sharp** HDR image, its depth map and the camera settings. Nothing else:
  no CoC map, no masks, no G-buffer.
- The target is the rendered DoF image under exactly those settings, with no masking. The model
  learns edges, occlusion and reflections by itself.
- All images are HDR (linear radiance). No JPEG or other LDR data anywhere.

## 1. One example

| Name | Shape | dtype | Content |
|---|---|---|---|
| `x` | `[10, H, W]` | float32 | input: sharp RGB (3) + inverse depth (1) + camera settings as constant planes (6) |
| `y` | `[3, H, W]` | float32 | target: DoF RGB, same colour transform as `x[0:3]` |
| `settings` | `[6]` | float32 | the same 6 settings as a vector (`x[4:10]` is this, broadcast) |
| `meta` | dict | | ids, raw settings in physical units, exposure scale; never an input |

A batch is `x: [B, 10, H, W]`, `y: [B, 3, H, W]`, `settings: [B, 6]` (NCHW, PyTorch order).
Training uses random `512 x 512` crops of the 1920 x 1080 renders, at native resolution.
Evaluation uses the full frame.

### Input channels

| Index | Name | Formula | Unit / range in the data |
|---|---|---|---|
| 0-2 | sharp R, G, B | `log(1 + max(L, 0) * e)` | `L` linear radiance of `sharp.exr`, `e` the sample's exposure scale (section 2) |
| 3 | inverse depth | `1 / z`, sky `0` | 1/m; `z` planar depth in meters (`depth.npy`); about 0 to 20 |
| 4 | focal length | `log(f / 50 mm)` | f 24 to 135 mm, so -0.73 to 0.99 |
| 5 | f-number | `log(N / 4)` | N 1.2 to 16, so -1.20 to 1.39 |
| 6 | aperture diameter | `log(A / 10 mm)`, `A = f / N` | A 1.5 to 112 mm, so -1.9 to 2.4 |
| 7 | inverse focus distance | `1 / s` | 1/m, the same unit as channel 3 |
| 8 | sensor height | `h / 24 mm` | 1.0 for all data so far |
| 9 | frame height | `H_px / 1080` | 1.0 for all data so far; the **full frame's** height, unchanged by crops |

Channels 4 to 9 are constant over the image: each setting is one number, **broadcast** (copied to
every pixel) and concatenated with the image channels. That is the contract's default conditioning:
the first convolution sees the settings at every pixel, next to that pixel's depth. The same numbers
also come as the vector `settings` [6], so a model may instead (or also) feed them through FiLM
(a small MLP that scales and shifts the features at every level); that is a model choice and changes
nothing in the data. Channels 8 and 9 are constant today and still part of the input: the format
must not change when a dataset at another resolution or sensor size is added.

### Why exactly these settings

For the thin lens the renderer uses (checked against rendered discs within 0.3%), a point at
planar depth `z` is blurred into a disc of diameter, in pixels,

    CoC = (H_px / h) * f * A * |1/s - 1/z|

So `x` contains everything the target depends on: the scene as seen (channels 0-2), where
everything is (3), and every lens and sensor quantity in that formula (4-9). Focus and depth both
enter as inverse distance in 1/m, the unit in which blur is linear. `A` is redundant given `f` and
`N` but is fed explicitly, because the aperture size is the camera setting that sets the blur.
Camera roll, position and field of view are not inputs: they are already in the image.

Reserved: when a second aperture shape exists (contract section 4), the vector grows by
`[one-hot shape, sin(rotation), cos(rotation)]` and the protocol version goes to v2.

## 2. Colour

Every image is linear HDR radiance, stored as EXR. The model sees

    T(L) = log(1 + max(L, 0) * e)

- `e` = `auto_exposure_scale` of the pose (in `pose.json` and the manifest): it brings the
  log-average luminance of the sharp image to 0.18, as a camera's meter does. It is one number per
  pose, used for the input and the target alike, so the ratio between them is untouched.
- `log(1 + .)` compresses highlights without clipping and is exactly invertible:
  `L = (exp(T) - 1) / e` recovers the rendered radiance.
- The sharp and DoF renders have the same mean luminance (within 1%), so one `e` fits both.

Gamma, sRGB and tone mapping are for viewing only (`render.tonemap`), never for data.

## 3. Target

`y = T(L_dof)` with the same `e` as the input: the DoF render under exactly the settings in `x`.
There is no mask and no reweighting. A model's output `t` turns into radiance as
`(exp(t) - 1) / e`.

## 4. From files to tensors

```
<scene>/s<seed>/<env>/manifest.jsonl              -> list of samples, exposure scale
p<index>_<hash>/sharp.exr                         -> x[0:3]
p<index>_<hash>/depth.npy                         -> x[3]   (float32 meters, inf = sky)
p<index>_<hash>/lens_<hash>/metadata.json         -> x[4:10], settings, meta  (camera.*)
p<index>_<hash>/lens_<hash>/dof.exr               -> y
```

Settings come from `metadata.json`: `camera.focal_length_mm`, `camera.f_number`,
`camera.focus_distance_m`, `camera.sensor_height_mm`, and `resolution[1]` for `H_px`. The
aperture diameter is computed as `f / N`; it equals `2 * camera.aperture_radius_m`.

Order in the loader: read, colour transform and inverse depth, random crop, flips and 90-degree
turns, settings planes. Flips and 90-degree turns are exact for a disc aperture and change none of
the settings. Vertical flips and 90-degree turns get probability 0.1 each (they put the sky
sideways); horizontal flips 0.5. No resizing and no arbitrary-angle rotation.

The renderer also writes `coc.npy`, `coc_valid.npy` and `gbuffer.exr`. They are for analysis only
and are not part of an example.

## 5. Formats

| Data | Format |
|---|---|
| sharp, dof | OpenEXR, float32 RGB, linear radiance, lossless compression |
| depth | `.npy` float32, planar meters, `inf` for sky |
| settings, ids | JSON (`metadata.json`, `pose.json`, `manifest.jsonl`) |
| previews | PNG, tone-mapped; never data |

Not used: Radiance `.hdr` (RGBE, about 1% precision), PFM (no compression, no metadata), PNG or
JPEG for data (low dynamic range).

## 6. Units and checks

- Every scene is in meters (SCENE_CONTRACT.md rule 2). Web assets are checked against their catalog
  size when loaded, and mis-authored ones are scaled in the scene.
- Depth is planar `z` along the optical axis, the quantity the thin lens focuses on; sky is `inf`
  on disk and `0` in channel 3.
- The loader refuses a sample with NaN or inf in an image, NaN depth, or a surface closer than 0.05 m.

## 7. Constants

| Constant | Value |
|---|---|
| render resolution | 1920 x 1080 |
| training crop | 512 x 512 |
| input channels | 10 (3 RGB, 1 inverse depth, 6 settings) |
| settings normalisation | `log(f/50)`, `log(N/4)`, `log(A/10)`, `1/s`, `h/24`, `H_px/1080` |
| colour | `log(1 + max(L, 0) * auto_exposure_scale)` |
| flips | horizontal 0.5, vertical 0.1, 90-degree turn 0.1 |

## 8. Before bulk generation

- [x] Dataset rendering on the fast frozen tracer (`lens_trace`), same images as `mi.render`
      (checked on Cornell, desk and workshop: method difference equals render noise).
- [x] `ai/data.py` implements this file; `python ai/data.py <root> --split train` checks every sample.
- [ ] Pilot (8 poses x 2 lenses per training scene) reviewed by eye.
- [ ] Held-out scenes (3 val, 1 test, 1 test_ood) from the other author.
- [ ] Sharding into archives (not needed for the first few thousand samples).
