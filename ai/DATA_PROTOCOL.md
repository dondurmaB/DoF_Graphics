# Data protocol (v0)

What a training example looks like when it reaches the model: file formats, colour,
tensor shapes, channels, units and normalisation, and how they come from the rendered
files. What is rendered and how it is stored on disk is `renderer/mitsuba/SCENE_CONTRACT.md`,
section 7. Both authors' scenes go through this one protocol, so a model never knows
whose scene a sample came from.

Agreed on 2026-10-20: signed CoC instead of depth plus settings, 512x512 crops, no depth
channel, linear (HDR) colour. Meeting notes it answers: image formats and raw values,
linear vs gamma colour, geometry buffers, depth units and outliers, U-Net, active learning.

## 1. One example

| Name | Shape (one example) | dtype | Meaning |
|---|---|---|---|
| `x` | `[4, H, W]` | float32 (float16 is fine) | model input: sharp image (3) + signed CoC (1) |
| `y` | `[3, H, W]` | float32 | target: DoF image |
| `mask` | `[1, H, W]` | float32 in {0, 1} | loss weight: 0 where the CoC is not trustworthy (`coc_valid`, behind glass or mirrors) |
| `meta` | dict | | scene, split, group, env, f, N, focus, CoC percentiles, `auto_exposure_scale`, `sample_id`; for evaluation by bin and for active learning, never an input |

A batch is `x: [B, 4, H, W]`, `y: [B, 3, H, W]`, `mask: [B, 1, H, W]` (NCHW, PyTorch order).
Training crops are `H = W = 512`, cut from 1920x1080 renders at native resolution.
Evaluation runs on the full frame, padded to a multiple of the network's stride.

| Channel of `x` | Content | Normalisation |
|---|---|---|
| 0-2 | sharp RGB | colour transform of section 3 |
| 3 | signed CoC diameter, in pixels **of this tensor's resolution** | divided by `COC_NORM = 16` (1.0 means a 16 px blur disc) |

There are **no extra scalar parameters** for a disc aperture. The lens settings (focal
length f, f-number N, focus distance s, sensor height h, image height H) and the depth z
enter only through the CoC map.

## 2. Why the signed CoC instead of depth plus settings

For the thin lens the renderer uses (checked against rendered discs within 0.3%),

    CoC_px(z) = K (1 - s / z),    K = H f^2 / (N s h)

positive behind the focus plane, negative in front, `+K` for sky (z = infinity).

- **It is the quantity the blur depends on.** Raw depth plus settings planes
  (3 + 1 + 5 = 9 channels) would make the network learn this formula, across
  depths from 0.3 m to kilometres and lens values with very different scales.
- **It keeps depth order.** The CoC increases with z (it is affine in 1/z), so
  "which surface is in front" at a silhouette is still in the map. Occlusion at
  object boundaries is the hard part of DoF, and it needs that order.
- **Crops do not change it, resizing scales it.** Resize by factor r, then multiply
  channel 3 by r. Flips and 90-degree turns leave the values unchanged.
- **It works on real photos.** A monocular depth estimator gives relative disparity
  `d = a / z + b` with unknown a and b. Because the CoC is affine in 1/z, it becomes
  `CoC = k (d_focus - d)`: the user picks the focus point and one blur strength `k`.
  No metric depth or lens data is needed, and the input means the same as in training.

Considered and not chosen:

| Layout | Channels | Why not |
|---|---|---|
| RGB + metric depth + lens values as constant planes | 9 | network must learn the CoC formula; depth scale differs per scene; real photos have no metric depth |
| RGB + depth, lens values through FiLM (a small MLP that scales and shifts features) | 4 + 5-vector | same, plus a second input path |
| RGB + CoC + disparity | 5 | disparity is recoverable from the CoC up to an affine map, so it adds no information |

**Reserved for other aperture shapes** (contract section 4): when a second shape exists, add a
global vector `[one-hot shape, sin(rot), cos(rot)]` through FiLM, or as constant planes.

## 3. Colour: linear HDR end to end

**Rule: every stored image is linear radiance, and the model works on linear values.**
Gamma (sRGB) and tone mapping are for looking at images only.

Why linear and not display colour:

- **Blur is an average of light, and light adds linearly.** A DoF pixel is the mean of the
  radiance arriving through the aperture. Averaging gamma-encoded values is a different
  operation: it darkens bright-to-dark edges and makes bokeh of lights too dim.
- **Display colour clips highlights.** A lamp, the sun on chrome or a window is far brighter
  than 1.0. Clipped, the network cannot know how much light is there, yet that energy is what
  makes the bright bokeh discs in the target. HDR keeps it.
- **One-way street.** An LDR (display) version can always be made from HDR data later, never
  the reverse.

The transform from file to model space, the same for `x[0:3]` and `y`:

1. `L = radiance * auto_exposure_scale`: the sample's meter scale (log-average luminance
   to 0.18, what a camera's meter does; in `metadata.json`). It removes the "this pose is
   in shade, that one in sun" range (raw medians differ by 50x between poses) without
   changing ratios inside the image. The sharp and DoF passes have equal mean luminance
   (checked within 1%), so one scale fits both.
2. `T(L) = log(1 + L)`: compresses highlights for the network, linear near black so
   shadows keep their resolution, exactly invertible: `L = exp(T) - 1`. Model outputs are
   inverted the same way. No clipping anywhere.

Loss: L1 in `T` space (masked), optionally plus a perceptual term on the display version.
Display version (previews, figures, LDR metrics): `render.tonemap` (ACES fit + sRGB).

Real photos at inference: a camera RAW is linear and goes in directly. A JPEG is
gamma-encoded and clipped: undo sRGB to get approximately linear values, and accept that
clipped highlights are lost. If that matters, train a variant whose inputs are clipped and
quantised like a JPEG while the target stays HDR.

## 4. File formats

| Data | Format | Why |
|---|---|---|
| sharp, dof | OpenEXR, float32 RGB, linear, lossless compression | lossless float, linear, compressed, standard in rendering |
| G-buffer | OpenEXR, float32, 10 named channels | many channels with names in one file |
| depth | `.npy` float32, meters, `inf` for sky | exact, fast to load |
| CoC | `.npy` float32, signed pixels | same |
| `coc_valid` | `.npy` uint8, 1 = trust the CoC | per pose |
| metadata | JSON | |
| previews | PNG 8-bit sRGB, tone-mapped | looking only, never training |

Not used: **Radiance `.hdr`** (RGBE: one shared exponent and 8-bit mantissas, about 1%
precision, no named channels), **PFM** (uncompressed float, at most 3 channels, no metadata),
**PNG for data** (8 or 16 bit, gamma-encoded, clips at 1.0). EXR covers what they do.
Half-float EXR would halve the size, but the sun disc and lamps may exceed half's range
(65504) before metering, so float32 stays until the maxima are measured.

## 5. Geometry buffers

Stored once per pose in `gbuffer.exr`: world position P (xyz), shading normal N (xyz),
albedo (rgb), shape index. One ray per pixel (point-sampled), so every value belongs to one
real surface, with no averages across silhouettes. Depth is planar z along the optical axis,
`dot(P - origin, forward)`.

v0 models do **not** take them as input: real photos have no G-buffers, and the CoC already
carries what depth gives. Uses: deriving `coc_valid` (first hit is glass or a mirror),
evaluation by region (edges, materials), and an "oracle" ablation (RGB + CoC + normals) that
shows how much geometry could help. P and N are in world space; convert with the pose's
extrinsics (in `metadata.json`) for camera space, and rotate them with any flip or rot90.

## 6. Depth units and outliers

- **Every scene is in meters.** Web assets are checked against their catalog size when
  loaded (`web_assets._check_size`), and a scene must correct an asset authored in the wrong
  unit (the workshop shelf is stored 10x too big and scaled by 0.1).
- **A mis-scaled scene does not break labels.** The rendered blur and the CoC come from the
  same geometry, so they stay consistent. It only makes the scene look like a miniature or a
  giant, so scale is checked for realism, not for label correctness.
- **No outliers by construction.** Sky is `inf` in `depth.npy`, never a large number; it gets
  the at-infinity CoC `+K`. Poses whose 1st-percentile depth is under 0.3 m are rejected
  by the sampler.
- **Loader checks per sample.** No NaN; finite depth within [0.05 m, scene `far_clip`];
  |CoC| p95 within the sampler's range (0.05% to 4% of image height). A sample that fails is
  logged and skipped, not clamped.

## 7. From files to tensors

```
p<index>_<hash>/sharp.exr                  -> x[0:3]
p<index>_<hash>/lens_<hash>/coc.npy        -> x[3]   (float32 [1080, 1920], signed px)
p<index>_<hash>/lens_<hash>/dof.exr        -> y
p<index>_<hash>/coc_valid.npy              -> mask   (uint8 [1080, 1920]; per pose, shared by its lenses)
p<index>_<hash>/lens_<hash>/metadata.json  -> auto_exposure_scale, meta
```

Order in the loader: read, meter and log transform (section 3), random crop, flip / rot90,
CoC normalisation. Rot90 and vertical flips get low probability (they put the sky sideways;
contract section 7). No arbitrary-angle rotation or resizing during training.

The beauty passes use Mitsuba's gaussian pixel filter (about 0.5 px), the same in the sharp
input and the DoF target, also when rendered with the fast frozen tracer.

## 8. Model notes

A U-Net (encoder-decoder with skip connections) fits: the output is the input image
re-blurred, so the skips carry the in-focus detail and the deep layers gather the blur.
Its receptive field must cover the largest blur disc in the data (4% of the image height =
43 px at 1080p) plus the neighbouring occluder, so at least about 90 px. A U-Net with 4
downsamplings already exceeds that. 512 crops divide by 16 and 32.

## 9. Active learning

Every sample is a pure function of `(scene, scene seed, env, sampler seed, index)`, and a
`--dry-run` draws poses and lenses without rendering. The model input needs only the sharp
image and the CoC, which are cheaper than the DoF target (1024 vs 2048+ spp). So:

1. Train on a first balanced set.
2. Evaluate the error per bin of `meta` (scene, env, CoC bin, focal length, edge density).
3. Draw a large pool of candidates with `--dry-run`; score them by the bin error, or by the
   disagreement of a small ensemble run on the cheap input.
4. Render the DoF pass only for the highest-scoring candidates; repeat.

The val and test scenes never enter this loop, so the selection cannot leak into the numbers.

## 10. Constants

| Constant | Value | Reason |
|---|---|---|
| render resolution | 1920x1080 | contract budget |
| training crop | 512x512 | holds the largest blur (43 px) with room for context; divides by 32 |
| `COC_NORM` | 16 px | puts the data's range (0.5 to 43 px) around [-3, 3] |
| colour | `log(1 + radiance * auto_exposure_scale)` | section 3 |
| lens variants per pose | 2 | sampler default; sharp, depth and G-buffer are shared by them |

## 11. Before bulk generation

- [x] Dataset rendering on the fast frozen renderer (`lens_trace`), gaussian filter splatted
      with Mitsuba's own ImageBlock. Checked against `mi.render` on the Cornell box, desk and
      workshop (2026-10-20): difference between methods equals the noise between two renders of
      one method (ratio 0.98 to 1.04, edges too), same per-pixel noise, means within 0.1%.
- [x] `coc_valid` mask in the renderer (contract section 7), from the first-hit BSDF flags: 0 where the
      surface has a specular or glossy lobe and no diffuse one (the rule 9 BSDFs) or is a `null`
      surface. Measured: pool views 23-50% masked (water), museum case views about 50% (display
      glass), desk under 5% (monitor stand, keyboard plate).
- [ ] Measure radiance maxima per scene (decides half vs float32 EXR).
- [ ] A loader (`ai/data.py`) that implements this file, with a test that loads one sample
      per scene and checks shapes, ranges and `mean(y) ~ mean(x[0:3])`.
- [ ] Held-out scenes (3 val, 1 test, 1 test_ood) from the other author; their generation can
      start later because splits are by scene.
- [ ] Sharding into archives (not needed for the first few thousand samples).
