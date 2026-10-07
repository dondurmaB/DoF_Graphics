# Scene contract (v1)

The agreement between everyone who writes scenes and the shared renderer,
sampler and dataset code. Scene authors own `scenes/<id>.py`. The shared side
owns `scene_api.py`, `scene_kit.py`, `render.py` and the sampler. Changing any
of those, or this file, needs both owners to agree and bumps `CONTRACT_VERSION`.

Split of work: artificial and artificial x nature scenes (Rui), natural scenes
(Beiba). A scene's `group` is `artificial`, `natural` or `hybrid`, and `owner`
records who maintains it.

Status legend: **[done]** exists and is tested, **[shared, todo]** the shared
side will build it, nothing a scene author needs to do.

## 1. What a scene is

One file, `renderer/mitsuba/scenes/<id>.py`, exporting `SCENE: SceneDef`
(see `scene_api.py`; `scenes/cafe.py` is the reference). `<id>` is lowercase
snake_case and matches `SCENE.id`.

```python
SCENE = SceneDef(
    id="desk", group="artificial", owner="rui", description="...",
    build=build,                                     # (BuildContext) -> SceneBundle
    views={"home": View(origin, target, focus="cup", roll_deg=0)}, default_view="home",
    camera_box=((x0, y0, z0), (x1, y1, z1)),         # where the sampler may put cameras
    target_box=((x0, y0, z0), (x1, y1, z1)),         # where it may aim them
    exclude_boxes=(((...), (...)),),                 # carve-outs: furniture, walls, props
    lens_mm=(24, 135), f_number=(1.4, 8.0),          # ranges the scene is tuned for
    envs=("clear", "overcast"),                      # presets this scene implements
    tags=("indoor", "clutter"),                      # from scene_api.TAGS only
    default_seed=7, asset_version=1,                 # bump asset_version when a recipe changes
    max_depth=12, rr_depth=6, spp_hint=2048, far_clip=1000.0,
)
```

`build(ctx)` returns a `SceneBundle(scene, focus_points, meta)`: a Mitsuba scene
dict, named focus points, and any facts worth recording (sun direction, ...).
Nothing else crosses the boundary. Shared building blocks (`Assets`,
`SceneBuilder`, `principled`, `bitmap`, `rgb`, `xf`) live in `scene_kit.py`:
import them, do not copy them.

## 2. Rules for `build`

1. **No sensor, no integrator.** The renderer adds the camera, lens, film and
   integrator. A scene dict containing either is rejected.
2. **Meters, y up.** The CoC and depth maths assume real scale. A 50 mm lens at
   f/1.8 must behave like a real one in your scene.
3. **Deterministic.** All randomness comes from `np.random.default_rng(ctx.seed)`.
   No `random`, no time, no unseeded numpy, no network. The same
   `(id, seed, env)` must give the same scene dict (checked by `fingerprint`).
   The scene is deterministic; the *images* are only statistically reproducible,
   because GPU sampling is not bit-exact (the pre-port and ported cafe, same seeds,
   agreed on mean radiance to 0.03%; per-pixel differences are sampling noise).
4. **Two cache directories.** Seed- and env-independent meshes and textures go in
   `ctx.shared_dir` (built once, reused by every seed). Anything that depends on
   the seed or env goes in `ctx.assets_dir`. Write nowhere else. Bump
   `asset_version` whenever a recipe changes, so stale caches are not reused.
5. **The caller sets the Mitsuba variant** before calling `build`. A scene must
   work on every `*_ad_rgb` variant and on `scalar_rgb`.
6. **Stay on the prebuilt wheel.** Stock Mitsuba plugins only. Nothing that needs
   compiling Mitsuba from source.
7. **Real assets come through `web_assets.py`.** Scanned CC0 models from Poly Haven
   (furniture, tools, plants, trees) usually beat procedural ones, so prefer them for
   recognisable objects. Pin each one locally (`python renderer/mitsuba/web_assets.py pin
   <id>`, which writes URLs and md5s to the committed `scenes/assets/web_manifest.json`),
   then `W.model(id)` downloads, checks and converts it on whatever machine builds the scene
   and `W.add(b, m, W.place(...))` adds it. The downloads are not committed. Other CC0
   sources need the same pin-and-verify treatment before use; nothing without a CC0
   licence. Mitsuba bitmaps put v = 0 at the image top, like glTF (measured).
8. **Every scene has light**, and each emitter gets a `sampling_weight`
   proportional to its power (see `SceneBuilder._shape`). Uniform picking made
   the cafe sun sparkle among about 40 lamps.
9. **Use these BSDFs for mirrors, glass and water**: `dielectric`,
   `thindielectric`, `roughdielectric`, `conductor`, `roughconductor`. The shared
   side builds the `coc_valid` mask from them (section 7), so a glass surface
   made with another plugin would be silently treated as matte.
10. **Named focus points** are `{"teapot": [x, y, z]}` or a list of points. Every
    scene needs at least one, and `View.focus` must name one.

## 3. Cameras

**Random camera sampling [done: `sampler.py`].** Sample `i` of a
`(scene, scene seed, env, sampler seed)` is a pure function of those and `i`
(`SeedSequence` keyed on all of them), so any index can be regenerated alone and
array jobs can split the range. It draws a position uniformly from `camera_box`
minus `exclude_boxes`, a target from `target_box` (so the direction is random too),
a roll, and a focal length (log-uniform in the intersection of 24 to 135 mm and the
scene's range). It renders the 1-spp G-buffer and rejects the pose when: pitch is
beyond 75 degrees; the target is under 0.5 m away; more than 50% of pixels are sky;
the 1st-percentile surface depth is under 0.3 m; or the view is one flat surface
(p95/p5 depth under 1.5). Rejection reasons are counted and printed. On the cafe,
with no `exclude_boxes`, about 1 attempt in 5 is rejected (mostly flat views; 51 of 251 over 200 poses), so
exclude boxes are only needed if a new scene's rate is much higher. The thresholds
are constants at the top of `sampler.py` to tune. You still guarantee the boxes are
honest: a tight box around places you checked beats a big box with bad corners.

**Camera roll.** The sampler also draws `roll_deg`, a rotation of the camera
about its optical axis (the tilted "Dutch angle" photographers use). It is part
of the camera, not an image rotation: the sharp and DoF passes are rendered with
the same rolled `up` vector, depth stays planar z along the optical axis, nothing
is resampled, and it is physically exact. Positive roll is a right-hand rotation
of `up` about the forward axis (camera rolled clockwise, picture content turns
counter-clockwise). Implemented in `render.py` (`--roll`) **[done]**. Pitch and yaw
come from the look-at, so together with roll the sampler covers all three camera
rotations while the scene stays upright in world space.

**Lens.** The sensor is 24 mm tall for every scene. `lens_mm` and `f_number`
are the ranges the scene looks right for, and the sampler stays inside them.
Focus distance is **planar**: `focus = dot(p - origin, forward)`, the same
quantity stored in `depth.npy`. Focus targets must be visible and unoccluded
(checked against the G-buffer).

**CoC.** Diameter in pixels at planar depth z, focused at s, with focal length f,
f-number N, image height H and sensor height h:

    CoC_px = H f^2 |z - s| / (N z s h)

This is what Mitsuba's `thinlens` renders (measured against rendered discs, within
0.3%). The textbook form divides by `(s - f)` instead of `s`; it was 16-29% too
large at 135 mm and is not used. `traditional_dof.py` keeps the textbook form on
purpose, because it ports the DoFScene shader.

## 4. Aperture shape (reserved)

Today every aperture is the circular disc that Mitsuba's `thinlens` gives. We want
to add other shapes later (polygonal blades, cat's eye, anamorphic), so the place
for them exists already, although nothing but `disc` is implemented:

- `scene_api.APERTURES = ("disc",)` lists the allowed shapes.
- `camera.aperture = {"shape": "disc", "rotation_deg": 0}` is in every `metadata.json`,
  and `"aperture"` is part of the `lens_<hash>` and `sample_id`, so samples of a new
  shape never collide with existing ones. A model that conditions on lens settings
  should take the shape as an input.
- A new shape is a change inside `Lens` in `render.py`, because `thinlens` has no
  shape parameter. Candidate approaches, none built or verified: accumulate pinhole
  renders from points sampled over the aperture shape, with an off-axis frustum so the
  focus plane stays registered; or a custom sensor plugin, which would break rule 6
  (no compiling Mitsuba) and needs both owners to agree.
- Augmentation changes with it. For a disc, flips and 90-degree turns are exact. For a
  polygon they are exact only if the transform helper also rotates (or mirrors)
  `aperture.rotation_deg`, and camera roll must add to it, because the aperture
  turns with the camera. `coc.npy` would then mean the area-equivalent disc diameter.
- The traditional gather method assumes a disc, so comparisons against it only hold
  for `disc`.

## 5. Environments

`ctx.env` is one of `clear, overcast, snow, sand, wet`. A scene declares what it
implements in `envs`, and `clear` is mandatory. An environment changes surface
albedo, ground, sky and light colour, never the depth structure. A scene raises
`ValueError` for an environment it did not declare (`build_checked` does this for
you). Do not fake an environment that makes no physical sense (snow in an
aquarium).

`haze` is **not** a scene preset. The shared renderer applies it to any `clear`
scene as a bounded fog volume with `volpath` (Mitsuba's `path` silently ignores
a camera medium, an unbounded one blacks out the sky, and the depth pass does not
see haze, so it needs one owner). Do not implement haze in a scene.

Use `env_kit.py` **[done, v0]** so every author gets the same sun, sky and ground:
`sky(env, sun_direction, scale, sampling_weight)` returns the emitter and
`ground(env)` a ground BSDF. `clear`, `snow` and `sand` use `sunsky`; `overcast` and
`wet` use a constant sky calibrated to be 1/4 as bright as the clear sky (1/6 gave
a pavement scene median of 0.014, under the band). `sun_direction` points toward the
sun, like the cafe's. Measured at scale 1.0 on a test ground plane and sphere
(128 spp, metal): clear 0.088, overcast 0.0225, snow 0.315, sand 0.169, wet 0.0145
(the dark asphalt keeps wet under the 0.02 band, so wet scenes need a brighter
ground or sky). Not verified: the sky colour and sun disc (the test cameras never
looked at the sky), and snow and sand differ from clear only by ground albedo and
turbidity. Do not copy the cafe's `DAYLIGHT_SCALE = 60`: it is tuned for an
interior lit through windows, and an outdoor scene at scale 1 already lands in the
band. Every environment should appear in both groups, so the model cannot learn
"artificial implies clear".

## 6. Splits

Scenes are assigned to exactly one split in `scenes/splits.json`, never inside
the scene file, so an author cannot grade their own scene:

| Split | Scenes | Purpose |
|---|---|---|
| `train` | the 10 artificial, the natural ones | training |
| `val` | 3, all in-distribution (like the training mix) | tuning and model selection, used freely |
| `test` | 1, in-distribution | final numbers, touched rarely |
| `test_ood` | 1, unlike anything in `train` | does it generalise? |

All seeds, environments, cameras and crops of a scene stay in that scene's split,
because seeds and presets of one scene are near-duplicates. `tests/test_scene_api.py`
fails if a scene is missing from `splits.json` or listed twice.

Held-out scenes should mirror the training mix (indoor and outdoor artificial, one
natural or hybrid), and **should be built by the other author**, so they do not
share one author's props, textures and style with the training scenes. The
`test_ood` scene must still be physically valid, but differ from training on depth
structure, material and lighting together (candidates: a macro closeup, an ice cave,
an underwater reef). The five scenes are still to be chosen.

## 7. What the shared side produces

For each sample (scene, scene seed, env, camera pose and roll, lens, render seed):

```
dataset/mitsuba/<scene_id>/s<seed>/<env>/
  manifest.jsonl                                          # one line per (pose, lens) sample
  p<index>_<hash>/                                        # a pose: position, target, roll, focal length
    pose.json, depth.npy, gbuffer.exr, sharp.exr, .done
    lens_<hash>/dof.exr, coc.npy, metadata.json           # a lens variant: focus distance, f-number
```

A pose includes the focal length, because it changes the field of view and so
the sharp image, depth and G-buffer. Variants within a pose differ only in focus
and f-number. `.done` makes a rerun skip finished poses. `coc.npy` is the signed
CoC diameter in pixels (positive behind the focus plane, negative in front, sky at
its at-infinity value), float32 at image resolution. `dataset/mitsuba/` is
gitignored.

Sharp, depth and G-buffer are stored once per pose and shared by its lens variants. Depth is kept once, as float32
`.npy`. World position is derivable from depth and the camera, and CoC is
analytic, so neither is stored twice. Expect about 100 MB per full sample at
1080p. Millions of tiny files are avoided with shards plus a manifest.

`metadata.json` records: `contract_version`; scene id, seed, env, `fingerprint`,
`asset_version`; git commit and a dirty flag; Mitsuba version and variant;
resolution, sensor height, focal length, f-number, focus distance, roll, camera
origin/target/up and intrinsics; render seeds and spp per pass; exposure; the CoC
formula; timings. `sample_id` hashes only the rounded sampler inputs, not the
git commit, and each sample's random stream is derived from its ids with
`SeedSequence`, never drawn from one shared stream. A changed `fingerprint`
marks old samples stale. **[done in `render.py` and `sampler.py`: metadata, `--roll`, poses, the manifest,
resume; shared, todo: sharding into archives, `coc_valid`]**

**`coc_valid`.** Depth is the first surface, but behind a mirror, window or glass
the eye focuses on the virtual image, so analytic CoC is wrong there. The mask
marks pixels whose first hit has a specular or transmissive BSDF (rule 9), and the
loss can ignore or down-weight them.

**Augmentation after rendering.** Only horizontal flip and 90-degree rotations
are applied to finished passes, because they are exact for beauty, depth, CoC and
masks (the thin-lens aperture is a disc and the pixel filter is isotropic).
World-space G-buffer P and N and the camera intrinsics and extrinsics need the
same transform, so a shared helper does it. Arbitrary-angle rotation is **not**
done on finished images: it resamples noise, interpolates depth across
silhouettes, and rescales CoC. Tilt comes from camera roll above. Vertical flips
and 90-degree turns produce sideways or upside-down pictures, which breaks
gravity and sun cues, so the training side should give them a low probability.
Crops are free and preserve everything (the CoC does not depend on the crop).

## 8. Exposure, noise and balance

- **Exposure.** All values are raw linear EXR radiance of the named pass.
  `--exposure` only affects PNG previews. There is no downstream exposure knob,
  because the EXRs are the data. The sharp and DoF passes have equal mean
  luminance (0.1805 vs 0.1798 on the cafe), so f-number does not change
  exposure. The median luminance of the sharp pass should be between 0.02 and 0.5.
  A median is a weak summary (the cafe is 0.04, p5 0.002, p99.9 21), so the check
  applies per `(view, env)`, and the sampler also gates each random pose and logs
  the number. The night scene is exempt from the band, by declaring it in its
  `tags`.
- **Noise.** Ground truth must be clean. The DoF pass uses at least 2048 spp
  (more where `spp_hint` says), and sharp and DoF use independent seeds.
  Fireflies are pixels more than 4x their 5x5 median plus 0.5, excluding emitter
  pixels; fewer than 0.1% is the target. Each sample stores a noise estimate
  (variance between two half-spp renders), and a check requires
  `mean(dof) = mean(sharp)` within 1%. There is no denoising unless we decide on
  one later.
- **Balance [done in `sampler.py` v0].** The target p95 blur (as a fraction of image
  height, 0.05% to 4%, which is 0.5 to 43 px at 1080p) is drawn log-uniformly from what
  the pose's focal length and the focus can reach with the allowed f-numbers, then the
  f-number is solved for it. Asking a wide lens for a huge blur is physically
  unreachable, so the achieved distribution is skewed toward mid and large blur
  (measured on the cafe over 400 lens draws: 4, 24, 45, 72, 73, 58, 63, 55 per bin from
  small to large blur; 6 of 400 fell outside the range). Focus comes from a named point
  (if visible) or a depth quantile, 50/50. Rebalancing is a later step: over-generate
  cheap poses and lenses with `--dry-run`, then render only a balanced subset. Every
  dataset report states counts by scene, group, env and CoC bin.
- **Exposure gate in practice.** `pose.json` records the sharp pass's median
  luminance and `exposure_gate_ok`. On the first two cafe poses, one was too dark
  (0.008). Random poses include dull dark views, so we still need to decide whether to
  reject at generation time with a cheap low-spp probe, or render everything and filter.

## 9. Budgets (per scene, RTX 5000 Ada)

| | Limit |
|---|---|
| Triangles | 8M (one scanned tree is about 2M) |
| Textures | 1 GB total on disk |
| `build` with a warm shared cache | 60 s |
| 1920x1080 at 1024 spp, one beauty pass | 30 s |
| GPU memory at 1080p with 256 spp chunks | under 24 GB (the card has 32) |

Render time per sample is budgeted by total pass time, because the DoF pass needs
more spp than the sharp pass.

## 10. Acceptance checklist

Run `python renderer/mitsuba/check_scene.py <id>` (add `--envs`, `--views`, `--spp`).
It prints PASS, WARN or FAIL for each item below and exits 1 on any FAIL.

- [ ] Definition and bundle validate, the scene is listed in `splits.json`, build time and triangles are within budget.
- [ ] Builds twice with the same seed with an identical `fingerprint`, and a different seed gives a different one.
- [ ] A preview renders for every view and every declared env, with no NaN or inf.
- [ ] Median luminance of the sharp pass is within [0.02, 0.5] (night scenes warn only), and `mean(dof)` matches `mean(sharp)` within 1%.
- [ ] Firefly fraction is reported (at preview spp it only warns; the check also flags three pixels at each corner of a rectangular emitter, a known quirk), and sky fraction, depth range and CoC percentiles look sane.
- [ ] By hand: `viewer.py --scene <id>` shows no camera in `camera_box` clipping into geometry, and `python renderer/mitsuba/sampler.py --scene <id> --count 200 --dry-run` shows a sane rejection rate.
- [ ] `tags` and `description` are filled in.

## 11. What is built

| Part | State |
|---|---|
| `scene_api.py` (types, loader, validators, fingerprint, splits) | done, tested |
| `scene_kit.py` (shared builder, atomic multi-part cache) | done |
| `scenes/cafe.py` (reference scene) | done, matches the pre-port render |
| `render.py --scene --env --roll --scene-seed`, richer metadata, signed `coc_map` | done |
| `viewer.py --scene` | done |
| `check_scene.py` (the checklist as one command) | done, tested on the cafe |
| `env_kit.py` (sun, sky and ground presets) | v0 done, tested; sky and sun disc unverified |
| `sampler.py` (poses, roll, lens draw, dataset layout, manifest, resume, `--dry-run`) | v0 done, tested on the cafe (2 samples end to end) |
| Haze volume, `coc_valid` mask, transform helper, sharding into archives | shared, todo |
| Balanced subset selection (over-generate with `--dry-run`, render a subset) | shared, todo |
| Aperture shapes other than `disc` | reserved, see section 4 |
| The 5 held-out scenes (3 `val`, 1 `test`, 1 `test_ood`) | to choose |

The cafe's `exclude_boxes` are empty on purpose: the sampler's rejection rate there
is low enough that measuring furniture is not worth it yet.
