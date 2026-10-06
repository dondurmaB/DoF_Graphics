# Mitsuba Cafe

A physically based cafe interior for [Mitsuba 3](https://mitsuba.readthedocs.io/), built to be the realistic counterpart of the OpenGL `DoFScene`. It uses path-traced global illumination, sun and sky light through real window openings, soft shadows, glass that refracts, metals, glazed ceramics and procedural textures. From one camera it renders three pixel-aligned passes:

| Output | What it is |
|---|---|
| `sharp.exr` / `.png` | Pinhole camera, all in focus: the input a post-process DoF method would get |
| `dof.exr` / `.png` | Thin-lens camera, the physically correct depth of field: the ground truth |
| `depth.exr` / `.npy` | Planar z-depth in meters along the optical axis (`inf` where the ray escapes) |
| `gbuffer.exr` | World position, shading normal, albedo, shape index |
| `normal.png`, `depth.png` | Previews |
| `metadata.json` | Lens, focus, aperture, intrinsics, extrinsics, spp, timings, CoC statistics |

The EXRs are linear radiance and are the data. The PNGs are ACES-tonemapped previews only.

## Quick start

```sh
conda env update -f environment.yml --prune     # adds mitsuba==3.9.1
conda activate dof-graphics

python renderer/mitsuba/render.py --preview                  # 640x360, 32 spp, seconds
python renderer/mitsuba/render.py --res 1280x720 --spp 1024  # clean enough to judge
python renderer/mitsuba/render.py --list-targets             # named focus points
python renderer/mitsuba/render.py --focus-target books --f-number 1.4 --view close
```

`--variant auto` picks `cuda_ad_rgb`, then `metal_ad_rgb` (Apple GPU), then `llvm_ad_rgb`, then `scalar_rgb`. On an M4 Max, 1280x720 at 2048 spp takes about 70 s per beauty pass.

Other options: `--scene <id>`, `--scene-seed`, `--env`, `--roll degrees` (tilt the camera about its optical axis), `--view` (a view the scene declares; the cafe has `home|close|wide`), `--lens mm`, `--sensor-height mm`, `--f-number`, `--focus meters` (overrides `--focus-target`), `--passes sharp,dof,gbuffer`, `--dof-spp`, `--max-depth`, `--seed`, `--exposure EV` (previews only), `--rebuild-assets`.

## Interactive viewer

```sh
python renderer/mitsuba/viewer.py      # then open http://localhost:8765
```

This is a progressive path tracer in the browser. The image refines while the camera is still and restarts when anything changes. Camera and lens values are pushed into the already-compiled kernel as opaque parameters, so moving or refocusing costs no recompile. At 960x540 on an M4 Max it reaches 64 spp in about 2 s.

- **Controls**: drag to look, `WASD` to move, `Q`/`E` for down/up, `Shift` for speed. Click to focus on the surface under the cursor, scroll to change the focus distance, `[` `]` to change the aperture, and `1`-`5` to switch between the DoF, sharp, depth, CoC and **traditional** views.
- **Traditional mode** blurs the pinhole render with `DoFScene`'s screen-space gather, running on the GPU in about 23 ms. The pinhole image doesn't depend on focus or aperture, so changing them re-blurs the samples already accumulated instead of restarting. Press `1` and `5` to flip between the physical lens and the traditional approximation.
- **Panel**: sliders for focus, f-number, focal length and exposure (exposure only re-tonemaps and does not restart the render), named focus targets, camera presets, and resolution.
- **Save snapshot**: writes the current accumulation as EXR and PNG plus camera metadata to `output/mitsuba/viewer/<timestamp>/`.

## Traditional DoF vs. path-traced DoF

```sh
python renderer/mitsuba/render.py --res 960x540 --spp 1024 --dof-spp 2048 --out output/mitsuba/trad
python renderer/mitsuba/traditional_dof.py output/mitsuba/trad
```

`traditional_dof.py` applies `DoFScene`'s depth-aware 64-tap gather (a faithful port of `shaders/dof_scene/screen.frag`) to `sharp.exr` and `depth.npy`. It writes `traditional.exr/.png`, a `compare.png` grid (sharp, traditional, path-traced, error amplified 4x), and `traditional_metrics.json`. The GPU version used by the viewer, `gather_dof_gpu`, matches the numpy reference to within 1/255.

Measured at the home view (50 mm, f/1.8, focus 1.89 m), in mean absolute error on a 0..255 scale against the path-traced thin lens:

| | Overall | Near depth edges (20% of pixels) |
|---|---|---|
| No blur (sharp image) | 13.45 | 17.24 |
| Traditional gather | 4.53 | 7.96 |

The error concentrates at silhouettes: bright halos around the midground chairs and lamp shades, and the rims of each bulb's defocus disc. A single pinhole image cannot show what the lens sees behind an edge.

## HPC

**Interactive viewer on a cluster GPU:**

```sh
sbatch renderer/mitsuba/hpc/viewer.slurm                   # on the login node, from the repo root
grep "tunnel:" logs/cafe-viewer-<jobid>.out                # prints the node and a token URL
ssh -N -L 8765:<node>:8765 login-student-lab.mbzu.ae       # on your laptop; keep it open
open "http://localhost:8765/?t=<token>"
scancel <jobid>                                            # when done
```

On an RTX 5000 Ada it reaches about 180 spp/s at 960x540, about 5.5x an M4 Max, so a still view converges to 4096 spp in about 23 s. On the cluster the server listens on the node's network interface, so it requires the random per-job token. Without the token it returns 403.

```sh
sbatch renderer/mitsuba/hpc/render_cafe.slurm
sbatch --export=ALL,RENDER_ARGS="--focus-target menu --f-number 1.4" renderer/mitsuba/hpc/render_cafe.slurm
```

The script is set up for `login-student-lab` (partition `ws-ia`, one RTX 5000 Ada 32 GB per node, driver 570). On that GPU, 1920x1080 at 1024 spp takes about 10 s per beauty pass, about 7x the M4 Max's throughput. Submit only from the login node; never render on it. The PyPI `mitsuba` wheel ships the CUDA variants for Linux and needs only an NVIDIA driver, not a CUDA toolkit. `llvm_ad_rgb`, the CPU fallback, needs `libLLVM` on the node. The job renders 1920x1080 at 4096 spp in chunks of 256 spp, so GPU memory stays bounded at any sample count.

The procedural assets (about 14 MB of PLY and PNG files) are generated into `renderer/mitsuba/generated/` on first run and reused afterwards. Every generator is seeded, so the laptop and the cluster build identical scenes.

## Scene

Units are meters, y is up, and the camera looks toward -z, the same convention as `DoFScene`.

- **Room**: 7.6 x 10.5 x 3.2 m with a plank floor, plaster walls, a green wainscot and ceiling beams. The left wall has two windows with thick reveals, mullions and sills. Outside there is a street and a facade across the road.
- **Hero table (foreground)**: a teapot, a cup of coffee on a saucer with a spoon, a wine glass, a croissant on a plate, and a vase of flowers.
- **Midground**: three more tables with bentwood chairs, a laptop with a lit screen, books, candles in frosted glass, and potted plants.
- **Background**: a marble-topped tiled counter with an espresso machine, a glass cake dome, cookie jars and a fruit bowl. Behind it, shelves of green, amber and clear bottles, a chalkboard menu, a round mirror, paintings and string lights.
- **Lights**: `sunsky` (sun and Hosek-Wilkie sky, late afternoon, from the street side), four enamel pendants, four bare globe bulbs, 28 string-light bulbs, two candles and the laptop screen.

About 626k triangles in total. Glassware is modelled as closed shells with real wall thickness, so the dielectrics refract correctly.

Files: `procedural.py` holds the numpy mesh primitives (lathe, sweep, box, leaf) and procedural textures. `props.py` holds the recipe for each object. `scenes/cafe.py` holds the cafe's materials, layout and lights, and declares it as a `SCENE` under the shared contract (`SCENE_CONTRACT.md`, `scene_api.py`, `scene_kit.py`). `render.py` holds the camera, the passes and the outputs.

## Design notes

- **Thin-lens model**: the same as `DoFScene`/`DoFApproaches`. The vertical FOV comes from focal length over a 24 mm sensor height, and the aperture diameter is focal length / f-number. Mitsuba's `thinlens` focuses on a plane, so `focus_distance` is planar z-depth, the same quantity written to `depth.exr`. With the defaults (50 mm, f/1.8, focus on the teapot at 1.89 m, 960x540), the CoC diameter at infinity is 17 px.
- **Emitter sampling weights**: Mitsuba picks emitters uniformly by default. With about 40 small lamps, the sun got 1 light sample in 40, and every sunlit surface sparkled no matter the spp. Each lamp now gets a `sampling_weight` proportional to its luminous power, and the sky gets as much weight as all the lamps together. At equal spp, this cut firefly pixels from 2.1% to 0.15%.
- **Daylight scale**: `sunsky` at unit scale puts sunlit plaster near radiance 0.1, about three orders of magnitude below a bulb. Sun and sky are scaled together by `DAYLIGHT_SCALE = 60`, which keeps their ratio physical and puts a sunlit patch a few stops below the bulbs, as in a real daytime interior.
- **G-buffer at 1 spp**: averaging many samples would blend foreground and background depth along silhouettes into depths that exist nowhere in the scene. That is the wrong ground truth for DoF, where silhouettes are exactly where methods fail. One jittered sample per pixel keeps every depth on a real surface.
- **What remains**: caustics through glass (for example, sunlight focused by the wine glass) converge slowly in a unidirectional path tracer. They are physically present and need high spp. Budget 2048 or more for the DoF pass.
