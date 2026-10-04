# Cycles inspection renders

The current checkpoint is the café scene and matched direct lighting. **These renders are not yet convergence-certified ground truth.** Every sidecar says `ground_truth: false`. Phase 3's independent convergence renders and error bars remain unfinished.

Run from the project root:

```sh
python tools/raytraced_reference/render_dof.py --dry-run
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
  --python-exit-code 1 --python tools/raytraced_reference/render_dof.py -- \
  --fstops 1.2 22 --sharp --samples 128 --no-denoise
```

`--scene` selects `scene/cafe.scene` by default; `scene/alley.scene` remains available. Camera pose, sensor, focus, capture dimensions and optional imported transform come from that file. The default café comparison deliberately uses an 85 mm lens, focus 2.5 m, and f/1.2 plus f/22: f/1.2 gives tens of pixels of far-wall blur at 1200×1200, while f/22 is the near-pinhole control. Non-default scenes inherit their own lens unless `--lens` overrides it. `--width`, `--height`, `--lens`, `--sensor-height`, `--focus`, `--position`, `--yaw`, and `--pitch` override the loaded settings. `--fstops` accepts discrete stops. At most 12 frames per invocation. `--preview` selects 32 samples; otherwise 128, unless `--samples` overrides it. `--device cpu` disables automatic Metal selection.

Each frame writes PNG display preview, 32-bit linear RGB EXR, lossless float PFM companion, and settings JSON. `--output` chooses the directory. PNGs clip HDR highlights and must not be treated as linear measurements. Repeated names overwrite prior outputs; use separate directories for inspection runs. Names encode focus, aperture, nondefault lens and sharp state: `rt_focus2.5m_f1.2.png`, `rt_focus2.5m_f1.2_sharp.png`, or `rt_focus2.5m_f1.2_85mm.png`. The executable contract test pins the C++ and Python name builders. Pose and sensor changes still require separate output directories: names alone do not establish a valid comparison.

Matched mode uses diffuse Lambert plus **primary-camera-only, unoccluded ambient emission**, with mesh-light sampling disabled for that fill. World radiance is camera-visible only. Diffuse/glossy continuation is disabled; emissive props do not illuminate surfaces. This deliberately non-physical model matches the OpenGL equation, including dark interior surfaces that cannot see the sky. Ordinary fill emission was measured to brighten other surfaces even with diffuse continuation disabled; the camera-ray restriction is essential.

`--full-gi` restores world illumination, indirect transport and emitter illumination. It is explicitly labelled `full_gi_not_comparable`, has finite bounce limits, and is not radiometrically equivalent to OpenGL. Neither mode has a convergence certificate yet. Inspection uses explicit seed 16, no adaptive sampling, no clamping, max 12 bounces; matched diffuse/glossy/transmission continuation is 0, full-GI uses 4/4/12. Denoising remains available by default for legacy previews; use `--no-denoise` for inspection measurements. This checkpoint's reported captures all used that flag.

Both loaders produce the geometry. Blender applies the one world conversion `(x,y,z) -> (x,-z,y)` and imports analytic normals; no geometry is hand-placed here. The café's teapot is off by default. `--import-mesh` explicitly adds the OBJ with the scene's `imported` transform, using `T*Ry*Rx*Rz*S` before the common world conversion. Source units are not inherently metres. The chosen scale/rotation maps them into the project's metre convention. Missing optional assets are fatal when requested.

Verified on Blender 5.2.2 LTS with Metal / Apple M4 Max. The normal API fallback is documented in source but was not exercised; it can alter smooth shading. Matched lighting still differs at raster/pixel and shadow boundaries: OpenGL uses a biased, filtered 4096² shadow map and Cycles uses ray intersections. They do **not** differ only in DoF. See [the checkpoint report](../../notes/graphics/cafe_phase1.md) for linear measurements, images, rejected specular mapping, and pending work.
