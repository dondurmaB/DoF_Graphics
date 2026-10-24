# Render noise in the dataset (deferred)

Status: known, not being fixed now (decided 2026-10-24). Revisit before or during model training.

## What we found

Inspection of 20 samples from `train10k` (2 poses per scene, downloaded to `inspect_data/`, previews in
`inspect_data/previews/`, 2x pixel crops in `inspect_data/previews/noise_crops_2x.png`) on 2026-10-24.

- Both the sharp input (1024 spp) and the DoF target (2048–4096 spp) carry Monte Carlo noise.
- It is worst on glossy and glazed materials in direct sun: the cafe teacup and the desk mug sparkle all over,
  and the sunlit desk surface has fireflies. The white cars in night_street are grainy.
- It is in the DoF target too: the defocused teacup keeps its pixel-scale sparkle, so this is render noise,
  not texture.
- Mild elsewhere: museum, rooftop and workshop show light grain only.
- Depth is fine: `depth.npy` is float32 with ~1.8M distinct values per frame. Banding seen in the depth previews
  came from the image viewer's colour reduction, not from the data.

## Why it matters

- Noise in the **target** is mostly harmless: it is unbiased, so an L2 loss learns the mean (Noise2Noise).
- Noise in the **sharp input** is the real issue: the model learns to expect sparkly inputs, which real
  photographs do not have, and may learn to pass the sparkle through.

## Options (none needs a re-render)

1. **Denoise the sharp input only** with the OptiX denoiser (`mi.OptixDenoiser`), guided by the albedo and
   normals already saved in each pose's G-buffer. Runs on the CIAI GPUs over the stored EXRs. Preferred.
2. Also denoise the DoF target. Riskier: denoisers can soften the very blur edges the model must learn.
3. Leave the data as is; add noise augmentation to the input during training, or rely on the model to cope.
4. For future renders: more sharp spp, or a per-sample radiance clamp (removes fireflies, slightly biased).

Keep the G-buffers until this is decided: option 1 depends on them.
