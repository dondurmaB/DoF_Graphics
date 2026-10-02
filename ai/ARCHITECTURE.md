# AI stage: input/output contract (draft for the next meeting)

## Goal
Learn to produce a depth-of-field image that is close to the ray-traced Cycles reference
(always correct, slow) but cheap like the OpenGL CoC pass (fast, but wrong at silhouettes,
occlusion, and bokeh shape). Both renderers load one scene, camera and light from
`scene/cafe.scene` (via `src/SceneFile.cpp` and `tools/scene/scene_loader.py`), so a matched
pair of images differs only in how the lens was simulated. Every Cycles sidecar JSON records
the scene file's sha256, which is what lets a stored pair be trusted months later.

## Proposed model: "learned DoF refinement"
| | Tensor | Source |
|---|---|---|
| Input 1 | Sharp RGB, `H x W x 3`, **linear, unbounded** | OpenGL scene FBO color (view 1, no blur) |
| Input 2 | Linear depth in meters, `H x W x 1` | OpenGL scene FBO depth, linearized (same math as screen.frag) |
| Input 3 | Signed CoC **diameter** in pixels, `H x W x 1` | Computed analytically from depth + lens (screen.frag) |
| Input 4 (optional) | Baseline DoF RGB, `H x W x 3`, linear | OpenGL BasicDoF pass output |
| Lens scalars | focus distance, f-number, focal length, sensor height | Fed as constant channels or via FiLM conditioning |
| **Output** | DoF RGB, `H x W x 3`, linear | Predicted |
| **Target** | Cycles DoF RGB, `H x W x 3` | `reports/raytraced_dof/rt_*.png`, same camera/pose/resolution |

### Colour space and range (changed in experiment 18)
Inputs 1 and 4 are **linear radiance from an `RGBA16F` target, not [0,1] sRGB**. The alley
contains emitters (windows, bulbs, the neon sign) whose radiance exceeds 1.0 by design, and the
whole reason the raster pass now renders to a float target is that clipping them before the
defocus gather destroys exactly the bokeh highlights the network is supposed to learn.

Consequences for the dataset:
- Dump inputs 1 and 4 as float (EXR or `.npy`), not PNG. A PNG dump would silently clip and
  gamma-encode them and the training pair would no longer be physically matched.
- Cycles targets are now 32-bit linear EXR with no view transform when rendered with
  `--ground-truth` (experiment 19), so **train in linear**. This decision is made: the earlier
  8-bit PNG path clipped every value above 1.0 and quantized to 256 levels, which destroyed the
  highlights whose bokeh is the thing being learned. The denoiser is also off in that mode, for
  the same reason it would be wrong to train against it: it invents plausible detail, and smooth
  out-of-focus discs are what it rewrites most.
- Every reference carries a measured error bar in its sidecar JSON (`--convergence`): the frame is
  rendered twice at different sample counts and the difference reported. Use it as a floor on the
  loss. A model that reaches the reference's own noise level has converged to the data, and
  further training is fitting noise.
- Input 3 is a signed *diameter*. Note that `screen.frag` halves it to get its gather radius;
  if the network is fed the diameter, keep that convention documented in the dataloader, because
  confusing the two was a real bug in the renderer for three experiments.

### Known systematic difference to model
The raster sky term has no occlusion, while Cycles darkens creases and undersides where the sky
is genuinely blocked. So a matched pair differs by more than lens simulation: there is a real
ambient-occlusion component in the residual. Either accept that the network learns it along with
the defocus behaviour, or add a screen-space AO term to the raster pass first so the residual is
defocus only. Worth deciding explicitly rather than discovering it in the loss curves.

Architecture suggestion: small U-Net taking inputs 1-3 (+4 as residual base), predicting a
residual added to the baseline DoF image. Loss: L1 + perceptual (or SSIM); evaluate with
PSNR/SSIM and a silhouette-region mask (where the baseline is known to fail).

## Why these inputs
Depth and CoC are things a rasterizer gets for free, so the network does not need to guess
geometry; it only learns what the gather blur cannot do (occlusion, foreground bleed,
bokeh shape). Monocular depth *estimation* (RGB -> depth) is a separate, later option if the
project must work on photos without a depth buffer; it changes Input 2 into a predicted value.

## Dataset generation (needs to be built)
1. Sample camera poses + lens settings (focus distance, f-stop).
2. For each: save OpenGL sharp color / depth / CoC / BasicDoF, and the matching Cycles render.
3. Keep resolution fixed per dataset (e.g. 512x512 crops from 1200x1200) and split by camera pose, not by pixel.

## Open decisions for the professor
- Refine the baseline (proposed) vs. replace the whole DoF pass?
- Depth from the rasterizer (proposed) vs. predicted from RGB?
- Dataset size / how many Cycles renders we can afford (each is slow).
- Leave the ambient-occlusion difference in the residual, or add screen-space AO to the raster
  pass so the network only has to learn defocus?
- Train on matched references (emitters light nothing, so the residual is defocus only) or on
  `--full-gi` ones (physically complete, but the residual then contains global illumination too)?
  Matched is the current default and is the cleaner experiment.
