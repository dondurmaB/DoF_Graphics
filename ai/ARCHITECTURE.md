# AI stage: input/output contract (draft for the next meeting)

## Goal
Learn to produce a depth-of-field image that is close to the ray-traced Cycles reference
(always correct, slow) but cheap like the OpenGL CoC pass (fast, but wrong at silhouettes,
occlusion, and bokeh shape). The three renderers share one scene, camera and light
(`src/main.cpp` <-> `tools/raytraced_reference/render_dof.py`).

## Proposed model: "learned DoF refinement"
| | Tensor | Source |
|---|---|---|
| Input 1 | Sharp RGB, `H x W x 3`, float [0,1] | OpenGL scene FBO color (view 1, no blur) |
| Input 2 | Linear depth in meters, `H x W x 1` | OpenGL scene FBO depth, linearized (same math as screen.frag) |
| Input 3 | Signed CoC in pixels, `H x W x 1` | Computed analytically from depth + lens (screen.frag) |
| Input 4 (optional) | Baseline DoF RGB, `H x W x 3` | OpenGL BasicDoF pass output |
| Lens scalars | focus distance, f-number, focal length, sensor height | Fed as constant channels or via FiLM conditioning |
| **Output** | DoF RGB, `H x W x 3` | Predicted |
| **Target** | Cycles DoF RGB, `H x W x 3` | `reports/raytraced_dof/rt_*.png`, same camera/pose/resolution |

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
