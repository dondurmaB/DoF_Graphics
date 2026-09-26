# Experiment 16 — Multi-View Depth of Field

## Purpose

Experiment 15 showed that a single-layer gather fails at silhouettes and that more taps do not fix it. This experiment renders the identical scene with the identical shading through a method that has neither failure, and identifies precisely which structural property removes them.

The method stops trying to reconstruct the lens integral from one image and evaluates it instead:

```text
I(p) = (1/A) ∫ L(p, u) du        u = a point on the aperture disc
              A
```

Each `u` is a pinhole camera translated onto the lens. Render the whole scene once per `u`, average the results. `renderMethodTwo()` in `src/dof_approaches/main.cpp` distributes N aperture points by the golden angle at radius `r·sqrt(i/N)`, which gives a uniform area distribution over the disc.

## Why the Focus Plane Stays Sharp

Translating the eye sideways is not enough on its own: it would swing the whole image and blur everything, focus plane included. `renderScenePass()` pairs each lateral eye offset with a projection **shear** that cancels that offset at one depth:

```cpp
glm::mat4 shear(1.0f);
shear[2][0] = -eyeOffset.x / focusDistance;
shear[2][1] = -eyeOffset.y / focusDistance;
const glm::mat4 viewProjection = projection * shear * view;
```

Camera space looks down −z, so a point on the focus plane has z = −focusDistance. Written as x″ = x + c·z, cancelling the offset there requires c = −offset/focusDistance, which is where the negative sign comes from. Two consequences follow, and they are the whole experiment:

1. **Focus-plane points project to the same pixel in every view.** Averaging cannot move them. Sharpness is not achieved by choosing a small radius — it is structurally impossible to lose. Experiment 15's Error 1 cannot occur.
2. **Defocus is parallax, not a filter.** An off-focus point lands on different pixels in different views, and the spread of those landings *is* the circle of confusion — correct size, correct shape, aperture shape included. Method 2 contains no circle-of-confusion formula anywhere; compare `postprocess.frag`, which is almost entirely that formula.

Because each pass runs its own complete depth test, background hidden from the lens centre is genuinely visible from off-centre aperture points and genuinely rendered. Averaging then produces correct partial coverage, so a thin foreground turns translucent. That is Experiment 15's Error 2, fixed with real data rather than extrapolation.

## Run It

```sh
BIN=./build/DoFApproaches.app/Contents/MacOS/DoFApproaches

$BIN --method 2 --samples 8  --size 1280x860 --capture output/dof3-m2-multiview-8.png
$BIN --method 2 --samples 64 --size 1280x860 --capture output/dof3-m2-multiview-64.png
```

`--samples` is the number of aperture views, capped at 64 for this method. Interactively, `Up`/`Down` double and halve it, which makes the banding-versus-cost trade visible in real time.

## Findings

Mean absolute error against `--method 3 --samples 512`, 0..255:

| Views | Overall | Zone A (halo) | Zone B (near occluder) | Highlights |
|---|---|---|---|---|
| 8 | 2.455 | 2.096 | 4.966 | 2.516 |
| 16 | 1.487 | — | — | — |
| 32 | 0.923 | — | — | — |
| 64 | **0.682** | **0.558** | **1.364** | **0.626** |

Compare method 1 at 64 taps: 4.946 / 2.925 / 12.641 / 4.014. At equal sample budget the multi-view result is 7x better overall and 9x better at the near occluder, and unlike the gather it keeps improving: 8 → 64 views is a 3.6x reduction, heading toward the 0.518 floor measured below.

On the halo profile from Experiment 15, method 2 matches the reference to **0.1 of 255 at every distance from the sharp bar**, including 2 px from the silhouette where the gather is wrong by 35.

`output/dof3-diff-m2_multiview_64-vs-reference.png` shows what is left: faint vertical striping across the near posts, and concentric rings on the focus sphere. Both are the same artifact — 64 discrete aperture positions instead of a continuous disc, so a large defocus disc resolves into 64 overlapping ghost copies. Hiding it needs roughly as many views as the defocus diameter in pixels, which is why wide apertures are expensive.

## Cost

N full rasterizations of the scene per frame: N times the geometry and shading work, for an error that falls as roughly 1/N. `--samples 8` and `--samples 64` differ by 8x in cost and 3.6x in error.

## What This Method Still Is Not

It is an exact evaluation of primary visibility through a finite aperture, sampled at N points. It is not light transport: reflection, refraction and indirect lighting are outside what the shared `shadeSurface()` computes, and no number of aperture views adds them.

## Verification

| Check | Result |
|---|---|
| Build | Clean with `-Wall -Wextra -Wpedantic` |
| Focus plane invariance | Railing bar centres identical to the pinhole capture (1076, 1428, 1780, 2132, 2485), confirming the shear holds the focus plane fixed across all 64 views |
| Convergence | Monotone 2.455 → 1.487 → 0.923 → 0.682 over 8 → 16 → 32 → 64 views |
| Captures | `output/dof3-m2-multiview-8.png`, `output/dof3-m2-multiview-64.png`, `output/dof3-diff-m2_multiview_64-vs-reference.png` |

## Related Notes

[Experiment 16 notes](../../../notes/graphics/16_multiview_dof.md)
