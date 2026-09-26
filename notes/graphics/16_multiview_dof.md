# Experiment 16 — Multi-View Depth of Field

Review time: approximately 4–5 minutes.

## The One Sentence

Stop filtering an image and re-render the world once per point on the lens: blur becomes the disagreement between views, and every view brings its own correct visibility.

## The Integral

A thin lens does not take one picture. It integrates over its aperture:

```text
I(p) = (1/A) ∫ L(p, u) du
              A
```

`p` is a sensor point, `u` a point on the aperture disc, `L(p, u)` the radiance arriving at `p` through `u`. Each fixed `u` is just a pinhole camera sitting at that spot on the lens. So the integral says something concrete: **a photograph is the average of every pinhole image you could take from anywhere on the lens.**

Experiment 15's method tries to reconstruct that average from a single `u` plus a filter. This experiment computes the average directly: pick N points on the disc, render the whole scene from each, average. `renderMethodTwo()` distributes them by the golden angle at radius `r·sqrt(i/N)`, which spreads them uniformly over *area* instead of crowding the centre.

## The Shear, and Why It Is the Whole Trick

Translating the eye sideways is not enough on its own. Do only that and the entire image swings between views; averaging would blur everything, focus plane included.

The fix is to pair each lateral eye offset with a projection shear that cancels it at one depth:

```cpp
shear[2][0] = -eyeOffset.x / focusDistance;
shear[2][1] = -eyeOffset.y / focusDistance;
viewProjection = projection * shear * view;
```

Camera space looks down −z, so a focus-plane point has z = −focusDistance. Writing the shear as x″ = x + c·z, cancelling the offset at that depth needs c = −offset/focusDistance — and the sign comes from z carrying it, which is the part that looks wrong until you write it out.

This is a real lens's behaviour, not a trick: a lens focused at distance F maps every point at F to one sensor point regardless of which part of the glass the light passed through. The shear is that property, expressed as a matrix.

## What Falls Out for Free

Everything Experiment 15 got wrong stops being possible, and it is worth being precise about *why* in each case.

**Sharp stays sharp structurally.** A focus-plane point projects to the *same pixel* in all N views. Averaging identical values changes nothing. There is no radius to choose and therefore no radius to choose wrongly. Experiment 15's Error 1 is not reduced here — it is unreachable. Measured, method 2 matches the reference to 0.1 of 255 at every distance from a sharp bar, including 2 px away where the gather is wrong by 35.

**Defocus is parallax.** An off-focus point lands on different pixels in different views. The spread of those landings *is* the circle of confusion: right size, right shape, aperture shape included. Method 2 contains no CoC formula at all. Open `postprocess.frag` next to it — that file is almost entirely the formula this method never needed.

**Hidden surfaces are actually rendered.** This is the one that matters most. Each pass runs a full, independent depth test. Background occluded from the lens centre *is visible* from off-centre aperture points, so it appears in those passes and contributes to the average. A thin foreground therefore comes out translucent, with the right amount of background showing through, because the right amount of the aperture really did see past it. Zone B error: 1.364, against the gather's 12.641.

**Occlusion ordering is correct per view.** Near and far defocus interact properly with no layer-compositing heuristic, because each view is a normal correctly-depth-tested render.

## What It Costs

N full rasterizations per frame. That is N times the geometry and shading, for an error that falls roughly as 1/N:

| Views | Mean absolute error |
|---|---|
| 8 | 2.455 |
| 16 | 1.487 |
| 32 | 0.923 |
| 64 | 0.682 |

The residual is visible in the difference map as faint vertical striping across the near posts and concentric rings on the focus sphere. Both are the same thing: 64 discrete aperture positions instead of a continuous disc, so a large defocus disc resolves into 64 overlapping ghost copies of the geometry. Hiding it takes roughly as many views as the defocus diameter in pixels — so the wider the aperture, the more views you need, exactly when each view is also blurrier. That is the method's real limitation.

Note this is *structured* error. Ghost copies look like geometry, so the eye finds them immediately. Experiment 17's noise at the same error magnitude is much less objectionable.

## What It Still Is Not

It is an exact evaluation of **primary visibility through a finite aperture**, sampled at N points. It is not light transport. Reflection, refraction and indirect lighting are outside what the shared `shadeSurface()` computes, and no number of aperture views will add them.

## Checkpoint

- [ ] Explain, without using the word "blur", why the focus plane survives the average.
- [ ] Explain why this method needs no circle-of-confusion formula, and what replaced it.
- [ ] Explain where the background seen through a defocused foreground comes from.
- [ ] Predict what 4 views look like versus 64 at the same aperture, then run both.

## Related

[Experiment 16 code and findings](../../experiments/graphics/16_multiview_dof/) · [Experiment 15 — Where It Breaks](15_screen_space_halo.md) · [Experiment 17 — Ray Traced](17_raytraced_dof.md)
