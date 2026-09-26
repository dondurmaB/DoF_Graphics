# Experiment 15 — Where Screen-Space Depth of Field Breaks

Review time: approximately 4–5 minutes.

## The One Sentence

A single-layer gather has one image, taken from a point, and it has to guess what a lens with area would have seen — so it fails wherever the guess matters, which is exactly at silhouettes.

## What the Method Actually Has

Experiment 14 built the pipeline this experiment stresses:

```text
Scene → FBO colour + depth → screen shader → linear depth → CoC → disk gather → displayed colour
```

Look at what the screen shader is holding when it runs. One colour texture. One depth texture. Both rendered from a **pinhole at the centre of the lens**. For each output pixel it reads *that pixel's own* depth, converts it to a circle-of-confusion radius, and averages neighbours inside that radius. In `postprocess.frag` the neighbour loop samples `uColor` and nothing else — neighbour depth is never consulted.

Everything below follows from those two facts.

## Error 1 — Spreading Light That Cannot Spread

Put a razor-sharp object in front of a heavily defocused one. The railing in this scene sits at exactly 3.80 m, the focus distance, so its circle of confusion is exactly zero. The wall behind it is at 37.6 m, near the far-field maximum.

Now take a wall pixel one texel from the railing's silhouette. *Its* depth is 37.6 m, so *its* radius is large, so its gather reaches across the silhouette and averages in the railing's near-black colour. The railing is in perfect focus. Physically its light converges to a single sensor point and cannot smear by even one pixel. The filter smears it anyway.

Measured on the bright wall, averaged over 170 rows:

| Distance from the bar | Method 1 | Reference | Error |
|---|---|---|---|
| 34 px | 180.7 | 187.4 | −6.7 |
| 18 px | 156.2 | 162.1 | −5.9 |
| 10 px | 144.3 | 162.1 | −17.8 |
| 2 px | 127.1 | 162.2 | **−35.1** |

The dark band is the width of the *wall's* blur radius, because the wall's radius is what decided how far to reach. Every darkened pixel is light the renderer invented.

Note which direction the error runs. The gather pulls the foreground's colour *outward* onto the background. It is not that the sharp object got blurred — it stayed sharp — it is that a halo of it appeared around it.

## Error 2 — Needing Data That Was Never Recorded

This one is more fundamental, and it is the one that cannot be patched.

A real lens is not a point, it is a disc. A thin foreground occluder that hides some background from the **centre** of the lens does not hide it from the **rim**. So in a real photograph an out-of-focus foreground goes semi-transparent: you see the background through it, because most of the aperture had an unobstructed view.

The pinhole image contains no record of what is behind that occluder. Zero. No filter can recover it, because filtering is a function of the pixels you have. The best a gather can do is smear the occluder's own colour around, which is why defocused foregrounds in screen-space DoF look like opaque smudges instead of haze.

The scene's Zone B tests exactly this: posts 27 px wide with a 63 px defocus diameter. The gather scores 12.641 mean absolute error there against the reference's structure; multi-view scores 1.364. It is by far the worst region in the frame.

## Why More Samples Do Not Help

This is the part worth internalising, because the instinct when an image looks wrong is to raise the sample count.

| Samples | Method 1 (taps) | Method 2 (views) |
|---|---|---|
| 8 | 5.626 | 2.455 |
| 16 | 5.238 | 1.487 |
| 32 | 5.022 | 0.923 |
| 64 | 4.946 | 0.682 |

Eight times the taps buys the gather 12%, and it is clearly flattening out near 4.9. Eight times the views buys multi-view a factor of 3.6, and it is still falling.

The gather is a **biased** estimator. It converges — to the wrong answer. More taps only measure that wrong answer more precisely. The bias is structural: it comes from having one image and one depth per pixel, not from having too few taps. No tap count, kernel shape or weighting scheme removes it, because none of them add the missing information.

That distinction — sampling error, which more work fixes, versus structural error, which it does not — is the reason Experiments 16 and 17 exist.

## What the Method Is Good For

It costs one extra full-screen pass. Multi-view costs N full scene renders. For a real-time renderer where the camera keeps moving and the artifacts sit in a few pixels along silhouettes, that trade is often correct, and it is why this family of methods is what shipped games actually use. The point of this experiment is not that the gather is bad. It is knowing precisely which pixels you should not believe.

## Checkpoint

- [ ] Explain why the halo appears on the *background* side of a sharp silhouette, not on the object.
- [ ] Explain why a defocused foreground should be see-through, using the aperture rather than the word "blur".
- [ ] Explain why raising the tap count from 16 to 64 changes the error by 6%.
- [ ] Predict what happens to the halo width if the f-number is halved, and check it.

## Related

[Experiment 15 code and findings](../../experiments/graphics/15_screen_space_halo/) · [Experiment 14 — Basic Depth of Field](14_basic_dof.md) · [Experiment 16 — Multi-View](16_multiview_dof.md)
