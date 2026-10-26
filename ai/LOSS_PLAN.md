# Loss plan: weight the loss where depth of field happens

Written 2026-10-26 after the first smoke run. Status: being tested (A/B in section 5).

## 1. The smoke baseline this starts from

`ai/model.py`, `ai/smoke_train.py`, commit 590bf36.

- **Input** `x` `[B, 10, 512, 512]` (DATA_PROTOCOL.md): sharp RGB as `log(1 + L e)` (3), inverse depth
  `1/z` with sky 0 (1), six camera settings as constant planes (6). No CoC map, no masks.
- **Output** `y_hat` `[B, 3, 512, 512]`: DoF RGB in the same log space.
- **Architecture**: residual U-Net, 8.1 M parameters. Blocks are two 3x3 conv + GroupNorm + SiLU. Four
  levels down (2x2 stride-2 conv, 32 -> 64 -> 128 -> 256 -> 512 channels), four up (2x2 transposed conv,
  skip concat). The 1x1 head is zero-initialised and added to the sharp RGB, so step 0 is the identity.
  Camera settings enter only at the input.
- **Training**: L1 in log space, AdamW 2e-4 one-cycle, clip 1.0, bf16, batch 8, random 512 crops, flips.
- **Result** (2,000 steps, 172 held-out samples from unseen poses): L1 0.0166 vs identity 0.0215,
  PSNR 27.8 vs 25.4 dB.

## 2. What the data looks like (40 samples from `inspect_data/`, 2026-10-26)

Regions from the lens's `coc.npy` (|CoC| diameter in px at 1080p):

| Region | Definition | Pixels (median) | Identity L1 (median) |
|---|---|---|---|
| in focus | own CoC < 1 px, not boundary | 29% | 0.0087 |
| defocused | own CoC >= 1 px | 70% | 0.0231 |
| boundary | own CoC < 1 px, but inside the blur disc of a neighbour with CoC >= 4 px | 0.1% (max 13%) | 0.0257 (max 0.33) |

CoC p95 per sample: median 7.9 px (5.6 to 38.6); max: median 13.6 px (up to 88).

What follows from it:

1. In-focus pixels do **not** dominate plain L1: the blur floor makes ~70% of pixels defocused, and
   in-focus pixels carry ~13% of the identity error.
2. In-focus identity error (0.0087) is almost all **render noise**: sharp and DoF are independent renders.
   About 0.009 of the 0.0204 identity L1 is a floor no model can learn (see `denoising.md`). The smoke
   model therefore closed roughly 40% of the learnable gap, not 23%.
3. The hardest pixels are the rarest: boundary pixels, where a blurred foreground spills over sharp
   background (the halo / occlusion case of Experiments 15-17), are ~0.1% of a frame with the largest
   error. Uniform L1 almost ignores them.
4. Within defocused pixels the error grows with blur size.

## 3. Metrics (fixed, independent of the loss)

On the held-out poses, with deterministic crops so every run sees the same pixels:

- L1 per region: in focus, boundary, defocused; and per own-CoC bin: 1-4, 4-16, 16-64, >= 64 px.
- The identity L1 on the same pixels next to each, and the ratio model / identity.
- The noise floor: identity L1 on in-focus pixels.

Judge a loss by these, not by overall L1 (which favours the unweighted loss by construction).

## 4. Weighted loss

```
D(p) = dilated CoC: the largest |coc(q)| whose blur disc (radius |coc(q)| / 2) covers p
w(p) = 1 + alpha * min(D(p) / c0, cap) + beta * boundary(p)
L    = sum(w * |y_hat - y|) / sum(w)
```

- **Dilated, not raw, CoC**: raw CoC gives ~0 weight to a sharp background pixel under a blurred
  foreground, which is exactly where the hard error is. Dilating each disc by its own radius puts the
  weight where the blur lands. Computed on the GPU with max-pools (square windows, radii 1 to 48 px) on
  a CoC crop with a 48 px margin, so discs from just outside the crop count.
- **Floor of 1**: in-focus pixels keep full weight, otherwise the model may blur everything for free.
- **alpha, c0, cap**: extra weight for large blur. Start c0 = 8 px (median p95), cap = 4.
- **beta * boundary**: extra weight for the spill zone (own CoC < 1, D >= 4 px).
- **Normalised by sum(w)**: same loss scale whether a crop is mostly sharp or mostly blurred.
- The CoC map is a loss weight and a metric only, never a model input (DATA_PROTOCOL.md v1.1).

It does not remove the noise floor; weighting towards large blur helps a little because DoF targets
are smoother there.

## 5. A/B test

Same data, seed, steps (2,000), model and crops; only the loss changes.

| Run | alpha | beta | Question |
|---|---|---|---|
| A | 0 | 0 | baseline (uniform L1) |
| B | 1 | 0 | does weighting large blur help? |
| C | 1 | 5 | does the boundary boost help on top? |
| D | 3 | 10 | does pushing harder cost in-focus quality? |

A good weighted loss improves boundary and large-blur L1 clearly while in-focus L1 stays at the noise
floor.

## 6. Open questions

1. Boundary definition: "in-focus pixel inside a blurred neighbour's disc" (used now), or depth-edge bands
   where either side is blurred (also catches blurred-on-blurred edges)?
2. Perceptual or gradient losses for bokeh shape: after this A/B.
3. Effective receptive field: if the >= 64 px bin stays bad under every loss, the architecture (not the
   loss) is the limit.

## 7. Result of the A/B (2026-10-26, jobs 344947-344950, 2,000 steps each, one seed)

Ratio of model L1 to identity L1 on the same held-out pixels (fixed crops), step 2,000. Below 1 = better
than leaving the sharp image unchanged.

| Region | Pixels | A (0, 0) | B (1, 0) | C (1, 5) | D (3, 10) |
|---|---|---|---|---|---|
| all | 100% | **0.772** | 0.812 | 0.812 | 0.852 |
| in focus | 31% | **1.034** | 1.191 | 1.187 | 1.327 |
| boundary | 0.8% | **1.291** | 1.391 | 1.316 | 1.363 |
| defocused | 68% | **0.703** | 0.715 | 0.717 | 0.733 |
| CoC 1-4 px | 35% | **0.795** | 0.877 | 0.866 | 0.949 |
| CoC 4-16 px | 27% | 0.654 | 0.626 | 0.632 | **0.618** |
| CoC 16-64 px | 6% | 0.661 | 0.650 | 0.668 | **0.641** |

Findings:

1. Weighting only helps where it pushes (CoC >= 4 px, 3-5%) and costs more elsewhere, most on in-focus
   pixels. Plain L1 is best overall. Single short runs: differences of a few percent are near the noise.
2. Every run is worse than identity on boundary pixels (1.29-1.39) and at or above identity in focus: the
   model softens pixels that should stay sharp, the halo failure of screen-space DoF (Experiments 15-17).
   Run A's boundary ratio grew during training (1.13 at step 500 -> 1.29).
3. The boundary definition ignores occlusion. A blurred *foreground* disc does spread over sharp pixels
   behind it (the target changes there); a blurred *background* disc cannot spread over a sharp object in
   front of it (the target stays sharp). The max-filter dilation mixes both. Fix: dilate only discs nearer
   than the pixel, i.e. negative signed CoC (in front of the focus plane) for in-focus pixels.
4. Likely architectural cause: settings enter only as input planes, so the network must infer where the
   focus plane is from them and depth alone, and gets the in-focus band wrong. FiLM conditioning on the
   settings at every level is the standard fix.

Next: occlusion-aware regions and weights (split boundary into foreground spill and background edge),
FiLM conditioning, and a longer run; judge by the in-focus and both boundary ratios.

## 8. Second A/B: occlusion-aware weights x FiLM conditioning (2026-10-26)

Correction to section 7.3: a loss weight only changes how much a pixel's error counts; it cannot push blur
in a direction. What was wrong with the first weights: the boundary metric mixed two opposite cases, the
boost (5 on < 1% of pixels) was too small to matter, and the alpha term spread a blurred *background*'s
weight onto the sharp objects in front of it.

Regions (signed CoC, negative = in front of the focus plane; every pixel in exactly one of the first four):

| Region | Definition | Target there |
|---|---|---|
| in_focus | own CoC < 1 px, no spill, no blurred background disc | unchanged (noise only) |
| fg_spill | own CoC < 1 px, inside the disc of an in-front pixel with CoC <= -4 px | changes: foreground blur over it |
| bg_edge | own CoC < 1 px, inside the disc of a behind pixel with CoC >= 4 px, not fg_spill | stays sharp (occludes it) |
| defocused | own CoC >= 1 px | blurred |

Weight: `w = 1 + alpha * min(blur / c0, cap) + beta * fg_spill + gamma * bg_edge`, where
`blur = max(|own CoC|, dilated in-front CoC)`: the blur that actually reaches the pixel.

FiLM: the 6 settings go through a 2-layer MLP to a per-channel scale and shift after every U-Net block
(zero-initialised, so the model still starts as the identity); +0.38 M parameters.

| Run | Loss | Conditioning |
|---|---|---|
| E | plain L1 | input planes only (= run A, with the new metrics) |
| F | alpha 1, beta 5, gamma 5 | input planes only |
| G | plain L1 | FiLM |
| H | alpha 1, beta 5, gamma 5 | FiLM |

Judge by the in_focus, fg_spill and bg_edge ratios (all should reach < 1) and the large-CoC bins.

## 9. Result of the second A/B (2026-10-26, jobs 345109-345112, 2,000 steps, one seed)

Ratio to identity at step 2,000 (below 1 = better than leaving the sharp image unchanged):

| Region | Pixels | E plain | F weighted | G FiLM | H FiLM + weighted |
|---|---|---|---|---|---|
| all | 100% | 0.772 | 0.814 | **0.757** | 0.818 |
| in_focus | 31% | **1.034** | 1.198 | **1.034** | 1.205 |
| fg_spill | 0.56% | 0.911 | **0.873** | 0.886 | **0.873** |
| bg_edge | 0.26% | 5.49 | 6.46 | **4.52** | 6.63 |
| defocused | 68% | 0.703 | 0.716 | **0.686** | 0.719 |
| CoC 16-64 px | 6% | 0.661 | 0.655 | **0.637** | 0.655 |

1. FiLM helps (G best or tied on almost every region, bg_edge -18%). Keep it.
2. The weighted loss (alpha 1, beta 5, gamma 5) helps fg_spill slightly and hurts in_focus and bg_edge in
   both pairs. Dropped in this form.
3. The halo is the failure: on sharp edges in front of a blurred background every run is 4.5-6.6x worse
   than identity, and it grows with training (G: 3.1, 3.3, 4.7, 4.5 at steps 500-2,000) as the model learns
   to blur backgrounds and bleeds that over the objects in front. The earlier "boundary > 1" was all bg_edge.

Next: gamma alone (no alpha), a longer FiLM run to see the bg_edge trend, then occlusion-aware architecture
(depth-ordered kernel prediction, or a sharpness gate) if neither helps.
