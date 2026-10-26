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
