# Experiment plan: learned depth of field, next phase

Drafted 2026-10-28. Status: **draft for discussion**. Results so far are in `ai/LOSS_PLAN.md` (sections 1-12); the
current best setting is in `notes.md`.

## 0. Goal and ground rules

- **Goal:** a learned model that **reaches the classic screen-space method's quality in the same or less time**.
  Beating it is not required.
- **Pure network at inference.** No fixed gather or hand-written blur step inside the model; only standard layers
  (convolutions, up/downsampling). Option D (network corrects the classic output) is a reference ceiling only.
- **The blur map (CoC) never enters the model.** It may shape losses, sampling and metrics (DATA_PROTOCOL.md v1.1).
- **Realistic, not uselessly hard.** Lens settings, blur sizes and aperture shapes follow how real cameras are
  used. We cover the realistic range densely (no sparse gaps, both large and small apertures) instead of adding
  extreme cases nobody would shoot. Large blur belongs in the data because real lenses produce it: an 85 mm f/1.4
  portrait focused at 2 m blurs a distant background into a disc about 120 px across at 1080p. Extreme settings
  outside real use are not a goal.
- **Screening vs final.** 10k-step, one-seed runs choose directions. Claims need long runs, several seeds, error
  bars, full frames and held-out scenes (section 5).

Where we stand (screening, ratio to doing nothing, lower is better; time = 1080p frame on an A100, unoptimised):

| | all pixels | blur >= 64 px | ms / frame |
|---|---|---|---|
| classic gather | 0.603 | **0.312** | 58 |
| U-Net width 24 | **0.593** | 0.599 | 96 |
| U-Net width 16 | 0.690 | 0.823 | 63 |

Main gaps: **large blur (>= 64 px)** and **speed**. Professor Yan's comments (`prof_comment.md`): the scenes look
good; **span the parameter space** (large and small apertures); use **mild denoising** such as Intel OIDN.

## 1. Why large blur fails (working hypotheses)

1. The model outputs "sharp + correction"; heavy blur needs the correction to erase all detail exactly, and what it
   misses stays visible (previews: ivy and shutter slats too crisp).
2. Its effective view is short; a 64 px disc needs even weight 32+ px away.
3. Convolutions make soft, Gaussian-like blur; a lens makes a flat disc with a crisp rim (bokeh).
4. Large blur is rare in training: the sampler targets p95 blur of 5.4-43 px per sample
   (`COC_FRAC_RANGE` in `renderer/mitsuba/sampler.py`), and random 512 crops seldom land on the tail.

E0 below tests 1; tracks D and T address 4; track M addresses 1-3.

## 2. Track D: data

### D1. Distribution audit (first, cheap)
- Histograms over all 9,999 samples, per scene: f-number, focal length, aperture diameter, focus distance, p95 and
  max blur, share of pixels with blur >= 16 / 32 / 64 px.
- Question: are wide apertures, close focus and large blur under-represented (Professor Yan's point)?
- Cost: one short CPU-side job reading `metadata.json` and `coc.npy`. Output: a figure and a table.

### D2. Realistic wide-aperture top-up render
- If D1 confirms the gap: draw extra lenses for existing poses from **realistic shooting setups** that naturally
  give large blur, instead of raising the blur target arbitrarily: portrait primes (50 / 85 / 135 mm at f/1.2-2.8)
  focused on a subject, close-up focus with a distant background, wide-angle at f/1.4-2 for mild blur.
- Keep the combinations physically plausible (lens and f-number pairs that exist; focus on something in the scene)
  and check the resulting blur sizes against what such photos show. The current 5.4-43 px p95 window stays for the
  existing data; the top-up fills the realistic large-blur range it cuts off, densely rather than with a few
  extreme samples.
- Reuse each pose's sharp image, depth and G-buffer; render only the new lens images. To verify: the generator can
  add lenses to an existing pose without re-rendering it.
- Fix the lens-collision bug in `draw_lens` (`sampler.py:149`) first.
- Size to decide: for example one extra lens on each of the 5,000 poses (+50%).

### D3. Aperture shapes
- Today only the circular disc exists. Contract section 4 and DATA_PROTOCOL.md reserve the slot.
- Add shapes real lenses have: polygonal blades (5-9 blades, random rotation, optionally rounded blades, which
  most modern lenses use), and the disc stays the most common. Cat's eye (edge-of-frame vignetting) only later, and
  only if it matches real lens behaviour. Plug-in point: the aperture sampling in `lens_trace._rays`, which
  already generates our own thin-lens rays.
- Validate on a point-light test scene: the rendered bokeh must match the polygon and rotation.
- Protocol v2: the settings vector grows by `[one-hot shape, sin(rotation), cos(rotation)]`; loader and FiLM follow.
- Render a shaped-aperture subset (size to decide), held-out split included.

### D4. Mild denoising (Professor Yan: OIDN)
- Run OIDN on the sharp input with the stored albedo and normal buffers (`denoising.md`).
- Decide: denoise input only, or input and target. Denoising the target lowers the noise floor in the metric but
  changes what "ground truth" means; denoising only the input keeps targets physical.
- Check: in-focus identity error before and after; visual check that no detail is smeared.

## 3. Track T: training signal

### T1. Large-blur crop oversampling
- A fraction p of training crops (start p = 0.25) centred on the largest blur in the frame (as the hard
  evaluation crops, threshold drawn at random from 16-64 px); the rest random.
- Applies to every later run. Keep p moderate so training still reflects the realistic mix; evaluation always
  reports the plain held-out set (realistic distribution) next to the hard crops.

### T2. Bokeh metric and loss
- Metric: error on bright pixels (top 1% luminance in the target) with blur >= 16 px, reported next to the regions.
- Loss: an extra L1 term on those pixels in **linear HDR**, where bokeh shape is visible (log space flattens
  highlights). Weight to tune; watch the other regions.

## 4. Track M: model

### M1. Two-arm v2 (main candidate)
```
input ─┬─ Arm 1: U-Net, full resolution ─► image A (small blur, edges, spill) + mask m in [0, 1]
       └─ Arm 2: 1/4 resolution, dilated / large-kernel blocks ─► image B (heavy blur), upsampled
output = m * B + (1 - m) * A
```
- Arm 2 outputs the blurred image **directly**, not a correction (fixes hypothesis 1). At 1/4 resolution a 64 px
  disc is 16 px, within reach of a few dilated layers (2), and larger kernels can form a flat disc (3). It costs
  about 1/16 of a full-resolution arm.
- The mask m comes from arm 1 at full resolution, so sharp-object / blurred-background boundaries stay pixel-exact.
- Losses: `L = L1_gamma(output) + l_B * L1(B, downscaled target | blur >= t) + l_A * L1(A, target | blur < t)`
  `[+ l_m * mask loss toward sigmoid((blur - t) / s)]`, all label-side.
- Why the old two-arm net failed (0.571 / 0.712): its wide arm only fed features into a "sharp + correction"
  output, so it still had to erase detail.
- Variants: arm 2 separate vs sharing arm 1's first downsampling layers; 1/4 vs 1/8 resolution; mask loss on/off;
  arm 1 width 16 vs 24.

### M2. Reference ceiling (option D, not a candidate)
- Classic gather output as the starting point; the network learns only the correction. One run, to show the
  quality reachable when large blur is solved, as a target for M1.

### M3. Speed
- Inference: `torch.compile`, fp16 / bf16 kernels, TensorRT, for the U-Net widths and M1.
- A fair classic time: an optimised implementation (the DoFScene shader or a fused CUDA kernel), not the PyTorch
  port.
- Decide the target hardware (A100 only, or also a consumer GPU).

## 5. Track E: evaluation hygiene (before any claim)
- Bootstrap error bars over held-out samples for every metric.
- Full-frame evaluation (not crops); PSNR, SSIM, LPIPS next to the L1 ratios.
- Held-out **scenes** (Beibarys's evaluation scenes) once available.
- A proper training script: resume, run config, checkpoints by best held-out score.

## 6. Experiments, in order

| # | Experiment | Question | Success | Cost |
|---|---|---|---|---|
| E0 | Under-blur check on run K and w24 (>= 64 px pixels: leftover detail vs target) | Is hypothesis 1 real? | clear yes / no | minutes, existing checkpoints |
| E1 | D1 distribution audit | Is large blur / wide aperture under-represented? | table + figure | < 1 h |
| E2 | T1 oversampling on U-Net w24 | Does more practice fix large blur? | >= 64 px ratio drops clearly; others hold | 1 run |
| E3 | T2 bokeh metric (+ loss) on U-Net w24 | Can we measure and improve bokeh shape? | metric works; loss helps it without hurting others | 1-2 runs |
| E4 | M1 two-arm v2 at about w16-24 cost | Does a dedicated large-blur arm close the gap? | >= 64 px at or below classic; all <= 0.603; time <= w24 | 2-4 runs |
| E5 | M2 reference ceiling | How good can it get when large blur is free? | upper bound for E4 | 1 run |
| E6 | D2 realistic wide-aperture top-up (if E1 says so), retrain best of E2-E4 | Does realistic large-blur data help further? | >= 64 px improves | render + 1 run |
| E7 | D4 OIDN on sharp input | Lower noise floor without losing detail? | in-focus error drops | preprocessing + 1 run |
| E8 | D3 aperture shapes (render, protocol v2, retrain) | Can the model follow aperture shape? | shape-specific bokeh error close to disc's | renderer work + render + runs |
| E9 | M3 speed: compiled models vs optimised classic | Same or less time? | candidate <= classic time | benchmarking |
| E10 | Long runs (100k+ steps, 3 seeds) of the chosen model, track E evaluation | Final numbers | error bars separate us from classic | days of GPU |

E0, E1 and E2 are independent and can run in parallel. E4 should start with T1 and T2 included. D2, D3 and D4 change
the data, so they come before the long runs (E10).

## 7. Open questions
1. Arm 2: separate arm or shared early layers?
2. Large-blur top-up size and blur range (needs E1).
3. Which aperture shapes, and what share of the data?
4. OIDN on input only, or input and target?
5. Speed target hardware.
6. Do Beibarys's natural scenes follow the same lens sampling, or do they need the same top-up?

## 8. Results log

### E1. Lens and blur audit (2026-10-29, job 352533; `runs/audit/` on CIAI)
- f-number: **only f/1.4-8 occurs**. None at f/1.2-1.4 and **none at f/8-16** (small apertures, deep focus).
  68% of samples are below f/2.8 (median f/2.05).
- Focal length 24-135 mm, weighted to wide (median 51 mm).
- Focus: median 7.2 m; **under 2 m in only 7.8%** of samples (1.4% under 1 m). Close-focus portraits are rare.
  91% of focus distances come from a random depth quantile, 9% from named focus points.
- Blur: p95 per sample is 4-43 px by design (`COC_FRAC_RANGE`); the largest blur in a frame reaches 64 px in
  **1.0%** of samples; **0.02% of all pixels** have blur >= 64 px, 1.6% >= 32 px. Far background blur is under
  16 px in 83% of samples.
- Reading: Professor Yan's point holds at both ends. Small apertures (nearly all-sharp photos) are missing
  entirely, and realistic large blur (wide aperture + close focus) is rare. Scene by scene the picture is
  similar; rooftop and desk have the most blur, terminal and pool the least.

### E0. Under-blur check (2026-10-29, job 352532; `runs/underblur.json`)
Pixels with blur >= 64 px in the hard crops (90k px from 18 crops, so indicative only):

| | detail vs target | coarse error | fine error | bias (log units) |
|---|---|---|---|---|
| sharp input | 2.27 | 0.353 | 0.132 | -0.164 |
| classic | 0.68 | **0.122** | **0.044** | +0.018 |
| run K (w32) | 1.06 | 0.212 | 0.056 | -0.056 |
| w24 | 1.15 | 0.236 | 0.068 | -0.049 |
| w64 | 1.15 | 0.259 | 0.069 | -0.061 |

- Hypothesis 1 (leftover fine detail) is **only a small part** at >= 64 px: the models keep 6-15% more detail
  than the target, and their fine-detail error is close to the classic method's.
- The main error is **coarse**: the models' output is too **dark** over large blur (bias -0.05 to -0.06,
  about 5% too dim), and their coarse error is about twice the classic method's. In the lens image, bright
  lights spread their light over large discs, raising the brightness of everything around them. The classic
  method averages light linearly and gets this right (bias +0.02); the models, working in log space, get only
  about two thirds of it, and wider models do not do better.
- At 16-64 px the leftover-detail problem is real: models keep 30-70% too much detail (the classic method too:
  45-67%).
- Consequences for the plan: T2 (a loss in linear HDR, where spreading light is measured correctly) moves up;
  M1's large-blur arm should **blur in linear light** (convert to linear inside the model, blur, convert back,
  all fixed elementwise steps, so still a pure network); D2 should also add realistic small-aperture samples.

### E2. Large-blur crop oversampling (2026-10-29, job 352534)
U-Net w24, FiLM, gamma 20, `--big-blur-prob 0.25`, otherwise as w24 (350202):

| | all | hard all | hard CoC >= 32 | hard CoC >= 64 |
|---|---|---|---|---|
| classic | 0.603 | 0.479 | 0.365 | 0.312 |
| w24 | 0.593 | 0.495 | 0.428 | 0.599 |
| w24 + big-blur crops | 0.597 | 0.485 | 0.411 | 0.585 |

Small gains on large blur (2-4%), nothing lost elsewhere; within one-seed noise. More practice alone does not fix
large blur, as E0 suggested (the error is light spreading, not rarity). Kept on for later runs (cheap).

### E3 / E4. Light-energy loss and two-arm v2 (2026-10-29, jobs 352592-352594; diagnostic 352766)
All with FiLM, gamma 20, big-blur crops 0.25. New metrics: `highlight_blur16` (brightest 1% of target pixels per
crop with blur >= 16 px, where bokeh shows) and `energy_*` (light box-averaged over 8 px in linear radiance:
too dark or too bright over an area). Older runs and the classic method re-scored (`ai/reeval.py`, job 352767).

| | ms | all | fg_spill | bg_edge | CoC 16-64 | highlight | energy | hard all | hard >= 64 | hard energy >= 32 |
|---|---|---|---|---|---|---|---|---|---|---|
| classic | 58 | 0.603 | 1.000 | 1.000 | 0.406 | **0.464** | 0.647 | 0.479 | **0.312** | **0.332** |
| run K (w32, 1 crop/frame) | 130 | **0.549** | **0.534** | 1.032 | 0.366 | 0.664 | 0.635 | **0.444** | 0.526 | 0.392 |
| w24 | 96 | 0.593 | 0.610 | 1.077 | 0.418 | 0.639 | 0.669 | 0.495 | 0.599 | 0.452 |
| w24 + big-blur crops (E2) | 96 | 0.597 | 0.615 | 1.049 | 0.411 | 0.669 | 0.668 | 0.485 | 0.585 | 0.429 |
| E3: + energy loss (weight 1) | 96 | 0.595 | 0.593 | 1.070 | 0.412 | 0.597 | 0.635 | 0.480 | 0.542 | 0.413 |
| E4a: two-arm v2, arm 1 w16 | 82 | 0.635 | 0.650 | 1.745 | 0.380 | 0.702 | 0.684 | 0.504 | 0.674 | 0.507 |
| E4b: two-arm v2, arm 1 w24 | 116 | 0.584 | 0.555 | 1.102 | **0.369** | 0.769 | 0.638 | 0.483 | 0.665 | 0.489 |

Diagnostic of E4b (ratio to identity per blur bin; m = mask weight on the large-blur arm B):

| bin | output | arm A alone | arm B alone | m | B closer |
|---|---|---|---|---|---|
| in focus | 0.874 | 0.877 | 4.28 | 0.01 | 21% |
| 4-16 px | 0.436 | 0.458 | 1.49 | 0.02 | 32% |
| 16-32 px | 0.315 | 0.459 | 0.345 | 0.54 | 66% |
| 32-64 px | 0.473 | 0.695 | 0.521 | 0.92 | 76% |
| >= 64 px (hard crops) | 0.665 | 0.797 | 0.665 | 1.00 | 71% |
| in focus (hard crops) | 0.865 | 0.850 | 3.12 | 0.16 | 15% |

1. The energy loss is the first change that moves the largest blur (0.585 -> 0.542) and bokeh highlights
   (0.669 -> 0.597) without cost elsewhere. Keep it.
2. Two-arm v2: the mask routes correctly and arm B cuts the error by about a third at 16-64 px, which gives the
   best width-24 overall score (0.584). At >= 64 px arm B itself is the limit (0.665 with all the weight): the
   training data has almost no such pixels (0.02%, E1), and the evaluation has only 90k of them (a few frames).
3. Halo: in the hard crops the mask leaks 16% of arm B onto in-focus pixels (bg_edge 1.10; 1.75 with a w16
   arm 1). Arm B at 1/4 resolution also smears small bright bokeh discs (highlight 0.77).
4. The classic method is far ahead on bokeh highlights (0.464) and light energy in large blur (0.332): it spreads
   light linearly over an exact disc. No model is within reach there yet.
5. Two-arm v2 costs 82 / 116 ms (w16 / w24), against 58 ms for the classic method (unoptimised).

Next: the realistic data top-up (D2) is now the main lever for >= 64 px, for training and for a usable test set;
model-side, a sharper mask at object edges and a finer arm B for highlights; keep the energy loss.

### D2. Realistic top-up `topup1` (started 2026-10-29)
- Code: `sampler.py --topup` (redraws each existing pose exactly, renders one extra lens), `generate.py plan-topup`,
  `merge_topup.py` (moves lenses into the dataset, recomputes CoC from the stored depth); loader `--no-topup`.
  Also fixes the lens-collision bug in the main draw (cafe p000281).
- Lens policy: 65% wide open (f/1.2-2.8 by what primes reach at the pose's focal length; focus on a named subject
  or the near part of the scene; stop down if p95 blur would pass 10% of the frame), 35% stopped down (f/8-16,
  focus a third of the way in). Overrides the scenes' f/1.4-8 range within real-lens limits.
- Check (old cluster, 3 cafe poses): pose ids, depth (byte-identical) and original lens ids reproduce exactly;
  about 20 s per lens on an RTX 5000 Ada.
- Dry run (300 poses, all scenes): wide lenses put 6.6% of pixels at >= 32 px and 1.6% at >= 64 px (originals:
  1.6% and 0.02%); 7% of wide samples reach 64 px. Stopped-down lenses: p95 blur median 2.2 px. Close focus stays
  limited by the views (10% under 2 m).
- Size: 5,000 lenses (one per pose), about 28 GPU-hours; rendering on the old cluster, shipped to CIAI
  `/nfs-stor/rui.gao/Researches/dof/dof_dataset/topup1`, merged into train10k when done.
- Evaluation: held-out (validation) poses get their top-up lens too; scored separately as `top_*`, so the original
  validation set and all earlier numbers stay unchanged. Beibarys's held-out test scenes are not rendered or touched.

### E4c-f. Mask fix and finer blur arms (2026-10-30, jobs 352979, 352986) — confounded, rerun as E5
Mask fix = `--aux-m 0.5 --aux-m-s 2 --aux-m-halo 1`; all with energy loss, big-blur crops, FiLM, gamma 20.

| | all | in_focus | fg_spill | bg_edge | CoC 16-64 | highlight | hard >= 64 |
|---|---|---|---|---|---|---|---|
| E4b two-arm w24 (aux-m 0.1) | **0.584** | **0.895** | **0.555** | **1.102** | **0.369** | 0.769 | 0.665 |
| E4c two-arm w24 + mask fix | 0.679 | 0.973 | 0.710 | 1.908 | 0.428 | 0.890 | 0.676 |
| E4d two-arm, arm B at 1/2 + mask fix | 0.660 | 0.949 | 0.698 | 1.399 | 0.450 | 0.868 | 0.671 |
| E4e pyramid w24 + mask fix | 0.645 | 0.973 | 0.579 | 1.101 | 0.371 | 0.772 | **0.661** |
| E4f pyramid w16 + mask fix | 0.750 | 0.994 | 0.629 | 1.152 | 0.388 | **0.744** | 0.672 |

Everything got worse, halos included. Cause: loss scale. The mask terms (BCE to a soft target, ~0.1-0.7) dwarf the
image L1 (~0.01): at the end of training the extra terms were about 90% of the pyramid's loss (0.188 total vs
0.016 L1) and about 60% for the two-arm runs, so the models fit the mask and lost image quality (in-focus 0.95-0.99
vs 0.90). The comparison says nothing yet about the designs. Within the same bad weighting, arm B at 1/2
resolution beat 1/4 on halo (1.40 vs 1.91) and highlights (0.87 vs 0.89), a weak hint for finer resolution.

Rerun as E5c-f (jobs 354643, 354644) with mask weights scaled to the image loss: `--aux-m 0.02 --aux-m-halo 0.05`
(aux-b / aux-a unchanged at 0.5, they are image L1 terms). Lesson: weight every extra term by its magnitude
relative to the main L1, and log each term separately.

### E5c-f. Point-2 fixes with scaled weights (2026-10-31, jobs 354643, 354644)
`--aux-m 0.02 --aux-m-halo 0.05` (logged terms at the end: main 0.008-0.014, energy 0.003-0.005, aux 0.004-0.018).

| | ms | all | in_focus | fg_spill | bg_edge | CoC 16-64 | highlight | energy | hard all | hard >= 32 | hard >= 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| classic | 59 | 0.603 | 1.000 | 1.000 | 1.000 | 0.406 | **0.464** | 0.647 | 0.479 | **0.365** | **0.312** |
| E4b two-arm w24 | 116 | 0.584 | 0.895 | 0.555 | 1.102 | 0.369 | 0.769 | 0.638 | 0.483 | 0.425 | 0.665 |
| E5c two-arm w24 + mask fix | 116 | 0.576 | 0.884 | 0.572 | 1.106 | 0.358 | 0.659 | 0.619 | 0.477 | 0.425 | 0.676 |
| E5d two-arm, arm B at 1/2 + mask fix | 153 | 0.591 | 0.902 | 0.609 | 1.229 | 0.383 | 0.680 | 0.640 | 0.496 | 0.436 | 0.662 |
| **E5e pyramid w24 + mask fix** | 151 | **0.551** | **0.879** | **0.546** | 1.043 | **0.312** | 0.578 | **0.612** | **0.434** | 0.375 | 0.621 |
| E5f pyramid w16 + mask fix | 118 | 0.575 | 0.909 | 0.567 | 1.249 | 0.327 | 0.645 | 0.637 | 0.445 | 0.386 | 0.599 |

1. The blur pyramid wins. At w24 it is the best learned model on almost every region: overall 0.551 (as good as
   run K's w32 at 0.549), the lowest halo of the learned models (1.043), medium blur 0.312 (best of any model), and
   bokeh highlights 0.578 (best learned). It beats the classic method overall, on light energy (0.612 vs 0.647) and
   on the strong-blur crops (0.434 vs 0.479), ties it at >= 32 px, and is still behind on highlights (0.578 vs
   0.464) and >= 64 px (0.621 vs 0.312, the data-limited bin).
2. The mask fix helps a little (E5c vs E4b: overall 0.584 -> 0.576, highlights 0.769 -> 0.659) but does not remove
   the halo leak (1.106). The pyramid's per-pixel choice of level does better on halos (1.043).
3. A single blur arm at 1/2 resolution (E5d) is worse than at 1/4 and slower: one level cannot serve all blur
   sizes. Several levels can.
4. Speed: pyramid w24 151 ms, w16 118 ms (2.6x / 2.0x the classic method, unoptimised). Next for speed: compile,
   fp16, fewer or narrower levels.

Next: retrain the pyramid on the top-up data (>= 64 px), then speed work on it.

### M3. Compiled speed (2026-10-31, job 356731, `ai/speed_compile.py`)
One 1920 x 1088 frame, A100, batch 1, median of 30; trained checkpoints. The classic gather (PyTorch port, fp32) is
compiled too. `max` = `torch.compile(mode="max-autotune")` (autotuned kernels + CUDA graphs).

| ms per frame | eager (before) | compile fp16 | max-autotune fp16 | max-autotune, fp16 weights | quality (all) |
|---|---|---|---|---|---|
| classic gather (fp32) | 57.7 | 19.5 | **19.5** | - | 0.603 |
| U-Net w16 | 63.1 | 7.1 | 6.3 | **5.9** | 0.690 |
| U-Net w24 (E3) | 96.3 | 10.4 | 10.2 | **9.6** | 0.595 |
| U-Net w32 | 130.1 | 15.0 | 15.0 | **14.3** | 0.563 |
| two-arm w16 (E4a) | 82.3 | **10.1** | 10.9 | fails | 0.635 |
| two-arm w24 (E5c) | 115.5 | 13.4 | **12.9** | fails | 0.576 |
| pyramid w16 (E5f) | 117.8 | 17.8 | **14.5** | fails | 0.575 |
| pyramid w24 (E5e) | 151.0 | 19.4 | **18.4** | fails | 0.551 |

- Compiling speeds the models up 6-10x (eager PyTorch was dominated by per-layer overhead and unfused GroupNorm
  / activation passes) and the classic gather 3x.
- **Every learned model now runs at or below the compiled classic method's time.** The best-quality model, the
  w24 pyramid (0.551), takes 18.4 ms against 19.5 ms. The w24 U-Net (0.595) takes 9.6 ms, half the classic time.
- Outputs are unchanged: compiled fp16 differs from fp32 by 0.2-0.4% of what the model changes (eager bf16, used in
  all training-time evaluations, differs by 2-5%). The fp16-weights mode fails for two-arm / pyramid because their
  light-space steps are deliberately fp32; mixed fp16 works for them.
- Caveats: A100, batch 1, PyTorch for both. A hand-written CUDA kernel or the DoFScene GLSL shader for the classic
  method could be faster than its compiled PyTorch port; that comparison (and a consumer GPU) is still to do.
