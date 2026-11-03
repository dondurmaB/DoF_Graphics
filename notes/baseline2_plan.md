# Next baseline run: plan (2026-11-03, not launched)

Owner: beiba. Builds on baseline 269282 (U-Net w24, 10k steps, batch 8, crop 512; 335 training samples from forest,
meadow and stream; 7 held-out poses from the same scenes as the only validation). Scores below are L1 in log1p space
on the riverbed test (80 samples), noise floor 0.00096.

## 1. Data

| Role | Data | Poses x lenses | Samples | Staged size |
|---|---|---|---|---|
| train | rocky_shore pilot (job 275606) | 12 x 3 | 36 | 1.14 GB |
| train | mossy_log_macro pilot (job 275652) | 12 x 3 | 36 | 1.46 GB |
| train | existing natural (forest_v3, meadow_v1, stream_v4), the poses that fit | 2 per pose | see section 2 | |
| val | dune_field pilot (job 275638), a whole held-out scene, replacing the held-out-pose slice | 12 x 2 | 24 | 0.98 GB |
| test | dry_riverbed (unchanged), staged only for evaluation after training | 40 x 2 | 80 | 3.29 GB |

A staged pose is sharp.exr, depth.npy, pose.json and coc_valid.npy; the G-buffer (~28 MB) is not read by the loader
and is not staged. Measured per pose / per lens: rocky 25.5 / 23.1 MB, dune 28.9 / 26.4 MB, existing natural scenes
29-31 / ~30 MB, mossy 32.3 / 29.7 MB. Train lenses per pose: lowest, middle and highest blur (as staged
for 269282); val and test: lowest and highest blur.

Pilot lens spread (dry runs): rocky has focus under 2 m in 7% of lenses and f/8-16 in 8%; mossy 21% / 29%; dune 19%
/ 25%. So the pilots add the close-focus and stopped-down lenses the existing data lacks (1-4% and 0%).

## 2. What fits under the 48 GB line

The cluster home is 41.61 GB apparent with no data staged (this project 11.7 GB, `.conda` 6.4 GB, `.local` 10.7 GB,
FinalProjectAI2210 10.3 GB). That leaves 6.39 GB; 0.6 GB stays free for two runs' checkpoints and logs (~0.2 GB per
run), so **5.8 GB for data**.

| Tier | Room for data | Train samples | What is in it |
|---|---|---|---|
| A, now | 5.8 GB | ~120 | dune val 0.98 + rocky 1.14 + mossy 1.46 GB + existing ~2.1 GB (23 poses x 2 lenses: forest 8, meadow 7, stream_v4 8) |
| B, after freeing 3.6 GB of my own scratch (your OK needed) | 9.4 GB | ~200 | tier A + 40 more existing poses x 2 lenses (forest 22, meadow 10, stream_v4 8) |
| C, 269282's set plus the pilots | 17.7 GB | ~410 | needs ~8.3 GB more than tier B: a quota increase, or a crop store (512 px fp16 crops, ~4.4 MB each, ~7x more pairs per GB; needs a crop-store reader next to Rui's loader) |

Tier B frees: scratch_scenes smokes 0.50, scratch_riverbed (noise-floor renders, numbers recorded) 0.48,
scratch_mossy/smoke_269633 0.26, train_smoke/data 0.80, code_new/_out (check sheets) 0.87, alpine_scree's generated
cache (parked scene, regenerable) 0.70.

Tier A trains on a third of 269282's samples, so a change in score mixes the new data, the smaller set and the val
swap. **I recommend tier B** (about 60% of 269282's set, more poses at 2 lenses each instead of 3), and comparing
against a control run on the same data, not against 269282 directly.

The riverbed test (3.29 GB) is not on the cluster during training: after training, delete the staged train copies
(masters on the Mac, SHA-verified) and stage the test set for evaluation.

## 3. Protecting in-focus pixels

Where 269282 lost: in focus 1.12x the sharp input, background edges (in-focus pixels in front of a blurred background)
2.4x, blurred highlights 22.7x. The model is already residual on the sharp image (`ai/model.py`: y = sharp + o,
zero-initialised), so the skip exists; the network just isn't kept from touching pixels it should leave alone.

- **Gate the residual (recommended).** y = sharp + g * o, with g = smoothstep(0.5, 1.5, blur px), where blur is the
  dilated CoC map smoke_train already computes (max of the pixel's own |CoC| and the blur of any nearer
  foreground that spreads over it). A pixel that is in focus and has no blurred foreground over it returns the sharp
  input exactly. That covers both in_focus and bg_edge (in focus, background blurred behind it). Foreground spill
  keeps g = 1, so it is still learned. At inference the CoC map comes from depth and the lens settings, which the
  model already gets. It is a small patch to Rui's `ai/model.py` and `ai/smoke_train.py`; this is a shared file, so
  Rui decides whether it goes in.
- **Loss weight (no code change).** `--gamma 5` (extra weight on bg_edge) with `--beta 5`. Rui's LOSS_PLAN.md found
  that weighting pushes the targeted regions but costs 1.19-1.33x in-focus error elsewhere. Kept as the fallback if
  the gate is not accepted.

## 4. Runs

Two runs, same data (tier B), same settings as 269282 (w24, 10k steps, batch 8, crop 512, `--save-every 1000`,
`--stop-at`): **A** control, **B** gated residual (or `--gamma 5 --beta 5` if the gate isn't accepted).
Each is ~85 min on one ws-ia GPU. Scored on dune val during training. After training, both are scored on the riverbed
test with `ai/eval_split.py` (per region, per blur bin, per distance to a depth edge), against the noise floor.

Decisions: tier B cleanup; gate patch (Rui) or loss weights; whether ray-work is merged first (not needed for these
runs).
