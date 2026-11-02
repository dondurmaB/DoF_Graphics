# Natural scene plan (2026-11-02)

Owner: beiba. Status: proposal. Inputs: the riverbed eval of the first baseline (U-Net w24, job 269282),
the depth and lens audit of the existing natural data, and the pinned-asset inventory
(`renderer/mitsuba/scenes/assets/web_manifest.json`, 434 assets). No new pins are assumed except where a row
says "needs pin".

## 1. What the eval says the next data should target

The baseline was scored full-frame on 80 riverbed test samples. L1 is in log1p space, and the Monte Carlo
noise floor of one target is 0.00096.

| Finding | Numbers (model / classic gather / sharp input) | What the data should add |
|---|---|---|
| Bright blurred highlights fail | highlight_blur16: 0.00905 / 0.00032 / 0.00040, so the model is 22.7x worse than the sharp input | Frames with bright sky and specular highlights behind the focus plane |
| Smooth areas far from depth edges lose to the classic gather | 65+ px from a depth edge (52% of the frame): 0.00370 / 0.00168 / 0.01155 | Large-sky, low-texture frames. Riverbed sky median is 0.30; the training scenes have 0.01-0.16 |
| In-focus silhouettes against far background bleed (bg_edge) | 0.19% of pixels: 0.01293 / 0.00531 / 0.00530 (2.4x the sharp input). The background is 3.8x deeper (p50), sky in 88% of cases, within 2 px of the depth edge | Near and mid subjects in focus against sky or a far background, at many depth ratios |
| In-focus pixels drift | in_focus: 0.00213 vs sharp 0.00189 (1.12x), 2.2x the noise floor | Mostly a loss question, not a data one; keep stopped-down lenses in the data (below) |
| The model wins where blur is large and near depth edges | CoC 32-64 px: 0.0114 vs 0.0183; on depth edges: 0.0131 vs 0.0285 | Keep this |

## 2. Common depth and lens spec

Audit: 10 poses per scene for depth, every lens for settings.

| Quantity | Spec for new scenes | Existing natural data | Out of spec |
|---|---|---|---|
| Focus distance | 0.3-60 m. At least 25% under 2 m, at least 50% under 10 m | Under 2 m: 1.3-3.9%. p50 14-40 m | All scenes; meadow (p50 39.7 m, p95 100 m) and riverbed (p95 135 m) most |
| f-number | f/1.4-16, with at least 15% at f/8-16 | f/1.4-8, none past f/8 | All scenes |
| Focal length | 24-135 mm | 24-135 mm | none |
| Depth p1 (per pose, median) | 0.3-3 m | 2.8-5.2 m | forest 5.15 m, meadow 4.36 m |
| Inverse-depth span p1-p99 (median) | at least 0.3 /m | 0.19-0.36 /m | forest 0.19, stream_v3 0.26, meadow 0.29 |
| Sky fraction (median) | 0.02-0.35, with some scenes at 0.2-0.35 | 0.01-0.30 | riverbed is alone at 0.30 (test); forest 0.01 |

How the spec is reached:
- Camera boxes low and close to subjects (mossy_log_macro: 0.12-1.0 m high, 0.3-3.5 m from the log).
- `SceneDef.f_number` up to 16.
- Rui's top-up "small" lenses (f/8-16) for the existing scenes, once ray-work is merged (it merges cleanly).

## 3. Roster

| Scene | Role | Buildable from pins? | Missing pins | Notes |
|---|---|---|---|---|
| forest_floor, meadow, stream | train | built | | Existing |
| dry_riverbed | test | built | | Built by beiba; the contract wants held-out scenes by the other author (open decision 1) |
| mossy_log_macro | train | yes; built 2026-11-02 | none | Log and twigs from the procedural kit. Moss shell, mushrooms and leaves procedural; bark, litter, fern, rock, stump and trees from pins |
| rocky_shore | train | partly | a sand or pebble texture, a shore boulder set | Water from `N.water_sheet`; rocks are `rock_moss_set_01` desaturated (as riverbed) |
| alpine_scree | train or val | yes, with recoloured rocks | a scree or talus texture (nice to have) | Rocks recoloured with `_dry_riverbed_props.bare_rock`; grass, periwinkle; no trees above the line; large sky |
| dune_field | train | yes, procedural | a sand texture (nice to have) | Terrain plus the procedural sand ground (`env_kit.ground("sand")`), `weed_plant_02` and `shrub_04` scrub. Low texture, large sky |
| snow_conifer | test_ood candidate | no | any conifer model | Procedural snow ground exists (`env_kit.ground("snow")`); no conifer in the manifest |
| marsh_reeds | train | no | a reed or cattail model | Water exists; there is no reed geometry |
| test_ood | | | | See below |

Asset inventory by category (pinned):

| Category | Pins |
|---|---|
| Broadleaf trees | island_tree_02, jacaranda_tree, tree_small_02, searsia_lucida |
| Conifers | none |
| Dead wood | tree_stump_01, plus the procedural `N.log`; bark textures bark_brown_02 and bark_willow_02 |
| Rocks | rock_moss_set_01; stone textures mossy_sandstone, old_sandstone_02 and others |
| Grass | grass_medium_02, weed_plant_02, nettle_plant; leafy_grass texture |
| Flowers | flower_gazania, periwinkle_plant, othonna_cerarioides |
| Ground cover | fern_02, shrub_02/03/04, moss_01; textures brown_mud_leaves_01, brown_mud_02, brown_mud_dry, farm_soil |
| Reeds | none |
| Water | none (procedural `N.water_sheet`) |
| Snow | none (procedural `env_kit.ground("snow")`) |
| Textures | 178 texture entries, mostly urban |

**test_ood suggestion: snow_conifer.** It differs from training in material (snow, needles), lighting (low
winter sun on white) and edge structure (dense needle silhouettes against sky) together, as the contract asks.
It needs one conifer pin, and the other author should build it.

If no new pin is allowed: dune_field (smooth depth, almost no texture, large sky).

A macro closeup is the contract's own test_ood example, but mossy_log_macro is now in `train` (open decision 2).

## 4. Splits

- **train:** forest_floor, meadow, stream, mossy_log_macro, plus rocky_shore, dune_field, marsh_reeds when built.
- **val:** alpine_scree, as the in-distribution natural val scene. Whole scene held out; this replaces the
  optimistic held-out-pose slice (7 samples) used today.
- **test:** dry_riverbed (unchanged).
- **test_ood:** snow_conifer (or dune_field, see above).

## 5. Pilot per new scene

12 poses x 6 lenses per environment-mixed scene: 72 samples.

Riverbed rates: 56-57 s and about 242 MB per pose, so about 12 min of GPU and 2.9 GB per scene before thinning.
Copy to the Mac, SHA-verify, then free the cluster copy, as for riverbed.

Per pilot, check:
- the focus and f-number spread against section 2 (dry run first; sampler `--dry-run` with `_jobs/sampler_record.py`)
- check_scene and the fingerprint guards
- the eval split tables (`ai/eval_split.py`), so each new scene is scored per blur size and per distance to a
  depth edge

## 6. Open decisions

1. Who builds the held-out scenes? The contract says the other author; riverbed (test) was built by beiba.
2. Is mossy_log_macro train, or the test_ood macro closeup? It is in `train` now, as asked.
3. May I pin one conifer and one reed model (shared manifest, both authors)?
4. Merge ray-work (clean; the sampler gains `--topup` and a lens-collision redraw) to get f/8-16 top-ups?
   The cluster guard's EXPECT_COMMIT then moves to the merge commit.
5. Storage. 512x512 fp16 crops (about 4.4 MB each with CoC, 3.7 MB without) hold about 6,800-8,100 crops in
   30 GB, against about 800 full-frame samples (37.5 MB each at 3 lenses per pose). This needs a crop-store
   reader next to Rui's loader.
