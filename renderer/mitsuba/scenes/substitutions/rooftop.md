# Rooftop substitution plan

**Summary**
- **Procedural groups:** 89, merged into 46 table rows (the 70 city-wall groups are one row): 7 SUBSTITUTE, 23 RETEXTURE (two of them keep the geometry rebuilt) and 16 KEEP. The KEEP rows are olive foliage, glassware, emitters, cables and railings, distant city, street and haze tiers.
- **Scanned groups:** the 13 already in use are kept, and 2 are adjusted.
- **Additions:** 13.

**Triangle budget.** Today the scene has 2.37M triangles, of which 1.06M are procedural, against a budget of 8M. After this plan it is about 3.1M.

**Inputs.**
- `output/dev/inventory/rooftop.json`
- `scenes/rooftop.py` and `_rooftop_props.py`
- `output/dev/audit/rooftop_clear.png` (24 random poses)
- Poly Haven models (521) and textures (866), and ambientCG materials (2013).

**Dimensions.** Catalog `dimensions` are Poly Haven's mm figures in arbitrary axis order. I checked sizes against the largest and smallest of the three, and `W.place(height=...)` should set the final scale.

## What the audit shows

These are the things that read as fake. Ranked by how many of the 24 tiles they hurt:

1. **The city is extruded boxes** (tiles 1, 4, 5, 6, 7, 8, 9, 12, 13, 17, 23).
   - **Flat-roofed slabs and window grids:** every block has a flat roof with a regular window grid baked into one texture.
   - **Same massing:** the same rectangle silhouette repeats.
   - **Bare roofs:** no parapets, setbacks or roof plant beyond the scanned AC units.
   - **Glass towers:** they read as dark grid boxes with no reflection.
2. **The deck reads reddish-brown and flat in shade** (tiles 0, 3, 11, 14, 19, 21). The `wood_floor_deck` tint goes red under golden-hour light. It is the single most-seen surface: in 14 of 24 tiles it fills over 30% of the frame.
3. **Procedural parasols** (tiles 6, 7, 9, 20). A flat octagon canopy on a pole with no ribs, valance or fabric texture.
4. **The `our_walls` / parapet top band** (tiles 12, 15, 22). The concrete upstand and the brick-tier wall below the deck show as smooth slabs.
5. **The hut** (tiles 10, 21). A plain box with a flat door and no hardware.

## Object table

Decision key:
- **SUBSTITUTE** means swap in a scanned model.
- **RETEXTURE** means keep the geometry and use the named texture (Poly Haven id unless written `aCG:`).
- **KEEP** gives the reason.

| object group | instances | tris | decision | asset id(s) | size check | notes |
|---|---|---|---|---|---|---|
| plants.leaf | 1 | 191k | KEEP (+RETEXTURE leaf colour) | — | — | Procedural olive leaves. No olive tree scan exists (closest are `searsia_lucida` at 0.84M triangles each and `tree_small_02` at 4.65M, both too heavy x3, and neither looks like an olive). Keep the procedural olives; give the leaves a two-tone silver-green (`thin` diff_trans already). |
| plants.bark | 1 | 48k | RETEXTURE | `bark_willow_02` (or `chinese_hackberry_bark`) | — | Olive bark is grey and fissured; procedural flat bark reads plastic up close (tiles 2, 9, 16). |
| plants.stem | 1 | 44k | KEEP | — | — | Lavender and grass stems; thin, fine at any distance. |
| plants.lavender_flower | 1 | 95k | KEEP | — | — | No lavender scan. The procedural spikes look right in tile 2. |
| plants.plume | 1 | 17k | SUBSTITUTE (partial) | `grass_bermuda_01` cut into tufts, or `shrub_04` | `shrub_04` is 0.75 x 0.28 x 0.19 m, a trough-sized clump | Replace about half the procedural grass plumes in the troughs with `shrub_04` / `shrub_03` (0.4 m) for real leaf shapes. `grass_medium_01/02` are 1.0 to 1.6M triangles, too heavy. |
| olive_planter | 3 | 144 | SUBSTITUTE | `planter_box_03` | 0.91 x 0.86 x 0.66 m vs the current corten cube of about 0.9 m | Real planter with edges and wear. If the corten look must stay: RETEXTURE `rusty_metal_02` (Poly Haven) or `aCG:Metal018`. |
| olive_soil | 3 | 6 | RETEXTURE | `brown_mud_dry` + a few `seeding_tray_01`-free pebbles via `bark_brown_02` mulch | — | Top dressing reads as flat colour today. |
| trough | 2 | 96 | RETEXTURE | `rusty_metal_02` (corten) | — | Long troughs have no scanned equivalent of the right length (`planter_box_02` is 1.25 m, troughs are about 10 m). Keep the geometry, use a real corten texture, and add a 3 cm lip. |
| trough_soil | 2 | 4 | RETEXTURE | `bark_brown_02` (bark mulch) | — | — |
| wine_glass | 6 | 31k | KEEP | — | — | Clear dielectric. No scanned wine glass in Poly Haven (`brass_goblets` is brass), and the procedural lathe is physically right. |
| candle.holder / wax / wick / flame | 4 each | 20k | SUBSTITUTE holder; KEEP wax, wick, flame | `wooden_candlestick` no; best is `brass_diya_lantern` (0.38 m) for the bar, keep the glass votive on tables | — | The votive is fine. Swap the bar-top candles for `brass_diya_lantern` or `Lantern_01` (0.29 m) for variety. |
| bottles.bottle | 4 | 79k | SUBSTITUTE | `wine_bottles_01` (already used on the back bar), `multi_cleaner_bottle` no | `wine_bottles_01` is a 0.68 m row of 0.33 m bottles | Replace the 4 procedural table and bar bottles with single bottles cut from the `wine_bottles_01` parts (one part per bottle), so glass and labels are real. |
| bar_counter | 1 | 756 | RETEXTURE | `teak_veneer` front, `marble_01` top (see bar_top) | — | No scanned bar counter. Add a kick plate in `metal_plate_02`. |
| bar_top | 1 | 12 | RETEXTURE | `marble_01` (or `aCG:Marble012`) | — | — |
| back_bar | 1 | 60 | RETEXTURE | `dark_wooden_planks` | — | Shelving; fill per Additions. |
| foot_rail | 1 | 40 | RETEXTURE | `metal_plate` (brushed brass via tint) | — | — |
| screen | 1 | 480 | RETEXTURE | `teak_veneer` | — | HVAC slat screen; real wood grain helps close views (tile 19). |
| coffee_table | 1 | 156 | SUBSTITUTE | `modern_coffee_table_01` | 1.2 x 0.6 x 0.39 m: matches a lounge coffee table | Teak block today. |
| armchair.teak + armchair.cushion | 2 + 2 | 2.5k | SUBSTITUTE | `mid_century_lounge_chair` (1.17 x 1.01 x 1.19 m) or `modern_arm_chair_01` (1.02 x 0.99 x 0.82 m); cushion via `throw_pillows_01` | Both are lounge-chair size | The previous pass rejected `ArmChair_01` as Victorian. `modern_arm_chair_01` is a plain modern upholstered chair; retexture its fabric to `rough_linen` for outdoor use. |
| rug | 1 | 2 | RETEXTURE | `aCG:Carpet012` (or `rough_linen`) | — | Flat rug quad; add 1 cm thickness. |
| umbrella.canvas / pole / base | 2 each | 1.3k | KEEP (geometry rebuild) + RETEXTURE | canvas `hessian_380` / `rough_linen` (tinted off-white); pole `metal_plate`; base `brushed_concrete` | — | No parasol scan exists in Poly Haven (searched umbrella, parasol, canopy). Rebuild procedurally with 8 ribs, a scalloped valance, a sagging fabric between ribs and a crank collar, then put real fabric on it. This is the only fix for tiles 6, 7, 9 and 20. |
| pergola | 1 | 276 | RETEXTURE | `weathered_brown_planks` | — | Beams are fine as geometry. |
| deck | 1 | 2 | RETEXTURE | `brown_planks_09` or `hinoki_planks` (grey-brown), tint neutral (1.0, 0.98, 0.95) | — | Fixes the reddish deck in shade. Also split into 3 board-direction zones (uv rotate 90 degrees under the lounge) so it isn't one tiled plane. |
| pavers | 1 | 2 | RETEXTURE | `concrete_tiles` is OK; better `checkered_pavement_tiles` / `aCG:PavingStones` | — | Keep if it reads well; minor. |
| upstand | 1 | 48 | RETEXTURE | `concrete_block_wall_02` + top coping in `brushed_concrete` | — | Tiles 12, 15 and 22: a 30 cm upstand with a separate coping strip, a drip edge and an algae streak (via `chipped_concrete`). |
| our_walls | 1 | 8 | RETEXTURE | `brick_wall_02` (+ window openings via facade map) | — | Seen straight down over the railing (tiles 12, 22). |
| near_ground | 1 | 2 | RETEXTURE | `asphalt_02` | — | — |
| hut | 1 | 10 | RETEXTURE | `concrete_block_wall` | — | — |
| hut_door | 1 | 12 | SUBSTITUTE | `rusted_shutter`-textured slab + handle: no door model; use `rollershutter_door` (3.08 x 2.4 m, too wide), so RETEXTURE door slab with `painted_metal_shutter` instead | — | Add `industrial_wall_lamp` (0.43 m) above the door and `fire_alarm` beside it. |
| hut_roof | 1 | 12 | RETEXTURE | `bitumen` | — | — |
| hut_tank / hut_tank_steel | 1 / 1 | 300 | KEEP + RETEXTURE | `dark_planks` stays for the timber tank; steel `rusty_metal_03` | — | NYC-style timber tank: no scan. The existing procedural tank with real planks is appropriate. |
| mast, pole_fl/fm/fr/ml/br | 6 | 192 | KEEP | — | — | Festoon poles; thin tubes, real-looking at all sizes. |
| festoon.cable | 1 | 11k | KEEP | — | — | — |
| festoon.bulbs_on [emitter] | 1 | 28k | KEEP | — | — | Emitter geometry; a scanned `lightbulb_01` would be 4,372 triangles x 170, about 0.74M, and adds nothing visible as a bokeh disc. |
| railing_0..3.post / cable / handrail | 12 | 1.4k | KEEP | — | — | Thin edges are what DoF needs. Handrail already teak. |
| city.walls_*_t (14 kinds x 5 haze tiers) [emitter] | 70 | 33k | RETEXTURE (+ geometry, see City) | see the City section | — | — |
| city.roof_t | 5 | 8.7k | RETEXTURE | `bitumen` for flat roofs, `box_profile_metal_sheet` for sheds, `clay_roof_tiles_02` for pitched pre-war roofs | — | — |
| city.roofstuff | 1 | 18k | SUBSTITUTE (near tier only) | `exterior_aircon_unit` (already), `modular_airduct_rectangular_01`, `modular_pipes`, `small_lpg_tank` | — | Only tiers 0 and 1 (under 500 m). Beyond that keep the procedural boxes, since at 1 to 3 px a scan is invisible. |
| city.tank | 1 | 15k | KEEP | — | — | Procedural rooftop water tanks across the city; at their distance they are a few pixels. |
| city.steel | 1 | 8.5k | KEEP | — | — | Tank stands and antenna; thin. |
| city.spire_t | 2 | 380 | KEEP | — | — | Landmark spire. |
| street.car / car_glass | 6 / 1 | 54k | KEEP | — | — | Seen from 42 m up at 2 to 10 px. The only Poly Haven vehicle is `covered_car`, which reads wrong in traffic. |
| street.canopy / trunk | 1 / 1 | 187k | KEEP | — | — | Street trees from above; procedural canopy blobs are right at that scale. |
| street.tyre | 1 | 69k | KEEP | — | — | Part of cars. 69k triangles is wasteful, but it's not a realism issue. |
| river.park_tree / park / water / bridge [emitter] | 4 | 60k | KEEP | — | — | Haze-tier geometry, distant. |
| hills / far_ground [emitter] | 2 | 35k | KEEP | — | — | Haze tiers, 3.4 to 10 km. |

**Already-scanned models:**

| model | instances | review |
|---|---|---|
| `exterior_aircon_unit` | 168 (city) + 2 (hut) | Good. It is 0.8M of the triangles. On city roofs past 300 m use a decimated box instead (they are under 4 px). |
| `potted_plant_02` | 6 | Good. |
| `wine_bottles_01` | 36 | Good on the back bar. Also reuse it for table bottles (above). |
| `bar_chair_round_01` | 5 | Good: 0.75 m bar stool. |
| `food_lime_01`, `wooden_bowl_01` | 5, 1 | Good. |
| `outdoor_table_chair_set_01` | 8 | Good. It is the hero of the table view. |
| `wooden_lantern_01` | 4 | Good. |
| `carved_wooden_plate`, `croissant` | 4 each | Good. |
| `potted_plant_04` | 1 | Small succulent, OK. |
| `sofa_02` | 1 | ADJUST: recover in `rough_linen` / `hessian_230` via overrides, since the indoor fabric reads wrong outdoors. |
| `modular_airduct_circular_01` | 2 | ADJUST: scale check, it is 4.17 m long; keep only if it fits the HVAC pen. |

## Big surfaces: textures

| surface | current | proposed | why |
|---|---|---|---|
| Deck (lounge and table area) | `wood_floor_deck` tinted 0.92 | `brown_planks_09` or `hinoki_planks`, neutral tint, 2 board directions | Kills the red cast; grey-brown weathered hardwood is what rooftop decks look like. |
| Pavers (bar end) | `concrete_tiles` | keep, or `checkered_pavement_tiles` | — |
| Parapet upstand + coping | `brushed_concrete_04` | `concrete_block_wall_02` (upstand) + `brushed_concrete` (coping) | Visible seam and coping read as built. |
| Planters / troughs | procedural corten | `rusty_metal_02` / `aCG:Metal018` | Real weathered steel. |
| Hut walls / roof / door | `beige_wall_001` / concrete / flat | `concrete_block_wall` / `bitumen` / `painted_metal_shutter` | — |
| Bar front / top / back bar | teak / marble / teak | `teak_veneer` / `marble_01` / `dark_wooden_planks` | — |
| Parasol canvas | flat | `hessian_380` tinted off-white | — |
| City flat roofs | `bitumen` tint | `bitumen` (near), `box_profile_metal_sheet` (industrial), `clay_roof_tiles_02` (pre-war blocks) | Roof variety is the most visible city cue from 42 m up. |

## City: making it read less like extruded boxes

The city is seen from above at 42 m, so the roofline and massing do more than facade detail does.

1. **Massing, which is the biggest win.** In `city_layout` / `R.walls`, give each building one of these profiles, chosen by seed:
   - flat box (40%);
   - podium + setback tower (25%): 3 to 5 storey base, tower inset 3 to 6 m;
   - stepped (15%): 2 to 3 setbacks;
   - pre-war block with a pitched or mansard roof (15%);
   - slender glass tower (5%).
   
   Also:
   - Add a 0.6 to 1.2 m parapet with coping on every flat roof.
   - Add a lift overrun or stair box (2.5 x 3 x 3 m) and a water-tank stand on half the roofs.
   - Add corner chamfers on 10% of towers.
2. **Facade by building type.** These use real textures; ambientCG has 26 real photographed facades with windows built in:
   - **Glass curtain-wall towers:** `aCG:Facade001`–`Facade006` (reflective glass) and `aCG:Facade018A/B/C`–`020A/B/C` (skyscraper). Replace the procedural `glass` and `dark` layouts; they show real mullions, spandrels and sky reflections.
   - **Office blocks:** `aCG:Facade012`–`Facade017` (window-wall offices; also good for lit windows at golden hour via emission from their luminance), plus `concrete_tile_facade` and `rectangular_facade_tiles_02`.
   - **Brick residential:** keep `exterior_wall_cladding_02`, `patterned_brick_wall_03` and `red_brick_plaster_patch_02` behind the procedural window layer, and add `brick_wall_02` and `aCG:Bricks047` for variety.
   - **Render / stucco:** `plastered_wall_02`, `blue_plaster_weathered`, `aCG:PaintedPlaster010`–`018`.
   - **Concrete brutalist:** `concrete_slab_wall_02`, `concrete_panels`.
3. **Ground-floor band.** Every near building gets a 4 to 5 m retail base: a darker storefront band, awnings, and lit shop windows at golden hour. This is the cue that separates "building" from "box" when looking down.
4. **Lit windows at golden hour.** Keep the emission map, but light 5 to 15% of windows, varied by tower type (offices lit cooler, residential warm). Light none on tiers 3 and 4.
5. **Roof clutter on tiers 0 and 1.** Swap the procedural `roofstuff` for scanned `exterior_aircon_unit`, `modular_airduct_rectangular_01`, `modular_pipes` and `small_lpg_tank`. Beyond 500 m keep the boxes.
6. **Keep the fake-haze tiers** (`blendbsdf` + emission per distance tier); all of the above slots into the per-tier material.

## Additions a real rooftop bar has (all exist in the catalogs)

1. `standing_chalkboard_01` (1.51 m): an A-frame menu board by the bar.
2. `metal_trash_can` (0.91 m) and `trashbag`: by the hut.
3. `industrial_wall_lamp` (0.43 m) on the hut and `security_light` on the HVAC screen.
4. `CoffeeCart_01` (1.72 m) or `wine_barrel_01` (0.87 m): a barrel used as a standing table near the bar.
5. `side_table_tall_01` (0.76 m) beside the lounge chairs.
6. `throw_pillows_01` on the sofa.
7. `tea_set_01` or `jug_01` + `metal_jug` on the lounge coffee table.
8. `wicker_basket_01/02` with `lemon` / `food_lime_01` on the bar.
9. `planter_box_01/02` (0.91 and 1.25 m) with `shrub_03` / `shrub_04` along the hut wall.
10. `garden_hose_wall_mounted_01` (0.56 m) on the hut wall: a real rooftop detail.
11. `propane_tank` (0.55 m) next to a patio heater. There is no patio heater scan; skip the heater and keep the tank by the bar.
12. `modular_pipes` / `modular_industrial_pipes_01` running along the parapet base to the HVAC pen.
13. `fire_alarm` + `wall_clock` on the hut wall facing the bar.

**Out of scope, noticed in passing:** the `exterior_aircon_unit` scan accounts for 0.8M of the 2.4M triangles. Distant instances could be swapped for a box LOD without any visible change.
