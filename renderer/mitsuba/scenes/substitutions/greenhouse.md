# Greenhouse substitution plan

Sources: `output/dev/inventory/greenhouse.json` (built scene, seed 3, clear), `scenes/greenhouse.py` + `_greenhouse_props.py`, Poly Haven model and texture catalogs, ambientCG. Current total is 7.02M triangles against an 8M budget.

**Summary.** There are 33 object groups (the merged procedural plant mesh is split into its 10 plant kinds):

| Decision | Groups |
|---|---|
| SUBSTITUTE | 8 |
| RETEXTURE | 8 |
| KEEP | 12 |
| Scanned, keep as is | 5 |

Plus 16 additions. Two shared-side prerequisites:

- **(P1) Instancing in `web_assets.add`.** Add `instance=True`, giving one Mitsuba `shapegroup` per model and an `instance` per placement, so repeated trees and shrubs cost their geometry once. Without it the tree and hedge substitutions do not fit the budget.
- **(P2) Budget counting.** `check_scene` and the contract's 8M budget should count unique geometry, not instanced copies.

Note: the inventory reports the `web_*_variants` groups as "procedural" only because they are cut from Poly Haven variant sheets into the scene's own cache. They are scanned assets.

## Object groups

Abbreviations: PH is Poly Haven; bed palm, pot palm and hero palm are the procedural palms in the beds, in the four big pots and at the hero position.

| object group | instances | tris | decision | asset id(s) | size check | notes |
|---|---|---|---|---|---|---|
| procedural palms (`plants…leaf/stem/trunk`): hero + 8 bed + 4 pot | 13 | ~250k (est., part of the 1.27M leaf group) | SUBSTITUTE the 4 pot palms and 4 of the 8 bed palms; KEEP the hero and 4 bed palms | pot palms → `potted_plant_01` / `potted_plant_02` (variants alternated); bed palms → `pachira_aquatica_01` | potted_plant_01 is 1.35 m tall (pot palm 1.4–1.9 m, scale ≤1.3×); pachira sheet has 1.9 m plants (bed palm 2.3–3.0 m, scale 1.2–1.5×) | PH has no palm or tree fern, so a palm silhouette has to stay procedural. Retexture the kept trunks with the `bark_brown_02` or `fever_tree_bark` texture instead of the flat `trunk` atlas. |
| procedural ferns (hero + 8 in beds) | 9 | ~60k (est.) | SUBSTITUTE | `fern_02` (4 variants, already pinned) | fern_02 variants are 0.43 m tall and 0.4–0.6 m wide; procedural ferns 0.6–0.9 m, so scale 1.3–1.6× | The hero fern keeps its position, so the `fern_near` focus still lands on it. Twelve `fern_02` are already in the beds; mix the variants. |
| climbers on ribs (`climber_on_rib`, leaf atlas) | 18 runs | ~200k (est.) | KEEP | — | — | No ivy, vine or climbing plant in PH or ambientCG models. Improve the leaf atlas colour variation and add a few dead or brown leaves. |
| hanging baskets: basket frame + chain + trailing plants | 5 | basket 1.4k, chain 0.3k, plants ~40k | KEEP frame and chain; SUBSTITUTE the plant fill | fill: `periwinkle_plant` (trailing, flowering), `shrub_04` variants, `flower_gazania` | basket radius 0.2 m; periwinkle variants 0.3–0.4 m wide, so scale 0.7–0.9× | No hanging-basket model exists. Line the wire frame with the `wicker_basket_01` look, or keep the wire and add a moss liner using the `moss_01` texture. |
| lily pads + blooms (`pad`, `flower_*`) | 13 + 3 | ~4k | KEEP | — | — | No aquatic plants in the catalogs. Raise pad translucency and add slight curl variation. |
| near garden trees (`tree`, 4 × 2600 leaves) | 4 | ~470k (est.) | SUBSTITUTE (needs P1) | `searsia_lucida` (4.8 m), `island_tree_02` (8.5 m, already loaded), `tree_small_02` (4.65 m, already loaded), `othonna_cerarioides` (4.0 m) | procedural trees are 6–8.5 m tall; island_tree_02 at 1.0×, searsia at 1.3×, tree_small_02 at 1.3–1.5× | With instancing, island_tree_02 and tree_small_02 add no unique triangles. searsia_lucida adds 0.84M once. |
| horizon tree ring (`tree`, 22 × 2200 leaves, 45–75 m) | 22 | ~400k (est.) | SUBSTITUTE if P1 lands, else KEEP | instances of `island_tree_01` / `island_tree_02` / `searsia_burchellii` | 8–13 m tall procedural; island_tree_01 is 12.5 m (1.0×), searsia_burchellii 8.4 m (1.0–1.5×) | Always strongly defocused, so low priority. Without instancing this would add over 20M triangles, so keep procedural. |
| hedges (`hedge`: core box + ~51k atlas leaves) | 3 (60 m sides + 26 m end) | ~400k (est.) + 36 core | SUBSTITUTE | dense rows of `shrub_02` variants (already loaded) on a dark core; optional `grass_medium_02` skirt | shrub_02 variants ~1.3 m wide and 2.0 m tall; hedges 1.3–1.9 m tall, so scale 0.65–0.95× and pitch 0.9 m | About 95 shrubs; with P1 this costs 0 unique triangles (without it ~1.2M). This is the user's "box-like hedges". |
| `iron_frame` (ribs, glazing bars) | 1 | 27,400 | KEEP + RETEXTURE | `green_metal_rust` (PH, 1.0 m) | — | The Victorian structure has no catalog match. Flat principled green iron reads as CG; chipped, weathered green paint fixes it. |
| `end_walls.iron` | 1 | 2,128 | KEEP + RETEXTURE | `green_metal_rust` | — | Same as the frame. |
| `end_walls.brick`, `dwarf` ×2 | 3 | 50 | RETEXTURE | `brick_moss_001` (2.2 m) or `mossy_brick` (1.4 m), replacing `brick_wall_02` | — | Damp greenhouse brick has moss at the base. |
| `floor` | 1 | 12 | RETEXTURE | `terracotta_floor_tiles` (2.1 m) or keep `patterned_terracotta_tiling` | — | Optional. The bigger win is a dirt or wet decal overlay near the beds (procedural mask, as the workshop does with oil). |
| `kerb_*` (bed edges) | 4 | 192 | RETEXTURE | `mossy_sandstone` (1.9 m), replacing `old_sandstone_02` | — | |
| `bed_soil_*` | 4 | 48 | RETEXTURE | `brown_mud_leaves_01` (1.3 m) or `forest_leaves_02` (leaf litter), replacing `farm_soil` | — | Real beds have fallen leaves and mulch, not field soil. `wood_chip_path` is an alternative (bark mulch). |
| `pot_big` + `potsoil_big` (procedural terracotta, flat colour) | 4 + 4 | 3,264 | SUBSTITUTE | covered by the pot-palm swap above: `potted_plant_01/02` bring their own pots | — | If a procedural palm stays in a big pot, use `planter_pot_clay` scaled ~2× (0.26 to 0.5 m) instead of the flat principled pot. |
| `urn` | 1 | 1,568 | KEEP | — | — | A fountain urn has no catalog match (`antique_ceramic_vase_01` is a 0.44 m vase). It is already on `old_sandstone_02`; switch to `mossy_sandstone`. |
| `pond.rim` | 1 | 1,440 | RETEXTURE | `mossy_sandstone` | — | |
| `bench.iron` + `bench.wood` | 1 | 352 | SUBSTITUTE | `painted_wooden_bench` (1.16 m) | procedural bench ~1.2 m; scale 1.0× | Keep the `bench_l` focus point on its seat. |
| `roof_panes` (shading-painted panes, `principledthin`) | 1 | 90 | KEEP | — | — | Glazing approach; open cells are the clear panes. Add a procedural grime/algae opacity map along the bottom of panes and the glazing bars (no catalog texture for dirty glass). |
| `rectangle:terrace` | 1 | 2 | KEEP | `stone_pathway_02` | — | Fine. |
| `rectangle:lawn_scan` | 1 | 2 | KEEP + ADD | `leafy_grass` | — | Fine at distance. Near the glasshouse, add a strip of `grass_bermuda_01` or `grass_medium_02` (needs P1) so the lawn edge isn't a flat texture in doorway views. |
| `disk:water`, `disk:pond_bed` | 2 | 0 | KEEP | — | — | Deliberate (no-caustics water). |
| scanned already: `island_tree_02`, `tree_small_02` | 2 trees | 4.2M | KEEP | — | heights 6.5–7 m and 6.0 m | These are most of the triangle budget. Convert them to instances (P1) so they can be reused for the substitutions above at no cost. |
| scanned already: `potted_plant_01/02/04`, `planter_box_02`, `planter_pot_clay` | 12 | 532k | KEEP | — | — | |
| scanned already: `WoodenTable_03`, `seeding_tray_01`, `trowel_01`, `garden_gloves_01`, `watering_can_metal_01`, `compost_bags`, `garden_sprinkler_01` | 8 | 70k | KEEP | — | — | The potting bench reads well. |
| scanned variant plants: anthurium, calathea, pachira, periwinkle, shrub_02, shrub_04, fern_02 | 84 | ~770k | KEEP | — | shrub_04 is used at 0.2–0.3 m as ground cover (the sheet is 0.19 m tall, so ≤1.6×) | Good. Shrub_04 at 7–9 per bed is fine. |

## Big surfaces

| surface | now | proposed | real width | why |
|---|---|---|---|---|
| iron frame, glazing bars, end-wall iron | flat principled green | `green_metal_rust` (PH) | 1.0 m | chipped, weathered paint on Victorian ironwork |
| dwarf walls, end-wall brick | `brick_wall_02` | `brick_moss_001` (PH) | 2.2 m | damp base, moss |
| floor | `patterned_terracotta_tiling` | keep, plus a procedural wet/dirt mask | 2.2 m | wear at bed edges and under pots |
| bed soil | `farm_soil` | `brown_mud_leaves_01` (PH) | 1.3 m | leaf litter and mulch |
| kerbs, pond rim, urn | `old_sandstone_02` | `mossy_sandstone` (PH) | 1.9 m | moss in joints |
| palm trunks (kept) | `trunk` atlas | `bark_brown_02` or `fever_tree_bark` (PH) | 1.0–1.4 m | real bark detail |
| lawn | `leafy_grass` (tinted) | keep | — | — |
| panes | flat thin | procedural grime/algae mask (no catalog asset) | — | condensation streaks and algae at pane bottoms |

## Additions (all in the PH catalog unless noted)

**Potting bench and rotunda**
- `wooden_crate_01` with `planter_pot_clay` ×6 stacked under the table.
- `compost_bag_02` (open bag) beside the table.
- `wicker_basket_01` on the table.
- A second `watering_can_metal_01`.
- `wooden_stool_01` at the bench.
- `plastic_crate_01` with seed trays.

**Doorway and inside**
- `garden_hose_wall_mounted_01` on the far end wall by the door.
- `rubber_boots` inside the door.
- `rusted_spade_01` leaning on the dwarf wall.
- `wooden_bucket_01` near the pond.
- `moss_01` clumps on kerbs and the pond rim (~246k tris each: one or two only, or instanced).

**Garden outside**
- `tree_stump_01`.
- `rock_moss_set_01` (small selection) by the terrace.
- `picke_dirty_01` and `wooden_bucket_02` on the terrace.
- `planter_box_01` / `planter_box_03` with periwinkle.
- `grass_medium_02` strip along the glasshouse base (P1).

**Procedural, not in any catalog**
- Plant labels (thin white plastic tags) in seed trays and bed pots.
- A coiled hose on the floor (sweep).

No wheelbarrow, palm, ivy or tree fern exists in PH or ambientCG models; these stay procedural or are left out.

## Triangle estimate

Current total is 7.02M. The estimates below split the merged 1.27M leaf mesh by builder counts.

| change | without P1 | with P1 (unique geometry) |
|---|---|---|
| remove procedural ferns, 8 palms, 4 near trees, hedges | −1.1M | −1.1M |
| fern_02 ×9 | +0.02M | +0 (already loaded) |
| pot plants: potted_plant_01/02 ×4 | +0.35M | +0 (already loaded) |
| pachira ×4 | +0.12M | +0 |
| near trees: island_tree_02 / tree_small_02 ×3 + searsia_lucida ×1 | +4.0M | +0.84M (searsia only) |
| hedge shrub_02 ×95 | +1.2M | +0 |
| painted_wooden_bench, additions (~15 props) | +0.2M | +0.2M |
| moss_01 ×2 | +0.5M | +0.25M |
| grass strip `grass_medium_02` | +1.0M | +1.0M |
| horizon ring substitution | not possible | +0 (instances of loaded trees) |
| **total** | **~12.3M (over budget)** | **~7.0M** |

Recommendation: land P1 (instancing in `web_assets.add` plus unique-geometry budget counting) first. Then apply every SUBSTITUTE above. Without P1, do only the ferns, pot plants, pachira, bench, retextures and small additions (~7.6M), and keep the procedural trees and hedges.
