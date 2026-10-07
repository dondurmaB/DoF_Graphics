# Courtyard: substitution plan

**Summary:** 108 procedural object groups plus 1 scanned group (`tree_small_02`, kept). Counts are by inventory group:
- 9 groups are SUBSTITUTE.
- 67 groups are RETEXTURE.
- 32 groups are KEEP, because they are structure fitted to openings, have no CC0 match, or the user approved them.
- About 30 new scanned additions are listed under "Additions".

Rows below gather families with the same decision. The instance and triangle counts are summed over the family.

## Sources

- Models: Poly Haven ids, checked in `output/dev/catalog/polyhaven_models.json`.
- Textures: Poly Haven ids (`W.surface`) or ambientCG ids (marked "aCG", once the loader supports them).
- `pinned` means the id is already in `scenes/assets/web_manifest.json`.
- Triangle budget: the scene is at about 2.9M now, 2.06M of it the scanned tree. The additions bring it to about 4M of 8M.

## Object groups

| object group | instances | tris | decision | asset id(s) | size check | notes |
|---|---|---|---|---|---|---|
| climbers ivy_leaf, creeper_leaf, stem | 3 | 527k | KEEP | – | – | User liked the ivy. Poly Haven has no ivy or creeper model. |
| plants citrus_leaf, stem, bark (5 potted citrus) | 3 | 61k | KEEP | – | – | No citrus or lemon-tree scan exists. They are a signature element. |
| plants lemon (fruit) | 1 | 20k | KEEP | (`lemon` as an option) | `lemon` is 0.07×0.10 m | Scan fruit only on the 2 citrus nearest the default views if ever needed. |
| plants flower_red/pink/salmon/white, geranium_leaf | 5 | 31k | KEEP | – | – | Window-box geraniums. Poly Haven's flowers are wild South African species, not geraniums. |
| plants grass (bed edges) | 1 | 4.5k | SUBSTITUTE | `shrub_04` (pinned) + `weed_plant_02` | 0.75×0.19×0.28 m, 2.1×0.3×0.08 m | Low ground cover along beds; weeds along wall bases. |
| plants fallen_leaf | 1 | 2.7k | KEEP | – | – | Procedural scatter, fine at ground level. |
| roof_n, roof_e, roof_s, roof_w | 4 | 99k | RETEXTURE | `clay_roof_tiles_02` (else `clay_roof_tiles`, pinned) | texture 2.5 m / 4.0 m | Keep the barrel-tile geometry. Swap the procedural tile texture for scanned albedo, roughness and normal. Check the UV pitch matches the tile pitch (currently 2.1×2.16 m), otherwise use `uv_scale`. Snow env stays snow. |
| chimney | 4 | 0.4k | RETEXTURE | `red_brick_plaster_patch_02` (pinned) | 1.5 m | Weathered rendered brick. |
| eave_n/e/s/w gutter | 4 | 0.2k | RETEXTURE | `rusty_painted_metal` (pinned) | 2.2 m | Visible in every upward audit pose. |
| eave_n/e/s/w fascia | 4 | 48 | RETEXTURE | `weathered_brown_planks` | 1.8 m | |
| downpipe | 2 | 0.1k | RETEXTURE | `rusty_painted_metal` (pinned) | – | Add 2 more, at the NE and SW corners (see Additions). |
| wall_n.face (ashlar) | 1 | 0.7k | RETEXTURE | `white_sandstone_blocks_02` or `large_sandstone_blocks_01` | 2.0 m / 3.0 m | The facade bakes openings and weathering into a per-wall texture. Composite the scanned albedo, normal and roughness at real scale with the existing procedural stain masks, so window drips and base grime stay. |
| wall_e.face (ochre stucco spalling to brick) | 1 | 0.1k | RETEXTURE | `yellow_plaster_02` + `red_brick_plaster_patch_02` (pinned) for spalled patches | 2.0 m / 1.5 m | Blend by the existing spall mask. |
| wall_w.face (cloister / arcade) | 1 | 0.8k | RETEXTURE | `old_sandstone_02` (pinned) | 1.5 m | |
| wall_s.face (brick wing) | 1 | 0.1k | RETEXTURE | `medieval_red_brick` (or `brick_wall_02`, pinned) | 2.0 m | A close brick wall was a blank tile in the audit (tile 22). Real brick normals fix that. |
| wall_n/e/s/w.reveal, .cap | 8 | 0.9k | RETEXTURE | stone of the matching face | – | Reveals are seen in every oblique pose. |
| wall_n/e/s/w.back | 4 | 1.7k | KEEP | – | – | Outside faces, never visible from the camera box. |
| trim_n/e/s/w.stone (sills, surrounds) | 4 | 3.5k | RETEXTURE | `old_sandstone_02` (pinned) | 1.5 m | |
| trim_n/e/s/w.frame (window frames) | 4 | 3.9k | RETEXTURE | aCG `PaintedWood004` (cream, tinted) | – | Fitted to each opening, so keep the geometry. |
| trim_n/e.iron (window grilles) | 2 | 0.8k | RETEXTURE | aCG `Rust004` / `rusty_painted_metal` (pinned) | – | |
| trim_n/e/s/w.glass | 4 | 0.1k | KEEP | – | – | Dark reflective glass, tuned for sun glints. |
| trim_n.door (main door, north) | 1 | 0.1k | SUBSTITUTE | `large_castle_door` | 2.01×0.34×2.96 m | Fits an arched doorway of about 2×3 m. Scale to the opening; if it is narrower than 1.8 m, RETEXTURE with `rough_pine_door` instead. |
| trim_e.door | 1 | 32 | RETEXTURE | `rough_pine_door` | 2.0 m | |
| shutters n/e/s/w | 4 | 26k | RETEXTURE | `distressed_painted_planks` tinted per `PAINTS` (`W.surface(..., tint=)`) | 1.6 m | Louvres are sized per window, so keep the geometry. No louvred-shutter scan exists. |
| walk_doors.door (3 cloister doors) | 1 | 0.3k | RETEXTURE | `rough_pine_door` | 2.0 m | |
| walk_doors.iron | 1 | 1.2k | RETEXTURE | `rusty_painted_metal` (pinned) | – | |
| walk_back.face (cloister back wall) | 1 | 0.3k | RETEXTURE | `worn_plaster_wall` | 1.8 m | Was flat plaster. |
| walk_back.reveal | 1 | 0.2k | RETEXTURE | `old_sandstone_02` (pinned) | – | |
| walk.plaster (cloister soffit) | 1 | 6 | RETEXTURE | `white_rough_plaster` | 1.0 m | |
| walk.beams | 1 | 0.2k | RETEXTURE | `weathered_brown_planks` | 1.8 m | |
| walk.floor | 1 | 2 | RETEXTURE | `terracotta_floor_tiles` (or `patterned_terracotta_tiling`, pinned) | 2.1 m | |
| passage.vault | 1 | 64 | RETEXTURE | `worn_cracked_plaster` | 1.8 m | Dark arch interiors were bland in the audit (tiles 2, 3, 4, 21). |
| passage.wall | 1 | 4 | RETEXTURE | `old_stone_wall` (pinned) | 2.0 m | |
| passage.floor | 1 | 2 | RETEXTURE | `cobblestone_floor_08` (or `cobblestone_02`, pinned) | 2.0 m | Wet env keeps the clearcoat override. |
| street_wall face/reveal/back | 3 | 0.2k | RETEXTURE | `old_stone_wall` (pinned) | 2.0 m | Seen through the arch and from the passage. |
| street_ground | 1 | 2 | RETEXTURE | `cobblestone_pavement` (road) + `concrete_pavement` (pinned) pavement strip | 2.5 m / 1.8 m | Replaces the flat `env_kit.ground`. |
| street_facade (flat 40×14 m painted quad) | 1 | 2 | SUBSTITUTE | `modular_urban_apartments_facade` | 51.5×6.7×17 m, 118k tris | Real 3D facade with depth. Place at z≈24 facing north. It is what the archway view shows. |
| gate (open carriage-gate leaf) | 1 | 16 | SUBSTITUTE | `large_castle_door` | scale ×0.81 to the 2.4 m arch | One leaf swung flat against the passage's west wall, as now. |
| paving_s (court flagstones) | 1 | 6.7k | RETEXTURE | `monastery_stone_floor` (or `stone_tiles_02`) as the albedo and normal base | 1.8 m | Keep the procedural stone layout, moss joints, worn paths and wet puddle and roughness masks; use the scan for stone surface detail. |
| ground (joint plane under paving) | 1 | 2 | KEEP | – | – | Mortar under the stones. |
| kerb | 1 | 1.7k | RETEXTURE | `old_sandstone_02` (pinned) | – | |
| bed_soil, pot_*_soil, trough_soil | 4 | 0.6k | RETEXTURE | `farm_soil` (pinned) | – | Snow env stays snow. |
| fountain.stone | 1 | 5.1k | RETEXTURE | `mossy_sandstone` | 1.9 m | No fountain scan exists on Poly Haven, so keep the lathe geometry. |
| fountain.water_low, water_high | 2 | 10k | KEEP | – | – | Rough dielectric, tuned against caustic fireflies. |
| fountain.lining | 1 | 1.2k | KEEP | – | – | Dark lining for the noise trick. |
| pot_small (6) | 1 | 4.6k | SUBSTITUTE | `planter_pot_clay` (pinned) | 0.27×0.26×0.22 m vs procedural r 0.14, h 0.20 m | Near-exact size match. |
| pot_big (5 citrus pots) | 1 | 3.8k | RETEXTURE | `terracotta` → `patterned_terracotta_tiling`'s plain clay albedo, or aCG `Plaster006` tinted terracotta | – | `ceramic_pot` (0.66×0.37 m) is too squat for a citrus tree, so keep the lathe pot. |
| trough (window boxes) | 2 | 0.1k | RETEXTURE | same clay as the pots | – | `planter_box_01/02` are 0.41–0.47 m tall, too tall for ledges. |
| bench.iron, bench.wood | 2 | 0.4k | SUBSTITUTE | `painted_wooden_bench` (pinned) | 1.16×0.50×0.89 m | Same footprint as the current bench. Recolour via `overrides` if needed. |
| bicycle rubber, chrome, paint, leather, dark | 5 | 7.6k | KEEP | – | – | No bicycle scan on Poly Haven. The procedural bike reads well. |
| lantern.iron + lantern.glass | 2 | 1.6k | SUBSTITUTE | `Lantern_01` (pinned) | 0.12×0.10×0.29 m | Hang the scanned lantern from the existing iron bracket arm (keep the arm). Scale ×1.3 to read as a wall lantern. |
| balcony.stone | 1 | 36 | RETEXTURE | `old_sandstone_02` (pinned) | – | |
| balcony.iron | 1 | 0.3k | RETEXTURE | `rusty_painted_metal` (pinned) | – | |
| laundry rope | 1 | 0.3k | KEEP | – | – | |
| laundry cloth | 2 | 0.2k | RETEXTURE | aCG `Fabric004`, `Fabric026` and `Fabric045`, tinted per `CLOTH` | – | Woven texture instead of flat colour. |
| line_bracket | 2 | 24 | KEEP | – | – | |
| web:tree_small_02 | 3 | 2.06M | KEEP | (already scanned) | 7.0 m | |

## Big-surface textures (summary)

| surface | texture | real width | wet env | snow env |
|---|---|---|---|---|
| North ashlar facade | `white_sandstone_blocks_02` | 2.0 m | roughness ×0.5 + clearcoat (existing) | unchanged walls, snow caps on ledges |
| East stucco | `yellow_plaster_02` + `red_brick_plaster_patch_02` | 2.0 / 1.5 m | same | same |
| West arcade | `old_sandstone_02` | 1.5 m | same | same |
| South brick | `medieval_red_brick` | 2.0 m | same | same |
| Court paving | `monastery_stone_floor` (detail) on the procedural layout | 1.8 m | procedural puddles kept | snow layer kept |
| Roofs | `clay_roof_tiles_02` | 2.5 m | clearcoat | snow material |
| Passage vault, walls, floor | `worn_cracked_plaster`, `old_stone_wall`, `cobblestone_floor_08` | 1.8 / 2.0 / 2.0 m | floor clearcoat | – |
| Cloister floor | `terracotta_floor_tiles` | 2.1 m | – | – |
| Street | `cobblestone_pavement` + `concrete_pavement` | 2.5 / 1.8 m | env_kit wet roughness | snow material |

## Additions, so every audit pose looks real

**Beyond the arch (street, z 13–28).** This is what the archway hero view and passage poses see.
- `modular_urban_apartments_facade` at z≈24, as the street facade substitute above.
- `street_lamp_01` on the far pavement.
- `water_manhole_cover` in the road.
- `fire_hydrant` on the pavement.
- `covered_car` (pinned), parked across the arch at an angle so its edge breaks the opening (depth layering).
- 2× `trashbag` (pinned) at the kerb.

**Inside the carriage passage.**
- The retextured vault, walls and floor above.
- `power_box_01` (pinned) at 1.5 m on the east wall.
- `utility_box_01` (pinned) at floor level.
- `wooden_broom` (pinned) leaning by the gate leaf.
- 2× `plastic_crate_01` (pinned), stacked.
- `security_light` above the arch on the street face.

**Corners where wings meet (NE, NW, SE, SW).**
- A downpipe to the ground at NE and SW (procedural, retextured).
- At each corner, a cluster of 2–3 `planter_pot_clay` with `potted_plant_04` and `periwinkle_plant` (pinned).
- `weed_plant_02` and `moss_01` at the wall base. Use at most 2 `moss_01`: it is 246k tris.

**North wall, eye level (main door).**
- `large_castle_door`.
- `Lantern_01` either side.
- `wicker_basket_02` (pinned) by the door.
- `wooden_bucket_01` (pinned).
- A small seating nook: `painted_wooden_chair_01` + `gallinera_table` under the balcony.

**East wall (ochre stucco).**
- `utility_box_02` (pinned) low on the wall.
- `garden_hose_wall_mounted_01` (pinned) near the bike.
- `watering_can_metal_01` (pinned) under the hose.
- `wooden_bucket_02`.

**West cloister walk.**
- `painted_wooden_bench` (pinned) under the third arch.
- `planter_box_02` (pinned) planted with `shrub_04` between two piers.
- `wooden_stool_01`.
- `Lantern_01` hanging in 2 bays.

**South wing (brick, court side).**
- `exterior_aircon_unit` (pinned) on a bracket at 3.5 m.
- `trashbag` + `metal_trash_can` (pinned) by the passage mouth, on the court side.

**Roof edges seen from below.** These are the upward audit poses (tiles 6, 18, 20).
- Retextured gutters and fascia.
- 4 chimneys already exist.
- Add a short procedural TV aerial on one chimney.

**Boxes.**
- Keep the exclude boxes in sync with the new props: car, bench, chair nook, crates.
- The passage exclusions already block cameras inside the walls.

## Notes for the executor

- Every `W.surface` used for a baked facade needs the procedural opening and stain masks composited on top. A plain tiled texture would hide window drips and make all four walls look alike.
- `modular_urban_apartments_facade` is 51.5 m wide. Crop its placement so its ends sit outside the arch's view frustum, or tile two at an angle.
- Re-run the audit (clear and snow, sampler seeds 0 and 1) after the change. The target is no blank or near-blank tile.
