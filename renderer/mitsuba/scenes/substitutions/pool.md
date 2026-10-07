# Substitution map: `pool`

Summary: 98 object groups. **SUBSTITUTE 6** (outside trees x3 species, hedges, bin, wall clock), **RETEXTURE 12** surfaces, **KEEP 80**. **ADDITIONS 11** catalog items.
The user rated the scene good, so indoor geometry that already reads right (water, lane ropes, blocks, ladders, flags, lifeguard chair, stands, signage) stays. The weakest part is the view through the glazing (cartoon trees, blob hedges, flat lawn, box building), plus flat deck and wall materials.

Sources: PH = Poly Haven (supported by web_assets.py now), ACG = ambientCG (loader support pending). Sizes from the catalogs in metres [w, d, h].

## Objects

| object group | instances | tris | decision | asset id(s) | size check | notes |
|---|---|---|---|---|---|---|
| water | 1 | 375,600 | KEEP | – | – | Dielectric heightfield; the point of the scene. |
| rope_0..4.c (lane ropes) | 15 | 361,600 | KEEP | – | – | Nothing in catalog; procedural floats read correctly. |
| seats.c (stadium seats) | 5 | 160,704 | KEEP | – | – | No stadium seat in PH; `plastic_monobloc_chair_01` is the wrong type. |
| seats.post, stand_rail | 2 | 6,560 | KEEP | – | – | Structural. |
| hedge | 3 | 180,000 | SUBSTITUTE | PH `shrub_02` (row of instances, scaled) or `wild_rooibos_bush` | shrub_02 5.3 x 2.1 x 2.1 m vs 12 x 0.9 x 1.2 m hedge | Current hedges are leaf blobs. Use 3–4 shrub_02 per hedge run, scaled to about 1.2–1.5 m tall. |
| tree_0/1/2 (leaf + trunk) | 7 | 203,616 | SUBSTITUTE | PH `island_tree_02`, `searsia_lucida`, `tree_small_02` | island_tree_02 8.5 m, searsia_lucida 4.8 m, tree_small_02 4.6 m vs procedural 6.5–8.5 m | Three species, seeded mix. Mind the triangle budget: island_tree_02 1.76M, searsia_lucida 0.84M, tree_small_02 2.06M. Use 2 near instances of a heavy one and reuse the PLY for the rest (instances share files). |
| treeline (900 m ring) | 1 | 1,440 | KEEP | – | – | Horizon closure at 900 m; too far to need detail. |
| box_lawn | 1 | 12 | RETEXTURE | PH `leafy_grass` (or ACG Grass004) | 2.0 m tile | Flat lawn; add grass ground cover near the glazing (see additions). |
| box_path | 1 | 12 | RETEXTURE | PH `concrete_pavement_02` | 1.8 m tile | |
| box_far_building | 1 | 12 | RETEXTURE + detail | PH `rectangular_facade_tiles` / ACG Facade006 | 2.0 m tile | Box with painted windows. Retexture, and add window reveals or a second massing so it isn't a single box. |
| bin (noodle bin) | 1 | 320 | SUBSTITUTE | PH `plastic_container` | 0.90 x 0.63 x 0.43 m vs 0.66 x 0.6 m lathe | A plastic tub reads better than the lathe cylinder. Keep the procedural noodles in it. |
| clock_bezel + rectangle:clockface | 2 | 290 | SUBSTITUTE | PH `wall_clock` | 0.32 m dia vs about 0.5 m pace clock | A pool pace clock is bigger: scale `wall_clock` to about 0.6 m, or KEEP if the pace-clock look matters. |
| toys.t (kickboards, noodles) | 6 | 17,000 | KEEP | – | – | Nothing in catalog. |
| rack (kickboard rack) | 1 | 648 | KEEP | (alt PH `steel_frame_shelves_02`) | 0.59 x 0.5 x 2.14 m vs 1.2 x 0.45 x 1.25 m | The procedural rack fits the kickboards; steel_frame_shelves is too tall. |
| pendant + pendant_cable (+ disk emitters) | 96 | 14,464 | KEEP | – | – | Area-emitter design; `modern_ceiling_lamp_01` would add 32 copies for no gain. |
| ladder.steel | 4 | 4,128 | KEEP | – | – | Pool ladders: none in PH. |
| block.* (starting blocks) | 24+ | 1,920 | KEEP | – | – | None in PH. |
| chair.frame/seat (lifeguard chair) | 2 | 952 | KEEP | – | – | None in PH. |
| bench.steel/wood | 6 | 396 | KEEP + retexture wood | wood: PH `wood_planks` or `brown_planks_04` | – | Swapping in `painted_wooden_bench` (1.16 m) would look out of place; the 2 m slat bench is right for a pool. |
| flag_pole, flags_*.cord/c | 12 | 1,490 | KEEP | – | – | |
| duct | 1 | 2,176 | KEEP (alt PH `modular_airduct_circular_01`) | – | – | The procedural duct already reads right. |
| cube:alu (window mullions) | 64 | 768 | KEEP | – | – | Structural. |
| box_beam / box_beam_- (glulam) | 9 | 108 | RETEXTURE | PH `roof_planks` or `beam_wall_01` | 1.5 / 1.0 m | Better lamination grain. |
| pool_floor, pool_wall, pool_band, lane_lines | 10 | 3,568 | RETEXTURE (optional) | ACG `Tiles132A` (clean blue pool tiles), ACG `Tiles020` (lane band) | – | The procedural mosaic is already convincing through the water; switch only once ACG works, and A/B it. |
| box_dado, box_lw_dado, cube:white_tile, box_coping | 8 | 96 | RETEXTURE | PH `square_tiled_wall` or ACG `Tiles071` | 2.0 m tile | Wall tile looks flat. |
| cube:coping_edge | 4 | 48 | KEEP | – | – | Solid blue nosing; correct as is. |
| deck_top | 1 | 8 | RETEXTURE | PH `anti_skid_tiles` | 1.8 m tile | The current checker reads as CG (see final_ladder). Real anti-slip tile is the most visible upgrade indoors. |
| box_tier (stand tiers) | 4 | 48 | RETEXTURE | PH `brushed_concrete` | 2.5 m tile | |
| box_endwall, box_rightwall, box_lw_head, cube:plaster | 14 | 168 | RETEXTURE | PH `beige_wall_001` (or ACG Plaster001 / PaintedPlaster017) | 3.0 m tile | Speckled procedural plaster. |
| box_ceiling | 1 | 12 | RETEXTURE | PH `roof_planks` (timber slat ceiling) | 1.5 m tile | Natatoria often have timber ceilings, which matches the glulam. |
| cube:door | 5 | 60 | KEEP | – | – | Flat panels at distance. |
| cube:block_body, apron/backplate | 12 | 144 | KEEP | – | – | Part of the starting blocks. |
| cube:backing (+ emitter) | 1 | 12 | KEEP | – | – | Scoreboard backing / emitter. |
| cube:steel | 4 | 48 | KEEP | – | – | Structural. |
| rectangle:* (lane numbers, markers, no-dive, banners, signs) | 31 | 62 | KEEP | – | – | Textured decals with legible text; better than any catalog item. |

## Textures for big surfaces

| surface | current | proposed | source | tile size |
|---|---|---|---|---|
| deck | procedural checker | `anti_skid_tiles` | PH | 1.8 m |
| hall walls (plaster) | procedural speckle | `beige_wall_001` | PH | 3.0 m |
| dado / white wall tile | procedural | `square_tiled_wall` (alt ACG Tiles071) | PH | 2.0 m |
| ceiling | flat diffuse | `roof_planks` | PH | 1.5 m |
| glulam beams | procedural grain | `beam_wall_01` | PH | 1.0 m |
| stand tiers | procedural concrete | `brushed_concrete` | PH | 2.5 m |
| bench wood | procedural planks | `brown_planks_04` | PH | – |
| pool tiles (optional A/B) | procedural mosaic | ACG `Tiles132A`; lane band ACG `Tiles020` | ACG | – |
| lawn outside | procedural grass | `leafy_grass` | PH | 2.0 m |
| path outside | flat colour | `concrete_pavement_02` | PH | 1.8 m |
| far building | procedural facade | `rectangular_facade_tiles` (alt ACG Facade006) | PH | 2.0 m |

## Additions (all in the PH catalog, all things a real indoor pool has)

1. `lifebuoy` (0.78 m) on a wall bracket near the lifeguard chair.
2. `korean_fire_extinguisher_01` (0.66 m) by the changing-rooms door.
3. `WetFloorSign_01` (0.63 m) on the deck near the ladder, seeded on or off.
4. `metal_trash_can` (check orientation; dims read 1.85 x 0.56 x 0.91, so verify the scale) or `plastic_container` as a deck bin.
5. `plastic_monobloc_chair_01` x 2–4 stacked or loose along the window side.
6. `potted_plant_01` / `potted_plant_04` near the entrance doors.
7. `multi_cleaner_5_litre`, `all_purpose_cleaner`, `bleach_bottle` on a low shelf by the equipment rack (pool chemicals).
8. `garden_hose_wall_mounted_01` (0.56 m) on the end wall (deck wash-down).
9. `utility_box_01` (1.12 m) as a plant/electrical cabinet by the doors.
10. Outside: `grass_medium_01` patches and `shrub_04` along the glazing foot, so the lawn edge isn't a hard line. Watch the triangles: grass_medium_01 is 1.6M, so use grass_medium_02 (1.0M) or shrub_04 (48k) instances.
11. Outside: `modular_street_seating` or `wooden_picnic_table` on the lawn, and `street_lamp_02` along the path.

Budget note: the scene is at 1.34M triangles. Trees (two heavy instances about 3.5M), hedges (shrub_02, 52k each) and the additions fit inside 8M only if tree PLYs are reused across instances and grass_medium_01 is avoided.
