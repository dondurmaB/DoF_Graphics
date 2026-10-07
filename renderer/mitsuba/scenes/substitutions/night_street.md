# night_street: substitution map (planning only)

Source: `output/dev/inventory/night_street.json` (187 groups, 1078 procedural instances, 0 scanned), scene source `scenes/night_street.py`, `_night_street_props.py`, `_night_street_facade.py`. Catalogs: Poly Haven models/textures, ambientCG materials. Poly Haven dimensions are Blender order `[width, depth, height]` in metres.

**Summary: 187 groups -> SUBSTITUTE 15 (trees 8 leaf/bark groups, 2 of 4 trees; lantern post + frame; manhole; AC units; fire-escape iron; shutters; doorstep plant), RETEXTURE 23 (facades x9, trims x2, roofs x3, road, pavement x4, kerbs x2, tree grates, tree-pit soil), KEEP 149. Plus 15 additions.** Nearly all KEEP rows are lit-window interiors, emitters, signage, number plates and car sub-parts. Those carry the night character, and the catalog has nothing equivalent (no street-legal cars, no bollards, no bicycles, no traffic signals).

Triangle budget: the scene is about 0.89M now. Four procedural trees (about 0.3M) become scanned ones at about 1.76M (island_tree_02) or 2.06M (tree_small_02, as converted) each, so four scanned trees would hit about 7.5–8.5M. Plan for 2 scanned hero trees on the near pavement and keep 2 far procedural ones. Alternatively, add instancing (shapegroup/instance) to `web_assets.add`, which would make 4 copies cost one. That is a promotion candidate for the shared loader.

## Object groups

| object group | instances | tris | decision | asset id(s) | size check | notes |
|---|---|---|---|---|---|---|
| tree_0..3.leaf, tree_0..3.bark | 4+4 | 291,768 + 12,040 | SUBSTITUTE (2 of 4) | island_tree_02 (8.5 m), tree_small_02 as alt | procedural tree 7.5 m; island_tree_02 scaled 0.88 to 7.5 m; tree_small_02 4.65 m would need 1.6x (too much) | Put scanned trees on the 2 nearest grates (lamps/puddle_car views). Keep 2 far ones procedural unless instancing lands. Leaves are already alpha-masked principledthin via web_assets. |
| tree_grate | 4 | 3,648 | KEEP + RETEXTURE | metal_grate_rusty (PH) | 1.2 m square, fine | Geometry is right; swap flat iron for the rusty grate texture. |
| lantern_post, lantern_frame, lantern_panels [emitter] | 5 each | 2,920 + 1,440 + 40 | SUBSTITUTE (post + frame) | street_lamp_01 | 3.87 m tall vs 4.2 m procedural: uniform scale 1.085 | Keep the procedural `lantern_panels` emitter, re-positioned inside the scanned lantern head (the scan's glass is not an emitter). Sodium colour and power unchanged. |
| cobra_post, cobra_head | 5 each | 8,640 + 2,000 | KEEP | none | 8 m LED cobra-heads | No cobra-head/highway luminaire in the catalog (street_lamp_02 is a 1.68 m lantern). |
| manhole_disc | 4 | 640 | SUBSTITUTE | water_manhole_cover | 0.69 m diameter, matches a standard cover | Flush with road (0.07 m thick: sink 0.065 m into the asphalt box). |
| gully_bars, rectangle:gully_hole | 4 each | 624 + 8 | KEEP | none | | No gully grate model; tiny. |
| ac_unit | 15 | 324 | SUBSTITUTE | exterior_aircon_unit | 1.8 x 0.37 x 0.93 m; uniform scale 0.45 gives 0.81 x 0.17 x 0.42 m vs procedural 0.78 x 0.34 x 0.48 box | Wall-mounted under windows. 15 x 19k tris is about 0.28M, fine. |
| iron (balconies + fire escapes) | 16 | 22,764 | SUBSTITUTE (fire escapes only) | modular_fire_escape | 6.66 x 1.42 x 9.76 m kit vs procedural escape about 2.2 m bays per storey | Use the modular pieces per storey on `fire_escape` buildings. Balcony railings stay procedural (they fit the window grid). |
| shutter (rolled-down shop shutters) | 2 | 1,452 | SUBSTITUTE | rollershutter_window_02 / _03 (3.6 / 2.98 m wide), rollershutter_door (3.08 x 2.4) | match each shopfront width; pick the closest and scale x within about 10% | Painted/grafitti-free metal; keep it closed on unlit shops. |
| drain (downpipes, hoppers, brackets, roof ladders) | 15 | 4,848 | KEEP | (modular_metal_gutter considered) | gutter kit is 2.27 x 1.51 x 1.84 m, a different system | Procedural pipes follow each facade's height exactly; scans would not fit. |
| bin.body, bin.lid (kerbside litter bins) | 3 each | 672 + 708 | KEEP | none | 0.6 m litter bins | metal_trash_can is a 1.85 m wide commercial bin, wrong scale for kerbside. It is listed as an addition instead. |
| bollard.body, bollard.band | 8 each | 1,920 + 384 | KEEP | none | | No bollard in the catalog. |
| meter (parking meters) | 2 | 296 | KEEP | none | | No parking meter in the catalog. |
| sign_pole, plate_round.*, plate_square.*, rectangle:street_name | 4 / 2+2 / 2+2 / 1 | 192 / 160 / 8 / 2 | KEEP | none | | Legible custom sign faces (no parking, speed, crossing, one way, street name). |
| signal_pole, signal_head.body, signal_head.hood, sphere:signal_lens_off | 2 / 2 / 2 / 4 | 96 + 48 + 264 + 0 | KEEP | none | | No traffic signal in the catalog. The emitters are part of the night look. |
| bike.metal, bike.tyre, bike.frame, bike.saddle, bike_rack | 1 each | 2,096 + 1,536 + 320 + 196 + 1,440 | KEEP | none | | No bicycle in the catalog. |
| cube:shelter_metal (bus shelter) | 10 | 120 | KEEP | none | | Glass removed for the NEE trap. Add modular_street_seating inside (see additions). |
| leaf_plant | 1 | 1,200 | SUBSTITUTE | potted_plant_02 | 0.84 m tall, fits a shop doorstep | |
| rectangle:soil (tree pits) | 4 | 8 | RETEXTURE | ambientCG Ground (dark soil) or PH brown_mud_* | flat quad | Low value. Do it only if the scanned trees go in. |
| car_suv.*, car_hatch.*, car_sedan.* (body, tyre, rim, disc, glass, bezel, trim, seam, head/tail/reverse lens, mirror, grille) | 6 / 5 / 4 per part | about 230k total | KEEP | covered_car (only vehicle) | covered_car 4.38 m long | No street car in the catalog. Optionally replace one kerb-parked car with covered_car (addition 14). |
| rectangle:plate_0..7 (number plates) | 1–6 each | 2–12 | KEEP | none | | Procedural plate atlas, legible. |
| products, shelf, riser, rim_brass (shop gondolas) | 16 / 16 / 16 / 16 | 91,044 + 4,932 + 312 + 492 | KEEP | none | | Lit interior seen through shop glass. Scans would add tris for little visible gain at night. |
| shop_wall_a/b/c/cool, shop_floor, shop_glass | 6/7/9/2, 16, 16 | 704, 52, 104 | KEEP | none | | Behind glass, lit by emitters. |
| awning (awning_0, awning_1) | 6 | 36 | KEEP | none | | Custom stripe texture; no awning scan. |
| letters_dark/white/gold, fascia_0..5 | 7/6/4, 27 | 16,560, 396 | KEEP | none | | Shop sign lettering and fascias carry the character. |
| emit_* emitters (neon red/amber/white/green/pink/violet, shop warm/cool, room warm/neutral/cool, sheer warm/neutral/cool, office, hall, goose, tv, display, screen, dormer, box_*), rectangle:emitter_backing, sphere:emitter_backing | 1–22 each | about 18k | KEEP | none | | The night lighting. Must stay procedural, with power-proportional weights. |
| room_a/b/c, room_ceiling, room_floor, furn_dark/light/red/wood, curtain_0..3, blind, shelf_books, art, door (interior) | 13–24 each | about 33k | KEEP | none | | Lit-window interiors seen through openings at 5–30 m. Low visual return for scans, high tri cost. |
| door (street entrances, door_0..3) | 15 | 900 | KEEP | (large_castle_door rejected) | castle door 2.0 x 2.96 m, wrong style | Residential doors are fine procedurally. |
| frame_white/dark/wood/alu, glass_dark | 6/15/5/13, 15 | 28,236, 420 | KEEP | none | | Window/door frames are cut to each opening. |
| wall_glass_frame, glass_curtain, spandrel, office_floor | 1 each | 586 + 44 + 12 + 16 | KEEP | none | | Curtain-wall office block: needs the specific dark reflective look. |
| wires (tram contact + span + utility cables) | 1 | 6,948 | KEEP | (modular_electricity_poles rejected) | poles are 13 m utility poles, different system | Thin wires are the DoF-edge feature. Keep exact catenaries. |
| wood_tank (rooftop water tanks) | 2 | 336 | KEEP | none | | No water tank in the catalog; seen far up. |
| decal_paint (lane markings, zebra), decal_crack, decal_asphalt_patch | 1 each | 19,356 + 2,154 + 10 | KEEP | none | | Worn markings and cracks are decals over the road. Keep them above the new asphalt texture. |
| tactile, tactile_- (blister paving) | 1 each | 6,482 x 2 | KEEP | none | | Specific yellow tactile geometry. |
| wall_brick_red | 3 | 1,530 | RETEXTURE | red_brick_03 (PH) / alt ambientCG Bricks0xx | wall uvs in metres (`wall_with_openings` u0) | Replace the procedural `wall_texture` albedo with scanned brick plus a normal map. Keep a seeded grime/soot multiplier at street level. |
| wall_brick_brown | 3 | 1,496 | RETEXTURE | brick_wall_02 (PH) | uv in metres | |
| wall_brick_buff | 1 | 564 | RETEXTURE | yellow_bricks (PH) | uv in metres | |
| wall_stone_grey | 2 | 1,124 | RETEXTURE | stone_brick_wall_001 (PH) | uv in metres | |
| wall_stone_beige | 2 | 970 | RETEXTURE | sandstone_blocks_05 (PH) | uv in metres | |
| wall_stucco_ochre, wall_stucco_sage, wall_stucco_pink, wall_stucco_cream | 1 each | 418 / 358 / 304 / 298 | RETEXTURE (x4) | white_plaster_rough_01 (PH) with `tint=` per colour | uv in metres | One download, four tints. |
| stone_trim (sills, cornices, string courses) | 14 | 9,336 | RETEXTURE | white_sandstone_bricks or beige_wall_001 (PH) | thin bands; check uv scale | Plain stone texture, no visible coursing at that scale. |
| stucco_trim | 6 | 4,104 | RETEXTURE | white_plaster_02 (PH) | | |
| roof, cube:roof (flat roofs) | 16 + 16 | 32 + 192 | RETEXTURE | bitumen (PH, already pinned by rooftop) | | Seen from tilted/high poses. |
| slate (mansard) | 2 | 8 | RETEXTURE | roof_slates_02 (PH) | uv in metres on the mansard quads | |
| box_road (carriageway) | 1 | 12 | RETEXTURE | asphalt_02 (PH) or ambientCG Asphalt0xx | 80 m x 9 m box; uv must be metres | Wet env: keep the seeded puddle roughness map as the roughness multiplier on top of the scanned albedo/normal. Puddles stay sharp mirrors. |
| box_pave_*, pave_drop, pave_drop_- (pavement slabs) | 2+2+1+1 | 24+24+30+30 | RETEXTURE | concrete_pavers_02 (PH) | slab size about 0.6 m in the scan, matches UK slabs | Wet variant darkened and lower roughness as now. |
| kerbs, kerbs_- (granite) | 1 each | 1,020 x 2 | RETEXTURE | granite_tile_02 (PH) | 0.6 m texture repeat | |
| tree_grate | 4 | 3,648 | (see above) | metal_grate_rusty | | Counted once under RETEXTURE. |

## Textures for big surfaces (all CC0)

| surface | current | replace with | source | fallback |
|---|---|---|---|---|
| carriageway | procedural aggregate asphalt + patches | asphalt_02 | Poly Haven | ambientCG Asphalt0xx (46 options) |
| pavement slabs | procedural paving | concrete_pavers_02 | Poly Haven | ambientCG PavingStones0xx |
| kerbs | procedural granite | granite_tile_02 | Poly Haven | granite_wall |
| red brick facades | procedural `wall_texture("brick_red")` | red_brick_03 | Poly Haven | ambientCG Bricks0xx |
| brown brick facades | procedural | brick_wall_02 | Poly Haven | brick_wall_005 |
| buff brick facade | procedural | yellow_bricks | Poly Haven | yellow_brick |
| grey stone facades | procedural | stone_brick_wall_001 | Poly Haven | stone_wall |
| beige stone facades | procedural | sandstone_blocks_05 | Poly Haven | white_sandstone_bricks |
| stucco facades (4 colours) | flat principled | white_plaster_rough_01 + tint | Poly Haven | ambientCG Plaster00x, Facade0xx |
| trims | flat principled | white_sandstone_bricks / white_plaster_02 | Poly Haven | beige_wall_001 |
| flat roofs | flat | bitumen | Poly Haven | roof_07 |
| mansard | flat slate | roof_slates_02 | Poly Haven | grey_roof_tiles |
| tree grates | flat iron | metal_grate_rusty | Poly Haven | ambientCG Grate001/002 |

## Additions a real night street has (all in the Poly Haven catalog)

1. **fire_hydrant** (0.8 m): 1–2 on the pavement edge near a lamp.
2. **trashbag** (0.57 m): clusters of 3–5 by shop doors and kerbside bins. Seeded placement.
3. **cardboard_box_01** (0.34 m): flattened or stacked beside a closed shopfront and the shutter.
4. **plastic_crate_01 / _02 / _03**: a stack outside the grocery shopfront.
5. **metal_trash_can** (1.85 m commercial bin): one at a gap between buildings or a service entrance, not at the kerb.
6. **utility_box_01 / utility_box_02** (1.12 m street cabinets): 2 on the pavement back edge.
7. **security_camera_01 / _02**: on shopfront fascias and the office block entrance.
8. **security_light / industrial_wall_lamp**: over service doors. As small emitters these add realistic warm or cool pools; give them power-proportional weights.
9. **modular_street_seating** (4.34 m): inside or beside the bus shelter (replaces the missing seat).
10. **potted_plant_02 / potted_plant_04 / planter_box_02**: on shop doorsteps and window sills of lit ground-floor rooms.
11. **street_lamp_02** (1.68 m lantern bracket): as wall-mounted lanterns at residential entrances, with the panel emitter re-used.
12. **concrete_road_barrier_02**: one short roadworks section by the patched asphalt, a common real-street detail.
13. **wooden_crate_01**: by the shop with produce shelves.
14. **covered_car** (4.38 m): one kerb-parked car under a cover, for variety; the only vehicle scan available.
15. **street_rat** (0.15 m): one by the trash bags; tiny, but a believable close-up subject for low poses.

Not in any catalog (keep procedural): cars, bicycles, bollards, parking meters, traffic signals, cobra-head lamps, road signs, bus shelter frame, rooftop water tanks.
