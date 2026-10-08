# Cafe: substitution plan

**78 object groups: 26 SUBSTITUTE, 13 RETEXTURE, 39 KEEP**, plus 13 additions below. Source: output/dev/inventory/cafe.json (scene seed 7). Catalog dimensions are Poly Haven's (meters, axis order as listed); confirm with `web_assets.py info` before placing.

Converter caveats that shape these choices: glTF glass/transmission is not converted (scanned glass turns opaque, so glass stays procedural); rectangle/cube primitives have 0..1 uv per face, so `W.surface` needs an explicit `uv_scale` or a switch to `box_mesh`; ambientCG ids need the planned ambientCG support in web_assets.

## Object groups

| object group | instances | tris | decision | asset id(s) | size check | notes | status |
|---|---|---|---|---|---|---|---|
| bistro_chair.frame | 7 | 54,208 | SUBSTITUTE | dining_chair_02 (first choice) or gallinera_chair | 0.43x0.58x0.97 m vs procedural bentwood about 0.42x0.5x0.9 m: OK at scale 1 | Real worn wood and joints. Check the style is bistro, not dining room; if wrong, use painted_wooden_chair_01 (0.43x0.54x0.96). Replaces both chair parts. | REVERTED: scanned dining/painted chairs are upholstered dining-room chairs and lose the bentwood bistro look |
| bistro_chair.seat | 7 | 22,848 | SUBSTITUTE | (same model as frame) | - | Seat comes with the scanned chair. | REVERTED: see bistro_chair.frame |
| plant_big.leaves | 1 | 5,832 | SUBSTITUTE | potted_plant_01 | 1.35 m vs 1.3 m: scale 1 | Replaces leaves, stems, pot and soil of the big floor plant. | DONE |
| plant_small.leaves | 2 | 4,320 | SUBSTITUTE | potted_plant_04 | 0.27 m vs 0.32 m: scale 1-1.15 | Window-sill plants; replaces all 4 parts. Mix in planter_pot_clay + a succulent for variety. | DONE |
| teapot.body | 1 | 4,096 | SUBSTITUTE | tea_set_01 (teapot + cups) if its parts separate; else KEEP | set is 0.92x0.48 m incl. tray: place teapot part only | Hero subject and default focus target: re-point focus_points['teapot'] at the new model. Verify parts with web_assets info first. | REVERTED: tea_set_01 is one merged mesh (tray, pot, cups); the teapot cannot be isolated |
| mirror_ring | 1 | 4,096 | SUBSTITUTE | ornate_mirror_01 | 0.49x0.74 m vs 0.9 m round: scale 1-1.2 | Vintage framed mirror suits the cafe; mirror glass part may need a conductor override (converter maps glTF metal/rough only). | DONE |
| plant_mid.leaves | 1 | 4,032 | SUBSTITUTE | potted_plant_02 | 0.84 m vs 0.8 m: scale 1 | Right-wall plant; replaces all 4 parts. | DONE |
| bowl | 1 | 3,888 | SUBSTITUTE | wooden_bowl_01 | 0.31 m vs about 0.2 m: scale 0.75 | Fruit bowl on the counter. | DONE |
| plant_small.stems | 2 | 3,840 | SUBSTITUTE | (potted_plant_04) | - |  | DONE |
| plant_big.stems | 1 | 3,456 | SUBSTITUTE | (potted_plant_01) | - |  | DONE |
| plant_small.pot | 2 | 3,328 | SUBSTITUTE | (potted_plant_04) | - |  | DONE |
| croissant | 1 | 3,072 | SUBSTITUTE | croissant | 0.2 m: scale 0.8 to look like a cafe croissant | Already proven on the rooftop. | DONE |
| plant_mid.stems | 1 | 2,688 | SUBSTITUTE | (potted_plant_02) | - |  | DONE |
| vase.vase | 1 | 2,304 | SUBSTITUTE | ceramic_vase_02 or antique_ceramic_vase_01 | 0.31 m / 0.44 m tall vs about 0.2 m: scale 0.6-0.7 OK | Keep procedural flowers placed in it; re-anchor stems to the new neck height. | DONE |
| plant_big.pot | 1 | 1,664 | SUBSTITUTE | (potted_plant_01) | - |  | DONE |
| plant_mid.pot | 1 | 1,664 | SUBSTITUTE | (potted_plant_02) | - |  | DONE |
| teapot.lid | 1 | 1,584 | SUBSTITUTE | (tea_set_01) | - |  | REVERTED: see teapot.body |
| cake_dome.cake | 1 | 576 | SUBSTITUTE | strawberry_chocolate_cake or carrot_cake | 0.24 m vs 0.22 m dome interior: scale 0.9 | Seen through the glass dome, so photographic texture pays off. | DONE |
| plant_small.soil | 2 | 128 | SUBSTITUTE | (potted_plant_04) | - |  | DONE |
| plant_big.soil | 1 | 64 | SUBSTITUTE | (potted_plant_01) | - |  | DONE |
| plant_mid.soil | 1 | 64 | SUBSTITUTE | (potted_plant_02) | - |  | DONE |
| cube:aluminium | 2 | 24 | SUBSTITUTE | classic_laptop | catalog 0.65x0.49x0.54 m looks like a large retro laptop: check bounds, scale to about 0.33 m wide | Keep the emissive screen rect, re-fit onto the scanned lid; if the model is too retro, KEEP. | REVERTED: classic_laptop is a 0.65 m retro machine; shrunk it reads as a toy |
| rectangle:art_a | 1 | 2 | SUBSTITUTE | hanging_picture_frame_01 / _02 / fancy_picture_frame_01 | 0.59x0.84 / 0.75x0.5 m: scale 1 | Scanned frames with their own pictures replace rect + black bars. | DONE |
| rectangle:art_b | 1 | 2 | SUBSTITUTE | (see art_a) | - |  | DONE |
| sphere:orange | 6 | 0 | SUBSTITUTE | lemon + food_apple_01 + food_pears_asian_01 | 0.07-0.1 m each: scale 1 | Scanned fruit pile replaces 6 shaded spheres. | DONE |
| disk:mirror | 1 | 0 | SUBSTITUTE | (ornate_mirror_01) | - | Replaced together with the ring. | DONE |
| table_top | 4 | 13,056 | RETEXTURE | marble_01 (PH) for marble tops; black_walnut_veneer_01 or mocha_oak_veneer (PH) for wood tops | - | No scanned round bistro table. Mesh is a disc: uv must be in meters (else pass uv_scale). | DONE |
| cube:dark_wood | 13 | 156 | RETEXTURE | dark_wood or black_walnut_veneer_01 (PH) | - | Skirting, rails, beams, chalkboard frame. Cubes have 0..1 uv per face: pass uv_scale=size/texture_width or switch to box_mesh. | DONE |
| cube:window_frame | 12 | 144 | RETEXTURE | PaintedWood007A (AC) tinted dark green | - | Painted window joinery; needs ambientCG support in web_assets. | DONE |
| cube:wall | 7 | 84 | RETEXTURE | painted_plaster_wall or beige_wall_001 (PH) | - | Window-wall pieces; must match rect:wall. | DONE |
| cube:shelf_wood | 5 | 60 | RETEXTURE | oak_wood_planks or brown_planks_04 (PH) | - | Back-bar shelves and sills. | DONE |
| cube:facade | 1 | 12 | RETEXTURE | Facade00x (AC, photographic facades with windows) or brick_wall_001 / white_stucco (PH) | - | Today a flat 2 m slab 17 m away; the window-view task rebuilds it as real facades (see additions), this is the minimum. | DONE |
| box_counter_body | 1 | 12 | RETEXTURE | long_white_tiles (PH) | - | Subway-style counter front; box_mesh already has uv in meters. | REVERTED: long_white_tiles rendered as a dull grey grid; green subway tiles kept for character |
| box_counter_top | 1 | 12 | RETEXTURE | marble_01 (PH) or Marble006 (AC) | - |  | DONE |
| rectangle:wall | 3 | 6 | RETEXTURE | painted_plaster_wall or beige_wall_001 (PH) | - | Main walls. rectangle uv is 0..1: pass uv_scale=(width/tex_w, height/tex_h) or rebuild as box_mesh. | DONE |
| rectangle:floor | 1 | 2 | RETEXTURE | old_wood_floor or wood_floor_worn (PH) | - | Biggest surface in most poses; replaces procedural planks. | DONE |
| rectangle:ceiling | 1 | 2 | RETEXTURE | white_stucco (PH) or painted_plaster_wall | - | Ceiling between beams; audits show it in upward poses. | DONE |
| rectangle:wall_green | 1 | 2 | RETEXTURE | green_rough_planks (PH) or PaintedWood (AC) tinted | - | Green wainscot. | DONE |
| rectangle:street | 1 | 2 | RETEXTURE | asphalt_02 (PH) road + brick_pavement_02 or PavingStones (AC) pavement | - | Split into road, kerb and pavement in the window-view task. | DONE |
| bottle | 52 | 157,248 | KEEP | - | - | Real dielectric glass in green/amber/clear; the converter ignores KHR_materials_transmission, so `wine_bottles_01` (0.33 m tall row) would turn opaque. Revisit only if the converter gains glass support; then swap about half the shelf bottles for it. | kept |
| tumbler | 35 | 87,360 | KEEP | - | - | Simple glass/ceramic/copper cylinders, convincing; no scanned tumbler. | kept |
| jar.lid | 20 | 70,400 | KEEP | - | - | Glass storage jars; no scanned glass jar. | kept |
| jar.glass | 20 | 49,920 | KEEP | - | - | As above (dielectric). | kept |
| cup.cup | 7 | 23,520 | KEEP | - | - | Clean ceramic cups read fine; tea_set_01 covers the hero table (see teapot). | kept |
| saucer | 8 | 13,312 | KEEP | - | - | No scanned saucer. | kept |
| dome_shade.outer | 4 | 13,248 | KEEP | - | - | Enamel pendants are on style and hold the emitters; hanging_industrial_lamp (0.55 m wide) would be too industrial. | kept |
| dome_shade.inner | 4 | 10,944 | KEEP | - | - | Part of pendant. | kept |
| spoon | 3 | 7,824 | KEEP | - | - | No metal teaspoon (wooden_spoon is a 0.28 m cooking spoon). | kept |
| cake_dome.dome | 1 | 7,680 | KEEP | - | - | Dielectric dome. | kept |
| table_base | 4 | 6,656 | KEEP | - | - | Cast-iron base, fine; small on screen. | kept |
| vase.hearts | 1 | 6,624 | KEEP | - | - | Procedural flowers; scanned flowers are wild plants at 1-3 m scale. | kept |
| wine_glass | 1 | 5,184 | KEEP | - | - | Dielectric hero glass. | kept |
| candle.holder | 2 | 4,992 | KEEP | - | - | Frosted glass around an emitter; scanned glass would block light. | kept |
| candle.flame [emitter] | 2 | 4,416 | KEEP | - | - | Emitter. | kept |
| socket | 8 | 3,072 | KEEP | - | - | Small brass socket under each shade. | kept |
| string_wire | 1 | 1,932 | KEEP | - | - | Festoon wire. | kept |
| plate | 1 | 1,728 | KEEP | - | - | White ceramic plate; carved_wooden_plate is wood. | kept |
| vase.petals | 1 | 1,512 | KEEP | - | - | Procedural flowers. | kept |
| vase.stems | 1 | 1,152 | KEEP | - | - | Procedural flowers. | kept |
| cake_dome.stand | 1 | 1,152 | KEEP | - | - | Ceramic stand. | kept |
| espresso.black | 1 | 972 | KEEP | - | - | No scanned espresso machine in Poly Haven; keep (consider a later Smithsonian/other CC0 search). | kept |
| espresso.chrome | 1 | 948 | KEEP | - | - |  | kept |
| candle.wax | 2 | 768 | KEEP | - | - |  | kept |
| cup.coffee | 7 | 672 | KEEP | - | - | Coffee surface inside cups. | kept |
| cord | 8 | 256 | KEEP | - | - | Pendant cords. | kept |
| vase.leaves | 1 | 216 | KEEP | - | - | Procedural flowers. | kept |
| cube:pastry | 10 | 120 | KEEP | - | - | 1 cm cookies inside jars, unreadable at any distance. | kept |
| cube:black | 9 | 108 | KEEP | - | - | Kick plate and picture-frame bars (frames go away with the art swap). | kept |
| cube:brass | 9 | 108 | KEEP | - | - | Shelf brackets. | kept |
| candle.wick | 2 | 72 | KEEP | - | - |  | kept |
| cube:paper | 4 | 48 | KEEP | - | - | Page blocks of the procedural book stack. | kept |
| cube:book_2 | 2 | 24 | KEEP | - | - |  | kept |
| cube:book_3 | 1 | 12 | KEEP | - | - |  | kept |
| cube:book_1 | 1 | 12 | KEEP | - | - | Seeded book stack on table 3; scanned book sets are shelf rows (see additions). | kept |
| espresso.panel | 1 | 12 | KEEP | - | - |  | kept |
| rectangle:emitter_backing [emitter] | 1 | 2 | KEEP | - | - | Laptop screen emitter (re-fit to the new laptop). | DONE (kept with procedural laptop) |
| rectangle:chalkboard | 1 | 2 | KEEP | - | - | Procedural chalk menu text is convincing and readable. | kept |
| sphere:emitter_backing [emitter] | 36 | 0 | KEEP | - | - | Globe bulbs, pendant and string-light emitters; glass bulb models would block NEE. | kept |

## Big surfaces (textures)

| surface | current | proposed | source |
|---|---|---|---|
| floor | procedural planks (G.wood_planks) | old_wood_floor (alt wood_floor_worn) | Poly Haven |
| walls | procedural plaster | painted_plaster_wall (alt beige_wall_001) | Poly Haven |
| green wainscot | procedural paint | green_rough_planks (alt PaintedWood tinted) | Poly Haven / ambientCG |
| ceiling | procedural plaster | white_stucco | Poly Haven |
| beams, skirting, rails | procedural walnut | dark_wood (alt black_walnut_veneer_01) | Poly Haven |
| counter front | procedural subway tiles | long_white_tiles | Poly Haven |
| counter top, marble tables | procedural marble | marble_01 (alt Marble006) | Poly Haven / ambientCG |
| wood table tops, shelves | procedural oak | mocha_oak_veneer, oak_wood_planks | Poly Haven |
| window joinery | flat dark green | PaintedWood007A tinted | ambientCG |
| street outside | flat procedural | asphalt_02 road, brick_pavement_02 pavement, granite kerb (PavingStones) | Poly Haven / ambientCG |
| facade across the street | flat plaster slab | Facade00x photographic facades, or brick_wall_001 / white_stucco with real window openings | ambientCG / Poly Haven |

## Additions

Status: all DONE except as noted. modular_street_seating -> REVERTED: two parts share one name in the glTF and
collide in conversion; two outdoor_table_chair_set_01 terrace sets cover it. decorative_book_set_01 -> REVERTED:
no glTF published; book_encyclopedia_set_01 used on the back bar instead. vintage_telephone_wall_clock and
hanging_picture_frame_03 went on the new entrance wall (door, coat stand, radiator, two hanging bulbs), which
the audits showed was bare.

- **CashRegister_01**: on the counter by the espresso machine (0.6 m: check bounds, likely scale 0.7)
- **bar_chair_round_01 x3**: counter stools (0.75 m seat: scale 1)
- **wall_clock or vintage_telephone_wall_clock**: back wall or right wall (0.32 m: scale 1)
- **standing_chalkboard_01**: on the pavement outside the door (classic cafe A-board) or by the entrance (1.51 m: scale 0.6-0.7)
- **jug_01, brass_vase_02, decorative_book_set_01**: back-bar shelves between bottles (small: scale 1)
- **vintage_electric_kettle**: behind the counter (0.3 m: scale 1)
- **hanging_picture_frame_03 / standing_picture_frame_01**: extra wall art so close poses on bare walls see something (scale 1)
- **potted_plant_04 / planter_pot_clay**: more sill and counter greenery (scale 1)
- **Outside: street_lamp_02 (or street_lamp_01), fire_hydrant, water_manhole_cover, trashbag**: pavement and road outside the windows (scale 1 (street_lamp_01 is 3.9 m))
- **Outside: outdoor_table_chair_set_01 x2, modular_street_seating**: a small terrace on the pavement under the windows (scale 1)
- **Outside: tree_small_02 or island_tree_02**: street tree across the road (keep it from shading the windows) (4.6 m / 8.5 m: scale 1)
- **Outside: rollershutter_window_01/02, planter_box_02**: shopfronts and planters on the facade across the road (scale 1)
- **Outside: covered_car (the only vehicle in the catalog)**: parked across the road, optional (4.4 m: scale 1)

Not in any catalog (keep or build procedurally): entrance door with glass panel, radiator, coat stand, bicycles, espresso machine, napkin holder, sugar bowl.
