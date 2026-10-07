# Desk: substitution map

Sources: `output/dev/inventory/desk.json` (78 groups, all procedural), `scenes/desk.py`, Poly Haven (PH) models and textures, ambientCG (ACG) materials.
Poly Haven `dimensions` are read as [width, depth, height] in m (verified against known objects such as street_lamp_01 at 3.87 m tall).

**Summary: 13 SUBSTITUTE, 14 RETEXTURE, 26 KEEP (plus 25 sub-parts that follow their parent's decision), 21 ADDITIONS (12 room, 9 outside).**

## Object groups

| object group | inst | tris | decision | asset id(s) | size check | notes |
|---|---|---|---|---|---|---|
| notebook.pages / .edges / .cover | 1 | 7.5k | KEEP | | | Procedural handwriting, diagram and coffee ring are the point of this scene |
| sheet (report, table, red-pen) | 3 | 3.1k | KEEP | | | Procedural printed text |
| cube:paper_plain (paper stack) | 3 | 36 | KEEP | | | Sits under the text sheets |
| rectangle:sticky_0..3 | 5 | 10 | KEEP | | | Handwritten notes |
| rectangle:pinboard | 1 | 2 | KEEP | | | Procedural pinned notes; optionally retexture the cork with ACG cork |
| plant_floor (leaves, stems, pot, soil) | 1 | 11k | SUBSTITUTE | potted_plant_01 | 0.59x0.64x1.35 m; a floor plant is 1.0-1.4 m, so it fits as is | Scanned leaves with alpha and translucency |
| plant_shelf (leaves, stems, pot, soil) | 2 | 13k | SUBSTITUTE | potted_plant_04 | 0.17x0.19x0.27 m; fits a 0.3 m shelf gap | Use 2 instances, yaw varied by seed |
| plant_side (on side cabinet) | 1 | 9.4k | SUBSTITUTE | potted_plant_02 | 0.70x0.66x0.84 m; scale 0.75 to 0.63 m on a 0.8 m cabinet | |
| plant_desk_1 | 1 | 7.4k | SUBSTITUTE | potted_plant_04 | 0.27 m; desk plant size | Different yaw and tint from the shelf ones |
| office_chair (fabric, chrome, plastic) | 1 | 7.4k | SUBSTITUTE | gallinera_chair (alt: dining_chair_02) | 0.58x0.60x1.03 m; seat height matches a 0.75 m desk | No swivel office chair in PH; a wooden chair suits a home study |
| lamp_arms / lamp_base / lamp_shade_out | 1 | 2.1k | SUBSTITUTE | desk_lamp_arm_01 | 0.62x0.41x0.88 m; a real architect lamp | Keep the procedural emitter (lamp_shade_in, sphere bulb) inside its shade |
| lamp_shade_in [emitter], sphere bulb [emitter] | 2 | 0.4k | KEEP | | | Emitters; reposition into the scanned lamp's shade |
| jar (glass, lid) on bookcase | 1 | 6k | SUBSTITUTE | brass_vase_03 or ceramic_vase_03 | 0.08x0.08x0.20 / 0.41 m; fits a shelf | No scanned glass jar exists |
| cube:black (picture frames, 2 artworks) | 8 | 96 | SUBSTITUTE | hanging_picture_frame_01, hanging_picture_frame_02 | 0.59x0.84 and 0.75x0.50 m vs art_a 0.6x0.8 and art_b 0.75x0.56: near exact | Scanned frames with their own art. Drop rect art_a/art_b, or keep them as the inset |
| rectangle:art_a / art_b | 2 | 4 | SUBSTITUTE | (with the frames above) | | |
| books (about 300 spines on the bookcase) | 1 | 3.2k | KEEP + mix | book_encyclopedia_set_01, decorative_book_set_01 | 0.55 m and 2.51 m runs; place 1-2 runs on shelves | Procedural spines are convincing at shelf distance; scanned sets add variety and depth |
| cube:book_0 / book_1 (desk book stack) | 3 | 36 | RETEXTURE | ACG Fabric0xx (book cloth), Leather0xx | | Boxes; real cloth or leather covers help close-ups |
| mug.mug / mug.coffee | 1 | 2.9k | KEEP | | | No single mug in PH (only tea_set_01); the procedural mug reads fine |
| headphones (cushion, plastic, metal) | 1 | 4.2k | KEEP | | | No headphones in PH |
| pen_cup, ballpoint (6), pencil (4) | 11 | 3k | KEEP | | | No pens in PH; small and fine-edged, which is the point |
| mouse, keyboard | 2 | 2k | KEEP | | | No match. The keyboard legends are procedural detail |
| monitor (body, stand, logo) | 1 | 0.7k | KEEP | | | The code-editor screen is procedural text. classic_laptop exists, but the monitor is the subject |
| cube:case_wood (bookcase carcass, cabinet dividers, jar lid) | 12 | 144 | RETEXTURE | PH oak_veneer or american_walnut_veneer | | The custom 3 m bookcase holds 300 books, so keep the geometry and use real wood |
| cube:desk_dark (desk panels, pedestal, modesty panel, pinboard frame, side cabinet) | 8 | 96 | RETEXTURE | PH black_walnut_veneer_01 | | Flat dark colour now |
| box_desk_top / cube:desk_wood (drawer fronts) | 4 | 48 | RETEXTURE | PH american_walnut_veneer (or european_walnut_veneer_04) | | Procedural wood_top now; PH veneer has a normal map |
| box_mat (leather desk mat) | 1 | 12 | RETEXTURE | PH brown_leather (alt ACG Leather0xx) | | |
| cube:brass (drawer pulls) | 3 | 36 | KEEP | | | Tiny; the roughconductor already reads as brass |
| cube:white_paint (skirting, window frame, mullions, sill) | 11 | 132 | RETEXTURE | ACG PaintedWood0xx (white) | | Painted-wood grain and normal help in close views of the window |
| cube:wall (left wall around window) | 4 | 48 | RETEXTURE | PH painted_plaster_wall (tint to match) | | |
| rectangle:wall (right and front walls) | 2 | 4 | RETEXTURE | PH painted_plaster_wall / beige_wall_001 | | Real plaster unevenness instead of flat procedural |
| rectangle:wall_back (dark wall behind bookcase) | 1 | 2 | RETEXTURE | PH painted_plaster_wall, tint dark green | | |
| rectangle:ceiling | 1 | 2 | RETEXTURE | PH white_plaster_02 (alt white_plaster_rough_01) | | Upward views hit it |
| rectangle:floor | 1 | 2 | RETEXTURE | PH herringbone_parquet (alt old_wood_floor) | | |
| rectangle:rug | 1 | 2 | KEEP | (opt. ACG Carpet0xx normal) | | Procedural woven pattern; no scanned rugs |
| rectangle:emitter_backing (4 ceiling panels, 1 shelf strip) | 5 | 10 | KEEP | | | Emitters. Add visible fixtures (below) |
| plant soils | 5 | 0.3k | (with plants) | | | |
| rectangle:ground (outside) | 1 | 2 | REPLACE | see outside | | |
| cube:hedge (outside) | 1 | 12 | SUBSTITUTE | shrub_02 x 2-3 | 5.3x2.1x2.1 m each | A real hedge line at 6-8 m |

## Big-surface textures

| surface | texture | source | scale note |
|---|---|---|---|
| floor | herringbone_parquet | PH | real-world scale via W.surface (uv in m) |
| walls (3 light) | painted_plaster_wall or beige_wall_001 | PH | tint per seed for colour variety |
| back wall | painted_plaster_wall, tint deep green | PH | |
| ceiling | white_plaster_02 | PH | |
| desk top and drawer fronts | american_walnut_veneer | PH | grain along the desk length |
| desk carcass and side cabinet | black_walnut_veneer_01 | PH | |
| bookcase | oak_veneer | PH | |
| skirting and window joinery | ACG PaintedWood (white) | ACG | needs the ambientCG loader |
| desk mat | brown_leather | PH | |
| outside: front garden | leafy_grass (alt ACG Grass004) | PH | |
| outside: pavement | brick_pavement or herringbone_pavement | PH | |
| outside: road | asphalt_02 | PH | |
| outside: facades across the road | brick_wall_006, white_stucco, plastered_stone_wall (one per building) | PH | on procedural facade boxes with recessed windows |

## Additions

**Room** (visible from random poses: corners, ceiling, the other walls)
1. ArmChair_01 or mid_century_lounge_chair in the empty corner, with throw_pillows_01 and side_table_01 (0.55 m).
2. modern_ceiling_lamp_01 as the visible housing over the existing ceiling emitter. Upward views currently see bare panels.
3. ceiling_fan (1.46 m) centred on the ceiling, for upward views.
4. alarm_clock_01 or mantel_clock_01 on a bookshelf; vintage_telephone_wall_clock on the right wall.
5. standing_picture_frame_01 and _02 on the desk and a shelf.
6. wicker_basket_01 under the desk as a wastepaper basket; wicker_basket_02 on the cabinet.
7. cardboard_box_01 (scale 0.7) on top of the bookcase.
8. antique_ceramic_vase_01, ceramic_vase_01/04 and brass_vase_04 spread over the shelves.
9. Camera_01 and cassette_player on shelves as personal objects.
10. binder_notebook and clipboard on the side cabinet.
11. A door in the front wall: procedural (frame plus panel door) with the oak veneer and a brass handle. No PH door exists.
12. Curtains beside the window: procedural cloth sweep with ACG Fabric. No PH curtains exist.

**Outside the window** (window in the left wall, x = -2.4 m, looking -x; keep geometry at real distances so depth stays valid)
1. 0-4 m: a front garden in leafy_grass, with a low brick wall (procedural, PH brick_wall_006) and planter_box_02.
2. Hedge line at about 6 m: shrub_02 x 2-3, replacing the hedge cube.
3. Street tree at about 5 m: tree_small_02 (4.6 m, about 2M tris; scale 1.3 is fine) or island_tree_02 (8.5 m). Keeping one tree holds the triangle budget.
4. 4-6 m: pavement in brick_pavement; 6-13 m: road in asphalt_02 with kerbs.
5. street_lamp_01 (3.87 m) at the kerb; fire_hydrant; utility_box_01.
6. painted_wooden_bench on the pavement.
7. Across the road at 16-20 m: 2-3 terraced facades (procedural boxes with recessed windows, sills, a door, a roofline), textured brick_wall_006, white_stucco and plastered_stone_wall. Reuse the night_street facade generator if it gets promoted.
8. Vehicles: covered_car (4.4 m) parked at the kerb is the only scanned vehicle. Reusing night_street's procedural car generator would give 1-2 more. Gap: PH has no bicycles or uncovered cars.
9. Sky: the existing env_kit clear sky (correctly oriented), already lighting through the window.

**Triangle estimate:** about 2.1M from the tree, 0.5M from the plants and other scans, plus the existing scene: about 3M of the 8M budget.
