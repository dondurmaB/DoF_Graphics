# terminal — substitution plan

**Summary:** 135 inventory groups → **31 SUBSTITUTE** (8 potted trees = 24 groups, 7 kiosk jars), **38 RETEXTURE** (structure, floors, walls, furniture, luggage shells, skyline), **66 KEEP** (60 emissive shop/poster/board/sign/clock quads, cables, stanchions, trolleys, clock bezels, info pole). Plus **14 additions** and a skyline rebuild.

Notes on sizes: Poly Haven `dimensions` are listed as [width, depth, height] (checked against converted `potted_plant_01` / `tree_small_02`). Several entries (pachira_aquatica_01, metal_trash_can, vintage_suitcase) are **variant sheets** (several objects side by side), so cut one variant out by part/bounds as the greenhouse did. "ACG" = ambientCG material (needs the loader's ambientCG support); "PH" = Poly Haven texture (works today via `W.surface`).

## Objects

| object group | instances | tris | decision | asset id(s) | size check | notes |
|---|---|---|---|---|---|---|
| tree_0..7 .leaf / .trunk / .pot (24 groups) | 8 trees | 93.4k | SUBSTITUTE | 2× `tree_small_02` in large planters; 6× `pachira_aquatica_01` (one variant each) or `potted_plant_01`, in `planter_box_03` / `ceramic_pot` | procedural 3.0–3.5 m. tree_small_02 4.56 m, use at 0.7–0.75 (3.2–3.4 m). pachira variants about 1.9 m. potted_plant_01 1.35 m | 2.06M tris per tree_small_02, so 2 of them is about 4.1M of the 8M budget; the rest stay small. Leaves become translucent automatically |
| cube:case_0..6 (kiosk jars) | 7 | 84 | SUBSTITUTE | `wine_bottles_01` (cut single bottles), `tea_set_01` on the counter, or drop the jars and put `CoffeeCart_01` behind the counter | jars are 0.14×0.34 m | coloured cubes read as CG on any close kiosk view |
| rib | 16 | 56.7k | RETEXTURE | ACG `PaintedMetal` (white, satin) | — | truss geometry is right; the flat colour is what reads as CG |
| column | 32 | 23.0k | RETEXTURE | ACG `PaintedMetal` (white) | — | keep geometry |
| pier, shell_0, shell_1 (vault) | 32 + 2 + 2 | 2.7k | RETEXTURE | PH `painted_concrete` or ACG `PaintedPlaster` (light) | — | subtle roller texture and slight stains on the vault |
| bar_5/10/15 (glazing bars) | 6 | 192 | RETEXTURE | same as rib | — | |
| balu, balu_bridge, cube:steel_dark (handrails, kick, fixtures, roof) | 3 + 2 + 39 | 32k | RETEXTURE | ACG `Metal` (dark brushed) / `PaintedMetal` (charcoal) | — | |
| cube:steel_white (ties) | 16 | 192 | RETEXTURE | ACG `PaintedMetal` (white) | — | |
| cube:frame (mullions, sign and board boxes, girder, belts, monitors) | 120 | 1.4k | RETEXTURE | ACG `PaintedMetal` (dark) | — | |
| cube:plaster (gallery walls, piers, lintels, shop backs, aisle ends, kiosk back) | 127 | 1.5k | RETEXTURE | PH `painted_plaster_wall` / ACG `PaintedPlaster` (off-white) | — | biggest close-range CG tell after the floor |
| cube:ceiling (aisle roof undersides) | 2 | 24 | RETEXTURE | ACG `PaintedPlaster` (white) or acoustic `Tiles` | — | seen in every upward aisle pose |
| box_floor (nave floor) | 1 | 12 | RETEXTURE | PH `terrazzo_tiles` (2 m) or `granite_tile_02` (1.88 m); keep the clearcoat polish | 2 m tile at real scale over a 22 × 90 m floor | replaces the procedural floor; keep the stripe or inlay as a second material if wanted |
| cube:gallery_floor (galleries, bridge deck) | 4 | 48 | RETEXTURE | PH `granite_tile_02` | — | |
| cube:ground_out (outside ground) | 1 | 12 | RETEXTURE | PH `concrete_pavement_03` plus `asphalt_02` road strips | — | ties into the skyline street level (below) |
| cube:stone_dark (counter tops, kiosk top) | 8 | 96 | RETEXTURE | ACG `Granite00xB` (dark) or PH `grey_cartago_03` | — | |
| cube:laminate (check-in counters, kiosk) | 8 | 96 | RETEXTURE | ACG `Plastic` (white satin) or a wood veneer from ACG `Wood` | — | |
| info_desk | 1 | 360 | RETEXTURE | body ACG `Plastic` / wood veneer, top `grey_cartago_03` | custom round desk, no catalog match | |
| cube:rubber (belt rubber), escalator.rubber | 7 + 2 | 212 | RETEXTURE | ACG `Rubber00x` | — | |
| escalator.metal / escalator.steps | 2 + 2 | 984 | RETEXTURE | ACG `Metal` (brushed) / ACG `DiamondPlate` grooved for the steps | — | no escalator in the catalogs; keep geometry |
| bench4.frame / bench4.seat | 32 + 32 | 8.3k | RETEXTURE (KEEP geometry) | frame ACG `Metal` (brushed); seat ACG `Leather` (dark) or `Plastic` | 4-seat beam bench 2.2 m | no beam seating in Poly Haven (`modular_street_seating` is an outdoor bench, wrong for an airside hall) |
| case_a/b/c .shell (suitcases) | 10 + 13 + 2 | 15.6k | RETEXTURE | ACG `Plastic` (hard shell) tinted with the existing PAINTS palette | 0.48–0.55 m wide, matches carry-on and check-in sizes | Poly Haven has only `vintage_suitcase`, wrong era. Keep procedural |
| case_a/b/c .trim, duffel.trim, backpack.trim | 10 + 13 + 2 + 6 + 13 | 12.5k | RETEXTURE | ACG `Plastic` (black) / `Leather` | — | |
| duffel.shell, backpack.shell | 6 + 13 | 8.9k | RETEXTURE | PH `denim_fabric_04` / ACG `Fabric` (nylon) | — | |
| box_tower (skyline, 73 towers) | 73 | 876 | RETEXTURE + rebuild geometry | ACG `Facade018A–C`, `019A–C`, `020A–C` (skyscraper day); `Facade001–006` (reflective curtain wall) | floors about 3.6 m; scale uv so a floor is 3.6 m | see the skyline section |
| trolley.frame / .wheel | 8 + 8 | 5.6k | KEEP | — | 1.55 m nested trolleys | airport trolley not in the catalogs; chrome and rubber read fine |
| stanchion, tape | 12 + 11 | 2.6k | KEEP | — | 1.02 m posts | convincing as is |
| cable_unit | 96 | 1.9k | KEEP | — | — | thin hangers, good DoF edges |
| bezel | 2 | 640 | KEEP | — | — | clock ring |
| info_pole | 1 | 32 | KEEP | — | — | |
| rectangle:backing [emitter] (boards, signs, clocks, screens) | 197 | 394 | KEEP | — | — | legible procedural text: the fine-detail target |
| rectangle:shop_±1_k, poster_±1_k [emitter] | 58 | 116 | KEEP | — | — | shopfront and poster atlases are the scene's identity. Could add depth later (recessed glazing, scanned shelf goods) |

## Big surfaces (texture table)

| surface | current | texture | real scale |
|---|---|---|---|
| nave floor | procedural stripes | PH `terrazzo_tiles` (or `granite_tile_02`), polished | 2.0 m (1.88 m) |
| gallery floors, bridge deck | flat grey | PH `granite_tile_02` | 1.88 m |
| walls, piers, lintels, shop backs | procedural plaster | PH `painted_plaster_wall` / ACG `PaintedPlaster` | about 2 m |
| vault shell and piers | flat diffuse | PH `painted_concrete` | about 2 m |
| aisle ceilings | flat diffuse | ACG `PaintedPlaster` (white) | about 2 m |
| ribs, columns, ties, glazing bars | flat white | ACG `PaintedMetal` (white satin) | 1–2 m |
| dark steel (rails, frames, girder) | flat black | ACG `PaintedMetal` / `Metal` (dark) | 1–2 m |
| counters / kiosk | flat laminate, black stone | ACG `Plastic` white; `Granite` dark tops | 1 m |
| outside ground | env_kit ground | PH `concrete_pavement_03` + `asphalt_02` | 2–3 m |
| skyline facades | flat tower_0/1/2 colours | ACG `Facade018–020` (opaque), `Facade001–006` (glass) | floor = 3.6 m |

## Skyline fix (tiles 6 and 22 in the audit)

1. **Geometry.**
   - Replace each `box_tower` with a tower mesh: a 2–4 floor podium (wider footprint, glazed retail base), a shaft with 0–2 setbacks, and a crown (mechanical penthouse, parapet).
   - Add vertical mullion fins or spandrel bands as real geometry on towers within about 150 m, so close frames have depth edges.
   - Add rooftop clutter from Poly Haven `exterior_aircon_unit` and `modular_airduct_rectangular_01` on the near towers.
2. **Materials.**
   - About 60% opaque office facades: ACG `Facade018A–C`, `019A–C`, `020A–C`, uv in meters with a floor at 3.6 m, varied with `tint=`.
   - About 40% curtain-wall towers: ACG `Facade001–006` with low roughness (about 0.08) and specular 0.5, so they genuinely reflect the sky and neighbours instead of reading as a painted grid.
3. **Street level** (visible through the open end):
   - Road and pavement: PH `asphalt_02` and `concrete_pavement_03`.
   - Kerbs: a procedural box with PH `granite_tile` or concrete.
   - Street furniture: `street_lamp_01` every 25 m, `island_tree_02` or `searsia_lucida` street trees, `fire_hydrant`, `concrete_road_barrier`, `water_manhole_cover`, `modular_street_seating`.
   - Distant trees can be the same few models instanced.
4. **Composition.** Keep the nearest facade at 40 m or more from the hall's open end. Close the gaps between towers with a second row, so a long lens never frames empty ground between boxes.

## Additions (catalog-backed; quality over quantity)

1. `metal_trash_can`: one variant, about 0.9 m tall, by every second column and at the seating ends.
2. `WetFloorSign_01` (0.63 m): one or two on the polished floor.
3. `korean_fire_extinguisher_01` (0.66 m): on column bases or wall brackets.
4. `security_camera_01` / `security_camera_02`: on columns and under the gallery slabs, which helps the upward views.
5. `fire_alarm`: on the gallery and shop-back walls at 1.4 m.
6. `CoffeeCart_01` (1.72 m): next to the kiosk or seating, as a second coffee point.
7. `standing_chalkboard_01` (1.51 m): kiosk menu board.
8. `bar_chair_round_01` (0.75 m) ×3–4: at a kiosk counter ledge.
9. `planter_box_01` / `planter_box_02` with `shrub_02` or `potted_plant_02`: low planters lining the check-in queue and seating.
10. `potted_plant_02` (0.84 m) / `potted_plant_04` (0.27 m): on counters and the info desk.
11. `modern_arm_chair_01` / `mid_century_lounge_chair`: a small lounge corner under the LOUNGE sign.
12. `cardboard_box_01` and `hand_truck` (1.4 m): a service corner by the kiosk back.
13. `plastic_broom` / `wooden_broom`: a cleaner's corner (one only).
14. Street-level set beyond the open end: see the skyline fix.

Not in the catalogs, so stay procedural: airport trolleys, beam seating, escalators, departure boards and signage, modern suitcases and backpacks, vending machines and check-in kiosks. Vending machines and kiosks could be simple procedural boxes with an emissive screen and a real texture if wanted.
