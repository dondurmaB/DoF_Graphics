# Workshop: substitution plan

Source: `output/dev/inventory/workshop.json` (167 groups: 112 procedural groups / 391 instances, 55 scanned groups / 125 instances), `scenes/workshop.py`, `_workshop_props.py`, catalogs in `output/dev/catalog/`. Poly Haven dimensions are W x D x H in meters (H last).

**Summary (procedural groups):** SUBSTITUTE 9 families (covering 63 groups, ~190 instances: all pegboard tools, the bench screwdriver pot, shelving bays, fluorescent battens, the work light), RETEXTURE 6, KEEP 13 families (structure, emitters, cables/chains, door, rag, parts cabinet). Scanned already in use: 55, 2 flagged. ADDITIONS: 24 (ceiling 8, far corner/behind car 9, walls/bench 7).

## Procedural groups

| object group | inst | tris | decision | asset id(s) | size check | notes |
|---|---|---|---|---|---|---|
| pegtool_pc_8..21 / po_9..16 (`.steel`) | 37 | 24k | SUBSTITUTE | `combination_wrench`, `adjustable_wrench` | scanned 0.25 m long / 0.25 m; procedural 0.12-0.25 m | Scale `combination_wrench` per slot length (0.5-1.0x) to keep the graded set; mix in `adjustable_wrench` for 1 in 4. Hang rotated so the long axis is vertical (both are modelled lying flat, Z-up). Keep `hook_pin` above each. |
| pegtool_adj_15 / adj_25 | 5 | 2k | SUBSTITUTE | `adjustable_wrench` | 0.25 m vs 0.15/0.25 m | Scale 0.6x / 1.0x. |
| pegtool_pl_16/18/20 (`.steel`, `.grip`) | 14 | 19k | SUBSTITUTE | `pliers`, `tongue_groove_pliers` | already pinned (used on bench) | Both exist in the scene; reuse for the board. Lose the seeded grip colours (scanned grips are fixed); acceptable, or override the grip part with a tinted copy. |
| pegtool_ham_26/30/36 (`.steel`, `.wood`) | 14 | 2k | SUBSTITUTE | `cross_pein_hammer`, `wooden_hammer_01` | 0.30 m / 0.28 m vs 0.26-0.36 m | Head up, handle down. |
| pegtool_saw_45/55 (`.steel`, `.wood`) | 8 | 31k | SUBSTITUTE | `handsaw_wood` | 0.63 m vs 0.45/0.55 m | Scale 0.75x/0.9x, or keep native 0.63 m on the widest slots (the board is 2.8 m wide). These are the most CG-looking tools at close range. |
| pegtool_hack (`.steel`, `.grip`) | 4 | 2k | SUBSTITUTE | `rusted_hacksaw` | 0.50 m vs ~0.45 m | Fits as is. |
| pegtool_sd_80..210 (`.handle`, `.steel`) | 36 | 18k | SUBSTITUTE | `screwdriver` (+ `flathead_screwdriver`) | 0.21 m vs 0.13-0.29 m overall | Scale 0.6-1.3x for the graded rack. Handles lose seeded colours; fine. |
| sd_100 (`.handle`, `.steel`) in the bench pot | 12 | 4k | SUBSTITUTE | `screwdriver`, `flathead_screwdriver`, `measuring_tape_01` | 0.21 m | Stand 4-5 in the pot, randomised tilt. |
| coffee_can (`.can`, `.handle`) | 2 | 0.7k | SUBSTITUTE | `cleaner_tin_01` (or `can_rusted`, already used) | 0.09 x 0.09 x 0.13 m vs r 0.055 x 0.15 m | Same size; reads as a real reused tin. |
| shelving_bay (`.post`, `.shelf`) | 8 | 1.4k | SUBSTITUTE | `steel_frame_shelves_01` (x2) + `worn_metal_rack` (x1) or `steel_frame_shelves_03` | 1.10 x 0.50 x 2.14 m vs bay 1.0 x 0.45 x 2.2 m | Levels differ from `SHELF_LEVELS`; read shelf heights from the model (ply_extent on the shelf part, or measure once and hardcode) and feed them to `fill_shelf`. `worn_metal_rack` (0.92 x 0.60 x 1.90) adds wear variety. |
| batten (`.housing`) + `rectangle:emitter_backing` | 10 + 10 | 0.6k | SUBSTITUTE housing only | `mounted_fluorescent_lights` | 0.91 x 0.65 x 0.04 m vs 0.72 m tube emitter | Scanned fixture replaces the black box housing; keep our rectangle emitter, resized/positioned to the scanned tube positions (keep tube parts non-emissive or hide them behind the emitter). If the fixture is a 4-tube panel, use it 1:1 and drop chains. Check orientation (it may be modelled flat for ceiling mount: perfect, mount directly under the purlins). |
| work_light (`.cage`, `.body`) + wl_cable + sphere emitter | 3 | 2k | SUBSTITUTE | `pull_chain_light_socket` + `lightbulb_01`, or `industrial_pipe_lamp` | 0.11 x 0.11 x 0.17 m / 0.06 x 0.06 x 0.10 m | A bare bulb on a pull-chain socket hanging on its cable over the car; keep the sphere emitter inside the scanned bulb glass (glass must not block NEE: give the bulb glass a `null`/thin override or drop the glass part). Alternative `portable_searchlight` clipped to the lift. |
| chain_short (hanger chains) | 20 | 53k | KEEP (remove if housings become ceiling-mounted) | — | — | No scanned chain. If `mounted_fluorescent_lights` mounts flat, delete these. |
| chain_long + chain_hook | 2 | 10k | KEEP | — | — | Hoist chain over the bench; no scanned chain block (`overhead_crane` is 12.5 m, too big for an 8 m room). |
| hook_pin | 96 | 6k | KEEP | — | — | Pegboard hooks; tiny, fine. |
| cable_a/b/c, wl_cable | 4 | 1.3k | KEEP + ADD | `modular_electric_cables` (add) | 2.08 x 0.21 x 0.76 m | Procedural drooping cables are fine; add the scanned wall-run cable bundle (see additions). |
| parts_cabinet (`.frame`, `.fronts`, `.pulls`) | 3 | 1.6k | KEEP + RETEXTURE frame | — | — | No small parts-bin cabinet in catalog (`drawer_cabinet` is 1.9 m tall furniture). Frame -> `blue_metal_plate`; fronts keep yellow. |
| rag | 1 | 1.4k | KEEP + RETEXTURE | ambientCG `Fabric0xx` (oily cotton) | — | No scanned rag; fabric texture on the procedural cloth. |
| box_pegboard | 1 | 12 | KEEP | — | — | Procedural pegboard texture works. |
| bench top/lower/legs/rails | 10 | 0.1k | KEEP (already textured) | `brown_planks_04`, `blue_metal_plate` | — | A heavy custom bench; no scanned workbench of this size. |
| ibeam | 5 | 0.2k | RETEXTURE | `rusty_painted_metal` or `green_metal_rust` | — | Currently flat `paint_grey`. |
| cube:paint_black (window frames, 18) | 18 | 0.2k | RETEXTURE | `rusty_metal_02` / ambientCG `Metal0xx` painted | — | Steel window frames. |
| cube:paint_grey (door guides, hood, bottom rail, 4) + drip_tray | 5 | 0.1k | RETEXTURE | `rusty_painted_metal` | — | Roller-door guides/hood box. |
| door_slats | 1 | 0.2k | KEEP (already `painted_metal_shutter`) | (`rollershutter_door` 3.08 x 0.30 x 2.40) | width matches the 3.0 m opening | The scanned door is fully closed and 2.4 m tall; ours is half-open. Keep procedural unless the opening is raised; then `rollershutter_door` translated up is a clean swap. |
| walls / floor / ceiling quads, reveals, jambs, lintel | 17 | — | KEEP (already scanned textures) | `whitewashed_brick`, `concrete_wall_007`, `concrete_layers_02`, `garage_floor`, `corrugated_iron_03` | — | Done. |
| rectangle:yard / ground / brick_far | 3 | — | RETEXTURE + ADD | `asphalt_floor` (done); add props outside | — | The yard reads flat and over-bright: lower ground albedo, add scanned yard props (see additions). |
| rectangle:mat_rubber | 1 | 2 | RETEXTURE | ambientCG rubber mat (`Rubber00x`) | — | Anti-fatigue mat in front of the bench. |
| sphere emitters (lamp bulbs) | 3 | 0 | KEEP | — | — | Emitters. |

## Scanned models already used: review

| asset | inst | flag |
|---|---|---|
| `exterior_aircon_unit` | 4 | Placed outside at z = 13.6 (line 227), fine as yard dressing; make sure it is not visible as indoor clutter. OK. |
| `covered_car` | 2 shapes | Fills the frame in `along`/`door` views (builder's own note). Move it 0.5 m toward the right wall or turn 15 degrees to open the floor; not a substitution issue. |
| `compost_bag_02`, `garden_hose_wall_mounted_01` | 1 each | Garden items, plausible in a garage. OK. |
| all others | — | Scale and context look right (tools, cans, carts, generator, barrels, crates, lamps). |

## Big surfaces (textures)

| surface | now | proposed |
|---|---|---|
| floor | `garage_floor` + oil mask | keep |
| back / left / right walls | `whitewashed_brick`, `concrete_wall_007`, `concrete_layers_02` | keep |
| ceiling (profiled roof) | `corrugated_iron_03` | keep; add purlins/trusses below (additions) |
| I-beams | flat grey paint | `rusty_painted_metal` |
| window frames, door guides | flat paint | `rusty_metal_02` / `rusty_painted_metal` |
| yard | `asphalt_floor` | keep, darker tint (`tint=(0.6,0.6,0.6)`) |
| rubber mat | flat | ambientCG `Rubber004`-type |
| rag | flat colour | ambientCG `Fabric0xx` |

## Additions (all in the Poly Haven catalog)

Ceiling (fixes the bare upward views):
1. `modular_airduct_circular_01` (4.17 x 1.46 x 1.32) run along one side under the roof.
2. `modular_industrial_pipes_01` (1.40 x 0.31 x 1.95) and/or `modular_pipes` (4.19 x 0.46 x 2.43) along the back wall top, painted pipes.
3. `modular_electric_cables` (2.08 x 0.21 x 0.76) wall-to-ceiling cable runs.
4. `ceiling_fan` (1.46 x 1.46 x 0.52) over the car.
5. `pull_chain_light_socket` + `lightbulb_01` near the door.
6. `caged_hanging_light` or `industrial_pipe_lamp` over the left side.
7. Procedural purlins/joists (rectangular steel sections between the I-beams, `rusty_painted_metal`) at 1 m spacing: no scanned truss exists; these are cheap boxes.
8. `mounted_fluorescent_lights` (see substitution) mounted flat on the purlins.

Far corner and behind the car:
9. `old_military_compressor` (0.60 x 1.68 x 1.18): a real air compressor, back-right corner.
10. `steel_frame_shelves_03` (2.35 x 0.72 x 2.31) or `worn_metal_rack` on the back wall right of the bench, filled via `fill_shelf`.
11. `small_lpg_tank` (0.41 x 0.41 x 0.64) next to `propane_tank`.
12. `wooden_military_crate` (1.24 x 0.52 x 0.46) / `wooden_crate_02` (1.17 x 0.53 x 0.46) stacked behind the car.
13. `plastic_broom` + `dustpan` leaning in the corner.
14. `wooden_ladder_02` (0.99 x 0.62 x 1.72) step ladder folded against the far wall.
15. `metal_trash_can` (check: 1.85 x 0.56 x 0.91 reads as a wheelie/skip bin; use at the door).
16. `rusted_spade_01`, `picke_dirty_01` leaning by the door.
17. `small_oil_can_01`, `retro_multimeter` on the tool cart / bench.

Walls and bench:
18. `wall_clock` (0.32 x 0.05 x 0.32) above the door.
19. `dartboard` (0.45 x 0.04 x 0.45) on the right wall.
20. `fire_alarm` (0.10 x 0.03 x 0.14) by the door; `utility_box_01` (0.52 x 0.43 x 1.12) as the fuse/distribution board on the left wall.
21. `boombox` (0.72 x 0.19 x 0.47) or `vintage_radio_transceiver` on a shelf.
22. `vintage_hand_drill`, `propane_torch_02`, `ammo_box` (used as a fastener box) on the bench.
23. `old_gas_mask` / `rubber_boots` hung or on the floor by the door.
24. Outside the door (yard): `utility_box_02`, `modular_chainlink_fence` (8.12 m) along the far side, `street_lamp_02`, `rusted_wheel_rim_02`, `wooden_barrels_01`; this fixes the flat bright yard.

Budget: current 1.26M triangles; substitutions roughly neutral (tools gain ~60k), additions ~0.6M (fence 89k, compressor 79k, pipes 47k, airduct 14k). Well under 8M.
