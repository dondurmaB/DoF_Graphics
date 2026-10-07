# Museum: substitution plan

**Summary:** 145 procedural object groups: SUBSTITUTE 13, RETEXTURE 52, KEEP 80 (KEEP is mostly the 18 spot mounts, 30 painting labels/canvas sides and structural gaps). Plus 19 catalogued additions in 8 groups. All sources CC0: Poly Haven models/textures, ambientCG materials, Smithsonian Open Access 3D (records checked: `metadata_usage.access = CC0`).

Loader work this plan needs (shared side): (1) Smithsonian OBJ zips in `web_assets` (download zip, unzip obj+mtl+jpg, `map_Kd` -> bitmap; the 150k-4096 variants are 1-10 MB each; bounds must be checked because the records rarely state units), (2) ambientCG materials (zip of Color/Roughness/NormalGL jpg, same as `surface()`), (3) CC0 painting images from Smithsonian Open Access for canvas faces. Poly Haven dimensions below are the API's mm values /1000.

## Objects

| object group | instances | tris | decision | asset id(s) / URL | size check | notes |
|---|---|---|---|---|---|---|
| `ammonite_big` | 1 | 38,016 | SUBSTITUTE | Smithsonian *Psychopyge elegans* trilobite (NMNH, CC0) https://3d-api.si.edu/content/document/3d_package:e4518faa-9603-4e1b-8375-e62d1fda079b/resources/Psychopyge_elegans_Termier_-_Termier__1950-150k-4096-obj_std.zip | proc 0.20 m; scan ~0.1 m long incl. matrix: check bounds, scale to catalogue (no dims in record) | real fossil on its own rock matrix; replaces ammonite+slab pair in the fossils flat case (glass 0.28 m: fits) |
| `ammonite_small` | 1 | 38,016 | KEEP | - | - | no CC0 scanned ammonite in Poly Haven or Smithsonian zips; the procedural spiral on the acrylic stand reads as a cast; retexture `fossil` with a real stone (ambientCG Rock/Ground) albedo |
| `spot_head.can` | 18 | 36,000 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `trefoil` | 1 | 17,280 | SUBSTITUTE | Poly Haven `bronze_ray_statue` | 0.71 x 0.54 x 0.59 m; low plinth 0.8 x 0.65, h 0.45 | modern bronze for the 'knot' slot; keep the steel/patina variation off (scan has its own PBR) |
| `necklace.gold` | 1 | 11,152 | KEEP | - | - | no CC0 scanned jewellery found (Smithsonian 'jewelry'/'necklace' return shells); procedural gold (roughconductor Au) + dielectric gems read right at vitrine scale |
| `muse` | 1 | 9,024 | SUBSTITUTE | Poly Haven `marble_bust_01` | 0.27 x 0.30 x 0.51 m (mm dims /1000); plinth 0.46 x 1.08: top at ~1.65 m, eye level | keep the marble socle cube under it |
| `urn.body` | 1 | 6,000 | SUBSTITUTE | Poly Haven `antique_ceramic_vase_01` | 0.26 x 0.26 x 0.44 m; on plinth 0.56 x 0.86 | real antique glaze and wear |
| `spot_head.lens` | 18 | 3,456 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `ring.gold` | 3 | 3,228 | KEEP | - | - | no CC0 scanned jewellery found (Smithsonian 'jewelry'/'necklace' return shells); procedural gold (roughconductor Au) + dielectric gems read right at vitrine scale |
| `coin_tray.gold` | 1 | 3,200 | KEEP | - | - | no CC0 scanned coins in either catalog; procedural relief coins are fine at tray scale |
| `stanchion` | 3 | 3,168 | KEEP | - | - | no scanned stanchion/rope post; brass principled is fine |
| `amphora.body` | 1 | 2,880 | SUBSTITUTE | Smithsonian Baluster vase F1980.191 (NMAA, porcelain, CC0) https://3d-api.si.edu/content/document/3d_package:d8c6393a-4ebc-11ea-b77f-2e728ce88125/resources/f1980_191-150k-obj.zip | proc 0.44 m tall; real 46.3 x 17.2 cm; tall-case glass 0.55 m: fits | blue-and-white porcelain scan with 4k texture; no amphora exists in either catalog |
| `manuscript.pages` | 1 | 2,714 | KEEP | - | - | procedural parchment with lettering is the fine-text target; no scanned manuscript; boards could take `brown_leather` |
| `neck_stand` | 1 | 1,792 | RETEXTURE | Poly Haven `velour_velvet` (tint dark) | - | real velvet weave and sheen under the hood light |
| `bird.bronze` | 1 | 1,792 | SUBSTITUTE | Poly Haven `bronze_shark_statue` | 0.35 x 0.76 x 0.43 m; widen that plinth to 0.9 x 0.5, h 0.75 | drum_stone base absorbed; real cast bronze |
| `amphora.handles` | 1 | 1,640 | SUBSTITUTE | Smithsonian Baluster vase F1980.191 (NMAA, porcelain, CC0) https://3d-api.si.edu/content/document/3d_package:d8c6393a-4ebc-11ea-b77f-2e728ce88125/resources/f1980_191-150k-obj.zip | proc 0.44 m tall; real 46.3 x 17.2 cm; tall-case glass 0.55 m: fits | blue-and-white porcelain scan with 4k texture; no amphora exists in either catalog |
| `bench.cushion` | 1 | 1,572 | RETEXTURE | ambientCG `Leather0xx` (dark) or Poly Haven `brown_leather` | - | real leather grain; Ottoman_01 is too tall (0.62) to replace a gallery bench |
| `jug.body` | 1 | 1,536 | SUBSTITUTE | Smithsonian Colonoware pot (NMAAHC 2017.108.1.1, CC0) https://3d-api.si.edu/content/document/3d_package:79da3e3f-3ad7-41de-8956-e891d88a3c5f/resources/2017_108_1_1-ColonoPottery-150k-4096-obj_std.zip | proc 0.24 m; real size not in record: scale by bounds to ~0.20-0.25 m | earthenware with real surface; pairs with the vase |
| `bowl` | 1 | 1,296 | SUBSTITUTE | Smithsonian lidded incense burner (boshanlu, NMAA, bronze with inlay, CC0) https://3d-api.si.edu/content/document/3d_package:ce850625-2cf1-4c6f-9086-0d5845d9a664/resources/incense-burner-in-the-form-of-a-mountain-(boshanlu)-150k-4096-obj.zip | real 17.9 x 10 cm; fits beside the vase | replaces the plain celadon bowl with a real inlaid bronze |
| `rope_-10.` | 1 | 1,120 | RETEXTURE | Poly Haven `velour_velvet` (tint red) | - | velvet rope |
| `rope_-9.` | 1 | 1,120 | RETEXTURE | Poly Haven `velour_velvet` (tint red) | - | velvet rope |
| `ring.gem` | 3 | 864 | KEEP | - | - | no CC0 scanned jewellery found (Smithsonian 'jewelry'/'necklace' return shells); procedural gold (roughconductor Au) + dielectric gems read right at vitrine scale |
| `jug.handles` | 1 | 820 | SUBSTITUTE | Smithsonian Colonoware pot (NMAAHC 2017.108.1.1, CC0) https://3d-api.si.edu/content/document/3d_package:79da3e3f-3ad7-41de-8956-e891d88a3c5f/resources/2017_108_1_1-ColonoPottery-150k-4096-obj_std.zip | proc 0.24 m; real size not in record: scale by bounds to ~0.20-0.25 m | earthenware with real surface; pairs with the vase |
| `coin_tray.bronze` | 1 | 800 | KEEP | - | - | no CC0 scanned coins in either catalog; procedural relief coins are fine at tray scale |
| `coin_tray.silver` | 1 | 800 | KEEP | - | - | no CC0 scanned coins in either catalog; procedural relief coins are fine at tray scale |
| `ring_cone` | 3 | 768 | RETEXTURE | Poly Haven `velour_velvet` (tint dark) | - | real velvet weave and sheen under the hood light |
| `necklace.gem` | 1 | 624 | KEEP | - | - | no CC0 scanned jewellery found (Smithsonian 'jewelry'/'necklace' return shells); procedural gold (roughconductor Au) + dielectric gems read right at vitrine scale |
| `bird.stone` | 1 | 480 | SUBSTITUTE | Poly Haven `bronze_shark_statue` | 0.35 x 0.76 x 0.43 m; widen that plinth to 0.9 x 0.5, h 0.75 | drum_stone base absorbed; real cast bronze |
| `cube:case_metal` | 24 | 288 | RETEXTURE | ambientCG `Metal051A` (brushed bronze/brass) | - | edge trims of vitrines |
| `cube:paint_white` | 18 | 216 | RETEXTURE | ambientCG `PaintedPlaster001` (tint per palette, satin) | - | white painted MDF/plaster with roller texture; resize plinths to substituted sculptures (+0.1 m) |
| `cube:glass` | 18 | 216 | KEEP | - | - | vitrine glass: the dielectric is the point of this scene |
| `frame_gilt_1.30x1.` | 2 | 208 | RETEXTURE | ambientCG gold metal (`Metal034`/`Metal048A`, pick by preview) | - | larger gilt frames: scaling the scanned frame non-uniformly would distort its profile |
| `label_block` | 15 | 180 | KEEP | - | - | procedural label text (fine-text detail) |
| `spot_mount_0_-2.40_-0.85_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_1_-2.40_-5.15_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_2_-2.40_-8.25_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_3_-2.40_-10.55_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_4_2.40_-1.15_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_5_2.40_-5.15_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_6_2.40_-9.15_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_7_2.40_-10.40_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_8_0.00_-18.00_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_9_-1.90_-14.45_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_10_-1.90_-17.65_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_11_1.90_-14.45_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_12_1.90_-17.65_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_13_-2.40_-10.40_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_14_-2.40_-4.20_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_15_2.40_-4.20_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_16_-2.40_-9.10_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `spot_mount_17_-1.90_-15.30_4.` | 1 | 172 | KEEP | - | - | track spots: no scanned gallery track head in Poly Haven; procedural can + lens is accurate; body could take black powder-coat roughness noise |
| `frame_gilt_0.62x0.` | 1 | 104 | SUBSTITUTE | Poly Haven `fancy_picture_frame_02` (override its picture with the canvas texture) | 0.66 x 0.77 m vs slot 0.62 x 0.80: scale ~1:1 | real carved gilt frame on the small portrait slot |
| `frame_gilt_1.25x0.` | 1 | 104 | RETEXTURE | ambientCG gold metal (`Metal034`/`Metal048A`, pick by preview) | - | larger gilt frames: scaling the scanned frame non-uniformly would distort its profile |
| `frame_gilt_1.00x1.` | 1 | 104 | RETEXTURE | ambientCG gold metal (`Metal034`/`Metal048A`, pick by preview) | - | larger gilt frames: scaling the scanned frame non-uniformly would distort its profile |
| `bench.steel` | 1 | 84 | RETEXTURE | ambientCG `Metal009` (brushed) | - |  |
| `cube:case_body` | 6 | 72 | RETEXTURE | walnut: Poly Haven `american_walnut_veneer`; white: ambientCG `PaintedPlaster001` (tinted white) | - | fixes flat brown case bases seen close up (audit tile 21) |
| `cornice_14.00_5.` | 2 | 48 | RETEXTURE | ambientCG `PaintedPlaster001` (tint per palette, satin) | - | white painted MDF/plaster with roller texture; resize plinths to substituted sculptures (+0.1 m) |
| `cornice_7.60_4.` | 2 | 48 | RETEXTURE | ambientCG `PaintedPlaster001` (tint per palette, satin) | - | white painted MDF/plaster with roller texture; resize plinths to substituted sculptures (+0.1 m) |
| `cube:track_black` | 4 | 48 | KEEP | - | - | track rail |
| `cube:shadow_gap` | 4 | 48 | KEEP | - | - | structural shadow gap / emitter backing |
| `frame_oak_0.90x1.` | 1 | 40 | RETEXTURE | Poly Haven `natural_walnut_veneer` / `ash_veneer` | - |  |
| `frame_oak_1.50x1.` | 1 | 40 | RETEXTURE | Poly Haven `natural_walnut_veneer` / `ash_veneer` | - |  |
| `frame_black_1.00x1.` | 1 | 40 | KEEP | - | - | black satin frames are correct |
| `frame_oak_1.00x0.` | 1 | 40 | RETEXTURE | Poly Haven `natural_walnut_veneer` / `ash_veneer` | - |  |
| `frame_float_2.20x1.` | 1 | 40 | KEEP | - | - | black satin frames are correct |
| `frame_oak_1.10x1.` | 1 | 40 | RETEXTURE | Poly Haven `natural_walnut_veneer` / `ash_veneer` | - |  |
| `frame_black_1.20x0.` | 1 | 40 | KEEP | - | - | black satin frames are correct |
| `frame_black_0.90x1.` | 1 | 40 | KEEP | - | - | black satin frames are correct |
| `frame_black_2.40x1.` | 1 | 40 | KEEP | - | - | black satin frames are correct |
| `frame_float_1.00x1.` | 1 | 40 | KEEP | - | - | black satin frames are correct |
| `cube:wall_feature` | 3 | 36 | RETEXTURE | ambientCG `PaintedPlaster002` (tint teal/oxblood/navy) | - | feature wall around the doorway |
| `cube:trim_stone` | 3 | 36 | RETEXTURE | ambientCG `Marble006` or limestone `Tiles142` | - | door threshold / trim |
| `skirt_7.` | 3 | 36 | KEEP | - | - | dark satin skirting; small |
| `rectangle:ceiling` | 16 | 32 | RETEXTURE | ambientCG `PaintedPlaster001` (white) | - |  |
| `skirt_14.` | 2 | 24 | KEEP | - | - | dark satin skirting; small |
| `cornice_8.00_5.` | 1 | 24 | RETEXTURE | ambientCG `PaintedPlaster001` (tint per palette, satin) | - | white painted MDF/plaster with roller texture; resize plinths to substituted sculptures (+0.1 m) |
| `cornice_7.00_4.` | 1 | 24 | RETEXTURE | ambientCG `PaintedPlaster001` (tint per palette, satin) | - | white painted MDF/plaster with roller texture; resize plinths to substituted sculptures (+0.1 m) |
| `skirt_b_2.` | 2 | 24 | KEEP | - | - | dark satin skirting; small |
| `manuscript.block` | 1 | 24 | KEEP | - | - | procedural parchment with lettering is the fine-text target; no scanned manuscript; boards could take `brown_leather` |
| `manuscript.boards` | 1 | 24 | KEEP | - | - | procedural parchment with lettering is the fine-text target; no scanned manuscript; boards could take `brown_leather` |
| `manuscript.cradle` | 1 | 24 | RETEXTURE | Poly Haven `velour_velvet` (tint dark) | - | real velvet weave and sheen under the hood light |
| `acrylic_stand` | 1 | 24 | KEEP | - | - | thindielectric acrylic is correct |
| `plinth_46_108.body` | 2 | 24 | RETEXTURE | ambientCG `PaintedPlaster001` (tint per palette, satin) | - | white painted MDF/plaster with roller texture; resize plinths to substituted sculptures (+0.1 m) |
| `plinth_46_108.gap` | 2 | 24 | KEEP | - | - | structural shadow gap / emitter backing |
| `canvas_100x130.sides` | 2 | 16 | KEEP | - | - | canvas edges |
| `canvas_130x100.sides` | 2 | 16 | KEEP | - | - | canvas edges |
| `rectangle:wall_g2` | 6 | 12 | RETEXTURE | ambientCG `PaintedPlaster001`/`002` (tint per palette) | - | main fix for blank-wall frames together with the additions below |
| `skirt_8.` | 1 | 12 | KEEP | - | - | dark satin skirting; small |
| `text_board` | 1 | 12 | RETEXTURE | ambientCG `PaintedPlaster001` (tint per palette, satin) | - | white painted MDF/plaster with roller texture; resize plinths to substituted sculptures (+0.1 m) |
| `cube:deck_0` | 1 | 12 | RETEXTURE | Poly Haven `velour_velvet` / ambientCG `Fabric0xx` tinted to the seeded cloth colours | - | case decks |
| `cube:deck_1` | 1 | 12 | RETEXTURE | Poly Haven `velour_velvet` / ambientCG `Fabric0xx` tinted to the seeded cloth colours | - | case decks |
| `cube:deck_2` | 1 | 12 | RETEXTURE | Poly Haven `velour_velvet` / ambientCG `Fabric0xx` tinted to the seeded cloth colours | - | case decks |
| `cube:deck_3` | 1 | 12 | RETEXTURE | Poly Haven `velour_velvet` / ambientCG `Fabric0xx` tinted to the seeded cloth colours | - | case decks |
| `fossil_slab` | 1 | 12 | SUBSTITUTE | absorbed by the trilobite scan's matrix (drop) | - | the scan includes the rock; keep a velvet deck under it |
| `cube:velvet_dark` | 1 | 12 | RETEXTURE | Poly Haven `velour_velvet` (tint dark) | - | real velvet weave and sheen under the hood light |
| `plinth_50_22.body` | 1 | 12 | RETEXTURE | ambientCG `PaintedPlaster001` (tint per palette, satin) | - | white painted MDF/plaster with roller texture; resize plinths to substituted sculptures (+0.1 m) |
| `plinth_50_22.gap` | 1 | 12 | KEEP | - | - | structural shadow gap / emitter backing |
| `plinth_56_86.body` | 1 | 12 | RETEXTURE | ambientCG `PaintedPlaster001` (tint per palette, satin) | - | white painted MDF/plaster with roller texture; resize plinths to substituted sculptures (+0.1 m) |
| `plinth_56_86.gap` | 1 | 12 | KEEP | - | - | structural shadow gap / emitter backing |
| `cube:marble` | 1 | 12 | RETEXTURE | ambientCG `Marble006` | - | bust socle |
| `rectangle:emitter_backing [emitter]` | 4 | 8 | KEEP | - | - | structural shadow gap / emitter backing |
| `canvas_90x115.sides` | 1 | 8 | KEEP | - | - | canvas edges |
| `canvas_150x112.sides` | 1 | 8 | KEEP | - | - | canvas edges |
| `canvas_100x128.sides` | 1 | 8 | KEEP | - | - | canvas edges |
| `canvas_62x80.sides` | 1 | 8 | KEEP | - | - | canvas edges |
| `canvas_100x76.sides` | 1 | 8 | KEEP | - | - | canvas edges |
| `canvas_125x94.sides` | 1 | 8 | KEEP | - | - | canvas edges |
| `canvas_220x155.sides` | 1 | 8 | KEEP | - | - | canvas edges |
| `canvas_110x140.sides` | 1 | 8 | KEEP | - | - | canvas edges |
| `canvas_120x90.sides` | 1 | 8 | KEEP | - | - | canvas edges |
| `canvas_90x118.sides` | 1 | 8 | KEEP | - | - | canvas edges |
| `canvas_240x170.sides` | 1 | 8 | KEEP | - | - | canvas edges |
| `rectangle:wall_g1` | 3 | 6 | RETEXTURE | ambientCG `PaintedPlaster001`/`002` (tint per palette) | - | main fix for blank-wall frames together with the additions below |
| `canvas_100x130.face` | 2 | 4 | RETEXTURE | CC0 painting images from Smithsonian Open Access (SAAM/NPG, `online_media_type:Images`, `metadata_usage.access == CC0`) | match aspect class P/L/W | real paintings beat procedural; needs an image fetch path in web_assets (planned); keep procedural as fallback |
| `canvas_130x100.face` | 2 | 4 | RETEXTURE | CC0 painting images from Smithsonian Open Access (SAAM/NPG, `online_media_type:Images`, `metadata_usage.access == CC0`) | match aspect class P/L/W | real paintings beat procedural; needs an image fetch path in web_assets (planned); keep procedural as fallback |
| `rectangle:floor_stone` | 1 | 2 | RETEXTURE | ambientCG `Tiles142` (limestone tiles) or Poly Haven `floor_tiles_06` | - | real joints and wear vs the procedural limestone |
| `rectangle:floor_oak` | 1 | 2 | RETEXTURE | Poly Haven `herringbone_parquet` | 3.4 m tile | gallery 2 |
| `rectangle:trim_stone` | 1 | 2 | RETEXTURE | ambientCG `Marble006` or limestone `Tiles142` | - | door threshold / trim |
| `canvas_90x115.face` | 1 | 2 | RETEXTURE | CC0 painting images from Smithsonian Open Access (SAAM/NPG, `online_media_type:Images`, `metadata_usage.access == CC0`) | match aspect class P/L/W | real paintings beat procedural; needs an image fetch path in web_assets (planned); keep procedural as fallback |
| `rectangle:label_P0` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `canvas_150x112.face` | 1 | 2 | RETEXTURE | CC0 painting images from Smithsonian Open Access (SAAM/NPG, `online_media_type:Images`, `metadata_usage.access == CC0`) | match aspect class P/L/W | real paintings beat procedural; needs an image fetch path in web_assets (planned); keep procedural as fallback |
| `rectangle:label_L6` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `canvas_100x128.face` | 1 | 2 | RETEXTURE | CC0 painting images from Smithsonian Open Access (SAAM/NPG, `online_media_type:Images`, `metadata_usage.access == CC0`) | match aspect class P/L/W | real paintings beat procedural; needs an image fetch path in web_assets (planned); keep procedural as fallback |
| `rectangle:label_P5` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `canvas_62x80.face` | 1 | 2 | RETEXTURE | CC0 painting images from Smithsonian Open Access (SAAM/NPG, `online_media_type:Images`, `metadata_usage.access == CC0`) | match aspect class P/L/W | real paintings beat procedural; needs an image fetch path in web_assets (planned); keep procedural as fallback |
| `rectangle:label_P6` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `canvas_100x76.face` | 1 | 2 | RETEXTURE | CC0 painting images from Smithsonian Open Access (SAAM/NPG, `online_media_type:Images`, `metadata_usage.access == CC0`) | match aspect class P/L/W | real paintings beat procedural; needs an image fetch path in web_assets (planned); keep procedural as fallback |
| `rectangle:label_L3` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `canvas_125x94.face` | 1 | 2 | RETEXTURE | CC0 painting images from Smithsonian Open Access (SAAM/NPG, `online_media_type:Images`, `metadata_usage.access == CC0`) | match aspect class P/L/W | real paintings beat procedural; needs an image fetch path in web_assets (planned); keep procedural as fallback |
| `rectangle:label_L1` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `canvas_220x155.face` | 1 | 2 | RETEXTURE | CC0 painting images from Smithsonian Open Access (SAAM/NPG, `online_media_type:Images`, `metadata_usage.access == CC0`) | match aspect class P/L/W | real paintings beat procedural; needs an image fetch path in web_assets (planned); keep procedural as fallback |
| `rectangle:label_W2` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `canvas_110x140.face` | 1 | 2 | RETEXTURE | CC0 painting images from Smithsonian Open Access (SAAM/NPG, `online_media_type:Images`, `metadata_usage.access == CC0`) | match aspect class P/L/W | real paintings beat procedural; needs an image fetch path in web_assets (planned); keep procedural as fallback |
| `rectangle:label_P7` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `canvas_120x90.face` | 1 | 2 | RETEXTURE | CC0 painting images from Smithsonian Open Access (SAAM/NPG, `online_media_type:Images`, `metadata_usage.access == CC0`) | match aspect class P/L/W | real paintings beat procedural; needs an image fetch path in web_assets (planned); keep procedural as fallback |
| `rectangle:label_L4` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `canvas_90x118.face` | 1 | 2 | RETEXTURE | CC0 painting images from Smithsonian Open Access (SAAM/NPG, `online_media_type:Images`, `metadata_usage.access == CC0`) | match aspect class P/L/W | real paintings beat procedural; needs an image fetch path in web_assets (planned); keep procedural as fallback |
| `rectangle:label_P4` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `canvas_240x170.face` | 1 | 2 | RETEXTURE | CC0 painting images from Smithsonian Open Access (SAAM/NPG, `online_media_type:Images`, `metadata_usage.access == CC0`) | match aspect class P/L/W | real paintings beat procedural; needs an image fetch path in web_assets (planned); keep procedural as fallback |
| `rectangle:label_W3` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `rectangle:label_P3` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `rectangle:label_L5` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `rectangle:label_L0` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `rectangle:label_P1` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |
| `rectangle:wall_text` | 1 | 2 | KEEP | - | - | procedural label text (fine-text detail) |

Sculpture slots are seeded (bird / knot / muse / concretion / urn pick 4); the seed in the inventory had no `concretion`. Plan for it too: **concretion -> Smithsonian "Kneeling winged monster" (NMAA, limestone, 88.4 x 47.3 x 28.5 cm, CC0)** https://3d-api.si.edu/content/document/3d_package:0dc68216-3651-44c7-99cf-18e5d4d1eb9f/resources/kneeling-winged-monster-150k-4096-obj.zip on a 0.45 m plinth.

## Big surfaces (textures)

| surface | now | texture | why |
|---|---|---|---|
| gallery 1 walls `wall_g1` | procedural plaster, flat colour | ambientCG `PaintedPlaster001`, tinted per seeded palette | roller texture and sheen so a wall filling the frame still reads as a real wall |
| feature / gallery 2 walls `wall_feature`, `wall_g2` | flat teal / crimson / slate | ambientCG `PaintedPlaster002`, tinted | same; the audit's blank teal and slate frames (tiles 2, 8, 14, 19, 23) |
| ceiling | flat 0.86 diffuse | ambientCG `PaintedPlaster001` white | |
| gallery 1 floor | procedural limestone | ambientCG `Tiles142` (limestone tiles) or Poly Haven `floor_tiles_06` | real joints, polish, wear |
| gallery 2 floor | procedural planks | Poly Haven `herringbone_parquet` (3.4 m tile) | classic gallery parquet |
| plinth bodies, cornices | flat white principled | ambientCG `PaintedPlaster001` satin + slight edge darkening | the audit's plain plinth faces |
| vitrine bases | flat walnut or white | Poly Haven `american_walnut_veneer` / painted plaster | audit tile 21 (flat brown cabinet face) |
| case decks, ropes, stands | flat cloth/velvet colours | Poly Haven `velour_velvet` (tinted) | real pile and sheen |
| bench cushion / steel | flat leather / steel | Poly Haven `brown_leather` or ambientCG `Leather0xx`; ambientCG `Metal009` | |

## Fixing the blank-wall and plain-plinth frames

Textures alone are not enough: random close poses need something meaningful on every wall run. Hang paintings at a realistic density (2-3 per 4 m run, one salon-style group on the feature wall), add wall-mounted objects from the list below at eye level between paintings, put labels on plinth fronts, and keep the camera box 0.6 m off the walls so a frame can no longer be one wall. Plinths get sized to their sculpture and the painted-plaster texture.

## Additions (exist in the catalogs, CC0)

1. **Gallery 2 centrepiece:** Smithsonian "Model of the Greek Slave" (SAAM, plaster, 168.6 x 54.6 x 46.6 cm) https://3d-api.si.edu/content/document/3d_package:8edffe56-c358-4c3a-a61f-019f615ccef0/resources/greek-slave-plaster-cast-150k-4096-obj.zip on a low 0.25 m plinth (alternative: Poly Haven `gothic_statue`, ~1.7 m).
2. **Busts on wall consoles / pedestals:** Smithsonian Rutherford B. Hayes plaster bust (NPG, 27.3 x 24.8 x 12.7 cm) https://3d-api.si.edu/content/document/3d_package:1f3700e8-6d01-4488-8ab9-ea14031ef641/resources/rutherford-b.-hayes-plaster-bust-150k-4096-obj.zip; Poly Haven `lion_head` (0.32 x 0.21 x 0.44), `horse_head` (0.21 x 0.33 x 0.42), `bull_head` (0.31 x 0.28 x 0.43) on brackets between paintings.
3. **Second tall case (garniture pair):** Smithsonian Beaker-shaped vase F1980.194 (NMAA, porcelain, ~45 cm) https://3d-api.si.edu/content/document/3d_package:d8c64b96-4ebc-11ea-b77f-2e728ce88125/resources/f1980_194-150k-4096-obj.zip next to the baluster vase.
4. **Natural-history flat case:** Smithsonian *Diictodon feliceps* skull https://3d-api.si.edu/content/document/3d_package:3b3add34-8d97-4a66-96fa-4e2d343db77c/resources/Diictodon_feliceps_Owen__1876-150k-4096-obj_std.zip and *Merycoidodon* skull https://3d-api.si.edu/content/document/3d_package:42461630-b2c2-415d-9049-14b5e3b0962b/resources/Merycoidodon_sp-150k-4096-obj_std.zip; shells *Conus textile* https://3d-api.si.edu/content/document/3d_package:c29a9ede-c224-46c9-8791-69ff36c23828/resources/Conus_textile-150k-4096-obj_std.zip, *Cristaria plicata* https://3d-api.si.edu/content/document/3d_package:13924473-268f-4596-83fd-73ca16516a0d/resources/Cristaria_plicata-150k-4096-obj_std.zip; second trilobite *Coltraneia oufatenensis* https://3d-api.si.edu/content/document/3d_package:03cb230f-8ad6-4d7f-bc41-aee9a05644fb/resources/Coltraneia_oufatenensis_Morzadec__2001-150k-4096-obj_std.zip. All small: scale by bounds to 5-25 cm.
5. **Decorative arts corner (gallery 2):** Poly Haven `vintage_grandfather_clock_01` (1.0 x 0.8 x 2.19, behind a stanchion rope), `mantel_clock_01` on a plinth, `brass_vase_01`/`brass_goblets` in a case.
6. **Visitor bench in gallery 2:** Poly Haven `Ottoman_01` (0.88 x 0.62 x 0.62).
7. **Building services on walls:** Poly Haven `fire_alarm` (0.10 x 0.03 x 0.14), `korean_fire_extinguisher_01` (in a corner by the door), `security_camera_01`/`02` in two ceiling corners. Procedural additions with real textures: exit sign over the door (emissive), HVAC grilles near the ceiling, a thermostat and humidity logger.
8. **Arms case (optional):** Poly Haven `antique_estoc` (1.49 m sword) in a wall vitrine.

## Not substituted, on purpose

Vitrine glass and the gem dielectrics (the scene's refraction purpose), procedural label/manuscript text (fine-text detail), black frames, track lighting, structural gaps and emitter backings. No CC0 scanned jewellery, coins, stanchions or manuscripts were found in Poly Haven or Smithsonian 3D.
