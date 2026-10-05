# Graphics Experiment Archive

| # | Experiment | Code | Notes |
|---|---|---|---|
| 01 | Vertex Attributes and Interpolation | [Code](01_vertex_attributes/) | [Notes](../../notes/graphics/01_vertex_attributes.md) |
| 02 | Indexed Drawing with an EBO | [Code](02_indexed_drawing/) | [Notes](../../notes/graphics/02_indexed_drawing.md) |
| 03 | Uniforms and Animation | [Code](03_uniforms_animation/) | [Notes](../../notes/graphics/03_uniforms_animation.md) |
| 04 | Transformation Matrices | [Code](04_transformation_matrices/) | [Notes](../../notes/graphics/04_transformation_matrices.md) |
| 05 | First 3D Cube | [Code](05_first_3d_cube/) | [Notes](../../notes/graphics/05_first_3d_cube.md) |
| 06 | Depth Testing and the Depth Buffer | [Code](06_depth_testing/) | [Notes](../../notes/graphics/06_depth_testing.md) |
| 07 | Model, View, and Projection | [Code](07_model_view_projection/) | [Notes](../../notes/graphics/07_model_view_projection.md) |
| 08 | Interactive Camera | [Code](08_interactive_camera/) | [Notes](../../notes/graphics/08_interactive_camera.md) |
| 11 | Off-Screen Framebuffers and Post-Processing | [Code](11_framebuffers/) | [Notes](../../notes/graphics/11_framebuffers.md) |
| 12 | Sampling and Linearizing the Depth Texture | [Code](12_depth_texture_postprocess/) | [Notes](../../notes/graphics/12_depth_texture_postprocess.md) |
| 13 | Physical Focus Distance and Circle of Confusion | [Code](13_coc_visualization/) | [Notes](../../notes/graphics/13_coc_visualization.md) |
| 14 | Basic Depth of Field | [Code](14_basic_dof/) | [Notes](../../notes/graphics/14_basic_dof.md) |
| 15 | Where Screen-Space Depth of Field Breaks | [Code](15_screen_space_halo/) | [Notes](../../notes/graphics/15_screen_space_halo.md) |
| 16 | Multi-View Depth of Field | [Code](16_multiview_dof/) | [Notes](../../notes/graphics/16_multiview_dof.md) |
| 17 | Lens-Sampled Ray Tracing | [Code](17_raytraced_dof/) | [Notes](../../notes/graphics/17_raytraced_dof.md) |

Experiments 01-14 archive a source snapshot each, because the active `src/` moved on after them. Experiments 15-17 do not: all three are one program, `DoFApproaches`, rendering one shared scene through three different visibility solves, selected with `--method 1|2|3`. Copying it three times would only let the copies drift apart and destroy the comparison, so those directories hold the setup, the commands and the measured findings instead.


## Qualified stage-2 comparison

[Production OpenGL gather on Mitsuba inputs](stage2_qualified_gather/): reproducible
settings, float32 references, both gather variants, linear metrics and measured
noise. This is separate from the historical numbered snapshots.
