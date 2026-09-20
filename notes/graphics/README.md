# Graphics Experiments

| # | Experiment | Purpose |
|---|---|---|
| 01 | [Vertex Attributes and Interpolation](01_vertex_attributes.md) | Understand vertex memory layout, attributes, and rasterizer interpolation. |
| 02 | [Indexed Drawing with an EBO](02_indexed_drawing.md) | Understand indices, vertex reuse, and indexed primitive assembly. |
| 03 | [Uniforms and Animation](03_uniforms_animation.md) | Understand shader uniforms, shared draw-call values, and time-based animation. |
| 04 | [Transformation Matrices](04_transformation_matrices.md) | Understand model-space geometry, matrix transforms, and transform order. |
| 05 | [First 3D Cube](05_first_3d_cube.md) | Understand cube triangle meshes, 3D rotations, and why depth testing is needed. |
| 06 | [Depth Testing and the Depth Buffer](06_depth_testing.md) | Understand per-fragment depth testing and how the depth buffer fixes 3D visibility. |
| 07 | [Model, View, and Projection](07_model_view_projection.md) | Understand the full MVP transform pipeline from local space to screen. |
| 08 | [Interactive Camera](08_interactive_camera.md) | Understand explicit first-person camera state, mouse-look, and delta-time WASD movement. |
| 09 | [Raw Depth Visualization](09_raw_depth_visualization.md) | Observe `gl_FragCoord.z`, window-space depth, and nonlinear perspective depth. |
| 10 | [Linear Depth Reconstruction](10_linear_depth.md) | Reconstruct linear view depth from raw perspective depth for cleaner depth visualization. |
| 11 | [Off-Screen Framebuffers and Post-Processing](11_framebuffers.md) | Render the scene into color/depth textures, then present it through a full-screen quad. |
| 12 | [Sampling and Linearizing the Depth Texture](12_depth_texture_postprocess.md) | Sample FBO color/depth textures in the screen pass and reconstruct linear depth. |
| 13 | [Physical Focus Distance and Circle of Confusion](13_coc_visualization.md) | Calculate and visualize pixel-space CoC from sampled linear depth. |
| 14 | [Basic Depth of Field](14_basic_dof.md) | Render the first DoF blur with a 17-tap disk gather driven by CoC magnitude. |
