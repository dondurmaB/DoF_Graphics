#pragma once

#include <cstddef>
#include <vector>

#include <glm/glm.hpp>

namespace dof_approaches {

// Every solid in this scene is an analytic primitive. That is deliberate: the
// rasterized path and the ray-traced path must describe exactly the same
// geometry, or the three depth-of-field methods are not comparable.
enum class PrimKind : int {
    Plane = 0,    // horizontal, infinite, y = position.y
    Box = 1,      // axis aligned, size = half extents
    Sphere = 2,   // size.x = radius
    Cylinder = 3  // y-aligned, size.x = radius, size.y = half height
};

struct Primitive {
    PrimKind kind = PrimKind::Box;
    glm::vec3 position{0.0f};
    glm::vec3 size{0.5f};
    glm::vec3 color{0.8f};
    float emissive = 0.0f;
    float checker = 0.0f;  // 0 = flat shading, otherwise world checker scale
};

struct SceneVertex {
    glm::vec3 position;  // already baked into world space
    glm::vec3 normal;    // outward normal, world space
};

struct MeshData {
    std::vector<SceneVertex> vertices;
    std::vector<unsigned int> indices;
};

struct Scene {
    std::vector<Primitive> primitives;
    std::vector<MeshData> meshes;  // parallel to primitives, baked into world space
    glm::vec3 sunDirection{-0.35f, -0.82f, 0.45f};
    glm::vec3 background{0.045f, 0.055f, 0.085f};
    glm::vec3 eye{0.0f, 1.60f, 3.60f};
    glm::vec3 target{0.0f, 1.45f, -8.40f};
    float focusDistance = 3.80f;
};

// A deliberately hostile scene for screen-space depth of field. It stages four
// failure modes at once: a razor-sharp railing on the focus plane in front of a
// bright wall (halo), bright near posts thinner than their own circle of
// confusion (missing see-through), emissive background lights (bokeh energy),
// and a receding corridor (continuous depth ramp). See scene.cpp for the zone
// layout and the reasoning behind each placement.
Scene createHaloScene();

struct PackedPrimitive {
    glm::vec4 params;    // xyz position, w kind
    glm::vec4 size;      // xyz size, w emissive
    glm::vec4 material;  // rgb color, w checker scale
};

std::vector<PackedPrimitive> packPrimitives(const Scene& scene);

}  // namespace dof_approaches
