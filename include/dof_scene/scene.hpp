#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <glm/glm.hpp>

namespace dof_scene {

// Scene convention: one world unit is one meter, y is up, and the camera looks
// into negative z from the café entrance side.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
};

enum class Shape {
    Box = 0,
    Sphere = 1,
    Cylinder = 2,
    Torus = 3,
    Cup = 4,
    Teapot = 5,
    Spout = 6,
    Vase = 7,
    Count = 8
};

struct Material {
    glm::vec3 color;
    float roughness = 0.5f;
    float metallic = 0.0f;
    int texture = 0;
    glm::vec3 emission{0.0f};
};

struct Object {
    Shape shape;
    glm::mat4 model{1.0f};
    Material material;
};

struct Scene {
    std::array<MeshData, static_cast<std::size_t>(Shape::Count)> meshes;
    std::vector<Object> objects;
    glm::vec3 focusPoint;
};

// Primitive mesh space:
// - Box is a 1m cube centered at the origin.
// - Sphere is centered at the origin with radius 0.5m.
// - Cylinder is y-axis aligned, radius 0.5m, height 1m.
// - Torus is centered at the origin, wrapped around the y axis.
// - Cup, Teapot, Spout, and Vase are normalized custom meshes with their bases
//   near local y=0 for direct tabletop placement.
// Object transforms place and scale the primitives into the café scene.
Scene createCafeScene();

} // namespace dof_scene
