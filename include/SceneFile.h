#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

// Loader for scene/*.scene, the one description the OpenGL pass and the Cycles
// reference share. tools/scene/scene_loader.py is the Python twin of this file:
// same grammar, same primitive tessellation, same transform order. If the two
// ever disagree, the two renderers are no longer drawing the same scene, so
// tests/scene_file.cpp and tests/test_scene_file.py check both against the same
// expected numbers.
//
// Deliberately free of glm and OpenGL: this compiles and runs as plain C++17 so
// the geometry can be tested without a GL context.

struct SceneVec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

// One GPU vertex, uploaded interleaved exactly in this order:
// attribute 0 = position, 1 = albedo, 2 = normal, 4 = emission.
// Attribute 3 is left to Mesh.cpp's imported-mesh UVs so both VAOs can feed the
// same shader program.
struct SceneVertex {
    float position[3]{};
    float albedo[3]{};  // Linear, not sRGB.
    float normal[3]{};
    float emission = 0.0f;  // > 0 means the surface emits albedo * emission and is not shaded.
};

struct SceneCamera {
    float position[3]{0.0f, 0.0f, 5.0f};
    float yawDegrees = -90.0f;
    float pitchDegrees = 0.0f;
    float focusDistanceMeters = 5.0f;
    float fNumber = 1.4f;
    float focalLengthMillimeters = 50.0f;
    float sensorHeightMillimeters = 24.0f;
};

struct SceneSun {
    float direction[3]{-0.45f, 0.78f, 0.44f};  // From a lit surface TOWARD the light.
    float color[3]{1.0f, 0.88f, 0.72f};
    float energy = 4.2f;  // Irradiance in W/m^2, matching Blender's sun strength.
    float angularDiameterDegrees = 0.6f;
};

struct SceneAmbient {
    float color[3]{0.42f, 0.52f, 0.72f};
    float strength = 0.24f;  // Uniform sky radiance = color * strength.
};

struct SceneDescription {
    SceneCamera camera;
    SceneSun sun;
    SceneAmbient ambient;

    std::vector<SceneVertex> vertices;
    std::vector<unsigned int> indices;

    SceneVec3 boundsMin;
    SceneVec3 boundsMax;
    // Region the shadow map should cover. Present only if the file says so;
    // otherwise the caller should fall back to the geometry bounds, which can
    // be much larger than the part of the scene that matters.
    bool hasShadowRegion = false;
    SceneVec3 shadowLow;
    SceneVec3 shadowHigh;

    std::size_t primitiveCount = 0;
    std::size_t emissiveTriangleCount = 0;
    std::size_t smoothTriangleCount = 0;

    std::size_t triangleCount() const { return indices.size() / 3; }
};

// Both return false and fill `error` (with a line number where possible)
// instead of throwing, matching loadObjMesh in Mesh.h.
bool parseSceneText(const std::string& text, SceneDescription& scene, std::string& error);
bool loadSceneFile(const std::filesystem::path& path, SceneDescription& scene, std::string& error);
