#include "SceneFile.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

// Cross-language check. The expected numbers below were printed by the Python
// builder (tools/scene/scene_loader.py summary()); tests/test_scene_file.py
// asserts the same numbers from the Python side. If one builder is changed
// without the other, exactly one of the two tests fails and says so, which is
// the only cheap way to notice that the OpenGL pass and the Cycles reference
// have stopped drawing the same scene.
//
// Regenerate them with:
//   python3 tests/test_scene_file.py --print-expected

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void requireClose(double actual, double expected, double tolerance, const std::string& message) {
    if (!(std::fabs(actual - expected) <= tolerance)) {
        throw std::runtime_error(message + " (expected " + std::to_string(expected) + ", got " +
                                 std::to_string(actual) + ")");
    }
}

// scene/cafe.scene is a committed build artefact, so the test reads the real
// file rather than a fixture: a regenerated scene must update the numbers here.
// scene/alley.scene is still a valid input to both renderers, just not the one
// stage 2 compares on.
const std::size_t kPrimitives = 2678;
const std::size_t kVertices = 134957;
const std::size_t kTriangles = 115240;
const std::size_t kEmissiveTriangles = 5820;
const std::size_t kSmoothTriangles = 50480;
const double kBoundsMin[3] = {-3.6, -1.32, -10.2};
const double kBoundsMax[3] = {3.6, 1.795, 6.2};
const double kPositionSum[3] = {76694.3756314, -25110.0486, -276524.7532};
const double kAreaSum = 1351.86003672;

bool rejects(const std::string& text, const std::string& expectedFragment) {
    SceneDescription scene;
    std::string error;
    if (parseSceneText(text, scene, error)) return false;
    if (error.find(expectedFragment) == std::string::npos) {
        std::cerr << "  wrong message: " << error << " (wanted \"" << expectedFragment << "\")\n";
        return false;
    }
    // A rejected file must leave nothing half-built behind.
    return scene.vertices.empty() && scene.indices.empty() && scene.primitiveCount == 0;
}

}  // namespace

int main() {
    try {
        // --- Grammar: every rule that scene_loader.py enforces, enforced here too.
        require(rejects("box pos 0 0 0\n", "missing 'version'"), "A file without a version must be rejected");
        require(rejects("version 2\nbox pos 0 0 0\n", "is not 1"), "A future version must be rejected");
        require(rejects("version 1\nbox pos 0 0 0 wibble 1\n", "unknown key 'wibble'"), "Unknown keys must be rejected");
        require(rejects("version 1\nbox seg 8\n", "not valid here"), "A box must not accept 'seg'");
        require(rejects("version 1\nbox pos 0 0 0 pos 1 1 1\n", "appears twice"), "Repeated keys must be rejected");
        require(rejects("version 1\nbox pos 0 0\n", "needs 3 number(s)"), "Short key lists must be rejected");
        require(rejects("version 1\nbox pos 0 0 x\n", "is not a number"), "Non-numeric values must be rejected");
        require(rejects("version 1\nbox pos 0 0 nan\n", "must be finite"), "Non-finite values must be rejected");
        require(rejects("version 1\nbox size 1 0 1\n", "size must not be zero"), "A zero scale has no inverse transpose");
        require(rejects("version 1\ncyl seg 2\n", "between 3 and 128"), "Two-segment cylinders must be rejected");
        require(rejects("version 1\ncyl taper 9\n", "taper must be between 0 and 8"), "A wild taper must be rejected");
        require(rejects("version 1\nbox taper 0.5\n", "not valid here"), "A box must not accept 'taper'");
        require(rejects("version 1\nbox emit -1\n", "must not be negative"), "Negative emission must be rejected");
        require(rejects("version 1\nshadow lo 0 0 0\n", "both 'lo' and 'hi'"), "A half shadow region must be rejected");
        require(rejects("version 1\nshadow lo 0 0 0 hi 0 1 1\n", "must exceed 'lo'"), "An empty shadow region must be rejected");
        require(rejects("version 1\ncamera lens 0\n", "lens and sensor must be positive"), "A zero focal length must be rejected");
        require(rejects("version 1\ncamera focus 0.01\n", "focus must exceed"), "Focus inside the lens must be rejected");
        require(rejects("version 1\nwibble 1 2 3\n", "unknown entry 'wibble'"), "Unknown entries must be rejected");
        require(rejects("version 1\ncamera pos 0 0 5\n", "no geometry"), "A scene with no geometry must be rejected");
        {
            SceneDescription scene;
            std::string error;
            require(!parseSceneText("version 1\n\n\nbox pos 0 0\n", scene, error), "Bad file must fail");
            require(error.find("line 4") == 0, "Errors must point at the offending line, got: " + error);
        }

        // --- Defaults: unset fields must equal the literals main.cpp falls back to.
        {
            SceneDescription scene;
            std::string error;
            require(parseSceneText("version 1\nbox pos 0 0 0\n", scene, error), "Minimal scene should load: " + error);
            require(scene.camera.position[2] == 5.0f && scene.camera.yawDegrees == -90.0f,
                    "Default camera must sit at (0,0,5) looking down -Z");
            require(scene.camera.focalLengthMillimeters == 50.0f && scene.camera.sensorHeightMillimeters == 24.0f &&
                    scene.camera.fNumber == 1.4f && scene.camera.focusDistanceMeters == 5.0f,
                    "Default lens must stay 50mm f/1.4 focused at 5m");
            require(!scene.hasShadowRegion, "A file without a shadow line must say so, not invent a region");
            require(scene.primitiveCount == 1 && scene.vertices.size() == 24 && scene.triangleCount() == 12,
                    "A unit box must be 6 faces x 4 corners and 12 triangles");
            require(scene.emissiveTriangleCount == 0 && scene.smoothTriangleCount == 0,
                    "A box is neither emissive nor smooth by default");
            for (const SceneVertex& vertex : scene.vertices) {
                require(std::fabs(std::fabs(vertex.position[0]) - 0.5f) < 1e-6f,
                        "A default box must be the unit cube");
            }
        }

        // --- taper: a cylinder's top radius, for turned objects. The property
        // that matters is that taper 1.0 is EXACTLY the old cylinder, so no
        // existing scene file changes.
        {
            SceneDescription plain, explicitTaper, cone;
            std::string error;
            require(parseSceneText("version 1\ncyl pos 0 0 0 seg 24\n", plain, error), error);
            require(parseSceneText("version 1\ncyl pos 0 0 0 seg 24 taper 1\n", explicitTaper, error), error);
            require(plain.vertices.size() == explicitTaper.vertices.size() &&
                    plain.indices == explicitTaper.indices,
                    "taper 1.0 must build the same mesh as a plain cylinder");
            for (std::size_t index = 0; index < plain.vertices.size(); ++index) {
                require(std::memcmp(&plain.vertices[index], &explicitTaper.vertices[index], sizeof(SceneVertex)) == 0,
                        "taper 1.0 must be byte-identical");
                for (int axis = 0; axis < 3; ++axis) {
                    requireClose(explicitTaper.vertices[index].position[axis],
                                 plain.vertices[index].position[axis], 1e-9, "taper 1.0 position");
                    requireClose(explicitTaper.vertices[index].normal[axis],
                                 plain.vertices[index].normal[axis], 1e-9, "taper 1.0 normal");
                }
            }
            require(parseSceneText("version 1\ncyl pos 0 0 0 size 1 1 1 seg 32 taper 0\n", cone, error), error);
            // A 45-degree cone: side normals tilt to match the slope.
            const double magnitude = std::sqrt(1.0 + 0.25);
            bool checked = false;
            for (const SceneVertex& vertex : cone.vertices) {
                // The bottom cap shares these positions, so skip anything
                // whose normal points along the axis.
                if (std::fabs(vertex.position[1] + 0.5) > 1e-6 ||
                    std::fabs(vertex.position[0] - 0.5) > 1e-6 ||
                    std::fabs(vertex.normal[1]) > 0.9f) continue;
                requireClose(vertex.normal[0], 1.0 / magnitude, 1e-5, "cone side normal x");
                requireClose(vertex.normal[1], 0.5 / magnitude, 1e-5, "cone side normal y");
                checked = true;
            }
            require(checked, "The cone should have a vertex at the bottom of its seam");
            // The degenerate top cap is skipped, not built as a zero-radius disc.
            for (const SceneVertex& vertex : cone.vertices) {
                if (std::fabs(vertex.position[1] - 0.5) < 1e-6) {
                    requireClose(vertex.position[0], 0.0, 1e-6, "cone apex x");
                }
            }
        }

        // Material defaults and the opt-in chamfer keep old geometry unchanged.
        for (const std::string kind : {"box", "cyl", "sph"}) {
            SceneDescription defaults, material;
            std::string error;
            require(parseSceneText("version 1\n" + kind + "\n", defaults, error), error);
            require(parseSceneText("version 1\n" + kind + " rough 0.2 spec 0.7\n", material, error), error);
            require(defaults.indices == material.indices, "Materials must not change topology");
            for (const auto& v : defaults.vertices)
                require(v.roughness == 0.5f && v.specular == 0.0f, "Legacy material defaults");
            for (const auto& v : material.vertices)
                require(v.roughness == 0.2f && v.specular == 0.7f, "Material attributes must propagate");
        }
        {
            SceneDescription bevel;
            std::string error;
            require(parseSceneText("version 1\nbox size 4 2 1 bevel 0.1 rough 0.2 spec 0.7\n", bevel, error), error);
            require(bevel.triangleCount() == 44, "Chamfered box has 44 triangles");
            requireClose(bevel.boundsMin.x, -2, 1e-6, "Chamfer preserves bounds");
            requireClose(bevel.boundsMax.z, 0.5, 1e-6, "Chamfer preserves bounds");
        }
        require(rejects("version 1\nbox rough 0\n", "rough must"), "Reject singular roughness");
        require(rejects("version 1\ncyl spec 1.1\n", "spec must"), "Reject invalid F0");
        require(rejects("version 1\nsph bevel 0.1\n", "not valid here"), "Only boxes take bevel");
        require(rejects("version 1\nbox bevel 0.5\n", "bevel must"), "Reject collapsed bevel");

        // --- Transform order: world = pos + Ry*Rx*Rz * (size * local), normals by inverse transpose.
        {
            SceneDescription scene;
            std::string error;
            // 90 degrees about Y turns local +X into world -Z, and the anisotropic
            // scale must not tilt the normals of an axis-aligned box.
            require(parseSceneText("version 1\nbox pos 1 2 3 size 4 1 1 rot 0 90 0\n", scene, error),
                    "Rotated box should load: " + error);
            requireClose(scene.boundsMin.x, 0.5, 1e-5, "Rotated box min x");
            requireClose(scene.boundsMax.x, 1.5, 1e-5, "Rotated box max x");
            requireClose(scene.boundsMin.z, 1.0, 1e-5, "Rotated box min z");
            requireClose(scene.boundsMax.z, 5.0, 1e-5, "Rotated box max z");
            for (const SceneVertex& vertex : scene.vertices) {
                const double magnitude = std::sqrt(vertex.normal[0] * vertex.normal[0] +
                                                   vertex.normal[1] * vertex.normal[1] +
                                                   vertex.normal[2] * vertex.normal[2]);
                requireClose(magnitude, 1.0, 1e-5, "Normals must stay unit length");
                int axes = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    if (std::fabs(vertex.normal[axis]) > 1e-5f) ++axes;
                }
                require(axes == 1, "A scaled, Y-rotated box must keep axis-aligned normals");
            }
        }

        // --- The real scene, against the Python builder's numbers.
        const std::filesystem::path scenePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "scene" / "cafe.scene";
        SceneDescription scene;
        std::string error;
        require(loadSceneFile(scenePath, scene, error), "scene/cafe.scene should load: " + error);

        require(scene.primitiveCount == kPrimitives, "Primitive count must match the Python builder");
        require(scene.vertices.size() == kVertices, "Vertex count must match the Python builder");
        require(scene.triangleCount() == kTriangles, "Triangle count must match the Python builder");
        require(scene.emissiveTriangleCount == kEmissiveTriangles, "Emissive triangle count must match");
        require(scene.smoothTriangleCount == kSmoothTriangles, "Smooth triangle count must match");
        require(scene.hasShadowRegion, "The cafe scene must name the region the shadow map covers");
        requireClose(scene.shadowLow.z, -10.6, 1e-5, "Shadow region far edge");
        requireClose(scene.shadowHigh.y, 1.8, 1e-5, "Shadow region top edge");
        requireClose(scene.shadowHigh.z, 5.2, 1e-5, "Shadow region near edge");

        const float boundsMin[3] = {scene.boundsMin.x, scene.boundsMin.y, scene.boundsMin.z};
        const float boundsMax[3] = {scene.boundsMax.x, scene.boundsMax.y, scene.boundsMax.z};
        double positionSum[3] = {0.0, 0.0, 0.0};
        for (const SceneVertex& vertex : scene.vertices) {
            for (int axis = 0; axis < 3; ++axis) {
                positionSum[axis] += vertex.position[axis];
            }
        }
        for (int axis = 0; axis < 3; ++axis) {
            requireClose(boundsMin[axis], kBoundsMin[axis], 1e-4, "Scene bounds minimum");
            requireClose(boundsMax[axis], kBoundsMax[axis], 1e-4, "Scene bounds maximum");
            // Tight enough that one misplaced prop shifts the sum past it, loose
            // enough to absorb float rounding over 66802 vertices.
            requireClose(positionSum[axis], kPositionSum[axis], 0.5, "Summed vertex positions");
        }

        // Cycles takes a triangle's facing from its winding, OpenGL from the
        // interpolated normal. Any triangle where the two disagree is lit from
        // behind in one renderer and not the other, so check every one.
        double areaSum = 0.0;
        std::size_t disagreements = 0;
        for (std::size_t triangle = 0; triangle < scene.indices.size(); triangle += 3) {
            const SceneVertex& a = scene.vertices.at(scene.indices[triangle]);
            const SceneVertex& b = scene.vertices.at(scene.indices[triangle + 1]);
            const SceneVertex& c = scene.vertices.at(scene.indices[triangle + 2]);
            const double u[3] = {b.position[0] - a.position[0], b.position[1] - a.position[1],
                                 b.position[2] - a.position[2]};
            const double v[3] = {c.position[0] - a.position[0], c.position[1] - a.position[1],
                                 c.position[2] - a.position[2]};
            const double cross[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                                     u[0] * v[1] - u[1] * v[0]};
            const double magnitude = std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
            areaSum += 0.5 * magnitude;
            if (magnitude < 1e-12) continue;  // Pole triangles are already skipped by the builder.
            double agreement = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double averaged = (a.normal[axis] + b.normal[axis] + c.normal[axis]) / 3.0;
                agreement += cross[axis] / magnitude * averaged;
            }
            if (agreement <= 0.0) ++disagreements;
        }
        require(disagreements == 0, "Winding and shading normals disagree on " +
                                        std::to_string(disagreements) + " triangle(s)");
        requireClose(areaSum, kAreaSum, 0.05, "Summed triangle area");

        // Every emissive vertex needs a colour to emit, or the bulbs render black.
        std::size_t emissiveVertices = 0;
        for (const SceneVertex& vertex : scene.vertices) {
            if (vertex.emission <= 0.0f) continue;
            ++emissiveVertices;
            require(vertex.albedo[0] + vertex.albedo[1] + vertex.albedo[2] > 0.0f,
                    "An emitter with a black albedo emits nothing");
        }
        require(emissiveVertices > 0, "The cafe scene must contain emitters");

        std::cout << "Scene grammar, transform order, winding, and cross-language geometry passed: "
                  << scene.primitiveCount << " primitives, " << scene.vertices.size() << " vertices, "
                  << scene.triangleCount() << " triangles.\n";
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "scene_file test failed: " << failure.what() << "\n";
        return 1;
    }
}
