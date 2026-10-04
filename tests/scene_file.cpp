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

// The alley scene is a committed build artefact, so the test reads the real
// file rather than a fixture: a regenerated scene must update the numbers here.
const std::size_t kPrimitives = 2551;
const std::size_t kVertices = 66802;
const std::size_t kTriangles = 38572;
const std::size_t kEmissiveTriangles = 5076;
const std::size_t kSmoothTriangles = 7148;
const double kBoundsMin[3] = {-15.5, -2.55, -47.0};
const double kBoundsMax[3] = {17.5, 20.65, 9.5};
const double kPositionSum[3] = {2229.83502904, 109495.9536, -916767.8813};
const double kAreaSum = 6184.47785031;

const std::size_t kCafePrimitives = 1415;
const std::size_t kCafeVertices = 132875;
const std::size_t kCafeTriangles = 131636;
const std::size_t kCafeEmissiveTriangles = 18384;
const std::size_t kCafeSmoothTriangles = 79696;
const double kCafeBoundsMin[3] = {-3.12, -1.59, -10.12};
const double kCafeBoundsMax[3] = {3.12, 1.7, 5.1};
const double kCafePositionSum[3] = {-16080.1974176, -90379.2084028, -341348.810441};
const double kCafeAreaSum = 842.636448915;

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

        require(rejects("version 1\n", "no geometry"), "Empty scene rejected");
        require(rejects("version 1 extra\nbox\n", "needs 1 number(s)"), "Version takes exactly one number");
        require(rejects("version 1.9\nbox\n", "is not 1"), "Fractional version rejected");
        require(rejects("version 1\ncyl taper -1\n", "taper must"), "Negative taper rejected");
        require(rejects("version 1\nbox taper 1\n", "not valid here"), "Only cylinders taper");
        {
            SceneDescription plain, explicitDefault, cone;
            std::string error;
            require(parseSceneText("version 1\ncyl size 2 3 4 seg 17\n", plain, error), error);
            require(parseSceneText("version 1\ncyl size 2 3 4 seg 17 taper 1\n", explicitDefault, error),
                    error);
            require(plain.indices == explicitDefault.indices, "Default taper preserves index bytes");
            require(plain.vertices.size() == explicitDefault.vertices.size(),
                    "Default taper preserves count");
            require(std::memcmp(plain.vertices.data(), explicitDefault.vertices.data(),
                                plain.vertices.size() * sizeof(SceneVertex)) == 0,
                    "Default taper preserves every vertex byte");
            require(parseSceneText("version 1\ncyl seg 17 taper 0\n", cone, error), error);
            require(cone.triangleCount() == 34, "Cone skips collapsed triangles and top cap");
            requireClose(cone.vertices[0].normal[1], .5 / std::sqrt(1.25), 1e-6, "Derived cone side normal");
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

        for (const bool cafe : {false, true}) {
            const auto expectedPrimitives = cafe ? kCafePrimitives : kPrimitives;
            const auto expectedVertices = cafe ? kCafeVertices : kVertices;
            const auto expectedTriangles = cafe ? kCafeTriangles : kTriangles;
            const auto expectedEmissiveTriangles = cafe ? kCafeEmissiveTriangles : kEmissiveTriangles;
            const auto expectedSmoothTriangles = cafe ? kCafeSmoothTriangles : kSmoothTriangles;
            const auto expectedBoundsMin = cafe ? kCafeBoundsMin : kBoundsMin;
            const auto expectedBoundsMax = cafe ? kCafeBoundsMax : kBoundsMax;
            const auto expectedPositionSum = cafe ? kCafePositionSum : kPositionSum;
            const auto expectedAreaSum = cafe ? kCafeAreaSum : kAreaSum;
            // --- The real scene, against the Python builder's numbers.
            const std::filesystem::path scenePath =
                std::filesystem::path(PROJECT_SOURCE_DIR) / "scene" / (cafe ? "cafe.scene" : "alley.scene");
            SceneDescription scene;
            std::string error;
            require(loadSceneFile(scenePath, scene, error), "scene/alley.scene should load: " + error);

            require(scene.primitiveCount == expectedPrimitives,
                    "Primitive count must match the Python builder");
            require(scene.vertices.size() == expectedVertices, "Vertex count must match the Python builder");
            require(scene.triangleCount() == expectedTriangles,
                    "Triangle count must match the Python builder");
            require(scene.emissiveTriangleCount == expectedEmissiveTriangles,
                    "Emissive triangle count must match");
            require(scene.smoothTriangleCount == expectedSmoothTriangles, "Smooth triangle count must match");
            require(scene.hasShadowRegion, "The alley scene must name the region the shadow map covers");
            requireClose(scene.shadowLow.z, cafe ? -10.5 : -34.0, 1e-5, "Shadow region far edge");
            requireClose(scene.shadowHigh.y, cafe ? 1.8 : 8.0, 1e-5, "Shadow region top edge");

            const float boundsMin[3] = {scene.boundsMin.x, scene.boundsMin.y, scene.boundsMin.z};
            const float boundsMax[3] = {scene.boundsMax.x, scene.boundsMax.y, scene.boundsMax.z};
            double positionSum[3] = {0.0, 0.0, 0.0};
            for (const SceneVertex& vertex : scene.vertices) {
                for (int axis = 0; axis < 3; ++axis) {
                    positionSum[axis] += vertex.position[axis];
                }
            }
            for (int axis = 0; axis < 3; ++axis) {
                requireClose(boundsMin[axis], expectedBoundsMin[axis], 1e-4, "Scene bounds minimum");
                requireClose(boundsMax[axis], expectedBoundsMax[axis], 1e-4, "Scene bounds maximum");
                // Tight enough that one misplaced prop shifts the sum past it, loose
                // enough to absorb float rounding over 66802 vertices.
                requireClose(positionSum[axis], expectedPositionSum[axis], 0.5, "Summed vertex positions");
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
                const double magnitude =
                    std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
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
            requireClose(areaSum, expectedAreaSum, 0.05, "Summed triangle area");

            // Every emissive vertex needs a colour to emit, or the bulbs render black.
            std::size_t emissiveVertices = 0;
            for (const SceneVertex& vertex : scene.vertices) {
                if (vertex.emission <= 0.0f) continue;
                ++emissiveVertices;
                require(vertex.albedo[0] + vertex.albedo[1] + vertex.albedo[2] > 0.0f,
                        "An emitter with a black albedo emits nothing");
            }
            require(emissiveVertices > 0, "The alley scene must contain emitters");

            std::cout << "Scene grammar, transform order, winding, and cross-language geometry passed: "
                      << scene.primitiveCount << " primitives, " << scene.vertices.size() << " vertices, "
                      << scene.triangleCount() << " triangles.\n";
        }
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "scene_file test failed: " << failure.what() << "\n";
        return 1;
    }
}
