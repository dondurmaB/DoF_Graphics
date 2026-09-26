#include "dof_scene/camera.hpp"
#include "dof_scene/scene.hpp"

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

namespace {

constexpr float kEpsilon = 1.0e-4f;

struct TestFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw TestFailure(message);
    }
}

void checkNear(float actual, float expected, float tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw TestFailure(message + " (actual " + std::to_string(actual) +
                          ", expected " + std::to_string(expected) + ")");
    }
}

bool finite(float value) {
    return std::isfinite(value);
}

bool finiteVec(const glm::vec2& value) {
    return finite(value.x) && finite(value.y);
}

bool finiteVec(const glm::vec3& value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finiteMat(const glm::mat4& value) {
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            if (!finite(value[column][row])) {
                return false;
            }
        }
    }
    return true;
}

float rawDepthForLinearDepth(const dof_scene::Lens& lens, float linearDepth) {
    const float nearPlane = lens.nearPlane;
    const float farPlane = lens.farPlane;
    const float ndcDepth = (farPlane + nearPlane -
                            (2.0f * nearPlane * farPlane / linearDepth)) /
                           (farPlane - nearPlane);
    return ndcDepth * 0.5f + 0.5f;
}

void cameraBasisIsNormalizedAndOrthogonal() {
    dof_scene::Camera camera;
    camera.yaw = -37.0f;
    camera.pitch = 23.0f;

    const glm::vec3 forward = camera.forward();
    const glm::vec3 right = camera.right();

    checkNear(glm::length(forward), 1.0f, kEpsilon, "forward vector is normalized");
    checkNear(glm::length(right), 1.0f, kEpsilon, "right vector is normalized");
    checkNear(glm::dot(forward, right), 0.0f, kEpsilon, "forward and right are orthogonal");
}

void cameraRotateClampsPitch() {
    dof_scene::Camera camera;

    camera.rotate(0.0f, 2000.0f);
    checkNear(camera.pitch, -85.0f, kEpsilon, "large upward rotation clamps pitch down");

    camera.rotate(0.0f, -4000.0f);
    checkNear(camera.pitch, 85.0f, kEpsilon, "large downward rotation clamps pitch up");
}

void cameraLookAtCentersTargetInViewSpace() {
    dof_scene::Camera camera;
    camera.position = glm::vec3(2.5f, 1.65f, 3.5f);
    const glm::vec3 target(0.0f, 1.08f, -0.55f);

    camera.lookAt(target);
    const glm::vec4 viewTarget = camera.view() * glm::vec4(target, 1.0f);

    checkNear(viewTarget.x, 0.0f, 2.0e-4f, "lookAt target is centered horizontally");
    checkNear(viewTarget.y, 0.0f, 2.0e-4f, "lookAt target is centered vertically");
    check(viewTarget.z < 0.0f, "lookAt target lies in front of the camera on negative view z");
}

void lensReconstructsNearAndFarDepths() {
    const dof_scene::Lens lens;

    checkNear(lens.linearDepth(0.0f), lens.nearPlane, 1.0e-5f, "raw depth 0 reconstructs near plane");
    checkNear(lens.linearDepth(1.0f), lens.farPlane, 2.0e-3f, "raw depth 1 reconstructs far plane");
}

void lensRoundTripsProjectedDepth() {
    const dof_scene::Lens lens;
    const float requestedDepth = 7.25f;

    const float rawDepth = rawDepthForLinearDepth(lens, requestedDepth);
    const float reconstructedDepth = lens.linearDepth(rawDepth);

    check(rawDepth > 0.0f && rawDepth < 1.0f, "projected mid-range depth stays inside raw depth range");
    checkNear(reconstructedDepth, requestedDepth, 1.0e-3f, "projected depth round trips through reconstruction");
}

void lensVerticalFovRespondsToFocalLength() {
    dof_scene::Lens wideLens;
    dof_scene::Lens telephotoLens;
    wideLens.focalLengthMm = 24.0f;
    telephotoLens.focalLengthMm = 85.0f;

    check(wideLens.verticalFov() > telephotoLens.verticalFov(),
          "shorter focal length produces a wider vertical field of view");
    check(wideLens.verticalFov() > 0.0f && wideLens.verticalFov() < 3.1415927f,
          "vertical field of view remains physically valid");
}

void lensCoCRadiusIsZeroAtFocus() {
    dof_scene::Lens lens;
    lens.focusDistance = 3.5f;

    checkNear(lens.signedRadiusPixels(lens.focusDistance, 1080.0f), 0.0f, kEpsilon,
              "circle of confusion radius is zero at focus distance");
}

void lensCoCRadiusSignsDistinguishNearAndFarBlur() {
    dof_scene::Lens lens;
    lens.focusDistance = 4.0f;

    check(lens.signedRadiusPixels(2.0f, 1080.0f) < 0.0f,
          "nearer-than-focus depth has negative signed blur radius");
    check(lens.signedRadiusPixels(8.0f, 1080.0f) > 0.0f,
          "farther-than-focus depth has positive signed blur radius");
}

void lensCoCRadiusScalesWithApertureAndResolution() {
    dof_scene::Lens base;
    base.focusDistance = 4.0f;

    dof_scene::Lens widerAperture = base;
    widerAperture.fNumber = base.fNumber * 0.5f;

    const float baseRadius = std::abs(base.signedRadiusPixels(8.0f, 720.0f));
    const float widerRadius = std::abs(widerAperture.signedRadiusPixels(8.0f, 720.0f));
    const float tallerFrameRadius = std::abs(base.signedRadiusPixels(8.0f, 1440.0f));

    checkNear(widerRadius / baseRadius, 2.0f, 1.0e-3f,
              "halving f-number doubles blur radius");
    checkNear(tallerFrameRadius / baseRadius, 2.0f, 1.0e-3f,
              "doubling render height doubles blur radius in pixels");
}

void lensGuardsInvalidCoCInputsAndClampsFocus() {
    dof_scene::Lens lens;

    checkNear(lens.signedRadiusPixels(-1.0f, 1080.0f), 0.0f, kEpsilon,
              "negative depth returns zero blur");

    lens.fNumber = 0.0f;
    checkNear(lens.signedRadiusPixels(8.0f, 1080.0f), 0.0f, kEpsilon,
              "invalid f-number returns zero blur");

    lens = dof_scene::Lens{};
    lens.sensorHeightMm = 0.0f;
    checkNear(lens.signedRadiusPixels(8.0f, 1080.0f), 0.0f, kEpsilon,
              "invalid sensor height returns zero blur");

    lens = dof_scene::Lens{};
    checkNear(lens.signedRadiusPixels(8.0f, 0.0f), 0.0f, kEpsilon,
              "invalid render height returns zero blur");

    lens = dof_scene::Lens{};
    const float minimumFocusDistance =
        std::max(lens.nearPlane + 0.01f, lens.focalLengthMm * 0.001f + 0.01f);
    lens.focusAt(-10.0f);
    checkNear(lens.focusDistance, minimumFocusDistance, kEpsilon,
              "focusAt clamps below the minimum focus distance");

    lens.focusAt(lens.farPlane * 10.0f);
    checkNear(lens.focusDistance, lens.farPlane, kEpsilon,
              "focusAt clamps above the far plane");
}

void sceneMeshesHaveFiniteVerticesAndUnitNormals() {
    const dof_scene::Scene scene = dof_scene::createCafeScene();

    for (const dof_scene::MeshData& mesh : scene.meshes) {
        check(!mesh.vertices.empty(), "scene mesh has vertices");
        for (const dof_scene::Vertex& vertex : mesh.vertices) {
            check(finiteVec(vertex.position), "mesh vertex position is finite");
            check(finiteVec(vertex.normal), "mesh vertex normal is finite");
            check(finiteVec(vertex.uv), "mesh vertex uv is finite");
            checkNear(glm::length(vertex.normal), 1.0f, 2.0e-3f, "mesh vertex normal has unit length");
        }
    }
}

void sceneMeshIndicesAreValidTriangles() {
    const dof_scene::Scene scene = dof_scene::createCafeScene();

    for (const dof_scene::MeshData& mesh : scene.meshes) {
        check(!mesh.indices.empty(), "scene mesh has indices");
        check(mesh.indices.size() % 3 == 0, "scene mesh indices form triangles");
        for (const unsigned int index : mesh.indices) {
            check(index < mesh.vertices.size(), "scene mesh index references an existing vertex");
        }
    }
}

void sceneMeshTrianglesAreNondegenerateAndOutwardWound() {
    const dof_scene::Scene scene = dof_scene::createCafeScene();

    for (const dof_scene::MeshData& mesh : scene.meshes) {
        for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
            const dof_scene::Vertex& a = mesh.vertices[mesh.indices[i + 0]];
            const dof_scene::Vertex& b = mesh.vertices[mesh.indices[i + 1]];
            const dof_scene::Vertex& c = mesh.vertices[mesh.indices[i + 2]];

            const glm::vec3 faceNormal = glm::cross(b.position - a.position, c.position - a.position);
            const float faceAreaTwice = glm::length(faceNormal);
            check(faceAreaTwice > 1.0e-6f, "scene mesh triangle is nondegenerate");

            const glm::vec3 averageNormal = glm::normalize(a.normal + b.normal + c.normal);
            check(glm::dot(glm::normalize(faceNormal), averageNormal) > 0.25f,
                  "scene mesh triangle winding agrees with averaged outward normals");
        }
    }
}

void sceneObjectsHaveFiniteNonsingularTransforms() {
    const dof_scene::Scene scene = dof_scene::createCafeScene();

    check(!scene.objects.empty(), "cafe scene has renderable objects");
    for (const dof_scene::Object& object : scene.objects) {
        const int shapeIndex = static_cast<int>(object.shape);
        check(shapeIndex >= 0 && shapeIndex < static_cast<int>(scene.meshes.size()),
              "object shape maps to an available mesh");
        check(finiteMat(object.model), "object transform matrix is finite");
        check(std::abs(glm::determinant(object.model)) > 1.0e-8f,
              "object transform matrix is nonsingular");
        check(finiteVec(object.material.color), "object material color is finite");
        check(finiteVec(object.material.emission), "object material emission is finite");
        check(finite(object.material.roughness), "object material roughness is finite");
        check(finite(object.material.metallic), "object material metallic is finite");
    }
}

void sceneFocusTargetIsVisibleWhenCameraLooksAtIt() {
    const dof_scene::Scene scene = dof_scene::createCafeScene();
    dof_scene::Camera camera;
    dof_scene::Lens lens;
    camera.lookAt(scene.focusPoint);

    const glm::vec4 viewFocus = camera.view() * glm::vec4(scene.focusPoint, 1.0f);
    const float viewDepth = -viewFocus.z;
    const glm::mat4 projection =
        glm::perspective(lens.verticalFov(), 16.0f / 9.0f, lens.nearPlane, lens.farPlane);
    const glm::vec4 clipFocus = projection * viewFocus;
    const glm::vec3 ndcFocus = glm::vec3(clipFocus) / clipFocus.w;

    check(finiteVec(scene.focusPoint), "scene focus point is finite");
    check(viewDepth > lens.nearPlane && viewDepth < lens.farPlane,
          "scene focus point lies between the near and far planes");
    checkNear(ndcFocus.x, 0.0f, 2.0e-4f, "lookAt places scene focus point at screen center x");
    checkNear(ndcFocus.y, 0.0f, 2.0e-4f, "lookAt places scene focus point at screen center y");
    check(ndcFocus.z >= -1.0f && ndcFocus.z <= 1.0f,
          "scene focus point lies inside the camera frustum");
}

struct TestCase {
    const char* name;
    void (*run)();
};

} // namespace

int main() {
    const std::vector<TestCase> tests = {
        {"camera basis is normalized and orthogonal", cameraBasisIsNormalizedAndOrthogonal},
        {"camera rotate clamps pitch", cameraRotateClampsPitch},
        {"camera lookAt centers target in view space", cameraLookAtCentersTargetInViewSpace},
        {"lens reconstructs near and far depths", lensReconstructsNearAndFarDepths},
        {"lens round trips projected depth", lensRoundTripsProjectedDepth},
        {"lens vertical FOV responds to focal length", lensVerticalFovRespondsToFocalLength},
        {"lens CoC radius is zero at focus", lensCoCRadiusIsZeroAtFocus},
        {"lens CoC radius signs distinguish near and far blur", lensCoCRadiusSignsDistinguishNearAndFarBlur},
        {"lens CoC radius scales with aperture and resolution", lensCoCRadiusScalesWithApertureAndResolution},
        {"lens guards invalid CoC inputs and clamps focus", lensGuardsInvalidCoCInputsAndClampsFocus},
        {"scene meshes have finite vertices and unit normals", sceneMeshesHaveFiniteVerticesAndUnitNormals},
        {"scene mesh indices are valid triangles", sceneMeshIndicesAreValidTriangles},
        {"scene mesh triangles are nondegenerate and outward wound", sceneMeshTrianglesAreNondegenerateAndOutwardWound},
        {"scene objects have finite nonsingular transforms", sceneObjectsHaveFiniteNonsingularTransforms},
        {"scene focus target is visible when camera looks at it", sceneFocusTargetIsVisibleWhenCameraLooksAtIt},
    };

    int failures = 0;
    for (const TestCase& test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        }
    }

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }

    std::cout << tests.size() << " test(s) passed\n";
    return 0;
}
