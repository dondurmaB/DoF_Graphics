#pragma once

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace dof_scene {

struct Camera {
    glm::vec3 position{0.25f, 1.55f, 3.8f};
    float yaw = -122.0f;
    float pitch = -7.0f;

    glm::vec3 forward() const {
        const float y = glm::radians(yaw), p = glm::radians(pitch);
        return glm::normalize(glm::vec3(std::cos(y) * std::cos(p), std::sin(p), std::sin(y) * std::cos(p)));
    }
    glm::vec3 right() const { return glm::normalize(glm::cross(forward(), glm::vec3(0, 1, 0))); }
    glm::mat4 view() const { return glm::lookAt(position, position + forward(), glm::vec3(0, 1, 0)); }
    void lookAt(const glm::vec3& target) {
        glm::vec3 direction = glm::normalize(target - position);
        yaw = glm::degrees(std::atan2(direction.z, direction.x));
        pitch = glm::degrees(std::asin(std::clamp(direction.y, -1.0f, 1.0f)));
    }
    void rotate(float dx, float dy) {
        yaw += dx * 0.12f;
        pitch = std::clamp(pitch - dy * 0.12f, -85.0f, 85.0f);
    }
};

struct Lens {
    float focalLengthMm = 50.0f;
    float sensorHeightMm = 24.0f;
    float fNumber = 1.2f;
    float focusDistance = 4.0f;
    float nearPlane = 0.05f;
    float farPlane = 40.0f;

    float verticalFov() const { return 2.0f * std::atan(sensorHeightMm / (2.0f * focalLengthMm)); }
    float linearDepth(float rawDepth) const {
        return 2.0f * nearPlane * farPlane /
            (farPlane + nearPlane - (2.0f * rawDepth - 1.0f) * (farPlane - nearPlane));
    }
    void focusAt(float depth) {
        focusDistance = std::clamp(depth, std::max(nearPlane + 0.01f, focalLengthMm * 0.001f + 0.01f), farPlane);
    }
    float signedRadiusPixels(float depth, float height) const {
        const float focal = focalLengthMm * 0.001f;
        if (depth <= 0 || focusDistance <= focal || fNumber <= 0 || sensorHeightMm <= 0 || height <= 0) return 0;
        // Thin-lens CoC is a diameter; the gather footprint needs its radius.
        return 0.5f * (focal * focal / fNumber) * (depth - focusDistance) /
            (depth * (focusDistance - focal)) * height / (sensorHeightMm * 0.001f);
    }
};

} // namespace dof_scene
