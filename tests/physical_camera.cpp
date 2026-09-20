#include "PhysicalCamera.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <limits>

int main() {
    const float fov = physicalVerticalFovRadians(50.0f, 24.0f);
    if (std::abs(glm::degrees(fov) - 26.991467f) > 0.0001f) return 1;
    if (!(physicalVerticalFovRadians(85.0f, 24.0f) < fov)) return 2;
    if (!(physicalVerticalFovRadians(50.0f, 36.0f) > fov)) return 3;
    // The physical sensor's top-right corner projects to NDC (1,1), even non-square.
    for (float aspect : {1.0f, 16.0f / 9.0f, 0.75f}) {
        const auto projection = glm::perspective(fov, aspect, 0.1f, 100.0f);
        const auto clip = projection * glm::vec4(0.24f * aspect * 5.0f, 0.24f * 5.0f, -5.0f, 1.0f);
        if (std::abs(clip.x / clip.w - 1.0f) > 0.00001f ||
            std::abs(clip.y / clip.w - 1.0f) > 0.00001f) return 4;
    }
    for (float invalid : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        for (bool invalidLens : {false, true}) {
            try {
                physicalVerticalFovRadians(invalidLens ? invalid : 50.0f, invalidLens ? 24.0f : invalid);
                return 5;
            } catch (const std::invalid_argument&) {}
        }
    }
    std::cout << "Physical FOV, lens/sensor response, non-square projection and invalid inputs passed.\n";
}
