#pragma once

#include <cmath>
#include <stdexcept>

// Both inputs are millimeters; their ratio is unitless. The result is a vertical angle.
inline float physicalVerticalFovRadians(float focalLengthMillimeters, float sensorHeightMillimeters) {
    if (!std::isfinite(focalLengthMillimeters) || focalLengthMillimeters <= 0.0f ||
        !std::isfinite(sensorHeightMillimeters) || sensorHeightMillimeters <= 0.0f) {
        throw std::invalid_argument("Physical focal length and sensor height must be finite and positive.");
    }
    return 2.0f * std::atan(sensorHeightMillimeters / (2.0f * focalLengthMillimeters));
}
