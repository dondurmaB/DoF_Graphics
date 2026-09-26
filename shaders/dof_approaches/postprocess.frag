// Assembled by the host as: #version + shading.glsl + this file.
//
// Method 1: single-layer screen-space gather. One sharp pass provides color
// and depth; the circle of confusion of the center pixel alone decides how far
// the gather reaches. Neighbor depths are never consulted, which is exactly
// why this method bleeds across silhouettes.

in vec2 vUV;

out vec4 fragColor;

uniform sampler2D uColor;
uniform sampler2D uDepth;
uniform float uNear;
uniform float uFar;
uniform float uFocusDistance;
uniform float uFocalLength;
uniform float uSensorHeight;
uniform float uFNumber;
uniform float uMaxRadiusPixels;
uniform vec2 uResolution;
uniform int uTapCount;

float linearizeDepth(float rawDepth) {
    float ndc = rawDepth * 2.0 - 1.0;
    return (2.0 * uNear * uFar) / max(uFar + uNear - ndc * (uFar - uNear), 1e-6);
}

float cocRadiusPixels(float viewDepth) {
    float apertureDiameter = uFocalLength / max(uFNumber, 1e-4);
    float denominator = viewDepth * (uFocusDistance - uFocalLength);
    if (viewDepth <= 0.0 || uFocusDistance <= uFocalLength || abs(denominator) < 1e-6) return 0.0;
    float cocSensor = apertureDiameter * uFocalLength * (viewDepth - uFocusDistance) / denominator;
    float cocPixels = abs(cocSensor / uSensorHeight) * uResolution.y;
    // Thin-lens CoC is a diameter; the gather needs a radius.
    return min(0.5 * cocPixels, uMaxRadiusPixels);
}

void main() {
    vec3 center = texture(uColor, vUV).rgb;
    float radius = cocRadiusPixels(linearizeDepth(texture(uDepth, vUV).r));
    if (radius < 0.5) {
        fragColor = vec4(center, 1.0);
        return;
    }
    vec2 texel = 1.0 / uResolution;
    vec2 inset = 0.5 * texel;
    vec3 sum = center;
    float weight = 1.0;
    for (int i = 0; i < 64; ++i) {
        if (i >= uTapCount) break;
        float index = float(i) + 0.5;
        float r = sqrt(index / float(uTapCount));
        float angle = index * 2.39996323;
        vec2 offset = vec2(cos(angle), sin(angle)) * r;
        vec2 uv = clamp(vUV + offset * radius * texel, inset, vec2(1.0) - inset);
        sum += texture(uColor, uv).rgb;
        weight += 1.0;
    }
    fragColor = vec4(sum / weight, 1.0);
}
