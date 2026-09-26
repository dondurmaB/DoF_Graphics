// Assembled by the host as: #version + shading.glsl + this file.
// Divides the accumulated radiance by the sample count, then tone maps and
// converts to sRGB for display.

in vec2 vUV;

out vec4 fragColor;

uniform sampler2D uAccum;
uniform float uSampleCount;

void main() {
    vec3 color = texture(uAccum, vUV).rgb / max(uSampleCount, 1.0);
    color = max(color, vec3(0.0));
    color = color * (2.51 * color + 0.03) / (color * (2.43 * color + 0.59) + 0.14);
    color = clamp(color, 0.0, 1.0);
    fragColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}
