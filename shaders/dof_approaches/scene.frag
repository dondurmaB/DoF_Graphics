// Assembled by the host as: #version + shading.glsl + this file.

in vec3 vWorld;
in vec3 vNormal;

out vec4 fragColor;

uniform vec3 uColor;
uniform float uEmissive;
uniform float uChecker;

void main() {
    fragColor = vec4(shadeSurface(uColor, vNormal, vWorld, uEmissive, uChecker), 1.0);
}
