// Assembled by the host as: #version + shading.glsl + this file.
// Copies one scene pass into the accumulation target; GL_ONE/GL_ONE blending
// performs the accumulation.

in vec2 vUV;

out vec4 fragColor;

uniform sampler2D uSource;

void main() {
    fragColor = vec4(texture(uSource, vUV).rgb, 1.0);
}
