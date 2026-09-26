#version 330 core

// Fullscreen triangle generated from gl_VertexID; no vertex buffer is needed
// beyond an empty VAO, which core profile still requires.

out vec2 vUV;

void main() {
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUV = corner;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
