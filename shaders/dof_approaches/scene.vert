#version 330 core

// Geometry is baked into world space on the CPU, so the model matrix is the
// identity and only the view-projection is uploaded per draw.

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uViewProjection;

out vec3 vWorld;
out vec3 vNormal;

void main() {
    vWorld = aPosition;
    vNormal = aNormal;
    gl_Position = uViewProjection * vec4(aPosition, 1.0);
}
