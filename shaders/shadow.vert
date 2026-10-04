#version 330 core

// Same vertex layout as basic.vert so this pass can render every mesh in the
// scene (the baked alley and the imported teapot) without a second VAO setup.
// Only position is read; the other attributes exist to keep the layouts equal.
layout(location = 0) in vec3 aPos;

uniform mat4 uModel;
uniform mat4 uLightSpaceMatrix;

void main()
{
    // This pass writes depth from the light's point of view, so colour,
    // normal and emission are not needed and are not declared.
    gl_Position = uLightSpaceMatrix * uModel * vec4(aPos, 1.0);
}
