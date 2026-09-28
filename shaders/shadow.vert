#version 330 core

// Same vertex layout as basic.vert so this can render every mesh in the scene
// (cubes and the imported teapot) without a second VAO setup.
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uLightSpaceMatrix;

void main()
{
    // Only position matters here: this pass writes depth from the light's
    // point of view, so color/normal attributes are read but unused.
    gl_Position = uLightSpaceMatrix * uModel * vec4(aPos, 1.0);
}
