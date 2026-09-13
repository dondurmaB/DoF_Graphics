#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

out vec3 interpolatedColor;

uniform mat4 uTransform;

void main()
{
    // w = 1.0 makes this a position, so translation affects it in homogeneous coordinates.
    gl_Position = uTransform * vec4(aPos, 1.0);

    // Vertex shader outputs are interpolated during rasterization.
    interpolatedColor = aColor;
}
