#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

out vec3 interpolatedColor;

uniform float uOffsetX;

void main()
{
    vec3 position = aPos;
    position.x += uOffsetX;
    gl_Position = vec4(position, 1.0);

    // Vertex shader outputs are interpolated during rasterization.
    interpolatedColor = aColor;
}
