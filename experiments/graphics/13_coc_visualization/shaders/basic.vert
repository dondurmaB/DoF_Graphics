#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

out vec3 interpolatedColor;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

void main()
{
    // Written as P * V * M * vertex, but the vertex experiences Model -> View -> Projection.
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);

    // Vertex shader outputs are interpolated during rasterization.
    interpolatedColor = aColor;
}
