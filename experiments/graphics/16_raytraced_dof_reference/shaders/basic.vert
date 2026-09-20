#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec3 aNormal;

out vec3 interpolatedColor;
out vec3 worldNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform bool uImportedMesh;

void main()
{
    // Written as P * V * M * vertex, but the vertex experiences Model -> View -> Projection.
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);

    // Vertex shader outputs are interpolated during rasterization.
    interpolatedColor = aColor;
    worldNormal = vec3(0.0);
    if (uImportedMesh) {
        // Inverse-transpose keeps normals perpendicular under non-uniform model scale.
        worldNormal = mat3(transpose(inverse(uModel))) * aNormal;
    }
}
