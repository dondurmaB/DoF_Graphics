#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec3 aNormal;

out vec3 interpolatedColor;
out vec3 worldPosition;
out vec3 worldNormal;
out vec4 fragPosLightSpace;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uLightSpaceMatrix;
uniform bool uImportedMesh;

void main()
{
    vec4 worldPos4 = uModel * vec4(aPos, 1.0);
    worldPosition = worldPos4.xyz;

    // Written as P * V * M * vertex, but the vertex experiences Model -> View -> Projection.
    gl_Position = uProjection * uView * worldPos4;

    // Vertex shader outputs are interpolated during rasterization.
    interpolatedColor = aColor;

    // Every draw now supplies a real normal: hardcoded per-face normals for the
    // cubes (attribute 2, added alongside position/color) and authored/flat
    // normals for imported meshes (Mesh.cpp attribute 2). Inverse-transpose keeps
    // normals perpendicular to the surface under non-uniform model scale.
    worldNormal = mat3(transpose(inverse(uModel))) * aNormal;

    // Project into the light's clip space so the fragment shader can sample the shadow map.
    fragPosLightSpace = uLightSpaceMatrix * worldPos4;
}
