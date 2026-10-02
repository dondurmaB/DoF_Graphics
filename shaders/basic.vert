#version 330 core

// Vertex layout shared by both VAOs so one program draws the whole scene:
//   0 = position          (SceneFile.cpp and Mesh.cpp)
//   1 = linear albedo     (SceneFile.cpp only; disabled for imported meshes)
//   2 = normal            (both)
//   3 = texture UV        (Mesh.cpp only; unused here, reserved so it is not clobbered)
//   4 = emission          (SceneFile.cpp only; a disabled attribute reads 0)
// A disabled attribute supplies a constant, which is why the imported teapot
// needs no albedo or emission data and takes uOverrideAlbedo instead.
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aAlbedo;
layout(location = 2) in vec3 aNormal;
layout(location = 4) in float aEmission;

out vec3 interpolatedAlbedo;
out vec3 worldPosition;
out vec3 worldNormal;
out vec4 fragPosLightSpace;
out float interpolatedEmission;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uLightSpaceMatrix;
// World-space width of one shadow-map texel. Used for normal-offset shadows:
// see the shadow lookup below.
uniform float uShadowWorldTexelSize;

void main()
{
    vec4 worldPos4 = uModel * vec4(aPos, 1.0);
    worldPosition = worldPos4.xyz;

    // Written as P * V * M * vertex, but the vertex experiences Model -> View -> Projection.
    gl_Position = uProjection * uView * worldPos4;

    // Vertex shader outputs are interpolated during rasterization.
    interpolatedAlbedo = aAlbedo;
    interpolatedEmission = aEmission;

    // Inverse-transpose keeps normals perpendicular to the surface under
    // non-uniform model scale. The scene mesh is already baked into world space
    // (uModel is the identity there), so this only really matters for the teapot.
    worldNormal = mat3(transpose(inverse(uModel))) * aNormal;

    // Normal-offset shadows: look the shadow map up from a point pushed out
    // along the surface normal by about one texel, rather than from the surface
    // itself. The alley is full of shallow relief (mortar courses, protruding
    // bricks, railings) where a depth-only bias either leaves acne or makes
    // contact shadows float, and this fixes both at once.
    vec3 offsetPosition = worldPosition + normalize(worldNormal) * uShadowWorldTexelSize * 1.5;
    fragPosLightSpace = uLightSpaceMatrix * vec4(offsetPosition, 1.0);
}
