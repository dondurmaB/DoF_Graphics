#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;

out vec3 vWorldPosition;
out vec3 vWorldNormal;
out vec3 vLocalPosition;
out vec2 vUv;
out vec4 vShadowPosition;

uniform mat4 uModel;
uniform mat4 uViewProjection;
uniform mat4 uLightMatrix;

void main()
{
    vec4 worldPosition = uModel * vec4(aPosition, 1.0);
    mat3 normalMatrix = transpose(inverse(mat3(uModel)));

    vWorldPosition = worldPosition.xyz;
    vWorldNormal = normalize(normalMatrix * aNormal);
    vLocalPosition = aPosition;
    vUv = aUv;
    vShadowPosition = uLightMatrix * worldPosition;

    gl_Position = uViewProjection * worldPosition;
}
