#version 330 core

in vec3 interpolatedColor;
in vec3 worldNormal;

out vec4 fragColor;

uniform float uIntensity;
uniform bool uImportedMesh;

void main()
{
    vec3 color = interpolatedColor;
    if (uImportedMesh) {
        vec3 lightDirection = normalize(vec3(-0.4, 0.8, 0.6));
        float brightness = 0.25 + 0.75 * max(dot(normalize(worldNormal), lightDirection), 0.0);
        color = vec3(0.85, 0.65, 0.35) * brightness;
    }
    fragColor = vec4(color * uIntensity, 1.0);
}
