#version 330 core

in vec3 interpolatedColor;
in vec3 worldPosition;
in vec3 worldNormal;
in vec4 fragPosLightSpace;

out vec4 fragColor;

uniform float uIntensity;
uniform bool uImportedMesh;

// Directional light, matching tools/raytraced_reference/render_dof.py's LIGHT_DIRECTION_GL
// so the OpenGL raster pass and the Cycles ground truth are lit the same way.
uniform vec3 uLightDirection; // points FROM the surface TOWARD the light, world space
uniform vec3 uLightColor;
uniform float uAmbientStrength;

uniform bool uUseShadows;
uniform sampler2D uShadowMap;

// PCF (percentage-closer filtering): average several neighboring shadow-map
// texels instead of one, so shadow edges are soft rather than aliased.
float sampleShadow(vec4 lightSpacePos, float nDotL)
{
    // Perspective divide: not strictly required for an orthographic light
    // projection (w stays 1), but kept so this still works if the light
    // projection is ever swapped for a perspective one.
    vec3 projected = lightSpacePos.xyz / lightSpacePos.w;
    projected = projected * 0.5 + 0.5; // NDC [-1,1] -> texture space [0,1]

    if (projected.z > 1.0 || projected.x < 0.0 || projected.x > 1.0 ||
        projected.y < 0.0 || projected.y > 1.0) {
        // Outside the light's frustum/far plane: treat as unshadowed rather than
        // guessing, so geometry far from the tracked scene bounds is not clipped dark.
        return 0.0;
    }

    // Slope-scaled bias fights shadow acne (self-shadowing) without a fixed
    // constant being too thin on grazing surfaces or too thick on flat ones.
    float bias = max(0.0025 * (1.0 - nDotL), 0.0006);

    float shadow = 0.0;
    vec2 texelSize = 1.0 / vec2(textureSize(uShadowMap, 0));
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float closestDepth = texture(uShadowMap, projected.xy + vec2(x, y) * texelSize).r;
            shadow += (projected.z - bias) > closestDepth ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
}

void main()
{
    vec3 albedo = interpolatedColor;
    if (uImportedMesh) {
        albedo = vec3(0.85, 0.65, 0.35);
    }

    vec3 normal = normalize(worldNormal);
    vec3 lightDir = normalize(uLightDirection);
    float nDotL = max(dot(normal, lightDir), 0.0);

    float shadow = (uUseShadows) ? sampleShadow(fragPosLightSpace, nDotL) : 0.0;

    // Ambient keeps unlit/shadowed faces visible instead of pure black.
    // Diffuse is attenuated by (1 - shadow) so fully shadowed fragments fall back to ambient only.
    vec3 lighting = uAmbientStrength + (1.0 - shadow) * (1.0 - uAmbientStrength) * nDotL * uLightColor;
    vec3 color = albedo * lighting;

    fragColor = vec4(color * uIntensity, 1.0);
}
