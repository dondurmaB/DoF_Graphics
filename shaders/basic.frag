#version 330 core

in vec3 interpolatedAlbedo;
in vec3 worldPosition;
in vec3 worldNormal;
in vec4 fragPosLightSpace;
in float interpolatedEmission;

// Linear radiance, written into an RGBA16F attachment. Exposure and the sRGB
// transfer function are applied once at the very end of screen.frag, so the
// defocus gather in between averages light the way a lens does. Averaging
// display-encoded values instead is what makes naive DoF look grey and flat.
out vec4 fragColor;

uniform float uIntensity;
uniform bool uImportedMesh;
// The imported mesh has no per-vertex albedo, so it gets a flat one.
uniform vec3 uOverrideAlbedo;

// Directional light read from the selected scene file, the same file
// tools/raytraced_reference/render_dof.py configures the Cycles sun from.
uniform vec3 uLightDirection; // points FROM the surface TOWARD the light, world space
uniform vec3 uLightColor;
uniform float uLightEnergy;   // Irradiance in W/m^2, as Blender's sun strength.
// Uniform sky radiance: ambient colour * strength from the scene file, which is
// also the OpenGL clear colour and the Cycles world background.
uniform vec3 uSkyRadiance;

uniform bool uUseShadows;
uniform sampler2D uShadowMap;

const float PI = 3.14159265359;

// PCF (percentage-closer filtering): average several neighbouring shadow-map
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
        // guessing, so geometry beyond the scene file's shadow region is not
        // clipped dark. That region is chosen in build_alley.py to contain
        // everything the camera can see.
        return 0.0;
    }

    // Slope-scaled bias fights the last of the shadow acne. Most of the work is
    // done by the normal offset in basic.vert, so this can stay small and not
    // detach contact shadows under the crates.
    float bias = max(0.0012 * (1.0 - nDotL), 0.0003);

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
    vec3 albedo = uImportedMesh ? uOverrideAlbedo : interpolatedAlbedo;

    // Emitters (window panes, bulbs, the neon sign) are radiance sources, not
    // surfaces: they are not shaded, and in Cycles they are marked invisible to
    // diffuse rays with max_bounces 0 so they do not light anything either.
    // Both renderers therefore show the same self-lit patch of colour.
    if (!uImportedMesh && interpolatedEmission > 0.0) {
        fragColor = vec4(albedo * interpolatedEmission * uIntensity, 1.0);
        return;
    }

    vec3 normal = normalize(worldNormal);
    vec3 lightDir = normalize(uLightDirection);
    float nDotL = max(dot(normal, lightDir), 0.0);

    float shadow = (uUseShadows) ? sampleShadow(fragPosLightSpace, nDotL) : 0.0;

    // Outgoing radiance of a Lambertian surface: albedo/pi times the incoming
    // irradiance. The 1/pi is what makes this agree with a Cycles Diffuse BSDF
    // lit by a sun of the same strength, instead of being an arbitrary "looks
    // about right" constant. The sky term is already a radiance, so a uniform
    // hemisphere of it reflects back as albedo * skyRadiance with no 1/pi.
    //
    // Cycles matched mode adds this same non-physical, unoccluded fill as
    // primary-camera-only emission. It is never allowed to illuminate another
    // surface. Direct sunlight shares this Lambert equation; visibility still
    // differs (biased PCF shadow map versus Cycles ray intersections).
    vec3 sunIrradiance = uLightColor * uLightEnergy * nDotL * (1.0 - shadow);
    vec3 radiance = albedo * (uSkyRadiance + sunIrradiance / PI);

    fragColor = vec4(radiance * uIntensity, 1.0);
}
