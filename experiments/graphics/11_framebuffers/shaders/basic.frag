#version 330 core

in vec3 interpolatedColor;

out vec4 fragColor;

uniform float uIntensity;
uniform float uVisualizationMode;
uniform float uNearPlane;
uniform float uFarPlane;
uniform float uDepthVisualizationMax;

float linearizeDepth(float rawDepth)
{
    // window depth [0, 1] -> NDC depth [-1, 1]
    float zNdc = rawDepth * 2.0 - 1.0;

    return (2.0 * uNearPlane * uFarPlane) /
        (uFarPlane + uNearPlane - zNdc * (uFarPlane - uNearPlane));
}

void main()
{
    if (uVisualizationMode < 0.5) {
        fragColor = vec4(interpolatedColor * uIntensity, 1.0);
        return;
    }

    if (uVisualizationMode < 1.5) {
        // gl_FragCoord.z is the post-projection/window-space depth in the default range near [0, 1].
        float rawDepth = gl_FragCoord.z;
        fragColor = vec4(vec3(rawDepth), 1.0);
        return;
    }

    float linearDepth = linearizeDepth(gl_FragCoord.z);
    // uDepthVisualizationMax only controls grayscale display; it does not affect projection, depth testing, or reconstruction.
    float normalizedDepth = clamp(linearDepth / uDepthVisualizationMax, 0.0, 1.0);
    fragColor = vec4(vec3(normalizedDepth), 1.0);
}
