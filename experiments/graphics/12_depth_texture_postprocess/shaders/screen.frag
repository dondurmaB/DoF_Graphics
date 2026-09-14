#version 330 core

in vec2 texCoord;

out vec4 fragColor;

uniform sampler2D uSceneColor;
uniform sampler2D uSceneDepth;
uniform int uScreenMode;
uniform float uNearPlane;
uniform float uFarPlane;
uniform float uDepthVisualizationMax;

float linearizeDepth(float rawDepth)
{
    float zNdc = rawDepth * 2.0 - 1.0;

    return (2.0 * uNearPlane * uFarPlane) /
        (uFarPlane + uNearPlane - zNdc * (uFarPlane - uNearPlane));
}

void main()
{
    vec4 sceneColor = texture(uSceneColor, texCoord);

    if (uScreenMode == 0) {
        fragColor = sceneColor;
        return;
    }

    // gl_FragCoord.z is the depth of this screen-quad fragment; this samples depth stored by the earlier scene pass.
    float rawDepth = texture(uSceneDepth, texCoord).r;

    if (uScreenMode == 1) {
        fragColor = vec4(vec3(rawDepth), 1.0);
        return;
    }

    float linearDepth = linearizeDepth(rawDepth);
    // uDepthVisualizationMax only controls grayscale display; it does not affect projection or the sampled depth texture.
    float displayedDepth = clamp(linearDepth / uDepthVisualizationMax, 0.0, 1.0);
    fragColor = vec4(vec3(displayedDepth), 1.0);
}
