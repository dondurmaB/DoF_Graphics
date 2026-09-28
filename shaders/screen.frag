#version 330 core

in vec2 texCoord;

out vec4 fragColor;

uniform sampler2D uSceneColor;
uniform sampler2D uSceneDepth;
uniform int uScreenMode;
uniform float uNearPlane;
uniform float uFarPlane;
uniform float uDepthVisualizationMax;
uniform float uFocusDistanceMeters;
uniform float uFocalLengthMillimeters;
uniform float uFNumber;
uniform float uSensorHeightMillimeters;
uniform float uCoCVisualizationMaxPixels;
uniform float uFramebufferHeightPixels;
uniform float uFramebufferWidthPixels;
uniform float uMaxBlurRadiusPixels;
// How many aperture samples BasicDoF gathers per pixel. Raising this reduces
// the banding/ringing that a large blur radius exposes with only a few taps,
// at a direct cost in texture fetches (samples x framebuffer pixels).
uniform int uCoCSampleCount;

const int MAX_COC_SAMPLES = 256; // GLSL loop bound: an upper cap, not the default count.
const float GOLDEN_ANGLE_RADIANS = 2.39996323; // ~137.5 degrees.

// A Vogel/Fermat spiral: conceptually the same idea as sampling points across
// a real lens aperture, which is what the Cycles reference in
// tools/raytraced_reference does by path tracing through the lens. sqrt()
// keeps sample density even per unit area instead of bunching samples near
// the disk center, and the golden angle keeps the spiral from lining up into
// visible rays as sampleIndex grows.
vec2 vogelDiskSample(int sampleIndex, int sampleCount)
{
    float r = sqrt((float(sampleIndex) + 0.5) / float(sampleCount));
    float theta = float(sampleIndex) * GOLDEN_ANGLE_RADIANS;
    return vec2(r * cos(theta), r * sin(theta));
}

float linearizeDepth(float rawDepth)
{
    float zNdc = rawDepth * 2.0 - 1.0;

    return (2.0 * uNearPlane * uFarPlane) /
        (uFarPlane + uNearPlane - zNdc * (uFarPlane - uNearPlane));
}

float calculateSignedCoCPixels(float linearDepthMeters)
{
    float focalLengthMeters = uFocalLengthMillimeters * 0.001;
    float sensorHeightMeters = uSensorHeightMillimeters * 0.001;

    if (linearDepthMeters <= 0.0 ||
        uFocusDistanceMeters <= focalLengthMeters ||
        uFNumber <= 0.0 ||
        sensorHeightMeters <= 0.0 ||
        uFramebufferHeightPixels <= 0.0) {
        return 0.0;
    }

    // Aperture diameter = focal length / f-number. CoC is zero at the focus distance.
    float apertureDiameter = focalLengthMeters / uFNumber;
    float denominator = linearDepthMeters * (uFocusDistanceMeters - focalLengthMeters);
    if (abs(denominator) < 0.000001) {
        return 0.0;
    }

    // Positive CoC here means background/far defocus; negative means foreground/near defocus.
    float cocSensorMeters =
        (apertureDiameter * focalLengthMeters * (linearDepthMeters - uFocusDistanceMeters)) /
        denominator;

    return (cocSensorMeters / sensorHeightMeters) * uFramebufferHeightPixels;
}

void main()
{
    vec4 sceneColor = texture(uSceneColor, texCoord);

    if (uScreenMode == 0) {
        fragColor = sceneColor;
        return;
    }

    // gl_FragCoord.z is this screen-quad fragment's depth; this samples depth stored by the scene pass.
    float rawDepth = texture(uSceneDepth, texCoord).r;

    if (uScreenMode == 1) {
        fragColor = vec4(vec3(rawDepth), 1.0);
        return;
    }

    // With the project convention, reconstructed linear view depth is interpreted as meters.
    float linearDepth = linearizeDepth(rawDepth);

    if (uScreenMode == 2) {
        float displayedDepth = clamp(linearDepth / uDepthVisualizationMax, 0.0, 1.0);
        fragColor = vec4(vec3(displayedDepth), 1.0);
        return;
    }

    float cocPixels = calculateSignedCoCPixels(linearDepth);

    if (uScreenMode == 5) {
        float blurRadiusPixels = min(abs(cocPixels), max(uMaxBlurRadiusPixels, 0.0));
        // Below half a framebuffer pixel, keep the original color and skip the gather.
        if (blurRadiusPixels < 0.5) {
            fragColor = sceneColor;
            return;
        }

        vec2 framebufferSizePixels = vec2(uFramebufferWidthPixels, uFramebufferHeightPixels);
        vec2 texelSize = 1.0 / max(framebufferSizePixels, vec2(1.0));
        vec2 edgeInset = 0.5 * texelSize;

        // Clamp so a bad uniform value (0, negative, or absurdly large) can't
        // divide by zero below or blow past the fixed GLSL loop bound.
        int sampleCount = clamp(uCoCSampleCount, 1, MAX_COC_SAMPLES);

        vec4 gatheredColor = vec4(0.0);
        // Cost scales with screen resolution x sample count (color + depth per tap).
        // Neighbors are not depth-tested here: silhouette bleeding is a baseline limitation,
        // same as before, just less visible now that the disk is densely sampled.
        for (int i = 0; i < sampleCount; ++i) {
            vec2 sampleUV = texCoord + vogelDiskSample(i, sampleCount) * blurRadiusPixels * texelSize;
            sampleUV = clamp(sampleUV, edgeInset, vec2(1.0) - edgeInset);
            gatheredColor += texture(uSceneColor, sampleUV);
        }
        fragColor = gatheredColor / float(sampleCount);
        return;
    }

    float displayedCoC = clamp(abs(cocPixels) / uCoCVisualizationMaxPixels, 0.0, 1.0);

    if (uScreenMode == 3) {
        fragColor = vec4(vec3(displayedCoC), 1.0);
        return;
    }

    // Debug colors only: red = foreground defocus, blue = background defocus, black = near focus.
    if (cocPixels < 0.0) {
        fragColor = vec4(displayedCoC, 0.0, 0.0, 1.0);
    } else {
        fragColor = vec4(0.0, 0.0, displayedCoC, 1.0);
    }
}
