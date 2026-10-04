#version 330 core

in vec2 texCoord;

out vec4 fragColor;

// uSceneColor is an RGBA16F attachment holding LINEAR radiance from basic.frag.
// Everything below gathers in that linear space and only encodes for display on
// the way out, which is the whole point of the floating-point attachment.
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
// Ceiling on the gather RADIUS in pixels, so a near-focus setting cannot turn
// one frame into a several-thousand-tap-per-pixel stall.
uniform float uMaxBlurRadiusPixels;
// How many aperture samples BasicDoF gathers per pixel. Raising this reduces
// the banding/ringing that a large blur radius exposes with only a few taps,
// at a direct cost in texture fetches (samples x framebuffer pixels).
uniform int uCoCSampleCount;
// Linear exposure applied before the sRGB transfer function. 1.0 means the
// scene's radiance is displayed as-is, matching Blender's "Standard" view
// transform with no exposure offset.
uniform float uExposure;
// Split-screen wipe position for mode 6, in [0,1] across the frame.
uniform float uSplitFraction;

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

// Returns the signed circle-of-confusion DIAMETER in framebuffer pixels.
// Positive means background/far defocus, negative foreground/near defocus.
// Diameter, not radius: that is the standard definition of CoC, and a bug in
// the previous version passed this value straight in as a gather radius, which
// blurred the OpenGL image twice as hard as the Cycles reference at the same
// f-number. Everything downstream now halves it explicitly.
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

    float cocSensorMeters =
        (apertureDiameter * focalLengthMeters * (linearDepthMeters - uFocusDistanceMeters)) /
        denominator;

    return (cocSensorMeters / sensorHeightMeters) * uFramebufferHeightPixels;
}

// Average the aperture disk in linear radiance. Neighbours are not depth-tested:
// silhouette bleeding is a known baseline limitation, kept on purpose so the
// error the learned refinement is supposed to fix is visible in the input.
vec3 gatherDefocus(vec3 centerColor, float cocDiameterPixels)
{
    float blurRadiusPixels = min(abs(cocDiameterPixels) * 0.5, max(uMaxBlurRadiusPixels, 0.0));
    // Below half a framebuffer pixel, keep the original colour and skip the gather.
    if (blurRadiusPixels < 0.5) {
        return centerColor;
    }

    vec2 framebufferSizePixels = vec2(uFramebufferWidthPixels, uFramebufferHeightPixels);
    vec2 texelSize = 1.0 / max(framebufferSizePixels, vec2(1.0));
    vec2 edgeInset = 0.5 * texelSize;

    // Clamp so a bad uniform value (0, negative, or absurdly large) can't
    // divide by zero below or blow past the fixed GLSL loop bound.
    int sampleCount = clamp(uCoCSampleCount, 1, MAX_COC_SAMPLES);

    vec3 gatheredColor = vec3(0.0);
    // Cost scales with screen resolution x sample count (one fetch per tap).
    for (int i = 0; i < sampleCount; ++i) {
        vec2 sampleUV = texCoord + vogelDiskSample(i, sampleCount) * blurRadiusPixels * texelSize;
        sampleUV = clamp(sampleUV, edgeInset, vec2(1.0) - edgeInset);
        gatheredColor += texture(uSceneColor, sampleUV).rgb;
    }
    return gatheredColor / float(sampleCount);
}

// sRGB transfer function, the same encoding Blender's "Standard" view transform
// writes. Applying it here instead of leaving linear values in an 8-bit buffer
// is what makes the raster image and the Cycles PNG comparable pixel for pixel;
// the old RGB8 scene target stored linear values and came out visibly dark.
vec3 encodeForDisplay(vec3 linearRadiance)
{
    vec3 exposed = clamp(linearRadiance * uExposure, vec3(0.0), vec3(1.0));
    vec3 low = exposed * 12.92;
    vec3 high = 1.055 * pow(exposed, vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, greaterThan(exposed, vec3(0.0031308)));
}

void main()
{
    vec3 sceneColor = texture(uSceneColor, texCoord).rgb;

    if (uScreenMode == 0) {
        fragColor = vec4(encodeForDisplay(sceneColor), 1.0);
        return;
    }

    // gl_FragCoord.z is this screen-quad fragment's depth; this samples depth stored by the scene pass.
    float rawDepth = texture(uSceneDepth, texCoord).r;

    // The depth and CoC views below are data, not pictures: they are written
    // without exposure or sRGB encoding so a pixel value read off the screen or
    // a screenshot still means the number it says it means.
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
        fragColor = vec4(encodeForDisplay(gatherDefocus(sceneColor, cocPixels)), 1.0);
        return;
    }

    // Side-by-side wipe: sharp on the left of the divider, defocused on the
    // right, same frame and same camera. Much easier to judge than flipping
    // between two screenshots, and it makes an over-strong blur obvious at the
    // seam where the two halves have to agree at the focus plane.
    if (uScreenMode == 6) {
        float divider = clamp(uSplitFraction, 0.0, 1.0);
        float dividerWidth = 1.0 / max(uFramebufferWidthPixels, 1.0);
        if (abs(texCoord.x - divider) < dividerWidth) {
            fragColor = vec4(1.0, 0.85, 0.2, 1.0);
            return;
        }
        vec3 shown = texCoord.x < divider ? sceneColor : gatherDefocus(sceneColor, cocPixels);
        fragColor = vec4(encodeForDisplay(shown), 1.0);
        return;
    }

    // Visualization scale is in CoC diameter pixels, matching the readout in the UI panel.
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
