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
// 0 = naive gather (every neighbour contributes equally), 1 = CoC-weighted.
// See gatherDefocus below for what each one is and why both are kept.
uniform int uGatherMode;

// GLSL loop bound: an upper cap, not the default count. Raised from 256
// because a 300 px gather radius at 160 taps is visibly undersampled, and
// seeing that undersampling is part of the comparison.
const int MAX_COC_SAMPLES = 512;
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

// A per-pixel angle, so neighbouring pixels do not all sample the aperture in
// the same places. Without it the Vogel spiral's arms line up across the whole
// frame and a large radius shows concentric rings and blocky structure; with
// it, the same error becomes fine high-frequency noise, which both looks far
// better and is more honest about where the estimate is uncertain.
// Deterministic in screen position, so a screenshot is still reproducible.
float spiralRotation(vec2 fragment)
{
    return fract(sin(dot(floor(fragment), vec2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
}

vec2 rotate(vec2 offset, float angle)
{
    float s = sin(angle), c = cos(angle);
    return vec2(offset.x * c - offset.y * s, offset.x * s + offset.y * c);
}

// Average the aperture disk in linear radiance.
//
// Two modes, both kept on purpose:
//
// uGatherMode 0, NAIVE. Every sample inside the disk contributes equally. This
// is the textbook screen-space gather and it is the baseline the learned stage
// is meant to repair. Its signature failure is visible from a long way off: a
// SHARP object surrounded by blurred background gets smeared outward into a
// halo, because a background pixel happily averages in a foreground pixel whose
// own circle of confusion is a fraction of a pixel wide and could never have
// reached it.
//
// uGatherMode 1, CoC-WEIGHTED. Each sample is weighted by whether its OWN
// circle of confusion is large enough to reach the pixel being written:
//
//     weight = clamp(sampleRadius - distance + 1, 0, 1)
//
// A sharp sample has sampleRadius near zero, so at any real distance its weight
// is zero and it cannot bleed. A genuinely defocused sample has a large radius
// and contributes across its whole disc. This is a gather approximating a
// scatter, and it removes the halo almost entirely at the same cost plus one
// depth fetch per tap.
//
// What mode 1 still does NOT fix, and what the comparison therefore still
// measures: the gather searches only out to the CENTRE pixel's own radius, so a
// heavily blurred foreground object does not spread onto a sharp background
// behind it (that needs a near-field prepass). And neither mode can recover the
// surfaces a real lens sees around an occluder, because the rasterizer never
// stored them. That is the irreducible part.
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
    int maxSamples = clamp(uCoCSampleCount, 1, MAX_COC_SAMPLES);
    // Taps needed for a smooth result grow with the disc's AREA, so a fixed
    // count that looks fine at 10 px is badly undersampled at 120 px. Scale up
    // to the ceiling and let the user see where the ceiling bites.
    int sampleCount = clamp(int(ceil(3.14159265 * blurRadiusPixels * blurRadiusPixels / 18.0)),
                            12, maxSamples);

    float angle = spiralRotation(gl_FragCoord.xy);

    vec3 gatheredColor = vec3(0.0);
    float totalWeight = 0.0;
    // Cost scales with screen resolution x sample count: one colour fetch per
    // tap in mode 0, plus one depth fetch per tap in mode 1.
    for (int i = 0; i < sampleCount; ++i) {
        vec2 diskOffset = rotate(vogelDiskSample(i, sampleCount), angle) * blurRadiusPixels;
        vec2 sampleUV = clamp(texCoord + diskOffset * texelSize, edgeInset, vec2(1.0) - edgeInset);
        vec3 sampleColor = texture(uSceneColor, sampleUV).rgb;

        float weight = 1.0;
        if (uGatherMode != 0) {
            float sampleDepth = linearizeDepth(texture(uSceneDepth, sampleUV).r);
            float sampleRadius = min(abs(calculateSignedCoCPixels(sampleDepth)) * 0.5,
                                     max(uMaxBlurRadiusPixels, 0.0));
            weight = clamp(sampleRadius - length(diskOffset) + 1.0, 0.0, 1.0);
        }
        gatheredColor += sampleColor * weight;
        totalWeight += weight;
    }

    // Every neighbour rejected (an isolated sharp pixel in mode 1): keep it sharp
    // rather than dividing by zero and writing black.
    if (totalWeight < 1e-4) {
        return centerColor;
    }
    return gatheredColor / totalWeight;
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
