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
