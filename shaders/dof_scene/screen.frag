#version 330 core

in vec2 texCoord;

out vec4 fragColor;

uniform sampler2D uColor;
uniform sampler2D uDepth;
uniform float uNear;
uniform float uFar;
uniform float uFocusDistance;
uniform float uFocalLengthMm;
uniform float uSensorHeightMm;
uniform float uFNumber;
uniform float uMaxRadius;
uniform vec2 uResolution;
uniform int uMode;

const float kEpsilon = 0.000001;
// Shared by the native viewer and the EXR experiment runner. Exactly 100
// disk samples; the centre is used only by the subpixel early exit.
const int kSamples = 100;
uniform int uGatherMode; // 0: equal weight, 1: depth weighted
uniform int uDepthIsLinear;
uniform int uLinearOutput;
vec2 diskSample(int i) {
    float radius = sqrt((float(i) + 0.5) / float(kSamples));
    float angle = float(i) * 2.39996323;
    return radius * vec2(cos(angle), sin(angle));
}

float linearizeDepth(float rawDepth)
{
    if (uDepthIsLinear != 0) return rawDepth;
    float zNdc = rawDepth * 2.0 - 1.0;
    return (2.0 * uNear * uFar) / max(uFar + uNear - zNdc * (uFar - uNear), kEpsilon);
}

float signedCoCDiameterPixels(float linearDepth)
{
    float focalLength = uFocalLengthMm * 0.001;
    float sensorHeight = uSensorHeightMm * 0.001;

    if (linearDepth <= 0.0 ||
        uFocusDistance <= focalLength ||
        uFNumber <= 0.0 ||
        sensorHeight <= 0.0 ||
        uResolution.y <= 0.0) {
        return 0.0;
    }

    float apertureDiameter = focalLength / uFNumber;
    // Fixed-FOV sensor convention: aperture rays intersect the focus plane.
    float denominator = linearDepth * uFocusDistance;
    if (abs(denominator) < kEpsilon) {
        return 0.0;
    }

    float cocSensor = apertureDiameter * focalLength * (linearDepth - uFocusDistance) / denominator;
    return (cocSensor / sensorHeight) * uResolution.y;
}

float signedCoCRadiusPixels(float linearDepth)
{
    return 0.5 * signedCoCDiameterPixels(linearDepth);
}

vec3 filmicTonemap(vec3 color)
{
    color = max(color, vec3(0.0));
    color = color * (2.51 * color + 0.03) / (color * (2.43 * color + 0.59) + 0.14);
    return clamp(color, 0.0, 1.0);
}

vec3 toSrgb(vec3 linearColor)
{
    return pow(clamp(linearColor, 0.0, 1.0), vec3(1.0 / 2.2));
}

vec2 clampUv(vec2 uv, vec2 texelSize)
{
    vec2 halfTexel = texelSize * 0.5;
    return clamp(uv, halfTexel, vec2(1.0) - halfTexel);
}

vec3 gatherDof(vec2 uv, float centerDepth, float centerRadius)
{
    vec2 texelSize = 1.0 / max(uResolution, vec2(1.0));
    float blurRadius = min(abs(centerRadius), min(max(uMaxRadius, 0.0), 120.0));

    if (blurRadius < 0.5) {
        return texture(uColor, uv).rgb;
    }

    vec3 sumColor = vec3(0.0);
    float sumWeight = 0.0;
    float centerIsForeground = centerRadius < 0.0 ? 1.0 : 0.0;

    for (int i = 0; i < kSamples; ++i) {
        vec2 sampleUv = clampUv(uv + diskSample(i) * blurRadius * texelSize, texelSize);
        float sampleDepth = linearizeDepth(texture(uDepth, sampleUv).r);
        float sampleRadius = signedCoCRadiusPixels(sampleDepth);
        vec3 sampleColor = texture(uColor, sampleUv).rgb;

        float depthDelta = sampleDepth - centerDepth;
        float similarDepth = 1.0 - smoothstep(0.05, 1.4, abs(depthDelta));
        float keepBackgroundFromBleedingOverSharpForeground =
            centerIsForeground > 0.5 ? 1.0 - smoothstep(-0.05, 0.25, depthDelta) : 1.0;
        float allowLargeForegroundOcclusion =
            sampleRadius < -0.5 ? smoothstep(0.2, 1.2, abs(sampleRadius)) : 0.0;
        float sampleCanReachCenter = smoothstep(length(diskSample(i)) * blurRadius - 0.75,
                                                length(diskSample(i)) * blurRadius + 0.75,
                                                abs(sampleRadius));
        float weight = mix(0.18, 1.0, similarDepth);
        weight *= max(keepBackgroundFromBleedingOverSharpForeground, allowLargeForegroundOcclusion * 0.85);
        weight *= max(sampleCanReachCenter, similarDepth * 0.65);

        if (uGatherMode == 0) weight = 1.0;
        sumColor += sampleColor * weight;
        sumWeight += weight;
    }

    return sumWeight > kEpsilon ? sumColor / sumWeight : texture(uColor, uv).rgb;
}

vec3 displayColor(vec3 hdrColor)
{
    return uLinearOutput != 0 ? hdrColor : toSrgb(filmicTonemap(hdrColor));
}

void main()
{
    vec2 uv = clampUv(texCoord, 1.0 / max(uResolution, vec2(1.0)));
    vec3 sharpHdr = texture(uColor, uv).rgb;
    float rawDepth = texture(uDepth, uv).r;
    float linearDepth = linearizeDepth(rawDepth);
    float signedRadius = signedCoCRadiusPixels(linearDepth);

    if (uMode == 1) {
        fragColor = vec4(displayColor(sharpHdr), 1.0);
        return;
    }

    if (uMode == 2) {
        // A fixed 15 m display range keeps the cafe depth pass readable when the camera far plane is much larger.
        float depthView = clamp(linearDepth / 15.0, 0.0, 1.0);
        fragColor = vec4(vec3(depthView), 1.0);
        return;
    }

    if (uMode == 3) {
        float radiusView = clamp(abs(signedRadius) / max(min(max(uMaxRadius, 1.0), 120.0), 1.0), 0.0, 1.0);
        vec3 nearColor = vec3(1.0, 0.22, 0.08);
        vec3 farColor = vec3(0.10, 0.36, 1.0);
        vec3 focusColor = vec3(0.02);
        fragColor = vec4(mix(focusColor, signedRadius < 0.0 ? nearColor : farColor, radiusView), 1.0);
        return;
    }

    vec3 dofHdr = gatherDof(uv, linearDepth, signedRadius);

    if (uMode == 4) {
        float split = step(uv.x, 0.5);
        vec3 color = mix(dofHdr, sharpHdr, split);
        vec3 srgb = displayColor(color);
        if (abs(uv.x - 0.5) < 1.5 / max(uResolution.x, 1.0)) {
            srgb = vec3(1.0);
        }
        fragColor = vec4(srgb, 1.0);
        return;
    }

    fragColor = vec4(displayColor(dofHdr), 1.0);
}
