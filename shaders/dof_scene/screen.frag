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
const vec2 diskSamples[64] = vec2[](
    vec2( 0.0884,  0.0000), vec2(-0.1129,  0.1034), vec2( 0.0173, -0.1969), vec2( 0.1423,  0.1856),
    vec2(-0.2611, -0.0462), vec2( 0.2473, -0.1573), vec2(-0.0827,  0.3078), vec2(-0.1578, -0.3038),
    vec2( 0.3423,  0.1250), vec2(-0.3561,  0.1470), vec2( 0.1717, -0.3669), vec2( 0.1269,  0.4045),
    vec2(-0.3824, -0.2216), vec2( 0.4486, -0.0986), vec2(-0.2738,  0.3894), vec2(-0.0632, -0.4880),
    vec2( 0.3883,  0.3272), vec2(-0.5225,  0.0216), vec2( 0.3811, -0.3792), vec2(-0.0255,  0.5514),
    vec2(-0.3626, -0.4345), vec2( 0.5744,  0.0773), vec2(-0.4867,  0.3386), vec2( 0.1330, -0.5912),
    vec2( 0.3076,  0.5368), vec2(-0.6014, -0.1918), vec2( 0.5841, -0.2699), vec2(-0.2531,  0.6047),
    vec2(-0.2259, -0.6279), vec2( 0.6010,  0.3159), vec2(-0.6675,  0.1760), vec2( 0.3794, -0.5901),
    vec2( 0.1207,  0.7023), vec2(-0.5720, -0.4430), vec2( 0.7317, -0.0606), vec2(-0.5058,  0.5467),
    vec2( 0.0037, -0.7552), vec2( 0.5143,  0.5669), vec2(-0.7723, -0.0716), vec2( 0.6258, -0.4750),
    vec2(-0.1424,  0.7827), vec2(-0.4289, -0.6815), vec2( 0.7859,  0.2154), vec2(-0.7335,  0.3764),
    vec2( 0.2899, -0.7819), vec2( 0.3179,  0.7809), vec2(-0.7703, -0.3650), vec2( 0.8233, -0.2538),
    vec2(-0.4401,  0.7511), vec2(-0.1847, -0.8598), vec2( 0.7242,  0.5144), vec2(-0.8902,  0.1109),
    vec2( 0.5870, -0.6897), vec2( 0.0333,  0.9137), vec2(-0.6477, -0.6573), vec2( 0.9300,  0.0475),
    vec2(-0.7243,  0.5985), vec2( 0.1310, -0.9388), vec2( 0.5422,  0.7874), vec2(-0.9397, -0.2162),
    vec2( 0.8459, -0.4793), vec2(-0.3025,  0.9324), vec2(-0.4101, -0.8991), vec2( 0.9170,  0.3890)
);

float linearizeDepth(float rawDepth)
{
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
    float denominator = linearDepth * (uFocusDistance - focalLength);
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
    float blurRadius = min(abs(centerRadius), min(max(uMaxRadius, 0.0), 32.0));

    if (blurRadius < 0.5) {
        return texture(uColor, uv).rgb;
    }

    vec3 sumColor = texture(uColor, uv).rgb;
    float sumWeight = 1.0;
    float centerIsForeground = centerRadius < 0.0 ? 1.0 : 0.0;

    for (int i = 0; i < 64; ++i) {
        vec2 sampleUv = clampUv(uv + diskSamples[i] * blurRadius * texelSize, texelSize);
        float sampleDepth = linearizeDepth(texture(uDepth, sampleUv).r);
        float sampleRadius = signedCoCRadiusPixels(sampleDepth);
        vec3 sampleColor = texture(uColor, sampleUv).rgb;

        float depthDelta = sampleDepth - centerDepth;
        float similarDepth = 1.0 - smoothstep(0.05, 1.4, abs(depthDelta));
        float keepBackgroundFromBleedingOverSharpForeground =
            centerIsForeground > 0.5 ? 1.0 - smoothstep(-0.05, 0.25, depthDelta) : 1.0;
        float allowLargeForegroundOcclusion =
            sampleRadius < -0.5 ? smoothstep(0.2, 1.2, abs(sampleRadius)) : 0.0;
        float sampleCanReachCenter = smoothstep(length(diskSamples[i]) * blurRadius - 0.75,
                                                length(diskSamples[i]) * blurRadius + 0.75,
                                                abs(sampleRadius));
        float weight = mix(0.18, 1.0, similarDepth);
        weight *= max(keepBackgroundFromBleedingOverSharpForeground, allowLargeForegroundOcclusion * 0.85);
        weight *= max(sampleCanReachCenter, similarDepth * 0.65);

        sumColor += sampleColor * weight;
        sumWeight += weight;
    }

    return sumColor / max(sumWeight, kEpsilon);
}

vec3 displayColor(vec3 hdrColor)
{
    return toSrgb(filmicTonemap(hdrColor));
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
        float radiusView = clamp(abs(signedRadius) / max(min(max(uMaxRadius, 1.0), 32.0), 1.0), 0.0, 1.0);
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
