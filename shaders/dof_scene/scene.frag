#version 330 core

in vec3 vWorldPosition;
in vec3 vWorldNormal;
in vec3 vLocalPosition;
in vec2 vUv;
in vec4 vShadowPosition;

out vec4 fragColor;

uniform vec3 uBaseColor;
uniform vec3 uEmission;
uniform vec3 uCameraPosition;
uniform vec3 uSunDirection;
uniform float uRoughness;
uniform float uMetallic;
uniform int uTextureType;
uniform sampler2D uShadowMap;

const float kPi = 3.14159265359;

float hash(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);

    float a = hash(i + vec2(0.0, 0.0));
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p)
{
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 5; ++i) {
        value += amplitude * noise(p);
        p = p * 2.03 + vec2(7.1, 3.7);
        amplitude *= 0.5;
    }
    return value;
}

vec3 materialColor()
{
    vec3 color = uBaseColor;

    if (uTextureType == 1) {
        float lengthAxis = vLocalPosition.x * 52.0 + fbm(vec2(vLocalPosition.z * 2.2, vUv.y * 0.7)) * 1.15;
        float fineGrain = 0.5 + 0.5 * sin(lengthAxis);
        float longBands = 0.5 + 0.5 * sin(vLocalPosition.x * 10.0 + fbm(vLocalPosition.xz * 3.0) * 1.4);
        float pores = 0.94 + 0.06 * noise(vUv * 72.0 + vLocalPosition.xz * 12.0);
        color *= vec3(0.90, 0.84, 0.76);
        color *= (0.86 + 0.08 * fineGrain + 0.06 * longBands) * pores;
        color += vec3(0.018, 0.013, 0.008) * smoothstep(0.76, 1.0, fbm(vec2(vLocalPosition.x * 18.0, vLocalPosition.z * 2.0)));
    } else if (uTextureType == 2) {
        float plaster = 0.88 + 0.16 * fbm(vWorldPosition.xy * 5.0 + vWorldPosition.zy * 1.3);
        color *= plaster;
        color += vec3(0.025, 0.018, 0.010) * fbm(vWorldPosition.xz * 14.0);
    } else if (uTextureType == 3) {
        float weaveA = 0.5 + 0.5 * sin(vUv.x * 95.0);
        float weaveB = 0.5 + 0.5 * sin(vUv.y * 110.0);
        float weave = 0.82 + 0.13 * weaveA * weaveB;
        color *= weave * (0.9 + 0.12 * fbm(vUv * 18.0));
    } else if (uTextureType == 4) {
        float speckle = smoothstep(0.88, 1.0, noise(vUv * 76.0 + vLocalPosition.xz * 9.0));
        float glaze = 0.985 + 0.025 * fbm(vLocalPosition.xz * 7.0 + vUv * 5.0);
        color = color * glaze + vec3(0.024, 0.021, 0.017) * speckle;
    }

    return max(color, vec3(0.0));
}

float distributionGGX(float nDotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / max(kPi * denom * denom, 0.0001);
}

float geometrySchlickGGX(float nDotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return nDotV / max(nDotV * (1.0 - k) + k, 0.0001);
}

float geometrySmith(float nDotV, float nDotL, float roughness)
{
    return geometrySchlickGGX(nDotV, roughness) * geometrySchlickGGX(nDotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float shadowFactor(vec3 normal, vec3 lightDir)
{
    vec3 projected = vShadowPosition.xyz / vShadowPosition.w;
    projected = projected * 0.5 + 0.5;

    if (projected.z > 1.0 || projected.x < 0.0 || projected.x > 1.0 || projected.y < 0.0 || projected.y > 1.0) {
        return 1.0;
    }

    vec2 texelSize = 1.0 / vec2(textureSize(uShadowMap, 0));
    float bias = max(0.0012 * (1.0 - dot(normal, lightDir)), 0.00035);
    float lit = 0.0;

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float closestDepth = texture(uShadowMap, projected.xy + vec2(x, y) * texelSize).r;
            lit += projected.z - bias <= closestDepth ? 1.0 : 0.0;
        }
    }

    return lit / 9.0;
}

vec3 evaluateLight(vec3 radiance, vec3 lightDir, vec3 normal, vec3 viewDir, vec3 albedo, float roughness, float metallic)
{
    vec3 halfDir = normalize(viewDir + lightDir);
    float nDotL = max(dot(normal, lightDir), 0.0);
    float nDotV = max(dot(normal, viewDir), 0.001);
    float nDotH = max(dot(normal, halfDir), 0.0);
    float hDotV = max(dot(halfDir, viewDir), 0.0);

    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    float d = distributionGGX(nDotH, roughness);
    float g = geometrySmith(nDotV, nDotL, roughness);
    vec3 f = fresnelSchlick(hDotV, f0);

    vec3 specular = (d * g * f) / max(4.0 * nDotV * nDotL, 0.0001);
    vec3 diffuse = (1.0 - f) * (1.0 - metallic) * albedo / kPi;

    return (diffuse + specular) * radiance * nDotL;
}

void main()
{
    vec3 normal = normalize(vWorldNormal);
    if (!gl_FrontFacing) {
        normal = -normal;
    }

    vec3 viewDir = normalize(uCameraPosition - vWorldPosition);
    vec3 albedo = materialColor();
    float roughness = clamp(uRoughness, 0.08, 1.0);
    float metallic = clamp(uMetallic, 0.0, 1.0);

    vec3 sunDir = normalize(uSunDirection);
    float shadow = shadowFactor(normal, sunDir);
    vec3 color = evaluateLight(vec3(2.9, 2.38, 1.84) * shadow, sunDir, normal, viewDir, albedo, roughness, metallic);

    vec3 bulbA = vec3(-2.0, 2.8, -3.0);
    vec3 bulbB = vec3(2.4, 2.8, -3.5);
    vec3 toBulbA = bulbA - vWorldPosition;
    vec3 toBulbB = bulbB - vWorldPosition;
    float distA = length(toBulbA);
    float distB = length(toBulbB);

    color += evaluateLight(vec3(9.5, 5.6, 2.5) / (1.0 + distA * distA * 1.15),
                           normalize(toBulbA), normal, viewDir, albedo, roughness, metallic);
    color += evaluateLight(vec3(8.8, 5.1, 2.3) / (1.0 + distB * distB * 1.15),
                           normalize(toBulbB), normal, viewDir, albedo, roughness, metallic);

    vec3 ambient = albedo * vec3(0.092, 0.084, 0.075);
    float bounce = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
    color += ambient * (0.80 + 0.24 * bounce);
    color += uEmission;

    fragColor = vec4(color, 1.0);
}
