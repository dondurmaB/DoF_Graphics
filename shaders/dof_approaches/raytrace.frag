// Assembled by the host as: #version + shading.glsl + this file.
//
// Method 3: lens-sampled ray tracing. Each frame traces one ray per pixel from
// a random point on the aperture disc through the focus plane, then adds the
// result to an accumulation buffer. Every sample runs its own visibility
// solve, so hidden surfaces are handled correctly at the cost of noise until
// enough samples accumulate.

// 64 primitives across three vec4 arrays is 768 uniform components, which
// stays under the 1024 that GL 3.3 guarantees for the fragment stage.
#define MAX_PRIMITIVES 64

in vec2 vUV;

out vec4 fragColor;

uniform vec3 uEye;
uniform vec3 uForward;
uniform vec3 uRight;
uniform vec3 uUp;
uniform vec3 uBackground;
uniform float uTanHalfFovy;
uniform float uAspect;
uniform float uApertureRadius;
uniform float uFocusDistance;
uniform vec2 uResolution;
uniform int uPrimitiveCount;
uniform int uFrameIndex;
uniform vec4 uPrimParams[MAX_PRIMITIVES];
uniform vec4 uPrimSize[MAX_PRIMITIVES];
uniform vec4 uPrimMaterial[MAX_PRIMITIVES];

const float kFar = 1e30;
const float kEps = 1e-4;

// Mixing the frame index into the third component keeps successive
// accumulation passes from landing on correlated aperture positions, which
// otherwise reads as concentric banding rather than noise.
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

bool hitSphere(vec3 ro, vec3 rd, vec3 center, float radius, inout float best, inout vec3 normal) {
    vec3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    float t = -b - h;
    if (t < kEps) t = -b + h;
    if (t < kEps || t >= best) return false;
    best = t;
    normal = normalize(ro + rd * t - center);
    return true;
}

bool hitPlane(vec3 ro, vec3 rd, float planeY, inout float best, inout vec3 normal) {
    if (abs(rd.y) < 1e-7) return false;
    float t = (planeY - ro.y) / rd.y;
    if (t < kEps || t >= best) return false;
    best = t;
    normal = vec3(0.0, 1.0, 0.0);
    return true;
}

bool hitBox(vec3 ro, vec3 rd, vec3 center, vec3 halfExtents, inout float best, inout vec3 normal) {
    vec3 inv = 1.0 / rd;
    vec3 t0 = (center - halfExtents - ro) * inv;
    vec3 t1 = (center + halfExtents - ro) * inv;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tNear = max(max(tmin.x, tmin.y), tmin.z);
    float tFar = min(min(tmax.x, tmax.y), tmax.z);
    if (tNear > tFar || tFar < kEps) return false;
    float t = tNear;
    if (t < kEps) t = tFar;
    if (t >= best) return false;
    vec3 local = ro + rd * t - center;
    vec3 scaled = abs(local) / max(halfExtents, vec3(1e-5));
    vec3 axis = vec3(0.0);
    if (scaled.x >= scaled.y && scaled.x >= scaled.z) axis = vec3(sign(local.x), 0.0, 0.0);
    else if (scaled.y >= scaled.z) axis = vec3(0.0, sign(local.y), 0.0);
    else axis = vec3(0.0, 0.0, sign(local.z));
    best = t;
    normal = axis;
    return true;
}

bool hitCylinder(vec3 ro, vec3 rd, vec3 center, float radius, float halfHeight, inout float best, inout vec3 normal) {
    vec3 oc = ro - center;
    bool found = false;
    float a = dot(rd.xz, rd.xz);
    if (a > 1e-8) {
        float b = dot(oc.xz, rd.xz);
        float c = dot(oc.xz, oc.xz) - radius * radius;
        float disc = b * b - a * c;
        if (disc >= 0.0) {
            float sq = sqrt(disc);
            float t = (-b - sq) / a;
            if (t < kEps) t = (-b + sq) / a;
            if (t > kEps && t < best) {
                float y = oc.y + t * rd.y;
                if (abs(y) <= halfHeight) {
                    vec3 p = oc + rd * t;
                    best = t;
                    normal = normalize(vec3(p.x, 0.0, p.z));
                    found = true;
                }
            }
        }
    }
    if (abs(rd.y) > 1e-8) {
        for (int cap = 0; cap < 2; ++cap) {
            float capY = (cap == 0) ? -halfHeight : halfHeight;
            float t = (capY - oc.y) / rd.y;
            if (t > kEps && t < best) {
                vec2 q = oc.xz + t * rd.xz;
                if (dot(q, q) <= radius * radius) {
                    best = t;
                    normal = vec3(0.0, (cap == 0) ? -1.0 : 1.0, 0.0);
                    found = true;
                }
            }
        }
    }
    return found;
}

bool traceScene(vec3 ro, vec3 rd, out vec3 normal, out vec3 albedo, out float emissive,
                out float checkerScale, out float distance) {
    float best = kFar;
    vec3 bestNormal = vec3(0.0);
    int bestIndex = -1;
    for (int i = 0; i < MAX_PRIMITIVES; ++i) {
        if (i >= uPrimitiveCount) break;
        vec4 params = uPrimParams[i];
        vec4 size = uPrimSize[i];
        int kind = int(params.w + 0.5);
        float before = best;
        bool hit = false;
        if (kind == 0) hit = hitPlane(ro, rd, params.y, best, bestNormal);
        else if (kind == 1) hit = hitBox(ro, rd, params.xyz, size.xyz, best, bestNormal);
        else if (kind == 2) hit = hitSphere(ro, rd, params.xyz, size.x, best, bestNormal);
        else hit = hitCylinder(ro, rd, params.xyz, size.x, size.y, best, bestNormal);
        if (hit && best < before) bestIndex = i;
    }
    if (bestIndex < 0) return false;
    vec4 material = uPrimMaterial[bestIndex];
    normal = bestNormal;
    albedo = material.rgb;
    checkerScale = material.w;
    emissive = uPrimSize[bestIndex].w;
    distance = best;
    return true;
}

void main() {
    vec2 pixel = vUV * uResolution;
    float jitter = float(uFrameIndex) + 0.5;
    float r1 = hash13(vec3(pixel, jitter));
    float r2 = hash13(vec3(pixel + vec2(37.0, 17.0), jitter + 11.0));
    float angle = 6.28318530718 * r1;
    vec2 lens = vec2(cos(angle), sin(angle)) * (uApertureRadius * sqrt(r2));

    // vUV is a texture coordinate: y = 0 is the bottom of the screen, which
    // must map to the camera's negative up direction.
    float a = (2.0 * vUV.x - 1.0) * uTanHalfFovy * uAspect;
    float b = (2.0 * vUV.y - 1.0) * uTanHalfFovy;
    vec3 focalPoint = uEye + uForward * uFocusDistance + uRight * (a * uFocusDistance) +
                      uUp * (b * uFocusDistance);
    vec3 origin = uEye + uRight * lens.x + uUp * lens.y;
    vec3 direction = normalize(focalPoint - origin);

    vec3 color = uBackground;
    vec3 normal;
    vec3 albedo;
    float emissive;
    float checkerScale;
    float distance;
    if (traceScene(origin, direction, normal, albedo, emissive, checkerScale, distance)) {
        color = shadeSurface(albedo, normal, origin + direction * distance, emissive, checkerScale);
    }
    fragColor = vec4(color, 1.0);
}
