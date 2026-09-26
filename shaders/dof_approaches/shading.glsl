// Shared surface shading. The host prepends this file to scene.frag and
// raytrace.frag, so the rasterized and ray-traced paths produce the same
// surface response and the three methods differ only in how they resolve
// visibility. Keep this file free of main() and of #version.

uniform vec3 uSunDirection;

vec3 applyWorldChecker(vec3 albedo, vec3 position, float scale) {
    if (scale <= 0.0) return albedo;
    // The +0.5 shifts cell boundaries off the world axes. Without it the ground
    // plane sits exactly on a boundary at y = 0: the rasterizer interpolates
    // that y as exactly 0.0 and stays on one side, but the ray tracer computes
    // it as ro.y + rd.y*t and lands on either side depending on rounding, so
    // successive aperture samples disagree about the parity and the ground
    // averages to a flat grey. Same shading, different visibility solve, and
    // only the stochastic one reveals the ambiguity.
    vec3 cell = floor(position / scale + 0.5);
    float pattern = mod(cell.x + cell.y + cell.z, 2.0);
    return mix(albedo, albedo * 0.45, pattern);
}

vec3 shadeSurface(vec3 albedo, vec3 normal, vec3 position, float emissive, float checkerScale) {
    albedo = applyWorldChecker(albedo, position, checkerScale);
    vec3 n = normalize(normal);
    vec3 l = normalize(uSunDirection);
    float lambert = max(dot(n, l), 0.0);
    float ambient = 0.18 + 0.14 * (0.5 + 0.5 * n.y);
    return albedo * (ambient + 0.9 * lambert) + albedo * emissive;
}
