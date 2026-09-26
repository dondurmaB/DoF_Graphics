#include "dof_approaches/scene.hpp"

#include <cmath>

namespace dof_approaches {
namespace {

constexpr float kPi = 3.14159265358979323846f;

void addQuad(MeshData& mesh, const glm::vec3& normal, const glm::vec3 corners[4]) {
    const unsigned int base = static_cast<unsigned int>(mesh.vertices.size());
    for (int i = 0; i < 4; ++i) mesh.vertices.push_back({corners[i], normal});
    mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
}

MeshData makeGround(float y, float extent) {
    MeshData mesh;
    const glm::vec3 n(0.0f, 1.0f, 0.0f);
    const glm::vec3 corners[4] = {
        {-extent, y, -extent}, {extent, y, -extent}, {extent, y, extent}, {-extent, y, extent}};
    addQuad(mesh, n, corners);
    return mesh;
}

MeshData makeBox(const glm::vec3& c, const glm::vec3& he) {
    MeshData mesh;
    // Normals are written explicitly instead of derived from winding order, so
    // back-face culling can stay off without any risk of inverted lighting.
    {
        const glm::vec3 n(1.0f, 0.0f, 0.0f);
        const glm::vec3 q[4] = {{c.x + he.x, c.y - he.y, c.z - he.z}, {c.x + he.x, c.y - he.y, c.z + he.z},
                                {c.x + he.x, c.y + he.y, c.z + he.z}, {c.x + he.x, c.y + he.y, c.z - he.z}};
        addQuad(mesh, n, q);
    }
    {
        const glm::vec3 n(-1.0f, 0.0f, 0.0f);
        const glm::vec3 q[4] = {{c.x - he.x, c.y - he.y, c.z + he.z}, {c.x - he.x, c.y - he.y, c.z - he.z},
                                {c.x - he.x, c.y + he.y, c.z - he.z}, {c.x - he.x, c.y + he.y, c.z + he.z}};
        addQuad(mesh, n, q);
    }
    {
        const glm::vec3 n(0.0f, 1.0f, 0.0f);
        const glm::vec3 q[4] = {{c.x - he.x, c.y + he.y, c.z - he.z}, {c.x + he.x, c.y + he.y, c.z - he.z},
                                {c.x + he.x, c.y + he.y, c.z + he.z}, {c.x - he.x, c.y + he.y, c.z + he.z}};
        addQuad(mesh, n, q);
    }
    {
        const glm::vec3 n(0.0f, -1.0f, 0.0f);
        const glm::vec3 q[4] = {{c.x - he.x, c.y - he.y, c.z + he.z}, {c.x + he.x, c.y - he.y, c.z + he.z},
                                {c.x + he.x, c.y - he.y, c.z - he.z}, {c.x - he.x, c.y - he.y, c.z - he.z}};
        addQuad(mesh, n, q);
    }
    {
        const glm::vec3 n(0.0f, 0.0f, 1.0f);
        const glm::vec3 q[4] = {{c.x + he.x, c.y - he.y, c.z + he.z}, {c.x - he.x, c.y - he.y, c.z + he.z},
                                {c.x - he.x, c.y + he.y, c.z + he.z}, {c.x + he.x, c.y + he.y, c.z + he.z}};
        addQuad(mesh, n, q);
    }
    {
        const glm::vec3 n(0.0f, 0.0f, -1.0f);
        const glm::vec3 q[4] = {{c.x - he.x, c.y - he.y, c.z - he.z}, {c.x + he.x, c.y - he.y, c.z - he.z},
                                {c.x + he.x, c.y + he.y, c.z - he.z}, {c.x - he.x, c.y + he.y, c.z - he.z}};
        addQuad(mesh, n, q);
    }
    return mesh;
}

MeshData makeSphere(const glm::vec3& c, float radius) {
    constexpr int kSegments = 32;
    constexpr int kRings = 20;
    MeshData mesh;
    for (int ring = 0; ring <= kRings; ++ring) {
        const float v = static_cast<float>(ring) / kRings;
        const float phi = v * kPi;
        for (int seg = 0; seg <= kSegments; ++seg) {
            const float u = static_cast<float>(seg) / kSegments;
            const float theta = u * 2.0f * kPi;
            const glm::vec3 n(std::sin(phi) * std::cos(theta), std::cos(phi), std::sin(phi) * std::sin(theta));
            mesh.vertices.push_back({c + n * radius, n});
        }
    }
    for (int ring = 0; ring < kRings; ++ring) {
        for (int seg = 0; seg < kSegments; ++seg) {
            const unsigned int a = static_cast<unsigned int>(ring * (kSegments + 1) + seg);
            const unsigned int b = a + kSegments + 1;
            mesh.indices.insert(mesh.indices.end(), {a, b, a + 1, a + 1, b, b + 1});
        }
    }
    return mesh;
}

MeshData makeCylinder(const glm::vec3& c, float radius, float halfHeight) {
    constexpr int kSegments = 32;
    MeshData mesh;
    // Side wall: bottom and top vertex per segment, sharing the radial normal.
    for (int seg = 0; seg <= kSegments; ++seg) {
        const float theta = 2.0f * kPi * static_cast<float>(seg) / kSegments;
        const glm::vec3 n(std::cos(theta), 0.0f, std::sin(theta));
        const glm::vec3 bottom(c.x + n.x * radius, c.y - halfHeight, c.z + n.z * radius);
        mesh.vertices.push_back({bottom, n});
        mesh.vertices.push_back({{bottom.x, c.y + halfHeight, bottom.z}, n});
    }
    for (int seg = 0; seg < kSegments; ++seg) {
        const unsigned int a = static_cast<unsigned int>(seg * 2);
        mesh.indices.insert(mesh.indices.end(), {a, a + 1, a + 2, a + 1, a + 3, a + 2});
    }
    // Caps: a centre vertex plus one ring per cap. The ring is indexed modulo
    // the segment count so the closing fan never reads past the last vertex.
    const unsigned int topCentre = static_cast<unsigned int>(mesh.vertices.size());
    mesh.vertices.push_back({{c.x, c.y + halfHeight, c.z}, {0.0f, 1.0f, 0.0f}});
    const unsigned int bottomCentre = static_cast<unsigned int>(mesh.vertices.size());
    mesh.vertices.push_back({{c.x, c.y - halfHeight, c.z}, {0.0f, -1.0f, 0.0f}});
    const unsigned int topRing = static_cast<unsigned int>(mesh.vertices.size());
    for (int seg = 0; seg < kSegments; ++seg) {
        const float theta = 2.0f * kPi * static_cast<float>(seg) / kSegments;
        mesh.vertices.push_back({{c.x + std::cos(theta) * radius, c.y + halfHeight, c.z + std::sin(theta) * radius},
                                 {0.0f, 1.0f, 0.0f}});
    }
    const unsigned int bottomRing = static_cast<unsigned int>(mesh.vertices.size());
    for (int seg = 0; seg < kSegments; ++seg) {
        const float theta = 2.0f * kPi * static_cast<float>(seg) / kSegments;
        mesh.vertices.push_back({{c.x + std::cos(theta) * radius, c.y - halfHeight, c.z + std::sin(theta) * radius},
                                 {0.0f, -1.0f, 0.0f}});
    }
    for (int seg = 0; seg < kSegments; ++seg) {
        const unsigned int current = static_cast<unsigned int>(seg);
        const unsigned int next = static_cast<unsigned int>((seg + 1) % kSegments);
        mesh.indices.insert(mesh.indices.end(), {topCentre, topRing + current, topRing + next});
        mesh.indices.insert(mesh.indices.end(), {bottomCentre, bottomRing + next, bottomRing + current});
    }
    return mesh;
}

void add(Scene& scene, const Primitive& primitive) {
    scene.primitives.push_back(primitive);
    switch (primitive.kind) {
        case PrimKind::Plane:
            scene.meshes.push_back(makeGround(primitive.position.y, primitive.size.x));
            break;
        case PrimKind::Box:
            scene.meshes.push_back(makeBox(primitive.position, primitive.size));
            break;
        case PrimKind::Sphere:
            scene.meshes.push_back(makeSphere(primitive.position, primitive.size.x));
            break;
        case PrimKind::Cylinder:
            scene.meshes.push_back(makeCylinder(primitive.position, primitive.size.x, primitive.size.y));
            break;
    }
}

}  // namespace

// A deliberately hostile scene for screen-space depth of field. Four failure
// modes are staged side by side so one frame exposes all of them at once:
//
//   Zone A (centre-right) - a near-black railing sitting exactly on the focus
//     plane, in front of a bright checkered wall 38 m away. The railing is
//     perfectly sharp, so a correct renderer cannot spread its colour at all;
//     a single-layer gather still reads the wall pixel's own circle of
//     confusion, reaches across the silhouette, and drags the dark bar colour
//     outward. That dark fringe is the halo.
//
//   Zone B (left third) - three bright posts far in front of the focus plane
//     and narrower than their own circle of confusion. A finite aperture sees
//     past them, so they must become translucent haze. A gather over one
//     pinhole image has no record of what is behind them and can only smear
//     them into opaque bands.
//
//   Zone C (background) - small emissive spheres, each far smaller than its
//     own defocus disc. Multi-view and ray tracing redistribute that energy
//     into bright bokeh discs; averaging neighbours inside one image only
//     dilutes them into grey mush.
//
//   Zone D (everything else) - a receding arcade of columns, spheres, blocks
//     and thin wires giving a continuous depth ramp on both sides of focus, so
//     the sharp-to-blurred transition reads as a gradient rather than as a few
//     isolated objects.
//
// Every placement below is expressed as a *depth from the eye*, because that
// is what the circle of confusion depends on. The eye sits at z = +3.60, so
// world z = 3.60 - depth. A 50 mm lens on a 24 mm sensor gives a 27 deg
// vertical field of view, which means the visible half-height at depth d is
// about 0.24*d, and the half-width about 0.36*d on the 1280x860 frame used for
// captures. Object sizes are chosen against those numbers; ignoring them is
// what makes a telephoto DoF scene look like a close-up of one object.
Scene createHaloScene() {
    Scene scene;

    scene.eye = glm::vec3(0.0f, 1.45f, 3.60f);
    scene.target = glm::vec3(0.0f, 1.05f, -8.40f);
    scene.focusDistance = 3.80f;

    // --- Room --------------------------------------------------------------
    add(scene, {PrimKind::Plane, {0.0f, 0.0f, 0.0f}, {200.0f, 0.0f, 0.0f}, {0.26f, 0.25f, 0.22f}, 0.0f, 2.0f});
    // The back wall is bright and high frequency on purpose: halo contrast is
    // proportional to how different the occluder and the background are.
    add(scene, {PrimKind::Box, {0.0f, 9.0f, -34.0f}, {30.0f, 9.0f, 0.5f}, {0.93f, 0.91f, 0.86f}, 0.0f, 1.2f});
    // Side walls enter the frame past ~15 m of depth and close the corridor.
    add(scene, {PrimKind::Box, {-5.5f, 2.6f, -15.0f}, {0.35f, 2.6f, 19.0f}, {0.34f, 0.33f, 0.39f}, 0.0f, 1.6f});
    add(scene, {PrimKind::Box, {5.5f, 2.6f, -15.0f}, {0.35f, 2.6f, 19.0f}, {0.34f, 0.33f, 0.39f}, 0.0f, 1.6f});

    // --- Zone A: sharp railing, exactly on the focus plane -----------------
    // Depth 3.80 m is the focus distance, so every bar here has a circle of
    // confusion of exactly zero and must survive any correct method untouched.
    const glm::vec3 barColor(0.018f, 0.018f, 0.024f);
    for (int i = 0; i < 5; ++i) {
        const float x = -0.22f + 0.38f * static_cast<float>(i);
        add(scene, {PrimKind::Box, {x, 1.35f, -0.20f}, {0.012f, 0.82f, 0.012f}, barColor});
    }
    for (int i = 0; i < 2; ++i) {
        const float y = 0.80f + 1.05f * static_cast<float>(i);
        add(scene, {PrimKind::Box, {0.54f, y, -0.20f}, {0.80f, 0.012f, 0.012f}, barColor});
    }

    // --- Zone B: thin bright posts well in front of the focus plane --------
    // Depth 1.05 m, where the thin-lens diameter is about 63 px on an 860 px
    // frame while the posts themselves are only about 27 px wide. A defocus
    // disc wider than the occluder is exactly the condition under which a real
    // aperture sees straight through, so these must turn into translucent haze
    // rather than into soft-edged bars.
    for (int i = 0; i < 3; ++i) {
        const float x = -0.30f + 0.08f * static_cast<float>(i);
        add(scene, {PrimKind::Box, {x, 1.40f, 2.55f}, {0.008f, 0.62f, 0.008f}, {0.95f, 0.52f, 0.14f}});
    }

    // --- Focus subject, just past the railing at depth 5.2 -----------------
    add(scene, {PrimKind::Box, {0.45f, 0.80f, -1.60f}, {0.50f, 0.14f, 0.38f}, {0.24f, 0.25f, 0.29f}});
    add(scene, {PrimKind::Sphere, {0.45f, 1.24f, -1.60f}, {0.30f, 0.0f, 0.0f}, {0.90f, 0.30f, 0.17f}});
    add(scene, {PrimKind::Cylinder, {-0.70f, 1.00f, -1.90f}, {0.11f, 0.42f, 0.0f}, {0.86f, 0.84f, 0.79f}});

    // --- Zone D: receding arcade -------------------------------------------
    // Columns sit on the lines x = +-0.22*depth, so they converge steadily
    // toward the vanishing point and the blur gradient is read along a line
    // rather than guessed from scattered objects.
    const Primitive columns[] = {
        {PrimKind::Cylinder, {-1.43f, 1.60f, -2.90f}, {0.14f, 1.60f, 0.0f}, {0.18f, 0.60f, 0.64f}},
        {PrimKind::Cylinder, {1.43f, 1.60f, -2.90f}, {0.14f, 1.60f, 0.0f}, {0.88f, 0.58f, 0.18f}},
        {PrimKind::Cylinder, {-2.42f, 1.90f, -7.40f}, {0.18f, 1.90f, 0.0f}, {0.48f, 0.32f, 0.78f}},
        {PrimKind::Cylinder, {2.42f, 1.90f, -7.40f}, {0.18f, 1.90f, 0.0f}, {0.30f, 0.70f, 0.36f}},
        {PrimKind::Cylinder, {-3.74f, 2.20f, -13.40f}, {0.24f, 2.20f, 0.0f}, {0.26f, 0.42f, 0.82f}},
        {PrimKind::Cylinder, {3.74f, 2.20f, -13.40f}, {0.24f, 2.20f, 0.0f}, {0.80f, 0.26f, 0.26f}},
        {PrimKind::Cylinder, {-4.90f, 2.50f, -20.40f}, {0.32f, 2.50f, 0.0f}, {0.72f, 0.70f, 0.30f}},
        {PrimKind::Cylinder, {4.90f, 2.50f, -20.40f}, {0.32f, 2.50f, 0.0f}, {0.22f, 0.66f, 0.62f}},
    };
    for (const Primitive& column : columns) add(scene, column);

    const Primitive spheres[] = {
        {PrimKind::Sphere, {-1.10f, 0.34f, -4.40f}, {0.34f, 0.0f, 0.0f}, {0.95f, 0.86f, 0.30f}},
        {PrimKind::Sphere, {1.60f, 0.46f, -8.40f}, {0.46f, 0.0f, 0.0f}, {0.36f, 0.86f, 0.50f}},
        {PrimKind::Sphere, {-2.30f, 0.52f, -12.40f}, {0.52f, 0.0f, 0.0f}, {0.82f, 0.46f, 0.90f}},
        {PrimKind::Sphere, {3.10f, 0.62f, -17.40f}, {0.62f, 0.0f, 0.0f}, {0.92f, 0.92f, 0.96f}},
        {PrimKind::Sphere, {-3.90f, 0.70f, -23.40f}, {0.70f, 0.0f, 0.0f}, {0.62f, 0.36f, 0.20f}},
        {PrimKind::Sphere, {4.40f, 0.80f, -27.40f}, {0.80f, 0.0f, 0.0f}, {0.30f, 0.60f, 0.92f}},
    };
    for (const Primitive& sphere : spheres) add(scene, sphere);

    const Primitive blocks[] = {
        {PrimKind::Box, {0.95f, 0.30f, -5.40f}, {0.30f, 0.30f, 0.30f}, {0.70f, 0.68f, 0.62f}},
        {PrimKind::Box, {-1.85f, 0.42f, -10.40f}, {0.42f, 0.42f, 0.42f}, {0.55f, 0.52f, 0.48f}},
        {PrimKind::Box, {2.70f, 0.55f, -16.40f}, {0.55f, 0.55f, 0.55f}, {0.66f, 0.62f, 0.58f}},
        {PrimKind::Box, {-3.30f, 0.70f, -22.40f}, {0.70f, 0.70f, 0.70f}, {0.58f, 0.56f, 0.52f}},
        {PrimKind::Box, {3.60f, 0.85f, -26.40f}, {0.85f, 0.85f, 0.85f}, {0.50f, 0.49f, 0.46f}},
    };
    for (const Primitive& block : blocks) add(scene, block);

    // Thin dark wires at depths where the circle of confusion is still growing,
    // so the bleed can be seen mid-frame and not only at the two extremes.
    const Primitive wires[] = {
        {PrimKind::Cylinder, {-0.55f, 1.60f, -3.40f}, {0.022f, 1.60f, 0.0f}, {0.05f, 0.05f, 0.06f}},
        {PrimKind::Cylinder, {0.80f, 1.80f, -6.40f}, {0.026f, 1.80f, 0.0f}, {0.05f, 0.05f, 0.06f}},
        {PrimKind::Cylinder, {-1.35f, 2.00f, -10.40f}, {0.030f, 2.00f, 0.0f}, {0.05f, 0.05f, 0.06f}},
    };
    for (const Primitive& wire : wires) add(scene, wire);

    // --- Zone C: emissive bokeh lights -------------------------------------
    // Each light is far smaller than its own defocus disc, so its peak
    // brightness falls roughly as (light radius / disc radius)^2 once blurred.
    // The emissive values compensate for that ratio, which is why the distant
    // lights are set brighter than the near ones.
    const Primitive lights[] = {
        {PrimKind::Sphere, {-2.67f, 2.43f, -6.40f}, {0.020f, 0.0f, 0.0f}, {1.00f, 0.83f, 0.55f}, 10.0f},
        {PrimKind::Sphere, {3.10f, 2.10f, -8.40f}, {0.022f, 0.0f, 0.0f}, {1.00f, 0.72f, 0.42f}, 12.0f},
        {PrimKind::Sphere, {2.20f, 2.80f, -9.40f}, {0.022f, 0.0f, 0.0f}, {0.72f, 0.86f, 1.00f}, 13.0f},
        {PrimKind::Sphere, {-1.50f, 4.10f, -11.40f}, {0.026f, 0.0f, 0.0f}, {1.00f, 0.90f, 0.70f}, 15.0f},
        {PrimKind::Sphere, {-3.60f, 3.20f, -12.40f}, {0.026f, 0.0f, 0.0f}, {1.00f, 0.66f, 0.36f}, 16.0f},
        {PrimKind::Sphere, {0.40f, 2.30f, -14.40f}, {0.030f, 0.0f, 0.0f}, {0.80f, 0.92f, 1.00f}, 18.0f},
        {PrimKind::Sphere, {1.30f, 3.60f, -15.40f}, {0.030f, 0.0f, 0.0f}, {1.00f, 0.85f, 0.58f}, 19.0f},
        {PrimKind::Sphere, {-1.10f, 4.00f, -18.40f}, {0.034f, 0.0f, 0.0f}, {1.00f, 0.74f, 0.45f}, 21.0f},
        {PrimKind::Sphere, {-3.20f, 5.00f, -19.40f}, {0.034f, 0.0f, 0.0f}, {0.86f, 0.90f, 1.00f}, 22.0f},
        {PrimKind::Sphere, {4.20f, 4.30f, -21.40f}, {0.038f, 0.0f, 0.0f}, {1.00f, 0.88f, 0.62f}, 24.0f},
        {PrimKind::Sphere, {-4.60f, 2.60f, -24.40f}, {0.040f, 0.0f, 0.0f}, {1.00f, 0.78f, 0.48f}, 26.0f},
        {PrimKind::Sphere, {4.90f, 3.80f, -25.40f}, {0.040f, 0.0f, 0.0f}, {0.94f, 0.94f, 1.00f}, 27.0f},
        {PrimKind::Sphere, {2.50f, 4.80f, -27.40f}, {0.044f, 0.0f, 0.0f}, {1.00f, 0.80f, 0.52f}, 29.0f},
        {PrimKind::Sphere, {-2.00f, 3.40f, -30.40f}, {0.044f, 0.0f, 0.0f}, {1.00f, 0.92f, 0.76f}, 31.0f},
    };
    for (const Primitive& light : lights) add(scene, light);

    return scene;
}

std::vector<PackedPrimitive> packPrimitives(const Scene& scene) {
    std::vector<PackedPrimitive> packed;
    packed.reserve(scene.primitives.size());
    for (const Primitive& primitive : scene.primitives) {
        packed.push_back({glm::vec4(primitive.position, static_cast<float>(static_cast<int>(primitive.kind))),
                          glm::vec4(primitive.size, primitive.emissive),
                          glm::vec4(primitive.color, primitive.checker)});
    }
    return packed;
}

}  // namespace dof_approaches
