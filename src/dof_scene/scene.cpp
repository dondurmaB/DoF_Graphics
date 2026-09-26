#include "dof_scene/scene.hpp"

#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

namespace dof_scene {
namespace {

constexpr float kPi = 3.14159265358979323846f;

void rebuildSmoothNormals(MeshData& mesh) {
    for (Vertex& vertex : mesh.vertices) {
        vertex.normal = glm::vec3(0.0f);
    }

    for (size_t index = 0; index + 2 < mesh.indices.size(); index += 3) {
        Vertex& a = mesh.vertices[mesh.indices[index]];
        Vertex& b = mesh.vertices[mesh.indices[index + 1]];
        Vertex& c = mesh.vertices[mesh.indices[index + 2]];
        const glm::vec3 faceNormal = glm::normalize(glm::cross(b.position - a.position, c.position - a.position));
        a.normal += faceNormal;
        b.normal += faceNormal;
        c.normal += faceNormal;
    }

    for (Vertex& vertex : mesh.vertices) {
        vertex.normal = glm::normalize(vertex.normal);
    }
}

Material makeMaterial(const glm::vec3& color,
                      float roughness,
                      float metallic,
                      int texture,
                      const glm::vec3& emission = glm::vec3(0.0f)) {
    Material material;
    material.color = color;
    material.roughness = roughness;
    material.metallic = metallic;
    material.texture = texture;
    material.emission = emission;
    return material;
}

glm::mat4 makeModel(const glm::vec3& translation,
                    const glm::vec3& rotationDegrees,
                    const glm::vec3& scale) {
    glm::mat4 model(1.0f);
    model = glm::translate(model, translation);
    model = glm::rotate(model, glm::radians(rotationDegrees.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rotationDegrees.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rotationDegrees.z), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::scale(model, scale);
    return model;
}

void addObject(Scene& scene,
               Shape shape,
               const glm::vec3& translation,
               const glm::vec3& rotationDegrees,
               const glm::vec3& scale,
               const Material& material) {
    Object object;
    object.shape = shape;
    object.model = makeModel(translation, rotationDegrees, scale);
    object.material = material;
    scene.objects.push_back(object);
}

MeshData createBoxMesh() {
    MeshData mesh;
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const glm::vec3 positions[8] = {
        {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f},
        {0.5f, 0.5f, -0.5f},   {-0.5f, 0.5f, -0.5f},
        {-0.5f, -0.5f, 0.5f},  {0.5f, -0.5f, 0.5f},
        {0.5f, 0.5f, 0.5f},    {-0.5f, 0.5f, 0.5f},
    };

    struct Face {
        int a;
        int b;
        int c;
        int d;
        glm::vec3 normal;
    };

    const Face faces[6] = {
        {5, 4, 7, 6, {0.0f, 0.0f, 1.0f}},
        {4, 0, 3, 7, {-1.0f, 0.0f, 0.0f}},
        {0, 1, 2, 3, {0.0f, 0.0f, -1.0f}},
        {1, 5, 6, 2, {1.0f, 0.0f, 0.0f}},
        {3, 2, 6, 7, {0.0f, 1.0f, 0.0f}},
        {4, 5, 1, 0, {0.0f, -1.0f, 0.0f}},
    };

    const glm::vec2 uvs[4] = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f},
    };

    for (const Face& face : faces) {
        const unsigned int base = static_cast<unsigned int>(mesh.vertices.size());
        mesh.vertices.push_back({positions[face.a], face.normal, uvs[0]});
        mesh.vertices.push_back({positions[face.b], face.normal, uvs[1]});
        mesh.vertices.push_back({positions[face.c], face.normal, uvs[2]});
        mesh.vertices.push_back({positions[face.d], face.normal, uvs[3]});

        mesh.indices.insert(mesh.indices.end(), {
            base + 0u, base + 2u, base + 1u,
            base + 0u, base + 3u, base + 2u,
        });
    }

    return mesh;
}

MeshData createSphereMesh(int rings = 24, int segments = 48) {
    MeshData mesh;
    mesh.vertices.reserve(static_cast<size_t>((rings + 1) * (segments + 1)));
    mesh.indices.reserve(static_cast<size_t>(rings * segments * 6));

    for (int ring = 0; ring <= rings; ++ring) {
        const float v = static_cast<float>(ring) / static_cast<float>(rings);
        const float phi = v * kPi;
        const float y = std::cos(phi);
        const float radius = std::sin(phi);

        for (int segment = 0; segment <= segments; ++segment) {
            const float u = static_cast<float>(segment) / static_cast<float>(segments);
            const float theta = u * kPi * 2.0f;
            glm::vec3 normal(radius * std::cos(theta), y, radius * std::sin(theta));
            mesh.vertices.push_back({normal * 0.5f, normal, {u, 1.0f - v}});
        }
    }

    for (int ring = 0; ring < rings; ++ring) {
        for (int segment = 0; segment < segments; ++segment) {
            const unsigned int a = static_cast<unsigned int>(ring * (segments + 1) + segment);
            const unsigned int b = static_cast<unsigned int>((ring + 1) * (segments + 1) + segment);
            const unsigned int c = b + 1u;
            const unsigned int d = a + 1u;

            if (ring > 0) {
                mesh.indices.insert(mesh.indices.end(), {a, d, b});
            }
            if (ring < rings - 1) {
                mesh.indices.insert(mesh.indices.end(), {d, c, b});
            }
        }
    }

    return mesh;
}

MeshData createCylinderMesh(int segments = 48) {
    MeshData mesh;
    mesh.vertices.reserve(static_cast<size_t>((segments + 1) * 2 + (segments + 2) * 2));
    mesh.indices.reserve(static_cast<size_t>(segments * 12));

    for (int segment = 0; segment <= segments; ++segment) {
        const float u = static_cast<float>(segment) / static_cast<float>(segments);
        const float theta = u * kPi * 2.0f;
        const glm::vec3 normal(std::cos(theta), 0.0f, std::sin(theta));
        const glm::vec3 bottom = normal * 0.5f + glm::vec3(0.0f, -0.5f, 0.0f);
        const glm::vec3 top = normal * 0.5f + glm::vec3(0.0f, 0.5f, 0.0f);
        mesh.vertices.push_back({bottom, normal, {u, 0.0f}});
        mesh.vertices.push_back({top, normal, {u, 1.0f}});
    }

    for (int segment = 0; segment < segments; ++segment) {
        const unsigned int a = static_cast<unsigned int>(segment * 2);
        const unsigned int b = a + 1u;
        const unsigned int c = a + 2u;
        const unsigned int d = a + 3u;
        mesh.indices.insert(mesh.indices.end(), {a, b, c, b, d, c});
    }

    const unsigned int topCenter = static_cast<unsigned int>(mesh.vertices.size());
    mesh.vertices.push_back({{0.0f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.5f, 0.5f}});
    for (int segment = 0; segment <= segments; ++segment) {
        const float u = static_cast<float>(segment) / static_cast<float>(segments);
        const float theta = u * kPi * 2.0f;
        const glm::vec3 rim(std::cos(theta) * 0.5f, 0.5f, std::sin(theta) * 0.5f);
        mesh.vertices.push_back({rim, {0.0f, 1.0f, 0.0f}, {rim.x + 0.5f, rim.z + 0.5f}});
    }

    for (int segment = 0; segment < segments; ++segment) {
        mesh.indices.insert(mesh.indices.end(), {
            topCenter,
            topCenter + static_cast<unsigned int>(segment + 2),
            topCenter + static_cast<unsigned int>(segment + 1),
        });
    }

    const unsigned int bottomCenter = static_cast<unsigned int>(mesh.vertices.size());
    mesh.vertices.push_back({{0.0f, -0.5f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.5f, 0.5f}});
    for (int segment = 0; segment <= segments; ++segment) {
        const float u = static_cast<float>(segment) / static_cast<float>(segments);
        const float theta = u * kPi * 2.0f;
        const glm::vec3 rim(std::cos(theta) * 0.5f, -0.5f, std::sin(theta) * 0.5f);
        mesh.vertices.push_back({rim, {0.0f, -1.0f, 0.0f}, {rim.x + 0.5f, rim.z + 0.5f}});
    }

    for (int segment = 0; segment < segments; ++segment) {
        mesh.indices.insert(mesh.indices.end(), {
            bottomCenter,
            bottomCenter + static_cast<unsigned int>(segment + 1),
            bottomCenter + static_cast<unsigned int>(segment + 2),
        });
    }

    return mesh;
}

MeshData createTorusMesh(int rings = 48, int segments = 18) {
    MeshData mesh;
    mesh.vertices.reserve(static_cast<size_t>((rings + 1) * (segments + 1)));
    mesh.indices.reserve(static_cast<size_t>(rings * segments * 6));

    constexpr float majorRadius = 0.35f;
    constexpr float minorRadius = 0.05f;

    for (int ring = 0; ring <= rings; ++ring) {
        const float u = static_cast<float>(ring) / static_cast<float>(rings);
        const float theta = u * kPi * 2.0f;
        const glm::vec3 radial(std::cos(theta), 0.0f, std::sin(theta));

        for (int segment = 0; segment <= segments; ++segment) {
            const float v = static_cast<float>(segment) / static_cast<float>(segments);
            const float phi = v * kPi * 2.0f;
            const glm::vec3 normal = glm::normalize(radial * std::cos(phi) + glm::vec3(0.0f, std::sin(phi), 0.0f));
            const glm::vec3 position = radial * (majorRadius + minorRadius * std::cos(phi)) +
                                       glm::vec3(0.0f, minorRadius * std::sin(phi), 0.0f);
            mesh.vertices.push_back({position, normal, {u, v}});
        }
    }

    for (int ring = 0; ring < rings; ++ring) {
        for (int segment = 0; segment < segments; ++segment) {
            const unsigned int a = static_cast<unsigned int>(ring * (segments + 1) + segment);
            const unsigned int b = static_cast<unsigned int>((ring + 1) * (segments + 1) + segment);
            const unsigned int c = b + 1u;
            const unsigned int d = a + 1u;
            mesh.indices.insert(mesh.indices.end(), {a, d, b, d, c, b});
        }
    }

    return mesh;
}

MeshData createLatheMesh(const std::vector<glm::vec2>& control, int segments = 64) {
    std::vector<glm::vec2> profile;
    for (std::size_t i = 0; i + 1 < control.size(); ++i) {
        const glm::vec2 a = control[i == 0 ? 0 : i - 1];
        const glm::vec2 b = control[i], c = control[i + 1];
        const glm::vec2 d = control[std::min(i + 2, control.size() - 1)];
        for (int sample = 0; sample < 6; ++sample) {
            const float t = sample / 6.0f;
            glm::vec2 point = 0.5f * ((2.0f*b) + (-a+c)*t +
                (2.0f*a-5.0f*b+4.0f*c-d)*t*t + (-a+3.0f*b-3.0f*c+d)*t*t*t);
            point.x = std::max(point.x, 0.004f);
            profile.push_back(point);
        }
    }
    profile.push_back(control.back());
    MeshData mesh;
    mesh.vertices.reserve(static_cast<size_t>(profile.size() * (segments + 1)));
    mesh.indices.reserve(static_cast<size_t>((profile.size() - 1) * segments * 6));

    for (int segment = 0; segment <= segments; ++segment) {
        const float u = static_cast<float>(segment) / static_cast<float>(segments);
        const float theta = u * kPi * 2.0f;
        const float cosTheta = std::cos(theta);
        const float sinTheta = std::sin(theta);

        for (const glm::vec2& point : profile) {
            mesh.vertices.push_back({{point.x * cosTheta, point.y, point.x * sinTheta},
                                     {cosTheta, 0.0f, sinTheta},
                                     {u, point.y}});
        }
    }

    const unsigned int stride = static_cast<unsigned int>(profile.size());
    for (int segment = 0; segment < segments; ++segment) {
        for (unsigned int point = 0; point + 1 < stride; ++point) {
            const unsigned int a = static_cast<unsigned int>(segment) * stride + point;
            const unsigned int b = a + 1u;
            const unsigned int c = static_cast<unsigned int>(segment + 1) * stride + point;
            const unsigned int d = c + 1u;
            mesh.indices.insert(mesh.indices.end(), {a, b, c, b, d, c});
        }
    }

    rebuildSmoothNormals(mesh);
    return mesh;
}

MeshData createCupMesh() {
    return createLatheMesh({
        {0.24f, 0.02f},
        {0.34f, 0.08f},
        {0.42f, 0.40f},
        {0.46f, 0.88f},
        {0.50f, 0.96f},
        {0.39f, 0.91f},
        {0.32f, 0.18f},
        {0.16f, 0.10f},
    });
}

MeshData createTeapotMesh() {
    return createLatheMesh({
        {0.22f,0.00f}, {0.35f,0.06f}, {0.47f,0.16f}, {0.54f,0.30f},
        {0.55f,0.44f}, {0.50f,0.60f}, {0.40f,0.73f}, {0.30f,0.80f},
        {0.305f,0.83f}, {0.27f,0.86f}, {0.18f,0.90f}, {0.085f,0.92f},
        {0.085f,0.98f}, {0.055f,1.03f}, {0.004f,1.035f}
    });
}

MeshData createVaseMesh() {
    return createLatheMesh({
        {0.22f, 0.02f},
        {0.30f, 0.12f},
        {0.26f, 0.42f},
        {0.18f, 0.76f},
        {0.23f, 0.91f},
        {0.29f, 0.98f},
        {0.21f, 1.03f},
        {0.16f, 0.92f},
    });
}

MeshData createSpoutMesh(int rings = 18, int segments = 18) {
    MeshData mesh;
    mesh.vertices.reserve(static_cast<size_t>((rings + 1) * (segments + 1)));
    mesh.indices.reserve(static_cast<size_t>(rings * segments * 6));

    for (int ring = 0; ring <= rings; ++ring) {
        const float t = static_cast<float>(ring) / static_cast<float>(rings);
        const glm::vec3 center(-0.08f - 0.92f * t,
                               0.18f + 0.20f * t + 0.10f * std::sin(kPi * t),
                               0.0f);
        const glm::vec3 tangent = glm::normalize(glm::vec3(-0.92f,
                                                           0.20f + 0.10f * kPi * std::cos(kPi * t),
                                                           0.0f));
        const glm::vec3 side(0.0f, 0.0f, 1.0f);
        const glm::vec3 up = glm::normalize(glm::cross(side, tangent));
        const float radius = 0.14f * (1.0f - t) + 0.045f * t;

        for (int segment = 0; segment <= segments; ++segment) {
            const float v = static_cast<float>(segment) / static_cast<float>(segments);
            const float theta = v * kPi * 2.0f;
            const glm::vec3 normal = glm::normalize(up * std::cos(theta) + side * std::sin(theta));
            mesh.vertices.push_back({center + radius * normal, normal, {t, v}});
        }
    }

    const unsigned int stride = static_cast<unsigned int>(segments + 1);
    for (int ring = 0; ring < rings; ++ring) {
        for (int segment = 0; segment < segments; ++segment) {
            const unsigned int a = static_cast<unsigned int>(ring) * stride + static_cast<unsigned int>(segment);
            const unsigned int b = a + 1u;
            const unsigned int c = static_cast<unsigned int>(ring + 1) * stride + static_cast<unsigned int>(segment);
            const unsigned int d = c + 1u;
            mesh.indices.insert(mesh.indices.end(), {a, c, b, b, c, d});
        }
    }

    rebuildSmoothNormals(mesh);
    return mesh;
}

void addBookStack(Scene& scene, const glm::vec3& base, float yawDegrees) {
    const Material forestCover = makeMaterial({0.10f, 0.25f, 0.18f}, 0.72f, 0.0f, 3);
    const Material ochreCover = makeMaterial({0.52f, 0.35f, 0.13f}, 0.74f, 0.0f, 3);
    const Material slateCover = makeMaterial({0.16f, 0.22f, 0.28f}, 0.76f, 0.0f, 3);
    const Material pages = makeMaterial({0.78f, 0.72f, 0.58f}, 0.88f, 0.0f, 0);

    addObject(scene, Shape::Box, base + glm::vec3(0.0f, 0.02f, 0.0f), {0.0f, yawDegrees, 0.0f},
              {0.50f, 0.026f, 0.32f}, slateCover);
    addObject(scene, Shape::Box, base + glm::vec3(0.015f, 0.045f, -0.005f), {0.0f, yawDegrees, 0.0f},
              {0.47f, 0.022f, 0.29f}, pages);
    addObject(scene, Shape::Box, base + glm::vec3(0.02f, 0.055f, -0.015f), {0.0f, yawDegrees + 5.0f, 0.0f},
              {0.46f, 0.026f, 0.29f}, forestCover);
    addObject(scene, Shape::Box, base + glm::vec3(-0.01f, 0.082f, 0.0f), {0.0f, yawDegrees + 5.0f, 0.0f},
              {0.42f, 0.020f, 0.26f}, pages);
    addObject(scene, Shape::Box, base + glm::vec3(-0.03f, 0.095f, 0.02f), {0.0f, yawDegrees - 8.0f, 0.0f},
              {0.42f, 0.032f, 0.28f}, ochreCover);
}

void addPlant(Scene& scene, const glm::vec3& base) {
    const Material pot = makeMaterial({0.64f, 0.28f, 0.15f}, 0.72f, 0.0f, 4);
    const Material soil = makeMaterial({0.09f, 0.055f, 0.035f}, 0.95f, 0.0f, 0);
    const Material leaf = makeMaterial({0.06f, 0.36f, 0.16f}, 0.62f, 0.0f, 0);

    addObject(scene, Shape::Cylinder, base + glm::vec3(0.0f, 0.12f, 0.0f), {0.0f, 0.0f, 0.0f},
              {0.34f, 0.24f, 0.34f}, pot);
    addObject(scene, Shape::Cylinder, base + glm::vec3(0.0f, 0.25f, 0.0f), {0.0f, 0.0f, 0.0f},
              {0.29f, 0.03f, 0.29f}, soil);

    for (int i = 0; i < 11; ++i) {
        const float angle = static_cast<float>(i) * 32.0f;
        const float lean = (i % 3 == 0) ? 28.0f : 16.0f;
        const float height = 0.28f + 0.035f * static_cast<float>(i % 4);
        addObject(scene, Shape::Cylinder, base + glm::vec3(0.0f, 0.34f + height * 0.22f, 0.0f),
                  {lean, angle, 0.0f}, {0.018f, height, 0.018f}, leaf);
        addObject(scene, Shape::Sphere, base + glm::vec3(std::cos(glm::radians(angle)) * 0.13f,
                                                         0.49f + height * 0.18f,
                                                         std::sin(glm::radians(angle)) * 0.13f),
                  {18.0f, angle, 38.0f}, {0.18f, 0.035f, 0.08f}, leaf);
    }
}

void addBackgroundChair(Scene& scene, const glm::vec3& base, float yawDegrees) {
    const Material chairWood = makeMaterial({0.43f, 0.23f, 0.11f}, 0.56f, 0.0f, 1);
    const glm::vec3 rotation(0.0f, yawDegrees, 0.0f);

    addObject(scene, Shape::Box, base + glm::vec3(0.0f, 0.45f, 0.0f), rotation, {0.42f, 0.08f, 0.42f}, chairWood);
    addObject(scene, Shape::Box, base + glm::vec3(0.0f, 0.88f, -0.19f), rotation, {0.46f, 0.58f, 0.07f}, chairWood);
    addObject(scene, Shape::Cylinder, base + glm::vec3(-0.16f, 0.22f, -0.16f), rotation, {0.045f, 0.44f, 0.045f}, chairWood);
    addObject(scene, Shape::Cylinder, base + glm::vec3(0.16f, 0.22f, -0.16f), rotation, {0.045f, 0.44f, 0.045f}, chairWood);
    addObject(scene, Shape::Cylinder, base + glm::vec3(-0.16f, 0.22f, 0.16f), rotation, {0.045f, 0.44f, 0.045f}, chairWood);
    addObject(scene, Shape::Cylinder, base + glm::vec3(0.16f, 0.22f, 0.16f), rotation, {0.045f, 0.44f, 0.045f}, chairWood);
}

} // namespace

Scene createCafeScene() {
    Scene scene;
    scene.meshes[static_cast<size_t>(Shape::Box)] = createBoxMesh();
    scene.meshes[static_cast<size_t>(Shape::Sphere)] = createSphereMesh();
    scene.meshes[static_cast<size_t>(Shape::Cylinder)] = createCylinderMesh();
    scene.meshes[static_cast<size_t>(Shape::Torus)] = createTorusMesh();
    scene.meshes[static_cast<size_t>(Shape::Cup)] = createCupMesh();
    scene.meshes[static_cast<size_t>(Shape::Teapot)] = createTeapotMesh();
    scene.meshes[static_cast<size_t>(Shape::Spout)] = createSpoutMesh();
    scene.meshes[static_cast<size_t>(Shape::Vase)] = createVaseMesh();

    const Material plaster = makeMaterial({0.74f, 0.70f, 0.63f}, 0.88f, 0.0f, 2);
    const Material darkPlaster = makeMaterial({0.46f, 0.43f, 0.38f}, 0.9f, 0.0f, 2);
    const Material wood = makeMaterial({0.55f, 0.32f, 0.14f}, 0.48f, 0.0f, 1);
    const Material darkWood = makeMaterial({0.25f, 0.13f, 0.06f}, 0.58f, 0.0f, 1);
    const Material ceramicWhite = makeMaterial({0.92f, 0.88f, 0.80f}, 0.38f, 0.0f, 4);
    const Material ceramicBlue = makeMaterial({0.15f, 0.33f, 0.52f}, 0.34f, 0.0f, 4);
    const Material coffee = makeMaterial({0.055f, 0.025f, 0.012f}, 0.82f, 0.0f, 0);
    const Material metal = makeMaterial({0.8f, 0.68f, 0.48f}, 0.26f, 0.75f, 0);
    const Material glass = makeMaterial({0.58f, 0.77f, 0.82f}, 0.12f, 0.0f, 0);
    const Material leaf = makeMaterial({0.08f, 0.32f, 0.15f}, 0.64f, 0.0f, 0);
    const Material orange = makeMaterial({0.92f, 0.44f, 0.11f}, 0.66f, 0.0f, 0);
    const Material pastry = makeMaterial({0.77f, 0.47f, 0.21f}, 0.72f, 0.0f, 0);
    const Material flower = makeMaterial({0.86f, 0.55f, 0.17f}, 0.72f, 0.0f, 0);
    const Material stem = makeMaterial({0.05f, 0.27f, 0.10f}, 0.74f, 0.0f, 0);
    const Material bulbGlow = makeMaterial({1.0f, 0.74f, 0.38f}, 0.18f, 0.0f, 0, {6.0f, 3.8f, 1.5f});
    const Material shade = makeMaterial({0.08f, 0.08f, 0.075f}, 0.42f, 0.15f, 0);

    // Open-topped room shell: leaves direct sun free to reach the tabletop.
    addObject(scene, Shape::Box, {0.0f, -0.03f, -2.2f}, {0.0f, 0.0f, 0.0f}, {7.5f, 0.06f, 8.6f}, darkPlaster);
    addObject(scene, Shape::Box, {0.0f, 1.65f, -6.02f}, {0.0f, 0.0f, 0.0f}, {7.5f, 3.3f, 0.08f}, plaster);
    addObject(scene, Shape::Box, {-3.75f, 1.65f, -2.2f}, {0.0f, 0.0f, 0.0f}, {0.08f, 3.3f, 8.6f}, plaster);
    addObject(scene, Shape::Box, {3.75f, 1.65f, -2.2f}, {0.0f, 0.0f, 0.0f}, {0.08f, 3.3f, 8.6f}, plaster);

    // Window and back wall shelving.
    addObject(scene, Shape::Box, {-1.9f, 1.65f, -5.96f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.25f, 0.04f}, glass);
    addObject(scene, Shape::Box, {-1.9f, 2.3f, -5.91f}, {0.0f, 0.0f, 0.0f}, {1.15f, 0.08f, 0.12f}, darkWood);
    addObject(scene, Shape::Box, {-1.9f, 1.0f, -5.91f}, {0.0f, 0.0f, 0.0f}, {1.15f, 0.08f, 0.12f}, darkWood);
    addObject(scene, Shape::Box, {-2.5f, 1.65f, -5.91f}, {0.0f, 0.0f, 0.0f}, {0.08f, 1.35f, 0.12f}, darkWood);
    addObject(scene, Shape::Box, {-1.3f, 1.65f, -5.91f}, {0.0f, 0.0f, 0.0f}, {0.08f, 1.35f, 0.12f}, darkWood);
    addObject(scene, Shape::Box, {1.65f, 1.45f, -5.72f}, {0.0f, 0.0f, 0.0f}, {2.2f, 0.08f, 0.34f}, darkWood);
    addObject(scene, Shape::Box, {1.65f, 2.05f, -5.72f}, {0.0f, 0.0f, 0.0f}, {2.2f, 0.08f, 0.34f}, darkWood);
    for (int i = 0; i < 8; ++i) {
        const float x = 0.75f + static_cast<float>(i % 4) * 0.45f;
        const float y = (i < 4) ? 1.62f : 2.22f;
        addObject(scene, Shape::Box, {x, y, -5.55f}, {0.0f, -7.0f + i * 4.0f, 0.0f},
                  {0.16f, 0.28f, 0.20f}, (i % 2 == 0) ? ceramicBlue : ceramicWhite);
    }

    // Main table.
    addObject(scene, Shape::Box, {0.0f, 0.78f, -0.35f}, {0.0f, 0.0f, 0.0f}, {3.3f, 0.12f, 2.5f}, wood);
    addObject(scene, Shape::Box, {-1.42f, 0.38f, -1.35f}, {0.0f, 0.0f, 0.0f}, {0.14f, 0.76f, 0.14f}, darkWood);
    addObject(scene, Shape::Box, {1.42f, 0.38f, -1.35f}, {0.0f, 0.0f, 0.0f}, {0.14f, 0.76f, 0.14f}, darkWood);
    addObject(scene, Shape::Box, {-1.42f, 0.38f, 0.65f}, {0.0f, 0.0f, 0.0f}, {0.14f, 0.76f, 0.14f}, darkWood);
    addObject(scene, Shape::Box, {1.42f, 0.38f, 0.65f}, {0.0f, 0.0f, 0.0f}, {0.14f, 0.76f, 0.14f}, darkWood);

    // Central focus subject: ceramic teapot with enough geometry to hold sharp focus.
    addObject(scene, Shape::Teapot, {0.0f, 0.87f, -0.55f}, {0.0f, -10.0f, 0.0f},
              {0.72f, 0.52f, 0.56f}, ceramicWhite);
    addObject(scene, Shape::Spout, {-0.26f, 1.00f, -0.55f}, {0.0f, -4.0f, 0.0f},
              {0.54f, 0.54f, 0.54f}, ceramicWhite);
    addObject(scene, Shape::Torus, {0.47f, 1.10f, -0.55f}, {90.0f, 0.0f, 0.0f},
              {0.60f, 0.50f, 0.70f}, ceramicWhite);
    addObject(scene, Shape::Cylinder, {0.0f, 0.855f, -0.55f}, {0.0f, 0.0f, 0.0f}, {0.86f, 0.025f, 0.66f}, ceramicBlue);

    // Foreground blur cue: mug close to the planned camera and in front of focus.
    addObject(scene, Shape::Cup, {0.98f, 0.84f, 0.40f}, {0.0f, -18.0f, 0.0f},
              {0.36f, 0.34f, 0.36f}, ceramicBlue);
    addObject(scene, Shape::Cylinder, {0.98f, 1.145f, 0.40f}, {0.0f, -18.0f, 0.0f},
              {0.265f, 0.012f, 0.265f}, coffee);
    addObject(scene, Shape::Torus, {1.16f, 1.025f, 0.40f}, {90.0f, 0.0f, 0.0f},
              {0.40f, 0.35f, 0.38f}, ceramicBlue);
    addObject(scene, Shape::Cylinder, {0.40f, 0.86f, 0.55f}, {0.0f, 30.0f, 90.0f}, {0.030f, 0.92f, 0.030f}, metal);

    addBookStack(scene, {-0.88f, 0.842f, -0.15f}, 12.0f);
    addBookStack(scene, {0.98f, 0.842f, -0.95f}, -18.0f);

    addObject(scene, Shape::Cylinder, {-0.62f, 0.94f, -0.88f}, {0.0f, 0.0f, 0.0f}, {0.38f, 0.08f, 0.38f}, ceramicWhite);
    addObject(scene, Shape::Sphere, {-0.78f, 1.04f, -0.86f}, {0.0f, 0.0f, 0.0f}, {0.20f, 0.17f, 0.20f}, orange);
    addObject(scene, Shape::Sphere, {-0.52f, 1.03f, -0.74f}, {0.0f, 0.0f, 0.0f}, {0.17f, 0.15f, 0.17f}, pastry);
    addObject(scene, Shape::Sphere, {-0.44f, 1.03f, -0.98f}, {0.0f, 0.0f, 0.0f}, {0.16f, 0.14f, 0.16f}, pastry);

    addObject(scene, Shape::Vase, {0.98f, 0.82f, -0.12f}, {0.0f, 0.0f, 0.0f}, {0.44f, 0.66f, 0.44f}, glass);
    for (int i = 0; i < 7; ++i) {
        const float angle = static_cast<float>(i) * 51.0f;
        const float radius = 0.06f + 0.025f * static_cast<float>(i % 3);
        const float headY = 1.55f + 0.07f * static_cast<float>(i % 3);
        const glm::vec3 start(0.98f, 1.22f, -0.12f);
        const glm::vec3 end(0.98f + std::cos(glm::radians(angle))*0.16f,
                            headY, -0.12f + std::sin(glm::radians(angle))*0.16f);
        const glm::vec3 delta = end-start;
        const glm::vec3 direction = glm::normalize(delta);
        const glm::vec3 axis = glm::normalize(glm::cross(glm::vec3(0,1,0),direction));
        Object stalk;
        stalk.shape=Shape::Cylinder;
        stalk.material=stem;
        stalk.model=glm::translate(glm::mat4(1), (start+end)*0.5f);
        stalk.model=glm::rotate(stalk.model, std::acos(direction.y), axis);
        stalk.model=glm::scale(stalk.model,glm::vec3(0.012f,glm::length(delta),0.012f));
        scene.objects.push_back(stalk);
        addObject(scene, Shape::Sphere, {0.98f + std::cos(glm::radians(angle)) * radius,
                                         1.34f + 0.04f * static_cast<float>(i % 2),
                                         -0.12f + std::sin(glm::radians(angle)) * radius},
                  {18.0f, angle + 35.0f, 32.0f}, {0.13f, 0.028f, 0.055f}, leaf);
        addObject(scene, Shape::Sphere, {0.98f + std::cos(glm::radians(angle)) * 0.16f,
                                         headY,
                                         -0.12f + std::sin(glm::radians(angle)) * 0.16f},
                  {0.0f, angle, 0.0f}, {0.105f, 0.045f, 0.105f}, flower);
    }

    addPlant(scene, {-1.25f, 0.82f, 0.40f});

    // Background depth cues: tables, chairs, pendant bulbs, shelves.
    addObject(scene, Shape::Cylinder, {-1.95f, 0.70f, -3.25f}, {0.0f, 0.0f, 0.0f}, {0.95f, 0.10f, 0.95f}, wood);
    addObject(scene, Shape::Cylinder, {-1.95f, 0.34f, -3.25f}, {0.0f, 0.0f, 0.0f}, {0.16f, 0.68f, 0.16f}, darkWood);
    addObject(scene, Shape::Cylinder, {2.15f, 0.70f, -3.65f}, {0.0f, 0.0f, 0.0f}, {0.85f, 0.10f, 0.85f}, wood);
    addObject(scene, Shape::Cylinder, {2.15f, 0.34f, -3.65f}, {0.0f, 0.0f, 0.0f}, {0.15f, 0.68f, 0.15f}, darkWood);
    addBackgroundChair(scene, {-2.55f, 0.0f, -3.05f}, 22.0f);
    addBackgroundChair(scene, {-1.34f, 0.0f, -3.58f}, -32.0f);
    addBackgroundChair(scene, {1.58f, 0.0f, -3.28f}, 18.0f);
    addBackgroundChair(scene, {2.76f, 0.0f, -3.95f}, -24.0f);

    addObject(scene, Shape::Cylinder, {-2.0f, 2.45f, -3.0f}, {0.0f, 0.0f, 0.0f}, {0.03f, 0.70f, 0.03f}, shade);
    addObject(scene, Shape::Sphere, {-2.0f, 2.80f, -3.0f}, {0.0f, 0.0f, 0.0f}, {0.20f, 0.20f, 0.20f}, bulbGlow);
    addObject(scene, Shape::Torus, {-2.0f, 2.70f, -3.0f}, {0.0f, 0.0f, 0.0f}, {0.68f, 0.40f, 0.68f}, shade);

    addObject(scene, Shape::Cylinder, {2.4f, 2.45f, -3.5f}, {0.0f, 0.0f, 0.0f}, {0.03f, 0.70f, 0.03f}, shade);
    addObject(scene, Shape::Sphere, {2.4f, 2.80f, -3.5f}, {0.0f, 0.0f, 0.0f}, {0.20f, 0.20f, 0.20f}, bulbGlow);
    addObject(scene, Shape::Torus, {2.4f, 2.70f, -3.5f}, {0.0f, 0.0f, 0.0f}, {0.68f, 0.40f, 0.68f}, shade);

    scene.focusPoint = glm::vec3(0.0f, 1.08f, -0.55f);
    return scene;
}

} // namespace dof_scene
