#include "Mesh.h"

#include <tiny_obj_loader.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <tuple>
#include <utility>

namespace {
bool finite(const glm::vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
}

bool loadObjMesh(const std::filesystem::path& path, MeshData& data,
                 std::string& error, std::string& warning) {
    data = {};
    error.clear();
    warning.clear();

    tinyobj::ObjReaderConfig config;
    config.triangulate = true;
    config.vertex_color = false;
    config.mtl_search_path = path.parent_path().string();
    tinyobj::ObjReader reader;
    const bool parsed = reader.ParseFromFile(path.string(), config);
    warning = reader.Warning();
    if (!parsed) {
        error = reader.Error().empty() ? "OBJ parsing failed." : reader.Error();
        return false;
    }

    const auto& attributes = reader.GetAttrib();
    MeshData loaded;
    // UV seams and normal seams require distinct GPU vertices. Generated flat
    // normals also split triangles, instead of incorrectly smoothing hard edges.
    using Key = std::tuple<int, int, int, size_t>;
    std::map<Key, unsigned int> verticesByIndex;
    size_t triangleSerial = 0;
    size_t skippedTriangles = 0;

    for (const auto& shape : reader.GetShapes()) {
        size_t offset = 0;
        for (const auto faceSize : shape.mesh.num_face_vertices) {
            if (faceSize != 3 || offset + 3 > shape.mesh.indices.size()) {
                error = "OBJ contains a face that could not be triangulated.";
                return false;
            }
            const auto* corners = shape.mesh.indices.data() + offset;
            offset += 3;
            ++triangleSerial;
            glm::vec3 positions[3];
            for (int corner = 0; corner < 3; ++corner) {
                const auto& index = corners[corner];
                if (index.vertex_index < 0 ||
                    static_cast<size_t>(index.vertex_index) >= attributes.vertices.size() / 3 ||
                    index.normal_index < -1 ||
                    (index.normal_index >= 0 && static_cast<size_t>(index.normal_index) >= attributes.normals.size() / 3) ||
                    index.texcoord_index < -1 ||
                    (index.texcoord_index >= 0 && static_cast<size_t>(index.texcoord_index) >= attributes.texcoords.size() / 2)) {
                    error = "OBJ face references an invalid position, normal, or UV index.";
                    return false;
                }
                const size_t base = static_cast<size_t>(index.vertex_index) * 3;
                positions[corner] = glm::vec3(attributes.vertices[base], attributes.vertices[base + 1],
                                              attributes.vertices[base + 2]);
                if (!finite(positions[corner])) {
                    error = "OBJ contains a non-finite position.";
                    return false;
                }
            }

            glm::vec3 faceNormal = glm::cross(positions[1] - positions[0], positions[2] - positions[0]);
            const float faceLengthSquared = glm::dot(faceNormal, faceNormal);
            if (!std::isfinite(faceLengthSquared)) {
                error = "OBJ positions exceed the supported numeric range.";
                return false;
            }
            if (faceLengthSquared <= 0.0f) {
                ++skippedTriangles;
                continue;
            }
            faceNormal /= std::sqrt(faceLengthSquared);

            for (int corner = 0; corner < 3; ++corner) {
                const auto& index = corners[corner];
                Vertex vertex{positions[corner], faceNormal, glm::vec2(0.0f)};
                bool useFaceNormal = true;
                if (index.normal_index >= 0) {
                    const size_t base = static_cast<size_t>(index.normal_index) * 3;
                    const glm::vec3 normal(attributes.normals[base], attributes.normals[base + 1], attributes.normals[base + 2]);
                    const float lengthSquared = glm::dot(normal, normal);
                    if (finite(normal) && std::isfinite(lengthSquared) && lengthSquared > 0.0f) {
                        vertex.normal = normal / std::sqrt(lengthSquared);
                        useFaceNormal = false;
                        loaded.hasNormals = true;
                    }
                }
                loaded.generatedNormals |= useFaceNormal;
                if (index.texcoord_index >= 0) {
                    const size_t base = static_cast<size_t>(index.texcoord_index) * 2;
                    vertex.texCoord = glm::vec2(attributes.texcoords[base], attributes.texcoords[base + 1]);
                    if (!std::isfinite(vertex.texCoord.x) || !std::isfinite(vertex.texCoord.y)) {
                        error = "OBJ contains a non-finite texture coordinate.";
                        return false;
                    }
                    loaded.hasTexCoords = true;
                }

                const Key key(index.vertex_index, index.normal_index, index.texcoord_index,
                              useFaceNormal ? triangleSerial : 0);
                auto found = verticesByIndex.find(key);
                if (found == verticesByIndex.end()) {
                    if (loaded.vertices.size() >= std::numeric_limits<unsigned int>::max()) {
                        error = "OBJ has too many vertices for 32-bit indices.";
                        return false;
                    }
                    const auto newIndex = static_cast<unsigned int>(loaded.vertices.size());
                    found = verticesByIndex.emplace(key, newIndex).first;
                    if (loaded.vertices.empty()) {
                        loaded.boundsMin = loaded.boundsMax = vertex.position;
                    } else {
                        loaded.boundsMin = glm::min(loaded.boundsMin, vertex.position);
                        loaded.boundsMax = glm::max(loaded.boundsMax, vertex.position);
                    }
                    loaded.vertices.push_back(vertex);
                }
                loaded.indices.push_back(found->second);
            }
        }
    }

    if (loaded.indices.empty()) {
        error = "OBJ contains no non-degenerate triangle geometry.";
        return false;
    }
    if (skippedTriangles > 0) {
        warning += "Skipped " + std::to_string(skippedTriangles) + " degenerate triangles.\n";
    }
    data = std::move(loaded);
    return true;
}

bool uploadMesh(const MeshData& data, Mesh& mesh, std::string& error) {
    error.clear();
    if (mesh.vao || mesh.vbo || mesh.ebo) {
        error = "Mesh buffers are already uploaded.";
        return false;
    }
    if (data.vertices.empty() || data.indices.empty() || data.indices.size() % 3 != 0 ||
        data.indices.size() > static_cast<size_t>(std::numeric_limits<GLsizei>::max()) ||
        data.vertices.size() > static_cast<size_t>(std::numeric_limits<GLsizeiptr>::max()) / sizeof(Vertex) ||
        data.indices.size() > static_cast<size_t>(std::numeric_limits<GLsizeiptr>::max()) / sizeof(unsigned int)) {
        error = "Mesh is empty or exceeds OpenGL buffer/draw limits.";
        return false;
    }
    for (const auto index : data.indices) {
        if (index >= data.vertices.size()) {
            error = "Mesh contains an out-of-range index.";
            return false;
        }
    }

    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vbo);
    glGenBuffers(1, &mesh.ebo);
    glBindVertexArray(mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.vertices.size() * sizeof(Vertex)),
                 data.vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.indices.size() * sizeof(unsigned int)),
                 data.indices.data(), GL_STATIC_DRAW);
    // Attribute 1 remains the cube's color. Imported geometry uses 2 for normals, 3 for UVs.
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, texCoord)));
    glEnableVertexAttribArray(3);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    mesh.indexCount = static_cast<GLsizei>(data.indices.size());
    return true;
}

void drawMesh(const Mesh& mesh) {
    if (mesh.indexCount == 0) return;
    glBindVertexArray(mesh.vao);
    glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

void destroyMesh(Mesh& mesh) {
    if (mesh.ebo) glDeleteBuffers(1, &mesh.ebo);
    if (mesh.vbo) glDeleteBuffers(1, &mesh.vbo);
    if (mesh.vao) glDeleteVertexArrays(1, &mesh.vao);
    mesh = {};
}
