#include "Mesh.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Fixtures {
    std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("dof-mesh-tests-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Fixtures() { std::filesystem::create_directory(directory); }
    ~Fixtures() { std::error_code error; std::filesystem::remove_all(directory, error); }

    std::filesystem::path write(const char* name, const std::string& source) {
        const auto path = directory / name;
        std::ofstream file(path);
        file << source;
        require(file.good(), "Could not write test fixture");
        return path;
    }
};

const std::string triangle = "v 0 0 0\nv 2 0 0\nv 0 2 0\n";
}

int main() {
    try {
        Fixtures fixtures;
        MeshData mesh;
        std::string error, warning;
        auto load = [&](const char* name, const std::string& source) {
            return loadObjMesh(fixtures.write(name, source), mesh, error, warning);
        };

        // One quad with independent OBJ attributes must become two indexed triangles.
        require(load("quad.obj", R"(v 0 0 0
v 200 0 0
v 200 100 0
v 0 100 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 2
f -4/1/1 -3/2/1 -2/3/1 -1/4/1
)"), "Quad/negative-index OBJ should load");
        require(mesh.vertices.size() == 4 && mesh.indices.size() == 6, "Quad must reuse four GPU vertices");
        require(mesh.hasNormals && mesh.hasTexCoords && !mesh.generatedNormals, "Source attributes must be retained");
        require(mesh.boundsMax.x == 200.0f && mesh.boundsMax.y == 100.0f, "Authored units must not be normalized");
        float area = 0.0f;
        for (size_t i = 0; i < mesh.indices.size(); i += 3) {
            auto a = mesh.vertices.at(mesh.indices[i]).position;
            auto b = mesh.vertices.at(mesh.indices[i + 1]).position;
            auto c = mesh.vertices.at(mesh.indices[i + 2]).position;
            area += glm::length(glm::cross(b - a, c - a)) * 0.5f;
        }
        require(std::abs(area - 20000.0f) < 0.1f, "Triangulation must preserve face area");
        for (const auto& vertex : mesh.vertices) {
            require(std::abs(vertex.normal.z - 1.0f) < 0.0001f, "Authored normals should be normalized");
            require(vertex.texCoord.x == vertex.position.x / 200.0f &&
                    vertex.texCoord.y == vertex.position.y / 100.0f, "UVs must match their independent OBJ indices");
        }

        require(load("flat.obj", triangle + "v 0 0 2\ng front\nf 1 2 3\ng side\nf 1 4 2\n"), "Multiple shapes without attributes should load");
        require(mesh.indices.size() == 6 && mesh.vertices.size() == 6, "Flat faces need split vertices");
        require(!mesh.hasNormals && !mesh.hasTexCoords && mesh.generatedNormals, "Missing attributes must be reported");
        require(mesh.vertices[0].normal.z == 1.0f && mesh.vertices[3].normal.y == 1.0f, "Generated normals must follow each face");
        require(mesh.vertices[0].texCoord == glm::vec2(0.0f), "Missing UVs should be initialized");

        require(load("seam.obj", triangle + "vn 0 0 1\nvt 0 0\nvt 1 1\nf 1/1/1 2/1/1 3/1/1\nf 1/2/1 2/1/1 3/1/1\n"), "UV seam should load");
        require(mesh.vertices.size() == 4, "UV seam must split only the differing corner");

        require(load("partial.obj", triangle + "vn 0 0 1\nvt 1 1\nf 1/1/1 2//1 3\n"), "Partial attributes should load");
        require(mesh.hasNormals && mesh.hasTexCoords && mesh.generatedNormals, "Partial attributes must generate only missing normals");
        require(load("zero_normal.obj", triangle + "vn 0 0 0\nf 1//1 2//1 3//1\n"), "Zero normals should get a flat fallback");
        require(mesh.generatedNormals && mesh.vertices[0].normal.z == 1.0f, "Zero normals must not create NaNs");

        require(load("missing_material.obj", "mtllib unavailable.mtl\n" + triangle + "usemtl missing\nf 1 2 3\n"), "Missing materials should not block geometry");
        require(!warning.empty(), "Missing material should produce a warning");
        require(load("degenerate.obj", triangle + "f 1 1 2\nf 1 2 3\n"), "A valid face must survive a degenerate neighbor");
        require(mesh.indices.size() == 3 && !warning.empty(), "Degenerate faces should be skipped and reported");

        for (const auto& source : {std::string(), triangle + "f 1 1 2\n", triangle + "f 1 2 99\n",
                                   triangle + "vn 0 0 1\nf 1//9 2//1 3//1\n",
                                   triangle + "vt 0 0\nf 1/9 2/1 3/1\n"}) {
            require(!load("invalid.obj", source), "Empty/invalid geometry must fail safely");
            require(!error.empty() && mesh.vertices.empty() && mesh.indices.empty(), "Failed loads must not expose partial/stale data");
        }
        require(!loadObjMesh(fixtures.directory / "absent.obj", mesh, error, warning), "Missing file must fail safely");
        require(!error.empty() && mesh.indices.empty(), "Missing asset must report an error and no geometry");

        // This validation must return before any OpenGL call, even without a context.
        Mesh gpuMesh;
        require(!uploadMesh(mesh, gpuMesh, error), "Empty mesh upload must be rejected");
        std::cout << "Mesh loading checks passed: triangulation, indices, normals, UV seams, units, warnings and failure paths.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Mesh loading check failed: " << error.what() << '\n';
        return 1;
    }
}
