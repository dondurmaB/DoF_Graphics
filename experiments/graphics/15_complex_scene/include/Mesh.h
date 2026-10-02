#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <filesystem>
#include <string>
#include <vector>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
};

// OBJ indices are converted into a single index stream for OpenGL.
// Positions remain in authored units; the model matrix supplies explicit scale.
struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    glm::vec3 boundsMin{0.0f};
    glm::vec3 boundsMax{0.0f};
    bool hasNormals = false;
    bool hasTexCoords = false;
    bool generatedNormals = false;
};

bool loadObjMesh(const std::filesystem::path& path, MeshData& data,
                 std::string& error, std::string& warning);

// Explicit lifetime: upload once with a current GL context; destroy before GLFW exits.
struct Mesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei indexCount = 0;
};

bool uploadMesh(const MeshData& data, Mesh& mesh, std::string& error);
void drawMesh(const Mesh& mesh);
void destroyMesh(Mesh& mesh);
