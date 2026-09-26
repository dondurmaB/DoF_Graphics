#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_NONE
#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include "dof_approaches/scene.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
namespace fs = std::filesystem;
using namespace dof_approaches;

constexpr int kMaxPrimitives = 64;
constexpr float kGoldenAngle = 2.39996323f;

std::string readText(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot read shader: " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

GLuint compileShader(GLenum type, const std::string& source) {
    GLuint shader = glCreateShader(type);
    const char* text = source.c_str();
    glShaderSource(shader, 1, &text, nullptr);
    glCompileShader(shader);
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(std::max(length, 1), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        glDeleteShader(shader);
        throw std::runtime_error("Shader compilation failed:\n" + log);
    }
    return shader;
}

struct Program {
    GLuint id = 0;
    std::unordered_map<std::string, GLint> locations;

    void build(const std::string& vertexSource, const std::string& fragmentSource) {
        const GLuint vs = compileShader(GL_VERTEX_SHADER, vertexSource);
        GLuint ps = 0;
        try {
            ps = compileShader(GL_FRAGMENT_SHADER, fragmentSource);
        } catch (...) {
            glDeleteShader(vs);
            throw;
        }
        id = glCreateProgram();
        glAttachShader(id, vs);
        glAttachShader(id, ps);
        glLinkProgram(id);
        glDeleteShader(vs);
        glDeleteShader(ps);
        GLint success = 0;
        glGetProgramiv(id, GL_LINK_STATUS, &success);
        if (!success) {
            GLint length = 0;
            glGetProgramiv(id, GL_INFO_LOG_LENGTH, &length);
            std::string log(std::max(length, 1), '\0');
            glGetProgramInfoLog(id, length, nullptr, log.data());
            throw std::runtime_error("Shader link failed:\n" + log);
        }
    }

    GLint location(const std::string& name) {
        const auto it = locations.find(name);
        if (it != locations.end()) return it->second;
        const GLint value = glGetUniformLocation(id, name.c_str());
        locations.emplace(name, value);
        return value;
    }

    void use() const { glUseProgram(id); }
    void set(const std::string& n, float v) { glUniform1f(location(n), v); }
    void set(const std::string& n, int v) { glUniform1i(location(n), v); }
    void set(const std::string& n, const glm::vec2& v) { glUniform2fv(location(n), 1, glm::value_ptr(v)); }
    void set(const std::string& n, const glm::vec3& v) { glUniform3fv(location(n), 1, glm::value_ptr(v)); }
    void set(const std::string& n, const glm::mat4& v) {
        glUniformMatrix4fv(location(n), 1, GL_FALSE, glm::value_ptr(v));
    }

    ~Program() {
        if (id) glDeleteProgram(id);
    }
};

struct GpuMesh {
    GLuint vao = 0, vbo = 0, ebo = 0;
    GLsizei count = 0;

    void upload(const MeshData& data) {
        count = static_cast<GLsizei>(data.indices.size());
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.vertices.size() * sizeof(SceneVertex)),
                     data.vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.indices.size() * sizeof(unsigned int)),
                     data.indices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SceneVertex),
                              reinterpret_cast<void*>(offsetof(SceneVertex, position)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SceneVertex),
                              reinterpret_cast<void*>(offsetof(SceneVertex, normal)));
    }

    void draw() const {
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, nullptr);
    }

    ~GpuMesh() { glDeleteVertexArrays(1, &vao); glDeleteBuffers(1, &vbo); glDeleteBuffers(1, &ebo); }
};

struct Framebuffer {
    GLuint fbo = 0, color = 0, depth = 0;
    int width = 0, height = 0;
    // Half floats are plenty for a single rendered frame, but not for a target
    // that sums hundreds of them: once the running total is large, fp16's 10
    // mantissa bits round each new sample's contribution away and the average
    // stops converging. The accumulation target overrides this to GL_RGBA32F.
    GLenum colorFormat = GL_RGBA16F;

    void resize(int w, int h) {
        if (w == width && h == height && fbo != 0) return;
        width = w;
        height = h;
        if (!fbo) {
            glGenFramebuffers(1, &fbo);
            glGenTextures(1, &color);
            glGenTextures(1, &depth);
        }
        glBindTexture(GL_TEXTURE_2D, color);
        glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(colorFormat), w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, depth);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Incomplete framebuffer");
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    ~Framebuffer() {
        if (fbo) glDeleteFramebuffers(1, &fbo);
        if (color) glDeleteTextures(1, &color);
        if (depth) glDeleteTextures(1, &depth);
    }
};

struct Options {
    int width = 1040, height = 700;
    int method = 1;
    int samples = 32;
    int frames = 0;
    float focus = -1.0f;
    float fNumber = 1.0f;
    float lens = 50.0f;
    float sensorHeight = 24.0f;
    float maxRadius = 40.0f;
    std::string capture;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
            return argv[++i];
        };
        if (flag == "--method") options.method = std::stoi(next());
        else if (flag == "--samples") options.samples = std::stoi(next());
        else if (flag == "--frames") options.frames = std::stoi(next());
        else if (flag == "--focus") options.focus = std::stof(next());
        else if (flag == "--f-number") options.fNumber = std::stof(next());
        else if (flag == "--lens") options.lens = std::stof(next());
        else if (flag == "--sensor-height") options.sensorHeight = std::stof(next());
        else if (flag == "--max-radius") options.maxRadius = std::stof(next());
        else if (flag == "--capture") options.capture = next();
        else if (flag == "--size") {
            const std::string size = next();
            const std::size_t separator = size.find('x');
            if (separator == std::string::npos) throw std::runtime_error("--size expects WIDTHxHEIGHT");
            options.width = std::stoi(size.substr(0, separator));
            options.height = std::stoi(size.substr(separator + 1));
        } else {
            throw std::runtime_error("Unknown option: " + flag);
        }
    }
    if (options.method < 1 || options.method > 3) throw std::runtime_error("--method must be 1, 2 or 3");
    if (options.samples < 1) throw std::runtime_error("--samples must be positive");
    if (options.fNumber <= 0.0f) throw std::runtime_error("--f-number must be positive");
    if (options.lens <= 0.0f) throw std::runtime_error("--lens must be positive");
    if (options.width < 320 || options.height < 200) throw std::runtime_error("--size is too small");
    return options;
}

struct Camera {
    glm::vec3 position{0.0f, 1.6f, 3.6f};
    float yawDegrees = -90.0f;
    float pitchDegrees = 0.0f;

    glm::vec3 forward() const {
        const float y = glm::radians(yawDegrees);
        const float p = glm::radians(pitchDegrees);
        return glm::normalize(glm::vec3(std::cos(y) * std::cos(p), std::sin(p), std::sin(y) * std::cos(p)));
    }
    glm::vec3 right() const { return glm::normalize(glm::cross(forward(), glm::vec3(0.0f, 1.0f, 0.0f))); }
    glm::vec3 up() const { return glm::normalize(glm::cross(right(), forward())); }
    glm::mat4 view() const { return glm::lookAt(position, position + forward(), glm::vec3(0.0f, 1.0f, 0.0f)); }
    void lookAt(const glm::vec3& target) {
        const glm::vec3 direction = glm::normalize(target - position);
        yawDegrees = glm::degrees(std::atan2(direction.z, direction.x));
        pitchDegrees = glm::degrees(std::asin(std::clamp(direction.y, -1.0f, 1.0f)));
    }
};

void savePng(const fs::path& path, int width, int height) {
    if (path.has_parent_path()) fs::create_directories(path.parent_path());
    std::vector<unsigned char> pixels(static_cast<std::size_t>(width) * height * 4), flipped(pixels.size());
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    for (int y = 0; y < height; ++y) {
        std::copy_n(pixels.data() + static_cast<std::size_t>(height - 1 - y) * width * 4, width * 4,
                    flipped.data() + static_cast<std::size_t>(y) * width * 4);
    }
    const std::string name = fs::absolute(path).string();
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(nullptr, reinterpret_cast<const UInt8*>(name.c_str()),
                                                           static_cast<CFIndex>(name.size()), false);
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr, flipped.data(), flipped.size(), nullptr);
    CGImageRef image = CGImageCreate(width, height, 8, 32, width * 4, space, kCGImageAlphaLast, provider, nullptr,
                                     false, kCGRenderingIntentDefault);
    CGImageDestinationRef destination = url ? CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, nullptr) : nullptr;
    bool success = false;
    if (destination && image) {
        CGImageDestinationAddImage(destination, image, nullptr);
        success = CGImageDestinationFinalize(destination);
    }
    if (destination) CFRelease(destination);
    if (image) CGImageRelease(image);
    if (provider) CGDataProviderRelease(provider);
    if (space) CGColorSpaceRelease(space);
    if (url) CFRelease(url);
    if (!success) throw std::runtime_error("Could not save screenshot: " + name);
    std::cout << "Saved " << name << " (" << width << "x" << height << ")\n";
}

std::string format(float value, int precision = 1) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

const char* methodName(int method) {
    switch (method) {
        case 1: return "1: screen-space gather";
        case 2: return "2: multi-view accumulation";
        default: return "3: lens-sampled ray tracing";
    }
}

int methodSampleLimit(int method) { return method == 3 ? 512 : 64; }

struct App {
    GLFWwindow* window = nullptr;
    Scene scene;
    std::vector<PackedPrimitive> packed;
    std::vector<GpuMesh> meshes;
    Program raster;
    Program postprocess;
    Program additive;
    Program raytrace;
    Program resolve;
    Framebuffer sceneFbo;
    Framebuffer accumFbo;
    GLuint emptyVao = 0;

    Camera camera;
    int method = 1;
    int samples = 32;
    float focusDistance = 12.0f;
    float fNumber = 1.0f;
    float focalLength = 0.05f;
    float sensorHeight = 0.024f;
    float maxRadiusPixels = 40.0f;
    float nearPlane = 0.1f;
    float farPlane = 240.0f;
    int accumulated = 0;
    bool accumClearPending = true;
    double lastCursorX = 0.0;
    double lastCursorY = 0.0;
    bool dragging = false;

    App(GLFWwindow* w, const Options& options, const fs::path& shaders) : window(w) {
        scene = createHaloScene();
        if (static_cast<int>(scene.primitives.size()) > kMaxPrimitives) {
            throw std::runtime_error("Scene has more primitives than the ray tracer supports");
        }
        packed = packPrimitives(scene);
        meshes.resize(scene.meshes.size());
        for (std::size_t i = 0; i < scene.meshes.size(); ++i) meshes[i].upload(scene.meshes[i]);
        glGenVertexArrays(1, &emptyVao);
        accumFbo.colorFormat = GL_RGBA32F;

        const std::string shading = readText(shaders / "shading.glsl");
        const std::string screenVertex = readText(shaders / "screen.vert");
        auto fragment = [&](const std::string& name) {
            return "#version 330 core\n" + shading + readText(shaders / name);
        };
        raster.build(readText(shaders / "scene.vert"), fragment("scene.frag"));
        postprocess.build(screenVertex, fragment("postprocess.frag"));
        additive.build(screenVertex, fragment("additive.frag"));
        raytrace.build(screenVertex, fragment("raytrace.frag"));
        resolve.build(screenVertex, fragment("resolve.frag"));

        camera.position = scene.eye;
        camera.lookAt(scene.target);
        focusDistance = options.focus > 0.0f ? options.focus : scene.focusDistance;
        fNumber = options.fNumber;
        focalLength = options.lens * 0.001f;
        sensorHeight = options.sensorHeight * 0.001f;
        maxRadiusPixels = options.maxRadius;
        method = options.method;
        samples = std::min(options.samples, methodSampleLimit(method));
    }

    ~App() { glDeleteVertexArrays(1, &emptyVao); }

    float verticalFov() const { return 2.0f * std::atan(sensorHeight / (2.0f * focalLength)); }
    float apertureRadius() const { return 0.5f * focalLength / std::max(fNumber, 0.1f); }

    void resetAccumulation() {
        accumulated = 0;
        accumClearPending = true;
    }

    void drawFullscreen() {
        glBindVertexArray(emptyVao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    // Renders the scene once from one aperture sample. The lateral eye offset
    // and the matching projection shear together keep the focus plane fixed,
    // so only defocused geometry moves between samples.
    void renderScenePass(const glm::vec3& eyeOffset, int width, int height) {
        // Camera space looks down -z, so a focus-plane point has z = -focusDistance.
        // Cancelling the lateral eye offset there needs x''/z to lose the
        // offset, which makes the shear coefficient negative once it is
        // written as x'' = x + c*z with z carrying the sign.
        const glm::mat4 shear = [&] {
            glm::mat4 matrix(1.0f);
            matrix[2][0] = -eyeOffset.x / focusDistance;
            matrix[2][1] = -eyeOffset.y / focusDistance;
            return matrix;
        }();
        const glm::vec3 eye = camera.position + eyeOffset;
        const glm::mat4 view = glm::lookAt(eye, eye + camera.forward(), glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::mat4 projection =
            glm::perspective(verticalFov(), static_cast<float>(width) / static_cast<float>(height), nearPlane, farPlane);
        const glm::mat4 viewProjection = projection * shear * view;

        glBindFramebuffer(GL_FRAMEBUFFER, sceneFbo.fbo);
        glViewport(0, 0, width, height);
        glClearColor(scene.background.r, scene.background.g, scene.background.b, 1.0f);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        raster.use();
        raster.set("uViewProjection", viewProjection);
        raster.set("uSunDirection", scene.sunDirection);
        for (std::size_t i = 0; i < meshes.size(); ++i) {
            const Primitive& primitive = scene.primitives[i];
            raster.set("uColor", primitive.color);
            raster.set("uEmissive", primitive.emissive);
            raster.set("uChecker", primitive.checker);
            meshes[i].draw();
        }
    }

    void addScenePassToAccumulation() {
        glBindFramebuffer(GL_FRAMEBUFFER, accumFbo.fbo);
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        additive.use();
        additive.set("uSource", 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sceneFbo.color);
        drawFullscreen();
        glDisable(GL_BLEND);
    }

    void renderMethodOne(int width, int height) {
        renderScenePass(glm::vec3(0.0f), width, height);
        glBindFramebuffer(GL_FRAMEBUFFER, accumFbo.fbo);
        glViewport(0, 0, width, height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        postprocess.use();
        postprocess.set("uColor", 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sceneFbo.color);
        postprocess.set("uDepth", 1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, sceneFbo.depth);
        postprocess.set("uNear", nearPlane);
        postprocess.set("uFar", farPlane);
        postprocess.set("uFocusDistance", focusDistance);
        postprocess.set("uFocalLength", focalLength);
        postprocess.set("uSensorHeight", sensorHeight);
        postprocess.set("uFNumber", fNumber);
        postprocess.set("uMaxRadiusPixels", maxRadiusPixels);
        postprocess.set("uResolution", glm::vec2(static_cast<float>(width), static_cast<float>(height)));
        postprocess.set("uTapCount", std::min(samples, 64));
        drawFullscreen();
    }

    void renderMethodTwo(int width, int height) {
        glBindFramebuffer(GL_FRAMEBUFFER, accumFbo.fbo);
        glViewport(0, 0, width, height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        const int views = std::min(samples, 64);
        const float radius = apertureRadius();
        const glm::vec3 right = camera.right();
        const glm::vec3 up = camera.up();
        for (int i = 0; i < views; ++i) {
            const float index = static_cast<float>(i) + 0.5f;
            const float r = radius * std::sqrt(index / static_cast<float>(views));
            const float angle = index * kGoldenAngle;
            renderScenePass(right * (std::cos(angle) * r) + up * (std::sin(angle) * r), width, height);
            addScenePassToAccumulation();
        }
    }

    void uploadPrimitives() {
        std::vector<glm::vec4> params, sizes, materials;
        params.reserve(packed.size());
        sizes.reserve(packed.size());
        materials.reserve(packed.size());
        for (const PackedPrimitive& primitive : packed) {
            params.push_back(primitive.params);
            sizes.push_back(primitive.size);
            materials.push_back(primitive.material);
        }
        const GLsizei count = static_cast<GLsizei>(packed.size());
        glUniform4fv(raytrace.location("uPrimParams[0]"), count, glm::value_ptr(params[0]));
        glUniform4fv(raytrace.location("uPrimSize[0]"), count, glm::value_ptr(sizes[0]));
        glUniform4fv(raytrace.location("uPrimMaterial[0]"), count, glm::value_ptr(materials[0]));
    }

    void renderMethodThree(int width, int height) {
        glBindFramebuffer(GL_FRAMEBUFFER, accumFbo.fbo);
        glViewport(0, 0, width, height);
        glDisable(GL_DEPTH_TEST);
        if (accumClearPending) {
            glDisable(GL_BLEND);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            accumClearPending = false;
        }
        if (accumulated >= samples) return;

        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        raytrace.use();
        raytrace.set("uEye", camera.position);
        raytrace.set("uForward", camera.forward());
        raytrace.set("uRight", camera.right());
        raytrace.set("uUp", camera.up());
        raytrace.set("uBackground", scene.background);
        raytrace.set("uSunDirection", scene.sunDirection);
        raytrace.set("uTanHalfFovy", std::tan(0.5f * verticalFov()));
        raytrace.set("uAspect", static_cast<float>(width) / static_cast<float>(height));
        raytrace.set("uApertureRadius", apertureRadius());
        raytrace.set("uFocusDistance", focusDistance);
        raytrace.set("uResolution", glm::vec2(static_cast<float>(width), static_cast<float>(height)));
        raytrace.set("uPrimitiveCount", static_cast<int>(packed.size()));
        raytrace.set("uFrameIndex", accumulated);
        uploadPrimitives();
        drawFullscreen();
        glDisable(GL_BLEND);
        ++accumulated;
    }

    void displayAccumulation(int width, int height, float sampleCount) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        resolve.use();
        resolve.set("uAccum", 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, accumFbo.color);
        resolve.set("uSampleCount", sampleCount);
        drawFullscreen();
    }

    void renderFrame(int width, int height) {
        sceneFbo.resize(width, height);
        accumFbo.resize(width, height);
        float displayed = 1.0f;
        switch (method) {
            case 2:
                renderMethodTwo(width, height);
                displayed = static_cast<float>(std::min(samples, 64));
                break;
            case 3:
                renderMethodThree(width, height);
                displayed = static_cast<float>(std::max(accumulated, 1));
                break;
            default:
                renderMethodOne(width, height);
                displayed = 1.0f;
                break;
        }
        displayAccumulation(width, height, displayed);

        std::ostringstream title;
        title << "DoF Approaches | " << methodName(method) << " | focus " << format(focusDistance, 2) << " m"
              << " | f/" << format(fNumber, 2) << " | " << format(focalLength * 1000.0f, 0) << " mm"
              << " | samples " << samples;
        if (method == 3) title << " | accumulated " << accumulated;
        glfwSetWindowTitle(window, title.str().c_str());
    }

    void onKey(int key) {
        switch (key) {
            case GLFW_KEY_1: method = 1; samples = std::min(samples, 64); resetAccumulation(); break;
            case GLFW_KEY_2: method = 2; samples = std::min(samples, 64); resetAccumulation(); break;
            case GLFW_KEY_3: method = 3; resetAccumulation(); break;
            case GLFW_KEY_UP:
                samples = std::min(methodSampleLimit(method), samples * 2);
                resetAccumulation();
                break;
            case GLFW_KEY_DOWN:
                samples = std::max(1, samples / 2);
                resetAccumulation();
                break;
            case GLFW_KEY_R: resetAccumulation(); break;
            case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, 1); break;
            default: break;
        }
    }

    void onScroll(double yOffset) {
        if (yOffset == 0.0) return;
        focusDistance = std::clamp(focusDistance * std::pow(1.08f, static_cast<float>(yOffset)), 0.5f, 200.0f);
        resetAccumulation();
    }

    void handleInput() {
        glm::vec3 move(0.0f);
        const glm::vec3 forward = camera.forward();
        const glm::vec3 right = camera.right();
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) move += forward;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) move -= forward;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) move += right;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) move -= right;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) move += glm::vec3(0.0f, 1.0f, 0.0f);
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) move -= glm::vec3(0.0f, 1.0f, 0.0f);
        if (glm::length(move) > 0.0f) {
            camera.position += glm::normalize(move) * 0.06f;
            resetAccumulation();
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS) {
            focusDistance = std::max(0.5f, focusDistance * 0.985f);
            resetAccumulation();
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) {
            focusDistance = std::min(200.0f, focusDistance * 1.015f);
            resetAccumulation();
        }
        if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS) {
            fNumber = std::max(0.7f, fNumber / 1.015f);
            resetAccumulation();
        }
        if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS) {
            fNumber = std::min(22.0f, fNumber * 1.015f);
            resetAccumulation();
        }
        if (dragging) {
            double x = 0.0, y = 0.0;
            glfwGetCursorPos(window, &x, &y);
            const float dx = static_cast<float>(x - lastCursorX);
            const float dy = static_cast<float>(y - lastCursorY);
            lastCursorX = x;
            lastCursorY = y;
            if (dx != 0.0f || dy != 0.0f) {
                camera.yawDegrees += dx * 0.12f;
                camera.pitchDegrees = std::clamp(camera.pitchDegrees - dy * 0.12f, -85.0f, 85.0f);
                resetAccumulation();
            }
        }
    }
};

int run(GLFWwindow* window, const Options& options, const fs::path& shaders) {
    App app(window, options, shaders);
    glfwSetWindowUserPointer(window, &app);
    glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int, int action, int) {
        if (action != GLFW_PRESS) return;
        if (auto* instance = static_cast<App*>(glfwGetWindowUserPointer(w))) instance->onKey(key);
    });
    glfwSetScrollCallback(window, [](GLFWwindow* w, double, double yOffset) {
        if (auto* instance = static_cast<App*>(glfwGetWindowUserPointer(w))) instance->onScroll(yOffset);
    });
    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int button, int action, int) {
        auto* instance = static_cast<App*>(glfwGetWindowUserPointer(w));
        if (!instance || button != GLFW_MOUSE_BUTTON_RIGHT) return;
        instance->dragging = action == GLFW_PRESS;
        glfwGetCursorPos(w, &instance->lastCursorX, &instance->lastCursorY);
    });

    const bool capturing = !options.capture.empty();
    int frames = options.frames;
    if (frames <= 0) frames = options.method == 3 ? options.samples + 2 : 3;

    auto step = [&]() {
        glfwPollEvents();
        app.handleInput();
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (width <= 0 || height <= 0) return false;
        app.renderFrame(width, height);
        return true;
    };

    if (capturing) {
        int lastWidth = options.width, lastHeight = options.height;
        for (int i = 0; i < frames && !glfwWindowShouldClose(window); ++i) {
            if (!step()) continue;
            glfwGetFramebufferSize(window, &lastWidth, &lastHeight);
        }
        glFinish();
        savePng(options.capture, lastWidth, lastHeight);
        return 0;
    }

    while (!glfwWindowShouldClose(window)) {
        if (!step()) continue;
        glfwSwapBuffers(window);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    GLFWwindow* window = nullptr;
    try {
        const Options options = parseOptions(argc, argv);
        glfwSetErrorCallback([](int code, const char* message) { std::cerr << "GLFW " << code << ": " << message << "\n"; });
        if (!glfwInit()) throw std::runtime_error("GLFW initialization failed");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        window = glfwCreateWindow(options.width, options.height, "DoF Approaches", nullptr, nullptr);
        if (!window) throw std::runtime_error("Cannot create OpenGL window");
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        std::cout << "Renderer: " << glGetString(GL_RENDERER) << "\n";
        fs::path shaders = fs::path(argv[0]).parent_path() / "shaders/dof_approaches";
        if (!fs::exists(shaders / "shading.glsl")) shaders = fs::path(PROJECT_SOURCE_DIR) / "shaders/dof_approaches";
        const int result = run(window, options, shaders);
        glfwDestroyWindow(window);
        glfwTerminate();
        return result;
    } catch (const std::exception& error) {
        std::cerr << "DoFApproaches: " << error.what() << "\n";
        if (window) glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
}
