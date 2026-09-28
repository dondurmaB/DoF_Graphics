#include <glad/glad.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include "Mesh.h"
#include "PhysicalCamera.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

#ifndef GL_TEXTURE_2D
#define GL_TEXTURE_2D 0x0DE1
#endif
#ifndef GL_RGBA
#define GL_RGBA 0x1908
#endif
#ifndef GL_TEXTURE_MIN_FILTER
#define GL_TEXTURE_MIN_FILTER 0x2801
#endif
#ifndef GL_TEXTURE_MAG_FILTER
#define GL_TEXTURE_MAG_FILTER 0x2800
#endif
#ifndef GL_TEXTURE_WRAP_S
#define GL_TEXTURE_WRAP_S 0x2802
#endif
#ifndef GL_TEXTURE_WRAP_T
#define GL_TEXTURE_WRAP_T 0x2803
#endif
#ifndef GL_LINEAR
#define GL_LINEAR 0x2601
#endif
#ifndef GL_NEAREST
#define GL_NEAREST 0x2600
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#ifndef GL_FRAMEBUFFER
#define GL_FRAMEBUFFER 0x8D40
#endif
#ifndef GL_COLOR_ATTACHMENT0
#define GL_COLOR_ATTACHMENT0 0x8CE0
#endif
#ifndef GL_DEPTH_ATTACHMENT
#define GL_DEPTH_ATTACHMENT 0x8D00
#endif
#ifndef GL_FRAMEBUFFER_COMPLETE
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#endif
#ifndef GL_DEPTH_COMPONENT
#define GL_DEPTH_COMPONENT 0x1902
#endif
#ifndef GL_DEPTH_COMPONENT24
#define GL_DEPTH_COMPONENT24 0x81A6
#endif
#ifndef GL_TEXTURE0
#define GL_TEXTURE0 0x84C0
#endif
#ifndef GL_TEXTURE1
#define GL_TEXTURE1 0x84C1
#endif
#ifndef GL_TEXTURE2
#define GL_TEXTURE2 0x84C2
#endif
#ifndef GL_NONE
#define GL_NONE 0x0000
#endif

// DEVELOPMENT SETTINGS
const int windowWidth = 1200;
const int windowHeight = 1200;
const int windowPosX = 50;
const int windowPosY = 100;

// ==============================
// EXPERIMENT 16 CONTROLS
// ==============================
enum class ScreenMode {
    Color = 0,
    RawDepth = 1,
    LinearDepth = 2,
    CoCMagnitude = 3,
    CoCSigned = 4,
    BasicDoF = 5
};

ScreenMode screenMode = ScreenMode::BasicDoF;

// Physical convention: 1 world unit = 1 meter.
bool usePhysicalCameraProjection = true;
float focusDistanceMeters = 5.0f;
float focalLengthMillimeters = 50.0f;
float fNumber = 10.0f;
float sensorHeightMillimeters = 24.0f;
// Debug visualization scaling only; does not change the rendered blur.
float cocVisualizationMaxPixels = 20.0f;
// Actual maximum gathering radius used by BasicDoF.
float maxBlurRadiusPixels = 120.0f;
// Aperture samples per pixel for the BasicDoF gather (shaders/screen.frag).
// At a 120 px radius, 16 taps band visibly; 100 is the meeting's target.
int cocSampleCount = 100;

// ==============================
// EXPERIMENT 17 CONTROLS: shading, shadows, live UI
// ==============================
// Points FROM a lit surface TOWARD the light. Matches
// tools/raytraced_reference/render_dof.py's LIGHT_DIRECTION_GL so the raster
// pass and the Cycles ground truth share one lighting setup.
glm::vec3 lightDirection = glm::vec3(-0.4f, 0.8f, 0.6f);
glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
float ambientStrength = 0.25f;
bool enableShadows = true;
const int shadowMapSize = 2048;
// Half-extents of the orthographic light frustum, world-space meters. Sized
// to cover the box layout below (roughly x[-4,3], y[-1,4], z[3,-22]) with
// margin; not derived automatically, same "explicit over inferred" approach
// the project already uses for camera and scene scale.
const float shadowFrustumHalfExtent = 20.0f;
const float shadowNearPlane = 0.1f;
const float shadowFarPlane = 60.0f;

bool showControlPanel = true;

// OBJ units are arbitrary. Choose meters per authored unit explicitly; never auto-normalize.
const string importedScenePath = "assets/models/scene.obj";
float importedSceneScale = 0.1f;
glm::vec3 importedScenePosition = glm::vec3(0.0f, -0.75f, 0.0f);
float importedSceneRotationYDegrees = 0.0f;
bool showEnvironment = true;

float fieldOfViewDegrees = 45.0f; // Retained for legacy/manual projection only.
// Near and far define the camera-space depth range that can appear after projection.
float nearPlane = 0.1f;
float farPlane = 100.0f;

// Controls only the displayed grayscale range for reconstructed linear depth.
float depthVisualizationMax = 25.0f;

bool renderThroughFramebuffer = true;

// ==============================
// EXPERIMENT 07 PROJECTION CONTROLS
// ==============================

// Try false to remove perspective projection; this is not orthographic projection.
bool usePerspectiveProjection = true;

// Initial camera is at z=5: these centers provide roughly 2, 5, 10, 15, 25 m depth.
glm::vec3 cubeAPosition = glm::vec3(-0.55f, -0.35f, 3.0f);
float cubeARotationXDegrees = 50.0f;
float cubeARotationYDegrees = 70.0f;
float cubeAUniformScale = 0.5f;

glm::vec3 cubeBPosition = glm::vec3(0.0f, 0.0f, 0.0f);
float cubeBRotationXDegrees = -20.0f;
float cubeBRotationYDegrees = -35.0f;
float cubeBUniformScale = 1.0f;

glm::vec3 cubeCPosition = glm::vec3(0.0f, 0.0f, -5.0f);
float cubeCRotationXDegrees = 15.0f;
float cubeCRotationYDegrees = 35.0f;
float cubeCUniformScale = 0.5f;

glm::vec3 cubeDPosition = glm::vec3(1.6f, 0.4f, -10.0f);
float cubeDRotationXDegrees = 35.0f;
float cubeDRotationYDegrees = -20.0f;
float cubeDUniformScale = 2.0f;

glm::vec3 cubeEPosition = glm::vec3(-3.8f, 0.4f, -20.0f);
float cubeERotationXDegrees = -15.0f;
float cubeERotationYDegrees = 60.0f;
float cubeEUniformScale = 2.0f;

// ==============================
// EXPERIMENT 17 SCENE ADDITIONS: extra depth layers for a richer composition
// ==============================
// A/D/E above are read by tests/test_reference_config.py, which regex-matches
// them against tools/raytraced_reference/render_dof.py's BOXES tuple; F and G
// are new and are mirrored there separately (BOXES entries midground_F,
// background_G) so the Cycles reference keeps the same environment.
glm::vec3 cubeFPosition = glm::vec3(2.4f, -0.3f, -2.0f);
float cubeFRotationXDegrees = 10.0f;
float cubeFRotationYDegrees = -25.0f;
float cubeFUniformScale = 0.35f;

glm::vec3 cubeGPosition = glm::vec3(-2.2f, 0.2f, -13.0f);
float cubeGRotationXDegrees = 0.0f;
float cubeGRotationYDegrees = 40.0f;
float cubeGUniformScale = 1.2f;

// ==============================
// EXPERIMENT 08 CAMERA CONTROLS
// ==============================
// Position is where the camera is in world space.
glm::vec3 cameraPosition = glm::vec3(0.0f, 0.0f, 5.0f);
// Front is the direction the camera is looking.
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
// worldUp is the global reference for vertical direction.
glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);

// Yaw rotates left/right. Pitch rotates up/down.
float yawDegrees = -90.0f;
float pitchDegrees = 0.0f;

// R restores the configured starting view for repeatable focus/aperture comparisons.
const glm::vec3 initialCameraPosition = cameraPosition;
const float initialYawDegrees = yawDegrees;
const float initialPitchDegrees = pitchDegrees;

float movementSpeed = 2.5f;
float mouseSensitivity = 0.1f;

bool printCameraState = false;

float deltaTime = 0.0f;
float lastFrameTime = 0.0f;
float lastMouseX = windowWidth / 2.0f;
float lastMouseY = windowHeight / 2.0f;
bool firstMouse = true;
float lastCameraPrintTime = 0.0f;

bool animateIntensity = false;
float intensitySpeed = 1.0f;
float staticIntensity = 1.0f;

// Draw filled cube faces by default. Try GL_LINE to inspect the triangle mesh.
bool wireframeMode = false;

string readFile(const string& path) {
    ifstream file(path);

    if (!file.is_open()) {
        cout << "Failed to open shader file: " << path << endl;
        return "";
    }

    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

const char* screenModeName() {
    switch (screenMode) {
        case ScreenMode::Color: return "Color";
        case ScreenMode::RawDepth: return "RawDepth";
        case ScreenMode::LinearDepth: return "LinearDepth";
        case ScreenMode::CoCMagnitude: return "CoCMagnitude";
        case ScreenMode::CoCSigned: return "CoCSigned";
        case ScreenMode::BasicDoF: return "BasicDoF";
    }
    return "Unknown";
}

void showVerificationStatus(GLFWwindow* window, bool printToConsole = true) {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    const bool mouseLook = glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;
    ostringstream title;
    title << "DOF_Research | " << screenModeName() << " | focus " << focusDistanceMeters
          << " m | f/" << fNumber << " | " << width << "x" << height << " px"
          << (usePhysicalCameraProjection ? " | physical" : " | legacy")
          << (mouseLook ? " | mouse look" : " | cursor free");
    glfwSetWindowTitle(window, title.str().c_str());
    if (printToConsole) {
        cout << title.str() << " | camera (" << cameraPosition.x << ", " << cameraPosition.y
             << ", " << cameraPosition.z << ") | yaw/pitch " << yawDegrees << "/" << pitchDegrees << endl;
        cout << "Physical camera: " << focalLengthMillimeters << " mm lens | "
             << sensorHeightMillimeters << " mm sensor height | vertical FOV "
             << glm::degrees(physicalVerticalFovRadians(focalLengthMillimeters, sensorHeightMillimeters))
             << " degrees | projection " << (usePhysicalCameraProjection ? "physical" : "legacy")
             << " | legacy FOV " << fieldOfViewDegrees << " degrees" << endl;
    }
}

void printVerificationHelp() {
    cout << "Experiment 17 verification keys (no rebuild needed):\n"
         << "  1 Color | 2 RawDepth | 3 LinearDepth | 4 CoCMagnitude | 5 CoCSigned | 6 BasicDoF\n"
         << "  7 focus 2 m | 8 focus 5 m | 9 focus 15 m | F toggle f/1.4 and f/8\n"
         << "  R reset camera | Tab release/capture cursor (pauses/resumes camera)\n"
         << "  T reference preset + release cursor | V physical/legacy projection | B f/2.8\n"
         << "  [ / ] focal length -/+5 mm | , / . sensor height -/+2 mm\n"
         << "  WASD + mouse navigate | P screenshot | H help/status | Escape exit\n"
         << "  G show/hide control panel (press Tab first to free the cursor for its sliders)\n";
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    showVerificationStatus(window, false);
}

void updateCameraFront() {
    glm::vec3 front;
    front.x = cos(glm::radians(yawDegrees)) * cos(glm::radians(pitchDegrees));
    front.y = sin(glm::radians(pitchDegrees));
    front.z = sin(glm::radians(yawDegrees)) * cos(glm::radians(pitchDegrees));
    cameraFront = glm::normalize(front);
}

// Callbacks are installed before the ImGui context exists, so an early event must not touch ImGui::GetIO().
static bool imguiWantsMouse() { return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse; }
static bool imguiWantsKeyboard() { return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureKeyboard; }

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (imguiWantsMouse()) {
        // The control panel wants this mouse motion (e.g. dragging a slider); don't also turn the camera.
        firstMouse = true;
        return;
    }
    if (glfwGetInputMode(window, GLFW_CURSOR) != GLFW_CURSOR_DISABLED) {
        firstMouse = true;
        return;
    }
    if (firstMouse) {
        lastMouseX = static_cast<float>(xpos);
        lastMouseY = static_cast<float>(ypos);
        firstMouse = false;
        return;
    }

    float xOffset = static_cast<float>(xpos) - lastMouseX;
    float yOffset = lastMouseY - static_cast<float>(ypos);
    lastMouseX = static_cast<float>(xpos);
    lastMouseY = static_cast<float>(ypos);

    xOffset *= mouseSensitivity;
    yOffset *= mouseSensitivity;

    yawDegrees += xOffset;
    pitchDegrees += yOffset;

    // Clamp pitch before +/-90 degrees to avoid unstable flipped camera behavior.
    pitchDegrees = clamp(pitchDegrees, -89.0f, 89.0f);

    updateCameraFront();
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // Ignore repeats: holding F or Tab must not toggle repeatedly.
    if (action != GLFW_PRESS) return;

    // Let the control panel consume keys while it has focus, except Tab (release/capture
    // the cursor, which is how the panel becomes reachable) and G (panel visibility itself).
    if (imguiWantsKeyboard() && key != GLFW_KEY_TAB && key != GLFW_KEY_G) return;

    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_6) {
        screenMode = static_cast<ScreenMode>(key - GLFW_KEY_1);
    } else if (key == GLFW_KEY_7) {
        focusDistanceMeters = 2.0f;
    } else if (key == GLFW_KEY_8) {
        focusDistanceMeters = 5.0f;
    } else if (key == GLFW_KEY_9) {
        focusDistanceMeters = 15.0f;
    } else if (key == GLFW_KEY_F) {
        fNumber = fNumber < 4.0f ? 8.0f : 1.4f;
    } else if (key == GLFW_KEY_B) {
        fNumber = 2.8f;
    } else if (key == GLFW_KEY_V) {
        usePhysicalCameraProjection = !usePhysicalCameraProjection;
    } else if (key == GLFW_KEY_LEFT_BRACKET || key == GLFW_KEY_RIGHT_BRACKET) {
        focalLengthMillimeters = clamp(focalLengthMillimeters + (key == GLFW_KEY_RIGHT_BRACKET ? 5.0f : -5.0f), 10.0f, 200.0f);
    } else if (key == GLFW_KEY_COMMA || key == GLFW_KEY_PERIOD) {
        sensorHeightMillimeters = clamp(sensorHeightMillimeters + (key == GLFW_KEY_PERIOD ? 2.0f : -2.0f), 4.0f, 70.0f);
    } else if (key == GLFW_KEY_T) {
        // Match the separate Cycles script; preserve actual framebuffer dimensions.
        cameraPosition = initialCameraPosition;
        yawDegrees = initialYawDegrees;
        pitchDegrees = initialPitchDegrees;
        updateCameraFront();
        focusDistanceMeters = 5.0f;
        focalLengthMillimeters = 50.0f;
        sensorHeightMillimeters = 24.0f;
        fNumber = 1.4f;
        usePhysicalCameraProjection = true;
        usePerspectiveProjection = true;
        screenMode = ScreenMode::BasicDoF;
        firstMouse = true;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    } else if (key == GLFW_KEY_R) {
        cameraPosition = initialCameraPosition;
        yawDegrees = initialYawDegrees;
        pitchDegrees = initialPitchDegrees;
        updateCameraFront();
        firstMouse = true;
    } else if (key == GLFW_KEY_TAB) {
        const bool mouseLook = glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;
        firstMouse = true;
        glfwSetInputMode(window, GLFW_CURSOR, mouseLook ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
    } else if (key == GLFW_KEY_H) {
        printVerificationHelp();
    } else if (key == GLFW_KEY_G) {
        showControlPanel = !showControlPanel;
    } else {
        return;
    }
    showVerificationStatus(window);
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    // While the control panel is focused, let WASD type/interact there instead of moving the camera.
    if (imguiWantsKeyboard()) return;

    // Keep the comparison view still while the cursor is released for window resizing.
    if (glfwGetInputMode(window, GLFW_CURSOR) != GLFW_CURSOR_DISABLED) return;

    // movementSpeed * deltaTime makes camera movement approximately independent of frame rate.
    float cameraMovement = movementSpeed * deltaTime;

    // front = direction camera looks.
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        cameraPosition += cameraFront * cameraMovement;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        cameraPosition -= cameraFront * cameraMovement;
    }

    // right = normalize(cross(front, worldUp)); up can conceptually be derived from cross(right, front).
    glm::vec3 cameraRight = glm::normalize(glm::cross(cameraFront, worldUp));
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        cameraPosition -= cameraRight * cameraMovement;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        cameraPosition += cameraRight * cameraMovement;
    }
}

bool checkShaderCompilation(unsigned int shader, const string& shaderName) {
    int success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        int logLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);

        string infoLog(logLength, '\0');
        glGetShaderInfoLog(shader, logLength, NULL, infoLog.data());

        cout << "Failed to compile shader: " << shaderName << endl;
        cout << infoLog << endl;
        return false;
    }

    return true;
}

bool checkProgramLinking(unsigned int shaderProgram) {
    int success = 0;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

    if (!success) {
        int logLength = 0;
        glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &logLength);

        string infoLog(logLength, '\0');
        glGetProgramInfoLog(shaderProgram, logLength, NULL, infoLog.data());

        cout << "Failed to link shader program" << endl;
        cout << infoLog << endl;
        return false;
    }

    return true;
}

unsigned int createShaderProgram(const string& vertexPath, const string& fragmentPath,
                                 const string& vertexName, const string& fragmentName) {
    string vertexCode = readFile(vertexPath);
    string fragmentCode = readFile(fragmentPath);
    if (vertexCode.empty() || fragmentCode.empty()) {
        return 0;
    }

    const char* vertexShaderSource = vertexCode.c_str();
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    if (!checkShaderCompilation(vertexShader, vertexName)) {
        glDeleteShader(vertexShader);
        return 0;
    }

    const char* fragmentShaderSource = fragmentCode.c_str();
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    if (!checkShaderCompilation(fragmentShader, fragmentName)) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return 0;
    }

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    if (!checkProgramLinking(shaderProgram)) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        glDeleteProgram(shaderProgram);
        return 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return shaderProgram;
}

void appendUint32BE(vector<unsigned char>& bytes, uint32_t value) {
    bytes.push_back(static_cast<unsigned char>((value >> 24) & 0xff));
    bytes.push_back(static_cast<unsigned char>((value >> 16) & 0xff));
    bytes.push_back(static_cast<unsigned char>((value >> 8) & 0xff));
    bytes.push_back(static_cast<unsigned char>(value & 0xff));
}

uint32_t crc32(const unsigned char* data, size_t size) {
    uint32_t crc = 0xffffffffu;

    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            if (crc & 1u) {
                crc = (crc >> 1u) ^ 0xedb88320u;
            } else {
                crc >>= 1u;
            }
        }
    }

    return crc ^ 0xffffffffu;
}

uint32_t adler32(const vector<unsigned char>& data) {
    uint32_t a = 1;
    uint32_t b = 0;

    for (unsigned char byte : data) {
        a = (a + byte) % 65521u;
        b = (b + a) % 65521u;
    }

    return (b << 16u) | a;
}

void appendPngChunk(vector<unsigned char>& png, const char type[4], const vector<unsigned char>& data) {
    appendUint32BE(png, static_cast<uint32_t>(data.size()));

    size_t chunkStart = png.size();
    png.insert(png.end(), type, type + 4);
    png.insert(png.end(), data.begin(), data.end());

    appendUint32BE(png, crc32(png.data() + chunkStart, png.size() - chunkStart));
}

bool writePng(const string& path, int width, int height, const vector<unsigned char>& rgbPixels) {
    vector<unsigned char> scanlines;
    const int rowSize = width * 3;
    scanlines.reserve(static_cast<size_t>((rowSize + 1) * height));

    for (int y = 0; y < height; ++y) {
        scanlines.push_back(0); // PNG filter type 0: no filter.
        const unsigned char* rowStart = rgbPixels.data() + static_cast<size_t>(y * rowSize);
        scanlines.insert(scanlines.end(), rowStart, rowStart + rowSize);
    }

    vector<unsigned char> compressed;
    compressed.push_back(0x78); // zlib header for uncompressed deflate data.
    compressed.push_back(0x01);

    size_t offset = 0;
    while (offset < scanlines.size()) {
        const uint16_t blockSize = static_cast<uint16_t>(min<size_t>(65535, scanlines.size() - offset));
        const bool finalBlock = offset + blockSize >= scanlines.size();

        compressed.push_back(finalBlock ? 0x01 : 0x00);
        compressed.push_back(static_cast<unsigned char>(blockSize & 0xff));
        compressed.push_back(static_cast<unsigned char>((blockSize >> 8) & 0xff));

        const uint16_t inverseBlockSize = static_cast<uint16_t>(~blockSize);
        compressed.push_back(static_cast<unsigned char>(inverseBlockSize & 0xff));
        compressed.push_back(static_cast<unsigned char>((inverseBlockSize >> 8) & 0xff));

        compressed.insert(compressed.end(), scanlines.begin() + static_cast<long>(offset),
                          scanlines.begin() + static_cast<long>(offset + blockSize));
        offset += blockSize;
    }

    appendUint32BE(compressed, adler32(scanlines));

    vector<unsigned char> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};

    vector<unsigned char> ihdr;
    appendUint32BE(ihdr, static_cast<uint32_t>(width));
    appendUint32BE(ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(8); // 8 bits per channel.
    ihdr.push_back(2); // RGB color.
    ihdr.push_back(0); // deflate compression.
    ihdr.push_back(0); // standard PNG filter method.
    ihdr.push_back(0); // no interlacing.

    appendPngChunk(png, "IHDR", ihdr);
    appendPngChunk(png, "IDAT", compressed);
    appendPngChunk(png, "IEND", {});

    ofstream file(path, ios::binary);
    if (!file.is_open()) {
        cout << "Failed to save screenshot: " << path << endl;
        return false;
    }

    file.write(reinterpret_cast<const char*>(png.data()), static_cast<streamsize>(png.size()));
    return file.good();
}

bool saveScreenshot(const string& path, int width, int height) {
    vector<unsigned char> pixels(static_cast<size_t>(width * height * 3));
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    vector<unsigned char> flippedPixels(pixels.size());
    const int rowSize = width * 3;
    for (int y = 0; y < height; ++y) {
        const unsigned char* source = pixels.data() + static_cast<size_t>((height - 1 - y) * rowSize);
        unsigned char* destination = flippedPixels.data() + static_cast<size_t>(y * rowSize);
        copy(source, source + rowSize, destination);
    }

    filesystem::create_directories(filesystem::path(path).parent_path());

    if (writePng(path, width, height, flippedPixels)) {
        cout << "Saved screenshot: " << path << endl;
        return true;
    }

    return false;
}

struct ExtraGlFunctions {
    void (*genFramebuffers)(GLsizei, GLuint*) = nullptr;
    void (*bindFramebuffer)(GLenum, GLuint) = nullptr;
    void (*framebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint) = nullptr;
    GLenum (*checkFramebufferStatus)(GLenum) = nullptr;
    void (*deleteFramebuffers)(GLsizei, const GLuint*) = nullptr;
    void (*genTextures)(GLsizei, GLuint*) = nullptr;
    void (*bindTexture)(GLenum, GLuint) = nullptr;
    void (*texImage2D)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*) = nullptr;
    void (*texParameteri)(GLenum, GLenum, GLint) = nullptr;
    void (*deleteTextures)(GLsizei, const GLuint*) = nullptr;
    void (*activeTexture)(GLenum) = nullptr;
    void (*uniform1i)(GLint, GLint) = nullptr;
    void (*uniform3fv)(GLint, GLsizei, const GLfloat*) = nullptr;
    // A depth-only FBO (the shadow map) needs both set to GL_NONE to be complete on strict drivers.
    void (*drawBuffer)(GLenum) = nullptr;
    void (*readBuffer)(GLenum) = nullptr;
};

template <typename FunctionPointer>
bool loadGlFunction(FunctionPointer& function, const char* name) {
    function = reinterpret_cast<FunctionPointer>(glfwGetProcAddress(name));
    if (!function) {
        cout << "Failed to load OpenGL function: " << name << endl;
        return false;
    }
    return true;
}

bool loadExtraGlFunctions(ExtraGlFunctions& functions) {
    return loadGlFunction(functions.genFramebuffers, "glGenFramebuffers") &&
           loadGlFunction(functions.bindFramebuffer, "glBindFramebuffer") &&
           loadGlFunction(functions.framebufferTexture2D, "glFramebufferTexture2D") &&
           loadGlFunction(functions.checkFramebufferStatus, "glCheckFramebufferStatus") &&
           loadGlFunction(functions.deleteFramebuffers, "glDeleteFramebuffers") &&
           loadGlFunction(functions.genTextures, "glGenTextures") &&
           loadGlFunction(functions.bindTexture, "glBindTexture") &&
           loadGlFunction(functions.texImage2D, "glTexImage2D") &&
           loadGlFunction(functions.texParameteri, "glTexParameteri") &&
           loadGlFunction(functions.deleteTextures, "glDeleteTextures") &&
           loadGlFunction(functions.activeTexture, "glActiveTexture") &&
           loadGlFunction(functions.uniform1i, "glUniform1i") &&
           loadGlFunction(functions.uniform3fv, "glUniform3fv") &&
           loadGlFunction(functions.drawBuffer, "glDrawBuffer") &&
           loadGlFunction(functions.readBuffer, "glReadBuffer");
}

int main() {
    try {
        cout << "Physical vertical FOV: "
             << glm::degrees(physicalVerticalFovRadians(focalLengthMillimeters, sensorHeightMillimeters))
             << " degrees (lens " << focalLengthMillimeters << " mm, sensor height "
             << sensorHeightMillimeters << " mm) | focus " << focusDistanceMeters
             << " m | f/" << fNumber << endl;
    } catch (const std::invalid_argument& error) {
        cerr << error.what() << endl;
        return 1;
    }
    MeshData importedData;
    string meshError;
    string meshWarning;
    const filesystem::path modelPath = filesystem::path(PROJECT_SOURCE_DIR) / importedScenePath;
    cout << "OBJ loader: tinyobjloader | model: " << importedScenePath
         << " | importedSceneScale: " << importedSceneScale << " meters/unit" << endl;
    bool hasImportedMesh = false;
    if (!std::isfinite(importedSceneScale) || importedSceneScale <= 0.0f) {
        meshError = "importedSceneScale must be finite and positive.";
    } else {
        hasImportedMesh = loadObjMesh(modelPath, importedData, meshError, meshWarning);
    }
    if (!meshWarning.empty()) cout << "OBJ warning: " << meshWarning << endl;
    if (!hasImportedMesh) {
        cout << "Failed to load " << importedScenePath << ": " << meshError
             << "\nUsing fallback scene." << endl;
    } else {
        cout << "Loaded " << importedScenePath << ": " << importedData.vertices.size()
             << " vertices, " << importedData.indices.size() << " indices, "
             << importedData.indices.size() / 3 << " triangles"
             << " | source normals: " << (importedData.hasNormals ? "present" : "absent")
             << " | generated flat normals: " << (importedData.generatedNormals ? "yes" : "no")
             << " | source UVs: " << (importedData.hasTexCoords ? "present (unused for shading)" : "absent") << endl;
        cout << "Authored bounds: (" << importedData.boundsMin.x << ", " << importedData.boundsMin.y
             << ", " << importedData.boundsMin.z << ") to (" << importedData.boundsMax.x << ", "
             << importedData.boundsMax.y << ", " << importedData.boundsMax.z << ")" << endl;
    }

    if (!glfwInit()) {
        cout << "Failed to initialize GLFW" << endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "DOF_Research", NULL, NULL);
    if (window == NULL) {
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return -1;
    }
    glfwSetWindowPos(window, windowPosX, windowPosY);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);  
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadfunc)glfwGetProcAddress)) {
        cout << "Failed to initialize GLAD" << endl;
        glfwTerminate();
        return -1;
    }

    ExtraGlFunctions extraGl;
    if (!loadExtraGlFunctions(extraGl)) {
        glfwTerminate();
        return -1;
    }

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    glViewport(0, 0, framebufferWidth, framebufferHeight);

    const string shaderDirectory = string(PROJECT_SOURCE_DIR) + "/shaders/";
    unsigned int shaderProgram = createShaderProgram(shaderDirectory + "basic.vert",
                                                     shaderDirectory + "basic.frag",
                                                     "basic.vert",
                                                     "basic.frag");
    if (shaderProgram == 0) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    unsigned int screenShaderProgram = createShaderProgram(shaderDirectory + "screen.vert",
                                                           shaderDirectory + "screen.frag",
                                                           "screen.vert",
                                                           "screen.frag");
    if (screenShaderProgram == 0) {
        glDeleteProgram(shaderProgram);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    unsigned int shadowShaderProgram = createShaderProgram(shaderDirectory + "shadow.vert",
                                                            shaderDirectory + "shadow.frag",
                                                            "shadow.vert",
                                                            "shadow.frag");
    if (shadowShaderProgram == 0) {
        glDeleteProgram(shaderProgram);
        glDeleteProgram(screenShaderProgram);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Uniform locations are queried once after linking. A -1 location can mean the uniform was optimized away.
    int intensityLocation = glGetUniformLocation(shaderProgram, "uIntensity");
    if (intensityLocation == -1) {
        cout << "Warning: could not find uniform uIntensity" << endl;
    }

    int importedMeshLocation = glGetUniformLocation(shaderProgram, "uImportedMesh");

    int modelLocation = glGetUniformLocation(shaderProgram, "uModel");
    if (modelLocation == -1) {
        cout << "Warning: could not find uniform uModel" << endl;
    }

    int viewLocation = glGetUniformLocation(shaderProgram, "uView");
    if (viewLocation == -1) {
        cout << "Warning: could not find uniform uView" << endl;
    }

    int projectionLocation = glGetUniformLocation(shaderProgram, "uProjection");
    if (projectionLocation == -1) {
        cout << "Warning: could not find uniform uProjection" << endl;
    }

    int lightSpaceMatrixLocation = glGetUniformLocation(shaderProgram, "uLightSpaceMatrix");
    if (lightSpaceMatrixLocation == -1) {
        cout << "Warning: could not find uniform uLightSpaceMatrix" << endl;
    }

    int lightDirectionLocation = glGetUniformLocation(shaderProgram, "uLightDirection");
    if (lightDirectionLocation == -1) {
        cout << "Warning: could not find uniform uLightDirection" << endl;
    }

    int lightColorLocation = glGetUniformLocation(shaderProgram, "uLightColor");
    if (lightColorLocation == -1) {
        cout << "Warning: could not find uniform uLightColor" << endl;
    }

    int ambientStrengthLocation = glGetUniformLocation(shaderProgram, "uAmbientStrength");
    if (ambientStrengthLocation == -1) {
        cout << "Warning: could not find uniform uAmbientStrength" << endl;
    }

    int useShadowsLocation = glGetUniformLocation(shaderProgram, "uUseShadows");
    if (useShadowsLocation == -1) {
        cout << "Warning: could not find uniform uUseShadows" << endl;
    }

    int shadowMapLocation = glGetUniformLocation(shaderProgram, "uShadowMap");
    if (shadowMapLocation == -1) {
        cout << "Warning: could not find uniform uShadowMap" << endl;
    }

    // The shadow pass shares the same uModel name so drawBox/drawCube can upload
    // it identically whichever program is currently bound.
    int shadowModelLocation = glGetUniformLocation(shadowShaderProgram, "uModel");
    if (shadowModelLocation == -1) {
        cout << "Warning: could not find uniform uModel (shadow.vert)" << endl;
    }

    int shadowLightSpaceMatrixLocation = glGetUniformLocation(shadowShaderProgram, "uLightSpaceMatrix");
    if (shadowLightSpaceMatrixLocation == -1) {
        cout << "Warning: could not find uniform uLightSpaceMatrix (shadow.vert)" << endl;
    }

    int sceneColorLocation = glGetUniformLocation(screenShaderProgram, "uSceneColor");
    if (sceneColorLocation == -1) {
        cout << "Warning: could not find uniform uSceneColor" << endl;
    }

    int sceneDepthLocation = glGetUniformLocation(screenShaderProgram, "uSceneDepth");
    if (sceneDepthLocation == -1) {
        cout << "Warning: could not find uniform uSceneDepth" << endl;
    }

    int screenModeLocation = glGetUniformLocation(screenShaderProgram, "uScreenMode");
    if (screenModeLocation == -1) {
        cout << "Warning: could not find uniform uScreenMode" << endl;
    }

    int screenNearPlaneLocation = glGetUniformLocation(screenShaderProgram, "uNearPlane");
    if (screenNearPlaneLocation == -1) {
        cout << "Warning: could not find uniform uNearPlane" << endl;
    }

    int screenFarPlaneLocation = glGetUniformLocation(screenShaderProgram, "uFarPlane");
    if (screenFarPlaneLocation == -1) {
        cout << "Warning: could not find uniform uFarPlane" << endl;
    }

    int screenDepthVisualizationMaxLocation = glGetUniformLocation(screenShaderProgram, "uDepthVisualizationMax");
    if (screenDepthVisualizationMaxLocation == -1) {
        cout << "Warning: could not find uniform uDepthVisualizationMax" << endl;
    }

    int focusDistanceLocation = glGetUniformLocation(screenShaderProgram, "uFocusDistanceMeters");
    if (focusDistanceLocation == -1) {
        cout << "Warning: could not find uniform uFocusDistanceMeters" << endl;
    }

    int focalLengthLocation = glGetUniformLocation(screenShaderProgram, "uFocalLengthMillimeters");
    if (focalLengthLocation == -1) {
        cout << "Warning: could not find uniform uFocalLengthMillimeters" << endl;
    }

    int fNumberLocation = glGetUniformLocation(screenShaderProgram, "uFNumber");
    if (fNumberLocation == -1) {
        cout << "Warning: could not find uniform uFNumber" << endl;
    }

    int sensorHeightLocation = glGetUniformLocation(screenShaderProgram, "uSensorHeightMillimeters");
    if (sensorHeightLocation == -1) {
        cout << "Warning: could not find uniform uSensorHeightMillimeters" << endl;
    }

    int cocVisualizationMaxLocation = glGetUniformLocation(screenShaderProgram, "uCoCVisualizationMaxPixels");
    if (cocVisualizationMaxLocation == -1) {
        cout << "Warning: could not find uniform uCoCVisualizationMaxPixels" << endl;
    }

    int framebufferHeightLocation = glGetUniformLocation(screenShaderProgram, "uFramebufferHeightPixels");
    if (framebufferHeightLocation == -1) {
        cout << "Warning: could not find uniform uFramebufferHeightPixels" << endl;
    }

    int framebufferWidthLocation = glGetUniformLocation(screenShaderProgram, "uFramebufferWidthPixels");
    if (framebufferWidthLocation == -1) {
        cout << "Warning: could not find uniform uFramebufferWidthPixels" << endl;
    }

    int maxBlurRadiusLocation = glGetUniformLocation(screenShaderProgram, "uMaxBlurRadiusPixels");
    if (maxBlurRadiusLocation == -1) {
        cout << "Warning: could not find uniform uMaxBlurRadiusPixels" << endl;
    }

    int cocSampleCountLocation = glGetUniformLocation(screenShaderProgram, "uCoCSampleCount");
    if (cocSampleCountLocation == -1) {
        cout << "Warning: could not find uniform uCoCSampleCount" << endl;
    }

    glUseProgram(screenShaderProgram);
    // Sampler uniforms store texture-unit indices, not texture object IDs.
    if (sceneColorLocation != -1) {
        extraGl.uniform1i(sceneColorLocation, 0);
    }
    if (sceneDepthLocation != -1) {
        extraGl.uniform1i(sceneDepthLocation, 1);
    }

    // Vertex attributes vary per vertex; uniforms are shared for the whole draw call.
    // The cube uses 24 face vertices so each face can have one clear color and one flat normal.
    // Each vertex has nine floats: position.xyz, color.rgb, normal.xyz (object space;
    // uModel's inverse-transpose in basic.vert carries this into world space).
    // Positions are fixed model-space geometry; transformation happens with a matrix uniform.
    float vertices[] = {
        // Front face: red, normal +Z
        -0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 0.0f,   0.0f, 0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 0.0f,   0.0f, 0.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f, 0.0f,   0.0f, 0.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f, 0.0f,   0.0f, 0.0f, 1.0f,

        // Back face: green, normal -Z
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f, 0.0f,   0.0f, 0.0f, -1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f, 0.0f,   0.0f, 0.0f, -1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f, 0.0f,   0.0f, 0.0f, -1.0f,
         0.5f,  0.5f, -0.5f,  0.0f, 1.0f, 0.0f,   0.0f, 0.0f, -1.0f,

        // Left face: blue, normal -X
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f, 1.0f,  -1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f, 1.0f,  -1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 0.0f, 1.0f,  -1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 0.0f, 1.0f,  -1.0f, 0.0f, 0.0f,

        // Right face: yellow, normal +X
         0.5f, -0.5f,  0.5f,  1.0f, 1.0f, 0.0f,   1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 1.0f, 0.0f,   1.0f, 0.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 0.0f,   1.0f, 0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 0.0f,   1.0f, 0.0f, 0.0f,

        // Top face: cyan, normal +Y
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f, 1.0f,   0.0f, 1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f, 1.0f, 1.0f,   0.0f, 1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f, 1.0f, 1.0f,   0.0f, 1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f, 1.0f,   0.0f, 1.0f, 0.0f,

        // Bottom face: magenta, normal -Y
        -0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 1.0f,   0.0f, -1.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 1.0f,   0.0f, -1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 1.0f,   0.0f, -1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 1.0f,   0.0f, -1.0f, 0.0f
    };

    // The index array describes two triangles per cube face: 6 faces * 2 triangles * 3 indices = 36.
    unsigned int indices[] = {
         0,  1,  2,   2,  3,  0,
         4,  5,  6,   6,  7,  4,
         8,  9, 10,  10, 11,  8,
        12, 13, 14,  14, 15, 12,
        16, 17, 18,  18, 19, 16,
        20, 21, 22,  22, 23, 20
    };
    const GLsizei indexCount = static_cast<GLsizei>(sizeof(indices) / sizeof(indices[0]));

    //create Vertex Buffer Object, Vertex Array Object, and Element Buffer Object
    unsigned int VBO, VAO, EBO;
    glGenBuffers(1, &VBO);  
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &EBO);
    // glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Vertex Attributes
    // 0. copy our vertices array in a buffer for OpenGL to use
    // The VAO remembers this attribute layout. The VBO stores the vertex bytes.
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // The EBO stores index data. Its binding is remembered by the currently bound VAO.
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // 1. then set the vertex attributes pointers
    // Stride is the byte distance from one vertex to the next: 9 floats here (pos, color, normal).
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);  

    // Color starts after the first three floats because position takes x, y, z.
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Normal starts after position and color (location 2, matching Mesh.cpp's imported-mesh layout).
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // 2. use our shader program when we want to render an object
    // glUseProgram(shaderProgram);
    // 3. now draw the object.
    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glBindVertexArray(0); 

    Mesh importedMesh;
    if (hasImportedMesh && !uploadMesh(importedData, importedMesh, meshError)) {
        cout << "Failed to upload " << importedScenePath << ": " << meshError
             << "\nUsing fallback scene." << endl;
        hasImportedMesh = false;
    }
    importedData = {}; // GPU buffers retain the uploaded data for every subsequent frame.

    // The screen quad is already in clip/NDC space, so it needs no Model/View/Projection transforms.
    float screenQuadVertices[] = {
        // position.xy   uv
        -1.0f, -1.0f,   0.0f, 0.0f,
         1.0f, -1.0f,   1.0f, 0.0f,
         1.0f,  1.0f,   1.0f, 1.0f,

        -1.0f, -1.0f,   0.0f, 0.0f,
         1.0f,  1.0f,   1.0f, 1.0f,
        -1.0f,  1.0f,   0.0f, 1.0f
    };

    unsigned int screenVAO = 0;
    unsigned int screenVBO = 0;
    glGenVertexArrays(1, &screenVAO);
    glGenBuffers(1, &screenVBO);
    glBindVertexArray(screenVAO);
    glBindBuffer(GL_ARRAY_BUFFER, screenVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(screenQuadVertices), screenQuadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    unsigned int sceneFBO = 0;
    unsigned int sceneColorTexture = 0;
    unsigned int sceneDepthTexture = 0;
    extraGl.genFramebuffers(1, &sceneFBO);
    extraGl.genTextures(1, &sceneColorTexture);
    extraGl.genTextures(1, &sceneDepthTexture);

    int sceneFramebufferWidth = 0;
    int sceneFramebufferHeight = 0;

    auto resizeSceneFramebuffer = [&](int width, int height) {
        if (width <= 0 || height <= 0) {
            return false;
        }
        if (width == sceneFramebufferWidth && height == sceneFramebufferHeight) {
            return true;
        }

        // The custom FBO is an alternate render target; its attachments receive the scene output.
        extraGl.bindFramebuffer(GL_FRAMEBUFFER, sceneFBO);

        // Scene color is rendered into this texture instead of directly into the window.
        extraGl.bindTexture(GL_TEXTURE_2D, sceneColorTexture);
        extraGl.texImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        extraGl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        extraGl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        extraGl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        extraGl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        extraGl.framebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sceneColorTexture, 0);

        // The depth texture stores scene depth now and will be sampled by later DoF passes.
        extraGl.bindTexture(GL_TEXTURE_2D, sceneDepthTexture);
        extraGl.texImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0,
                           GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
        extraGl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        extraGl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        extraGl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        extraGl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        extraGl.framebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, sceneDepthTexture, 0);

        GLenum framebufferStatus = extraGl.checkFramebufferStatus(GL_FRAMEBUFFER);
        extraGl.bindFramebuffer(GL_FRAMEBUFFER, 0); // FBO 0 is the default/window framebuffer.
        extraGl.bindTexture(GL_TEXTURE_2D, 0);

        if (framebufferStatus != GL_FRAMEBUFFER_COMPLETE) {
            cout << "Framebuffer is incomplete. Status: 0x" << hex << framebufferStatus << dec << endl;
            return false;
        }

        cout << "Scene framebuffer complete: " << width << " x " << height << endl;
        sceneFramebufferWidth = width;
        sceneFramebufferHeight = height;
        return true;
    };

    // ==============================
    // EXPERIMENT 17: shadow map framebuffer
    // ==============================
    // Depth-only, fixed resolution (unlike the scene FBO, this never tracks
    // the window size): the light's view of the scene doesn't change with
    // the window, only with the scene and light direction.
    unsigned int shadowFBO = 0;
    unsigned int shadowDepthTexture = 0;
    extraGl.genFramebuffers(1, &shadowFBO);
    extraGl.genTextures(1, &shadowDepthTexture);

    extraGl.bindTexture(GL_TEXTURE_2D, shadowDepthTexture);
    extraGl.texImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, shadowMapSize, shadowMapSize, 0,
                       GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
    extraGl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    extraGl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    extraGl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    extraGl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    extraGl.bindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
    extraGl.framebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowDepthTexture, 0);
    // No color attachment exists on this FBO; without these two calls some
    // drivers report GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER / READ_BUFFER.
    extraGl.drawBuffer(GL_NONE);
    extraGl.readBuffer(GL_NONE);
    GLenum shadowFramebufferStatus = extraGl.checkFramebufferStatus(GL_FRAMEBUFFER);
    extraGl.bindFramebuffer(GL_FRAMEBUFFER, 0);
    extraGl.bindTexture(GL_TEXTURE_2D, 0);
    if (shadowFramebufferStatus != GL_FRAMEBUFFER_COMPLETE) {
        cout << "Shadow framebuffer is incomplete. Status: 0x" << hex << shadowFramebufferStatus << dec << endl;
        glDeleteProgram(shaderProgram);
        glDeleteProgram(screenShaderProgram);
        glDeleteProgram(shadowShaderProgram);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }
    cout << "Shadow framebuffer complete: " << shadowMapSize << " x " << shadowMapSize << endl;

    // The scene (boxes + imported mesh) doesn't move once running, only the
    // camera does, so one orthographic light-space matrix covers every frame.
    // If objects ever animate, recompute this inside the render loop instead.
    auto computeLightSpaceMatrix = [&]() {
        glm::vec3 lightDir = glm::normalize(lightDirection);
        // The scene sits roughly around z = -8; center the light's look-at
        // there instead of at the origin so the frustum isn't wasted on empty space.
        glm::vec3 sceneCenter = glm::vec3(0.0f, 0.0f, -8.0f);
        glm::vec3 lightEye = sceneCenter + lightDir * shadowFrustumHalfExtent;
        glm::vec3 upHint = (std::abs(lightDir.y) > 0.99f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
        glm::mat4 lightView = glm::lookAt(lightEye, sceneCenter, upHint);
        glm::mat4 lightProjection = glm::ortho(-shadowFrustumHalfExtent, shadowFrustumHalfExtent,
                                               -shadowFrustumHalfExtent, shadowFrustumHalfExtent,
                                               shadowNearPlane, shadowFarPlane);
        return lightProjection * lightView;
    };
    glm::mat4 lightSpaceMatrix = computeLightSpaceMatrix();

    if (wireframeMode) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    glEnable(GL_DEPTH_TEST);
    // GL_LESS keeps the nearest fragment for each framebuffer location.
    glDepthFunc(GL_LESS);

    bool screenshotKeyWasPressed = false;

    // ==============================
    // EXPERIMENT 17: Dear ImGui control panel
    // ==============================
    // Callbacks were installed on the window above, so install_callbacks = true
    // makes ImGui chain to them instead of replacing them.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    printVerificationHelp();
    showVerificationStatus(window);

    while (!glfwWindowShouldClose(window)) {
        float currentFrameTime = static_cast<float>(glfwGetTime());
        deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;

        //input
        processInput(window);

        float intensity = animateIntensity
            ? 0.6f + 0.4f * sin(currentFrameTime * intensitySpeed)
            : staticIntensity;

        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
        if (framebufferWidth <= 0 || framebufferHeight <= 0) {
            glfwSwapBuffers(window);
            glfwPollEvents();
            continue;
        }

        const float aspectRatio = framebufferHeight > 0
            ? static_cast<float>(framebufferWidth) / static_cast<float>(framebufferHeight)
            : 1.0f;

        // View transforms world-space geometry into the camera's coordinate system.
        // lookAt uses camera position, the point it looks toward, and a reference up direction.
        glm::mat4 view = glm::lookAt(cameraPosition, cameraPosition + cameraFront, worldUp);

        if (printCameraState && currentFrameTime - lastCameraPrintTime > 1.0f) {
            cout << "Camera position: "
                 << cameraPosition.x << ", " << cameraPosition.y << ", " << cameraPosition.z
                 << " | front: "
                 << cameraFront.x << ", " << cameraFront.y << ", " << cameraFront.z
                 << " | yaw: " << yawDegrees
                 << " | pitch: " << pitchDegrees << endl;
            lastCameraPrintTime = currentFrameTime;
        }

        // Projection transforms camera/view space into clip space. Perspective creates w for the later perspective divide.
        const float verticalFovRadians = usePhysicalCameraProjection
            ? physicalVerticalFovRadians(focalLengthMillimeters, sensorHeightMillimeters)
            : glm::radians(fieldOfViewDegrees);
        glm::mat4 projection = usePerspectiveProjection
            ? glm::perspective(verticalFovRadians, aspectRatio, nearPlane, farPlane)
            : glm::mat4(1.0f);

        // ---- Control panel (Experiment 17) ----
        // Immediate-mode UI: this code runs every frame and edits the same globals the hotkeys edit,
        // so keyboard and panel always agree. Press Tab to release the cursor before clicking it.
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        if (showControlPanel) {
            ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(360.0f, 560.0f), ImGuiCond_FirstUseEver);
            ImGui::Begin("DoF Controls (G hides, Tab frees cursor)", &showControlPanel);

            const char* screenModeNames[] = {
                "Color", "Raw depth", "Linear depth", "CoC magnitude", "CoC signed", "Basic DoF"
            };
            int screenModeIndex = static_cast<int>(screenMode);
            if (ImGui::Combo("View", &screenModeIndex, screenModeNames, IM_ARRAYSIZE(screenModeNames))) {
                screenMode = static_cast<ScreenMode>(screenModeIndex);
            }

            if (ImGui::CollapsingHeader("Camera / lens", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Focus distance (m)", &focusDistanceMeters, 0.5f, 40.0f);
                ImGui::SliderFloat("f-number", &fNumber, 1.0f, 22.0f);
                ImGui::SliderFloat("Focal length (mm)", &focalLengthMillimeters, 18.0f, 200.0f);
                ImGui::SliderFloat("Sensor height (mm)", &sensorHeightMillimeters, 8.0f, 36.0f);
                ImGui::Checkbox("Physical camera FOV", &usePhysicalCameraProjection);
            }

            if (ImGui::CollapsingHeader("Blur gather", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Max blur radius (px)", &maxBlurRadiusPixels, 0.0f, 200.0f);
                ImGui::SliderInt("CoC samples", &cocSampleCount, 4, 256);
                ImGui::SliderFloat("CoC debug max (px)", &cocVisualizationMaxPixels, 1.0f, 200.0f);
            }

            if (ImGui::CollapsingHeader("Lighting / shadows", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Shadows", &enableShadows);
                ImGui::SliderFloat("Ambient", &ambientStrength, 0.0f, 1.0f);
                ImGui::SliderFloat3("Light dir (to light)", glm::value_ptr(lightDirection), -1.0f, 1.0f);
                if (glm::length(lightDirection) < 0.05f) {
                    lightDirection = glm::vec3(-0.4f, 0.8f, 0.6f); // Avoid normalizing a zero vector.
                }
                ImGui::ColorEdit3("Light color", glm::value_ptr(lightColor));
                ImGui::Checkbox("Show floor and wall", &showEnvironment);
            }

            ImGui::Separator();
            ImGui::Text("%.1f FPS (%.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
            ImGui::Text("Blur cost: %d taps x %d x %d px", cocSampleCount, framebufferWidth, framebufferHeight);
            ImGui::End();
        }

        // Recomputed every frame so the light-direction slider in the control panel takes effect immediately.
        lightSpaceMatrix = computeLightSpaceMatrix();

        // shadowPass = true renders depth only from the light's point of view (shadow.vert/frag);
        // false is the normal lit camera pass. Both walk the exact same geometry list below.
        auto renderScene = [&](bool shadowPass) {
            const int activeModelLocation = shadowPass ? shadowModelLocation : modelLocation;
            if (shadowPass) {
                glUseProgram(shadowShaderProgram);
                if (shadowLightSpaceMatrixLocation != -1) {
                    glUniformMatrix4fv(shadowLightSpaceMatrixLocation, 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
                }
            } else {
                glUseProgram(shaderProgram);
                // Uniforms are shared values for this draw call and are uploaded to the active shader program.
                if (intensityLocation != -1) {
                    glUniform1f(intensityLocation, intensity);
                }
                if (viewLocation != -1) {
                    glUniformMatrix4fv(viewLocation, 1, GL_FALSE, glm::value_ptr(view));
                }
                if (projectionLocation != -1) {
                    glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, glm::value_ptr(projection));
                }
                if (lightSpaceMatrixLocation != -1) {
                    glUniformMatrix4fv(lightSpaceMatrixLocation, 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
                }
                if (lightDirectionLocation != -1) {
                    extraGl.uniform3fv(lightDirectionLocation, 1, glm::value_ptr(lightDirection));
                }
                if (lightColorLocation != -1) {
                    extraGl.uniform3fv(lightColorLocation, 1, glm::value_ptr(lightColor));
                }
                if (ambientStrengthLocation != -1) {
                    glUniform1f(ambientStrengthLocation, ambientStrength);
                }
                if (useShadowsLocation != -1) {
                    extraGl.uniform1i(useShadowsLocation, enableShadows ? 1 : 0);
                }
                // Unit 0/1 belong to the screen pass (color/depth); the shadow map takes unit 2.
                extraGl.activeTexture(GL_TEXTURE2);
                extraGl.bindTexture(GL_TEXTURE_2D, shadowDepthTexture);
                if (shadowMapLocation != -1) {
                    extraGl.uniform1i(shadowMapLocation, 2);
                }
                extraGl.activeTexture(GL_TEXTURE0);
            }
            glBindVertexArray(VAO);
            if (!shadowPass) {
                extraGl.uniform1i(importedMeshLocation, 0); // Preserve the original cube vertex colors.
            }

            auto drawBox = [&](glm::vec3 position, float rotationX, float rotationY, glm::vec3 scale) {
                // Model transforms this cube from local/object space into world space.
                glm::mat4 model(1.0f);
                model = glm::translate(model, position);
                model = glm::rotate(model, glm::radians(rotationY), glm::vec3(0.0f, 1.0f, 0.0f));
                model = glm::rotate(model, glm::radians(rotationX), glm::vec3(1.0f, 0.0f, 0.0f));
                model = glm::scale(model, scale);

                if (activeModelLocation != -1) {
                    glUniformMatrix4fv(activeModelLocation, 1, GL_FALSE, glm::value_ptr(model));
                }
                glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, nullptr);
            };
            auto drawCube = [&](glm::vec3 position, float rotationX, float rotationY, float scale) {
                drawBox(position, rotationX, rotationY, glm::vec3(scale));
            };

            // P * V * M is uploaded as three uniforms; perspective divide happens after the vertex shader and produces NDC.
            drawCube(cubeAPosition, cubeARotationXDegrees, cubeARotationYDegrees, cubeAUniformScale);
            if (!hasImportedMesh) {
                drawCube(cubeBPosition, cubeBRotationXDegrees, cubeBRotationYDegrees, cubeBUniformScale);
                drawCube(cubeCPosition, cubeCRotationXDegrees, cubeCRotationYDegrees, cubeCUniformScale);
            }
            drawCube(cubeDPosition, cubeDRotationXDegrees, cubeDRotationYDegrees, cubeDUniformScale);
            drawCube(cubeEPosition, cubeERotationXDegrees, cubeERotationYDegrees, cubeEUniformScale);
            // Experiment 17 extra depth layers (mirrored in render_dof.py BOXES).
            drawCube(cubeFPosition, cubeFRotationXDegrees, cubeFRotationYDegrees, cubeFUniformScale);
            drawCube(cubeGPosition, cubeGRotationXDegrees, cubeGRotationYDegrees, cubeGUniformScale);
            if (showEnvironment) {
                drawBox(glm::vec3(0.0f, -0.85f, -7.5f), 0.0f, 0.0f, glm::vec3(12.0f, 0.1f, 25.0f));
                drawBox(glm::vec3(0.0f, 1.2f, -22.0f), 0.0f, 0.0f, glm::vec3(14.0f, 4.0f, 0.1f));
            }

            if (hasImportedMesh) {
                glm::mat4 model = glm::translate(glm::mat4(1.0f), importedScenePosition);
                model = glm::rotate(
                    model,
                    glm::radians(-90.0f),
                    glm::vec3(1.0f, 0.0f, 0.0f)
                );
                model = glm::rotate(model, glm::radians(importedSceneRotationYDegrees), glm::vec3(0.0f, 1.0f, 0.0f));
                model = glm::scale(model, glm::vec3(importedSceneScale));
                glUniformMatrix4fv(activeModelLocation, 1, GL_FALSE, glm::value_ptr(model));
                if (!shadowPass) {
                    extraGl.uniform1i(importedMeshLocation, 1);
                }
                // Same scene pass, depth test and FBO attachments as the cubes. No special depth path.
                drawMesh(importedMesh);
            }

            glBindVertexArray(0);
        };

        // Pass 0: render depth from the light into the shadow map. Front faces are kept
        // (no culling tricks); the slope-scaled bias in basic.frag handles self-shadow acne.
        if (enableShadows) {
            extraGl.bindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
            glViewport(0, 0, shadowMapSize, shadowMapSize);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glClear(GL_DEPTH_BUFFER_BIT);
            renderScene(true);
            extraGl.bindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        if (renderThroughFramebuffer) {
            if (!resizeSceneFramebuffer(framebufferWidth, framebufferHeight)) {
                glfwSwapBuffers(window);
                glfwPollEvents();
                continue;
            }

            // Pass 1: render the 3D scene into the off-screen FBO's color and depth textures.
            extraGl.bindFramebuffer(GL_FRAMEBUFFER, sceneFBO);
            glViewport(0, 0, sceneFramebufferWidth, sceneFramebufferHeight);
            glEnable(GL_DEPTH_TEST); // Scene geometry still needs depth testing for visibility.
            glDepthFunc(GL_LESS);
            glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            renderScene(false);

            // Pass 2: present color, depth/CoC diagnostics, or BasicDoF through a screen quad.
            extraGl.bindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, framebufferWidth, framebufferHeight);
            glDisable(GL_DEPTH_TEST); // The quad covers the screen and only presents an already-rendered image.
            glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(screenShaderProgram);
            if (sceneColorLocation != -1) {
                extraGl.uniform1i(sceneColorLocation, 0);
            }
            if (sceneDepthLocation != -1) {
                extraGl.uniform1i(sceneDepthLocation, 1);
            }
            if (screenModeLocation != -1) {
                extraGl.uniform1i(screenModeLocation, static_cast<int>(screenMode));
            }
            // Screen-pass depth reconstruction must use the same near/far values as the projection.
            if (screenNearPlaneLocation != -1) {
                glUniform1f(screenNearPlaneLocation, nearPlane);
            }
            if (screenFarPlaneLocation != -1) {
                glUniform1f(screenFarPlaneLocation, farPlane);
            }
            if (screenDepthVisualizationMaxLocation != -1) {
                glUniform1f(screenDepthVisualizationMaxLocation, depthVisualizationMax);
            }
            // Linear depth is interpreted as meters; BasicDoF uses absolute CoC as its blur radius.
            if (focusDistanceLocation != -1) {
                glUniform1f(focusDistanceLocation, focusDistanceMeters);
            }
            if (focalLengthLocation != -1) {
                glUniform1f(focalLengthLocation, focalLengthMillimeters);
            }
            if (fNumberLocation != -1) {
                glUniform1f(fNumberLocation, fNumber);
            }
            if (sensorHeightLocation != -1) {
                glUniform1f(sensorHeightLocation, sensorHeightMillimeters);
            }
            if (cocVisualizationMaxLocation != -1) {
                glUniform1f(cocVisualizationMaxLocation, cocVisualizationMaxPixels);
            }
            if (framebufferHeightLocation != -1) {
                glUniform1f(framebufferHeightLocation, static_cast<float>(framebufferHeight));
            }
            if (framebufferWidthLocation != -1) {
                glUniform1f(framebufferWidthLocation, static_cast<float>(framebufferWidth));
            }
            if (maxBlurRadiusLocation != -1) {
                glUniform1f(maxBlurRadiusLocation, maxBlurRadiusPixels);
            }
            if (cocSampleCountLocation != -1) {
                extraGl.uniform1i(cocSampleCountLocation, cocSampleCount);
            }
            // A texture object is bound to a texture unit; the sampler chooses which unit to read.
            extraGl.activeTexture(GL_TEXTURE0);
            extraGl.bindTexture(GL_TEXTURE_2D, sceneColorTexture);
            extraGl.activeTexture(GL_TEXTURE1);
            extraGl.bindTexture(GL_TEXTURE_2D, sceneDepthTexture);
            glBindVertexArray(screenVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);
        } else {
            extraGl.bindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, framebufferWidth, framebufferHeight);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            // Clearing the color buffer does not clear stored depth values; reset both buffers every frame.
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            renderScene(false);
        }

        const bool screenshotKeyIsPressed = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
        if (screenshotKeyIsPressed && !screenshotKeyWasPressed) {
            glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
            saveScreenshot("output/latest.png", framebufferWidth, framebufferHeight);
        }
        screenshotKeyWasPressed = screenshotKeyIsPressed;

        // Drawn after the screenshot read so saved PNGs never contain the control panel.
        extraGl.bindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, framebufferWidth, framebufferHeight);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        //check and call events and swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents(); //check updates

    }

    //Delete everything
    destroyMesh(importedMesh);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &screenVAO);
    glDeleteBuffers(1, &screenVBO);
    extraGl.deleteFramebuffers(1, &sceneFBO);
    extraGl.deleteTextures(1, &sceneColorTexture);
    extraGl.deleteTextures(1, &sceneDepthTexture);
    extraGl.deleteFramebuffers(1, &shadowFBO);
    extraGl.deleteTextures(1, &shadowDepthTexture);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glDeleteProgram(shaderProgram);
    glDeleteProgram(screenShaderProgram);
    glDeleteProgram(shadowShaderProgram);

    glfwTerminate();
    return 0;
}
