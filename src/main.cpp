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
#include "SceneFile.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
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
// Half-float colour attachment. The scene pass writes linear radiance, which an
// RGB8 target would both clip at 1.0 and quantize before the defocus gather
// ever runs; 16 bits per channel is enough headroom for the lit windows and
// bulbs without the bandwidth of full floats.
#ifndef GL_RGBA16F
#define GL_RGBA16F 0x881A
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
    BasicDoF = 5,
    // Sharp on the left of a draggable divider, defocused on the right, in one
    // frame. Flipping between two screenshots hides exactly the thing worth
    // looking at, which is how much the blur changes at a given depth.
    SplitSharpDoF = 6
};
const int screenModeCount = 7;

ScreenMode screenMode = ScreenMode::BasicDoF;

// Physical convention: 1 world unit = 1 meter.
bool usePhysicalCameraProjection = true;
float focusDistanceMeters = 5.0f;
float focalLengthMillimeters = 50.0f;
// f/1.4 wide open by default, matching the `camera` line in scene/alley.scene
// and the first Cycles reference job. The scene file overrides these at startup;
// the literals stay here so the renderer still runs if the file is missing.
float fNumber = 1.4f;
float sensorHeightMillimeters = 24.0f;
// Debug visualization scaling only; does not change the rendered blur. In CoC
// DIAMETER pixels, the same unit the panel reports.
float cocVisualizationMaxPixels = 20.0f;
// Ceiling on the gather RADIUS used by BasicDoF. Note the unit change from the
// previous version: screen.frag used to pass the CoC diameter straight in as a
// radius, which blurred the raster image twice as hard as the Cycles reference
// at the same f-number. See notes/graphics/18_alley_scene_and_ui.md.
float maxBlurRadiusPixels = 120.0f;
// Linear exposure applied when the HDR scene target is encoded for display.
// 1.0 means "show the radiance as rendered", which is what makes the image
// comparable to Cycles with Blender's Standard view transform.
float exposure = 1.0f;
// Divider position for ScreenMode::SplitSharpDoF, as a fraction of frame width.
float splitFraction = 0.5f;
// Aperture samples per pixel for the BasicDoF gather (shaders/screen.frag).
// At a 120 px radius, 16 taps band visibly; 100 is the meeting's target.
int cocSampleCount = 100;

// ==============================
// EXPERIMENT 17 CONTROLS: shading, shadows, live UI
// ==============================
// ==============================
// EXPERIMENT 18: the whole environment comes from one shared scene file
// ==============================
// scene/alley.scene is the single description both renderers read: this
// executable through src/SceneFile.cpp, and the Cycles ground truth through
// tools/scene/scene_loader.py. Neither one authors geometry any more, so the
// two cannot drift apart. Regenerate the file with tools/scene/build_alley.py.
const string scenePath = "scene/alley.scene";

// Points FROM a lit surface TOWARD the light. Overwritten by the scene file's
// `sun` line at startup; the literals match its current contents so the
// fallback scene is lit the same way.
glm::vec3 lightDirection = glm::vec3(-0.45f, 0.78f, 0.44f);
glm::vec3 lightColor = glm::vec3(1.0f, 0.88f, 0.72f);
// Sun irradiance in W/m^2, the same number Blender's sun strength takes.
// basic.frag divides by pi to match a Lambert diffuse BSDF, so this is a
// physical quantity rather than a tuning knob.
float lightEnergy = 4.2f;
// Uniform sky radiance: also the clear colour and the Cycles world background.
glm::vec3 ambientColor = glm::vec3(0.42f, 0.52f, 0.72f);
float ambientStrength = 0.24f;
bool enableShadows = true;
// 4096 rather than 2048: the alley is full of shallow relief (mortar courses,
// railings, fire-escape slats) whose shadows disappear into the texel grid at
// the lower resolution.
const int shadowMapSize = 4096;
// The scene file's `shadow lo/hi` region drives the light frustum. Fitting it
// to that box instead of the geometry bounds matters here, because the geometry
// also contains the distant facade at z = -47 and the frustum would spend most
// of its texels on scenery the camera never sees up close.
const float shadowFrustumFallbackHalfExtent = 20.0f;
const float shadowNearPlane = 0.1f;
const float shadowFarPlane = 120.0f;
// World-space size of one shadow-map texel, passed to basic.vert for the
// normal-offset shadow lookup. Recomputed whenever the frustum is refitted.
float shadowWorldTexelSize = 0.01f;

// The panel has three sizes rather than a visibility toggle: Compact is what
// the parameter sweeps actually need, Full exposes everything, Hidden gets it
// out of the way for screenshots. G cycles them.
enum class PanelMode {
    Hidden = 0,
    Compact = 1,
    Full = 2
};
PanelMode panelMode = PanelMode::Compact;

// Set by the panel's mouse-look button or by Tab, serviced in the render loop
// because only there is the GLFWwindow handle in scope.
bool cursorToggleRequested = false;

// Set by the L key, serviced by the render loop, which owns the GPU buffers.
// Editing a line in scene/alley.scene and pressing L is much faster than a
// rebuild when placing props or retuning the light.
bool sceneReloadRequested = false;

// OBJ units are arbitrary. Choose meters per authored unit explicitly; never auto-normalize.
const string importedScenePath = "assets/models/scene.obj";
float importedSceneScale = 0.1f;
glm::vec3 importedScenePosition = glm::vec3(0.0f, -0.75f, 0.0f);
float importedSceneRotationYDegrees = 0.0f;
// The imported mesh has no per-vertex albedo (Mesh.cpp supplies position,
// normal and UV only), so basic.frag takes this flat linear albedo for it.
glm::vec3 importedSceneAlbedo = glm::vec3(0.62f, 0.44f, 0.20f);
// Both halves of the scene can be switched off independently, which is the
// quickest way to tell whether a blur artifact comes from the alley geometry
// or from the subject at the focus plane.
bool showSceneGeometry = true;
bool showImportedMesh = true;

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

// The seven hand-placed cubes that used to live here are gone. They were
// duplicated in three places (these globals, render_dof.py's BOXES tuple, and
// the regex checks in tests/test_reference_config.py), and every change had to
// be made three times or the two renderers silently diverged. The alley is
// authored once in tools/scene/build_alley.py and loaded from
// scene/alley.scene by both, so there is nothing left to keep in sync by hand.

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

// R restores the configured starting view for repeatable focus/aperture
// comparisons. Not const: the scene file's `camera` line overwrites these at
// startup, so R returns to the pose the file defines rather than to whatever
// happened to be compiled in.
glm::vec3 initialCameraPosition = cameraPosition;
float initialYawDegrees = yawDegrees;
float initialPitchDegrees = pitchDegrees;

float movementSpeed = 2.5f;
float mouseSensitivity = 0.1f;

// The cursor starts free so the panel is clickable straight away, and the
// camera turns while the RIGHT mouse button is held. The old behaviour (cursor
// captured on launch, Tab to release it before touching the UI) meant every
// parameter change cost two extra keystrokes. Tab still toggles a sticky
// captured-cursor mode for long navigation.
bool rightMouseLook = false;

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
        case ScreenMode::SplitSharpDoF: return "Split sharp|DoF";
    }
    return "Unknown";
}

// Short filename tag per view, so a depth or CoC capture is not mistaken for a
// colour one. BasicDoF gets no tag: it is the mode that gets compared.
const char* screenModeFileTag(ScreenMode mode) {
    switch (mode) {
        case ScreenMode::Color: return "sharp";
        case ScreenMode::RawDepth: return "rawdepth";
        case ScreenMode::LinearDepth: return "depth";
        case ScreenMode::CoCMagnitude: return "coc";
        case ScreenMode::CoCSigned: return "cocsigned";
        case ScreenMode::BasicDoF: return "dof";
        case ScreenMode::SplitSharpDoF: return "split";
    }
    return "view";
}

// "%g"-style: 5 not 5.000000, 1.4 not 1.400000. Matches how render_dof.py's
// f-strings format the same numbers into its output names.
string formatCompact(float value) {
    std::ostringstream stream;
    stream << std::defaultfloat << static_cast<double>(value);
    return stream.str();
}

// Signed circle-of-confusion DIAMETER in pixels at a given depth, the same
// formula shaders/screen.frag evaluates per fragment. Duplicated on the CPU so
// the panel can report the numbers that actually come out of the current lens
// settings, instead of leaving the user to guess why a 120 px blur ceiling is
// never reached at f/1.4 focused 5 m away.
float signedCoCDiameterPixels(float depthMeters, int framebufferHeightPixels) {
    const float focalLengthMeters = focalLengthMillimeters * 0.001f;
    const float sensorHeightMeters = sensorHeightMillimeters * 0.001f;
    if (depthMeters <= 0.0f || focusDistanceMeters <= focalLengthMeters || fNumber <= 0.0f ||
        sensorHeightMeters <= 0.0f || framebufferHeightPixels <= 0) {
        return 0.0f;
    }
    const float apertureDiameter = focalLengthMeters / fNumber;
    const float denominator = depthMeters * (focusDistanceMeters - focalLengthMeters);
    if (std::abs(denominator) < 1e-6f) {
        return 0.0f;
    }
    const float cocSensorMeters =
        (apertureDiameter * focalLengthMeters * (depthMeters - focusDistanceMeters)) / denominator;
    return (cocSensorMeters / sensorHeightMeters) * static_cast<float>(framebufferHeightPixels);
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
    cout << "Experiment 18 verification keys (no rebuild needed):\n"
         << "  1 Color | 2 RawDepth | 3 LinearDepth | 4 CoCMagnitude | 5 CoCSigned | 6 BasicDoF\n"
         << "  0 split sharp|DoF (drag the divider in the panel)\n"
         << "  7 focus 2 m | 8 focus 5 m | 9 focus 15 m | F toggle f/1.4 and f/8 | B f/2.8\n"
         << "  K strong-DoF preset (85 mm f/1.4 focused 1.6 m: an obvious, large blur)\n"
         << "  T reference preset (50 mm f/1.4 focused 5 m, matches the Cycles jobs)\n"
         << "  L reload scene/alley.scene | R reset camera | V physical/legacy projection\n"
         << "  [ / ] focal length -/+5 mm | , / . sensor height -/+2 mm\n"
         << "  WASD move | Tab or the panel button frees/captures the mouse\n"
         << "  (holding the right mouse button is a quick look without toggling)\n"
         << "  G cycle panel: compact -> full -> hidden | H help | Escape exit\n"
         << "  P screenshot: output/latest.png and a settings-named copy for comparisons\n";
}

void updateCameraFront();

// Copies the scene file's camera, lens and lighting into the globals the
// renderer and the panel already read, so the file is the source of truth for
// both renderers and the hotkeys/sliders keep working unchanged on top of it.
void applySceneSettings(const SceneDescription& scene) {
    cameraPosition = glm::vec3(scene.camera.position[0], scene.camera.position[1],
                               scene.camera.position[2]);
    yawDegrees = scene.camera.yawDegrees;
    pitchDegrees = scene.camera.pitchDegrees;
    initialCameraPosition = cameraPosition;
    initialYawDegrees = yawDegrees;
    initialPitchDegrees = pitchDegrees;
    updateCameraFront();

    focusDistanceMeters = scene.camera.focusDistanceMeters;
    fNumber = scene.camera.fNumber;
    focalLengthMillimeters = scene.camera.focalLengthMillimeters;
    sensorHeightMillimeters = scene.camera.sensorHeightMillimeters;

    lightDirection = glm::vec3(scene.sun.direction[0], scene.sun.direction[1], scene.sun.direction[2]);
    lightColor = glm::vec3(scene.sun.color[0], scene.sun.color[1], scene.sun.color[2]);
    lightEnergy = scene.sun.energy;
    ambientColor = glm::vec3(scene.ambient.color[0], scene.ambient.color[1], scene.ambient.color[2]);
    ambientStrength = scene.ambient.strength;
}

// Uniform sky radiance. This one expression is the OpenGL clear colour, the
// ambient term in basic.frag, and the Cycles world background strength, which
// is why the background of both images ends up the same colour.
glm::vec3 skyRadiance() {
    return ambientColor * ambientStrength;
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

// True while the camera should follow mouse motion: either Tab captured the
// cursor, or the right button is being held for a quick look-around.
static bool cameraFollowsMouse(GLFWwindow* window) {
    if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) return true;
    return rightMouseLook;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (imguiWantsMouse() && !rightMouseLook) {
        // The control panel wants this mouse motion (e.g. dragging a slider); don't also turn the camera.
        firstMouse = true;
        return;
    }
    if (!cameraFollowsMouse(window)) {
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

void mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/) {
    if (button != GLFW_MOUSE_BUTTON_RIGHT) return;
    if (action == GLFW_PRESS) {
        // Don't start a look if the press landed on the panel.
        if (imguiWantsMouse()) return;
        rightMouseLook = true;
        firstMouse = true;
        // Hide the cursor for the duration so it doesn't drift to a screen edge
        // and stop reporting motion mid-turn.
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    } else if (action == GLFW_RELEASE && rightMouseLook) {
        rightMouseLook = false;
        firstMouse = true;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // Ignore repeats: holding F or Tab must not toggle repeatedly.
    if (action != GLFW_PRESS) return;

    // Let the control panel consume keys while it has focus, except Tab (release/capture
    // the cursor, which is how the panel becomes reachable) and G (panel visibility itself).
    if (imguiWantsKeyboard() && key != GLFW_KEY_TAB && key != GLFW_KEY_G) return;

    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_6) {
        screenMode = static_cast<ScreenMode>(key - GLFW_KEY_1);
    } else if (key == GLFW_KEY_0) {
        screenMode = ScreenMode::SplitSharpDoF;
    } else if (key == GLFW_KEY_K) {
        // An unmistakable depth of field, for when the question is "is the blur
        // working" rather than "does it match Cycles". A long lens focused close
        // is what actually produces a large circle of confusion: at 50 mm f/1.4
        // focused 5 m away the background CoC radius tops out around 9 px, so
        // the 120 px ceiling never comes into play.
        focalLengthMillimeters = 85.0f;
        fNumber = 1.4f;
        focusDistanceMeters = 1.6f;
        usePhysicalCameraProjection = true;
        usePerspectiveProjection = true;
        screenMode = ScreenMode::BasicDoF;
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
        exposure = 1.0f;
        firstMouse = true;
    } else if (key == GLFW_KEY_L) {
        // Handled in the render loop, which owns the GPU buffers.
        sceneReloadRequested = true;
    } else if (key == GLFW_KEY_R) {
        cameraPosition = initialCameraPosition;
        yawDegrees = initialYawDegrees;
        pitchDegrees = initialPitchDegrees;
        updateCameraFront();
        firstMouse = true;
    } else if (key == GLFW_KEY_TAB) {
        cursorToggleRequested = true;
    } else if (key == GLFW_KEY_H) {
        printVerificationHelp();
    } else if (key == GLFW_KEY_G) {
        // Compact -> Full -> Hidden -> Compact.
        panelMode = static_cast<PanelMode>((static_cast<int>(panelMode) + 1) % 3);
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

    // WASD works whenever the panel isn't typing, cursor captured or not. The
    // old code required a captured cursor, which meant the camera could not be
    // nudged without first taking the mouse away from the sliders.

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
    // The shared scene file is read first because it supplies the camera and
    // lens the FOV report below prints. Everything about the environment comes
    // from here; if it fails to load the renderer still starts, with the
    // literals at the top of this file and the imported mesh alone.
    const filesystem::path sceneFilePath = filesystem::path(PROJECT_SOURCE_DIR) / scenePath;
    SceneDescription sceneDescription;
    string sceneError;
    bool hasSceneGeometry = loadSceneFile(sceneFilePath, sceneDescription, sceneError);
    if (!hasSceneGeometry) {
        cout << "Failed to load " << scenePath << ": " << sceneError
             << "\nFalling back to the imported mesh and the built-in camera settings." << endl;
    } else {
        applySceneSettings(sceneDescription);
        cout << "Loaded " << scenePath << ": " << sceneDescription.primitiveCount
             << " primitives, " << sceneDescription.vertices.size() << " vertices, "
             << sceneDescription.triangleCount() << " triangles ("
             << sceneDescription.emissiveTriangleCount << " emissive)" << endl;
        cout << "Scene bounds: (" << sceneDescription.boundsMin.x << ", " << sceneDescription.boundsMin.y
             << ", " << sceneDescription.boundsMin.z << ") to (" << sceneDescription.boundsMax.x << ", "
             << sceneDescription.boundsMax.y << ", " << sceneDescription.boundsMax.z << ")" << endl;
        cout << "Sun: energy " << lightEnergy << " W/m^2 | sky radiance "
             << skyRadiance().x << ", " << skyRadiance().y << ", " << skyRadiance().z << endl;
    }

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
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    // Cursor free on launch: the panel is the primary way to drive the
    // parameter sweeps, so it should be clickable without pressing Tab first.
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

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

    int lightEnergyLocation = glGetUniformLocation(shaderProgram, "uLightEnergy");
    if (lightEnergyLocation == -1) {
        cout << "Warning: could not find uniform uLightEnergy" << endl;
    }

    int skyRadianceLocation = glGetUniformLocation(shaderProgram, "uSkyRadiance");
    if (skyRadianceLocation == -1) {
        cout << "Warning: could not find uniform uSkyRadiance" << endl;
    }

    int overrideAlbedoLocation = glGetUniformLocation(shaderProgram, "uOverrideAlbedo");
    if (overrideAlbedoLocation == -1) {
        cout << "Warning: could not find uniform uOverrideAlbedo" << endl;
    }

    int shadowWorldTexelSizeLocation = glGetUniformLocation(shaderProgram, "uShadowWorldTexelSize");
    if (shadowWorldTexelSizeLocation == -1) {
        cout << "Warning: could not find uniform uShadowWorldTexelSize" << endl;
    }

    int useShadowsLocation = glGetUniformLocation(shaderProgram, "uUseShadows");
    if (useShadowsLocation == -1) {
        cout << "Warning: could not find uniform uUseShadows" << endl;
    }

    int shadowMapLocation = glGetUniformLocation(shaderProgram, "uShadowMap");
    if (shadowMapLocation == -1) {
        cout << "Warning: could not find uniform uShadowMap" << endl;
    }

    // The shadow pass shares the same uModel name so renderScene can upload
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

    int exposureLocation = glGetUniformLocation(screenShaderProgram, "uExposure");
    if (exposureLocation == -1) {
        cout << "Warning: could not find uniform uExposure" << endl;
    }

    int splitFractionLocation = glGetUniformLocation(screenShaderProgram, "uSplitFraction");
    if (splitFractionLocation == -1) {
        cout << "Warning: could not find uniform uSplitFraction" << endl;
    }

    glUseProgram(screenShaderProgram);
    // Sampler uniforms store texture-unit indices, not texture object IDs.
    if (sceneColorLocation != -1) {
        extraGl.uniform1i(sceneColorLocation, 0);
    }
    if (sceneDepthLocation != -1) {
        extraGl.uniform1i(sceneDepthLocation, 1);
    }

    // ==============================
    // EXPERIMENT 18: the alley as one static mesh
    // ==============================
    // The 24-vertex cube and its seven model matrices are gone. The scene file
    // is already baked into world space by SceneFile.cpp, so the whole
    // environment is a single VBO drawn with one glDrawElements per pass,
    // rather than one draw call per prop. At 2551 primitives that difference
    // matters: per-prop draws would be ~2500 state changes a frame for
    // geometry that never moves.
    //
    // Vertex attributes vary per vertex; uniforms are shared for the whole draw
    // call. Each vertex is ten floats: position.xyz, albedo.rgb, normal.xyz,
    // emission. Attribute 3 is deliberately skipped so Mesh.cpp's imported-mesh
    // UVs can keep it and both VAOs feed the same shader program.
    unsigned int VBO = 0, VAO = 0, EBO = 0;
    glGenBuffers(1, &VBO);
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &EBO);
    GLsizei indexCount = 0;

    // Re-uploads the whole scene mesh. Called once at startup and again on
    // every L reload, so a one-line edit to the scene file shows up without a
    // rebuild. Uploading the entire buffer is fine here because the scene is
    // static: there is no per-frame cost to pay for the simplicity.
    auto uploadSceneGeometry = [&](const SceneDescription& scene) {
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(scene.vertices.size() * sizeof(SceneVertex)),
                     scene.vertices.data(), GL_STATIC_DRAW);

        // The EBO stores index data. Its binding is remembered by the currently bound VAO.
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(scene.indices.size() * sizeof(unsigned int)),
                     scene.indices.data(), GL_STATIC_DRAW);

        // Stride is the byte distance from one vertex to the next. Offsets come
        // from offsetof rather than hand-counted floats so the layout cannot
        // drift out of step with struct SceneVertex.
        const GLsizei stride = static_cast<GLsizei>(sizeof(SceneVertex));
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
                              (void*)offsetof(SceneVertex, position));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                              (void*)offsetof(SceneVertex, albedo));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride,
                              (void*)offsetof(SceneVertex, normal));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride,
                              (void*)offsetof(SceneVertex, emission));
        glEnableVertexAttribArray(4);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        indexCount = static_cast<GLsizei>(scene.indices.size());
    };

    if (hasSceneGeometry) {
        uploadSceneGeometry(sceneDescription);
    }

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
        // RGBA16F, not RGB8: basic.frag writes linear radiance, and the lit
        // windows and bulbs go well above 1.0. In an 8-bit target they would be
        // clipped and quantized before the defocus gather ever averaged them,
        // which is what turns a bright out-of-focus bulb into a flat grey disc
        // instead of a bokeh highlight.
        extraGl.bindTexture(GL_TEXTURE_2D, sceneColorTexture);
        extraGl.texImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
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

    // The scene never moves, only the camera does, so one orthographic
    // light-space matrix covers every frame. It is fitted to the scene file's
    // `shadow lo/hi` region rather than to the geometry bounds: the alley also
    // contains the facade at z = -47, and a frustum large enough to hold that
    // would spend almost all of its 4096 texels on scenery the camera only
    // ever sees as a distant silhouette.
    auto computeLightSpaceMatrix = [&]() {
        glm::vec3 lightDir = glm::normalize(lightDirection);

        glm::vec3 regionLow(-shadowFrustumFallbackHalfExtent);
        glm::vec3 regionHigh(shadowFrustumFallbackHalfExtent);
        if (hasSceneGeometry && sceneDescription.hasShadowRegion) {
            regionLow = glm::vec3(sceneDescription.shadowLow.x, sceneDescription.shadowLow.y,
                                  sceneDescription.shadowLow.z);
            regionHigh = glm::vec3(sceneDescription.shadowHigh.x, sceneDescription.shadowHigh.y,
                                   sceneDescription.shadowHigh.z);
        }
        const glm::vec3 regionCenter = 0.5f * (regionLow + regionHigh);
        // A sphere around the region: its radius is the same from every light
        // angle, so the shadow map does not change resolution (or start
        // clipping) as the light direction slider moves.
        const float regionRadius = 0.5f * glm::length(regionHigh - regionLow);

        const glm::vec3 lightEye = regionCenter + lightDir * (regionRadius + 2.0f);
        const glm::vec3 upHint = (std::abs(lightDir.y) > 0.99f) ? glm::vec3(0.0f, 0.0f, 1.0f)
                                                                : glm::vec3(0.0f, 1.0f, 0.0f);
        const glm::mat4 lightView = glm::lookAt(lightEye, regionCenter, upHint);
        const glm::mat4 lightProjection = glm::ortho(-regionRadius, regionRadius,
                                                     -regionRadius, regionRadius,
                                                     shadowNearPlane,
                                                     min(2.0f * regionRadius + 4.0f, shadowFarPlane));
        // basic.vert offsets the shadow lookup by about one texel along the
        // normal, so it needs to know how big a texel is in meters.
        shadowWorldTexelSize = 2.0f * regionRadius / static_cast<float>(shadowMapSize);
        return lightProjection * lightView;
    };
    glm::mat4 lightSpaceMatrix = computeLightSpaceMatrix();
    cout << "Shadow map: " << shadowMapSize << " px across "
         << shadowWorldTexelSize * static_cast<float>(shadowMapSize) << " m ("
         << shadowWorldTexelSize * 100.0f << " cm per texel)" << endl;

    if (wireframeMode) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    glEnable(GL_DEPTH_TEST);
    // GL_LESS keeps the nearest fragment for each framebuffer location.
    glDepthFunc(GL_LESS);

    bool screenshotKeyWasPressed = false;

    // Shadow-map invalidation state. Initialised so the first frame builds it.
    bool shadowMapDirty = true;
    glm::vec3 shadowBuiltLightDirection = lightDirection;
    bool shadowBuiltSceneVisible = showSceneGeometry;
    bool shadowBuiltMeshVisible = showImportedMesh;

    // ==============================
    // EXPERIMENT 17: Dear ImGui control panel
    // ==============================
    // Callbacks were installed on the window above, so install_callbacks = true
    // makes ImGui chain to them instead of replacing them.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    // Slightly smaller text and tighter padding: the panel has to fit in ~250
    // px of width without wrapping its readout lines.
    ImGui::GetIO().FontGlobalScale = 0.92f;
    ImGui::GetStyle().WindowPadding = ImVec2(8.0f, 6.0f);
    ImGui::GetStyle().ItemSpacing = ImVec2(6.0f, 4.0f);
    ImGui::GetStyle().WindowRounding = 4.0f;
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

        // Mouse-look mode, toggled by the panel button or by Tab. Captured means
        // the cursor is hidden and locked and the camera follows every mouse
        // movement; free means the cursor is a normal pointer for the panel.
        if (cursorToggleRequested) {
            cursorToggleRequested = false;
            firstMouse = true;
            rightMouseLook = false;
            const bool wasCaptured = glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;
            glfwSetInputMode(window, GLFW_CURSOR,
                             wasCaptured ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
        }
        const bool cursorCaptured = glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;

        // L (or the panel button) re-reads the scene file. A failed reload
        // leaves the previously loaded scene on the GPU rather than emptying
        // the window, so a typo while editing is recoverable.
        if (sceneReloadRequested) {
            sceneReloadRequested = false;
            SceneDescription reloaded;
            string reloadError;
            if (!loadSceneFile(sceneFilePath, reloaded, reloadError)) {
                cout << "Reload failed, keeping the current scene: " << reloadError << endl;
            } else {
                sceneDescription = std::move(reloaded);
                applySceneSettings(sceneDescription);
                uploadSceneGeometry(sceneDescription);
                hasSceneGeometry = true;
                shadowMapDirty = true;
                cout << "Reloaded " << scenePath << ": " << sceneDescription.primitiveCount
                     << " primitives, " << sceneDescription.triangleCount() << " triangles" << endl;
            }
        }

        // The shadow map only has to be redrawn when the light moves or the
        // geometry changes. Comparing against what it was last built with is
        // less error-prone than setting a dirty flag at every call site that
        // could matter, and there are now several (panel sliders, hotkeys,
        // reloads).
        if (lightDirection != shadowBuiltLightDirection ||
            showSceneGeometry != shadowBuiltSceneVisible ||
            showImportedMesh != shadowBuiltMeshVisible) {
            shadowBuiltLightDirection = lightDirection;
            shadowBuiltSceneVisible = showSceneGeometry;
            shadowBuiltMeshVisible = showImportedMesh;
            shadowMapDirty = true;
        }

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
        if (panelMode != PanelMode::Hidden) {
            const bool full = panelMode == PanelMode::Full;
            ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
            // AlwaysAutoResize instead of a fixed 360x560: the compact panel is
            // then only as tall as the controls it actually shows, which leaves
            // the frame visible. The old panel covered a ninth of a 1200 px
            // window while most of it was collapsed headers.
            ImGui::SetNextWindowSizeConstraints(ImVec2(240.0f, 0.0f), ImVec2(260.0f, FLT_MAX));
            ImGui::Begin(full ? "DoF Controls - full (G)" : "DoF Controls (G)", nullptr,
                         ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNav);

            // Mouse look as a button, because holding the right button to turn
            // is awkward while also reaching for a slider. One click swaps
            // between "pointer for the panel" and "camera follows the mouse".
            if (ImGui::Button(cursorCaptured ? "Mouse: camera (click to free)"
                                             : "Mouse: pointer (click to look)",
                              ImVec2(-1.0f, 0.0f))) {
                cursorToggleRequested = true;
            }
            if (cursorCaptured) {
                // While the cursor is captured the panel cannot receive clicks,
                // so say plainly what gets the pointer back. This is the same
                // deal any engine viewport makes.
                ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.25f, 1.0f), "Press Tab to free the mouse");
            }
            ImGui::Separator();

            // The four things a parameter sweep actually needs, as buttons
            // rather than sliders: a slider cannot be set to exactly f/1.4
            // twice in a row, and the Cycles reference renders at fixed stops.
            ImGui::TextUnformatted("Aperture");
            const struct { const char* label; float value; } apertures[] = {
                {"f/1.4", 1.4f}, {"f/2.8", 2.8f}, {"f/8", 8.0f}
            };
            for (int index = 0; index < 3; ++index) {
                if (index > 0) ImGui::SameLine();
                const bool active = std::abs(fNumber - apertures[index].value) < 0.01f;
                if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.30f, 0.50f, 0.75f, 1.0f));
                if (ImGui::Button(apertures[index].label, ImVec2(66.0f, 0.0f))) {
                    fNumber = apertures[index].value;
                }
                if (active) ImGui::PopStyleColor();
            }

            ImGui::TextUnformatted("Focus");
            const struct { const char* label; float value; } focusPresets[] = {
                {"1.6 m", 1.6f}, {"5 m", 5.0f}, {"15 m", 15.0f}
            };
            for (int index = 0; index < 3; ++index) {
                if (index > 0) ImGui::SameLine();
                const bool active = std::abs(focusDistanceMeters - focusPresets[index].value) < 0.01f;
                if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.30f, 0.50f, 0.75f, 1.0f));
                if (ImGui::Button(focusPresets[index].label, ImVec2(66.0f, 0.0f))) {
                    focusDistanceMeters = focusPresets[index].value;
                }
                if (active) ImGui::PopStyleColor();
            }

            ImGui::SliderFloat("##focus", &focusDistanceMeters, 0.4f, 40.0f, "focus %.2f m",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("##fnumber", &fNumber, 1.0f, 22.0f, "f/%.1f");
            ImGui::SliderFloat("##lens", &focalLengthMillimeters, 18.0f, 200.0f, "lens %.0f mm");

            const char* screenModeNames[] = {
                "Color", "Raw depth", "Linear depth", "CoC magnitude", "CoC signed",
                "Basic DoF", "Split sharp|DoF"
            };
            int screenModeIndex = static_cast<int>(screenMode);
            ImGui::SetNextItemWidth(-1.0f);
            if (ImGui::Combo("##view", &screenModeIndex, screenModeNames, IM_ARRAYSIZE(screenModeNames))) {
                screenMode = static_cast<ScreenMode>(screenModeIndex);
            }
            if (screenMode == ScreenMode::SplitSharpDoF) {
                ImGui::SliderFloat("##split", &splitFraction, 0.0f, 1.0f, "wipe %.2f");
            }

            // The readout that makes the blur numbers legible. The professor's
            // 120 px target is a RADIUS, and at 50 mm f/1.4 focused 5 m away
            // the far background only reaches about 9 px: the ceiling is not
            // the thing limiting the blur, the lens is. Showing both numbers
            // side by side is the quickest way to see that.
            const float farCoCRadius =
                0.5f * signedCoCDiameterPixels(farPlane, framebufferHeight);
            const float nearCoCRadius =
                0.5f * signedCoCDiameterPixels(nearPlane + 0.2f, framebufferHeight);
            const float reachedRadius = min(max(std::abs(farCoCRadius), std::abs(nearCoCRadius)),
                                            maxBlurRadiusPixels);
            ImGui::Separator();
            ImGui::Text("CoC radius: bg %.1f px, fg %.1f px", std::abs(farCoCRadius),
                        std::abs(nearCoCRadius));
            ImGui::Text("gather uses %.0f px of %.0f max", reachedRadius, maxBlurRadiusPixels);
            if (reachedRadius < maxBlurRadiusPixels * 0.25f) {
                ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.25f, 1.0f),
                                   "lens-limited: press K for a strong blur");
            }
            ImGui::Text("%.0f FPS | %d taps", ImGui::GetIO().Framerate, cocSampleCount);

            if (full) {
                ImGui::Separator();
                if (ImGui::CollapsingHeader("Blur gather")) {
                    ImGui::SliderFloat("Max radius (px)", &maxBlurRadiusPixels, 0.0f, 300.0f);
                    ImGui::SliderInt("CoC samples", &cocSampleCount, 4, 256);
                    ImGui::SliderFloat("CoC debug max (px)", &cocVisualizationMaxPixels, 1.0f, 200.0f);
                    ImGui::SliderFloat("Depth view max (m)", &depthVisualizationMax, 1.0f, 100.0f);
                }
                if (ImGui::CollapsingHeader("Sensor / exposure")) {
                    ImGui::SliderFloat("Sensor height (mm)", &sensorHeightMillimeters, 8.0f, 36.0f);
                    ImGui::SliderFloat("Exposure", &exposure, 0.1f, 8.0f, "%.2f x",
                                       ImGuiSliderFlags_Logarithmic);
                    ImGui::Checkbox("Physical camera FOV", &usePhysicalCameraProjection);
                }
                if (ImGui::CollapsingHeader("Lighting / shadows")) {
                    ImGui::Checkbox("Shadows", &enableShadows);
                    ImGui::SliderFloat3("Sun dir (to sun)", glm::value_ptr(lightDirection), -1.0f, 1.0f);
                    if (glm::length(lightDirection) < 0.05f) {
                        lightDirection = glm::vec3(-0.45f, 0.78f, 0.44f); // Never normalize a zero vector.
                    }
                    ImGui::ColorEdit3("Sun color", glm::value_ptr(lightColor));
                    ImGui::SliderFloat("Sun energy (W/m2)", &lightEnergy, 0.0f, 20.0f);
                    ImGui::ColorEdit3("Sky color", glm::value_ptr(ambientColor));
                    ImGui::SliderFloat("Sky strength", &ambientStrength, 0.0f, 1.0f);
                    ImGui::TextDisabled("Edits here diverge from the scene file");
                }
                if (ImGui::CollapsingHeader("Scene")) {
                    ImGui::Checkbox("Alley geometry", &showSceneGeometry);
                    ImGui::Checkbox("Imported mesh", &showImportedMesh);
                    ImGui::ColorEdit3("Mesh albedo", glm::value_ptr(importedSceneAlbedo));
                    if (ImGui::Button("Reload scene file (L)")) {
                        sceneReloadRequested = true;
                    }
                    ImGui::TextDisabled("%zu prims, %zu tris", sceneDescription.primitiveCount,
                                        sceneDescription.triangleCount());
                }
            }
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
                if (lightEnergyLocation != -1) {
                    glUniform1f(lightEnergyLocation, lightEnergy);
                }
                if (skyRadianceLocation != -1) {
                    const glm::vec3 sky = skyRadiance();
                    extraGl.uniform3fv(skyRadianceLocation, 1, glm::value_ptr(sky));
                }
                if (shadowWorldTexelSizeLocation != -1) {
                    glUniform1f(shadowWorldTexelSizeLocation, shadowWorldTexelSize);
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

            // The alley: already baked into world space by SceneFile.cpp, so
            // its model matrix is the identity and the whole environment is one
            // draw call. P * V * M is still uploaded as three uniforms; the
            // perspective divide happens after the vertex shader and produces NDC.
            if (hasSceneGeometry && showSceneGeometry && indexCount > 0) {
                const glm::mat4 identity(1.0f);
                if (activeModelLocation != -1) {
                    glUniformMatrix4fv(activeModelLocation, 1, GL_FALSE, glm::value_ptr(identity));
                }
                if (!shadowPass) {
                    // Per-vertex albedo and emission come from the buffer.
                    extraGl.uniform1i(importedMeshLocation, 0);
                }
                glBindVertexArray(VAO);
                glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, nullptr);
                glBindVertexArray(0);
            }

            // The hero subject at the focus plane, standing on the crate the
            // scene file puts there. Its VAO leaves attributes 1 and 4
            // disabled, so basic.frag takes uOverrideAlbedo and sees no
            // emission for it.
            if (hasImportedMesh && showImportedMesh) {
                glm::mat4 model = glm::translate(glm::mat4(1.0f), importedScenePosition);
                model = glm::rotate(
                    model,
                    glm::radians(-90.0f),
                    glm::vec3(1.0f, 0.0f, 0.0f)
                );
                model = glm::rotate(model, glm::radians(importedSceneRotationYDegrees), glm::vec3(0.0f, 1.0f, 0.0f));
                model = glm::scale(model, glm::vec3(importedSceneScale));
                if (activeModelLocation != -1) {
                    glUniformMatrix4fv(activeModelLocation, 1, GL_FALSE, glm::value_ptr(model));
                }
                if (!shadowPass) {
                    extraGl.uniform1i(importedMeshLocation, 1);
                    extraGl.uniform3fv(overrideAlbedoLocation, 1, glm::value_ptr(importedSceneAlbedo));
                }
                // Same scene pass, depth test and FBO attachments as the alley. No special depth path.
                drawMesh(importedMesh);
            }
        };

        // Pass 0: render depth from the light into the shadow map. Front faces are
        // kept (no culling tricks); the normal offset in basic.vert plus the
        // slope-scaled bias in basic.frag handle self-shadow acne.
        //
        // Only redrawn when something it depends on actually changes. The scene
        // is static, so at 38572 triangles and 4096 px this would otherwise be a
        // second full geometry pass every frame for an identical result.
        if (enableShadows && shadowMapDirty) {
            extraGl.bindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
            glViewport(0, 0, shadowMapSize, shadowMapSize);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glClear(GL_DEPTH_BUFFER_BIT);
            renderScene(true);
            extraGl.bindFramebuffer(GL_FRAMEBUFFER, 0);
            shadowMapDirty = false;
        }

        // The sky colour is the clear colour, the ambient term and the Cycles
        // world background, all from the same expression. It is a linear
        // radiance here because the scene target is RGBA16F; screen.frag
        // encodes it for display along with everything else.
        const glm::vec3 sky = skyRadiance();

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
            glClearColor(sky.r, sky.g, sky.b, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            renderScene(false);

            // Pass 2: present color, depth/CoC diagnostics, BasicDoF or the split view through a screen quad.
            extraGl.bindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, framebufferWidth, framebufferHeight);
            glDisable(GL_DEPTH_TEST); // The quad covers the screen and only presents an already-rendered image.
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
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
            if (exposureLocation != -1) {
                glUniform1f(exposureLocation, exposure);
            }
            if (splitFractionLocation != -1) {
                glUniform1f(splitFractionLocation, splitFraction);
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
            glClearColor(sky.r, sky.g, sky.b, 1.0f);
            // Clearing the color buffer does not clear stored depth values; reset both buffers every frame.
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            // Direct-to-window path: no screen quad, so nothing applies exposure
            // or the sRGB curve and the linear radiance looks dark. Kept only as
            // a way to check that the FBO round-trip is not what broke an image.
            renderScene(false);
        }

        const bool screenshotKeyIsPressed = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
        if (screenshotKeyIsPressed && !screenshotKeyWasPressed) {
            glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
            // Two files per press. `output/latest.png` stays, because it is what
            // the workflow notes tell you to open in the editor. The second name
            // encodes the settings in exactly the form render_dof.py uses for
            // its outputs (gl_focus5m_f1.4.png against rt_focus5m_f1.4.png), so
            // a matched pair can be found by name months later instead of by
            // remembering which screenshot was which.
            saveScreenshot("output/latest.png", framebufferWidth, framebufferHeight);
            std::ostringstream matched;
            matched << "output/gl_focus" << formatCompact(focusDistanceMeters) << "m_f"
                    << formatCompact(fNumber);
            if (std::abs(focalLengthMillimeters - 50.0f) > 0.01f) {
                matched << "_" << formatCompact(focalLengthMillimeters) << "mm";
            }
            if (screenMode != ScreenMode::BasicDoF) {
                matched << "_" << screenModeFileTag(screenMode);
            }
            matched << ".png";
            saveScreenshot(matched.str(), framebufferWidth, framebufferHeight);
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
