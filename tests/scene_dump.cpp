// A small CPU-only bridge lets Python compare actual C++ vertices, not summaries.
#include "SceneFile.h"
#include <iomanip>
#include <iostream>
#include <limits>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    SceneDescription scene;
    std::string error;
    if (!loadSceneFile(argv[1], scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    std::cout << "{\"capture\":[" << scene.captureWidth << ',' << scene.captureHeight << "],\"imported\":[";
    bool firstSetting = true;
    for (const auto& value :
         {scene.importPosition, scene.importRotation, scene.importScale, scene.importAlbedo}) {
        if (!firstSetting) std::cout << ',';
        firstSetting = false;
        std::cout << value.x << ',' << value.y << ',' << value.z;
    }
    std::cout << "],\"vertices\":[";
    bool first = true;
    for (const auto& vertex : scene.vertices) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << '[';
        for (int i = 0; i < 3; ++i) std::cout << vertex.position[i] << ',';
        for (int i = 0; i < 3; ++i) std::cout << vertex.normal[i] << ',';
        for (int i = 0; i < 3; ++i) std::cout << vertex.albedo[i] << ',';
        std::cout << vertex.emission << ']';
    }
    std::cout << "],\"indices\":[";
    first = true;
    for (auto index : scene.indices) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << index;
    }
    std::cout << "]}\n";
}
