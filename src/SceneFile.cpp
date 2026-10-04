#include "SceneFile.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Twin of tools/scene/scene_loader.py. Every rule here exists in that file too:
// the grammar, the key arity table, the primitive tessellation, the winding and
// the transform order. Arithmetic runs in double and is narrowed to float only
// when a vertex is stored, so the two builders agree to more than float
// precision and tests/scene_file.cpp can compare against numbers produced by
// the Python side.

namespace {

struct SceneParseError : std::runtime_error {
    explicit SceneParseError(const std::string& message) : std::runtime_error(message) {}
};

struct Vec3d {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    double operator[](int axis) const { return axis == 0 ? x : (axis == 1 ? y : z); }
    double& operator[](int axis) { return axis == 0 ? x : (axis == 1 ? y : z); }
};

// Every key takes a fixed number of floats, so parsing needs no lookahead.
const std::map<std::string, int>& keyArity() {
    static const std::map<std::string, int> table = {
        {"pos", 3},  {"size", 3},    {"rot", 3},  {"rgb", 3},    {"emit", 1},
        {"seg", 1},  {"smooth", 1},  {"yaw", 1},  {"pitch", 1},  {"focus", 1},
        {"fnumber", 1}, {"lens", 1}, {"sensor", 1}, {"dir", 3},  {"color", 3},
        {"energy", 1}, {"angle", 1}, {"strength", 1}, {"value", 1},
        {"lo", 3},   {"hi", 3},
    };
    return table;
}

std::string lineTag(std::size_t lineNumber) {
    return "line " + std::to_string(lineNumber) + ": ";
}

double toNumber(const std::string& token, std::size_t lineNumber, const std::string& key) {
    const char* begin = token.c_str();
    char* end = nullptr;
    const double value = std::strtod(begin, &end);
    if (end != begin + token.size() || token.empty()) {
        throw SceneParseError(lineTag(lineNumber) + "'" + token + "' after '" + key + "' is not a number");
    }
    if (!std::isfinite(value)) {
        throw SceneParseError(lineTag(lineNumber) + "'" + key + "' must be finite");
    }
    return value;
}

// `key v.. key v..` pairs. Unknown, misplaced or repeated keys are errors, so a
// typo in the scene file is reported instead of being silently ignored.
std::map<std::string, std::vector<double>> readKeyValues(const std::vector<std::string>& tokens,
                                                         std::size_t lineNumber,
                                                         const std::set<std::string>& allowed) {
    std::map<std::string, std::vector<double>> found;
    std::size_t index = 0;
    while (index < tokens.size()) {
        const std::string& key = tokens[index];
        const auto arityEntry = keyArity().find(key);
        if (arityEntry == keyArity().end()) {
            throw SceneParseError(lineTag(lineNumber) + "unknown key '" + key + "'");
        }
        if (allowed.find(key) == allowed.end()) {
            throw SceneParseError(lineTag(lineNumber) + "key '" + key + "' is not valid here");
        }
        if (found.find(key) != found.end()) {
            throw SceneParseError(lineTag(lineNumber) + "key '" + key + "' appears twice");
        }
        const int arity = arityEntry->second;
        if (index + 1 + static_cast<std::size_t>(arity) > tokens.size()) {
            throw SceneParseError(lineTag(lineNumber) + "key '" + key + "' needs " +
                                  std::to_string(arity) + " number(s)");
        }
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(arity));
        for (int offset = 0; offset < arity; ++offset) {
            values.push_back(toNumber(tokens[index + 1 + static_cast<std::size_t>(offset)], lineNumber, key));
        }
        found.emplace(key, std::move(values));
        index += 1 + static_cast<std::size_t>(arity);
    }
    return found;
}

bool has(const std::map<std::string, std::vector<double>>& found, const char* key) {
    return found.find(key) != found.end();
}

Vec3d vectorOf(const std::map<std::string, std::vector<double>>& found, const char* key) {
    const auto& values = found.at(key);
    return Vec3d{values[0], values[1], values[2]};
}

double scalarOf(const std::map<std::string, std::vector<double>>& found, const char* key) {
    return found.at(key)[0];
}

double length(const Vec3d& vector) {
    return std::sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

Vec3d normalize(const Vec3d& vector) {
    const double magnitude = length(vector);
    if (magnitude < 1e-12) {
        return Vec3d{0.0, 1.0, 0.0};  // Same fallback as the Python builder.
    }
    return Vec3d{vector.x / magnitude, vector.y / magnitude, vector.z / magnitude};
}

struct Matrix3 {
    double m[3][3]{};
};

Matrix3 multiply(const Matrix3& a, const Matrix3& b) {
    Matrix3 result;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += a.m[row][k] * b.m[k][column];
            }
            result.m[row][column] = sum;
        }
    }
    return result;
}

Vec3d apply(const Matrix3& matrix, const Vec3d& vector) {
    Vec3d result;
    for (int row = 0; row < 3; ++row) {
        result[row] = matrix.m[row][0] * vector.x + matrix.m[row][1] * vector.y +
                      matrix.m[row][2] * vector.z;
    }
    return result;
}

// R = Ry * Rx * Rz, the one order both builders use.
Matrix3 rotationMatrix(const Vec3d& degrees) {
    const double x = degrees.x * M_PI / 180.0;
    const double y = degrees.y * M_PI / 180.0;
    const double z = degrees.z * M_PI / 180.0;
    const double cx = std::cos(x), sx = std::sin(x);
    const double cy = std::cos(y), sy = std::sin(y);
    const double cz = std::cos(z), sz = std::sin(z);
    Matrix3 rotateX;
    rotateX.m[0][0] = 1.0; rotateX.m[1][1] = cx; rotateX.m[1][2] = -sx;
    rotateX.m[2][1] = sx;  rotateX.m[2][2] = cx;
    Matrix3 rotateY;
    rotateY.m[0][0] = cy;  rotateY.m[0][2] = sy; rotateY.m[1][1] = 1.0;
    rotateY.m[2][0] = -sy; rotateY.m[2][2] = cy;
    Matrix3 rotateZ;
    rotateZ.m[0][0] = cz;  rotateZ.m[0][1] = -sz; rotateZ.m[1][0] = sz;
    rotateZ.m[1][1] = cz;  rotateZ.m[2][2] = 1.0;
    return multiply(rotateY, multiply(rotateX, rotateZ));
}

struct Primitive {
    std::string kind;
    Vec3d pos{0.0, 0.0, 0.0};
    Vec3d size{1.0, 1.0, 1.0};
    Vec3d rot{0.0, 0.0, 0.0};
    Vec3d rgb{0.8, 0.8, 0.8};
    double emit = 0.0;
    int segments = 0;
    bool smooth = true;
};

// Unit cube: 6 faces x 4 corners, wound counter-clockwise seen from outside.
// Same face order and corner order as the hand-written cube this scene replaced.
struct BoxFace {
    double corners[4][3];
    double normal[3];
};

const BoxFace kBoxFaces[6] = {
    {{{-0.5, -0.5, 0.5}, {0.5, -0.5, 0.5}, {0.5, 0.5, 0.5}, {-0.5, 0.5, 0.5}}, {0.0, 0.0, 1.0}},
    {{{0.5, -0.5, -0.5}, {-0.5, -0.5, -0.5}, {-0.5, 0.5, -0.5}, {0.5, 0.5, -0.5}}, {0.0, 0.0, -1.0}},
    {{{-0.5, -0.5, -0.5}, {-0.5, -0.5, 0.5}, {-0.5, 0.5, 0.5}, {-0.5, 0.5, -0.5}}, {-1.0, 0.0, 0.0}},
    {{{0.5, -0.5, 0.5}, {0.5, -0.5, -0.5}, {0.5, 0.5, -0.5}, {0.5, 0.5, 0.5}}, {1.0, 0.0, 0.0}},
    {{{-0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, 0.5, -0.5}, {-0.5, 0.5, -0.5}}, {0.0, 1.0, 0.0}},
    {{{-0.5, -0.5, -0.5}, {0.5, -0.5, -0.5}, {0.5, -0.5, 0.5}, {-0.5, -0.5, 0.5}}, {0.0, -1.0, 0.0}},
};

// Appends one primitive's triangles to the description. The vertex order has to
// match scene_loader.py primitive for primitive, which is why the loops below
// are written out rather than shared between the three shapes.
class GeometryBuilder {
public:
    explicit GeometryBuilder(SceneDescription& scene) : scene_(scene) {}

    void emit(const Primitive& primitive) {
        rotation_ = rotationMatrix(primitive.rot);
        current_ = &primitive;
        if (primitive.kind == "box") {
            emitBox();
        } else if (primitive.kind == "cyl") {
            emitCylinder();
        } else {
            emitSphere();
        }
        current_ = nullptr;
    }

private:
    unsigned int place(const Vec3d& localPosition, const Vec3d& localNormal) {
        const Vec3d& size = current_->size;
        const Vec3d scaled{localPosition.x * size.x, localPosition.y * size.y, localPosition.z * size.z};
        const Vec3d rotated = apply(rotation_, scaled);
        // Inverse transpose of R*S for a diagonal S: divide by the scale, then rotate.
        const Vec3d unscaled{localNormal.x / size.x, localNormal.y / size.y, localNormal.z / size.z};
        const Vec3d normal = normalize(apply(rotation_, unscaled));

        SceneVertex vertex;
        vertex.position[0] = static_cast<float>(rotated.x + current_->pos.x);
        vertex.position[1] = static_cast<float>(rotated.y + current_->pos.y);
        vertex.position[2] = static_cast<float>(rotated.z + current_->pos.z);
        vertex.albedo[0] = static_cast<float>(current_->rgb.x);
        vertex.albedo[1] = static_cast<float>(current_->rgb.y);
        vertex.albedo[2] = static_cast<float>(current_->rgb.z);
        vertex.normal[0] = static_cast<float>(normal.x);
        vertex.normal[1] = static_cast<float>(normal.y);
        vertex.normal[2] = static_cast<float>(normal.z);
        vertex.emission = static_cast<float>(current_->emit);

        for (int axis = 0; axis < 3; ++axis) {
            float& low = axis == 0 ? scene_.boundsMin.x : (axis == 1 ? scene_.boundsMin.y : scene_.boundsMin.z);
            float& high = axis == 0 ? scene_.boundsMax.x : (axis == 1 ? scene_.boundsMax.y : scene_.boundsMax.z);
            if (scene_.vertices.empty()) {
                low = vertex.position[axis];
                high = vertex.position[axis];
            } else {
                low = std::min(low, vertex.position[axis]);
                high = std::max(high, vertex.position[axis]);
            }
        }
        scene_.vertices.push_back(vertex);
        return static_cast<unsigned int>(scene_.vertices.size() - 1);
    }

    void addTriangle(unsigned int i0, unsigned int i1, unsigned int i2, bool smooth) {
        scene_.indices.push_back(i0);
        scene_.indices.push_back(i1);
        scene_.indices.push_back(i2);
        if (smooth) {
            ++scene_.smoothTriangleCount;
        }
        if (current_->emit > 0.0) {
            ++scene_.emissiveTriangleCount;
        }
    }

    void emitBox() {
        for (const BoxFace& face : kBoxFaces) {
            const Vec3d normal{face.normal[0], face.normal[1], face.normal[2]};
            unsigned int corner[4];
            for (int index = 0; index < 4; ++index) {
                corner[index] = place(Vec3d{face.corners[index][0], face.corners[index][1],
                                            face.corners[index][2]}, normal);
            }
            addTriangle(corner[0], corner[1], corner[2], false);
            addTriangle(corner[2], corner[3], corner[0], false);
        }
    }

    void emitCylinder() {
        const int segments = current_->segments;
        std::vector<double> angles(static_cast<std::size_t>(segments));
        for (int index = 0; index < segments; ++index) {
            angles[static_cast<std::size_t>(index)] = 2.0 * M_PI * index / segments;
        }

        if (current_->smooth) {
            std::vector<unsigned int> bottom, top;
            bottom.reserve(static_cast<std::size_t>(segments));
            top.reserve(static_cast<std::size_t>(segments));
            for (const double angle : angles) {
                const Vec3d direction{std::cos(angle), 0.0, std::sin(angle)};
                bottom.push_back(place(Vec3d{0.5 * direction.x, -0.5, 0.5 * direction.z}, direction));
                top.push_back(place(Vec3d{0.5 * direction.x, 0.5, 0.5 * direction.z}, direction));
            }
            for (int index = 0; index < segments; ++index) {
                const std::size_t self = static_cast<std::size_t>(index);
                const std::size_t next = static_cast<std::size_t>((index + 1) % segments);
                addTriangle(bottom[self], top[self], top[next], true);
                addTriangle(bottom[self], top[next], bottom[next], true);
            }
        } else {
            for (int index = 0; index < segments; ++index) {
                const double a = angles[static_cast<std::size_t>(index)];
                const double b = angles[static_cast<std::size_t>((index + 1) % segments)];
                // The seam quad wraps past 2*pi, so its mid-angle needs the turn added.
                const double mid = index + 1 < segments ? 0.5 * (a + b) : 0.5 * (a + b + 2.0 * M_PI);
                const Vec3d faceNormal{std::cos(mid), 0.0, std::sin(mid)};
                const unsigned int quad[4] = {
                    place(Vec3d{0.5 * std::cos(a), -0.5, 0.5 * std::sin(a)}, faceNormal),
                    place(Vec3d{0.5 * std::cos(a), 0.5, 0.5 * std::sin(a)}, faceNormal),
                    place(Vec3d{0.5 * std::cos(b), 0.5, 0.5 * std::sin(b)}, faceNormal),
                    place(Vec3d{0.5 * std::cos(b), -0.5, 0.5 * std::sin(b)}, faceNormal),
                };
                addTriangle(quad[0], quad[1], quad[2], false);
                addTriangle(quad[0], quad[2], quad[3], false);
            }
        }

        for (const double sign : {1.0, -1.0}) {
            const Vec3d normal{0.0, sign, 0.0};
            const unsigned int center = place(Vec3d{0.0, 0.5 * sign, 0.0}, normal);
            std::vector<unsigned int> ring;
            ring.reserve(static_cast<std::size_t>(segments));
            for (const double angle : angles) {
                ring.push_back(place(Vec3d{0.5 * std::cos(angle), 0.5 * sign, 0.5 * std::sin(angle)}, normal));
            }
            for (int index = 0; index < segments; ++index) {
                const std::size_t self = static_cast<std::size_t>(index);
                const std::size_t next = static_cast<std::size_t>((index + 1) % segments);
                if (sign > 0.0) {
                    addTriangle(center, ring[next], ring[self], false);
                } else {
                    addTriangle(center, ring[self], ring[next], false);
                }
            }
        }
    }

    // Sphere: UV grid with a duplicated seam column so both builders index alike.
    void emitSphere() {
        const int segments = current_->segments;
        const int rings = std::max(2, segments / 2);
        std::vector<std::vector<unsigned int>> grid;
        grid.reserve(static_cast<std::size_t>(rings) + 1);
        for (int ring = 0; ring <= rings; ++ring) {
            const double phi = -0.5 * M_PI + M_PI * ring / rings;
            std::vector<unsigned int> row;
            row.reserve(static_cast<std::size_t>(segments) + 1);
            for (int longitude = 0; longitude <= segments; ++longitude) {
                const double theta = 2.0 * M_PI * longitude / segments;
                const Vec3d normal{std::cos(phi) * std::cos(theta), std::sin(phi),
                                   std::cos(phi) * std::sin(theta)};
                row.push_back(place(Vec3d{0.5 * normal.x, 0.5 * normal.y, 0.5 * normal.z}, normal));
            }
            grid.push_back(std::move(row));
        }
        for (int ring = 0; ring < rings; ++ring) {
            for (int longitude = 0; longitude < segments; ++longitude) {
                const std::size_t low = static_cast<std::size_t>(ring);
                const std::size_t high = low + 1;
                const std::size_t left = static_cast<std::size_t>(longitude);
                const std::size_t right = left + 1;
                if (ring > 0) {  // Degenerate at the south pole, where the low row collapses.
                    addTriangle(grid[low][left], grid[high][right], grid[low][right], true);
                }
                if (ring + 1 < rings) {  // Degenerate at the north pole.
                    addTriangle(grid[low][left], grid[high][left], grid[high][right], true);
                }
            }
        }
    }

    SceneDescription& scene_;
    Matrix3 rotation_;
    const Primitive* current_ = nullptr;
};

std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream stream(line);
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

void parseCamera(const std::vector<std::string>& tokens, std::size_t lineNumber, SceneCamera& camera) {
    const auto found = readKeyValues(tokens, lineNumber,
                                     {"pos", "yaw", "pitch", "focus", "fnumber", "lens", "sensor"});
    if (has(found, "pos")) {
        const Vec3d position = vectorOf(found, "pos");
        camera.position[0] = static_cast<float>(position.x);
        camera.position[1] = static_cast<float>(position.y);
        camera.position[2] = static_cast<float>(position.z);
    }
    if (has(found, "yaw")) camera.yawDegrees = static_cast<float>(scalarOf(found, "yaw"));
    if (has(found, "pitch")) camera.pitchDegrees = static_cast<float>(scalarOf(found, "pitch"));
    if (has(found, "focus")) camera.focusDistanceMeters = static_cast<float>(scalarOf(found, "focus"));
    if (has(found, "fnumber")) camera.fNumber = static_cast<float>(scalarOf(found, "fnumber"));
    if (has(found, "lens")) camera.focalLengthMillimeters = static_cast<float>(scalarOf(found, "lens"));
    if (has(found, "sensor")) camera.sensorHeightMillimeters = static_cast<float>(scalarOf(found, "sensor"));
    if (camera.focalLengthMillimeters <= 0.0f || camera.sensorHeightMillimeters <= 0.0f) {
        throw SceneParseError(lineTag(lineNumber) + "lens and sensor must be positive");
    }
    if (camera.fNumber <= 0.0f ||
        camera.focusDistanceMeters <= camera.focalLengthMillimeters * 0.001f) {
        throw SceneParseError(lineTag(lineNumber) +
                              "f-number must be positive and focus must exceed the focal length");
    }
}

void parseSun(const std::vector<std::string>& tokens, std::size_t lineNumber, SceneSun& sun) {
    const auto found = readKeyValues(tokens, lineNumber, {"dir", "color", "energy", "angle"});
    if (has(found, "dir")) {
        const Vec3d direction = vectorOf(found, "dir");
        if (length(direction) < 1e-6) {
            throw SceneParseError(lineTag(lineNumber) + "sun direction must not be zero");
        }
        sun.direction[0] = static_cast<float>(direction.x);
        sun.direction[1] = static_cast<float>(direction.y);
        sun.direction[2] = static_cast<float>(direction.z);
    }
    if (has(found, "color")) {
        const Vec3d color = vectorOf(found, "color");
        sun.color[0] = static_cast<float>(color.x);
        sun.color[1] = static_cast<float>(color.y);
        sun.color[2] = static_cast<float>(color.z);
    }
    if (has(found, "energy")) sun.energy = static_cast<float>(scalarOf(found, "energy"));
    if (has(found, "angle")) sun.angularDiameterDegrees = static_cast<float>(scalarOf(found, "angle"));
}

void parseAmbient(const std::vector<std::string>& tokens, std::size_t lineNumber, SceneAmbient& ambient) {
    const auto found = readKeyValues(tokens, lineNumber, {"color", "strength"});
    if (has(found, "color")) {
        const Vec3d color = vectorOf(found, "color");
        ambient.color[0] = static_cast<float>(color.x);
        ambient.color[1] = static_cast<float>(color.y);
        ambient.color[2] = static_cast<float>(color.z);
    }
    if (has(found, "strength")) ambient.strength = static_cast<float>(scalarOf(found, "strength"));
}

void parseShadow(const std::vector<std::string>& tokens, std::size_t lineNumber, SceneDescription& scene) {
    const auto found = readKeyValues(tokens, lineNumber, {"lo", "hi"});
    if (has(found, "lo") != has(found, "hi")) {
        throw SceneParseError(lineTag(lineNumber) + "shadow needs both 'lo' and 'hi'");
    }
    if (!has(found, "lo")) {
        return;
    }
    const Vec3d low = vectorOf(found, "lo");
    const Vec3d high = vectorOf(found, "hi");
    for (int axis = 0; axis < 3; ++axis) {
        if (high[axis] <= low[axis]) {
            throw SceneParseError(lineTag(lineNumber) + "shadow 'hi' must exceed 'lo' on every axis");
        }
    }
    scene.hasShadowRegion = true;
    scene.shadowLow = SceneVec3{static_cast<float>(low.x), static_cast<float>(low.y),
                                static_cast<float>(low.z)};
    scene.shadowHigh = SceneVec3{static_cast<float>(high.x), static_cast<float>(high.y),
                                 static_cast<float>(high.z)};
}

Primitive parsePrimitive(const std::string& kind, const std::vector<std::string>& tokens,
                         std::size_t lineNumber) {
    static const std::set<std::string> boxKeys = {"pos", "size", "rot", "rgb", "emit"};
    static const std::set<std::string> cylinderKeys = {"pos", "size", "rot", "rgb", "emit", "seg", "smooth"};
    static const std::set<std::string> sphereKeys = {"pos", "size", "rot", "rgb", "emit", "seg"};

    Primitive primitive;
    primitive.kind = kind;
    if (kind == "cyl") {
        primitive.segments = 16;
    } else if (kind == "sph") {
        primitive.segments = 12;
    }
    const auto& allowed = kind == "box" ? boxKeys : (kind == "cyl" ? cylinderKeys : sphereKeys);
    const auto found = readKeyValues(tokens, lineNumber, allowed);

    if (has(found, "pos")) primitive.pos = vectorOf(found, "pos");
    if (has(found, "size")) primitive.size = vectorOf(found, "size");
    if (has(found, "rot")) primitive.rot = vectorOf(found, "rot");
    if (has(found, "rgb")) primitive.rgb = vectorOf(found, "rgb");
    if (has(found, "emit")) primitive.emit = scalarOf(found, "emit");
    if (has(found, "seg")) primitive.segments = static_cast<int>(scalarOf(found, "seg"));
    if (has(found, "smooth")) primitive.smooth = scalarOf(found, "smooth") != 0.0;

    for (int axis = 0; axis < 3; ++axis) {
        if (std::fabs(primitive.size[axis]) < 1e-9) {
            throw SceneParseError(lineTag(lineNumber) + "size must not be zero on any axis");
        }
    }
    if (kind != "box" && (primitive.segments < 3 || primitive.segments > 128)) {
        throw SceneParseError(lineTag(lineNumber) + "seg must be between 3 and 128");
    }
    if (primitive.emit < 0.0) {
        throw SceneParseError(lineTag(lineNumber) + "emit must not be negative");
    }
    return primitive;
}

}  // namespace

bool parseSceneText(const std::string& text, SceneDescription& scene, std::string& error) {
    scene = SceneDescription{};
    error.clear();

    GeometryBuilder builder(scene);
    bool seenVersion = false;
    std::istringstream stream(text);
    std::string raw;
    std::size_t lineNumber = 0;

    try {
        while (std::getline(stream, raw)) {
            ++lineNumber;
            const std::size_t comment = raw.find('#');
            const std::string line = comment == std::string::npos ? raw : raw.substr(0, comment);
            std::vector<std::string> tokens = tokenize(line);
            if (tokens.empty()) {
                continue;
            }
            const std::string kind = tokens.front();
            tokens.erase(tokens.begin());

            if (kind == "version") {
                if (tokens.size() != 1) {
                    throw SceneParseError(lineTag(lineNumber) + "key 'version' needs 1 number(s)");
                }
                const double value = toNumber(tokens[0], lineNumber, "version");
                if (static_cast<int>(value) != 1) {
                    throw SceneParseError(lineTag(lineNumber) + "scene version " + tokens[0] + " is not 1");
                }
                seenVersion = true;
            } else if (kind == "camera") {
                parseCamera(tokens, lineNumber, scene.camera);
            } else if (kind == "sun") {
                parseSun(tokens, lineNumber, scene.sun);
            } else if (kind == "ambient") {
                parseAmbient(tokens, lineNumber, scene.ambient);
            } else if (kind == "shadow") {
                parseShadow(tokens, lineNumber, scene);
            } else if (kind == "box" || kind == "cyl" || kind == "sph") {
                // Emitted straight away: parse order is primitive order, which is
                // what keeps the vertex indices equal to the Python builder's.
                const Primitive primitive = parsePrimitive(kind, tokens, lineNumber);
                builder.emit(primitive);
                ++scene.primitiveCount;
            } else {
                throw SceneParseError(lineTag(lineNumber) + "unknown entry '" + kind + "'");
            }
        }
        if (!seenVersion) {
            throw SceneParseError("missing 'version' line; refusing to guess the scene format");
        }
    } catch (const SceneParseError& failure) {
        error = failure.what();
        scene = SceneDescription{};
        return false;
    }

    if (scene.vertices.empty()) {
        error = "scene file contains no geometry";
        scene = SceneDescription{};
        return false;
    }
    if (scene.vertices.size() > 0xFFFFFFFFull) {
        error = "scene file needs more vertices than a 32-bit index can address";
        scene = SceneDescription{};
        return false;
    }
    return true;
}

bool loadSceneFile(const std::filesystem::path& path, SceneDescription& scene, std::string& error) {
    scene = SceneDescription{};
    error.clear();

    std::ifstream file(path);
    if (!file) {
        error = "could not open scene file: " + path.string();
        return false;
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    if (!parseSceneText(contents.str(), scene, error)) {
        error = path.filename().string() + ": " + error;
        return false;
    }
    return true;
}
