#include <glad/glad.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// DEVELOPMENT SETTINGS
const int windowWidth = 600;
const int windowHeight = 600;
const int windowPosX = 50;
const int windowPosY = 100;

// ==============================
// EXPERIMENT 02 CONTROLS
// ==============================
float rectangleScale = 1.0f;

float rectangleOffsetX = 0.0f;
float rectangleOffsetY = 0.0f;

float topRightColor[3] = {1.0f, 0.0f, 0.0f};
float bottomRightColor[3] = {0.0f, 1.0f, 0.0f};
float bottomLeftColor[3] = {0.0f, 0.0f, 1.0f};
float topLeftColor[3] = {1.0f, 1.0f, 0.0f};

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

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
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

int main() {
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

    if (!gladLoadGLLoader((GLADloadfunc)glfwGetProcAddress)) {
        cout << "Failed to initialize GLAD" << endl;
        glfwTerminate();
        return -1;
    }

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    glViewport(0, 0, framebufferWidth, framebufferHeight);

    // basic.vert shader
    string vertexCode = readFile(string(PROJECT_SOURCE_DIR) + "/shaders/basic.vert");
    const char* vertexShaderSource = vertexCode.c_str();
    
    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);

    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    if (!checkShaderCompilation(vertexShader, "basic.vert")) {
        glDeleteShader(vertexShader);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }
    
    // basic.frag shader
    string fragmentCode = readFile(string(PROJECT_SOURCE_DIR) + "/shaders/basic.frag");
    const char* fragmentShaderSource = fragmentCode.c_str();

    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    if (!checkShaderCompilation(fragmentShader, "basic.frag")) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    //Shader Program
    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    if (!checkProgramLinking(shaderProgram)) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        glDeleteProgram(shaderProgram);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // glUseProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // The vertex array stores four unique vertices.
    // Each vertex has six floats: position.xyz followed by color.rgb.
    float vertices[] = {
        (0.5f * rectangleScale) + rectangleOffsetX, (0.5f * rectangleScale) + rectangleOffsetY, 0.0f,
        topRightColor[0], topRightColor[1], topRightColor[2],

        (0.5f * rectangleScale) + rectangleOffsetX, (-0.5f * rectangleScale) + rectangleOffsetY, 0.0f,
        bottomRightColor[0], bottomRightColor[1], bottomRightColor[2],

        (-0.5f * rectangleScale) + rectangleOffsetX, (-0.5f * rectangleScale) + rectangleOffsetY, 0.0f,
        bottomLeftColor[0], bottomLeftColor[1], bottomLeftColor[2],

        (-0.5f * rectangleScale) + rectangleOffsetX, (0.5f * rectangleScale) + rectangleOffsetY, 0.0f,
        topLeftColor[0], topLeftColor[1], topLeftColor[2]
    };

    // The index array describes which vertices make up each triangle.
    // Vertices 1 and 3 are reused by both triangles.
    unsigned int indices[] = {
        0, 1, 3,
        1, 2, 3
    };

    // Without indices, glDrawArrays() would need duplicate vertices:
    // top-right, bottom-right, top-left, bottom-right, bottom-left, top-left.

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
    // Stride is the byte distance from one vertex to the next: 6 floats here.
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);  

    // Color starts after the first three floats because position takes x, y, z.
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // 2. use our shader program when we want to render an object
    // glUseProgram(shaderProgram);
    // 3. now draw the object 
    // someOpenGLFunctionThatDrawsOurRectangle();   
    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glBindVertexArray(0); 

    if (wireframeMode) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    bool screenshotKeyWasPressed = false;

    while (!glfwWindowShouldClose(window)) {
        //input
        processInput(window);

        //rendering
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // draw rectangle
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        const bool screenshotKeyIsPressed = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
        if (screenshotKeyIsPressed && !screenshotKeyWasPressed) {
            glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
            saveScreenshot("output/latest.png", framebufferWidth, framebufferHeight);
        }
        screenshotKeyWasPressed = screenshotKeyIsPressed;

        //check and call events and swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents(); //check updates

    }

    //Delete everything
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}
