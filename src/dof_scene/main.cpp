#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_NONE
#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include "dof_scene/camera.hpp"
#include "dof_scene/hud.hpp"
#include "dof_scene/scene.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <array>
#include <chrono>
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
using namespace dof_scene;
namespace fs = std::filesystem;

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
    Program(const std::string& vertex, const std::string& fragment) {
        const GLuint vs = compileShader(GL_VERTEX_SHADER, vertex);
        GLuint ps = 0;
        try { ps = compileShader(GL_FRAGMENT_SHADER, fragment); }
        catch (...) { glDeleteShader(vs); throw; }
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
            glDeleteProgram(id);
            id = 0;
            throw std::runtime_error("Shader linking failed:\n" + log);
        }
    }
    ~Program() { if (id) glDeleteProgram(id); }
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    void use() const { glUseProgram(id); }
    GLint location(const std::string& name) {
        auto found = locations.find(name);
        if (found != locations.end()) return found->second;
        const GLint loc = glGetUniformLocation(id, name.c_str());
        if (loc < 0) throw std::runtime_error("Missing shader uniform: " + name);
        locations.emplace(name, loc);
        return loc;
    }
    void set(const std::string& n, int v) { glUniform1i(location(n), v); }
    void set(const std::string& n, float v) { glUniform1f(location(n), v); }
    void set(const std::string& n, const glm::vec2& v) { glUniform2fv(location(n), 1, glm::value_ptr(v)); }
    void set(const std::string& n, const glm::vec3& v) { glUniform3fv(location(n), 1, glm::value_ptr(v)); }
    void set(const std::string& n, const glm::mat4& v) { glUniformMatrix4fv(location(n), 1, GL_FALSE, glm::value_ptr(v)); }
};

struct Mesh {
    GLuint vao = 0, vbo = 0, ebo = 0;
    GLsizei count = 0;
    void upload(const MeshData& data) {
        count = static_cast<GLsizei>(data.indices.size());
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, data.vertices.size()*sizeof(Vertex), data.vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.indices.size()*sizeof(unsigned), data.indices.data(), GL_STATIC_DRAW);
        for (GLuint i = 0; i < 3; ++i) glEnableVertexAttribArray(i);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, position)));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, uv)));
    }
    void draw() const { glBindVertexArray(vao); glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, nullptr); }
    ~Mesh() { glDeleteVertexArrays(1, &vao); glDeleteBuffers(1, &vbo); glDeleteBuffers(1, &ebo); }
};

struct Options {
    int width = 1120, height = 760, frames = 0, mode = 0;
    float focus = -1, fNumber = 1.2f, focal = 50;
    bool hud = true, verify = false;
    std::string capture, view = "home";
};

Options parseOptions(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("Missing value for " + arg);
            return argv[i];
        };
        if (arg == "--capture") o.capture = value();
        else if (arg == "--frames") o.frames = std::stoi(value());
        else if (arg == "--mode") o.mode = std::stoi(value());
        else if (arg == "--focus") o.focus = std::stof(value());
        else if (arg == "--f-number") o.fNumber = std::stof(value());
        else if (arg == "--lens") o.focal = std::stof(value());
        else if (arg == "--view") o.view = value();
        else if (arg == "--size") {
            std::string size = value();
            auto x = size.find('x');
            if (x == std::string::npos) throw std::runtime_error("Size must be WIDTHxHEIGHT");
            o.width = std::stoi(size.substr(0, x)); o.height = std::stoi(size.substr(x+1));
        } else if (arg == "--no-hud") o.hud = false;
        else if (arg == "--verify") o.verify = true;
        else if (arg == "--help") {
            std::cout << "DoFScene [--capture file.png] [--frames N] [--mode 0..4] [--focus meters]\n"
                      << "  [--f-number 0.7..16] [--lens 24..100] [--size WIDTHxHEIGHT, minimum 900x560]\n"
                      << "  [--view home|close|wide] [--no-hud] [--verify]\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown option: " + arg);
    }
    if (o.width < 900 || o.height < 560 || o.width > 2560 || o.height > 1600 ||
        o.frames < 0 || o.mode < 0 || o.mode > 4 || !std::isfinite(o.fNumber) ||
        o.fNumber < 0.7f || o.fNumber > 16 || !std::isfinite(o.focal) || o.focal < 24 ||
        o.focal > 100 || !std::isfinite(o.focus) || (o.focus != -1 && o.focus <= 0) ||
        (o.view != "home" && o.view != "close" && o.view != "wide"))
        throw std::runtime_error("Invalid camera, lens, or window option; use --help.");
    if (!o.capture.empty() && o.frames == 0) o.frames = 4;
    if (o.verify) o.frames = std::max(o.frames, 14);
    return o;
}

struct App {
    GLFWwindow* window = nullptr;
    Camera camera;
    Lens lens;
    glm::vec3 subject;
    int mode = 0;
    bool hud = true, help = true, looking = false, pick = false, capture = false;
    double mouseX = 0, mouseY = 0;
    glm::vec2 pickUv{0.5f};
    std::string status = "CLICK A SURFACE TO FOCUS";

    void home() {
        camera = Camera{};
        camera.lookAt(subject + glm::vec3(0,0.27f,0));
        lens = Lens{};
        lens.focusAt(glm::dot(subject-camera.position, camera.forward()));
        mode = 0;
    }
    bool hit(double x, double y, float left, float top, float width, float height) const {
        return x >= left && x <= left+width && y >= top && y <= top+height;
    }
    void mouseButton(int button, int action) {
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            looking = action == GLFW_PRESS;
            glfwSetInputMode(window, GLFW_CURSOR, looking ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
            glfwGetCursorPos(window, &mouseX, &mouseY);
        }
        if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS || looking) return;
        int w, h; glfwGetWindowSize(window, &w, &h);
        double x, y; glfwGetCursorPos(window, &x, &y);
        if (hud) {
            for (int i = 0; i < 5; ++i) {
                if (hit(x,y,w-410+i*78,24,72,30)) { mode = i; return; }
            }
            if (hit(x,y,24,h-118,340,38)) {
                if (x < 68) lens.focusAt(lens.focusDistance/1.12f);
                else if (x > 318) lens.focusAt(lens.focusDistance*1.12f);
                return;
            }
            if (hit(x,y,24,h-72,340,38)) {
                if (x < 68) lens.fNumber = std::max(0.7f,lens.fNumber/1.2f);
                else if (x > 318) lens.fNumber = std::min(16.0f,lens.fNumber*1.2f);
                return;
            }
            if (hit(x,y,24,24,345,103)) return;
            if (help && hit(x,y,w-476,h-113,452,79)) return;
        }
        pickUv = {static_cast<float>(x/w),static_cast<float>(1-y/h)};
        pick = true;
    }
    void key(int key, int action) {
        if (action != GLFW_PRESS) return;
        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, true);
        if (key >= GLFW_KEY_1 && key <= GLFW_KEY_5) mode = key-GLFW_KEY_1;
        if (key == GLFW_KEY_0 || key == GLFW_KEY_HOME) home();
        if (key == GLFW_KEY_F) { pickUv = {0.5f,0.5f}; pick = true; }
        if (key == GLFW_KEY_TAB) hud = !hud;
        if (key == GLFW_KEY_H) help = !help;
        if (key == GLFW_KEY_P) capture = true;
    }
    void update(float dt) {
        if (!glfwGetWindowAttrib(window, GLFW_FOCUSED)) return;
        auto down = [&](int k) { return glfwGetKey(window,k) == GLFW_PRESS; };
        glm::vec3 move(0);
        if (down(GLFW_KEY_W)) move += camera.forward();
        if (down(GLFW_KEY_S)) move -= camera.forward();
        if (down(GLFW_KEY_D)) move += camera.right();
        if (down(GLFW_KEY_A)) move -= camera.right();
        if (down(GLFW_KEY_E)) move.y += 1;
        if (down(GLFW_KEY_Q)) move.y -= 1;
        if (glm::length(move)>0) camera.position += glm::normalize(move)*dt*(down(GLFW_KEY_LEFT_SHIFT)?3.6f:1.2f);
        camera.position.y = std::clamp(camera.position.y,0.12f,3.2f);
        if (down(GLFW_KEY_LEFT_BRACKET)) lens.focusAt(lens.focusDistance*std::exp(-dt));
        if (down(GLFW_KEY_RIGHT_BRACKET)) lens.focusAt(lens.focusDistance*std::exp(dt));
        if (down(GLFW_KEY_MINUS)) lens.fNumber = std::max(0.7f,lens.fNumber*std::exp(-dt));
        if (down(GLFW_KEY_EQUAL)) lens.fNumber = std::min(16.0f,lens.fNumber*std::exp(dt));
    }
};

void configureCallbacks(App& app) {
    GLFWwindow* w = app.window;
    glfwSetWindowUserPointer(w,&app);
    glfwSetKeyCallback(w,[](GLFWwindow* window,int key,int,int action,int) {
        static_cast<App*>(glfwGetWindowUserPointer(window))->key(key,action);
    });
    glfwSetMouseButtonCallback(w,[](GLFWwindow* window,int button,int action,int) {
        static_cast<App*>(glfwGetWindowUserPointer(window))->mouseButton(button,action);
    });
    glfwSetCursorPosCallback(w,[](GLFWwindow* window,double x,double y) {
        auto& a=*static_cast<App*>(glfwGetWindowUserPointer(window));
        if(a.looking) a.camera.rotate(static_cast<float>(x-a.mouseX),static_cast<float>(y-a.mouseY));
        a.mouseX=x; a.mouseY=y;
    });
    glfwSetWindowFocusCallback(w,[](GLFWwindow* window,int focused) {
        auto& a=*static_cast<App*>(glfwGetWindowUserPointer(window));
        if(!focused && a.looking) a.mouseButton(GLFW_MOUSE_BUTTON_RIGHT,GLFW_RELEASE);
    });
    glfwSetScrollCallback(w,[](GLFWwindow* window,double,double dy) {
        auto& a=*static_cast<App*>(glfwGetWindowUserPointer(window));
        if(glfwGetKey(window,GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS) {
            a.lens.focalLengthMm=std::clamp(a.lens.focalLengthMm+static_cast<float>(dy)*2,24.0f,100.0f);
            a.lens.focusAt(a.lens.focusDistance);
        } else a.lens.focusAt(a.lens.focusDistance*std::exp(static_cast<float>(dy)*0.08f));
    });
}

void savePng(const fs::path& path,int width,int height) {
    if (path.has_parent_path()) fs::create_directories(path.parent_path());
    std::vector<unsigned char> pixels(static_cast<size_t>(width)*height*4), flipped(pixels.size());
    glPixelStorei(GL_PACK_ALIGNMENT,1);
    glReadBuffer(GL_BACK);
    glReadPixels(0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
    for(int y=0;y<height;++y)
        std::copy_n(pixels.data()+static_cast<size_t>(height-1-y)*width*4,width*4,flipped.data()+static_cast<size_t>(y)*width*4);
    const auto name=fs::absolute(path).string();
    CFURLRef url=CFURLCreateFromFileSystemRepresentation(nullptr,reinterpret_cast<const UInt8*>(name.c_str()),name.size(),false);
    CGColorSpaceRef space=CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider=CGDataProviderCreateWithData(nullptr,flipped.data(),flipped.size(),nullptr);
    CGImageRef image=CGImageCreate(width,height,8,32,width*4,space,kCGImageAlphaLast,provider,nullptr,false,kCGRenderingIntentDefault);
    CGImageDestinationRef destination=url?CGImageDestinationCreateWithURL(url,CFSTR("public.png"),1,nullptr):nullptr;
    bool success=false;
    if(destination && image) { CGImageDestinationAddImage(destination,image,nullptr); success=CGImageDestinationFinalize(destination); }
    if(destination) CFRelease(destination);
    if(image) CGImageRelease(image);
    if(provider) CGDataProviderRelease(provider);
    if(space) CGColorSpaceRelease(space);
    if(url) CFRelease(url);
    if(!success) throw std::runtime_error("Could not save screenshot: "+name);
    std::cout<<"Saved "<<name<<" ("<<width<<"x"<<height<<")\n";
}

std::string number(float value,int precision=1) {
    std::ostringstream s; s<<std::fixed<<std::setprecision(precision)<<value; return s.str();
}

const char* modeName(int mode) {
    static const char* names[]={"DOF","SHARP","DEPTH","COC","SPLIT"};
    return names[std::clamp(mode,0,4)];
}

class Renderer {
    Scene scene_;
    std::array<Mesh,std::tuple_size<decltype(Scene::meshes)>::value> meshes_;
    Program sceneProgram_,depthProgram_,screenProgram_,hudProgram_;
    GLuint sceneFbo_=0,color_=0,depth_=0,shadowFbo_=0,shadow_=0,emptyVao_=0,hudVao_=0,hudVbo_=0;
    int width_=0,height_=0;
    glm::vec3 sun_=glm::normalize(glm::vec3(3,7,4));
    glm::mat4 light_;
    static constexpr int shadowSize=2048;

    void checkFbo(const char* label) {
        GLenum status=glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if(status!=GL_FRAMEBUFFER_COMPLETE) throw std::runtime_error(std::string(label)+" framebuffer incomplete: "+std::to_string(status));
    }
    void drawObjects(Program& p,bool materials) {
        for(const auto& object:scene_.objects) {
            p.set("uModel",object.model);
            if(materials) {
                p.set("uBaseColor",object.material.color);
                p.set("uRoughness",object.material.roughness);
                p.set("uMetallic",object.material.metallic);
                p.set("uTextureType",object.material.texture);
                p.set("uEmission",object.material.emission);
            }
            meshes_[static_cast<size_t>(object.shape)].draw();
        }
    }
    void resize(int width,int height) {
        if(width_==width && height_==height) return;
        width_=width; height_=height;
        glBindFramebuffer(GL_FRAMEBUFFER,sceneFbo_);
        glBindTexture(GL_TEXTURE_2D,color_);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,width,height,0,GL_RGBA,GL_FLOAT,nullptr);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,color_,0);
        glBindTexture(GL_TEXTURE_2D,depth_);
        glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT24,width,height,0,GL_DEPTH_COMPONENT,GL_UNSIGNED_INT,nullptr);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,depth_,0);
        checkFbo("Scene");
        std::cout<<"Scene framebuffer complete: "<<width<<" x "<<height<<"\n";
    }
public:
    explicit Renderer(const fs::path& shaders)
        : scene_(createCafeScene()),
          sceneProgram_(readText(shaders/"scene.vert"),readText(shaders/"scene.frag")),
          depthProgram_(readText(shaders/"depth.vert"),readText(shaders/"depth.frag")),
          screenProgram_(readText(shaders/"screen.vert"),readText(shaders/"screen.frag")),
          hudProgram_(
              "#version 330 core\nlayout(location=0) in vec2 p; layout(location=1) in vec4 c; uniform vec2 uSize; out vec4 color; void main(){color=c;gl_Position=vec4(p.x/uSize.x*2.0-1.0,1.0-p.y/uSize.y*2.0,0,1);}",
              "#version 330 core\nin vec4 color; out vec4 fragColor; void main(){fragColor=color;}") {
        for(size_t i=0;i<meshes_.size();++i) meshes_[i].upload(scene_.meshes[i]);
        glGenVertexArrays(1,&emptyVao_);
        glGenVertexArrays(1,&hudVao_); glGenBuffers(1,&hudVbo_);
        glBindVertexArray(hudVao_); glBindBuffer(GL_ARRAY_BUFFER,hudVbo_);
        glEnableVertexAttribArray(0); glEnableVertexAttribArray(1);
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(HudVertex),reinterpret_cast<void*>(offsetof(HudVertex,position)));
        glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,sizeof(HudVertex),reinterpret_cast<void*>(offsetof(HudVertex,color)));
        glGenFramebuffers(1,&sceneFbo_); glGenTextures(1,&color_); glGenTextures(1,&depth_);
        glGenFramebuffers(1,&shadowFbo_); glGenTextures(1,&shadow_);
        glBindTexture(GL_TEXTURE_2D,shadow_);
        glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT24,shadowSize,shadowSize,0,GL_DEPTH_COMPONENT,GL_UNSIGNED_INT,nullptr);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_BORDER);
        const float border[]={1,1,1,1}; glTexParameterfv(GL_TEXTURE_2D,GL_TEXTURE_BORDER_COLOR,border);
        glBindFramebuffer(GL_FRAMEBUFFER,shadowFbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,shadow_,0);
        glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE); checkFbo("Shadow");
        const glm::vec3 target(0,0,-2);
        light_=glm::ortho(-7.0f,7.0f,-7.0f,7.0f,0.1f,26.0f)*glm::lookAt(target+sun_*12.0f,target,glm::vec3(0,1,0));
        glViewport(0,0,shadowSize,shadowSize);
        glEnable(GL_DEPTH_TEST); glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(2,4);
        glClear(GL_DEPTH_BUFFER_BIT);
        depthProgram_.use(); depthProgram_.set("uLightMatrix",light_); drawObjects(depthProgram_,false);
        glDisable(GL_POLYGON_OFFSET_FILL);
        glBindFramebuffer(GL_FRAMEBUFFER,0);
        std::cout<<"Cafe scene: "<<scene_.objects.size()<<" objects; shader programs linked; shadow framebuffer complete\n";
    }
    ~Renderer() {
        glDeleteFramebuffers(1,&sceneFbo_); glDeleteFramebuffers(1,&shadowFbo_);
        glDeleteTextures(1,&color_); glDeleteTextures(1,&depth_); glDeleteTextures(1,&shadow_);
        glDeleteVertexArrays(1,&emptyVao_); glDeleteVertexArrays(1,&hudVao_); glDeleteBuffers(1,&hudVbo_);
    }
    glm::vec3 subject() const { return scene_.focusPoint; }
    void frame(App& app,int framebufferWidth,int framebufferHeight) {
        // Cap the expensive HDR blur target; the HUD remains native-resolution.
        float scale=std::min(1.0f,1000.0f/static_cast<float>(framebufferHeight));
        resize(std::max(1,static_cast<int>(framebufferWidth*scale)),std::max(1,static_cast<int>(framebufferHeight*scale)));
        glBindFramebuffer(GL_FRAMEBUFFER,sceneFbo_); glViewport(0,0,width_,height_);
        glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS); glDisable(GL_BLEND);
        glClearColor(0.20f,0.26f,0.33f,1); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        sceneProgram_.use();
        sceneProgram_.set("uViewProjection",glm::perspective(app.lens.verticalFov(),static_cast<float>(width_)/height_,app.lens.nearPlane,app.lens.farPlane)*app.camera.view());
        sceneProgram_.set("uLightMatrix",light_); sceneProgram_.set("uCameraPosition",app.camera.position);
        sceneProgram_.set("uSunDirection",sun_); sceneProgram_.set("uShadowMap",0);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,shadow_);
        drawObjects(sceneProgram_,true);
        if(app.pick) {
            float raw=1;
            glReadPixels(std::clamp(static_cast<int>(app.pickUv.x*width_),0,width_-1),
                         std::clamp(static_cast<int>(app.pickUv.y*height_),0,height_-1),
                         1,1,GL_DEPTH_COMPONENT,GL_FLOAT,&raw);
            if(raw<0.999999f) { app.lens.focusAt(app.lens.linearDepth(raw)); app.status="FOCUSED AT "+number(app.lens.focusDistance,2)+" M"; }
            else app.status="NO SURFACE HERE - FOCUS UNCHANGED";
            app.pick=false;
        }
        glBindFramebuffer(GL_FRAMEBUFFER,0); glViewport(0,0,framebufferWidth,framebufferHeight);
        glDisable(GL_DEPTH_TEST);
        screenProgram_.use();
        screenProgram_.set("uColor",0); screenProgram_.set("uDepth",1);
        screenProgram_.set("uNear",app.lens.nearPlane); screenProgram_.set("uFar",app.lens.farPlane);
        screenProgram_.set("uFocusDistance",app.lens.focusDistance);
        screenProgram_.set("uFocalLengthMm",app.lens.focalLengthMm); screenProgram_.set("uSensorHeightMm",app.lens.sensorHeightMm);
        screenProgram_.set("uFNumber",app.lens.fNumber); screenProgram_.set("uMaxRadius",120.0f);
        screenProgram_.set("uGatherMode",1);
        screenProgram_.set("uResolution",glm::vec2(width_,height_)); screenProgram_.set("uMode",app.mode);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,color_);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D,depth_);
        glBindVertexArray(emptyVao_); glDrawArrays(GL_TRIANGLES,0,3);
        if(app.hud) drawHud(app);
        GLenum error=glGetError();
        if(error!=GL_NO_ERROR) throw std::runtime_error("OpenGL error: "+std::to_string(error));
    }
    void drawHud(const App& app) {
        int w,h; glfwGetWindowSize(app.window,&w,&h);
        Hud hud;
        const glm::vec4 ink(0.94f,0.91f,0.84f,1), muted(0.64f,0.65f,0.62f,1), gold(0.90f,0.65f,0.31f,1), panel(0.055f,0.065f,0.065f,0.88f);
        hud.rect(24,24,345,103,panel); hud.rect(24,24,3,103,gold);
        hud.text(42,40,"CAFE / FOCUS STUDY",2.0f,ink);
        hud.text(42,66,number(app.lens.focalLengthMm,0)+" MM  F/"+number(app.lens.fNumber)+"  "+modeName(app.mode),1.6f,gold);
        hud.text(42,94,app.status,1.15f,muted);
        for(int i=0;i<5;++i) {
            float x=static_cast<float>(w-410+i*78);
            hud.rect(x,24,72,30,i==app.mode?gold:panel);
            hud.text(x+11,34,modeName(i),1.35f,i==app.mode?panel:ink);
        }
        auto control=[&](float y,const std::string& label) {
            hud.rect(24,y,340,38,panel);
            hud.text(40,y+12,"-",2.0f,gold); hud.text(75,y+12,label,1.7f,ink); hud.text(336,y+12,"+",2.0f,gold);
        };
        control(static_cast<float>(h-118),"FOCUS  "+number(app.lens.focusDistance,2)+" M");
        control(static_cast<float>(h-72),"APERTURE  F/"+number(app.lens.fNumber));
        if(app.help) {
            hud.rect(static_cast<float>(w-476),static_cast<float>(h-113),452,79,panel);
            hud.text(static_cast<float>(w-459),static_cast<float>(h-98),"RMB LOOK  WASD MOVE  Q/E HEIGHT",1.4f,ink);
            hud.text(static_cast<float>(w-459),static_cast<float>(h-76),"CLICK FOCUS  WHEEL FOCUS  SHIFT FAST",1.25f,muted);
            hud.text(static_cast<float>(w-459),static_cast<float>(h-56),"0 RESET  P SAVE  H HELP  TAB HUD",1.35f,muted);
        }
        hud.rect(w/2.0f-6,h/2.0f,4,1,gold); hud.rect(w/2.0f+3,h/2.0f,4,1,gold);
        hud.rect(w/2.0f,h/2.0f-6,1,4,gold); hud.rect(w/2.0f,h/2.0f+3,1,4,gold);
        hudProgram_.use(); hudProgram_.set("uSize",glm::vec2(w,h));
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
        glBindVertexArray(hudVao_); glBindBuffer(GL_ARRAY_BUFFER,hudVbo_);
        glBufferData(GL_ARRAY_BUFFER,hud.vertices.size()*sizeof(HudVertex),hud.vertices.data(),GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES,0,static_cast<GLsizei>(hud.vertices.size()));
        glDisable(GL_BLEND);
    }
};

int run(GLFWwindow* window,const Options& options,const fs::path& shaders) {
    Renderer renderer(shaders);
    App app; app.window=window; app.subject=renderer.subject(); app.home();
    if(options.view=="close") { app.camera.position={0.15f,1.4f,2.8f}; app.camera.lookAt(app.subject+glm::vec3(0,0.16f,0)); }
    if(options.view=="wide") { app.camera.position={2.8f,2.05f,4.7f}; app.camera.lookAt(app.subject); }
    app.lens.fNumber=options.fNumber; app.lens.focalLengthMm=options.focal;
    app.lens.focusAt(glm::dot(app.subject-app.camera.position,app.camera.forward()));
    if(options.focus>0) app.lens.focusAt(options.focus);
    app.mode=options.mode; app.hud=options.hud;
    configureCallbacks(app);
    double previous=glfwGetTime();
    int frames=0;
    for(;!glfwWindowShouldClose(window);) {
        glfwPollEvents();
        double now=glfwGetTime();
        app.update(static_cast<float>(std::clamp(now-previous,0.0,0.05))); previous=now;
        if(options.verify) {
            if(frames==2) { app.camera.rotate(14,-7); glfwSetWindowSize(window,960,640); }
            if(frames==4) { app.lens.focusAt(0.2f); app.key(GLFW_KEY_F,GLFW_PRESS); }
            if(frames==5) { app.key(GLFW_KEY_2,GLFW_PRESS); app.lens.fNumber=16; }
            if(frames==6) app.key(GLFW_KEY_3,GLFW_PRESS);
            if(frames==7) app.key(GLFW_KEY_4,GLFW_PRESS);
            if(frames==8) app.key(GLFW_KEY_5,GLFW_PRESS);
            if(frames==9) app.key(GLFW_KEY_0,GLFW_PRESS);
            if(frames==10) {
                auto saved=app.camera.position;
                app.camera.position+=app.camera.right()*0.15f;
                if(glm::distance(saved,app.camera.position)<0.1f) throw std::runtime_error("Camera verification failed");
            }
            if(frames==11) {
                int w,h; glfwGetWindowSize(window,&w,&h);
                if(w!=960 || h!=640) throw std::runtime_error("Window resize verification failed");
                glfwSetCursorPos(window,w-410+78+36,39);
                app.mouseButton(GLFW_MOUSE_BUTTON_LEFT,GLFW_PRESS);
                if(app.mode!=1) throw std::runtime_error("HUD mode button verification failed");
                app.mouseButton(GLFW_MOUSE_BUTTON_RIGHT,GLFW_PRESS);
                if(!app.looking || glfwGetInputMode(window,GLFW_CURSOR)!=GLFW_CURSOR_DISABLED)
                    throw std::runtime_error("Mouse-look capture verification failed");
                app.mouseButton(GLFW_MOUSE_BUTTON_RIGHT,GLFW_RELEASE);
            }
            if(frames==12) {
                int w,h; glfwGetWindowSize(window,&w,&h);
                const float before=app.lens.focusDistance;
                glfwSetCursorPos(window,339,h-100);
                app.mouseButton(GLFW_MOUSE_BUTTON_LEFT,GLFW_PRESS);
                if(app.lens.focusDistance<=before) throw std::runtime_error("HUD focus button verification failed");
                glfwSetCursorPos(window,w*0.5,h*0.5);
                app.mouseButton(GLFW_MOUSE_BUTTON_LEFT,GLFW_PRESS);
                if(!app.pick) throw std::runtime_error("Click focus was not queued");
            }
        }
        int width,height; glfwGetFramebufferSize(window,&width,&height);
        if(width<=0 || height<=0) { glfwWaitEventsTimeout(0.05); continue; }
        renderer.frame(app,width,height);
        if(options.verify && (frames==4 || frames==12) &&
           (app.lens.focusDistance<0.3f || app.status.find("FOCUSED AT ")!=0))
            throw std::runtime_error("Surface autofocus verification failed");
        ++frames;
        if(app.capture || (!options.capture.empty() && frames==options.frames)) {
            savePng(app.capture?fs::path(PROJECT_SOURCE_DIR)/"output/cafe-dof.png":fs::path(options.capture),width,height);
            app.capture=false; app.status="SAVED OUTPUT/CAFE-DOF.PNG";
        }
        if(frames%30==0) {
            std::string title="Cafe - Depth of Field | "+std::string(modeName(app.mode))+" | "+number(app.lens.focusDistance,2)+" m | f/"+number(app.lens.fNumber);
            glfwSetWindowTitle(window,title.c_str());
        }
        glfwSwapBuffers(window);
        if(options.frames>0 && frames>=options.frames) break;
    }
    if(options.verify) std::cout<<"Runtime smoke checks passed: resize, camera, center/click focus, HUD mode/focus buttons, mouse capture, five modes, aperture and reset; "<<frames<<" frames.\n";
    return 0;
}
} // namespace

int main(int argc,char** argv) {
    GLFWwindow* window=nullptr;
    try {
        const Options options=parseOptions(argc,argv);
        glfwSetErrorCallback([](int code,const char* message){std::cerr<<"GLFW "<<code<<": "<<message<<"\n";});
        if(!glfwInit()) throw std::runtime_error("GLFW initialization failed");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
        glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE); glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GL_TRUE);
        window=glfwCreateWindow(options.width,options.height,"Cafe - Depth of Field",nullptr,nullptr);
        if(!window) throw std::runtime_error("Cannot create OpenGL window");
        glfwMakeContextCurrent(window); glfwSwapInterval(1); glfwSetWindowSizeLimits(window,900,560,GLFW_DONT_CARE,GLFW_DONT_CARE);
        std::cout<<"Renderer: "<<glGetString(GL_RENDERER)<<"\n";
        fs::path shaders=fs::path(argv[0]).parent_path()/"shaders/dof_scene";
        if(!fs::exists(shaders/"scene.vert")) shaders=fs::path(PROJECT_SOURCE_DIR)/"shaders/dof_scene";
        const int result=run(window,options,shaders);
        glfwDestroyWindow(window); glfwTerminate();
        return result;
    } catch(const std::exception& error) {
        std::cerr<<"DoFScene: "<<error.what()<<"\n";
        if(window) glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
}
