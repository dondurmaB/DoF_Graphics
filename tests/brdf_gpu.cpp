// Compile the PRODUCTION GLSL function, read linear float pixels (no tone mapping).
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <OpenGL/gl3.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
GLuint shader(GLenum type,const std::string& source) {
    GLuint s=glCreateShader(type);const char* p=source.c_str();glShaderSource(s,1,&p,nullptr);glCompileShader(s);
    GLint ok;glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok) {char log[4096];glGetShaderInfoLog(s,4096,nullptr,log);throw std::runtime_error(log);}return s;
}
int main() {try {
    if(!glfwInit()) return 77;
    glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GL_TRUE);
    auto w=glfwCreateWindow(16,16,"BRDF verification",nullptr,nullptr);if(!w){glfwTerminate();return 77;}
    glfwMakeContextCurrent(w);
    std::ifstream input(std::string(PROJECT_SOURCE_DIR)+"/shaders/basic.frag");
    std::string source((std::istreambuf_iterator<char>(input)),{});
    auto a=source.find("// BEGIN MATCHED_SPECULAR"),b=source.find("// END MATCHED_SPECULAR");
    if(a==std::string::npos||b==std::string::npos)throw std::runtime_error("Production BRDF markers missing");
    std::string frag="#version 330 core\nuniform vec3 L,V;uniform float R,F;out vec4 result;\n"+source.substr(a,b-a)+"\nvoid main(){result=vec4(matchedSpecular(vec3(0,0,1),L,V,R,F),0,0,1);}";
    auto vs=shader(GL_VERTEX_SHADER,"#version 330 core\nvoid main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2-1,0,1);}");
    auto fs=shader(GL_FRAGMENT_SHADER,frag);auto program=glCreateProgram();glAttachShader(program,vs);glAttachShader(program,fs);glLinkProgram(program);
    GLint ok;glGetProgramiv(program,GL_LINK_STATUS,&ok);if(!ok)throw std::runtime_error("BRDF link failed");
    GLuint vao,tex,fbo;glGenVertexArrays(1,&vao);glBindVertexArray(vao);
    glGenTextures(1,&tex);glBindTexture(GL_TEXTURE_2D,tex);glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,1,1,0,GL_RGBA,GL_FLOAT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glGenFramebuffers(1,&fbo);glBindFramebuffer(GL_FRAMEBUFFER,fbo);glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,0);
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)throw std::runtime_error("BRDF FBO incomplete");
    glViewport(0,0,1,1);glUseProgram(program);
    float lx,ly,lz,vx,vy,vz,r,f;
    while(std::cin>>lx>>ly>>lz>>vx>>vy>>vz>>r>>f) {
        glUniform3f(glGetUniformLocation(program,"L"),lx,ly,lz);glUniform3f(glGetUniformLocation(program,"V"),vx,vy,vz);
        glUniform1f(glGetUniformLocation(program,"R"),r);glUniform1f(glGetUniformLocation(program,"F"),f);
        glDrawArrays(GL_TRIANGLES,0,3);float pixel[4];glReadPixels(0,0,1,1,GL_RGBA,GL_FLOAT,pixel);
        std::cout<<std::setprecision(10)<<pixel[0]<<'\n';
    }
    glDeleteFramebuffers(1,&fbo);glDeleteTextures(1,&tex);glDeleteVertexArrays(1,&vao);glDeleteProgram(program);glDeleteShader(vs);glDeleteShader(fs);
    glfwDestroyWindow(w);glfwTerminate();return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';glfwTerminate();return 1;}}
