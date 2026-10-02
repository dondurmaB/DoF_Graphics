#include <glad/glad.h>

#if defined(__APPLE__)
#include <dlfcn.h>
#endif

PFNGLGETSTRINGPROC glad_glGetString = 0;
PFNGLENABLEPROC glad_glEnable = 0;
PFNGLDISABLEPROC glad_glDisable = 0;
PFNGLDEPTHFUNCPROC glad_glDepthFunc = 0;
PFNGLVIEWPORTPROC glad_glViewport = 0;
PFNGLCLEARCOLORPROC glad_glClearColor = 0;
PFNGLCLEARPROC glad_glClear = 0;
PFNGLGENVERTEXARRAYSPROC glad_glGenVertexArrays = 0;
PFNGLBINDVERTEXARRAYPROC glad_glBindVertexArray = 0;
PFNGLDELETEVERTEXARRAYSPROC glad_glDeleteVertexArrays = 0;
PFNGLGENBUFFERSPROC glad_glGenBuffers = 0;
PFNGLBINDBUFFERPROC glad_glBindBuffer = 0;
PFNGLBUFFERDATAPROC glad_glBufferData = 0;
PFNGLDELETEBUFFERSPROC glad_glDeleteBuffers = 0;
PFNGLVERTEXATTRIBPOINTERPROC glad_glVertexAttribPointer = 0;
PFNGLENABLEVERTEXATTRIBARRAYPROC glad_glEnableVertexAttribArray = 0;
PFNGLCREATESHADERPROC glad_glCreateShader = 0;
PFNGLSHADERSOURCEPROC glad_glShaderSource = 0;
PFNGLCOMPILESHADERPROC glad_glCompileShader = 0;
PFNGLGETSHADERIVPROC glad_glGetShaderiv = 0;
PFNGLGETSHADERINFOLOGPROC glad_glGetShaderInfoLog = 0;
PFNGLDELETESHADERPROC glad_glDeleteShader = 0;
PFNGLCREATEPROGRAMPROC glad_glCreateProgram = 0;
PFNGLATTACHSHADERPROC glad_glAttachShader = 0;
PFNGLLINKPROGRAMPROC glad_glLinkProgram = 0;
PFNGLGETPROGRAMIVPROC glad_glGetProgramiv = 0;
PFNGLGETPROGRAMINFOLOGPROC glad_glGetProgramInfoLog = 0;
PFNGLDELETEPROGRAMPROC glad_glDeleteProgram = 0;
PFNGLUSEPROGRAMPROC glad_glUseProgram = 0;
PFNGLGETUNIFORMLOCATIONPROC glad_glGetUniformLocation = 0;
PFNGLUNIFORM1FPROC glad_glUniform1f = 0;
PFNGLUNIFORMMATRIX4FVPROC glad_glUniformMatrix4fv = 0;
PFNGLDRAWARRAYSPROC glad_glDrawArrays = 0;
PFNGLDRAWELEMENTSPROC glad_glDrawElements = 0;
PFNGLPIXELSTOREIPROC glad_glPixelStorei = 0;
PFNGLREADPIXELSPROC glad_glReadPixels = 0;
PFNGLPOLYGONMODEPROC glad_glPolygonMode = 0;

static void *get_proc(GLADloadfunc load, const char *name)
{
    void *proc = 0;

    if (load) {
        proc = (void *)load(name);
    }

#if defined(__APPLE__)
    if (!proc) {
        static void *framework = 0;
        if (!framework) {
            framework = dlopen("/System/Library/Frameworks/OpenGL.framework/OpenGL", RTLD_LAZY | RTLD_LOCAL);
        }
        if (framework) {
            proc = dlsym(framework, name);
        }
    }
#endif

    return proc;
}

int gladLoadGLLoader(GLADloadfunc load)
{
    glad_glGetString = (PFNGLGETSTRINGPROC)get_proc(load, "glGetString");
    glad_glEnable = (PFNGLENABLEPROC)get_proc(load, "glEnable");
    glad_glDisable = (PFNGLDISABLEPROC)get_proc(load, "glDisable");
    glad_glDepthFunc = (PFNGLDEPTHFUNCPROC)get_proc(load, "glDepthFunc");
    glad_glViewport = (PFNGLVIEWPORTPROC)get_proc(load, "glViewport");
    glad_glClearColor = (PFNGLCLEARCOLORPROC)get_proc(load, "glClearColor");
    glad_glClear = (PFNGLCLEARPROC)get_proc(load, "glClear");
    glad_glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)get_proc(load, "glGenVertexArrays");
    glad_glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)get_proc(load, "glBindVertexArray");
    glad_glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)get_proc(load, "glDeleteVertexArrays");
    glad_glGenBuffers = (PFNGLGENBUFFERSPROC)get_proc(load, "glGenBuffers");
    glad_glBindBuffer = (PFNGLBINDBUFFERPROC)get_proc(load, "glBindBuffer");
    glad_glBufferData = (PFNGLBUFFERDATAPROC)get_proc(load, "glBufferData");
    glad_glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)get_proc(load, "glDeleteBuffers");
    glad_glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)get_proc(load, "glVertexAttribPointer");
    glad_glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)get_proc(load, "glEnableVertexAttribArray");
    glad_glCreateShader = (PFNGLCREATESHADERPROC)get_proc(load, "glCreateShader");
    glad_glShaderSource = (PFNGLSHADERSOURCEPROC)get_proc(load, "glShaderSource");
    glad_glCompileShader = (PFNGLCOMPILESHADERPROC)get_proc(load, "glCompileShader");
    glad_glGetShaderiv = (PFNGLGETSHADERIVPROC)get_proc(load, "glGetShaderiv");
    glad_glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)get_proc(load, "glGetShaderInfoLog");
    glad_glDeleteShader = (PFNGLDELETESHADERPROC)get_proc(load, "glDeleteShader");
    glad_glCreateProgram = (PFNGLCREATEPROGRAMPROC)get_proc(load, "glCreateProgram");
    glad_glAttachShader = (PFNGLATTACHSHADERPROC)get_proc(load, "glAttachShader");
    glad_glLinkProgram = (PFNGLLINKPROGRAMPROC)get_proc(load, "glLinkProgram");
    glad_glGetProgramiv = (PFNGLGETPROGRAMIVPROC)get_proc(load, "glGetProgramiv");
    glad_glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)get_proc(load, "glGetProgramInfoLog");
    glad_glDeleteProgram = (PFNGLDELETEPROGRAMPROC)get_proc(load, "glDeleteProgram");
    glad_glUseProgram = (PFNGLUSEPROGRAMPROC)get_proc(load, "glUseProgram");
    glad_glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)get_proc(load, "glGetUniformLocation");
    glad_glUniform1f = (PFNGLUNIFORM1FPROC)get_proc(load, "glUniform1f");
    glad_glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC)get_proc(load, "glUniformMatrix4fv");
    glad_glDrawArrays = (PFNGLDRAWARRAYSPROC)get_proc(load, "glDrawArrays");
    glad_glDrawElements = (PFNGLDRAWELEMENTSPROC)get_proc(load, "glDrawElements");
    glad_glPixelStorei = (PFNGLPIXELSTOREIPROC)get_proc(load, "glPixelStorei");
    glad_glReadPixels = (PFNGLREADPIXELSPROC)get_proc(load, "glReadPixels");
    glad_glPolygonMode = (PFNGLPOLYGONMODEPROC)get_proc(load, "glPolygonMode");

    return glad_glGetString &&
           glad_glEnable &&
           glad_glDisable &&
           glad_glDepthFunc &&
           glad_glViewport &&
           glad_glClearColor &&
           glad_glClear &&
           glad_glGenVertexArrays &&
           glad_glBindVertexArray &&
           glad_glDeleteVertexArrays &&
           glad_glGenBuffers &&
           glad_glBindBuffer &&
           glad_glBufferData &&
           glad_glDeleteBuffers &&
           glad_glVertexAttribPointer &&
           glad_glEnableVertexAttribArray &&
           glad_glCreateShader &&
           glad_glShaderSource &&
           glad_glCompileShader &&
           glad_glGetShaderiv &&
           glad_glGetShaderInfoLog &&
           glad_glDeleteShader &&
           glad_glCreateProgram &&
           glad_glAttachShader &&
           glad_glLinkProgram &&
           glad_glGetProgramiv &&
           glad_glGetProgramInfoLog &&
           glad_glDeleteProgram &&
           glad_glUseProgram &&
           glad_glGetUniformLocation &&
           glad_glUniform1f &&
           glad_glUniformMatrix4fv &&
           glad_glDrawArrays &&
           glad_glDrawElements &&
           glad_glPixelStorei &&
           glad_glReadPixels &&
           glad_glPolygonMode;
}
