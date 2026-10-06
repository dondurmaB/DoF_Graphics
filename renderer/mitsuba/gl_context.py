"""Offscreen OpenGL 3.3 core contexts: CGL on macOS, EGL on Linux.

`gl_gather.py` runs the production `shaders/dof_scene/screen.frag` on linear
Mitsuba inputs. It did that through CGL with the macOS OpenGL framework path
hard-coded, which means it could only ever run on a Mac. The dataset for stage 3
is generated on a Linux GPU cluster, where that path does not exist and there is
no display server, so the gather had to become portable before anything could be
rendered at scale.

The alternative was a numpy or CuPy reimplementation of the gather. That was
rejected: the audit measured the existing Python port as NOT equivalent to the
shader (its sample disk is rotated about 68.75 degrees, and it bilinearly
interpolates packed depth and precomputed radius where the shader samples depth
nearest-neighbour and evaluates the circle of confusion afterwards, which invents
depths belonging to neither surface at silhouettes). A dataset generated with a
near-copy of the method under study would be quietly wrong. Running the actual
shader keeps the baseline honest on both platforms.

EGL notes, since this is the part that tends to fail silently on a cluster:

  * The device is selected through EGL_EXT_platform_device rather than
    EGL_DEFAULT_DISPLAY. A default display assumes a window system; compute nodes
    have none, and the call either fails or quietly picks a software renderer.
  * EGL_OPENGL_API is bound explicitly. The default is OpenGL ES, which does not
    accept a 3.3 core profile and would reject the shader.
  * The context is made current with no surface (EGL_KHR_surfaceless_context).
    All rendering targets an application framebuffer object, so a pbuffer would
    only be a dummy. A pbuffer fallback is used where surfaceless is missing.
  * Entry points are resolved through eglGetProcAddress first. Which symbols
    libGL exports directly varies between NVIDIA and Mesa, and a missing
    glGenVertexArrays shows up as a null-pointer crash rather than an error.

Verify a node before submitting an array job:

    python renderer/mitsuba/gl_context.py
"""
from __future__ import annotations

import ctypes as C
import os
import platform
import sys

# --- EGL constants. Spelled out because there is no EGL header to parse here.
EGL_SUCCESS = 0x3000
EGL_NO_CONTEXT = C.c_void_p(0)
EGL_NO_SURFACE = C.c_void_p(0)
EGL_NO_DISPLAY = C.c_void_p(0)
EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_OPENGL_API = 0x30A2
EGL_OPENGL_BIT = 0x0008
EGL_PBUFFER_BIT = 0x0001
EGL_SURFACE_TYPE = 0x3033
EGL_RENDERABLE_TYPE = 0x3040
EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_BLUE_SIZE, EGL_ALPHA_SIZE = 0x3024, 0x3023, 0x3022, 0x3021
EGL_DEPTH_SIZE = 0x3025
EGL_WIDTH, EGL_HEIGHT = 0x3057, 0x3056
EGL_CONTEXT_MAJOR_VERSION = 0x3098
EGL_CONTEXT_MINOR_VERSION = 0x30FB
EGL_CONTEXT_OPENGL_PROFILE_MASK = 0x30FD
EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x00000001
EGL_EXTENSIONS = 0x3055
EGL_NONE = 0x3038

GL_VENDOR, GL_RENDERER, GL_VERSION = 0x1F00, 0x1F01, 0x1F02


class ContextError(RuntimeError):
    """Raised with enough detail to tell a cluster problem from a code problem."""


class _Backend:
    """Common entry-point lookup. Subclasses provide creation and make_current."""

    def __init__(self):
        self.library = None
        self.get_proc = None
        self._cache = {}

    def function(self, name, result, *args):
        """Resolve a GL entry point, preferring the platform's proc-address call.

        Cached because every uniform upload would otherwise pay a dlsym and,
        on EGL, an eglGetProcAddress round trip.
        """
        key = (name, result, args)
        if key in self._cache:
            return self._cache[key]
        pointer = None
        if self.get_proc is not None:
            pointer = self.get_proc(name.encode())
        if not pointer:
            try:
                function = getattr(self.library, name)
            except AttributeError:
                raise ContextError(
                    f"OpenGL entry point {name} is unavailable. On a cluster this usually "
                    f"means the context fell back to a software or OpenGL ES driver; check "
                    f"that a GPU is allocated and that libEGL comes from the vendor driver."
                ) from None
            function.restype, function.argtypes = result, list(args)
            self._cache[key] = function
            return function
        prototype = C.CFUNCTYPE(result, *args)
        function = prototype(pointer)
        self._cache[key] = function
        return function

    def describe(self):
        getString = self.function("glGetString", C.c_char_p, C.c_uint)
        return {
            "vendor": (getString(GL_VENDOR) or b"?").decode(),
            "renderer": (getString(GL_RENDERER) or b"?").decode(),
            "version": (getString(GL_VERSION) or b"?").decode(),
            "backend": self.name,
        }


class CGLBackend(_Backend):
    """macOS. Unchanged behaviour from the original implementation."""

    name = "CGL"

    def __init__(self):
        super().__init__()
        path = "/System/Library/Frameworks/OpenGL.framework/OpenGL"
        try:
            self.library = C.CDLL(path)
        except OSError as error:
            raise ContextError(f"Could not load the macOS OpenGL framework: {error}") from None
        self.context = C.c_void_p()
        pixel_format, count = C.c_void_p(), C.c_int()
        # 99 = kCGLPFAOpenGLProfile, 0x3200 = core profile 3.2+.
        attributes = (C.c_int * 3)(99, 0x3200, 0)
        choose = self._cgl("CGLChoosePixelFormat", C.c_int, C.POINTER(C.c_int),
                           C.POINTER(C.c_void_p), C.POINTER(C.c_int))
        if choose(attributes, C.byref(pixel_format), C.byref(count)):
            raise ContextError("CGL pixel format failed; macOS graphics access is required")
        create = self._cgl("CGLCreateContext", C.c_int, C.c_void_p, C.c_void_p,
                           C.POINTER(C.c_void_p))
        error = create(pixel_format, None, C.byref(self.context))
        self._cgl("CGLDestroyPixelFormat", C.c_int, C.c_void_p)(pixel_format)
        if error:
            raise ContextError(f"CGL context creation failed: {error}")
        self.make_current()

    def _cgl(self, name, result, *args):
        function = getattr(self.library, name)
        function.restype, function.argtypes = result, list(args)
        return function

    def make_current(self):
        self._cgl("CGLSetCurrentContext", C.c_int, C.c_void_p)(self.context)

    def destroy(self):
        if self.context:
            self._cgl("CGLSetCurrentContext", C.c_int, C.c_void_p)(None)
            self._cgl("CGLDestroyContext", C.c_int, C.c_void_p)(self.context)
            self.context = C.c_void_p()


class EGLBackend(_Backend):
    """Linux, headless. Selects a GPU device explicitly; no display server."""

    name = "EGL"

    def __init__(self, device_index=None):
        super().__init__()
        self.egl = self._load("libEGL.so.1", "libEGL.so")
        self.library = self._load("libGL.so.1", "libGL.so", "libOpenGL.so.0")
        get_proc = self.egl.eglGetProcAddress
        get_proc.restype, get_proc.argtypes = C.c_void_p, [C.c_char_p]
        self.get_proc = get_proc

        self.display = self._open_display(device_index)
        major, minor = C.c_int(), C.c_int()
        initialize = self._egl("eglInitialize", C.c_uint, C.c_void_p,
                               C.POINTER(C.c_int), C.POINTER(C.c_int))
        if not initialize(self.display, C.byref(major), C.byref(minor)):
            raise ContextError(f"eglInitialize failed ({self._error()})")
        self.egl_version = (major.value, minor.value)

        # Desktop GL, not GL ES. The default is ES, which rejects a core profile
        # and would then refuse the 3.3 shader with a confusing compile error.
        if not self._egl("eglBindAPI", C.c_uint, C.c_uint)(EGL_OPENGL_API):
            raise ContextError("eglBindAPI(EGL_OPENGL_API) failed; this driver has no desktop GL")

        config = C.c_void_p()
        count = C.c_int()
        attributes = (C.c_int * 15)(
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
            EGL_DEPTH_SIZE, 0,
            EGL_NONE)
        choose = self._egl("eglChooseConfig", C.c_uint, C.c_void_p, C.POINTER(C.c_int),
                           C.POINTER(C.c_void_p), C.c_int, C.POINTER(C.c_int))
        if not choose(self.display, attributes, C.byref(config), 1, C.byref(count)) or not count.value:
            raise ContextError(f"No EGL config with desktop OpenGL ({self._error()})")

        context_attributes = (C.c_int * 7)(
            EGL_CONTEXT_MAJOR_VERSION, 3,
            EGL_CONTEXT_MINOR_VERSION, 3,
            EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            EGL_NONE)
        create = self._egl("eglCreateContext", C.c_void_p, C.c_void_p, C.c_void_p,
                           C.c_void_p, C.POINTER(C.c_int))
        self.context = create(self.display, config, EGL_NO_CONTEXT, context_attributes)
        if not self.context:
            raise ContextError(f"eglCreateContext for GL 3.3 core failed ({self._error()})")

        # Surfaceless where supported: everything renders into an FBO, so a
        # pbuffer would be an unused allocation. Older drivers need the pbuffer.
        self.surface = EGL_NO_SURFACE
        extensions = self._egl("eglQueryString", C.c_char_p, C.c_void_p, C.c_int)(
            self.display, EGL_EXTENSIONS) or b""
        if b"EGL_KHR_surfaceless_context" not in extensions:
            surface_attributes = (C.c_int * 5)(EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE)
            self.surface = self._egl("eglCreatePbufferSurface", C.c_void_p, C.c_void_p,
                                     C.c_void_p, C.POINTER(C.c_int))(
                self.display, config, surface_attributes)
            if not self.surface:
                raise ContextError(f"eglCreatePbufferSurface failed ({self._error()})")
        self.make_current()

    @staticmethod
    def _load(*names):
        for name in names:
            try:
                return C.CDLL(name)
            except OSError:
                continue
        raise ContextError(
            f"Could not load any of {names}. On a GPU node these come from the vendor "
            f"driver; if only Mesa is present the gather will run on llvmpipe and be "
            f"unusably slow. Check `ldconfig -p | grep -E 'libEGL|libGL'`."
        )

    def _egl(self, name, result, *args):
        function = getattr(self.egl, name)
        function.restype, function.argtypes = result, list(args)
        return function

    def _error(self):
        code = self._egl("eglGetError", C.c_uint)()
        return f"EGL error 0x{code:04x}" if code != EGL_SUCCESS else "no EGL error reported"

    def _open_display(self, device_index):
        """Pick a GPU explicitly through EGL_EXT_platform_device.

        EGL_DEFAULT_DISPLAY assumes a window system. On a compute node that
        either fails outright or silently lands on a software rasterizer, which
        would produce correct images at roughly a thousandth of the speed. Being
        explicit turns that into an error instead of a mystery.
        """
        query_devices = self.get_proc(b"eglQueryDevicesEXT")
        get_platform_display = self.get_proc(b"eglGetPlatformDisplayEXT")
        if query_devices and get_platform_display:
            query = C.CFUNCTYPE(C.c_uint, C.c_int, C.POINTER(C.c_void_p),
                                C.POINTER(C.c_int))(query_devices)
            count = C.c_int()
            if query(0, None, C.byref(count)) and count.value:
                devices = (C.c_void_p * count.value)()
                query(count.value, devices, C.byref(count))
                index = 0 if device_index is None else device_index
                if not 0 <= index < count.value:
                    raise ContextError(
                        f"EGL device {index} requested but only {count.value} are visible. "
                        f"Under SLURM, CUDA_VISIBLE_DEVICES masks GPUs for CUDA but not "
                        f"always for EGL; pass --egl-device or set DOF_EGL_DEVICE."
                    )
                platform_display = C.CFUNCTYPE(
                    C.c_void_p, C.c_uint, C.c_void_p, C.POINTER(C.c_int))(get_platform_display)
                display = platform_display(EGL_PLATFORM_DEVICE_EXT, devices[index], None)
                if display:
                    self.device_count = count.value
                    self.device_index = index
                    return display
        # No device enumeration: fall back, and say so, because this is the path
        # that can quietly give a software renderer.
        display = self._egl("eglGetDisplay", C.c_void_p, C.c_void_p)(EGL_NO_DISPLAY)
        if not display:
            raise ContextError(
                "Neither EGL_EXT_platform_device nor EGL_DEFAULT_DISPLAY produced a display. "
                "This node has no usable EGL driver."
            )
        self.device_count, self.device_index = 0, -1
        return display

    def make_current(self):
        if not self._egl("eglMakeCurrent", C.c_uint, C.c_void_p, C.c_void_p,
                         C.c_void_p, C.c_void_p)(
                self.display, self.surface, self.surface, self.context):
            raise ContextError(f"eglMakeCurrent failed ({self._error()})")

    def destroy(self):
        if getattr(self, "context", None):
            self._egl("eglMakeCurrent", C.c_uint, C.c_void_p, C.c_void_p, C.c_void_p,
                      C.c_void_p)(self.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)
            self._egl("eglDestroyContext", C.c_uint, C.c_void_p, C.c_void_p)(
                self.display, self.context)
            self.context = None
        if getattr(self, "surface", None):
            self._egl("eglDestroySurface", C.c_uint, C.c_void_p, C.c_void_p)(
                self.display, self.surface)
            self.surface = EGL_NO_SURFACE


def create_context(backend=None, device_index=None):
    """Return an offscreen GL 3.3 core context for this machine.

    `backend` forces 'cgl' or 'egl'; the default follows the platform. The
    device index can also come from DOF_EGL_DEVICE, so a SLURM array task can
    select a GPU without editing the submit script.
    """
    if device_index is None and os.environ.get("DOF_EGL_DEVICE"):
        device_index = int(os.environ["DOF_EGL_DEVICE"])
    backend = (backend or os.environ.get("DOF_GL_BACKEND") or
               ("cgl" if platform.system() == "Darwin" else "egl")).lower()
    if backend == "cgl":
        return CGLBackend()
    if backend == "egl":
        return EGLBackend(device_index)
    raise ContextError(f"Unknown GL backend {backend!r}; expected 'cgl' or 'egl'")


def main():
    """Node check. Run this on a compute node before submitting an array job."""
    try:
        context = create_context()
    except ContextError as error:
        print(f"FAIL: {error}")
        return 1
    info = context.describe()
    for key in ("backend", "vendor", "renderer", "version"):
        print(f"  {key:9s} {info[key]}")
    software = any(mark in info["renderer"].lower()
                   for mark in ("llvmpipe", "softpipe", "swrast", "software"))
    context.destroy()
    if software:
        print("FAIL: this is a software rasterizer. The gather would be unusably slow and "
              "the node is not using its GPU driver.")
        return 1
    if not info["version"].startswith(("3.3", "4.")):
        print(f"FAIL: need OpenGL 3.3 core or newer, got {info['version']}")
        return 1
    print("OK: offscreen GL context is usable for the production gather.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
