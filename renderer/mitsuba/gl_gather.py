"""Run the production DoFScene fragment shader on linear Mitsuba inputs.

Offscreen RGBA32F; no window, display transform, or Python blur. Arrays use
top-left row order. Upload/readback flip rows so native OpenGL bottom-left
texture coordinates and its disk orientation are preserved.

The context comes from gl_context.py: CGL on macOS, EGL on Linux. This file used
to create a CGL context inline with the framework path hard-coded, which meant
the production gather could not run on the GPU cluster the stage-3 dataset is
generated on.
"""
from pathlib import Path
import ctypes as C
import numpy as np

from gl_context import ContextError, create_context  # noqa: F401  (re-exported)

ROOT = Path(__file__).resolve().parents[2]
SHADER = ROOT / "shaders/dof_scene/screen.frag"
VERTEX = """#version 330 core
out vec2 texCoord;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    texCoord = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""


class Gather:
    def __init__(self, fragment=None, backend=None, device_index=None):
        self.backend = create_context(backend, device_index)
        info = self.backend.describe()
        self.renderer = info['renderer']
        self.gl_info = info
        self.program = self.f('glCreateProgram', C.c_uint)()
        for kind, source in ((0x8b31, VERTEX), (0x8b30, fragment or SHADER.read_text())):
            shader = self.compile(kind, source)
            self.f('glAttachShader', None, C.c_uint, C.c_uint)(self.program, shader)
            self.f('glDeleteShader', None, C.c_uint)(shader)
        self.f('glLinkProgram', None, C.c_uint)(self.program)
        ok = C.c_int()
        self.f('glGetProgramiv', None, C.c_uint, C.c_uint, C.POINTER(C.c_int))(
            self.program, 0x8b82, C.byref(ok))
        if not ok.value:
            raise RuntimeError('GL program link failed')
        self.f('glUseProgram', None, C.c_uint)(self.program)
        vao = C.c_uint()
        self.f('glGenVertexArrays', None, C.c_int, C.POINTER(C.c_uint))(1, C.byref(vao))
        self.f('glBindVertexArray', None, C.c_uint)(vao)
        self.vao = vao

    def f(self, name, result, *args):
        """Resolve a GL entry point through the active backend.

        EGL drivers do not reliably export every GL 3.3 symbol from libGL, so
        the backend tries eglGetProcAddress first. A missing glGenVertexArrays
        otherwise appears as a null-pointer crash rather than an error.
        """
        return self.backend.function(name, result, *args)

    def compile(self, kind, source):
        shader = self.f('glCreateShader', C.c_uint, C.c_uint)(kind)
        string = C.c_char_p(source.encode())
        self.f('glShaderSource', None, C.c_uint, C.c_int, C.POINTER(C.c_char_p),
               C.c_void_p)(shader, 1, C.byref(string), None)
        self.f('glCompileShader', None, C.c_uint)(shader)
        ok = C.c_int()
        self.f('glGetShaderiv', None, C.c_uint, C.c_uint, C.POINTER(C.c_int))(
            shader, 0x8b81, C.byref(ok))
        if not ok.value:
            log = C.create_string_buffer(16384)
            self.f('glGetShaderInfoLog', None, C.c_uint, C.c_int, C.c_void_p,
                   C.c_char_p)(shader, len(log), None, log)
            raise RuntimeError(log.value.decode())
        return shader

    def uniform(self, name, value):
        location = self.f('glGetUniformLocation', C.c_int, C.c_uint, C.c_char_p)(
            self.program, name.encode())
        if isinstance(value, (int, bool)):
            self.f('glUniform1i', None, C.c_int, C.c_int)(location, int(value))
        elif isinstance(value, tuple):
            self.f('glUniform2f', None, C.c_int, C.c_float, C.c_float)(location, *value)
        else:
            self.f('glUniform1f', None, C.c_int, C.c_float)(location, float(value))

    def run(self, rgb, depth, camera, variant='naive', max_radius=120., raw_depth=False, mode=0):
        rgb, depth = np.asarray(rgb, np.float32), np.asarray(depth, np.float32)
        h, w = depth.shape
        if rgb.shape != (h, w, 3) or not np.isfinite(rgb).all():
            raise ValueError('Expected finite HxWx3 linear RGB matching depth dimensions')
        if not np.isfinite(depth).all() or (depth <= 0).any():
            raise ValueError('Depth must be finite and positive; map sky explicitly before upload')
        if variant not in ('naive', 'weighted') or not 0 < max_radius <= 120:
            raise ValueError('Invalid gather settings')
        self.backend.make_current()
        self.f('glUseProgram', None, C.c_uint)(self.program)
        self.f('glBindVertexArray', None, C.c_uint)(self.vao)
        ids = (C.c_uint * 3)()
        self.f('glGenTextures', None, C.c_int, C.POINTER(C.c_uint))(3, ids)
        for i, (data, fmt, internal, filtering) in enumerate((
                (rgb, 0x1907, 0x8815, 0x2601),
                (depth, 0x1903, 0x822e, 0x2600),
                (None, 0x1908, 0x8814, 0x2600))):
            self.f('glActiveTexture', None, C.c_uint)(0x84c0 + i)
            self.f('glBindTexture', None, C.c_uint, C.c_uint)(0xde1, ids[i])
            for pname, value in ((0x2801, filtering), (0x2800, filtering),
                                  (0x2802, 0x812f), (0x2803, 0x812f)):
                self.f('glTexParameteri', None, C.c_uint, C.c_uint, C.c_int)(0xde1, pname, value)
            pixels = None if data is None else np.ascontiguousarray(data[::-1])
            self.f('glTexImage2D', None, C.c_uint, C.c_int, C.c_int, C.c_int, C.c_int,
                   C.c_int, C.c_uint, C.c_uint, C.c_void_p)(
                       0xde1, 0, internal, w, h, 0, fmt, 0x1406,
                       None if pixels is None else pixels.ctypes.data)
        fbo = C.c_uint()
        self.f('glGenFramebuffers', None, C.c_int, C.POINTER(C.c_uint))(1, C.byref(fbo))
        self.f('glBindFramebuffer', None, C.c_uint, C.c_uint)(0x8d40, fbo)
        self.f('glFramebufferTexture2D', None, C.c_uint, C.c_uint, C.c_uint, C.c_uint,
               C.c_int)(0x8d40, 0x8ce0, 0xde1, ids[2], 0)
        if self.f('glCheckFramebufferStatus', C.c_uint, C.c_uint)(0x8d40) != 0x8cd5:
            raise RuntimeError('Float gather framebuffer incomplete')
        self.f('glViewport', None, C.c_int, C.c_int, C.c_int, C.c_int)(0, 0, w, h)
        settings = dict(uColor=0, uDepth=1, uMode=mode, uLinearOutput=1,
                        uDepthIsLinear=int(not raw_depth), uGatherMode=int(variant == 'weighted'),
                        uResolution=(float(w), float(h)), uMaxRadius=float(max_radius),
                        uNear=.01, uFar=1000., uFocusDistance=camera['focus_distance_m'],
                        uFocalLengthMm=camera['focal_length_mm'], uSensorHeightMm=camera['sensor_height_mm'],
                        uFNumber=camera['f_number'])
        for k, v in settings.items():
            self.uniform(k, v)
        self.f('glDrawArrays', None, C.c_uint, C.c_int, C.c_int)(4, 0, 3)
        result = np.empty((h, w, 4), np.float32)
        self.f('glReadPixels', None, C.c_int, C.c_int, C.c_int, C.c_int,
               C.c_uint, C.c_uint, C.c_void_p)(0, 0, w, h, 0x1908, 0x1406, result.ctypes.data)
        error = self.f('glGetError', C.c_uint)()
        self.f('glDeleteTextures', None, C.c_int, C.POINTER(C.c_uint))(3, ids)
        self.f('glDeleteFramebuffers', None, C.c_int, C.POINTER(C.c_uint))(1, C.byref(fbo))
        if error or not np.isfinite(result).all():
            raise RuntimeError(f'GL gather failed: {error}')
        return result[::-1, :, :3].copy()

    def close(self):
        self.backend.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
