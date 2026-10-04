// ggml OpenGL backend (skeleton, plan item P52)
//
// OpenGL 4.3 compute through WGL, for drivers where D3D compute is weak or missing:
//  - one device per GL context; the context lives on a hidden window, core profile 4.3 or newer
//  - every GL function past 1.1 comes from wglGetProcAddress (no GLAD/GLEW)
//  - tensor buffers are SSBOs; each tensor is bound at an aligned offset at or before it, the rest of the
//    offset is passed to the kernel in elements (same scheme as the D3D11 backend)
//  - kernels are GLSL 430 ports of the D3D11 HLSL, compiled by the driver at run time; the program
//    binaries are cached on disk
//  - GL contexts are thread-bound: every entry point makes the context current on the calling thread
//    and restores the previous one when it returns
// Ops: MUL_MAT (f32/f16 weights; Q4_0/Q8_0/Q4_K/Q6_K weights through the mat-vec kernel), CPY, ADD, MUL, SCALE, GET_ROWS; everything else stays on the CPU.

#include "ggml-opengl.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ggml-opengl-shaders.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <GL/gl.h>
#else
// Linux: EGL without a window; no GL or EGL headers needed, libEGL is opened at run time
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cstddef>
typedef unsigned int  GLenum;
typedef unsigned int  GLuint;
typedef int           GLint;
typedef int           GLsizei;
typedef unsigned char GLubyte;
typedef unsigned int  GLbitfield;
#define APIENTRY
#define GL_NO_ERROR      0
#define GL_TRUE          1
#define GL_UNSIGNED_BYTE 0x1401
#define GL_RENDERER      0x1F01
#define GL_VERSION       0x1F02
#define GL_EXTENSIONS    0x1F03
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Below 16 columns the matvec kernel is as fast or faster (measured for the D3D11 kernels, not for GL)
#define GGML_GL_TILED_DEFAULT        16
#define GGML_GL_WG_SIZE              256
#ifndef GGML_GL_MAX_TPR
#define GGML_GL_MAX_TPR              GGML_GL_WG_SIZE   // matrix-vector threads per row cap (power of two)
#endif
#define GGML_GL_MAX_WG_PER_DIM       65535
#define GGML_GL_BINDING_ALIGNMENT    256   // minimum tensor binding alignment; raised to the driver's SSBO offset alignment
#define GGML_GL_PARAM_SLOT_SIZE      256
#define GGML_GL_FLASH_ATTN_TMP_MAX   (64ull * 1024 * 1024)   // cap on the block results buffer
#define GGML_GL_PARAM_SLOT_COUNT     1024
#define GGML_GL_BUFFER_SLACK         256   // past the end of every buffer: unaligned loads read one word further

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

/* GL enums past 1.1, prefixed E_ so they never collide with GL/gl.h */

enum : GLenum {
    E_COMPUTE_SHADER                       = 0x91B9,
    E_SHADER_STORAGE_BUFFER                = 0x90D2,
    E_UNIFORM_BUFFER                       = 0x8A11,
    E_COPY_READ_BUFFER                     = 0x8F36,
    E_COPY_WRITE_BUFFER                    = 0x8F37,
    E_SHADER_STORAGE_BARRIER_BIT           = 0x2000,
    E_ALL_BARRIER_BITS                     = 0xFFFFFFFF,
    E_MAX_SHADER_STORAGE_BLOCK_SIZE        = 0x90DE,
    E_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT = 0x90DF,
    E_UNIFORM_BUFFER_OFFSET_ALIGNMENT      = 0x8A34,
    E_NUM_EXTENSIONS                       = 0x821D,
    E_SHADING_LANGUAGE_VERSION             = 0x8B8C,
    E_CONTEXT_FLAGS                        = 0x821E,
    E_CONTEXT_FLAG_DEBUG_BIT               = 0x0002,
    E_COMPILE_STATUS                       = 0x8B81,
    E_LINK_STATUS                          = 0x8B82,
    E_INFO_LOG_LENGTH                      = 0x8B84,
    E_DYNAMIC_COPY                         = 0x88EA,
    E_DYNAMIC_DRAW                         = 0x88E8,
    E_DEBUG_OUTPUT                         = 0x92E0,
    E_DEBUG_OUTPUT_SYNCHRONOUS             = 0x8242,
    E_DEBUG_SEVERITY_NOTIFICATION          = 0x826B,
    E_R8UI                                 = 0x8232,
    E_RED_INTEGER                          = 0x8D94,
    E_PROGRAM_BINARY_RETRIEVABLE_HINT      = 0x8257,
    E_PROGRAM_BINARY_LENGTH                = 0x8741,
    E_NUM_PROGRAM_BINARY_FORMATS           = 0x87FE,
    E_VBO_FREE_MEMORY_ATI                  = 0x87FB,
    E_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX = 0x9047,
    E_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX = 0x9049,
    E_TIME_ELAPSED                         = 0x88BF,
    E_QUERY_RESULT                         = 0x8866,
    E_MAP_WRITE_BIT                        = 0x0002,
    E_MAP_PERSISTENT_BIT                   = 0x0040,
    E_MAP_COHERENT_BIT                     = 0x0080,
    E_SYNC_GPU_COMMANDS_COMPLETE           = 0x9117,
    E_SYNC_FLUSH_COMMANDS_BIT              = 0x0001,
    E_TIMEOUT_EXPIRED                      = 0x911B,
    E_WAIT_FAILED                          = 0x911D,
};
enum : int {
    E_WGL_CONTEXT_MAJOR_VERSION = 0x2091,
    E_WGL_CONTEXT_MINOR_VERSION = 0x2092,
    E_WGL_CONTEXT_FLAGS         = 0x2094,
    E_WGL_CONTEXT_DEBUG_BIT     = 0x0001,
    E_WGL_CONTEXT_PROFILE_MASK  = 0x9126,
    E_WGL_CONTEXT_CORE_BIT      = 0x0001,
};

typedef struct gl_sync_s * gl_sync_t;   // GLsync
typedef void (APIENTRY * gl_debug_cb_t)(GLenum, GLenum, GLuint, GLenum, GLsizei, const char *, const void *);

// required functions past GL 1.1; the device is skipped when one is missing
#define GGML_GL_FUNCS(X) \
    X(const GLubyte *, glGetStringi, (GLenum, GLuint)) \
    X(void, glGetInteger64v, (GLenum, long long *)) \
    X(GLuint, glCreateShader, (GLenum)) \
    X(void, glShaderSource, (GLuint, GLsizei, const char * const *, const GLint *)) \
    X(void, glCompileShader, (GLuint)) \
    X(void, glGetShaderiv, (GLuint, GLenum, GLint *)) \
    X(void, glGetShaderInfoLog, (GLuint, GLsizei, GLsizei *, char *)) \
    X(GLuint, glCreateProgram, (void)) \
    X(void, glAttachShader, (GLuint, GLuint)) \
    X(void, glDetachShader, (GLuint, GLuint)) \
    X(void, glLinkProgram, (GLuint)) \
    X(void, glGetProgramiv, (GLuint, GLenum, GLint *)) \
    X(void, glGetProgramInfoLog, (GLuint, GLsizei, GLsizei *, char *)) \
    X(void, glDeleteShader, (GLuint)) \
    X(void, glDeleteProgram, (GLuint)) \
    X(void, glUseProgram, (GLuint)) \
    X(void, glGenBuffers, (GLsizei, GLuint *)) \
    X(void, glDeleteBuffers, (GLsizei, const GLuint *)) \
    X(void, glBindBuffer, (GLenum, GLuint)) \
    X(void, glBufferData, (GLenum, ptrdiff_t, const void *, GLenum)) \
    X(void, glBufferSubData, (GLenum, ptrdiff_t, ptrdiff_t, const void *)) \
    X(void, glGetBufferSubData, (GLenum, ptrdiff_t, ptrdiff_t, void *)) \
    X(void, glBindBufferRange, (GLenum, GLuint, GLuint, ptrdiff_t, ptrdiff_t)) \
    X(void, glCopyBufferSubData, (GLenum, GLenum, ptrdiff_t, ptrdiff_t, ptrdiff_t)) \
    X(void, glClearBufferSubData, (GLenum, GLenum, ptrdiff_t, ptrdiff_t, GLenum, GLenum, const void *)) \
    X(void, glDispatchCompute, (GLuint, GLuint, GLuint)) \
    X(void, glMemoryBarrier, (GLbitfield))

// optional functions: debug output, program binaries, parallel compilation
#define GGML_GL_FUNCS_OPT(X) \
    X(void, glDebugMessageCallback, (gl_debug_cb_t, const void *)) \
    X(void, glGetProgramBinary, (GLuint, GLsizei, GLsizei *, GLenum *, void *)) \
    X(void, glProgramBinary, (GLuint, GLenum, const void *, GLsizei)) \
    X(void, glProgramParameteri, (GLuint, GLenum, GLint)) \
    X(void, glMaxShaderCompilerThreadsARB, (GLuint)) \
    X(void, glMaxShaderCompilerThreadsKHR, (GLuint)) \
    X(void, glGenQueries, (GLsizei, GLuint *)) \
    X(void, glBeginQuery, (GLenum, GLuint)) \
    X(void, glEndQuery, (GLenum)) \
    X(void, glGetQueryObjectui64v, (GLuint, GLenum, unsigned long long *)) \
    X(void, glBufferStorage, (GLenum, ptrdiff_t, const void *, GLbitfield)) \
    X(void *, glMapBufferRange, (GLenum, ptrdiff_t, ptrdiff_t, GLbitfield)) \
    X(gl_sync_t, glFenceSync, (GLenum, GLbitfield)) \
    X(GLenum, glClientWaitSync, (gl_sync_t, GLbitfield, unsigned long long)) \
    X(void, glDeleteSync, (gl_sync_t))

// one device only (WGL has no adapter choice), so one global table
#define GGML_GL_DECL(ret, name, args) static ret (APIENTRY * p_##name) args = nullptr;
GGML_GL_FUNCS(GGML_GL_DECL)
GGML_GL_FUNCS_OPT(GGML_GL_DECL)
#undef GGML_GL_DECL

#ifdef _WIN32
static HMODULE g_opengl32 = nullptr;

static PROC ggml_opengl_get_proc(const char * name) {
    PROC     p = wglGetProcAddress(name);
    intptr_t v = (intptr_t) p;
    if (v == 0 || v == 1 || v == 2 || v == 3 || v == -1) {
        p = g_opengl32 ? GetProcAddress(g_opengl32, name) : nullptr;
    }
    return p;
}
#else
// GL 1.1 entry points too come from eglGetProcAddress (EGL 1.5 / EGL_KHR_get_all_proc_addresses)
#define GGML_GL_FUNCS_11(X) \
    X(GLenum, glGetError, (void)) \
    X(const GLubyte *, glGetString, (GLenum)) \
    X(void, glGetIntegerv, (GLenum, GLint *)) \
    X(void, glEnable, (GLenum)) \
    X(void, glFinish, (void)) \
    X(void, glFlush, (void))

#define GGML_GL_DECL(ret, name, args) static ret (APIENTRY * p_##name) args = nullptr;
GGML_GL_FUNCS_11(GGML_GL_DECL)
#undef GGML_GL_DECL
#define glGetError    p_glGetError
#define glGetString   p_glGetString
#define glGetIntegerv p_glGetIntegerv
#define glEnable      p_glEnable
#define glFinish      p_glFinish
#define glFlush       p_glFlush

typedef void * EGLDisplay;
typedef void * EGLContext;
typedef void * EGLConfig;
typedef void * EGLSurface;
typedef int    EGLint;
typedef unsigned int EGLBoolean;
typedef unsigned int EGLenum;
enum : EGLint {
    E_EGL_NONE                     = 0x3038,
    E_EGL_EXTENSIONS               = 0x3055,
    E_EGL_OPENGL_API               = 0x30A2,
    E_EGL_CONTEXT_MAJOR_VERSION    = 0x3098,
    E_EGL_CONTEXT_MINOR_VERSION    = 0x30FB,
    E_EGL_CONTEXT_OPENGL_PROFILE_MASK = 0x30FD,
    E_EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x0001,
    E_EGL_CONTEXT_OPENGL_DEBUG     = 0x31B0,
    E_EGL_PLATFORM_SURFACELESS_MESA = 0x31DD,
};

// EGL entry points, from libEGL.so.1
#define GGML_EGL_FUNCS(X) \
    X(void *, eglGetProcAddress, (const char *)) \
    X(EGLDisplay, eglGetPlatformDisplay, (EGLenum, void *, const intptr_t *)) \
    X(EGLBoolean, eglInitialize, (EGLDisplay, EGLint *, EGLint *)) \
    X(EGLBoolean, eglTerminate, (EGLDisplay)) \
    X(const char *, eglQueryString, (EGLDisplay, EGLint)) \
    X(EGLBoolean, eglBindAPI, (EGLenum)) \
    X(EGLContext, eglCreateContext, (EGLDisplay, EGLConfig, EGLContext, const EGLint *)) \
    X(EGLBoolean, eglDestroyContext, (EGLDisplay, EGLContext)) \
    X(EGLBoolean, eglMakeCurrent, (EGLDisplay, EGLSurface, EGLSurface, EGLContext)) \
    X(EGLContext, eglGetCurrentContext, (void)) \
    X(EGLDisplay, eglGetCurrentDisplay, (void)) \
    X(EGLint, eglGetError, (void))

#define GGML_GL_DECL(ret, name, args) static ret (* p_##name) args = nullptr;
GGML_EGL_FUNCS(GGML_GL_DECL)
#undef GGML_GL_DECL

static void * g_libegl = nullptr;

static void * ggml_opengl_get_proc(const char * name) {
    return p_eglGetProcAddress ? p_eglGetProcAddress(name) : nullptr;
}
#endif

// returns the number of missing required functions
static int ggml_opengl_load_funcs() {
    int missing = 0;
#define GGML_GL_LOAD(ret, name, args) \
    p_##name = (ret (APIENTRY *) args) (void *) ggml_opengl_get_proc(#name); \
    if (!p_##name) { GGML_LOG_WARN("ggml_opengl: missing %s\n", #name); missing++; }
#ifndef _WIN32
    GGML_GL_FUNCS_11(GGML_GL_LOAD)
#endif
    GGML_GL_FUNCS(GGML_GL_LOAD)
#undef GGML_GL_LOAD
#define GGML_GL_LOAD_OPT(ret, name, args) p_##name = (ret (APIENTRY *) args) (void *) ggml_opengl_get_proc(#name);
    GGML_GL_FUNCS_OPT(GGML_GL_LOAD_OPT)
#undef GGML_GL_LOAD_OPT
    return missing;
}

static double ggml_opengl_time_us() {
#ifdef _WIN32
    static LARGE_INTEGER freq = {};
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double) now.QuadPart * 1e6 / (double) freq.QuadPart;
#else
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

// GGML_OPENGL_<name>, else GGML_D3D12_<name>: the box test runner passes only GGML_D3D12_* variables
static const char * ggml_opengl_getenv(const char * name) {
    const char * v = getenv((std::string("GGML_OPENGL_") + name).c_str());
    return v ? v : getenv((std::string("GGML_D3D12_") + name).c_str());
}

// All device buffers report the same fake host base pointer; a tensor's byte offset inside its
// buffer is recovered from tensor->data (same scheme as the WebGPU and D3D11 backends).
static void * const gl_ptr_base = (void *) (uintptr_t) 0x1000;  // NOLINT

static size_t ggml_opengl_tensor_offset(const ggml_tensor * tensor) {
    const ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    return (size_t) ((uintptr_t) base_tensor->data - (uintptr_t) gl_ptr_base) + tensor->view_offs;
}

/* Structs */

struct gl_pipeline {
    GLuint      prog = 0;
    std::string name;
};

struct gl_device_ctx {
    std::string name;      // "OpenGL0"
    std::string desc;      // GL_RENDERER
    std::string version;   // GL_VERSION
#ifdef _WIN32
    HWND        hwnd = nullptr;
    HDC         dc   = nullptr;
    HGLRC       ctx  = nullptr;
#else
    EGLDisplay  dpy  = nullptr;
    EGLContext  ctx  = nullptr;
#endif

    size_t   max_alloc    = 0;
    size_t   bind_align   = GGML_GL_BINDING_ALIGNMENT;
    size_t   param_slot   = GGML_GL_PARAM_SLOT_SIZE;
    bool     debug        = false;
    bool     trace        = false;   // GGML_OPENGL_TRACE: print + flush every node, alloc and copy, glFinish per node
    bool     parallel     = false;   // GL_ARB/KHR_parallel_shader_compile
    bool     prog_binary  = false;   // at least one program binary format
    bool     ati_meminfo  = false;
    bool     nvx_meminfo  = false;
    uint32_t tiled_min_cols = GGML_GL_TILED_DEFAULT;
    uint32_t tiled_ksplit   = 0;   // tiles of k per tiled-matmul dispatch (1, 2, 4 or 8); 0 = one dispatch, loop in the shader
    uint32_t max_tpr        = GGML_GL_MAX_TPR;
    bool     no_fuse        = false;   // GGML_OPENGL_NO_FUSE: every node on its own

    GLuint   ubo       = 0;   // kernel parameters, one slot per dispatch, reused round robin
    // GL 4.4 buffer storage: the parameter buffer stays mapped (persistent, coherent) and is written with memcpy;
    // one fence per quarter of the ring keeps a slot from being rewritten while the GPU may still read it
    uint8_t *              ubo_map = nullptr;
    gl_sync_t              ubo_fence[4] = {};
    // last program and SSBO bindings set by a dispatch, so repeats are skipped; forgotten at graph start and
    // whenever a buffer is deleted (a new buffer can get the deleted one's name)
    struct bind_state {
        GLuint buf    = 0;
        size_t offset = 0;
        size_t size   = 0;
    };
    GLuint                  cur_prog = 0;
    std::vector<bind_state> cur_bind;

    GLuint   fa_tmp      = 0;   // flash attention block results, grown on demand
    size_t   fa_tmp_size = 0;
    GLuint   mmid_scratch      = 0;   // mul_mat_id expert tiles and pair list, grown on demand
    size_t   mmid_scratch_size = 0;
    uint32_t next_slot = 0;

    std::unordered_map<std::string, gl_pipeline> pipelines;

    // parallel compilation: a collect pass over a new graph queues the missing pipelines, which are then
    // compiled and linked together (the driver compiles them in parallel with parallel_shader_compile)
    struct pipeline_job {
        std::string key;
        std::string name;
        std::string text;   // full GLSL source with #version and defines
        std::vector<std::string> defines;
    };
    bool                      collecting       = false;
    std::vector<pipeline_job> pipeline_jobs;
    int                       last_graph_nodes = -1;

    // counters printed at exit when GGML_OPENGL_STATS is set
    bool     stats          = false;
    uint64_t n_graphs       = 0;
    uint64_t n_nodes        = 0;
    uint64_t n_dispatches   = 0;
    uint64_t n_barriers     = 0;
    uint64_t n_compiles     = 0;   // programs compiled from source
    uint64_t n_cache_hits   = 0;   // programs loaded from the disk cache
    double   t_compile_us   = 0;
    double   t_graph_us     = 0;
    uint64_t n_set_tensor   = 0;
    uint64_t n_get_tensor   = 0;
    uint64_t bytes_set      = 0;
    uint64_t bytes_get      = 0;
    double   t_set_us       = 0;
    double   t_get_us       = 0;

    // GGML_OPENGL_PROFILE: GPU time per pipeline from GL_TIME_ELAPSED queries around every dispatch; the
    // queries of one graph are read at the start of the next one (by then they are done in generation)
    struct prof_entry {
        uint64_t ns = 0;
        uint64_t n  = 0;
    };
    bool                                          profile = false;
    std::vector<GLuint>                           q_free;
    std::vector<std::pair<GLuint, std::string>>   q_pending;
    std::unordered_map<std::string, prof_entry>   prof;

    std::recursive_mutex mutex;

    ggml_backend_buffer_type buft = {};
};

struct ggml_backend_opengl_buffer_context {
    std::shared_ptr<gl_device_ctx> dev;
    GLuint                         buf  = 0;
    size_t                         size = 0;   // allocated size, slack included
};

struct ggml_backend_opengl_context {
    std::shared_ptr<gl_device_ctx> dev;
    std::string                    name;
};

static std::shared_ptr<gl_device_ctx> ggml_opengl_shared_dev(gl_device_ctx * dev);

// Takes the device lock and makes the device's context current on this thread; restores the previous
// context (or none) on exit. Nests: an inner scope finds the context already current.
#ifdef _WIN32
struct gl_scope {
    gl_device_ctx &                        dev;
    std::lock_guard<std::recursive_mutex> lock;
    HGLRC                                  prev_ctx;
    HDC                                    prev_dc;
    bool                                   switched;

    explicit gl_scope(gl_device_ctx & d) : dev(d), lock(d.mutex) {
        prev_ctx = wglGetCurrentContext();
        prev_dc  = wglGetCurrentDC();
        switched = prev_ctx != dev.ctx;
        if (switched && !wglMakeCurrent(dev.dc, dev.ctx)) {
            GGML_ABORT("ggml_opengl: wglMakeCurrent failed (error %lu)", (unsigned long) GetLastError());
        }
    }
    ~gl_scope() {
        if (switched) {
            wglMakeCurrent(prev_dc, prev_ctx);
        }
    }
};

static void ggml_opengl_release_current(gl_device_ctx &) {
    wglMakeCurrent(nullptr, nullptr);
}
#else
struct gl_scope {
    gl_device_ctx &                        dev;
    std::lock_guard<std::recursive_mutex> lock;
    EGLContext                             prev_ctx;
    EGLDisplay                             prev_dpy;
    bool                                   switched;

    explicit gl_scope(gl_device_ctx & d) : dev(d), lock(d.mutex) {
        prev_ctx = p_eglGetCurrentContext();
        prev_dpy = p_eglGetCurrentDisplay();
        switched = prev_ctx != dev.ctx;
        if (switched && !p_eglMakeCurrent(dev.dpy, nullptr, nullptr, dev.ctx)) {
            GGML_ABORT("ggml_opengl: eglMakeCurrent failed (error 0x%x)", (unsigned) p_eglGetError());
        }
    }
    ~gl_scope() {
        if (switched) {
            p_eglMakeCurrent(prev_ctx ? prev_dpy : dev.dpy, nullptr, nullptr, prev_ctx);
        }
    }
};

static void ggml_opengl_release_current(gl_device_ctx & dev) {
    p_eglMakeCurrent(dev.dpy, nullptr, nullptr, nullptr);
}
#endif

static void ggml_opengl_check_error(const char * what) {
    for (int i = 0; i < 8; i++) {
        const GLenum e = glGetError();
        if (e == GL_NO_ERROR) {
            return;
        }
        GGML_LOG_ERROR("ggml_opengl: GL error 0x%04x after %s\n", (unsigned) e, what);
    }
}

/* Shader compilation and the program binary cache */

#ifndef _WIN32
// the cache code uses the Win32 wide-path calls; on Linux the paths are plain bytes
static std::string ggml_opengl_narrow(const wchar_t * s) {
    std::string r;
    for (; *s; s++) {
        r += (char) *s;
    }
    return r;
}
static FILE * _wfopen(const wchar_t * path, const wchar_t * mode) {
    return fopen(ggml_opengl_narrow(path).c_str(), ggml_opengl_narrow(mode).c_str());
}
static bool CreateDirectoryW(const wchar_t * path, void *) {
    return mkdir(ggml_opengl_narrow(path).c_str(), 0755) == 0;
}
static bool DeleteFileW(const wchar_t * path) {
    return unlink(ggml_opengl_narrow(path).c_str()) == 0;
}
#define MOVEFILE_REPLACE_EXISTING 0
static bool MoveFileExW(const wchar_t * from, const wchar_t * to, int) {
    return rename(ggml_opengl_narrow(from).c_str(), ggml_opengl_narrow(to).c_str()) == 0;
}
static unsigned GetCurrentProcessId() {
    return (unsigned) getpid();
}
#endif

// program binaries are cached in opengl-shader-cache next to ggml-opengl.dll; GGML_OPENGL_NO_SHADER_CACHE
// disables it. Unwritable folders just skip the cache.
static const std::wstring & ggml_opengl_shader_cache_dir() {
    static const std::wstring dir = []() -> std::wstring {
        if (getenv("GGML_OPENGL_NO_SHADER_CACHE") != nullptr) {
            return L"";
        }
#ifdef _WIN32
        HMODULE module = nullptr;
        wchar_t path[MAX_PATH];
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                (LPCWSTR) &ggml_opengl_shader_cache_dir, &module) ||
            GetModuleFileNameW(module, path, MAX_PATH) - 1 >= MAX_PATH - 1) {
            return L"";
        }
        std::wstring d = path;
#else
        Dl_info info = {};
        if (!dladdr((void *) &ggml_opengl_shader_cache_dir, &info) || !info.dli_fname) {
            return L"";
        }
        std::string  p = info.dli_fname;
        std::wstring d(p.begin(), p.end());
#endif
        d = d.substr(0, d.find_last_of(L"\\/") + 1) + L"opengl-shader-cache";
        CreateDirectoryW(d.c_str(), nullptr);
        return d;
    }();
    return dir;
}

// the file name is a hash of the full GLSL text (defines included), GL_RENDERER and GL_VERSION, so a
// kernel change recompiles only the programs it touches and a driver update recompiles everything.
// Do not add a global salt.
static std::wstring ggml_opengl_shader_cache_file(const gl_device_ctx & dev, const std::string & text) {
    const std::wstring & dir = ggml_opengl_shader_cache_dir();
    if (dir.empty() || !dev.prog_binary) {
        return L"";
    }
    uint64_t h = 0xcbf29ce484222325ULL;
    auto hash_bytes = [&h](const void * data, size_t size) {
        const unsigned char * b = (const unsigned char *) data;
        for (size_t i = 0; i < size; i++) {
            h = (h ^ b[i]) * 0x100000001b3ULL;
        }
    };
    hash_bytes(text.c_str(), text.size() + 1);
    hash_bytes(dev.desc.c_str(), dev.desc.size() + 1);
    hash_bytes(dev.version.c_str(), dev.version.size() + 1);
    std::wstring name = L"/0000000000000000.glbin";
    for (int i = 16; i > 0; i--, h >>= 4) {
        name[i] = L"0123456789abcdef"[h & 0xf];
    }
    return dir + name;
}

static std::vector<uint8_t> ggml_opengl_read_file(const std::wstring & file) {
    std::vector<uint8_t> data;
    FILE * f = file.empty() ? nullptr : _wfopen(file.c_str(), L"rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (size > 0) {
            data.resize((size_t) size);
            if (fread(data.data(), 1, data.size(), f) != data.size()) {
                data.clear();
            }
        }
        fclose(f);
    }
    return data;
}

// write to a per-process temp name and rename, so concurrent processes never see a partial file
static void ggml_opengl_write_file(const std::wstring & file, const void * data, size_t size) {
    if (file.empty()) {
        return;
    }
    const std::wstring tmp = file + L"." + std::to_wstring(GetCurrentProcessId()) + L".tmp";
    FILE * f = _wfopen(tmp.c_str(), L"wb");
    if (!f) {
        return;
    }
    const bool ok = fwrite(data, 1, size, f) == size;
    fclose(f);
    if (!ok || !MoveFileExW(tmp.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        DeleteFileW(tmp.c_str());
    }
}

// cache file layout: "GLPB", the binary format (4 bytes), the program binary
static bool ggml_opengl_load_binary(const std::wstring & file, GLuint prog) {
    const std::vector<uint8_t> data = ggml_opengl_read_file(file);
    if (data.size() <= 8 || memcmp(data.data(), "GLPB", 4) != 0) {
        return false;
    }
    GLenum fmt;
    memcpy(&fmt, data.data() + 4, 4);
    p_glProgramBinary(prog, fmt, data.data() + 8, (GLsizei) (data.size() - 8));
    GLint ok = 0;
    p_glGetProgramiv(prog, E_LINK_STATUS, &ok);
    while (glGetError() != GL_NO_ERROR) {}   // a rejected binary (driver update) may leave an error
    if (!ok) {
        DeleteFileW(file.c_str());
    }
    return ok != 0;
}

static void ggml_opengl_save_binary(const std::wstring & file, GLuint prog) {
    if (file.empty()) {
        return;
    }
    GLint len = 0;
    p_glGetProgramiv(prog, E_PROGRAM_BINARY_LENGTH, &len);
    if (len <= 0) {
        return;
    }
    std::vector<uint8_t> data(8 + (size_t) len);
    GLenum  fmt = 0;
    GLsizei got = 0;
    p_glGetProgramBinary(prog, len, &got, &fmt, data.data() + 8);
    if (got <= 0) {
        return;
    }
    memcpy(data.data(), "GLPB", 4);
    memcpy(data.data() + 4, &fmt, 4);
    ggml_opengl_write_file(file, data.data(), 8 + (size_t) got);
}

static std::string ggml_opengl_info_log(GLuint h, bool prog) {
    GLint n = 0;
    if (prog) {
        p_glGetProgramiv(h, E_INFO_LOG_LENGTH, &n);
    } else {
        p_glGetShaderiv(h, E_INFO_LOG_LENGTH, &n);
    }
    std::string s(std::max(n, 1), '\0');
    if (prog) {
        p_glGetProgramInfoLog(h, n, nullptr, &s[0]);
    } else {
        p_glGetShaderInfoLog(h, n, nullptr, &s[0]);
    }
    return s;
}

// the full source: #version, the work group size, the variant defines, then the kernel
static std::string ggml_opengl_shader_text(const char * source, const std::vector<std::string> & defines) {
    std::string text = "#version 430\n#define WG_SIZE " + std::to_string(GGML_GL_WG_SIZE) + "\n#define GGML_OPENGL 1\n";
    for (const auto & d : defines) {
        const size_t eq = d.find('=');
        text += "#define " + (eq == std::string::npos ? d + " 1" : d.substr(0, eq) + " " + d.substr(eq + 1)) + "\n";
    }
    return text + "#line 1\n" + source;
}

// each used shader adds a line "name<TAB>define..." to this file; the next process builds the listed
// programs at device init, all at once, so the driver compiles them in parallel
static std::wstring ggml_opengl_shader_list_file() {
    const std::wstring & dir = ggml_opengl_shader_cache_dir();
    return dir.empty() ? L"" : dir + L"/shaders.txt";
}

static std::mutex                  g_listed_mutex;
static std::map<std::string, bool> g_listed;

static void ggml_opengl_list_shader(const std::string & key, const std::string & name, const std::vector<std::string> & defines) {
    const std::wstring file = ggml_opengl_shader_list_file();
    std::lock_guard<std::mutex> lock(g_listed_mutex);
    if (file.empty() || g_listed[key]) {
        return;
    }
    g_listed[key] = true;
    std::string line = name;
    for (const auto & d : defines) {
        line += "\t" + d;
    }
    line += "\n";
    if (FILE * f = _wfopen(file.c_str(), L"ab")) {
        fwrite(line.data(), 1, line.size(), f);
        fclose(f);
    }
}

static std::string ggml_opengl_pipeline_key(const std::string & name, const std::vector<std::string> & defines) {
    std::string key = name;
    for (const auto & d : defines) {
        key += " -D" + d;
    }
    return key;
}

// Builds the queued programs: cache hits load their binary, the rest are all compiled and linked before
// any status is read, so a driver with parallel_shader_compile works on all of them at once. Needs the
// context current. skip_failed (prewarm): a program that no longer builds (a key listed by an older build) is
// dropped with a warning instead of aborting.
static void ggml_opengl_build_pipeline_jobs(gl_device_ctx & dev, bool skip_failed = false) {
    std::vector<gl_device_ctx::pipeline_job> jobs;
    jobs.swap(dev.pipeline_jobs);
    if (jobs.empty()) {
        return;
    }
    const double t0 = ggml_opengl_time_us();
    struct pending {
        size_t       job;
        GLuint       shader;
        GLuint       prog;
        std::wstring cache_file;
    };
    std::vector<pending> compiling;
    for (size_t i = 0; i < jobs.size(); i++) {
        const auto & j = jobs[i];
        if (dev.pipelines.count(j.key)) {
            continue;
        }
        const std::wstring cache_file = ggml_opengl_shader_cache_file(dev, j.text);
        GLuint             prog       = p_glCreateProgram();
        if (!cache_file.empty() && ggml_opengl_load_binary(cache_file, prog)) {
            dev.pipelines[j.key] = { prog, j.key };
            dev.n_cache_hits++;
            continue;
        }
        GLuint       shader = p_glCreateShader(E_COMPUTE_SHADER);
        const char * src    = j.text.c_str();
        p_glShaderSource(shader, 1, &src, nullptr);
        p_glCompileShader(shader);
        p_glAttachShader(prog, shader);
        if (!cache_file.empty() && p_glProgramParameteri) {
            p_glProgramParameteri(prog, E_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
        }
        p_glLinkProgram(prog);
        compiling.push_back({ i, shader, prog, cache_file });
    }
    for (const auto & c : compiling) {
        const auto & j  = jobs[c.job];
        GLint        ok = 0;
        p_glGetShaderiv(c.shader, E_COMPILE_STATUS, &ok);
        if (!ok && skip_failed) {
            GGML_LOG_WARN("ggml_opengl: listed program %s does not build any more, skipped\n", j.key.c_str());
            p_glDeleteProgram(c.prog);
            p_glDeleteShader(c.shader);
            continue;
        }
        if (!ok) {
            GGML_LOG_ERROR("ggml_opengl: shader compilation failed for %s:\n%s\n", j.key.c_str(),
                           ggml_opengl_info_log(c.shader, false).c_str());
            fflush(stderr);
            GGML_ABORT("ggml_opengl: shader compilation failed");
        }
        p_glGetProgramiv(c.prog, E_LINK_STATUS, &ok);
        if (!ok) {
            GGML_LOG_ERROR("ggml_opengl: program link failed for %s:\n%s\n", j.key.c_str(),
                           ggml_opengl_info_log(c.prog, true).c_str());
            fflush(stderr);
            GGML_ABORT("ggml_opengl: program link failed");
        }
        p_glDetachShader(c.prog, c.shader);
        p_glDeleteShader(c.shader);
        ggml_opengl_save_binary(c.cache_file, c.prog);
        dev.pipelines[j.key] = { c.prog, j.key };
        dev.n_compiles++;
    }
    for (const auto & j : jobs) {
        ggml_opengl_list_shader(j.key, j.name, j.defines);
    }
    dev.t_compile_us += ggml_opengl_time_us() - t0;
    if (!compiling.empty() && dev.debug) {
        GGML_LOG_INFO("ggml_opengl: built %zu programs from source in %.1f ms\n", compiling.size(),
                      (ggml_opengl_time_us() - t0) / 1000.0);
    }
}

static void ggml_opengl_queue_job(gl_device_ctx & dev, const std::string & key, const char * name, const char * source,
                                  const std::vector<std::string> & defines) {
    const bool queued = std::any_of(dev.pipeline_jobs.begin(), dev.pipeline_jobs.end(),
                                    [&](const gl_device_ctx::pipeline_job & j) { return j.key == key; });
    if (!queued) {
        dev.pipeline_jobs.push_back({ key, name, ggml_opengl_shader_text(source, defines), defines });
    }
}

// defines: list of "NAME" or "NAME=VALUE"
static gl_pipeline & ggml_opengl_get_pipeline(gl_device_ctx & dev, const char * name, const char * source,
                                              const std::vector<std::string> & defines) {
    const std::string key = ggml_opengl_pipeline_key(name, defines);
    auto it = dev.pipelines.find(key);
    if (it != dev.pipelines.end()) {
        return it->second;
    }
    ggml_opengl_queue_job(dev, key, name, source, defines);
    if (dev.collecting) {
        static gl_pipeline placeholder;
        return placeholder;
    }
    ggml_opengl_build_pipeline_jobs(dev);
    return dev.pipelines.at(key);
}

// builds the programs listed by earlier processes (see ggml_opengl_list_shader); needs the context current
static void ggml_opengl_prewarm(gl_device_ctx & dev) {
    const std::wstring file = ggml_opengl_shader_list_file();
    FILE * f = file.empty() ? nullptr : _wfopen(file.c_str(), L"rb");
    if (!f) {
        return;
    }
    char buf[4096];
    while (fgets(buf, sizeof(buf), f)) {
        std::string line = buf;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        std::vector<std::string> fields;
        for (size_t pos = 0; pos <= line.size();) {
            const size_t tab = std::min(line.find('\t', pos), line.size());
            fields.push_back(line.substr(pos, tab - pos));
            pos = tab + 1;
        }
        for (const auto & t : glsl_table) {
            if (fields[0] == t.name) {
                const std::vector<std::string> defines(fields.begin() + 1, fields.end());
                const std::string              key = ggml_opengl_pipeline_key(t.name, defines);
                {
                    std::lock_guard<std::mutex> lock(g_listed_mutex);
                    g_listed[key] = true;
                }
                ggml_opengl_queue_job(dev, key, t.name, t.source, defines);
            }
        }
    }
    fclose(f);
    if (!dev.pipeline_jobs.empty()) {
        GGML_LOG_INFO("ggml_opengl: building %zu listed programs\n", dev.pipeline_jobs.size());
        ggml_opengl_build_pipeline_jobs(dev, true);
    }
}

/* Dispatch encoding */

struct gl_binding {
    GLuint   buf;
    size_t   offset;        // aligned byte offset the SSBO range starts at
    size_t   size;          // to the end of the buffer
    uint32_t elem_offset;   // misalignment in elements, passed to the kernel
};

// Bind at the largest aligned offset at or before the tensor such that the distance is a whole number
// of elements, so the kernel can index the remainder in elements.
static gl_binding ggml_opengl_bind_tensor(const gl_device_ctx & dev, const ggml_tensor * t) {
    auto *       buf_ctx   = (ggml_backend_opengl_buffer_context *) t->buffer->context;
    const size_t offset    = ggml_opengl_tensor_offset(t);
    const size_t type_size = ggml_type_size(t->type);
    size_t       aligned   = offset & ~(dev.bind_align - 1);
    while ((offset - aligned) % type_size != 0) {
        GGML_ASSERT(aligned >= dev.bind_align);
        aligned -= dev.bind_align;
    }
    return { buf_ctx->buf, aligned, buf_ctx->size - aligned, (uint32_t) ((offset - aligned) / type_size) };
}

static inline uint32_t ggml_opengl_u32_from_f32(float value) {
    uint32_t u;
    memcpy(&u, &value, sizeof(u));
    return u;
}

// one dispatch (params get nwg_x appended), followed by a storage barrier so the next dispatch sees
// its writes. Needs the context current.
static void ggml_opengl_dispatch(gl_device_ctx & dev, gl_pipeline & pipeline, std::vector<uint32_t> params,
                                 const std::vector<gl_binding> & bindings, uint32_t total_wg) {
    if (dev.collecting) {
        return;
    }
    const uint32_t wg_y = std::max(1u, CEIL_DIV(total_wg, (uint32_t) GGML_GL_MAX_WG_PER_DIM));
    const uint32_t wg_x = CEIL_DIV(total_wg, wg_y);
    params.push_back(wg_x);
    GGML_ASSERT(params.size() * sizeof(uint32_t) <= GGML_GL_PARAM_SLOT_SIZE);

    const size_t slot_off = (size_t) dev.next_slot * dev.param_slot;
    if (dev.ubo_map) {
        constexpr uint32_t quarter = GGML_GL_PARAM_SLOT_COUNT / 4;
        if (dev.next_slot % quarter == 0) {
            // entering quarter q: fence the work so far (it covers the previous quarter's slots), then wait for
            // the fence set when quarter q was last left
            const uint32_t q    = dev.next_slot / quarter;
            gl_sync_t &    prev = dev.ubo_fence[(q + 3) % 4];
            if (prev) {
                p_glDeleteSync(prev);
            }
            prev = p_glFenceSync(E_SYNC_GPU_COMMANDS_COMPLETE, 0);
            if (dev.ubo_fence[q]) {
                GLenum r;
                while ((r = p_glClientWaitSync(dev.ubo_fence[q], E_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull)) == E_TIMEOUT_EXPIRED) {
                }
                GGML_ASSERT(r != E_WAIT_FAILED);
                p_glDeleteSync(dev.ubo_fence[q]);
                dev.ubo_fence[q] = nullptr;
            }
        }
        memcpy(dev.ubo_map + slot_off, params.data(), params.size() * sizeof(uint32_t));
    } else {
        p_glBindBuffer(E_UNIFORM_BUFFER, dev.ubo);
        p_glBufferSubData(E_UNIFORM_BUFFER, (ptrdiff_t) slot_off, (ptrdiff_t) (params.size() * sizeof(uint32_t)), params.data());
    }
    dev.next_slot = (dev.next_slot + 1) % GGML_GL_PARAM_SLOT_COUNT;
    p_glBindBufferRange(E_UNIFORM_BUFFER, 0, dev.ubo, (ptrdiff_t) slot_off, GGML_GL_PARAM_SLOT_SIZE);

    if (dev.cur_prog != pipeline.prog) {
        p_glUseProgram(pipeline.prog);
        dev.cur_prog = pipeline.prog;
    }
    if (dev.cur_bind.size() < bindings.size()) {
        dev.cur_bind.resize(bindings.size());
    }
    for (size_t i = 0; i < bindings.size(); i++) {
        const gl_binding &           b = bindings[i];
        gl_device_ctx::bind_state & c = dev.cur_bind[i];
        if (c.buf != b.buf || c.offset != b.offset || c.size != b.size) {
            p_glBindBufferRange(E_SHADER_STORAGE_BUFFER, (GLuint) i, b.buf, (ptrdiff_t) b.offset, (ptrdiff_t) b.size);
            c = { b.buf, b.offset, b.size };
        }
    }
    if (total_wg > 0) {
        GLuint q = 0;
        if (dev.profile && p_glBeginQuery) {
            if (dev.q_free.empty()) {
                dev.q_free.resize(256);
                p_glGenQueries(256, dev.q_free.data());
            }
            q = dev.q_free.back();
            dev.q_free.pop_back();
            p_glBeginQuery(E_TIME_ELAPSED, q);
        }
        p_glDispatchCompute(wg_x, wg_y, 1);
        if (q) {
            p_glEndQuery(E_TIME_ELAPSED);
            dev.q_pending.emplace_back(q, pipeline.name);
        }
        dev.n_dispatches++;
    }
    p_glMemoryBarrier(E_SHADER_STORAGE_BARRIER_BIT);
    dev.n_barriers++;
    if (dev.debug) {
        ggml_opengl_check_error(pipeline.name.c_str());
    }
}

/* Op encoders */

static std::string ggml_opengl_type_define(ggml_type type, const char * prefix) {
    std::string s = prefix;
    switch (type) {
        case GGML_TYPE_F32: s += "_F32"; break;
        case GGML_TYPE_F16: s += "_F16"; break;
        case GGML_TYPE_BF16: s += "_BF16"; break;
        case GGML_TYPE_I32: s += "_I32"; break;
        case GGML_TYPE_Q4_0: s += "_Q4_0"; break;
        case GGML_TYPE_Q4_1: s += "_Q4_1"; break;
        case GGML_TYPE_Q5_0: s += "_Q5_0"; break;
        case GGML_TYPE_Q5_1: s += "_Q5_1"; break;
        case GGML_TYPE_Q5_K: s += "_Q5_K"; break;
        case GGML_TYPE_Q4_K: s += "_Q4_K"; break;
        case GGML_TYPE_Q8_0: s += "_Q8_0"; break;
        case GGML_TYPE_Q6_K: s += "_Q6_K"; break;
        case GGML_TYPE_Q2_K: s += "_Q2_K"; break;
        case GGML_TYPE_Q3_K: s += "_Q3_K"; break;
        case GGML_TYPE_IQ4_NL: s += "_IQ4_NL"; break;
        case GGML_TYPE_IQ4_XS: s += "_IQ4_XS"; break;
        case GGML_TYPE_IQ3_S: s += "_IQ3_S"; break;
        case GGML_TYPE_IQ2_S: s += "_IQ2_S"; break;
        case GGML_TYPE_IQ2_XXS: s += "_IQ2_XXS"; break;
        case GGML_TYPE_IQ2_XS: s += "_IQ2_XS"; break;
        case GGML_TYPE_IQ3_XXS: s += "_IQ3_XXS"; break;
        case GGML_TYPE_IQ1_S: s += "_IQ1_S"; break;
        case GGML_TYPE_IQ1_M: s += "_IQ1_M"; break;
        case GGML_TYPE_MXFP4: s += "_MXFP4"; break;
        case GGML_TYPE_TQ2_0: s += "_TQ2_0"; break;
        case GGML_TYPE_Q1_0: s += "_Q1_0"; break;
        case GGML_TYPE_Q2_0: s += "_Q2_0"; break;
        default: GGML_ABORT("ggml_opengl: unsupported type %s", ggml_type_name(type));
    }
    return s;
}

static void ggml_opengl_cpy(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    const std::vector<std::string> defines = { ggml_opengl_type_define(src->type, "SRC"), ggml_opengl_type_define(dst->type, "DST") };
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "cpy", glsl_cpy, defines);

    const gl_binding bsrc = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bdst = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   ne   = (uint32_t) ggml_nelements(dst);
    const size_t     ts   = ggml_type_size(src->type);
    const size_t     td   = ggml_type_size(dst->type);

    std::vector<uint32_t> params = {
        ne, bsrc.elem_offset, bdst.elem_offset,
        (uint32_t) (src->nb[0] / ts), (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (dst->nb[0] / td), (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bsrc, bdst }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_binary_op(gl_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    const std::string op_define = std::string("OP_") + ggml_op_name(dst->op);   // OP_ADD, OP_SUB, OP_MUL, OP_DIV
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "binary", glsl_binary,
                                                      { ggml_opengl_type_define(dst->type, "TYPE"), op_define });

    const gl_binding b0 = ggml_opengl_bind_tensor(dev, src0);
    const gl_binding b1 = ggml_opengl_bind_tensor(dev, src1);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   ne = (uint32_t) ggml_nelements(dst);
    const size_t     t0 = ggml_type_size(src0->type);
    const size_t     t1 = ggml_type_size(src1->type);

    std::vector<uint32_t> params = {
        ne, b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[0] / t0), (uint32_t) (src0->nb[1] / t0), (uint32_t) (src0->nb[2] / t0), (uint32_t) (src0->nb[3] / t0),
        (uint32_t) (src1->nb[0] / t1), (uint32_t) (src1->nb[1] / t1), (uint32_t) (src1->nb[2] / t1), (uint32_t) (src1->nb[3] / t1),
        (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0], (uint32_t) src1->ne[1], (uint32_t) src1->ne[2], (uint32_t) src1->ne[3],
    };
    ggml_opengl_dispatch(dev, pipeline, params, { b0, b1, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_scale(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "scale", glsl_scale, {});

    const gl_binding bs = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   ne = (uint32_t) ggml_nelements(dst);
    const size_t     ts = ggml_type_size(src->type);
    const size_t     td = ggml_type_size(dst->type);

    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        ne, (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2],
        ggml_opengl_u32_from_f32(ggml_get_op_params_f32(dst, 0)),   // scale
        ggml_opengl_u32_from_f32(ggml_get_op_params_f32(dst, 1)),   // bias
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_fill(gl_device_ctx & dev, ggml_tensor * dst) {
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "fill", glsl_fill,
                                                      { dst->type == GGML_TYPE_F16 ? "DST_F16" : "DST_F32" });

    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        bd.elem_offset, ne, ggml_opengl_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_concat(gl_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "concat", glsl_concat,
                                                      { ggml_type_size(dst->type) == 2 ? "ELEM16" : "ELEM32" });

    const gl_binding b0  = ggml_opengl_bind_tensor(dev, src0);
    const gl_binding b1  = ggml_opengl_bind_tensor(dev, src1);
    const gl_binding bd  = ggml_opengl_bind_tensor(dev, dst);
    const int32_t    dim = ggml_get_op_params_i32(dst, 0);
    const uint32_t   ne  = (uint32_t) ggml_nelements(dst);
    const size_t     ts  = ggml_type_size(dst->type);

    std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[0] / ts), (uint32_t) (src0->nb[1] / ts), (uint32_t) (src0->nb[2] / ts), (uint32_t) (src0->nb[3] / ts),
        (uint32_t) (src1->nb[0] / ts), (uint32_t) (src1->nb[1] / ts), (uint32_t) (src1->nb[2] / ts), (uint32_t) (src1->nb[3] / ts),
        (uint32_t) (dst->nb[0] / ts), (uint32_t) (dst->nb[1] / ts), (uint32_t) (dst->nb[2] / ts), (uint32_t) (dst->nb[3] / ts),
        ne, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) dim, (uint32_t) src0->ne[dim],
    };
    ggml_opengl_dispatch(dev, pipeline, params, { b0, b1, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_pad(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (ggml_get_op_params_i32(dst, 8) != 0) {
        defines.push_back("CIRCULAR");
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "pad", glsl_pad, defines);

    const gl_binding bs = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[0] / 4), (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], ne,
        (uint32_t) ggml_get_op_params_i32(dst, 0), (uint32_t) ggml_get_op_params_i32(dst, 2),
        (uint32_t) ggml_get_op_params_i32(dst, 4), (uint32_t) ggml_get_op_params_i32(dst, 6),
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

// ROLL: cyclic shift along all four axes, one workgroup per destination row (port of the D3D12 encoder)
static void ggml_opengl_roll(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    gl_pipeline &    pipeline = ggml_opengl_get_pipeline(dev, "roll", glsl_roll, {});
    const gl_binding bs       = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bd       = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   n_rows   = (uint32_t) ggml_nrows(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3],
        (uint32_t) ggml_get_op_params_i32(dst, 0), (uint32_t) ggml_get_op_params_i32(dst, 1),
        (uint32_t) ggml_get_op_params_i32(dst, 2), (uint32_t) ggml_get_op_params_i32(dst, 3),
        n_rows,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bd }, n_rows);
}

static void ggml_opengl_im2col(gl_device_ctx & dev, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * src    = dst->src[1];
    const int32_t *     op     = (const int32_t *) dst->op_params;
    const bool          is_2D  = op[6] == 1;

    std::vector<std::string> defines;
    if (dst->type == GGML_TYPE_F16) {
        defines.push_back("DST_F16");
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "im2col", glsl_im2col, defines);

    const gl_binding bs = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[is_2D ? 3 : 2] / 4), (uint32_t) (src->nb[is_2D ? 2 : 1] / 4),
        is_2D ? (uint32_t) (src->nb[1] / 4) : 0u, (uint32_t) (src->nb[0] / 4),
        (uint32_t) op[0], (uint32_t) op[1], (uint32_t) op[2], (uint32_t) op[3], (uint32_t) op[4], (uint32_t) op[5],
        (uint32_t) (is_2D ? src->ne[2] : src->ne[1]), (uint32_t) (is_2D ? src->ne[1] : 1), (uint32_t) src->ne[0],
        (uint32_t) (is_2D ? kernel->ne[1] : 1), (uint32_t) kernel->ne[0],
        (uint32_t) (is_2D ? dst->ne[2] : 1), (uint32_t) dst->ne[1], ne,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_upscale(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    const int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    float sf0 = (float) dst->ne[0] / src->ne[0];
    float sf1 = (float) dst->ne[1] / src->ne[1];
    const float sf2 = (float) dst->ne[2] / src->ne[2];
    const float sf3 = (float) dst->ne[3] / src->ne[3];
    float pixel_offset = 0.5f;
    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = dst->ne[0] > 1 && src->ne[0] > 1 ? (float) (dst->ne[0] - 1) / (src->ne[0] - 1) : sf0;
        sf1 = dst->ne[1] > 1 && src->ne[1] > 1 ? (float) (dst->ne[1] - 1) / (src->ne[1] - 1) : sf1;
    }
    std::vector<std::string> defines;
    if ((mode_flags & 0xFF) == GGML_SCALE_MODE_BILINEAR) {
        defines.push_back("BILINEAR");
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "upscale", glsl_upscale, defines);

    const gl_binding bs = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[0] / 4), (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], ne,
        ggml_opengl_u32_from_f32(sf0), ggml_opengl_u32_from_f32(sf1), ggml_opengl_u32_from_f32(sf2), ggml_opengl_u32_from_f32(sf3),
        ggml_opengl_u32_from_f32(pixel_offset),
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_pool_2d(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    const int32_t * op = (const int32_t *) dst->op_params;
    std::vector<std::string> defines;
    if (src->type == GGML_TYPE_F16) {
        defines.push_back("SRC_F16");
    }
    if ((ggml_op_pool) op[0] == GGML_OP_POOL_MAX) {
        defines.push_back("POOL_MAX");
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "pool_2d", glsl_pool_2d, defines);

    const gl_binding bs = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const size_t     ts = ggml_type_size(src->type);
    const uint32_t   ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
        (uint32_t) op[1], (uint32_t) op[2], (uint32_t) op[3], (uint32_t) op[4], (uint32_t) op[5], (uint32_t) op[6],
        ne,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_ssm_conv(gl_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "ssm_conv", glsl_ssm_conv, {});

    const gl_binding b0 = ggml_opengl_bind_tensor(dev, src0);
    const gl_binding b1 = ggml_opengl_bind_tensor(dev, src1);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src1->nb[1] / 4),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4),
        (uint32_t) src1->ne[0], (uint32_t) src0->ne[1], (uint32_t) dst->ne[1], ne,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { b0, b1, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

// GROUP_NORM (f32), port of the D3D12 path: one workgroup per (group, i03)
static void ggml_opengl_group_norm(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "group_norm", glsl_group_norm, {});

    const gl_binding bs = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t n_channels = (uint32_t) src->ne[2];
    const uint32_t n_groups   = (uint32_t) ggml_get_op_params_i32(dst, 0);
    const uint32_t n_batches  = (uint32_t) src->ne[3];
    float eps;
    memcpy(&eps, (const int32_t *) dst->op_params + 1, sizeof(float));

    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1],
        n_channels, n_groups, CEIL_DIV(n_channels, n_groups), n_batches,
        ggml_opengl_u32_from_f32(eps),
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bd }, n_groups * n_batches);
}

// ADD_ID (f32), port of the D3D12 path: dst[i0, i1, i2] = src0[i0, i1, i2] + src1[i0, ids[i1, i2]]
static void ggml_opengl_add_id(gl_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * ids,
                               ggml_tensor * dst) {
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "add_id", glsl_add_id, {});

    const gl_binding b0 = ggml_opengl_bind_tensor(dev, src0);
    const gl_binding b1 = ggml_opengl_bind_tensor(dev, src1);
    const gl_binding bi = ggml_opengl_bind_tensor(dev, ids);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src1->nb[1] / 4),
        (uint32_t) (ids->nb[1] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], ne,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { b0, b1, bi, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

// SSM_SCAN, port of the D3D12 path: one thread per (sequence, head, dim); the token loop stays inside the thread
static void ggml_opengl_ssm_scan(gl_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * s0  = dst->src[0];
    ggml_tensor * x   = dst->src[1];
    ggml_tensor * dt  = dst->src[2];
    ggml_tensor * A   = dst->src[3];
    ggml_tensor * B   = dst->src[4];
    ggml_tensor * C   = dst->src[5];
    ggml_tensor * ids = dst->src[6];

    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "ssm_scan", glsl_ssm_scan, {});

    const gl_binding b0 = ggml_opengl_bind_tensor(dev, s0);
    const gl_binding b1 = ggml_opengl_bind_tensor(dev, x);
    const gl_binding b2 = ggml_opengl_bind_tensor(dev, dt);
    const gl_binding b3 = ggml_opengl_bind_tensor(dev, A);
    const gl_binding b4 = ggml_opengl_bind_tensor(dev, B);
    const gl_binding b5 = ggml_opengl_bind_tensor(dev, C);
    const gl_binding b6 = ggml_opengl_bind_tensor(dev, ids);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);

    const uint32_t nc     = (uint32_t) s0->ne[0];
    const uint32_t nr     = (uint32_t) s0->ne[1];
    const uint32_t nh     = (uint32_t) x->ne[1];
    const uint32_t ng     = (uint32_t) B->ne[1];
    const uint32_t nt     = (uint32_t) x->ne[2];
    const uint32_t ns     = (uint32_t) x->ne[3];
    const uint32_t n_jobs = ns * nh * nr;

    std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, b2.elem_offset, b3.elem_offset,
        b4.elem_offset, b5.elem_offset, b6.elem_offset, bd.elem_offset,
        (uint32_t) (s0->nb[3] / 4),
        (uint32_t) (x->nb[2] / 4), (uint32_t) (x->nb[3] / 4),
        (uint32_t) (dt->nb[1] / 4), (uint32_t) (dt->nb[2] / 4),
        (uint32_t) (B->nb[2] / 4), (uint32_t) (B->nb[3] / 4),
        (uint32_t) (C->nb[2] / 4), (uint32_t) (C->nb[3] / 4),
        nc, nr, nh, ng, nt, ns,
        (uint32_t) ggml_get_op_params_i32(dst, 0),
        (uint32_t) ggml_nelements(x),          // s_off, in elements
        (uint32_t) (A->ne[0] == 1 ? 1 : 0),    // Mamba-2 has a scalar decay per head
        n_jobs,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { b0, b1, b2, b3, b4, b5, b6, bd },
                         CEIL_DIV(n_jobs, (uint32_t) GGML_GL_WG_SIZE));
}

// CONV_2D_DW (WHCN, f32 or f16 kernel), port of the D3D12 path: one thread per destination element
static void ggml_opengl_conv_2d_dw(gl_device_ctx & dev, ggml_tensor * knl, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (knl->type == GGML_TYPE_F16) {
        defines = { "KNL_F16" };
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, knl->type == GGML_TYPE_F16 ? "conv_2d_dw_f16" : "conv_2d_dw",
                                                      glsl_conv_2d_dw, defines);

    const gl_binding bk   = ggml_opengl_bind_tensor(dev, knl);
    const gl_binding bs   = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bd   = ggml_opengl_bind_tensor(dev, dst);
    const int32_t *  opts = (const int32_t *) dst->op_params;
    const uint32_t   ne   = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        bk.elem_offset, bs.elem_offset, bd.elem_offset,
        (uint32_t) src->ne[0], (uint32_t) src->ne[1],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
        (uint32_t) knl->ne[0], (uint32_t) knl->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) opts[0], (uint32_t) opts[1], (uint32_t) opts[2],
        (uint32_t) opts[3], (uint32_t) opts[4], (uint32_t) opts[5],
        ne,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bk, bs, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

static bool ggml_opengl_unary_supported(ggml_unary_op op) {
    switch (op) {
        case GGML_UNARY_OP_ABS: case GGML_UNARY_OP_SGN: case GGML_UNARY_OP_NEG: case GGML_UNARY_OP_STEP:
        case GGML_UNARY_OP_TANH: case GGML_UNARY_OP_ELU: case GGML_UNARY_OP_RELU: case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_GELU: case GGML_UNARY_OP_GELU_QUICK: case GGML_UNARY_OP_GELU_ERF: case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_HARDSWISH: case GGML_UNARY_OP_HARDSIGMOID: case GGML_UNARY_OP_EXP: case GGML_UNARY_OP_SOFTPLUS:
        case GGML_UNARY_OP_EXPM1: case GGML_UNARY_OP_FLOOR: case GGML_UNARY_OP_CEIL: case GGML_UNARY_OP_ROUND:
        case GGML_UNARY_OP_TRUNC:
            return true;
        default:
            return false;
    }
}

// GGML_OP_UNARY and the standalone element-wise ops (CLAMP, SQR, SQRT, SIN, COS, LOG): the op name is the define
static void ggml_opengl_unary(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    const char *  op_define = dst->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(dst)) : ggml_op_name(dst->op);
    gl_pipeline & pipeline  = ggml_opengl_get_pipeline(dev, "unary", glsl_unary, { ggml_opengl_type_define(dst->type, "TYPE"), op_define });
    const bool    clamp     = dst->op == GGML_OP_CLAMP;

    const gl_binding bs = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const size_t     ts = ggml_type_size(src->type);
    const uint32_t   ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne, bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[0] / ts), (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2],
        clamp ? ggml_opengl_u32_from_f32(ggml_get_op_params_f32(dst, 0)) : 0u,   // clamp min
        clamp ? ggml_opengl_u32_from_f32(ggml_get_op_params_f32(dst, 1)) : 0u,   // clamp max
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_repeat(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "repeat", glsl_repeat, {});

    const gl_binding bs = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[0] / 4), (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], ne,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

// one workgroup per state row (see gated_delta_net.comp), all tokens in one dispatch
static void ggml_opengl_hc_post(gl_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * x    = dst->src[0];
    ggml_tensor * r    = dst->src[1];
    ggml_tensor * post = dst->src[2];
    ggml_tensor * comb = dst->src[3];

    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "hc_post", glsl_hc_post, { comb ? "HAS_COMB" : "NO_COMB" });

    const ggml_tensor * c     = comb ? comb : r;   // without comb the slot is unused; bind residual again
    const gl_binding    bx    = ggml_opengl_bind_tensor(dev, x);
    const gl_binding    br    = ggml_opengl_bind_tensor(dev, r);
    const gl_binding    bp    = ggml_opengl_bind_tensor(dev, post);
    const gl_binding    bc    = ggml_opengl_bind_tensor(dev, c);
    const gl_binding    bd    = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t      n_out = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        bx.elem_offset, br.elem_offset, bp.elem_offset, bc.elem_offset, bd.elem_offset,
        (uint32_t) (x->nb[1] / 4), (uint32_t) (r->nb[1] / 4), (uint32_t) (r->nb[2] / 4), (uint32_t) (post->nb[1] / 4),
        (uint32_t) (c->nb[1] / 4), (uint32_t) (c->nb[2] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4),
        (uint32_t) x->ne[0], (uint32_t) r->ne[1], n_out,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bx, br, bp, bc, bd }, CEIL_DIV(n_out, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_hc_comb(gl_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * m = dst->src[0];
    ggml_tensor * s = dst->src[1];
    ggml_tensor * b = dst->src[2];

    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "hc_comb", glsl_hc_comb, {});

    const gl_binding bm       = ggml_opengl_bind_tensor(dev, m);
    const gl_binding bs       = ggml_opengl_bind_tensor(dev, s);
    const gl_binding bb       = ggml_opengl_bind_tensor(dev, b);
    const gl_binding bd       = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   n_tokens = (uint32_t) m->ne[1];

    std::vector<uint32_t> params = {
        bm.elem_offset, bs.elem_offset, bb.elem_offset, bd.elem_offset,
        (uint32_t) (m->nb[1] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4),
        n_tokens, ggml_opengl_u32_from_f32(ggml_get_op_params_f32(dst, 0)), (uint32_t) ggml_get_op_params_i32(dst, 1),
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bm, bs, bb, bd }, CEIL_DIV(n_tokens, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_hc_pre(gl_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * x     = dst->src[0];
    ggml_tensor * w     = dst->src[1];
    const bool    gated = ggml_get_op_params_i32(dst, 1) != 0;

    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "hc_pre", glsl_hc_pre, { gated ? "GATED" : "PLAIN" });

    const gl_binding bx    = ggml_opengl_bind_tensor(dev, x);
    const gl_binding bw    = ggml_opengl_bind_tensor(dev, w);
    const gl_binding bd    = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   n_out = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        bx.elem_offset, bw.elem_offset, bd.elem_offset,
        (uint32_t) (x->nb[1] / 4), (uint32_t) (x->nb[2] / 4), (uint32_t) (w->nb[1] / 4), (uint32_t) (w->nb[2] / 4),
        (uint32_t) (dst->nb[1] / 4),
        (uint32_t) x->ne[0], (uint32_t) x->ne[1], n_out, ggml_opengl_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bx, bw, bd }, CEIL_DIV(n_out, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_lightning_indexer(gl_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * q = dst->src[0];
    ggml_tensor * k = dst->src[1];
    ggml_tensor * w = dst->src[2];
    ggml_tensor * m = dst->src[3];

    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "lightning_indexer", glsl_lightning_indexer,
                                                      { k->type == GGML_TYPE_F16 ? "K_F16" : "K_F32" });

    const gl_binding bq    = ggml_opengl_bind_tensor(dev, q);
    const gl_binding bk    = ggml_opengl_bind_tensor(dev, k);
    const gl_binding bw    = ggml_opengl_bind_tensor(dev, w);
    const gl_binding bm    = ggml_opengl_bind_tensor(dev, m);
    const gl_binding bd    = ggml_opengl_bind_tensor(dev, dst);
    const size_t     tk    = ggml_type_size(k->type);
    const uint32_t   n_out = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        bq.elem_offset, bk.elem_offset, bw.elem_offset, bm.elem_offset, bd.elem_offset,
        (uint32_t) (q->nb[1] / 4), (uint32_t) (q->nb[2] / 4), (uint32_t) (q->nb[3] / 4),
        (uint32_t) (k->nb[2] / tk), (uint32_t) (k->nb[3] / tk),
        (uint32_t) (w->nb[1] / 4), (uint32_t) (w->nb[3] / 4),
        (uint32_t) (m->nb[1] / 2), (uint32_t) (m->nb[3] / 2),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) q->ne[0], (uint32_t) q->ne[1], (uint32_t) k->ne[2], (uint32_t) q->ne[2], (uint32_t) m->ne[3], n_out,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bq, bk, bw, bm, bd }, CEIL_DIV(n_out, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_gated_delta_net(gl_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * q = dst->src[0];
    ggml_tensor * k = dst->src[1];
    ggml_tensor * v = dst->src[2];
    ggml_tensor * g = dst->src[3];
    ggml_tensor * b = dst->src[4];
    ggml_tensor * s = dst->src[5];

    const uint32_t S_v      = (uint32_t) v->ne[0];
    const uint32_t H        = (uint32_t) v->ne[1];
    const uint32_t n_tokens = (uint32_t) v->ne[2];
    const uint32_t n_seqs   = (uint32_t) v->ne[3];
    const uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    uint32_t       red      = 1;
    while (red < S_v) {
        red *= 2;
    }

    std::vector<std::string> defines = { "S_V=" + std::to_string(S_v), "RED=" + std::to_string(red) };
    if (g->ne[0] == S_v) {
        defines.push_back("KDA");
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "gated_delta_net", glsl_gated_delta_net, defines);

    const gl_binding bq = ggml_opengl_bind_tensor(dev, q);
    const gl_binding bk = ggml_opengl_bind_tensor(dev, k);
    const gl_binding bv = ggml_opengl_bind_tensor(dev, v);
    const gl_binding bg = ggml_opengl_bind_tensor(dev, g);
    const gl_binding bb = ggml_opengl_bind_tensor(dev, b);
    const gl_binding bs = ggml_opengl_bind_tensor(dev, s);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   n_rows = n_seqs * H * S_v;

    std::vector<uint32_t> params = {
        bq.elem_offset, bk.elem_offset, bv.elem_offset, bg.elem_offset, bb.elem_offset, bs.elem_offset, bd.elem_offset,
        (uint32_t) (q->nb[1] / 4), (uint32_t) (q->nb[2] / 4), (uint32_t) (q->nb[3] / 4),
        (uint32_t) (k->nb[1] / 4), (uint32_t) (k->nb[2] / 4), (uint32_t) (k->nb[3] / 4),
        (uint32_t) (v->nb[1] / 4), (uint32_t) (v->nb[2] / 4), (uint32_t) (v->nb[3] / 4),
        (uint32_t) (g->nb[1] / 4), (uint32_t) (g->nb[2] / 4), (uint32_t) (g->nb[3] / 4),
        (uint32_t) (b->nb[1] / 4), (uint32_t) (b->nb[2] / 4), (uint32_t) (b->nb[3] / 4),
        H, n_tokens, (uint32_t) q->ne[1], (uint32_t) k->ne[1],
        (uint32_t) (v->ne[3] / q->ne[3]), (uint32_t) (v->ne[3] / k->ne[3]),
        (uint32_t) (s->nb[3] / 4), K,
        n_rows, ggml_opengl_u32_from_f32(1.0f / sqrtf((float) S_v)),
        bd.elem_offset + S_v * H * n_tokens * n_seqs, S_v * S_v * H * n_seqs,   // offset_st, st_slot_stride
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bq, bk, bv, bg, bb, bs, bd }, n_rows);
}

// RMS_NORM, and with the op name as define L2_NORM, NORM, SUM_ROWS, MEAN (all one workgroup per row).
// With w, an RMS_NORM fused with the following MUL by one f32 weight row: dst is the MUL's output.
static void ggml_opengl_rms_norm(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst, ggml_tensor * norm = nullptr,
                                 ggml_tensor * w = nullptr) {
    norm = norm ? norm : dst;   // the node that holds the op and eps
    std::vector<std::string> defines;
    if (norm->op != GGML_OP_RMS_NORM) {
        defines.push_back(ggml_op_name(norm->op));
    }
    if (w) {
        defines.push_back("MUL_W");
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "rms_norm", glsl_rms_norm, defines);

    const gl_binding bs     = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bd     = ggml_opengl_bind_tensor(dev, dst);
    const gl_binding bw     = w ? ggml_opengl_bind_tensor(dev, w) : bs;
    const uint32_t   n_rows = (uint32_t) ggml_nrows(src);

    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], n_rows,
        ggml_opengl_u32_from_f32(ggml_get_op_params_f32(norm, 0)),   // eps
        bw.elem_offset,
    };
    // one workgroup per row
    if (w) {
        ggml_opengl_dispatch(dev, pipeline, params, { bs, bd, bw }, n_rows);
    } else {
        ggml_opengl_dispatch(dev, pipeline, params, { bs, bd }, n_rows);
    }
}

static void ggml_opengl_glu(gl_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    std::vector<std::string> defines = { ggml_opengl_type_define(dst->type, "TYPE"),
                                         std::string("OP_") + ggml_glu_op_name(ggml_get_glu_op(dst)) };
    if (!src1) {
        defines.push_back("NO_SPLIT");
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "glu", glsl_glu, defines);

    const gl_binding    b0 = ggml_opengl_bind_tensor(dev, src0);
    const ggml_tensor * s1 = src1 ? src1 : src0;
    // NO_SPLIT reads only src0; bind it again in the unused slot
    const gl_binding    b1 = ggml_opengl_bind_tensor(dev, s1);
    const gl_binding    bd = ggml_opengl_bind_tensor(dev, dst);
    const size_t        ts = ggml_type_size(dst->type);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[1] / ts), (uint32_t) (src0->nb[2] / ts), (uint32_t) (src0->nb[3] / ts),
        (uint32_t) (s1->nb[1] / ts), (uint32_t) (s1->nb[2] / ts), (uint32_t) (s1->nb[3] / ts),
        (uint32_t) (dst->nb[1] / ts), (uint32_t) (dst->nb[2] / ts), (uint32_t) (dst->nb[3] / ts),
        ne, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) ((int32_t *) dst->op_params)[1],                  // swapped
        ggml_opengl_u32_from_f32(ggml_get_op_params_f32(dst, 2)),   // alpha
        ggml_opengl_u32_from_f32(ggml_get_op_params_f32(dst, 3)),   // limit
    };
    ggml_opengl_dispatch(dev, pipeline, params, { b0, b1, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_set_rows(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
    if (ggml_is_empty(src) || ggml_is_empty(idx)) {
        return;
    }
    std::vector<std::string> defines = { ggml_opengl_type_define(dst->type, "DST") };
    if (idx->type == GGML_TYPE_I64) {
        defines.push_back("I64_IDX");
    }
    if (src->type == GGML_TYPE_F16) {
        defines.push_back("SRC_F16");
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "set_rows", glsl_set_rows, defines);

    const gl_binding bs = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bi = ggml_opengl_bind_tensor(dev, idx);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const size_t     ts = ggml_type_size(src->type);
    const size_t     ti = ggml_type_size(idx->type);
    const size_t     td = ggml_type_size(dst->type);
    const uint32_t   ne = (uint32_t) ggml_nelements(src);

    std::vector<uint32_t> params = {
        bs.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (idx->nb[0] / ti), (uint32_t) (idx->nb[1] / ti), (uint32_t) (idx->nb[2] / ti),
        (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        (uint32_t) idx->ne[1], (uint32_t) idx->ne[2],
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bi, bd }, CEIL_DIV(ne, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_soft_max(gl_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * src2, ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (src1) {
        defines.push_back("HAS_MASK");
        defines.push_back(src1->type == GGML_TYPE_F16 ? "MASK_F16" : "MASK_F32");
    }
    if (src2) {
        defines.push_back("HAS_SINK");
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "soft_max", glsl_soft_max, defines);

    // a missing mask or sink tensor: src0 is bound in its slot and never read
    const gl_binding b0 = ggml_opengl_bind_tensor(dev, src0);
    const gl_binding b1 = ggml_opengl_bind_tensor(dev, src1 ? src1 : src0);
    const gl_binding b2 = ggml_opengl_bind_tensor(dev, src2 ? src2 : src0);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const size_t     t1 = src1 ? ggml_type_size(src1->type) : 4;

    const float    max_bias    = ggml_get_op_params_f32(dst, 1);
    const float    n_head_log2 = (float) (1u << (uint32_t) floor(log2((double) src0->ne[2])));
    const float    m0          = powf(2.0f, -(max_bias) / n_head_log2);
    const float    m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const uint32_t n_rows      = (uint32_t) ggml_nrows(dst);

    std::vector<uint32_t> params = {
        b0.elem_offset, src1 ? b1.elem_offset : 0u, src2 ? b2.elem_offset : 0u, bd.elem_offset,
        (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src0->nb[3] / 4),
        src1 ? (uint32_t) (src1->nb[1] / t1) : 0u, src1 ? (uint32_t) (src1->nb[2] / t1) : 0u, src1 ? (uint32_t) (src1->nb[3] / t1) : 0u,
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], (uint32_t) src0->ne[2],
        src1 ? (uint32_t) src1->ne[2] : 1u, src1 ? (uint32_t) src1->ne[3] : 1u,
        ggml_opengl_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
        ggml_opengl_u32_from_f32(max_bias), ggml_opengl_u32_from_f32(n_head_log2),
        ggml_opengl_u32_from_f32(m0), ggml_opengl_u32_from_f32(m1),
        n_rows,
    };
    // one workgroup per row
    ggml_opengl_dispatch(dev, pipeline, params, { b0, b1, b2, bd }, n_rows);
}

static void ggml_opengl_rope(gl_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * src2, ggml_tensor * dst) {
    std::vector<std::string> defines = { ggml_opengl_type_define(dst->type, "TYPE") };
    if (src2) {
        defines.push_back("FF_FUNC");
    }
    if (dst->op == GGML_OP_ROPE_BACK) {
        defines.push_back("BACKWARD");
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "rope", glsl_rope, defines);

    const gl_binding b0 = ggml_opengl_bind_tensor(dev, src0);
    const gl_binding b1 = ggml_opengl_bind_tensor(dev, src1);
    // no frequency factors: src1 is bound in their slot and never read
    const gl_binding b2 = ggml_opengl_bind_tensor(dev, src2 ? src2 : src1);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const size_t     ts = ggml_type_size(src0->type);

    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params + 5,  sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params + 6,  sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params + 7,  sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8,  sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params + 9,  sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    int sections[4];
    memcpy(sections, (int32_t *) dst->op_params + 11, 4 * sizeof(int));
    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const uint32_t n_threads = (uint32_t) (ggml_nelements(src0) / 2);
    std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, src2 ? b2.elem_offset : 0u, bd.elem_offset,
        (uint32_t) (src0->nb[1] / ts), (uint32_t) (src0->nb[2] / ts), (uint32_t) (src0->nb[3] / ts),
        (uint32_t) (dst->nb[1] / ts), (uint32_t) (dst->nb[2] / ts), (uint32_t) (dst->nb[3] / ts),
        n_threads, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], (uint32_t) src0->ne[2],
        (uint32_t) n_dims, (uint32_t) mode,
        ggml_opengl_u32_from_f32(theta_scale), ggml_opengl_u32_from_f32(attn_factor),
        ggml_opengl_u32_from_f32(freq_scale), ggml_opengl_u32_from_f32(ext_factor),
        ggml_opengl_u32_from_f32(corr_dims[0]), ggml_opengl_u32_from_f32(corr_dims[1]),
        (uint32_t) sections[0], (uint32_t) sections[1], (uint32_t) sections[2], (uint32_t) sections[3],
        (uint32_t) n_offs,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { b0, b1, b2, bd }, CEIL_DIV(n_threads, (uint32_t) GGML_GL_WG_SIZE));
}

// FLASH_ATTN_EXT in two passes (block results, then combine); see flash_attn.comp. Port of the D3D11 encoder
// without its work-per-submit chunking: rows are split only by the size of the block results buffer.
static void ggml_opengl_flash_attn_ext(gl_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * q     = dst->src[0];
    ggml_tensor * k     = dst->src[1];
    ggml_tensor * v     = dst->src[2];
    ggml_tensor * mask  = dst->src[3];
    ggml_tensor * sinks = dst->src[4];

    float       scale         = ggml_get_op_params_f32(dst, 0);
    const float max_bias      = ggml_get_op_params_f32(dst, 1);
    const float logit_softcap = ggml_get_op_params_f32(dst, 2);
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    std::vector<std::string> defines = {
        "DK=" + std::to_string(k->ne[0]), "DV=" + std::to_string(v->ne[0]),
        k->type == GGML_TYPE_F16 ? "K_F16" : k->type == GGML_TYPE_Q8_0 ? "K_Q8_0" : "K_F32",
        v->type == GGML_TYPE_F16 ? "V_F16" : v->type == GGML_TYPE_Q8_0 ? "V_Q8_0" : "V_F32",
    };
    if (mask) {
        defines.push_back("HAS_MASK");
    }
    if (sinks) {
        defines.push_back("HAS_SINKS");
    }
    if (logit_softcap != 0.0f) {
        defines.push_back("SOFTCAP");
    }
    // pass 1 runs a workgroup per row and block of WG_SIZE KV entries: a thread per row with DV/4 vec4
    // accumulators each was wrong at head size >= 128 on the AMD GL compiler (exp27) or crashed it (exp28-exp30)
    const uint32_t blk = GGML_GL_WG_SIZE;
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "flash_attn", glsl_flash_attn, defines);
    // prompts: 4 query rows of one head per workgroup share each K / V load (MQ4 in flash_attn.comp)
    const uint32_t n_q_rows = (uint32_t) q->ne[1];
    const bool     mq4      = n_q_rows >= 4 && n_q_rows % 4 == 0 && k->ne[0] <= 256 && v->ne[0] <= 256;
    std::vector<std::string> mq4_defines = defines;
    mq4_defines.push_back("MQ4");
    gl_pipeline * pipeline_mq4 = mq4 ? &ggml_opengl_get_pipeline(dev, "flash_attn", glsl_flash_attn, mq4_defines) : nullptr;

    std::vector<std::string> combine_defines = { "DV=" + std::to_string(v->ne[0]), "COMBINE" };
    if (sinks) {
        combine_defines.push_back("HAS_SINKS");
    }
    gl_pipeline & combine = ggml_opengl_get_pipeline(dev, "flash_attn", glsl_flash_attn, combine_defines);

    // a missing mask or sinks tensor: q is bound in its slot and never read
    const gl_binding bq = ggml_opengl_bind_tensor(dev, q);
    const gl_binding bk = ggml_opengl_bind_tensor(dev, k);
    const gl_binding bv = ggml_opengl_bind_tensor(dev, v);
    const gl_binding bm = ggml_opengl_bind_tensor(dev, mask ? mask : q);
    const gl_binding bs = ggml_opengl_bind_tensor(dev, sinks ? sinks : q);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const size_t     tk = ggml_type_size(k->type);
    const size_t     tv = ggml_type_size(v->type);

    const uint32_t n_head      = (uint32_t) q->ne[2];
    const float    n_head_log2 = (float) (1u << (uint32_t) floor(log2((double) n_head)));
    const float    m0          = powf(2.0f, -(max_bias) / n_head_log2);
    const float    m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const uint32_t n_kv        = (uint32_t) k->ne[1];
    const uint32_t n_blocks    = CEIL_DIV(n_kv, blk);

    std::vector<uint32_t> params = {
        bq.elem_offset, bk.elem_offset, bv.elem_offset, mask ? bm.elem_offset : 0u, sinks ? bs.elem_offset : 0u, bd.elem_offset,
        (uint32_t) (q->nb[1] / 4), (uint32_t) (q->nb[2] / 4), (uint32_t) (q->nb[3] / 4),
        (uint32_t) (k->nb[1] / tk), (uint32_t) (k->nb[2] / tk), (uint32_t) (k->nb[3] / tk),
        (uint32_t) (v->nb[1] / tv), (uint32_t) (v->nb[2] / tv), (uint32_t) (v->nb[3] / tv),
        mask ? (uint32_t) (mask->nb[1] / 2) : 0u, mask ? (uint32_t) (mask->nb[2] / 2) : 0u, mask ? (uint32_t) (mask->nb[3] / 2) : 0u,
        mask ? (uint32_t) mask->ne[2] : 1u, mask ? (uint32_t) mask->ne[3] : 1u,
        (uint32_t) q->ne[1], n_head, n_kv,
        (uint32_t) (q->ne[2] / k->ne[2]), (uint32_t) (q->ne[3] / k->ne[3]),
        (uint32_t) (q->ne[2] / v->ne[2]), (uint32_t) (q->ne[3] / v->ne[3]),
        ggml_opengl_u32_from_f32(scale), ggml_opengl_u32_from_f32(max_bias), ggml_opengl_u32_from_f32(logit_softcap),
        ggml_opengl_u32_from_f32(n_head_log2), ggml_opengl_u32_from_f32(m0), ggml_opengl_u32_from_f32(m1),
        blk, n_blocks, 0, 0,   // blk_size, n_blocks, row0, n_rows
    };
    const size_t row0_idx = params.size() - 2;

    const uint64_t n_rows    = (uint64_t) ggml_nrows(dst);   // dst is [DV, n_head, n_q, n_batch]
    const uint64_t row_bytes = (uint64_t) n_blocks * (uint64_t) (v->ne[0] + 1) * sizeof(float);
    uint64_t       tmp_rows  = std::max<uint64_t>(1, GGML_GL_FLASH_ATTN_TMP_MAX / row_bytes);
    // MQ4 groups must not cross a dispatch: rows are split in multiples of 4 heads' rows
    const uint64_t mq4_span  = 4ull * n_head;
    const bool     use_mq4   = mq4 && tmp_rows >= mq4_span;
    if (use_mq4) {
        tmp_rows -= tmp_rows % mq4_span;
    }
    const size_t   tmp_need  = (size_t) (std::min(tmp_rows, n_rows) * row_bytes);
    if (dev.fa_tmp_size < tmp_need) {
        // GL keeps a deleted buffer alive until the dispatches that use it are done
        if (dev.fa_tmp) {
            p_glDeleteBuffers(1, &dev.fa_tmp);
            dev.cur_bind.clear();
        }
        size_t size = 1ull << 20;
        while (size < tmp_need) {
            size *= 2;
        }
        p_glGenBuffers(1, &dev.fa_tmp);
        p_glBindBuffer(E_COPY_WRITE_BUFFER, dev.fa_tmp);
        p_glBufferData(E_COPY_WRITE_BUFFER, (ptrdiff_t) size, nullptr, E_DYNAMIC_COPY);
        dev.fa_tmp_size = size;
    }
    const gl_binding bt = { dev.fa_tmp, 0, dev.fa_tmp_size, 0 };

    for (uint64_t row0 = 0, n = 0; row0 < n_rows; row0 += n) {
        n = std::min(n_rows - row0, tmp_rows);
        params[row0_idx]     = (uint32_t) row0;
        params[row0_idx + 1] = (uint32_t) n;
        if (use_mq4) {
            ggml_opengl_dispatch(dev, *pipeline_mq4, params, { bq, bk, bv, bm, bs, bd, bt },
                                 (uint32_t) (n / 4) * n_blocks);
        } else {
            ggml_opengl_dispatch(dev, pipeline, params, { bq, bk, bv, bm, bs, bd, bt },
                                 (uint32_t) n * n_blocks);
        }
        ggml_opengl_dispatch(dev, combine, params, { bq, bk, bv, bm, bs, bd, bt }, CEIL_DIV((uint32_t) n, (uint32_t) GGML_GL_WG_SIZE));
    }
}

// GET_ROWS of a quantized source: the dequant paths of the matrix-vector kernel, TPR threads per row
static void ggml_opengl_get_rows_quant(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
    const uint32_t tpr      = 16;
    gl_pipeline &  pipeline = ggml_opengl_get_pipeline(dev, "get_rows_q", glsl_get_rows_q,
                                                       { ggml_opengl_type_define(src->type, "SRC0"), "TPR=" + std::to_string(tpr) });

    const gl_binding bs     = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bi     = ggml_opengl_bind_tensor(dev, idx);
    const gl_binding bd     = ggml_opengl_bind_tensor(dev, dst);
    const size_t     ts     = ggml_type_size(src->type);
    const size_t     td     = ggml_type_size(dst->type);
    const uint32_t   n_rows = (uint32_t) ggml_nrows(dst);

    std::vector<uint32_t> params = {
        bs.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (idx->nb[0] / 4), (uint32_t) (idx->nb[1] / 4), (uint32_t) (idx->nb[2] / 4),
        (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        (uint32_t) dst->ne[0], (uint32_t) idx->ne[0], (uint32_t) idx->ne[1], n_rows,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bi, bd }, CEIL_DIV(n_rows * tpr, (uint32_t) GGML_GL_WG_SIZE));
}

static void ggml_opengl_get_rows(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
    if (ggml_is_quantized(src->type)) {
        ggml_opengl_get_rows_quant(dev, src, idx, dst);
        return;
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "get_rows", glsl_get_rows, { ggml_opengl_type_define(src->type, "SRC") });

    const gl_binding bs      = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bi      = ggml_opengl_bind_tensor(dev, idx);
    const gl_binding bd      = ggml_opengl_bind_tensor(dev, dst);
    const size_t     ts      = ggml_type_size(src->type);
    const size_t     td      = ggml_type_size(dst->type);
    const uint32_t   n_units = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        bs.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (idx->nb[0] / 4), (uint32_t) (idx->nb[1] / 4), (uint32_t) (idx->nb[2] / 4),
        (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        (uint32_t) dst->ne[0], (uint32_t) idx->ne[0], (uint32_t) idx->ne[1], n_units,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bi, bd }, CEIL_DIV(n_units, (uint32_t) GGML_GL_WG_SIZE));
}

// dst = src0 * src1 with a 64 x 32 tile of dst per workgroup; see mul_mat_tiled.comp
static void ggml_opengl_mul_mat_tiled(gl_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    std::vector<std::string> defines = { ggml_opengl_type_define(src0->type, "SRC0") };
    if (src1->type == GGML_TYPE_F16) {
        defines.push_back("SRC1_F16");
    }

    const gl_binding b0 = ggml_opengl_bind_tensor(dev, src0);
    const gl_binding b1 = ggml_opengl_bind_tensor(dev, src1);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const size_t     t0 = ggml_type_size(src0->type);
    const size_t     t1 = ggml_type_size(src1->type);

    std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) src0->ne[0],
        (uint32_t) (src0->nb[1] / t0), (uint32_t) (src0->nb[2] / t0), (uint32_t) (src0->nb[3] / t0),
        (uint32_t) (src1->nb[1] / t1), (uint32_t) (src1->nb[2] / t1), (uint32_t) (src1->nb[3] / t1),
        (uint32_t) dst->ne[2], (uint32_t) (src1->ne[2] / src0->ne[2]), (uint32_t) (src1->ne[3] / src0->ne[3]),
        (uint32_t) (dst->ne[2] * dst->ne[3]),
    };
    // tile sizes must match TILE_M and TILE_N in mul_mat_tiled.comp
    const uint32_t tiles_m = CEIL_DIV((uint32_t) dst->ne[0], 64u);
    const uint32_t tiles_n = CEIL_DIV((uint32_t) dst->ne[1], 32u);
    const uint32_t batches = (uint32_t) (dst->ne[2] * dst->ne[3]);
    if (dev.tiled_ksplit == 0) {
        gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "mul_mat_tiled", glsl_mul_mat_tiled, defines);
        ggml_opengl_dispatch(dev, pipeline, params, { b1, b0, bd }, tiles_m * tiles_n * batches);
        return;
    }
    // no loop around barriers (MTT S80): one dispatch per tiled_ksplit tiles of k, single tiles for the rest; the
    // passes after the first add to dst
    const uint32_t k = (uint32_t) src0->ne[0];
    defines.push_back("KSPLIT");
    for (uint32_t k0 = 0; k0 < k;) {
        const uint32_t steps = k - k0 >= dev.tiled_ksplit * 32 ? dev.tiled_ksplit : 1;
        std::vector<std::string> d = defines;
        d.push_back("K_STEPS=" + std::to_string(steps));
        gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "mul_mat_tiled", glsl_mul_mat_tiled, d);
        std::vector<uint32_t> p = params;
        p.push_back(k0);
        ggml_opengl_dispatch(dev, pipeline, p, { b1, b0, bd }, tiles_m * tiles_n * batches);
        k0 += steps * 32;
    }
}

// long prompts go to the tiled kernel; it needs k in whole tiles of 32
static bool ggml_opengl_mul_mat_use_tiled(const gl_device_ctx & dev, const ggml_tensor * src0, const ggml_tensor * dst) {
    return dev.tiled_min_cols != 0 && (uint32_t) dst->ne[1] >= dev.tiled_min_cols && src0->ne[0] % 32 == 0;
}

// with bias: MUL_MAT fused with the following ADD of one f32 row (mat-vec path only); dst is the ADD's output
static void ggml_opengl_mul_mat(gl_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst,
                                ggml_tensor * bias = nullptr) {
    const bool quant = ggml_is_quantized(src0->type);
    if (ggml_opengl_mul_mat_use_tiled(dev, src0, dst)) {
        GGML_ASSERT(!bias);
        ggml_opengl_mul_mat_tiled(dev, src0, src1, dst);
        return;
    }
    // threads per row: one unit (4 floats or one 32-wide block) per thread, power of two, at most WG_SIZE
    const uint32_t units = (uint32_t) (quant ? src0->ne[0] / 32 : src0->ne[0] / 4);
    uint32_t       tpr   = 1;
    while (tpr < units && tpr < dev.max_tpr) {
        tpr *= 2;
    }
    std::vector<std::string> defines = { ggml_opengl_type_define(src0->type, "SRC0"), "TPR=" + std::to_string(tpr) };
    if (src1->type == GGML_TYPE_F16) {
        defines.push_back("SRC1_F16");
    }
    if (bias) {
        defines.push_back("ADD_BIAS");
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "mul_mat_vec", glsl_mul_mat_vec, defines);

    const gl_binding b0 = ggml_opengl_bind_tensor(dev, src0);
    const gl_binding b1 = ggml_opengl_bind_tensor(dev, src1);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const gl_binding bb = bias ? ggml_opengl_bind_tensor(dev, bias) : bd;
    const size_t     t0 = ggml_type_size(src0->type);   // block size in bytes for quant types
    const size_t     t1 = ggml_type_size(src1->type);

    std::vector<uint32_t> params = {
        b1.elem_offset, (uint32_t) dst->ne[1], (uint32_t) src0->ne[0],
        (uint32_t) (src1->nb[1] / t1), (uint32_t) (src1->nb[2] / t1), (uint32_t) (src1->nb[3] / t1),
        (uint32_t) src0->ne[2], (uint32_t) src0->ne[3],
        (uint32_t) (src1->ne[2] / src0->ne[2]), (uint32_t) (src1->ne[3] / src0->ne[3]),
        0,   // col0, set per chunk below
        b0.elem_offset, bd.elem_offset, (uint32_t) dst->ne[0],
        (uint32_t) (src0->nb[1] / t0), (uint32_t) (src0->nb[2] / t0), (uint32_t) (src0->nb[3] / t0),
    };
    if (bias) {
        params.push_back(bb.elem_offset);
    }
    const size_t   col0_idx = 10;
    const uint32_t total_wg = CEIL_DIV((uint32_t) dst->ne[0], GGML_GL_WG_SIZE / tpr) * (uint32_t) (dst->ne[2] * dst->ne[3]);
    // columns are processed 4 at a time (MAX_COLS in mul_mat_vec.comp); every chunk re-reads src0
    for (uint32_t col0 = 0; col0 < (uint32_t) dst->ne[1]; col0 += 4) {
        params[col0_idx] = col0;
        if (bias) {
            ggml_opengl_dispatch(dev, pipeline, params, { b1, b0, bd, bb }, total_wg);
        } else {
            ggml_opengl_dispatch(dev, pipeline, params, { b1, b0, bd }, total_wg);
        }
    }
}

// MUL_MAT_ID: the matrix-vector kernel with MMID, one (id slot, token) pair per group of workgroups
// mul_mat_id for long prompts (port of the D3D11 plan): group the (token, slot) pairs by expert, then one tiled
// product per expert tile, as mm_ids_helper + mmq in the CUDA backend. Returns false when not eligible.
static bool ggml_opengl_mul_mat_id_tiled(gl_device_ctx & dev, ggml_tensor * as, ggml_tensor * src1, ggml_tensor * ids,
                                         ggml_tensor * dst) {
    const uint32_t n_used    = (uint32_t) ids->ne[0];
    const uint32_t n_tokens  = (uint32_t) ids->ne[1];
    const uint32_t n_experts = (uint32_t) as->ne[2];
    // MAX_EXPERTS in mul_mat_id_prep.comp
    if (dev.tiled_min_cols == 0 || dev.tiled_ksplit != 0 || n_tokens < dev.tiled_min_cols || n_experts > 1024 || as->ne[3] != 1 ||
        as->ne[0] % 32 != 0 || src1->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float)) {
        return false;
    }
    const uint32_t n_pairs   = n_used * n_tokens;
    const uint32_t max_tiles = CEIL_DIV(n_pairs, 32u) + std::min(n_experts, n_pairs);
    const uint32_t list_base = 1 + 3 * max_tiles;
    const size_t   need      = (size_t) (list_base + n_pairs) * 4;
    if (dev.mmid_scratch_size < need) {
        // GL keeps a deleted buffer alive until the dispatches that use it are done
        if (dev.mmid_scratch) {
            p_glDeleteBuffers(1, &dev.mmid_scratch);
            dev.cur_bind.clear();
        }
        size_t size = 1ull << 16;
        while (size < need) {
            size *= 2;
        }
        p_glGenBuffers(1, &dev.mmid_scratch);
        p_glBindBuffer(E_COPY_WRITE_BUFFER, dev.mmid_scratch);
        p_glBufferData(E_COPY_WRITE_BUFFER, (ptrdiff_t) size, nullptr, E_DYNAMIC_COPY);
        dev.mmid_scratch_size = size;
    }
    const gl_binding bs = { dev.mmid_scratch, 0, dev.mmid_scratch_size, 0 };

    const gl_binding bi   = ggml_opengl_bind_tensor(dev, ids);
    gl_pipeline &    prep = ggml_opengl_get_pipeline(dev, "mul_mat_id_prep", glsl_mul_mat_id_prep, {});
    ggml_opengl_dispatch(dev, prep, { bi.elem_offset, (uint32_t) (ids->nb[1] / 4), n_used, n_tokens, n_experts, list_base },
                         { bi, bs }, 1);

    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "mul_mat_tiled", glsl_mul_mat_tiled,
                                                      { ggml_opengl_type_define(as->type, "SRC0"), "MMID" });
    const gl_binding b0 = ggml_opengl_bind_tensor(dev, as);
    const gl_binding b1 = ggml_opengl_bind_tensor(dev, src1);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const size_t     t0 = ggml_type_size(as->type);

    std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) dst->ne[0], 0u, (uint32_t) as->ne[0],
        (uint32_t) (as->nb[1] / t0), (uint32_t) (as->nb[2] / t0), 0u,
        (uint32_t) (src1->nb[1] / 4), (uint32_t) (src1->nb[2] / 4), 0u,
        1u, 1u, 1u, 1u,
        n_used, (uint32_t) src1->ne[1], (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), list_base,
    };
    // TILE_M in mul_mat_tiled.comp; workgroups past the real tile count do nothing
    const uint32_t tiles_m = CEIL_DIV((uint32_t) dst->ne[0], 64u);
    ggml_opengl_dispatch(dev, pipeline, params, { b1, b0, bd, bs }, tiles_m * max_tiles);
    return true;
}

static void ggml_opengl_mul_mat_id(gl_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor *  as    = dst->src[0];
    ggml_tensor *  src1  = dst->src[1];
    ggml_tensor *  ids   = dst->src[2];
    if (ggml_opengl_mul_mat_id_tiled(dev, as, src1, ids, dst)) {
        return;
    }
    const bool     quant = ggml_is_quantized(as->type);
    const uint32_t units = (uint32_t) (quant ? as->ne[0] / 32 : as->ne[0] / 4);
    uint32_t       tpr   = 1;
    while (tpr < units && tpr < dev.max_tpr) {
        tpr *= 2;
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "mul_mat_vec", glsl_mul_mat_vec,
                                                      { ggml_opengl_type_define(as->type, "SRC0"), "TPR=" + std::to_string(tpr), "MMID" });

    const gl_binding b0 = ggml_opengl_bind_tensor(dev, as);
    const gl_binding b1 = ggml_opengl_bind_tensor(dev, src1);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
    const gl_binding bi = ggml_opengl_bind_tensor(dev, ids);
    const size_t     t0 = ggml_type_size(as->type);
    const size_t     t1 = ggml_type_size(src1->type);

    std::vector<uint32_t> params = {
        b1.elem_offset, 1, (uint32_t) as->ne[0],
        (uint32_t) (src1->nb[1] / t1), (uint32_t) (src1->nb[2] / t1), (uint32_t) (src1->nb[3] / t1),
        1, 1, 1, 1, 0,
        b0.elem_offset, bd.elem_offset, (uint32_t) as->ne[1],
        (uint32_t) (as->nb[1] / t0), (uint32_t) (as->nb[2] / t0), (uint32_t) (as->nb[3] / t0),
        bi.elem_offset, (uint32_t) (ids->nb[1] / 4), (uint32_t) ids->ne[0], (uint32_t) ids->ne[1],
        (uint32_t) src1->ne[1], (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4),
    };
    const uint32_t total_wg = CEIL_DIV((uint32_t) as->ne[1], GGML_GL_WG_SIZE / tpr) * (uint32_t) (ids->ne[0] * ids->ne[1]);
    ggml_opengl_dispatch(dev, pipeline, params, { b1, b0, bd, bi }, total_wg);
}

// ARGSORT and TOP_K, one workgroup per row
static void ggml_opengl_argsort(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    const bool desc = dst->op == GGML_OP_TOP_K || (ggml_sort_order) ggml_get_op_params_i32(dst, 0) == GGML_SORT_ORDER_DESC;
    std::vector<std::string> defines;
    if (desc) {
        defines.push_back("SORT_DESC");
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "argsort", glsl_argsort, defines);

    const gl_binding bs     = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bd     = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   n_rows = (uint32_t) ggml_nrows(src);

    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset, (uint32_t) (src->nb[1] / 4), (uint32_t) src->ne[0], (uint32_t) dst->ne[0], n_rows,
    };
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bd }, n_rows);
}

static void ggml_opengl_encode_node(gl_device_ctx & dev, ggml_tensor * node) {
    if (ggml_is_empty(node)) {
        return;
    }
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            return;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            ggml_opengl_cpy(dev, node->src[0], node);
            return;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            ggml_opengl_binary_op(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_SCALE:
            ggml_opengl_scale(dev, node->src[0], node);
            return;
        case GGML_OP_FILL:
            ggml_opengl_fill(dev, node);
            return;
        case GGML_OP_GET_ROWS:
            ggml_opengl_get_rows(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_NORM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            ggml_opengl_rms_norm(dev, node->src[0], node);
            return;
        case GGML_OP_CONCAT:
            ggml_opengl_concat(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_UNARY:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            ggml_opengl_unary(dev, node->src[0], node);
            return;
        case GGML_OP_REPEAT:
            ggml_opengl_repeat(dev, node->src[0], node);
            return;
        case GGML_OP_GATED_DELTA_NET:
            ggml_opengl_gated_delta_net(dev, node);
            return;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_opengl_lightning_indexer(dev, node);
            return;
        case GGML_OP_DSV4_HC_PRE:
            ggml_opengl_hc_pre(dev, node);
            return;
        case GGML_OP_DSV4_HC_POST:
            ggml_opengl_hc_post(dev, node);
            return;
        case GGML_OP_DSV4_HC_COMB:
            ggml_opengl_hc_comb(dev, node);
            return;
        case GGML_OP_PAD:
            ggml_opengl_pad(dev, node->src[0], node);
            return;
        case GGML_OP_ROLL:
            ggml_opengl_roll(dev, node->src[0], node);
            return;
        case GGML_OP_IM2COL:
            ggml_opengl_im2col(dev, node);
            return;
        case GGML_OP_UPSCALE:
            ggml_opengl_upscale(dev, node->src[0], node);
            return;
        case GGML_OP_POOL_2D:
            ggml_opengl_pool_2d(dev, node->src[0], node);
            return;
        case GGML_OP_SSM_CONV:
            ggml_opengl_ssm_conv(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_GROUP_NORM:
            ggml_opengl_group_norm(dev, node->src[0], node);
            return;
        case GGML_OP_ADD_ID:
            ggml_opengl_add_id(dev, node->src[0], node->src[1], node->src[2], node);
            return;
        case GGML_OP_CONV_2D_DW:
            ggml_opengl_conv_2d_dw(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_SSM_SCAN:
            ggml_opengl_ssm_scan(dev, node);
            return;
        case GGML_OP_GLU:
            ggml_opengl_glu(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            ggml_opengl_rope(dev, node->src[0], node->src[1], node->src[2], node);
            return;
        case GGML_OP_SOFT_MAX:
            ggml_opengl_soft_max(dev, node->src[0], node->src[1], node->src[2], node);
            return;
        case GGML_OP_SET_ROWS:
            ggml_opengl_set_rows(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_MUL_MAT:
            ggml_opengl_mul_mat(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_MUL_MAT_ID:
            ggml_opengl_mul_mat_id(dev, node);
            return;
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            ggml_opengl_argsort(dev, node->src[0], node);
            return;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_opengl_flash_attn_ext(dev, node);
            return;
        default:
            GGML_ABORT("ggml_opengl: unsupported op %s", ggml_op_name(node->op));
    }
}

/* GGML Backend Interface */

static const char * ggml_backend_opengl_name(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_opengl_context *) backend->context;
    return ctx->name.c_str();
}

static void ggml_opengl_read_queries(gl_device_ctx & dev) {
    for (auto & p : dev.q_pending) {
        unsigned long long ns = 0;
        p_glGetQueryObjectui64v(p.first, E_QUERY_RESULT, &ns);
        auto & e = dev.prof[p.second];
        e.ns += ns;
        e.n++;
        dev.q_free.push_back(p.first);
    }
    dev.q_pending.clear();
}

static void ggml_opengl_print_stats(gl_device_ctx & dev) {
    fprintf(stderr, "ggml_opengl stats [%s]: graphs %llu, nodes %llu, dispatches %llu, barriers %llu | graph_compute %.1f ms "
                    "| shader compile %.1f ms (%llu from source, %llu from the disk cache) | set_tensor %llu calls %.1f MB %.1f ms "
                    "| get_tensor %llu calls %.1f MB %.1f ms | params %s\n",
            dev.name.c_str(), (unsigned long long) dev.n_graphs, (unsigned long long) dev.n_nodes,
            (unsigned long long) dev.n_dispatches, (unsigned long long) dev.n_barriers, dev.t_graph_us / 1000.0,
            dev.t_compile_us / 1000.0, (unsigned long long) dev.n_compiles, (unsigned long long) dev.n_cache_hits,
            (unsigned long long) dev.n_set_tensor, dev.bytes_set / 1e6, dev.t_set_us / 1000.0,
            (unsigned long long) dev.n_get_tensor, dev.bytes_get / 1e6, dev.t_get_us / 1000.0,
            dev.ubo_map ? "mapped" : "subdata");
    if (dev.profile) {
        std::vector<std::pair<std::string, gl_device_ctx::prof_entry>> v(dev.prof.begin(), dev.prof.end());
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.second.ns > b.second.ns; });
        uint64_t total = 0, n = 0;
        for (const auto & e : v) {
            total += e.second.ns;
            n += e.second.n;
        }
        fprintf(stderr, "ggml_opengl profile [%s]: GPU time %.1f ms in %llu timed dispatches (the last graph is not counted)\n",
                dev.name.c_str(), total / 1e6, (unsigned long long) n);
        for (const auto & e : v) {
            fprintf(stderr, "  %7.1f ms %5.1f%% %8llu x %8.1f us  %s\n", e.second.ns / 1e6, total ? 100.0 * e.second.ns / total : 0.0,
                    (unsigned long long) e.second.n, e.second.n ? e.second.ns / 1e3 / e.second.n : 0.0, e.first.c_str());
        }
    }
    fflush(stderr);
}

static void ggml_backend_opengl_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_opengl_context *) backend->context;
    delete ctx;
    delete backend;
}

// RMS_NORM followed by a MUL with one f32 weight row of the same width, or a mat-vec MUL_MAT followed by an ADD
// of one f32 bias row -> one dispatch; returns the nodes used
static int ggml_opengl_try_fuse(gl_device_ctx & dev, const ggml_cgraph * cgraph, int i) {
    if (dev.no_fuse) {
        return 0;
    }
    ggml_tensor * mm = cgraph->nodes[i];
    if (mm->op == GGML_OP_MUL_MAT && ggml_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
        ggml_tensor * add  = cgraph->nodes[i + 1];
        ggml_tensor * bias = add->src[0] == mm ? add->src[1] : add->src[0];
        if (mm->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(add, mm) || !ggml_is_contiguous(add) || ggml_nrows(bias) != 1 ||
            bias->ne[0] != mm->ne[0] || bias->nb[0] != sizeof(float) || ggml_is_empty(mm) ||
            ggml_opengl_mul_mat_use_tiled(dev, mm->src[0], mm)) {
            return 0;
        }
        ggml_opengl_mul_mat(dev, mm->src[0], mm->src[1], add, bias);
        return 2;
    }
    ggml_tensor * rms = cgraph->nodes[i];
    if (rms->op != GGML_OP_RMS_NORM || !ggml_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        return 0;
    }
    ggml_tensor * mul = cgraph->nodes[i + 1];
    ggml_tensor * w   = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    if (rms->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
        rms->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(mul, rms) || ggml_nrows(w) != 1 ||
        w->ne[0] != rms->ne[0] || w->nb[0] != sizeof(float) || ggml_is_empty(rms)) {
        return 0;
    }
    ggml_opengl_rms_norm(dev, rms->src[0], mul, rms, w);
    return 2;
}

static void ggml_opengl_encode_graph(gl_device_ctx & dev, const ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (dev.trace && !dev.collecting) {
            const ggml_tensor * n = cgraph->nodes[i];
            fprintf(stderr, "gl-trace: node %d/%d %s %s %s ne [%lld %lld %lld %lld] src0 %s src1 %s\n", i, cgraph->n_nodes,
                    ggml_op_desc(n), n->name, ggml_type_name(n->type), (long long) n->ne[0], (long long) n->ne[1],
                    (long long) n->ne[2], (long long) n->ne[3], n->src[0] ? ggml_type_name(n->src[0]->type) : "-",
                    n->src[1] ? ggml_type_name(n->src[1]->type) : "-");
            fflush(stderr);
        }
        const int used = ggml_opengl_try_fuse(dev, cgraph, i);
        if (dev.trace && !dev.collecting) {
            glFinish();
        }
        if (used > 0) {
            i += used - 1;
            continue;
        }
        ggml_opengl_encode_node(dev, cgraph->nodes[i]);
        if (dev.trace && !dev.collecting) {
            glFinish();
            fprintf(stderr, "gl-trace: node %d done\n", i);
            fflush(stderr);
        }
    }
}

static ggml_status ggml_backend_opengl_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto *          ctx = (ggml_backend_opengl_context *) backend->context;
    gl_device_ctx & dev = *ctx->dev;
    gl_scope        scope(dev);
    if (dev.trace) {
        fprintf(stderr, "gl-trace: graph_compute %d nodes\n", cgraph->n_nodes);
        fflush(stderr);
    }
    const double    t0 = ggml_opengl_time_us();
    ggml_opengl_read_queries(dev);
    if (dev.n_graphs < 4 || cgraph->n_nodes != dev.last_graph_nodes) {
        // a graph of a new shape: find the programs it needs first and build the missing ones together
        dev.collecting = true;
        ggml_opengl_encode_graph(dev, cgraph);
        dev.collecting = false;
        ggml_opengl_build_pipeline_jobs(dev);
    }
    dev.last_graph_nodes = cgraph->n_nodes;
    // buffers and programs may have been deleted (and their names reused) since the last graph
    dev.cur_prog = 0;
    dev.cur_bind.clear();
    ggml_opengl_encode_graph(dev, cgraph);
    // later buffer reads, writes and copies see what the kernels wrote
    p_glMemoryBarrier(E_ALL_BARRIER_BITS);
    dev.n_barriers++;
    glFlush();
    dev.n_graphs++;
    dev.n_nodes += cgraph->n_nodes;
    dev.t_graph_us += ggml_opengl_time_us() - t0;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_opengl_synchronize(ggml_backend_t backend) {
    auto *   ctx = (ggml_backend_opengl_context *) backend->context;
    gl_scope scope(*ctx->dev);
    glFinish();
}

static ggml_backend_i ggml_backend_opengl_i = {
    /* .get_name                = */ ggml_backend_opengl_name,
    /* .free                    = */ ggml_backend_opengl_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_opengl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_opengl_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_opengl_guid(void) {
    static ggml_guid guid = { 0x0e, 0x6c, 0x4a, 0x31, 0x9d, 0x52, 0x4f, 0x08,
                              0xb3, 0x7e, 0x15, 0xc2, 0x60, 0x4d, 0xa9, 0x87 };
    return &guid;
}

/* GGML Backend Buffer Interface */

static void ggml_backend_opengl_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_opengl_buffer_context *) buffer->context;
    {
        gl_scope scope(*ctx->dev);
        p_glDeleteBuffers(1, &ctx->buf);
    }
    delete ctx;
}

static void * ggml_backend_opengl_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return gl_ptr_base;
}

static void ggml_opengl_buffer_memset(ggml_backend_opengl_buffer_context * ctx, size_t offset, size_t size, uint8_t value) {
    if (size == 0) {
        return;
    }
    gl_scope scope(*ctx->dev);
    p_glBindBuffer(E_COPY_WRITE_BUFFER, ctx->buf);
    p_glClearBufferSubData(E_COPY_WRITE_BUFFER, E_R8UI, (ptrdiff_t) offset, (ptrdiff_t) size, E_RED_INTEGER,
                           GL_UNSIGNED_BYTE, &value);
}

static void ggml_backend_opengl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                     size_t offset, size_t size) {
    auto * ctx = (ggml_backend_opengl_buffer_context *) buffer->context;
    ggml_opengl_buffer_memset(ctx, ggml_opengl_tensor_offset(tensor) + offset, size, value);
}

static void ggml_backend_opengl_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                                  size_t offset, size_t size) {
    auto *          ctx = (ggml_backend_opengl_buffer_context *) buffer->context;
    gl_device_ctx & dev = *ctx->dev;
    if (size == 0) {
        return;
    }
    gl_scope     scope(dev);
    if (dev.trace) {
        fprintf(stderr, "gl-trace: set_tensor %s %s offset %zu size %zu buf %zu\n", tensor->name, ggml_type_name(tensor->type),
                (size_t) ggml_opengl_tensor_offset(tensor) + offset, size, ctx->size);
        fflush(stderr);
    }
    const double t0 = ggml_opengl_time_us();
    p_glBindBuffer(E_COPY_WRITE_BUFFER, ctx->buf);
    p_glBufferSubData(E_COPY_WRITE_BUFFER, (ptrdiff_t) (ggml_opengl_tensor_offset(tensor) + offset), (ptrdiff_t) size, data);
    dev.n_set_tensor++;
    dev.bytes_set += size;
    dev.t_set_us += ggml_opengl_time_us() - t0;
}

static void ggml_backend_opengl_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
                                                  size_t offset, size_t size) {
    auto *          ctx = (ggml_backend_opengl_buffer_context *) buffer->context;
    gl_device_ctx & dev = *ctx->dev;
    if (size == 0) {
        return;
    }
    gl_scope     scope(dev);
    if (dev.trace) {
        fprintf(stderr, "gl-trace: get_tensor %s offset %zu size %zu buf %zu\n", tensor->name,
                (size_t) ggml_opengl_tensor_offset(tensor) + offset, size, ctx->size);
        fflush(stderr);
    }
    const double t0 = ggml_opengl_time_us();
    p_glBindBuffer(E_COPY_READ_BUFFER, ctx->buf);
    p_glGetBufferSubData(E_COPY_READ_BUFFER, (ptrdiff_t) (ggml_opengl_tensor_offset(tensor) + offset), (ptrdiff_t) size, data);
    if (dev.debug) {
        ggml_opengl_check_error("get_tensor");
    }
    dev.n_get_tensor++;
    dev.bytes_get += size;
    dev.t_get_us += ggml_opengl_time_us() - t0;
}

static bool ggml_backend_buffer_is_opengl(ggml_backend_buffer_t buffer);

static bool ggml_backend_opengl_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_buffer_is_opengl(src->buffer)) {
        return false;
    }
    auto * src_ctx = (ggml_backend_opengl_buffer_context *) src->buffer->context;
    auto * dst_ctx = (ggml_backend_opengl_buffer_context *) buffer->context;
    if (src_ctx->dev != dst_ctx->dev) {
        return false;
    }
    gl_scope scope(*dst_ctx->dev);
    p_glBindBuffer(E_COPY_READ_BUFFER, src_ctx->buf);
    p_glBindBuffer(E_COPY_WRITE_BUFFER, dst_ctx->buf);
    p_glCopyBufferSubData(E_COPY_READ_BUFFER, E_COPY_WRITE_BUFFER, (ptrdiff_t) ggml_opengl_tensor_offset(src),
                          (ptrdiff_t) ggml_opengl_tensor_offset(dst), (ptrdiff_t) ggml_nbytes(src));
    return true;
}

static void ggml_backend_opengl_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_opengl_buffer_context *) buffer->context;
    ggml_opengl_buffer_memset(ctx, 0, ctx->size, value);
}

static ggml_backend_buffer_i ggml_backend_opengl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_opengl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_opengl_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_opengl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_opengl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_opengl_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_opengl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_opengl_buffer_clear,
    /* .reset           = */ NULL,
};

static bool ggml_backend_buffer_is_opengl(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_opengl_buffer_free_buffer;
}

/* GGML Backend Buffer Type Interface */

static const char * ggml_backend_opengl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * dev = (gl_device_ctx *) buft->context;
    return dev->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_opengl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto *   dev = (gl_device_ctx *) buft->context;
    gl_scope scope(*dev);

    // slack past the end: kernels read unaligned values as two whole words
    const size_t alloc_size = std::max(dev->bind_align, (size + dev->bind_align - 1) & ~(dev->bind_align - 1)) + GGML_GL_BUFFER_SLACK;
    if (dev->trace) {
        fprintf(stderr, "gl-trace: alloc_buffer %zu bytes (alloc %zu)\n", size, alloc_size);
        fflush(stderr);
    }
    while (glGetError() != GL_NO_ERROR) {}
    GLuint buf = 0;
    p_glGenBuffers(1, &buf);
    p_glBindBuffer(E_COPY_WRITE_BUFFER, buf);
    p_glBufferData(E_COPY_WRITE_BUFFER, (ptrdiff_t) alloc_size, nullptr, E_DYNAMIC_COPY);
    const GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        GGML_LOG_ERROR("ggml_opengl: allocating %zu bytes failed with GL error 0x%04x\n", alloc_size, (unsigned) err);
        p_glDeleteBuffers(1, &buf);
        return nullptr;
    }
    auto * ctx = new ggml_backend_opengl_buffer_context();
    ctx->dev   = ggml_opengl_shared_dev(dev);
    ctx->buf   = buf;
    ctx->size  = alloc_size;
    return ggml_backend_buffer_init(buft, ggml_backend_opengl_buffer_interface, ctx, size);
}

static size_t ggml_backend_opengl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    auto * dev = (gl_device_ctx *) buft->context;
    return dev->bind_align;
}

static size_t ggml_backend_opengl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * dev = (gl_device_ctx *) buft->context;
    return dev->max_alloc;
}

/* GGML Backend Device Interface */

static const char * ggml_backend_opengl_device_get_name(ggml_backend_dev_t dev) {
    auto * ctx = (gl_device_ctx *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_opengl_device_get_description(ggml_backend_dev_t dev) {
    auto * ctx = (gl_device_ctx *) dev->context;
    return ctx->desc.c_str();
}

static void ggml_backend_opengl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto *   ctx = (gl_device_ctx *) dev->context;
    gl_scope scope(*ctx);
    // GL core has no memory query; vendor extensions report KiB. Without one, report 4 GiB so the
    // device is still used (an assumption, not a measurement).
    *free  = 4ull << 30;
    *total = 4ull << 30;
    // Mesa's software renderers expose both extensions with made-up values (llvmpipe: NVX dedicated ~1.8 TiB,
    // avail and ATI free 32767 KiB), which would make llama.cpp fit almost nothing on the device
    if (ctx->desc.find("llvmpipe") != std::string::npos || ctx->desc.find("softpipe") != std::string::npos) {
        return;
    }
    GLint dedicated = 0, avail = 0;
    if (ctx->nvx_meminfo) {
        glGetIntegerv(E_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX, &dedicated);
        glGetIntegerv(E_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &avail);
    }
    // KiB: 64 MiB .. 1 TiB, available within dedicated
    if (dedicated >= (64 << 10) && dedicated <= (1 << 30) && avail >= 0 && avail <= dedicated) {
        *total = (size_t) dedicated * 1024;
        *free  = (size_t) avail * 1024;
    } else if (ctx->ati_meminfo) {
        GLint m[4] = {};
        glGetIntegerv(E_VBO_FREE_MEMORY_ATI, m);
        *free  = (size_t) m[0] * 1024;
        *total = *free;
    }
}

static enum ggml_backend_dev_type ggml_backend_opengl_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_opengl_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_opengl_device_get_name(dev);
    props->description = ggml_backend_opengl_device_get_description(dev);
    props->type        = ggml_backend_opengl_device_get_type(dev);
    ggml_backend_opengl_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_opengl_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * dev_ctx = (gl_device_ctx *) dev->context;

    auto * ctx = new ggml_backend_opengl_context();
    ctx->dev   = ggml_opengl_shared_dev(dev_ctx);
    ctx->name  = dev_ctx->name;

    auto * backend = new ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_opengl_guid(),
        /* .interface = */ ggml_backend_opengl_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_opengl_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * ctx = (gl_device_ctx *) dev->context;
    return &ctx->buft;
}

static bool ggml_backend_opengl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_opengl_buffer_type_get_name && buft->device == dev;
}

static bool ggml_backend_opengl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    auto type_ok = [](ggml_type t) {
        return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_I32;
    };
    // kernels index elements with 32-bit integers
    if (ggml_nelements(op) > UINT32_MAX) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            return true;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            return type_ok(op->type) && type_ok(src0->type);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                   src0->type == op->type && src1->type == op->type;
        case GGML_OP_SCALE:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
        case GGML_OP_FILL:
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ggml_is_contiguous(op);
        case GGML_OP_DSV4_HC_POST:
        case GGML_OP_DSV4_HC_COMB:
            // f32, element 0 contiguous, strides in whole floats (comb of HC_POST may be absent)
            for (int i = 0; i < 4; i++) {
                const ggml_tensor * t = op->src[i];
                if (t && (t->type != GGML_TYPE_F32 || t->nb[0] != 4 || t->nb[1] % 4 != 0 || t->nb[2] % 4 != 0)) {
                    return false;
                }
            }
            return op->type == GGML_TYPE_F32 && op->nb[0] == 4 && op->nb[1] % 4 == 0 && op->nb[2] % 4 == 0;
        case GGML_OP_DSV4_HC_PRE:
            // f32, element 0 contiguous, strides in whole floats
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                   op->src[0]->nb[0] == 4 && op->src[1]->nb[0] == 4 && op->nb[0] == 4 &&
                   op->src[0]->nb[1] % 4 == 0 && op->src[0]->nb[2] % 4 == 0 &&
                   op->src[1]->nb[1] % 4 == 0 && op->src[1]->nb[2] % 4 == 0 && op->nb[1] % 4 == 0;
        case GGML_OP_LIGHTNING_INDEXER:
            // rows of q, k, w, mask contiguous (element 0 stride = type size, as the CPU op asserts)
            return (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   op->src[0]->nb[0] == 4 && op->src[1]->nb[0] == ggml_type_size(op->src[1]->type) &&
                   op->src[2]->nb[0] == 4 && op->src[3]->nb[0] == 2 && op->nb[0] == 4;
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_NORM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src0->nb[0] == sizeof(float);
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return src0->type == op->type && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                   ggml_is_contiguous(op);
        case GGML_OP_CONCAT:
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32 || op->type == GGML_TYPE_F16) &&
                   src0->type == op->type && src1->type == op->type;
        case GGML_OP_UNARY:
            // the kernel writes dst by flat index
            return ggml_opengl_unary_supported(ggml_get_unary_op(op)) && src0->type == op->type &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ggml_is_contiguous(op);
        case GGML_OP_REPEAT:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
        case GGML_OP_GATED_DELTA_NET:
            {
                // f32 everywhere, rows of S_v (<= 512, one workgroup) elements, q/k head size equal to S_v
                const ggml_tensor * v = op->src[2];
                bool ok = op->type == GGML_TYPE_F32 && v->ne[0] <= 512 &&
                          op->src[0]->ne[0] == v->ne[0] && op->src[1]->ne[0] == v->ne[0];
                for (int i = 0; i < 6; i++) {
                    ok = ok && op->src[i]->type == GGML_TYPE_F32 && op->src[i]->nb[0] == sizeof(float);
                }
                return ok;
            }
        case GGML_OP_ROLL:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   ggml_are_same_shape(src0, op);
        case GGML_OP_PAD:
            // dst is addressed by a flat index, as the CPU does; src keeps its strides
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
        case GGML_OP_IM2COL:
            // src0 (the kernel) only gives the shape; the f32 image keeps its strides, dst is contiguous
            return src1->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                   ggml_is_contiguous(op);
        case GGML_OP_UPSCALE:
            {
                const int32_t mode_flags = ggml_get_op_params_i32(op, 0);
                const int32_t mode       = mode_flags & 0xFF;
                return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && !(mode_flags & GGML_SCALE_FLAG_ANTIALIAS) &&
                       (mode == GGML_SCALE_MODE_NEAREST || mode == GGML_SCALE_MODE_BILINEAR);
            }
        case GGML_OP_POOL_2D:
            {
                const ggml_op_pool pool = (ggml_op_pool) ggml_get_op_params_i32(op, 0);
                return op->type == GGML_TYPE_F32 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                       (pool == GGML_OP_POOL_AVG || pool == GGML_OP_POOL_MAX) && ggml_is_contiguous(op) &&
                       src0->nb[0] == ggml_type_size(src0->type);
            }
        case GGML_OP_GROUP_NORM:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) && ggml_are_same_shape(src0, op);
        case GGML_OP_ADD_ID:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   op->src[2]->type == GGML_TYPE_I32 && ggml_is_contiguous(src1) && ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_CONV_2D_DW:
            // only the WHCN path; the CWHN variant the CPU also handles has a different kernel layout
            return op->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(op) &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_SSM_SCAN:
            {
                // one thread per (sequence, head, dim); the rows it indexes directly must be packed
                const ggml_tensor * dt  = op->src[2];
                const ggml_tensor * A   = op->src[3];
                const ggml_tensor * B   = op->src[4];
                const ggml_tensor * C   = op->src[5];
                const ggml_tensor * ids = op->src[6];
                const int64_t nc = src0->ne[0];
                const int64_t nr = src0->ne[1];
                const int64_t nh = src1->ne[1];
                const int64_t ng = B->ne[1];
                bool ok = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                          dt->type == GGML_TYPE_F32 && A->type == GGML_TYPE_F32 && B->type == GGML_TYPE_F32 &&
                          C->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32;
                ok = ok && ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float) && dt->nb[0] == sizeof(float) &&
                     src1->nb[1] == nr * sizeof(float) && A->nb[1] == A->ne[0] * sizeof(float) &&
                     B->nb[1] == nc * sizeof(float) && C->nb[1] == nc * sizeof(float);
                // every stride is passed in float units, and the flat job index is 32 bit
                ok = ok && ng != 0 && nh % ng == 0 && ggml_nelements(op) <= UINT32_MAX;
                return ok;
            }
        case GGML_OP_SSM_CONV:
            // same layout requirements as the CPU kernel
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) &&
                   src0->nb[1] == src0->ne[0] * sizeof(float) && src1->nb[1] == src1->ne[0] * sizeof(float);
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU: case GGML_GLU_OP_GEGLU: case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF: case GGML_GLU_OP_GEGLU_QUICK: case GGML_GLU_OP_SWIGLU_CLAMP:
                    return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           src0->type == op->type && (!src1 || src1->type == op->type);
                case GGML_GLU_OP_SWIGLU_OAI:
                    return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && (!src1 || src1->type == GGML_TYPE_F32);
                default:
                    return false;
            }
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && src0->type == op->type &&
                   src1->type == GGML_TYPE_I32 && src0->ne[0] % 2 == 0 &&
                   (!op->src[2] || op->src[2]->type == GGML_TYPE_F32);
        case GGML_OP_SOFT_MAX:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                   (!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                   (!op->src[2] || op->src[2]->type == GGML_TYPE_F32);
        case GGML_OP_FLASH_ATTN_EXT:
            {
                // f32/f16/q8_0 KV, f16 mask, head sizes up to 576 and multiples of 4 (32 for q8_0); contiguous
                // rows and a contiguous dst
                const ggml_tensor * k = op->src[1];
                const ggml_tensor * v = op->src[2];
                const ggml_tensor * m = op->src[3];
                const ggml_tensor * s = op->src[4];
                auto kv_ok = [](const ggml_tensor * t) {
                    if (t->type == GGML_TYPE_Q8_0) {
                        return t->ne[0] <= 576 && t->ne[0] % 32 == 0 && t->nb[1] % ggml_type_size(t->type) == 0 &&
                               t->nb[2] % ggml_type_size(t->type) == 0 && t->nb[3] % ggml_type_size(t->type) == 0;
                    }
                    return (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && t->nb[0] == ggml_type_size(t->type) &&
                           t->ne[0] <= 576 && t->ne[0] % 4 == 0;
                };
                return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src0->nb[0] == sizeof(float) &&
                       kv_ok(k) && kv_ok(v) && (!m || (m->type == GGML_TYPE_F16 && m->nb[0] == 2)) &&
                       (!s || s->type == GGML_TYPE_F32) && ggml_is_contiguous(op);
            }
        case GGML_OP_SET_ROWS:
            // src f32 into f32 / f16, or f16 into f16
            return ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && src0->type == GGML_TYPE_F32 ||
                    op->type == GGML_TYPE_F16 && src0->type == GGML_TYPE_F16) &&
                   (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);
        case GGML_OP_GET_ROWS:
            if (src1->type != GGML_TYPE_I32) {
                return false;
            }
            switch (src0->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_BF16:
                    return op->type == GGML_TYPE_F32;
                case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
                case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_TQ2_0:
                case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ2_S:
                case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ1_S:
                case GGML_TYPE_IQ1_M: case GGML_TYPE_MXFP4: case GGML_TYPE_Q1_0: case GGML_TYPE_Q2_0:
                    // whole blocks per row, as the matrix-vector kernel reads them
                    return op->type == GGML_TYPE_F32 && src0->ne[0] % ggml_blck_size(src0->type) == 0 &&
                           src0->nb[1] % ggml_type_size(src0->type) == 0 && src0->nb[2] % ggml_type_size(src0->type) == 0 &&
                           src0->nb[3] % ggml_type_size(src0->type) == 0;
                case GGML_TYPE_I32:
                    return op->type == GGML_TYPE_I32;
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
            // contiguous rows. Float weights (f32 / f16 / bf16): any k, f32 or f16 columns. Q4_0 / Q4_1 / Q5_0 / Q5_1 /
            // Q8_0 / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / IQ* / MXFP4 / TQ2_0 / Q1_0 / Q2_0 weights: whole blocks, f32 columns
            if (src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 || src0->type == GGML_TYPE_Q5_0 ||
                src0->type == GGML_TYPE_Q5_1 || src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_K ||
                src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q6_K ||
                src0->type == GGML_TYPE_Q2_K || src0->type == GGML_TYPE_Q3_K || src0->type == GGML_TYPE_IQ4_NL ||
                src0->type == GGML_TYPE_IQ4_XS || src0->type == GGML_TYPE_IQ3_S || src0->type == GGML_TYPE_IQ2_S ||
                src0->type == GGML_TYPE_IQ2_XXS || src0->type == GGML_TYPE_IQ2_XS || src0->type == GGML_TYPE_IQ3_XXS ||
                src0->type == GGML_TYPE_IQ1_S || src0->type == GGML_TYPE_IQ1_M || src0->type == GGML_TYPE_MXFP4 ||
                src0->type == GGML_TYPE_TQ2_0 || src0->type == GGML_TYPE_Q1_0 || src0->type == GGML_TYPE_Q2_0) {
                return src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                       src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == ggml_type_size(src1->type) &&
                       src0->ne[0] % ggml_blck_size(src0->type) == 0;
            }
            return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
                   (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == ggml_type_size(src1->type);
        case GGML_OP_MUL_MAT_ID:
            // same weight types as MUL_MAT, f32 columns, i32 ids
            if (op->src[2]->type != GGML_TYPE_I32 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
                src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != 4 || op->nb[0] != 4) {
                return false;
            }
            switch (src0->type) {
                case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ2_S:
                case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ1_S:
                case GGML_TYPE_IQ1_M: case GGML_TYPE_MXFP4: case GGML_TYPE_TQ2_0: case GGML_TYPE_Q1_0:
                case GGML_TYPE_Q2_0:
                    return src0->ne[0] % ggml_blck_size(src0->type) == 0;
                case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            // f32 rows of at most 1024 (the rank loop is O(ne0^2) per row), i32 indices, contiguous
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 && ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(op) && src0->ne[0] <= 1024;
        default:
            return false;
    }
}

static struct ggml_backend_device_i ggml_backend_opengl_device_i = {
    /* .get_name             = */ ggml_backend_opengl_device_get_name,
    /* .get_description      = */ ggml_backend_opengl_device_get_description,
    /* .get_memory           = */ ggml_backend_opengl_device_get_memory,
    /* .get_type             = */ ggml_backend_opengl_device_get_type,
    /* .get_props            = */ ggml_backend_opengl_device_get_props,
    /* .init_backend         = */ ggml_backend_opengl_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_opengl_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_opengl_device_supports_op,
    /* .supports_buft        = */ ggml_backend_opengl_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

/* Registry: context creation and device initialization */

struct ggml_backend_opengl_reg_context {
    std::vector<std::shared_ptr<gl_device_ctx>> devs;
    std::vector<ggml_backend_device>            devices;
};

static ggml_backend_opengl_reg_context * g_reg_ctx = nullptr;

static void ggml_opengl_atexit() {
    if (!g_reg_ctx) {
        return;
    }
    for (auto & dev : g_reg_ctx->devs) {
        if (dev->stats) {
            ggml_opengl_print_stats(*dev);
        }
    }
}

static std::shared_ptr<gl_device_ctx> ggml_opengl_shared_dev(gl_device_ctx * dev) {
    for (auto & d : g_reg_ctx->devs) {
        if (d.get() == dev) {
            return d;
        }
    }
    GGML_ABORT("ggml_opengl: unknown device context");
}

static void APIENTRY ggml_opengl_debug_cb(GLenum src, GLenum type, GLuint id, GLenum sev, GLsizei, const char * msg, const void *) {
    if (sev == E_DEBUG_SEVERITY_NOTIFICATION) {
        return;
    }
    GGML_LOG_WARN("ggml_opengl: [gl debug] src 0x%x type 0x%x id %u sev 0x%x: %s\n", src, type, id, sev, msg);
}

#ifndef _WIN32
static void ggml_opengl_destroy_context(gl_device_ctx & dev) {
    if (dev.dpy) {
        p_eglMakeCurrent(dev.dpy, nullptr, nullptr, nullptr);
        if (dev.ctx) {
            p_eglDestroyContext(dev.dpy, dev.ctx);
            dev.ctx = nullptr;
        }
        p_eglTerminate(dev.dpy);
        dev.dpy = nullptr;
    }
}

// Surfaceless EGL display (EGL_MESA_platform_surfaceless), core 4.6 .. 4.3 context without a config
// (EGL_KHR_no_config_context) and without a surface (EGL_KHR_surfaceless_context).
// Leaves the new context current on this thread.
static bool ggml_opengl_create_context(gl_device_ctx & dev) {
    if (!g_libegl) {
        g_libegl = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!g_libegl) {
            GGML_LOG_WARN("ggml_opengl: libEGL.so.1 not found, no OpenGL device\n");
            return false;
        }
#define GGML_EGL_LOAD(ret, name, args) p_##name = (ret (*) args) dlsym(g_libegl, #name);
        GGML_EGL_FUNCS(GGML_EGL_LOAD)
#undef GGML_EGL_LOAD
    }
#define GGML_EGL_CHECK(ret, name, args) \
    if (!p_##name) { GGML_LOG_WARN("ggml_opengl: libEGL has no %s, no OpenGL device\n", #name); return false; }
    GGML_EGL_FUNCS(GGML_EGL_CHECK)
#undef GGML_EGL_CHECK
    dev.dpy = p_eglGetPlatformDisplay(E_EGL_PLATFORM_SURFACELESS_MESA, nullptr, nullptr);
    EGLint major = 0, minor = 0;
    if (!dev.dpy || !p_eglInitialize(dev.dpy, &major, &minor)) {
        GGML_LOG_WARN("ggml_opengl: no surfaceless EGL display (error 0x%x), no OpenGL device\n", (unsigned) p_eglGetError());
        dev.dpy = nullptr;
        return false;
    }
    const char * exts = p_eglQueryString(dev.dpy, E_EGL_EXTENSIONS);
    if (!exts || !strstr(exts, "EGL_KHR_no_config_context") || !strstr(exts, "EGL_KHR_surfaceless_context") ||
        !p_eglBindAPI(E_EGL_OPENGL_API)) {
        GGML_LOG_WARN("ggml_opengl: EGL %d.%d lacks desktop GL, no_config or surfaceless contexts\n", major, minor);
        ggml_opengl_destroy_context(dev);
        return false;
    }
    const int vers[][2] = { { 4, 6 }, { 4, 5 }, { 4, 4 }, { 4, 3 } };
    for (const auto & v : vers) {
        const EGLint attr[] = { E_EGL_CONTEXT_MAJOR_VERSION, v[0], E_EGL_CONTEXT_MINOR_VERSION, v[1],
                                E_EGL_CONTEXT_OPENGL_PROFILE_MASK, E_EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                                E_EGL_CONTEXT_OPENGL_DEBUG, dev.debug ? 1 : 0, E_EGL_NONE };
        dev.ctx = p_eglCreateContext(dev.dpy, nullptr, nullptr, attr);
        if (dev.ctx) {
            break;
        }
    }
    if (!dev.ctx || !p_eglMakeCurrent(dev.dpy, nullptr, nullptr, dev.ctx)) {
        GGML_LOG_WARN("ggml_opengl: no core 4.3+ context (EGL error 0x%x)\n", (unsigned) p_eglGetError());
        ggml_opengl_destroy_context(dev);
        return false;
    }
    return true;
}
#else
static LRESULT CALLBACK ggml_opengl_wndproc(HWND h, UINT m, WPARAM w, LPARAM l) {
    return DefWindowProcA(h, m, w, l);
}

static void ggml_opengl_destroy_context(gl_device_ctx & dev) {
    wglMakeCurrent(nullptr, nullptr);
    if (dev.ctx) {
        wglDeleteContext(dev.ctx);
        dev.ctx = nullptr;
    }
    if (dev.dc) {
        ReleaseDC(dev.hwnd, dev.dc);
        dev.dc = nullptr;
    }
    if (dev.hwnd) {
        DestroyWindow(dev.hwnd);
        dev.hwnd = nullptr;
    }
}

// Hidden window, legacy context (to get wglCreateContextAttribsARB), then a core 4.6 .. 4.3 context.
// Leaves the new context current on this thread.
static bool ggml_opengl_create_context(gl_device_ctx & dev) {
    g_opengl32 = LoadLibraryA("opengl32.dll");
    HINSTANCE inst = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR) &ggml_opengl_wndproc, &inst);
    WNDCLASSA wc     = {};
    wc.style         = CS_OWNDC;
    wc.lpfnWndProc   = ggml_opengl_wndproc;
    wc.hInstance     = inst;
    wc.lpszClassName = "ggml_opengl";
    if (!RegisterClassA(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        GGML_LOG_WARN("ggml_opengl: RegisterClass failed (error %lu)\n", (unsigned long) GetLastError());
        return false;
    }
    dev.hwnd = CreateWindowA("ggml_opengl", "ggml_opengl", WS_OVERLAPPEDWINDOW, 0, 0, 64, 64, nullptr, nullptr, inst, nullptr);
    dev.dc   = dev.hwnd ? GetDC(dev.hwnd) : nullptr;
    if (!dev.dc) {
        GGML_LOG_WARN("ggml_opengl: hidden window creation failed (error %lu)\n", (unsigned long) GetLastError());
        ggml_opengl_destroy_context(dev);
        return false;
    }
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize      = sizeof(pfd);
    pfd.nVersion   = 1;
    pfd.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    const int pf = ChoosePixelFormat(dev.dc, &pfd);
    if (!pf || !SetPixelFormat(dev.dc, pf, &pfd)) {
        GGML_LOG_WARN("ggml_opengl: no pixel format (error %lu)\n", (unsigned long) GetLastError());
        ggml_opengl_destroy_context(dev);
        return false;
    }
    HGLRC legacy = wglCreateContext(dev.dc);
    if (!legacy || !wglMakeCurrent(dev.dc, legacy)) {
        GGML_LOG_WARN("ggml_opengl: wglCreateContext failed (error %lu)\n", (unsigned long) GetLastError());
        if (legacy) {
            wglDeleteContext(legacy);
        }
        ggml_opengl_destroy_context(dev);
        return false;
    }
    const char * renderer = (const char *) glGetString(GL_RENDERER);
    if (!renderer || strstr(renderer, "GDI Generic")) {
        // Microsoft's GL 1.1 software renderer: no vendor driver in this session
        GGML_LOG_WARN("ggml_opengl: only the GDI Generic software renderer is available, no OpenGL device\n");
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(legacy);
        ggml_opengl_destroy_context(dev);
        return false;
    }
    typedef HGLRC (WINAPI * create_attribs_t)(HDC, HGLRC, const int *);
    auto create_attribs = (create_attribs_t) (void *) ggml_opengl_get_proc("wglCreateContextAttribsARB");
    const int vers[][2] = { { 4, 6 }, { 4, 5 }, { 4, 4 }, { 4, 3 } };
    for (const auto & v : vers) {
        if (!create_attribs) {
            break;
        }
        const int attr[] = { E_WGL_CONTEXT_MAJOR_VERSION, v[0], E_WGL_CONTEXT_MINOR_VERSION, v[1],
                             E_WGL_CONTEXT_FLAGS, dev.debug ? E_WGL_CONTEXT_DEBUG_BIT : 0,
                             E_WGL_CONTEXT_PROFILE_MASK, E_WGL_CONTEXT_CORE_BIT, 0 };
        dev.ctx = create_attribs(dev.dc, nullptr, attr);
        if (dev.ctx) {
            break;
        }
    }
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(legacy);
    if (!dev.ctx || !wglMakeCurrent(dev.dc, dev.ctx)) {
        GGML_LOG_WARN("ggml_opengl: no core 4.3+ context on %s (%s)\n", renderer,
                      create_attribs ? "wglCreateContextAttribsARB failed" : "no wglCreateContextAttribsARB");
        ggml_opengl_destroy_context(dev);
        return false;
    }
    return true;
}
#endif

static bool ggml_opengl_init_device(gl_device_ctx & dev, ggml_backend_dev_t ggml_dev) {
    dev.debug = ggml_opengl_getenv("DEBUG") != nullptr;
    dev.stats = ggml_opengl_getenv("STATS") != nullptr;
    dev.profile = ggml_opengl_getenv("PROFILE") != nullptr;
    dev.trace   = ggml_opengl_getenv("TRACE") != nullptr;
#ifdef GGML_OPENGL_TRACE_DEFAULT
    // diagnostic builds for the test boxes, whose launcher cannot set environment variables
    dev.trace = true;
#endif
#ifdef GGML_OPENGL_PROFILE_DEFAULT
    // diagnostic builds for the test boxes, whose launcher cannot set environment variables
    dev.profile = true;
#endif
    dev.stats   = dev.stats || dev.profile;
    if (!ggml_opengl_create_context(dev)) {
        return false;
    }
    if (ggml_opengl_load_funcs() != 0) {
        GGML_LOG_WARN("ggml_opengl: required GL functions missing, no OpenGL device\n");
        ggml_opengl_destroy_context(dev);
        return false;
    }
    dev.desc    = (const char *) glGetString(GL_RENDERER);
    dev.version = (const char *) glGetString(GL_VERSION);
    // Moore Threads: the driver's compiler aborts on the tiled matmul's barriers inside a loop, with a runtime or a
    // constant bound ("LLVM ERROR: barrierOp in dynamic branch", exp43/exp44), so the tiled matmul runs its k loop
    // as one dispatch per 4 tiles there (exp91); mul_mat_id stays on the mat-vec kernel
    if (dev.desc.find("Moore Threads") != std::string::npos) {
        dev.tiled_ksplit = 4;
    }

    std::map<std::string, bool> exts;
    GLint n_ext = 0;
    glGetIntegerv(E_NUM_EXTENSIONS, &n_ext);
    for (GLint i = 0; i < n_ext; i++) {
        exts[(const char *) p_glGetStringi(GL_EXTENSIONS, (GLuint) i)] = true;
    }
    dev.ati_meminfo = exts.count("GL_ATI_meminfo") != 0;
    dev.nvx_meminfo = exts.count("GL_NVX_gpu_memory_info") != 0;

    GLint cflags = 0;
    glGetIntegerv(E_CONTEXT_FLAGS, &cflags);
    if (dev.debug && (cflags & E_CONTEXT_FLAG_DEBUG_BIT) && p_glDebugMessageCallback) {
        glEnable(E_DEBUG_OUTPUT);
        glEnable(E_DEBUG_OUTPUT_SYNCHRONOUS);
        p_glDebugMessageCallback(ggml_opengl_debug_cb, nullptr);
    }

    // parallel compilation: all cores, the driver decides how many threads that is
    if (p_glMaxShaderCompilerThreadsARB && exts.count("GL_ARB_parallel_shader_compile")) {
        p_glMaxShaderCompilerThreadsARB(0xFFFFFFFFu);
        dev.parallel = true;
    } else if (p_glMaxShaderCompilerThreadsKHR && exts.count("GL_KHR_parallel_shader_compile")) {
        p_glMaxShaderCompilerThreadsKHR(0xFFFFFFFFu);
        dev.parallel = true;
    }
    GLint n_formats = 0;
    glGetIntegerv(E_NUM_PROGRAM_BINARY_FORMATS, &n_formats);
    dev.prog_binary = n_formats > 0 && p_glGetProgramBinary && p_glProgramBinary;

    // one SSBO binding cannot be larger than MAX_SHADER_STORAGE_BLOCK_SIZE, so neither can one buffer
    long long max_block = 0;
    p_glGetInteger64v(E_MAX_SHADER_STORAGE_BLOCK_SIZE, &max_block);
    GLint ssbo_align = 0, ubo_align = 0;
    glGetIntegerv(E_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &ssbo_align);
    glGetIntegerv(E_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &ubo_align);
    // the tensor binding alignment is a power of two (mask arithmetic); drivers report powers of two
    while (dev.bind_align < (size_t) ssbo_align) {
        dev.bind_align *= 2;
    }
    GGML_ASSERT(ssbo_align <= 0 || dev.bind_align % (size_t) ssbo_align == 0);
    if (ubo_align > 0) {
        dev.param_slot = CEIL_DIV((size_t) GGML_GL_PARAM_SLOT_SIZE, (size_t) ubo_align) * (size_t) ubo_align;
    }
    dev.max_alloc = ((size_t) std::max(0ll, max_block) & ~(dev.bind_align - 1));
    dev.max_alloc = dev.max_alloc > GGML_GL_BUFFER_SLACK + dev.bind_align ? dev.max_alloc - GGML_GL_BUFFER_SLACK - dev.bind_align : 0;
    if (const char * env = ggml_opengl_getenv("MAX_ALLOC_MB")) {
        dev.max_alloc = std::min(dev.max_alloc, (size_t) atoll(env) * 1024 * 1024);
    }
    if (dev.max_alloc == 0) {
        GGML_LOG_WARN("ggml_opengl: MAX_SHADER_STORAGE_BLOCK_SIZE is %lld, no OpenGL device\n", max_block);
        ggml_opengl_destroy_context(dev);
        return false;
    }
    if (const char * env = ggml_opengl_getenv("TILED")) {
        dev.tiled_min_cols = (uint32_t) std::max(0, atoi(env));
    }
    if (const char * env = ggml_opengl_getenv("KSPLIT")) {
        const int v = atoi(env);
        dev.tiled_ksplit = v == 1 || v == 2 || v == 4 || v == 8 ? (uint32_t) v : 0;
    }
    if (const char * env = ggml_opengl_getenv("NO_FUSE")) {
        dev.no_fuse = atoi(env) != 0;
    }
    if (const char * env = ggml_opengl_getenv("MAX_TPR")) {
        // power of two in 1..WG_SIZE
        const int v = atoi(env);
        dev.max_tpr = 1;
        while (dev.max_tpr * 2 <= (uint32_t) std::max(1, std::min(v, GGML_GL_WG_SIZE))) {
            dev.max_tpr *= 2;
        }
    }

    p_glGenBuffers(1, &dev.ubo);
    p_glBindBuffer(E_UNIFORM_BUFFER, dev.ubo);
    const size_t ubo_size = dev.param_slot * GGML_GL_PARAM_SLOT_COUNT;
    const char * persist  = ggml_opengl_getenv("PARAM_PERSIST");
    if (p_glBufferStorage && p_glMapBufferRange && p_glFenceSync && p_glClientWaitSync && p_glDeleteSync &&
        !(persist && atoi(persist) == 0)) {
        const GLbitfield flags = E_MAP_WRITE_BIT | E_MAP_PERSISTENT_BIT | E_MAP_COHERENT_BIT;
        p_glBufferStorage(E_UNIFORM_BUFFER, (ptrdiff_t) ubo_size, nullptr, flags);
        if (glGetError() == GL_NO_ERROR) {
            dev.ubo_map = (uint8_t *) p_glMapBufferRange(E_UNIFORM_BUFFER, 0, (ptrdiff_t) ubo_size, flags);
        }
        if (!dev.ubo_map) {
            // storage is immutable once set: start over with a new buffer
            while (glGetError() != GL_NO_ERROR) {
            }
            p_glDeleteBuffers(1, &dev.ubo);
            p_glGenBuffers(1, &dev.ubo);
            p_glBindBuffer(E_UNIFORM_BUFFER, dev.ubo);
        }
    }
    if (!dev.ubo_map) {
        p_glBufferData(E_UNIFORM_BUFFER, (ptrdiff_t) ubo_size, nullptr, E_DYNAMIC_DRAW);
    }
    if (glGetError() != GL_NO_ERROR) {
        GGML_LOG_WARN("ggml_opengl: parameter buffer allocation failed, no OpenGL device\n");
        ggml_opengl_destroy_context(dev);
        return false;
    }

    if (dev.stats) {
        static bool registered = false;
        if (!registered) {
            registered = true;
            atexit(ggml_opengl_atexit);
        }
    }

    dev.buft = {
        /* .iface = */ {
            /* .get_name       = */ ggml_backend_opengl_buffer_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_opengl_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_opengl_buffer_type_get_alignment,
            /* .get_max_size   = */ ggml_backend_opengl_buffer_type_get_max_size,
            /* .get_alloc_size = */ NULL,
            /* .is_host        = */ NULL,
        },
        /* .device  = */ ggml_dev,
        /* .context = */ &dev,
    };

    ggml_opengl_prewarm(dev);
    if (dev.debug) {
        ggml_opengl_check_error("device init");
    }
    GGML_LOG_WARN("ggml_opengl: %s = %s | %s | GLSL %s | max buffer %zu MiB | SSBO align %d | parallel compile %s | "
                  "program binaries %s | build %s %s\n",
                  dev.name.c_str(), dev.desc.c_str(), dev.version.c_str(),
                  (const char *) glGetString(E_SHADING_LANGUAGE_VERSION), dev.max_alloc >> 20, ssbo_align,
                  dev.parallel ? "yes" : "no", dev.prog_binary ? "yes" : "no", __DATE__, __TIME__);
    {
        // raw vendor memory info (KiB), the source of get_memory; none -> get_memory reports 4 GiB
        GLint nvx_total = -1, nvx_avail = -1, ati[4] = { -1, -1, -1, -1 };
        if (dev.nvx_meminfo) {
            glGetIntegerv(E_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX, &nvx_total);
            glGetIntegerv(E_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &nvx_avail);
        }
        if (dev.ati_meminfo) {
            glGetIntegerv(E_VBO_FREE_MEMORY_ATI, ati);
        }
        GGML_LOG_INFO("ggml_opengl: %s meminfo: NVX %s dedicated %d KiB avail %d KiB | ATI %s vbo free %d KiB largest %d KiB\n",
                      dev.name.c_str(), dev.nvx_meminfo ? "yes" : "no", nvx_total, nvx_avail,
                      dev.ati_meminfo ? "yes" : "no", ati[0], ati[1]);
    }
    // other threads make the context current when they need it
    ggml_opengl_release_current(dev);
    return true;
}

#ifdef _WIN32
static int __cdecl ggml_opengl_process_exit() {
    if (g_reg_ctx) {
        for (auto & dev : g_reg_ctx->devs) {
            std::lock_guard<std::recursive_mutex> lock(dev->mutex);
            ggml_opengl_destroy_context(*dev);
        }
    }
    return 0;
}
#endif

static void ggml_opengl_enumerate(ggml_backend_opengl_reg_context & reg_ctx, ggml_backend_reg_t reg) {
    // WGL gives one context on the driver of the primary display: one device
    auto dev  = std::make_shared<gl_device_ctx>();
    dev->name = GGML_OPENGL_NAME "0";
    reg_ctx.devs.push_back(dev);
    reg_ctx.devices.reserve(1);
    reg_ctx.devices.push_back({
        /* .iface   = */ ggml_backend_opengl_device_i,
        /* .reg     = */ reg,
        /* .context = */ dev.get(),
    });
    if (!ggml_opengl_init_device(*dev, &reg_ctx.devices.back())) {
        reg_ctx.devices.clear();
        reg_ctx.devs.clear();
        return;
    }
    // At ExitProcess the loader detaches the ICD (loaded last) before OPENGL32, whose own detach then cleans up
    // live contexts through the detached ICD: a call to 0 on the MTT S80 (exp33). Delete the contexts in the
    // process exit handlers instead, which run before ExitProcess; a DLL's own atexit only runs at its detach.
    // Pin this DLL so the handler cannot outlive it.
#ifdef _WIN32
    HMODULE self = nullptr;
    auto crt_onexit = (_onexit_t (__cdecl *)(_onexit_t)) (void *) GetProcAddress(GetModuleHandleA("msvcrt.dll"), "_onexit");
    if (crt_onexit && GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                         (LPCSTR) &ggml_opengl_process_exit, &self)) {
        crt_onexit(ggml_opengl_process_exit);
    }
#endif
}

static const char * ggml_backend_opengl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_OPENGL_NAME;
}

static size_t ggml_backend_opengl_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * ctx = (ggml_backend_opengl_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_opengl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = (ggml_backend_opengl_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return &ctx->devices[index];
}

static const struct ggml_backend_reg_i ggml_backend_opengl_reg_i = {
    /* .get_name         = */ ggml_backend_opengl_reg_get_name,
    /* .get_device_count = */ ggml_backend_opengl_reg_get_device_count,
    /* .get_device       = */ ggml_backend_opengl_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_opengl_reg() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    // leaked on purpose: GL objects must not be torn down during static destruction
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_opengl_reg_i,
        /* .context     = */ nullptr,
    };
    if (g_reg_ctx == nullptr) {
        g_reg_ctx   = new ggml_backend_opengl_reg_context();
        reg.context = g_reg_ctx;
        ggml_opengl_enumerate(*g_reg_ctx, &reg);
    }
    return &reg;
}

ggml_backend_t ggml_backend_opengl_init(int device) {
    ggml_backend_reg_t reg = ggml_backend_opengl_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_opengl_device_init_backend(ggml_backend_reg_dev_get(reg, device), nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_opengl_reg)
