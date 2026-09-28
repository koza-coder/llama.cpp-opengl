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

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <GL/gl.h>

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
#define GGML_GL_MAX_WG_PER_DIM       65535
#define GGML_GL_BINDING_ALIGNMENT    256   // minimum tensor binding alignment; raised to the driver's SSBO offset alignment
#define GGML_GL_PARAM_SLOT_SIZE      256
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
};
enum : int {
    E_WGL_CONTEXT_MAJOR_VERSION = 0x2091,
    E_WGL_CONTEXT_MINOR_VERSION = 0x2092,
    E_WGL_CONTEXT_FLAGS         = 0x2094,
    E_WGL_CONTEXT_DEBUG_BIT     = 0x0001,
    E_WGL_CONTEXT_PROFILE_MASK  = 0x9126,
    E_WGL_CONTEXT_CORE_BIT      = 0x0001,
};

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
    X(void, glMaxShaderCompilerThreadsKHR, (GLuint))

// one device only (WGL has no adapter choice), so one global table
#define GGML_GL_DECL(ret, name, args) static ret (APIENTRY * p_##name) args = nullptr;
GGML_GL_FUNCS(GGML_GL_DECL)
GGML_GL_FUNCS_OPT(GGML_GL_DECL)
#undef GGML_GL_DECL

static HMODULE g_opengl32 = nullptr;

static PROC ggml_opengl_get_proc(const char * name) {
    PROC     p = wglGetProcAddress(name);
    intptr_t v = (intptr_t) p;
    if (v == 0 || v == 1 || v == 2 || v == 3 || v == -1) {
        p = g_opengl32 ? GetProcAddress(g_opengl32, name) : nullptr;
    }
    return p;
}

// returns the number of missing required functions
static int ggml_opengl_load_funcs() {
    int missing = 0;
#define GGML_GL_LOAD(ret, name, args) \
    p_##name = (ret (APIENTRY *) args) (void *) ggml_opengl_get_proc(#name); \
    if (!p_##name) { GGML_LOG_WARN("ggml_opengl: missing %s\n", #name); missing++; }
    GGML_GL_FUNCS(GGML_GL_LOAD)
#undef GGML_GL_LOAD
#define GGML_GL_LOAD_OPT(ret, name, args) p_##name = (ret (APIENTRY *) args) (void *) ggml_opengl_get_proc(#name);
    GGML_GL_FUNCS_OPT(GGML_GL_LOAD_OPT)
#undef GGML_GL_LOAD_OPT
    return missing;
}

static double ggml_opengl_time_us() {
    static LARGE_INTEGER freq = {};
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double) now.QuadPart * 1e6 / (double) freq.QuadPart;
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
    HWND        hwnd = nullptr;
    HDC         dc   = nullptr;
    HGLRC       ctx  = nullptr;

    size_t   max_alloc    = 0;
    size_t   bind_align   = GGML_GL_BINDING_ALIGNMENT;
    size_t   param_slot   = GGML_GL_PARAM_SLOT_SIZE;
    bool     debug        = false;
    bool     parallel     = false;   // GL_ARB/KHR_parallel_shader_compile
    bool     prog_binary  = false;   // at least one program binary format
    bool     ati_meminfo  = false;
    bool     nvx_meminfo  = false;
    uint32_t tiled_min_cols = GGML_GL_TILED_DEFAULT;

    GLuint   ubo       = 0;   // kernel parameters, one slot per dispatch, reused round robin
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

// program binaries are cached in opengl-shader-cache next to ggml-opengl.dll; GGML_OPENGL_NO_SHADER_CACHE
// disables it. Unwritable folders just skip the cache.
static const std::wstring & ggml_opengl_shader_cache_dir() {
    static const std::wstring dir = []() -> std::wstring {
        if (getenv("GGML_OPENGL_NO_SHADER_CACHE") != nullptr) {
            return L"";
        }
        HMODULE module = nullptr;
        wchar_t path[MAX_PATH];
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                (LPCWSTR) &ggml_opengl_shader_cache_dir, &module) ||
            GetModuleFileNameW(module, path, MAX_PATH) - 1 >= MAX_PATH - 1) {
            return L"";
        }
        std::wstring d = path;
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
    std::wstring name = L"\\0000000000000000.glbin";
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
    return dir.empty() ? L"" : dir + L"\\shaders.txt";
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
// context current.
static void ggml_opengl_build_pipeline_jobs(gl_device_ctx & dev) {
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
        ggml_opengl_build_pipeline_jobs(dev);
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
    dev.next_slot         = (dev.next_slot + 1) % GGML_GL_PARAM_SLOT_COUNT;
    p_glBindBuffer(E_UNIFORM_BUFFER, dev.ubo);
    p_glBufferSubData(E_UNIFORM_BUFFER, (ptrdiff_t) slot_off, (ptrdiff_t) (params.size() * sizeof(uint32_t)), params.data());
    p_glBindBufferRange(E_UNIFORM_BUFFER, 0, dev.ubo, (ptrdiff_t) slot_off, GGML_GL_PARAM_SLOT_SIZE);

    p_glUseProgram(pipeline.prog);
    for (size_t i = 0; i < bindings.size(); i++) {
        const gl_binding & b = bindings[i];
        p_glBindBufferRange(E_SHADER_STORAGE_BUFFER, (GLuint) i, b.buf, (ptrdiff_t) b.offset, (ptrdiff_t) b.size);
    }
    if (total_wg > 0) {
        p_glDispatchCompute(wg_x, wg_y, 1);
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
        case GGML_TYPE_I32: s += "_I32"; break;
        case GGML_TYPE_Q4_0: s += "_Q4_0"; break;
        case GGML_TYPE_Q4_K: s += "_Q4_K"; break;
        case GGML_TYPE_Q8_0: s += "_Q8_0"; break;
        case GGML_TYPE_Q6_K: s += "_Q6_K"; break;
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
    const char * op_define = dst->op == GGML_OP_ADD ? "OP_ADD" : "OP_MUL";
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

static void ggml_opengl_rms_norm(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "rms_norm", glsl_rms_norm, {});

    const gl_binding bs     = ggml_opengl_bind_tensor(dev, src);
    const gl_binding bd     = ggml_opengl_bind_tensor(dev, dst);
    const uint32_t   n_rows = (uint32_t) ggml_nrows(src);

    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], n_rows,
        ggml_opengl_u32_from_f32(ggml_get_op_params_f32(dst, 0)),   // eps
    };
    // one workgroup per row
    ggml_opengl_dispatch(dev, pipeline, params, { bs, bd }, n_rows);
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

static void ggml_opengl_get_rows(gl_device_ctx & dev, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
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
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "mul_mat_tiled", glsl_mul_mat_tiled, defines);

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
    ggml_opengl_dispatch(dev, pipeline, params, { b1, b0, bd }, tiles_m * tiles_n * batches);
}

static void ggml_opengl_mul_mat(gl_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    const bool quant = ggml_is_quantized(src0->type);
    // long prompts go to the tiled kernel; it needs k in whole tiles of 32
    if (dev.tiled_min_cols != 0 && (uint32_t) dst->ne[1] >= dev.tiled_min_cols && src0->ne[0] % 32 == 0) {
        ggml_opengl_mul_mat_tiled(dev, src0, src1, dst);
        return;
    }
    // threads per row: one unit (4 floats or one 32-wide block) per thread, power of two, at most WG_SIZE
    const uint32_t units = (uint32_t) (quant ? src0->ne[0] / 32 : src0->ne[0] / 4);
    uint32_t       tpr   = 1;
    while (tpr < units && tpr < GGML_GL_WG_SIZE) {
        tpr *= 2;
    }
    std::vector<std::string> defines = { ggml_opengl_type_define(src0->type, "SRC0"), "TPR=" + std::to_string(tpr) };
    if (src1->type == GGML_TYPE_F16) {
        defines.push_back("SRC1_F16");
    }
    gl_pipeline & pipeline = ggml_opengl_get_pipeline(dev, "mul_mat_vec", glsl_mul_mat_vec, defines);

    const gl_binding b0 = ggml_opengl_bind_tensor(dev, src0);
    const gl_binding b1 = ggml_opengl_bind_tensor(dev, src1);
    const gl_binding bd = ggml_opengl_bind_tensor(dev, dst);
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
    const size_t   col0_idx = 10;
    const uint32_t total_wg = CEIL_DIV((uint32_t) dst->ne[0], GGML_GL_WG_SIZE / tpr) * (uint32_t) (dst->ne[2] * dst->ne[3]);
    // columns are processed 4 at a time (MAX_COLS in mul_mat_vec.comp); every chunk re-reads src0
    for (uint32_t col0 = 0; col0 < (uint32_t) dst->ne[1]; col0 += 4) {
        params[col0_idx] = col0;
        ggml_opengl_dispatch(dev, pipeline, params, { b1, b0, bd }, total_wg);
    }
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
        case GGML_OP_MUL:
            ggml_opengl_binary_op(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_SCALE:
            ggml_opengl_scale(dev, node->src[0], node);
            return;
        case GGML_OP_GET_ROWS:
            ggml_opengl_get_rows(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_RMS_NORM:
            ggml_opengl_rms_norm(dev, node->src[0], node);
            return;
        case GGML_OP_GLU:
            ggml_opengl_glu(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_ROPE:
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
        default:
            GGML_ABORT("ggml_opengl: unsupported op %s", ggml_op_name(node->op));
    }
}

/* GGML Backend Interface */

static const char * ggml_backend_opengl_name(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_opengl_context *) backend->context;
    return ctx->name.c_str();
}

static void ggml_opengl_print_stats(gl_device_ctx & dev) {
    fprintf(stderr, "ggml_opengl stats [%s]: graphs %llu, nodes %llu, dispatches %llu, barriers %llu | graph_compute %.1f ms "
                    "| shader compile %.1f ms (%llu from source, %llu from the disk cache) | set_tensor %llu calls %.1f MB %.1f ms "
                    "| get_tensor %llu calls %.1f MB %.1f ms\n",
            dev.name.c_str(), (unsigned long long) dev.n_graphs, (unsigned long long) dev.n_nodes,
            (unsigned long long) dev.n_dispatches, (unsigned long long) dev.n_barriers, dev.t_graph_us / 1000.0,
            dev.t_compile_us / 1000.0, (unsigned long long) dev.n_compiles, (unsigned long long) dev.n_cache_hits,
            (unsigned long long) dev.n_set_tensor, dev.bytes_set / 1e6, dev.t_set_us / 1000.0,
            (unsigned long long) dev.n_get_tensor, dev.bytes_get / 1e6, dev.t_get_us / 1000.0);
    fflush(stderr);
}

static void ggml_backend_opengl_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_opengl_context *) backend->context;
    delete ctx;
    delete backend;
}

static ggml_status ggml_backend_opengl_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto *          ctx = (ggml_backend_opengl_context *) backend->context;
    gl_device_ctx & dev = *ctx->dev;
    gl_scope        scope(dev);
    const double    t0 = ggml_opengl_time_us();
    if (dev.n_graphs < 4 || cgraph->n_nodes != dev.last_graph_nodes) {
        // a graph of a new shape: find the programs it needs first and build the missing ones together
        dev.collecting = true;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_opengl_encode_node(dev, cgraph->nodes[i]);
        }
        dev.collecting = false;
        ggml_opengl_build_pipeline_jobs(dev);
    }
    dev.last_graph_nodes = cgraph->n_nodes;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_opengl_encode_node(dev, cgraph->nodes[i]);
    }
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
    if (ctx->nvx_meminfo) {
        GLint dedicated = 0, avail = 0;
        glGetIntegerv(E_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX, &dedicated);
        glGetIntegerv(E_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &avail);
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
        case GGML_OP_MUL:
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                   src0->type == op->type && src1->type == op->type;
        case GGML_OP_SCALE:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
        case GGML_OP_RMS_NORM:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src0->nb[0] == sizeof(float);
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU: case GGML_GLU_OP_GEGLU: case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF: case GGML_GLU_OP_GEGLU_QUICK:
                    return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           src0->type == op->type && (!src1 || src1->type == op->type);
                case GGML_GLU_OP_SWIGLU_OAI:
                    return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && (!src1 || src1->type == GGML_TYPE_F32);
                default:
                    return false;
            }
        case GGML_OP_ROPE:
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && src0->type == op->type &&
                   src1->type == GGML_TYPE_I32 && src0->ne[0] % 2 == 0 &&
                   (!op->src[2] || op->src[2]->type == GGML_TYPE_F32);
        case GGML_OP_SOFT_MAX:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                   (!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                   (!op->src[2] || op->src[2]->type == GGML_TYPE_F32);
        case GGML_OP_SET_ROWS:
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && src0->type == GGML_TYPE_F32 &&
                   (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);
        case GGML_OP_GET_ROWS:
            if (src1->type != GGML_TYPE_I32) {
                return false;
            }
            switch (src0->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                    return op->type == GGML_TYPE_F32;
                case GGML_TYPE_I32:
                    return op->type == GGML_TYPE_I32;
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
            // contiguous rows. Float weights: k in units of 4, f32 or f16 columns. Q4_0 / Q8_0 / Q4_K / Q6_K
            // weights: whole blocks, f32 columns
            if (src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q8_0 ||
                src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K) {
                return src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                       src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == ggml_type_size(src1->type) &&
                       src0->ne[0] % ggml_blck_size(src0->type) == 0;
            }
            return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                   (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == ggml_type_size(src1->type) &&
                   src0->ne[0] % 4 == 0;
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

static bool ggml_opengl_init_device(gl_device_ctx & dev, ggml_backend_dev_t ggml_dev) {
    dev.debug = ggml_opengl_getenv("DEBUG") != nullptr;
    dev.stats = ggml_opengl_getenv("STATS") != nullptr;
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

    p_glGenBuffers(1, &dev.ubo);
    p_glBindBuffer(E_UNIFORM_BUFFER, dev.ubo);
    p_glBufferData(E_UNIFORM_BUFFER, (ptrdiff_t) (dev.param_slot * GGML_GL_PARAM_SLOT_COUNT), nullptr, E_DYNAMIC_DRAW);
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
    // other threads make the context current when they need it
    wglMakeCurrent(nullptr, nullptr);
    return true;
}

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
    }
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
