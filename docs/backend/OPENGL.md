# llama.cpp for OpenGL

This repository adds an **OpenGL 4.3 compute backend** to ggml / llama.cpp. It runs models on any Windows GPU whose
driver offers an OpenGL 4.3 (or newer) core context, including GPUs without Vulkan, CUDA or ROCm support.

- Backend source: [ggml/src/ggml-opengl](../../ggml/src/ggml-opengl) (host code `ggml-opengl.cpp`, GLSL compute shaders in `glsl/`)
- Device name in llama.cpp: `OpenGL0`, `OpenGL1`, ...
- Platform: Windows x64 (WGL). A Linux build (surfaceless EGL) compiles and passes the op tests on Mesa llvmpipe;
  it has not been tested on a Linux GPU yet.

## Status

Tested with release v0.0.3:

| GPU | Result |
|---|---|
| AMD Radeon AI PRO R9700 | new op groups pass (flash attention 3260, MUL_MAT_ID 768, GET_ROWS, IM2COL, FILL, ROPE_BACK, SWIGLU_CLAMP); qwen2.5-0.5b q4_0 PPL 17.29, pp512 5948 / tg64 86 tok/s; SmolVLM-500M (image) and Qwen3-ASR (speech) correct |
| Intel Iris Xe | new op groups pass (MUL_MAT not run, too slow on that machine); qwen2.5-0.5b q4_0 PPL 17.29, pp512 323 / tg64 6.5 tok/s; SmolVLM-500M and Qwen3-ASR correct |
| Moore Threads MTT S80 | not tested with v0.0.3 (offline). v0.0.2: flash attention and quantized MUL_MAT pass; 37 f32 MUL_MAT broadcast cases fail; the tiled prompt kernel is off on this GPU |

Ops the backend does not support run on the CPU (the ggml scheduler does this automatically). The authoritative list
of supported ops and types is `ggml_backend_opengl_device_supports_op` in `ggml-opengl.cpp`.

## Download

Prebuilt Windows x64 zips are on the [releases page](https://github.com/koza-coder/llama.cpp-opengl/releases). The
binaries are statically linked, need no installer and target plain x86-64 (no AVX requirement).

```
llama-cli.exe -m model.gguf -ngl 99
```

## Build

The tested way is to cross-compile on Linux (Debian/Ubuntu, x86-64 or arm64) with MinGW-w64:

```
sudo apt install cmake python3 gcc-mingw-w64-x86-64-posix g++-mingw-w64-x86-64-posix binutils-mingw-w64-x86-64
```

```
git clone https://github.com/koza-coder/llama.cpp-opengl
cd llama.cpp-opengl
```

```
cmake -B build-win \
    -DCMAKE_TOOLCHAIN_FILE=cmake/x86_64-w64-mingw32.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENGL=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="-static -static-libgcc -static-libstdc++"
```

```
cmake --build build-win -j 4 --target llama-cli llama-server llama-bench test-backend-ops
```

The configure step warns that OpenSSL and ccache are missing; that is expected (no HTTPS model download). The .exe files are in `build-win/bin/`. Copy them to the Windows machine; they need no extra DLLs.

Python 3 is needed at build time only: `glsl/embed_glsl.py` embeds the shaders into the binary. A native Windows build
(MSYS2 MinGW or MSVC) has not been tested.

## Test

```
test-backend-ops.exe -b OpenGL0
```

Compares every supported op against the CPU backend. `-o MUL_MAT` (or any op name) limits the run to one op.

## Environment variables

| Variable | Effect |
|---|---|
| `GGML_OPENGL_NO_SHADER_CACHE` | do not read or write the compiled shader cache (`opengl-shader-cache` next to the exe) |
| `GGML_OPENGL_TILED=<n>` | use the tiled matmul from `n` columns up (0 = off); default 16, off on Moore Threads |
| `GGML_OPENGL_MAX_ALLOC_MB=<n>` | cap the size of one GPU buffer |
| `GGML_OPENGL_STATS` | print counters (graphs, dispatches, barriers, graph time, shader compiles vs. cache hits, transfers) |
| `GGML_OPENGL_DEBUG` | enable the GL debug output callback (when the driver provides a debug context) |
| `GGML_OPENGL_PROFILE` | GPU time per shader from timer queries, printed at exit with the STATS line (slows every dispatch) |
| `GGML_OPENGL_NO_FUSE` | do not fuse RMS_NORM + MUL and mat-vec MUL_MAT + bias ADD |
| `GGML_OPENGL_PARAM_PERSIST=0` | write dispatch parameters with glBufferSubData instead of the persistent mapped ring |
| `GGML_OPENGL_MAX_TPR=<n>` | cap the matrix-vector threads per row (power of two, default 256) |

The first run compiles the shaders; later runs load them from the disk cache.
