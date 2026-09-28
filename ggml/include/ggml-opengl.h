#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_OPENGL_NAME "OpenGL"

GGML_BACKEND_API ggml_backend_t ggml_backend_opengl_init(int device);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opengl_reg(void);

#ifdef  __cplusplus
}
#endif
