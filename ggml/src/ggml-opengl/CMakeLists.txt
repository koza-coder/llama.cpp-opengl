cmake_minimum_required(VERSION 3.13)

find_package(Python3 REQUIRED)

# GLSL 430 compute kernels, compiled at runtime by the OpenGL driver
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/glsl")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(SHADER_HEADER "${SHADER_OUTPUT_DIR}/ggml-opengl-shaders.hpp")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

file(GLOB GLSL_SHADER_FILES
    "${SHADER_DIR}/*.comp"
    "${SHADER_DIR}/*.glsl"
)

add_custom_command(
    OUTPUT ${SHADER_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONIOENCODING=utf-8
        ${Python3_EXECUTABLE} ${SHADER_DIR}/embed_glsl.py
            --input_dir "${SHADER_DIR}"
            --output_file "${SHADER_HEADER}"
    DEPENDS ${GLSL_SHADER_FILES} ${SHADER_DIR}/embed_glsl.py
    VERBATIM
)

add_custom_target(generate_opengl_shaders DEPENDS ${SHADER_HEADER})

ggml_add_backend_library(ggml-opengl
    ggml-opengl.cpp
    ${SHADER_HEADER}
    ../../include/ggml-opengl.h
)

add_dependencies(ggml-opengl generate_opengl_shaders)

target_compile_features(ggml-opengl PRIVATE cxx_std_17)
target_include_directories(ggml-opengl PRIVATE ${SHADER_OUTPUT_DIR})
# every GL function past 1.1 is loaded with wglGetProcAddress (Windows) or eglGetProcAddress (Linux, libEGL
# opened at run time); no GLAD/GLEW and no GL/EGL headers needed
if (WIN32)
    target_link_libraries(ggml-opengl PRIVATE opengl32 gdi32 user32)
else()
    target_link_libraries(ggml-opengl PRIVATE ${CMAKE_DL_LIBS})
endif()
