#ifndef GLAD_CONFIG_H
#define GLAD_CONFIG_H

#define GLAD_GL_IMPLEMENTATION
#include <glad/glad.h>

// OpenGL constants if not defined by GLAD
#ifndef GL_MAP_PERSISTENT_BIT
#define GL_MAP_PERSISTENT_BIT     0x0040
#endif

#ifndef GL_MAP_COHERENT_BIT
#define GL_MAP_COHERENT_BIT       0x0080
#endif

#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER  0x90D2
#endif

#endif