#pragma once

#include <GL/glew.h>

namespace Graphics
{
   GLuint compileShader(const char* vertex_source, const char* fragment_source);
}
