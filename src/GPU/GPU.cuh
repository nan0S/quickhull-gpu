#pragma once

#include <vector>
#include <GL/glew.h>

#include "Config.h"

namespace GPU
{
   void init(
            Config config,
            const std::vector<int>& n_points,
            GLuint gl_buffer);
   int calculate(int n);
   void cleanup();
}
