#pragma once

#include <vector>
#include <GL/glew.h>

#include "Config.h"

namespace CPU
{
   void init(Config config, const std::vector<int>& num_points);
   int calculate(int n);
   void cleanup();
}
