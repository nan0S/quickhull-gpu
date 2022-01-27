#pragma once

#include <vector>
#include <GL/glew.h>

#include "Config.h"

namespace GPU
{
	void init(Config config, const std::vector<int>& ns, GLuint vbo);
	int calculate(int n);
	void terminate();
}
