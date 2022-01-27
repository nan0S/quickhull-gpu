#pragma once

#include <vector>
#include <GL/glew.h>

#include "Config.h"

namespace CPU
{
	void init(Config config, const std::vector<int>& ns);
	int calculate(int n);
	void terminate();
}
