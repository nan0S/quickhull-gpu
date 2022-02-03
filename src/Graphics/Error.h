#pragma once

#include <cassert>

#define GL_CALL(x) \
   glClearError(); \
   x; \
   assert(glLogError(#x, __FILE__, __LINE__))

void glClearError();
bool glLogError(const char* call, const char* file, int line);
