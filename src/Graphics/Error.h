#pragma once

#include <cassert>

#define glCall(x) \
   glClearError(); \
   x; \
   glCheckErrorExpr(#x)

#define glCheckError() \
   glCheckErrorExpr("")

#define glCheckErrorExpr(x) \
   assert(glLogError(x, __FILE__, __LINE__));

void glClearError();
bool glLogError(const char* call, const char* file, int line);
