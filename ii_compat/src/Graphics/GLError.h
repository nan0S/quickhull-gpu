#pragma once

#include "Debug//Assert.h"

#define glCall(x) \
	glClearError(); \
	x; \
	glCheckErrorExpr(#x)

#define glCheckError() \
	glCheckErrorExpr("")

#define glCheckErrorExpr(x) \
	ASSERT(glLogError(x, __FILE__, __LINE__));

void glClearError();
bool glLogError(const char* call, const char* file, int line);
