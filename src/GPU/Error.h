#pragma once

#include <cassert>

#define cudaCall(x) \
   x; \
   cudaCheckErrors(#x, __FILE__, __LINE__)

void cudaCheckErrors(const char* call, const char* file, int line);
