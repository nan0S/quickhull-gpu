#pragma once

#include <cassert>

#include <curand.h>

#define curandCall(x) \
   if ((x) != CURAND_STATUS_SUCCESS) \
      ERROR("cuRAND failure at ", __FILE__, ":", __LINE__)

#define cudaCall(x) \
   x; \
   cudaCheckErrors(#x, __FILE__, __LINE__)

void cudaCheckErrors(const char* call, const char* file, int line);
