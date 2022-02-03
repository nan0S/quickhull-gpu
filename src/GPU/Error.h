#pragma once

#include <cassert>

#define CUDA_CALL(...) \
   __VA_ARGS__; \
   cudaCheckErrors(#__VA_ARGS__, __FILE__, __LINE__)

void cudaCheckErrors(const char* call, const char* file, int line);
