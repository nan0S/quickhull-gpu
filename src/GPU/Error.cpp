#include "Error.h"

#include <filesystem>

#include <cuda_runtime.h>

#include "Utils/Log.h"

namespace fs = std::filesystem;

void cudaCheckErrors(const char* call, const char* file, int line)
{
   cudaError_t code = cudaGetLastError();
   if (code == cudaSuccess)
      return;
   const std::string& filename = fs::path(file).filename().string();
   const char* err = cudaGetErrorString(code);
   ERROR("CUDA call failure: ", call, " '", err, "' (", code, ") ", "at ", filename, ":", line);
}
