#include "CUDAError.h"

void cudaCheckErrors(const char* call, const char* file, int line)
{
	cudaError_t code = cudaGetLastError();
	if (code == cudaSuccess)
		return;
	const char* err = cudaGetErrorString(code);
	ERROR("CUDA call failure: ", call, " '", err, "' (", code, ") ", "at ",
		file, ":", line);
}
