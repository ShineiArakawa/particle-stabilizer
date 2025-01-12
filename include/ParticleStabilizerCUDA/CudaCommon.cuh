#pragma once

#include <cuda_runtime.h>

#include <Util/Logging.hpp>
#include <limits>

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCK_DIM 32
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

using uint = unsigned int;

#define CUDA_CHECK_ERROR(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    LOG_CRITICAL("GPUassert: : " + std::string(cudaGetErrorString(code)) + " " + std::string(file) + " " + std::to_string(line));
    if (abort) {
      exit(code);
    }
  }
}

inline void deviceInfo(const int deviceID = 0) {
  cudaDeviceProp dProp;
  cudaGetDeviceProperties(&dProp, deviceID);
  LOG_INFO("Device                                       : " + std::string(dProp.name));
  LOG_INFO("Max number of threads per block              : " + std::to_string(dProp.maxThreadsPerBlock));
  LOG_INFO("Max dimension size of a thread block (x,y,z) : (" + std::to_string(dProp.maxThreadsDim[0]) + ", " + std::to_string(dProp.maxThreadsDim[1]) + ", " + std::to_string(dProp.maxThreadsDim[2]) + ")");
  LOG_INFO("Max dimension size of a grid size    (x,y,z) : (" + std::to_string(dProp.maxGridSize[0]) + ", " + std::to_string(dProp.maxGridSize[1]) + ", " + std::to_string(dProp.maxGridSize[2]) + ")");
}

inline uint divRoundUp(const uint value, const uint radix) { return (value + radix - 1) / radix; };

#ifdef __cplusplus
}
#endif
