#pragma once

// This implementation is from
// https://github.com/mark-poscablo/gpu-prefix-sum

#include <ParticleStabilizerCUDA/CudaCommon.cuh>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_BLOCK_SZ 1024
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
  ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

// ==============================================================================================================
// Kernels
// ==============================================================================================================

__global__ void gpu_sum_scan_blelloch(uint* const d_out,
                                      const uint* const d_in,
                                      uint* const d_block_sums,
                                      const size_t numElems);

__global__ void gpu_add_block_sums(uint* const d_out,
                                   const uint* const d_in,
                                   uint* const d_block_sums,
                                   const size_t numElems);

__global__ void gpu_prescan(uint* const d_out,
                            const uint* const d_in,
                            uint* const d_block_sums,
                            const uint len,
                            const uint shmem_sz,
                            const uint max_elems_per_block);

// ==============================================================================================================
// Dispatchers
// ==============================================================================================================

class PrefixScan {
 public:
  PrefixScan(const size_t arrayLength);
  ~PrefixScan();

  void sum_scan_blelloch(uint* const d_out,
                         const uint* const d_in);

 private:
  size_t _arrayLength;
  uint _block_sz;
  uint _max_elems_per_block;
  uint _grid_sz;
  uint _shmem_sz;

  uint* _d_block_sums;
  uint* _d_dummy_blocks_sums;
  uint* _d_in_block_sums;
};

#ifdef __cplusplus
}
#endif
