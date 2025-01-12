// This implementation is from
// https://github.com/mark-poscablo/gpu-prefix-sum

#include <cooperative_groups.h>

#include <ParticleStabilizerCUDA/PrefixScan.cuh>

__global__ void gpu_sum_scan_blelloch(uint* const d_out,
                                      const uint* const d_in,
                                      uint* const d_block_sums,
                                      const size_t numElems) {
  cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

  extern __shared__ uint s_out[];

  // Zero out shared memory
  s_out[threadIdx.x] = 0;
  s_out[threadIdx.x + blockDim.x] = 0;

  cta.sync();

  // Copy d_in to shared memory per block
  uint cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (cpy_idx < numElems) {
    s_out[threadIdx.x] = d_in[cpy_idx];
    if (cpy_idx + blockDim.x < numElems)
      s_out[threadIdx.x + blockDim.x] = d_in[cpy_idx + blockDim.x];
  }

  cta.sync();

  // Reduce/Upsweep step

  // 2^11 = 2048, the max amount of data a block can blelloch scan
  uint max_steps = 11;

  uint r_idx = 0;
  uint l_idx = 0;
  uint sum = 0;  // global sum can be passed to host if needed
  uint t_active = 0;
  for (int s = 0; s < max_steps; ++s) {
    t_active = 0;

    // calculate necessary indexes
    // right index must be (t+1) * 2^(s+1)) - 1
    r_idx = ((threadIdx.x + 1) * (1 << (s + 1))) - 1;
    if (r_idx < 2048)
      t_active = 1;

    if (t_active) {
      // left index must be r_idx - 2^s
      l_idx = r_idx - (1 << s);

      // do the actual add operation
      sum = s_out[l_idx] + s_out[r_idx];
    }

    cta.sync();

    if (t_active) {
      s_out[r_idx] = sum;
    }

    cta.sync();
  }

  if (threadIdx.x == 0) {
    d_block_sums[blockIdx.x] = s_out[r_idx];
    s_out[r_idx] = 0;
  }

  cta.sync();

  // Downsweep step

  for (int s = max_steps - 1; s >= 0; --s) {
    // calculate necessary indexes
    // right index must be (t+1) * 2^(s+1)) - 1
    r_idx = ((threadIdx.x + 1) * (1 << (s + 1))) - 1;
    if (r_idx < 2048) {
      t_active = 1;
    }

    uint r_cpy = 0;
    uint lr_sum = 0;
    if (t_active) {
      // left index must be r_idx - 2^s
      l_idx = r_idx - (1 << s);

      // do the downsweep operation
      r_cpy = s_out[r_idx];
      lr_sum = s_out[l_idx] + s_out[r_idx];
    }

    cta.sync();

    if (t_active) {
      s_out[l_idx] = r_cpy;
      s_out[r_idx] = lr_sum;
    }

    cta.sync();
  }

  // Copy the results to global memory
  if (cpy_idx < numElems) {
    d_out[cpy_idx] = s_out[threadIdx.x];
    if (cpy_idx + blockDim.x < numElems)
      d_out[cpy_idx + blockDim.x] = s_out[threadIdx.x + blockDim.x];
  }
}

__global__ void gpu_add_block_sums(uint* const d_out,
                                   const uint* const d_in,
                                   uint* const d_block_sums,
                                   const size_t numElems) {
  uint d_block_sum_val = d_block_sums[blockIdx.x];

  uint cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;

  if (cpy_idx < numElems) {
    d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
    if (cpy_idx + blockDim.x < numElems)
      d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
  }
}

__global__ void gpu_prescan(uint* const d_out,
                            const uint* const d_in,
                            uint* const d_block_sums,
                            const uint len,
                            const uint shmem_sz,
                            const uint max_elems_per_block) {
  cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

  // Allocated on invocation
  extern __shared__ uint s_out[];

  int thid = threadIdx.x;
  int ai = thid;
  int bi = thid + blockDim.x;

  // Zero out the shared memory
  s_out[thid] = 0;
  s_out[thid + blockDim.x] = 0;

  if (thid + max_elems_per_block < shmem_sz) {
    s_out[thid + max_elems_per_block] = 0;
  }

  cta.sync();

  // Copy d_in to shared memory
  uint cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
  if (cpy_idx < len) {
    s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
    if (cpy_idx + blockDim.x < len)
      s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
  }

  // Upsweep/Reduce step
  int offset = 1;
  for (int d = max_elems_per_block >> 1; d > 0; d >>= 1) {
    cta.sync();

    if (thid < d) {
      int ai = offset * ((thid << 1) + 1) - 1;
      int bi = offset * ((thid << 1) + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      s_out[bi] += s_out[ai];
    }

    offset <<= 1;
  }

  if (thid == 0) {
    d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1 + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
    s_out[max_elems_per_block - 1 + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
  }

  // Downsweep step
  for (int d = 1; d < max_elems_per_block; d <<= 1) {
    offset >>= 1;
    cta.sync();

    if (thid < d) {
      int ai = offset * ((thid << 1) + 1) - 1;
      int bi = offset * ((thid << 1) + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      uint temp = s_out[ai];
      s_out[ai] = s_out[bi];
      s_out[bi] += temp;
    }
  }

  cta.sync();

  // Copy contents of shared memory to global memory
  if (cpy_idx < len) {
    d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
    if (cpy_idx + blockDim.x < len)
      d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
  }
}

PrefixScan::PrefixScan(const size_t arrayLength)
    : _arrayLength(arrayLength),
      _block_sz(MAX_BLOCK_SZ / 2),
      _max_elems_per_block(2 * _block_sz),
      _shmem_sz(_max_elems_per_block + ((_max_elems_per_block - 1) >> LOG_NUM_BANKS)) {
  _grid_sz = _arrayLength / _max_elems_per_block;

  if (_arrayLength % _max_elems_per_block != 0) {
    _grid_sz += 1;
  }

  CUDA_CHECK_ERROR(cudaMalloc(&_d_block_sums, sizeof(uint) * _grid_sz));
  CUDA_CHECK_ERROR(cudaMalloc(&_d_dummy_blocks_sums, sizeof(uint)));
  CUDA_CHECK_ERROR(cudaMalloc(&_d_in_block_sums, sizeof(uint) * _grid_sz));
}

PrefixScan::~PrefixScan() {
  CUDA_CHECK_ERROR(cudaFree(_d_block_sums));
  CUDA_CHECK_ERROR(cudaFree(_d_dummy_blocks_sums));
  CUDA_CHECK_ERROR(cudaFree(_d_in_block_sums));
}

void PrefixScan::sum_scan_blelloch(uint* const d_out,
                                   const uint* const d_in) {
  CUDA_CHECK_ERROR(cudaMemset(d_out, 0, _arrayLength * sizeof(uint)));
  CUDA_CHECK_ERROR(cudaMemset(_d_block_sums, 0, sizeof(uint) * _grid_sz));

  gpu_prescan<<<_grid_sz, _block_sz, sizeof(uint) * _shmem_sz>>>(d_out,
                                                                 d_in,
                                                                 _d_block_sums,
                                                                 _arrayLength,
                                                                 _shmem_sz,
                                                                 _max_elems_per_block);

  if (_grid_sz <= _max_elems_per_block) {
    CUDA_CHECK_ERROR(cudaMemset(_d_dummy_blocks_sums, 0, sizeof(uint)));

    gpu_prescan<<<1, _block_sz, sizeof(uint) * _shmem_sz>>>(_d_block_sums,
                                                            _d_block_sums,
                                                            _d_dummy_blocks_sums,
                                                            _grid_sz,
                                                            _shmem_sz,
                                                            _max_elems_per_block);
  } else {
    CUDA_CHECK_ERROR(cudaMemcpy(_d_in_block_sums, _d_block_sums, sizeof(uint) * _grid_sz, cudaMemcpyDeviceToDevice));

    // Recursive
    PrefixScan pefixScan(_grid_sz);
    pefixScan.sum_scan_blelloch(_d_block_sums,
                                _d_in_block_sums);
  }

  gpu_add_block_sums<<<_grid_sz, _block_sz>>>(d_out,
                                              d_out,
                                              _d_block_sums,
                                              _arrayLength);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}
