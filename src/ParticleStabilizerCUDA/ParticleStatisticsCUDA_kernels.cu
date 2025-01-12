#include <cooperative_groups.h>

#include <ParticleStabilizerCUDA/ParticleStatisticsCUDA_kernels.cuh>

// #define DEBUG_PARTICLE_ESTATISTICS_CUDA_KERNELS

template <uint blockSize>
static __device__ __forceinline__ void warpReduceSum(volatile double* sdata, uint tid) {
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduceSum(double* g_idata, double* g_odata, uint n) {
  cooperative_groups::thread_block ctx = cooperative_groups::this_thread_block();

  extern __shared__ double sdata[];

  const uint tid = threadIdx.x;
  const uint gridSize = blockSize * 2 * gridDim.x;
  uint i = blockIdx.x * (blockSize * 2) + tid;

  sdata[tid] = 0.0;
  ctx.sync();

  while (i < n) {
    sdata[tid] += g_idata[i] + g_idata[i + blockSize];
    i += gridSize;
  }
  ctx.sync();

  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    ctx.sync();
  }

  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    ctx.sync();
  }

  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    ctx.sync();
  }

  if (tid < 32) {
    warpReduceSum<blockSize>(sdata, tid);
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

template <uint blockSize>
static __device__ __forceinline__ void warpReduceMax(volatile double* sdata, uint tid) {
  if (blockSize >= 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
  if (blockSize >= 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
  if (blockSize >= 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
  if (blockSize >= 8) sdata[tid] = max(sdata[tid], sdata[tid + 4]);
  if (blockSize >= 4) sdata[tid] = max(sdata[tid], sdata[tid + 2]);
  if (blockSize >= 2) sdata[tid] = max(sdata[tid], sdata[tid + 1]);
}

template <unsigned int blockSize>
__global__ void reduceMax(double* g_idata, double* g_odata, uint n) {
  cooperative_groups::thread_block ctx = cooperative_groups::this_thread_block();

  extern __shared__ double sdata[];

  const uint tid = threadIdx.x;
  const uint gridSize = blockSize * 2 * gridDim.x;
  uint i = blockIdx.x * (blockSize * 2) + tid;

  sdata[tid] = g_idata[0];
  ctx.sync();

  while (i < n) {
    sdata[tid] = max(sdata[tid], max(g_idata[i], g_idata[i + blockSize]));
    i += gridSize;
  }
  ctx.sync();

  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] = max(sdata[tid], sdata[tid + 256]);
    }
    ctx.sync();
  }

  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] = max(sdata[tid], sdata[tid + 128]);
    }
    ctx.sync();
  }

  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] = max(sdata[tid], sdata[tid + 64]);
    }
    ctx.sync();
  }

  if (tid < 32) {
    warpReduceMax<blockSize>(sdata, tid);
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

template <uint blockSize>
static __device__ __forceinline__ void warpReduceMin(volatile double* sdata, uint tid) {
  if (blockSize >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
  if (blockSize >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
  if (blockSize >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
  if (blockSize >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
  if (blockSize >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
  if (blockSize >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

template <unsigned int blockSize>
__global__ void reduceMin(double* g_idata, double* g_odata, uint n) {
  cooperative_groups::thread_block ctx = cooperative_groups::this_thread_block();

  extern __shared__ double sdata[];

  const uint tid = threadIdx.x;
  const uint gridSize = blockSize * 2 * gridDim.x;
  uint i = blockIdx.x * (blockSize * 2) + tid;

  sdata[tid] = g_idata[0];
  ctx.sync();

  while (i < n) {
    sdata[tid] = min(sdata[tid], min(g_idata[i], g_idata[i + blockSize]));
    i += gridSize;
  }
  ctx.sync();

  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] = min(sdata[tid], sdata[tid + 256]);
    }
    ctx.sync();
  }

  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] = min(sdata[tid], sdata[tid + 128]);
    }
    ctx.sync();
  }

  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] = min(sdata[tid], sdata[tid + 64]);
    }
    ctx.sync();
  }

  if (tid < 32) {
    warpReduceMin<blockSize>(sdata, tid);
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

__global__ void calcKineticEnergyKernel(const ParticleCUDA* particles,
                                        const int64_t nParticles,
                                        double* kineticEnergy) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nParticles) {
    const ParticleCUDA particle = particles[idx];
    kineticEnergy[idx] = 0.5 * particle.mass * (particle.velocity.x * particle.velocity.x + particle.velocity.y * particle.velocity.y + particle.velocity.z * particle.velocity.z);
  }
}

__global__ void calcMaxPosKernel(const ParticleCUDA* particles,
                                 const int64_t nParticles,
                                 double* maxPosX,
                                 double* maxPosY,
                                 double* maxPosZ) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nParticles) {
    maxPosX[idx] = particles[idx].position.x + particles[idx].radius;
    maxPosY[idx] = particles[idx].position.y + particles[idx].radius;
    maxPosZ[idx] = particles[idx].position.z + particles[idx].radius;
  }
}

__global__ void calcMinPosKernel(const ParticleCUDA* particles,
                                 const int64_t nParticles,
                                 double* minPosX,
                                 double* minPosY,
                                 double* minPosZ) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nParticles) {
    const ParticleCUDA particle = particles[idx];

    minPosX[idx] = particle.position.x - particle.radius;
    minPosY[idx] = particle.position.y - particle.radius;
    minPosZ[idx] = particle.position.z - particle.radius;
  }
}

__global__ void calcCollisionDistanceKernel(const ParticleCUDA* particles,
                                            const int64_t nParticles,
                                            double* maxCollisionDistance) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nParticles) {
    double tmp_maxCollisionDistance = 0.0;

    const ParticleCUDA particle = particles[idx];

    for (int i = idx + 1; i < nParticles; ++i) {
      const ParticleCUDA otherParticle = particles[i];

      const double reletiveX = particle.position.x - otherParticle.position.x;
      const double reletiveY = particle.position.y - otherParticle.position.y;
      const double reletiveZ = particle.position.z - otherParticle.position.z;

      const double distance = sqrt(reletiveX * reletiveX +
                                   reletiveY * reletiveY +
                                   reletiveZ * reletiveZ);

      const double collisionDistance = particle.radius + otherParticle.radius - distance;

      if (collisionDistance > 0.0) {
        tmp_maxCollisionDistance = max(tmp_maxCollisionDistance, collisionDistance);
      }
    }

    maxCollisionDistance[idx] = tmp_maxCollisionDistance;
  }
}

double cpuReduceSum(const double* data, const int64_t n) {
  double sum = 0.0;

  for (int i = 0; i < n; ++i) {
    sum += data[i];
  }

  return sum;
}

double cpuReduceMax(const double* data, const int64_t n) {
  double max = data[0];

  for (int i = 0; i < n; ++i) {
    max = std::max(max, data[i]);
  }

  return max;
}

double cpuReduceMin(const double* data, const int64_t n) {
  double min = data[0];

  for (int i = 0; i < n; ++i) {
    min = std::min(min, data[i]);
  }

  return min;
}

double launchCalcKineticEnergyKernel(const ParticleCUDA* particles,
                                     const int64_t nParticles) {
  // Calc kinetic energy
  double* deviceKineticEnergy;
  CUDA_CHECK_ERROR(cudaMalloc(&deviceKineticEnergy, nParticles * sizeof(double)));

  const uint nBlocksPerGrid = divRoundUp(nParticles, BLOCK_DIM);

  calcKineticEnergyKernel<<<nBlocksPerGrid, BLOCK_DIM>>>(particles, nParticles, deviceKineticEnergy);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  CUDA_CHECK_ERROR(cudaGetLastError());

  // Reduce kinetic energy on device
  double* deviceReducedKineticEnergy;
  CUDA_CHECK_ERROR(cudaMalloc(&deviceReducedKineticEnergy, nBlocksPerGrid * sizeof(double)));

  reduceSum<BLOCK_DIM><<<nBlocksPerGrid, BLOCK_DIM, BLOCK_DIM * sizeof(double)>>>(deviceKineticEnergy, deviceReducedKineticEnergy, nParticles);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  CUDA_CHECK_ERROR(cudaGetLastError());

  // Copy reduced kinetic energy to host
  double* hostReducedKineticEnergy = new double[nBlocksPerGrid];
  CUDA_CHECK_ERROR(cudaMemcpy(hostReducedKineticEnergy, deviceReducedKineticEnergy, nBlocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));

  // Reduce kinetic energy on host
  double kineticEnergy = cpuReduceSum(hostReducedKineticEnergy, nBlocksPerGrid);

  // Free memory
  CUDA_CHECK_ERROR(cudaFree(deviceKineticEnergy));
  CUDA_CHECK_ERROR(cudaFree(deviceReducedKineticEnergy));
  free(hostReducedKineticEnergy);

  return kineticEnergy;
}

double3 launchCalcMaxPosKernel(const ParticleCUDA* particles,
                               const int64_t nParticles) {
  // Calc max pos
  double* deviceMaxPosX;
  double* deviceMaxPosY;
  double* deviceMaxPosZ;

  CUDA_CHECK_ERROR(cudaMalloc(&deviceMaxPosX, nParticles * sizeof(double)));
  CUDA_CHECK_ERROR(cudaMalloc(&deviceMaxPosY, nParticles * sizeof(double)));
  CUDA_CHECK_ERROR(cudaMalloc(&deviceMaxPosZ, nParticles * sizeof(double)));

  const uint nBlocksPerGrid = divRoundUp(nParticles, BLOCK_DIM);

  calcMaxPosKernel<<<nBlocksPerGrid, BLOCK_DIM>>>(particles, nParticles, deviceMaxPosX, deviceMaxPosY, deviceMaxPosZ);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  CUDA_CHECK_ERROR(cudaGetLastError());

#ifdef DEBUG_PARTICLE_ESTATISTICS_CUDA_KERNELS
  {
    double* hostMaxPosX = new double[nParticles];
    double* hostMaxPosY = new double[nParticles];
    double* hostMaxPosZ = new double[nParticles];

    CUDA_CHECK_ERROR(cudaMemcpy(hostMaxPosX, deviceMaxPosX, nParticles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(hostMaxPosY, deviceMaxPosY, nParticles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(hostMaxPosZ, deviceMaxPosZ, nParticles * sizeof(double), cudaMemcpyDeviceToHost));

    double maxPosX = hostMaxPosX[0];
    double maxPosY = hostMaxPosY[0];
    double maxPosZ = hostMaxPosZ[0];

    for (int i = 0; i < nParticles; ++i) {
      maxPosX = std::max(maxPosX, hostMaxPosX[i]);
      maxPosY = std::max(maxPosY, hostMaxPosX[i]);
      maxPosZ = std::max(maxPosZ, hostMaxPosX[i]);
    }

    LOG_INFO("maxPosX: " + std::to_string(maxPosX));
    LOG_INFO("maxPosY: " + std::to_string(maxPosY));
    LOG_INFO("maxPosZ: " + std::to_string(maxPosZ));

    free(hostMaxPosX);
    free(hostMaxPosY);
    free(hostMaxPosZ);
  }
#endif

  // Reduce max pos on device
  double* deviceReducedMaxPosX;
  double* deviceReducedMaxPosY;
  double* deviceReducedMaxPosZ;

  CUDA_CHECK_ERROR(cudaMalloc(&deviceReducedMaxPosX, nBlocksPerGrid * sizeof(double)));
  CUDA_CHECK_ERROR(cudaMalloc(&deviceReducedMaxPosY, nBlocksPerGrid * sizeof(double)));
  CUDA_CHECK_ERROR(cudaMalloc(&deviceReducedMaxPosZ, nBlocksPerGrid * sizeof(double)));

  reduceMax<BLOCK_DIM><<<nBlocksPerGrid, BLOCK_DIM, BLOCK_DIM * sizeof(double)>>>(deviceMaxPosX, deviceReducedMaxPosX, nParticles);
  reduceMax<BLOCK_DIM><<<nBlocksPerGrid, BLOCK_DIM, BLOCK_DIM * sizeof(double)>>>(deviceMaxPosY, deviceReducedMaxPosY, nParticles);
  reduceMax<BLOCK_DIM><<<nBlocksPerGrid, BLOCK_DIM, BLOCK_DIM * sizeof(double)>>>(deviceMaxPosZ, deviceReducedMaxPosZ, nParticles);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  CUDA_CHECK_ERROR(cudaGetLastError());

#ifdef DEBUG_PARTICLE_ESTATISTICS_CUDA_KERNELS
  {
    double* hostMaxPosX = new double[nBlocksPerGrid];
    double* hostMaxPosY = new double[nBlocksPerGrid];
    double* hostMaxPosZ = new double[nBlocksPerGrid];

    CUDA_CHECK_ERROR(cudaMemcpy(hostMaxPosX, deviceReducedMaxPosX, nBlocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(hostMaxPosY, deviceReducedMaxPosY, nBlocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(hostMaxPosZ, deviceReducedMaxPosZ, nBlocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));

    double maxPosX = hostMaxPosX[0];
    double maxPosY = hostMaxPosY[0];
    double maxPosZ = hostMaxPosZ[0];

    for (int i = 0; i < nBlocksPerGrid; ++i) {
      maxPosX = std::max(maxPosX, hostMaxPosX[i]);
      maxPosY = std::max(maxPosY, hostMaxPosX[i]);
      maxPosZ = std::max(maxPosZ, hostMaxPosX[i]);
    }

    LOG_INFO("maxPosX: " + std::to_string(maxPosX));
    LOG_INFO("maxPosY: " + std::to_string(maxPosY));
    LOG_INFO("maxPosZ: " + std::to_string(maxPosZ));

    free(hostMaxPosX);
    free(hostMaxPosY);
    free(hostMaxPosZ);
  }
#endif

  // Copy reduced max pos to host
  double* hostReducedMaxPosX = new double[nBlocksPerGrid];
  double* hostReducedMaxPosY = new double[nBlocksPerGrid];
  double* hostReducedMaxPosZ = new double[nBlocksPerGrid];

  CUDA_CHECK_ERROR(cudaMemcpy(hostReducedMaxPosX, deviceReducedMaxPosX, nBlocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK_ERROR(cudaMemcpy(hostReducedMaxPosY, deviceReducedMaxPosY, nBlocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK_ERROR(cudaMemcpy(hostReducedMaxPosZ, deviceReducedMaxPosZ, nBlocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));

  // Reduce max pos on host
  const double maxPosX = cpuReduceMax(hostReducedMaxPosX, nBlocksPerGrid);
  const double maxPosY = cpuReduceMax(hostReducedMaxPosY, nBlocksPerGrid);
  const double maxPosZ = cpuReduceMax(hostReducedMaxPosZ, nBlocksPerGrid);

#ifdef DEBUG_PARTICLE_ESTATISTICS_CUDA_KERNELS
  LOG_INFO("maxPosX: " + std::to_string(maxPosX));
  LOG_INFO("maxPosY: " + std::to_string(maxPosY));
  LOG_INFO("maxPosZ: " + std::to_string(maxPosZ));
#endif

  // Free memory
  CUDA_CHECK_ERROR(cudaFree(deviceMaxPosX));
  CUDA_CHECK_ERROR(cudaFree(deviceMaxPosY));
  CUDA_CHECK_ERROR(cudaFree(deviceMaxPosZ));
  CUDA_CHECK_ERROR(cudaFree(deviceReducedMaxPosX));
  CUDA_CHECK_ERROR(cudaFree(deviceReducedMaxPosY));
  CUDA_CHECK_ERROR(cudaFree(deviceReducedMaxPosZ));

  free(hostReducedMaxPosX);
  free(hostReducedMaxPosY);
  free(hostReducedMaxPosZ);

  return make_double3(maxPosX, maxPosY, maxPosZ);
}

double3 launchCalcMinPosKernel(const ParticleCUDA* particles,
                               const int64_t nParticles) {
  // Calc min pos
  double* deviceMinPosX;
  double* deviceMinPosY;
  double* deviceMinPosZ;

  CUDA_CHECK_ERROR(cudaMalloc(&deviceMinPosX, nParticles * sizeof(double)));
  CUDA_CHECK_ERROR(cudaMalloc(&deviceMinPosY, nParticles * sizeof(double)));
  CUDA_CHECK_ERROR(cudaMalloc(&deviceMinPosZ, nParticles * sizeof(double)));

  const uint nBlocksPerGrid = divRoundUp(nParticles, BLOCK_DIM);

  calcMinPosKernel<<<nBlocksPerGrid, BLOCK_DIM>>>(particles, nParticles, deviceMinPosX, deviceMinPosY, deviceMinPosZ);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  CUDA_CHECK_ERROR(cudaGetLastError());

  // Reduce min pos on device
  double* deviceReducedMinPosX;
  double* deviceReducedMinPosY;
  double* deviceReducedMinPosZ;

  CUDA_CHECK_ERROR(cudaMalloc(&deviceReducedMinPosX, nBlocksPerGrid * sizeof(double)));
  CUDA_CHECK_ERROR(cudaMalloc(&deviceReducedMinPosY, nBlocksPerGrid * sizeof(double)));
  CUDA_CHECK_ERROR(cudaMalloc(&deviceReducedMinPosZ, nBlocksPerGrid * sizeof(double)));

  reduceMin<BLOCK_DIM><<<nBlocksPerGrid, BLOCK_DIM, BLOCK_DIM * sizeof(double)>>>(deviceMinPosX, deviceReducedMinPosX, nParticles);
  reduceMin<BLOCK_DIM><<<nBlocksPerGrid, BLOCK_DIM, BLOCK_DIM * sizeof(double)>>>(deviceMinPosY, deviceReducedMinPosY, nParticles);
  reduceMin<BLOCK_DIM><<<nBlocksPerGrid, BLOCK_DIM, BLOCK_DIM * sizeof(double)>>>(deviceMinPosZ, deviceReducedMinPosZ, nParticles);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  CUDA_CHECK_ERROR(cudaGetLastError());

  // Copy reduced min pos to host
  double* hostReducedMinPosX = new double[nBlocksPerGrid];
  double* hostReducedMinPosY = new double[nBlocksPerGrid];
  double* hostReducedMinPosZ = new double[nBlocksPerGrid];

  CUDA_CHECK_ERROR(cudaMemcpy(hostReducedMinPosX, deviceReducedMinPosX, nBlocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK_ERROR(cudaMemcpy(hostReducedMinPosY, deviceReducedMinPosY, nBlocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK_ERROR(cudaMemcpy(hostReducedMinPosZ, deviceReducedMinPosZ, nBlocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));

  // Reduce min pos on host
  double minPosX = cpuReduceMin(hostReducedMinPosX, nBlocksPerGrid);
  double minPosY = cpuReduceMin(hostReducedMinPosY, nBlocksPerGrid);
  double minPosZ = cpuReduceMin(hostReducedMinPosZ, nBlocksPerGrid);

  // Free memory
  CUDA_CHECK_ERROR(cudaFree(deviceMinPosX));
  CUDA_CHECK_ERROR(cudaFree(deviceMinPosY));
  CUDA_CHECK_ERROR(cudaFree(deviceMinPosZ));
  CUDA_CHECK_ERROR(cudaFree(deviceReducedMinPosX));
  CUDA_CHECK_ERROR(cudaFree(deviceReducedMinPosY));
  CUDA_CHECK_ERROR(cudaFree(deviceReducedMinPosZ));

  free(hostReducedMinPosX);
  free(hostReducedMinPosY);
  free(hostReducedMinPosZ);

  return make_double3(minPosX, minPosY, minPosZ);
}

void launchCalcCollisionsKernel(const ParticleCUDA* particles,
                                const int64_t nParticles,
                                double& maxCollisionDistance,
                                int64_t& nCollisions) {
  // Calc collision distance
  double* deviceMaxCollisionDistance;
  CUDA_CHECK_ERROR(cudaMalloc(&deviceMaxCollisionDistance, nParticles * sizeof(double)));

  const uint nBlocksPerGrid = divRoundUp(nParticles, BLOCK_DIM);

  calcCollisionDistanceKernel<<<nBlocksPerGrid, BLOCK_DIM>>>(particles, nParticles, deviceMaxCollisionDistance);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  CUDA_CHECK_ERROR(cudaGetLastError());

  // Reduce collision distance on device
  double* deviceReducedMaxCollisionDistance;
  CUDA_CHECK_ERROR(cudaMalloc(&deviceReducedMaxCollisionDistance, nBlocksPerGrid * sizeof(double)));

  reduceMax<BLOCK_DIM><<<nBlocksPerGrid, BLOCK_DIM, BLOCK_DIM * sizeof(double)>>>(deviceMaxCollisionDistance, deviceReducedMaxCollisionDistance, nParticles);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  CUDA_CHECK_ERROR(cudaGetLastError());

  // Copy reduced collision distance to host
  double* hostReducedMaxCollisionDistance = new double[nBlocksPerGrid];
  CUDA_CHECK_ERROR(cudaMemcpy(hostReducedMaxCollisionDistance, deviceReducedMaxCollisionDistance, nBlocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));

  // Reduce collision distance on host
  maxCollisionDistance = std::max(0.0, cpuReduceMax(hostReducedMaxCollisionDistance, nBlocksPerGrid));

  // Count collisions
  double* hostMaxCollisionDistance = new double[nParticles];
  CUDA_CHECK_ERROR(cudaMemcpy(hostMaxCollisionDistance, deviceMaxCollisionDistance, nParticles * sizeof(double), cudaMemcpyDeviceToHost));

  nCollisions = 0LL;
  for (int i = 0; i < nParticles; ++i) {
    if (hostMaxCollisionDistance[i] > 0.0) {
      ++nCollisions;
    }
  }

  // Free memory
  CUDA_CHECK_ERROR(cudaFree(deviceMaxCollisionDistance));
  CUDA_CHECK_ERROR(cudaFree(deviceReducedMaxCollisionDistance));

  free(hostReducedMaxCollisionDistance);
  free(hostMaxCollisionDistance);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}
