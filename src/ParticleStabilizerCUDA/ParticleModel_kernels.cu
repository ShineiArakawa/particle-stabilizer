#include <cooperative_groups.h>

#include <ParticleStabilizerCUDA/ParticleModel_kernels.cuh>

#define VECTORIZE_OPTIMIZATION

#if defined(VECTORIZE_OPTIMIZATION)
constexpr int NUM_LOOK_UP_BUCKETS = 27;
__constant__ int LOOK_UP_BUCKET_OFFSETS[NUM_LOOK_UP_BUCKETS][3] = {
    // ===================================
    {0, 0, 0},
    {0, 0, 1},
    {0, 0, 2},
    {0, 1, 0},
    {0, 1, 1},
    {0, 1, 2},
    {0, 2, 0},
    {0, 2, 1},
    {0, 2, 2},
    // ===================================
    {1, 0, 0},
    {1, 0, 1},
    {1, 0, 2},
    {1, 1, 0},
    {1, 1, 1},
    {1, 1, 2},
    {1, 2, 0},
    {1, 2, 1},
    {1, 2, 2},
    // ===================================
    {2, 0, 0},
    {2, 0, 1},
    {2, 0, 2},
    {2, 1, 0},
    {2, 1, 1},
    {2, 1, 2},
    {2, 2, 0},
    {2, 2, 1},
    {2, 2, 2}};
#endif

static __device__ __forceinline__ int64_t getGlobalBucketIndexKernelFunc(const int i,
                                                                         const int j,
                                                                         const int k,
                                                                         const BucketContext* bucketContext) {
  return (static_cast<int64_t>(i) * static_cast<int64_t>(bucketContext->nBuckets.y) + static_cast<int64_t>(j)) * static_cast<int64_t>(bucketContext->nBuckets.z) + static_cast<int64_t>(k);
}

static __device__ __forceinline__ int64_t getBucketIndexKernelFunc(const double3 position,
                                                                   const BucketContext* bucketContext) {
  const int i = static_cast<int>((position.x - bucketContext->minCoords.x) / bucketContext->interval);
  const int j = static_cast<int>((position.y - bucketContext->minCoords.y) / bucketContext->interval);
  const int k = static_cast<int>((position.z - bucketContext->minCoords.z) / bucketContext->interval);

  return getGlobalBucketIndexKernelFunc(i, j, k, bucketContext);
}

static __device__ __forceinline__ int toBucketIndexEachDimKernelFunc(const double coord,
                                                                     const double minCoord,
                                                                     const double interval,
                                                                     const int64_t nBuckets) {
  int index = static_cast<int>((coord - minCoord) / interval);

  if (index < 0) {
    index = 0;
  }

  if (index >= static_cast<int>(nBuckets)) {
    index = nBuckets - 1;
  }

  return index;
}

__global__ void countParticlesInEachBucketKernel(const ParticleCUDA* particles,
                                                 const int64_t nParticles,
                                                 const BucketContext bucketContext,
                                                 uint* bucketCounter) {
  const uint iParticle = blockIdx.x * blockDim.x + threadIdx.x;

  if (iParticle < nParticles) {
    const int64_t iBucket = getBucketIndexKernelFunc(particles[iParticle].position, &bucketContext);
    atomicAdd(&bucketCounter[iBucket], 1);
  }
}

#ifdef USE_NATIVE_PREFIX_SCAN
__global__ void naivePrefixSumKernel(const uint* bucketCounter,
                                     const int64_t nBuckets,
                                     uint* bucketCumsumCounter) {
  const uint iBucket = blockIdx.x * blockDim.x + threadIdx.x;

  if (iBucket < nBuckets) {
    uint sum = 0;

    for (int64_t i = 0; i < iBucket; ++i) {
      sum += bucketCounter[i];
    }

    bucketCumsumCounter[iBucket] = sum;
  }
}
#endif

__global__ void registerToBucketKernel(const ParticleCUDA* particles,
                                       const int64_t nParticles,
                                       const BucketContext bucketContext,
                                       const uint* bucketCounter,
                                       const uint* bucketCumsumCounter,
                                       uint* bucketCounterBuffer,
                                       int64_t* buckets) {
  const uint iParticle = blockIdx.x * blockDim.x + threadIdx.x;

  if (iParticle < nParticles) {
    const int64_t iBucket = getBucketIndexKernelFunc(particles[iParticle].position, &bucketContext);

    // Offset to the bucket
    const uint offset = bucketCumsumCounter[iBucket];

    // Index in the bucket
    const uint i = atomicAdd(&bucketCounterBuffer[iBucket], 1);

    buckets[offset + i] = iParticle;
  }
}

__global__ void resolveCollisionsKernel(const ParticleCUDA* particles,
                                        const int64_t nParticles,
                                        const BucketContext bucketContext,
                                        const uint* bucketCounter,
                                        const uint* bucketCumsumCounter,
                                        const int64_t* buckets,
                                        const double coefficientOfSpring,
                                        const double coefficientOfRestitution,
                                        ParticleCUDA* newParticles) {
#if defined(VECTORIZE_OPTIMIZATION)
  const uint iParticle = blockIdx.x * blockDim.x + threadIdx.x;
  const uint iBucketLocal = blockIdx.y * blockDim.y + threadIdx.y;

  if (iParticle >= nParticles || iBucketLocal >= NUM_LOOK_UP_BUCKETS) {
    return;
  }

  // Find the bucket index
  const int minBucketIndexX = toBucketIndexEachDimKernelFunc(particles[iParticle].position.x, bucketContext.minCoords.x, bucketContext.interval, bucketContext.nBuckets.x) - 1;
  const int minBucketIndexY = toBucketIndexEachDimKernelFunc(particles[iParticle].position.y, bucketContext.minCoords.y, bucketContext.interval, bucketContext.nBuckets.y) - 1;
  const int minBucketIndexZ = toBucketIndexEachDimKernelFunc(particles[iParticle].position.z, bucketContext.minCoords.z, bucketContext.interval, bucketContext.nBuckets.z) - 1;

  const int bucketIndexX = minBucketIndexX + LOOK_UP_BUCKET_OFFSETS[iBucketLocal][0];
  const int bucketIndexY = minBucketIndexY + LOOK_UP_BUCKET_OFFSETS[iBucketLocal][1];
  const int bucketIndexZ = minBucketIndexZ + LOOK_UP_BUCKET_OFFSETS[iBucketLocal][2];

  if (bucketIndexX < 0 || bucketIndexX >= bucketContext.nBuckets.x ||
      bucketIndexY < 0 || bucketIndexY >= bucketContext.nBuckets.y ||
      bucketIndexZ < 0 || bucketIndexZ >= bucketContext.nBuckets.z) {
    return;
  }

  const int64_t iBucket = getGlobalBucketIndexKernelFunc(bucketIndexX,
                                                         bucketIndexY,
                                                         bucketIndexZ,
                                                         &bucketContext);

  const uint offset = bucketCumsumCounter[iBucket];

  for (uint i = 0; i < bucketCounter[iBucket]; ++i) {
    const int64_t jParticle = buckets[offset + i];

    if (iParticle == jParticle) {
      // Skip self
      continue;
    }

    const double deltaCoordX = particles[iParticle].position.x - particles[jParticle].position.x;
    const double deltaCoordY = particles[iParticle].position.y - particles[jParticle].position.y;
    const double deltaCoordZ = particles[iParticle].position.z - particles[jParticle].position.z;

    const double distanceSquared = deltaCoordX * deltaCoordX + deltaCoordY * deltaCoordY + deltaCoordZ * deltaCoordZ;
    const double distance = sqrt(distanceSquared);

    const double iRadius = particles[iParticle].radius;
    const double jRadius = particles[jParticle].radius;

    const double overlap = iRadius + jRadius - distance;

    if (overlap > 0.0) {
      // Resolve collision
      const double relVelocityX = particles[iParticle].velocity.x - particles[jParticle].velocity.x;
      const double relVelocityY = particles[iParticle].velocity.y - particles[jParticle].velocity.y;
      const double relVelocityZ = particles[iParticle].velocity.z - particles[jParticle].velocity.z;

      const double relVecDotDeltaCoord = relVelocityX * deltaCoordX + relVelocityY * deltaCoordY + relVelocityZ * deltaCoordZ;

      const double coeffRelative = 2.0 * particles[jParticle].mass * relVecDotDeltaCoord * coefficientOfRestitution / ((particles[iParticle].mass + particles[jParticle].mass) * distanceSquared);
      const double coeffSpringForce = coefficientOfSpring * overlap / (distance * particles[iParticle].mass);

      atomicAdd(&(newParticles[iParticle].velocity.x), (coeffSpringForce - coeffRelative) * deltaCoordX);
      atomicAdd(&(newParticles[iParticle].velocity.y), (coeffSpringForce - coeffRelative) * deltaCoordY);
      atomicAdd(&(newParticles[iParticle].velocity.z), (coeffSpringForce - coeffRelative) * deltaCoordZ);
    }
  }
#else
  cooperative_groups::thread_block ctx = cooperative_groups::this_thread_block();

  __shared__ ParticleCUDA sharedParticles[BLOCK_DIM];
  const uint tid = threadIdx.x;

  const uint iParticle = blockIdx.x * blockDim.x + threadIdx.x;

  if (iParticle < nParticles) {
    // Copy particle to shared memory
    sharedParticles[tid] = particles[iParticle];
    ctx.sync();

    // Find the bucket index
    const int minBucketIndexX = toBucketIndexEachDimKernelFunc(sharedParticles[tid].position.x, bucketContext.minCoords.x, bucketContext.interval, bucketContext.nBuckets.x) - 1;
    const int minBucketIndexY = toBucketIndexEachDimKernelFunc(sharedParticles[tid].position.y, bucketContext.minCoords.y, bucketContext.interval, bucketContext.nBuckets.y) - 1;
    const int minBucketIndexZ = toBucketIndexEachDimKernelFunc(sharedParticles[tid].position.z, bucketContext.minCoords.z, bucketContext.interval, bucketContext.nBuckets.z) - 1;

    // Iterate over the 3x3x3 neighborhood
    for (int bucketIndexOffsetX = 0; bucketIndexOffsetX < 3; ++bucketIndexOffsetX) {
      const int bucketIndexX = minBucketIndexX + bucketIndexOffsetX;

      if (bucketIndexX < 0 || bucketIndexX >= bucketContext.nBuckets.x) {
        // Out of bound
        continue;
      }

      for (int bucketIndexOffsetY = 0; bucketIndexOffsetY < 3; ++bucketIndexOffsetY) {
        const int bucketIndexY = minBucketIndexY + bucketIndexOffsetY;

        if (bucketIndexY < 0 || bucketIndexY >= bucketContext.nBuckets.y) {
          // Out of bound
          continue;
        }

        for (int bucketIndexOffsetZ = 0; bucketIndexOffsetZ < 3; ++bucketIndexOffsetZ) {
          const int bucketIndexZ = minBucketIndexZ + bucketIndexOffsetZ;

          if (bucketIndexZ < 0 || bucketIndexZ >= bucketContext.nBuckets.z) {
            // Out of bound
            continue;
          }

          const int64_t iBucket = getGlobalBucketIndexKernelFunc(bucketIndexX, bucketIndexY, bucketIndexZ, &bucketContext);
          const uint offset = bucketCumsumCounter[iBucket];

          for (uint i = 0; i < bucketCounter[iBucket]; ++i) {
            const int64_t jParticle = buckets[offset + i];

            if (iParticle == jParticle) {
              // Skip self
              continue;
            }

            const double deltaCoordX = sharedParticles[tid].position.x - particles[jParticle].position.x;
            const double deltaCoordY = sharedParticles[tid].position.y - particles[jParticle].position.y;
            const double deltaCoordZ = sharedParticles[tid].position.z - particles[jParticle].position.z;

            const double distanceSquared = deltaCoordX * deltaCoordX + deltaCoordY * deltaCoordY + deltaCoordZ * deltaCoordZ;
            const double distance = sqrt(distanceSquared);

            const double iRadius = sharedParticles[tid].radius;
            const double jRadius = particles[jParticle].radius;

            const double overlap = iRadius + jRadius - distance;

            if (overlap > 0.0) {
              // Resolve collision
              const double relVelocityX = sharedParticles[tid].velocity.x - particles[jParticle].velocity.x;
              const double relVelocityY = sharedParticles[tid].velocity.y - particles[jParticle].velocity.y;
              const double relVelocityZ = sharedParticles[tid].velocity.z - particles[jParticle].velocity.z;

              const double relVecDotDeltaCoord = relVelocityX * deltaCoordX + relVelocityY * deltaCoordY + relVelocityZ * deltaCoordZ;

              const double coeffRelative = 2.0 * particles[jParticle].mass * relVecDotDeltaCoord * coefficientOfRestitution / ((sharedParticles[tid].mass + particles[jParticle].mass) * distanceSquared);
              const double coeffSpringForce = coefficientOfSpring * overlap / (distance * sharedParticles[tid].mass);

              newParticles[iParticle].velocity.x += (coeffSpringForce - coeffRelative) * deltaCoordX;
              newParticles[iParticle].velocity.y += (coeffSpringForce - coeffRelative) * deltaCoordY;
              newParticles[iParticle].velocity.z += (coeffSpringForce - coeffRelative) * deltaCoordZ;
            }
          }
        }
      }
    }
  }
#endif
}

void launchCountParticlesInEachBucketKernel(const ParticleCUDA* particles,
                                            const int64_t nParticles,
                                            const BucketContext bucketContext,
                                            uint* bucketCounter) {
  const uint nThreadsPerBlock = BLOCK_DIM;
  const uint nBlocksPerGrid = divRoundUp(nParticles, nThreadsPerBlock);

  countParticlesInEachBucketKernel<<<nBlocksPerGrid, nThreadsPerBlock>>>(particles,
                                                                         nParticles,
                                                                         bucketContext,
                                                                         bucketCounter);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

#ifdef USE_NATIVE_PREFIX_SCAN
void launchPrefixSumKernel(const uint* bucketCounter,
                           const int64_t nBuckets,
                           uint* bucketCumsumCounter) {
  const uint nThreadsPerBlock = BLOCK_DIM;
  const uint nBlocksPerGrid = divRoundUp(nBuckets, nThreadsPerBlock);

  naivePrefixSumKernel<<<nBlocksPerGrid, nThreadsPerBlock>>>(bucketCounter,
                                                             nBuckets,
                                                             bucketCumsumCounter);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}
#endif

void launchRegisterToBucketKernel(const ParticleCUDA* particles,
                                  const int64_t nParticles,
                                  const BucketContext bucketContext,
                                  const uint* bucketCounter,
                                  const uint* bucketCumsumCounter,
                                  uint* bucketCounterBuffer,
                                  int64_t* buckets) {
  const uint nThreadsPerBlock = BLOCK_DIM;
  const uint nBlocksPerGrid = divRoundUp(nParticles, nThreadsPerBlock);

  registerToBucketKernel<<<nBlocksPerGrid, nThreadsPerBlock>>>(particles,
                                                               nParticles,
                                                               bucketContext,
                                                               bucketCounter,
                                                               bucketCumsumCounter,
                                                               bucketCounterBuffer,
                                                               buckets);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

void launchResolveCollisionsKernel(const ParticleCUDA* particles,
                                   const int64_t nParticles,
                                   const BucketContext bucketContext,
                                   const uint* bucketCounter,
                                   const uint* bucketCumsumCounter,
                                   const int64_t* buckets,
                                   const double coefficientOfSpring,
                                   const double coefficientOfRestitution,
                                   ParticleCUDA* newParticles) {
#if defined(VECTORIZE_OPTIMIZATION)
  const dim3 nThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  // Look up 3x3x3 buckets
  const dim3 nBlocksPerGrid(divRoundUp(nParticles, nThreadsPerBlock.x),
                            divRoundUp(NUM_LOOK_UP_BUCKETS, nThreadsPerBlock.y));
#else
  const uint nThreadsPerBlock = BLOCK_DIM;
  const uint nBlocksPerGrid = divRoundUp(nParticles, nThreadsPerBlock);
#endif

  resolveCollisionsKernel<<<nBlocksPerGrid, nThreadsPerBlock>>>(particles,
                                                                nParticles,
                                                                bucketContext,
                                                                bucketCounter,
                                                                bucketCumsumCounter,
                                                                buckets,
                                                                coefficientOfSpring,
                                                                coefficientOfRestitution,
                                                                newParticles);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}