#pragma once

#include <ParticleStabilizerCUDA/ParticleModelCUDA.hpp>

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================================================================================
// Kernels
// ==============================================================================================================

/**
 * @brief Calculate the numbers of particles in each bucket
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @param bucketContext Bucket configuration
 * @param bucketCounter Device pointer to the counter array
 */
__global__ void countParticlesInEachBucketKernel(const ParticleCUDA* particles,
                                                 const int64_t nParticles,
                                                 const BucketContext bucketContext,
                                                 uint* bucketCounter);

#ifdef USE_NATIVE_PREFIX_SCAN
__global__ void naivePrefixSumKernel(const uint* bucketCounter,
                                     const int64_t nBuckets,
                                     uint* bucketCumsumCounter);
#endif

/**
 * @brief Register particles to each bucket
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @param bucketContext Bucket configuration
 * @param bucketCounter Device pointer to the counter array
 * @param bucketCumsumCounter Device pointer to the cumulative counter array
 * @param bucketCounterBuffer Device pointer to the another counter buffer array
 * @param buckets Device pointer to the bucket array
 */
__global__ void registerToBucketKernel(const ParticleCUDA* particles,
                                       const int64_t nParticles,
                                       const BucketContext bucketContext,
                                       const uint* bucketCounter,
                                       const uint* bucketCumsumCounter,
                                       uint* bucketCounterBuffer,
                                       int64_t* buckets);

/**
 * @brief Detect particle-particle collision and modifiy positions and velocities of particles.
 *
 * @param particles Device pointer to particles (read-only)
 * @param nParticles The number of particles
 * @param bucketContext Bucket configuration
 * @param bucketCounter Device pointer to the counter array
 * @param bucketCumsumCounter Device pointer to the cumulative counter array
 * @param buckets Device pointer to the bucket array
 * @param coefficientOfSpring Spring coefficient for particle-particle interaction
 * @param coefficientOfRestitution Restitution coefficient
 * @param newParticles Device pointer to particles (write-only)
 */
__global__ void resolveCollisionsKernel(const ParticleCUDA* particles,
                                        const int64_t nParticles,
                                        const BucketContext bucketContext,
                                        const uint* bucketCounter,
                                        const uint* bucketCumsumCounter,
                                        const int64_t* buckets,
                                        const double coefficientOfSpring,
                                        const double coefficientOfRestitution,
                                        ParticleCUDA* newParticles);

// ==============================================================================================================
// Dispatchers
// ==============================================================================================================

/**
 * @brief Calculate the numbers of particles in each bucket
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @param bucketContext Bucket configuration
 * @param bucketCounter Device pointer to the counter array
 */
void launchCountParticlesInEachBucketKernel(const ParticleCUDA* particles,
                                            const int64_t nParticles,
                                            const BucketContext bucketContext,
                                            uint* bucketCounter);

/**
 * @brief Calculate prefix sum of counter array
 *
 * @param bucketCounter Device pointer to the counter array
 * @param nBuckets The total number of buckets
 * @param bucketCumsumCounter Device pointer to the cumulative counter array
 */
void launchPrefixSumKernel(const uint* bucketCounter,
                           const int64_t nBuckets,
                           uint* bucketCumsumCounter);

/**
 * @brief Register particles to each bucket
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @param bucketContext Bucket configuration
 * @param bucketCounter Device pointer to the counter array
 * @param bucketCumsumCounter Device pointer to the cumulative counter array
 * @param bucketCounterBuffer Device pointer to the another counter buffer array
 * @param buckets Device pointer to the bucket array
 */
void launchRegisterToBucketKernel(const ParticleCUDA* particles,
                                  const int64_t nParticles,
                                  const BucketContext bucketContext,
                                  const uint* bucketCounter,
                                  const uint* bucketCumsumCounter,
                                  uint* bucketCounterBuffer,
                                  int64_t* buckets);

/**
 * @brief Detect particle-particle collision and modifiy positions and velocities of particles.
 *
 * @param particles Device pointer to particles (read-only)
 * @param nParticles The number of particles
 * @param bucketContext Bucket configuration
 * @param bucketCounter Device pointer to the counter array
 * @param bucketCumsumCounter Device pointer to the cumulative counter array
 * @param buckets Device pointer to the bucket array
 * @param coefficientOfSpring Spring coefficient for particle-particle interaction
 * @param coefficientOfRestitution Restitution coefficient
 * @param newParticles Device pointer to particles (write-only)
 */
void launchResolveCollisionsKernel(const ParticleCUDA* particles,
                                   const int64_t nParticles,
                                   const BucketContext bucketContext,
                                   const uint* bucketCounter,
                                   const uint* bucketCumsumCounter,
                                   const int64_t* buckets,
                                   const double coefficientOfSpring,
                                   const double coefficientOfRestitution,
                                   ParticleCUDA* newParticles);

#ifdef __cplusplus
}
#endif
