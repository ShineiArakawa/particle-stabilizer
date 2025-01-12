#pragma once

#include <ParticleStabilizerCUDA/ParticleModelCUDA.hpp>

// ==============================================================================================================
// Kernels
// ==============================================================================================================

template <unsigned int blockSize>
/**
 * @brief Sum reduction in the warp unit
 *
 * @param sdata Device point to the source array
 * @param tid Thread index
 */
static __device__ __forceinline__ void warpReduceSum(volatile double* sdata, uint tid);

template <unsigned int blockSize>
/**
 * @brief Sum reduction
 *
 * @param g_idata Device point to the input array
 * @param g_odata Device point to the output array
 * @param n The number of elements
 */
__global__ void reduceSum(double* g_idata, double* g_odata, uint n);

template <unsigned int blockSize>
/**
 * @brief Max reduction in the warp unit
 *
 * @param sdata Device point to the source array
 * @param tid Thread index
 */
static __device__ __forceinline__ void warpReduceMax(volatile double* sdata, uint tid);

template <unsigned int blockSize>
/**
 * @brief Max reduction
 *
 * @param g_idata Device point to the input array
 * @param g_odata Device point to the output array
 * @param n The number of elements
 */
__global__ void reduceMax(double* g_idata, double* g_odata, uint n);

template <unsigned int blockSize>
/**
 * @brief Min reduction in the warp unit
 *
 * @param sdata Device point to the source array
 * @param tid Thread index
 */
static __device__ __forceinline__ void warpReduceMin(volatile double* sdata, uint tid);

template <unsigned int blockSize>
/**
 * @brief Min reduction
 *
 * @param g_idata Device point to the input array
 * @param g_odata Device point to the output array
 * @param n The number of elements
 */
__global__ void reduceMin(double* g_idata, double* g_odata, uint n);

/**
 * @brief Calculate kinetic energy of each particle
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @param kineticEnergy Device pointer to the resulting kinetic energy array
 */
__global__ void calcKineticEnergyKernel(const ParticleCUDA* particles,
                                        const int64_t nParticles,
                                        double* kineticEnergy);

/**
 * @brief Calculate max positions of each particle
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @param maxPosX Device pointer to the max X coords array
 * @param maxPosY Device pointer to the max Y coords array
 * @param maxPosZ Device pointer to the max Z coords array
 */
__global__ void calcMaxPosKernel(const ParticleCUDA* particles,
                                 const int64_t nParticles,
                                 double* maxPosX,
                                 double* maxPosY,
                                 double* maxPosZ);

/**
 * @brief Calculate min positions of each particle
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @param minPosX Device pointer to the min X coords array
 * @param minPosY Device pointer to the min Y coords array
 * @param minPosZ Device pointer to the min Z coords array
 */
__global__ void calcMinPosKernel(const ParticleCUDA* particles,
                                 const int64_t nParticles,
                                 double* minPosX,
                                 double* minPosY,
                                 double* minPosZ);

/**
 * @brief Calculate the maximum collision distance of each particle
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @param maxCollisionDistance Device pointer to the resulting maximum collision distance
 */
__global__ void calcCollisionDistanceKernel(const ParticleCUDA* particles,
                                            const int64_t nParticles,
                                            double* maxCollisionDistance);

/**
 * @brief Sum reduction on CPU
 *
 * @param data Host pointer to the input array
 * @param n The number of elements
 * @return double
 */
double cpuReduceSum(const double* data, const int64_t n);

/**
 * @brief Max reduction on CPU
 *
 * @param data Host pointer to the input array
 * @param n The number of elements
 * @return double
 */
double cpuReduceMax(const double* data, const int64_t n);

/**
 * @brief Min reduction on CPU
 *
 * @param data Host pointer to the input array
 * @param n The number of elements
 * @return double
 */
double cpuReduceMin(const double* data, const int64_t n);

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================================================================================
// Dispatchers
// ==============================================================================================================

/**
 * @brief Calculate kinetic energy of each particle
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @return double
 */
double launchCalcKineticEnergyKernel(const ParticleCUDA* particles,
                                     const int64_t nParticles);

/**
 * @brief Calculate max positions of each particle
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @return double3
 */
double3 launchCalcMaxPosKernel(const ParticleCUDA* particles,
                               const int64_t nParticles);

/**
 * @brief Calculate min positions of each particle
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @return double3
 */
double3 launchCalcMinPosKernel(const ParticleCUDA* particles,
                               const int64_t nParticles);

/**
 * @brief Calculate the maximum collision distance of each particle
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @param maxCollisionDistance Maximum collision distance
 * @param nCollisions The number of collided particles
 */
void launchCalcCollisionsKernel(const ParticleCUDA* particles,
                                const int64_t nParticles,
                                double& maxCollisionDistance,
                                int64_t& nCollisions);

#ifdef __cplusplus
}
#endif
