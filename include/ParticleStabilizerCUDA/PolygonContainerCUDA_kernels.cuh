#pragma once

#include <ParticleStabilizerCUDA/ParticleModelCUDA.hpp>

#ifdef __cplusplus
extern "C" {
#endif

// ==================================================================================================================
// Kernels
// ==================================================================================================================

/**
 * @brief Kernel function to initialize buffers.
 *
 * This kernel initializes the intersection counter, positional modification array,
 * and velocity buffer for each particle. All values are set to zero.
 *
 * @param[out] intersectCounter Array to store intersection counters, initialized to zero.
 * @param[out] posModification Array to store positional modification amounts, initialized to zero.
 * @param[out] velocityBuffer Array to store velocity buffers, initialized to zero.
 * @param[in] nParticles Number of particles.
 *
 * @note This is a CUDA kernel function, intended to be executed on the GPU.
 */
__global__ void initBuffersKernel(int* intersectCounter,
                                  double3* posModification,
                                  double3* velocityBuffer,
                                  const int64_t nParticles);

/**
 * @brief Kernel function to check for intersections between particles and polygons.
 *
 * This kernel checks whether each particle intersects with each polygon and updates
 * the intersection status and counter. The intersection test includes an Axis-Aligned
 * Bounding Box (AABB) test followed by a detailed intersection test.
 *
 * @param[in] particles Array of particles.
 * @param[in] nParticles Number of particles.
 * @param[in] polygonCoords Array of coordinates for the polygons.
 * @param[in] polygonNormals Array of face normals for the polygons.
 * @param[in] polygonMinCoords Array of minimum coordinates for the polygons' AABB.
 * @param[in] polygonMaxCoords Array of maximum coordinates for the polygons' AABB.
 * @param[in] nPolygons Number of polygons.
 * @param[out] isIntersected Array indicating if a particle intersects with a polygon.
 * @param[out] intersectCounter Array counting the number of intersections per particle.
 *
 * @note This is a CUDA kernel function, intended to be executed on the GPU.
 *
 * @see ParticleCUDA
 * @see getClosestPointOnPolygonKernelFunc
 */
__global__ void isIntersectedKernel(const ParticleCUDA* particles,
                                    const int64_t nParticles,
                                    const double3* polygonCoords,
                                    const double3* polygonNormals,
                                    const double3* polygonMinCoords,
                                    const double3* polygonMaxCoords,
                                    const int64_t nPolygons,
                                    bool* isIntersected,
                                    int* intersectCounter);

/**
 * @brief Kernel function to calculate the sum of modified velocities for intersected particles.
 *
 * This kernel computes the reflected velocity for particles that intersect with polygons and
 * accumulates the modified velocities into a buffer. The reflection is computed using the
 * coefficient of restitution.
 *
 * @param[in] particles Array of particles.
 * @param[in] nParticles Number of particles.
 * @param[in] polygonNormals Array of face normals for the polygons.
 * @param[in] nPolygons Number of polygons.
 * @param[in] isIntersected Array indicating if a particle intersects with a polygon.
 * @param[in] coefficientOfRestitution Coefficient of restitution for velocity reflection.
 * @param[out] velocityBuffer Array to store the sum of modified velocities.
 *
 * @note This is a CUDA kernel function, intended to be executed on the GPU.
 *
 * @see ParticleCUDA
 */
__global__ void calcModifiedVelocitySumKernel(const ParticleCUDA* particles,
                                              const int64_t nParticles,
                                              const double3* polygonNormals,
                                              const int64_t nPolygons,
                                              const bool* isIntersected,
                                              const double coefficientOfRestitution,
                                              double3* velocityBuffer);

/**
 * @brief Kernel function to calculate the averaged velocity for each particle.
 *
 * This kernel computes the averaged velocity of particles based on the velocity
 * buffer and intersection counter. Each particle's velocity is updated if it has
 * been intersected at least once.
 *
 * @param[in] nParticles Number of particles.
 * @param[in] velocityBuffer Array of velocity vectors.
 * @param[in] intersectCounter Array of intersection counters.
 * @param[out] particles Array of particles to update with averaged velocities.
 *
 * @note This is a CUDA kernel function, intended to be executed on the GPU.
 *
 * @see ParticleCUDA
 */
__global__ void calcAveragedVelocityKernel(const int64_t nParticles,
                                           const double3* velocityBuffer,
                                           const int* intersectCounter,
                                           ParticleCUDA* particles);

/**
 * @brief Kernel function to calculate the position modification amount for intersected particles.
 *
 * This kernel computes the amount of position modification needed for particles that intersect
 * with polygons and accumulates these modifications into a buffer. The modification is based on
 * the distance to the polygon plane and the particle's radius.
 *
 * @param[in] particles Array of particles.
 * @param[in] nParticles Number of particles.
 * @param[in] polygonCoords Array of coordinates for the polygons.
 * @param[in] polygonNormals Array of normals for the polygons.
 * @param[in] nPolygons Number of polygons.
 * @param[in] isIntersected Array indicating if a particle intersects with a polygon.
 * @param[out] modification Array to store the accumulated position modifications.
 *
 * @note This is a CUDA kernel function, intended to be executed on the GPU.
 *
 * @see ParticleCUDA
 */
__global__ void calcPosModificationAmountKernel(const ParticleCUDA* particles,
                                                const int64_t nParticles,
                                                const double3* polygonCoords,
                                                const double3* polygonNormals,
                                                const int64_t nPolygons,
                                                const bool* isIntersected,
                                                double3* modification);

/**
 * @brief Kernel function to modify the position of particles.
 *
 * This kernel updates the position of each particle based on the given modification amounts.
 *
 * @param[in] nParticles Number of particles.
 * @param[in] modification Array of position modification vectors.
 * @param[in,out] particles Array of particles to be updated with new positions.
 *
 * @note This is a CUDA kernel function, intended to be executed on the GPU.
 *
 * @see ParticleCUDA
 */
__global__ void modifyPositionKernel(const int64_t nParticles,
                                     const double3* modification,
                                     ParticleCUDA* particles);

// ==================================================================================================================
// Dispacthers
// ==================================================================================================================

/**
 * @brief Launches a CUDA kernel to initialize particle buffers.
 *
 * This function sets up and launches a CUDA kernel to initialize the particle
 * buffers. After the kernel execution, it synchronizes the device.
 *
 * @param[in,out] intersectCounter Pointer to an integer counter for intersections.
 * @param[in,out] posModification Pointer to the buffer for position modifications (double3 type).
 * @param[in,out] velocityBuffer Pointer to the buffer for velocity values (double3 type).
 * @param[in] nParticles The total number of particles to initialize.
 */
void launchInitBuffersKernel(int* intersectCounter,
                             double3* posModification,
                             double3* velocityBuffer,
                             const int64_t nParticles);

/**
 * @brief Launches a CUDA kernel to check for intersections between particles and polygons.
 *
 * This function sets up and launches a CUDA kernel to determine if particles
 * intersect with any polygons. After the kernel execution, it synchronizes
 * the device.
 *
 * @param[in] particles Pointer to the array of particles (ParticleCUDA type).
 * @param[in] nParticles The total number of particles.
 * @param[in] polygonCoords Pointer to the array of polygon coordinates (double3 type).
 * @param[in] polygonNormals Pointer to the array of polygon normals (double3 type).
 * @param[in] polygonMinCoords Pointer to the array of minimum coordinates of the polygons (double3 type).
 * @param[in] polygonMaxCoords Pointer to the array of maximum coordinates of the polygons (double3 type).
 * @param[in] nPolygons The total number of polygons.
 * @param[out] isIntersected Pointer to the array that will store intersection results (bool type).
 * @param[out] intersectCounter Pointer to an counter for intersections.
 */
void launchIsIntersectedKernel(const ParticleCUDA* particles,
                               const int64_t nParticles,
                               const double3* polygonCoords,
                               const double3* polygonNormals,
                               const double3* polygonMinCoords,
                               const double3* polygonMaxCoords,
                               const int64_t nPolygons,
                               bool* isIntersected,
                               int* intersectCounter);

/**
 * @brief Launches a CUDA kernel to calculate the modified velocity sum for particles.
 *
 * This function sets up and launches a CUDA kernel to compute the modified velocity
 * sum for particles based on their intersections with polygons and a given coefficient
 * of restitution. After the kernel execution, it synchronizes the device.
 *
 * @param[in] particles Pointer to the array of particles (ParticleCUDA type).
 * @param[in] nParticles The total number of particles.
 * @param[in] polygonNormals Pointer to the array of polygon normals (double3 type).
 * @param[in] nPolygons The total number of polygons.
 * @param[in] isIntersected Pointer to the array that stores intersection results (bool type).
 * @param[in] coefficientOfRestitution The coefficient of restitution for collisions.
 * @param[out] velocityBuffer Pointer to the buffer for velocity values (double3 type).
 */
void launchCalcModifiedVelocitySumKernel(const ParticleCUDA* particles,
                                         const int64_t nParticles,
                                         const double3* polygonNormals,
                                         const int64_t nPolygons,
                                         const bool* isIntersected,
                                         const double coefficientOfRestitution,
                                         double3* velocityBuffer);

/**
 * @brief Launches a CUDA kernel to calculate the averaged velocity for particles.
 *
 * This function sets up and launches a CUDA kernel to compute the averaged velocity
 * for particles. After the kernel execution, it synchronizes the device.
 *
 * @param[in] nParticles The total number of particles.
 * @param[in] velocityBuffer Pointer to the buffer for summed velocity values (double3 type).
 * @param[in] intersectCounter Pointer to an integer counter for intersections.
 * @param[in,out] particles Pointer to the array of particles (ParticleCUDA type).
 */
void launchCalcAveragedVelocityKernel(const int64_t nParticles,
                                      const double3* velocityBuffer,
                                      const int* intersectCounter,
                                      ParticleCUDA* particles);

/**
 * @brief Launches a CUDA kernel to calculate position modifications for particles.
 *
 * This function sets up and launches a CUDA kernel to compute position modification amounts
 * for particles based on their intersections with polygons. After the kernel execution, it
 * synchronizes the device.
 *
 * @param[in] particles Pointer to the array of particles (ParticleCUDA type).
 * @param[in] nParticles The total number of particles.
 * @param[in] polygonCoords Pointer to the array of polygon coordinates (double3 type).
 * @param[in] polygonNormals Pointer to the array of polygon normals (double3 type).
 * @param[in] nPolygons The total number of polygons.
 * @param[in] isIntersected Pointer to the array that stores intersection results (bool type).
 * @param[out] modification Pointer to the buffer for position modification amounts (double3 type).
 */
void launchCalcPositionModificationKernel(const ParticleCUDA* particles,
                                          const int64_t nParticles,
                                          const double3* polygonCoords,
                                          const double3* polygonNormals,
                                          const int64_t nPolygons,
                                          const bool* isIntersected,
                                          double3* modification);

/**
 * @brief Launches a CUDA kernel to modify the positions of particles.
 *
 * This function sets up and launches a CUDA kernel to modify the positions of particles
 * based on the given position modifications. After the kernel execution, it synchronizes the device.
 *
 * @param[in] nParticles The total number of particles.
 * @param[out] modification Pointer to the buffer for position modification amounts (double3 type).
 * @param[in,out] particles Pointer to the array of particles (ParticleCUDA type).
 */
void launchModifyPositionKernel(const int64_t nParticles,
                                const double3* modification,
                                ParticleCUDA* particles);

#ifdef __cplusplus
}
#endif
