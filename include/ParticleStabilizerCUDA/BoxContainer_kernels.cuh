#pragma once

#include <ParticleStabilizerCUDA/BoxContainerCUDA.hpp>
#include <ParticleStabilizerCUDA/ParticleModelCUDA.hpp>

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================================================================================
// Kernels
// ==============================================================================================================

/**
 * @brief Detect particle-wall collision and modifiy positions and velocities of particles.
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @param min Min coords of the box
 * @param max Max coords of the box
 * @param coefficientOfRestitutionWall Restitution coefficient for each sides of the box
 */
__global__ void resolveWallCollisionKernel(ParticleCUDA* particles,
                                           const int64_t nParticles,
                                           const double3 min,
                                           const double3 max,
                                           const CoefficientOfRestitutionWall coefficientOfRestitutionWall);

// ==============================================================================================================
// Dispatchers
// ==============================================================================================================

/**
 * @brief Detect particle-wall collision and modifiy positions and velocities of particles.
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @param min Min coords of the box
 * @param max Max coords of the box
 * @param coefficientOfRestitutionWall Restitution coefficient for each sides of the box
 *
 * @see resolveWallCollisionKernel()
 */
void launchResolveWallCollisionKernel(ParticleCUDA* particles,
                                      const int64_t nParticles,
                                      const double3 min,
                                      const double3 max,
                                      const CoefficientOfRestitutionWall coefficientOfRestitutionWall);

#ifdef __cplusplus
}
#endif
