#pragma once

#include <ParticleStabilizerCUDA/PhysicsEngineCUDA.hpp>

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================================================================================
// Kernels
// ==============================================================================================================

/**
 * @brief Integrate acceleration and velocity according to the equation of motion
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @param gravity Gravitational acceleration
 * @param dt Delta time
 */
__global__ void updatePositionAndVelocityKernel(ParticleCUDA* particles,
                                                const int64_t nParticles,
                                                const double3 gravity,
                                                const double dt);

// ==============================================================================================================
// Dispatchers
// ==============================================================================================================

/**
 * @brief Integrate acceleration and velocity according to the equation of motion
 *
 * @param particles Device pointer to particles
 * @param nParticles The number of particles
 * @param gravity Gravitational acceleration
 * @param dt Delta time
 */
void lanchUpdatePositionAndVelocityKernel(ParticleCUDA* particles,
                                          const int64_t nParticles,
                                          const double3 gravity,
                                          const double dt);

#ifdef __cplusplus
}
#endif
