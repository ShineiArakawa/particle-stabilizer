#pragma once

#include <ParticleStabilizerCUDA/ParticleModelCUDA.hpp>
#include <SimView/core.hpp>

class ContainerBaseCUDA {
 public:
  virtual ~ContainerBaseCUDA() = default;

  /**
   * @brief Detect particle-wall collision and modifiy positions and velocities of particles.
   *
   * @param particle Device pointer to particles
   */
  virtual void resolveCollisions(ParticleCUDA* particles) = 0;

  /**
   * @brief Get the min coords of the box
   *
   * @return double3
   */
  virtual double3 getMinCoords() = 0;

  /**
   * @brief Get the max coords of the box
   *
   * @return double3
   */
  virtual double3 getMaxCoords() = 0;

  /**
   * @brief Get the Simview Primitive object
   *
   * @return simview::model::Primitive_t
   */
  virtual simview::model::Primitive_t getSimviewPrimitive() = 0;
};
