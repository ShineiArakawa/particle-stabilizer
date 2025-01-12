#pragma once

#include <ParticleStabilizer/ContainerBase.hpp>
#include <ParticleStabilizerCUDA/ContainerBaseCUDA.hpp>

#ifdef __cplusplus
extern "C" {
#endif

class BoxContainerCUDA : public ContainerBaseCUDA {
 public:
  /**
   * @brief Construct a new box container object
   *
   * @param min Min coords of the box
   * @param max Max coords of the box
   * @param coefficientOfRestitution Restitution coefficient for each sides of the box
   * @param nParticles The number of particles
   */
  BoxContainerCUDA(const double3 min,
                   const double3 max,
                   const CoefficientOfRestitutionWall coefficientOfRestitution,
                   const int64_t nParticles);

  ~BoxContainerCUDA() = default;

  /**
   * @brief Detect particle-wall collision and modifiy positions and velocities of particles.
   *
   * @param particle Device pointer to particles
   */
  void resolveCollisions(ParticleCUDA* particle) override;

  /**
   * @brief Get the min coords of the box
   *
   * @return double3
   */
  double3 getMinCoords() override;

  /**
   * @brief Get the max coords of the box
   *
   * @return double3
   */
  double3 getMaxCoords() override;

  /**
   * @brief Get the Simview Primitive object
   *
   * @return simview::model::Primitive_t
   */
  simview::model::Primitive_t getSimviewPrimitive() override;

 private:
  double3 _min;
  double3 _max;

  double3 _minCoordsBuffer;
  double3 _maxCoordsBuffer;

  int64_t _nParticles;

  CoefficientOfRestitutionWall _coefficientOfRestitution;
};

#ifdef __cplusplus
}
#endif
