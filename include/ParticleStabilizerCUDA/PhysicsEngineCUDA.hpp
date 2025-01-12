#pragma once

#if defined(_WIN64)
#define DLLEXPORT_DECL extern "C" __declspec(dllexport)
#else
#define DLLEXPORT_DECL extern "C"
#endif

#include <ParticleStabilizer/ContainerBase.hpp>
#include <ParticleStabilizerCUDA/ContainerBaseCUDA.hpp>

#ifdef __cplusplus
extern "C" {
#endif

class PhysicsEngineCUDA {
 public:
  PhysicsEngineCUDA(std::shared_ptr<ParticleModelCUDA> particleModel,
                    std::shared_ptr<ContainerBaseCUDA> container,
                    const double3 gravity,
                    const double dt);

  void startSimulationLoop(const int64_t maxSteps);

 private:
  std::shared_ptr<ParticleModelCUDA> _particleModel;
  std::shared_ptr<ContainerBaseCUDA> _container;

  double3 _gravity;

  double _dt;

  void updatePositionAndVelocity();
};

#ifdef __cplusplus
}
#endif

// =================================================================================================
// Simulation runner
// =================================================================================================
DLLEXPORT_DECL void runSimulateCUDA(const std::vector<ParticlePtr>& particles,
                                    const glm::dvec3 dvec3_boxMin,
                                    const glm::dvec3 dvec3_boxMax,
                                    const std::string containerPath,
                                    const glm::dvec3 dvec3_gravity,
                                    const CoefficientOfRestitutionWall coefficientOfRestitutionWall,
                                    const double coefficientOfSpring,
                                    const double coefficientOfRestitutionSphere,
                                    const double dt,
                                    const size_t maxSteps,
                                    const bool toVisualize,
                                    std::vector<Particle>& resultParticles);
