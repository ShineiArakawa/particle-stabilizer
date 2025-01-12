#pragma once

#include <ParticleStabilizer/ContainerBase.hpp>
#include <ParticleStabilizer/ParticleModel.hpp>

class PhysicsEngine {
 public:
  PhysicsEngine(std::shared_ptr<ParticleModel> particleModel,
                std::shared_ptr<ContainerBase> container,
                const glm::dvec3 gravity,
                const double dt);

  void startSimulationLoop(const size_t maxSteps);

 private:
  std::shared_ptr<ParticleModel> _particleModel;
  std::shared_ptr<ContainerBase> _container;

  glm::dvec3 _gravity;

  double _dt;

  void updatePositionAndVelocity();

  std::shared_ptr<ParticleModel> getParticleModel();
};

// =================================================================================================
// Simulation runner
// =================================================================================================
void runSimulate(const std::vector<ParticlePtr>& particles,
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
