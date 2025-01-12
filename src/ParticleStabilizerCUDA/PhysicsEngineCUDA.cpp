#include <ParticleStabilizerCUDA/BoxContainerCUDA.hpp>
#include <ParticleStabilizerCUDA/ParticleStatisticsCUDA.hpp>
#include <ParticleStabilizerCUDA/PhysicsEngineCUDA.hpp>
#include <ParticleStabilizerCUDA/PhysicsEngine_kernels.cuh>
#include <ParticleStabilizerCUDA/PolygonContainerCUDA.hpp>
#include <ParticleStabilizerCUDA/VisualizationCUDA.hpp>

PhysicsEngineCUDA::PhysicsEngineCUDA(std::shared_ptr<ParticleModelCUDA> particleModel,
                                     std::shared_ptr<ContainerBaseCUDA> container,
                                     const double3 gravity,
                                     const double dt)
    : _particleModel(particleModel),
      _container(container),
      _gravity(gravity),
      _dt(dt) {
}

void PhysicsEngineCUDA::updatePositionAndVelocity() {
  lanchUpdatePositionAndVelocityKernel(_particleModel->getDeviceParticles(),
                                       _particleModel->getNumParticles(),
                                       _gravity,
                                       _dt);
}

void PhysicsEngineCUDA::startSimulationLoop(const int64_t maxSteps) {
  bool toContinue = true;

  const auto startTimeStat = std::chrono::high_resolution_clock::now();
  auto previousTimeStat = startTimeStat;

  int64_t nSteps = 0;

  while (toContinue) {
    ++nSteps;

    {
      // Simulate the physics
      updatePositionAndVelocity();

      _container->resolveCollisions(_particleModel->getDeviceParticles());

      _particleModel->resolveCollisions();
    }

    {
      // Statistics
      const auto currentTime = std::chrono::high_resolution_clock::now();
      const double duration = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - previousTimeStat).count() * 1e-6;

      if (duration > 10.0 || nSteps == maxSteps) {
        const double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTimeStat).count() * 1e-6;

#if 1
        const auto paricles = _particleModel->getParticles();
        const auto& stat = ParticleStatisticsCUDA::calculateStatistics(paricles, _particleModel->getNumParticles());
        free(paricles);
#else
        ParticleCUDA* const paricles = _particleModel->getDeviceParticles();  // NOTE: This is a device pointer
        const auto& stat = ParticleStatisticsCUDA::calculateStatisticsOnDevice(paricles, _particleModel->getNumParticles());
#endif

        LOG_INFO("######################## Step " + std::to_string(nSteps) + ", Elapsed time: " + std::to_string(elapsedTime) + " [sec] ########################");
        LOG_INFO("Total kinetic energy         : " + std::to_string(stat.totalKineticEnergy));
        LOG_INFO("Max position                 : (" + std::to_string(stat.maxPosition.x) + ", " + std::to_string(stat.maxPosition.y) + ", " + std::to_string(stat.maxPosition.z) + ")");
        LOG_INFO("Min position                 : (" + std::to_string(stat.minPosition.x) + ", " + std::to_string(stat.minPosition.y) + ", " + std::to_string(stat.minPosition.z) + ")");
        LOG_INFO("Number of collided particles : " + std::to_string(stat.nCollidedParticles) + " / " + std::to_string(_particleModel->getNumParticles()));
        LOG_INFO("Max collision distance       : " + std::to_string(stat.maxCollisionDistance));

        previousTimeStat = currentTime;
      }
    }

    if (nSteps >= maxSteps) {
      toContinue = false;
    }
  }

  const auto currentTime = std::chrono::high_resolution_clock::now();
  const double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTimeStat).count() * 1e-6;
  LOG_INFO("Total elapsed time: " + std::to_string(elapsedTime) + " [sec]");
}

// =================================================================================================
// Simulation runner
// =================================================================================================
void runSimulateCUDA(const std::vector<ParticlePtr>& particles,
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
                     std::vector<Particle>& resultParticles) {
  // Create particle model
  std::shared_ptr<ParticleModelCUDA> particleModel = std::make_shared<ParticleModelCUDA>(particles,
                                                                                         coefficientOfSpring,
                                                                                         coefficientOfRestitutionSphere);

  // Create container
  std::shared_ptr<ContainerBaseCUDA> container = nullptr;
  if (containerPath.empty()) {
    // Simple box
    container = std::make_shared<BoxContainerCUDA>(make_double3(dvec3_boxMin.x, dvec3_boxMin.y, dvec3_boxMin.z),
                                                   make_double3(dvec3_boxMax.x, dvec3_boxMax.y, dvec3_boxMax.z),
                                                   coefficientOfRestitutionWall,
                                                   particleModel->getNumParticles());
  } else {
    container = std::make_shared<PolygonContainerCUDA>(containerPath,
                                                       coefficientOfRestitutionWall.zMin,
                                                       particleModel->getNumParticles());
  }

  particleModel->initBuckets(container->getMinCoords(), container->getMaxCoords());

  // Create the physics engine
  std::shared_ptr<PhysicsEngineCUDA> physicsEngine = std::make_shared<PhysicsEngineCUDA>(particleModel,
                                                                                         container,
                                                                                         make_double3(dvec3_gravity.x, dvec3_gravity.y, dvec3_gravity.z),
                                                                                         dt);

  if (toVisualize) {
    // Launch the viewer
    auto viewer = std::make_shared<simview::app::ViewerGUIApp>();
    std::vector<simview::model::Sphere_t> spheres;
    launchViewerCUDA(spheres, particleModel, container, viewer);

    // Create stream executor
    auto streamExecutor = std::make_shared<simview::util::StreamExecutor>();

    // NOTE: Run simulation loop in a separate thread
    streamExecutor->enqueue([physicsEngine, maxSteps]() {
      // Start the simulation loop
      physicsEngine->startSimulationLoop(maxSteps);
    });

    // NOTE: Run viewer loop in the main thread
    // NOTE: OpenGL context should be created in the main thread
    while (viewer->shouldUpdateWindow()) {
      // Update the spheres
      updateSpheresCUDA(particleModel, spheres);

      // Update the viewer
      viewer->paint();
    }
  } else {
    // Start the simulation loop
    physicsEngine->startSimulationLoop(maxSteps);
  }

  {
    // Copy the result particles
    const size_t nParticles = particleModel->getNumParticles();

    ParticleCUDA* ptr_resultParticles = particleModel->getParticles();

    for (size_t iParticle = 0; iParticle < nParticles; ++iParticle) {
      const auto& particle = ptr_resultParticles[iParticle];

      resultParticles[iParticle] = Particle(glm::dvec3(particle.position.x,
                                                       particle.position.y,
                                                       particle.position.z),
                                            glm::dvec3(particle.velocity.x,
                                                       particle.velocity.y,
                                                       particle.velocity.z),
                                            particle.radius,
                                            particle.mass);
    }

    free(ptr_resultParticles);
  }
}