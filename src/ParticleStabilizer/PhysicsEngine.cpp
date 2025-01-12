#include <ParticleStabilizer/BoxContainer.hpp>
#include <ParticleStabilizer/ParticleStatistics.hpp>
#include <ParticleStabilizer/PhysicsEngine.hpp>
#include <ParticleStabilizer/PolygonContainer.hpp>
#include <ParticleStabilizer/Visualization.hpp>
#include <Util/Logging.hpp>
#include <chrono>

PhysicsEngine::PhysicsEngine(std::shared_ptr<ParticleModel> particleModel,
                             std::shared_ptr<ContainerBase> container,
                             const glm::dvec3 gravity,
                             const double dt)
    : _particleModel(particleModel),
      _container(container),
      _gravity(gravity),
      _dt(dt) {
}

void PhysicsEngine::updatePositionAndVelocity() {
  const auto& particles = _particleModel->getParticles();

  for (auto& particle : particles) {
    particle->position += particle->velocity * _dt;
    particle->velocity += _gravity * _dt;
  }
}

void PhysicsEngine::startSimulationLoop(const size_t maxSteps) {
  bool toContinue = true;

  const auto startTime = std::chrono::high_resolution_clock::now();
  auto previousTime = startTime;
  size_t nSteps = 0;

  while (toContinue) {
    ++nSteps;

    {
      // Simulate the physics
      updatePositionAndVelocity();

      _particleModel->updateBuckets();

      auto particles = _particleModel->getParticles();
      _container->resolveCollisions(particles);

      _particleModel->resolveCollisions();
    }

    // Statistics
    const auto currentTime = std::chrono::high_resolution_clock::now();
    const double duration = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - previousTime).count() * 1e-6;

    if (duration > 10.0 || nSteps == maxSteps) {
      const double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime).count() * 1e-6;

      const auto& stat = ParticleStatistics::calculateStatistics(_particleModel->getParticles());

      LOG_INFO("######################## Step " + std::to_string(nSteps) + ", Elapsed time: " + std::to_string(elapsedTime) + " [sec] ########################");
      LOG_INFO("Total kinetic energy         : " + std::to_string(stat.totalKineticEnergy));
      LOG_INFO("Max position                 : (" + std::to_string(stat.maxPosition.x) + ", " + std::to_string(stat.maxPosition.y) + ", " + std::to_string(stat.maxPosition.z) + ")");
      LOG_INFO("Min position                 : (" + std::to_string(stat.minPosition.x) + ", " + std::to_string(stat.minPosition.y) + ", " + std::to_string(stat.minPosition.z) + ")");
      LOG_INFO("Number of collided particles : " + std::to_string(stat.nCollidedParticles) + " / " + std::to_string(_particleModel->getNumParticles()));
      LOG_INFO("Max collision distance       : " + std::to_string(stat.maxCollisionDistance));

      previousTime = currentTime;
    }

    if (nSteps >= maxSteps) {
      toContinue = false;
    }
  }

  const auto currentTime = std::chrono::high_resolution_clock::now();
  const double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime).count() * 1e-6;
  LOG_INFO("Total elapsed time: " + std::to_string(elapsedTime) + " [sec]");
}

// =================================================================================================
// Simulation runner
// =================================================================================================
void runSimulate(const std::vector<ParticlePtr>& particles,
                 const glm::dvec3 boxMin,
                 const glm::dvec3 boxMax,
                 const std::string containerPath,
                 const glm::dvec3 gravity,
                 const CoefficientOfRestitutionWall coefficientOfRestitutionWall,
                 const double coefficientOfSpring,
                 const double coefficientOfRestitutionSphere,
                 const double dt,
                 const size_t maxSteps,
                 const bool toVisualize,
                 std::vector<Particle>& resultParticles) {
  // Create particle model
  std::shared_ptr<ParticleModel> particleModel = std::make_shared<ParticleModel>(particles,
                                                                                 coefficientOfSpring,
                                                                                 coefficientOfRestitutionSphere);

  // Create container
  std::shared_ptr<ContainerBase> container = nullptr;
  if (containerPath.empty()) {
    // Simple box
    container = std::make_shared<BoxContainer>(boxMin,
                                               boxMax,
                                               coefficientOfRestitutionWall);
  } else {
    container = std::make_shared<PolygonContainer>(containerPath,
                                                   coefficientOfRestitutionWall.zMin,
                                                   particleModel->getNumParticles());
  }

  particleModel->initBuckets(container->getMinCoords(), container->getMaxCoords());

  // Create the physics engine
  std::shared_ptr<PhysicsEngine> physicsEngine = std::make_shared<PhysicsEngine>(particleModel,
                                                                                 container,
                                                                                 gravity,
                                                                                 dt);

  if (toVisualize) {
    // Launch the viewer
    std::vector<simview::model::Sphere_t> spheres;
    auto viewer = launchViewer(spheres, particleModel, container);

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
      updateSpheres(particleModel->getParticles(), spheres);

      // Update the viewer
      viewer->paint();
    }
  } else {
    // Start the simulation loop
    physicsEngine->startSimulationLoop(maxSteps);
  }

  // Copy the result particles
  const auto tmp_particles = particleModel->getParticles();
  for (size_t iParticle = 0; iParticle < particles.size(); ++iParticle) {
    resultParticles[iParticle] = Particle(glm::dvec3(tmp_particles[iParticle]->position.x,
                                                     tmp_particles[iParticle]->position.y,
                                                     tmp_particles[iParticle]->position.z),
                                          glm::dvec3(tmp_particles[iParticle]->velocity.x,
                                                     tmp_particles[iParticle]->velocity.y,
                                                     tmp_particles[iParticle]->velocity.z),
                                          tmp_particles[iParticle]->radius,
                                          tmp_particles[iParticle]->mass);
  }
}
