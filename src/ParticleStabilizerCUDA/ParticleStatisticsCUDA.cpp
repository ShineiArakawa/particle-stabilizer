#include <ParticleStabilizerCUDA/ParticleStatisticsCUDA.hpp>
#include <ParticleStabilizerCUDA/ParticleStatisticsCUDA_kernels.cuh>

static double distance(const double3& a, const double3& b) {
  const double diffX = a.x - b.x;
  const double diffY = a.y - b.y;
  const double diffZ = a.z - b.z;
  return std::sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ);
}

ParticleStatisticsCUDA ParticleStatisticsCUDA::calculateStatistics(const ParticleCUDA* particles,
                                                                   const int64_t nParticles) {
  double totalKineticEnergy = 0.0;

  double minPositionX = particles[0].position.x - particles[0].radius;
  double minPositionY = particles[0].position.y - particles[0].radius;
  double minPositionZ = particles[0].position.z - particles[0].radius;

  double maxPositionX = particles[0].position.x + particles[0].radius;
  double maxPositionY = particles[0].position.y + particles[0].radius;
  double maxPositionZ = particles[0].position.z + particles[0].radius;

  for (int64_t i = 0; i < nParticles; ++i) {
    const auto particle = particles[i];

    totalKineticEnergy += 0.5 * particle.mass * (particle.velocity.x * particle.velocity.x + particle.velocity.y * particle.velocity.y + particle.velocity.z * particle.velocity.z);

    minPositionX = std::min(minPositionX, particle.position.x - particle.radius);
    minPositionY = std::min(minPositionY, particle.position.y - particle.radius);
    minPositionZ = std::min(minPositionZ, particle.position.z - particle.radius);
    maxPositionX = std::max(maxPositionX, particle.position.x + particle.radius);
    maxPositionY = std::max(maxPositionY, particle.position.y + particle.radius);
    maxPositionZ = std::max(maxPositionZ, particle.position.z + particle.radius);
  }

  int64_t nCollidedParticles = 0;
  double maxDistance = 0.0;

  for (int64_t iParticle = 0; iParticle < nParticles; ++iParticle) {
    bool isCollided = false;

    for (int64_t jParticle = iParticle + 1; jParticle < nParticles; ++jParticle) {
      const double dist = distance(particles[iParticle].position, particles[jParticle].position);
      const double collisionDistance = particles[iParticle].radius + particles[jParticle].radius - dist;

      if (collisionDistance > 0.0) {
        isCollided = true;
        maxDistance = std::max(maxDistance, collisionDistance);
      }
    }

    if (isCollided) {
      ++nCollidedParticles;
    }
  }

  return ParticleStatisticsCUDA(totalKineticEnergy,
                                make_double3(minPositionX, minPositionY, minPositionZ),
                                make_double3(maxPositionX, maxPositionY, maxPositionZ),
                                nCollidedParticles,
                                maxDistance);
}

ParticleStatisticsCUDA ParticleStatisticsCUDA::calculateStatisticsOnDevice(const ParticleCUDA* particles,
                                                                           const int64_t nParticles) {
  const double totalKineticEnergy = launchCalcKineticEnergyKernel(particles, nParticles);
  const double3 minPosition = launchCalcMinPosKernel(particles, nParticles);
  const double3 maxPosition = launchCalcMaxPosKernel(particles, nParticles);

  double maxDistance;
  int64_t nCollidedParticles;
  launchCalcCollisionsKernel(particles, nParticles, maxDistance, nCollidedParticles);

  return ParticleStatisticsCUDA(totalKineticEnergy,
                                minPosition,
                                maxPosition,
                                nCollidedParticles,
                                maxDistance);
}