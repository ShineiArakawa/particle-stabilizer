#include <ParticleStabilizer/ParticleStatistics.hpp>

ParticleStatistics ParticleStatistics::calculateStatistics(const std::vector<ParticlePtr> &particles) {
  double totalKineticEnergy = 0.0;
  glm::dvec3 minPosition = particles[0]->position - particles[0]->radius;
  glm::dvec3 maxPosition = particles[0]->position + particles[0]->radius;

  for (const auto &particle : particles) {
    totalKineticEnergy += 0.5 * particle->mass * glm::dot(particle->velocity, particle->velocity);

    minPosition = glm::min(minPosition, particle->position - particle->radius);
    maxPosition = glm::max(maxPosition, particle->position + particle->radius);
  }

  const auto &nParticles = particles.size();

  size_t nCollidedParticles = 0;
  double maxDistance = 0.0;

  for (size_t iParticle = 0; iParticle < nParticles; ++iParticle) {
    bool isCollided = false;

    for (size_t jParticle = iParticle + 1; jParticle < nParticles; ++jParticle) {
      const double distance = glm::distance(particles[iParticle]->position, particles[jParticle]->position);
      const double collisionDistance = particles[iParticle]->radius + particles[jParticle]->radius - distance;

      if (collisionDistance > 0.0) {
        isCollided = true;
        maxDistance = std::max(maxDistance, collisionDistance);
      }
    }

    if (isCollided) {
      ++nCollidedParticles;
    }
  }

  return ParticleStatistics(totalKineticEnergy,
                            minPosition,
                            maxPosition,
                            nCollidedParticles,
                            maxDistance);
}