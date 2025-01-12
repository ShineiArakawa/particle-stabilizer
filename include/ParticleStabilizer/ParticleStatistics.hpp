#pragma once

#include <ParticleStabilizer/ParticleModel.hpp>
#include <vector>

enum class VerticalDirection {
  X,
  Y,
  Z
};

struct ParticleStatistics {
  ParticleStatistics(const double totalKineticEnergy_,
                     const glm::dvec3 minPosition_,
                     const glm::dvec3 maxPosition_,
                     const size_t nCollidedParticles_,
                     const double maxCollisionDistance_)
      : totalKineticEnergy(totalKineticEnergy_),
        minPosition(minPosition_),
        maxPosition(maxPosition_),
        nCollidedParticles(nCollidedParticles_),
        maxCollisionDistance(maxCollisionDistance_){};
  double totalKineticEnergy;
  glm::dvec3 minPosition;
  glm::dvec3 maxPosition;
  size_t nCollidedParticles;
  double maxCollisionDistance;

  static ParticleStatistics calculateStatistics(const std::vector<ParticlePtr> &particles);
};
