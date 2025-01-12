#pragma once

#include <ParticleStabilizerCUDA/ParticleModelCUDA.hpp>

#ifdef __cplusplus
extern "C" {
#endif

struct ParticleStatisticsCUDA {
  ParticleStatisticsCUDA(const double totalKineticEnergy_,
                         const double3 minPosition_,
                         const double3 maxPosition_,
                         const int64_t nCollidedParticles_,
                         const double maxCollisionDistance_)
      : totalKineticEnergy(totalKineticEnergy_),
        minPosition(minPosition_),
        maxPosition(maxPosition_),
        nCollidedParticles(nCollidedParticles_),
        maxCollisionDistance(maxCollisionDistance_){};
  double totalKineticEnergy;
  double3 minPosition;
  double3 maxPosition;
  int64_t nCollidedParticles;
  double maxCollisionDistance;

  static ParticleStatisticsCUDA calculateStatistics(const ParticleCUDA* particles,
                                                    const int64_t nParticles);

  static ParticleStatisticsCUDA calculateStatisticsOnDevice(const ParticleCUDA* particles,
                                                            const int64_t nParticles);
};

#ifdef __cplusplus
}
#endif
