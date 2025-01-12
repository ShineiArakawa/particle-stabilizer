#include <ParticleStabilizer/ParticleModel.hpp>
#include <Util/Logging.hpp>
#include <algorithm>
#include <cmath>

ParticleModel::ParticleModel(const std::vector<ParticlePtr>& particles,
                             const double coefficientOfSpring,
                             const double coefficientOfRestitution)
    : _particles(particles),
      _particlesBuffer(),
      _coefficientOfSpring(coefficientOfSpring),
      _coefficientOfRestitution(coefficientOfRestitution),
      _particleBucket(nullptr) {
  // =====================================================================================================
  // Initialize _particlesBuffer
  // =====================================================================================================
  const size_t nParticles = _particles.size();
  _particlesBuffer.resize(nParticles);

  for (size_t i = 0; i < nParticles; ++i) {
    _particlesBuffer[i] = std::make_shared<Particle>(_particles[i]->position,
                                                     _particles[i]->velocity,
                                                     _particles[i]->radius,
                                                     _particles[i]->mass);
  }
}

void ParticleModel::initBuckets(const glm::dvec3 boxMinCoords,
                                const glm::dvec3 boxMaxCoords) {
  // =====================================================================================================
  // Initialize ParticleBucket
  // =====================================================================================================
  const size_t nParticles = _particles.size();

  double maxRadius = 0.0;
  for (const auto& particle : _particles) {
    maxRadius = std::max(maxRadius, particle->radius);
  }

  _particleBucket = std::make_shared<ParticleBucket>();
  _particleBucket->initBucket(maxRadius, boxMinCoords, boxMaxCoords);

  for (size_t i = 0; i < nParticles; ++i) {
    const glm::dvec3& position = _particles[i]->position;
    _particleBucket->addParticleToBucket(position.x, position.y, position.z, i);
  }
}

void ParticleModel::updateBuckets() {
  // Clear buckets
  _particleBucket->clearBuckets();

  const size_t nParticles = _particles.size();

  for (size_t iParticle = 0; iParticle < nParticles; ++iParticle) {
    // Add particle to bucket
    _particleBucket->addParticleToBucket(_particles[iParticle]->position.x,
                                         _particles[iParticle]->position.y,
                                         _particles[iParticle]->position.z,
                                         iParticle);
  }
}

bool ParticleModel::isHit(const int64_t& i,
                          const int64_t& j) const {
  return glm::distance(_particles[i]->position, _particles[j]->position) < _particles[i]->radius + _particles[j]->radius;
}

void ParticleModel::reboundWithOtherParticle(const int64_t& i,
                                             const int64_t& j) {
  if (isHit(i, j)) {
    const glm::dvec3& iCoord = _particles[i]->position;
    const glm::dvec3& jCoord = _particles[j]->position;
    const glm::dvec3& iVelocity = _particles[i]->velocity;
    const glm::dvec3& jVelocity = _particles[j]->velocity;
    const double iRadius = _particles[i]->radius;
    const double jRadius = _particles[j]->radius;
    const double& iMass = _particles[i]->mass;
    const double& jMass = _particles[j]->mass;

    const glm::dvec3 deltaCoord = iCoord - jCoord;

    const double overlapDist = iRadius + jRadius - glm::length(deltaCoord);

    const glm::dvec3 reltive = (2.0 * jMass / (iMass + jMass)) * (glm::dot(iVelocity - jVelocity, deltaCoord) / glm::dot(deltaCoord, deltaCoord)) * deltaCoord * _coefficientOfRestitution;

    const glm::dvec3 springForce = _coefficientOfSpring * overlapDist * glm::normalize(deltaCoord);

    _particlesBuffer[i]->velocity += -reltive + springForce;

    {
      // // めり込みの修正
      // const glm::dvec3 correction = deltaCoord * overlapDist / 2.0;

      // _particles[i]->position += correction;
    }
  }
}

void ParticleModel::resolveCollisions() {
  copyParticleData(_particles, _particlesBuffer);

  const auto nParticles = static_cast<int64_t>(_particles.size());

  for (int64_t iParticle = 0; iParticle < nParticles; ++iParticle) {
    const auto& iCoord = _particles[iParticle]->position;

    const auto minBucketIndexX = _particleBucket->toBucketIndexX(iCoord.x) - 1LL;
    const auto minBucketIndexY = _particleBucket->toBucketIndexY(iCoord.y) - 1LL;
    const auto minBucketIndexZ = _particleBucket->toBucketIndexZ(iCoord.z) - 1LL;

    for (int64_t bucketIndexOffsetX = 0; bucketIndexOffsetX < 3LL; ++bucketIndexOffsetX) {
      const int64_t bucketIndexX = minBucketIndexX + bucketIndexOffsetX;

      for (int64_t bucketIndexOffsetY = 0; bucketIndexOffsetY < 3LL; ++bucketIndexOffsetY) {
        const int64_t bucketIndexY = minBucketIndexY + bucketIndexOffsetY;

        for (int64_t bucketIndexOffsetZ = 0; bucketIndexOffsetZ < 3LL; ++bucketIndexOffsetZ) {
          const int64_t bucketIndexZ = minBucketIndexZ + bucketIndexOffsetZ;

          const auto& particleIds = _particleBucket->getPaticleIdsInBucket(bucketIndexX,
                                                                           bucketIndexY,
                                                                           bucketIndexZ);

          if (particleIds == nullptr) {
            // Bucket is out of range
            continue;
          }

          for (const auto& jParticle : *particleIds) {
            if (iParticle == jParticle) {
              // Not to compare with itself
              continue;
            }

            reboundWithOtherParticle(iParticle, jParticle);
          }
        }
      }
    }
  }

  copyParticleData(_particlesBuffer, _particles);
}

size_t ParticleModel::getNumParticles() const {
  return _particles.size();
}

std::vector<ParticlePtr> ParticleModel::getParticles() const {
  return _particles;
}

void ParticleModel::resolveOverlaps(std::vector<Particle>& particles, const int axis) {
  LOG_INFO("### Resolving overlaps");

  int64_t nParticles = particles.size();
  std::vector<int64_t> rankingMap(nParticles);  // (rank) -> (index)

  {
    // Z座標のランキングを作成
    // インデックスで初期化
    for (int64_t i = 0; i < nParticles; ++i) {
      rankingMap[i] = i;
    }

    // position.zについてソート
    std::sort(rankingMap.begin(),
              rankingMap.end(),
              [particles, axis](int64_t i1, int64_t i2) {
                return particles[i1].position[axis] < particles[i2].position[axis];
              });
  }

  std::vector<int> otherAxes = {0, 1, 2};
  otherAxes.erase(std::remove(otherAxes.begin(), otherAxes.end(), axis), otherAxes.end());

  for (int64_t i = 0; i < nParticles; ++i) {
    LOG_INFO("i: " + std::to_string(i) + " / " + std::to_string(nParticles));
    const int64_t& iParticle = rankingMap[i];

    int64_t minRank = 0;
    for (int64_t j = 0; j < i; ++j) {
      const int64_t& jParticle = rankingMap[j];
      if (particles[iParticle].position[axis] - particles[iParticle].radius <= particles[jParticle].position[axis] + particles[jParticle].radius) {
        minRank = j;
        break;
      }
    }

    for (int64_t j = minRank; j < i; ++j) {
      const int64_t& jParticle = rankingMap[j];

      const double sumRadius = particles[iParticle].radius + particles[jParticle].radius;
      const double sumRadiusSquared = sumRadius * sumRadius;
      const double deltaX = particles[iParticle].position.x - particles[jParticle].position.x;
      const double deltaY = particles[iParticle].position.y - particles[jParticle].position.y;
      const double deltaZ = particles[iParticle].position.z - particles[jParticle].position.z;
      const double distanceSquared = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ;

      if (sumRadiusSquared - distanceSquared > 0.0) {
        const double delta0 = particles[iParticle].position[otherAxes[0]] - particles[jParticle].position[otherAxes[0]];
        const double delta1 = particles[iParticle].position[otherAxes[1]] - particles[jParticle].position[otherAxes[1]];

        particles[iParticle].position[axis] = particles[jParticle].position[axis] + std::sqrt(sumRadius * sumRadius - delta0 * delta0 - delta1 * delta1);

        int64_t maxRankChanged = i;
        for (int64_t k = i + 1; k < nParticles; ++k) {  // ランキングの変更が必要な最大のインデックスを求める
          if (particles[rankingMap[k - 1]].position[axis] < particles[iParticle].position[axis] &&
              particles[iParticle].position[axis] < particles[rankingMap[k]].position[axis]) {
            maxRankChanged = k;
            break;
          }
        }

        if (maxRankChanged != i) {
          std::sort(rankingMap.begin() + i,
                    rankingMap.begin() + maxRankChanged + 1,
                    [particles, axis](int64_t i1, int64_t i2) {
                      return particles[i1].position[axis] < particles[i2].position[axis];
                    });
          --i;  // やり直し
          break;
        }
      }
    }
  }

  // Check overlaps
  int64_t nCollidedParticles = 0;
  double maxDistance = 0.0;

  for (int64_t iParticle = 0; iParticle < nParticles; ++iParticle) {
    bool isCollided = false;

    for (int64_t jParticle = iParticle + 1; jParticle < nParticles; ++jParticle) {
      const glm::dvec3 delta = particles[iParticle].position - particles[jParticle].position;
      const double distance = std::sqrt(glm::dot(delta, delta));
      const double collisionDistance = particles[iParticle].radius + particles[jParticle].radius - distance;

      if (collisionDistance > 0.0) {
        isCollided = true;
        maxDistance = std::max(maxDistance, collisionDistance);
      }
    }

    if (isCollided) {
      ++nCollidedParticles;
    }
  }

  LOG_INFO("Number of collided particles: " + std::to_string(nCollidedParticles) + " / " + std::to_string(nParticles));

  char buffer[256];
  sprintf(buffer, "Max collision distance: %.15lf", maxDistance);
  LOG_INFO(std::string(buffer));

  LOG_INFO("### Finished resolving overlaps");
}

void ParticleModel::copyParticleData(const std::vector<ParticlePtr>& particlesFrom, std::vector<ParticlePtr>& particlesTo) {
  const size_t nParticles = particlesFrom.size();

  for (size_t i = 0; i < nParticles; ++i) {
    particlesTo[i]->position = particlesFrom[i]->position;
    particlesTo[i]->velocity = particlesFrom[i]->velocity;
    particlesTo[i]->radius = particlesFrom[i]->radius;
    particlesTo[i]->mass = particlesFrom[i]->mass;
  }
}
