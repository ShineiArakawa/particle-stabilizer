#pragma once

#include <ParticleStabilizer/ParticleBucket.hpp>
#include <memory>
#include <vector>

template <typename Vector3>
struct ParticlePrimitive {
  ParticlePrimitive() = default;
  ParticlePrimitive(Vector3 position_,
                    Vector3 velocity_,
                    double radius_,
                    double mass_)
      : position(position_),
        velocity(velocity_),
        radius(radius_),
        mass(mass_){};

  ~ParticlePrimitive() = default;

  Vector3 position;
  Vector3 velocity;
  double radius;
  double mass;
};

using Particle = ParticlePrimitive<glm::dvec3>;
using ParticlePtr = std::shared_ptr<Particle>;

class ParticleModel {
 private:
  std::vector<ParticlePtr> _particles;
  std::vector<ParticlePtr> _particlesBuffer;

  double _coefficientOfSpring;
  double _coefficientOfRestitution;

  std::shared_ptr<ParticleBucket> _particleBucket;

  bool isHit(const int64_t& i,
             const int64_t& j) const;

  void reboundWithOtherParticle(const int64_t& i,
                                const int64_t& j);

 public:
  ParticleModel(const std::vector<ParticlePtr>& particles,
                const double coefficientOfSpring,
                const double coefficientOfRestitution);

  ~ParticleModel() = default;

  void initBuckets(const glm::dvec3 boxMinCoords,
                   const glm::dvec3 boxMaxCoords);

  void updateBuckets();

  void resolveCollisions();

  size_t getNumParticles() const;

  std::vector<ParticlePtr> getParticles() const;

  static void resolveOverlaps(std::vector<Particle>& particles, const int axis = 2);

  static void copyParticleData(const std::vector<ParticlePtr>& particlesFrom, std::vector<ParticlePtr>& particlesTo);
};