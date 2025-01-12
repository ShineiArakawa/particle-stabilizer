#pragma once

#include <ParticleStabilizer/ParticleModel.hpp>
#include <ParticleStabilizerCUDA/CudaCommon.cuh>
#include <set>

#ifndef USE_NATIVE_PREFIX_SCAN
#include <ParticleStabilizerCUDA/PrefixScan.cuh>
#endif

#ifdef __cplusplus
extern "C" {
#endif

using ParticleCUDA = ParticlePrimitive<double3>;

struct BucketContext {
  BucketContext() = default;
  BucketContext(double3 minCoords_,
                double3 maxCoords_,
                double interval_,
                int3 nBuckets_)
      : minCoords(minCoords_),
        maxCoords(maxCoords_),
        interval(interval_),
        nBuckets(nBuckets_){};

  double3 minCoords;
  double3 maxCoords;
  double interval;
  int3 nBuckets;
};

class ParticleModelCUDA {
 private:
  static inline const int MAX_NUM_BUCKETS_PER_DIM = 256;

  ParticleCUDA* _deviceParticles;
  ParticleCUDA* _deviceParticlesBuffer;

  int64_t _nParticles;
  std::set<double> _radiuses;

  int64_t* _deviceBucket;
  uint* _deviceBucketCounter;
  uint* _deviceBucketCounterBuffer;
  uint* _deviceBucketCumsumCounter;

  double _coefficientOfSpring;
  double _coefficientOfRestitution;

  BucketContext _bucketContext;

#ifndef USE_NATIVE_PREFIX_SCAN
  std::shared_ptr<PrefixScan> _prefixScanner;
#endif

  void updateBuckets();

  void copyParticleData(const ParticleCUDA* particlesFrom, ParticleCUDA* particlesTo);

 public:
  ParticleModelCUDA(const std::vector<ParticlePtr>& particles,
                    const double coefficientOfSpring,
                    const double coefficientOfRestitution);

  ~ParticleModelCUDA();

  void initBuckets(const double3 boxMinCoords,
                   const double3 boxMaxCoords);

  void resolveCollisions();

  int64_t getNumParticles() const;

  ParticleCUDA* getParticles() const;

  ParticleCUDA* getDeviceParticles() const;
};

#ifdef __cplusplus
}
#endif
