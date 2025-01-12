#pragma once

#include <Util/Math.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <vector>

class ParticleBucket {
 public:
  template <class DType>
  using Vector_t = std::shared_ptr<std::vector<DType>>;
  using Bucket_t = Vector_t<Vector_t<Vector_t<Vector_t<int64_t>>>>;

 private:
  inline static const int64_t INVALID_ID = -1;
  inline static const double COEFF_FOR_INTERVAL_OF_BUCKETS = 1.05f;

  double _interval;
  glm::dvec3 _minCoords;
  glm::dvec3 _maxCoords;
  int64_t _nBucketsX;
  int64_t _nBucketsY;
  int64_t _nBucketsZ;

  Bucket_t _bucket;

 public:
  ParticleBucket();
  ~ParticleBucket();

  void initBucket(const double &maxRadius,
                  const glm::dvec3 &minCoords,
                  const glm::dvec3 &maxCoords);
  void resetBuckets();
  void clearBuckets();

  int64_t toBucketIndexX(const double &) const;
  int64_t toBucketIndexY(const double &) const;
  int64_t toBucketIndexZ(const double &) const;

  Vector_t<int64_t> getPaticleIdsInBucket(const int64_t &,
                                          const int64_t &,
                                          const int64_t &) const;
  Vector_t<int64_t> getPaticleIdsInBucket(const double &,
                                          const double &,
                                          const double &) const;
  int64_t getNumBucketsX() const;
  int64_t getNumBucketsY() const;
  int64_t getNumBucketsZ() const;

  bool isInBucket(const int64_t &,
                  const int64_t &,
                  const int64_t &,
                  const int64_t &) const;
  void addParticleToBucket(const double &,
                           const double &,
                           const double &,
                           const int64_t &);
};
