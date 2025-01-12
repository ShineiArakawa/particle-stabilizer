#include <ParticleStabilizer/ParticleBucket.hpp>
#include <Util/Logging.hpp>

ParticleBucket::ParticleBucket()
    : _interval(0.0),
      _minCoords({0.0, 0.0, 0.0}),
      _maxCoords({0.0, 0.0, 0.0}),
      _nBucketsX(0),
      _nBucketsY(0),
      _nBucketsZ(0),
      _bucket(nullptr) {
}

ParticleBucket::~ParticleBucket() {
}

void ParticleBucket::initBucket(const double &maxRadius,
                                const glm::dvec3 &minCoords,
                                const glm::dvec3 &maxCoords) {
  _interval = maxRadius * 2.0 * COEFF_FOR_INTERVAL_OF_BUCKETS;

  _minCoords = minCoords;
  _maxCoords = maxCoords;

  _nBucketsX = (int64_t)((_maxCoords.x - _minCoords.x) / _interval) + 1;
  _nBucketsY = (int64_t)((_maxCoords.y - _minCoords.y) / _interval) + 1;
  _nBucketsZ = (int64_t)((_maxCoords.z - _minCoords.z) / _interval) + 1;

  LOG_INFO("############################ Bucket Info ############################");
  LOG_INFO("maxRadius = " + std::to_string(maxRadius));
  LOG_INFO("interval  = " + std::to_string(_interval));
  LOG_INFO("minCoords = (" + std::to_string(_minCoords[0]) + ", " + std::to_string(_minCoords[1]) + ", " + std::to_string(_minCoords[2]) + ")");
  LOG_INFO("maxCoords = (" + std::to_string(_maxCoords[0]) + ", " + std::to_string(_maxCoords[1]) + ", " + std::to_string(_maxCoords[2]) + ")");
  LOG_INFO("nBuckets  = (" + std::to_string(_nBucketsX) + ", " + std::to_string(_nBucketsY) + ", " + std::to_string(_nBucketsZ) + ")");
  LOG_INFO("#####################################################################");

  resetBuckets();
}

void ParticleBucket::resetBuckets() {
  _bucket = std::make_shared<std::vector<Vector_t<Vector_t<Vector_t<int64_t>>>>>();

  for (int64_t x = 0; x < _nBucketsX; ++x) {
    _bucket->push_back(std::make_shared<std::vector<Vector_t<Vector_t<int64_t>>>>());

    for (int64_t y = 0; y < _nBucketsY; ++y) {
      (*_bucket)[x]->push_back(std::make_shared<std::vector<Vector_t<int64_t>>>());

      for (int64_t z = 0; z < _nBucketsZ; ++z) {
        (*(*_bucket)[x])[y]->push_back(std::make_shared<std::vector<int64_t>>());
      }
    }
  }
}

void ParticleBucket::clearBuckets() {
  for (int64_t x = 0; x < _nBucketsX; ++x) {
    for (int64_t y = 0; y < _nBucketsY; ++y) {
      for (int64_t z = 0; z < _nBucketsZ; ++z) {
        (*(*(*_bucket)[x])[y])[z]->clear();
      }
    }
  }
}

int64_t ParticleBucket::toBucketIndexX(const double &x) const {
  int64_t index = (int64_t)((x - _minCoords[0]) / _interval);

  if (index < 0) {
    index = 0;
  }

  if (index >= _nBucketsX) {
    index = _nBucketsX - 1;
  }

  return index;
}

int64_t ParticleBucket::toBucketIndexY(const double &y) const {
  int64_t index = (int64_t)((y - _minCoords[1]) / _interval);

  if (index < 0) {
    index = 0;
  }

  if (index >= _nBucketsY) {
    index = _nBucketsY - 1;
  }

  return index;
}

int64_t ParticleBucket::toBucketIndexZ(const double &z) const {
  int64_t index = (int64_t)((z - _minCoords[2]) / _interval);

  if (index < 0) {
    index = 0;
  }

  if (index >= _nBucketsZ) {
    index = _nBucketsZ - 1;
  }

  return index;
}

ParticleBucket::Vector_t<int64_t> ParticleBucket::getPaticleIdsInBucket(const int64_t &x,
                                                                        const int64_t &y,
                                                                        const int64_t &z) const {
  Vector_t<int64_t> paticleIds = nullptr;

  if (x >= 0 && x < _nBucketsX && y >= 0 && y < _nBucketsY && z >= 0 && z < _nBucketsZ) {
    paticleIds = (*(*(*_bucket)[x])[y])[z];
  }

  return paticleIds;
}

ParticleBucket::Vector_t<int64_t> ParticleBucket::getPaticleIdsInBucket(const double &x,
                                                                        const double &y,
                                                                        const double &z) const {
  const int64_t bucketIndexX = toBucketIndexX(x);
  const int64_t bucketIndexY = toBucketIndexY(y);
  const int64_t bucketIndexZ = toBucketIndexZ(z);

  return getPaticleIdsInBucket(bucketIndexX, bucketIndexY, bucketIndexZ);
}

int64_t ParticleBucket::getNumBucketsX() const {
  return _nBucketsX;
};

int64_t ParticleBucket::getNumBucketsY() const {
  return _nBucketsY;
};

int64_t ParticleBucket::getNumBucketsZ() const {
  return _nBucketsZ;
};

bool ParticleBucket::isInBucket(const int64_t &x,
                                const int64_t &y,
                                const int64_t &z,
                                const int64_t &index) const {
  return std::find((*(*(*_bucket)[x])[y])[z]->begin(),
                   (*(*(*_bucket)[x])[y])[z]->end(),
                   index) != (*(*(*_bucket)[x])[y])[z]->end();
}

void ParticleBucket::addParticleToBucket(const double &x,
                                         const double &y,
                                         const double &z,
                                         const int64_t &index) {
  const int64_t bucketIndexX = toBucketIndexX(x);
  const int64_t bucketIndexY = toBucketIndexY(y);
  const int64_t bucketIndexZ = toBucketIndexZ(z);

  (*(*(*_bucket)[bucketIndexX])[bucketIndexY])[bucketIndexZ]->push_back(index);
}
