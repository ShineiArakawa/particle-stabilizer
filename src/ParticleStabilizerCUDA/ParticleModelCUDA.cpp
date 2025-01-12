#include <ParticleStabilizerCUDA/ParticleModelCUDA.hpp>
#include <ParticleStabilizerCUDA/ParticleModel_kernels.cuh>

ParticleModelCUDA::ParticleModelCUDA(const std::vector<ParticlePtr>& particles,
                                     const double coefficientOfSpring,
                                     const double coefficientOfRestitution)
    : _deviceParticles(),
      _deviceParticlesBuffer(),
      _nParticles(particles.size()),
      _radiuses(),
      _deviceBucket(),
      _deviceBucketCounter(),
      _deviceBucketCounterBuffer(),
      _deviceBucketCumsumCounter(),
      _coefficientOfSpring(coefficientOfSpring),
      _coefficientOfRestitution(coefficientOfRestitution) {
  // =====================================================================================================
  // Allocate memory and set data for particles
  // =====================================================================================================
  std::vector<ParticleCUDA> particlesCUDA;
  particlesCUDA.resize(_nParticles);

  for (int64_t iParticle = 0; iParticle < _nParticles; ++iParticle) {
    const auto& particle = particles[iParticle];

    particlesCUDA[iParticle] = ParticleCUDA{make_double3(particle->position.x,
                                                         particle->position.y,
                                                         particle->position.z),
                                            make_double3(particle->velocity.x,
                                                         particle->velocity.y,
                                                         particle->velocity.z),
                                            particle->radius,
                                            particle->mass};
  }

  {
    // Particles
    CUDA_CHECK_ERROR(cudaMalloc(&_deviceParticles, sizeof(ParticleCUDA) * _nParticles));
    CUDA_CHECK_ERROR(cudaMemcpy(_deviceParticles, particlesCUDA.data(), sizeof(ParticleCUDA) * _nParticles, cudaMemcpyHostToDevice));

    // Buffers for calculation
    CUDA_CHECK_ERROR(cudaMalloc(&_deviceParticlesBuffer, sizeof(ParticleCUDA) * _nParticles));
  }

  // =====================================================================================================
  // Set radius types
  // =====================================================================================================
  for (int64_t iParticle = 0; iParticle < _nParticles; ++iParticle) {
    const double radius = particles[iParticle]->radius;
    _radiuses.emplace(radius);
  }
}

ParticleModelCUDA::~ParticleModelCUDA() {
  CUDA_CHECK_ERROR(cudaFree(_deviceParticles));
  CUDA_CHECK_ERROR(cudaFree(_deviceParticlesBuffer));
  CUDA_CHECK_ERROR(cudaFree(_deviceBucket));
  CUDA_CHECK_ERROR(cudaFree(_deviceBucketCounter));
  CUDA_CHECK_ERROR(cudaFree(_deviceBucketCounterBuffer));
}

void ParticleModelCUDA::initBuckets(const double3 boxMinCoords, const double3 boxMaxCoords) {
  // =====================================================================================================
  // Initialize ParticleBucket
  // =====================================================================================================
  double maxRadius = 0.0;

  for (const auto& radius : _radiuses) {
    maxRadius = std::max(maxRadius, radius);
  }

  const double3 maxInterval = {(boxMaxCoords.x - boxMinCoords.x) / static_cast<double>(MAX_NUM_BUCKETS_PER_DIM),
                               (boxMaxCoords.y - boxMinCoords.y) / static_cast<double>(MAX_NUM_BUCKETS_PER_DIM),
                               (boxMaxCoords.z - boxMinCoords.z) / static_cast<double>(MAX_NUM_BUCKETS_PER_DIM)};

  float interval = maxRadius * 2.0f * 1.05f;
  if (interval < maxInterval.x ||
      interval < maxInterval.y ||
      interval < maxInterval.z) {
    LOG_WARN("Maximum bucket interval exceeded! Clipped the interval.");
    interval = std::max(maxInterval.x, std::max(maxInterval.y, maxInterval.z));
  }

  const double bucketInterval = maxRadius * 2.0 * 1.05;

  _bucketContext = BucketContext(boxMinCoords,
                                 boxMaxCoords,
                                 bucketInterval,
                                 make_int3((int)((boxMaxCoords.x - boxMinCoords.x) / bucketInterval) + 1,
                                           (int)((boxMaxCoords.y - boxMinCoords.y) / bucketInterval) + 1,
                                           (int)((boxMaxCoords.z - boxMinCoords.z) / bucketInterval) + 1));

  LOG_INFO("################################################ Bucket info #################################################");
  LOG_INFO("# (minX, minY, minZ)                : (" + std::to_string(_bucketContext.minCoords.x) + ", " + std::to_string(_bucketContext.minCoords.y) + ", " + std::to_string(_bucketContext.minCoords.z) + ")");
  LOG_INFO("# (maxX, maxY, maxZ)                : (" + std::to_string(_bucketContext.maxCoords.x) + ", " + std::to_string(_bucketContext.maxCoords.y) + ", " + std::to_string(_bucketContext.maxCoords.z) + ")");
  LOG_INFO("# (nBucketsX, nBucketsY, nBucketsZ) : (" + std::to_string(_bucketContext.nBuckets.x) + ", " + std::to_string(_bucketContext.nBuckets.y) + ", " + std::to_string(_bucketContext.nBuckets.z) + ")");
  LOG_INFO("# interval                          : " + std::to_string(_bucketContext.interval));
  LOG_INFO("##############################################################################################################");

  const int64_t nTotalBuckets = _bucketContext.nBuckets.x * _bucketContext.nBuckets.y * _bucketContext.nBuckets.z;

  CUDA_CHECK_ERROR(cudaMalloc(&_deviceBucket, sizeof(int64_t) * _nParticles));
  CUDA_CHECK_ERROR(cudaMalloc(&_deviceBucketCounter, sizeof(uint) * nTotalBuckets));
  CUDA_CHECK_ERROR(cudaMalloc(&_deviceBucketCounterBuffer, sizeof(uint) * nTotalBuckets));
  CUDA_CHECK_ERROR(cudaMalloc(&_deviceBucketCumsumCounter, sizeof(uint) * nTotalBuckets));

  // =====================================================================================================
  // Fast prefix scan implementation
  // =====================================================================================================
#ifndef USE_NATIVE_PREFIX_SCAN
  _prefixScanner = std::make_shared<PrefixScan>(nTotalBuckets);
#endif
}

void ParticleModelCUDA::updateBuckets() {
  const int64_t nTotalBuckets = _bucketContext.nBuckets.x * _bucketContext.nBuckets.y * _bucketContext.nBuckets.z;

  {
    // Count particles in each bucket
    CUDA_CHECK_ERROR(cudaMemset(_deviceBucketCounter, 0, sizeof(uint) * nTotalBuckets));

    launchCountParticlesInEachBucketKernel(_deviceParticles,
                                           _nParticles,
                                           _bucketContext,
                                           _deviceBucketCounter);
  }

  {
    // Cumsum
    CUDA_CHECK_ERROR(cudaMemset(_deviceBucketCumsumCounter, 0, sizeof(uint) * nTotalBuckets));
#ifdef USE_NATIVE_PREFIX_SCAN
    launchPrefixSumKernel(_deviceBucketCounter,
                          nTotalBuckets,
                          _deviceBucketCumsumCounter);
#else
    _prefixScanner->sum_scan_blelloch(_deviceBucketCumsumCounter, _deviceBucketCounter);
#endif
  }

  {
    // Fill buckets
    CUDA_CHECK_ERROR(cudaMemset(_deviceBucketCounterBuffer, 0, sizeof(uint) * nTotalBuckets));
    CUDA_CHECK_ERROR(cudaMemset(_deviceBucket, -1, sizeof(int64_t) * _nParticles));

    launchRegisterToBucketKernel(_deviceParticles,
                                 _nParticles,
                                 _bucketContext,
                                 _deviceBucketCounter,
                                 _deviceBucketCumsumCounter,
                                 _deviceBucketCounterBuffer,
                                 _deviceBucket);
  }

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

void ParticleModelCUDA::resolveCollisions() {
  // Update buckets
  updateBuckets();

  // Copy the particle data to the buffer
  // From now on, _deviceParticles is read-only
  // _deviceParticlesBuffer is write-only
  copyParticleData(_deviceParticles, _deviceParticlesBuffer);

  launchResolveCollisionsKernel(_deviceParticles,
                                _nParticles,
                                _bucketContext,
                                _deviceBucketCounter,
                                _deviceBucketCumsumCounter,
                                _deviceBucket,
                                _coefficientOfSpring,
                                _coefficientOfRestitution,
                                _deviceParticlesBuffer);

  copyParticleData(_deviceParticlesBuffer, _deviceParticles);
}

int64_t ParticleModelCUDA::getNumParticles() const {
  return _nParticles;
}

ParticleCUDA* ParticleModelCUDA::getParticles() const {
  ParticleCUDA* hostParticles = new ParticleCUDA[_nParticles];

  CUDA_CHECK_ERROR(cudaMemcpy(hostParticles, _deviceParticles, sizeof(ParticleCUDA) * _nParticles, cudaMemcpyDeviceToHost));

  return hostParticles;
}

ParticleCUDA* ParticleModelCUDA::getDeviceParticles() const {
  return _deviceParticles;
}

void ParticleModelCUDA::copyParticleData(const ParticleCUDA* particlesFrom, ParticleCUDA* particlesTo) {
  CUDA_CHECK_ERROR(cudaMemcpy(particlesTo, particlesFrom, sizeof(ParticleCUDA) * _nParticles, cudaMemcpyDeviceToDevice));
}
