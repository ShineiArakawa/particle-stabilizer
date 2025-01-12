#include <ParticleStabilizerCUDA/PolygonContainerCUDA_kernels.cuh>

// ==================================================================================================================
// Kernels
// ==================================================================================================================
static __device__ __forceinline__ double distance(const double3 vector0, const double3 vector1) {
  const double relVecX = vector0.x - vector1.x;
  const double relVecY = vector0.y - vector1.y;
  const double relVecZ = vector0.z - vector1.z;
  return sqrt(relVecX * relVecX + relVecY * relVecY + relVecZ * relVecZ);
}

static __device__ __forceinline__ double dot(const double3 vector0, const double3 vector1) {
  return vector0.x * vector1.x +
         vector0.y * vector1.y +
         vector0.z * vector1.z;
}

static __device__ __forceinline__ double3 operator+(const double3 vector0, const double3 vector1) {
  return make_double3(vector0.x + vector1.x,
                      vector0.y + vector1.y,
                      vector0.z + vector1.z);
}

static __device__ __forceinline__ double3 operator+(const double3 vector, const double scalar) {
  return make_double3(vector.x + scalar,
                      vector.y + scalar,
                      vector.z + scalar);
}

static __device__ __forceinline__ double3 operator+(const double scalar, const double3 vector) {
  return vector + scalar;
}

static __device__ __forceinline__ double3 operator-(const double3 vector0, const double3 vector1) {
  return make_double3(vector0.x - vector1.x,
                      vector0.y - vector1.y,
                      vector0.z - vector1.z);
}

static __device__ __forceinline__ double3 operator-(const double3 vector, const double scalar) {
  return make_double3(vector.x - scalar,
                      vector.y - scalar,
                      vector.z - scalar);
}

static __device__ __forceinline__ double3 operator-(const double scalar, const double3 vector) {
  return make_double3(scalar - vector.x,
                      scalar - vector.y,
                      scalar - vector.z);
}

static __device__ __forceinline__ double3 operator*(const double3 vector, const double scalar) {
  return make_double3(vector.x * scalar,
                      vector.y * scalar,
                      vector.z * scalar);
}

static __device__ __forceinline__ double3 operator*(const double scalar, const double3 vector) {
  return vector * scalar;
}

static __device__ __forceinline__ double3 operator/(const double3 vector, const double scalar) {
  return make_double3(vector.x / scalar,
                      vector.y / scalar,
                      vector.z / scalar);
}

static __device__ __forceinline__ double3 projectPointOntoPlaneKernelFunc(const double3 point,      /* OP */
                                                                          const double3 planePoint, /* OA */
                                                                          const double3 planeNormal /* N  */) {
  // AP
  const double3 v = point - planePoint;

  const double innerProduct = dot(v, planeNormal);

  return point - planeNormal * innerProduct;
}

static __device__ __forceinline__ bool isPointOnPolygonKernelFunc(const double3 p,
                                                                  const double3 a,
                                                                  const double3 b,
                                                                  const double3 c) {
  const double3 v0 = c - a;
  const double3 v1 = b - a;
  const double3 v2 = p - a;

  const double dot00 = dot(v0, v0);
  const double dot01 = dot(v0, v1);
  const double dot02 = dot(v0, v2);
  const double dot11 = dot(v1, v1);
  const double dot12 = dot(v1, v2);

  const double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
  const double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  const double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

  return (u >= 0.0) && (v >= 0.0) && (u + v <= 1.0);
}

static __device__ __forceinline__ double3 closestPointOnLineSegmentKernelFunc(const double3& p,
                                                                              const double3& a,
                                                                              const double3& b) {
  const double3 ab = b - a;
  const double3 ap = p - a;

  double t = dot(ap, ab) / dot(ab, ab);

  t = max(0.0, min(1.0, t));

  return a + ab * t;
}

static __device__ __forceinline__ double3 getClosestPointOnPolygonKernelFunc(const double3 vertex0,  // A
                                                                             const double3 vertex1,  // B
                                                                             const double3 vertex2,  // C
                                                                             const double3 normal,
                                                                             const double3 center) {
  // Project the point onto the plane of the triangle
  double3 projection = projectPointOntoPlaneKernelFunc(center, vertex0, normal);

  // Check if the projection is inside the triangle
  if (isPointOnPolygonKernelFunc(projection, vertex0, vertex1, vertex2)) {
    return projection;
  }

  // Find the closest point on each edge
  // Edge AB
  const double3 closestPointAB = closestPointOnLineSegmentKernelFunc(center, vertex0, vertex1);
  double minDistSquared = dot(center, closestPointAB);

  projection = closestPointAB;

  // Edge BC
  const double3 closestPointBC = closestPointOnLineSegmentKernelFunc(center, vertex1, vertex2);
  const double distBCSquared = dot(center, closestPointBC);

  if (distBCSquared < minDistSquared) {
    minDistSquared = distBCSquared;
    projection = closestPointBC;
  }

  // Edge CA
  const double3 closestPointCA = closestPointOnLineSegmentKernelFunc(center, vertex2, vertex0);
  const double distCASquared = dot(center, closestPointCA);

  if (distCASquared < minDistSquared) {
    minDistSquared = distCASquared;
    projection = closestPointCA;
  }

  return projection;
}

__global__ void initBuffersKernel(int* intersectCounter,
                                  double3* posModification,
                                  double3* velocityBuffer,
                                  const int64_t nParticles) {
  const int64_t iParticle = static_cast<int64_t>(threadIdx.x + blockIdx.x * blockDim.x);

  if (iParticle < nParticles) {
    intersectCounter[iParticle] = 0L;
    posModification[iParticle].x = 0.0;
    posModification[iParticle].y = 0.0;
    posModification[iParticle].z = 0.0;
    velocityBuffer[iParticle].x = 0.0;
    velocityBuffer[iParticle].y = 0.0;
    velocityBuffer[iParticle].z = 0.0;
  }
}

__global__ void isIntersectedKernel(const ParticleCUDA* particles,
                                    const int64_t nParticles,
                                    const double3* polygonCoords,
                                    const double3* polygonNormals,
                                    const double3* polygonMinCoords,
                                    const double3* polygonMaxCoords,
                                    const int64_t nPolygons,
                                    bool* isIntersected,
                                    int* intersectCounter) {
  const int64_t iParticle = static_cast<int64_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const int64_t iPolygon = static_cast<int64_t>(threadIdx.y + blockIdx.y * blockDim.y);

  if (iParticle < nParticles && iPolygon < nPolygons) {
    const int64_t offsetPolygon = 3L * iPolygon;
    bool tmp_isIntersected = false;

    const double3 minCoords = particles[iParticle].position - particles[iParticle].radius;
    const double3 maxCoords = particles[iParticle].position + particles[iParticle].radius;

    // Axis-aligned Bounding Box test
    if (minCoords.x <= polygonMaxCoords[iPolygon].x &&
        maxCoords.x >= polygonMinCoords[iPolygon].x &&
        minCoords.y <= polygonMaxCoords[iPolygon].y &&
        maxCoords.y >= polygonMinCoords[iPolygon].y &&
        minCoords.z <= polygonMaxCoords[iPolygon].z &&
        maxCoords.z >= polygonMinCoords[iPolygon].z) {
      // Detailed test
      const double3 closestPoint = getClosestPointOnPolygonKernelFunc(polygonCoords[offsetPolygon + 0L],
                                                                      polygonCoords[offsetPolygon + 1L],
                                                                      polygonCoords[offsetPolygon + 2L],
                                                                      polygonNormals[iPolygon],
                                                                      particles[iParticle].position);

      const double3 relVec = closestPoint - particles[iParticle].position;
      const double distSquared = dot(relVec, relVec);

      tmp_isIntersected = distSquared < particles[iParticle].radius * particles[iParticle].radius;
    }

    const int64_t index = iParticle * nPolygons + iPolygon;
    isIntersected[index] = tmp_isIntersected;

    if (tmp_isIntersected) {
      atomicAdd(&intersectCounter[iParticle], 1);
    }
  }
}

__global__ void calcModifiedVelocitySumKernel(const ParticleCUDA* particles,
                                              const int64_t nParticles,
                                              const double3* polygonNormals,
                                              const int64_t nPolygons,
                                              const bool* isIntersected,
                                              const double coefficientOfRestitution,
                                              double3* velocityBuffer) {
  const int64_t iParticle = static_cast<int64_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const int64_t iPolygon = static_cast<int64_t>(threadIdx.y + blockIdx.y * blockDim.y);

  if (iParticle < nParticles && iPolygon < nPolygons) {
    const int64_t index = iParticle * nPolygons + iPolygon;
    const bool tmp_isIntersected = isIntersected[index];

    if (tmp_isIntersected) {
      const double3 reflectedVelocity = particles[iParticle].velocity - 2.0 * dot(particles[iParticle].velocity, polygonNormals[iPolygon]) * polygonNormals[iPolygon];

      atomicAdd(&(velocityBuffer[iParticle].x), coefficientOfRestitution * reflectedVelocity.x);
      atomicAdd(&(velocityBuffer[iParticle].y), coefficientOfRestitution * reflectedVelocity.y);
      atomicAdd(&(velocityBuffer[iParticle].z), coefficientOfRestitution * reflectedVelocity.z);
    }
  }
}

__global__ void calcAveragedVelocityKernel(const int64_t nParticles,
                                           const double3* velocityBuffer,
                                           const int* intersectCounter,
                                           ParticleCUDA* particles) {
  const int64_t iParticle = static_cast<int64_t>(threadIdx.x + blockIdx.x * blockDim.x);

  if (iParticle < nParticles) {
    const bool tmp_isIntersectedAtLeast = intersectCounter[iParticle] > 0;

    if (tmp_isIntersectedAtLeast) {
      const double intersectionCount = static_cast<double>(intersectCounter[iParticle]);

      particles[iParticle].velocity = velocityBuffer[iParticle] / intersectionCount;
    }
  }
}

__global__ void calcPosModificationAmountKernel(const ParticleCUDA* particles,
                                                const int64_t nParticles,
                                                const double3* polygonCoords,
                                                const double3* polygonNormals,
                                                const int64_t nPolygons,
                                                const bool* isIntersected,
                                                double3* modification) {
  const int64_t iParticle = static_cast<int64_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const int64_t iPolygon = static_cast<int64_t>(threadIdx.y + blockIdx.y * blockDim.y);

  if (iParticle < nParticles && iPolygon < nPolygons) {
    const int64_t index = iParticle * nPolygons + iPolygon;
    const bool tmp_isIntersected = isIntersected[index];

    if (tmp_isIntersected) {
      const double distanceToPlane = dot(-1.0 * polygonNormals[iPolygon], particles[iParticle].position - polygonCoords[3L * iPolygon]);
      const double scale = particles[iParticle].radius - distanceToPlane;

      atomicAdd(&(modification[iParticle].x), -scale * polygonNormals[iPolygon].x);
      atomicAdd(&(modification[iParticle].y), -scale * polygonNormals[iPolygon].y);
      atomicAdd(&(modification[iParticle].z), -scale * polygonNormals[iPolygon].z);
    }
  }
}

__global__ void modifyPositionKernel(const int64_t nParticles,
                                     const double3* modification,
                                     ParticleCUDA* particles) {
  const int64_t iParticle = static_cast<int64_t>(threadIdx.x + blockIdx.x * blockDim.x);

  if (iParticle < nParticles) {
    particles[iParticle].position = particles[iParticle].position + modification[iParticle];
  }
}

// ==================================================================================================================
// Dispacther
// ==================================================================================================================

void launchInitBuffersKernel(int* intersectCounter,
                             double3* posModification,
                             double3* velocityBuffer,
                             const int64_t nParticles) {
  const uint nThreadsPerBlock = BLOCK_DIM;
  const uint nBlocksPerGrid = divRoundUp(nParticles, nThreadsPerBlock);

  initBuffersKernel<<<nBlocksPerGrid, nThreadsPerBlock>>>(intersectCounter,
                                                          posModification,
                                                          velocityBuffer,
                                                          nParticles);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

void launchIsIntersectedKernel(const ParticleCUDA* particles,
                               const int64_t nParticles,
                               const double3* polygonCoords,
                               const double3* polygonNormals,
                               const double3* polygonMinCoords,
                               const double3* polygonMaxCoords,
                               const int64_t nPolygons,
                               bool* isIntersected,
                               int* intersectCounter) {
  const dim3 nThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  const dim3 nBlocksPerGrid(divRoundUp(nParticles, nThreadsPerBlock.x),
                            divRoundUp(nPolygons, nThreadsPerBlock.y));

  isIntersectedKernel<<<nBlocksPerGrid, nThreadsPerBlock>>>(particles,
                                                            nParticles,
                                                            polygonCoords,
                                                            polygonNormals,
                                                            polygonMinCoords,
                                                            polygonMaxCoords,
                                                            nPolygons,
                                                            isIntersected,
                                                            intersectCounter);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

void launchCalcModifiedVelocitySumKernel(const ParticleCUDA* particles,
                                         const int64_t nParticles,
                                         const double3* polygonNormals,
                                         const int64_t nPolygons,
                                         const bool* isIntersected,
                                         const double coefficientOfRestitution,
                                         double3* velocityBuffer) {
  const dim3 nThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  const dim3 nBlocksPerGrid(divRoundUp(nParticles, nThreadsPerBlock.x),
                            divRoundUp(nPolygons, nThreadsPerBlock.y));

  calcModifiedVelocitySumKernel<<<nBlocksPerGrid, nThreadsPerBlock>>>(particles,
                                                                      nParticles,
                                                                      polygonNormals,
                                                                      nPolygons,
                                                                      isIntersected,
                                                                      coefficientOfRestitution,
                                                                      velocityBuffer);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

void launchCalcAveragedVelocityKernel(const int64_t nParticles,
                                      const double3* velocityBuffer,
                                      const int* intersectCounter,
                                      ParticleCUDA* particles) {
  const uint nThreadsPerBlock = BLOCK_DIM;
  const uint nBlocksPerGrid = divRoundUp(nParticles, nThreadsPerBlock);

  calcAveragedVelocityKernel<<<nBlocksPerGrid, nThreadsPerBlock>>>(nParticles,
                                                                   velocityBuffer,
                                                                   intersectCounter,
                                                                   particles);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

void launchCalcPositionModificationKernel(const ParticleCUDA* particles,
                                          const int64_t nParticles,
                                          const double3* polygonCoords,
                                          const double3* polygonNormals,
                                          const int64_t nPolygons,
                                          const bool* isIntersected,
                                          double3* modification) {
  const dim3 nThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  const dim3 nBlocksPerGrid(divRoundUp(nParticles, nThreadsPerBlock.x),
                            divRoundUp(nPolygons, nThreadsPerBlock.y));

  calcPosModificationAmountKernel<<<nBlocksPerGrid, nThreadsPerBlock>>>(particles,
                                                                        nParticles,
                                                                        polygonCoords,
                                                                        polygonNormals,
                                                                        nPolygons,
                                                                        isIntersected,
                                                                        modification);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

void launchModifyPositionKernel(const int64_t nParticles,
                                const double3* modification,
                                ParticleCUDA* particles) {
  const uint nThreadsPerBlock = BLOCK_DIM;
  const uint nBlocksPerGrid = divRoundUp(nParticles, nThreadsPerBlock);

  modifyPositionKernel<<<nBlocksPerGrid, nThreadsPerBlock>>>(nParticles,
                                                             modification,
                                                             particles);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}
