#include <ParticleStabilizerCUDA/BoxContainer_kernels.cuh>

__global__ void resolveWallCollisionKernel(ParticleCUDA* particles,
                                           const int64_t nParticles,
                                           const double3 min,
                                           const double3 max,
                                           const CoefficientOfRestitutionWall coefficientOfRestitutionWall) {
  __shared__ ParticleCUDA sharedParticles[BLOCK_DIM];

  const uint iParticle = blockIdx.x * blockDim.x + threadIdx.x;
  const uint iThread = threadIdx.x;

  if (iParticle < nParticles) {
    sharedParticles[iThread] = particles[iParticle];
    // NOTE: No need to synchronize

    const double3 minCoords = make_double3(sharedParticles[iThread].position.x - sharedParticles[iThread].radius,
                                           sharedParticles[iThread].position.y - sharedParticles[iThread].radius,
                                           sharedParticles[iThread].position.z - sharedParticles[iThread].radius);
    const double3 maxCoords = make_double3(sharedParticles[iThread].position.x + sharedParticles[iThread].radius,
                                           sharedParticles[iThread].position.y + sharedParticles[iThread].radius,
                                           sharedParticles[iThread].position.z + sharedParticles[iThread].radius);

    if (minCoords.x < min.x) {
      sharedParticles[iThread].position.x = min.x + sharedParticles[iThread].radius;
      sharedParticles[iThread].velocity.x = -sharedParticles[iThread].velocity.x * coefficientOfRestitutionWall.xMin;
    }

    if (minCoords.y < min.y) {
      sharedParticles[iThread].position.y = min.y + sharedParticles[iThread].radius;
      sharedParticles[iThread].velocity.y = -sharedParticles[iThread].velocity.y * coefficientOfRestitutionWall.yMin;
    }

    if (minCoords.z < min.z) {
      sharedParticles[iThread].position.z = min.z + sharedParticles[iThread].radius;
      sharedParticles[iThread].velocity.z = -sharedParticles[iThread].velocity.z * coefficientOfRestitutionWall.zMin;
    }

    if (maxCoords.x > max.x) {
      sharedParticles[iThread].position.x = max.x - sharedParticles[iThread].radius;
      sharedParticles[iThread].velocity.x = -sharedParticles[iThread].velocity.x * coefficientOfRestitutionWall.xMax;
    }

    if (maxCoords.y > max.y) {
      sharedParticles[iThread].position.y = max.y - sharedParticles[iThread].radius;
      sharedParticles[iThread].velocity.y = -sharedParticles[iThread].velocity.y * coefficientOfRestitutionWall.yMax;
    }

    if (maxCoords.z > max.z) {
      sharedParticles[iThread].position.z = max.z - sharedParticles[iThread].radius;
      sharedParticles[iThread].velocity.z = -sharedParticles[iThread].velocity.z * coefficientOfRestitutionWall.zMax;
    }

    particles[iParticle] = sharedParticles[iThread];
  }
}

void launchResolveWallCollisionKernel(ParticleCUDA* particles,
                                      const int64_t nParticles,
                                      const double3 min,
                                      const double3 max,
                                      const CoefficientOfRestitutionWall coefficientOfRestitutionWall) {
  const uint nThreadsPerBlock = BLOCK_DIM;
  const uint nBlocksPerGrid = divRoundUp(nParticles, nThreadsPerBlock);

  resolveWallCollisionKernel<<<nBlocksPerGrid, nThreadsPerBlock>>>(particles,
                                                                   nParticles,
                                                                   min,
                                                                   max,
                                                                   coefficientOfRestitutionWall);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}