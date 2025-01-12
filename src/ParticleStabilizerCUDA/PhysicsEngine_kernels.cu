#include <ParticleStabilizerCUDA/PhysicsEngine_kernels.cuh>

__global__ void updatePositionAndVelocityKernel(ParticleCUDA* particles,
                                                const int64_t nParticles,
                                                const double3 gravity,
                                                const double dt) {
  const uint iParticle = blockIdx.x * blockDim.x + threadIdx.x;

  if (iParticle < nParticles) {
    particles[iParticle].position.x += particles[iParticle].velocity.x * dt;
    particles[iParticle].position.y += particles[iParticle].velocity.y * dt;
    particles[iParticle].position.z += particles[iParticle].velocity.z * dt;

    particles[iParticle].velocity.x += gravity.x * dt;
    particles[iParticle].velocity.y += gravity.y * dt;
    particles[iParticle].velocity.z += gravity.z * dt;
  }
}

void lanchUpdatePositionAndVelocityKernel(ParticleCUDA* particles,
                                          const int64_t nParticles,
                                          const double3 gravity,
                                          const double dt) {
  const uint nThreadsPerBlock = BLOCK_DIM;
  const uint nBlocksPerGrid = divRoundUp(nParticles, nThreadsPerBlock);

  updatePositionAndVelocityKernel<<<nBlocksPerGrid, nThreadsPerBlock>>>(particles,
                                                                        nParticles,
                                                                        gravity,
                                                                        dt);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}