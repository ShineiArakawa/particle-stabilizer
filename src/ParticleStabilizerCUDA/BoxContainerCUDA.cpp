#include <ParticleStabilizerCUDA/BoxContainerCUDA.hpp>
#include <ParticleStabilizerCUDA/BoxContainer_kernels.cuh>

BoxContainerCUDA::BoxContainerCUDA(const double3 min,
                                   const double3 max,
                                   const CoefficientOfRestitutionWall coefficientOfRestitution,
                                   const int64_t nParticles)
    : _min(min),
      _max(max),
      _coefficientOfRestitution(coefficientOfRestitution),
      _nParticles(nParticles) {
}

void BoxContainerCUDA::resolveCollisions(ParticleCUDA* particle) {
  launchResolveWallCollisionKernel(particle,
                                   _nParticles,
                                   _min,
                                   _max,
                                   _coefficientOfRestitution);
}

double3 BoxContainerCUDA::getMinCoords() {
  return _min;
}

double3 BoxContainerCUDA::getMaxCoords() {
  return _max;
}

simview::model::Primitive_t BoxContainerCUDA::getSimviewPrimitive() {
  const auto object = std::make_shared<simview::model::Box>(
      static_cast<float>((_min.x + _max.x) * 0.5),  // x
      static_cast<float>((_min.y + _max.y) * 0.5),  // y
      static_cast<float>((_min.z + _max.z) * 0.5),  // z
      static_cast<float>(_max.x - _min.x),          // width
      static_cast<float>(_max.y - _min.y),          // height
      static_cast<float>(_max.z - _min.z)           // depth
  );

  object->initVAO();

  return object;
}
