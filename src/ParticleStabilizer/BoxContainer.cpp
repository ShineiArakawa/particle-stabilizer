#include <ParticleStabilizer/BoxContainer.hpp>
#include <ParticleStabilizer/ParticleModel.hpp>

BoxContainer::BoxContainer(const glm::dvec3& min,
                           const glm::dvec3& max,
                           const CoefficientOfRestitutionWall coefficientOfRestitution)
    : _min(min),
      _max(max),
      _faceIDsBuffer(),
      _minCoordsBuffer(),
      _maxCoordsBuffer(),
      _distanceToWall(),
      _coefficientOfRestitution() {
  _coefficientOfRestitution[0] = coefficientOfRestitution.zMin;
  _coefficientOfRestitution[1] = coefficientOfRestitution.zMax;
  _coefficientOfRestitution[2] = coefficientOfRestitution.xMin;
  _coefficientOfRestitution[3] = coefficientOfRestitution.xMax;
  _coefficientOfRestitution[4] = coefficientOfRestitution.yMin;
  _coefficientOfRestitution[5] = coefficientOfRestitution.yMax;
}

void BoxContainer::resolveCollisions(std::vector<ParticlePtr>& particles) {
  const int64_t numParticles = particles.size();

  for (int64_t iParticle = 0; iParticle < numParticles; ++iParticle) {
    resolveCollisions(particles[iParticle]);
  }
}

void BoxContainer::resolveCollisions(ParticlePtr& particle) {
  _faceIDsBuffer.clear();

  _minCoordsBuffer = particle->position - particle->radius;
  _maxCoordsBuffer = particle->position + particle->radius;

  if (_minCoordsBuffer.x < _min.x) {
    particle->position.x = _min.x + particle->radius;
    particle->velocity.x = -particle->velocity.x * _coefficientOfRestitution[LABEL_FACE_X_MIN];
  }

  if (_minCoordsBuffer.y < _min.y) {
    particle->position.y = _min.y + particle->radius;
    particle->velocity.y = -particle->velocity.y * _coefficientOfRestitution[LABEL_FACE_Y_MIN];
  }

  if (_minCoordsBuffer.z < _min.z) {
    particle->position.z = _min.z + particle->radius;
    particle->velocity.z = -particle->velocity.z * _coefficientOfRestitution[LABEL_FACE_Z_MIN];
  }

  if (_maxCoordsBuffer.x > _max.x) {
    particle->position.x = _max.x - particle->radius;
    particle->velocity.x = -particle->velocity.x * _coefficientOfRestitution[LABEL_FACE_X_MAX];
  }

  if (_maxCoordsBuffer.y > _max.y) {
    particle->position.y = _max.y - particle->radius;
    particle->velocity.y = -particle->velocity.y * _coefficientOfRestitution[LABEL_FACE_Y_MAX];
  }

  if (_maxCoordsBuffer.z > _max.z) {
    particle->position.z = _max.z - particle->radius;
    particle->velocity.z = -particle->velocity.z * _coefficientOfRestitution[LABEL_FACE_Z_MAX];
  }
}

glm::dvec3 BoxContainer::calcSymmetricVector(const glm::dvec3& vector, const glm::dvec3& pole) {
  const double lenPole = glm::length(pole);
  const double len = glm::dot(vector, pole) / (lenPole * lenPole);

  return -vector + 2.0 * len * pole;
}

glm::dvec3 BoxContainer::getMinCoords() {
  return _min;
}

glm::dvec3 BoxContainer::getMaxCoords() {
  return _max;
}

simview::model::Primitive_t BoxContainer::getSimviewPrimitive() {
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
