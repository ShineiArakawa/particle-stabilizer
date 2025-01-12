#pragma once

#include <ParticleStabilizer/ContainerBase.hpp>
#include <Util/Math.hpp>
#include <vector>

class BoxContainer : public ContainerBase {
 public:
  inline static const int LABEL_FACE_INVALID = -1;
  inline static const int LABEL_FACE_Z_MIN = 0;
  inline static const int LABEL_FACE_Z_MAX = 1;
  inline static const int LABEL_FACE_X_MIN = 2;
  inline static const int LABEL_FACE_X_MAX = 3;
  inline static const int LABEL_FACE_Y_MIN = 4;
  inline static const int LABEL_FACE_Y_MAX = 5;
  inline static const int LIST_LABEL_FACE[] = {LABEL_FACE_Z_MIN, LABEL_FACE_Z_MAX, LABEL_FACE_X_MIN,
                                               LABEL_FACE_X_MAX, LABEL_FACE_Y_MIN, LABEL_FACE_Y_MAX};
  inline static const glm::dvec3 UNIT_PERPENDICULAR_VECTOR_XY_MIN = {0.0f, 0.0f, 1.0f};
  inline static const glm::dvec3 UNIT_PERPENDICULAR_VECTOR_XY_MAX = {0.0f, 0.0f, -1.0f};
  inline static const glm::dvec3 UNIT_PERPENDICULAR_VECTOR_YZ_MIN = {1.0f, 0.0f, 0.0f};
  inline static const glm::dvec3 UNIT_PERPENDICULAR_VECTOR_YZ_MAX = {-1.0f, 0.0f, 0.0f};
  inline static const glm::dvec3 UNIT_PERPENDICULAR_VECTOR_ZX_MIN = {0.0f, 1.0f, 0.0f};
  inline static const glm::dvec3 UNIT_PERPENDICULAR_VECTOR_ZX_MAX = {0.0f, -1.0f, 0.0f};
  inline static const glm::dvec3 LIST_UNIT_PERPENDICULAR_VECTOR[] = {UNIT_PERPENDICULAR_VECTOR_XY_MIN,
                                                                     UNIT_PERPENDICULAR_VECTOR_XY_MAX,
                                                                     UNIT_PERPENDICULAR_VECTOR_YZ_MIN,
                                                                     UNIT_PERPENDICULAR_VECTOR_YZ_MAX,
                                                                     UNIT_PERPENDICULAR_VECTOR_ZX_MIN,
                                                                     UNIT_PERPENDICULAR_VECTOR_ZX_MAX};

  BoxContainer(const glm::dvec3& min,
               const glm::dvec3& max,
               const CoefficientOfRestitutionWall coefficientOfRestitution);

  ~BoxContainer() override = default;

  void resolveCollisions(std::vector<ParticlePtr>& particles) override;

  static glm::dvec3 calcSymmetricVector(const glm::dvec3& vector, const glm::dvec3& pole);

  glm::dvec3 getMinCoords() override;

  glm::dvec3 getMaxCoords() override;

  simview::model::Primitive_t getSimviewPrimitive() override;

 private:
  glm::dvec3 _min;
  glm::dvec3 _max;

  std::vector<int> _faceIDsBuffer;
  glm::dvec3 _minCoordsBuffer;
  glm::dvec3 _maxCoordsBuffer;
  std::array<double, 6> _distanceToWall;

  std::array<double, 6> _coefficientOfRestitution;

  void resolveCollisions(ParticlePtr& particle);
};