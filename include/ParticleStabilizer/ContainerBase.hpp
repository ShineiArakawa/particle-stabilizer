#pragma once

#include <ParticleStabilizer/ParticleModel.hpp>
#include <SimView/core.hpp>
#include <Util/Math.hpp>
#include <sstream>
#include <string>
#include <vector>

struct CoefficientOfRestitutionWall {
  CoefficientOfRestitutionWall(const double zMin,
                               const double zMax,
                               const double xMin,
                               const double xMax,
                               const double yMin,
                               const double yMax)
      : zMin(zMin), zMax(zMax), xMin(xMin), xMax(xMax), yMin(yMin), yMax(yMax) {};

  double zMin;
  double zMax;
  double xMin;
  double xMax;
  double yMin;
  double yMax;

  std::string toString() const {
    return "zMin: " + std::to_string(zMin) + ", zMax: " + std::to_string(zMax) + ", xMin: " + std::to_string(xMin) + ", xMax: " + std::to_string(xMax) + ", yMin: " + std::to_string(yMin) + ", yMax: " + std::to_string(yMax);
  }

  friend std::ostream& operator<<(std::ostream& os, const CoefficientOfRestitutionWall& obj) {
    os << "zMin: " << obj.zMin << ", zMax: " << obj.zMax << ", xMin: " << obj.xMin << ", xMax: " << obj.xMax << ", yMin: " << obj.yMin << ", yMax: " << obj.yMax;
    return os;
  }
};

class ContainerBase {
 public:
  virtual ~ContainerBase() = default;
  virtual void resolveCollisions(std::vector<ParticlePtr>& particles) = 0;
  virtual glm::dvec3 getMinCoords() = 0;
  virtual glm::dvec3 getMaxCoords() = 0;

  virtual simview::model::Primitive_t getSimviewPrimitive() = 0;
};
