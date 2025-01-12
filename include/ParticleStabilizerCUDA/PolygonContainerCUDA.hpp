#pragma once

#include <ParticleStabilizer/ContainerBase.hpp>
#include <ParticleStabilizerCUDA/ContainerBaseCUDA.hpp>

#ifdef __cplusplus
extern "C" {
#endif

class PolygonContainerCUDA : public ContainerBaseCUDA {
 public:
  PolygonContainerCUDA(const std::string filePath,
                       const double coefficientOfRestitution,
                       const int64_t nParticles);

  ~PolygonContainerCUDA();

  void resolveCollisions(ParticleCUDA* particle) override;

  double3 getMinCoords() override;

  double3 getMaxCoords() override;

  simview::model::Primitive_t getSimviewPrimitive() override;

  static void readFromFile(const std::string& filePath,
                           std::vector<glm::dvec3>& vertices,
                           std::vector<int64_t>& indices);

 private:
  std::string _containerFilePath;
  double _coefficientOfRestitution;

  int64_t _nParticles;
  int64_t _nPolygons;

  double3* _hostPolygonCoords;
  double3* _hostNormals;

  double3* _devicePolygonCoords;
  double3* _devicePolygonMinCoords;
  double3* _devicePolygonMaxCoords;
  double3* _deviceNormals;
  bool* _deviceIsIntersected;
  int* _deviceIntersectCounter;
  double3* _deviceParticleVelBuffer;
  double3* _devicePositionModification;
};

#ifdef __cplusplus
}
#endif
