#pragma once

#include <ParticleStabilizer/ContainerBase.hpp>
#include <vector>

class PolygonContainer : public ContainerBase {
 public:
  PolygonContainer(const std::string filePath,
                   const double coefficientOfRestitution,
                   const size_t nParticles);

  ~PolygonContainer() override = default;

  void resolveCollisions(std::vector<ParticlePtr>& particles) override;

  glm::dvec3 getMinCoords() override;

  glm::dvec3 getMaxCoords() override;

  simview::model::Primitive_t getSimviewPrimitive() override;

  static void readFromFile(const std::string& filePath,
                           std::vector<glm::dvec3>& vertices,
                           std::vector<int64_t>& indices);

 private:
  glm::dvec3 projectPointOntoPlane(const glm::dvec3& point,               // OP
                                   const glm::dvec3& planePoint,          // OA
                                   const glm::dvec3& planeNormal) const;  // N

  bool isPointOnPolygon(const glm::dvec3& p,
                        const glm::dvec3& a,
                        const glm::dvec3& b,
                        const glm::dvec3& c) const;

  glm::dvec3 closestPointOnLineSegment(const glm::dvec3& p,
                                       const glm::dvec3& a,
                                       const glm::dvec3& b) const;

  glm::dvec3 getClosestPointOnPolygon(const glm::dvec3& vertex0,  // A
                                      const glm::dvec3& vertex1,  // B
                                      const glm::dvec3& vertex2,  // C
                                      const glm::dvec3& normal,
                                      const glm::dvec3& center) const;

  bool resolveCollision(const glm::dvec3& vertex0,
                        const glm::dvec3& vertex1,
                        const glm::dvec3& vertex2,
                        const glm::dvec3& normal,
                        const ParticlePtr& particle,
                        ParticlePtr& particleBuffer);

  void copyParticleBufferPosition(const std::vector<ParticlePtr>& particles);
  void clearParticleBufferVelocity();
  void clearIntersectCounter();

  std::string _containerFilePath;
  double _coefficientOfRestitution;
  size_t _nParticles;

  size_t _nPolygons;
  std::vector<glm::dvec3> _polygonCoords;
  std::vector<glm::dvec3> _polygonMinCoords;
  std::vector<glm::dvec3> _polygonMaxCoords;
  std::vector<glm::dvec3> _normals;
  std::vector<int64_t> _intersectCounter;
  std::vector<ParticlePtr> _particleBuffer;
};
