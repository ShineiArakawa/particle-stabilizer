#include <assimp/postprocess.h>  // Post processing flags
#include <assimp/scene.h>        // Output data structure

#include <ParticleStabilizer/PolygonContainer.hpp>
#include <Util/Logging.hpp>
#include <assimp/Importer.hpp>  // C++ importer interface

PolygonContainer::PolygonContainer(const std::string filePath,
                                   const double coefficientOfRestitution,
                                   const size_t nParticles)
    : _containerFilePath(filePath),
      _coefficientOfRestitution(coefficientOfRestitution),
      _nParticles(nParticles),
      _nPolygons(),
      _polygonCoords(),
      _polygonMinCoords(),
      _polygonMaxCoords(),
      _normals(),
      _intersectCounter(),
      _particleBuffer() {
  std::vector<glm::dvec3> vertices;
  std::vector<int64_t> indices;

  readFromFile(filePath, vertices, indices);

  // Number of triangles
  _nPolygons = indices.size() / 3ULL;
  LOG_INFO("nPolygons = " + std::to_string(_nPolygons));

  _polygonCoords.resize(3 * _nPolygons);

  for (size_t iPolygons = 0ULL; iPolygons < _nPolygons; ++iPolygons) {
    const size_t offset = 3ULL * iPolygons;

    const int64_t offsetID0 = indices[offset + 0ULL];
    const int64_t offsetID1 = indices[offset + 1ULL];
    const int64_t offsetID2 = indices[offset + 2ULL];

    _polygonCoords[offset + 0ULL] = vertices[offsetID0];
    _polygonCoords[offset + 1ULL] = vertices[offsetID1];
    _polygonCoords[offset + 2ULL] = vertices[offsetID2];
  }

  _intersectCounter.resize(_nParticles);

  // Initialize particle buffer
  _particleBuffer.resize(_nParticles);
  for (size_t iParticle = 0ULL; iParticle < _nParticles; ++iParticle) {
    _particleBuffer[iParticle] = std::make_shared<Particle>();
  }

  // Calc normals
  _normals.resize(_nPolygons);
  for (size_t iPolygon = 0ULL; iPolygon < _nPolygons; ++iPolygon) {
    const size_t offset = 3ULL * iPolygon;

    const glm::dvec3 vertex0 = {_polygonCoords[offset + 0ULL].x,
                                _polygonCoords[offset + 0ULL].y,
                                _polygonCoords[offset + 0ULL].z};
    const glm::dvec3 vertex1 = {_polygonCoords[offset + 1ULL].x,
                                _polygonCoords[offset + 1ULL].y,
                                _polygonCoords[offset + 1ULL].z};
    const glm::dvec3 vertex2 = {_polygonCoords[offset + 2ULL].x,
                                _polygonCoords[offset + 2ULL].y,
                                _polygonCoords[offset + 2ULL].z};

    _normals[iPolygon] = glm::normalize(glm::cross(vertex1 - vertex0, vertex2 - vertex0));
  }

  // Calc AABB
  _polygonMinCoords.resize(_nPolygons);
  _polygonMaxCoords.resize(_nPolygons);

  for (size_t iPolygon = 0ULL; iPolygon < _nPolygons; ++iPolygon) {
    const size_t offset = 3ULL * iPolygon;

    const glm::dvec3 vertex0 = {_polygonCoords[offset + 0ULL].x,
                                _polygonCoords[offset + 0ULL].y,
                                _polygonCoords[offset + 0ULL].z};
    const glm::dvec3 vertex1 = {_polygonCoords[offset + 1ULL].x,
                                _polygonCoords[offset + 1ULL].y,
                                _polygonCoords[offset + 1ULL].z};
    const glm::dvec3 vertex2 = {_polygonCoords[offset + 2ULL].x,
                                _polygonCoords[offset + 2ULL].y,
                                _polygonCoords[offset + 2ULL].z};

    _polygonMinCoords[iPolygon] = glm::min(glm::min(vertex0, vertex1), vertex2);
    _polygonMaxCoords[iPolygon] = glm::max(glm::max(vertex0, vertex1), vertex2);
  }
}

glm::dvec3 PolygonContainer::projectPointOntoPlane(const glm::dvec3& point,      /* OP */
                                                   const glm::dvec3& planePoint, /* OA */
                                                   const glm::dvec3& planeNormal /* N  */) const {
  // AP
  const glm::dvec3& v = point - planePoint;

  const double innerProduct = glm::dot(v, planeNormal);

  return point - planeNormal * innerProduct;
}

bool PolygonContainer::isPointOnPolygon(const glm::dvec3& p,
                                        const glm::dvec3& a,
                                        const glm::dvec3& b,
                                        const glm::dvec3& c) const {
  const glm::dvec3& v0 = c - a;
  const glm::dvec3& v1 = b - a;
  const glm::dvec3& v2 = p - a;

  const double dot00 = glm::dot(v0, v0);
  const double dot01 = glm::dot(v0, v1);
  const double dot02 = glm::dot(v0, v2);
  const double dot11 = glm::dot(v1, v1);
  const double dot12 = glm::dot(v1, v2);

  const double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
  const double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  const double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

  return (u >= 0.0) && (v >= 0.0) && (u + v <= 1.0);
}

glm::dvec3 PolygonContainer::closestPointOnLineSegment(const glm::dvec3& p,
                                                       const glm::dvec3& a,
                                                       const glm::dvec3& b) const {
  const glm::dvec3& ab = b - a;
  const glm::dvec3& ap = p - a;  // CHECK

  double t = glm::dot(ap, ab) / glm::dot(ab, ab);

  t = std::max(0.0, std::min(1.0, t));

  return a + ab * t;
}

glm::dvec3 PolygonContainer::getClosestPointOnPolygon(const glm::dvec3& vertex0,  // A
                                                      const glm::dvec3& vertex1,  // B
                                                      const glm::dvec3& vertex2,  // C
                                                      const glm::dvec3& normal,
                                                      const glm::dvec3& center) const {
  // Project the point onto the plane of the triangle
  glm::dvec3 projection = projectPointOntoPlane(center, vertex0, normal);

  // Check if the projection is inside the triangle
  if (isPointOnPolygon(projection, vertex0, vertex1, vertex2)) {
    return projection;
  }

  // Find the closest point on each edge
  // Edge AB
  const glm::dvec3& closestPointAB = closestPointOnLineSegment(center, vertex0, vertex1);
  double minDist = glm::distance(center, closestPointAB);

  projection = closestPointAB;

  // Edge BC
  const glm::dvec3& closestPointBC = closestPointOnLineSegment(center, vertex1, vertex2);
  const double distBC = glm::distance(center, closestPointBC);

  if (distBC < minDist) {
    minDist = distBC;
    projection = closestPointBC;
  }

  // Edge CA
  const glm::dvec3& closestPointCA = closestPointOnLineSegment(center, vertex2, vertex0);
  const double distCA = glm::distance(center, closestPointCA);

  if (distCA < minDist) {
    minDist = distCA;
    projection = closestPointCA;
  }

  return projection;
}

bool PolygonContainer::resolveCollision(const glm::dvec3& vertex0,
                                        const glm::dvec3& vertex1,
                                        const glm::dvec3& vertex2,
                                        const glm::dvec3& normal,
                                        const ParticlePtr& particle,
                                        ParticlePtr& particleBuffer) {
  const glm::dvec3& closestPoint = getClosestPointOnPolygon(vertex0,
                                                            vertex1,
                                                            vertex2,
                                                            normal,
                                                            particle->position);

  const glm::dvec3& relvec = closestPoint - particle->position;
  const double distance = glm::length(relvec);

  const bool isIntersect = distance < particle->radius;

  if (isIntersect) {
    {
      // Modify velocity
      // const glm::dvec3& reflectedVelocity = particle->velocity - 2.0 * glm::dot(particle->velocity, normal) * normal;
      const glm::dvec3& reflectedVelocity = particle->velocity - 2.0 * glm::dot(particle->velocity, normal) * normal;

      particleBuffer->velocity += _coefficientOfRestitution * reflectedVelocity;
    }

    {
      // Modify position
      const double distanceToPlane = glm::dot(-normal, particleBuffer->position - vertex0);

      const double modification = particle->radius - distanceToPlane;

      particleBuffer->position = particleBuffer->position - modification * normal;
    }
  }

  return isIntersect;
}

void PolygonContainer::copyParticleBufferPosition(const std::vector<ParticlePtr>& particles) {
  for (size_t iParticle = 0; iParticle < _nParticles; ++iParticle) {
    _particleBuffer[iParticle]->position = particles[iParticle]->position;
  }
}

void PolygonContainer::clearParticleBufferVelocity() {
  for (size_t iParticle = 0; iParticle < _nParticles; ++iParticle) {
    _particleBuffer[iParticle]->velocity.x = 0.0;
    _particleBuffer[iParticle]->velocity.y = 0.0;
    _particleBuffer[iParticle]->velocity.z = 0.0;
  }
}

void PolygonContainer::clearIntersectCounter() {
  for (size_t iParticle = 0; iParticle < _nParticles; ++iParticle) {
    _intersectCounter[iParticle] = 0LL;
  }
}

void PolygonContainer::resolveCollisions(std::vector<ParticlePtr>& particles) {
  // From now on, particles is read-only
  // _particleBuffer is write-only
  copyParticleBufferPosition(particles);
  clearParticleBufferVelocity();
  clearIntersectCounter();

#pragma omp parallel for
  for (int64_t iParticle = 0; iParticle < static_cast<int64_t>(_nParticles); ++iParticle) {
    const glm::dvec3 minCoords = particles[iParticle]->position - particles[iParticle]->radius;
    const glm::dvec3 maxCoords = particles[iParticle]->position + particles[iParticle]->radius;

    for (int64_t iPolygon = 0; iPolygon < static_cast<int64_t>(_nPolygons); ++iPolygon) {
      // AABB test
      if (minCoords.x > _polygonMaxCoords[iPolygon].x ||
          maxCoords.x < _polygonMinCoords[iPolygon].x ||
          minCoords.y > _polygonMaxCoords[iPolygon].y ||
          maxCoords.y < _polygonMinCoords[iPolygon].y ||
          minCoords.z > _polygonMaxCoords[iPolygon].z ||
          maxCoords.z < _polygonMinCoords[iPolygon].z) {
        // No intersection
        continue;
      }

      // Detailed test
      const int64_t offsetPolygonCoords = 3LL * iPolygon;

      const glm::dvec3& vertex0 = _polygonCoords[offsetPolygonCoords + 0LL];
      const glm::dvec3& vertex1 = _polygonCoords[offsetPolygonCoords + 1LL];
      const glm::dvec3& vertex2 = _polygonCoords[offsetPolygonCoords + 2LL];
      const glm::dvec3& normal = _normals[iPolygon];

      if (resolveCollision(vertex0,
                           vertex1,
                           vertex2,
                           normal,
                           particles[iParticle],
                           _particleBuffer[iParticle])) {
        _intersectCounter[iParticle] += 1LL;
      }
    }
  }

  for (size_t iParticle = 0; iParticle < _nParticles; ++iParticle) {
    if (_intersectCounter[iParticle] > 0LL) {
      particles[iParticle]->position = _particleBuffer[iParticle]->position;
      particles[iParticle]->velocity = _particleBuffer[iParticle]->velocity / static_cast<double>(_intersectCounter[iParticle]);
    }
  }
}

glm::dvec3 PolygonContainer::getMinCoords() {
  glm::dvec3 minCoords = {};

  const size_t nVertices = _polygonCoords.size();

  for (size_t i = 0; i < nVertices; ++i) {
    if (i == 0ULL) {
      minCoords = _polygonCoords[i];
    } else {
      minCoords = glm::min(minCoords, _polygonCoords[i]);
    }
  }

  return minCoords;
}

glm::dvec3 PolygonContainer::getMaxCoords() {
  glm::dvec3 maxCoords = {};

  const size_t nVertices = _polygonCoords.size();

  for (size_t i = 0; i < nVertices; ++i) {
    if (i == 0ULL) {
      maxCoords = _polygonCoords[i];
    } else {
      maxCoords = glm::max(maxCoords, _polygonCoords[i]);
    }
  }

  return maxCoords;
}

simview::model::Primitive_t PolygonContainer::getSimviewPrimitive() {
  const simview::vec3f_t color = {0.0f, 0.0f, 1.0f};

  std::shared_ptr<std::vector<simview::vec3f_t>> positions = std::make_shared<std::vector<simview::vec3f_t>>();
  std::shared_ptr<std::vector<simview::vec3f_t>> colors = std::make_shared<std::vector<simview::vec3f_t>>();
  std::shared_ptr<std::vector<simview::vec3f_t>> normals = std::make_shared<std::vector<simview::vec3f_t>>();

  // Alloc
  const size_t nVertices = _polygonCoords.size();
  positions->resize(nVertices);
  colors->resize(nVertices);
  normals->resize(nVertices);

  for (size_t i = 0; i < _nPolygons; ++i) {
    const size_t offset = 3 * i;

    const glm::dvec3 vertex0 = {_polygonCoords[offset + 0].x,
                                _polygonCoords[offset + 0].y,
                                _polygonCoords[offset + 0].z};
    const glm::dvec3 vertex1 = {_polygonCoords[offset + 1].x,
                                _polygonCoords[offset + 1].y,
                                _polygonCoords[offset + 1].z};
    const glm::dvec3 vertex2 = {_polygonCoords[offset + 2].x,
                                _polygonCoords[offset + 2].y,
                                _polygonCoords[offset + 2].z};
    const glm::dvec3 normal = glm::normalize(glm::cross(vertex1 - vertex0, vertex2 - vertex0));

    // Positions
    (*positions)[offset + 0] = {static_cast<float>(vertex0.x),
                                static_cast<float>(vertex0.y),
                                static_cast<float>(vertex0.z)};
    (*positions)[offset + 1] = {static_cast<float>(vertex1.x),
                                static_cast<float>(vertex1.y),
                                static_cast<float>(vertex1.z)};
    (*positions)[offset + 2] = {static_cast<float>(vertex2.x),
                                static_cast<float>(vertex2.y),
                                static_cast<float>(vertex2.z)};
    // Colors
    (*colors)[offset + 0] = color;
    (*colors)[offset + 1] = color;
    (*colors)[offset + 2] = color;

    // Normals
    (*normals)[offset + 0] = {static_cast<float>(normal.x),
                              static_cast<float>(normal.y),
                              static_cast<float>(normal.z)};
    (*normals)[offset + 1] = {static_cast<float>(normal.x),
                              static_cast<float>(normal.y),
                              static_cast<float>(normal.z)};
    (*normals)[offset + 2] = {static_cast<float>(normal.x),
                              static_cast<float>(normal.y),
                              static_cast<float>(normal.z)};
  }

  simview::model::Object_t object = std::make_shared<simview::model::Object>();

  object->initVAO(positions, colors, normals);

  return object;
}

void PolygonContainer::readFromFile(const std::string& filePath,
                                    std::vector<glm::dvec3>& vertices,
                                    std::vector<int64_t>& indices) {
  indices.clear();
  vertices.clear();

  Assimp::Importer importer;
  unsigned int flag = 0;
  flag |= aiProcess_Triangulate;
  flag |= aiProcess_CalcTangentSpace;
  flag |= aiProcess_RemoveRedundantMaterials;
  flag |= aiProcess_GenNormals;

  const aiScene* scene = importer.ReadFile(filePath, flag);

  if (scene == nullptr) {
    LOG_ERROR("[ERROR] " + std::string(importer.GetErrorString()));
    return;
  }

  for (unsigned int iMesh = 0; iMesh < scene->mNumMeshes; ++iMesh) {
    aiMesh* mesh = scene->mMeshes[iMesh];

    for (unsigned int iFace = 0; iFace < mesh->mNumFaces; ++iFace) {
      const aiFace& face = mesh->mFaces[iFace];

      for (unsigned int iVertex = 0; iVertex < face.mNumIndices; ++iVertex) {
        const unsigned int index = face.mIndices[iVertex];

        indices.push_back(static_cast<int64_t>(vertices.size()));
        vertices.push_back(glm::dvec3(static_cast<double>(mesh->mVertices[index].x),
                                      static_cast<double>(mesh->mVertices[index].y),
                                      static_cast<double>(mesh->mVertices[index].z)));
      }
    }
  }
}