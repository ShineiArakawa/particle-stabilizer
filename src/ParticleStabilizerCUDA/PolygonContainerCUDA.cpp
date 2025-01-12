#include <assimp/postprocess.h>  // Post processing flags
#include <assimp/scene.h>        // Output data structure

#include <ParticleStabilizer/PolygonContainer.hpp>
#include <ParticleStabilizerCUDA/PolygonContainerCUDA.hpp>
#include <ParticleStabilizerCUDA/PolygonContainerCUDA_kernels.cuh>
#include <assimp/Importer.hpp>  // C++ importer interface

PolygonContainerCUDA::PolygonContainerCUDA(const std::string filePath,
                                           const double coefficientOfRestitution,
                                           const int64_t nParticles)
    : _containerFilePath(filePath),
      _coefficientOfRestitution(coefficientOfRestitution),
      _nParticles(nParticles),
      _nPolygons(),
      _hostPolygonCoords(),
      _hostNormals(),
      _devicePolygonCoords(),
      _devicePolygonMinCoords(),
      _devicePolygonMaxCoords(),
      _deviceNormals(),
      _deviceIsIntersected(),
      _deviceIntersectCounter(),
      _deviceParticleVelBuffer(),
      _devicePositionModification() {
  std::vector<glm::dvec3> vertices;
  std::vector<int64_t> indices;

  readFromFile(filePath, vertices, indices);

  // Number of triangles
  _nPolygons = static_cast<int64_t>(indices.size() / 3ULL);
  LOG_INFO("nPolygons = " + std::to_string(_nPolygons));

  _hostPolygonCoords = new double3[3L * _nPolygons];

  for (int64_t iPolygons = 0L; iPolygons < _nPolygons; ++iPolygons) {
    const int64_t offset = 3L * iPolygons;

    const uint32_t offsetID0 = indices[offset + 0L];
    const uint32_t offsetID1 = indices[offset + 1L];
    const uint32_t offsetID2 = indices[offset + 2L];

    _hostPolygonCoords[offset + 0L] = make_double3(vertices[offsetID0].x,
                                                   vertices[offsetID0].y,
                                                   vertices[offsetID0].z);
    _hostPolygonCoords[offset + 1L] = make_double3(vertices[offsetID1].x,
                                                   vertices[offsetID1].y,
                                                   vertices[offsetID1].z);
    _hostPolygonCoords[offset + 2L] = make_double3(vertices[offsetID2].x,
                                                   vertices[offsetID2].y,
                                                   vertices[offsetID2].z);
  }

  double3* hostPolygonMinCoords = (double3*)malloc(sizeof(double3) * _nPolygons);
  double3* hostPolygonMaxCoords = (double3*)malloc(sizeof(double3) * _nPolygons);
  _hostNormals = new double3[_nPolygons];

  for (int64_t iPolygons = 0L; iPolygons < _nPolygons; ++iPolygons) {
    const int64_t offset = 3L * iPolygons;

    const double3& vertex0 = _hostPolygonCoords[offset + 0L];
    const double3& vertex1 = _hostPolygonCoords[offset + 1L];
    const double3& vertex2 = _hostPolygonCoords[offset + 2L];

    hostPolygonMinCoords[iPolygons] = make_double3(std::min(std::min(vertex0.x, vertex1.x), vertex2.x),
                                                   std::min(std::min(vertex0.y, vertex1.y), vertex2.y),
                                                   std::min(std::min(vertex0.z, vertex1.z), vertex2.z));
    hostPolygonMaxCoords[iPolygons] = make_double3(std::max(std::max(vertex0.x, vertex1.x), vertex2.x),
                                                   std::max(std::max(vertex0.y, vertex1.y), vertex2.y),
                                                   std::max(std::max(vertex0.z, vertex1.z), vertex2.z));

    const double3 relVec0 = make_double3(vertex1.x - vertex0.x,
                                         vertex1.y - vertex0.y,
                                         vertex1.z - vertex0.z);
    const double3 relVec1 = make_double3(vertex2.x - vertex0.x,
                                         vertex2.y - vertex0.y,
                                         vertex2.z - vertex0.z);
    const double3 normal = make_double3(relVec0.y * relVec1.z - relVec0.z * relVec1.y,
                                        relVec0.z * relVec1.x - relVec0.x * relVec1.z,
                                        relVec0.x * relVec1.y - relVec0.y * relVec1.x);
    const double length = std::sqrt(normal.x * normal.x +
                                    normal.y * normal.y +
                                    normal.z * normal.z);

    _hostNormals[iPolygons] = make_double3(normal.x / length,
                                           normal.y / length,
                                           normal.z / length);
  }

  // Malloc & copy
  CUDA_CHECK_ERROR(cudaMalloc(&_devicePolygonCoords, sizeof(double3) * 3L * _nPolygons));
  CUDA_CHECK_ERROR(cudaMemcpy(_devicePolygonCoords, _hostPolygonCoords, sizeof(double3) * 3L * _nPolygons, cudaMemcpyHostToDevice));

  CUDA_CHECK_ERROR(cudaMalloc(&_devicePolygonMinCoords, sizeof(double3) * _nPolygons));
  CUDA_CHECK_ERROR(cudaMemcpy(_devicePolygonMinCoords, hostPolygonMinCoords, sizeof(double3) * _nPolygons, cudaMemcpyHostToDevice));

  CUDA_CHECK_ERROR(cudaMalloc(&_devicePolygonMaxCoords, sizeof(double3) * _nPolygons));
  CUDA_CHECK_ERROR(cudaMemcpy(_devicePolygonMaxCoords, hostPolygonMaxCoords, sizeof(double3) * _nPolygons, cudaMemcpyHostToDevice));

  CUDA_CHECK_ERROR(cudaMalloc(&_deviceNormals, sizeof(double3) * _nPolygons));
  CUDA_CHECK_ERROR(cudaMemcpy(_deviceNormals, _hostNormals, sizeof(double3) * _nPolygons, cudaMemcpyHostToDevice));

  CUDA_CHECK_ERROR(cudaMalloc(&_deviceIsIntersected, sizeof(bool) * _nParticles * _nPolygons));

  CUDA_CHECK_ERROR(cudaMalloc(&_deviceIntersectCounter, sizeof(int) * _nParticles));

  CUDA_CHECK_ERROR(cudaMalloc(&_deviceParticleVelBuffer, sizeof(double3) * _nParticles));

  CUDA_CHECK_ERROR(cudaMalloc(&_devicePositionModification, sizeof(double3) * _nParticles));

  // Clean up
  free(hostPolygonMinCoords);
  free(hostPolygonMaxCoords);
}

PolygonContainerCUDA::~PolygonContainerCUDA() {
  free(_hostPolygonCoords);
  free(_hostNormals);
  CUDA_CHECK_ERROR(cudaFree(_devicePolygonCoords));
  CUDA_CHECK_ERROR(cudaFree(_devicePolygonMinCoords));
  CUDA_CHECK_ERROR(cudaFree(_devicePolygonMaxCoords));
  CUDA_CHECK_ERROR(cudaFree(_deviceNormals));
  CUDA_CHECK_ERROR(cudaFree(_deviceIsIntersected));
  CUDA_CHECK_ERROR(cudaFree(_deviceIntersectCounter));
  CUDA_CHECK_ERROR(cudaFree(_deviceParticleVelBuffer));
  CUDA_CHECK_ERROR(cudaFree(_devicePositionModification));
}

void PolygonContainerCUDA::resolveCollisions(ParticleCUDA* particles) {
  {
    // Initialize buffer
    launchInitBuffersKernel(_deviceIntersectCounter,
                            _devicePositionModification,
                            _deviceParticleVelBuffer,
                            _nParticles);
  }

  {
    // Check sphere-polygon intersection
    launchIsIntersectedKernel(particles,
                              _nParticles,
                              _devicePolygonCoords,
                              _deviceNormals,
                              _devicePolygonMinCoords,
                              _devicePolygonMaxCoords,
                              _nPolygons,
                              _deviceIsIntersected,
                              _deviceIntersectCounter);
  }

  // {
  //   bool* hostIsIntersected = new bool[_nParticles * _nPolygons];
  //   CUDA_CHECK_ERROR(cudaMemcpy(hostIsIntersected, _deviceIsIntersected, sizeof(bool) * _nParticles * _nPolygons, cudaMemcpyDeviceToHost));
  //   bool xxx = false;

  //   for (int i = 0; i < _nParticles * _nPolygons; ++i) {
  //     if (hostIsIntersected[i]) {
  //       xxx = true;
  //     }
  //   }

  //   if (xxx) {
  //     std::cout << "intersected!" << std::endl;
  //   }
  // }

  {
    // Modifiy velocity
    launchCalcModifiedVelocitySumKernel(particles,
                                        _nParticles,
                                        _deviceNormals,
                                        _nPolygons,
                                        _deviceIsIntersected,
                                        _coefficientOfRestitution,
                                        _deviceParticleVelBuffer);

    launchCalcAveragedVelocityKernel(_nParticles,
                                     _deviceParticleVelBuffer,
                                     _deviceIntersectCounter,
                                     particles);
  }

  // {
  //   ParticleCUDA* hostParticles = new ParticleCUDA[_nParticles];
  //   CUDA_CHECK_ERROR(cudaMemcpy(hostParticles, particles, sizeof(ParticleCUDA) * _nParticles, cudaMemcpyDeviceToHost));

  //   printf("###\n");
  //   printf("%.8lf              %.8lf              %.8lf\n", hostParticles[0].position.x, hostParticles[0].position.y, hostParticles[0].position.z);
  //   printf("%.8lf              %.8lf              %.8lf\n", hostParticles[0].velocity.x, hostParticles[0].velocity.y, hostParticles[0].velocity.z);

  //   if (std::abs(hostParticles[0].velocity.y) < 1.0) {
  //     exit(0);
  //   }
  // }

  // printf("%lf              %lf              %lf\n", particles[iParticle].velocity.x, particles[iParticle].velocity.y, particles[iParticle].velocity.z);

  {
    // Modifiy position
    launchCalcPositionModificationKernel(particles,
                                         _nParticles,
                                         _devicePolygonCoords,
                                         _deviceNormals,
                                         _nPolygons,
                                         _deviceIsIntersected,
                                         _devicePositionModification);

    launchModifyPositionKernel(_nParticles,
                               _devicePositionModification,
                               particles);
  }

  // {
  //   ParticleCUDA* hostParticles = new ParticleCUDA[_nParticles];
  //   CUDA_CHECK_ERROR(cudaMemcpy(hostParticles, particles, sizeof(ParticleCUDA) * _nParticles, cudaMemcpyDeviceToHost));

  //   printf("###\n");
  //   printf("%.8lf              %.8lf              %.8lf\n", hostParticles[0].position.x, hostParticles[0].position.y, hostParticles[0].position.z);
  //   printf("%.8lf              %.8lf              %.8lf\n", hostParticles[0].velocity.x, hostParticles[0].velocity.y, hostParticles[0].velocity.z);

  //   if (std::abs(hostParticles[0].velocity.y) < 1.0) {
  //     exit(0);
  //   }
  // }
}

double3 PolygonContainerCUDA::getMinCoords() {
  double3 minCoords = make_double3(0.0, 0.0, 0.0);

  const int64_t nVertices = 3L * _nPolygons;

  for (int64_t i = 0L; i < nVertices; ++i) {
    if (i == 0L) {
      minCoords = _hostPolygonCoords[i];
    } else {
      minCoords.x = std::min(minCoords.x, _hostPolygonCoords[i].x);
      minCoords.y = std::min(minCoords.y, _hostPolygonCoords[i].y);
      minCoords.z = std::min(minCoords.z, _hostPolygonCoords[i].z);
    }
  }

  return minCoords;
}

double3 PolygonContainerCUDA::getMaxCoords() {
  double3 maxCoords = make_double3(0.0, 0.0, 0.0);

  const int64_t nVertices = 3L * _nPolygons;

  for (int64_t i = 0L; i < nVertices; ++i) {
    if (i == 0L) {
      maxCoords = _hostPolygonCoords[i];
    } else {
      maxCoords.x = std::max(maxCoords.x, _hostPolygonCoords[i].x);
      maxCoords.y = std::max(maxCoords.y, _hostPolygonCoords[i].y);
      maxCoords.z = std::max(maxCoords.z, _hostPolygonCoords[i].z);
    }
  }

  return maxCoords;
}

simview::model::Primitive_t PolygonContainerCUDA::getSimviewPrimitive() {
  const simview::vec3f_t color = {0.0f, 0.0f, 1.0f};

  std::shared_ptr<std::vector<simview::vec3f_t>> positions = std::make_shared<std::vector<simview::vec3f_t>>();
  std::shared_ptr<std::vector<simview::vec3f_t>> colors = std::make_shared<std::vector<simview::vec3f_t>>();
  std::shared_ptr<std::vector<simview::vec3f_t>> normals = std::make_shared<std::vector<simview::vec3f_t>>();

  // Alloc
  const int64_t nVertices = 3L * _nPolygons;
  positions->resize(nVertices);
  colors->resize(nVertices);
  normals->resize(nVertices);

  for (int64_t i = 0L; i < _nPolygons; ++i) {
    const int64_t offset = 3L * i;

    // Positions
    (*positions)[offset + 0L] = {static_cast<float>(_hostPolygonCoords[offset + 0L].x),
                                 static_cast<float>(_hostPolygonCoords[offset + 0L].y),
                                 static_cast<float>(_hostPolygonCoords[offset + 0L].z)};
    (*positions)[offset + 1L] = {static_cast<float>(_hostPolygonCoords[offset + 1L].x),
                                 static_cast<float>(_hostPolygonCoords[offset + 1L].y),
                                 static_cast<float>(_hostPolygonCoords[offset + 1L].z)};
    (*positions)[offset + 2L] = {static_cast<float>(_hostPolygonCoords[offset + 2L].x),
                                 static_cast<float>(_hostPolygonCoords[offset + 2L].y),
                                 static_cast<float>(_hostPolygonCoords[offset + 2L].z)};
    // Colors
    (*colors)[offset + 0L] = color;
    (*colors)[offset + 1L] = color;
    (*colors)[offset + 2L] = color;

    // Normals
    (*normals)[offset + 0L] = {static_cast<float>(_hostNormals[i].x),
                               static_cast<float>(_hostNormals[i].y),
                               static_cast<float>(_hostNormals[i].z)};
    (*normals)[offset + 1L] = {static_cast<float>(_hostNormals[i].x),
                               static_cast<float>(_hostNormals[i].y),
                               static_cast<float>(_hostNormals[i].z)};
    (*normals)[offset + 2L] = {static_cast<float>(_hostNormals[i].x),
                               static_cast<float>(_hostNormals[i].y),
                               static_cast<float>(_hostNormals[i].z)};
  }

  simview::model::Object_t object = std::make_shared<simview::model::Object>();

  object->initVAO(positions, colors, normals);

  return object;
}

void PolygonContainerCUDA::readFromFile(const std::string& filePath,
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

  for (uint32_t iMesh = 0; iMesh < scene->mNumMeshes; ++iMesh) {
    aiMesh* mesh = scene->mMeshes[iMesh];

    for (uint32_t iFace = 0; iFace < mesh->mNumFaces; ++iFace) {
      const aiFace& face = mesh->mFaces[iFace];

      for (uint32_t iVertex = 0; iVertex < face.mNumIndices; ++iVertex) {
        const unsigned int index = face.mIndices[iVertex];

        indices.push_back(static_cast<int64_t>(vertices.size()));
        vertices.push_back(glm::dvec3(static_cast<double>(mesh->mVertices[index].x),
                                      static_cast<double>(mesh->mVertices[index].y),
                                      static_cast<double>(mesh->mVertices[index].z)));
      }
    }
  }
}
