#include <ParticleStabilizer/ObjData.hpp>
#include <Util/FileUtil.hpp>
#include <Util/Logging.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

void ObjData::write(const std::string& filePath, const std::vector<Particle>& particles) {
  LOG_INFO("Writing to file " + filePath);

  const std::string dirPath = FileUtil::dirPath(filePath);

  if (!FileUtil::exists(dirPath)) {
    FileUtil::mkdirs(dirPath);
  }

  // ==============================================================================================
  // Generate primitive
  // ==============================================================================================
  const size_t nParticles = particles.size();
  const size_t MAX_NUM_VERTICES = 1000000;
  int ndivs = (int)sqrt((double)MAX_NUM_VERTICES / (6.0 + (double)nParticles)) + 1;
  ndivs = std::max(ndivs, 6);

  LOG_INFO("Number of divisions: " + std::to_string(ndivs));

  std::vector<std::array<float, 3>> vertices;
  std::vector<std::array<float, 3>> normals;
  size_t nFaces = 0;

  {
    const int nDivsTheta = ndivs;
    const int nDivsPhi = 2 * ndivs;
    const float deltaTheta = M_PI / (float)(nDivsTheta - 1);
    const float deltaPhi = 2 * M_PI / (float)(nDivsPhi - 1);

    for (int iTheta = 0; iTheta < nDivsTheta - 1; ++iTheta) {
      for (int iPhi = 0; iPhi < nDivsPhi - 1; ++iPhi) {
        const float theta = (float)iTheta * deltaTheta;
        const float thetaNext = theta + deltaTheta;
        const float phi = (float)iPhi * deltaPhi;
        const float phiNext = phi + deltaPhi;

        const std::array<float, 3> position0 = {std::sin(theta) * std::cos(phi),
                                                std::sin(theta) * std::sin(phi),
                                                std::cos(theta)};
        const std::array<float, 3> position1 = {std::sin(thetaNext) * std::cos(phi),
                                                std::sin(thetaNext) * std::sin(phi),
                                                std::cos(thetaNext)};
        const std::array<float, 3> position2 = {std::sin(thetaNext) * std::cos(phiNext),
                                                std::sin(thetaNext) * std::sin(phiNext),
                                                std::cos(thetaNext)};
        const std::array<float, 3> position3 = {std::sin(theta) * std::cos(phiNext),
                                                std::sin(theta) * std::sin(phiNext),
                                                std::cos(theta)};

        vertices.push_back(position0);
        vertices.push_back(position1);
        vertices.push_back(position2);
        normals.push_back(position0);
        normals.push_back(position1);
        normals.push_back(position2);
        nFaces++;

        vertices.push_back(position2);
        vertices.push_back(position3);
        vertices.push_back(position0);
        normals.push_back(position2);
        normals.push_back(position3);
        normals.push_back(position0);
        nFaces++;
      }
    }
  }

  // ==============================================================================================
  // Save particles
  // ==============================================================================================
  if (std::ofstream file = std::ofstream(filePath)) {
    file << "o Generated particles\n";

    // write 'v'
    for (size_t iParticle = 0ULL; iParticle < nParticles; ++iParticle) {
      const auto& particle = particles[iParticle];

      const float centerX = particle.position.x;
      const float centerY = particle.position.y;
      const float centerZ = particle.position.z;
      const float radius = particle.radius;

      for (size_t iFace = 0ULL; iFace < nFaces; ++iFace) {
        const size_t offset = 3ULL * iFace;
        const size_t index0 = offset + 0ULL;
        const size_t index1 = offset + 1ULL;
        const size_t index2 = offset + 2ULL;

        file << "v " << (radius * vertices[index0][0] + centerX) << " " << (radius * vertices[index0][1] + centerY) << " " << (radius * vertices[index0][2] + centerZ) << "\n";
        file << "v " << (radius * vertices[index1][0] + centerX) << " " << (radius * vertices[index1][1] + centerY) << " " << (radius * vertices[index1][2] + centerZ) << "\n";
        file << "v " << (radius * vertices[index2][0] + centerX) << " " << (radius * vertices[index2][1] + centerY) << " " << (radius * vertices[index2][2] + centerZ) << "\n";
      }
    }

    // write 'vn'
    for (size_t iFace = 0ULL; iFace < nFaces; ++iFace) {
      const size_t offset = 3ULL * iFace;
      const size_t index0 = offset + 0ULL;
      const size_t index1 = offset + 1ULL;
      const size_t index2 = offset + 2ULL;

      file << "vn " << normals[index0][0] << " " << normals[index0][1] << " " << normals[index0][2] << "\n";
      file << "vn " << normals[index1][0] << " " << normals[index1][1] << " " << normals[index1][2] << "\n";
      file << "vn " << normals[index2][0] << " " << normals[index2][1] << " " << normals[index2][2] << "\n";
    }

    // write 'f'
    for (size_t iParticle = 0ULL; iParticle < nParticles; ++iParticle) {
      const size_t particleOffset = iParticle * nFaces * 3ULL;

      for (size_t iFace = 0ULL; iFace < nFaces; ++iFace) {
        const size_t faceOffset = 3ULL * iFace;
        const size_t offset = particleOffset + faceOffset;

        const size_t index0 = offset + 1ULL;
        const size_t index1 = offset + 2ULL;
        const size_t index2 = offset + 3ULL;

        file << "f ";
        file << index0 << "//" << faceOffset + 1LL << " ";
        file << index1 << "//" << faceOffset + 2LL << " ";
        file << index2 << "//" << faceOffset + 3LL << "\n";
      }
    }

    file.close();

    LOG_INFO("Finished writing to file " + filePath);
  } else {
    LOG_ERROR("Could not open file " + filePath);
  }
}
