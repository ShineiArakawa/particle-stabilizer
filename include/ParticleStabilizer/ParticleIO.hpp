#pragma once

#include <ParticleStabilizer/ParticleModel.hpp>
#include <Util/Math.hpp>
#include <fstream>
#include <string>
#include <vector>

class ParticleData {
 private:
  static std::string format(const size_t& value, const int& nChars);
  static std::string format(const double& value, const int& nChars);

 public:
  ParticleData(const std::vector<ParticlePtr>& particles);

  ~ParticleData();

  std::vector<ParticlePtr> particles;

  void writeAsCSV(const std::string& filePath);
  static ParticleData readFromCSV(const std::string& filePath,
                                  const glm::dvec3& velocity,
                                  const double& density);
};
