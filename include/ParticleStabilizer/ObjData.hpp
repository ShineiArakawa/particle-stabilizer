#pragma once

#include <ParticleStabilizer/ParticleModel.hpp>
#include <fstream>
#include <string>
#include <vector>

class ObjData {
 public:
  static void write(const std::string& filePath, const std::vector<Particle>& particles);
};
