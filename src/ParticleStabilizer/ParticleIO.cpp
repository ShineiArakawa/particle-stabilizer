#include <ParticleStabilizer/ParticleIO.hpp>
#include <Util/FileUtil.hpp>
#include <Util/Logging.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>

static std::vector<std::string> splitText(const std::string& s, const char delim) {
  std::vector<std::string> tokens;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    if (!item.empty() && item != "null") {
      tokens.push_back(item);
    }
  }
  return tokens;
}

ParticleData::ParticleData(const std::vector<ParticlePtr>& particles)
    : particles(particles) {}

ParticleData::~ParticleData() {
#ifdef DEBUG_BUILD
  LOG_INFO("Destruct DatData");
#endif
}

std::string ParticleData::format(const size_t& value, const int& nChars) {
  std::ostringstream oss;
  oss << std::setw(nChars) << std::setfill(' ') << value;

  const std::string str = oss.str();

  return str;
}

std::string ParticleData::format(const double& value, const int& nChars) {
  std::ostringstream oss;

  int integerPartLength = std::to_string(static_cast<int>(std::abs(value))).length();
  if (value < 0) {
    // Negative sign
    integerPartLength += 1;
  }

  int decimalPartLength = nChars - integerPartLength - 1;
  if (decimalPartLength < 0) {
    decimalPartLength = 0;
  }

  oss << std::fixed << std::setprecision(decimalPartLength) << value;

  return oss.str();
}

void ParticleData::writeAsCSV(const std::string& filePath) {
  LOG_INFO("Writing to file " + filePath);

  const std::string dirPath = FileUtil::dirPath(filePath);

  if (!FileUtil::exists(dirPath)) {
    FileUtil::mkdirs(dirPath);
  }

  if (std::ofstream file = std::ofstream(filePath)) {
    const size_t nParticles = particles.size();

    // Write the header
    file << "x,y,z,radius" << std::endl;

    // Write each particle
    for (size_t i = 0; i < nParticles; ++i) {
      const ParticlePtr& particle = particles[i];

      file << format(particle->position.x, 10) << ","
           << format(particle->position.y, 10) << ","
           << format(particle->position.z, 10) << ","
           << format(particle->radius, 10) << std::endl;
    }

    file.close();
    LOG_INFO("Finished writing to file " + filePath);
  } else {
    LOG_ERROR("Could not open file " + filePath);
  }
}

ParticleData ParticleData::readFromCSV(const std::string& filePath,
                                       const glm::dvec3& velocity,
                                       const double& density) {
  // =====================================================================================================
  // Read particles from file
  // =====================================================================================================
  std::vector<ParticlePtr> vec_particles;

  LOG_INFO("Reading particles from file: " + filePath);

  std::ifstream file(filePath);

  if (!file.is_open()) {
    LOG_CRITICAL("Could not open file: " + filePath);
    exit(EXIT_FAILURE);
  }

  std::string line;
  std::vector<std::string> tokens;

  // Read the header
  std::getline(file, line);

  // Read each particle
  // CSV format: x, y, z, radius

  while (std::getline(file, line)) {
    std::istringstream iss(line);

    tokens = splitText(line, ',');

    if (tokens.size() != 4) {
      LOG_ERROR("Invalid line: " + line);
      continue;
    }

    const double x = std::stod(tokens[0]);
    const double y = std::stod(tokens[1]);
    const double z = std::stod(tokens[2]);
    const double radius = std::stod(tokens[3]);

    const double mass = density * 4.0 / 3.0 * M_PI * std::pow(radius, 3);

    ParticlePtr particle = std::make_shared<Particle>(glm::dvec3(x, y, z), velocity, radius, mass);

    vec_particles.push_back(std::move(particle));
  }

  file.close();

  const size_t nParticles = vec_particles.size();
  LOG_INFO("Read " + std::to_string(nParticles) + " particles from file: " + filePath);

  return ParticleData(std::move(vec_particles));
}