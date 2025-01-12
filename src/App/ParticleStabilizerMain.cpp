#include <ParticleStabilizer/ContainerBase.hpp>
#include <ParticleStabilizer/DynamicLoader.hpp>
#include <ParticleStabilizer/ObjData.hpp>
#include <ParticleStabilizer/ParticleIO.hpp>
#include <ParticleStabilizer/PhysicsEngine.hpp>
#include <ParticleStabilizer/Visualization.hpp>
#include <Util/Logging.hpp>

// =================================================================================================
// Type definitions of the shred library
// =================================================================================================
#if defined(_WIN64) && defined(DEBUG_BUILD)  // Windows Debug
constexpr const char* CUDA_SHARED_LIB = "particle_stabilizer_cudad.dll";
#elif defined(_WIN64)       // Windows Release
constexpr const char* CUDA_SHARED_LIB = "particle_stabilizer_cuda.dll";
#elif defined(__APPLE__)    // Apple
constexpr const char* CUDA_SHARED_LIB = "particle_stabilizer_cuda.dylib";
#elif defined(DEBUG_BUILD)  // Linux Debug
constexpr const char* CUDA_SHARED_LIB = "./libparticle_stabilizer_cuda.so";
#else                       // Linux Release
constexpr const char* CUDA_SHARED_LIB = "./libparticle_stabilizer_cuda.so";
#endif

using runSimulateCUDA_t = void (*)(const std::vector<ParticlePtr>&,
                                   const glm::dvec3,
                                   const glm::dvec3,
                                   const std::string,
                                   const glm::dvec3,
                                   const CoefficientOfRestitutionWall,
                                   const double,
                                   const double,
                                   const double,
                                   const size_t,
                                   const bool,
                                   std::vector<Particle>&);

// =================================================================================================
// Argument parser
// =================================================================================================
struct Argument {
  Argument()
      : particleFilePath(),
        objOutPath(),
        initialVelocity({0.0, 0.0, -10.0}),
        density(1.0),
        coefficientOfRestitutionWall(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        coefficientOfSpring(1.0),
        coefficientOfRestitutionSphere(0.1),
        gravity({0.0, 0.0, -10.0}),
        boxMin({0.0, 0.0, 0.0}),
        boxMax({10.0, 20.0, 30.0}),
        containerPath(),
        dt(0.00001),
        toVisualize(false),
        maxSteps(100000),
        resolveOverlapAxis(-1),
        toUseCUDA(false) {};

  std::string particleFilePath;
  std::string objOutPath;
  std::string particleOutPath;
  glm::dvec3 initialVelocity;
  double density;
  CoefficientOfRestitutionWall coefficientOfRestitutionWall;
  double coefficientOfSpring;
  double coefficientOfRestitutionSphere;
  glm::dvec3 gravity;
  glm::dvec3 boxMin;
  glm::dvec3 boxMax;
  std::string containerPath;
  double dt;
  bool toVisualize;
  int64_t maxSteps;
  int resolveOverlapAxis;
  bool toUseCUDA;

  void print() const {
    LOG_INFO("### Argument");
    LOG_INFO("  particleFilePath               : " + particleFilePath);
    LOG_INFO("  objOutPath                     : " + objOutPath);
    LOG_INFO("  particleOutPath                : " + particleOutPath);
    LOG_INFO("  initialVelocity                : " + glm::to_string(initialVelocity));
    LOG_INFO("  density                        : " + std::to_string(density));
    LOG_INFO("  coefficientOfRestitutionWall   : " + coefficientOfRestitutionWall.toString());
    LOG_INFO("  coefficientOfSpring            : " + std::to_string(coefficientOfSpring));
    LOG_INFO("  coefficientOfRestitutionSphere : " + std::to_string(coefficientOfRestitutionSphere));
    LOG_INFO("  gravity                        : " + glm::to_string(gravity));
    LOG_INFO("  boxMin                         : " + glm::to_string(boxMin));
    LOG_INFO("  boxMax                         : " + glm::to_string(boxMax));
    LOG_INFO("  containerPath                  : " + containerPath);
    LOG_INFO("  dt                             : " + std::to_string(dt));
    LOG_INFO("  toVisualize                    : " + std::string(toVisualize ? "true" : "false"));
    LOG_INFO("  maxSteps                       : " + std::to_string(maxSteps));
    LOG_INFO("  resolveOverlapAxis             : " + std::to_string(resolveOverlapAxis));
    LOG_INFO("  toUseCUDA                      : " + std::string(toUseCUDA ? "true" : "false"));
  }

  static Argument parseArgs(int argc, char* argv[]) {
    Argument args;

    bool toShowHelp = false;

    if (argc < 2) {
      toShowHelp = true;
    }

    for (int i = 1; i < argc; ++i) {
      std::string arg = std::string(argv[i]);

      if (arg == "-h") {
        toShowHelp = true;
        break;
      } else if (arg == "--out-obj-path") {
        args.objOutPath = std::string(argv[++i]);
      } else if (arg == "--out-csv-path") {
        args.particleOutPath = std::string(argv[++i]);
      } else if (arg == "--initial-velocity") {
        args.initialVelocity = {std::stod(argv[++i]), std::stod(argv[++i]), std::stod(argv[++i])};
      } else if (arg == "--density") {
        args.density = std::stod(argv[++i]);
      } else if (arg == "--coeff-of-restitution-wall") {
        args.coefficientOfRestitutionWall = CoefficientOfRestitutionWall(std::stod(argv[++i]),
                                                                         std::stod(argv[++i]),
                                                                         std::stod(argv[++i]),
                                                                         std::stod(argv[++i]),
                                                                         std::stod(argv[++i]),
                                                                         std::stod(argv[++i]));
      } else if (arg == "--coeff-of-spring") {
        args.coefficientOfSpring = std::stod(argv[++i]);
      } else if (arg == "--coeff-of-restitution-sphere") {
        args.coefficientOfRestitutionSphere = std::stod(argv[++i]);
      } else if (arg == "--gravity") {
        args.gravity = {std::stod(argv[++i]), std::stod(argv[++i]), std::stod(argv[++i])};
      } else if (arg == "--box-min") {
        args.boxMin = {std::stod(argv[++i]), std::stod(argv[++i]), std::stod(argv[++i])};
      } else if (arg == "--box-max") {
        args.boxMax = {std::stod(argv[++i]), std::stod(argv[++i]), std::stod(argv[++i])};
      } else if (arg == "--container") {
        args.containerPath = std::string(argv[++i]);
      } else if (arg == "--dt") {
        args.dt = std::stod(argv[++i]);
      } else if (arg == "--visualize") {
        args.toVisualize = true;
      } else if (arg == "--max-steps") {
        args.maxSteps = std::stoll(argv[++i]);
      } else if (arg == "--resolve-overlap-axis") {
        args.resolveOverlapAxis = std::stoi(argv[++i]);
      } else if (arg == "--cuda") {
        args.toUseCUDA = true;
      } else {
        args.particleFilePath = arg;
      }
    }

    if (args.particleFilePath.empty()) {
      toShowHelp = true;
    }

    if (toShowHelp) {
      std::cout << "################################################# Particle Stabilizer #################################################\n";
      std::cout << "                                                                                                                       \n";
      std::cout << "A simulator for particle dynamics in a container.                                                                      \n";
      std::cout << "                                                                                                                       \n";
      std::cout << "usege: ./particle_stabilizer_main [Options] particleFilePath                                                           \n";
      std::cout << "                                                                                                                       \n";
      std::cout << "[Options]                                                                                                              \n";
      std::cout << "  General                                                                                                              \n";
      std::cout << "    -h                                                                  Show this help message                         \n";
      std::cout << "    --cuda                                                              Use CUDA for the simulation                    \n";
      std::cout << "                                                                                                                       \n";
      std::cout << "  Particle Constants                                                                                                   \n";
      std::cout << "    --initial-velocity             X Y Z                                Initial velocity of the particles              \n";
      std::cout << "    --density                      MASS                                 Density of the particles                       \n";
      std::cout << "                                                                                                                       \n";
      std::cout << "  Simulation Constants                                                                                                 \n";
      std::cout << "    --coeff-of-restitution-wall    Z_MIN Z_MAX X_MIN X_MAX Y_MIN Y_MAX  Coefficient of restitution for the wall        \n";
      std::cout << "    --coeff-of-spring              SPRING                               Coefficient of spring                          \n";
      std::cout << "    --coeff-of-restitution-sphere  RESTITUTION                          Coefficient of restitution for the sphere      \n";
      std::cout << "    --gravity                      X Y Z                                Gravity                                        \n";
      std::cout << "    --dt                           DT                                   Time step                                      \n";
      std::cout << "    --max-steps                    MAX_STEPS                            Maximum number of steps                        \n";
      std::cout << "                                                                                                                       \n";
      std::cout << "  Box Container                                                                                                        \n";
      std::cout << "    --box-min                      X Y Z                                Minimum coordinates of the box container       \n";
      std::cout << "    --box-max                      X Y Z                                Maximum coordinates of the box container       \n";
      std::cout << "    --container                    CONTAINER_PATH                       Path to the container file.                    \n";
      std::cout << "                                                                        If this option is specified, 0th value of      \n";
      std::cout << "                                                                        '--coeff-of-restitution-wall' will be used for \n";
      std::cout << "                                                                        the polygon restitution coefficient.           \n";
      std::cout << "                                                                                                                       \n";
      std::cout << "  Visualization                                                                                                        \n";
      std::cout << "    --visualize                                                         Visualize the simulation                       \n";
      std::cout << "                                                                        (NOTICE: Significant performance drop)         \n";
      std::cout << "                                                                                                                       \n";
      std::cout << "  Postprocess                                                                                                          \n";
      std::cout << "    --resolve-overlap-axis         AXIS                                 Resolve overlaps along the specified axis      \n";
      std::cout << "                                                                                                                       \n";
      std::cout << "  Save                                                                                                                 \n";
      std::cout << "    --out-obj-path                 OBJ_PATH                             Path to save the simulation as .obj file       \n";
      std::cout << "    --out-csv-path                 CSV_PATH                             Path to save the simulation as .csv file       \n";
      std::cout << "                                                                                                                       \n";
      exit(EXIT_SUCCESS);
    }

    return args;
  }
};

// =================================================================================================
// Main function
// =================================================================================================
int main(int argc, char* argv[]) {
  // =================================================================================================
  // Parse the arguments
  // =================================================================================================
  const Argument args = Argument::parseArgs(argc, argv);
  args.print();

  // Read particle data
  // NOTE: If you want to read the particles from other file formats, you need to implement the reader.
  ParticleData particleData = ParticleData::readFromCSV(args.particleFilePath,
                                                        args.initialVelocity,
                                                        args.density);

  // Create particle container for results
  std::vector<Particle> resultParticles(particleData.particles.size());

  if (args.toUseCUDA) {
    // =================================================================================================
    // CUDA implementation
    // =================================================================================================
    // Load the CUDA shared library
    LOG_INFO("Using CUDA for the simulation.");

    DynamicLoader loader(CUDA_SHARED_LIB);

    runSimulateCUDA_t func_runSimulateCUDA = reinterpret_cast<runSimulateCUDA_t>(loader.getSymbol("runSimulateCUDA"));

    if (!func_runSimulateCUDA) {
      LOG_CRITICAL("Cannot load symbol: 'runSimulateCUDA' from the shared library.");
      exit(EXIT_FAILURE);
    }

    // Run the simulation
    func_runSimulateCUDA(particleData.particles,
                         args.boxMin,
                         args.boxMax,
                         args.containerPath,
                         args.gravity,
                         args.coefficientOfRestitutionWall,
                         args.coefficientOfSpring,
                         args.coefficientOfRestitutionSphere,
                         args.dt,
                         args.maxSteps,
                         args.toVisualize,
                         resultParticles);
  } else {
    // =================================================================================================
    // CPU implementation
    // =================================================================================================
    runSimulate(particleData.particles,
                args.boxMin,
                args.boxMax,
                args.containerPath,
                args.gravity,
                args.coefficientOfRestitutionWall,
                args.coefficientOfSpring,
                args.coefficientOfRestitutionSphere,
                args.dt,
                args.maxSteps,
                args.toVisualize,
                resultParticles);
  }

  if (args.resolveOverlapAxis >= 0) {
    // Resolve overlaps along the specified axis
    ParticleModel::resolveOverlaps(resultParticles, args.resolveOverlapAxis);
  }

  // Update particleData with the result particles
  for (size_t iParticle = 0ULL; iParticle < particleData.particles.size(); ++iParticle) {
    particleData.particles[iParticle]->position = resultParticles[iParticle].position;
    particleData.particles[iParticle]->velocity = resultParticles[iParticle].velocity;
    particleData.particles[iParticle]->radius = resultParticles[iParticle].radius;
    particleData.particles[iParticle]->mass = resultParticles[iParticle].mass;
  }

  // Save the simulation result as .csv file
  if (!args.particleOutPath.empty()) {
    particleData.writeAsCSV(args.particleOutPath);
  }

  // Save the simulation result as .obj file
  if (!args.objOutPath.empty()) {
    ObjData::write(args.objOutPath, resultParticles);
  }

  LOG_INFO("Simulation completed successfully.");

  return EXIT_SUCCESS;
}
