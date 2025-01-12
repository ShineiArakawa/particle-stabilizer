# particle-stabilizer

![demo](assets/demo.gif)

**particle-stabilizer** is a C++ and CUDA-based program for simulating the motion of particles. It is designed for high-performance simulations, leveraging the power of GPU acceleration to handle large-scale particle systems efficiently.

## Features

- **GPU Acceleration:** Utilizes CUDA for efficient computation.
- **High Performance:** Optimized for large-scale particle simulations.
- **Flexible Configuration:** Easily customizable parameters for diverse simulation scenarios.
- **Visualization Ready:** Includes built-in tools for real-time and post-simulation visualization of particle movements and interactions.

## Requirements

To build and run particle-stabilizer, ensure the following dependencies are installed:

- **C++ Compiler** supporting C++17 or later (e.g., GCC, Clang, MSVC).
- **CUDA Toolkit** (tested on version 12.4).
- **CMake** (tested on version 3.30).
- **NVIDIA GPU** (tested on GeForce RTX 3090).

## Build and Run

1. Clone the repository:
   ```bash
   git clone --recursive https://github.com/ShineiArakawa/particle-stabilizer
   cd particle-stabilizer
   ```

2. Create a build directory:
   ```bash
   mkdir build && cd build
   ```

3. Configure the build system with CMake:
   ```bash
   cmake ..
   ```

4. Build the project:
   ```bash
   make
   ```

5. Run the program:
   ```bash
   ./src/App/particle_stabilizer_main ../samples/sample.csv --visualize
   ```

### Command-Line Options

| Option                                                            | Description                                                                                                                             |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **General**                                                       |                                                                                                                                         |
| `-h, --help`                                                      | Display help information                                                                                                                |
| `--cuda`                                                          | Use CUDA for the simulation                                                                                                             |
| **Conditions for the particles**                                  |                                                                                                                                         |
| `--initial-velocity X Y Z`                                        | Set the initial velocity of the particles                                                                                               |
| `--density MASS`                                                  | Define the density of the particles                                                                                                     |
| **Conditions for the simulation**                                 |                                                                                                                                         |
| `--coeff-of-restitution-wall Z_MIN Z_MAX X_MIN X_MAX Y_MIN Y_MAX` | Coefficient of restitution for the walls                                                                                                |
| `--coeff-of-spring SPRING`                                        | Coefficient of spring                                                                                                                   |
| `--coeff-of-restitution-sphere RESTITUTION`                       | Coefficient of restitution for the spheres                                                                                              |
| `--gravity X Y Z`                                                 | Define the gravity vector                                                                                                               |
| `--dt DT`                                                         | Set the time step for the simulation                                                                                                    |
| `--max-steps MAX_STEPS`                                           | Maximum number of simulation steps                                                                                                      |
| **Container**                                                     |                                                                                                                                         |
| `--box-min X Y Z`                                                 | Minimum coordinates of the box container                                                                                                |
| `--box-max X Y Z`                                                 | Maximum coordinates of the box container                                                                                                |
| `--container CONTAINER_PATH`                                      | Path to the container file. When specified, 0th value of `--coeff-of-restitution-wall` will be used for polygon restitution coefficient |
| **Visualization**                                                 |                                                                                                                                         |
| `--visualize`                                                     | Enable real-time visualization of the simulation (may reduce performance)                                                               |
| **Postprocess**                                                   |                                                                                                                                         |
| `--resolve-overlap-axis AXIS`                                     | Resolve overlaps along the specified axis                                                                                               |
| **Save**                                                          |                                                                                                                                         |
| `--out-obj-path OBJ_PATH`                                         | Path to save the simulation as a `.obj` file                                                                                            |
| `--out-csv-path CSV_PATH`                                         | Path to save the simulation as a `.csv` file                                                                                            |

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
