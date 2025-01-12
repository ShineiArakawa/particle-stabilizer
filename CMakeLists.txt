cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)
endif()

project(ParticleStabilizer LANGUAGES C CXX)

############################################################################################################
# Compile Options 
############################################################################################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE on)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
elseif(UNIX)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(-DDEBUG_BUILD)
endif()

############################################################################################################
# Envirionment Variables
############################################################################################################
set(PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(PROJECT_EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/external")

# Install
set(ORIGINAL_SYSTEM_NAME ${CMAKE_SYSTEM_NAME})
string(TOLOWER ${ORIGINAL_SYSTEM_NAME} LOWERCASE_SYSTEM_NAME)

set(ORIGINAL_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR})
string(TOLOWER ${ORIGINAL_SYSTEM_PROCESSOR} LOWERCASE_SYSTEM_PROCESSOR)

set(PROJECT_ARCHITECTURE_NAME "${LOWERCASE_SYSTEM_PROCESSOR}-${LOWERCASE_SYSTEM_NAME}")

if (NOT CMAKE_INSTALL_LIBDIR)
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()

if (NOT CMAKE_INSTALL_DATADIR)
set(CMAKE_INSTALL_DATADIR "share")
endif()

if (NOT CMAKE_INSTALL_INCLUDEDIR)
  set(CMAKE_INSTALL_INCLUDEDIR "include")
endif()

if (NOT CMAKE_INSTALL_BINDIR)
  set(CMAKE_INSTALL_BINDIR "bin")
endif()

############################################################################################################
# External Libraries
############################################################################################################
add_subdirectory(external)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  message(STATUS "OpenMP found")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# CUDA
find_package(CUDAToolkit)

set(VISUALIZER_LIBS SimView_static)

set(
  EXTERNAL_INCLUDE_DIR
  ${SimpleObjectViewer_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${SPD_LOG_INCLUDE_DIR}
  ${ASSIMP_INCLUDE_DIR}
  ${GLM_INCLUDE_DIR}
)

set(
  EXTERNAL_LIBS
  ${VISUALIZER_LIBS}
  ${ASSIMP_LIBS}
)

############################################################################################################
# Add subdirectory
############################################################################################################
add_subdirectory(src)

############################################################################################################
# Message
############################################################################################################
message(STATUS "# =======================================================================================================")
message(STATUS "#  '${PROJECT_NAME}' build configuration ")
message(STATUS "# =======================================================================================================")
message(STATUS "#  [CMake]")
message(STATUS "#    CMake version                        : ${CMAKE_VERSION}")
message(STATUS "#    CMake build type                     : ${CMAKE_BUILD_TYPE}")
message(STATUS "#    CMAKE_CXX_FLAGS                      : ${CMAKE_CXX_FLAGS}")
message(STATUS "#    CMAKE_CXX_FLAGS_DEBUG                : ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "#    CMAKE_CXX_FLAGS_RELEASE              : ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "#    CMAKE_INSTALL_PREFIX                 : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "# ")
message(STATUS "#  [C/C++]")
message(STATUS "#    C   Compiler                         : ${CMAKE_C_COMPILER_ID} | ${CMAKE_C_COMPILER_VERSION} | ${CMAKE_C_COMPILER}")
message(STATUS "#    C++ Compiler                         : ${CMAKE_CXX_COMPILER_ID} | ${CMAKE_CXX_COMPILER_VERSION} | ${CMAKE_CXX_COMPILER}")
message(STATUS "#    CXX STANDARD                         : ${CMAKE_CXX_STANDARD}")
message(STATUS "# ")
message(STATUS "# [CUDA]")
message(STATUS "#    CMAKE_CUDA_COMPILER                  : ${CMAKE_CUDA_COMPILER}")
message(STATUS "#    CMAKE_CUDA_ARCHITECTURES             : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "#    CUDA_ARCH_LIST                       : ${CUDA_ARCH_LIST}")
message(STATUS "#    CUDA_NVCC_FLAGS                      : ${CUDA_NVCC_FLAGS}")
message(STATUS "# ")
message(STATUS "# [OpenMP]")
message(STATUS "#    OpenMP_C_FLAGS                       : ${OpenMP_C_FLAGS}")
message(STATUS "#    OpenMP_CXX_FLAGS                     : ${OpenMP_CXX_FLAGS}")
message(STATUS "# ")
message(STATUS "#  [Environment Variables]")
message(STATUS "#    PROJECT_INCLUDE_DIR                  : ${PROJECT_INCLUDE_DIR}")
message(STATUS "#    PROJECT_EXTERNAL_DIR                 : ${PROJECT_EXTERNAL_DIR}")
message(STATUS "#    PROJECT_ARCHITECTURE_NAME            : ${PROJECT_ARCHITECTURE_NAME}")
message(STATUS "# ")
message(STATUS "#  [Install]")
message(STATUS "#    CMAKE_INSTALL_PREFIX                 : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "#    CMAKE_INSTALL_BINDIR                 : ${CMAKE_INSTALL_BINDIR}")
message(STATUS "#    CMAKE_INSTALL_LIBDIR                 : ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "# =======================================================================================================")
