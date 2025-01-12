project(particle_stabilizer_cuda CXX CUDA)

add_library(
  ${PROJECT_NAME}
  SHARED
  "PhysicsEngineCUDA.cpp"
  "ParticleModelCUDA.cpp"
  "BoxContainerCUDA.cpp"
  "ParticleStatisticsCUDA.cpp"
  "VisualizationCUDA.cpp"
  "PolygonContainerCUDA.cpp"
  "BoxContainer_kernels.cu"
  "ParticleModel_kernels.cu"
  "PhysicsEngine_kernels.cu"
  "ParticleStatisticsCUDA_kernels.cu"
  "PrefixScan.cu"
  "PolygonContainerCUDA_kernels.cu"
)

# =========================================================
# Set Libraries ===========================================
# =========================================================
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION
  ON
)

set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS
  ON
)

# Include
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
  ${PROJECT_INCLUDE_DIR}
  ${EXTERNAL_INCLUDE_DIR}
)

# Link
target_link_directories(
  ${PROJECT_NAME}
  PRIVATE
  ${CUDAToolkit_LIBRARY_DIR}
)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
  CUDA::cudart_static
  ${EXTERNAL_LIBS}
)

############################################################################################################
# Install
############################################################################################################
install(
  	DIRECTORY "${PROJECT_INCLUDE_DIR}/ParticleStabilizerCUDA"
  	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
	  FILES_MATCHING PATTERN "*.hpp"
)

install(
  	DIRECTORY "${PROJECT_INCLUDE_DIR}/ParticleStabilizerCUDA"
  	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
	  FILES_MATCHING PATTERN "*.cuh"
)

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_ARCHITECTURE_NAME}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/${PROJECT_ARCHITECTURE_NAME}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_ARCHITECTURE_NAME}"
)