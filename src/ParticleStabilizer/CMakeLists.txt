project(particle_stabilizer CXX)

add_library(
  ${PROJECT_NAME}
  STATIC
  "PhysicsEngine.cpp"
  "ParticleModel.cpp"
  "BoxContainer.cpp"
  "ParticleBucket.cpp"
  "ParticleStatistics.cpp"
  "Visualization.cpp"
  "DynamicLoader.cpp"
  "ParticleIO.cpp"
  "ObjData.cpp"
  "PolygonContainer.cpp"
)

# =========================================================
# Set Libraries ===========================================
# =========================================================
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
  ${PROJECT_INCLUDE_DIR}
  ${EXTERNAL_INCLUDE_DIR}
)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
  ${EXTERNAL_LIBS}
)

############################################################################################################
# Install
############################################################################################################
install(
  	DIRECTORY "${PROJECT_INCLUDE_DIR}/ParticleStabilizer"
  	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
	  FILES_MATCHING PATTERN "*.hpp"
)

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_ARCHITECTURE_NAME}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/${PROJECT_ARCHITECTURE_NAME}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_ARCHITECTURE_NAME}"
)
