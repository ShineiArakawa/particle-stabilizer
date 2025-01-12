project(particle_stabilizer_main CXX)

add_executable(
  ${PROJECT_NAME}
  "ParticleStabilizerMain.cpp"
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
  $<TARGET_OBJECTS:particle_stabilizer>
)

############################################################################################################
# Install
############################################################################################################
install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_ARCHITECTURE_NAME}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/${PROJECT_ARCHITECTURE_NAME}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_ARCHITECTURE_NAME}"
)
