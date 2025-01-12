if (CUDAToolkit_FOUND)
    add_subdirectory(
        "ParticleStabilizerCUDA"
    )
endif()

add_subdirectory(
    "ParticleStabilizer"
)

add_subdirectory(
    "App"
)

