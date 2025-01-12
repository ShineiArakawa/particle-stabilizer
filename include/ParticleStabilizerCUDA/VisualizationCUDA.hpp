#pragma once

#include <ParticleStabilizerCUDA/ContainerBaseCUDA.hpp>
#include <ParticleStabilizerCUDA/ParticleModelCUDA.hpp>
#include <SimView/core.hpp>

#ifdef __cplusplus
extern "C" {
#endif

// C linkage can not return pointer, in MXVC
void launchViewerCUDA(std::vector<simview::model::Sphere_t>& spheres,
                      const std::shared_ptr<ParticleModelCUDA>& particleModel,
                      const std::shared_ptr<ContainerBaseCUDA>& container,
                      simview::app::ViewerGUIApp_t& viewer);

void updateSpheresCUDA(const std::shared_ptr<ParticleModelCUDA>& particleModel,
                       std::vector<simview::model::Sphere_t>& spheres);

#ifdef __cplusplus
}
#endif
