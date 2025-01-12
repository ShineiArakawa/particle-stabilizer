#pragma once

#include <ParticleStabilizer/ContainerBase.hpp>
#include <ParticleStabilizer/ParticleModel.hpp>
#include <ParticleStabilizer/PhysicsEngine.hpp>
#include <SimView/core.hpp>

simview::app::ViewerGUIApp_t launchViewer(std::vector<simview::model::Sphere_t>& spheres,
                                          const std::shared_ptr<ParticleModel>& particleModel,
                                          const std::shared_ptr<ContainerBase>& container);

void updateSpheres(const std::vector<ParticlePtr>& particles,
                   std::vector<simview::model::Sphere_t>& spheres);