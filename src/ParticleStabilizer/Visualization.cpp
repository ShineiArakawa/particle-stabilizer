
#include <ParticleStabilizer/Visualization.hpp>

simview::app::ViewerGUIApp_t launchViewer(std::vector<simview::model::Sphere_t>& spheres,
                                          const std::shared_ptr<ParticleModel>& particleModel,
                                          const std::shared_ptr<ContainerBase>& container) {
  auto viewer = std::make_shared<simview::app::ViewerGUIApp>();
  const glm::dvec3 modelMin = container->getMinCoords();
  const glm::dvec3 modelMax = container->getMaxCoords();

  {
    // ========================================================================================================================
    // Add box
    // ========================================================================================================================
    simview::model::Primitive_t model = container->getSimviewPrimitive();
    model->setVisible(false);
    viewer->addObject(model, false);
  }

  {
    // ========================================================================================================================
    // Add box wireframe
    // ========================================================================================================================
    std::shared_ptr<std::vector<simview::vec3f_t>> boxVertices = std::make_shared<std::vector<simview::vec3f_t>>();

    {
      boxVertices->push_back({(float)modelMin.x, (float)modelMin.y, (float)modelMin.z});
      boxVertices->push_back({(float)modelMax.x, (float)modelMin.y, (float)modelMin.z});

      boxVertices->push_back({(float)modelMax.x, (float)modelMin.y, (float)modelMin.z});
      boxVertices->push_back({(float)modelMax.x, (float)modelMax.y, (float)modelMin.z});

      boxVertices->push_back({(float)modelMax.x, (float)modelMax.y, (float)modelMin.z});
      boxVertices->push_back({(float)modelMin.x, (float)modelMax.y, (float)modelMin.z});

      boxVertices->push_back({(float)modelMin.x, (float)modelMax.y, (float)modelMin.z});
      boxVertices->push_back({(float)modelMin.x, (float)modelMin.y, (float)modelMin.z});
    }

    {
      boxVertices->push_back({(float)modelMin.x, (float)modelMin.y, (float)modelMax.z});
      boxVertices->push_back({(float)modelMax.x, (float)modelMin.y, (float)modelMax.z});

      boxVertices->push_back({(float)modelMax.x, (float)modelMin.y, (float)modelMax.z});
      boxVertices->push_back({(float)modelMax.x, (float)modelMax.y, (float)modelMax.z});

      boxVertices->push_back({(float)modelMax.x, (float)modelMax.y, (float)modelMax.z});
      boxVertices->push_back({(float)modelMin.x, (float)modelMax.y, (float)modelMax.z});

      boxVertices->push_back({(float)modelMin.x, (float)modelMax.y, (float)modelMax.z});
      boxVertices->push_back({(float)modelMin.x, (float)modelMin.y, (float)modelMax.z});
    }

    {
      boxVertices->push_back({(float)modelMin.x, (float)modelMin.y, (float)modelMin.z});
      boxVertices->push_back({(float)modelMin.x, (float)modelMin.y, (float)modelMax.z});

      boxVertices->push_back({(float)modelMax.x, (float)modelMin.y, (float)modelMin.z});
      boxVertices->push_back({(float)modelMax.x, (float)modelMin.y, (float)modelMax.z});

      boxVertices->push_back({(float)modelMax.x, (float)modelMax.y, (float)modelMin.z});
      boxVertices->push_back({(float)modelMax.x, (float)modelMax.y, (float)modelMax.z});

      boxVertices->push_back({(float)modelMin.x, (float)modelMax.y, (float)modelMin.z});
      boxVertices->push_back({(float)modelMin.x, (float)modelMax.y, (float)modelMax.z});
    }

    simview::model::LineSet_t lineSet = std::make_shared<simview::model::LineSet>(boxVertices,
                                                                                  glm::vec3(0.0f, 1.0f, 0.0f),
                                                                                  1.0f);

    viewer->addObject(lineSet);
  }

  // ========================================================================================================================
  // Add spheres
  // ========================================================================================================================
  const size_t nParticles = particleModel->getParticles().size();

  const size_t MAX_NUM_VERTICES = 1000000;
  int ndivs = (int)sqrt((double)MAX_NUM_VERTICES / (6.0 + (double)nParticles)) + 1;
  ndivs = std::max(ndivs, 3);

  LOG_INFO("Number of divisions: " + std::to_string(ndivs));

  for (const auto& particle : particleModel->getParticles()) {
    simview::model::Sphere_t sphere = std::make_shared<simview::model::Sphere>(ndivs,
                                                                               0.0,
                                                                               0.0,
                                                                               0.0,
                                                                               2.0 * particle->radius,
                                                                               2.0 * particle->radius,
                                                                               2.0 * particle->radius,
                                                                               glm::vec3(0.0f, 0.0f, 1.0f));

    spheres.push_back(sphere);
    viewer->addObject(sphere);
  }

  // ========================================================================================================================
  // Set viewer properties
  // ========================================================================================================================
  viewer->setRenderType(simview::model::Primitive::RenderType::SHADE);
  viewer->setSideBarVisibility(false);
  viewer->setCameraPose({-2.0 * (modelMax.z - modelMin.z), 0.0f, (float)modelMax.z},
                        {0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 1.0f});
  viewer->setWindowSubTitle("Static Particles");

  return viewer;
}

void updateSpheres(const std::vector<ParticlePtr>& particles,
                   std::vector<simview::model::Sphere_t>& spheres) {
  const size_t nSpheres = spheres.size();

  for (size_t i = 0; i < nSpheres; ++i) {
    spheres[i]->setPosition(particles[i]->position.x,
                            particles[i]->position.y,
                            particles[i]->position.z);
  }
}
