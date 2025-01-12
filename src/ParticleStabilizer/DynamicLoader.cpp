#include <ParticleStabilizer/DynamicLoader.hpp>
#include <Util/Logging.hpp>
#include <iostream>

DynamicLoader::DynamicLoader(const char* libraryName) {
#ifdef _WIN32
  _handle = LoadLibrary(libraryName);
  if (!_handle) {
    LOG_ERROR("Cannot open library: " + std::to_string(GetLastError()));
  }
#else
  _handle = dlopen(libraryName, RTLD_LAZY);
  if (!_handle) {
    LOG_ERROR("Cannot open library: " + std::string(dlerror()));
  }
#endif
}

DynamicLoader::~DynamicLoader() {
  if (_handle) {
#ifdef _WIN32
    FreeLibrary(_handle);
#else
    dlclose(_handle);
#endif
  }
}

void* DynamicLoader::getSymbol(const char* symbolName) {
  if (!_handle) return nullptr;

#ifdef _WIN32
  void* symbol = GetProcAddress(_handle, symbolName);
  if (!symbol) {
    LOG_ERROR("Cannot load symbol: " + std::to_string(GetLastError()));
  }
#else
  void* symbol = dlsym(_handle, symbolName);
  const char* dlsym_error = dlerror();
  if (dlsym_error) {
    LOG_ERROR("Cannot load symbol: " + std::string(dlsym_error));
  }
#endif

  return symbol;
}
