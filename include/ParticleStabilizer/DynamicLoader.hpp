#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

class DynamicLoader {
 public:
  DynamicLoader(const char* libraryName);
  ~DynamicLoader();

  void* getSymbol(const char* symbolName);

 private:
#ifdef _WIN32
  HMODULE _handle;
#else
  void* _handle;
#endif
};
