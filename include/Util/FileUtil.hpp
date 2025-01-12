#pragma once

#include <filesystem>
#include <string>

namespace generic_fs = std::filesystem;

class FileUtil {
  using Path_t = std::filesystem::path;

 public:
  static std::string join(const std::string basePath, const std::string additional) {
    Path_t path_basePath = generic_fs::absolute(Path_t(basePath));
    return (path_basePath / Path_t(additional)).string();
  }

  static std::string absPath(const std::string path) {
    return generic_fs::absolute(Path_t(path)).string();
  }

  static std::string dirPath(const std::string path) {
    return generic_fs::absolute(Path_t(path)).parent_path().string();
  }

  static std::string baseName(const std::string path) {
    return generic_fs::absolute(Path_t(path)).filename().string();
  }

  static std::string extension(const std::string path) {
    return Path_t(path).extension().string();
  }

  static std::string cwd() {
    return generic_fs::current_path().string();
  }

  static void mkdirs(const std::string path) {
    Path_t path_basePath = generic_fs::absolute(Path_t(path));
    generic_fs::create_directories(path_basePath);
  }

  static bool exists(const std::string path) {
    return generic_fs::exists(generic_fs::absolute(Path_t(path)));
  }

  static bool isFile(const std::string path) {
    return generic_fs::is_regular_file(generic_fs::absolute(Path_t(path)));
  }

  static bool isAbsolute(const std::string path) {
    return Path_t(path).is_absolute();
  }

  static std::string getTimeStamp() {
    const time_t t = time(NULL);
    const tm *local = localtime(&t);

    char buf[128];
    strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", local);

    std::string timeStamp(buf);

    return timeStamp;
  }
};
