#pragma once

#include <spdlog/spdlog.h>

#include <iostream>
#include <string>

#define LOG_TRACE(str) SPDLOG_TRACE(str)
#define LOG_DEBUG(str) SPDLOG_DEBUG(str)
#define LOG_INFO(str) SPDLOG_INFO(str)
#define LOG_WARN(str) SPDLOG_WARN(str)
#define LOG_ERROR(str) SPDLOG_ERROR(str)
#define LOG_CRITICAL(str) SPDLOG_CRITICAL(str)

// Define the loggers with stream input
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE
#define LOG_TRACE_STREAM(stream_input) \
  do {                                 \
    std::ostringstream oss;            \
    oss << stream_input;               \
    std::string message = oss.str();   \
    LOG_TRACE(message);                \
  } while (0)
#else
#define LOG_TRACE_STREAM(stream_input) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
#define LOG_DEBUG_STREAM(stream_input) \
  do {                                 \
    std::ostringstream oss;            \
    oss << stream_input;               \
    std::string message = oss.str();   \
    LOG_DEBUG(message);                \
  } while (0)
#else
#define LOG_DEBUG_STREAM(stream_input) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
#define LOG_INFO_STREAM(stream_input) \
  do {                                \
    std::ostringstream oss;           \
    oss << stream_input;              \
    std::string message = oss.str();  \
    LOG_INFO(message);                \
  } while (0)
#else
#define LOG_INFO_STREAM(stream_input) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_WARN
#define LOG_WARN_STREAM(stream_input) \
  do {                                \
    std::ostringstream oss;           \
    oss << stream_input;              \
    std::string message = oss.str();  \
    LOG_WARN(message);                \
  } while (0)
#else
#define LOG_WARN_STREAM(stream_input) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_ERROR
#define LOG_ERROR_STREAM(stream_input) \
  do {                                 \
    std::ostringstream oss;            \
    oss << stream_input;               \
    std::string message = oss.str();   \
    LOG_ERROR(message);                \
  } while (0)
#else
#define LOG_ERROR_STREAM(stream_input) (void)0
#endif

class Logging {
 private:
  // nothing
 public:
  inline static const std::string LOG_FORMAT = "[%Y-%m-%d %H:%M:%S] [%s:%#] [%!] [%l] %v";

  static void setFormat(std::string format) { spdlog::set_pattern(format); }

  static void setLevel(std::string str_level) {
    spdlog::level::level_enum level = spdlog::level::info;
    if (str_level == "trace") {
      level = spdlog::level::trace;
    } else if (str_level == "debug") {
      level = spdlog::level::debug;
    } else if (str_level == "info") {
      level = spdlog::level::info;
    } else if (str_level == "warn") {
      level = spdlog::level::warn;
    } else if (str_level == "error") {
      level = spdlog::level::err;
    } else if (str_level == "critical") {
      level = spdlog::level::critical;
    } else {
      std::cerr << "Unknown log level: " << str_level << std::endl;
      std::cerr << "Proceeding with log level: " << level << std::endl;
    }
    spdlog::set_level(level);
  }
};
