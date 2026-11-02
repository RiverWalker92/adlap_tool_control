#include "adlap_tool_control/saving_utils.hpp"

#include <cstdlib>
#include <ctime>

namespace saving_utils
{
std::filesystem::path get_state_directory()
{
  if (const char* ros_home = std::getenv("ROS_HOME"))
  {
    return std::filesystem::path(ros_home) / "adlap_tool_control";
  }
  if (const char* home = std::getenv("HOME"))
  {
    return std::filesystem::path(home) / ".ros" / "adlap_tool_control";
  }
  return std::filesystem::path(".") / "adlap_tool_control";
}

std::string make_timestamp()
{
  std::time_t now = std::time(nullptr);
  std::tm tm{};
  localtime_r(&now, &tm);
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm);
  return std::string(buffer);
}
}  // namespace saving_utils
