#pragma once

#include <filesystem>
#include <string>

namespace saving_utils
{
std::filesystem::path get_state_directory();
std::string make_timestamp();
}  // namespace saving_utils
