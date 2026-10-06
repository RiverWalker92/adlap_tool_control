#include "adlap_tool_control/motor_controller.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <string>
#include <system_error>
#include <time.h>

namespace
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

nlohmann::json make_position_json(const std::array<int, 4>& positions)
{
  return nlohmann::json({positions[0], positions[1], positions[2], positions[3]});
}

std::optional<std::array<int, 4>> parse_positions_field(const nlohmann::json& json, const char* field_name)
{
  if (!json.contains(field_name) || !json[field_name].is_array() || json[field_name].size() != 4)
  {
    return std::nullopt;
  }

  std::array<int, 4> values{0, 0, 0, 0};
  for (size_t i = 0; i < 4; ++i)
  {
    if (!json[field_name][i].is_number_integer())
    {
      return std::nullopt;
    }
    values[i] = json[field_name][i].get<int>();
  }
  return values;
}

std::string make_random_hex_key(size_t bytes = 16)
{
  static const char hex[] = "0123456789abcdef";
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFFFFFFFFULL);
  std::string s;
  s.reserve(bytes * 2);
  for (size_t generated = 0; generated < bytes; ++generated)
  {
    const uint64_t v = dist(gen);
    const unsigned byte = static_cast<unsigned>(v & 0xFFu);
    s.push_back(hex[(byte >> 4) & 0xF]);
    s.push_back(hex[byte & 0xF]);
  }
  return s;
}
}  // namespace

/** Save shutdown snapshot for restart recovery. */
void MotorController::save_state_to_json() const
{
  std::array<int, 4> starting_positions_snapshot{0, 0, 0, 0};
  std::array<int, 4> current_positions_snapshot{0, 0, 0, 0};
  std::string key_copy;
  {
    std::lock_guard<std::mutex> lock(state_mtx_);
    starting_positions_snapshot = starting_positions;
    current_positions_snapshot = response_positions_;
    key_copy = starting_positions_key_;
  }

  std::error_code ec;
  const auto output_dir = get_state_directory();
  std::filesystem::create_directories(output_dir, ec);
  if (ec)
  {
    RCLCPP_WARN(logger_, "Failed to create state directory %s: %s",
                output_dir.string().c_str(), ec.message().c_str());
    return;
  }

  const auto output_file = output_dir / "motor_state_shutdown_last.json";
  std::ofstream out(output_file, std::ios::out | std::ios::trunc);
  if (!out)
  {
    RCLCPP_WARN(logger_, "Failed to open shutdown state file %s", output_file.string().c_str());
    return;
  }

  nlohmann::json j;
  j["saved_at"] = make_timestamp();
  if (!key_copy.empty())
  {
    j["state_key"] = key_copy;
  }
  j["starting_positions"] = make_position_json(starting_positions_snapshot);
  j["current_positions"] = make_position_json(current_positions_snapshot);
  out << j.dump(2) << "\n";
}

/** Save the current calibration starting positions for the next restart. */
void MotorController::save_starting_positions_to_json(const std::array<int, 4>& starting_positions_snapshot)
{
  std::error_code ec;
  const auto output_dir = get_state_directory();
  std::filesystem::create_directories(output_dir, ec);
  if (ec)
  {
    RCLCPP_WARN(logger_, "Failed to create state directory %s: %s",
                output_dir.string().c_str(), ec.message().c_str());
    return;
  }

  const auto output_file = output_dir / "starting_positions_last.json";
  std::ofstream out(output_file, std::ios::out | std::ios::trunc);
  if (!out)
  {
    RCLCPP_WARN(logger_, "Failed to open starting positions file %s", output_file.string().c_str());
    return;
  }

  const std::string key = make_random_hex_key(16);
  {
    std::lock_guard<std::mutex> lock(state_mtx_);
    starting_positions_key_ = key;
  }

  nlohmann::json j;
  j["saved_at"] = make_timestamp();
  j["state_key"] = key;
  j["starting_positions"] = make_position_json(starting_positions_snapshot);
  out << j.dump(2) << "\n";
}

/** Read both saved JSON files and validate required fields. */
bool MotorController::load_saved_state_files(MotorController::LoadedSavedState& loaded_state) const
{
  const auto dir = get_state_directory();
  const auto start_file = dir / "starting_positions_last.json";
  const auto shutdown_file = dir / "motor_state_shutdown_last.json";

  if (!std::filesystem::exists(start_file))
  {
    RCLCPP_WARN(logger_, "Starting positions file not found: %s", start_file.string().c_str());
    return false;
  }
  if (!std::filesystem::exists(shutdown_file))
  {
    RCLCPP_WARN(logger_, "Shutdown state file not found: %s", shutdown_file.string().c_str());
    return false;
  }

  try
  {
    std::ifstream start_stream(start_file);
    if (!start_stream)
    {
      RCLCPP_WARN(logger_, "Failed to open starting positions file: %s", start_file.string().c_str());
      return false;
    }

    std::ifstream shutdown_stream(shutdown_file);
    if (!shutdown_stream)
    {
      RCLCPP_WARN(logger_, "Failed to open shutdown state file: %s", shutdown_file.string().c_str());
      return false;
    }

    nlohmann::json start_json;
    nlohmann::json shutdown_json;
    start_stream >> start_json;
    shutdown_stream >> shutdown_json;

    if (!start_json.contains("state_key") || !start_json["state_key"].is_string() ||
        !shutdown_json.contains("state_key") || !shutdown_json["state_key"].is_string())
    {
      RCLCPP_WARN(logger_, "Missing or invalid state_key in saved state files");
      return false;
    }

    const auto start_positions = parse_positions_field(start_json, "starting_positions");
    const auto shutdown_start_positions = parse_positions_field(shutdown_json, "starting_positions");
    const auto shutdown_current_positions = parse_positions_field(shutdown_json, "current_positions");
    if (!start_positions || !shutdown_start_positions || !shutdown_current_positions)
    {
      RCLCPP_WARN(logger_, "Invalid starting/current position values in saved state files");
      return false;
    }

    loaded_state.start_file_key = start_json["state_key"].get<std::string>();
    loaded_state.shutdown_file_key = shutdown_json["state_key"].get<std::string>();
    loaded_state.start_file_starting_positions = *start_positions;
    loaded_state.shutdown_file_starting_positions = *shutdown_start_positions;
    loaded_state.shutdown_file_current_positions = *shutdown_current_positions;
    return true;
  }
  catch (const std::exception& e)
  {
    RCLCPP_WARN(logger_, "Failed to load saved state files: %s", e.what());
    return false;
  }
}

/** Ensure both JSON files belong to the same saved state. */
bool MotorController::compare_saved_state_files(const MotorController::LoadedSavedState& loaded_state) const
{
  if (loaded_state.start_file_key != loaded_state.shutdown_file_key)
  {
    RCLCPP_WARN(logger_, "State keys do not match: %s != %s",
                loaded_state.start_file_key.c_str(), loaded_state.shutdown_file_key.c_str());
    return false;
  }

  if (loaded_state.start_file_starting_positions != loaded_state.shutdown_file_starting_positions)
  {
    RCLCPP_WARN(logger_, "Starting positions differ between start and shutdown files");
    return false;
  }

  return true;
}

/** Check whether the live encoder positions still match the saved shutdown snapshot. */
bool MotorController::compare_saved_state_to_current_positions(const MotorController::LoadedSavedState& loaded_state) const
{
  const auto current = get_positions();
  for (size_t i = 0; i < 4; ++i)
  {
    const int diff = std::abs(current[i] - loaded_state.shutdown_file_current_positions[i]);
    if (diff > 10)
    {
      RCLCPP_WARN(logger_, "Saved current position differs too much from live motor position on motor %zu: saved=%d live=%d diff=%d",
                  i, loaded_state.shutdown_file_current_positions[i], current[i], diff);
      return false;
    }
  }

  return true;
}

/** Restore saved starting positions; offset them if the encoder counters were reset. */
bool MotorController::restore_saved_starting_positions(bool force_restore)
{
  LoadedSavedState loaded_state;
  if (!load_saved_state_files(loaded_state))
  {
    return false;
  }

  if (!compare_saved_state_files(loaded_state))
  {
    return false;
  }

  const bool current_positions_match = compare_saved_state_to_current_positions(loaded_state);
  if (!current_positions_match && !force_restore)
  {
    return false;
  }

  auto starting_positions_to_apply = loaded_state.start_file_starting_positions;
  const bool used_force_offset_restore = !current_positions_match && force_restore;
  if (!current_positions_match && force_restore)
  {
    const auto current_positions = get_positions();
    for (size_t i = 0; i < 4; ++i)
    {
      starting_positions_to_apply[i] =
          loaded_state.start_file_starting_positions[i] +
          (current_positions[i] - loaded_state.shutdown_file_current_positions[i]);
    }
    RCLCPP_WARN(logger_,
                "Saved current positions do not match live encoders. force_restore enabled; applying offset-adjusted starting positions.");
  }

  {
    std::lock_guard<std::mutex> lock(state_mtx_);
    starting_positions = starting_positions_to_apply;
    if (!used_force_offset_restore)
    {
      starting_positions_key_ = loaded_state.start_file_key;
    }
    starting_positions_initialized_.store(true, std::memory_order_release);
  }

  if (used_force_offset_restore)
  {
    save_starting_positions_to_json(starting_positions_to_apply);
  }

  return true;
}
