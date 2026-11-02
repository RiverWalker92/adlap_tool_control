#include "adlap_tool_control/gearbox.hpp"
#include "adlap_tool_control/saving_utils.hpp"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <system_error>


void Gearbox::save_state_to_json() const
{
  std::error_code ec;
  const auto output_dir = saving_utils::get_state_directory();
  std::filesystem::create_directories(output_dir, ec);
  if (ec)
  {
    RCLCPP_WARN(logger_, "Failed to create state directory %s: %s",
                output_dir.string().c_str(), ec.message().c_str());
    return;
  }

  const auto output_file = output_dir / "gearbox_state_last.json";
  std::ofstream out(output_file, std::ios::out | std::ios::trunc);
  if (!out)
  {
    RCLCPP_WARN(logger_, "Failed to open gearbox state file %s", output_file.string().c_str());
    return;
  }

  nlohmann::json j;
  j["saved_at"] = saving_utils::make_timestamp();
  j["session_key"] = motor_controller.get_session_key();
  j["m1_m2_offset"] = m1_m2_offset;
  out << j.dump(2) << "\n";
}

bool Gearbox::load_state_from_json()
{
  const auto input_file = saving_utils::get_state_directory() / "gearbox_state_last.json";
  if (!std::filesystem::exists(input_file))
  {
    RCLCPP_WARN(logger_, "Gearbox state file not found: %s", input_file.string().c_str());
    return false;
  }

  try
  {
    std::ifstream in(input_file);
    if (!in)
    {
      RCLCPP_WARN(logger_, "Failed to open gearbox state file: %s", input_file.string().c_str());
      return false;
    }

    const nlohmann::json j = nlohmann::json::parse(in);
    if (!j.contains("session_key") || !j["session_key"].is_string() ||
        !j.contains("m1_m2_offset") || !j["m1_m2_offset"].is_number_integer())
    {
      RCLCPP_WARN(logger_, "Invalid fields in gearbox state file: %s", input_file.string().c_str());
      return false;
    }

    const std::string file_session_key = j["session_key"].get<std::string>();
    if (file_session_key != motor_controller.get_session_key())
    {
      RCLCPP_WARN(logger_, "Gearbox state session key does not match the current session");
      return false;
    }

    m1_m2_offset = j["m1_m2_offset"].get<int>();
    return true;
  }
  catch (const std::exception& e)
  {
    RCLCPP_WARN(logger_, "Failed to load gearbox state file: %s", e.what());
    return false;
  }
}
