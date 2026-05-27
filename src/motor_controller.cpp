#pragma once
#include "adlap_tool_control/motor_controller.hpp"
#include <sstream>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

MotorController::MotorController(std::shared_ptr<SerialPort> serial, rclcpp::Logger logger, const std::array<Motor, 4>& motors)
    : serial_(serial), logger_(logger), motors(motors)
{
  // Streaming mode: don't expect request/response anymore.
  start_stream_reader();

  try
  {
    // Optional: wait for first frame so downstream code has valid data
    const auto start_seq = get_rx_seq();
    if (!wait_for_next_frame(start_seq, std::chrono::milliseconds(1000)))
    {
      throw std::runtime_error("No streaming data received from motor within timeout");
    }

    for (int i = 0; i < 4; ++i)
    {
      send_motor_configuration(i, /*verbose=*/true);
    }
    update_target_positions();
    update_starting_positions();
  }
  catch (...)
  {
    stop_stream_reader();
    throw;
  }
}

MotorController::~MotorController()
{
  stop_stream_reader();
}


// ------------------------------------------------------------
// Commands for motor control and configuration
// ------------------------------------------------------------

/**
 * @brief Format motor position command message for the given motor positions
 * @param m_array Array of 4 integers representing the target motor positions
 * @return Formatted command string to send to the motor controller
 */
std::string MotorController::motor_message(const std::array<int, 4> &m_array)
{
  std::string response = "targets " + std::to_string(m_array[0]) + ", " + std::to_string(m_array[1]) + ", " + std::to_string(m_array[2]) + ", " + std::to_string(m_array[3]) + "\n";
  return response;
}

/**
 * @brief Send new motor positions relative to current target positions
 * @param m_array Array of relative positions for motors 0-3
 */
void MotorController::send_relative_motor_positions(const std::array<int, 4> &m_array, bool verbose)
{
  send_relative_motor_positions(m_array[0], m_array[1], m_array[2], m_array[3], verbose);
}

/**
 * @brief Send new motor positions relative to current target positions
 * @param m0 Relative position for motor 0
 * @param m1 Relative position for motor 1
 * @param m2 Relative position for motor 2
 * @param m3 Relative position for motor 3
 */
void MotorController::send_relative_motor_positions(int m0, int m1, int m2, int m3, bool verbose)
{
  std::array<int, 4> new_positions = {0, 0, 0, 0};
  new_positions[0] = target_positions_[0] + m0;
  new_positions[1] = target_positions_[1] + m1;
  new_positions[2] = target_positions_[2] + m2;
  new_positions[3] = target_positions_[3] + m3;

  send_motor_positions(new_positions, verbose);
}

/**
 * @brief Send new absolute motor positions to the motor controller
 * @param new_positions Array of 4 integers representing the new motor positions
 */
void MotorController::send_motor_positions(const std::array<int, 4> &new_positions, bool verbose)
{
  std::string message = motor_message(new_positions);
  if (verbose)
  {
    std::string log_message = message.substr(0, message.size() - 1);
    RCLCPP_DEBUG(logger_, "Writing: '%s'", log_message.c_str());
  }
  serial_->write_data(message);
  rclcpp::sleep_for(std::chrono::milliseconds(min_wait_time_ms_));
  target_positions_ = new_positions;

}

/// @brief Send new duty cycle values to the motor controller 0-100% represented as 0 to 100
void MotorController::send_duty_cycle(const std::array<int, 4> &duty_cycle_array, bool verbose)
{
  std::string message = "duty " + std::to_string(duty_cycle_array[0]) + ", " + std::to_string(duty_cycle_array[1]) + ", " + std::to_string(duty_cycle_array[2]) + ", " + std::to_string(duty_cycle_array[3]) + "\n";
  if (verbose)
  {
    std::string log_message = message.substr(0, message.size() - 1);
    RCLCPP_DEBUG(logger_, "Writing duty cycle: '%s'", log_message.c_str());
  }
  serial_->write_data(message);
  rclcpp::sleep_for(std::chrono::milliseconds(min_wait_time_ms_));
}

/// @brief Send new encoder mode values to the motor controller (1, 2, or 4)
void MotorController::send_encoder_mode(const std::array<int, 4> &encoder_mode_array, bool verbose)
{
  std::string message = "encoder " + std::to_string(encoder_mode_array[0]) + ", " + std::to_string(encoder_mode_array[1]) + ", " + std::to_string(encoder_mode_array[2]) + ", " + std::to_string(encoder_mode_array[3]) + "\n";
  if (verbose)
  {
    std::string log_message = message.substr(0, message.size() - 1);
    RCLCPP_DEBUG(logger_, "Writing encoder mode: '%s'", log_message.c_str());
  }
  serial_->write_data(message);
  rclcpp::sleep_for(std::chrono::milliseconds(min_wait_time_ms_));
}

/// @brief Send motor configuration parameters to the motor controller for a specific motor index
void MotorController::send_motor_configuration(int motor_index, bool verbose)
{
  if (motor_index < 0 || motor_index > 3)
  {
    RCLCPP_ERROR(logger_, "Invalid motor index %d for configuration", motor_index);
    return;
  }
  int reverse_int = static_cast<int>(motors[motor_index].reverse_direction);
  int inverse_driven_int = static_cast<int>(motors[motor_index].inverse_driven);
  std::string message = "config " + std::to_string(motor_index) + ", " +
                        std::to_string(0) + ", " + // Min duty cycle (fix with proper value)
                        std::to_string(motors[motor_index].duty_cycle_percentage) + ", " + // Max duty cycle 
                        std::to_string(motors[motor_index].max_current) + ", " +
                        std::to_string(motors[motor_index].emergency_current) + ", " +
                        std::to_string(static_cast<int>(motors[motor_index].gear_ratio)) + ", " +
                        std::to_string(motors[motor_index].magnets) + ", " +
                        std::to_string(motors[motor_index].encoder_mode) + ", " +
                        std::to_string(reverse_int) + ", " + 
                        std::to_string(inverse_driven_int) + "\n";
  if (verbose)
  {
    std::string log_message = message.substr(0, message.size() - 1);
    RCLCPP_DEBUG(logger_, "Writing motor configuration: '%s'", log_message.c_str());
  }
  serial_->write_data(message);
  rclcpp::sleep_for(std::chrono::milliseconds(min_wait_time_ms_));
}

std::array<int, 4>& MotorController::get_duty_cycles() const
{
  static std::array<int, 4> duty_cycle_array;
  for (size_t i = 0; i < 4; ++i)
  {
    duty_cycle_array[i] = motors[i].duty_cycle_percentage;
  }
  return duty_cycle_array;
}



