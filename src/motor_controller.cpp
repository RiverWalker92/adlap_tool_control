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

MotorController::MotorController(std::shared_ptr<SerialPort> serial, rclcpp::Logger logger, const Motor &config)
    : serial_(serial), logger_(logger), motor_(config)
{
  // Streaming mode: don't expect request/response anymore.
  start_stream_reader();

  // Optional: wait for first frame so downstream code has valid data
  const auto start_seq = get_rx_seq();
  if (!wait_for_next_frame(start_seq, std::chrono::milliseconds(1000)))
  {
    throw std::runtime_error("No streaming data received from motor within timeout");
  }
  //send_duty_cycle({motor_.duty_cycle_percentage, motor_.duty_cycle_percentage, motor_.duty_cycle_percentage, motor_.duty_cycle_percentage}, /*verbose=*/true);
  //send_encoder_mode({motor_.encoder_mode, motor_.encoder_mode, motor_.encoder_mode, motor_.encoder_mode}, /*verbose=*/true);
  for (int i = 0; i < 4; ++i)
  {
    send_motor_configuration(i, /*verbose=*/true);
  }
  update_target_positions();
  update_starting_positions();
}

MotorController::~MotorController()
{
  stop_stream_reader();
}

std::string MotorController::motor_message(const std::array<int, 4> &m_array)
{
  std::string response = "targets " + std::to_string(m_array[0]) + ", " + std::to_string(m_array[1]) + ", " + std::to_string(m_array[2]) + ", " + std::to_string(m_array[3]) + "\n";
  return response;
}

void MotorController::send_relative_motor_positions(const std::array<int, 4> &m_array, bool verbose)
{
  send_relative_motor_positions(m_array[0], m_array[1], m_array[2], m_array[3], verbose);
}

void MotorController::send_relative_motor_positions(int m0, int m1, int m2, int m3, bool verbose)
{
  std::array<int, 4> new_positions = {0, 0, 0, 0};
  new_positions[0] = target_positions_[0] + m0;
  new_positions[1] = target_positions_[1] + m1;
  new_positions[2] = target_positions_[2] + m2;
  new_positions[3] = target_positions_[3] + m3;

  send_motor_positions(new_positions, verbose);
}

/// @brief Send new absolute motor positions to the motor controller
/// @param new_positions Array of 4 integers representing the new motor positions
/// @throws std::runtime_error if the motor response is not ok after reversing movement
void MotorController::send_motor_positions(const std::array<int, 4> &new_positions, bool verbose)
{
  std::string message = motor_message(new_positions);
  if (verbose)
  {
    RCLCPP_INFO(logger_, "Writing: '%s'", message.c_str());
  }
  serial_->write_data(message);
  rclcpp::sleep_for(std::chrono::milliseconds(motor_.min_wait_time_ms));
  if (verbose)
  {

    std::array<bool, 4> blocked = get_blocked();
    RCLCPP_INFO(logger_, "Blocked status: %d, %d, %d, %d", blocked[0], blocked[1], blocked[2], blocked[3]);
    std::array<bool, 2> hall_values = get_hall_sensors();
    RCLCPP_DEBUG(logger_, "Hall sensors: %d, %d", hall_values[0], hall_values[1]);
  }
  target_positions_ = new_positions;

}

/// @brief Send new duty cycle values to the motor controller 0-100% represented as 0 to 100
void MotorController::send_duty_cycle(const std::array<int, 4> &duty_cycle_array, bool verbose)
{
  std::string message = "duty " + std::to_string(duty_cycle_array[0]) + ", " + std::to_string(duty_cycle_array[1]) + ", " + std::to_string(duty_cycle_array[2]) + ", " + std::to_string(duty_cycle_array[3]) + "\n";
  if (verbose)
  {
    RCLCPP_INFO(logger_, "Writing duty cycle: '%s'", message.c_str());
  }
  serial_->write_data(message);
  rclcpp::sleep_for(std::chrono::milliseconds(motor_.min_wait_time_ms));
}

/// @brief Send new encoder mode values to the motor controller (1, 2, or 4)
void MotorController::send_encoder_mode(const std::array<int, 4> &encoder_mode_array, bool verbose)
{
  std::string message = "encoder " + std::to_string(encoder_mode_array[0]) + ", " + std::to_string(encoder_mode_array[1]) + ", " + std::to_string(encoder_mode_array[2]) + ", " + std::to_string(encoder_mode_array[3]) + "\n";
  if (verbose)
  {
    RCLCPP_INFO(logger_, "Writing encoder mode: '%s'", message.c_str());
  }
  serial_->write_data(message);
  rclcpp::sleep_for(std::chrono::milliseconds(motor_.min_wait_time_ms));
}

/// @brief Send motor configuration parameters to the motor controller for a specific motor index
void MotorController::send_motor_configuration(int motor_index, bool verbose)
{
  if (motor_index < 0 || motor_index > 3)
  {
    RCLCPP_ERROR(logger_, "Invalid motor index %d for configuration", motor_index);
    return;
  }
  int reverse_int = static_cast<int>(motor_.reverse_direction);
  std::string message = "config " + std::to_string(motor_index) + ", " +
                        std::to_string(motor_.duty_cycle_percentage) + ", " + // Min duty cycle (same as duty cycle for now, but could be different)
                        std::to_string(motor_.duty_cycle_percentage) + ", " + // Max duty cycle (same as duty cycle for now, but could be different)
                        std::to_string(motor_.max_current) + ", " +
                        std::to_string(motor_.emergency_current) + ", " +
                        std::to_string(static_cast<int>(motor_.gear_ratio)) + ", " +
                        std::to_string(motor_.magnets) + ", " +
                        std::to_string(motor_.encoder_mode) + ", " +
                        std::to_string(reverse_int) + "\n";
  if (verbose)
  {
    RCLCPP_INFO(logger_, "Writing motor configuration: '%s'", message.c_str());
  }
  serial_->write_data(message);
  rclcpp::sleep_for(std::chrono::milliseconds(motor_.min_wait_time_ms));
}

/// @brief Couple the motors to the gearbox
void MotorController::couple_sequence()
{
  // Making some rotations to get the pins do drop into the holes of the gearbox
  for (int i : {0, 1, 2, 3})
  {
    int step_size = get_pulses_per_degree(is_upper_motor(i)) * 20; // Step size of 20 degrees in pulses
    // Move the motor till it is blocked (coupled and reached its end position)
    std::array<int, 4> forward = {0, 0, 0, 0};
    forward[i] = step_size;
    std::array<int, 4> backward = {0, 0, 0, 0};
    backward[i] = -step_size;
    do
    {
      send_relative_motor_positions(forward);
      send_relative_motor_positions(forward);
      rclcpp::sleep_for(std::chrono::milliseconds(300));
      send_relative_motor_positions(backward);
      rclcpp::sleep_for(std::chrono::milliseconds(100));
      if (!rclcpp::ok()) throw std::runtime_error("Shutdown requested (rclcpp::ok() == false)");
    } while (!get_blocked()[i]);
    RCLCPP_DEBUG(logger_, "Motor %d blocked at position %d", i, get_positions()[i]);

    do
    {
      send_relative_motor_positions(backward);
      rclcpp::sleep_for(std::chrono::milliseconds(150));
      if (!rclcpp::ok()) throw std::runtime_error("Shutdown requested (rclcpp::ok() == false)");
    } while (!get_blocked()[i]);
    RCLCPP_DEBUG(logger_, "Motor %d blocked at position %d", i, get_positions()[i]);

    // Unblock the motor
    std::array<int, 4> unblock_values = get_positions();
    unblock_values[i] += step_size; // Move back a bit to unblock the motor
    send_motor_positions(unblock_values);
    rclcpp::sleep_for(std::chrono::milliseconds(500));
  }
  RCLCPP_DEBUG(logger_, "Coupling sequence completed");
}

/// @brief Set up the lower motors in the correct position for instrument operation to start.
void MotorController::setup_motors()
{
  // Move motor 1 to both extremes and position it in the middle of that.

  int motor_nr = 1;
  int step_size = get_pulses_per_degree(); // Step size of a degree in pulses
  int max_position = 0;
  int min_position = 0;
  for (int j : {-step_size, step_size})
  {
    
    update_target_positions();  // reset target positions to current
    // Move the motor till it is blocked
    std::array<int, 4> driving_values = {0, j, 0, 0};
    do
    {
      send_relative_motor_positions(driving_values);
      rclcpp::sleep_for(std::chrono::milliseconds(20));
      if (!rclcpp::ok()) throw std::runtime_error("Shutdown requested (rclcpp::ok() == false)");
    } while (!get_blocked()[motor_nr]);
    if (j > 0)
    {
      max_position = get_positions()[motor_nr];
      RCLCPP_DEBUG(logger_, "Motor %d blocked at max position %d", motor_nr, max_position);
    }
    else
    {
      min_position = get_positions()[motor_nr];
      RCLCPP_DEBUG(logger_, "Motor %d blocked at min position %d", motor_nr, min_position);
    }
  }
  // Place the motor in the middle between the two blocked positions
  std::array<int, 4> setup_values = get_positions();
  lower_motor_start_offset = 20 * get_pulses_per_degree();
  setup_values[motor_nr] = (max_position + min_position) / 2 - lower_motor_start_offset; // Add some extra to compensate for backlash
  send_motor_positions(setup_values);
  rclcpp::sleep_for(std::chrono::milliseconds(500));
  // Drive the motor back to remove tension and put it in the neutral starting position for the instrument operation
  setup_values[motor_nr] += 2 * get_pulses_lower_motors_play();
  send_motor_positions(setup_values);
  rclcpp::sleep_for(std::chrono::milliseconds(500));

  // TODO: Move motor 1 and 2 together so motor 2 has the hall magnet down

  // Move motor 3 to the starting position so the collet can be tightened around the instrument shaft
  motor_nr = 3;
  std::array<int, 4> driving_values = {0, 0, 0, -get_pulses_per_degree(true) * 5};
  do
  {
    send_relative_motor_positions(driving_values);
    rclcpp::sleep_for(std::chrono::milliseconds(50));
    if (!rclcpp::ok()) throw std::runtime_error("Shutdown requested (rclcpp::ok() == false)");
  } while (!get_blocked()[motor_nr]);
  // Move back a bit to remove tension and put it in the neutral starting position for the instrument operation
  setup_values = get_positions();
  gripper_start_offset = 200 * get_pulses_per_degree(true);
  setup_values[motor_nr] += gripper_start_offset; // Add some extra to compensate for backlash
  send_motor_positions(setup_values);
  rclcpp::sleep_for(std::chrono::milliseconds(500));

  RCLCPP_DEBUG(logger_, "Motor setup sequence completed with lower_motor_start_offset %d and gripper_start_offset %d", lower_motor_start_offset, gripper_start_offset);
}

void MotorController::find_hall_sensor_positions()
{
  // Couple the lower motors
  // These motors also control the front gears, which have to be aligned for the tool to be able to be inserted
  // The hall sensors can be used to find the correct position

  int hall1_value = response_values_[4];
  int hall1_up = 0;
  int hall1_down = 0;
  int hall2_value = response_values_[5];
  int hall2_up = 0;
  int hall2_down = 0;
  int step_size = 10;
  for (int i = 0; i < 3 * motor_.get_pulses_per_rotation() / step_size; i++)
  {
    send_relative_motor_positions(0, step_size, step_size, 0, false);
    std::array<bool, 2> hall_values = get_hall_sensors();
    if (hall_values[0] == 1 and hall1_value == 0)
    {
      hall1_up = target_positions_[1] % motor_.get_pulses_per_rotation();
      RCLCPP_INFO(logger_, "Hall1 up: '%d'", hall1_up);
    }
    if (hall_values[0] == 0 and hall1_value == 1)
    {
      hall1_down = target_positions_[1] % motor_.get_pulses_per_rotation();
      RCLCPP_INFO(logger_, "Hall1 down: '%d'", hall1_down);
    }
    if (hall_values[1] == 1 and hall2_value == 0)
    {
      hall2_up = target_positions_[2] % motor_.get_pulses_per_rotation();
      RCLCPP_INFO(logger_, "Hall2 up: '%d'", hall2_up);
    }
    if (hall_values[1] == 0 and hall2_value == 1)
    {
      hall2_down = target_positions_[2] % motor_.get_pulses_per_rotation();
      RCLCPP_INFO(logger_, "Hall2 down: '%d'", hall2_down);
    }
    hall1_value = hall_values[0];
    hall2_value = hall_values[1];
  }
  // Move back to the position where the hall sensor triggers (assumed to be the correct position for instrument insertion)
  if (hall1_up > hall1_down)
  {
    hall1_down += motor_.get_pulses_per_rotation();
  }
  if (hall2_up > hall2_down)
  {
    hall2_down += motor_.get_pulses_per_rotation();
  }
  int hall1_position = (hall1_up + hall1_down) / 2 + HALL1_OFFSET;
  int hall1_correction = hall1_position - target_positions_[1] % motor_.get_pulses_per_rotation();
  RCLCPP_INFO(logger_, "Setting hall1 position: '%d'", hall1_correction);
  int hall2_position = (hall2_up + hall2_down) / 2 + HALL2_OFFSET;
  int hall2_correction = hall2_position - target_positions_[2] % motor_.get_pulses_per_rotation();
  RCLCPP_INFO(logger_, "Setting hall2 position: '%d'", hall2_correction);
  send_relative_motor_positions(0, hall1_correction, hall2_correction, 0);
}
