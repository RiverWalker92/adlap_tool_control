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
  send_duty_cycle({motor_.duty_cycle_percentage, motor_.duty_cycle_percentage, motor_.duty_cycle_percentage, motor_.duty_cycle_percentage}, /*verbose=*/true);
  send_encoder_mode({motor_.encoder_mode, motor_.encoder_mode, motor_.encoder_mode, motor_.encoder_mode}, /*verbose=*/true);
  update_target_positions();
  update_starting_positions();
}

MotorController::~MotorController()
{
  stop_stream_reader();
}

void MotorController::start_stream_reader()
{
  if (reader_running_.exchange(true))
  {
    return; // already running
  }
  reader_thread_ = std::thread([this]()
                               { reader_loop(); });
}

void MotorController::stop_stream_reader()
{
  if (!reader_running_.exchange(false))
  {
    return; // already stopped
  }
  if (reader_thread_.joinable())
  {
    reader_thread_.join();
  }
}

void MotorController::reader_loop()
{
  last_message_time_ = std::chrono::steady_clock::now();

  while (rclcpp::ok() && reader_running_.load())
  {
    const std::string chunk = serial_->read_data();
    if (chunk.empty())
    {
      // Check for timeout
      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_message_time_).count();
      if (elapsed > MESSAGE_TIMEOUT_MS)
      {
        std::string error_msg = "Motor communication timeout: No messages received for " + std::to_string(elapsed) + " ms";
        RCLCPP_ERROR(logger_, "%s", error_msg.c_str());
        reader_running_.store(false);
        throw std::runtime_error(error_msg);
      }
    }

    rx_buffer_ += chunk;

    // Process all complete lines currently in the buffer
    size_t nl = std::string::npos;
    while ((nl = rx_buffer_.find('\n')) != std::string::npos)
    {
      std::string line = rx_buffer_.substr(0, nl);
      rx_buffer_.erase(0, nl + 1);

      // Drop CR if sender uses "\r\n"
      if (!line.empty() && line.back() == '\r')
      {
        line.pop_back();
      }
      if (line.empty())
      {
        continue;
      }

      const bool ok = set_response_values_from_stream(line, /*verbose=*/false);
      if (ok)
      {
        last_message_time_ = std::chrono::steady_clock::now(); // Update last message time
        rx_seq_.fetch_add(1, std::memory_order_relaxed);
        state_cv_.notify_all();
      }
    }
  }
}

bool MotorController::set_response_values_from_stream(const std::string &line, bool verbose)
{
  if (verbose)
  {
    RCLCPP_INFO(logger_, "Raw response: '%s'", line.c_str());
  }

  if (line.rfind("INFO:", 0) == 0)
  { // starts with "INFO:"
    RCLCPP_INFO(logger_, "Motor controller info: %s", line.c_str());
    return true;
  }

  if (line.rfind("ERROR:", 0) == 0)
  { // starts with "ERROR:"
    RCLCPP_ERROR(logger_, "Motor controller: %s", line.c_str());
    return false;
  }

  std::vector<int> values;
  values.reserve(10);

  try
  {
    std::stringstream ss(line);
    std::string item;

    while (std::getline(ss, item, ','))
    {
      if (item.empty())
        continue;
      values.push_back(std::stoi(item));
    }
  }
  catch (const std::exception &e)
  {
    // Streaming: don't crash on one bad/partial frame
    RCLCPP_WARN(logger_, "Failed to parse motor stream frame: %s", e.what());
    return false;
  }

  if (values.size() < 10)
  {
    RCLCPP_WARN(logger_, "Motor stream frame too short (%zu values): '%s'",
                values.size(), line.c_str());
    return false;
  }

  bool response_ok = true;
  {
    std::lock_guard<std::mutex> lock(state_mtx_);

    for (int i = 0; i < 10; ++i)
    {
      response_values_[i] = values[i];
    }

    for (int i = 0; i < 4; ++i)
    {
      if (response_values_[i+4] > motor_.blocking_current)
      {
        blocked_[i] = true;
      }
      else
      {
        blocked_[i] = false;
      }

      if (response_values_[i+4] > motor_.max_current)
      {
        maxed_[i] = true;
        response_ok = false;
      }
      else
      {
        maxed_[i] = false;
      }
    }
  }

  return response_ok;
}

std::array<int, 10> MotorController::get_response_values() const
{
  std::lock_guard<std::mutex> lock(state_mtx_);
  return response_values_;
}

void MotorController::update_starting_positions()
{
  std::lock_guard<std::mutex> lock(state_mtx_);
  for (size_t i = 0; i < 4; ++i)
  {
    starting_positions[i] = response_values_[i];
  }
}

void MotorController::update_target_positions()
{
  std::lock_guard<std::mutex> lock(state_mtx_);
  for (size_t i = 0; i < 4; ++i)
  {
    target_positions_[i] = response_values_[i];
  }
}

std::array<bool, 4> MotorController::get_blocked() const
{
  std::array<bool, 4> blocked_copy;
  bool any_blocked = false;
  {
    std::lock_guard<std::mutex> lock(state_mtx_);
    for (size_t i = 0; i < 4; ++i)
    {
      blocked_copy[i] = blocked_[i];
    }
  }
  for (size_t i = 0; i < 4; ++i){
    if (!blocked_copy[i])
    {
      if (target_positions_[i] > response_values_[i] + get_pulses_per_degree(is_upper_motor(i)) * 10 || target_positions_[i] < response_values_[i] - get_pulses_per_degree(is_upper_motor(i)) * 10)
      {
        blocked_copy[i] = true; // Treat as blocked if position is way off
        RCLCPP_DEBUG(logger_, "Motor %zu position off by more than 10 degrees, treating as blocked (target: %d, actual: %d)", i, target_positions_[i], response_values_[i]);
        any_blocked = true;
      }
    }
    else
    {
      any_blocked = true;
    }
  }
  if (any_blocked)
  {
    std::array<int, 4> current_values = get_currents();
    RCLCPP_DEBUG(logger_, "Blocked with current values: %d, %d, %d, %d", current_values[0], current_values[1], current_values[2], current_values[3]);
  }
  return blocked_copy;
}

std::array<bool, 4> MotorController::get_maxed() const
{
  std::lock_guard<std::mutex> lock(state_mtx_);
  return maxed_;
}

bool MotorController::any_maxed() const
{
  std::lock_guard<std::mutex> lock(state_mtx_);
  for (const auto &is_maxed : maxed_)
  {
    if (is_maxed)
    {
      return true;
    }
  }
  return false;
}

/// @brief Get a thread-safe snapshot of the actual current motor positions
std::array<int, 4> MotorController::get_positions() const
{
  std::array<int, 4> positions_copy;
  std::lock_guard<std::mutex> lock(state_mtx_);
  for (size_t i = 0; i < 4; ++i)
  {
    positions_copy[i] = response_values_[i];
  }
  return positions_copy;
}

std::array<int, 4> MotorController::get_currents() const
{
  std::array<int, 4> currents_copy;
  std::lock_guard<std::mutex> lock(state_mtx_);
  for (size_t i = 0; i < 4; ++i)
  {
    currents_copy[i] = response_values_[i+4];
  }
  return currents_copy;
}

uint64_t MotorController::get_rx_seq() const
{
  return rx_seq_.load(std::memory_order_relaxed);
}

bool MotorController::wait_for_next_frame(uint64_t last_seq, std::chrono::milliseconds timeout)
{
  std::unique_lock<std::mutex> lock(state_mtx_);
  return state_cv_.wait_for(lock, timeout, [&]()
                            { return rx_seq_.load(std::memory_order_relaxed) > last_seq; });
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

void MotorController::send_motor_positions(int m0, int m1, int m2, int m3, bool verbose)
{
  std::array<int, 4> new_positions = {m0, m1, m2, m3};
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
    std::array<int, 4> current_values = get_currents();
    RCLCPP_INFO(logger_, "Current values: %d, %d, %d, %d", current_values[0], current_values[1], current_values[2], current_values[3]);
  }
  if (any_maxed()) // Check if any motor is maxed after the movement
  {
    RCLCPP_ERROR(logger_, "Motor response not ok, reversing movement");
    serial_->write_data(motor_message(target_positions_));
    rclcpp::sleep_for(std::chrono::milliseconds(200));
    if (any_maxed())
    {
      // TODO: Read actual positions and reverse to those instead of just the target positions, which might be off if the motor was blocked for a while
      std::array<int, 10> current_response_values = get_response_values();
      RCLCPP_DEBUG(logger_, "Current values: %d, %d, %d, %d", current_response_values[4], current_response_values[5], current_response_values[6], current_response_values[7]);
      RCLCPP_DEBUG(logger_, "Positions after reversal: %d, %d, %d, %d", target_positions_[0], target_positions_[1], target_positions_[2], target_positions_[3]);
      RCLCPP_DEBUG(logger_, "New positions attempted: %d, %d, %d, %d", new_positions[0], new_positions[1], new_positions[2], new_positions[3]);
      RCLCPP_DEBUG(logger_, "Received positions: %d, %d, %d, %d", current_response_values[0], current_response_values[1], current_response_values[2], current_response_values[3]);
      send_duty_cycle({0, 0, 0, 0}, true);               // Stop all motors immediately
      rclcpp::sleep_for(std::chrono::milliseconds(200)); // Give it a moment to stop and log the final state
      throw std::runtime_error("Motor response still not ok after reversing movement");
    }
  }
  else
  {
    target_positions_ = new_positions;
  }
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

/// @brief Couple the motors to the gearbox
void MotorController::couple_sequence()
{
  // Making some rotations to get the pins do drop into the holes of the gearbox
  for (int i : {0, 1, 2, 3})
  {
    int step_size = get_pulses_per_degree(is_upper_motor(i)) * 20; // Step size of 20 degrees in pulses
    for (int j : {step_size, -step_size})
    {
      // Move the motor till it is blocked (coupled and reached its end position)
      std::array<int, 4> driving_values = {0, 0, 0, 0};
      driving_values[i] = 2 * j;
      std::array<int, 4> step_back = {0, 0, 0, 0};
      step_back[i] = -j;
      do
      {
        send_relative_motor_positions(driving_values);
        rclcpp::sleep_for(std::chrono::milliseconds(100));
        send_relative_motor_positions(step_back);
        rclcpp::sleep_for(std::chrono::milliseconds(200));
      } while (!get_blocked()[i]);
      RCLCPP_DEBUG(logger_, "Motor %d blocked at position %d", i, get_positions()[i]);
      // Unblock the motor
      std::array<int, 4> unblock_values = get_positions();
      unblock_values[i] -= 10 * j;
      send_motor_positions(unblock_values);
      rclcpp::sleep_for(std::chrono::milliseconds(500));
    }
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
    // Move the motor till it is blocked
    std::array<int, 4> driving_values = {0, j, 0, 0};
    do
    {
      send_relative_motor_positions(driving_values);
      rclcpp::sleep_for(std::chrono::milliseconds(20));
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
    std::array<int, 10> current_response_values = get_response_values();
    if (current_response_values[8] == 1 and hall1_value == 0)
    {
      hall1_up = target_positions_[1] % motor_.get_pulses_per_rotation();
      RCLCPP_INFO(logger_, "Hall1 up: '%d'", hall1_up);
    }
    if (current_response_values[8] == 0 and hall1_value == 1)
    {
      hall1_down = target_positions_[1] % motor_.get_pulses_per_rotation();
      RCLCPP_INFO(logger_, "Hall1 down: '%d'", hall1_down);
    }
    if (current_response_values[9] == 1 and hall2_value == 0)
    {
      hall2_up = target_positions_[2] % motor_.get_pulses_per_rotation();
      RCLCPP_INFO(logger_, "Hall2 up: '%d'", hall2_up);
    }
    if (current_response_values[9] == 0 and hall2_value == 1)
    {
      hall2_down = target_positions_[2] % motor_.get_pulses_per_rotation();
      RCLCPP_INFO(logger_, "Hall2 down: '%d'", hall2_down);
    }
    hall1_value = current_response_values[8];
    hall2_value = current_response_values[9];
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
