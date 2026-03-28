#include "adlap_tool_control/motor_controller.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

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

  {
    std::lock_guard<std::mutex> lock(state_mtx_);
    const uint8_t flags = static_cast<uint8_t>(values[8] & 0xFF);
    const uint8_t hall  = static_cast<uint8_t>(values[9] & 0xFF);

    for (int i = 0; i < 4; ++i)
    {
      response_positions_[i] = values[i];
      response_currents_[i] = values[i + 4];
      stalled_[i] = (flags >> (i)) & 0x01; // Stall flags are in bits 0-3
      overcurrent_[i] = (flags >> (i + 4)) & 0x01; // Overcurrent flags are in bits 4-7
    }
    hall_[0] = hall & 0x01; // Hall sensor 1 in bit 0
    hall_[1] = (hall >> 1) & 0x01; // Hall sensor 2 in bit 1
  }

  return true;
}

/// @brief Get a thread-safe snapshot of the actual current motor positions
std::array<int, 4> MotorController::get_positions() const
{
  std::lock_guard<std::mutex> lock(state_mtx_);
  return response_positions_;
}

std::array<int, 4> MotorController::get_currents() const
{
  std::lock_guard<std::mutex> lock(state_mtx_);
  return response_currents_;
}

std::array<bool, 4> MotorController::get_stalled() const
{
  std::lock_guard<std::mutex> lock(state_mtx_);
  return stalled_;
}

std::array<bool, 4> MotorController::get_overcurrent() const
{
  std::lock_guard<std::mutex> lock(state_mtx_);
  return overcurrent_;
}

std::array<bool, 2> MotorController::get_hall_sensors() const
{
  std::lock_guard<std::mutex> lock(state_mtx_);
  return hall_;
}

void MotorController::update_starting_positions()
{
  std::lock_guard<std::mutex> lock(state_mtx_);
  for (size_t i = 0; i < 4; ++i)
  {
    starting_positions[i] = response_positions_[i];
  }
}

void MotorController::update_target_positions()
{
  std::lock_guard<std::mutex> lock(state_mtx_);
  for (size_t i = 0; i < 4; ++i)
  {
    target_positions_[i] = response_positions_[i];
  }
}

std::array<bool, 4> MotorController::get_blocked() const
{
  std::array<bool, 4> blocked;
  std::array<int, 4> target_positions_copy;
  std::array<int, 4> response_positions_copy;
  std::array<int, 4> response_currents_copy;
  std::array<bool, 4> stalled_copy;
  std::array<bool, 4> overcurrent_copy;
  bool any_blocked = false;
  {
    std::lock_guard<std::mutex> lock(state_mtx_);
    for (size_t i = 0; i < 4; ++i)
    {
      target_positions_copy[i] = target_positions_[i];
      response_positions_copy[i] = response_positions_[i];
      response_currents_copy[i] = response_currents_[i];
      stalled_copy[i] = stalled_[i];
      overcurrent_copy[i] = overcurrent_[i];
      blocked[i] = stalled_copy[i] || overcurrent_copy[i]; // Consider blocked if stalled or overcurrent
    }
  }

  for (size_t i = 0; i < 4; ++i)
  {
    if (!blocked[i])
    {
      const float margin = get_pulses_per_degree(is_upper_motor(i)) * 30;
      if (target_positions_copy[i] > response_positions_copy[i] + margin ||
          target_positions_copy[i] < response_positions_copy[i] - margin)
      {
        blocked[i] = true; // Treat as blocked if position is way off
        RCLCPP_DEBUG(logger_, "Motor %zu position off by more than 30 degrees, treating as blocked (target: %d, actual: %d)",
                     i, target_positions_copy[i], response_positions_copy[i]);
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
    RCLCPP_DEBUG(logger_, "Blocked with current values: %d, %d, %d, %d",
                 response_currents_copy[0], response_currents_copy[1], response_currents_copy[2], response_currents_copy[3]);
    RCLCPP_DEBUG(logger_, "Flag values stalled: %d, %d, %d, %d - overcurrent: %d, %d, %d, %d",
                 stalled_copy[0], stalled_copy[1], stalled_copy[2], stalled_copy[3],
                 overcurrent_copy[0], overcurrent_copy[1], overcurrent_copy[2], overcurrent_copy[3]);
  }

  return blocked;
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
