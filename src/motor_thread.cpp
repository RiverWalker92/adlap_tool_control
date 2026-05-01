#include "adlap_tool_control/motor_controller.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <cstdio>

namespace
{
constexpr uint8_t SYNC0 = 0xAA;
constexpr uint8_t SYNC1 = 0x55;
constexpr uint8_t MSG_TYPE_TELEMETRY = 1;
constexpr uint8_t MSG_TYPE_LOG = 2;
constexpr size_t TELEMETRY_PAYLOAD_LEN = 26;
constexpr size_t FRAME_HEADER_LEN = 5; // sync0, sync1, msg_type, seq, len
constexpr size_t FRAME_MIN_LEN = FRAME_HEADER_LEN + 1; // + crc

uint8_t crc8_atm(const uint8_t *data, size_t len)
{
  uint8_t crc = 0x00;
  for (size_t i = 0; i < len; ++i)
  {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit)
    {
      if (crc & 0x80)
      {
        crc = static_cast<uint8_t>((crc << 1) ^ 0x07);
      }
      else
      {
        crc = static_cast<uint8_t>(crc << 1);
      }
    }
  }
  return crc;
}

int32_t read_i32_le(const uint8_t *data)
{
  const uint32_t value =
      static_cast<uint32_t>(data[0]) |
      (static_cast<uint32_t>(data[1]) << 8) |
      (static_cast<uint32_t>(data[2]) << 16) |
      (static_cast<uint32_t>(data[3]) << 24);
  return static_cast<int32_t>(value);
}

uint16_t read_u16_le(const uint8_t *data)
{
  return static_cast<uint16_t>(
      static_cast<uint16_t>(data[0]) |
      (static_cast<uint16_t>(data[1]) << 8));
}
} // namespace

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
  auto last_backlog_log_time = std::chrono::steady_clock::now();
  auto last_backlog_summary_time = std::chrono::steady_clock::now();
  size_t peak_pending_bytes = 0;
  size_t peak_parser_buffer_bytes = 0;

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
        state_cv_.notify_all();
        rclcpp::shutdown();
        break;
      }
    }

    rx_buffer_ += chunk;

    // Log backlog status periodically if we have a growing backlog
    const auto now = std::chrono::steady_clock::now();
    if (now - last_backlog_log_time >= std::chrono::microseconds(200))
    {
      const size_t pending_bytes = serial_->pending_input_bytes();
      peak_pending_bytes = std::max(peak_pending_bytes, pending_bytes);
      peak_parser_buffer_bytes = std::max(peak_parser_buffer_bytes, rx_buffer_.size());

      if (now - last_backlog_summary_time >= std::chrono::seconds(2))
      {
        RCLCPP_DEBUG(logger_,
                     "Serial backlog summary (2s): current=%zuB, peak=%zuB, parser_buffer_peak=%zuB",
                     pending_bytes, peak_pending_bytes, peak_parser_buffer_bytes);
        last_backlog_summary_time = now;
        peak_pending_bytes = pending_bytes;
        peak_parser_buffer_bytes = rx_buffer_.size();
      }
      last_backlog_log_time = now;
    }

    while (true)
    {
      if (rx_buffer_.size() < FRAME_MIN_LEN)
      {
        break;
      }

      size_t sync_pos = std::string::npos;
      for (size_t i = 0; i + 1 < rx_buffer_.size(); ++i)
      {
        if (static_cast<uint8_t>(rx_buffer_[i]) == SYNC0 &&
            static_cast<uint8_t>(rx_buffer_[i + 1]) == SYNC1)
        {
          sync_pos = i;
          break;
        }
      }

      if (sync_pos == std::string::npos)
      {
        if (!rx_buffer_.empty() && static_cast<uint8_t>(rx_buffer_.back()) == SYNC0)
        {
          rx_buffer_.erase(0, rx_buffer_.size() - 1);
        }
        else
        {
          rx_buffer_.clear();
        }
        break;
      }

      if (sync_pos > 0)
      {
        rx_buffer_.erase(0, sync_pos);
      }

      if (rx_buffer_.size() < FRAME_HEADER_LEN)
      {
        break;
      }

      const uint8_t len = static_cast<uint8_t>(rx_buffer_[4]);
      const size_t frame_len = FRAME_HEADER_LEN + static_cast<size_t>(len) + 1;
      if (rx_buffer_.size() < frame_len)
      {
        break;
      }

      const std::string frame = rx_buffer_.substr(0, frame_len);
      const bool ok = parse_frame(frame, /*verbose=*/false);
      if (ok)
      {
        rx_buffer_.erase(0, frame_len);
        last_message_time_ = std::chrono::steady_clock::now();
        rx_seq_.fetch_add(1, std::memory_order_relaxed);
        state_cv_.notify_all();
      }
      else
      {
        rx_buffer_.erase(0, 1);
      }
    }
  }
}

bool MotorController::parse_frame(const std::string &frame, bool verbose)
{
  if (frame.size() < FRAME_MIN_LEN)
  {
    return false;
  }

  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(frame.data());
  if (bytes[0] != SYNC0 || bytes[1] != SYNC1)
  {
    return false;
  }

  const uint8_t msg_type = bytes[2];
  const uint8_t seq = bytes[3];
  const uint8_t len = bytes[4];
  const size_t expected_len = FRAME_HEADER_LEN + static_cast<size_t>(len) + 1;
  if (frame.size() != expected_len)
  {
    return false;
  }

  const uint8_t expected_crc = crc8_atm(&bytes[2], 3 + static_cast<size_t>(len));
  const uint8_t actual_crc = bytes[expected_len - 1];
  if (expected_crc != actual_crc)
  {
    RCLCPP_WARN(logger_, "Dropping frame with CRC mismatch (type=%u, seq=%u, len=%u, expected=0x%02X, got=0x%02X)",
                msg_type, seq, len, expected_crc, actual_crc);
    return false;
  }

  if (verbose)
  {
    RCLCPP_INFO(logger_, "Received frame type=%u seq=%u len=%u", msg_type, seq, len);
  }

  const uint8_t *payload = &bytes[5];

  if (msg_type == MSG_TYPE_LOG)
  {
    return parse_log_message(payload, len);
  }
  else if (msg_type == MSG_TYPE_TELEMETRY)
  {
    return parse_telemetry_message(payload, len);
  }
  else
  {
    RCLCPP_WARN(logger_, "Unknown message type %u (seq=%u, len=%u)", msg_type, seq, len);
    return true;
  }
}

bool MotorController::parse_telemetry_message(const uint8_t *payload, uint8_t len)
{
  if (len != TELEMETRY_PAYLOAD_LEN)
  {
    RCLCPP_WARN(logger_, "Telemetry frame has invalid payload length %u (expected %zu)",
                len, TELEMETRY_PAYLOAD_LEN);
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(state_mtx_);

    std::string hex;
    char buf[4];

    for (int j = 0; j < len; ++j)
    {
      snprintf(buf, sizeof(buf), "%02X ", payload[j]);
      hex += buf;
    }


    for (int i = 0; i < 4; ++i)
    {
      response_positions_[i] = read_i32_le(payload + (i * 4));
      response_currents_[i] = static_cast<int>(read_u16_le(payload + 16 + (i * 2)));
      
    }
    const uint8_t flags = payload[24];
    const uint8_t hall = payload[25];

    for (int i = 0; i < 4; ++i)
    {
      stalled_[i] = (flags >> i) & 0x01;
      overcurrent_[i] = (flags >> (i + 4)) & 0x01;
    }
    hall_[0] = hall & 0x01;
    hall_[1] = (hall >> 1) & 0x01;
  }

  return true;
}

bool MotorController::parse_log_message(const uint8_t *payload, uint8_t len)
{    
  if (len < 1)
  {
    RCLCPP_WARN(logger_, "Log frame has invalid payload length: %u", len);
    return false;
  }

  const uint8_t log_level = payload[0];
  std::string text(reinterpret_cast<const char *>(payload + 1), static_cast<size_t>(len - 1));
  for (char &ch : text)
  {
    if (ch == '\0')
    {
      ch = ' ';
    }
  }
  const rclcpp::Logger logger = rclcpp::get_logger("motor_unit");
  if (log_level == 1)
  {
    RCLCPP_DEBUG(logger, "%s", text.c_str());
  }
  else if (log_level == 2)
  {
    RCLCPP_INFO(logger, "%s", text.c_str());
  }
  else if (log_level == 3)
  {
    RCLCPP_WARN(logger, "%s", text.c_str());
  }
  else if (log_level == 4)
  {
    RCLCPP_ERROR(logger, "%s", text.c_str());
  }
  else if (log_level == 5)
  {
    RCLCPP_FATAL(logger, "%s", text.c_str());
  }
  else
  {
    RCLCPP_WARN(logger, "Unknown level %u: %s", log_level, text.c_str());
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
