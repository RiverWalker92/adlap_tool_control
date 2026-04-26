#include "adlap_tool_control/motor_controller.hpp"

#include <utility>

MotorController::MotorController(std::shared_ptr<SerialPort> serial, rclcpp::Logger logger, const Motor& config)
    : serial_(std::move(serial)), logger_(logger), motor_(config)
{
  starting_positions = {0, 0, 0, 0};
  target_positions_ = {0, 0, 0, 0};
  response_positions_ = {0, 0, 0, 0};
}

MotorController::~MotorController() = default;

void MotorController::send_motor_positions(const std::array<int, 4>& new_positions, bool)
{
  target_positions_ = new_positions;
  response_positions_ = new_positions;
}

void MotorController::send_relative_motor_positions(const std::array<int, 4>& m_array, bool verbose)
{
  send_relative_motor_positions(m_array[0], m_array[1], m_array[2], m_array[3], verbose);
}

void MotorController::send_relative_motor_positions(int m0, int m1, int m2, int m3, bool verbose)
{
  std::array<int, 4> next = {
    target_positions_[0] + m0,
    target_positions_[1] + m1,
    target_positions_[2] + m2,
    target_positions_[3] + m3
  };
  send_motor_positions(next, verbose);
}

void MotorController::send_duty_cycle(const std::array<int, 4>&, bool) {}

void MotorController::send_encoder_mode(const std::array<int, 4>&, bool) {}

void MotorController::send_motor_configuration(int, bool) {}

void MotorController::couple_sequence() {}

void MotorController::setup_motors() {}

void MotorController::update_target_positions()
{
  target_positions_ = response_positions_;
}

void MotorController::update_starting_positions()
{
  starting_positions = response_positions_;
}

bool MotorController::set_response_values(bool)
{
  return true;
}

std::string MotorController::motor_message(const std::array<int, 4>& m_array)
{
  return "targets " + std::to_string(m_array[0]) + ", " + std::to_string(m_array[1]) + ", " +
         std::to_string(m_array[2]) + ", " + std::to_string(m_array[3]) + "\n";
}

void MotorController::start_stream_reader() {}

void MotorController::stop_stream_reader() {}

std::array<bool, 4> MotorController::get_blocked() const
{
  return stalled_;
}

std::array<bool, 4> MotorController::get_maxed() const
{
  return overcurrent_;
}

bool MotorController::any_maxed() const
{
  return overcurrent_[0] || overcurrent_[1] || overcurrent_[2] || overcurrent_[3];
}

std::array<int, 4> MotorController::get_positions() const
{
  return response_positions_;
}

std::array<int, 4> MotorController::get_currents() const
{
  return response_currents_;
}

std::array<bool, 4> MotorController::get_stalled() const
{
  return stalled_;
}

std::array<bool, 4> MotorController::get_overcurrent() const
{
  return overcurrent_;
}

std::array<bool, 2> MotorController::get_hall_sensors() const
{
  return hall_;
}

uint64_t MotorController::get_rx_seq() const
{
  return rx_seq_.load();
}

bool MotorController::wait_for_next_frame(uint64_t, std::chrono::milliseconds)
{
  return true;
}

void MotorController::reader_loop() {}

bool MotorController::parse_frame(const std::string&, bool)
{
  return true;
}

bool MotorController::parse_telemetry_message(const uint8_t*, uint8_t)
{
  return true;
}

bool MotorController::parse_log_message(const uint8_t*, uint8_t)
{
  return true;
}

void MotorController::find_hall_sensor_positions() {}
