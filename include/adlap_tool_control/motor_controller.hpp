#pragma once

#include "adlap_tool_control/serial_port.hpp"
#include "rclcpp/rclcpp.hpp"
#include <array>
#include <memory>

class MotorController
{
public:
    MotorController(std::shared_ptr<SerialPort> serial, rclcpp::Logger logger);
    
    // Motor position control
    void send_motor_positions(const std::array<int, 4>& new_positions, bool verbose = true);
    void send_motor_positions(int m0, int m1, int m2, int m3, bool verbose = true);
    void send_relative_motor_positions(const std::array<int, 4>& m_array, bool verbose = true);
    void send_relative_motor_positions(int m0, int m1, int m2, int m3, bool verbose = true);
    void couple_sequence();
    void update_starting_positions();

    // Motor status and communication
    bool set_response_values(bool verbose = true);
    std::string motor_message(const std::array<int, 4>& m_array);
    
    // Getters for current state
    const std::array<int, 4>& get_starting_positions() const { return starting_positions; }
    const std::array<int, 4>& get_positions() const { return positions_; }
    const std::array<int, 6>& get_response_values() const { return response_values_; }
    const std::array<bool, 4>& get_blocked_status() const { return blocked_; }
    const std::array<bool, 4>& get_maxed_status() const { return maxed_; }
    const int get_lower_motors_play() const { return LOWER_MOTORS_PLAY; }

private:
    void find_hall_sensor_positions();

    std::shared_ptr<SerialPort> serial_;
    rclcpp::Logger logger_;
    
    // Motor state
    std::array<int, 4>  starting_positions = {0, 0, 0, 0}; // Starting positions for the motors
    std::array<int, 4> positions_{0, 0, 0, 0};
    std::array<int, 6> response_values_{0, 0, 0, 0, 0, 0};
    std::array<bool, 4> blocked_{false, false, false, false};
    std::array<bool, 4> maxed_{false, false, false, false};
    
    // Constants
    // Motor unit defining constants
    static constexpr int MIN_WAIT_TIME = 4; // Minimum wait time in milliseconds
    static constexpr int BLOCKING_CURRENT = 30;
    static constexpr int MAX_CURRENT = 60;
    /* Motor gearbox is 150:1. The motors are equipped with a 12 pulse per rotation hall encoder.
    This means 450 pulses equals 1 full rotation.*/
    const float UPPER_MOTOR_FACTOR = 15.0f/25;
    const int MOTOR_PULSES_PER_ROTATION = 1800; // pulses

    const int LOWER_MOTORS_PLAY = 15; // in pulses. When changing directions, you need to move twice the play before the gearbox starts moving

    const int HALL1_OFFSET = -120; // in pulses, to align the hall sensor with the physical 0 degree position
    const int HALL2_OFFSET = -90; // in pulses, to align the hall sensor with the physical 0 degree position

};