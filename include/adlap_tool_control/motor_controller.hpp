#pragma once

#include "adlap_tool_control/serial_port.hpp"
#include "rclcpp/rclcpp.hpp"
#include <array>
#include <memory>
#include <string>

/// @brief Configuration parameters for different motor types
class Motor
{
public:
    int magnets;                 // Number of magnets in the encoder
    float gear_ratio;            // Gear ratio (e.g., 150:1)
    int encoder_mode;            // Read rise and fall of both channels for higher resolution (1, 2, or 4)
 
    int duty_cycle_percentage;   // Duty cycle percentage for motor control (0-100)
    int max_current;             // Current threshold to detect motor is blocked (mA)
    int emergency_current;       // Maximum safe current threshold (mA)
    
    float upper_motor_factor;    // Gear ratio factor for upper motors (if different)
    int lower_motors_play;       // Backlash compensation in degrees
    int min_wait_time_ms;        // Minimum wait time after commands (ms)
    bool reverse_direction = true; // Whether to reverse the direction of the motor clockwise or counterclockwise
    bool inverse_driven = false; // Whether the motor is driven inversely + signal makes encoder count down instead of up

        Motor(int magnets,
                    float gear_ratio,
                    int encoder_mode,
                    int duty_cycle_percentage,
                    int max_current,
                    int emergency_current,
                    float upper_motor_factor,
                    int lower_motors_play,
                    int min_wait_time_ms,
                    bool reverse_direction,
                    bool inverse_driven)
                : magnets(magnets),
                    gear_ratio(gear_ratio),
                    encoder_mode(encoder_mode),
                    duty_cycle_percentage(duty_cycle_percentage),
                    max_current(max_current),
                    emergency_current(emergency_current),
                    upper_motor_factor(upper_motor_factor),
                    lower_motors_play(lower_motors_play),
                    min_wait_time_ms(min_wait_time_ms),
                    reverse_direction(reverse_direction),
                    inverse_driven(inverse_driven)
        {}
    
    /// @brief Calculate pulses per rotation from hardware parameters
    int get_pulses_per_rotation() const {
        return static_cast<int>(std::lround(magnets * gear_ratio * encoder_mode));
    }
    
    /// @brief Create motor configuration from gear ratio, encoder resolution, and mode
    static Motor create(int magnets, float gear_ratio, int encoder_mode,
                             int duty_cycle = 40, int max_current = 500, int emergency_current = 1000,
                             float upper_motor_factor = 25.0f / 15.0f, int lower_motors_play = 15,
                             int min_wait_time_ms = 4, bool reverse_direction = true, bool inverse_driven = false) {
        return {
            magnets,
            gear_ratio,
            encoder_mode,
            duty_cycle,
            max_current,
            emergency_current,
            upper_motor_factor,
            lower_motors_play,
            min_wait_time_ms,
            reverse_direction,
            inverse_driven
        };
    }
    
    /// @brief Create default motor configuration
    static Motor create_default() {
        return create(3, 150.58f, 2);  // 3 magnets * 150:1 gearbox * 2 mode = 900 pulses
    }
};

class MotorController
{
public:
    MotorController(std::shared_ptr<SerialPort> serial, rclcpp::Logger logger, const Motor& config = Motor::create_default());
    ~MotorController();
    
    // Motor position control
    void send_motor_positions(const std::array<int, 4>& new_positions, bool verbose = false);
    void send_relative_motor_positions(const std::array<int, 4>& m_array, bool verbose = false);
    void send_relative_motor_positions(int m0, int m1, int m2, int m3, bool verbose = false);
    void send_duty_cycle(const std::array<int, 4>& duty_cycle_array, bool verbose = false);
    void send_encoder_mode(const std::array<int, 4>& duty_cycle_array, bool verbose = false);
    void send_motor_configuration(int motor_index, bool verbose = false);
    void couple_sequence();
    void setup_motors();
    void update_target_positions();
    void update_starting_positions();

    // Motor status and communication
    bool set_response_values(bool verbose = false);
    std::string motor_message(const std::array<int, 4>& m_array);
    
    // Getters for current state
    const std::array<int, 4>& get_starting_positions() const { return starting_positions; }
    const std::array<int, 4>& get_target_positions() const { return target_positions_; }
    const std::array<bool, 4>& get_blocked_status() const { return stalled_; }
    const std::array<bool, 4>& get_maxed_status() const { return overcurrent_; }
    int get_pulses_per_rotation(bool upper) const { return upper ? static_cast<int>(motor_.get_pulses_per_rotation() * motor_.upper_motor_factor) : motor_.get_pulses_per_rotation(); }
    float get_pulses_per_degree(bool upper = false) const { 
        int pulses = get_pulses_per_rotation(upper);
        return static_cast<float>(pulses) / 360.0f;
    }
    int get_pulses_lower_motors_play() const { return static_cast<int>(std::lround(motor_.lower_motors_play * get_pulses_per_degree())); }
    bool is_upper_motor(int motor_index) const { return motor_index == 0 || motor_index == 3; }

    void start_stream_reader();
    void stop_stream_reader();

    int lower_motor_start_offset = 0; // Compensation used
    int gripper_start_offset = 0; // Compensation used for the gripper motor

    // Thread-safe snapshots for your main thread
    std::array<bool, 4> get_blocked() const;
    std::array<bool, 4> get_maxed() const;
    bool any_maxed() const;
    std::array<int, 4> get_positions() const;
    std::array<int, 4> get_currents() const;
    std::array<bool, 4> get_stalled() const;
    std::array<bool, 4> get_overcurrent() const;
    std::array<bool, 2> get_hall_sensors() const;
    uint64_t get_rx_seq() const;

    // Optional: wait for fresh data (e.g., after sending a command)
    bool wait_for_next_frame(uint64_t last_seq, std::chrono::milliseconds timeout);


private:
    void reader_loop();

    // Streaming-friendly: parse a line you already read (no serial read inside)
    bool parse_frame(const std::string& response, bool verbose);
    bool parse_telemetry_message(const uint8_t *payload, uint8_t len);
    bool parse_log_message(const uint8_t *payload, uint8_t len);

    mutable std::mutex state_mtx_;
    mutable std::condition_variable state_cv_;
    std::atomic<bool> reader_running_{false};
    std::thread reader_thread_;
    std::atomic<uint64_t> rx_seq_{0};
    std::string rx_buffer_;
    std::chrono::steady_clock::time_point last_message_time_;
    static constexpr int MESSAGE_TIMEOUT_MS = 5000;  // Timeout for no messages received

    void find_hall_sensor_positions();

    std::shared_ptr<SerialPort> serial_;
    rclcpp::Logger logger_;
    Motor motor_;  // Motor configuration parameters
    
    // Motor state
    std::array<int, 4>  starting_positions = {0, 0, 0, 0}; // Starting positions for the motors
    std::array<int, 4> target_positions_{0, 0, 0, 0};
    std::array<int, 10> response_values_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::array<int, 4> response_positions_{0, 0, 0, 0};
    std::array<int, 4> response_currents_{0, 0, 0, 0};
    std::array<bool, 4> stalled_{false, false, false, false};
    std::array<bool, 4> overcurrent_{false, false, false, false};
    std::array<bool, 2> hall_{false, false};
    
    // Calibration constants (hardware-specific offsets)
    const int HALL1_OFFSET = 4; // in degrees from 6 o'clock position
    const int HALL2_OFFSET = 10; // in degrees from 6 o'clock position
};