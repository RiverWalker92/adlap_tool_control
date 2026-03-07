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
    int encoder_resolution;      // Encoder resolution (pulses per motor rotation)
    int gear_ratio;              // Gear ratio (e.g., 150:1)
    int encoder_mode;            // Read rise and fall of both channels for higher resolution (1, 2, or 4)
 
    int duty_cycle_percentage;   // Duty cycle percentage for motor control (0-100)
    int blocking_current;        // Current threshold to detect motor is blocked (mA)
    int max_current;             // Maximum safe current threshold (mA)
    
    float upper_motor_factor;    // Gear ratio factor for upper motors (if different)
    int lower_motors_play;       // Backlash compensation in pulses
    int min_wait_time_ms;        // Minimum wait time after commands (ms)

        Motor() = default;
        Motor(int encoder_resolution,
                    int gear_ratio,
                    int encoder_mode,
                    int duty_cycle_percentage,
                    int blocking_current,
                    int max_current,
                    float upper_motor_factor,
                    int lower_motors_play,
                    int min_wait_time_ms)
                : encoder_resolution(encoder_resolution),
                    gear_ratio(gear_ratio),
                    encoder_mode(encoder_mode),
                    duty_cycle_percentage(duty_cycle_percentage),
                    blocking_current(blocking_current),
                    max_current(max_current),
                    upper_motor_factor(upper_motor_factor),
                    lower_motors_play(lower_motors_play),
                    min_wait_time_ms(min_wait_time_ms)
        {}
    
    /// @brief Calculate pulses per rotation from hardware parameters
    int get_pulses_per_rotation() const {
        return encoder_resolution * gear_ratio * encoder_mode;
    }
    
    /// @brief Create motor configuration from gear ratio, encoder resolution, and mode
    static Motor create(int encoder_resolution, int gear_ratio, int encoder_mode,
                             int duty_cycle = 7, int blocking_current = 30, int max_current = 60,
                             float upper_motor_factor = 25.0f / 15.0f, int lower_motors_play = 15,
                             int min_wait_time_ms = 4) {
        return {
            encoder_resolution,
            gear_ratio,
            encoder_mode,
            duty_cycle,
            blocking_current,
            max_current,
            upper_motor_factor,
            lower_motors_play,
            min_wait_time_ms
        };
    }
    
    /// @brief Create default motor configuration (150:1 gear ratio, 12 pulse encoder, mode=1)
    static Motor create_default() {
        return create(3, 150, 4);  // 3 pulses/rotation * 150:1 gearbox * 4 mode = 1800 pulses
    }
};

class MotorController
{
public:
    MotorController(std::shared_ptr<SerialPort> serial, rclcpp::Logger logger, const Motor& config = Motor::create_default());
    ~MotorController();
    
    // Motor position control
    void send_motor_positions(const std::array<int, 4>& new_positions, bool verbose = true);
    void send_motor_positions(int m0, int m1, int m2, int m3, bool verbose = true);
    void send_relative_motor_positions(const std::array<int, 4>& m_array, bool verbose = true);
    void send_relative_motor_positions(int m0, int m1, int m2, int m3, bool verbose = true);
    void send_duty_cycle(const std::array<int, 4>& duty_cycle_array, bool verbose = true);
    void send_encoder_mode(const std::array<int, 4>& duty_cycle_array, bool verbose = true);
    void couple_sequence();
    void update_target_positions();
    void update_starting_positions();

    // Motor status and communication
    bool set_response_values(bool verbose = true);
    std::string motor_message(const std::array<int, 4>& m_array);
    
    // Getters for current state
    const std::array<int, 4>& get_starting_positions() const { return starting_positions; }
    const std::array<int, 4>& get_target_positions() const { return target_positions_; }
    const std::array<bool, 4>& get_blocked_status() const { return blocked_; }
    const std::array<bool, 4>& get_maxed_status() const { return maxed_; }
    int get_lower_motors_play() const { return motor_.lower_motors_play; }
    int get_motor_pulses_per_rotation(bool upper) const { return upper ? motor_.get_pulses_per_rotation(): static_cast<int>(motor_.get_pulses_per_rotation() * motor_.upper_motor_factor); }
    float get_pulses_per_degree(bool upper = false) const { 
        int pulses = get_motor_pulses_per_rotation(upper);
        return static_cast<float>(pulses) / 360.0f;
    }
    bool is_upper_motor(int motor_index) const { return motor_index == 0 || motor_index == 3; }

    void start_stream_reader();
    void stop_stream_reader();

    // Thread-safe snapshots for your main thread
    std::array<int, 10> get_response_values() const;
    std::array<bool, 4> get_blocked() const;
    std::array<bool, 4> get_maxed() const;
    bool any_maxed() const;
    std::array<int, 4> get_positions() const;
    uint64_t get_rx_seq() const;

    // Optional: wait for fresh data (e.g., after sending a command)
    bool wait_for_next_frame(uint64_t last_seq, std::chrono::milliseconds timeout);


private:
    void reader_loop();

    // Streaming-friendly: parse a line you already read (no serial read inside)
    bool set_response_values_from_stream(const std::string& response, bool verbose);

    mutable std::mutex state_mtx_;
    mutable std::condition_variable state_cv_;
    std::atomic<bool> reader_running_{false};
    std::thread reader_thread_;
    std::atomic<uint64_t> rx_seq_{0};
    std::string rx_buffer_;
    std::chrono::steady_clock::time_point last_message_time_;
    static constexpr int MESSAGE_TIMEOUT_MS = 10000;  // Timeout for no messages received

    void find_hall_sensor_positions();

    std::shared_ptr<SerialPort> serial_;
    rclcpp::Logger logger_;
    Motor motor_;  // Motor configuration parameters
    
    // Motor state
    std::array<int, 4>  starting_positions = {0, 0, 0, 0}; // Starting positions for the motors
    std::array<int, 4> target_positions_{0, 0, 0, 0};
    std::array<int, 10> response_values_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::array<bool, 4> blocked_{false, false, false, false};
    std::array<bool, 4> maxed_{false, false, false, false};
    
    // Calibration constants (hardware-specific offsets)
    const int HALL1_OFFSET = -120; // in pulses, to align the hall sensor with the physical 0 degree position
    const int HALL2_OFFSET = -90; // in pulses, to align the hall sensor with the physical 0 degree position
};