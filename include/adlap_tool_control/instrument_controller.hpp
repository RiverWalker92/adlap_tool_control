#pragma once

#include "adlap_tool_control/motor_controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <array>
#include <deque>
#include <memory>
#include <limits>





class InstrumentController
{
public:
    InstrumentController(MotorController& motor_controller, rclcpp::Logger logger);

    // Instrument control methods
    void manual_adjustment();
    void set_angles(double roll, double pitch, double yaw, double gripper);
    std::array<double, 4> angles_from_motors(const std::array<int, 4>& m_array);
private:
    // Calculation methods
    int get_motor2_value_for_angle(double radians);
    std::array<int, 4> calculate_motor_positions_from_angles();

    /// Push a new sample, cap history to smoothing_factor_, and return the mean.
    static double update_history_and_get_mean(std::deque<double>& history, double sample, std::size_t max_size,
        double min_value = -std::numeric_limits<double>::infinity(),
        double max_value = std::numeric_limits<double>::infinity());
    
    // Member variables
    MotorController& motor_controller_;
    rclcpp::Logger logger_;
    std::deque<double> roll_history_; // History for smoothing
    std::deque<double> pitch_history_; // History for smoothing
    std::deque<double> yaw_history_; // History for smoothing
    std::deque<double> gripper_history_; // History for smoothing

    int smoothing_factor_ = 1; // Number of samples to average for smoothing
    int bend_play_compensation_ = 0; // Current compensation for the bend play, updated after each motor command
    
    // Instrument state
    double smoothed_roll_ = 0.0;
    double smoothed_pitch_ = 0.01;
    double smoothed_yaw_ = 0.0;
    double smoothed_gripper_ = 0.0;
    double absolute_omega_ = 0.0; // Absolute omega value for shortest rotation calculation

    // Constants
    const double TWO_PI = 2.0 * M_PI;   

    /* The lower motors bend the shaft. This factor relates the motor position to the bend angle.*/
    const double BEND_FACTOR = 1.5;
    const int MAX_BEND_ANGLE_DEGREES = 45; // Maximum bend angle in degrees, used for clamping the input
};