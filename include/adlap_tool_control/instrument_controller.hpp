#pragma once

#include "adlap_tool_control/motor_controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <array>
#include <deque>
#include <memory>





class InstrumentController
{
public:
    InstrumentController(MotorController& motor_controller, rclcpp::Logger logger);

private:
    // Instrument control methods
    void manual_adjustment();
    
    // Calculation methods
    int get_relative_motor1_value_for_angle(double radians);
    void set_angles(double roll, double pitch, double yaw, double gripper);
    std::array<int, 4>& calculate_motor_positions_based_on_smoothed_angles();
    void drive_motors(const std::array<int, 4>& m_array);

    /// Push a new sample, cap history to smoothing_factor_, and return the mean.
    static double update_history_and_get_mean(std::deque<double>& history, double sample, std::size_t max_size);
    
    // Member variables
    MotorController motor_controller_;
    rclcpp::Logger logger_;
    std::deque<double> roll_history_; // History for smoothing
    std::deque<double> pitch_history_; // History for smoothing
    std::deque<double> yaw_history_; // History for smoothing
    std::deque<double> gripper_history_; // History for smoothing

    int smoothing_factor_ = 20;
    
    // Instrument state
    double smoothed_roll_ = 0.0;
    double smoothed_pitch_ = 0.0;
    double smoothed_yaw_ = 0.0;
    double smoothed_gripper_ = 0.0;
    int bend_angle_ = 0; // Current articulation angles
    double absolute_omega_ = 0.0; // Absolute omega value for shortest rotation calculation

    // Constants
    const double TWO_PI = 2.0 * M_PI;   

    /* The lower motors bend the shaft. The difference approximates the max bending.*/
    const int LOWER_MOTORS_MAX_DIFFERENCE = 160; // in pulses, should be changed to radians
    const double LOWER_MOTORS_MAX_BEND_ANGLE = 50.0; // degrees
};