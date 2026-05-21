#pragma once

#include "adlap_tool_control/motor_controller.hpp"
#include "rclcpp/rclcpp.hpp"

class Gearbox
{
public:
    // Member variables
    MotorController& motor_controller;
        
    float upper_motor_factor;    // Gear ratio factor for upper motors (if different)
    int lower_motors_play;       // Backlash compensation in degrees

    // Constructor
    Gearbox(MotorController& motor_controller, rclcpp::Logger logger):motor_controller(motor_controller), logger_(logger) {}

    static Gearbox version_1(MotorController& motor_controller, rclcpp::Logger logger, float upper_motor_factor = 25.0f / 15.0f, int lower_motors_play = 15) {
        Gearbox gearbox(motor_controller, logger);
        gearbox.upper_motor_factor = upper_motor_factor;
        gearbox.lower_motors_play = lower_motors_play;

        gearbox.lower_motor_start_offset_ = 20 * gearbox.get_pulses_per_degree(1); // Add some extra to compensate for backlash
        gearbox.gripper_start_offset_ = 200 * gearbox.get_pulses_per_degree(3); // Move it away from the hard stop

        RCLCPP_INFO(logger, "Initialized Gearbox with upper_motor_factor=%.2f and lower_motors_play=%d degrees", upper_motor_factor, lower_motors_play);
        return gearbox;
    }

    bool is_upper_motor(int motor_index) const { return motor_index == 0 || motor_index == 3; }

    int get_pulses_per_rotation(int motor_index) const { 
        int base_pulses = motor_controller.motors[motor_index].get_pulses_per_rotation();
        return is_upper_motor(motor_index) ? static_cast<int>(base_pulses * upper_motor_factor) : base_pulses;
    }
    float get_pulses_per_degree(int motor_index) const { 
        int pulses = get_pulses_per_rotation(motor_index);
        return static_cast<float>(pulses) / 360.0f;
    }
    int get_pulses_lower_motors_play() const { return static_cast<int>(std::lround(lower_motors_play * get_pulses_per_degree(1))); } // Using motor 1, since it doesn't matter which of the lower motors we use for this compensation

    void couple_sequence();
    void setup_motors();
    

private:
    // Private member variables
    rclcpp::Logger logger_;

    // Hardware-specific offsets       
    int lower_motor_start_offset_;
    int gripper_start_offset_;
    const int HALL1_OFFSET = 4; // in degrees from 6 o'clock position
    const int HALL2_OFFSET = 10; // in degrees from 6 o'clock position
};