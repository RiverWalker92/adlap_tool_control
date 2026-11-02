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
    int bend_play_compensation = 0; // Current compensation for the bend play, updated after each motor command
    int m1_m2_offset = 0;

    // Constructor
    Gearbox(MotorController& motor_controller, rclcpp::Logger logger, float upper_motor_factor = 25.0f / 15.0f, int lower_motors_play = 15);
    ~Gearbox();

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
    void save_state_to_json() const;
    bool load_state_from_json();
    

private:
    // Private member variables
    rclcpp::Logger logger_;

    // Hardware-specific offsets       
    int lower_motor_start_offset_;
    int gripper_start_offset_;
    const int HALL1_OFFSET = 4; // in degrees from 6 o'clock position
    const int HALL2_OFFSET = 10; // in degrees from 6 o'clock position
};