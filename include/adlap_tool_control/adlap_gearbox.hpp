#pragma once

#include "adlap_tool_control/motor_controller.hpp"
#include "rclcpp/rclcpp.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

class GearboxParameters
{
public:
    float upper_motor_factor;
    float lower_motor_factor;

    std::vector<int> upper_motors;
    std::vector<int> lower_motors;

    int lower_motors_play_deg;
    int lower_motor_start_offset_deg;
    int gripper_start_offset_deg;
    int hall1_offset_deg;
    int hall2_offset_deg;

    // static GearboxParameters create_default();
    static GearboxParameters from_yaml(const std::string& yaml_path);
};

class Gearbox
{
public:
    // Member variables
    MotorController& motor_controller;
    GearboxParameters params;
        
    // float upper_motor_factor;    // Gear ratio factor for upper motors (if different)
    // int lower_motors_play;       // Backlash compensation in degrees

    // Constructor
    Gearbox(MotorController& motor_controller, const GearboxParameters& params, rclcpp::Logger logger):motor_controller(motor_controller), params(params), logger_(logger)
    {
        lower_motor_start_offset_ =
            params.lower_motor_start_offset_deg * get_pulses_per_degree(1);

        gripper_start_offset_ =
            params.gripper_start_offset_deg * get_pulses_per_degree(3);

        RCLCPP_INFO(
            logger_,
            "Initialized Gearbox with upper_motor_factor=%.3f and lower_motors_play=%d degrees",
            params.upper_motor_factor,
            params.lower_motors_play_deg
        );
    }

    // static Gearbox version_1(MotorController& motor_controller, rclcpp::Logger logger)
    // // float upper_motor_factor = 25.0f / 15.0f, int lower_motors_play = 15) 
    // {
    //     return Gearbox(motor_controller, GearboxParameters::create_default(), logger);
    // }
        // Gearbox gearbox(motor_controller, logger);
        // gearbox.upper_motor_factor = upper_motor_factor;
        // gearbox.lower_motors_play = lower_motors_play;

        // gearbox.lower_motor_start_offset_ = 20 * gearbox.get_pulses_per_degree(1); // Add some extra to compensate for backlash
        // gearbox.gripper_start_offset_ = 200 * gearbox.get_pulses_per_degree(3); // Move it away from the hard stop

        // RCLCPP_INFO(logger, "Initialized Gearbox with upper_motor_factor=%.2f and lower_motors_play=%d degrees", upper_motor_factor, lower_motors_play);
        // return gearbox;

    bool is_upper_motor(int motor_index) const 
    // { return motor_index == 0 || motor_index == 3; }
    {
        return std::find(
            params.upper_motors.begin(),
            params.upper_motors.end(),
            motor_index
        ) != params.upper_motors.end();
    }
    int get_pulses_per_rotation(int motor_index) const { 
        int base_pulses = motor_controller.motors[motor_index].get_pulses_per_rotation();
        return is_upper_motor(motor_index) ? static_cast<int>(base_pulses * params.upper_motor_factor) : static_cast<int>(base_pulses * params.lower_motor_factor);
    }
    float get_pulses_per_degree(int motor_index) const { 
        int pulses = get_pulses_per_rotation(motor_index);
        return static_cast<float>(pulses) / 360.0f;
    }
    int get_pulses_lower_motors_play() const { return static_cast<int>(std::lround(params.lower_motors_play_deg * get_pulses_per_degree(1))); } // Using motor 1, since it doesn't matter which of the lower motors we use for this compensation

    void couple_sequence();
    void setup_motors();
    

private:
    // Private member variables
    rclcpp::Logger logger_;

    // Hardware-specific offsets       
    int lower_motor_start_offset_;
    int gripper_start_offset_;
    // const int HALL1_OFFSET = 4; // in degrees from 6 o'clock position
    // const int HALL2_OFFSET = 10; // in degrees from 6 o'clock position
};