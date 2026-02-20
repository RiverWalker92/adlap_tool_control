#include "adlap_tool_control/motor_controller.hpp"
#include <sstream>
#include <stdexcept>
#include <thread>
#include <chrono>

MotorController::MotorController(std::shared_ptr<SerialPort> serial, rclcpp::Logger logger)
    : serial_(serial), logger_(logger)
{   
    //TODO: read starting positions from the motor and set them as the current positions
    // Test the serial port connection
    auto message = motor_message(positions_);
    RCLCPP_INFO(logger_, "Writing: '%s'", message.c_str());
    serial_->write_data(message);

    // Wait for a short period to allow the device to respond
    std::this_thread::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
    std::string response = serial_->read_data();
    if (!response.empty()) {
      RCLCPP_INFO(logger_, "Received: '%s'", response.c_str());
    }else {
      throw std::runtime_error("Failed to read from serial port");
    }
}

std::string MotorController::motor_message(const std::array<int, 4>& m_array)
{
    std::string response = 
        std::to_string(m_array[0]) + ", " 
        + std::to_string(m_array[1]) + ", " 
        + std::to_string(m_array[2]) + ", " 
        + std::to_string(m_array[3]) + "\n";
    return response;
}

void MotorController::send_relative_motor_positions(const std::array<int, 4>& m_array, bool verbose)
{
    send_relative_motor_positions(m_array[0], m_array[1], m_array[2], m_array[3], verbose);
}

void MotorController::send_relative_motor_positions(int m0, int m1, int m2, int m3, bool verbose)
{
    std::array<int, 4> new_positions = {0, 0, 0, 0};
    new_positions[0] = positions_[0] + m0;
    new_positions[1] = positions_[1] + m1;
    new_positions[2] = positions_[2] + m2;
    new_positions[3] = positions_[3] + m3;
    
    send_motor_positions(new_positions, verbose);
}

void MotorController::send_motor_positions(int m0, int m1, int m2, int m3, bool verbose)
{
    std::array<int, 4> new_positions = {m0, m1, m2, m3};
    send_motor_positions(new_positions, verbose);
}

/// @brief Send new absolute motor positions to the motor controller
/// @param new_positions Array of 4 integers representing the new motor positions
/// @throws std::runtime_error if the motor response is not ok after reversing movement
void MotorController::send_motor_positions(const std::array<int, 4>& new_positions, bool verbose)
{
    std::string message = motor_message(new_positions);

    if (verbose) {
        RCLCPP_INFO(logger_, "Writing: '%s'", message.c_str());
    }
    
    serial_->write_data(message);
    rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
    bool motor_status = set_response_values(verbose);

    // TODO: This is a workaround, cause the current return values are one loop behind
    serial_->write_data(message);
    rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
    motor_status = set_response_values(verbose);
    // End of workaround

    if (!motor_status) {
        RCLCPP_ERROR(logger_, "Motor response not ok, reversing movement");
        serial_->write_data(motor_message(positions_));
        rclcpp::sleep_for(std::chrono::milliseconds(500));
        motor_status = set_response_values(verbose);

        // TODO: This is a workaround, cause the current return values are one loop behind
        serial_->write_data(motor_message(positions_));
        rclcpp::sleep_for(std::chrono::milliseconds(500));
        motor_status = set_response_values(verbose);
        // End of workaround

        if (!motor_status) {
            throw std::runtime_error("Motor response still not ok after reversing movement");
        }
    } else {
        positions_ = new_positions;
    }
}

/// @brief Read and set response values from the motor
/// @return True if the currents are within limits, false otherwise
bool MotorController::set_response_values(bool verbose)
{
    bool response_ok = true;
    std::string response = serial_->read_data();
    
    if (!response.empty()) {
        if (verbose) {
            RCLCPP_INFO(logger_, "Raw response: '%s'", response.c_str());
        }
    } else {
        RCLCPP_ERROR(logger_, "Failed to read from serial port");
    }
    
    std::vector<int> values;
    std::stringstream ss(response);
    std::string item;
    while (std::getline(ss, item, ' ')) {
        values.push_back(std::stoi(item));
    }
    
    if (values.size() < 6) {
        RCLCPP_ERROR(logger_, "Received less than 6 values from the motor: %zu", values.size());
        throw std::runtime_error("Received less than 6 values from the motor");
    }
    
    for (int i = 0; i < 6; ++i) {
        response_values_[i] = values[i];
    }
    
    for (int i = 0; i < 4; ++i) {
        // Check if the current value exceeds the blocking current
        if (response_values_[i] > BLOCKING_CURRENT) {
            blocked_[i] = true;
            RCLCPP_WARN(logger_, "Motor %d value out of range: %d", i, response_values_[i]);
        } else {
            blocked_[i] = false;
        }
        
        // Check if the current value exceeds the maximum current
        if (response_values_[i] > MAX_CURRENT) {
            maxed_[i] = true;
            response_ok = false;
            RCLCPP_ERROR(logger_, "Motor %d current exceeded maximum: %d", i, response_values_[i]);
        } else {
            maxed_[i] = false;
        }
    }
    
    return response_ok;
}

/// @brief Couple the motors to the gearbox
void MotorController::couple_sequence(){
    // Making some rotations to get the pins do drop into the holes of the gearbox
    for (int i : {0,1,2,3}){
      for (int j : {10, -10}){
        // Move the motor till it is blocked (coupled and reached its end position)
        std::array<int, 4> driving_values  = {0, 0, 0, 0};
        driving_values[i] = j;
        do{
          send_relative_motor_positions(driving_values);
        } while (!blocked_[i]);
        // Unblock the motor
        driving_values[i] = -5*j;
        send_relative_motor_positions(driving_values);
      }
    }
  }

void MotorController::update_starting_positions() {
    starting_positions = positions_;
}  

void MotorController::find_hall_sensor_positions() {
    // Couple the lower motors
    // These motors also control the front gears, which have to be aligned for the tool to be able to be inserted
    // The hall sensors can be used to find the correct position

    int hall1_value = response_values_[4];
    int hall1_up = 0;
    int hall1_down = 0;
    int hall2_value = response_values_[5];
    int hall2_up = 0;
    int hall2_down = 0;
    int step_size = 10;
    for (int i = 0; i < 3*MOTOR_PULSES_PER_ROTATION/step_size; i++) {
      send_relative_motor_positions(0,step_size,step_size,0, false);
      if (response_values_[4] == 1 and hall1_value == 0) {
        hall1_up = positions_[1] % MOTOR_PULSES_PER_ROTATION;
        RCLCPP_INFO(logger_, "Hall1 up: '%d'", hall1_up);
      }
      if (response_values_[4] == 0 and hall1_value == 1) {
        hall1_down = positions_[1] % MOTOR_PULSES_PER_ROTATION;
        RCLCPP_INFO(logger_, "Hall1 down: '%d'", hall1_down);
      }
      if (response_values_[5] == 1 and hall2_value == 0) {
        hall2_up = positions_[2] % MOTOR_PULSES_PER_ROTATION;
        RCLCPP_INFO(logger_, "Hall2 up: '%d'", hall2_up);
      }
      if (response_values_[5] == 0 and hall2_value == 1) {
        hall2_down = positions_[2] % MOTOR_PULSES_PER_ROTATION;
        RCLCPP_INFO(logger_, "Hall2 down: '%d'", hall2_down);
      }
      hall1_value = response_values_[4];
      hall2_value = response_values_[5];
    }  
    // Move back to the position where the hall sensor triggers (assumed to be the correct position for instrument insertion)
    if (hall1_up > hall1_down) {
      hall1_down += MOTOR_PULSES_PER_ROTATION;
    }
    if (hall2_up > hall2_down) {
      hall2_down += MOTOR_PULSES_PER_ROTATION;
    }
    int hall1_position = (hall1_up + hall1_down) / 2 + HALL1_OFFSET;
    int hall1_correction = hall1_position - positions_[1] % MOTOR_PULSES_PER_ROTATION;
    RCLCPP_INFO(logger_, "Setting hall1 position: '%d'", hall1_correction);
    int hall2_position = (hall2_up + hall2_down) / 2 + HALL2_OFFSET;
    int hall2_correction = hall2_position - positions_[2] % MOTOR_PULSES_PER_ROTATION;
    RCLCPP_INFO(logger_, "Setting hall2 position: '%d'", hall2_correction);
    send_relative_motor_positions(0,hall1_correction,hall2_correction,0);
  }