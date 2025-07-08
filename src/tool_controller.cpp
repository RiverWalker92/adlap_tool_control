#include "adlap_tool_control/serial_port.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cmath>
#include <format>
//#include <signal.h>
//#include <stdio.h>
#include <termios.h>
//#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

using namespace std::chrono_literals;

const std::string STATUS_TOPIC = "/tool_status";
const std::string TOOL_CONTROL_TOPIC = "/instrument_angles";

const float UPPER_MOTOR_FACTOR = 15.0f/25;
/* Motor gearbox is 150:1. The motors are equipped with a 3 pulse per rotation hall encoder.
This means 450 pulses equals 1 full rotation.*/
const int MOTOR_PULSES_PER_ROTATION = 450;
/* The lower motors bend the shaft. The difference approximates the max bending.*/
const int LOWER_MOTORS_MAX_DIFFERENCE = 160;
const int MIN_WAIT_TIME = 40; // Minimum wait time in milliseconds for the motors to respond

std::array<int, 4>  starting_positions = {0, 0, 0, 0}; // Starting positions for the motors
std::array<int, 4>  positions = {0, 0, 0, 0}; // Current positions of the motors
std::array<int, 6>  response_values = {0, 0, 0, 0, 0, 0}; // Response values from the motors
int blocking_current = 3000; // Amperage threshold for blocking the motors
bool blocked[4] = {false, false, false, false}; // Flag to indicate if the motors are blocked
int max_current = 8000; // Maximum current value for the motors
bool maxed[4] = {false, false, false, false}; // Flag to indicate if the motors are at maximum current

constexpr int8_t KEYCODE_RIGHT = 0x43;
constexpr int8_t KEYCODE_LEFT = 0x44;
constexpr int8_t KEYCODE_UP = 0x41;
constexpr int8_t KEYCODE_DOWN = 0x42;
constexpr int8_t KEYCODE_W = 0x77;
constexpr int8_t KEYCODE_A = 0x61;
constexpr int8_t KEYCODE_S = 0x73;
constexpr int8_t KEYCODE_D = 0x64;
constexpr int8_t KEYCODE_Q = 0x71;
constexpr int8_t KEYCODE_R = 0x72;
constexpr int8_t KEYCODE_J = 0x6A;
constexpr int8_t KEYCODE_T = 0x74;
constexpr int8_t KEYCODE_E = 0x65;
// A class for reading the key inputs from the terminal
class KeyboardReader
{
public:
  KeyboardReader() : file_descriptor_(0)
  {
    // get the console in raw mode
    tcgetattr(file_descriptor_, &cooked_);
    struct termios raw;
    memcpy(&raw, &cooked_, sizeof(struct termios));
    raw.c_lflag &= ~(ICANON | ECHO);
    // Setting a new line, then end of file
    raw.c_cc[VEOL] = 1;
    raw.c_cc[VEOF] = 2;
    tcsetattr(file_descriptor_, TCSANOW, &raw);
  }
  void readOne(char* c)
  {
    int rc = read(file_descriptor_, c, 1);
    if (rc < 0)
    {
      throw std::runtime_error("read failed");
    }
  }
  void shutdown()
  {
    tcsetattr(file_descriptor_, TCSANOW, &cooked_);
  }

private:
  int file_descriptor_;
  struct termios cooked_;
};

class ToolController : public rclcpp::Node
{
  public:
  ToolController(std::shared_ptr<SerialPort> serial)
    : Node("tool_control_node"), serial_(serial)
    {
      this->declare_parameter("side", "right");

      publisher_ = this->create_publisher<std_msgs::msg::String>("~" + STATUS_TOPIC, 10);
      subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "~" + TOOL_CONTROL_TOPIC, 10, std::bind(&ToolController::topic_callback, this, std::placeholders::_1));  

      auto message = motor_message(positions);
      RCLCPP_INFO(this->get_logger(), "Writing: '%s'", message.c_str());
      serial_->writeData(message);

      // Wait for a short period to allow the device to respond
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      std::string response = serial_->readData();
      if (!response.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received: '%s'", response.c_str());
      }else {
        RCLCPP_ERROR(this->get_logger(), "Failed to read from serial port");
      }

      initialize();
    }

  private:
    std::string motor_message(const std::array<int, 4>& m_array){
      std::string response = 
        std::to_string(m_array[0]) + ", " 
        + std::to_string(m_array[1]) + ", " 
        + std::to_string(m_array[2]) + ", " 
        + std::to_string(m_array[3]) + "\n";
      return response;
    }

    void send_relative_motor_positions(const std::array<int, 4>& m_array) {
      send_relative_motor_positions(m_array[0], m_array[1], m_array[2], m_array[3]);
    }

    void send_relative_motor_positions(int m0, int m1, int m2, int m3) {
      std::array<int, 4>  new_positions = {0, 0, 0, 0};
      new_positions[0] = positions[0] + m0;
      new_positions[1] = positions[1] + m1;
      new_positions[2] = positions[2] + m2;
      new_positions[3] = positions[3] + m3;
      send_motor_positions(new_positions);
    }

    void send_motor_positions(int m0, int m1, int m2, int m3) {
      std::array<int, 4>  new_positions = {m0, m1, m2, m3};
      send_motor_positions(new_positions);
    }

    void send_motor_positions(const std::array<int, 4>& new_positions) {
      std::string message = motor_message(new_positions);
      RCLCPP_INFO(this->get_logger(), "Writing: '%s'", message.c_str());
      serial_->writeData(message);
      rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
      bool motor_status = set_response_values();
      if (!motor_status) {
        RCLCPP_ERROR(this->get_logger(), "Motor response not ok, reversing movement");
        serial_->writeData(motor_message(positions));
        rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
        motor_status = set_response_values();

        
        // TODO: This is a workaround for the motors not responding the first loop.
        // Write the positions again, cause the current return values are one loop behind
        
        if (!motor_status) {
          RCLCPP_ERROR(this->get_logger(), "Motor response still not ok, reversing further");
          std::array<int, 4> rev_positions;
          for (size_t i = 0; i < new_positions.size(); ++i) {
              rev_positions[i] = 2 * positions[i] - new_positions[i];
          }
          message = motor_message(rev_positions);
          RCLCPP_INFO(this->get_logger(), "Writing: '%s'", message.c_str());
          serial_->writeData(message);
          rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
          motor_status = set_response_values();
          // Write again, cause the current return values are one loop behind
          serial_->writeData(message);
          rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
          motor_status = set_response_values();
        }
        // End of workaround
        
        if (!motor_status) {
          throw std::runtime_error("Motor response still not ok after reversing movement");
        }
      }else{
        positions = new_positions;
      }
      return;
    }

    bool set_response_values(){
      bool response_ok = true;
      std::string response = serial_->readData();
      if (!response.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received: '%s'", response.c_str());
      }else {
        RCLCPP_ERROR(this->get_logger(), "Failed to read from serial port");
      }
      std::vector<int> values;
      std::stringstream ss(response);
      std::string item;
      while (std::getline(ss, item, ',')) {
        values.push_back(std::stoi(item)); // Convert to int and store
      }
      if (values.size() < 6) {
        RCLCPP_ERROR(this->get_logger(), "Received less than 6 values from the motor: %zu", values.size());
        throw std::runtime_error("Received less than 6 values from the motor");
      }
      for (int i = 0; i < 6; ++i) {
        // Update the response values

        //TODO: the order of the current values is not the same as the order of the motors
        if (i == 2) {
          response_values[3] = values[2];
        }
        else if (i == 3)
        {
          response_values[2] = values[3];
        }
        else{
          response_values[i] = values[i];
        }
      }
      for (int i = 0; i < 4; ++i) {
        // Check if the current value exceeds the blocking current
        if (response_values[i] > blocking_current){
          blocked[i] = true;
          RCLCPP_WARN(this->get_logger(), "Motor %d value out of range: %d", i, response_values[i]);
        }
        else {
          blocked[i] = false;
        }
        // Check if the current value exceeds the maximum current
        if(response_values[i] > max_current){
          maxed[i] = true;
          response_ok = false;
          RCLCPP_ERROR(this->get_logger(), "Motor %d current exceeded maximum: %d", i, response_values[i]);
        }
        else {
          maxed[i] = false;
        }
      }
      return response_ok;
    }

    void initialize(){

      // Couple the upper motors
      for (int i : {0,3}){
        for (int j : {10, -10}){
          // Move the motor till it is blocked (coupled and reached its end position)
          std::array<int, 4> driving_values  = {0, 0, 0, 0};
          driving_values[i] = j;
          do{
            send_relative_motor_positions(driving_values);
          } while (!blocked[i]);
          // Unblock the motor
          driving_values[i] = -j;
          send_relative_motor_positions(driving_values);
        }
      }

      // TODO: one hall value is low (motor1) when passing and the other is high (motor2).
      int hall1_position = 0;
      int hall1_value = 1000;
      int hall2_position = 0;
      int hall2_value = 0;
      for (int i = 0; i < 135; i++) {
        send_relative_motor_positions(0,10,10,0);
        if (response_values[4] < hall1_value) {
          hall1_value = response_values[4];
          hall1_position = positions[1] % MOTOR_PULSES_PER_ROTATION;
          RCLCPP_INFO(this->get_logger(), "Hall1 position: '%d'", hall1_position);
        }
        if (response_values[5] > hall2_value) {
          hall2_value = response_values[5];
          hall2_position = positions[2] % MOTOR_PULSES_PER_ROTATION;
          RCLCPP_INFO(this->get_logger(), "Hall2 position: '%d'", hall2_position);
        }
      }  

      int hall1_correction = hall1_position - positions[1] % MOTOR_PULSES_PER_ROTATION;
      RCLCPP_INFO(this->get_logger(), "Setting hall1 position: '%d'", hall1_correction);
      int hall2_correction = hall2_position - positions[2] % MOTOR_PULSES_PER_ROTATION;
      RCLCPP_INFO(this->get_logger(), "Setting hall1 position: '%d'", hall1_correction);
      send_relative_motor_positions(0,hall1_correction,hall2_correction,0);

      KeyboardReader input;
      char c;
      for (;;)
      {
        // get the next event from the keyboard
        try
        {
          input.readOne(&c);
        }
        catch (const std::runtime_error&)
        {
          perror("read():");
          return;
        }
        RCLCPP_INFO(this->get_logger(), "value: 0x%02X\n", c);
        switch (c)
        {
          case KEYCODE_RIGHT:
            RCLCPP_DEBUG(this->get_logger(), "RIGHT");
            send_relative_motor_positions(0,10,0,0);
            break;
          case KEYCODE_LEFT:
            RCLCPP_DEBUG(this->get_logger(), "LEFT");
            send_relative_motor_positions(0,-10,0,0);
            break;
          case KEYCODE_UP:
            RCLCPP_DEBUG(this->get_logger(), "UP");
            send_relative_motor_positions(0,0,10,0);
            break;
          case KEYCODE_DOWN:
            RCLCPP_DEBUG(this->get_logger(), "DOWN");
            send_relative_motor_positions(0,0,-10,0);
            break;
          case KEYCODE_W:
            RCLCPP_DEBUG(this->get_logger(), "W");
            send_relative_motor_positions(10,0,0,0);
            break;
          case KEYCODE_S:
            RCLCPP_DEBUG(this->get_logger(), "S");
            send_relative_motor_positions(-10,0,0,0);
            break;
          case KEYCODE_D:
            RCLCPP_DEBUG(this->get_logger(), "D");
            send_relative_motor_positions(10,0,0,10);
            break;
          case KEYCODE_A:
            RCLCPP_DEBUG(this->get_logger(), "A");
            send_relative_motor_positions(-10,0,0,-10);
            break;
          case KEYCODE_E:
            send_relative_motor_positions(0,10,10,0);
            RCLCPP_DEBUG(this->get_logger(), "E");
            break;
          case KEYCODE_Q:
            send_relative_motor_positions(0,-10,-10,0);
            RCLCPP_DEBUG(this->get_logger(), "Q");
            break;
          case KEYCODE_J:
            RCLCPP_DEBUG(this->get_logger(), "J");
            input.shutdown();
            starting_positions = positions;
            return;
          default:
            RCLCPP_DEBUG(this->get_logger(), "Unknown key pressed: 0x%02X", c);
            continue;
        }
      }
    }

    void topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
      if (msg->data.size() < 4) {
        RCLCPP_WARN(this->get_logger(), "Received data size less than 4");
        return;
      }
      std::vector<double> angles = msg->data;
      RCLCPP_INFO(this->get_logger(), "I heard array: '%f' '%f' '%f' '%f'", angles[0], angles[1], angles[2], angles[3]);

      // Tip rotation
      int tip_rot = static_cast<int>(std::round(angles[0] * MOTOR_PULSES_PER_ROTATION / (2 * UPPER_MOTOR_FACTOR * M_PI))); // Convert degrees to radians

      int gripper_factor = static_cast<int>(std::round(0.6f * MOTOR_PULSES_PER_ROTATION / UPPER_MOTOR_FACTOR));
      //gripper position
      int gripper_pos = static_cast<int>(std::round(8 * angles[3] * MOTOR_PULSES_PER_ROTATION / (2 * UPPER_MOTOR_FACTOR * M_PI))); // Convert degrees to radians

      RCLCPP_INFO(this->get_logger(), "tip_rot, gripper_factor, gripper_pos: '%d' '%d' '%d'", tip_rot, gripper_factor, gripper_pos);
      send_motor_positions(
        static_cast<int>(std::round(starting_positions[0] + tip_rot - gripper_factor - gripper_pos)),
        static_cast<int>(std::round(starting_positions[1])),
        static_cast<int>(std::round(starting_positions[2])),
        static_cast<int>(std::round(starting_positions[3] + tip_rot))
      );

      auto status_msg = std_msgs::msg::String();
      status_msg.data = "Currents: ";
      for (size_t i = 0; i < 6; ++i){
        status_msg.data += std::to_string(response_values[i]) + (i < 5 ? ", " : ";");
      } 
      RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", status_msg.data.c_str());
      publisher_->publish(status_msg);
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
    std::shared_ptr<SerialPort> serial_;
};

int main(int argc, char * argv[])
{
  int ret = 0;
  rclcpp::init(argc, argv);

  try {
    auto serial = std::make_shared<SerialPort>("/dev/ttyUSB0", 115200);

    if (!serial->openPort()) {
      throw std::runtime_error("Failed to open serial port");
    }

    rclcpp::spin(std::make_shared<ToolController>(serial));


    serial->closePort();
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Exception: %s", e.what());
    ret = 1;
  }  
  rclcpp::shutdown();
  return ret;
}