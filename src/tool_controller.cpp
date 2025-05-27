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
#include "geometry_msgs/msg/pose_stamped.hpp"

using namespace std::chrono_literals;

constexpr char TOPIC[] = "tool_status";
constexpr char POSE_TOPIC[] = "/servo_node/pose_target_cmds";

const float UPPER_MOTOR_FACTOR = 15/25;
/* Motor gearbox is 150:1. The motors are equipped with a 3 pulse per rotation hall encoder.
This means 450 pulses equals 1 full rotation.*/
const int MOTOR_PULSES_PER_ROTATION = 450;
const int LOWER_MOTORS_MAX_DIFFERENCE = 160;
const int MIN_WAIT_TIME = 40; // Minimum wait time in milliseconds for the motors to respond

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

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
    : Node("tool_controller"), serial_(serial)
    {
      this->declare_parameter("side", "right");

      publisher_ = this->create_publisher<std_msgs::msg::String>(TOPIC, 10);
      auto side = this->get_parameter("side").as_string();
      //TODO: use side to determine the topic name
      auto topic_name = "/servo_node/right_pose_target_cmds";
      subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        topic_name, 10, std::bind(&ToolController::topic_callback, this, std::placeholders::_1));  

      auto message = motor_message(0,0,0,0);
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
    std::string motor_message(int motor0, int motor1, int motor2, int motor3){
      std::string response = 
        std::to_string(motor0) + ", " 
        + std::to_string(motor1) + ", " 
        + std::to_string(motor2) + ", " 
        + std::to_string(motor3) + "\n";
      return response;
    }

    std::vector<int> response_values(){
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
      return values;
    }

    bool not_blocked()
    {
      std::vector<int> values = response_values();
      for (int i = 0; i < 4; i++)
      {
        if (values[i] > 3000) {
          RCLCPP_ERROR(this->get_logger(), "Motor %d value out of range: %d", i, values[i]);
          return false;
        }
      }
      return true;   
    }

    void initialize(){
      std::string response;
      int m0 = 0, m1 = 0, m2 = 0, m3 = 0;
      
      do{
        m0 += 10;
        serial_->writeData(motor_message(m0,m1,m2,m3));
        rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
      } while (not_blocked());
      do {
        m0 -= 10;
        serial_->writeData(motor_message(m0,m1,m2,m3));
        rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
      } while (!not_blocked());

      do {
        m3 -= 10;
        serial_->writeData(motor_message(m0,m1,m2,m3));
        rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
      } while (not_blocked());
      do {
        m3 += 10;
        serial_->writeData(motor_message(m0,m1,m2,m3));
        rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
      } while (!not_blocked());

      // TODO: one hall value is low (motor1) when passing and the other is high (motor2).
      int hall1_position = 0;
      int hall1_value = 1000;
      int hall2_position = 0;
      int hall2_value = 0;
      for (int i = 0; i < 135; i++) {
        m1 += 10;
        m2 += 10;
        serial_->writeData(motor_message(m0,m1,m2,m3));
        rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
        std::vector<int> values = response_values();
        if (values[4] < hall1_value) {
          hall1_value = values[4];
          hall1_position = m1 % MOTOR_PULSES_PER_ROTATION;
          RCLCPP_INFO(this->get_logger(), "Hall1 position: '%d'", hall1_position);
        }
        if (values[5] > hall2_value) {
          hall2_value = values[5];
          hall2_position = m2 % MOTOR_PULSES_PER_ROTATION;
          RCLCPP_INFO(this->get_logger(), "Hall2 position: '%d'", hall2_position);
        }
      }  

      RCLCPP_INFO(this->get_logger(), "Setting hall1 position: '%d'", hall1_position - m1 % MOTOR_PULSES_PER_ROTATION);
      m1 += hall1_position - m1 % MOTOR_PULSES_PER_ROTATION;
      m2 += hall2_position - m2 % MOTOR_PULSES_PER_ROTATION;
      serial_->writeData(motor_message(m0,m1,m2,m3));

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
            m1 += 10;
            break;
          case KEYCODE_LEFT:
            RCLCPP_DEBUG(this->get_logger(), "LEFT");
            m1 -= 10;
            break;
          case KEYCODE_UP:
            RCLCPP_DEBUG(this->get_logger(), "UP");
            m2 += 10;
            break;
          case KEYCODE_DOWN:
            RCLCPP_DEBUG(this->get_logger(), "DOWN");
            m2 -= 10;
            break;
          case KEYCODE_W:
            RCLCPP_DEBUG(this->get_logger(), "W");
            m0 += 10;
            break;
          case KEYCODE_S:
            RCLCPP_DEBUG(this->get_logger(), "S");
            m0 -= 10;
            break;
          case KEYCODE_D:
            RCLCPP_DEBUG(this->get_logger(), "D");
            m3 += 10;
            m0 += 10;
            break;
          case KEYCODE_A:
            RCLCPP_DEBUG(this->get_logger(), "A");
            m3 -= 10;
            m0 -= 10;
            break;
          case KEYCODE_E:
            m1 += 10;
            m2 += 10;
            RCLCPP_DEBUG(this->get_logger(), "Q");
            break;
          case KEYCODE_Q:
            m1 -= 10;
            m2 -= 10;
            RCLCPP_DEBUG(this->get_logger(), "E");
            break;
          
          case KEYCODE_J:
            RCLCPP_DEBUG(this->get_logger(), "Q");
            input.shutdown();
            return;
          default:
            RCLCPP_DEBUG(this->get_logger(), "Unknown key pressed: 0x%02X", c);
            continue;
        }
        RCLCPP_INFO(this->get_logger(), "Setting motor positions: m0: %d, m1: %d, m2: %d, m3: %d", m0, m1, m2, m3);
        serial_->writeData(motor_message(m0,m1,m2,m3));
        rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
        response_values();
      }
    }

    void topic_callback(const geometry_msgs::msg::PoseStamped msg) const
    {
      RCLCPP_INFO(this->get_logger(), "I heard quaternion: '%f' '%f' '%f' '%f'", msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w);

      std::string motor_positions = std::to_string(msg.pose.orientation.x / M_PI * 180) + ","
        + std::to_string(msg.pose.orientation.y / M_PI * 180) + ","
        + std::to_string(msg.pose.orientation.z / M_PI * 180) + ","
        + std::to_string(msg.pose.orientation.w / M_PI * 180) + "\n";
      RCLCPP_INFO(this->get_logger(), "Writing: '%s'", motor_positions.c_str());
      serial_->writeData(motor_positions);

      int i = 0;
      do {
        // Wait for a short period to allow the device to respond
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        std::string response = serial_->readData();
        if (!response.empty()) {
          RCLCPP_INFO(this->get_logger(), "Received: '%s'", response.c_str());
          break;
        }else {
          RCLCPP_INFO(this->get_logger(), "Failed to read from serial port on attempt %d", i);
        }
        i++;
      } while (i < 10 && serial_->isOpen());

      if (i == 10) {
        RCLCPP_ERROR(this->get_logger(), "Failed to read from serial port after 10 attempts");
      }

      auto message = std_msgs::msg::String();
      message.data = "Hello, world!";
      RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
      publisher_->publish(message);
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr subscription_;
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