#include "adlap_tool_control/serial_port.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

using namespace std::chrono_literals;

constexpr char TOPIC[] = "tool_status";
constexpr char POSE_TOPIC[] = "/servo_node/pose_target_cmds";

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

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

      auto message = "0,0,0,0\n";
      RCLCPP_INFO(this->get_logger(), "Writing: '%s'", message);
      serial_->writeData(message);

      // Wait for a short period to allow the device to respond
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      std::string response = serial_->readData();
      if (!response.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received: '%s'", response.c_str());
      }else {
        RCLCPP_ERROR(this->get_logger(), "Failed to read from serial port");
      }
    }

  private:
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