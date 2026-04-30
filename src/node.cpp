#include "adlap_tool_control/instrument_controller.hpp"
#include "adlap_tool_control/keyboard_reader.hpp"
#include "adlap_tool_control/motor_controller.hpp"
#include "adlap_tool_control/serial_port.hpp"

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

const std::string STATUS_TOPIC = "/tool_status";
const std::string CURRENT_ANGLES_TOPIC = "/current_instrument_angles";
const std::string TOOL_CONTROL_TOPIC = "/instrument_angles";

class ToolController : public rclcpp::Node {
public:
  ToolController(std::shared_ptr<SerialPort> serial)
      : Node("tool_control_node"), serial_(serial),
        // motor_controller_(serial_, this->get_logger(), Motor{7, 158.9f, 2,
        // 30, 50, 90, 25.0f / 15.0f, 20, 4, false}), // AE motor config
        motor_controller_(serial_, this->get_logger()), // Default motor config
        instrument_controller_(motor_controller_, this->get_logger()) {
    // The publisher and subscriber topics are relative, so they are mapped to
    // left or right with the node namespace
    publisher_ =
        this->create_publisher<std_msgs::msg::String>("~" + STATUS_TOPIC, 10);
    publisher_angles_ =
        this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "~" + CURRENT_ANGLES_TOPIC, 10);
    subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "~" + TOOL_CONTROL_TOPIC, 10,
        std::bind(&ToolController::topic_callback, this,
                  std::placeholders::_1));

    // Start manual control for testing
    instrument_controller_.manual_adjustment();
  }

  void topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (msg->data.size() < 4) {
      RCLCPP_WARN(this->get_logger(), "Received data size less than 4");
      return;
    }
    std::vector<double> angles = msg->data;
    // data values are: roll, pitch, yaw, gripper angle
    RCLCPP_INFO(this->get_logger(), "I heard array: '%f' '%f' '%f' '%f'",
                angles[0], angles[1], angles[2], angles[3]);
    instrument_controller_.set_euler_angles(angles[0], angles[1], angles[2],
                                      angles[3]);
    // Publish the response values for now, later publish the actual status of
    // the tool
    auto current_angles = instrument_controller_.euler_angles_from_motors(
        motor_controller_.get_target_positions());

    auto status_msg = std_msgs::msg::String();
    // Create a status message with the current angles
    // TODO: change this to actual status information, and publish the angles in
    // a separate topic if needed
    status_msg.data = "Current angles: ";
    for (size_t i = 0; i < current_angles.size(); ++i) {
      status_msg.data += std::to_string(current_angles[i]) +
                         (i < current_angles.size() - 1 ? ", " : "");
    }
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'",
                status_msg.data.c_str());
    publisher_->publish(status_msg);

    auto current_angles_msg = std_msgs::msg::Float64MultiArray();
    for (size_t i = 0; i < current_angles.size(); ++i) {
      current_angles_msg.data.push_back(current_angles[i]);
    }
    publisher_angles_->publish(current_angles_msg);
  }

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
      publisher_angles_;

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr
      subscription_;
  std::shared_ptr<SerialPort> serial_;
  MotorController motor_controller_;
  InstrumentController instrument_controller_;
};

int main(int argc, char *argv[]) {
  int ret = 0;
  rclcpp::init(argc, argv);

  try {
    std::string device_path =
        SerialPort::find_device_by_manufacturer_product("Raspberry Pi", "Pico");

    if (device_path.empty()) {
      throw std::runtime_error("Could not find Raspberry Pi Pico device. "
                               "Please check if the device is connected.");
    }
    RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "Using device: %s",
                 device_path.c_str());
    auto serial = std::make_shared<SerialPort>(device_path);
    if (!serial->open_port()) {
      throw std::runtime_error("Failed to open serial port: " + device_path);
    }

    rclcpp::spin(std::make_shared<ToolController>(serial));
    serial->close_port();

  } catch (const std::exception &e) {
    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Exception: %s", e.what());
    ret = 1;
  }

  rclcpp::shutdown();
  return ret;
}