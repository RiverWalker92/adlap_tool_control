#include "adlap_tool_control/serial_port.hpp"
#include "adlap_tool_control/keyboard_reader.hpp"
#include "adlap_tool_control/motor_controller.hpp"
#include "adlap_tool_control/instrument_controller.hpp"

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include <thread>

using namespace std::chrono_literals;

const std::string STATUS_TOPIC = "/tool_status";
const std::string TOOL_CONTROL_TOPIC = "/instrument_angles";
const std::string MOTOR_CURRENTS_TOPIC = "/motor_currents";
const std::string MOTOR_POSITIONS_TOPIC = "/motor_positions"; //feedback from motors 
const std::string COMMANDED_MOTOR_POSITIONS_TOPIC = "/commanded_motor_positions";
const std::string COMMANDED_ANGLES_TOPIC = "/commanded_instrument_angles";
const std::string TASK_TOPIC = "/task_label";


class ToolController : public rclcpp::Node
{
  public:
  ToolController(std::shared_ptr<SerialPort> serial)
    : Node("tool_control_node"), serial_(serial),
      //motor_controller_(serial_, this->get_logger(), Motor{7, 158.9f, 2, 30, 50, 90, 25.0f / 15.0f, 20, 4, false}), // AE motor config
      motor_config_publisher_(this->create_publisher<std_msgs::msg::String>(
          "~/motor_config",
          rclcpp::QoS(10).transient_local()
      )),
      motor_controller_(serial_, this->get_logger(), motor_config_publisher_), // Default motor config
      task_publisher_(this->create_publisher<std_msgs::msg::String>("~" + TASK_TOPIC, 10)),
      instrument_controller_(motor_controller_, this->get_logger(), task_publisher_)
    {
      // The publisher and subscriber topics are relative, so they are mapped to left or right with the node namespace
      publisher_ = this->create_publisher<std_msgs::msg::String>("~" + STATUS_TOPIC, 10);
      subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "~" + TOOL_CONTROL_TOPIC, 10, std::bind(&ToolController::topic_callback, this, std::placeholders::_1));  

      // Subscriber for commanded motor positions   
      motor_command_subscription_ = this->create_subscription<std_msgs::msg::Int32MultiArray>(
        "~" + COMMANDED_MOTOR_POSITIONS_TOPIC,
        10,
        std::bind(&ToolController::motor_command_callback, this, std::placeholders::_1));
        
        // Publisher for commanded angles
      commanded_angles_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "~" + COMMANDED_ANGLES_TOPIC, 10);

      // Publisher for motor currents
      motor_currents_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "~" + MOTOR_CURRENTS_TOPIC, 10);
      motor_currents_timer_ = this->create_wall_timer(100ms,
      std::bind(&ToolController::publish_motor_currents, this));

      // Publisher for motor positions
      motor_positions_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "~" + MOTOR_POSITIONS_TOPIC, 10);


      motor_positions_timer_ = this->create_wall_timer(
        100ms,
        std::bind(&ToolController::publish_motor_positions, this));

      // Start manual control for testing in separate thread
      manual_thread_ = std::thread([this]() {
        instrument_controller_.manual_adjustment();
      });
    }


    void topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
      if (msg->data.size() < 4) {
        RCLCPP_WARN(this->get_logger(), "Received data size less than 4");
        return;
      }
      std::vector<double> angles = msg->data;
      // data values are: roll, pitch, yaw, gripper angle
      RCLCPP_INFO(this->get_logger(), "I heard array: '%f' '%f' '%f' '%f'", angles[0], angles[1], angles[2], angles[3]);
      instrument_controller_.set_angles(angles[0], angles[1], angles[2], angles[3]);
      
      // auto task_msg = std_msgs::msg::String();
      // task_msg.data = "ros_command";
      // task_publisher_->publish(task_msg);

      auto command_msg = std_msgs::msg::Float64MultiArray();
      command_msg.data = {
        angles[0],
        angles[1],
        angles[2],
        angles[3]
      };

      commanded_angles_publisher_->publish(command_msg);
      // Publish the response values for now, later publish the actual status of the tool

      auto status_msg = std_msgs::msg::String();
      auto current_angles = instrument_controller_.angles_from_motors(motor_controller_.get_target_positions());

      // Create a status message with the current angles
      // TODO: change this to actual status information, and publish the angles in a separate topic if needed
      status_msg.data = "Current angles: ";
      for (size_t i = 0; i < current_angles.size(); ++i) {
        status_msg.data += std::to_string(current_angles[i]) + (i < current_angles.size() - 1 ? ", " : "");
      }
      RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", status_msg.data.c_str());
      publisher_->publish(status_msg);

    }

    void motor_command_callback(const std_msgs::msg::Int32MultiArray::SharedPtr msg)
    {
      if (msg->data.size() < 4) {
        RCLCPP_WARN(this->get_logger(), "Received motor command with less than 4 values");
        return;
      }

      motor_controller_.send_relative_motor_positions(
        msg->data[0],
        msg->data[1],
        msg->data[2],
        msg->data[3],
        true
      );

    }


    void publish_motor_currents()
    {
      auto currents = motor_controller_.get_currents();

      auto current_msg = std_msgs::msg::Float64MultiArray();
      current_msg.data = {
        static_cast<double>(currents[0]),
        static_cast<double>(currents[1]),
        static_cast<double>(currents[2]),
        static_cast<double>(currents[3])
      };

      motor_currents_publisher_->publish(current_msg);
    }

    void publish_motor_positions()
    {
      auto positions = motor_controller_.get_positions();

      auto position_msg = std_msgs::msg::Float64MultiArray();
      position_msg.data = {
        static_cast<double>(positions[0]),
        static_cast<double>(positions[1]),
        static_cast<double>(positions[2]),
        static_cast<double>(positions[3])
      };

      motor_positions_publisher_->publish(position_msg);
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;

    rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr motor_command_subscription_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr commanded_angles_publisher_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr motor_currents_publisher_;
    rclcpp::TimerBase::SharedPtr motor_currents_timer_;
 
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr motor_positions_publisher_;
    rclcpp::TimerBase::SharedPtr motor_positions_timer_;

  
    
    std::shared_ptr<SerialPort> serial_;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr motor_config_publisher_;

    MotorController motor_controller_;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr task_publisher_;

    InstrumentController instrument_controller_;

    std::thread manual_thread_;
};

int main(int argc, char * argv[])
{
  int ret = 0;
  rclcpp::init(argc, argv);

  try {
    std::string device_path = SerialPort::find_device_by_manufacturer_product("Raspberry Pi", "Pico");
    
    if (device_path.empty()) {
      throw std::runtime_error("Could not find Raspberry Pi Pico device. Please check if the device is connected.");
    }
    RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "Using device: %s", device_path.c_str());
    auto serial = std::make_shared<SerialPort>(device_path);
    if (!serial->open_port()) {
      throw std::runtime_error("Failed to open serial port: " + device_path);
    }
    
    rclcpp::spin(std::make_shared<ToolController>(serial));
    serial->close_port();
    
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Exception: %s", e.what());
    ret = 1;
  }  
  
  rclcpp::shutdown();
  return ret;
}