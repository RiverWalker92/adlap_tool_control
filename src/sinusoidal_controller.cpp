#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

using namespace std::chrono_literals;

class ToolController : public rclcpp::Node {
public:
  ToolController() : Node("tool_controller") {
    // Global params
    this->declare_parameter<std::string>("topic", "/tool_cmd");
    this->declare_parameter<double>("publish_rate", 100.0);

    declare_dof("roll");
    declare_dof("pitch");
    declare_dof("yaw");
    declare_dof("aperture");

    topic_ = this->get_parameter("topic").as_string();
    double rate = this->get_parameter("publish_rate").as_double();

    publisher_ =
        this->create_publisher<std_msgs::msg::Float64MultiArray>(topic_, 10);

    timer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / rate),
                                     std::bind(&ToolController::update, this));

    start_time_ = this->now();
  }

private:
  struct DOF {
    std::string mode; // "sinusoid" or "constant"
    double min;
    double max;
    double frequency;
    double value;
  };

  void declare_dof(const std::string &name) {
    this->declare_parameter<std::string>(name + ".mode", "constant");
    this->declare_parameter<double>(name + ".min", 0.0);
    this->declare_parameter<double>(name + ".max", 0.0);
    this->declare_parameter<double>(name + ".frequency", 1.0);
    this->declare_parameter<double>(name + ".value", 0.0);
  }

  DOF get_dof(const std::string &name) {
    DOF d;
    d.mode = this->get_parameter(name + ".mode").as_string();
    d.min = this->get_parameter(name + ".min").as_double();
    d.max = this->get_parameter(name + ".max").as_double();
    d.frequency = this->get_parameter(name + ".frequency").as_double();
    d.value = this->get_parameter(name + ".value").as_double();
    return d;
  }

  double compute(const DOF &d, double t) {
    if (d.mode == "constant") {
      return d.value;
    } else if (d.mode == "sinusoid") {
      double center = (d.max + d.min) / 2.0;
      double amplitude = (d.max - d.min) / 2.0;
      return center + amplitude * std::sin(2.0 * M_PI * d.frequency * t);
    } else if (d.mode == "triangle") {

      double center = (d.max + d.min) / 2.0;
      double amplitude = (d.max - d.min) / 2.0;

      double phase = t * d.frequency;
      double frac = phase - std::floor(phase + 0.5);

      double tri = 2.0 * std::abs(2.0 * frac) - 1.0;

      return center + amplitude * tri;
    } else {
      RCLCPP_WARN(this->get_logger(), "Unknown mode '%s', returning 0",
                  d.mode.c_str());
      return 0.0;
    }
  }

  void update() {
    double t = (this->now() - start_time_).seconds();

    double roll = compute(get_dof("roll"), t);
    double pitch = compute(get_dof("pitch"), t);
    double yaw = compute(get_dof("yaw"), t);
    double aperture = compute(get_dof("aperture"), t);

    std_msgs::msg::Float64MultiArray msg;
    msg.data = {roll, pitch, yaw, aperture};

    publisher_->publish(msg);
  }

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time start_time_;
  std::string topic_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ToolController>());
  rclcpp::shutdown();
  return 0;
}