#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

using namespace std::chrono_literals;

class ToolControllerNode : public rclcpp::Node
{
public:
  ToolControllerNode() : Node("tool_controller_node")
  {
    // Global params
    this->declare_parameter<std::string>("topic", "/tool_cmd");
    this->declare_parameter<double>("publish_rate", 100.0);

    declare_dof("dof1");
    declare_dof("dof2");
    declare_dof("dof3");
    declare_dof("dof4");

    topic_ = this->get_parameter("topic").as_string();
    double rate = this->get_parameter("publish_rate").as_double();

    publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(topic_, 10);

    timer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / rate),
                                     std::bind(&ToolControllerNode::update, this));

    start_time_ = this->now();
  }

private:
  struct DOF
  {
    std::string mode;  // "sinusoid" or "constant"
    double min;
    double max;
    double frequency;
    double value;
  };

  void declare_dof(const std::string& name)
  {
    this->declare_parameter<std::string>(name + ".mode", "constant");
    this->declare_parameter<double>(name + ".min", 0.0);
    this->declare_parameter<double>(name + ".max", 0.0);
    this->declare_parameter<double>(name + ".frequency", 1.0);
    this->declare_parameter<double>(name + ".value", 0.0);
  }

  DOF get_dof(const std::string& name)
  {
    DOF d;
    d.mode = this->get_parameter(name + ".mode").as_string();
    d.min = this->get_parameter(name + ".min").as_double();
    d.max = this->get_parameter(name + ".max").as_double();
    d.frequency = this->get_parameter(name + ".frequency").as_double();
    d.value = this->get_parameter(name + ".value").as_double();
    return d;
  }

  double compute(const DOF& d, double t)
  {
    if (d.mode == "constant")
    {
      return d.value;
    }
    else if (d.mode == "sinusoid")
    {
      double center = (d.max + d.min) / 2.0;
      double amplitude = (d.max - d.min) / 2.0;
      return center + amplitude * std::sin(2.0 * M_PI * d.frequency * t);
    }
    else if (d.mode == "triangle")
    {
      double center = (d.max + d.min) / 2.0;
      double amplitude = (d.max - d.min) / 2.0;

      double phase = t * d.frequency;
      double frac = phase - std::floor(phase + 0.5);

      double tri = 2.0 * std::abs(2.0 * frac) - 1.0;

      return center + amplitude * tri;
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "Unknown mode '%s', returning 0", d.mode.c_str());
      return 0.0;
    }
  }

  void update()
  {
    double t = (this->now() - start_time_).seconds();

    double dof1 = compute(get_dof("dof1"), t);
    double dof2 = compute(get_dof("dof2"), t);
    double dof3 = compute(get_dof("dof3"), t);
    double dof4 = compute(get_dof("dof4"), t);

    std_msgs::msg::Float64MultiArray msg;
    msg.data = { dof1, dof2, dof3, dof4 };

    publisher_->publish(msg);
  }

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time start_time_;
  std::string topic_;
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ToolControllerNode>());
  rclcpp::shutdown();
  return 0;
}