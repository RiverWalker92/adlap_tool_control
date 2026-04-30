#include <chrono>
#include <cmath>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

using namespace std::chrono_literals;

class CircularTrajectoryNode : public rclcpp::Node {
public:
  CircularTrajectoryNode() : Node("circular_trajectory") {
    // Parameters
    this->declare_parameter<std::string>("topic", "/tool_cmd");
    this->declare_parameter<double>("publish_rate", 100.0);
    this->declare_parameter<double>("radius", 0.5);
    this->declare_parameter<double>("frequency", 0.5); // Hz

    topic_ = this->get_parameter("topic").as_string();
    double rate = this->get_parameter("publish_rate").as_double();

    publisher_ =
        this->create_publisher<std_msgs::msg::Float64MultiArray>(topic_, 10);

    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / rate),
        std::bind(&CircularTrajectoryNode::update, this));

    start_time_ = this->now();
  }

private:
  void update() {
    double t = (this->now() - start_time_).seconds();

    double R = this->get_parameter("radius").as_double();
    double f = this->get_parameter("frequency").as_double();

    double omega = 2.0 * M_PI * f;

    double pitch = R * std::cos(omega * t); // x
    double yaw = R * std::sin(omega * t);   // y

    double roll = 0.0;
    double aperture = 0.0;

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
  rclcpp::spin(std::make_shared<CircularTrajectoryNode>());
  rclcpp::shutdown();
  return 0;
}