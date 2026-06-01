#!/usr/bin/env python3

import math
import time
from datetime import datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from pathlib import Path


class PatternRunner(Node):
    def __init__(self):
        super().__init__("tool_controller_node")

        default_config_path = (
            Path.home()
            / "ros2_ws"
            / "src"
            / "adlap_tool_control"
            / "config"
            / "tool_params.yaml"
        )
        self.declare_parameter("topic", "/right/tool_control_node/instrument_angles")
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("duration", 30.0)
        self.duration = float(self.get_parameter("duration").value)

        for i in range(1, 5):
            prefix = f"dof{i}"
            self.declare_parameter(f"{prefix}.mode", "constant")
            self.declare_parameter(f"{prefix}.value", 0.0)
            self.declare_parameter(f"{prefix}.min", 0.0)
            self.declare_parameter(f"{prefix}.max", 0.0)
            self.declare_parameter(f"{prefix}.frequency", 0.1)

        self.topic = self.get_parameter("topic").value
        self.publish_rate = self.get_parameter("publish_rate").value

        self.pub = self.create_publisher(Float64MultiArray, self.topic, 10)
        self.task_pub = self.create_publisher(
            String,
            "/right/tool_control_node/task_label",
            10,
        )

        self.start_time = time.time()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = self.generate_test_name()

        task = String()
        task.data = (
            f"setup_03|"
            f"gearbox_instrument|"
            f"{test_name}|"
            f"trial_01|"
            f"{timestamp}"
        )
        self.task_msg = task
        self.task_publish_count = 0

        self.task_timer = self.create_timer(
            0.2,
            self.publish_task_label_repeatedly,
        )
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.timer_callback,
        )

        self.get_logger().info(f"Pattern runner publishing to {self.topic}")
        self.get_logger().info(f"Publish rate: {self.publish_rate} Hz")
        self.get_logger().info(f"task label: {task.data}")
    
    def publish_task_label_repeatedly(self):
        if self.task_publish_count >= 10:
            self.destroy_timer(self.task_timer)
            return

        self.task_pub.publish(self.task_msg)
        self.task_publish_count += 1
        self.get_logger().info(f"Published task label: {self.task_msg.data}")
        
    def generate_test_name(self):
        active = []

        for i in range(1, 5):
            mode = self.get_parameter(f"dof{i}.mode").value

            if mode != "constant":
                active.append(f"dof{i}_{mode}")

        if not active:
            return "all_constant"

        return "continuous_" + "_".join(active)

    def compute_dof(self, dof_name, t):
        mode = self.get_parameter(f"{dof_name}.mode").value
        value = float(self.get_parameter(f"{dof_name}.value").value)
        min_value = float(self.get_parameter(f"{dof_name}.min").value)
        max_value = float(self.get_parameter(f"{dof_name}.max").value)
        frequency = float(self.get_parameter(f"{dof_name}.frequency").value)

        if mode == "constant":
            return value

        if mode == "sinusoid":
            offset = 0.5 * (max_value + min_value)
            amplitude = 0.5 * (max_value - min_value)
            return offset + amplitude * math.sin(2.0 * math.pi * frequency * t)

        if mode == "triangle":
            period = 1.0 / frequency
            phase = (t % period) / period

            if phase < 0.5:
                return min_value + 2.0 * phase * (max_value - min_value)
            else:
                return max_value - 2.0 * (phase - 0.5) * (max_value - min_value)

        self.get_logger().warn(f"Unknown mode '{mode}' for {dof_name}, using constant value")
        return value

    def timer_callback(self):
        t = time.time() - self.start_time
        if t > self.duration:
            self.get_logger().info("Pattern finished")
            self.destroy_timer(self.timer)
            return

        values = [
            self.compute_dof("dof1", t),
            self.compute_dof("dof2", t),
            self.compute_dof("dof3", t),
            self.compute_dof("dof4", t),
        ]

        msg = Float64MultiArray()
        msg.data = values
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PatternRunner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()