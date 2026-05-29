#!/usr/bin/env python3

from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from gearbox_digital_twin import GearboxDigitalTwin


class GearboxStateNode(Node):
    def __init__(self):
        # Make ROS node and publishers/subscribers
        super().__init__("gearbox_state_node")

        # Same YAML config
        config_path = (
            Path.home()
            / "ros2_ws_roel_split"
            / "src"
            / "adlap_tool_control"
            / "config"
            / "gearbox_params.yaml"
        )

        self.dt = GearboxDigitalTwin(config_path=config_path)
        self.start_positions = None # Start positions will be set at the first callback, so that we can calculate deltas from there

        # Subscribe to motor positions and publish gearbox state
        self.motor_position_sub = self.create_subscription(
            Float64MultiArray,
            "/right/tool_control_node/motor_positions",
            self.motor_position_callback,
            10,
        )
        self.gearbox_state_pub = self.create_publisher(
            Float64MultiArray,
            "/right/tool_control_node/gearbox_state",
            10,
        )

        self.get_logger().info("Gearbox state node started")

    # Runs every time we get new motor positions, and publishes the predicted gearbox state
    def motor_position_callback(self, msg):
        current_positions = [int(x) for x in msg.data]

        if len(current_positions) != 4:
            self.get_logger().warn("Expected 4 motor positions")
            return

        if self.start_positions is None:
            self.start_positions = current_positions
            self.get_logger().info(f"Set gearbox DT start positions: {self.start_positions}")
            return

        delta_motor_pulses = [
            current_positions[i] - self.start_positions[i]
            for i in range(4)
        ]

        state = self.dt.predict_instrument_shaft_inputs(delta_motor_pulses)

        out = Float64MultiArray()
        out.data = [
            state["inner_shaft_rotation_deg"],
            state["inner_shaft_translation_index_deg"],
            state["inner_shaft_translation_mm"],
            state["middle_shaft_rotation_deg"],
            state["outer_shaft_rotation_deg"],
            state["middle_outer_relative_rotation_deg"],
        ]

        self.gearbox_state_pub.publish(out)

        self.get_logger().info(f"Delta motor pulses: {delta_motor_pulses}")
        self.get_logger().info(f"Gearbox state: {out.data}")


def main(args=None):
    rclpy.init(args=args)
    node = GearboxStateNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()