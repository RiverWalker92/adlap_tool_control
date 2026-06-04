#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import yaml
from pathlib import Path
from instrument_digital_twin import InstrumentDigitalTwin, InstrumentDigitalTwinParams

class InstrumentStateNode(Node):
    def __init__(self):
        super().__init__("instrument_state_node")

        self.declare_parameter(
            "gearbox_output_topic",
            "/right/gearbox_digital_twin/gearbox_state",
        )
        self.declare_parameter(
            "predicted_instrument_angles_topic",
            "/right/instrument_digital_twin/predicted_instrument_angles",
        )

        self.gearbox_output_topic = self.get_parameter("gearbox_output_topic").value
        self.predicted_topic = self.get_parameter(
            "predicted_instrument_angles_topic"
        ).value

        self.declare_parameter(
            "instrument_params_file",
            str(
                Path.home()
                / "ros2_ws"
                / "src"
                / "adlap_tool_control"
                / "config"
                / "instruments"
                / "gripper_params.yaml"
            ),
        )

        self.instrument_params_file = Path(
            self.get_parameter("instrument_params_file").value
        )

        with open(self.instrument_params_file, "r") as f:
            config = yaml.safe_load(f)
        instrument_config = config["instrument"]

        self.instrument_dt = InstrumentDigitalTwin(
            InstrumentDigitalTwinParams(
                bend_factor=instrument_config["bend_factor"],
                articulation_factor=instrument_config["articulation_factor"],
                shaft_roll_sign=instrument_config["shaft_roll_sign"],
                bend_sign=instrument_config["bend_sign"],                
                tip_rotation_sign=instrument_config["tip_rotation_sign"],
                articulation_sign=instrument_config["articulation_sign"],
                jaw_angle_deg_per_mm=instrument_config["jaw_angle_deg_per_mm"],
            )
        )


        self.predicted_pub = self.create_publisher(
            Float64MultiArray,
            self.predicted_topic,
            10,
        )

        self.gearbox_sub = self.create_subscription(
            Float64MultiArray,
            self.gearbox_output_topic,
            self.gearbox_output_callback,
            10,
        )

        self.get_logger().info(
            f"Instrument state node started. Listening to {self.gearbox_output_topic}"
        )
        self.get_logger().info(
            f"Loaded instrument parameters from {self.instrument_params_file}"
        )

    def gearbox_output_callback(self, msg):
        self.get_logger().info(f"Received gearbox output: {list(msg.data)}")

        if len(msg.data) != 6:
            self.get_logger().warn(
                f"Expected 6 gearbox output values, got {len(msg.data)}"
            )
            return

        predicted = self.instrument_dt.euler_angles_from_gearbox_output(msg.data)

        out = Float64MultiArray()
        out.data = [
            float(predicted["tip_rotation"]),
            float(predicted["pitch"]),
            float(predicted["yaw"]),
            float(predicted["articulation"]),
        ]

        self.predicted_pub.publish(out)
        self.get_logger().info(f"Published predicted angles: {list(out.data)}")

def main(args=None):
    rclpy.init(args=args)
    node = InstrumentStateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()