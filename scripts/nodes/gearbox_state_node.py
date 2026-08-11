#!/usr/bin/env python3

from pathlib import Path
import json
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

# Make imports work after moving files into scripts/nodes and scripts/digital_twins.
SCRIPT_DIR = Path(__file__).resolve().parent

IMPORT_DIRS = [
    SCRIPT_DIR,                         # install: beide bestanden naast elkaar
    SCRIPT_DIR.parent / "digital_twins" # source: nodes -> digital_twins
]

for import_dir in IMPORT_DIRS:
    if import_dir.exists() and str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from gearbox_digital_twin import GearboxDigitalTwin


class GearboxStateNode(Node):
    def __init__(self):
        # Make ROS node and publishers/subscribers
        super().__init__("gearbox_state_node")

        # Same YAML config
        config_path = (
            Path.home()
            / "ros2_ws"
            / "src"
            / "adlap_tool_control"
            / "config"
            / "gearbox_params.yaml"
        )

        self.declare_parameter("gearbox_variant", "")
        gearbox_variant = (
            self.get_parameter("gearbox_variant")
            .get_parameter_value()
            .string_value
        )

        if gearbox_variant == "":
            gearbox_variant = None
        self.get_logger().info(f"Gearbox variant from launch parameter: {gearbox_variant}")

        self.dt = GearboxDigitalTwin(
            config_path=config_path,
            active_variant=gearbox_variant,
        )
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

        self.gearbox_debug_pub = self.create_publisher(
            String,
            "/right/tool_control_node/gearbox_debug_state",
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
            state["inner_shaft_relative_rotation_deg"],
            state["inner_shaft_translation_mm"],
            state["middle_shaft_rotation_deg"],
            state["outer_shaft_rotation_deg"],
            state["middle_outer_relative_rotation_deg"],
        ]

        self.gearbox_state_pub.publish(out)

        self.get_logger().info(f"Delta motor pulses: {delta_motor_pulses}")
        self.get_logger().info(f"Gearbox state: {out.data}")

        debug = String()
        debug.data = json.dumps({
            "motor_delta_pulses": list(delta_motor_pulses),

            "gear_l1_deg": state["gear_l1_deg"],
            "gear_l2_deg": state["gear_l2_deg"],
            "gear_r1_deg": state["gear_r1_deg"],
            "gear_r2_deg": state["gear_r2_deg"],

            "inner_shaft_rotation_deg": state["inner_shaft_rotation_deg"],
            "inner_shaft_relative_rotation_deg": state["inner_shaft_relative_rotation_deg"],
            "inner_shaft_translation_mm_raw": state["inner_shaft_translation_mm_raw"],
            "inner_shaft_translation_mm": state["inner_shaft_translation_mm"],

            "middle_shaft_rotation_deg": state["middle_shaft_rotation_deg"],
            "outer_shaft_rotation_deg": state["outer_shaft_rotation_deg"],
            "middle_outer_relative_rotation_deg": state["middle_outer_relative_rotation_deg"],
        })
        self.gearbox_debug_pub.publish(debug)


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