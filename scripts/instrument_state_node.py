#!/usr/bin/env python3

import math
from pathlib import Path

import joblib
import pandas as pd
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from instrument_digital_twin import InstrumentDigitalTwin

class HybridBendResidualModel:
    """
    Loads a trained gearbox-only no-history residual model.

    The model predicts:
        residual_deg = video_angle_deg - physics_DT_pitch_deg

    Forward prediction:
        hybrid_pitch = physics_DT_pitch + residual
    """

    def __init__(self, model_path):
        model_path = Path(model_path).expanduser()

        saved = joblib.load(model_path)

        self.model = saved["model"]
        self.feature_columns = list(saved["feature_columns"])

        allowed_columns = [f"gearbox_{i}" for i in range(6)]

        unsupported_columns = [
            col for col in self.feature_columns
            if col not in allowed_columns
        ]

        if unsupported_columns:
            raise RuntimeError(
                "This online hybrid model currently only supports "
                "gearbox-only no-history models. Unsupported columns: "
                f"{unsupported_columns}"
            )

    def predict_residual_deg(self, gearbox_output):
        feature_values = {
            f"gearbox_{i}": float(gearbox_output[i])
            for i in range(6)
        }

        row = {
            col: feature_values[col]
            for col in self.feature_columns
        }

        x = pd.DataFrame([row], columns=self.feature_columns)
        residual_deg = float(self.model.predict(x)[0])

        return residual_deg

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
        self.declare_parameter(
            "hybrid_predicted_instrument_angles_topic",
            "/right/instrument_digital_twin/hybrid_predicted_instrument_angles",
        )
#!!!!!!!!!!!!PAS DIT AAN
        self.declare_parameter("hybrid_bend_enabled", True) #!!!! hierin aanpassen

        self.declare_parameter(
            "hybrid_bend_model_file",
            "",
        )

        self.gearbox_output_topic = self.get_parameter("gearbox_output_topic").value
        self.predicted_topic = self.get_parameter(
            "predicted_instrument_angles_topic"
        ).value

        self.hybrid_predicted_topic = self.get_parameter(
            "hybrid_predicted_instrument_angles_topic"
        ).value

        self.hybrid_bend_enabled = bool(
            self.get_parameter("hybrid_bend_enabled").value
        )

        self.hybrid_bend_model_file = self.get_parameter(
            "hybrid_bend_model_file"
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

        self.instrument_dt = InstrumentDigitalTwin(config_path=self.instrument_params_file)

        self.hybrid_bend_model = None

        if self.hybrid_bend_enabled:
            if self.hybrid_bend_model_file == "":
                raise RuntimeError(
                    "hybrid_bend_enabled is True, but hybrid_bend_model_file is empty"
                )

            self.hybrid_bend_model = HybridBendResidualModel(
                self.hybrid_bend_model_file
            )

            self.get_logger().info(
                f"Loaded hybrid bend residual model from {self.hybrid_bend_model_file}"
            )

        self.predicted_pub = self.create_publisher(
            Float64MultiArray,
            self.predicted_topic,
            10,
        )

        self.hybrid_predicted_pub = self.create_publisher(
            Float64MultiArray,
            self.hybrid_predicted_topic,
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
            float(predicted["shaft_roll"]),          # 0 = DOF1
            float(predicted["bend"]),                # 1 = DOF2
            float(predicted["tip_rotation"]),        # 2 = DOF3
            float(predicted["articulation"]),        # 3 = DOF4
            float(predicted["pitch"]),               # 4 extra
            float(predicted["yaw"]),                 # 5 extra
            float(predicted["raw_bend"]),            # 6 extra
            float(predicted["articulation_one_jaw"]) # 7 extra
        ]

        self.predicted_pub.publish(out)
        self.get_logger().info(f"Published predicted angles: {list(out.data)}")

        if self.hybrid_bend_enabled:
            residual_deg = self.hybrid_bend_model.predict_residual_deg(msg.data)
            residual_rad = math.radians(residual_deg)

            hybrid_out = Float64MultiArray()

            hybrid_pitch = float(predicted["pitch"]) + residual_rad

            # Voor jouw DOF2 bend model corrigeren we ook bend met dezelfde residual.
            # De training gebruikte pred_instr_1, dus pitch is de belangrijkste output.
            hybrid_bend = float(predicted["bend"]) + residual_rad

            hybrid_out.data = [
                float(predicted["shaft_roll"]),          # 0 = DOF1
                float(hybrid_bend),                      # 1 = DOF2 corrected
                float(predicted["tip_rotation"]),        # 2 = DOF3
                float(predicted["articulation"]),        # 3 = DOF4
                float(hybrid_pitch),                     # 4 extra corrected pitch
                float(predicted["yaw"]),                 # 5 extra
                float(predicted["raw_bend"]),            # 6 extra
                float(predicted["articulation_one_jaw"]) # 7 extra
            ]
            self.hybrid_predicted_pub.publish(hybrid_out)

            self.get_logger().info(
                f"Published hybrid predicted angles: {list(hybrid_out.data)}, "
                f"residual={residual_deg:.3f} deg"
            )

def main(args=None):
    rclpy.init(args=args)
    node = InstrumentStateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()