#!/usr/bin/env python3

import math
import json
import yaml
from pathlib import Path
import sys
from ament_index_python.packages import get_package_share_directory

# Make imports work after moving files into scripts/nodes and scripts/digital_twins.
SCRIPT_DIR = Path(__file__).resolve().parent

IMPORT_DIRS = [
    SCRIPT_DIR,                         # install folder
    SCRIPT_DIR / "digital_twins",       # install folder if subfolder is preserved
    SCRIPT_DIR.parent / "digital_twins" # source folder: scripts/nodes -> scripts/digital_twins
]

DEFAULT_TRAINING_CONFIG_PATH = (
    Path(get_package_share_directory("adlap_tool_control"))
    / "config"
    / "training_config.yaml"
)

for import_dir in IMPORT_DIRS:
    if import_dir.exists() and str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import joblib
import pandas as pd
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from instrument_digital_twin import InstrumentDigitalTwin

# Helper class to load a trained hybrid bend residual model and predict the residual from gearbox output.
class HybridBendResidualModel:
    """
    Loads a trained gearbox-only no-history residual model.
    This model predicts:
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

    # Predict the residual in degrees from the gearbox output.
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


# Determine the instrument parameters file from the detected instrument configuration.
def resolve_instrument_params_file(instrument_config: str) -> Path:
    """
    Converts the detected instrument configuration to its parameter file.
    """

    instruments_dir = (
        Path(get_package_share_directory("adlap_tool_control"))
        / "config"
        / "instruments"
    )

    mapping = {
        "gripper": instruments_dir / "gripper_params.yaml",
        "scissors": instruments_dir / "scissors_params.yaml",
    }

    if instrument_config not in mapping:
        raise RuntimeError(
            f"Unknown or missing instrument_config '{instrument_config}'. "
            f"Available options are: {list(mapping.keys())}"
        )

    selected_path = mapping[instrument_config]

    if not selected_path.exists():
        raise RuntimeError(
            f"Instrument config '{instrument_config}' maps to missing file: "
            f"{selected_path}"
        )

    return selected_path

def resolve_selected_hybrid_bend_model(training_config_file: Path) -> Path:
    """
    Resolve the Hybrid Bend model selected by the latest configured training.
    """
    training_config_file = Path(training_config_file).expanduser()

    with open(training_config_file, "r") as f:
        config = yaml.safe_load(f)

    test_data_root = Path(
        config["test_data_root"]
    ).expanduser()

    training_data_root = Path(
        config["training_data_root"]
    ).expanduser()

    if not training_data_root.is_absolute():
        training_data_root = (
            test_data_root / training_data_root
        )

    hybrid_output_dir = Path(
        config["hybrid_bend_dt"]["output_dir"]
    ).expanduser()

    if not hybrid_output_dir.is_absolute():
        hybrid_output_dir = (
            training_data_root / hybrid_output_dir
        )

    metrics_file = (
        hybrid_output_dir
        / "hybrid_bend_metrics.json"
    )

    if not metrics_file.exists():
        raise RuntimeError(
            "Hybrid Bend training metrics not found: "
            f"{metrics_file}. Run Hybrid Bend training first."
        )

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    selected_model = metrics.get("selected_model")
    selected_model_file = metrics.get("selected_model_file")

    if not selected_model or not selected_model_file:
        raise RuntimeError(
            "Hybrid Bend metrics do not contain a valid "
            "selected_model and selected_model_file."
        )

    model_file = (
        hybrid_output_dir
        / selected_model_file
    )

    if not model_file.exists():
        raise RuntimeError(
            f"Selected Hybrid Bend model not found: {model_file}"
        )

    return model_file


class InstrumentStateNode(Node):
    def __init__(self):
        super().__init__("instrument_state_node")

        # -------------------------------------------------------------------------
        # Declare parameters
        # -------------------------------------------------------------------------

        self.declare_parameter(
            "gearbox_output_topic",
            "/right/tool_control_node/gearbox_state",
        )

        self.declare_parameter(
            "predicted_instrument_angles_topic",
            "/right/instrument_digital_twin/predicted_instrument_angles",
        )

        self.declare_parameter(
            "hybrid_predicted_instrument_angles_topic",
            "/right/instrument_digital_twin/hybrid_predicted_instrument_angles",
        )

        self.declare_parameter(
            "test_data_root",
            "~/ros2_ws/test_data",
        )

        self.declare_parameter(
            "hybrid_bend_enabled",
            False,
        )

        self.declare_parameter(
            "hybrid_bend_model_file",
            "",
        )

        self.declare_parameter(
            "training_config_file",
            str(DEFAULT_TRAINING_CONFIG_PATH),
        )

        self.declare_parameter(
            "instrument_config",
            "",
        )

        # -------------------------------------------------------------------------
        # Read parameter values
        # -------------------------------------------------------------------------

        self.gearbox_output_topic = self.get_parameter(
            "gearbox_output_topic"
        ).value

        self.predicted_topic = self.get_parameter(
            "predicted_instrument_angles_topic"
        ).value

        self.hybrid_predicted_topic = self.get_parameter(
            "hybrid_predicted_instrument_angles_topic"
        ).value

        self.test_data_root = Path(
            self.get_parameter("test_data_root").value
        ).expanduser()

        self.hybrid_bend_enabled = bool(
            self.get_parameter("hybrid_bend_enabled").value
        )

        hybrid_bend_model_file_raw = self.get_parameter(
            "hybrid_bend_model_file"
        ).value

        training_config_file = Path(
            self.get_parameter("training_config_file").value
        ).expanduser()

        instrument_config = self.get_parameter(
            "instrument_config"
        ).value

        # -------------------------------------------------------------------------
        # Resolve paths
        # -------------------------------------------------------------------------
        if self.hybrid_bend_enabled:
            if hybrid_bend_model_file_raw:
                # Optional manual model override.
                hybrid_bend_model_file = Path(
                    hybrid_bend_model_file_raw
                ).expanduser()

                if hybrid_bend_model_file.is_absolute():
                    self.hybrid_bend_model_file = hybrid_bend_model_file
                else:
                    self.hybrid_bend_model_file = (
                        self.test_data_root
                        / hybrid_bend_model_file
                    )

            else:
                # Automatically load the model selected during Hybrid Bend training.
                self.hybrid_bend_model_file = (
                    resolve_selected_hybrid_bend_model(
                        training_config_file
                    )
                )
        else:
            self.hybrid_bend_model_file = None

        self.instrument_params_file = resolve_instrument_params_file(
            instrument_config=instrument_config
        )
                
        # -------------------------------------------------------------------------
        # Initialize Digital Twin models
        # -------------------------------------------------------------------------

        self.instrument_dt = InstrumentDigitalTwin(
            config_path=self.instrument_params_file
        )

        self.hybrid_bend_model = None

        if self.hybrid_bend_enabled:
            if not self.hybrid_bend_model_file.exists():
                raise RuntimeError(
                    f"Hybrid bend model not found: {self.hybrid_bend_model_file}"
                )

            self.hybrid_bend_model = HybridBendResidualModel(
                self.hybrid_bend_model_file
            )

            self.get_logger().info(
                f"Loaded hybrid bend residual model from "
                f"{self.hybrid_bend_model_file}"
            )
        
        # -------------------------------------------------------------------------
        # Publishers and subscribers
        # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Callback for gearbox output messages
    # -------------------------------------------------------------------------
    def gearbox_output_callback(self, msg):
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
        if self.hybrid_bend_enabled:
            residual_deg = self.hybrid_bend_model.predict_residual_deg(msg.data)
            residual_rad = math.radians(residual_deg)

            hybrid_out = Float64MultiArray()

            hybrid_pitch = float(predicted["pitch"]) + residual_rad

            # Apply the learned DOF2 residual to both bend and pitch.
            # The residual model was trained against the pitch-related output.
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

            # self.get_logger().info(
            #     f"Published hybrid predicted angles: {list(hybrid_out.data)}, "
            #     f"residual={residual_deg:.3f} deg"
            # )

def main(args=None):
    rclpy.init(args=args)
    node = InstrumentStateNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()