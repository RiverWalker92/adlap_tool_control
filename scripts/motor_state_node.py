#!/usr/bin/env python3

from pathlib import Path

import joblib
import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray, Int32MultiArray, String

# Default model directory for the trained Motor Digital Twin models
DEFAULT_MODEL_DIR = (
    Path.home()
    / "ros2_ws"
    / "test_data"
    / "automated_trials"
    / "setup_01_motors"
    / "training motor data"
    / "motor_dt_results"
    / "models"
)


class MotorDigitalTwinNode(Node):
    def __init__(self):
        super().__init__("motor_digital_twin_node")

        self.declare_parameter("model_dir", str(DEFAULT_MODEL_DIR))
        self.declare_parameter("publish_rate_hz", 100.0)
        # self.declare_parameter("publish_rate_hz", 10.0)
        model_dir = Path(self.get_parameter("model_dir").value).expanduser()
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.models = self.load_models(model_dir)
        # Initialize state variables for the Motor Digital Twin
        self.current_command = np.zeros(4, dtype=float)
        self.previous_command = np.zeros(4, dtype=float)
        self.commanded_target = np.zeros(4, dtype=float)
        self.previous_predicted_encoder = np.zeros(4, dtype=float)
        self.measured_positions = np.full(4, np.nan, dtype=float)
        self.measured_currents = np.full(4, np.nan, dtype=float)
        self.last_stamped_command_age_ms = np.nan
        self.last_prediction_compute_time_ms = np.nan

        now = self.get_time_s()
        self.previous_publish_time = now
        self.last_command_change_time = np.full(4, now, dtype=float)
        self.last_target_change_time = np.full(4, now, dtype=float)

        # Set up ROS2 subscriptions and publishers
        # self.command_subscriber = self.create_subscription(
        #     Int32MultiArray,
        #     "/right/tool_control_node/commanded_motor_positions",
        #     self.command_callback,
        #     10,
        # )
        self.command_subscriber = self.create_subscription(
            Float64MultiArray,
            "/right/tool_control_node/commanded_motor_positions_stamped",
            self.stamped_command_callback,
            10,
        )
        self.task_subscriber = self.create_subscription(
            String,
            "/right/tool_control_node/task_label",
            self.task_callback,
            10,
        )


        self.predicted_state_publisher = self.create_publisher(
            Float64MultiArray,
            "/right/motor_digital_twin/predicted_motor_state",
            10,
        )
        
        self.predicted_position_publisher = self.create_publisher(
            Float64MultiArray,
            "/right/motor_digital_twin/predicted_motor_positions",
            10,
        )

        self.predicted_current_publisher = self.create_publisher(
            Float64MultiArray,
            "/right/motor_digital_twin/predicted_motor_currents",
            10,
        )

        self.measured_position_subscriber = self.create_subscription(
            Float64MultiArray,
            "/right/tool_control_node/motor_positions",
            self.measured_position_callback,
            10,
        )

        self.measured_current_subscriber = self.create_subscription(
            Float64MultiArray,
            "/right/tool_control_node/motor_currents",
            self.measured_current_callback,
            10,
        )

        # Set up a timer to periodically publish predictions at the specified rate
        timer_period = 1.0 / publish_rate_hz
        self.timer = self.create_timer(timer_period, self.publish_prediction)
        self.get_logger().info(f"Loaded Motor DT models from: {model_dir}")
        self.get_logger().info("Motor Digital Twin node started.")

    def get_time_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def load_models(self, model_dir):
        models = {}

        for motor_index in range(4):
            model_path = model_dir / f"motor_dt_m{motor_index}.pkl"

            if not model_path.exists():
                raise RuntimeError(f"Missing Motor DT model: {model_path}")

            model_package = joblib.load(model_path)

            current_feature_names = model_package.get("current_feature_names", [])
            encoder_feature_names = model_package.get("encoder_feature_names", [])

            self.get_logger().info(
                f"Motor {motor_index}: "
                f"encoder_features={len(encoder_feature_names)}, "
                f"current_features={len(current_feature_names)}, "
                f"current_model={model_package.get('selected_current_model', 'unknown')}"
            )

            if len(encoder_feature_names) != 8:
                raise RuntimeError(
                    f"Motor {motor_index} encoder model has {len(encoder_feature_names)} features, "
                    "but the online Motor DT expects 8 command/target encoder features."
                )

            if len(current_feature_names) != 8:
                raise RuntimeError(
                    f"Motor {motor_index} current model has {len(current_feature_names)} features, "
                    "but the online Motor DT expects 8 command-only current features."
                )
            models[motor_index] = model_package

        return models

    def task_callback(self, msg):
        if msg.data in ["motor_tests_finished", "tests_finished"]:
            return

        self.reset_segment_state()


    def reset_segment_state(self):
        now = self.get_time_s()

        self.current_command = np.zeros(4, dtype=float)
        self.previous_command = np.zeros(4, dtype=float)
        self.commanded_target = np.zeros(4, dtype=float)
        self.previous_predicted_encoder = np.zeros(4, dtype=float)

        self.last_target_change_time = np.full(4, now, dtype=float)
        self.last_command_change_time = np.full(4, now, dtype=float)
        self.previous_publish_time = now
        self.last_stamped_command_age_ms = np.nan
        self.last_prediction_compute_time_ms = np.nan

        self.get_logger().info("Motor DT reset for new task segment")

    def command_callback(self, msg):
        """
        Fallback voor oude Int32 command topic.
        Deze gebruikt de ontvangsttijd als command time.
        """
        command_receive_time = self.get_time_s()
        command = np.array(msg.data[:4], dtype=float)

        self.update_command_state(
            command=command,
            command_source_time=command_receive_time,
        )


    def stamped_command_callback(self, msg):
        """
        Nieuwe command topic:
        msg.data = [command_source_time, m0, m1, m2, m3]
        """
        if len(msg.data) < 5:
            self.get_logger().warn(
                "Received stamped command with fewer than 5 values."
            )
            return

        command_source_time = float(msg.data[0])
        command = np.array(msg.data[1:5], dtype=float)

        self.update_command_state(
            command=command,
            command_source_time=command_source_time,
        )


    def update_command_state(self, command, command_source_time):
        """
        Update Motor DT command state using the original command source time.
        This prevents delayed Motor DT callbacks from introducing artificial
        prediction delay.
        """
        for motor_index in range(4):
            if command[motor_index] != self.previous_command[motor_index]:

                # Current prediction: every raw command change counts,
                # including command -> 0.
                self.last_command_change_time[motor_index] = command_source_time

                # Position target: only non-zero relative commands are accumulated.
                if command[motor_index] != 0.0:
                    self.commanded_target[motor_index] += command[motor_index]
                    self.last_target_change_time[motor_index] = command_source_time

                self.previous_command[motor_index] = command[motor_index]

        self.current_command = command

        # self.publish_prediction()
    def measured_position_callback(self, msg):
        if len(msg.data) >= 4:
            self.measured_positions = np.array(msg.data[:4], dtype=float)

    def measured_current_callback(self, msg):
        if len(msg.data) >= 4:
            self.measured_currents = np.array(msg.data[:4], dtype=float)
            
    def build_encoder_features(
        self,
        target,
        raw_command,
        time_since_target_change,
        time_since_command_change,
    ):
        is_command_active = 1.0 if raw_command != 0.0 else 0.0

        return np.array([[
            target,
            raw_command,
            abs(raw_command),
            np.sign(raw_command),
            time_since_target_change,
            time_since_command_change,
            abs(target),
            is_command_active,
        ]], dtype=float)

    def build_current_features(
        self,
        target,
        raw_command,
        time_since_command_change,
    ):
        is_command_active = 1.0 if raw_command != 0.0 else 0.0
        command_based_is_moving = is_command_active

        return np.array([[
            target,
            raw_command,
            abs(raw_command),
            np.sign(raw_command),
            time_since_command_change,
            abs(target),
            is_command_active,
            command_based_is_moving,
        ]], dtype=float)

    # def build_current_features(
    #     self,
    #     segment_time,
    #     target,
    #     raw_command,
    #     target_change,
    #     target_velocity,
    #     time_since_change,
    #     encoder,
    #     encoder_velocity,
    #     encoder_acceleration,
    # ):
    #     encoder_error = target - encoder

    #     is_command_active = 1.0 if raw_command != 0.0 else 0.0
    #     is_moving = 1.0 if (abs(encoder_velocity) > 5.0 or raw_command != 0.0) else 0.0

    #     return np.array([[
    #         segment_time,
    #         target,
    #         raw_command,
    #         abs(raw_command),
    #         target_change,
    #         target_velocity,
    #         abs(target_velocity),
    #         np.sign(target_velocity),
    #         time_since_change,
    #         abs(target),
    #         is_command_active,

    #         encoder,
    #         encoder_velocity,
    #         encoder_acceleration,
    #         encoder_error,
    #         abs(encoder_error),
    #         is_moving,
    #     ]], dtype=float)

    def publish_prediction(self):

        # active = self.get_active_motor_indices()
        # self.get_logger().info(
        #     f"active={list(active)}  dt={dt*1000:.1f} ms"
        # )
        prediction_start_time = self.get_time_s()
        now = prediction_start_time
        dt = now - self.previous_publish_time

        if dt <= 0.0:
            dt = 1e-3

        predicted_positions = self.previous_predicted_encoder.copy()
        predicted_currents = np.zeros(4, dtype=float)

        for motor_index in range(4):
            target = self.commanded_target[motor_index]
            raw_command = self.current_command[motor_index]

            # target_change = target - self.previous_target[motor_index]
            # target_velocity = target_change / dt

            # time_since_change = now - self.last_target_change_time[motor_index]
            time_since_target_change = now - self.last_target_change_time[motor_index]
            time_since_command_change = now - self.last_command_change_time[motor_index]
            model_package = self.models[motor_index]
            encoder_model = model_package["encoder_model"]
            current_model = model_package["current_model"]

            # x_encoder = self.build_encoder_features(
            #     segment_time=segment_time,
            #     target=target,
            #     raw_command=raw_command,
            #     target_change=target_change,
            #     target_velocity=target_velocity,
            #     time_since_change=time_since_change,
            # )
            x_encoder = self.build_encoder_features(
                target=target,
                raw_command=raw_command,
                time_since_target_change=time_since_target_change,
                time_since_command_change=time_since_command_change,
            )

            predicted_encoder = float(encoder_model.predict(x_encoder)[0])

            x_current = self.build_current_features(
                target=target,
                raw_command=raw_command,
                time_since_command_change=time_since_command_change,
            )

            # x_current = self.build_current_features(
            #     segment_time=segment_time,
            #     target=target,
            #     raw_command=raw_command,
            #     target_change=target_change,
            #     target_velocity=target_velocity,
            #     time_since_change=time_since_change,
            #     encoder=predicted_encoder,
            #     encoder_velocity=encoder_velocity,
            #     encoder_acceleration=encoder_acceleration,
            # )

            predicted_current = float(current_model.predict(x_current)[0])
            predicted_positions[motor_index] = predicted_encoder
            predicted_currents[motor_index] = predicted_current

        time_since_command_change_array = now - self.last_command_change_time
        time_since_target_change_array = now - self.last_target_change_time
        prediction_compute_end_time = self.get_time_s()
        self.last_prediction_compute_time_ms = (
            prediction_compute_end_time - prediction_start_time
        ) * 1000.0
        
        state_msg = Float64MultiArray()
        prediction_publish_time = self.get_time_s()

        state_msg.data = (
            predicted_positions.tolist()
            + predicted_currents.tolist()
            + self.current_command.tolist()
            + self.commanded_target.tolist()
            + time_since_command_change_array.tolist()
            + time_since_target_change_array.tolist()
            + self.measured_positions.tolist()
            + self.measured_currents.tolist()
            + [
                self.last_stamped_command_age_ms,
                prediction_start_time,
                prediction_publish_time,
                self.last_prediction_compute_time_ms,
            ]
        )
        self.predicted_state_publisher.publish(state_msg)
        positions_msg = Float64MultiArray()
        positions_msg.data = predicted_positions.tolist()
        self.predicted_position_publisher.publish(positions_msg)

        currents_msg = Float64MultiArray()
        currents_msg.data = predicted_currents.tolist()
        self.predicted_current_publisher.publish(currents_msg)

        self.previous_publish_time = now


def main(args=None):
    rclpy.init(args=args)

    node = MotorDigitalTwinNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()