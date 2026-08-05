#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32MultiArray, String, Bool


class ToolReader(Node):

    def __init__(self):
        super().__init__('tool_reader_node')
        self.get_logger().error("FOR MOTOR-ONLY TESTS: instrument should NOT be coupled.")

        # General state variables
        self.last_task = "unlabeled"
        self.stop_requested = False
        self.start_time = self.get_clock().now().nanoseconds / 1e9

        # State variables for motor and instrument data
        self.last_currents = []
        self.last_positions = []
        self.last_instrument_angles = []            # Commanded instrument angles
        self.last_measured_instrument_angles = []    # Encoder-based instrument angles
        self.last_motor_commands = []
        self.last_led_command = None

        # Latest Digital Twin variables
        self.last_gearbox_state = []
        self.last_predicted_instrument_angles = []
        self.last_hybrid_predicted_instrument_angles = []
        self.last_predicted_motor_positions = []
        self.last_predicted_motor_currents = []

        # Timestamps used for timing/delay analysis of the received messages. 
        self.last_currents_time = None
        self.last_positions_time = None
        self.last_motor_commands_time = None
        self.last_predicted_motor_positions_time = None
        self.last_predicted_motor_currents_time = None

        # self.last_motor_config = [] 
        # self.last_predicted_motor_state = []
        # self.last_motor_dt_state_time = None
        # self.last_instrument_angles_time = None
        # self.last_measured_instrument_angles_time = None
        # self.last_motor_dt_debug_command_state_time = None
        # self.last_motor_dt_debug_command_state = []


        # Control/task subscriptions
        self.task_sub = self.create_subscription(
            String,
            '/right/tool_control_node/task_label',
            self.task_callback,
            10
        )

        self.control_sub = self.create_subscription(
            String,
            '/right/tool_control_node/control',
            self.control_callback,
            10
        )

        self.led_sub = self.create_subscription(
            Bool,
            '/right/tool_control_node/led_control',
            self.led_callback,
            10
        )

        # Measured/controller subscriptions
        self.current_sub = self.create_subscription(
            Float64MultiArray,
            '/right/tool_control_node/motor_currents',
            self.current_callback,
            10
        )

        self.position_sub = self.create_subscription(
            Float64MultiArray,
            '/right/tool_control_node/motor_positions',
            self.position_callback,
            10
        )

        self.instrument_command_sub = self.create_subscription(
            Float64MultiArray,
            '/right/tool_control_node/instrument_angles',
            self.instrument_command_callback,
            10
        ) 

        self.measured_instrument_sub = self.create_subscription(
            Float64MultiArray,
            '/right/tool_control_node/current_instrument_angles',
            self.measured_instrument_callback,
            10
        )   #current_instrument_angles from node, but renamed to measured_instrument_ for clarity

        self.motor_command_sub = self.create_subscription(
            Int32MultiArray,
            '/right/tool_control_node/commanded_motor_positions',
            self.motor_command_callback,
            10
        )

        # Digital Twin subscriptions
        self.gearbox_state_sub = self.create_subscription(
            Float64MultiArray,
            '/right/tool_control_node/gearbox_state',
            self.gearbox_state_callback,
            10
        )

        self.predicted_motor_position_sub = self.create_subscription(
            Float64MultiArray,
            '/right/motor_digital_twin/predicted_motor_positions',
            self.predicted_motor_position_callback,
            10
        )

        self.predicted_motor_current_sub = self.create_subscription(
            Float64MultiArray,
            '/right/motor_digital_twin/predicted_motor_currents',
            self.predicted_motor_current_callback,
            10
        )

        self.predicted_instrument_sub = self.create_subscription(
            Float64MultiArray,
            '/right/instrument_digital_twin/predicted_instrument_angles',
            self.predicted_instrument_callback,
            10
        )

        self.hybrid_predicted_instrument_sub = self.create_subscription(
            Float64MultiArray,
            '/right/instrument_digital_twin/hybrid_predicted_instrument_angles',
            self.hybrid_predicted_instrument_callback,
            10
        )

        # self.motor_dt_debug_command_state_sub = self.create_subscription(
        #     Float64MultiArray,
        #     "/right/motor_digital_twin/debug_command_state",
        #     self.motor_dt_debug_command_state_callback,
        #     10,
        # )
        # self.predicted_motor_state_sub = self.create_subscription(
        #     Float64MultiArray,
        #     "/right/motor_digital_twin/predicted_motor_state",
        #     self.predicted_motor_state_callback,
        #     10,
        # )

        # self.motor_config_sub = self.create_subscription(
        #     String,
        #     '/right/tool_control_node/motor_config',
        #     self.motor_config_callback,
        #     10
        # )

        # Logging setup
        self.declare_parameter(
            "output_dir",
            str(Path.home() / "ros2_ws" / "test_data" / "manual_logs"),
        )
        self.declare_parameter(
            "log_file_name",
            "",
        )

        output_dir_param = self.get_parameter("output_dir").value
        log_file_name_param = self.get_parameter("log_file_name").value

        self.output_dir = Path(output_dir_param).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if log_file_name_param:
            self.log_path = self.output_dir / log_file_name_param
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = self.output_dir / f"manual_log_{timestamp}.jsonl"

        self.log_file = open(self.log_path, "a", buffering=1)

        self.get_logger().info(f"Logging to: {self.log_path}")
        self.log_rate_hz = 100.0
        self.log_timer = self.create_timer(
            1.0 / self.log_rate_hz,
            self.print_and_log_state
        )

    def now_s(self):
        return self.get_clock().now().nanoseconds / 1e9

    def control_callback(self, msg):
        if msg.data == "stop_logging":
            self.get_logger().info("Stopping ToolReader")
            self.stop_requested = True

    def task_callback(self, msg):
        self.last_task = msg.data
        # self.get_log_file(self.last_task)   

    def current_callback(self, msg):
        self.last_currents = list(msg.data)
        self.last_currents_time = self.now_s()

    def position_callback(self, msg):
        self.last_positions = list(msg.data)
        self.last_positions_time = self.now_s()

    def instrument_command_callback(self, msg):
        self.last_instrument_angles = list(msg.data)

    def measured_instrument_callback(self, msg):
        self.last_measured_instrument_angles = list(msg.data)

    def motor_command_callback(self, msg):
        self.last_motor_commands = list(msg.data)
        self.last_motor_commands_time = self.now_s()
    
    def led_callback(self, msg):
        self.last_led_command = msg.data

    def gearbox_state_callback(self, msg):
        self.last_gearbox_state = list(msg.data)

    def predicted_motor_position_callback(self, msg):
        self.last_predicted_motor_positions = list(msg.data)
        self.last_predicted_motor_positions_time = self.now_s()

    def predicted_motor_current_callback(self, msg):
        self.last_predicted_motor_currents = list(msg.data)
        self.last_predicted_motor_currents_time = self.now_s()
    
    # def predicted_motor_state_callback(self, msg):
    #     self.last_predicted_motor_state = list(msg.data)
    #     self.last_motor_dt_state_time = self.now_s()

    # def motor_dt_debug_command_state_callback(self, msg):
    #     self.last_motor_dt_debug_command_state = list(msg.data)
    #     self.last_motor_dt_debug_command_state_time = self.now_s()

    # def motor_config_callback(self, msg):
    #     self.last_motor_config.append(msg.data)
    #     self.get_logger().warn(f"Received motor config: {msg.data}")

    def predicted_instrument_callback(self, msg):
        self.last_predicted_instrument_angles = list(msg.data)

    def hybrid_predicted_instrument_callback(self, msg):
        self.last_hybrid_predicted_instrument_angles = list(msg.data)
    
    def print_and_log_state(self):
        if self.stop_requested:
            if rclpy.ok():
                rclpy.shutdown()
            return
        # self.get_logger().info(
        #     f"Check lengths: currents={len(self.last_currents)}, positions={len(self.last_positions)}, task={self.last_task}"
        # )
        if len(self.last_currents) == 4 and len(self.last_positions) == 4: #and len(self.last_commands) == 4:
            timestamp = self.get_clock().now().nanoseconds / 1e9
            relative_time = timestamp - self.start_time

            sample = {
                "time": relative_time,
                "ros_timestamp": timestamp,

                "commanded_instrument_angles": self.last_instrument_angles if len(self.last_instrument_angles) == 4 else None,
                "commanded_motor_positions": self.last_motor_commands if len(self.last_motor_commands) == 4 else None,
                
                "measured_instrument_angles": self.last_measured_instrument_angles if len(self.last_measured_instrument_angles) == 4 else None,
                "measured_motor_positions": self.last_positions,
                "measured_currents": self.last_currents,
                
                # "motor_config": self.last_motor_config if self.last_motor_config else None,
                "led_command": self.last_led_command,
                
                "gearbox_state": self.last_gearbox_state if len(self.last_gearbox_state) == 6 else None,
                "predicted_motor_positions": (
                    self.last_predicted_motor_positions
                    if self.last_predicted_motor_positions else None),
                "predicted_motor_currents": (
                    self.last_predicted_motor_currents
                    if self.last_predicted_motor_currents else None),
                "predicted_instrument_angles": (self.last_predicted_instrument_angles
                    if self.last_predicted_instrument_angles else None),
                "hybrid_predicted_instrument_angles": (
                    self.last_hybrid_predicted_instrument_angles
                    if self.last_hybrid_predicted_instrument_angles else None),
                # "motor_dt_debug_command_state": (
                #     self.last_motor_dt_debug_command_state
                #     if len(self.last_motor_dt_debug_command_state) == 20 else None),
                # "motor_dt_state": (
                #     self.last_predicted_motor_state
                #     if len(self.last_predicted_motor_state) >= 24 else None),
                
                "measured_currents_timestamp": self.last_currents_time,
                "measured_motor_positions_timestamp": self.last_positions_time,
                "commanded_motor_positions_timestamp": self.last_motor_commands_time,
                "predicted_motor_positions_timestamp": self.last_predicted_motor_positions_time,
                "predicted_motor_currents_timestamp": self.last_predicted_motor_currents_time,
                # "motor_dt_state_timestamp": self.last_motor_dt_state_time,
                # "motor_dt_debug_command_state_timestamp": self.last_motor_dt_debug_command_state_time,
            #     "measured_currents_age": (
            #         timestamp - self.last_currents_time
            #         if self.last_currents_time is not None else None
            #     ),
            #     "measured_motor_positions_age": (
            #         timestamp - self.last_positions_time
            #         if self.last_positions_time is not None else None
            #     ),
            #     "commanded_motor_positions_age": (
            #         timestamp - self.last_motor_commands_time
            #         if self.last_motor_commands_time is not None else None
            #     ),
            #     "motor_dt_state_age": (
            #         timestamp - self.last_motor_dt_state_time
            #         if self.last_motor_dt_state_time is not None else None
            #     ),
            }

            sample["task"] = self.last_task
            sample["task_label"] = self.last_task
            self.log_file.write(json.dumps(sample) + "\n")
            # self.get_logger().info(f"Logging to: {self.log_path}")


    def destroy_node(self):
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    node = ToolReader()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user")
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()