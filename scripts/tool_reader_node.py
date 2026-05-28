#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32MultiArray, String




# !!!!!!!!!!!!!!!!!!!!!!!!!!
# ADD LOGGER DAT VOOR NU INSTRUMENT NIET GEKOPPELD MOET ZIJN! 
# !!!!!!!!!!!!!!!!!!!!!!!!!!




class ToolReader(Node):

    def __init__(self):
        super().__init__('tool_reader_node')

        self.get_logger().error("FOR MOTOR-ONLY TESTS: instrument should NOT be coupled.")

        self.last_task = "unlabeled"
        self.log_files = {}
        self.base_log_dir = Path.home() / "ros2_ws_roel_split" / "test_data"
        # Moet dit veranderd worden? 

        self.task_sub = self.create_subscription(
            String,
            '/right/tool_control_node/task_label',
            self.task_callback,
            10
        )

        self.last_currents = []
        self.last_positions = []
        self.last_instrument_angles = [] # based on commands, not encoder feedback
        self.last_current_instrument_angles = [] # based on motor encoder
        self.last_motor_commands = []
        # self.last_duty_cycle = []
        self.last_motor_config = []
        self.last_led_command = None

        self.control_sub = self.create_subscription(
            String,
            '/right/tool_control_node/control',
            self.control_callback,
            10
        )

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
        ) #commanded eruit gehaald, want dat is hetzelfde als instrument angles topic

        self.current_instrument_sub = self.create_subscription(
            Float64MultiArray,
            '/right/tool_control_node/current_instrument_angles',
            self.current_instrument_callback,
            10
        )

        self.motor_command_sub = self.create_subscription(
            Int32MultiArray,
            '/right/tool_control_node/commanded_motor_positions',
            self.motor_command_callback,
            10
        )

        # self.duty_cycle_sub = self.create_subscription(
        #     Int32MultiArray,
        #     '/right/tool_control_node/commanded_duty_cycle',
        #     self.duty_cycle_callback,
        #     10
        # )

        self.motor_config_sub = self.create_subscription(
            String,
            '/right/tool_control_node/motor_config',
            self.motor_config_callback,
            10
        )

        self.led_sub = self.create_subscription(
            String,
            '/right/tool_control_node/led_control',
            self.led_callback,
            10
        )


    def control_callback(self, msg):
        if msg.data == "stop_logging":
            self.get_logger().info("Stopping ToolReader")
            self.destroy_node()
            rclpy.shutdown()

    def task_callback(self, msg):
        self.last_task = msg.data
        self.get_log_file(self.last_task)   

    def current_callback(self, msg):
        self.last_currents = list(msg.data)
        self.print_and_log_state()

    def position_callback(self, msg):
        self.last_positions = list(msg.data)
        self.print_and_log_state()
        
    def instrument_command_callback(self, msg):
        self.last_instrument_angles = list(msg.data)
        self.print_and_log_state()

    def current_instrument_callback(self, msg):
        self.last_current_instrument_angles = list(msg.data)
        self.print_and_log_state()

    def motor_command_callback(self, msg):
        self.last_motor_commands = list(msg.data)
        self.print_and_log_state()

    # def duty_cycle_callback(self, msg):
    #     self.last_duty_cycle = list(msg.data)
    #     self.print_and_log_state()

    def motor_config_callback(self, msg):
        self.last_motor_config.append(msg.data)
        self.get_logger().warn(f"Received motor config: {msg.data}")
        self.print_and_log_state()

    def led_callback(self, msg):
        self.last_led_command = msg.data
        self.print_and_log_state()

    def get_log_file(self, task):
        parts = task.split("|")

        if len(parts) == 5:
            setup_number = parts[0]
            setup_type = parts[1]
            # instrument_id = parts[2]
            test_name = parts[2]
            trial_name = parts[3]
            timestamp = parts[4]

            task_dir = self.base_log_dir / setup_number / setup_type / test_name
            task_dir.mkdir(parents=True, exist_ok=True)

            log_path = task_dir / f"{trial_name}_{timestamp}.jsonl"

        else:
            safe_task = task.replace(" ", "_").replace("/", "_")
            task_dir = self.base_log_dir / "unstructured" / safe_task
            task_dir.mkdir(parents=True, exist_ok=True)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = task_dir / f"{safe_task}_{timestamp_str}.jsonl"

        if task not in self.log_files:
            self.log_files[task] = open(log_path, "a", buffering=1)
            self.get_logger().info(f"Logging task '{task}' to: {log_path}")

        return self.log_files[task]

    def print_and_log_state(self):
        if len(self.last_currents) == 4 and len(self.last_positions) == 4: #and len(self.last_commands) == 4:
            timestamp = self.get_clock().now().nanoseconds / 1e9

            sample = {
                "timestamp": timestamp,
                "commanded_instrument_angles": self.last_instrument_angles if len(self.last_instrument_angles) == 4 else None,
                "current_instrument_angles": self.last_current_instrument_angles if len(self.last_current_instrument_angles) == 4 else None,
                "commanded_motor_positions": self.last_motor_commands if len(self.last_motor_commands) == 4 else None,
                # "commanded_duty_cycle": self.last_duty_cycle if len(self.last_duty_cycle) == 4 else None,
                "measured_motor_positions": self.last_positions,
                "measured_currents": self.last_currents,
                "motor_config": self.last_motor_config if self.last_motor_config else None,
                "led_command": self.last_led_command
            }

            sample["task"] = self.last_task
            log_file = self.get_log_file(self.last_task)
            log_file.write(json.dumps(sample) + "\n")

            self.get_logger().info(
                f"{timestamp:.3f} | "
                f"Instrument Angles: {self.last_instrument_angles} | "
                f"Motor Commands: {self.last_motor_commands} | "
                # f"Duty Cycle: {self.last_duty_cycle}"
                f"Currents: {self.last_currents} | "
                f"Positions: {self.last_positions} | "
                f"Positions: {self.last_positions} | "
                f"Motor Config: {self.last_motor_config} | "
            )

    def destroy_node(self):
        for log_file in self.log_files.values():
            log_file.close()
        super().destroy_node()

# add someting for task etc
#log JSON file 1 whole and 1 divided

#print pas als alle 3 beschikbaar zijn

#heeel vaak gelogd, miss iedere ... ms?

def main(args=None):
    rclpy.init(args=args)
    node = ToolReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()