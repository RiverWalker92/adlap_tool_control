#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


class ToolReader(Node):

    def __init__(self):
        super().__init__('tool_reader_node')

        self.last_task = "unlabeled"
        self.log_files = {}
        self.base_log_dir = Path.home() / "ros2_ws" / "test_data"

        self.task_sub = self.create_subscription(
            String,
            '/right/tool_control_node/task_label',
            self.task_callback,
            10
        )

        self.last_currents = []
        self.last_positions = []
        self.last_commands = []

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

        self.command_sub = self.create_subscription(
            Float64MultiArray,
            '/right/tool_control_node/commanded_instrument_angles',
            self.command_callback,
            10
        )
    def task_callback(self, msg):
        self.last_task = msg.data
        self.get_log_file(self.last_task)   

    def current_callback(self, msg):
        self.last_currents = list(msg.data)
        self.print_and_log_state()

    def position_callback(self, msg):
        self.last_positions = list(msg.data)
        self.print_and_log_state()

    def command_callback(self, msg):
        self.last_commands = list(msg.data)
        self.print_and_log_state()

    def get_log_file(self, task):
        parts = task.split("|")

        if len(parts) == 6:
            setup_number = parts[0]
            setup_type = parts[1]
            instrument_id = parts[2]
            test_name = parts[3]
            trial_name = parts[4]
            timestamp = parts[5]

            task_dir = self.base_log_dir / setup_number / setup_type / instrument_id / test_name
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
                "commanded_instrument_angles": self.last_commands if len(self.last_commands) == 4 else None,
                "measured_motor_positions": self.last_positions,
                "measured_currents": self.last_currents,
            }

            sample["task"] = self.last_task
            log_file = self.get_log_file(self.last_task)
            log_file.write(json.dumps(sample) + "\n")

            self.get_logger().info(
                f"{timestamp:.3f} | "
                f"Command: {self.last_commands} | "
                f"Currents: {self.last_currents} | "
                f"Positions: {self.last_positions}"
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