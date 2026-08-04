#!/usr/bin/env python3

import time
from datetime import datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray,  Float64MultiArray, String

# Helper functions to create test sequences for motor patterns
def make_step(motor_index, step_size, duration):
    steps = [0, 0, 0, 0]
    steps[motor_index] = step_size
    return {"steps": steps, "duration": duration}


def make_idle_test(duration=3.0):
    return [
        {"steps": [0, 0, 0, 0], "duration": duration},
    ]

def make_back_and_forth_test(motor_index, step_size, hold_duration=2.0):
    return [
        make_step(motor_index, step_size, hold_duration),
        make_step(motor_index, -step_size, hold_duration),
        {"steps": [0, 0, 0, 0], "duration": 0.5},
    ]


def make_reversal_test(motor_index, step_size):
    return [
        make_step(motor_index, step_size, 0.3),
        {"steps": [0, 0, 0, 0], "duration": 0.3},

        make_step(motor_index, -step_size, 2.0),
        {"steps": [0, 0, 0, 0], "duration": 0.3},

        make_step(motor_index, -step_size, 0.3),
        {"steps": [0, 0, 0, 0], "duration": 0.3},

        make_step(motor_index, step_size, 2.0),
        {"steps": [0, 0, 0, 0], "duration": 0.5},
    ]


def make_cyclic_test(motor_index, step_size, cycles=5, hold_duration=1.0):
    sequence = []

    for _ in range(cycles):
        sequence.append(make_step(motor_index, step_size, hold_duration))
        sequence.append(make_step(motor_index, -step_size, hold_duration))

    sequence.append({"steps": [0, 0, 0, 0], "duration": 0.5})
    return sequence

def make_full_turn_test(motor_index, turns=3, pulses_per_turn=903, hold_duration=2.0):
    step_size = int(turns * pulses_per_turn)

    return [
        make_step(motor_index, step_size, hold_duration),
        {"steps": [0, 0, 0, 0], "duration": 1.5},
        make_step(motor_index, -step_size, hold_duration),
        {"steps": [0, 0, 0, 0], "duration": 1.5},
    ]

# New ROS node to run a series of predefined test sequences
class MotorPatternRunner(Node):
    def __init__(self):
        super().__init__("motor_pattern_runner_node")

        # Publishers that send motor commands and task labels
        self.motor_pub = self.create_publisher(
            Int32MultiArray,
            "/right/tool_control_node/commanded_motor_positions",
            10
        )

        self.task_pub = self.create_publisher(
            String,
            "/right/tool_control_node/task_label",
            10
        )
        self.control_pub = self.create_publisher(
            String,
            "/right/tool_control_node/control",
            10
        )
        self.motor_command_stamped_publisher = self.create_publisher(
            Float64MultiArray,
            "/right/tool_control_node/commanded_motor_positions_stamped",
            10,
        )

        # For later use in logging setup and instrument
        self.setup_type = "motors_only"
        self.instrument_id = "no_instrument"
        self.setup_number = "setup_01"

        self.get_logger().info("Setup test runner started")

    # Helper functions to publish task labels and commanded positions
    def publish_task_label(self, label):
        msg = String()
        msg.data = label
        self.task_pub.publish(msg)
        self.get_logger().info(f"Task label: {label}")
    # Helper function to publish motor commands and log them with a timestamp
    
    def publish_motor_command(self, m0, m1, m2, m3):
        stamp = self.get_clock().now().nanoseconds * 1e-9

        msg = Int32MultiArray()
        msg.data = [int(m0), int(m1), int(m2), int(m3)]
        self.motor_pub.publish(msg)

        stamped_msg = Float64MultiArray()
        stamped_msg.data = (
            [stamp]
            + [float(x) for x in msg.data[:4]]
        )

        self.motor_command_stamped_publisher.publish(stamped_msg)

        self.get_logger().info(
            f"Motor command: m0={m0}, m1={m1}, m2={m2}, m3={m3}"
        )

    def send_command_and_wait(self, motor_steps, duration):
        self.publish_motor_command(*motor_steps)
        time.sleep(duration)

    # Main function to run a single test trial with a given name, trial number, and sequence of commands
    def run_trial(self, test_type, motor_name, trial_number, sequence):  
        self.publish_task_label("between_trials")  # Publish a task label to indicate the end of the previous trial and the start of a pause
        time.sleep(0.9)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        task_label = (
            f"{self.setup_number}_{self.setup_type}|"
            f"{test_type}|"
            f"{motor_name}|"
        #   f"{self.instrument_id}|"
            f"trial_{trial_number:02d}|"
            f"{timestamp}"
        )

        self.publish_task_label(task_label)     # Publish the task label at the start of the trial so it gets logged by the tool reader node
        self.get_logger().info(f"Starting trial: {task_label}")

        time.sleep(0.1)  # Short pause to ensure the task label is logged before sending motor commands
        
        # Run through the sequence of commands for this trial, holding each command for the specified duration
        for step in sequence:
            self.send_command_and_wait(
                step["steps"],
                step["duration"],
            )

        # Trial is nu klaar; de sequence bevat zelf al settling
        self.publish_task_label("between_trials")
        self.get_logger().info(f"Finished trial: {task_label}")

        time.sleep(1.0)

    def run_all_tests(self):
        self.get_logger().info("Starting motor-only pattern tests...")
        time.sleep(1.0)

        number_of_trials = 3
        pause_between_trials = 0.5
        pause_between_tests = 0.5

        tests = []

        tests.append({
            "test_type": "idle_baseline",
            "motor_name": "all_motors",
            "sequence": make_idle_test(duration=3.0),
        })

        for motor_index in range(4):
            motor_name = f"m{motor_index}"

            tests.extend([
                {
                    "test_type": "back_and_forth_small",
                    "motor_name": motor_name,
                    "sequence": make_back_and_forth_test(motor_index, step_size=100),
                },
                {
                    "test_type": "back_and_forth_medium",
                    "motor_name": motor_name,
                    "sequence": make_back_and_forth_test(motor_index, step_size=300),
                },
                {
                    "test_type": "back_and_forth_large",
                    "motor_name": motor_name,
                    "sequence": make_back_and_forth_test(motor_index, step_size=900),
                },
                {
                    "test_type": "reversal_medium",
                    "motor_name": motor_name,
                    "sequence": make_reversal_test(motor_index, step_size=500),
                },
                {
                    "test_type": "cyclic_medium",
                    "motor_name": motor_name,
                    "sequence": make_cyclic_test(
                        motor_index,
                        step_size=500,
                        cycles=5,
                        hold_duration=1.0,
                    ),
                },
                {                
                    "test_type": "full_turn_single_3x",
                    "motor_name": motor_name,
                    "sequence": make_full_turn_test(
                        motor_index,
                        turns=3,
                        pulses_per_turn=903,
                        hold_duration=2.0,
                    ),
                },
            ])

        for test in tests:
            for trial_number in range(1, number_of_trials + 1):
                self.run_trial(
                    test_type=test["test_type"],
                    motor_name=test["motor_name"],
                    trial_number=trial_number,
                    sequence=test["sequence"],
                )

                time.sleep(pause_between_trials)

            time.sleep(pause_between_tests)

        stop_msg = String()
        stop_msg.data = "stop_logging"
        self.control_pub.publish(stop_msg)

        time.sleep(1.0)
        self.publish_task_label("motor_tests_finished")
        time.sleep(1.0)

        self.get_logger().info("All motor-only tests finished")


def main(args=None):
    rclpy.init(args=args)

    node = MotorPatternRunner()

    try:
        node.run_all_tests()
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

