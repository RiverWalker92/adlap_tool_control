#!/usr/bin/env python3

# import time
# from datetime import datetime
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Int32MultiArray, String

# PULSES_PER_ROTATION = 903

# def make_step(motor_index, step_size, duration):
#     steps = [0, 0, 0, 0]
#     steps[motor_index] = step_size
#     return {"steps": steps, "duration": duration}


# def make_m1_m2_same_direction_test(step_size=50, direction=1):
#     step = direction * step_size

#     return [
#         {"steps": [0, 0, -step, 0], "duration": 1.5},
#         {"steps": [0, 0, step, 0], "duration": 1.5},
#     ]

# def make_lower_ratio_check(motor_index, direction=-1):
#     step = direction * 150

#     return [
#         make_step(motor_index, step, 1.5),
#         make_step(motor_index, step, 1.5),
#         make_step(motor_index, step, 1.5),
#         make_step(motor_index, step, 1.5),
#         make_step(motor_index, step, 1.5),
#         make_step(motor_index, direction * 153, 1.5),
#     ]

# class SetupTestRunner(Node):
#     def __init__(self):
#         super().__init__("setup_test_runner_node")

#         self.motor_pub = self.create_publisher(
#             Int32MultiArray,
#             "/right/tool_control_node/commanded_motor_positions",
#             10
#         )
        
#         self.control_pub = self.create_publisher(
#             String,
#             "/right/tool_control_node/control",
#             10
#         )

#         self.task_pub = self.create_publisher(
#             String,
#             "/right/tool_control_node/task_label",
#             10
#         )

#         self.led_control_pub = self.create_publisher(
#             String,
#             "/right/tool_control_node/led_control",
#             10
#         )

#         self.setup_type = "motors_only"
#         self.instrument_id = "no_instrument"
#         self.setup_number = "setup_01"

#         self.get_logger().info("Visual motor rotation test runner started")

#     def publish_task_label(self, label):
#         msg = String()
#         msg.data = label
#         self.task_pub.publish(msg)
#         self.get_logger().info(f"Task label: {label}")

#     def publish_motor_command(self, m0, m1, m2, m3):
#         msg = Int32MultiArray()
#         msg.data = [int(m0), int(m1), int(m2), int(m3)]
#         self.motor_pub.publish(msg)

#         self.get_logger().info(
#             f"Motor command: m0={m0}, m1={m1}, m2={m2}, m3={m3}"
#         )

#     def send_command_and_wait(self, motor_steps, duration):
#         self.publish_motor_command(*motor_steps)
#         time.sleep(duration)

#     def set_led(self, state):
#         msg = String()
#         msg.data = f"led {int(state)}"
#         self.led_control_pub.publish(msg)

#         self.get_logger().info(f"LED command sent: {msg.data}")

#     def run_trial(self, test_type, motor_name, trial_number, sequence):
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#         task_label = (
#             f"{self.setup_number}_{self.setup_type}|"
#             f"{test_type}|"
#             f"{motor_name}|"
#             f"trial_{trial_number:02d}|"
#             f"{timestamp}"
#         )

#         self.publish_task_label(task_label)
#         self.get_logger().info(f"Starting trial: {task_label}")

#         # Led on for 2 seconds (start marker)
#         self.set_led(1)
#         time.sleep(2.0)

#         # Led off before movement
#         self.set_led(0)

#         time.sleep(0.5)

#         time.sleep(1.0)

#         for step in sequence:
#             steps = step["steps"]
#             duration = step["duration"]
#             self.send_command_and_wait(steps, duration)

#         time.sleep(1.0)

#         self.get_logger().info(f"Finished trial: {task_label}")

#     def run_all_tests(self):
#         self.get_logger().info("Starting m1 + m2 same-direction test...")
#         time.sleep(1.0)

#         self.run_trial(
#             "m1_m2_same_direction_check",
#             "m1_m2",
#             1,
#             make_m1_m2_same_direction_test(step_size=100, direction=1)
#         )

#         stop_msg = String()
#         stop_msg.data = "stop_logging"
#         self.control_pub.publish(stop_msg)
#         time.sleep(1.0)

#         self.publish_task_label("m1_m2_same_direction_check_finished")
#         self.get_logger().info("m1 + m2 same-direction check finished")

# def main(args=None):
#     rclpy.init(args=args)

#     node = SetupTestRunner()

#     try:
#         node.run_all_tests()
#     except KeyboardInterrupt:
#         node.get_logger().warn("Interrupted by user")
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()

# # !/usr/bin/env python3

import time
from datetime import datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, String

def make_step(motor_index, step_size, duration):
    steps = [0, 0, 0, 0]
    steps[motor_index] = step_size
    return {"steps": steps, "duration": duration}


def make_single_step_test(motor_index, step_size):
    return [
        make_step(motor_index, step_size, 2.0),
    ]


def make_backlash_test(motor_index, step_size):
    return [
        make_step(motor_index, step_size, 2.0),
        make_step(motor_index, -step_size, 2.0),
    ]


def make_reversal_test(motor_index, step_size):
    return [
        make_step(motor_index, step_size, 0.3),
        make_step(motor_index, -step_size, 2.0),
        make_step(motor_index, -step_size, 0.3),
        make_step(motor_index, step_size, 2.0),
    ]


def make_cyclic_test(motor_index, step_size, cycles=15):
    sequence = []
    for _ in range(cycles):
        sequence.append(make_step(motor_index, step_size, 1.0))
        sequence.append(make_step(motor_index, -step_size, 1.0))
    return sequence

# New ROS node to run a series of predefined test sequences
class SetupTestRunner(Node):
    def __init__(self):
        super().__init__("setup_test_runner_node")

        # Two publishers that send motor commands and task labels
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

    def publish_motor_command(self, m0, m1, m2, m3):
        msg = Int32MultiArray()
        msg.data = [int(m0), int(m1), int(m2), int(m3)]
        self.motor_pub.publish(msg)

        self.get_logger().info(
            f"Motor command: m0={m0}, m1={m1}, m2={m2}, m3={m3}"
        )

    def send_command_and_wait(self, motor_steps, duration):
        self.publish_motor_command(*motor_steps)
        time.sleep(duration)

    # Main function to run a single test trial with a given name, trial number, and sequence of commands
    def run_trial(self, test_type, motor_name, trial_number, sequence):  
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

        # short pause before movement
        time.sleep(1.0)

        # Run through the sequence of commands for this trial, holding each command for the specified duration
        for step in sequence:
            steps = step["steps"]
            duration = step["duration"]
            self.send_command_and_wait(steps, duration)

        time.sleep(1.0)

        self.get_logger().info(f"Finished trial: {task_label}")



    def run_all_tests(self):
        self.get_logger().info("Waiting before starting tests...")
        time.sleep(1.0)

        tests = []

        idle_baseline = [
            {"steps": [0, 0, 0, 0], "duration": 5.0},
        ]
        tests.append(("idle_baseline", "all_motors", idle_baseline))

        for motor_index in range(4):
            motor_name = f"m{motor_index + 1}"

            tests.append((
                "single_step_small",
                motor_name,
                make_single_step_test(motor_index, 100)
            ))

            tests.append((
                "single_step_medium",
                motor_name,
                make_single_step_test(motor_index, 300)
            ))

            tests.append((
                "backlash_small",
                motor_name,
                make_backlash_test(motor_index, 100)
            ))

            tests.append((
                "reversal_medium",
                motor_name,
                make_reversal_test(motor_index, 300)
            ))

            tests.append((
                "cyclic_medium",
                motor_name,
                make_cyclic_test(motor_index, 300, cycles=15)
            ))

        for test_type, motor_name, sequence in tests:
            for trial_number in range(1, 2): # meot 4 zijn
                self.run_trial(test_type, motor_name, trial_number, sequence)
                time.sleep(1.0)
       
        stop_msg = String()
        stop_msg.data = "stop_logging"
        self.control_pub.publish(stop_msg)
        time.sleep(1.0)        

        self.publish_task_label("tests_finished")
        time.sleep(1.0)

        self.get_logger().info("All setup tests finished")


def main(args=None):
    rclpy.init(args=args)

    node = SetupTestRunner()

    try:
        node.run_all_tests()
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()