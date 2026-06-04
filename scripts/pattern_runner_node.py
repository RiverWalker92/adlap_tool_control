#!/usr/bin/env python3

import math
import time
from datetime import datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String, Bool
from pathlib import Path


class PatternRunner(Node):
    def __init__(self):
        super().__init__("tool_controller_node")

        default_config_path = (
            Path.home()
            / "ros2_ws"
            / "src"
            / "adlap_tool_control"
            / "config"
            / "tool_params.yaml"
        )
        self.declare_parameter("topic", "/right/tool_control_node/instrument_angles")
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("duration", 30.0)
        self.duration = float(self.get_parameter("duration").value)

        for i in range(1, 5):
            prefix = f"dof{i}"
            self.declare_parameter(f"{prefix}.mode", "constant")
            self.declare_parameter(f"{prefix}.value", 0.0)
            self.declare_parameter(f"{prefix}.min", 0.0)
            self.declare_parameter(f"{prefix}.max", 0.0)
            self.declare_parameter(f"{prefix}.frequency", 0.1)

        self.topic = self.get_parameter("topic").value
        self.publish_rate = self.get_parameter("publish_rate").value

        self.dof4_active = self.get_parameter("dof4.mode").value != "constant"
        self.dof4_preroll_angle = 1.7 #90.0
        print("### USING PREROLL ANGLE =", self.dof4_preroll_angle, "###", flush=True)
        self.get_logger().info(f"DOF4 preroll angle: {self.dof4_preroll_angle}")
        self.dof4_preroll_wait = 2.0  # seconden wachten na draaien
        self.dof4_sync_wait = 2.0  # extra wachten na synchronisatie voordat patroon begint

        self.pub = self.create_publisher(Float64MultiArray, self.topic, 10)
        self.task_pub = self.create_publisher(
            String,
            "/right/tool_control_node/task_label",
            10,
        )

        self.led_pub = self.create_publisher(
            Bool,
            "/right/tool_control_node/led_control",
            10
        )

        self.start_time = time.time()
        led_msg = Bool()

        self.led_pulse_start_delay = 1.0
        self.led_pulse_duration = 2.0
        self.led_command_repeat_duration = 0.2
        self.led_on_sent = False
        self.led_off_sent = False
        self.dof4_motion_started = False
        self.motion_after_led_off_wait = 0.2
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = self.generate_test_name()

        task = String()
        task.data = (
            f"setup_03|"
            f"gearbox_instrument|"
            f"{test_name}|"
            f"trial_01|"
            f"{timestamp}"
        )
        self.task_msg = task
        self.task_publish_count = 0

        self.task_timer = self.create_timer(
            0.2,
            self.publish_task_label_repeatedly,
        )
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.timer_callback,
        )

        self.get_logger().info(f"Pattern runner publishing to {self.topic}")
        self.get_logger().info(f"Publish rate: {self.publish_rate} Hz")
        self.get_logger().info(f"task label: {task.data}")
    
    def publish_task_label_repeatedly(self):
        if self.task_publish_count >= 10:
            self.destroy_timer(self.task_timer)
            return

        self.task_pub.publish(self.task_msg)
        self.task_publish_count += 1
        self.get_logger().info(f"Published task label: {self.task_msg.data}")
        
    def generate_test_name(self):
        active = []

        for i in range(1, 5):
            mode = self.get_parameter(f"dof{i}.mode").value

            if mode != "constant":
                active.append(f"dof{i}_{mode}")

        if not active:
            return "all_constant"

        return "continuous_" + "_".join(active)

    def compute_dof(self, dof_name, t):
        mode = self.get_parameter(f"{dof_name}.mode").value
        value = float(self.get_parameter(f"{dof_name}.value").value)
        min_value = float(self.get_parameter(f"{dof_name}.min").value)
        max_value = float(self.get_parameter(f"{dof_name}.max").value)
        frequency = float(self.get_parameter(f"{dof_name}.frequency").value)

        if mode == "constant":
            return value

        if mode == "sinusoid":
            offset = 0.5 * (max_value + min_value)
            amplitude = 0.5 * (max_value - min_value)
            return offset + amplitude * math.sin(2.0 * math.pi * frequency * t)

        if mode == "triangle":
            period = 1.0 / frequency
            phase = (t % period) / period

            if phase < 0.5:
                return min_value + 2.0 * phase * (max_value - min_value)
            else:
                return max_value - 2.0 * (phase - 0.5) * (max_value - min_value)

        self.get_logger().warn(f"Unknown mode '{mode}' for {dof_name}, using constant value")
        return value

    def timer_callback(self):
        elapsed = time.time() - self.start_time

        if self.dof4_active:
            preroll_end = self.dof4_preroll_wait
            motion_start = self.dof4_preroll_wait + self.dof4_sync_wait

            # Fase 1 + 2: preroll en daarna stil wachten
            if elapsed < motion_start:
                values = [0.0, 0.0, self.dof4_preroll_angle, 0.0]

                # LED aan tijdens de wachttijd NA preroll
                led_should_be_on = (
                    preroll_end
                    <= elapsed
                    <
                    preroll_end + self.led_pulse_duration
                )

                led_msg = Bool()
                led_msg.data = led_should_be_on
                self.led_pub.publish(led_msg)

                if led_should_be_on and not self.led_on_sent:
                    self.led_on_sent = True
                    self.get_logger().info("LED ON")

                if not led_should_be_on and self.led_on_sent and not self.led_off_sent:
                    self.led_off_sent = True
                    self.get_logger().info("LED OFF")

                msg = Float64MultiArray()
                msg.data = values
                self.pub.publish(msg)
                return

            # Fase 3: echte continuous test begint hier pas
            t = elapsed - motion_start
            if not self.dof4_motion_started:
                self.dof4_motion_started = True

                led_msg = Bool()
                led_msg.data = False
                for _ in range(5):
                    self.led_pub.publish(led_msg)

                self.get_logger().info("DOF4 motion started after preroll/sync wait")
        # else:
        #     t = elapsed
        #     led_should_be_on = (
        #         self.led_pulse_start_delay
        #         <= t
        #         <
        #         self.led_pulse_start_delay + self.led_pulse_duration
        #     )

        #     led_msg = Bool()
        #     led_msg.data = led_should_be_on
        #     self.led_pub.publish(led_msg)

        #     if led_should_be_on and not self.led_on_sent:
        #         self.led_on_sent = True
        #         self.get_logger().info("LED ON")

        #     if not led_should_be_on and self.led_on_sent and not self.led_off_sent:
        #         self.led_off_sent = True
        #         self.get_logger().info("LED OFF")

        # if t > self.duration:
        #     led_msg = Bool()
        #     led_msg.data = False

        #     for _ in range(5):
        #         self.led_pub.publish(led_msg)

        #     self.get_logger().info("LED OFF at pattern end")
        #     self.get_logger().info("Pattern finished")
        #     self.destroy_timer(self.timer)
        #     return

        # values = [
        #     self.compute_dof("dof1", t),
        #     self.compute_dof("dof2", t),
        #     self.dof4_preroll_angle if self.dof4_active else self.compute_dof("dof3", t),
        #     self.compute_dof("dof4", t),
        # ]

        # msg = Float64MultiArray()
        # msg.data = values
        # self.pub.publish(msg)
        else:
            led_should_be_on = (
                self.led_pulse_start_delay
                <= elapsed
                <
                self.led_pulse_start_delay + self.led_pulse_duration
            )

            led_msg = Bool()
            led_msg.data = led_should_be_on
            self.led_pub.publish(led_msg)

            if led_should_be_on and not self.led_on_sent:
                self.led_on_sent = True
                self.get_logger().info("LED ON")

            if not led_should_be_on and self.led_on_sent and not self.led_off_sent:
                self.led_off_sent = True
                self.get_logger().info("LED OFF")

            motion_start = (
                self.led_pulse_start_delay
                + self.led_pulse_duration
                + self.motion_after_led_off_wait
            )

            # Voor motion_start: instrument stil houden
            if elapsed < motion_start:
                values = [
                    float(self.get_parameter("dof1.value").value),
                    float(self.get_parameter("dof2.value").value),
                    float(self.get_parameter("dof3.value").value),
                    float(self.get_parameter("dof4.value").value),
                ]

                msg = Float64MultiArray()
                msg.data = values
                self.pub.publish(msg)
                return

            # Echte continuous test begint pas na LED uit
            t = elapsed - motion_start
        if t > self.duration:
            led_msg = Bool()
            led_msg.data = False

            for _ in range(5):
                self.led_pub.publish(led_msg)

            self.get_logger().info("LED OFF at pattern end")
            self.get_logger().info("Pattern finished")
            self.destroy_timer(self.timer)
            return

        values = [
            self.compute_dof("dof1", t),
            self.compute_dof("dof2", t),
            self.dof4_preroll_angle if self.dof4_active else self.compute_dof("dof3", t),
            self.compute_dof("dof4", t),
        ]

        msg = Float64MultiArray()
        msg.data = values
        self.pub.publish(msg)
def main(args=None):
    rclpy.init(args=args)
    node = PatternRunner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        led_msg = Bool()
        led_msg.data = False

        for _ in range(5):
            node.led_pub.publish(led_msg)
            rclpy.spin_once(node, timeout_sec=0.05)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()