#!/usr/bin/env python3

"""
ROS 2 DOF pattern runner for the SATA Digital Twin diagnostic setup.

This node generates standardized multi-condition motion sequences for the four
SATA degrees of freedom. Each trial keeps inactive DOFs at a constant value and
executes the configured waveform sequence for the selected DOF.

During a trial, the node publishes instrument commands, synchronization LED
signals, task labels, and sequence-condition labels for logging and subsequent
Digital Twin analysis.
"""

import math
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Float64MultiArray, String


class PatternRunner(Node):
    """
    ROS 2 node that generates and publishes standardized SATA DOF test patterns.

    The node executes a multi-stage waveform sequence for one selected DOF while
    keeping the remaining DOFs at their configured constant values. It also
    publishes synchronization signals and trial labels for logging.
    """
    def __init__(self):
        super().__init__("dof_pattern_runner_node")
        self.declare_parameter("topic", "/right/tool_control_node/instrument_angles")
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("duration", 11.0)
        self.declare_parameter("coupling_mode", "full_setup")

        # active_dof=0 uses the DOF configuration from the parameter file;
        # values 1-4 explicitly select one DOF.
        self.declare_parameter("active_dof", 0)
        self.coupling_mode = str(self.get_parameter("coupling_mode").value)
        self.active_dof_override = int(
            self.get_parameter("active_dof").value
        )
        if self.active_dof_override not in [0, 1, 2, 3, 4]:
            raise ValueError(
                "active_dof must be 0, 1, 2, 3 or 4"
            )
        self.duration = float(self.get_parameter("duration").value)

        # Declare parameters for each DOF
        for i in range(1, 5):
            prefix = f"dof{i}"
            self.declare_parameter(f"{prefix}.mode", "constant")
            self.declare_parameter(f"{prefix}.value", 0.0)
            self.declare_parameter(f"{prefix}.min", 0.0)
            self.declare_parameter(f"{prefix}.max", 0.0)
            self.declare_parameter(f"{prefix}.frequency", 0.1)

        # Declare parameters for DOF4 tip compensation
        self.declare_parameter("dof4.tip_compensation_pulses", 0)
        self.declare_parameter("dof4.tip_compensation_sign", -1)
        self.dof4_tip_compensation_pulses = int(
            self.get_parameter("dof4.tip_compensation_pulses").value
        )
        self.dof4_tip_compensation_sign = int(
            self.get_parameter("dof4.tip_compensation_sign").value
        )
        self.dof4_tip_compensation_offset_rad = 0.0
        self.previous_dof4_value_for_tip_compensation = None
        self.previous_dof4_direction_for_tip_compensation = 0

        self.get_logger().info(
            f"DOF4 tip compensation: "
            f"pulses={self.dof4_tip_compensation_pulses}, "
            f"sign={self.dof4_tip_compensation_sign}"
        )
        
        # -------------------------------------------------------------------------
        # Sequence parameters
        # -------------------------------------------------------------------------
        for dof_name in ["dof1", "dof2", "dof3", "dof4"]:
            self.declare_parameter(f"{dof_name}.sequence_modes")
            self.declare_parameter(f"{dof_name}.sequence_cycles")
            self.declare_parameter(f"{dof_name}.sequence_pause_duration")
            self.declare_parameter(f"{dof_name}.sequence_pause_led_start")
            self.declare_parameter(f"{dof_name}.sequence_pause_led_duration")

            self.declare_parameter(f"{dof_name}.sequence_frequency_factors")
            self.declare_parameter(f"{dof_name}.sequence_between_frequency_pause")

            self.declare_parameter(f"{dof_name}.sequence_range_factors")
            self.declare_parameter(f"{dof_name}.sequence_between_range_pause")
            self.declare_parameter(f"{dof_name}.sequence_final_pause_duration")
        
        self.sequence_enabled = {}
        self.sequence_modes = {}
        self.sequence_cycles = {}
        self.sequence_pause_duration = {}
        self.sequence_pause_led_start = {}
        self.sequence_pause_led_duration = {}
        self.sequence_frequency_factors = {}
        self.sequence_between_frequency_pause = {}
        self.sequence_range_factors = {}
        self.sequence_between_range_pause = {}
        self.sequence_final_pause_duration = {}

        for dof_name in ["dof1", "dof2", "dof3", "dof4"]:
            mode = self.get_parameter(f"{dof_name}.mode").value
            dof_number = int(dof_name[-1])

            # An explicit active_dof from the launch file overrides the sequence mode
            # configured for the other DOFs.
            if self.active_dof_override != 0:
                self.sequence_enabled[dof_name] = (
                    dof_number == self.active_dof_override
                )
            else:
                self.sequence_enabled[dof_name] = mode == "sequence"

            if self.sequence_enabled[dof_name]:
                self.sequence_modes[dof_name] = list(
                    self.get_required_parameter_value(f"{dof_name}.sequence_modes")
                )

                self.sequence_cycles[dof_name] = [
                    int(x)
                    for x in self.get_required_parameter_value(f"{dof_name}.sequence_cycles")
                ]

                self.sequence_pause_duration[dof_name] = float(
                    self.get_required_parameter_value(f"{dof_name}.sequence_pause_duration")
                )

                self.sequence_pause_led_start[dof_name] = float(
                    self.get_required_parameter_value(f"{dof_name}.sequence_pause_led_start")
                )

                self.sequence_pause_led_duration[dof_name] = float(
                    self.get_required_parameter_value(f"{dof_name}.sequence_pause_led_duration")
                )

                self.sequence_frequency_factors[dof_name] = [
                    float(x)
                    for x in self.get_required_parameter_value(
                        f"{dof_name}.sequence_frequency_factors"
                    )
                ]

                self.sequence_between_frequency_pause[dof_name] = float(
                    self.get_required_parameter_value(
                        f"{dof_name}.sequence_between_frequency_pause"
                    )
                )

                self.sequence_range_factors[dof_name] = [
                    float(x)
                    for x in self.get_required_parameter_value(
                        f"{dof_name}.sequence_range_factors"
                    )
                ]

                self.sequence_between_range_pause[dof_name] = float(
                    self.get_required_parameter_value(
                        f"{dof_name}.sequence_between_range_pause"
                    )
                )
                self.sequence_final_pause_duration[dof_name] = float(
                    self.get_required_parameter_value(
                        f"{dof_name}.sequence_final_pause_duration"
                    )
                )

                if len(self.sequence_modes[dof_name]) != len(self.sequence_cycles[dof_name]):
                    raise ValueError(
                        f"{dof_name}.sequence_modes and {dof_name}.sequence_cycles must have the same length"
                    )

                self.get_logger().info(
                    f"Loaded {dof_name} sequence: "
                    f"modes={self.sequence_modes[dof_name]}, "
                    f"cycles={self.sequence_cycles[dof_name]}, "
                    f"frequency_factors={self.sequence_frequency_factors[dof_name]}, "
                    f"range_factors={self.sequence_range_factors[dof_name]}"
                )

            else:
                self.sequence_modes[dof_name] = []
                self.sequence_cycles[dof_name] = []
                self.sequence_pause_duration[dof_name] = 0.0
                self.sequence_pause_led_start[dof_name] = 0.0
                self.sequence_pause_led_duration[dof_name] = 0.0
                self.sequence_frequency_factors[dof_name] = []
                self.sequence_between_frequency_pause[dof_name] = 0.0
                self.sequence_range_factors[dof_name] = []
                self.sequence_between_range_pause[dof_name] = 0.0
                self.sequence_final_pause_duration[dof_name] = 0.0

        self.sequence_led_active = False
        self.current_sequence_condition = "none"
        self.last_published_sequence_condition = None

        active_sequence_dofs = [
            dof_name
            for dof_name, enabled in self.sequence_enabled.items()
            if enabled
        ]

        if len(active_sequence_dofs) > 1:
            raise ValueError(
                "Only one DOF sequence may be active per trial. "
                f"Active sequences: {active_sequence_dofs}"
            )

        self.active_sequence_dof = (
            active_sequence_dofs[0]
            if active_sequence_dofs
            else None
        )
        self.topic = self.get_parameter("topic").value
        self.publish_rate = self.get_parameter("publish_rate").value
        for i in range(1, 5):
            mode = self.get_parameter(f"dof{i}.mode").value
            value = self.get_parameter(f"dof{i}.value").value
            min_value = self.get_parameter(f"dof{i}.min").value
            max_value = self.get_parameter(f"dof{i}.max").value
            frequency = self.get_parameter(f"dof{i}.frequency").value

            self.get_logger().info(
                f"Loaded dof{i}: mode={mode}, value={value}, "
                f"min={min_value}, max={max_value}, frequency={frequency}"
            )
            
        # -------------------------------------------------------------------------
        # ROS publishers
        # -------------------------------------------------------------------------
        self.pub = self.create_publisher(Float64MultiArray, self.topic, 10)
        self.task_pub = self.create_publisher(
            String,
            "/right/tool_control_node/task_label",
            10,
        )
        self.sequence_condition_pub = self.create_publisher(
            String,
            "/right/tool_control_node/sequence_condition",
            10,
        )

        self.led_pub = self.create_publisher(
            Bool,
            "/right/tool_control_node/led_control",
            10
        )

        self.start_time = time.time()
        self.finished = False
        
        # Pre-motion synchronization: wait 1 s, pulse the LED for 2 s, then wait
        # another 0.2 s before starting the motion pattern.
        self.led_pulse_start_delay = 1.0
        self.led_pulse_duration = 2.0
        self.led_on_sent = False
        self.led_off_sent = False
        self.motion_after_led_off_wait = 0.2
        
        # -------------------------------------------------------------------------
        # Generate the task name
        # -------------------------------------------------------------------------
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = self.generate_test_name()

        task = String()
        if self.coupling_mode == "motor_only":
            setup_label = "setup_01_motors"

        elif self.coupling_mode == "gearbox_only":
            setup_label = "setup_02_motors_gearbox"

        elif self.coupling_mode == "full_setup":
            setup_label = "setup_03_motors_gearbox_instrument"        
        else:
            raise RuntimeError(
                f"Unsupported coupling mode for DOF pattern runner: "
                f"{self.coupling_mode}"
            )
        
        task.data = (
            f"{setup_label}|"
            f"{self.coupling_mode}|"
            f"{test_name}|"
            f"trial_01|"
            f"{self.timestamp}"
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
    
    def get_required_parameter_value(self, parameter_name):
        """
        Return a required ROS parameter value.

        Raise a RuntimeError if the parameter was not provided in the DOF pattern
        configuration.
        """
        parameter = self.get_parameter(parameter_name)

        if parameter.type_ == Parameter.Type.NOT_SET:
            raise RuntimeError(
                f"Required parameter '{parameter_name}' is missing from dof_pattern_params.yaml"
            )

        return parameter.value

    def publish_task_label_repeatedly(self):
        """
        Publish the task label repeatedly at trial start to ensure it is received
        by the trial logger.
        """
        if self.task_publish_count >= 10:
            self.destroy_timer(self.task_timer)
            return

        self.task_pub.publish(self.task_msg)
        self.task_publish_count += 1
        self.get_logger().info(f"Published task label: {self.task_msg.data}")

    def publish_sequence_condition(self):
        """
        Publish the active sequence condition when it changes.
        """
        condition = self.current_sequence_condition

        if condition == self.last_published_sequence_condition:
            return

        msg = String()
        msg.data = condition

        self.sequence_condition_pub.publish(msg)

        self.last_published_sequence_condition = condition

    def generate_test_name(self):
        """
        Generate a descriptive test name based on the active DOFs and their modes.
        """
        active = []

        for i in range(1, 5):
            mode = self.get_parameter(f"dof{i}.mode").value

            sequence_active = (
                i == self.active_dof_override
                if self.active_dof_override != 0
                else mode == "sequence"
            )

            if sequence_active:
                modes = list(self.get_parameter(f"dof{i}.sequence_modes").value)
                frequency_factors = list(
                    self.get_parameter(f"dof{i}.sequence_frequency_factors").value
                )
                range_factors = list(
                    self.get_parameter(f"dof{i}.sequence_range_factors").value
                )

                frequency_text = "_".join(
                    f"{float(f):.1f}".replace(".", "p")
                    for f in frequency_factors
                )

                range_text = "_".join(
                    f"{float(r):.1f}".replace(".", "p")
                    for r in range_factors
                )

                active.append(
                    f"dof{i}_sequence_{'_'.join(modes)}"
                    f"_freqx{frequency_text}"
                    f"_rangex{range_text}"
                )
            
            elif mode != "constant":
                active.append(f"dof{i}_{mode}")

        if not active:
            return "all_constant"

        return "continuous_" + "_".join(active)
        
    def compute_dof(
        self,
        dof_name,
        t,
        mode=None,
        frequency_override=None,
        range_factor_override=None,
    ):
        """
        Compute the commanded value for one DOF.

        Optional mode, frequency, and range overrides are used by sequence trials.
        """
        value = float(
            self.get_parameter(f"{dof_name}.value").value
        )

        if mode is None:
            mode = self.get_parameter(f"{dof_name}.mode").value

            if (
                self.active_dof_override != 0
                and dof_name != f"dof{self.active_dof_override}"
            ):
                return value

        min_value = float(
            self.get_parameter(f"{dof_name}.min").value
        )
        max_value = float(
            self.get_parameter(f"{dof_name}.max").value
        )

        if range_factor_override is not None:
            range_factor = float(range_factor_override)
            min_value = value + range_factor * (min_value - value)
            max_value = value + range_factor * (max_value - value)

        if frequency_override is None:
            frequency = float(
                self.get_parameter(f"{dof_name}.frequency").value
            )
        else:
            frequency = float(frequency_override)

        if mode == "constant":
            return value

        if frequency <= 0.0:
            raise ValueError(
                f"Frequency must be greater than zero for mode '{mode}'."
            )

        offset = 0.5 * (max_value + min_value)
        amplitude = 0.5 * (max_value - min_value)

        if mode == "sinusoid":
            return offset - amplitude * math.cos(
                2.0 * math.pi * frequency * t
            )

        if mode == "sinusoid_mid":
            return offset + amplitude * math.sin(
                2.0 * math.pi * frequency * t
            )

        if mode == "triangle":
            period = 1.0 / frequency
            phase = (t % period) / period

        elif mode == "triangle_mid":
            period = 1.0 / frequency
            phase = ((t % period) / period + 0.25) % 1.0

        else:
            self.get_logger().warn(
                f"Unknown mode '{mode}' for {dof_name}, "
                "using constant value"
            )
            return value

        if phase < 0.5:
            return min_value + 2.0 * phase * (
                max_value - min_value
            )

        return max_value - 2.0 * (phase - 0.5) * (
            max_value - min_value
        )

    def compute_dof_sequence_block(
        self,
        dof_name,
        t,
        frequency,
        frequency_factor,
        range_factor,
    ):
        """
        Evaluate one sequence block for a fixed frequency and range factor.

        The block may contain multiple waveform stages separated by configured pauses.
        """
        value = float(self.get_parameter(f"{dof_name}.value").value)

        elapsed_in_sequence = t
        self.sequence_led_active = False

        for idx, (mode, cycles) in enumerate(
            zip(self.sequence_modes[dof_name], self.sequence_cycles[dof_name])
        ):
            stage_duration = float(cycles) / frequency

            # Motion stage
            if elapsed_in_sequence <= stage_duration:

                self.current_sequence_condition = (
                    f"{str(mode).replace('_mid', '')}_"
                    f"f{float(frequency_factor):.1f}_"
                    f"r{float(range_factor):.1f}"
                ).replace(".", "p")
                                
                return self.compute_dof(
                    dof_name,
                    elapsed_in_sequence,
                    mode=mode,
                    frequency_override=frequency,
                    range_factor_override=range_factor,
                )

            elapsed_in_sequence -= stage_duration

            # Pause only between waveform stages, not after final waveform
            is_last_stage = idx == len(self.sequence_modes[dof_name]) - 1

            if not is_last_stage:
                pause_duration = self.sequence_pause_duration[dof_name]

                if elapsed_in_sequence <= pause_duration:
                    self.current_sequence_condition = "pause"
                    self.sequence_led_active = (
                        self.sequence_pause_led_start[dof_name]
                        <= elapsed_in_sequence
                        <
                        self.sequence_pause_led_start[dof_name]
                        + self.sequence_pause_led_duration[dof_name]
                    )

                    return value

                elapsed_in_sequence -= pause_duration

        return None

    def compute_dof_sequence(self, dof_name, t):
        """
        Compute the active DOF value for the complete standardized sequence.

        The sequence iterates over configured range and frequency factors and inserts
        the configured pauses between waveform, frequency, and range blocks.
        """
        base_frequency = float(self.get_parameter(f"{dof_name}.frequency").value)
        value = float(self.get_parameter(f"{dof_name}.value").value)

        elapsed = t
        self.sequence_led_active = False

        frequency_factors = self.sequence_frequency_factors[dof_name]
        range_factors = self.sequence_range_factors[dof_name]

        for range_index, range_factor in enumerate(range_factors):
            for frequency_index, frequency_factor in enumerate(frequency_factors):
                frequency = base_frequency * float(frequency_factor)

                block_duration = self.get_sequence_block_duration(
                    dof_name,
                    frequency,
                )

                if elapsed <= block_duration:
                    return self.compute_dof_sequence_block(
                        dof_name,
                        elapsed,
                        frequency,
                        frequency_factor=frequency_factor,
                        range_factor=range_factor,
                    )

                elapsed -= block_duration

                # Pause between frequency blocks within the same range block.
                is_last_frequency = frequency_index == len(frequency_factors) - 1

                if not is_last_frequency:
                    pause_duration = self.sequence_between_frequency_pause[dof_name]

                    if elapsed <= pause_duration:
                        self.current_sequence_condition = "pause"
                        self.sequence_led_active = True
                        return value

                    elapsed -= pause_duration

            # Pause between range blocks.
            is_last_range = range_index == len(range_factors) - 1

            if not is_last_range:
                pause_duration = self.sequence_between_range_pause[dof_name]

                if elapsed <= pause_duration:
                    self.current_sequence_condition = "pause"
                    self.sequence_led_active = True
                    return value

                elapsed -= pause_duration
        final_pause_duration = self.sequence_final_pause_duration[dof_name]

        if elapsed <= final_pause_duration:
            self.current_sequence_condition = "pause"
            self.sequence_led_active = False
            return value

        elapsed -= final_pause_duration
        return None

    def get_sequence_block_duration(self, dof_name, frequency):
        """
        Compute the total duration of a single-frequency sequence block for one DOF.
        """
        motion_duration = sum(
            float(cycles) / frequency
            for cycles in self.sequence_cycles[dof_name]
        )

        number_of_stage_pauses = max(0, len(self.sequence_modes[dof_name]) - 1)
        stage_pause_duration = (
            number_of_stage_pauses
            * self.sequence_pause_duration[dof_name]
        )

        return motion_duration + stage_pause_duration

    def get_sequence_duration(self, dof_name):
        """
        Compute the total duration of the complete sequence for one DOF, including
        all waveform stages, pauses between stages, and pauses between frequency and range blocks.
        """
        base_frequency = float(self.get_parameter(f"{dof_name}.frequency").value)

        frequency_factors = self.sequence_frequency_factors[dof_name]
        range_factors = self.sequence_range_factors[dof_name]

        total_duration = 0.0

        for range_index, _range_factor in enumerate(range_factors):
            for frequency_index, frequency_factor in enumerate(frequency_factors):
                frequency = base_frequency * float(frequency_factor)

                total_duration += self.get_sequence_block_duration(
                    dof_name,
                    frequency,
                )

                is_last_frequency = frequency_index == len(frequency_factors) - 1

                if not is_last_frequency:
                    total_duration += self.sequence_between_frequency_pause[dof_name]

            is_last_range = range_index == len(range_factors) - 1

            if not is_last_range:
                total_duration += self.sequence_between_range_pause[dof_name]
        total_duration += self.sequence_final_pause_duration[dof_name]
        
        return total_duration

    def update_dof4_tip_compensation(self, dof4_value):
        if self.dof4_tip_compensation_pulses == 0:
            return

        if self.previous_dof4_value_for_tip_compensation is None:
            self.previous_dof4_value_for_tip_compensation = dof4_value
            return

        delta = dof4_value - self.previous_dof4_value_for_tip_compensation
        self.previous_dof4_value_for_tip_compensation = dof4_value

        if abs(delta) < 1e-4:
            return

        direction = 1 if delta > 0 else -1
        self.get_logger().info(
            f"delta={delta:.5f}, direction={direction}, previous={self.previous_dof4_direction_for_tip_compensation}"
        )
        if self.previous_dof4_direction_for_tip_compensation == 0:
            self.previous_dof4_direction_for_tip_compensation = direction
            return

        if direction == self.previous_dof4_direction_for_tip_compensation:
            return

        self.previous_dof4_direction_for_tip_compensation = direction

        pulses_per_rotation = 903.0
        compensation_rad = (
            self.dof4_tip_compensation_pulses
            * 2.0
            * math.pi
            / pulses_per_rotation
        )

        self.dof4_tip_compensation_offset_rad = (
            self.dof4_tip_compensation_sign
            * direction
            * compensation_rad
        )

        self.get_logger().info(
            f"DOF4 tip compensation offset: {self.dof4_tip_compensation_offset_rad:.4f} rad"
        )
        
    def timer_callback(self):
        """
        Update synchronization signals and publish the current DOF command.

        This callback runs at the configured publish rate, holds the instrument at its
        start position during LED synchronization, executes the configured motion
        pattern, and stops the trial when the pattern duration has elapsed.
        """
        elapsed = time.time() - self.start_time

        # Same pre-motion LED sync for all DOFs, including DOF4.
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

        # Before motion_start: keep all DOFs at their neutral/start value.
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

        # Real continuous test starts only after LED off + small wait.
        t = elapsed - motion_start

        if self.active_sequence_dof is not None:
            pattern_duration = self.get_sequence_duration(self.active_sequence_dof)
        else:
            pattern_duration = self.duration

        if t > pattern_duration:
            led_msg = Bool()
            led_msg.data = False

            for _ in range(5):
                self.led_pub.publish(led_msg)

            self.get_logger().info("LED OFF at pattern end")
            self.get_logger().info("Pattern finished")

            self.finished = True

            try:
                self.destroy_timer(self.timer)
            except Exception as exc:
                self.get_logger().warning(
                    f"Could not destroy pattern timer: {exc}"
                )

            return

        dof_values = {}

        for dof_name in ["dof1", "dof2", "dof3", "dof4"]:

            if self.sequence_enabled.get(dof_name, False):
                dof_value = self.compute_dof_sequence(dof_name, t)
                self.publish_sequence_condition()

                if dof_value is None:
                    dof_value = float(self.get_parameter(f"{dof_name}.value").value)
            else:
                dof_value = self.compute_dof(dof_name, t)

            dof_values[dof_name] = dof_value
        led_msg = Bool()
        led_msg.data = self.sequence_led_active
        self.led_pub.publish(led_msg)
            
        self.update_dof4_tip_compensation(dof_values["dof4"])
        values = [
            dof_values["dof1"],
            dof_values["dof2"],
            dof_values["dof3"] + self.dof4_tip_compensation_offset_rad,
            dof_values["dof4"],
        ]
        msg = Float64MultiArray()
        msg.data = values
        self.pub.publish(msg)


def main(args=None):
    """
    Initialize ROS 2, run the pattern node, and perform a safe shutdown.
    """
    rclpy.init(args=args)
    node = PatternRunner()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        pass

    finally:
        # Ensure the synchronization LED is turned off and the node is destroyed before shutting down ROS.
        led_msg = Bool()
        led_msg.data = False

        for _ in range(10):
            node.led_pub.publish(led_msg)
            rclpy.spin_once(node, timeout_sec=0.05)

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()