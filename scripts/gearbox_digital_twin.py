#!/usr/bin/env python3
import yaml
from pathlib import Path

# This script defines a digital twin for the gearbox, 
# which can be used to predict the state of the output shaft 
# based on the raw motor encoder readings. 
# This is a preliminary model and can be refined with physical calibration
def load_active_gearbox_config(config_path):
    """
    Load the active gearbox variant from the gearbox YAML file.

    Supports two formats:

    New format:
        gearbox:
          active_variant: gearbox_2
          variants:
            gearbox_1:
              ...
            gearbox_2:
              ...

    Old format:
        gearbox:
          version: 1
          encoder:
            ...
    """
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    gearbox_root = data["gearbox"]

    # New format: gearbox -> active_variant -> variants
    if "variants" in gearbox_root:
        active_variant = gearbox_root.get("active_variant")

        if active_variant is None:
            raise ValueError(
                "No 'active_variant' specified in gearbox YAML."
            )

        variants = gearbox_root["variants"]

        if active_variant not in variants:
            available = list(variants.keys())
            raise ValueError(
                f"Active gearbox variant '{active_variant}' not found. "
                f"Available variants are: {available}"
            )

        config = variants[active_variant]
        config["active_variant"] = active_variant

    # Old format fallback
    else:
        config = gearbox_root
        config["active_variant"] = f"version_{config.get('version', 'unknown')}"

    return config


class GearboxDigitalTwin:
    def __init__(self, config_path="config/gearbox_params.yaml"):
        # Load configuration from gearbox selection of YAML file
        config = load_active_gearbox_config(config_path)

        self.active_variant = config.get("active_variant", "unknown")
        self.version = config.get("version", None)
        print(f"Loaded gearbox variant: {self.active_variant}, version: {self.version}")

        self.pulses_per_motor_rotation = config["encoder"]["pulses_per_motor_rotation"]
        self.upper_motor_factor = config["ratios"]["upper_motor_factor"]
        self.lower_motor_factor = config["ratios"]["lower_motor_factor"]

        self.upper_motors = config["motor_groups"]["upper_motors"]
        self.lower_motors = config["motor_groups"]["lower_motors"]

        # self.lower_play_deg = config["backlash"].get("lower_motors_play_deg", 0.0)        
        # Load backlash compensation parameters for each gear.
        # The deadband avoids treating small encoder jitter as a real direction reversal.
        backlash = config["backlash"]
        self.direction_deadband_pulses = backlash.get("direction_deadband_pulses", 5)

        self.gear_backlash_pulses = {
            "gear_l1": {
                "positive": backlash["gear_l1_backlash_positive_pulses"],
                "negative": backlash["gear_l1_backlash_negative_pulses"],
            },
            "gear_l2": {
                "positive": backlash["gear_l2_backlash_positive_pulses"],
                "negative": backlash["gear_l2_backlash_negative_pulses"],
            },
            "gear_r1": {
                "positive": backlash["gear_r1_backlash_positive_pulses"],
                "negative": backlash["gear_r1_backlash_negative_pulses"],
            },
            "gear_r2": {
                "positive": backlash["gear_r2_backlash_positive_pulses"],
                "negative": backlash["gear_r2_backlash_negative_pulses"],
            },
        }
        print("Loaded direction-dependent backlash pulses:", self.gear_backlash_pulses)
        print("Config path:", config_path)
        mapping = config["gearbox_mapping"]
        
        self.inner_shaft_lead_mm_per_rotation = (
            config.get("linear_conversion", {}).get("inner_shaft_lead_mm_per_rotation")
        )
        linear_config = config.get("linear_conversion", {})

        self.inner_shaft_translation_start_mm = linear_config.get(
            "inner_shaft_translation_start_mm",
            0.0,
        )

        self.inner_shaft_translation_min_mm = linear_config.get(
            "inner_shaft_translation_min_mm",
            None,
        )

        self.inner_shaft_translation_max_mm = linear_config.get(
            "inner_shaft_translation_max_mm",
            None,
        )

        self.gear_l1_motor = mapping["gear_l1_motor"]
        self.gear_l2_motor = mapping["gear_l2_motor"]
        self.gear_r1_motor = mapping["gear_r1_motor"]
        self.gear_r2_motor = mapping["gear_r2_motor"]

        # Precompute how many encoder pulses correspond to one full gear rotation.
        self.pulses_per_upper_gear_rotation = (
            self.pulses_per_motor_rotation * self.upper_motor_factor
        )

        self.pulses_per_lower_gear_rotation = (
            self.pulses_per_motor_rotation * self.lower_motor_factor
        )
        # Store motion history for each gear.
        # This is required because backlash depends on previous 
        # direction and previous position.

        self.inner_shaft_translation_state = 0.0
        self.last_direction = {
            "gear_l1": 0,
            "gear_l2": 0,
            "gear_r1": 0,
            "gear_r2": 0,
        }

        self.last_input_pulses = {
            "gear_l1": 0.0,
            "gear_l2": 0.0,
            "gear_r1": 0.0,
            "gear_r2": 0.0,
        }

        self.last_output_pulses = {
            "gear_l1": 0.0,
            "gear_l2": 0.0,
            "gear_r1": 0.0,
            "gear_r2": 0.0,
        }

        self.remaining_backlash_pulses = {
            "gear_l1": 0.0,
            "gear_l2": 0.0,
            "gear_r1": 0.0,
            "gear_r2": 0.0,
        }

        self.pending_direction = {
            "gear_l1": 0,
            "gear_l2": 0,
            "gear_r1": 0,
            "gear_r2": 0,
        }

        self.reversal_reference_pulses = {
            "gear_l1": 0.0,
            "gear_l2": 0.0,
            "gear_r1": 0.0,
            "gear_r2": 0.0,
        }

    # Apply a history-dependent backlash model.
    # After a confirmed direction reversal, part of the motor 
    # motion is lost before the gear output moves again.
    def get_backlash_pulses(self, gear_name, direction):
        """
        Return backlash for the direction in which the motor/gear input is now moving.

        direction > 0 uses the positive-direction backlash.
        direction < 0 uses the negative-direction backlash.
        """
        backlash_values = self.gear_backlash_pulses[gear_name]

        if direction > 0:
            return backlash_values["positive"]
        elif direction < 0:
            return backlash_values["negative"]
        else:
            return 0.0

    def apply_backlash(self, gear_name, current_pulses):        
        previous_input = self.last_input_pulses[gear_name]
        previous_output = self.last_output_pulses[gear_name]
        previous_direction = self.last_direction[gear_name]

        delta = current_pulses - previous_input
        #Determines movement direction
        if delta > 0:
            direction = 1
        elif delta < 0:
            direction = -1
        else:
            return previous_output

        # First movement: no backlash yet, just follow input
        if previous_direction == 0:
            effective_output = previous_output + delta

            self.last_input_pulses[gear_name] = current_pulses
            self.last_output_pulses[gear_name] = effective_output
            self.last_direction[gear_name] = direction
            return effective_output

        # Possible direction reversal
        if direction != previous_direction:
            # Start or update pending reversal
            if self.pending_direction[gear_name] != direction:
                self.pending_direction[gear_name] = direction
                self.reversal_reference_pulses[gear_name] = previous_input

            reversal_movement = abs(
                current_pulses - self.reversal_reference_pulses[gear_name]
            )

            # Ignore tiny reversals as encoder jitter/noise.
            if reversal_movement < self.direction_deadband_pulses:
                return previous_output

            # Real reversal confirmed: activate direction-dependent backlash
            backlash_pulses = self.get_backlash_pulses(gear_name, direction)

            self.remaining_backlash_pulses[gear_name] = backlash_pulses
            self.last_direction[gear_name] = direction
            self.pending_direction[gear_name] = 0

            movement = reversal_movement

        else:
            # Same direction as before: normal movement
            self.pending_direction[gear_name] = 0
            movement = abs(delta)

        # Apply remaining backlash if active
        if self.remaining_backlash_pulses[gear_name] > 0:
            lost = min(movement, self.remaining_backlash_pulses[gear_name])
            self.remaining_backlash_pulses[gear_name] -= lost
            movement_after_backlash = movement - lost
        else:
            movement_after_backlash = movement

        effective_output = previous_output + direction * movement_after_backlash

        self.last_input_pulses[gear_name] = current_pulses
        self.last_output_pulses[gear_name] = effective_output
        self.last_direction[gear_name] = direction

        return effective_output

    def motor_delta_pulses_to_gear_degrees(self, delta_motor_pulses):
        """
        Converts motor displacement relative to the start position
        to gear rotations.

        Input:
            delta_motor_pulses = [dm0, dm1, dm2, dm3]

        Gear naming:
            Left to right in gearbox: left side = collet screw side
                                      right side = instrument / outer shaft side
            left side:  gear_l1, gear_l2
            right side: gear_r1, gear_r2

        Mapping:
            gear_l1 <- dm3, upper ratio
            gear_l2 <- dm0, upper ratio
            gear_r1 <- dm1, lower/direct ratio
            gear_r2 <- dm2, lower/direct ratio
        """
        def motor_to_degrees(motor_index):
            # not used now.
            factor = (
                self.upper_motor_factor
                if motor_index in self.upper_motors
                else self.lower_motor_factor
            )
            pulses_per_rotation = self.pulses_per_motor_rotation * factor
            return delta_motor_pulses[motor_index] * 360.0 / pulses_per_rotation

        gear_l1_pulses = self.apply_backlash(
            "gear_l1",
            delta_motor_pulses[self.gear_l1_motor],
        )

        gear_l2_pulses = self.apply_backlash(
            "gear_l2",
            delta_motor_pulses[self.gear_l2_motor],
        )

        gear_r1_pulses = self.apply_backlash(
            "gear_r1",
            delta_motor_pulses[self.gear_r1_motor],
        )

        gear_r2_pulses = self.apply_backlash(
            "gear_r2",
            delta_motor_pulses[self.gear_r2_motor],
        )

        return {
            "gear_l1_deg": gear_l1_pulses * 360.0 / self.pulses_per_upper_gear_rotation,
            "gear_l2_deg": gear_l2_pulses * 360.0 / self.pulses_per_upper_gear_rotation,
            "gear_r1_deg": gear_r1_pulses * 360.0 / self.pulses_per_lower_gear_rotation,
            "gear_r2_deg": gear_r2_pulses * 360.0 / self.pulses_per_lower_gear_rotation,
        }

    def predict_instrument_shaft_inputs(self, delta_motor_pulses):
        """
        Converts motor encoder positions to gearbox outputs.

        Gearbox/collet model:
            gear_l1 = axially fixed collet/screw gear
            gear_l2 = gear coupled to the inner shaft

        Within the collet translation range:
            gear_l1 - gear_l2 causes inner shaft translation

        At a translation stop:
            translation is clamped
            extra gear_l1 motion is interpreted as shaft rotation
        """

        gears = self.motor_delta_pulses_to_gear_degrees(delta_motor_pulses)

        gear_l1 = gears["gear_l1_deg"]
        gear_l2 = gears["gear_l2_deg"]
        gear_r1 = gears["gear_r1_deg"]
        gear_r2 = gears["gear_r2_deg"]

        # Raw screw/collet relation
        inner_shaft_relative_rotation_deg_raw = gear_l1 - gear_l2

        inner_shaft_translation_mm_raw = (
            self.inner_shaft_translation_start_mm
            + inner_shaft_relative_rotation_deg_raw / 360.0
            * self.inner_shaft_lead_mm_per_rotation
        )

        # Clamp the predicted translation to the physical min/max 
        # travel of the collet mechanism.
        inner_shaft_translation_mm = inner_shaft_translation_mm_raw

        if self.inner_shaft_translation_min_mm is not None:
            inner_shaft_translation_mm = max(
                self.inner_shaft_translation_min_mm,
                inner_shaft_translation_mm,
            )

        if self.inner_shaft_translation_max_mm is not None:
            inner_shaft_translation_mm = min(
                self.inner_shaft_translation_max_mm,
                inner_shaft_translation_mm,
            )

        # Convert the physically possible translation back to effective relative rotation
        if self.inner_shaft_lead_mm_per_rotation != 0:
            inner_shaft_relative_rotation_deg = (
                (inner_shaft_translation_mm - self.inner_shaft_translation_start_mm)
                / self.inner_shaft_lead_mm_per_rotation
                * 360.0
            )
        else:
            inner_shaft_relative_rotation_deg = inner_shaft_relative_rotation_deg_raw

        # Effective inner shaft rotation
        # Inside translation range this equals gear_l2.
        # At a stop, relative rotation is clamped, so gear_l1 motion
        # becomes shaft rotation.
        inner_shaft_rotation_deg_raw = gear_l2
        inner_shaft_rotation_deg = gear_l1 - inner_shaft_relative_rotation_deg

        inner_shaft_translation_blocked_mm = (
            inner_shaft_translation_mm_raw - inner_shaft_translation_mm
        )

        collet_at_min_stop = (
            self.inner_shaft_translation_min_mm is not None
            and inner_shaft_translation_mm_raw < self.inner_shaft_translation_min_mm
        )

        collet_at_max_stop = (
            self.inner_shaft_translation_max_mm is not None
            and inner_shaft_translation_mm_raw > self.inner_shaft_translation_max_mm
        )
        # Return all gearbox output states 
        # needed by the instrument Digital Twin layer.
        instrument_shaft_inputs = {
            # Individual gear rotations
            "gear_l1_deg": gear_l1,
            "gear_l2_deg": gear_l2,
            "gear_r1_deg": gear_r1,
            "gear_r2_deg": gear_r2,

            # Left gear block / collet
            "inner_shaft_rotation_deg_raw": inner_shaft_rotation_deg_raw,
            "inner_shaft_rotation_deg": inner_shaft_rotation_deg,

            "inner_shaft_relative_rotation_deg_raw": inner_shaft_relative_rotation_deg_raw,
            "inner_shaft_relative_rotation_deg": inner_shaft_relative_rotation_deg,

            "inner_shaft_translation_mm_raw": inner_shaft_translation_mm_raw,
            "inner_shaft_translation_mm": inner_shaft_translation_mm,
            "inner_shaft_translation_blocked_mm": inner_shaft_translation_blocked_mm,

            "collet_at_min_stop": collet_at_min_stop,
            "collet_at_max_stop": collet_at_max_stop,

            # Right gear block
            "middle_shaft_rotation_deg": gear_r1,
            "outer_shaft_rotation_deg": gear_r2,
            "middle_outer_relative_rotation_deg": gear_r1 - gear_r2,
        }

        return instrument_shaft_inputs