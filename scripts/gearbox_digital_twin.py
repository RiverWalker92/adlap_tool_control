#!/usr/bin/env python3
import yaml
from pathlib import Path

# This script defines a digital twin for the gearbox, which can be used to predict the state of the output shaft based on the raw motor encoder readings. This is a preliminary model and can be refined with physical calibration
class GearboxDigitalTwin:
    def __init__(self, config_path="config/gearbox_params.yaml"):
        # Load configuration from YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)["gearbox"]

        self.pulses_per_motor_rotation = config["encoder"]["pulses_per_motor_rotation"]
        self.upper_motor_factor = config["ratios"]["upper_motor_factor"]
        self.lower_motor_factor = config["ratios"]["lower_motor_factor"]

        self.upper_motors = config["motor_groups"]["upper_motors"]
        self.lower_motors = config["motor_groups"]["lower_motors"]

        self.lower_play_deg = config["backlash"].get("lower_motors_play_deg", 0.0)        
        backlash = config["backlash"]

        self.gear_backlash_pulses = {
            "gear_l1": backlash.get("gear_l1_backlash_pulses", 0),
            "gear_l2": backlash.get("gear_l2_backlash_pulses", 0),
            "gear_r1": backlash.get("gear_r1_backlash_pulses", 0),
            "gear_r2": backlash.get("gear_r2_backlash_pulses", 0),
        }   
        print("Loaded backlash pulses:", self.gear_backlash_pulses)
        print("Config path:", config_path)
        mapping = config["gearbox_mapping"]
        
        self.inner_shaft_lead_mm_per_rotation = (
            config.get("linear_conversion", {}).get("inner_shaft_lead_mm_per_rotation")
            or 3.0
        )
        self.gear_l1_motor = mapping["gear_l1_motor"]
        self.gear_l2_motor = mapping["gear_l2_motor"]
        self.gear_r1_motor = mapping["gear_r1_motor"]
        self.gear_r2_motor = mapping["gear_r2_motor"]

        self.pulses_per_upper_gear_rotation = (
            self.pulses_per_motor_rotation * self.upper_motor_factor
        )

        self.pulses_per_lower_gear_rotation = (
            self.pulses_per_motor_rotation * self.lower_motor_factor
        )
    
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

    def apply_backlash(self, gear_name, current_pulses, backlash_pulses):
        """
        Simple backlash model:
        the first part of the motion is lost in mechanical play.
        Simple static deadband approximation.
        Real backlash should only be applied after direction reversal.
        """
        previous_input = self.last_input_pulses[gear_name]
        previous_output = self.last_output_pulses[gear_name]
        previous_direction = self.last_direction[gear_name]
        delta = current_pulses - previous_input

        if delta > 0:
            direction = 1
        elif delta < 0:
            direction = -1
        else:
            return previous_output

        if previous_direction != 0 and direction != previous_direction:
            self.remaining_backlash_pulses[gear_name] = backlash_pulses

        movement = abs(delta)

        if self.remaining_backlash_pulses[gear_name] > 0:
            lost = min(movement, self.remaining_backlash_pulses[gear_name])
            self.remaining_backlash_pulses[gear_name] -= lost
            movement -= lost

        effective_output = previous_output + direction * movement
        if (
            direction != 0
            and self.last_direction[gear_name] != 0
            and direction != self.last_direction[gear_name]
        ):
            print(
                f"BACKLASH {gear_name}: "
                f"dir {self.last_direction[gear_name]} -> {direction}"
            )
            
        self.last_input_pulses[gear_name] = current_pulses
        self.last_output_pulses[gear_name] = effective_output
        self.last_direction[gear_name] = direction

        return effective_output

    def motor_delta_pulses_to_gear_degrees(self, delta_motor_pulses):
        """
        Converts motor displacement relative to the start position to gear rotations.

        Input:
            delta_motor_pulses = [dm0, dm1, dm2, dm3]

        Gear naming:
            Left to right in gearbox:   left side = collet screw side
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
            self.gear_backlash_pulses["gear_l1"]
        )
        # gear_l1_pulses = delta_motor_pulses[self.gear_l1_motor]
        # gear_l2_pulses = delta_motor_pulses[self.gear_l2_motor]
        # gear_r1_pulses = delta_motor_pulses[self.gear_r1_motor]
        # gear_r2_pulses = delta_motor_pulses[self.gear_r2_motor]
        
        gear_l2_pulses = self.apply_backlash(
            "gear_l2",
            delta_motor_pulses[self.gear_l2_motor],
            self.gear_backlash_pulses["gear_l2"]
        )

        gear_r1_pulses = self.apply_backlash(
            "gear_r1",
            delta_motor_pulses[self.gear_r1_motor],
            self.gear_backlash_pulses["gear_r1"]
        )

        gear_r2_pulses = self.apply_backlash(
            "gear_r2",
            delta_motor_pulses[self.gear_r2_motor],
            self.gear_backlash_pulses["gear_r2"]
        )

        return {
            "gear_l1_deg": gear_l1_pulses * 360.0 / self.pulses_per_upper_gear_rotation,
            "gear_l2_deg": gear_l2_pulses * 360.0 / self.pulses_per_upper_gear_rotation,
            "gear_r1_deg": gear_r1_pulses * 360.0 / self.pulses_per_lower_gear_rotation,
            "gear_r2_deg": gear_r2_pulses * 360.0 / self.pulses_per_lower_gear_rotation,
        }

    def predict_instrument_shaft_inputs(self, delta_motor_pulses):
        """
        Converts motor encoder positions to the gearbox outputs that drive
        the coupled instrument shafts.

        This model describes only the gearbox/transmission output:
            - inner shaft rotation
            - inner shaft translation index
            - middle shaft rotation
            - outer shaft rotation

        Instrument-specific effects such as jaw opening, bending, or tip angle
        are not included here.
        """
        gears = self.motor_delta_pulses_to_gear_degrees(delta_motor_pulses)

        gear_l1 = gears["gear_l1_deg"]
        gear_l2 = gears["gear_l2_deg"]
        gear_r1 = gears["gear_r1_deg"]
        gear_r2 = gears["gear_r2_deg"]

        inner_shaft_relative_rotation_deg = gear_l1 - gear_l2

        # inner_shaft_translation_mm_raw = (
        #     None
        #     if self.inner_shaft_lead_mm_per_rotation is None
        #     else inner_shaft_relative_rotation_deg / 360.0 * self.inner_shaft_lead_mm_per_rotation
        # )

        # if inner_shaft_translation_mm_raw is None:
        #     inner_shaft_translation_mm = None
        # else:
        #     inner_shaft_translation_mm = max(
        #         0.0,
        #         min(6.0, inner_shaft_translation_mm_raw)
            # )

        inner_shaft_translation_mm_raw = (
            inner_shaft_relative_rotation_deg / 360.0
            * self.inner_shaft_lead_mm_per_rotation
        )

        inner_shaft_translation_mm = inner_shaft_translation_mm_raw
        
        instrument_shaft_inputs = {
                
            # Individual gear rotations
            "gear_l1_deg": gear_l1,
            "gear_l2_deg": gear_l2,
            "gear_r1_deg": gear_r1,
            "gear_r2_deg": gear_r2,

            # Left gear block: inner shaft
            # Positive = clockwise inner shaft rotation
            "inner_shaft_rotation_deg": gear_l2, #0.5 * (gear_l1 + gear_l2),

            # relative = gear_l1 - gear_l2
            # translation_raw = relative / 360 * lead

            # if translation_raw <= lower_stop:
            #     translation = lower_stop
            #     rotation = gear_l1   # of gear_l2, afhankelijk van sign
            # elif translation_raw >= upper_stop:
            #     translation = upper_stop
            #     rotation = gear_l1   # of gear_l2
            # else:
            #     translation = translation_raw
            #     rotation = previous_rotation

            # m3 motion relative to m0 produces inner shaft translation.
            # If m0 is stationary and m3 moves, this gives pure translation.
            "inner_shaft_relative_rotation_deg": inner_shaft_relative_rotation_deg,
            "inner_shaft_translation_mm_raw": inner_shaft_translation_mm_raw,
            "inner_shaft_translation_mm": inner_shaft_translation_mm,

            # Right gear block: middle and outer shaft rotations
            "middle_shaft_rotation_deg": gear_r1, #this one not included in instrument controller script
            "outer_shaft_rotation_deg": gear_r2,

            # Relative rotation between middle and outer shaft. = bend and rot instrument 
            # Instrument effect depends on the coupled instrument.
            "middle_outer_relative_rotation_deg": gear_r1 - gear_r2,
        }

        return instrument_shaft_inputs
