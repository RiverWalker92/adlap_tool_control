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

        self.lower_play_deg = config["backlash"]["lower_motors_play_deg"]

        mapping = config["gearbox_mapping"]
        self.inner_shaft_lead_mm_per_rotation = config["linear_conversion"]["inner_shaft_lead_mm_per_rotation"]
        
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
    
        # # Encoder / motor conversion
        # self.pulses_per_motor_rotation = 903

        # # Additional transmission factor from gearbox design (e.g. upper motors have a 25:15 gear ratio and lower motors have a 15:15 ratio)
        # self.upper_motor_factor = 25.0 / 15.0

        # self.pulses_per_upper_gear_rotation = (
        #     self.pulses_per_motor_rotation * self.upper_motor_factor
        # )

        # self.pulses_per_lower_gear_rotation = self.pulses_per_motor_rotation

        # # Play if we want to include
        # self.lower_play_deg = 15.0
        # self.lower_play_pulses = self.lower_play_deg * self.pulses_per_lower_gear_rotation / 360.0

        # # starting position of translation shaft
        self.inner_shaft_translation_state = 0.0

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

        return {
            "gear_l1_deg": motor_to_degrees(self.gear_l1_motor),
            "gear_l2_deg": motor_to_degrees(self.gear_l2_motor),
            "gear_r1_deg": motor_to_degrees(self.gear_r1_motor),
            "gear_r2_deg": motor_to_degrees(self.gear_r2_motor),
        }
        # dm0, dm1, dm2, dm3 = delta_motor_pulses

        # return {
        #     "gear_l1_deg": dm3 * 360.0 / self.pulses_per_upper_gear_rotation,
        #     "gear_l2_deg": dm0 * 360.0 / self.pulses_per_upper_gear_rotation,
        #     "gear_r1_deg": dm1 * 360.0 / self.pulses_per_lower_gear_rotation,
        #     "gear_r2_deg": dm2 * 360.0 / self.pulses_per_lower_gear_rotation,
        # }

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

        # translation_increment = gear_l1 - gear_l2
        # self.inner_shaft_translation_state += translation_increment

        # gear_l2_left_limit = a #still to define
        # gear_l2_right_limit = b #still to define

        # self.inner_shaft_translation_state = np.clip(
        #     self.inner_shaft_translation_state,
        #     gear_l2_left_limit,
        #     gear_l2_right_limit)

        instrument_shaft_inputs = {
            # Individual gear rotations
            "gear_l1_deg": gear_l1,
            "gear_l2_deg": gear_l2,
            "gear_r1_deg": gear_r1,
            "gear_r2_deg": gear_r2,

            # Left gear block: inner shaft
            # Positive = clockwise inner shaft rotation
            "inner_shaft_rotation_deg": 0.5 * (gear_l1 + gear_l2),

            # m3 motion relative to m0 produces inner shaft translation.
            # If m0 is stationary and m3 moves, this gives pure translation.
            # Rot-to-linear conversion is not included yet.
            # "inner_shaft_translation_index": gear_l1 - gear_l2,
            
            # "inner_shaft_pure_translation_index": gear_l2,
            # "inner_shaft_translation_state": self.inner_shaft_translation_state,

        # Still to be fixed!!
            "inner_shaft_translation_index_deg": gear_l1 - gear_l2,
            "inner_shaft_translation_mm": None if self.inner_shaft_lead_mm_per_rotation is None else (gear_l1 - gear_l2) / 360.0 * self.inner_shaft_lead_mm_per_rotation,

            # "inner_shaft_translation_state": self.inner_shaft_translation_state,
            
            # Right gear block: middle and outer shaft rotations
            "middle_shaft_rotation_deg": gear_r1, #this one not included in instrument controller script
            "outer_shaft_rotation_deg": gear_r2,

            # Relative rotation between middle and outer shaft. = bend and rot instrument 
            # Instrument effect depends on the coupled instrument.
            "middle_outer_relative_rotation_deg": gear_r1 - gear_r2,
        }

        return instrument_shaft_inputs
