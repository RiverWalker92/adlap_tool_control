#!/usr/bin/env python3

import math
import yaml
from typing import Sequence, Dict


class InstrumentDigitalTwin:
    """
    Converts gearbox output states to predicted instrument output angles.

    Input order:
    [
        inner_shaft_rotation_deg,
        inner_shaft_relative_rotation_deg,
        inner_shaft_translation_mm,
        middle_shaft_rotation_deg,
        outer_shaft_rotation_deg,
        middle_outer_relative_rotation_deg,
    ]

    Output order:
    [   dof1 = shaft_roll, 
        dof2 = bend, 
        dof3 = tip_rotation,
        dof4 = articulation
    ]
    all in radians
    """

    # Load instrument calibration parameters from the YAML 
    # configuration file.
    def __init__(self, config_path="config/tool_params.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["instrument"]

        # self.articulation_factor = config["articulation_factor"]
        self.jaw_angle_deg_per_mm = config["jaw_angle_deg_per_mm"]
        self.between_jaws_deg_per_mm = config["between_jaws_deg_per_mm"]
        self.jaw_opening_threshold_mm = config["jaw_opening_threshold_mm"]
        self.jaw_closing_threshold_mm = config["jaw_closing_threshold_mm"]
        
        self.jaw_opening_backlash_mm = config["jaw_opening_backlash_mm"]
        self.jaw_closing_backlash_mm = config["jaw_closing_backlash_mm"]

        self.bend_factor = config["bend_factor"]
        self.instrument_bend_offset_deg = config["instrument_bend_offset_deg"]
        self.instrument_bend_backlash_positive_deg = config["instrument_bend_backlash_positive_deg"]
        self.instrument_bend_backlash_negative_deg = config["instrument_bend_backlash_negative_deg"]

        self.jaw_translation_min_mm = config.get("jaw_translation_min_mm", None)
        self.jaw_translation_max_mm = config.get("jaw_translation_max_mm", None)
        self.usable_jaw_translation_min_mm = config.get(
            "usable_jaw_translation_min_mm",
            0.0,
        )
        self.usable_jaw_translation_max_mm = config.get(
            "usable_jaw_translation_max_mm",
            self.jaw_translation_max_mm,
        )

        self.shaft_roll_sign = config["shaft_roll_sign"]
        self.bend_sign = config["bend_sign"]
        self.tip_rotation_sign = config["tip_rotation_sign"]
        self.articulation_sign = config["articulation_sign"]

        self._jaw_effective_translation_mm = None
        self._effective_bend_deg = None
        self._last_usable_translation_mm = None
        self._last_jaw_direction = 0
    
    # Reset backlash memory before starting a new 
    # trial or independent simulation.
    def reset(self):
        self._jaw_effective_translation_mm = None
        self._effective_bend_deg = None
        self._last_usable_translation_mm = None
        self._last_jaw_direction = 0

    def apply_jaw_backlash(self, translation_mm: float) -> float:
        opening_backlash = self.jaw_opening_backlash_mm
        closing_backlash = self.jaw_closing_backlash_mm

        if self._jaw_effective_translation_mm is None:
            self._jaw_effective_translation_mm = translation_mm
            return translation_mm

        previous_effective = self._jaw_effective_translation_mm
        # If the jaw is opening enough to overcome backlash, 
        # the effective translation starts moving.
        if translation_mm > previous_effective + opening_backlash:
            effective_translation = translation_mm - opening_backlash

        #If the jaw is closing enough to overcome backlash, the 
        # effective translation starts moving in the other direction.
        elif translation_mm < previous_effective - closing_backlash:
            effective_translation = translation_mm + closing_backlash

        # Small movements are absorbed by backlash and do not change the output.
        else:
            effective_translation = previous_effective

        self._jaw_effective_translation_mm = effective_translation
        return effective_translation

    # Apply direction-dependent bend backlash.
    # The effective bend only changes once the raw bend 
    # exceeds the backlash threshold.
    def apply_instrument_bend_backlash(self, raw_bend_deg: float) -> float:
        if self._effective_bend_deg is None:
            self._effective_bend_deg = raw_bend_deg
            return raw_bend_deg

        previous = self._effective_bend_deg

        if raw_bend_deg > previous:
            backlash = self.instrument_bend_backlash_positive_deg
        else:
            backlash = self.instrument_bend_backlash_negative_deg

        if raw_bend_deg > previous + backlash:
            effective = raw_bend_deg - backlash
        elif raw_bend_deg < previous - backlash:
            effective = raw_bend_deg + backlash
        else:
            effective = previous

        self._effective_bend_deg = effective
        return effective

    # Main conversion from gearbox output vector 
    # to instrument output angles.
    def euler_angles_from_gearbox_output(
        self,
        gearbox_output: Sequence[float],
    ) -> Dict[str, float]:

        # Unpack gearbox states. The order must match the output 
        # order of the gearbox digital twin.
        inner_shaft_rotation_deg = gearbox_output[0]
        inner_shaft_relative_rotation_deg = gearbox_output[1]
        inner_shaft_translation_mm = gearbox_output[2]
        middle_shaft_rotation_deg = gearbox_output[3]
        outer_shaft_rotation_deg = gearbox_output[4]
        middle_outer_relative_rotation_deg = gearbox_output[5]

        shaft_roll = math.radians(
            self.shaft_roll_sign * middle_shaft_rotation_deg
        )

        # bend = math.radians(
        #     self.params.bend_sign
        #     * middle_outer_relative_rotation_deg
        #     / self.params.bend_factor
        # )

        raw_bend_deg = (
            self.bend_sign
            * middle_outer_relative_rotation_deg
            / self.bend_factor
        )
        # A fixed bend offset is added > is now 0 so delete?
        raw_bend_deg = raw_bend_deg + self.instrument_bend_offset_deg
        # Backlash is applied
        bend_deg = self.apply_instrument_bend_backlash(raw_bend_deg)

        bend = math.radians(bend_deg)

        tip_rotation = math.radians(
            self.tip_rotation_sign * inner_shaft_rotation_deg
        )

        signed_translation_mm = (
            self.articulation_sign
            * inner_shaft_translation_mm
        )

        # Clamp instrument jaw translation to the 
        # usable/physical instrument range
        jaw_translation_mm = signed_translation_mm

        if self.jaw_translation_min_mm is not None:
            jaw_translation_mm = max(
                self.jaw_translation_min_mm,
                jaw_translation_mm,
            )

        if self.jaw_translation_max_mm is not None:
            jaw_translation_mm = min(
                self.jaw_translation_max_mm,
                jaw_translation_mm,
            )

        # Usable opening range: below usable_jaw_translation_min_mm,
        # the shaft may translate, but the jaws do not visibly open yet.
        usable_translation_mm = jaw_translation_mm - self.usable_jaw_translation_min_mm

        if self.usable_jaw_translation_max_mm is not None:
            usable_range_mm = (
                self.usable_jaw_translation_max_mm
                - self.usable_jaw_translation_min_mm
            )
            usable_translation_mm = min(
                usable_range_mm,
                usable_translation_mm,
            )

        usable_translation_mm = max(0.0, usable_translation_mm)

        # Determine whether the jaw mechanism is opening or closing
        # based on the raw usable translation before backlash.
        if self._last_usable_translation_mm is None:
            jaw_direction = 0
        else:
            delta_translation = usable_translation_mm - self._last_usable_translation_mm

            if delta_translation > 1e-6:
                jaw_direction = 1      # opening
            elif delta_translation < -1e-6:
                jaw_direction = -1     # closing
            else:
                jaw_direction = self._last_jaw_direction

        self._last_usable_translation_mm = usable_translation_mm
        self._last_jaw_direction = jaw_direction

        # Apply direction-dependent jaw backlash.
        backlash_translation_mm = self.apply_jaw_backlash(usable_translation_mm)

        # Apply direction-dependent threshold, but keep one shared gain.
        if jaw_direction < 0:
            jaw_threshold_mm = self.jaw_closing_threshold_mm
        else:
            jaw_threshold_mm = self.jaw_opening_threshold_mm

        opening_translation_mm = max(
            0.0,
            backlash_translation_mm - jaw_threshold_mm
        )

        articulation_one_jaw_deg = (
            opening_translation_mm
            * self.jaw_angle_deg_per_mm
        )

        articulation_deg = (
            opening_translation_mm
            * self.between_jaws_deg_per_mm
        )
        articulation_one_jaw = math.radians(articulation_one_jaw_deg)
        articulation = math.radians(articulation_deg)

        l1 = math.tan(bend)
        h = -l1 * math.cos(shaft_roll)
        w = -l1 * math.sin(shaft_roll)
        # Resolve the bend direction into pitch and yaw based 
        # on the current shaft roll angle.
        pitch = -math.atan(h)
        yaw = -math.atan(w)

        # Resolve the bend direction into pitch and yaw based 
        # on the current shaft roll angle.
        return {
            "tip_rotation": tip_rotation,
            "pitch": pitch,
            "yaw": yaw,
            "articulation": articulation, #between_jaws
            "articulation_one_jaw": articulation_one_jaw,
            "shaft_roll": shaft_roll,
            "raw_bend": math.radians(raw_bend_deg),
            "bend": bend,
        }