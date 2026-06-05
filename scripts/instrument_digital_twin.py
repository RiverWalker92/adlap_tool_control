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
    [tip_rotation, pitch, yaw, articulation]
    all in radians
    """

    def __init__(self, config_path="config/tool_params.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["instrument"]

        # self.articulation_factor = config["articulation_factor"]
        self.jaw_angle_deg_per_mm = config["jaw_angle_deg_per_mm"]
        self.jaw_opening_threshold_mm = config["jaw_opening_threshold_mm"]
        self.jaw_opening_backlash_mm = config["jaw_opening_backlash_mm"]
        self.jaw_closing_backlash_mm = config["jaw_closing_backlash_mm"]

        self.bend_factor = config["bend_factor"]
        self.instrument_bend_offset_deg = config["instrument_bend_offset_deg"]
        self.instrument_bend_backlash_positive_deg = config["instrument_bend_backlash_positive_deg"]
        self.instrument_bend_backlash_negative_deg = config["instrument_bend_backlash_negative_deg"]

        self.shaft_roll_sign = config["shaft_roll_sign"]
        self.bend_sign = config["bend_sign"]
        self.tip_rotation_sign = config["tip_rotation_sign"]
        self.articulation_sign = config["articulation_sign"]

        self._jaw_effective_translation_mm = None
        self._effective_bend_deg = None

    def reset(self):
        self._jaw_effective_translation_mm = None
        self._effective_bend_deg = None

    def apply_jaw_backlash(self, translation_mm: float) -> float:
        opening_backlash = self.jaw_opening_backlash_mm
        closing_backlash = self.jaw_closing_backlash_mm

        if self._jaw_effective_translation_mm is None:
            self._jaw_effective_translation_mm = translation_mm
            return translation_mm

        previous_effective = self._jaw_effective_translation_mm

        if translation_mm > previous_effective + opening_backlash:
            effective_translation = translation_mm - opening_backlash

        elif translation_mm < previous_effective - closing_backlash:
            effective_translation = translation_mm + closing_backlash

        else:
            effective_translation = previous_effective

        self._jaw_effective_translation_mm = effective_translation
        return effective_translation

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

    def euler_angles_from_gearbox_output(
        self,
        gearbox_output: Sequence[float],
    ) -> Dict[str, float]:

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

        raw_bend_deg = raw_bend_deg + self.instrument_bend_offset_deg

        bend_deg = self.apply_instrument_bend_backlash(raw_bend_deg)

        bend = math.radians(bend_deg)

        tip_rotation = math.radians(
            self.tip_rotation_sign * inner_shaft_rotation_deg
        )

        # articulation_deg = (
        #     self.params.articulation_sign
        #     * inner_shaft_translation_mm
        #     * self.params.jaw_angle_deg_per_mm
        # )

        # articulation = math.radians(articulation_deg)
        signed_translation_mm = (
            self.articulation_sign
            * inner_shaft_translation_mm
        )

        backlash_translation_mm = self.apply_jaw_backlash(signed_translation_mm)

        opening_translation_mm = max(
            0.0,
            backlash_translation_mm - self.jaw_opening_threshold_mm
        )

        articulation_deg = (
            opening_translation_mm
            * self.jaw_angle_deg_per_mm
        )

        articulation = math.radians(articulation_deg)

        l1 = math.tan(bend)
        h = -l1 * math.cos(shaft_roll)
        w = -l1 * math.sin(shaft_roll)

        pitch = -math.atan(h)
        yaw = -math.atan(w)

        return {
            "tip_rotation": tip_rotation,
            "pitch": pitch,
            "yaw": yaw,
            "articulation": articulation,
            "shaft_roll": shaft_roll,
            "bend": bend,
        }