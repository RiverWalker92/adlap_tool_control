#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Sequence, Dict


@dataclass
class InstrumentDigitalTwinParams:
    bend_factor: float = 3.5 #1.5
    articulation_factor: float = 16.0
    jaw_angle_deg_per_mm: float =  6 #9.167 #18.333
    jaw_opening_threshold_mm: float = 0.1
    jaw_backlash_mm: float = 0.05
    jaw_opening_backlash_mm: float = 0.08
    jaw_closing_backlash_mm: float = 0.05

    shaft_roll_sign: float = 1.0
    bend_sign: float = 1.0
    tip_rotation_sign: float = 1.0
    articulation_sign: float = 1.0


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

    def __init__(self, params: InstrumentDigitalTwinParams):
        self.params = params
        self._jaw_effective_translation_mm = None
    def reset(self):
        self._jaw_effective_translation_mm = None
    def apply_jaw_backlash(self, translation_mm: float) -> float:
        opening_backlash = self.params.jaw_opening_backlash_mm
        closing_backlash = self.params.jaw_closing_backlash_mm

        if self._jaw_effective_translation_mm is None:
            self._jaw_effective_translation_mm = translation_mm
            return translation_mm

        previous_effective = self._jaw_effective_translation_mm

        # Opening direction: more play/slack before jaw follows
        if translation_mm > previous_effective + opening_backlash:
            effective_translation = translation_mm - opening_backlash

        # Closing direction: smaller play
        elif translation_mm < previous_effective - closing_backlash:
            effective_translation = translation_mm + closing_backlash

        else:
            effective_translation = previous_effective

        self._jaw_effective_translation_mm = effective_translation
        return effective_translation
                
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
            self.params.shaft_roll_sign * middle_shaft_rotation_deg
        )

        bend = math.radians(
            self.params.bend_sign
            * middle_outer_relative_rotation_deg
            / self.params.bend_factor
        )

        tip_rotation = math.radians(
            self.params.tip_rotation_sign * inner_shaft_rotation_deg
        )

        # articulation_deg = (
        #     self.params.articulation_sign
        #     * inner_shaft_translation_mm
        #     * self.params.jaw_angle_deg_per_mm
        # )

        # articulation = math.radians(articulation_deg)
        signed_translation_mm = (
            self.params.articulation_sign
            * inner_shaft_translation_mm
        )

        backlash_translation_mm = self.apply_jaw_backlash(signed_translation_mm)

        opening_translation_mm = max(
            0.0,
            backlash_translation_mm - self.params.jaw_opening_threshold_mm
        )

        articulation_deg = (
            opening_translation_mm
            * self.params.jaw_angle_deg_per_mm
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