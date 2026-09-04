#!/usr/bin/env python3
"""Simple Digital Twin residual-band diagnosis for motor_only tests.

Diagnosis is based on healthy residual bands for:

1. Motor position:
       encoder_deviation = measured position - predicted position

2. Motor current:
       filtered_current_deviation =
           filtered measured current - predicted current

For each:
    test_type x motor x signal

the residual is compared with a predefined healthy band.

Short excursions are allowed. A signal is classified as deviating when either:

1. More than MAX_OUTSIDE_FRACTION of its samples are outside the healthy band.
2. The residual remains continuously outside the band for at least
   the signal-specific duration threshold:
   - position: POSITION_MAX_CONTINUOUS_OUTSIDE_S
   - current: CURRENT_MAX_CONTINUOUS_OUTSIDE_S

The overall motor/test result is deviating when position OR current deviates.
"""
# TODO: add motor-only m0 dictiorary in dof3
# TODO: add localisation past
# TODO: add instrument output deviation 

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import subprocess

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DETECTOR_NAME = "digital_twin_residual_band_diagnostics"
DETECTOR_VERSION = "2.1.0"

MOTOR_ONLY_BASELINE_NAME = "motor_only_residual_bands_healthy_v1"
GEARBOX_BASELINE_NAME = "gearbox_only_residual_bands_provisional_v1"
FULL_SETUP_BASELINE_NAME = "full_setup_residual_bands_healthy_v1"

# Detection tolerance.
MAX_OUTSIDE_FRACTION = 0.10

# POSITION_MAX_CONTINUOUS_OUTSIDE_S = 0.75  # 750 ms
CURRENT_MAX_CONTINUOUS_OUTSIDE_S = 0.10   # 100 ms

# ---------------------------------------------------------------------------
# Motors-Gearbox configuration
# ---------------------------------------------------------------------------

DOF_RELEVANT_MOTORS = {
    1: [1, 2],
    2: [1, 2],
    3: [0, 3],
    4: [0, 3],
}

# ---------------------------------------------------------------------------
# Motor only Position healthy bands
# ---------------------------------------------------------------------------

# Residual ranges [lower, upper] in encoder pulses.
#
# Structure:
# test_type -> motor -> (lower, upper)

MOTOR_ONLY_POSITION_BANDS = {
    "back_and_forth_large": {
        0: (-75.0, 20.0),
        1: (-40.0, 35.0),
        2: (-40.0, 35.0),
        3: (-15.0, 40.0),
    },
    "back_and_forth_medium": {
        0: (-30.0, 15.0),
        1: (-20.0, 15.0),
        2: (-25.0, 20.0),
        3: (-10.0, 20.0),
    },
    "back_and_forth_small": {
        0: (-20.0, 10.0),
        1: (-15.0, 10.0),
        2: (-15.0, 15.0),
        3: (-10.0, 15.0),
    },
    "cyclic_medium": {
        0: (-25.0, 20.0),
        1: (-30.0, 30.0),
        2: (-25.0, 25.0),
        3: (-15.0, 25.0),
    },
    "full_turn_single_3x": {
        0: (-70.0, 75.0),
        1: (-100.0, 95.0),
        2: (-80.0, 120.0),
        3: (-50.0, 90.0),
    },
    "reversal_medium": {
        0: (-35.0, 20.0),
        1: (-25.0, 25.0),
        2: (-25.0, 25.0),
        3: (-15.0, 30.0),
    },
}

# ---------------------------------------------------------------------------
# Motor only Current healthy bands
# ---------------------------------------------------------------------------

# Residual ranges [lower, upper] in mA.
#
# These bands are deliberately rounded and relatively tolerant because
# healthy current residuals contain short transient peaks.
#
# The 10 % / 100 ms rule prevents isolated short peaks from immediately
# producing a fault classification.

MOTOR_ONLY_CURRENT_BANDS = {
    "back_and_forth_large": {
        0: (-15.0, 15.0),
        1: (-20.0, 15.0),
        2: (-15.0, 15.0),
        3: (-10.0, 15.0),
    },
    "back_and_forth_medium": {
        0: (-10.0, 15.0),
        1: (-15.0, 10.0),
        2: (-15.0, 10.0),
        3: (-5.0, 15.0),
    },
    "back_and_forth_small": {
        0: (-5.0, 10.0),
        1: (-5.0, 10.0),
        2: (-10.0, 5.0),
        3: (-5.0, 5.0),
    },
    "cyclic_medium": {
        0: (-20.0, 20.0),
        1: (-20.0, 20.0),
        2: (-20.0, 20.0),
        3: (-15.0, 20.0),
    },
    "full_turn_single_3x": {
        0: (-10.0, 15.0),
        1: (-20.0, 15.0),
        2: (-10.0, 15.0),
        3: (-10.0, 15.0),
    },
    "reversal_medium": {
        0: (-15.0, 15.0),
        1: (-15.0, 15.0),
        2: (-15.0, 15.0),
        3: (-10.0, 15.0),
    },
}
#----------------------------------------------------------------------------
# Motor-only DOF pattern healthy bands
#----------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Motor-only DOF pattern healthy bands
#
# Structure:
# dof -> condition -> motor -> (lower, upper)
#
# condition:
#   waveform_frequency_range
# ---------------------------------------------------------------------------

MOTOR_ONLY_DOF_POSITION_BANDS = {
    1: {
        "sinusoid_f0p5_r0p5": {
            1: (-40.0, 45.0),
            2: (-35.0, 40.0),
        },
        "sinusoid_f1p0_r1p0": {
            1: (-55.0, 60.0),
            2: (-50.0, 55.0),
        },
        "triangle_f0p5_r0p5": {
            1: (-50.0, 55.0),
            2: (-45.0, 50.0),
        },
        "triangle_f1p0_r1p0": {
            1: (-70.0, 75.0),
            2: (-65.0, 70.0),
        },
    },

    2: {
        "sinusoid_f0p5_r0p5": {
            1: (-45.0, 50.0),
            2: (-40.0, 45.0),
        },
        "sinusoid_f1p0_r1p0": {
            1: (-65.0, 70.0),
            2: (-60.0, 65.0),
        },
        "triangle_f0p5_r0p5": {
            1: (-55.0, 60.0),
            2: (-50.0, 55.0),
        },
        "triangle_f1p0_r1p0": {
            1: (-80.0, 85.0),
            2: (-75.0, 80.0),
        },
    },

    3: {
        "sinusoid_f0p5_r0p5": {
            0: (-40.0, 45.0),
        },
        "sinusoid_f1p0_r1p0": {
            0: (-60.0, 65.0),
        },
        "triangle_f0p5_r0p5": {
            0: (-50.0, 55.0),
        },
        "triangle_f1p0_r1p0": {
            0: (-75.0, 80.0),
        },
    },

    4: {
        "sinusoid_f0p5_r0p5": {
            0: (-45.0, 50.0),
            3: (-40.0, 45.0),
        },
        "sinusoid_f1p0_r1p0": {
            0: (-65.0, 70.0),
            3: (-60.0, 65.0),
        },
        "triangle_f0p5_r0p5": {
            0: (-55.0, 60.0),
            3: (-50.0, 55.0),
        },
        "triangle_f1p0_r1p0": {
            0: (-80.0, 85.0),
            3: (-75.0, 80.0),
        },
    },
}

MOTOR_ONLY_DOF_CURRENT_BANDS = {
    1: {
        "sinusoid_f0p5_r0p5": {
            1: (-10.0, 15.0),
            2: (-10.0, 15.0),
        },
        "sinusoid_f1p0_r1p0": {
            1: (-15.0, 20.0),
            2: (-15.0, 20.0),
        },
        "triangle_f0p5_r0p5": {
            1: (-15.0, 20.0),
            2: (-15.0, 20.0),
        },
        "triangle_f1p0_r1p0": {
            1: (-20.0, 25.0),
            2: (-20.0, 25.0),
        },
    },

    2: {
        "sinusoid_f0p5_r0p5": {
            1: (-10.0, 15.0),
            2: (-10.0, 15.0),
        },
        "sinusoid_f1p0_r1p0": {
            1: (-15.0, 20.0),
            2: (-15.0, 20.0),
        },
        "triangle_f0p5_r0p5": {
            1: (-15.0, 20.0),
            2: (-15.0, 20.0),
        },
        "triangle_f1p0_r1p0": {
            1: (-20.0, 25.0),
            2: (-20.0, 25.0),
        },
    },

    3: {
        "sinusoid_f0p5_r0p5": {
            0: (-10.0, 15.0),
        },
        "sinusoid_f1p0_r1p0": {
            0: (-15.0, 20.0),
        },
        "triangle_f0p5_r0p5": {
            0: (-15.0, 20.0),
        },
        "triangle_f1p0_r1p0": {
            0: (-20.0, 25.0),
        },
    },

    4: {
        "sinusoid_f0p5_r0p5": {
            0: (-10.0, 15.0),
            3: (-10.0, 15.0),
        },
        "sinusoid_f1p0_r1p0": {
            0: (-15.0, 20.0),
            3: (-15.0, 20.0),
        },
        "triangle_f0p5_r0p5": {
            0: (-15.0, 20.0),
            3: (-15.0, 20.0),
        },
        "triangle_f1p0_r1p0": {
            0: (-20.0, 25.0),
            3: (-20.0, 25.0),
        },
    },
}


MOTOR_ONLY_DOF_CURRENT_DURATION_LIMITS = {
    1: {
        "sinusoid_f0p5_r0p5": {
            1: 0.20,
            2: 0.20,
        },
        "sinusoid_f1p0_r1p0": {
            1: 0.15,
            2: 0.15,
        },
        "triangle_f0p5_r0p5": {
            1: 0.25,
            2: 0.25,
        },
        "triangle_f1p0_r1p0": {
            1: 0.20,
            2: 0.20,
        },
    },

    2: {
        "sinusoid_f0p5_r0p5": {
            1: 0.20,
            2: 0.20,
        },
        "sinusoid_f1p0_r1p0": {
            1: 0.15,
            2: 0.15,
        },
        "triangle_f0p5_r0p5": {
            1: 0.25,
            2: 0.25,
        },
        "triangle_f1p0_r1p0": {
            1: 0.20,
            2: 0.20,
        },
    },

    3: {
        "sinusoid_f0p5_r0p5": {
            0: 0.20,
        },
        "sinusoid_f1p0_r1p0": {
            0: 0.15,
        },
        "triangle_f0p5_r0p5": {
            0: 0.25,
        },
        "triangle_f1p0_r1p0": {
            0: 0.20,
        },
    },

    4: {
        "sinusoid_f0p5_r0p5": {
            0: 0.20,
            3: 0.20,
        },
        "sinusoid_f1p0_r1p0": {
            0: 0.15,
            3: 0.15,
        },
        "triangle_f0p5_r0p5": {
            0: 0.25,
            3: 0.25,
        },
        "triangle_f1p0_r1p0": {
            0: 0.20,
            3: 0.20,
        },
    },
}
# ---------------------------------------------------------------------------
# Gearbox-only healthy bands
# ---------------------------------------------------------------------------

# IMPORTANT:
# These values still need to be filled from healthy gearbox-only data.
#
# Structure:
# DOF -> motor -> (lower, upper)

GEARBOX_POSITION_BANDS = {
    1: {
        1: (-150.0, 150.0),
        2: (-150.0, 150.0),
    },
    2: {
        1: (-150.0, 150.0),
        2: (-150.0, 150.0),
    },
    3: {
        0: (-150.0, 150.0),
    },
    4: {
        0: (-150.0, 150.0),
        3: (-150.0, 150.0),
    },
}

GEARBOX_CURRENT_BANDS = {
    1: {
        1: (-120.0, 120.0),
        2: (-120.0, 120.0),
    },
    2: {
        1: (-120.0, 120.0),
        2: (-120.0, 120.0),
    },
    3: {
        0: (-120.0, 120.0),
    },
    4: {
        0: (-120.0, 120.0),
        3: (-120.0, 120.0),
    },
}

# ---------------------------------------------------------------------------
# Full setup healthy position bands
# ---------------------------------------------------------------------------
FULL_SETUP_DOF_POSITION_BANDS = {
    1: {
        "sinusoid_f0p5_r1p0": {
            1: (-20.0, 20.0),
            2: (-20.0, 20.0),
        },
        "sinusoid_f1p0_r1p0": {
            1: (-30.0, 75.0),
            2: (-50.0, 125.0),
        },
        "triangle_f0p5_r1p0": {
            1: (-20.0, 15.0),
            2: (-20.0, 20.0),
        },
        "triangle_f1p0_r1p0": {
            1: (-30.0, 25.0),
            2: (-25.0, 30.0),
        },
    },

    2: {
        "sinusoid_f0p5_r0p2": {
            1: (-60.0, 80.0),
            2: (-90.0, 35.0),
        },
        "sinusoid_f0p5_r0p5": {
            1: (-15.0, 15.0),
            2: (-15.0, 15.0),
        },
        "sinusoid_f0p5_r1p0": {
            1: (-15.0, 15.0),
            2: (-20.0, 15.0),
        },

        "sinusoid_f1p0_r0p2": {
            1: (-20.0, 20.0),
            2: (-15.0, 20.0),
        },
        "sinusoid_f1p0_r0p5": {
            1: (-20.0, 20.0),
            2: (-20.0, 20.0),
        },
        "sinusoid_f1p0_r1p0": {
            1: (-20.0, 15.0),
            2: (-25.0, 25.0),
        },

        "sinusoid_f2p0_r0p2": {
            1: (-15.0, 20.0),
            2: (-15.0, 20.0),
        },
        "sinusoid_f2p0_r0p5": {
            1: (-20.0, 15.0),
            2: (-30.0, 25.0),
        },
        "sinusoid_f2p0_r1p0": {
            1: (-20.0, 15.0),
            2: (-20.0, 20.0),
        },

        "triangle_f0p5_r0p2": {
            1: (-15.0, 15.0),
            2: (-15.0, 15.0),
        },
        "triangle_f0p5_r0p5": {
            1: (-15.0, 15.0),
            2: (-10.0, 15.0),
        },
        "triangle_f0p5_r1p0": {
            1: (-20.0, 15.0),
            2: (-10.0, 20.0),
        },

        "triangle_f1p0_r0p2": {
            1: (-20.0, 20.0),
            2: (-15.0, 20.0),
        },
        "triangle_f1p0_r0p5": {
            1: (-15.0, 15.0),
            2: (-20.0, 15.0),
        },
        "triangle_f1p0_r1p0": {
            1: (-15.0, 15.0),
            2: (-20.0, 20.0),
        },

        "triangle_f2p0_r0p2": {
            1: (-25.0, 20.0),
            2: (-20.0, 20.0),
        },
        "triangle_f2p0_r0p5": {
            1: (-20.0, 25.0),
            2: (-25.0, 30.0),
        },
        "triangle_f2p0_r1p0": {
            1: (-15.0, 15.0),
            2: (-25.0, 35.0),
        },
    },

    3: {
        "sinusoid_f0p5_r1p0": {
            0: (-25.0, 25.0),
            3: (-25.0, 20.0),
        },
        "sinusoid_f1p0_r1p0": {
            0: (-35.0, 35.0),
            3: (-35.0, 35.0),
        },
        "triangle_f0p5_r1p0": {
            0: (-20.0, 20.0),
            3: (-20.0, 20.0),
        },
        "triangle_f1p0_r1p0": {
            0: (-35.0, 35.0),
            3: (-30.0, 35.0),
        },
    },

    4: {
        "sinusoid_f0p5_r0p5": {
            0: (-10.0, 10.0),
            3: (-10.0, 15.0),
        },
        "sinusoid_f1p0_r0p5": {
            0: (-10.0, 10.0),
            3: (-15.0, 15.0),
        },
        "sinusoid_f2p0_r0p5": {
            0: (-15.0, 15.0),
            3: (-70.0, 60.0),
        },

        "triangle_f0p5_r0p5": {
            0: (-15.0, 10.0),
            3: (-15.0, 10.0),
        },
        "triangle_f1p0_r0p5": {
            0: (-15.0, 10.0),
            3: (-20.0, 15.0),
        },
        "triangle_f2p0_r0p5": {
            0: (-15.0, 15.0),
            3: (-30.0, 35.0),
        },
    },
}
# -----------------------------------------------------------------------------
# Full setup healthy current bands
# -----------------------------------------------------------------------------
FULL_SETUP_DOF_CURRENT_BANDS = {
    1: {
        "sinusoid_f0p5_r1p0": {
            1: (-30.0, 20.0),
            2: (-25.0, 35.0),
        },
        "sinusoid_f1p0_r1p0": {
            1: (-25.0, 30.0),
            2: (-25.0, 25.0),
        },
        "triangle_f0p5_r1p0": {
            1: (-25.0, 20.0),
            2: (-20.0, 35.0),
        },
        "triangle_f1p0_r1p0": {
            1: (-25.0, 20.0),
            2: (-25.0, 35.0),
        },
    },

    2: {
        "sinusoid_f0p5_r0p2": {
            1: (-25.0, 20.0),
            2: (-60.0, 60.0),
        },
        "sinusoid_f0p5_r0p5": {
            1: (-20.0, 15.0),
            2: (-60.0, 50.0),
        },
        "sinusoid_f0p5_r1p0": {
            1: (-15.0, 15.0),
            2: (-40.0, 60.0),
        },

        "sinusoid_f1p0_r0p2": {
            1: (-25.0, 35.0),
            2: (-55.0, 40.0),
        },
        "sinusoid_f1p0_r0p5": {
            1: (-25.0, 30.0),
            2: (-40.0, 40.0),
        },
        "sinusoid_f1p0_r1p0": {
            1: (-35.0, 30.0),
            2: (-55.0, 75.0),
        },

        "sinusoid_f2p0_r0p2": {
            1: (-25.0, 25.0),
            2: (-40.0, 40.0),
        },
        "sinusoid_f2p0_r0p5": {
            1: (-25.0, 30.0),
            2: (-45.0, 35.0),
        },
        "sinusoid_f2p0_r1p0": {
            1: (-35.0, 45.0),
            2: (-45.0, 40.0),
        },

        "triangle_f0p5_r0p2": {
            1: (-15.0, 15.0),
            2: (-50.0, 40.0),
        },
        "triangle_f0p5_r0p5": {
            1: (-20.0, 15.0),
            2: (-55.0, 45.0),
        },
        "triangle_f0p5_r1p0": {
            1: (-20.0, 20.0),
            2: (-40.0, 50.0),
        },

        "triangle_f1p0_r0p2": {
            1: (-30.0, 35.0),
            2: (-60.0, 45.0),
        },
        "triangle_f1p0_r0p5": {
            1: (-20.0, 20.0),
            2: (-40.0, 35.0),
        },
        "triangle_f1p0_r1p0": {
            1: (-20.0, 25.0),
            2: (-30.0, 35.0),
        },

        "triangle_f2p0_r0p2": {
            1: (-35.0, 45.0),
            2: (-35.0, 40.0),
        },
        "triangle_f2p0_r0p5": {
            1: (-45.0, 65.0),
            2: (-35.0, 55.0),
        },
        "triangle_f2p0_r1p0": {
            1: (-35.0, 40.0),
            2: (-40.0, 35.0),
        },
    },

    3: {
        "sinusoid_f0p5_r1p0": {
            0: (-30.0, 25.0),
            3: (-30.0, 35.0),
        },
        "sinusoid_f1p0_r1p0": {
            0: (-25.0, 25.0),
            3: (-20.0, 25.0),
        },
        "triangle_f0p5_r1p0": {
            0: (-25.0, 20.0),
            3: (-20.0, 20.0),
        },
        "triangle_f1p0_r1p0": {
            0: (-25.0, 25.0),
            3: (-20.0, 25.0),
        },
    },

    4: {
        "sinusoid_f0p5_r0p5": {
            0: (-10.0, 10.0),
            3: (-30.0, 30.0),
        },
        "sinusoid_f1p0_r0p5": {
            0: (-10.0, 10.0),
            3: (-35.0, 25.0),
        },
        "sinusoid_f2p0_r0p5": {
            0: (-20.0, 25.0),
            3: (-75.0, 30.0),
        },

        "triangle_f0p5_r0p5": {
            0: (-10.0, 10.0),
            3: (-30.0, 20.0),
        },
        "triangle_f1p0_r0p5": {
            0: (-10.0, 10.0),
            3: (-40.0, 20.0),
        },
        "triangle_f2p0_r0p5": {
            0: (-20.0, 20.0),
            3: (-40.0, 20.0),
        },
    },
}
FULL_SETUP_DOF_CURRENT_DURATION_LIMITS = {
    1: {
        "sinusoid_f0p5_r1p0": {
            1: 0.15,
            2: 0.15,
        },
        "sinusoid_f1p0_r1p0": {
            1: 0.15,
            2: 0.15,
        },
        "triangle_f0p5_r1p0": {
            1: 0.10,
            2: 0.20,
        },
        "triangle_f1p0_r1p0": {
            1: 0.20,
            2: 0.15,
        },
    },

    2: {
        "sinusoid_f0p5_r0p2": {
            1: 0.20,
            2: 0.15,
        },
        "sinusoid_f0p5_r0p5": {
            1: 0.15,
            2: 0.10,
        },
        "sinusoid_f0p5_r1p0": {
            1: 0.15,
            2: 0.15,
        },

        "sinusoid_f1p0_r0p2": {
            1: 0.15,
            2: 0.10,
        },
        "sinusoid_f1p0_r0p5": {
            1: 0.10,
            2: 0.10,
        },
        "sinusoid_f1p0_r1p0": {
            1: 0.15,
            2: 0.10,
        },

        "sinusoid_f2p0_r0p2": {
            1: 0.10,
            2: 0.10,
        },
        "sinusoid_f2p0_r0p5": {
            1: 0.10,
            2: 0.10,
        },
        "sinusoid_f2p0_r1p0": {
            1: 0.10,
            2: 0.10,
        },

        "triangle_f0p5_r0p2": {
            1: 0.15,
            2: 0.15,
        },
        "triangle_f0p5_r0p5": {
            1: 0.15,
            2: 0.10,
        },
        "triangle_f0p5_r1p0": {
            1: 0.15,
            2: 0.10,
        },

        "triangle_f1p0_r0p2": {
            1: 0.10,
            2: 0.10,
        },
        "triangle_f1p0_r0p5": {
            1: 0.15,
            2: 0.10,
        },
        "triangle_f1p0_r1p0": {
            1: 0.15,
            2: 0.10,
        },

        "triangle_f2p0_r0p2": {
            1: 0.10,
            2: 0.10,
        },
        "triangle_f2p0_r0p5": {
            1: 0.10,
            2: 0.10,
        },
        "triangle_f2p0_r1p0": {
            1: 0.10,
            2: 0.10,
        },
    },

    3: {
        "sinusoid_f0p5_r1p0": {
            0: 0.15,
            3: 0.50,
        },
        "sinusoid_f1p0_r1p0": {
            0: 0.20,
            3: 0.20,
        },
        "triangle_f0p5_r1p0": {
            0: 0.50,
            3: 0.25,
        },
        "triangle_f1p0_r1p0": {
            0: 0.15,
            3: 0.10,
        },
    },

    4: {
        "sinusoid_f0p5_r0p5": {
            0: 0.15,
            3: 0.15,
        },
        "sinusoid_f1p0_r0p5": {
            0: 0.20,
            3: 0.10,
        },
        "sinusoid_f2p0_r0p5": {
            0: 0.15,
            3: 0.10,
        },
        "triangle_f0p5_r0p5": {
            0: 0.20,
            3: 0.10,
        },
        "triangle_f1p0_r0p5": {
            0: 0.15,
            3: 0.10,
        },
        "triangle_f2p0_r0p5": {
            0: 0.15,
            3: 0.10,
        },
    },
}
# ---------------------------------------------------------------------------
# Pause baseline
# ---------------------------------------------------------------------------
FULL_SETUP_DOF_PAUSE_POSITION_BANDS = {
    1: {
        1: (-10.0, 10.0),
        2: (-10.0, 10.0),
    },
    2: {
        1: (-30.0, 10.0),
        2: (-15.0, 10.0),
    },
    3: {
        0: (-15.0, 10.0),
        3: (-10.0, 10.0),
    },
    4: {
        0: (-10.0, 10.0),
        3: (-10.0, 10.0),
    },
}

FULL_SETUP_DOF_PAUSE_CURRENT_BANDS = {
    1: {
        1: (-10.0, 10.0),
        2: (-10.0, 10.0),
    },
    2: {
        1: (-5.0, 10.0),
        2: (-10.0, 10.0),
    },
    3: {
        0: (-10.0, 10.0),
        3: (-10.0, 10.0),
    },
    4: {
        0: (-10.0, 10.0),
        3: (-10.0, 10.0),
    },
}
# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

    if not rows:
        raise ValueError("Replay contains no rows.")

    return rows


def validate_replay(rows: list[dict[str, Any]]) -> None:
    required = {
        "test_type",
        "task_label",
        "coupling_mode",
        "segment_time",

        "offline_commanded_target",

        "offline_predicted_positions",
        "measured_positions",
        "encoder_deviation",

        "offline_predicted_currents",
        "filtered_measured_currents",
        "filtered_current_deviation",
    }

    missing = sorted(required - rows[0].keys())

    if missing:
        raise ValueError(
            "Replay misses required fields: "
            + ", ".join(missing)
        )

    coupling_modes = {
        str(row["coupling_mode"])
        for row in rows
    }

    supported_modes = {
        "motor_only",
        "gearbox_only",
        "full_setup",
    }

    if (
        len(coupling_modes) != 1
        or not coupling_modes.issubset(supported_modes)
    ):
        raise ValueError(
            "This detector supports coupling_mode=motor_only, "
            "gearbox_only and full_setup. "
            f"Found: {sorted(coupling_modes)}"
        )

def detect_dof(
    replay_path: Path,
) -> int:

    name = replay_path.name.lower()

    for dof in range(1, 5):

        if f"dof{dof}" in name:
            return dof

    raise ValueError(
        "Could not determine DOF from replay filename."
    )

def detect_dof_condition(
    row: dict[str, Any],
) -> str:

    condition = str(
        row.get("sequence_condition", "")
    ).strip().lower()

    if not condition or condition == "none":
        raise ValueError(
            "Missing sequence_condition in replay."
        )

    if condition == "pause":
        raise ValueError(
            "Sequence pause is not a diagnostic motion condition."
        )

    return condition

def analyse_pause_rows(
    rows: list[dict[str, Any]],
    motors: list[int],
    coupling_mode: str,
    dof: int | None,
) -> dict[str, Any]:

    pause_rows = [
        row
        for row in rows
        if str(
            row.get("sequence_condition", "")
        ).strip().lower() == "pause"
    ]

    result = {
        "status": "not_evaluated",
        "samples": len(pause_rows),
        "affected_motors": [],
        "motor_results": {},
    }

    if not pause_rows:
        return result

    # At the moment healthy pause bands are only
    # defined for full_setup.
    evaluate_pause = (
        coupling_mode == "full_setup"
        and dof is not None
    )

    any_evaluated = False

    for motor in motors:

        position_values = []
        current_values = []

        for row in pause_rows:

            try:
                position = (
                    row["encoder_deviation"][motor]
                )

                current = (
                    row["filtered_current_deviation"][motor]
                )

                if (
                    position is None
                    or current is None
                ):
                    continue

                position = float(position)
                current = float(current)

                if (
                    not np.isfinite(position)
                    or not np.isfinite(current)
                ):
                    continue

                position_values.append(
                    position
                )

                current_values.append(
                    current
                )

            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ):
                continue

        if not position_values:
            continue

        position_values = np.asarray(
            position_values,
            dtype=float,
        )

        current_values = np.asarray(
            current_values,
            dtype=float,
        )

        # -----------------------------------------------------------
        # If no pause baseline exists for this configuration,
        # keep descriptive statistics only.
        # -----------------------------------------------------------

        if not evaluate_pause:

            result["motor_results"][
                f"motor_{motor}"
            ] = {
                "status": "not_evaluated",

                "position": {
                    "mean_residual":
                        float(np.mean(position_values)),
                    "minimum_residual":
                        float(np.min(position_values)),
                    "maximum_residual":
                        float(np.max(position_values)),
                },

                "current": {
                    "mean_residual":
                        float(np.mean(current_values)),
                    "minimum_residual":
                        float(np.min(current_values)),
                    "maximum_residual":
                        float(np.max(current_values)),
                },
            }

            continue

        # -----------------------------------------------------------
        # Full-setup pause healthy bands
        # -----------------------------------------------------------

        try:
            position_lower, position_upper = (
                FULL_SETUP_DOF_PAUSE_POSITION_BANDS[
                    dof
                ][motor]
            )

            current_lower, current_upper = (
                FULL_SETUP_DOF_PAUSE_CURRENT_BANDS[
                    dof
                ][motor]
            )

        except KeyError:

            result["motor_results"][
                f"motor_{motor}"
            ] = {
                "status": "not_evaluated",
            }

            continue

        any_evaluated = True

        # -----------------------------------------------------------
        # Position pause baseline
        # -----------------------------------------------------------

        position_outside = (
            (position_values < position_lower)
            |
            (position_values > position_upper)
        )

        position_outside_count = int(
            np.sum(position_outside)
        )

        position_fraction = (
            position_outside_count
            / len(position_values)
        )

        position_deviating = (
            position_fraction
            > MAX_OUTSIDE_FRACTION
        )

        # -----------------------------------------------------------
        # Current pause baseline
        # -----------------------------------------------------------

        current_outside = (
            (current_values < current_lower)
            |
            (current_values > current_upper)
        )

        current_outside_count = int(
            np.sum(current_outside)
        )

        current_fraction = (
            current_outside_count
            / len(current_values)
        )

        current_deviating = (
            current_fraction
            > MAX_OUTSIDE_FRACTION
        )

        motor_deviating = (
            position_deviating
            or current_deviating
        )

        if motor_deviating:
            result["affected_motors"].append(
                motor
            )

        result["motor_results"][
            f"motor_{motor}"
        ] = {
            "status":
                "deviating"
                if motor_deviating
                else "healthy",

            "position": {
                "status":
                    "deviating"
                    if position_deviating
                    else "healthy",

                "healthy_band": {
                    "lower": position_lower,
                    "upper": position_upper,
                    "unit": "pulses",
                },

                "samples":
                    len(position_values),

                "samples_outside_band":
                    position_outside_count,

                "fraction_outside_band":
                    position_fraction,

                "percentage_outside_band":
                    100.0 * position_fraction,

                "mean_residual":
                    float(np.mean(position_values)),

                "minimum_residual":
                    float(np.min(position_values)),

                "maximum_residual":
                    float(np.max(position_values)),

                "fraction_rule_triggered":
                    position_deviating,
            },

            "current": {
                "status":
                    "deviating"
                    if current_deviating
                    else "healthy",

                "healthy_band": {
                    "lower": current_lower,
                    "upper": current_upper,
                    "unit": "mA",
                },

                "samples":
                    len(current_values),

                "samples_outside_band":
                    current_outside_count,

                "fraction_outside_band":
                    current_fraction,

                "percentage_outside_band":
                    100.0 * current_fraction,

                "mean_residual":
                    float(np.mean(current_values)),

                "minimum_residual":
                    float(np.min(current_values)),

                "maximum_residual":
                    float(np.max(current_values)),

                "fraction_rule_triggered":
                    current_deviating,
            },
        }

    if any_evaluated:

        result["status"] = (
            "deviating"
            if result["affected_motors"]
            else "healthy"
        )

    return result

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def longest_true_duration(
    mask: np.ndarray,
    times: np.ndarray,
) -> float:
    """Return longest continuous True period in seconds."""

    if len(mask) == 0:
        return 0.0

    if len(times) > 1:
        sample_period = float(
            np.median(np.diff(times))
        )
    else:
        sample_period = 0.0

    longest = 0.0
    start_index = None

    for i, value in enumerate(mask):

        if value and start_index is None:
            start_index = i

        if not value and start_index is not None:
            end_index = i - 1

            duration = (
                float(
                    times[end_index]
                    - times[start_index]
                )
                + sample_period
            )

            longest = max(
                longest,
                duration,
            )

            start_index = None

    if start_index is not None:
        end_index = len(mask) - 1

        duration = (
            float(
                times[end_index]
                - times[start_index]
            )
            + sample_period
        )

        longest = max(
            longest,
            duration,
        )

    return longest


def analyse_signal(
    residual: np.ndarray,
    times: np.ndarray,
    task_labels: np.ndarray,
    lower: float,
    upper: float,
    unit: str,
    max_continuous_outside_s: float | None,
) -> dict[str, Any]:
    """Analyse one residual signal against its healthy band."""

    outside = (
        (residual < lower)
        | (residual > upper)
    )

    sample_count = len(residual)
    outside_count = int(np.sum(outside))

    outside_fraction = (
        outside_count / sample_count
        if sample_count
        else 0.0
    )

    # ---------------------------------------------------------------
    # Longest continuous excursion:
    # calculate separately for every task_label
    # ---------------------------------------------------------------

    longest_outside_s = 0.0

    for task_label in np.unique(task_labels):

        task_mask = (
            task_labels == task_label
        )

        task_outside = outside[
            task_mask
        ]

        task_times = times[
            task_mask
        ]

        if len(task_times) == 0:
            continue

        order = np.argsort(
            task_times
        )

        task_outside = task_outside[
            order
        ]

        task_times = task_times[
            order
        ]

        task_duration = longest_true_duration(
            task_outside,
            task_times,
        )

        longest_outside_s = max(
            longest_outside_s,
            task_duration,
        )

    exceeds_fraction = (
        outside_fraction > MAX_OUTSIDE_FRACTION
    )

    exceeds_duration = (
        max_continuous_outside_s is not None
        and longest_outside_s >= max_continuous_outside_s
    )

    deviating = (
        exceeds_fraction
        or exceeds_duration
    )

    return {
        "status":
            "deviating"
            if deviating
            else "healthy",

        "healthy_band": {
            "lower": lower,
            "upper": upper,
            "unit": unit,
        },

        "samples":
            sample_count,

        "samples_outside_band":
            outside_count,

        "fraction_outside_band":
            outside_fraction,

        "percentage_outside_band":
            100.0 * outside_fraction,

        "maximum_continuous_outside_s":
            max_continuous_outside_s,

        "longest_continuous_outside_s":
            longest_outside_s,

        "minimum_residual":
            float(np.min(residual)),

        "maximum_residual":
            float(np.max(residual)),

        "fraction_rule_triggered":
            exceeds_fraction,

        "duration_rule_triggered":
            exceeds_duration,
    }

def get_healthy_bands(
    coupling_mode: str,
    motor: int,
    test_type: str,
    dof: int | None,
    condition: str | None = None,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
]:

    # ---------------------------------------------------------------
    # Motor-only baseline
    # ---------------------------------------------------------------

    if coupling_mode == "motor_only":

        if dof is not None:

            if condition is None:
                raise ValueError(
                    "DOF condition is required for motor-only DOF diagnosis."
                )

            try:
                position_band = (
                    MOTOR_ONLY_DOF_POSITION_BANDS[
                        dof
                    ][condition][motor]
                )

                current_band = (
                    MOTOR_ONLY_DOF_CURRENT_BANDS[
                        dof
                    ][condition][motor]
                )

            except KeyError as exc:
                raise ValueError(
                    f"No healthy bands defined for "
                    f"DOF{dof}, condition '{condition}', "
                    f"motor {motor}."
                ) from exc

            return (
                position_band,
                current_band,
            )

        position_band = (
            MOTOR_ONLY_POSITION_BANDS[
                test_type
            ][motor]
        )

        current_band = (
            MOTOR_ONLY_CURRENT_BANDS[
                test_type
            ][motor]
        )

        return (
            position_band,
            current_band,
        )
    # ---------------------------------------------------------------
    # Gearbox-only baseline
    # ---------------------------------------------------------------

    if coupling_mode == "gearbox_only":

        if dof is None:
            raise ValueError(
                "DOF is required for gearbox_only diagnosis."
            )

        position_band = (
            GEARBOX_POSITION_BANDS[
                dof
            ][motor]
        )

        current_band = (
            GEARBOX_CURRENT_BANDS[
                dof
            ][motor]
        )

        if (
            position_band is None
            or current_band is None
        ):
            raise ValueError(
                f"No gearbox healthy band defined "
                f"for DOF{dof}, motor {motor}."
            )

        return (
            position_band,
            current_band,
        )
    # ---------------------------------------------------------------
    # Full-setup baseline
    # ---------------------------------------------------------------

    if coupling_mode == "full_setup":

        if dof is None:
            raise ValueError(
                "DOF is required for full_setup diagnosis."
            )

        if condition is None:
            raise ValueError(
                "DOF condition is required for full_setup diagnosis."
            )

        try:
            position_band = (
                FULL_SETUP_DOF_POSITION_BANDS[
                    dof
                ][condition][motor]
            )

            current_band = (
                FULL_SETUP_DOF_CURRENT_BANDS[
                    dof
                ][condition][motor]
            )

        except KeyError as exc:
            raise ValueError(
                f"No full-setup healthy bands defined for "
                f"DOF{dof}, condition '{condition}', "
                f"motor {motor}."
            ) from exc

        return (
            position_band,
            current_band,
        )

    raise ValueError(
        f"Unsupported coupling mode: {coupling_mode}"
    )

def get_current_duration_limit(
    coupling_mode: str,
    motor: int,
    dof: int | None,
    condition: str | None,
) -> float | None:

    # Only enable condition-specific duration detection
    # for motor-only DOF pattern tests for now.
    if (
        coupling_mode in (
            "motor_only",
            "full_setup",
        )
        and dof is not None
        and condition is not None
    ):

        try:

            if coupling_mode == "full_setup":
                return (
                    FULL_SETUP_DOF_CURRENT_DURATION_LIMITS[
                        dof
                    ][condition][motor]
                )

            return (
                MOTOR_ONLY_DOF_CURRENT_DURATION_LIMITS[
                    dof
                ][condition][motor]
            )

        except KeyError as exc:
            raise ValueError(
                f"No current duration limit defined for "
                f"{coupling_mode}, DOF{dof}, "
                f"condition '{condition}', motor {motor}."
            ) from exc

    return None

def analyse_motor_test(
    rows: list[dict[str, Any]],
    motor: int,
    test_type: str,
    coupling_mode: str,
    dof: int | None = None,
    condition: str | None = None,
) -> dict[str, Any]:
    
    valid_rows = []

    for row in rows:
        try:
            position_residual = row["encoder_deviation"][motor]
            current_residual = row["filtered_current_deviation"][motor]

            if position_residual is None or current_residual is None:
                continue

            if not np.isfinite(float(position_residual)):
                continue

            if not np.isfinite(float(current_residual)):
                continue

            valid_rows.append(row)

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ):
            continue

    if not valid_rows:
        raise ValueError(
            f"No valid residual samples available for "
            f"motor {motor}, test '{test_type}'."
        )

    rows = sorted(
        valid_rows,
        key=lambda row: float(row["segment_time"]),
    )

    times = np.asarray(
        [
            float(row["segment_time"])
            for row in rows
        ],
        dtype=float,
    )
    
    task_labels = np.asarray(
        [
            str(row["task_label"])
            for row in rows
        ],
        dtype=str,
    )


    position_residual = np.asarray(
        [
            float(row["encoder_deviation"][motor])
            for row in rows
        ],
        dtype=float,
    )

    current_residual = np.asarray(
        [
            float(row["filtered_current_deviation"][motor])
            for row in rows
        ],
        dtype=float,
    )

    (
        position_band,
        current_band,
    ) = get_healthy_bands(
        coupling_mode=coupling_mode,
        motor=motor,
        test_type=test_type,
        dof=dof,
        condition=condition,
    )

    position_lower, position_upper = (
        position_band
    )

    current_lower, current_upper = (
        current_band
    )

    position_result = analyse_signal(
        residual=position_residual,
        times=times,
        task_labels=task_labels,
        lower=position_lower,
        upper=position_upper,
        unit="pulses",
        max_continuous_outside_s= None,
    )

    current_duration_limit = (
        get_current_duration_limit(
            coupling_mode=coupling_mode,
            motor=motor,
            dof=dof,
            condition=condition,
        )
    )

    current_result = analyse_signal(
        residual=current_residual,
        times=times,
        task_labels=task_labels,
        lower=current_lower,
        upper=current_upper,
        unit="mA",
        max_continuous_outside_s=current_duration_limit,
    )

    deviating = (
        position_result["status"] == "deviating"
        or
        current_result["status"] == "deviating"
    )

    triggered_signals = []

    if position_result["status"] == "deviating":
        triggered_signals.append("position")

    if current_result["status"] == "deviating":
        triggered_signals.append("current")

    return {
        "status":
            "deviating"
            if deviating
            else "healthy",

        "motor":
            motor,

        "test_type":
            test_type,

        "condition":
            condition,

        "triggered_signals":
            triggered_signals,

        "position":
            position_result,

        "current":
            current_result,
    }


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def invalid_diagnosis(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "detector": DETECTOR_NAME,
        "detector_version": DETECTOR_VERSION,
        "evaluated_at_utc":
            datetime.now(timezone.utc).isoformat(),

        "coupling_mode": "unknown",
        "status": "invalid",

        "probable_fault_location":
            "undetermined",

        "affected_motors": [],

        "invalid_reasons": [reason],
    }


def diagnose_replay(
    replay_path: Path,
) -> dict[str, Any]:

    try:
        rows = load_jsonl(replay_path)
        validate_replay(rows)

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
    ) as exc:
        return invalid_diagnosis(str(exc))

    coupling_mode = str(
        rows[0]["coupling_mode"]
    )
    
    if coupling_mode == "motor_only":

        try:
            dof = detect_dof(
                replay_path
            )

            motors_to_evaluate = (
                DOF_RELEVANT_MOTORS[dof]
            )

        except ValueError:

            dof = None

            motors_to_evaluate = [
                0,
                1,
                2,
                3,
            ]

    elif coupling_mode == "gearbox_only":

        try:
            dof = detect_dof(
                replay_path
            )

        except ValueError as exc:
            return invalid_diagnosis(
                str(exc)
            )

        motors_to_evaluate = (
            DOF_RELEVANT_MOTORS[
                dof
            ]
        )

    elif coupling_mode == "full_setup":

        try:
            dof = detect_dof(
                replay_path
            )

        except ValueError as exc:
            return invalid_diagnosis(
                str(exc)
            )

        motors_to_evaluate = (
            DOF_RELEVANT_MOTORS[
                dof
            ]
        )

    else:

        return invalid_diagnosis(
            f"Unsupported coupling mode: "
            f"{coupling_mode}"
        )
    
    grouped = defaultdict(list)

    for row in rows:

        test_type = str(
            row["test_type"]
        )

        if test_type == "idle_baseline":
            continue

        # ---------------------------------------------------------------
        # Motor-only DOF pattern:
        # group per waveform + frequency + range
        # ---------------------------------------------------------------

        if (
            coupling_mode in (
                "motor_only",
                "full_setup",
            )
            and dof is not None
        ):

            sequence_condition = str(
                row.get("sequence_condition", "")
            ).strip().lower()

            # Pause samples are evaluated separately
            # by analyse_pause_rows().
            if sequence_condition == "pause":
                continue

            try:
                condition = detect_dof_condition(
                    row
                )

            except ValueError as exc:
                print(
                    f"WARNING: {exc}",
                    file=sys.stderr,
                )
                continue

            group_key = (
                test_type,
                condition,
            )

        # ---------------------------------------------------------------
        # Normal motor-only pattern / gearbox
        # ---------------------------------------------------------------

        else:

            if coupling_mode == "motor_only":

                if (
                    test_type not in MOTOR_ONLY_POSITION_BANDS
                    or test_type not in MOTOR_ONLY_CURRENT_BANDS
                ):
                    continue

            group_key = (
                test_type,
                None,
            )

        grouped[
            group_key
        ].append(
            row
        )
    skipped_conditions = set()
    results = []

    for (test_type, condition,), test_rows in grouped.items():

        for motor in motors_to_evaluate:

            try:
                result = analyse_motor_test(
                    rows=test_rows,
                    motor=motor,
                    test_type=test_type,
                    coupling_mode=coupling_mode,
                    dof=dof,
                    condition=condition,
                )

            except ValueError as exc:

                if condition is not None:
                    skipped_conditions.add(
                        (
                            condition,
                            motor,
                        )
                    )

                print(
                    f"WARNING: skipping diagnostics for "
                    f"DOF{dof if dof is not None else '-'}, "
                    f"motor {motor}, test '{test_type}', "
                    f"condition '{condition}': {exc}",
                    file=sys.stderr,
                )
                continue

            results.append(result)

    if not results:
        return invalid_diagnosis(
            "No valid motor/test combinations were evaluated."
        )
    pause_analysis = None

    if (
        coupling_mode in (
            "motor_only",
            "full_setup",
        )
        and dof is not None
    ):
        pause_analysis = analyse_pause_rows(
            rows=rows,
            motors=motors_to_evaluate,
            coupling_mode=coupling_mode,
            dof=dof,
        )

    evaluation_coverage = (
        "partial"
        if skipped_conditions
        else "complete"
    )

    deviating_results = [
        result
        for result in results
        if result["status"] == "deviating"
    ]

    pause_deviating = (
        pause_analysis is not None
        and pause_analysis.get("status")
            == "deviating"
    )

    motion_affected_motors = {
        result["motor"]
        for result in deviating_results
    }

    pause_affected_motors = set(
        pause_analysis.get(
            "affected_motors",
            [],
        )
        if pause_analysis
        else []
    )

    affected_motors = sorted(
        motion_affected_motors
        | pause_affected_motors
    )

    deviating = (
        bool(deviating_results)
        or pause_deviating
    )

    motor_results = {}

    for motor in motors_to_evaluate:

        motor_tests = [
            result
            for result in results
            if result["motor"] == motor
        ]

        motor_deviations = [
            result
            for result in motor_tests
            if result["status"] == "deviating"
        ]

        pause_motor_result = (
            pause_analysis
            .get(
                "motor_results",
                {},
            )
            .get(
                f"motor_{motor}"
            )
            if pause_analysis
            else None
        )

        pause_motor_deviating = (
            pause_motor_result is not None
            and pause_motor_result.get("status")
                == "deviating"
        )

        motor_is_deviating = (
            bool(motor_deviations)
            or pause_motor_deviating
        )

        motor_results[f"motor_{motor}"] = {
            "status":
                "deviating"
                if motor_is_deviating
                else "healthy",

            "tests":
                motor_tests,

            "pause_baseline":
                pause_motor_result,
        }

    position_deviations = sum(
        result["position"]["status"] == "deviating"
        for result in results
    )

    current_deviations = sum(
        result["current"]["status"] == "deviating"
        for result in results
    )

    return {
        "schema_version": 3,

        "detector":
            DETECTOR_NAME,

        "detector_version":
            DETECTOR_VERSION,

        "evaluated_at_utc":
            datetime.now(timezone.utc).isoformat(),

        "coupling_mode":
            coupling_mode,

        "dof":
            dof,

        "evaluation_coverage":
            evaluation_coverage,

        "skipped_conditions": [
            {
                "condition": condition,
                "motor": motor,
            }
            for condition, motor
            in sorted(skipped_conditions)
        ],

        "pause_analysis":
            pause_analysis,

        "status":
            "deviating"
            if deviating
            else "healthy",

        "probable_fault_location":
            (
                "motor_or_gearbox"
                if (
                    deviating
                    and coupling_mode == "gearbox_only"
                )
                else (
                    "motor"
                    if deviating
                    else "none"
                )
            ),

        "affected_motors":
            affected_motors,

        "baseline_reference": {
            "name":
                (
                    MOTOR_ONLY_BASELINE_NAME
                    if coupling_mode == "motor_only"
                    else (
                        FULL_SETUP_BASELINE_NAME
                        if coupling_mode == "full_setup"
                        else GEARBOX_BASELINE_NAME
                    )
                ),

            "method":
                "fixed healthy position and current "
                "residual bands per motor and test type",

            "maximum_fraction_outside":
                MAX_OUTSIDE_FRACTION,

            "position_maximum_continuous_outside_s":
                None, #POSITION_MAX_CONTINUOUS_OUTSIDE_S,

            "current_maximum_continuous_outside_s":
                None, #CURRENT_MAX_CONTINUOUS_OUTSIDE_S,

            "current_signal":
                "filtered_current_deviation",
        },

        "summary": {
            "evaluated_motor_test_combinations":
                len(results),

            "deviating_motor_test_combinations":
                len(deviating_results),

            "position_deviations":
                int(position_deviations),

            "current_deviations":
                int(current_deviations),

            "pause_baseline_status":
                pause_analysis.get("status", "not_evaluated")
                if pause_analysis
                else "not_evaluated",
            
        },

        "motor_results":
            motor_results,
    }

# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def create_diagnostic_plots(
    replay_path: Path,
    output_dir: Path,
) -> list[str]:
    """Create one diagnostic plot per motor using only that motor's active tasks."""

    rows = load_jsonl(replay_path)
    validate_replay(rows)

    coupling_mode = str(
        rows[0]["coupling_mode"]
    )

    if coupling_mode == "gearbox_only":

        dof = detect_dof(
            replay_path
        )

        motors_to_plot = (
            DOF_RELEVANT_MOTORS[dof]
        )

    elif coupling_mode == "motor_only":

        try:
            dof = detect_dof(
                replay_path
            )

            motors_to_plot = (
                DOF_RELEVANT_MOTORS[dof]
            )

        except ValueError:

            dof = None

            motors_to_plot = [
                0,
                1,
                2,
                3,
            ]

    elif coupling_mode == "full_setup":

        dof = detect_dof(
            replay_path
        )

        motors_to_plot = (
            DOF_RELEVANT_MOTORS[
                dof
            ]
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_paths = []

    # Group complete replay by task_label.
    task_segments = defaultdict(list)

    for row in rows:
        task_segments[
            str(row["task_label"])
        ].append(row)

    for motor in motors_to_plot:

        # ---------------------------------------------------------------
        # Select only tasks in which this motor is actually commanded
        # ---------------------------------------------------------------

        motor_rows = []

        for task_label, segment in task_segments.items():

            segment = sorted(
                segment,
                key=lambda row: float(row["segment_time"]),
            )

            if not segment:
                continue

            test_type = str(
                segment[0]["test_type"]
            )

            # Skip idle baseline.
            if test_type == "idle_baseline":
                continue

            # Skip tests for which no diagnostic band exists.
            if coupling_mode == "motor_only" and dof is None:

                if (
                    test_type not in MOTOR_ONLY_POSITION_BANDS
                    or test_type not in MOTOR_ONLY_CURRENT_BANDS
                ):
                    continue

            if coupling_mode == "motor_only" and dof is None:

                commanded_target = np.asarray(
                    [
                        float(
                            row["offline_commanded_target"][motor]
                        )
                        for row in segment
                    ],
                    dtype=float,
                )

                if len(commanded_target) < 2:
                    continue

                target_change = np.diff(
                    commanded_target
                )

                motor_is_active = np.any(
                    np.abs(target_change) > 1e-12
                )

                if not motor_is_active:
                    continue

            # For gearbox_only, the relevant motors are already selected
            # through DOF_RELEVANT_MOTORS[dof].

            motor_rows.extend(
                segment
            )

        if not motor_rows:
            print(
                f"WARNING: no active diagnostic rows found "
                f"for motor {motor}.",
                file=sys.stderr,
            )
            continue

        # ---------------------------------------------------------------
        # Remove rows with missing / invalid values required for plotting
        # ---------------------------------------------------------------

        valid_motor_rows = []

        for row in motor_rows:

            try:

                values = [
                    row["offline_predicted_positions"][motor],
                    row["measured_positions"][motor],
                    row["encoder_deviation"][motor],

                    row["offline_predicted_currents"][motor],
                    row["filtered_measured_currents"][motor],
                    row["filtered_current_deviation"][motor],
                ]

                if any(
                    value is None
                    for value in values
                ):
                    continue

                if not all(
                    np.isfinite(float(value))
                    for value in values
                ):
                    continue

                valid_motor_rows.append(
                    row
                )

            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ):
                continue

        motor_rows = valid_motor_rows

        if not motor_rows:

            print(
                f"WARNING: no valid plot rows found "
                f"for motor {motor}.",
                file=sys.stderr,
            )

            continue

        # ---------------------------------------------------------------
        # Continuous plot x-axis for only this motor's own tasks
        # ---------------------------------------------------------------

        # Keep only DOF motion conditions for which healthy bands exist.
        if (
            coupling_mode in (
                "motor_only",
                "full_setup",
            )
            and dof is not None
        ):

            filtered_rows = []

            for row in motor_rows:
                condition = str(
                    row.get("sequence_condition", "")
                ).strip().lower()

                if condition in ("", "none", "pause"):
                    continue

                try:
                    get_healthy_bands(
                        coupling_mode=coupling_mode,
                        motor=motor,
                        test_type=str(row["test_type"]),
                        dof=dof,
                        condition=condition,
                    )
                except ValueError:
                    continue

                filtered_rows.append(row)

            motor_rows = filtered_rows

            if not motor_rows:
                print(
                    f"WARNING: no DOF rows with defined healthy bands "
                    f"for motor {motor}.",
                    file=sys.stderr,
                )
                continue


        x = np.arange(
            len(motor_rows)
        )

        test_types = [
            str(row["test_type"])
            for row in motor_rows
        ]

        conditions = []

        for row in motor_rows:

            if (
                coupling_mode in (
                    "motor_only",
                    "full_setup",
                )
                and dof is not None
            ):
                condition = detect_dof_condition(
                    row
                )
            else:
                condition = None

            conditions.append(
                condition
            )
        # ---------------------------------------------------------------
        # Position signals
        # ---------------------------------------------------------------

        predicted_position = np.asarray(
            [
                float(
                    row["offline_predicted_positions"][motor]
                )
                for row in motor_rows
            ],
            dtype=float,
        )

        measured_position = np.asarray(
            [
                float(
                    row["measured_positions"][motor]
                )
                for row in motor_rows
            ],
            dtype=float,
        )

        position_residual = np.asarray(
            [
                float(
                    row["encoder_deviation"][motor]
                )
                for row in motor_rows
            ],
            dtype=float,
        )

        # ---------------------------------------------------------------
        # Current signals
        # ---------------------------------------------------------------

        predicted_current = np.asarray(
            [
                float(
                    row["offline_predicted_currents"][motor]
                )
                for row in motor_rows
            ],
            dtype=float,
        )

        measured_current = np.asarray(
            [
                float(
                    row["filtered_measured_currents"][motor]
                )
                for row in motor_rows
            ],
            dtype=float,
        )

        current_residual = np.asarray(
            [
                float(
                    row["filtered_current_deviation"][motor]
                )
                for row in motor_rows
            ],
            dtype=float,
        )

        # ---------------------------------------------------------------
        # Healthy bands for every selected sample
        # ---------------------------------------------------------------

        position_lower = np.full(
            len(motor_rows),
            np.nan,
        )

        position_upper = np.full(
            len(motor_rows),
            np.nan,
        )

        current_lower = np.full(
            len(motor_rows),
            np.nan,
        )

        current_upper = np.full(
            len(motor_rows),
            np.nan,
        )

        for i, (
            test_type,
            condition,
        ) in enumerate(
            zip(
                test_types,
                conditions,
            )
        ):

            (
                position_band,
                current_band,
            ) = get_healthy_bands(
                coupling_mode=coupling_mode,
                motor=motor,
                test_type=test_type,
                condition=condition,
                dof=dof,
            )

            position_lower[i] = (
                position_band[0]
            )

            position_upper[i] = (
                position_band[1]
            )

            current_lower[i] = (
                current_band[0]
            )

            current_upper[i] = (
                current_band[1]
            )

        # ---------------------------------------------------------------
        # Healthy envelopes around DT prediction
        # ---------------------------------------------------------------

        position_prediction_lower = (
            predicted_position
            + position_lower
        )

        position_prediction_upper = (
            predicted_position
            + position_upper
        )

        current_prediction_lower = (
            predicted_current
            + current_lower
        )

        current_prediction_upper = (
            predicted_current
            + current_upper
        )

        # ---------------------------------------------------------------
        # Samples outside the healthy residual bands
        # ---------------------------------------------------------------

        position_outside = (
            (position_residual < position_lower)
            |
            (position_residual > position_upper)
        )

        current_outside = (
            (current_residual < current_lower)
            |
            (current_residual > current_upper)
        )

        # ---------------------------------------------------------------
        # Plot
        # ---------------------------------------------------------------

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(16, 14),
            sharex=True,
        )

        # ---------------------------------------------------------------
        # Position prediction
        # ---------------------------------------------------------------

        # axes[0].fill_between(
        #     x,
        #     position_prediction_lower,
        #     position_prediction_upper,
        #     alpha=0.20,
        #     label="Healthy prediction envelope",
        # )

        axes[0].plot(
            x,
            measured_position,
            linewidth=1.0,
            label="Measured position",
        )

        axes[0].plot(
            x,
            predicted_position,
            linestyle="--",
            linewidth=1.0,
            label="DT predicted position",
        )

        axes[0].set_ylabel(
            "Position [pulses]"
        )

        axes[0].set_title(
            f"Motor {motor} - Position prediction"
        )

        axes[0].grid(
            True,
            alpha=0.25,
        )

        axes[0].legend(
            loc="upper right"
        )

        # ---------------------------------------------------------------
        # Position residual
        # ---------------------------------------------------------------

        axes[1].fill_between(
            x,
            position_lower,
            position_upper,
            alpha=0.20,
            label="Healthy residual band",
        )

        axes[1].plot(
            x,
            position_residual,
            linewidth=0.9,
            label="Position residual",
        )

        if np.any(
            position_outside
        ):
            axes[1].scatter(
                x[position_outside],
                position_residual[
                    position_outside
                ],
                s=10,
                label="Outside healthy band",
            )

        axes[1].axhline(
            0.0,
            linewidth=0.8,
        )

        axes[1].set_ylabel(
            "Residual [pulses]"
        )

        axes[1].set_title(
            "Position residual diagnosis"
        )

        axes[1].grid(
            True,
            alpha=0.25,
        )

        axes[1].legend(
            loc="upper right"
        )

        # ---------------------------------------------------------------
        # Current prediction
        # ---------------------------------------------------------------

        # axes[2].fill_between(
        #     x,
        #     current_prediction_lower,
        #     current_prediction_upper,
        #     alpha=0.20,
        #     label="Healthy prediction envelope",
        # )

        axes[2].plot(
            x,
            measured_current,
            linewidth=0.9,
            label="Filtered measured current",
        )

        axes[2].plot(
            x,
            predicted_current,
            linestyle="--",
            linewidth=0.9,
            label="DT predicted current",
        )

        axes[2].set_ylabel(
            "Current [mA]"
        )

        axes[2].set_title(
            f"Motor {motor} - Current prediction"
        )

        axes[2].grid(
            True,
            alpha=0.25,
        )

        axes[2].legend(
            loc="upper right"
        )

        # ---------------------------------------------------------------
        # Current residual
        # ---------------------------------------------------------------

        axes[3].fill_between(
            x,
            current_lower,
            current_upper,
            alpha=0.20,
            label="Healthy residual band",
        )

        axes[3].plot(
            x,
            current_residual,
            linewidth=0.9,
            label="Current residual",
        )

        if np.any(
            current_outside
        ):
            axes[3].scatter(
                x[current_outside],
                current_residual[
                    current_outside
                ],
                s=10,
                label="Outside healthy band",
            )

        axes[3].axhline(
            0.0,
            linewidth=0.8,
        )

        axes[3].set_xlabel(
            "Sample"
        )

        axes[3].set_ylabel(
            "Residual [mA]"
        )

        axes[3].set_title(
            "Current residual diagnosis"
        )

        axes[3].grid(
            True,
            alpha=0.25,
        )

        axes[3].legend(
            loc="upper right"
        )

        fig.suptitle(
            f"Digital Twin diagnosis - Motor {motor}",
            fontsize=16,
        )

        fig.tight_layout()

        plot_path = (
            output_dir
            / f"digital_twin_diagnosis_motor_{motor}.png"
        )

        fig.savefig(
            plot_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        plot_paths.append(
            str(plot_path)
        )

    return plot_paths

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def update_metadata(
    metadata_path: Path,
    diagnosis: dict[str, Any],
) -> None:

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    metadata["diagnostics"] = diagnosis

    temporary_name = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=metadata_path.parent,
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:

            temporary_name = handle.name

            json.dump(
                metadata,
                handle,
                indent=2,
                allow_nan=False,
            )

            handle.write("\n")

        os.replace(
            temporary_name,
            metadata_path,
        )

    finally:

        if (
            temporary_name
            and os.path.exists(temporary_name)
        ):
            os.unlink(temporary_name)

def write_diagnostics_file(
    diagnostics_path: Path,
    diagnosis: dict[str, Any],
) -> None:

    diagnostics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with diagnostics_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            diagnosis,
            handle,
            indent=2,
            allow_nan=False,
        )

        handle.write("\n")

# ---------------------------------------------------------------------------
# Popup
# ---------------------------------------------------------------------------

def show_diagnosis_popup(
    diagnosis: dict[str, Any],
) -> bool:

    try:
        import tkinter as tk
        from tkinter import ttk

    except ImportError:
        print(
            "WARNING: diagnosis popup unavailable; "
            "install python3-tk, "
            "for Ubuntu use sudo apt install python3-tk.",
            file=sys.stderr,
        )
        return False

    status = str(
        diagnosis.get(
            "status",
            "invalid",
        )
    )

    colors = {
        "healthy":
            ("#176B35", "#F1FFF5"),

        "deviating":
            ("#8B1E1E", "#FFF4F4"),

        "invalid":
            ("#9A5B00", "#FFF8E8"),
    }

    accent, background = colors.get(
        status,
        colors["invalid"],
    )

    try:
        root = tk.Tk()

    except tk.TclError as exc:
        print(
            f"WARNING: diagnosis popup unavailable: {exc}",
            file=sys.stderr,
        )
        return False

    root.title(
        f"Digital Twin diagnosis - {status.upper()}"
    )

    root.geometry(
        "1450x750"
    )

    root.minsize(
        1050,
        600,
    )

    root.configure(
        bg=background,
    )

    root.attributes(
        "-topmost",
        True,
    )

    heading = {
        "healthy":
            "TEST HEALTHY",

        "deviating":
            "DEVIATION DETECTED",

        "invalid":
            "TEST INVALID",
    }.get(
        status,
        "DIAGNOSIS",
    )

    tk.Label(
        root,
        text=heading,
        font=(
            "DejaVu Sans",
            20,
            "bold",
        ),
        fg="white",
        bg=accent,
        pady=12,
    ).pack(
        fill="x",
    )

    notebook = ttk.Notebook(
        root
    )

    notebook.pack(
        fill="both",
        expand=True,
        padx=15,
        pady=15,
    )

    # ------------------------------------------------------------------
    # SUMMARY TAB
    # ------------------------------------------------------------------

    summary_tab = tk.Frame(
        notebook,
        bg=background,
    )

    notebook.add(
        summary_tab,
        text="Summary",
    )

    affected = diagnosis.get(
        "affected_motors",
        [],
    )

    affected_text = (
        ", ".join(
            f"M{motor}"
            for motor in affected
        )
        if affected
        else "none"
    )

    summary_data = diagnosis.get(
        "summary",
        {},
    )

    summary_lines = [
        f"Overall result: {status.upper()}",
        "",
        (
            "Configuration: "
            f'{diagnosis.get("coupling_mode", "unknown")}'
        ),
        (
            f'DOF: {diagnosis.get("dof")}'
            if diagnosis.get("dof") is not None
            else ""
        ),
        (
            "Evaluation coverage: "
                f'{diagnosis.get("evaluation_coverage", "complete").upper()}'
        ),

        f"Affected motor(s): {affected_text}",
        "",
        (
            "Position deviations: "
            f'{summary_data.get("position_deviations", 0)}'
        ),
        (
            "Current deviations: "
            f'{summary_data.get("current_deviations", 0)}'
        ),
        (
            "Pause baseline: "
            f'{summary_data.get(
                "pause_baseline_status",
                "not_evaluated",
            ).upper()}'
        ),
        "",
        (
            "Detection rule: more than 10% of samples "
            "outside the healthy band"
        ),
        (
            "Current DOF tests can additionally trigger when "
            "the continuous excursion exceeds the "
            "condition-specific duration limit."
        ),
    ]

    tk.Label(
        summary_tab,
        text="\n".join(
            summary_lines
        ),
        justify="left",
        anchor="nw",
        font=(
            "DejaVu Sans",
            13,
        ),
        bg=background,
        padx=25,
        pady=25,
    ).pack(
        fill="both",
        expand=True,
    )

    # ------------------------------------------------------------------
    # DETAILS TAB
    # ------------------------------------------------------------------

    details_tab = tk.Frame(
        notebook,
        bg=background,
    )

    notebook.add(
        details_tab,
        text="Details",
    )

    columns = (
        "motor",
        "test",
        "condition",
        "signal",
        "band",
        "outside",
        "duration",
        "trigger",
        "range",
        "status",
    )

    tree = ttk.Treeview(
        details_tab,
        columns=columns,
        show="headings",
    )

    labels = {
        "motor":
            "Motor",

        "test":
            "Test type",

        "condition":
            "Test condition",

        "signal":
            "Signal",

        "band":
            "Healthy band",

        "outside":
            "Samples outside band",

        "duration":
            "Longest excursion",

        "trigger":
            "Trigger",

        "range":
            "Observed residual",

        "status":
            "Result",
    }

    widths = {
        "motor": 60,
        "test": 210,
        "condition": 220,
        "signal": 90,
        "band": 180,
        "outside": 200,
        "duration": 160,
        "trigger": 110,
        "range": 210,
        "status": 110,
    }

    for column in columns:

        tree.heading(
            column,
            text=labels[column],
        )

        tree.column(
            column,
            width=widths[column],
            anchor="w",
        )

    scrollbar = ttk.Scrollbar(
        details_tab,
        orient="vertical",
        command=tree.yview,
    )

    tree.configure(
        yscrollcommand=scrollbar.set,
    )

    tree.pack(
        side="left",
        fill="both",
        expand=True,
        padx=10,
        pady=10,
    )

    scrollbar.pack(
        side="right",
        fill="y",
        pady=10,
    )

    for motor_name, motor_result in (
        diagnosis
        .get("motor_results", {})
        .items()
    ):

        motor_label = motor_name.replace(
            "motor_",
            "M",
        )

        for test in motor_result.get(
            "tests",
            [],
        ):

            for signal_name in (
                "position",
                "current",
            ):

                signal = test[
                    signal_name
                ]

                band = signal[
                    "healthy_band"
                ]

                unit = band["unit"]

                band_text = (
                    f'{band["lower"]:.0f} '
                    f'to '
                    f'{band["upper"]:.0f} '
                    f'{unit}'
                )

                outside_text = (
                    f'{signal["samples_outside_band"]}'
                    f'/{signal["samples"]} '
                    f'('
                    f'{signal["percentage_outside_band"]:.1f}%'
                    f')'
                )

                duration_limit = signal.get(
                    "maximum_continuous_outside_s"
                )

                if duration_limit is not None:

                    duration_text = (
                        f'{1000.0 * signal["longest_continuous_outside_s"]:.0f} '
                        f'/ '
                        f'{1000.0 * duration_limit:.0f} ms'
                    )

                else:

                    duration_text = (
                        f'{1000.0 * signal["longest_continuous_outside_s"]:.0f} ms'
                    )

                range_text = (
                    f'{signal["minimum_residual"]:.1f} '
                    f'to '
                    f'{signal["maximum_residual"]:.1f} '
                    f'{unit}'
                )
                fraction_triggered = signal.get(
                    "fraction_rule_triggered",
                    False,
                )

                duration_triggered = signal.get(
                    "duration_rule_triggered",
                    False,
                )

                if fraction_triggered and duration_triggered:
                    trigger_text = "fraction + duration"

                elif fraction_triggered:
                    trigger_text = "fraction"

                elif duration_triggered:
                    trigger_text = "continuous duration"

                else:
                    trigger_text = "-"

                tree.insert(
                    "",
                    "end",
                    values=(
                        motor_label,
                        test["test_type"],
                        test.get("condition") or "-",
                        signal_name,
                        band_text,
                        outside_text,
                        duration_text,
                        trigger_text,
                        range_text,
                        signal["status"],
                    ),
                )

    # ------------------------------------------------------------------
    # OUTPUT FILES TAB
    # ------------------------------------------------------------------

    files_tab = tk.Frame(
        notebook,
        bg=background,
    )

    notebook.add(
        files_tab,
        text="Output files",
    )

    plot_paths = diagnosis.get(
        "plot_files",
        [],
    )

    diagnostics_file = diagnosis.get(
        "diagnostics_file",
        "not written",
    )

    file_lines = [
        "Diagnostic plots:",
        "",
    ]

    file_lines.extend(
        plot_paths
    )

    file_lines.extend([
        "",
        "Detailed diagnostics JSON:",
        str(diagnostics_file),
    ])

    tk.Label(
        files_tab,
        text="\n".join(
            file_lines
        ),
        justify="left",
        anchor="nw",
        font=(
            "DejaVu Sans Mono",
            10,
        ),
        bg=background,
        padx=20,
        pady=20,
    ).pack(
        fill="both",
        expand=True,
    )

    def close_popup():
        root.quit()
        root.destroy()


    tk.Button(
        root,
        text="Close and continue",
        command=close_popup,
        font=(
            "DejaVu Sans",
            12,
            "bold",
        ),
        bg=accent,
        fg="white",
        padx=22,
        pady=8,
    ).pack(
        pady=12,
    )

    root.protocol(
        "WM_DELETE_WINDOW",
        close_popup,
    )

    root.mainloop()

    return True

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--replay-file",
        "--file",
        dest="replay_file",
        help="Motor Digital Twin replay JSONL",
    )

    parser.add_argument(
        "--popup-file",
        help="Show popup from an existing diagnostics JSON file",
    )

    parser.add_argument(
        "--metadata-file",
        help="Existing run metadata JSON to update",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diagnosis without modifying metadata",
    )

    parser.add_argument(
        "--no-popup",
        action="store_true",
        help="Do not show diagnosis popup",
    )

    return parser


def main() -> None:

    args = build_parser().parse_args()

    # Standalone popup mode.
    if args.popup_file:

        popup_path = (
            Path(args.popup_file)
            .expanduser()
            .resolve()
        )

        with popup_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            diagnosis = json.load(handle)

        show_diagnosis_popup(
            diagnosis
        )

        return

    if not args.replay_file or not args.metadata_file:
        raise SystemExit(
            "--replay-file and --metadata-file are required "
            "unless --popup-file is used."
        )

    replay_path = (
        Path(args.replay_file)
        .expanduser()
        .resolve()
    )

    metadata_path = (
        Path(args.metadata_file)
        .expanduser()
        .resolve()
    )

    # ------------------------------------------------------------------
    # Run diagnosis
    # ------------------------------------------------------------------

    diagnosis = diagnose_replay(
        replay_path,
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    diagnostics_dir = (
        metadata_path.parent
        / "diagnostics"
    )

    diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Create diagnostic plots
    # ------------------------------------------------------------------

    plot_files = []

    if diagnosis.get("status") != "invalid":

        try:

            plot_files = create_diagnostic_plots(
                replay_path,
                diagnostics_dir,
            )

        except Exception as exc:

            print(
                f"WARNING: could not create "
                f"diagnostic plots: {exc}",
                file=sys.stderr,
            )

    diagnosis["plot_files"] = (
        plot_files
    )

    # ------------------------------------------------------------------
    # Separate diagnostics JSON
    # ------------------------------------------------------------------

    diagnostics_path = (
        diagnostics_dir
        / "digital_twin_diagnosis.json"
    )

    diagnosis["diagnostics_file"] = (
        str(diagnostics_path)
    )

    if not args.dry_run:

        try:

            write_diagnostics_file(
                diagnostics_path,
                diagnosis,
            )

        except Exception as exc:

            print(
                f"WARNING: could not write "
                f"diagnostics JSON: {exc}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    if not args.dry_run:

        try:

            update_metadata(
                metadata_path,
                diagnosis,
            )

        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:

            raise SystemExit(
                f"Could not update metadata: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Terminal output
    # ------------------------------------------------------------------

    print(
        json.dumps(
            diagnosis,
            indent=2,
            allow_nan=False,
        )
    )

    if args.dry_run:

        print(
            "Dry run: metadata was not changed."
        )

    else:

        print(
            f"Updated metadata: {metadata_path}"
        )

        print(
            f"Diagnostics directory: "
            f"{diagnostics_dir}"
        )

    # ------------------------------------------------------------------
    # Start popup as separate process
    # ------------------------------------------------------------------

    if (
        not args.dry_run
        and not args.no_popup
    ):

        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--popup-file",
                str(diagnostics_path),
            ],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    main()