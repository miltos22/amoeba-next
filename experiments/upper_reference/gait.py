"""A steerable travelling-wave gait for the stage-0 body.

This is a measuring instrument, not part of any organism: nothing under
amoeba/ or baselines/ may import it.

The wave started from the one measured in an earlier gait check:

    target(k, j, t) = AMP * sin(2*pi*FREQ*t - j*pi/2 - k*pi/2)

That pattern repeats every four arms (4 * 90 = 360 degrees), so shifting
it by one arm is not a rotation of the physical pattern for a six-armed
body; measured, it did not steer the body.  With a phase step of 260
degrees per arm and 45 degrees per joint the pattern is closed around the
six arms, and shifting it by m arms (arm index taken modulo 6) rotates
the swim direction by exactly m * 60 degrees in the body frame.  The six
gaits measured over 30 s swims (measure_gait.py, seed 0, lean, no food)
travel at

    chord direction (body frame) = 5.6513076245987355 deg - m * 60 deg
    net displacement            = 0.8994460811348363 m in 30 s
    mean speed                  = 2.9981536037827876 cm/s
    yaw during the swim         = -35.39180053613916 deg

The function below takes a direction in the body frame and returns the
targets of the nearest of the six gaits (within 30 degrees of the
commanded direction; the measured error is 5.65 degrees).
"""

from __future__ import annotations

import math

AMP = 1.0
FREQ = 0.75
N_ARMS = 6
JOINTS_PER_ARM = 2
ARM_STEP_RAD = math.pi / 3.0

RING_STEP_DEG = 260.0
ARM_STEP_DEG = 45.0

# Measured by measure_gait.py: the six chords are ZERO - m * 60 degrees
# in the body frame, m = 0..5.
SWIM_DIR_ZERO_DEG = 5.6513076245987355
SWIM_DIR_ZERO_RAD = math.radians(SWIM_DIR_ZERO_DEG)
SWIM_DIR_PER_ARM_RAD = -ARM_STEP_RAD


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def arm_shift_for_direction(direction_body: float) -> int:
    """Nearest of the six gaits, as an arm shift m in 0..5."""
    step = (direction_body - SWIM_DIR_ZERO_RAD) / SWIM_DIR_PER_ARM_RAD
    return int(round(step)) % N_ARMS


def targets_for_arm_shift(arm_shift: int, t: float) -> dict[str, float]:
    """The 12 joint targets of the travelling-wave gait shifted by m arms."""
    base = 2.0 * math.pi * FREQ * t
    ring = math.radians(RING_STEP_DEG)
    arm = math.radians(ARM_STEP_DEG)
    out: dict[str, float] = {}
    for k in range(N_ARMS):
        for j in range(JOINTS_PER_ARM):
            phase = base - j * arm - ((k + arm_shift) % N_ARMS) * ring
            out[f"arm{k}.j{j + 1}.target"] = AMP * math.sin(phase)
    return out


def swim_targets(direction_body: float, t: float) -> dict[str, float]:
    """The 12 joint targets for a swim toward a body-frame angle (rad)."""
    return targets_for_arm_shift(arm_shift_for_direction(direction_body), t)


def in_phase_targets(t: float) -> dict[str, float]:
    """All 12 joints driven in phase: the reciprocal flap that must not swim."""
    value = AMP * math.sin(2.0 * math.pi * FREQ * t)
    return {
        f"arm{k}.j{j + 1}.target": value
        for k in range(N_ARMS)
        for j in range(JOINTS_PER_ARM)
    }


def stop_targets() -> dict[str, float]:
    """All 12 targets at zero: holding still."""
    return {
        f"arm{k}.j{j + 1}.target": 0.0
        for k in range(N_ARMS)
        for j in range(JOINTS_PER_ARM)
    }
