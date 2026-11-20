"""Energy and its feeling.

The numbers are the two named presets of the design (section 3.4.4).
They bracket constants nobody can measure; nothing here is tuned to a
behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

NEED_LOW = -1.0
NEED_HIGH = 1.0


@dataclass(frozen=True, slots=True)
class Preset:
    name: str
    basal_rate: float
    effort_rate: float
    intake_rate: float
    strength_floor: float
    energy_set_point: float


PRESETS = {
    "lean": Preset(
        name="lean",
        basal_rate=1 / 600,
        effort_rate=1 / 600,
        intake_rate=0.05,
        strength_floor=0.3,
        energy_set_point=0.8,
    ),
    "rich": Preset(
        name="rich",
        basal_rate=1 / 1800,
        effort_rate=1 / 1800,
        intake_rate=0.15,
        strength_floor=0.5,
        energy_set_point=0.8,
    ),
    "valve_room": Preset(
        name="valve_room",
        basal_rate=1 / 600,
        effort_rate=1 / 600,
        intake_rate=0.05,
        strength_floor=0.3,
        energy_set_point=0.8,
    ),
}


def resolve_preset(preset) -> Preset:
    if isinstance(preset, Preset):
        return preset
    try:
        return PRESETS[preset]
    except KeyError:
        raise ValueError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}") from None


def clamp_energy(energy: float) -> float:
    return 0.0 if energy < 0.0 else (1.0 if energy > 1.0 else energy)


def strength(energy: float, preset: Preset, weakness: bool = True) -> float:
    """Multiplier on actuator gain and force limit (3.4.2)."""
    if not weakness:
        return 1.0
    e = clamp_energy(energy)
    return preset.strength_floor + (1.0 - preset.strength_floor) * e


def energy_need(energy: float, preset: Preset) -> float:
    """The feeling item: set point minus energy, clipped to its range (3.4.3)."""
    need = preset.energy_set_point - clamp_energy(energy)
    return NEED_LOW if need < NEED_LOW else (NEED_HIGH if need > NEED_HIGH else need)


def fatigue_step(
    fatigue: float,
    effort: float,
    dt: float,
    threshold: float,
    gain_per_s: float,
    tau_s: float,
) -> float:
    """Advance the tiredness store by ``dt`` with effort held constant.

    The rule is ``dF/dt = load - F / tau_s`` with
    ``load = gain_per_s * max(0, effort - threshold)``, so the exact
    solution over the interval is ``F_inf + (F - F_inf) * exp(-dt / tau_s)``
    with ``F_inf = load * tau_s``; the result is clipped to [0, 1].
    """
    load = gain_per_s * max(0.0, effort - threshold)
    f_inf = load * tau_s
    value = f_inf + (fatigue - f_inf) * math.exp(-dt / tau_s)
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
