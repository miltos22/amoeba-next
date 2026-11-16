"""Item records: the only thing a brain sees of a body or a world.

An item is a named value with a kind, a range and a unit.  Senses and
feelings are read; actuators are written with a target that stays in
force until the actuator is written again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

KINDS = ("sense", "actuator", "feeling")


@dataclass(frozen=True, slots=True)
class Item:
    """One named value of a body: a sense, an actuator or a feeling."""

    name: str
    kind: str
    low: float
    high: float
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("item name must be a non-empty string")
        if self.kind not in KINDS:
            raise ValueError(f"item kind must be one of {KINDS}, got {self.kind!r}")
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError("item range must be finite")
        if not (self.high > self.low):
            raise ValueError("item high must be greater than item low")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("item unit must be a non-empty string")
