"""Reference forager for food kinds: SmellForager on a chosen subset of kinds.

This is a measuring instrument, not an organism: nothing under amoeba/ or
baselines/ may import it.  It is built on
experiments.upper_reference.policies.SmellForager (imported, not copied):
the only change is that the arm smell readings it sees are the SUM of the
per-kind readings of the kinds in ``approach``.  It reads nothing except
what ``world.read()`` returns; the world object is used only for the
config (energy set point, arm count, kind names) and for that read call.
"""

from __future__ import annotations

from experiments.upper_reference.policies import SmellForager


class KindForager(SmellForager):
    """SmellForager that steers by the summed smell of the chosen kinds."""

    def __init__(
        self,
        world,
        approach: set[str],
        eat_below: float = 0.78,
        leave_above: float = 0.82,
        smell_min: float = 0.2,
        seed: int = 0,
        reverse: bool = False,
    ) -> None:
        names = [kind.name for kind in world.config.food_kinds]
        chosen = set(approach)
        missing = chosen - set(names)
        if missing:
            raise ValueError(f"approach names not in the config: {sorted(missing)!r}")
        self.world = world
        self.approach = tuple(name for name in names if name in chosen)
        self._raw_read = world.read
        super().__init__(
            self._read,
            world.config.preset_constants.energy_set_point,
            eat_below=eat_below,
            leave_above=leave_above,
            smell_min=smell_min,
            seed=seed,
            reverse=reverse,
            n_arms=world.config.n_arms,
        )

    def _read(self) -> dict[str, float]:
        reading = self._raw_read()
        if len(self.world.config.food_kinds) == 1:
            return reading
        out = dict(reading)
        for k in range(self.n_arms):
            out[f"arm{k}.smell"] = sum(
                float(reading[f"arm{k}.smell.{name}"]) for name in self.approach
            )
        return out
