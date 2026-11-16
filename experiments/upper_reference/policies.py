"""Hand-written forager policies: measuring instruments, not organisms.

Nothing under amoeba/ or baselines/ may import this.  Two policies:

OracleForager  reads the world's internals (true patch positions and
               remaining food, body centre and heading).  This is the
               ceiling: it sees more than any brain will.
SmellForager   reads only what world.read() returns.  It is handed a
               read callable and a write-free decision interface, so it
               cannot touch anything else.

Both implement the same three rules:

 1. energy < eat_below: steer toward the nearest patch that still has
    food and swim.  When food is inside the mouth radius, hold the
    current targets (freeze the gait).
 2. energy > leave_above and food is inside the mouth radius: steer
    directly away from that food and swim until no food is inside the
    mouth radius, then hold still.
 3. otherwise: hold still (all targets zero).

Modes handed to the runner:
    "swim"  write a gait toward self.direction
    "hold"  write nothing (the actuators keep the targets they have)
    "idle"  write all targets zero
"""

from __future__ import annotations

import math

import numpy as np

from experiments.upper_reference.gait import _wrap, stop_targets, swim_targets

N_ARMS = 6
ARM_ANGLE = 2.0 * math.pi / N_ARMS


class OracleForager:
    """The ceiling policy: true state, true food, no readings needed."""

    def __init__(self, world, eat_below: float = 0.78, leave_above: float = 0.82,
                 reverse: bool = False) -> None:
        self.world = world
        self.eat_below = float(eat_below)
        self.leave_above = float(leave_above)
        self.reverse = bool(reverse)
        self.mode = "idle"
        self.direction = 0.0
        self.leaving = False
        self.leave_patch = None

    def _mouth_food(self) -> np.ndarray:
        world = self.world
        centre = np.asarray(world.body.centre(), dtype=float)
        pos = world.food.pos
        amt = world.food.amt
        if pos.size == 0:
            return np.zeros(0, dtype=int)
        d2 = ((pos - centre[None, :]) ** 2).sum(axis=1)
        mask = (d2 < world.config.mouth_radius ** 2) & (amt > 0.0)
        return np.flatnonzero(mask)

    def _nearest_patch(self):
        world = self.world
        totals = world.food.totals()
        candidates = np.flatnonzero(totals > 0.0)
        if candidates.size == 0:
            return None
        centre = np.asarray(world.body.centre(), dtype=float)
        d2 = ((world.food.centres[candidates] - centre[None, :]) ** 2).sum(axis=1)
        return int(candidates[int(np.argmin(d2))])

    def _swim_direction(self, target_xy) -> float:
        world = self.world
        cx, cy = world.body.centre()
        dx = float(target_xy[0]) - cx
        dy = float(target_xy[1]) - cy
        if dx * dx + dy * dy < 1e-18:
            return self.direction
        return _wrap(math.atan2(dy, dx) - world.body.heading())

    def _start_leave(self, mouth: np.ndarray) -> None:
        """Latch the patch that is being eaten and the escape direction.

        A patch is a disc of 20 particles; steering away from whichever
        particle is nearest at each update flips the direction and keeps
        the mouth inside the patch, so the escape is measured from the
        patch centre of the food that triggered the leave.
        """
        world = self.world
        centre = np.asarray(world.body.centre(), dtype=float)
        pos = world.food.pos[mouth]
        d2 = ((pos - centre[None, :]) ** 2).sum(axis=1)
        particle = int(mouth[int(np.argmin(d2))])
        self.leaving = True
        self.leave_patch = particle // world.food.npp

    def update(self) -> None:
        world = self.world
        energy = float(world.energy)
        mouth = self._mouth_food()
        has_mouth = mouth.size > 0
        if energy < self.eat_below:
            self.leaving = False
            self.leave_patch = None
            if has_mouth:
                self.mode = "hold"
            else:
                patch = self._nearest_patch()
                if patch is None:
                    self.mode = "idle"
                else:
                    self.direction = self._swim_direction(world.food.centres[patch])
                    self.mode = "swim"
        elif self.leaving or (energy > self.leave_above and has_mouth):
            if not self.leaving:
                self._start_leave(mouth)
            if has_mouth:
                centre = np.asarray(world.body.centre(), dtype=float)
                patch_centre = world.food.centres[self.leave_patch]
                dx = centre[0] - patch_centre[0]
                dy = centre[1] - patch_centre[1]
                if dx * dx + dy * dy < 1e-12:
                    pos = world.food.pos[mouth]
                    d2 = ((pos - centre[None, :]) ** 2).sum(axis=1)
                    nearest = pos[int(np.argmin(d2))]
                    dx = centre[0] - nearest[0]
                    dy = centre[1] - nearest[1]
                away_world = math.atan2(dy, dx)
                self.direction = _wrap(away_world - world.body.heading())
                self.mode = "swim"
            else:
                self.leaving = False
                self.leave_patch = None
                self.mode = "idle"
        else:
            self.leaving = False
            self.leave_patch = None
            self.mode = "idle"

    def targets(self, t: float):
        if self.mode == "hold":
            return None
        if self.mode == "idle":
            return stop_targets()
        direction = self.direction + (math.pi if self.reverse else 0.0)
        return swim_targets(direction, t)


class SmellForager:
    """Readings-only policy: arm smell gives the direction, energy the state.

    Food inside the mouth radius is not a reading, so the leaving rule uses
    the only readable trace of eating: energy rising between two updates.
    """

    def __init__(self, read, set_point: float, eat_below: float = 0.78,
                 leave_above: float = 0.82, smell_min: float = 0.2,
                 seed: int = 0, reverse: bool = False, n_arms: int = N_ARMS) -> None:
        self.read = read
        self.set_point = float(set_point)
        self.eat_below = float(eat_below)
        self.leave_above = float(leave_above)
        self.smell_min = float(smell_min)
        self.reverse = bool(reverse)
        self.n_arms = int(n_arms)
        self.rng = np.random.default_rng(int(seed))
        self.mode = "idle"
        self.direction = 0.0
        self.leaving = False
        self.prev_energy = None
        self.prev_touch = False
        self.last_arm = None

    def update(self) -> None:
        reading = self.read()
        need = float(reading["energy.need"])
        energy = self.set_point - need
        smells = [float(reading[f"arm{k}.smell"]) for k in range(self.n_arms)]
        touches = [float(reading[f"arm{k}.touch"]) for k in range(self.n_arms)]
        any_touch = any(t == 1.0 for t in touches)
        arm = int(np.argmax(smells))
        strongest = smells[arm]
        eating = self.prev_energy is not None and energy > self.prev_energy

        if energy < self.eat_below:
            self.leaving = False
            self.last_arm = arm
            if strongest >= self.smell_min:
                self.direction = _wrap(arm * ARM_ANGLE)
            elif any_touch and not self.prev_touch:
                self.direction = float(self.rng.uniform(-math.pi, math.pi))
            self.mode = "swim"
        elif self.leaving:
            if eating:
                if strongest >= self.smell_min:
                    self.direction = _wrap(arm * ARM_ANGLE + math.pi)
                self.mode = "swim"
            else:
                self.leaving = False
                self.mode = "idle"
        elif energy > self.leave_above and eating:
            self.leaving = True
            self.last_arm = arm
            if strongest >= self.smell_min:
                self.direction = _wrap(arm * ARM_ANGLE + math.pi)
            self.mode = "swim"
        else:
            self.mode = "idle"
        self.prev_energy = energy
        self.prev_touch = any_touch

    def targets(self, t: float):
        if self.mode == "hold":
            return None
        if self.mode == "idle":
            return stop_targets()
        direction = self.direction + (math.pi if self.reverse else 0.0)
        return swim_targets(direction, t)


class BlindSwimmer:
    """Blind reference arm: the foragers' gait steered at random.

    Reads nothing at all: no reading, no energy, no position, and no time
    except its own update count.  It holds one steer command for a
    duration drawn from an exponential distribution with mean ``hold_s``,
    then redraws the command uniformly over the gait's full steer range,
    [-pi, pi).

    One ``update()`` call is one step of ``control_s`` world seconds.  The
    exponential hold is counted down in seconds, ``control_s`` per
    update; the class itself never reads the world clock.  The waveform
    phase in ``targets`` comes from the runner's clock exactly as it does
    for the forager classes; no decision here depends on it.
    """

    def __init__(self, hold_s: float = 10.0, seed: int = 0, control_s: float = 0.1) -> None:
        if not hold_s > 0.0:
            raise ValueError(f"hold_s must be positive, got {hold_s!r}")
        if not control_s > 0.0:
            raise ValueError(f"control_s must be positive, got {control_s!r}")
        self.hold_s = float(hold_s)
        self.control_s = float(control_s)
        self.rng = np.random.default_rng(int(seed))
        self.hold_left_s = 0.0
        self.direction = 0.0
        self._redraw()

    def _redraw(self) -> None:
        self.direction = float(self.rng.uniform(-math.pi, math.pi))
        self.hold_left_s = float(self.rng.exponential(self.hold_s))

    def update(self) -> None:
        self.hold_left_s -= self.control_s
        if self.hold_left_s <= 0.0:
            self._redraw()

    def targets(self, t: float):
        return swim_targets(self.direction, t)


class BlindGatedSwimmer(BlindSwimmer):
    """Blind swimmer plus only the energy feeling: rest while full.

    Same steering as BlindSwimmer.  The two levels and the hysteresis are
    the foragers': hungry again below ``eat_below``, full again above
    ``leave_above``, the state kept in between.  While full every target
    is held at the gait's resting pose, which for the foragers is all
    targets zero (their "idle" mode, ``stop_targets()``).
    """

    def __init__(self, read, set_point: float, hold_s: float = 10.0,
                 seed: int = 0, control_s: float = 0.1,
                 eat_below: float = 0.78, leave_above: float = 0.82) -> None:
        super().__init__(hold_s=hold_s, seed=seed, control_s=control_s)
        self.read = read
        self.set_point = float(set_point)
        self.eat_below = float(eat_below)
        self.leave_above = float(leave_above)
        self.hungry = False

    def update(self) -> None:
        energy = self.set_point - float(self.read()["energy.need"])
        if energy < self.eat_below:
            self.hungry = True
        elif energy > self.leave_above:
            self.hungry = False
        if self.hungry:
            super().update()

    def targets(self, t: float):
        if not self.hungry:
            return stop_targets()
        return super().targets(t)
