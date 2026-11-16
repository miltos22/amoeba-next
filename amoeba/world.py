"""Stage-0 world: a round dish of thick fluid, food patches, smell, eating.

One simulated organism lives one unbroken life here.  The world owns the
MuJoCo model (through its body), the food field, energy and world time.
Nothing in it steers the body: it only supplies consequences.
"""

from __future__ import annotations

import copy
import heapq
import math
import pickle
from dataclasses import dataclass, field, fields

import mujoco
import numpy as np

from .body import Body, BodyShape, make_body
from .feelings import PRESETS, Preset, clamp_energy, resolve_preset

# Slack used when a gut entry's due time is compared with the food time.
# Food time accumulates as a sum of floats, so a delay that is a whole
# number of food steps can land a few ulps below the accumulated clock.
_GUT_TIME_EPS = 1e-9

_SNAPSHOT_FORMAT = "amoeba-world-snapshot/1"


@dataclass(frozen=True, slots=True)
class FoodKind:
    """One kind of food: when its energy arrives, how much, how much of the dish."""

    name: str
    delay_s: float
    gain: float
    share: float


@dataclass
class Config:
    """Everything that is fixed at the start of a life."""

    seed: int = 0
    timestep: float = 0.002
    preset: str = "lean"
    weakness: bool = True
    initial_energy: float = 1.0

    food_kinds: tuple[FoodKind, ...] = (FoodKind("plain", 0.0, 1.0, 1.0),)
    gut_sense: bool = False

    dish_radius: float = 1.5
    n_wall_planes: int = 32

    n_arms: int = 6
    joints_per_arm: int = 2
    center_radius: float = 0.15
    arm_radius: float = 0.035
    segment_length: float = 0.15
    center_mass: float = 1.0
    link_mass: float = 0.15
    joint_limit: float = 2.6
    joint_damping: float = 0.02
    joint_armature: float = 0.002
    kp: float = 2.0
    kv: float = 0.2
    force_limit: float = 2.0
    fluid_density: float = 1000.0
    fluid_viscosity: float = 1.0

    mouth_radius: float = 0.08
    smell_length: float = 0.35

    n_patches: int = 12
    fixed_fraction: float = 0.5
    drift_interval_s: float = 60.0
    patch_radius: float = 0.25
    particles_per_patch: int = 20
    patch_capacity: float = 3.0
    regrowth_tau_s: float = 120.0

    food_step_interval_s: float = 0.01
    food_record_interval_s: float = 1.0

    rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.timestep <= 0.0:
            raise ValueError("timestep must be positive")
        if self.dish_radius <= 0.0:
            raise ValueError("dish_radius must be positive")
        if self.n_arms < 1 or self.joints_per_arm < 1:
            raise ValueError("n_arms and joints_per_arm must be at least 1")
        if self.n_patches < 0 or self.particles_per_patch < 1:
            raise ValueError("need n_patches >= 0 and particles_per_patch >= 1")
        if self.patch_radius <= 0.0 or self.regrowth_tau_s <= 0.0:
            raise ValueError("patch_radius and regrowth_tau_s must be positive")
        if not 0.0 <= self.fixed_fraction <= 1.0:
            raise ValueError("fixed_fraction must be in [0, 1]")
        if self.food_step_interval_s <= 0.0 or self.food_record_interval_s <= 0.0:
            raise ValueError("food intervals must be positive")
        if self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}; known: {sorted(PRESETS)}")
        kinds = self.food_kinds
        if len(kinds) < 1:
            raise ValueError("at least one food kind is required")
        names = [kind.name for kind in kinds]
        if len(set(names)) != len(names):
            raise ValueError(f"food kind names must be unique, got {names!r}")
        for kind in kinds:
            if not isinstance(kind, FoodKind):
                raise ValueError(f"food_kinds must hold FoodKind values, got {kind!r}")
            if not math.isfinite(kind.delay_s) or kind.delay_s < 0.0:
                raise ValueError(f"food kind {kind.name!r} delay_s must be finite and >= 0")
            if not math.isfinite(kind.gain):
                raise ValueError(f"food kind {kind.name!r} gain must be finite")
            if not math.isfinite(kind.share) or kind.share <= 0.0:
                raise ValueError(f"food kind {kind.name!r} share must be finite and > 0")
        total_share = math.fsum(kind.share for kind in kinds)
        if abs(total_share - 1.0) > 1e-9:
            raise ValueError(f"food kind shares must sum to 1, got {total_share!r}")
        self.rng = np.random.default_rng(self.seed)

    @property
    def preset_constants(self) -> Preset:
        return PRESETS[self.preset]

    def as_dict(self) -> dict:
        out = {}
        for f in fields(self):
            if f.name == "rng":
                continue
            out[f.name] = getattr(self, f.name)
        out["food_kinds"] = [
            {
                "name": kind.name,
                "delay_s": kind.delay_s,
                "gain": kind.gain,
                "share": kind.share,
            }
            for kind in self.food_kinds
        ]
        out["preset_constants"] = {
            "basal_rate": self.preset_constants.basal_rate,
            "effort_rate": self.preset_constants.effort_rate,
            "intake_rate": self.preset_constants.intake_rate,
            "strength_floor": self.preset_constants.strength_floor,
            "energy_set_point": self.preset_constants.energy_set_point,
        }
        return out


class FoodField:
    """Food particles in patches that are eaten down and regrow."""

    def __init__(self, config: Config, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.n = int(config.n_patches)
        self.npp = int(config.particles_per_patch)
        self.cap_per_particle = config.patch_capacity / self.npp
        self.centres = np.zeros((self.n, 2))
        self.offsets = np.zeros((self.n, self.npp, 2))
        self.amounts = np.zeros((self.n, self.npp))
        order = rng.permutation(self.n) if self.n else np.array([], dtype=int)
        n_fixed = int(round(config.fixed_fraction * self.n))
        self.fixed = np.zeros(self.n, dtype=bool)
        if n_fixed:
            self.fixed[order[:n_fixed]] = True
        for i in range(self.n):
            self._place(i, refill=True)
        self.kind = self._assign_kinds()
        self._rebuild()
        self.time = 0.0
        self.next_drift = float(config.drift_interval_s)

    def _kind_counts(self) -> list[int]:
        """Largest-remainder rounding of share x n_patches, ties by kind order."""
        n = self.n
        shares = [kind.share for kind in self.config.food_kinds]
        quotas = [share * n for share in shares]
        counts = [int(math.floor(q)) for q in quotas]
        left = n - sum(counts)
        order = sorted(range(len(shares)), key=lambda i: (-(quotas[i] - counts[i]), i))
        for i in order[:left]:
            counts[i] += 1
        return counts

    def _assign_kinds(self) -> np.ndarray:
        """One kind per patch for its whole life; the default kind draws nothing."""
        n_kinds = len(self.config.food_kinds)
        kind = np.zeros(self.n, dtype=int)
        if n_kinds == 1 or self.n == 0:
            return kind
        counts = self._kind_counts()
        order = self.rng.permutation(self.n)
        start = 0
        for k, count in enumerate(counts):
            kind[order[start:start + count]] = k
            start += count
        return kind

    def _random_centre(self) -> np.ndarray:
        maxr = max(0.0, self.config.dish_radius - self.config.patch_radius)
        r = math.sqrt(float(self.rng.random())) * maxr
        theta = 2.0 * math.pi * float(self.rng.random())
        return np.array([r * math.cos(theta), r * math.sin(theta)])

    def _random_offsets(self) -> np.ndarray:
        r = np.sqrt(self.rng.random(self.npp)) * self.config.patch_radius
        theta = 2.0 * math.pi * self.rng.random(self.npp)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    def _place(self, i: int, refill: bool) -> None:
        self.centres[i] = self._random_centre()
        self.offsets[i] = self._random_offsets()
        if refill:
            self.amounts[i] = self.cap_per_particle

    def _rebuild(self) -> None:
        if self.n == 0:
            self.pos = np.zeros((0, 2))
        else:
            self.pos = (self.centres[:, None, :] + self.offsets).reshape(-1, 2)
        self.amt = self.amounts.reshape(-1)
        self.cap = np.full(self.amt.shape, self.cap_per_particle)
        self.particle_kind = np.repeat(self.kind, self.npp)

    def _drift(self) -> None:
        if self.n == 0 or self.config.drift_interval_s <= 0.0:
            return
        moved = False
        while self.time >= self.next_drift:
            for i in np.flatnonzero(~self.fixed):
                self.centres[i] = self._random_centre()
                self.offsets[i] = self._random_offsets()
            self.next_drift += float(self.config.drift_interval_s)
            moved = True
        if moved:
            self._rebuild()

    def step(
        self,
        dt: float,
        mouth_xy,
        intake_rate: float,
        energy: float,
        gut_load: float = 0.0,
    ) -> tuple[float, np.ndarray]:
        """Regrow, let drifting patches jump, eat what is within the mouth.

        Returns the food (energy units) taken in this interval and the part
        of it that belongs to each kind, in proportion to what each kind
        had inside the mouth radius.
        """
        n_kinds = len(self.config.food_kinds)
        self.time += dt
        self._drift()
        if self.n == 0:
            return 0.0, np.zeros(n_kinds)
        self.amt += (self.cap - self.amt) * (1.0 - math.exp(-dt / self.config.regrowth_tau_s))
        dx = self.pos[:, 0] - float(mouth_xy[0])
        dy = self.pos[:, 1] - float(mouth_xy[1])
        mask = (dx * dx + dy * dy) < self.config.mouth_radius**2
        take = 0.0
        by_kind = np.zeros(n_kinds)
        if bool(mask.any()):
            avail = float(self.amt[mask].sum())
            want = intake_rate * max(0.0, 1.0 - energy - gut_load) * dt
            take = want if want < avail else avail
            if take > 0.0:
                if n_kinds == 1:
                    by_kind[0] = take
                else:
                    avail_by_kind = np.bincount(
                        self.particle_kind[mask], weights=self.amt[mask], minlength=n_kinds
                    )
                    by_kind = np.where(avail_by_kind > 0.0, take * avail_by_kind / avail, 0.0)
                self.amt[mask] *= 1.0 - take / avail
        return take, by_kind

    def pin_patch(self, i: int, point) -> None:
        """Test hook: hold patch i at a point, full."""
        if self.n == 0:
            return
        self.centres[i] = np.asarray(point, dtype=float)
        self.offsets[i] = 0.0
        self.amounts[i] = self.cap_per_particle
        self._rebuild()

    def smell(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if self.amt.size == 0:
            return np.zeros(pts.shape[0])
        d = np.sqrt(((pts[:, None, :] - self.pos[None, :, :]) ** 2).sum(axis=-1))
        return np.exp(-d / self.config.smell_length) @ self.amt

    def smell_kinds(self, points) -> np.ndarray:
        """Smell per kind: one column per kind, each from its own particles."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        n_kinds = len(self.config.food_kinds)
        if n_kinds == 1:
            return self.smell(pts)[:, None]
        out = np.zeros((pts.shape[0], n_kinds))
        if self.amt.size == 0:
            return out
        d = np.sqrt(((pts[:, None, :] - self.pos[None, :, :]) ** 2).sum(axis=-1))
        base = np.exp(-d / self.config.smell_length)
        for k in range(n_kinds):
            mask = self.particle_kind == k
            if bool(mask.any()):
                out[:, k] = base[:, mask] @ self.amt[mask]
        return out

    def totals(self) -> np.ndarray:
        return self.amounts.sum(axis=1)

    def centres_list(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in self.centres]


class World:
    """The dish, its one body, its food and their shared clock."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = Config() if config is None else config
        self.preset = resolve_preset(self.config.preset)
        self.rng = self.config.rng
        self.energy = float(clamp_energy(self.config.initial_energy))
        self._steps = 0
        self._leftover = 0.0
        self._effort_sample = 0.0
        self._steps_since_food = 0
        self._food_every = max(1, int(round(self.config.food_step_interval_s / self.config.timestep)))
        self.food = FoodField(self.config, self.rng)
        n_kinds = len(self.config.food_kinds)
        self.food_eaten = 0.0
        self.food_eaten_by_kind = np.zeros(n_kinds)
        self.food_delivered = 0.0
        self.food_delivered_amount = 0.0
        self.food_delivered_by_kind = np.zeros(n_kinds)
        self.gut: list[tuple[float, int, float]] = []
        self._gut_load = 0.0
        self._hold_food = False
        self.recorder = None
        self.body = self._make_body(self.config.n_arms, self.config.joints_per_arm)

    # -- construction ------------------------------------------------------
    def _make_body(self, n_arms: int, joints_per_arm: int) -> Body:
        cfg = self.config
        shape = BodyShape(
            center_radius=cfg.center_radius,
            arm_radius=cfg.arm_radius,
            segment_length=cfg.segment_length,
            center_mass=cfg.center_mass,
            link_mass=cfg.link_mass,
            joint_limit=cfg.joint_limit,
            joint_damping=cfg.joint_damping,
            joint_armature=cfg.joint_armature,
            kp=cfg.kp,
            kv=cfg.kv,
            force_limit=cfg.force_limit,
        )
        body = make_body(
            n_arms,
            joints_per_arm,
            dish_radius=cfg.dish_radius,
            timestep=cfg.timestep,
            fluid_density=cfg.fluid_density,
            fluid_viscosity=cfg.fluid_viscosity,
            n_wall_planes=cfg.n_wall_planes,
            shape=shape,
            smell_names=tuple(kind.name for kind in cfg.food_kinds),
            gut_sense=cfg.gut_sense,
            smell_bound=max(1.0, cfg.n_patches * cfg.patch_capacity),
            preset=cfg.preset,
            weakness=cfg.weakness,
            initial_energy=self.energy,
        )
        body.smell_at = self.food.smell
        body.smell_at_kinds = self.food.smell_kinds
        return body

    # -- item interface ----------------------------------------------------
    def items(self):
        return self.body.items()

    @property
    def gut_load(self) -> float:
        """Amount of food (energy units) waiting in the gut.

        Kept as a running sum (in gut order of removal) so a large delay
        does not cost a scan of the queue at every food step.
        """
        return self._gut_load

    def read(self) -> dict[str, float]:
        self.body.set_energy(self.energy)
        self.body.gut_load = self.gut_load
        return self.body.read()

    def write(self, targets: dict[str, float]) -> None:
        self.body.write(targets)

    def targets(self) -> dict[str, float]:
        return self.body.targets()

    # -- clock and physics -------------------------------------------------
    @property
    def time(self) -> float:
        return self._steps * self.config.timestep

    def advance(self, seconds: float) -> None:
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError(f"advance needs a finite, non-negative time, got {seconds!r}")
        total = self._leftover + seconds
        n = int(math.floor(total / self.config.timestep + 0.5))
        self._leftover = total - n * self.config.timestep
        for _ in range(n):
            self._step_once()
        if self.recorder is not None:
            self.recorder.note()

    def _step_once(self) -> None:
        body = self.body
        if self._steps_since_food == 0:
            body.set_energy(self.energy)
        body.apply()
        mujoco.mj_step(body.model, body.data)
        self._steps += 1
        self._steps_since_food += 1
        if self._steps_since_food >= self._food_every:
            self._effort_sample = body.effort()
            self._update_food()

    def _update_food(self) -> None:
        cfg = self.config
        dtc = cfg.timestep * self._food_every
        centre = self.body.centre()
        if self._hold_food and self.food.n > 0:
            self.food.pin_patch(0, centre)
        taken, taken_by_kind = self.food.step(
            dtc, centre, self.preset.intake_rate, self.energy, self.gut_load
        )
        food_time = self.food.time
        for k, amount in enumerate(taken_by_kind):
            if amount > 0.0:
                amount = float(amount)
                heapq.heappush(self.gut, (food_time + cfg.food_kinds[k].delay_s, k, amount))
                self._gut_load += amount
        delivered = 0.0
        delivered_amount = 0.0
        delivered_by_kind = np.zeros(len(cfg.food_kinds))
        while self.gut and self.gut[0][0] <= food_time + _GUT_TIME_EPS:
            _due_time, k, amount = heapq.heappop(self.gut)
            value = cfg.food_kinds[k].gain * amount
            delivered += value
            delivered_amount += amount
            delivered_by_kind[k] += value
            self._gut_load -= amount
        self.food_eaten += taken
        self.food_eaten_by_kind += taken_by_kind
        self.food_delivered += delivered
        self.food_delivered_amount += delivered_amount
        self.food_delivered_by_kind += delivered_by_kind
        self._steps_since_food = 0
        drain = dtc * (self.preset.basal_rate + self.preset.effort_rate * self._effort_sample)
        self.energy = clamp_energy(self.energy + delivered - drain)

    # -- snapshot and restore ----------------------------------------------
    def snapshot(self) -> dict:
        """Everything needed to continue this world bit for bit.

        The physics state is MuJoCo's own ``mjSTATE_INTEGRATION``: it holds
        the time, positions, velocities, actuator activations, history
        buffers, warmstart acceleration, controls, applied forces,
        equality/mocap state, user data, plugin state and plastic state.
        That is the documented state needed to resume integration; plain
        ``mjSTATE_PHYSICS`` would leave the solver warmstart out.
        """
        model = self.body.model
        data = self.body.data
        sig = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(mujoco.mj_stateSize(model, sig), dtype=np.float64)
        mujoco.mj_getState(model, data, state, sig)
        return {
            "format": _SNAPSHOT_FORMAT,
            "config": self.config.as_dict(),
            "body_shape": {
                "n_arms": self.body.n_arms,
                "joints_per_arm": self.body.joints_per_arm,
            },
            "mujoco_state": state,
            "mujoco_state_sig": sig,
            "targets": self.body.targets(),
            "energy": self.energy,
            "steps": self._steps,
            "leftover": self._leftover,
            "effort_sample": self._effort_sample,
            "steps_since_food": self._steps_since_food,
            "hold_food": self._hold_food,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "food": {
                "time": self.food.time,
                "next_drift": self.food.next_drift,
                "centres": self.food.centres.copy(),
                "offsets": self.food.offsets.copy(),
                "amounts": self.food.amounts.copy(),
                "fixed": self.food.fixed.copy(),
                "kind": self.food.kind.copy(),
            },
            "food_eaten": self.food_eaten,
            "food_eaten_by_kind": self.food_eaten_by_kind.copy(),
            "food_delivered": self.food_delivered,
            "food_delivered_amount": self.food_delivered_amount,
            "food_delivered_by_kind": self.food_delivered_by_kind.copy(),
            "gut": list(self.gut),
            "gut_load": self._gut_load,
        }

    def restore(self, snap: dict) -> None:
        """Put this world back into the state of ``snap``.

        Raises ValueError (and changes nothing) if the snapshot is not a
        world snapshot, if its Config differs from this world's, or if its
        body shape (n_arms, joints_per_arm) differs.  An open recording
        file is not rewound: rows already written stay written, and rows
        after the restore continue from the restored clock.
        """
        if not isinstance(snap, dict) or snap.get("format") != _SNAPSHOT_FORMAT:
            raise ValueError("not an amoeba world snapshot")
        if snap["config"] != self.config.as_dict():
            raise ValueError("snapshot Config differs from this world's Config")
        shape = snap["body_shape"]
        if (int(shape["n_arms"]), int(shape["joints_per_arm"])) != (
            self.body.n_arms,
            self.body.joints_per_arm,
        ):
            raise ValueError(
                "snapshot body shape differs from this world's body: "
                f"snapshot {shape['n_arms']}x{shape['joints_per_arm']}, "
                f"this body {self.body.n_arms}x{self.body.joints_per_arm}"
            )
        sig = int(snap["mujoco_state_sig"])
        if sig != int(mujoco.mjtState.mjSTATE_INTEGRATION):
            raise ValueError(f"unsupported snapshot state signature {sig!r}")

        food = snap["food"]
        self.food.centres = np.array(food["centres"], dtype=float, copy=True)
        self.food.offsets = np.array(food["offsets"], dtype=float, copy=True)
        self.food.amounts = np.array(food["amounts"], dtype=float, copy=True)
        self.food.fixed = np.array(food["fixed"], dtype=bool, copy=True)
        self.food.kind = np.array(food["kind"], dtype=int, copy=True)
        self.food.time = float(food["time"])
        self.food.next_drift = float(food["next_drift"])
        self.food._rebuild()

        self.energy = float(snap["energy"])
        self._steps = int(snap["steps"])
        self._leftover = float(snap["leftover"])
        self._effort_sample = float(snap["effort_sample"])
        self._steps_since_food = int(snap["steps_since_food"])
        self._hold_food = bool(snap["hold_food"])
        self.food_eaten = float(snap["food_eaten"])
        self.food_eaten_by_kind = np.array(snap["food_eaten_by_kind"], dtype=float, copy=True)
        self.food_delivered = float(snap["food_delivered"])
        self.food_delivered_amount = float(snap["food_delivered_amount"])
        self.food_delivered_by_kind = np.array(
            snap["food_delivered_by_kind"], dtype=float, copy=True
        )
        self.gut = list(snap["gut"])
        heapq.heapify(self.gut)
        self._gut_load = float(snap["gut_load"])
        self.rng.bit_generator.state = copy.deepcopy(snap["rng_state"])

        self.body.write(dict(snap["targets"]))
        self.body.set_energy(self.energy, force=True)
        state = np.array(snap["mujoco_state"], dtype=np.float64)
        mujoco.mj_setState(self.body.model, self.body.data, state, sig)

    def save_snapshot(self, path) -> None:
        """Pickle a snapshot to ``path`` (write it under the project tree)."""
        with open(path, "wb") as file:
            pickle.dump(self.snapshot(), file)

    def load_snapshot(self, path) -> None:
        """Read a snapshot written by :meth:`save_snapshot` and restore it."""
        with open(path, "rb") as file:
            snap = pickle.load(file)
        self.restore(snap)

    # -- hooks and changes -------------------------------------------------
    def hold_food_at_mouth(self, enabled: bool = True) -> None:
        """Test hook (4.4): keep a full patch at the mouth."""
        self._hold_food = bool(enabled)

    def change_body(self, n_arms: int, joints_per_arm: int) -> None:
        """Replace the body in place; time and energy are untouched (3.3.4)."""
        old = self.body
        new = self._make_body(n_arms, joints_per_arm)
        root_q, root_v = old.root_state()
        new.set_root_state(root_q, root_v)
        for name, (angle, velocity) in old.joint_state().items():
            if name in new.joint_names:
                new.set_joint_state(name, angle, velocity)
        old_targets = old.targets()
        keep = {name: value for name, value in old_targets.items() if name in new.actuator_names}
        new.write(keep)
        new.set_energy(self.energy, force=True)
        mujoco.mj_forward(new.model, new.data)
        self.body = new

    def enable_recording(self, path):
        from .record import Recorder

        self.recorder = Recorder(path, self)
        return self.recorder

    def disable_recording(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
