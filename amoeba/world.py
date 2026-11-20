"""Stage-0 world: a round dish of thick fluid, food patches, smell, eating.

One simulated organism lives one unbroken life here.  The world owns the
MuJoCo model (through its body), the food field, energy and world time.
Nothing in it steers the body: it only supplies consequences.
"""

from __future__ import annotations

import bisect
import copy
import heapq
import math
import pickle
from dataclasses import dataclass, field, fields

import mujoco
import numpy as np

from .body import Body, BodyShape, make_body
from .feelings import PRESETS, Preset, clamp_energy, fatigue_step, resolve_preset
from .items import Item

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
    regrowth_mode: str = "exp"
    regrowth_seed: float = 1e-3

    fatigue_weight: float = 0.0
    # None retains the legacy weakening-dependent guard; False freezes fatigue.
    fatigue_enabled: bool | None = None
    fatigue_threshold: float = 0.5
    fatigue_gain_per_s: float = 0.02
    fatigue_tau_s: float = 600.0
    fatigue_sense: bool = False

    pulse_patches: int = 0
    pulse_period_s: float = 600.0
    pulse_on_s: float = 60.0
    pulse_phase_s: float = 0.0

    valves_enabled: bool = False
    valve_positions: tuple = ((1.0, 0.0), (-1.0, 0.0))
    valve_radius: float = 0.12
    valve_effective: int = 0
    valve_delay_s: float = 300.0
    valve_initial_open: tuple = (True, True)
    valve_mode: str = "toggle"
    valve_pulse_s: float = 300.0
    valve_pulse_effect: str = "stop"

    patch_positions: tuple | None = None

    sense_mode: str = "instant"
    holding_noise: float = 0.0
    need_split: bool = False

    food_step_interval_s: float = 0.01
    food_record_interval_s: float = 1.0

    rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.fatigue_enabled is not None and type(self.fatigue_enabled) is not bool:
            raise TypeError("fatigue_enabled must be bool or None")
        if self.sense_mode not in ("instant", "holding"):
            raise ValueError(
                f"sense_mode must be 'instant' or 'holding', got {self.sense_mode!r}"
            )
        if not math.isfinite(self.holding_noise) or self.holding_noise < 0.0:
            raise ValueError("holding_noise must be finite and >= 0")
        if self.preset == "valve_room":
            # The preset means exactly this room: one pinned patch and valves on.
            self.n_patches = 1
            self.fixed_fraction = 1.0
            self.valves_enabled = True
            if self.patch_positions is None:
                self.patch_positions = ((0.0, 1.0),)
        if not math.isfinite(self.valve_radius) or self.valve_radius <= 0.0:
            raise ValueError("valve_radius must be finite and > 0")
        if not math.isfinite(self.valve_delay_s) or self.valve_delay_s < 0.0:
            raise ValueError("valve_delay_s must be finite and >= 0")
        if self.valve_mode not in ("toggle", "pulse"):
            raise ValueError(
                f"valve_mode must be 'toggle' or 'pulse', got {self.valve_mode!r}"
            )
        if not math.isfinite(self.valve_pulse_s) or self.valve_pulse_s <= 0.0:
            raise ValueError("valve_pulse_s must be finite and > 0")
        if self.valve_pulse_effect not in ("stop", "empty"):
            raise ValueError(
                "valve_pulse_effect must be 'stop' or 'empty', got "
                f"{self.valve_pulse_effect!r}"
            )
        if type(self.need_split) is not bool:
            raise TypeError("need_split must be bool")
        if self.valves_enabled:
            positions = tuple(
                tuple(float(value) for value in point) for point in self.valve_positions
            )
            if len(positions) != 2:
                raise ValueError(
                    f"exactly two valve positions are needed, got {len(positions)}"
                )
            for x, y in positions:
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError(f"valve positions must be finite, got {(x, y)!r}")
                if math.hypot(x, y) > self.dish_radius:
                    raise ValueError(
                        f"valve position {(x, y)!r} is outside the dish of radius "
                        f"{self.dish_radius!r}"
                    )
            if len(self.valve_initial_open) != len(positions):
                raise ValueError(
                    "valve_initial_open must have one entry per valve position"
                )
            if any(type(open_) is not bool for open_ in self.valve_initial_open):
                raise ValueError("valve_initial_open entries must be bool")
            if type(self.valve_effective) is not int or not (
                0 <= self.valve_effective < len(positions)
            ):
                raise ValueError(
                    "valve_effective must be an index into valve_positions, got "
                    f"{self.valve_effective!r}"
                )
            self.valve_positions = positions
            self.valve_initial_open = tuple(self.valve_initial_open)
        if self.patch_positions is not None:
            points = tuple(
                (float(point[0]), float(point[1])) for point in self.patch_positions
            )
            for x, y in points:
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError(f"patch positions must be finite, got {(x, y)!r}")
            self.patch_positions = points
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
        if self.regrowth_mode not in ("exp", "logistic"):
            raise ValueError(
                f"regrowth_mode must be 'exp' or 'logistic', got {self.regrowth_mode!r}"
            )
        if self.regrowth_mode == "logistic":
            if not math.isfinite(self.regrowth_seed) or self.regrowth_seed <= 0.0:
                raise ValueError("regrowth_seed must be finite and > 0 in logistic mode")
        if not 0.0 <= self.fatigue_weight < 1.0:
            raise ValueError("fatigue_weight must be in [0, 1)")
        if not math.isfinite(self.fatigue_threshold) or self.fatigue_threshold < 0.0:
            raise ValueError("fatigue_threshold must be finite and >= 0")
        if not math.isfinite(self.fatigue_gain_per_s) or self.fatigue_gain_per_s < 0.0:
            raise ValueError("fatigue_gain_per_s must be finite and >= 0")
        if not math.isfinite(self.fatigue_tau_s) or self.fatigue_tau_s <= 0.0:
            raise ValueError("fatigue_tau_s must be finite and > 0")
        if self.pulse_patches < 0:
            raise ValueError("pulse_patches must be >= 0")
        if not math.isfinite(self.pulse_period_s) or self.pulse_period_s <= 0.0:
            raise ValueError("pulse_period_s must be finite and > 0")
        if (
            not math.isfinite(self.pulse_on_s)
            or self.pulse_on_s <= 0.0
            or self.pulse_on_s >= self.pulse_period_s
        ):
            raise ValueError("pulse_on_s must satisfy 0 < pulse_on_s < pulse_period_s")
        if not math.isfinite(self.pulse_phase_s) or self.pulse_phase_s < 0.0:
            raise ValueError("pulse_phase_s must be finite and >= 0")
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
        self.n_ordinary = int(config.n_patches)
        self.n_pulse = int(config.pulse_patches)
        self.n = self.n_ordinary + self.n_pulse
        self.npp = int(config.particles_per_patch)
        self.cap_per_particle = config.patch_capacity / self.npp
        self.centres = np.zeros((self.n, 2))
        self.offsets = np.zeros((self.n, self.npp, 2))
        self.amounts = np.zeros((self.n, self.npp))
        order = (
            rng.permutation(self.n_ordinary)
            if self.n_ordinary
            else np.array([], dtype=int)
        )
        n_fixed = int(round(config.fixed_fraction * self.n_ordinary))
        self.fixed = np.zeros(self.n, dtype=bool)
        if n_fixed:
            self.fixed[order[:n_fixed]] = True
        pinned = self.config.patch_positions
        for i in range(self.n_ordinary):
            if pinned is not None and i < len(pinned):
                self._place_at(i, pinned[i], refill=True)
            else:
                self._place(i, refill=True)
        self.kind = self._assign_kinds()
        for i in range(self.n_ordinary, self.n):
            self._place(i, refill=False)
        self.fixed[self.n_ordinary:] = True
        self.kind[self.n_ordinary:] = 0
        self._rebuild()
        self.time = 0.0
        self.next_drift = float(config.drift_interval_s)
        self.pulse_time = 0.0
        self.pulse_on = False

    def _kind_counts(self) -> list[int]:
        """Largest-remainder rounding of share x n_patches, ties by kind order."""
        n = self.n_ordinary
        shares = [kind.share for kind in self.config.food_kinds]
        quotas = [share * n for share in shares]
        counts = [int(math.floor(q)) for q in quotas]
        left = n - sum(counts)
        order = sorted(range(len(shares)), key=lambda i: (-(quotas[i] - counts[i]), i))
        for i in order[:left]:
            counts[i] += 1
        return counts

    def _assign_kinds(self) -> np.ndarray:
        """One kind per ordinary patch for its whole life; the default kind draws nothing."""
        n_kinds = len(self.config.food_kinds)
        kind = np.zeros(self.n, dtype=int)
        if n_kinds == 1 or self.n_ordinary == 0:
            return kind
        counts = self._kind_counts()
        order = self.rng.permutation(self.n_ordinary)
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

    def _place_at(self, i: int, point, refill: bool) -> None:
        """Place a patch at a pinned centre; its scatter still draws the RNG."""
        self.centres[i] = np.asarray(point, dtype=float)
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
        regrow_allowed: bool = True,
    ) -> tuple[float, np.ndarray]:
        """Regrow, let drifting patches jump, eat what is within the mouth.

        Returns the food (energy units) taken in this interval and the part
        of it that belongs to each kind, in proportion to what each kind
        had inside the mouth radius.  ``regrow_allowed=False`` leaves every
        patch amount untouched (the valve gate) while eating still works.
        """
        n_kinds = len(self.config.food_kinds)
        self.time += dt
        self._drift()
        if self.n == 0:
            return 0.0, np.zeros(n_kinds)
        if self.n_pulse:
            self._pulse_update(dt)
        if regrow_allowed:
            self._regrow(dt)
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

    def _pulse_active(self, t: float) -> bool:
        """True when food time ``t`` is inside a pulse window, phase + k period."""
        if self.n_pulse == 0:
            return False
        phase = self.config.pulse_phase_s
        if t < phase:
            return False
        period = self.config.pulse_period_s
        k = math.floor((t - phase) / period)
        return (t - (phase + k * period)) < self.config.pulse_on_s

    def _pulse_update(self, dt: float) -> None:
        """Advance the pulse clock and set the pulse patches at a window edge."""
        self.pulse_time += dt
        active = self._pulse_active(self.pulse_time)
        if active != self.pulse_on:
            if active:
                self.amounts[self.n_ordinary:] = self.cap_per_particle
            else:
                self.amounts[self.n_ordinary:] = 0.0
            self.pulse_on = active

    def _logistic_x(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Exact step of dx/dt = (x + sigma) * (1 - x) / tau over ``dt``."""
        sigma = self.config.regrowth_seed
        tau = self.config.regrowth_tau_s
        full = x >= 1.0 - 1e-15
        safe = np.where(full, 0.0, x)
        u = (safe + sigma) / (1.0 - safe)
        u = u * math.exp(dt * (1.0 + sigma) / tau)
        x_new = 1.0 - (1.0 + sigma) / (1.0 + u)
        x_new[full] = 1.0
        return np.clip(x_new, 0.0, 1.0)

    def _regrow(self, dt: float) -> None:
        """Regrow the patches that are not switched-off pulse patches."""
        n_regrow = self.n if (self.n_pulse == 0 or self.pulse_on) else self.n_ordinary
        stop = n_regrow * self.npp
        amt = self.amt[:stop]
        cap = self.cap[:stop]
        if self.config.regrowth_mode == "logistic":
            amt[:] = cap * self._logistic_x(amt / cap, dt)
        else:
            amt += (cap - amt) * (1.0 - math.exp(-dt / self.config.regrowth_tau_s))

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
        self.fatigue = 0.0
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

        self._valve_xy = np.array(self.config.valve_positions, dtype=float)
        self._valve_open = [bool(open_) for open_ in self.config.valve_initial_open]
        self._valve_prev_touch = [False] * len(self._valve_open)
        self._valve_touched = [False] * len(self._valve_open)
        self._valve_toggle_times: list[list[float]] = [
            [] for _ in self._valve_open
        ]
        self._valve_geom_ids: np.ndarray | None = None
        self._valve_half = 0.5 * self.config.segment_length
        self._regrowth_allowed = True
        self._pulse_stop_active = False
        if self.config.valves_enabled:
            self._init_valve_geometry()
            mujoco.mj_kinematics(self.body.model, self.body.data)

        self._hold_acc: dict[str, float] | None = (
            {} if self.config.sense_mode == "holding" else None
        )
        self._hold_prev: dict[str, float] | None = None
        self._hold_last_time = 0.0
        self._noise_rng = np.random.default_rng((int(self.config.seed), 7919))
        self._last_noise: dict[str, float] = {}

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
            smell_bound=max(
                1.0, (cfg.n_patches + cfg.pulse_patches) * cfg.patch_capacity
            ),
            preset=cfg.preset,
            weakness=cfg.weakness,
            initial_energy=self.energy,
        )
        body.smell_at = self.food.smell
        body.smell_at_kinds = self.food.smell_kinds
        return body

    def _init_valve_geometry(self) -> None:
        model = self.body.model
        self._valve_geom_ids = np.array(
            [
                model.geom(f"arm{k}.l{m + 1}").id
                for k in range(self.body.n_arms)
                for m in range(self.body.joints_per_arm)
            ],
            dtype=int,
        )
        self._valve_half = 0.5 * self.config.segment_length

    # -- item interface ----------------------------------------------------
    def items(self):
        items = self.body.items()
        if self.config.need_split:
            split = []
            for item in items:
                if item.name == "energy.need":
                    split.append(Item("energy.need_pos", "feeling", 0.0, 1.0, "need"))
                    split.append(Item("energy.need_neg", "feeling", 0.0, 1.0, "need"))
                else:
                    split.append(item)
            items = split
        if self.config.fatigue_sense:
            items.append(Item("tired", "sense", 0.0, 1.0, "fatigue"))
        if self.config.valves_enabled:
            for i in range(len(self._valve_xy)):
                items.append(Item(f"valve{i}_touch", "sense", 0.0, 1.0, "touch"))
                items.append(Item(f"valve{i}_smell", "sense", 0.0, 1.0, "smell"))
        return items

    @property
    def gut_load(self) -> float:
        """Amount of food (energy units) waiting in the gut.

        Kept as a running sum (in gut order of removal) so a large delay
        does not cost a scan of the queue at every food step.
        """
        return self._gut_load

    @property
    def effort_sample(self) -> float:
        """The body effort sampled at the most recent food step."""
        return self._effort_sample

    def _apply_strength(self, force: bool = False) -> None:
        """Put the current energy and fatigue into the actuators' strength."""
        cfg = self.config
        if cfg.fatigue_weight > 0.0:
            self.body.fatigue_mult = 1.0 - cfg.fatigue_weight * self.fatigue
        self.body.set_energy(self.energy, force=force)

    def _valve_segments(self) -> tuple[np.ndarray, np.ndarray]:
        """Segment endpoints of every arm capsule in the plane."""
        if self._valve_geom_ids is None:
            self._init_valve_geometry()
        data = self.body.data
        ids = self._valve_geom_ids
        pos = np.asarray(data.geom_xpos[ids], dtype=float)[:, :2]
        mat = np.asarray(data.geom_xmat[ids], dtype=float).reshape(-1, 3, 3)
        axis = mat[:, :, 2][:, :2]
        half = self._valve_half
        return pos - axis * half, pos + axis * half

    def _valve_touch_flags(self) -> list[bool]:
        """True for each valve when the centre or an arm segment is inside its radius."""
        centre = np.array(self.body.centre(), dtype=float)
        radius = self.config.valve_radius
        offsets = self._valve_xy - centre[None, :]
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
        flags = [bool(distance <= radius) for distance in distances]
        cfg = self.config
        reach = (
            cfg.center_radius
            + cfg.joints_per_arm * cfg.segment_length
            + cfg.arm_radius
            + radius
        )
        if all(distance > reach for distance in distances):
            return flags
        starts, ends = self._valve_segments()
        ab = ends - starts
        denom = (ab * ab).sum(axis=1)
        ap = self._valve_xy[:, None, :] - starts[None, :, :]
        dot = (ap * ab[None, :, :]).sum(axis=2)
        t = np.where(denom[None, :] > 0.0, dot / denom[None, :], 0.0)
        t = np.clip(t, 0.0, 1.0)
        closest = starts[None, :, :] + t[:, :, None] * ab[None, :, :]
        gaps = self._valve_xy[:, None, :] - closest
        segment_distances = np.sqrt((gaps * gaps).sum(axis=2))
        for i in range(len(flags)):
            if not flags[i] and bool(np.any(segment_distances[i] <= radius)):
                flags[i] = True
        return flags

    def _valve_senses(self) -> list[tuple[float, float]]:
        flags = self._valve_touch_flags()
        centre = np.array(self.body.centre(), dtype=float)
        length = self.config.smell_length
        out: list[tuple[float, float]] = []
        for i, valve in enumerate(self._valve_xy):
            distance = math.hypot(centre[0] - valve[0], centre[1] - valve[1])
            out.append((1.0 if flags[i] else 0.0, math.exp(-distance / length)))
        return out

    def _update_valves(self) -> None:
        touched = self._valve_touch_flags()
        for i, now in enumerate(touched):
            if now and not self._valve_prev_touch[i]:
                self._valve_open[i] = not self._valve_open[i]
                self._valve_toggle_times[i].append(self.time)
            self._valve_prev_touch[i] = now
        self._valve_touched = touched

    def _valve_delayed_open(self, index: int, when: float | None = None) -> bool:
        """The valve state ``valve_delay_s`` world seconds before ``when``."""
        moment = self.time if when is None else float(when)
        cutoff = moment - self.config.valve_delay_s
        toggles = self._valve_toggle_times[index]
        state = bool(self.config.valve_initial_open[index])
        if bisect.bisect_right(toggles, cutoff) % 2:
            state = not state
        return state

    def _valve_pulse_active(self, index: int, when: float | None = None) -> bool:
        """True when ``when`` lies inside a stop window made by one touch.

        A touch at ``t0`` stops regrowth during
        ``[t0 + valve_delay_s, t0 + valve_delay_s + valve_pulse_s]``.  The
        union of overlapping windows is a single stop stretch, so only the
        latest touch at or before ``when - valve_delay_s`` can matter: if it
        is older than one pulse, every earlier touch is older still.
        """
        moment = self.time if when is None else float(when)
        cutoff = moment - self.config.valve_delay_s
        touches = self._valve_toggle_times[index]
        i = bisect.bisect_right(touches, cutoff)
        if i == 0:
            return False
        return touches[i - 1] >= cutoff - self.config.valve_pulse_s

    @property
    def regrowth_allowed(self) -> bool:
        """The valve gate used at the most recent food step."""
        return self._regrowth_allowed

    def valve_states(self) -> tuple[bool, ...]:
        return tuple(self._valve_open)

    def valve_toggles(self, index: int) -> tuple[float, ...]:
        return tuple(self._valve_toggle_times[index])

    def holding_integral(self) -> dict[str, float] | None:
        """The running per-step integrals in holding mode (None in instant mode)."""
        return None if self._hold_acc is None else dict(self._hold_acc)

    @property
    def last_holding_noise(self) -> dict[str, float]:
        """The noise added at the most recent holding look (empty when off)."""
        return dict(self._last_noise)

    def _raw_reading(self) -> dict[str, float]:
        self._apply_strength()
        self.body.gut_load = self.gut_load
        out = self.body.read()
        if self.config.need_split:
            need = out.pop("energy.need")
            out["energy.need_pos"] = need if need > 0.0 else 0.0
            out["energy.need_neg"] = -need if need < 0.0 else 0.0
        if self.config.fatigue_sense:
            out["tired"] = float(self.fatigue)
        if self.config.valves_enabled:
            for i, (touch, proximity) in enumerate(self._valve_senses()):
                out[f"valve{i}_touch"] = touch
                out[f"valve{i}_smell"] = proximity
        return out

    def _accumulate_holding(self) -> None:
        raw = self._raw_reading()
        acc = self._hold_acc
        if not acc:
            acc.update({name: 0.0 for name in raw})
        dt = self.config.timestep
        for name, value in raw.items():
            acc[name] = acc.get(name, 0.0) + float(value) * dt

    def _holding_amounts(self, out: dict[str, float]) -> dict[str, float]:
        acc = self._hold_acc
        if self._hold_prev is None:
            self._hold_prev = {name: 0.0 for name in out}
        elapsed = self.time - self._hold_last_time
        if elapsed < 0.0:
            elapsed = 0.0
        sd = self.config.holding_noise * math.sqrt(elapsed)
        amounts: dict[str, float] = {}
        self._last_noise = {}
        for name in out:
            amount = acc.get(name, 0.0) - self._hold_prev.get(name, 0.0)
            if self.config.holding_noise > 0.0:
                draw = float(self._noise_rng.normal(0.0, sd))
                amount += draw
                self._last_noise[name] = draw
            amounts[name] = amount
        self._hold_prev = {name: acc.get(name, 0.0) for name in out}
        self._hold_last_time = self.time
        return amounts

    def read(self) -> dict[str, float]:
        out = self._raw_reading()
        if self._hold_acc is not None:
            out = self._holding_amounts(out)
        return out

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
            self._apply_strength()
        body.apply()
        mujoco.mj_step(body.model, body.data)
        self._steps += 1
        self._steps_since_food += 1
        if self.config.valves_enabled:
            self._update_valves()
        if self._steps_since_food >= self._food_every:
            self._effort_sample = body.effort()
            self._update_food()
        if self._hold_acc is not None:
            self._accumulate_holding()

    def _update_food(self) -> None:
        cfg = self.config
        dtc = cfg.timestep * self._food_every
        if (cfg.fatigue_weight > 0.0 if cfg.fatigue_enabled is None else cfg.fatigue_enabled):
            self.fatigue = fatigue_step(
                self.fatigue,
                self._effort_sample,
                dtc,
                cfg.fatigue_threshold,
                cfg.fatigue_gain_per_s,
                cfg.fatigue_tau_s,
            )
        centre = self.body.centre()
        if self._hold_food and self.food.n > 0:
            self.food.pin_patch(0, centre)
        if cfg.valves_enabled:
            if cfg.valve_mode == "pulse":
                active = self._valve_pulse_active(cfg.valve_effective)
                if (
                    active
                    and not self._pulse_stop_active
                    and cfg.valve_pulse_effect == "empty"
                ):
                    self.food.amounts[:] = 0.0
                    self.food._rebuild()
                self._pulse_stop_active = active
                allow = not active
            else:
                self._pulse_stop_active = False
                allow = self._valve_delayed_open(cfg.valve_effective)
        else:
            self._pulse_stop_active = False
            allow = True
        self._regrowth_allowed = allow
        taken, taken_by_kind = self.food.step(
            dtc,
            centre,
            self.preset.intake_rate,
            self.energy,
            self.gut_load,
            regrow_allowed=allow,
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
            "fatigue": self.fatigue,
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
                "pulse": {
                    "time": self.food.pulse_time,
                    "on": self.food.pulse_on,
                },
            },
            "food_eaten": self.food_eaten,
            "food_eaten_by_kind": self.food_eaten_by_kind.copy(),
            "food_delivered": self.food_delivered,
            "food_delivered_amount": self.food_delivered_amount,
            "food_delivered_by_kind": self.food_delivered_by_kind.copy(),
            "gut": list(self.gut),
            "gut_load": self._gut_load,
            "valves": {
                "open": list(self._valve_open),
                "prev_touch": list(self._valve_prev_touch),
                "touched": list(self._valve_touched),
                "toggle_times": [list(times) for times in self._valve_toggle_times],
                "regrowth_allowed": self._regrowth_allowed,
                "pulse_stop_active": self._pulse_stop_active,
            },
            "holding": {
                "acc": None if self._hold_acc is None else dict(self._hold_acc),
                "prev": None if self._hold_prev is None else dict(self._hold_prev),
                "last_time": self._hold_last_time,
                "noise_rng_state": copy.deepcopy(self._noise_rng.bit_generator.state),
                "last_noise": dict(self._last_noise),
            },
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
        pulse = food["pulse"]
        self.food.pulse_time = float(pulse["time"])
        self.food.pulse_on = bool(pulse["on"])
        self.food._rebuild()

        self.energy = float(snap["energy"])
        self.fatigue = float(snap["fatigue"])
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

        valves = snap.get("valves")
        if valves is not None:
            self._valve_open = [bool(open_) for open_ in valves["open"]]
            self._valve_prev_touch = [bool(t) for t in valves["prev_touch"]]
            self._valve_touched = [bool(t) for t in valves["touched"]]
            self._valve_toggle_times = [
                [float(time) for time in times] for times in valves["toggle_times"]
            ]
            self._regrowth_allowed = bool(valves["regrowth_allowed"])
            self._pulse_stop_active = bool(valves.get("pulse_stop_active", False))
        holding = snap.get("holding")
        if holding is not None:
            self._hold_acc = (
                None if holding["acc"] is None else dict(holding["acc"])
            )
            self._hold_prev = (
                None if holding["prev"] is None else dict(holding["prev"])
            )
            self._hold_last_time = float(holding["last_time"])
            self._noise_rng.bit_generator.state = copy.deepcopy(
                holding["noise_rng_state"]
            )
            self._last_noise = dict(holding["last_noise"])

        self.body.write(dict(snap["targets"]))
        self._apply_strength(force=True)
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
        if self.config.fatigue_weight > 0.0:
            new.fatigue_mult = 1.0 - self.config.fatigue_weight * self.fatigue
        new.set_energy(self.energy, force=True)
        mujoco.mj_forward(new.model, new.data)
        self.body = new
        if self.config.valves_enabled:
            self._init_valve_geometry()

    def enable_recording(self, path):
        from .record import Recorder

        self.recorder = Recorder(path, self)
        return self.recorder

    def disable_recording(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
