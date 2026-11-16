"""The stage-0 body: a round centre with jointed arms, held in the plane.

The body is the only thing the brain ever touches.  It exposes items:
senses (joint angles and speeds, smell at each arm tip and at the centre,
touch at each arm tip) and one feeling (energy need).  Every joint has
one actuator item; a written target is held until written again.

MuJoCo's ellipsoid fluid model is used on every body geom so that drag
depends on the direction a segment moves (3.3.1).  The dish wall is a
ring of planes tangent to the dish radius: smooth enough for an arm of
this size, and it cannot trap because every crease points away from the
interior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .feelings import NEED_HIGH, NEED_LOW, Preset, energy_need, resolve_preset, strength
from .items import Item


@dataclass(frozen=True, slots=True)
class BodyShape:
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


def make_body(
    n_arms: int = 6,
    joints_per_arm: int = 2,
    *,
    dish_radius: float = 1.5,
    timestep: float = 0.002,
    fluid_density: float = 1000.0,
    fluid_viscosity: float = 1.0,
    n_wall_planes: int = 32,
    shape: BodyShape | None = None,
    smell_names: tuple[str, ...] = ("plain",),
    gut_sense: bool = False,
    smell_bound: float = 36.0,
    preset: Preset | str = "lean",
    weakness: bool = True,
    initial_energy: float = 1.0,
) -> "Body":
    """Generate the MJCF for a body in its dish and return the live Body."""
    shape = BodyShape() if shape is None else shape
    if n_arms < 1 or joints_per_arm < 1:
        raise ValueError("a body needs at least one arm with at least one joint")
    xml = _body_xml(
        n_arms=n_arms,
        joints_per_arm=joints_per_arm,
        dish_radius=dish_radius,
        timestep=timestep,
        fluid_density=fluid_density,
        fluid_viscosity=fluid_viscosity,
        n_wall_planes=n_wall_planes,
        shape=shape,
    )
    return Body(
        xml,
        n_arms=n_arms,
        joints_per_arm=joints_per_arm,
        dish_radius=dish_radius,
        n_wall_planes=n_wall_planes,
        shape=shape,
        smell_names=smell_names,
        gut_sense=gut_sense,
        smell_bound=smell_bound,
        preset=preset,
        weakness=weakness,
        initial_energy=initial_energy,
    )


def _body_xml(
    *,
    n_arms: int,
    joints_per_arm: int,
    dish_radius: float,
    timestep: float,
    fluid_density: float,
    fluid_viscosity: float,
    n_wall_planes: int,
    shape: BodyShape,
) -> str:
    s = shape
    lines = [
        "<mujoco model=\"amoeba\">",
        "  <compiler angle=\"radian\" autolimits=\"true\"/>",
        f"  <option timestep=\"{timestep!r}\" gravity=\"0 0 0\" density=\"{fluid_density!r}\" "
        f"viscosity=\"{fluid_viscosity!r}\" integrator=\"implicitfast\"/>",
        "  <worldbody>",
    ]
    for i in range(n_wall_planes):
        theta = 2.0 * math.pi * i / n_wall_planes
        cx, cy = math.cos(theta), math.sin(theta)
        lines.append(
            f"    <geom name=\"wall{i}\" type=\"plane\" pos=\"{dish_radius * cx!r} {dish_radius * cy!r} 0\" "
            f"zaxis=\"{-cx!r} {-cy!r} 0\" size=\"2 2 0.5\" contype=\"1\" conaffinity=\"1\" rgba=\"0.6 0.6 0.6 0.15\"/>"
        )
    lines += [
        "    <body name=\"core\" pos=\"0 0 0\">",
        f"      <joint name=\"core.x\" type=\"slide\" axis=\"1 0 0\" range=\"{-dish_radius!r} {dish_radius!r}\" damping=\"0\"/>",
        f"      <joint name=\"core.y\" type=\"slide\" axis=\"0 1 0\" range=\"{-dish_radius!r} {dish_radius!r}\" damping=\"0\"/>",
        "      <joint name=\"core.th\" type=\"hinge\" axis=\"0 0 1\" range=\"-3.141592653589793 3.141592653589793\" damping=\"0\"/>",
        f"      <geom name=\"core\" type=\"sphere\" size=\"{s.center_radius!r}\" mass=\"{s.center_mass!r}\" "
        "fluidshape=\"ellipsoid\"/>",
    ]
    joint_names = []
    actuator_xml = []
    for k in range(n_arms):
        a = 2.0 * math.pi * k / n_arms
        c, sn = math.cos(a), math.sin(a)
        indent = "      "
        for m in range(joints_per_arm):
            jname = f"arm{k}.j{m + 1}"
            joint_names.append(jname)
            if m == 0:
                pos = f"{s.center_radius * c!r} {s.center_radius * sn!r} 0"
                euler = f"0 0 {a!r}"
            else:
                pos = f"{s.segment_length!r} 0 0"
                euler = "0 0 0"
            lines.append(f"{indent}<body name=\"arm{k}.l{m + 1}\" pos=\"{pos}\" euler=\"{euler}\">")
            lines.append(
                f"{indent}  <joint name=\"{jname}\" type=\"hinge\" axis=\"0 0 1\" "
                f"range=\"{-s.joint_limit!r} {s.joint_limit!r}\" damping=\"{s.joint_damping!r}\" "
                f"armature=\"{s.joint_armature!r}\"/>"
            )
            lines.append(
                f"{indent}  <geom name=\"arm{k}.l{m + 1}\" type=\"capsule\" fromto=\"0 0 0 {s.segment_length!r} 0 0\" "
                f"size=\"{s.arm_radius!r}\" mass=\"{s.link_mass!r}\" fluidshape=\"ellipsoid\"/>"
            )
            indent += "  "
        for _ in range(joints_per_arm):
            lines.append(f"{indent[:-2]}</body>")
        for m in range(joints_per_arm):
            jname = f"arm{k}.j{m + 1}"
            actuator_xml.append(
                f"    <position name=\"{jname}.target\" joint=\"{jname}\" kp=\"{s.kp!r}\" kv=\"{s.kv!r}\" "
                f"ctrlrange=\"{-s.joint_limit!r} {s.joint_limit!r}\" "
                f"forcerange=\"{-s.force_limit!r} {s.force_limit!r}\"/>"
            )
    lines.append("    </body>")
    lines.append("  </worldbody>")
    lines.append("  <actuator>")
    lines += actuator_xml
    lines.append("  </actuator>")
    lines.append("</mujoco>")
    return "\n".join(lines) + "\n"


class Body:
    """Live body: MJCF model/data plus the item interface."""

    def __init__(
        self,
        xml: str,
        *,
        n_arms: int,
        joints_per_arm: int,
        dish_radius: float,
        n_wall_planes: int,
        shape: BodyShape,
        smell_names: tuple[str, ...] = ("plain",),
        gut_sense: bool = False,
        smell_bound: float,
        preset: Preset | str,
        weakness: bool,
        initial_energy: float,
    ) -> None:
        self.xml = xml
        self.n_arms = n_arms
        self.joints_per_arm = joints_per_arm
        self.dish_radius = float(dish_radius)
        self.shape = shape
        self.preset = resolve_preset(preset)
        self.weakness = bool(weakness)
        self.smell_names = tuple(smell_names)
        self.gut_sense = bool(gut_sense)
        self.gut_load = 0.0
        self.smell_at = None
        self.smell_at_kinds = None
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.joint_names: list[str] = []
        for k in range(n_arms):
            for m in range(joints_per_arm):
                self.joint_names.append(f"arm{k}.j{m + 1}")
        self.actuator_names: list[str] = [f"{j}.target" for j in self.joint_names]

        self._qpos = np.array([self.model.joint(j).qposadr[0] for j in self.joint_names], dtype=int)
        self._dof = np.array([self.model.joint(j).dofadr[0] for j in self.joint_names], dtype=int)
        self._act = np.array([self.model.actuator(a).id for a in self.actuator_names], dtype=int)
        self._tip_geoms = np.array(
            [self.model.geom(f"arm{k}.l{joints_per_arm}").id for k in range(n_arms)], dtype=int
        )
        self._half_seg = 0.5 * shape.segment_length

        root = [self.model.joint(n).qposadr[0] for n in ("core.x", "core.y", "core.th")]
        if sorted(int(x) for x in root) != [0, 1, 2]:
            raise RuntimeError(f"unexpected root qpos layout: {root}")

        theta = 2.0 * math.pi * np.arange(n_wall_planes) / n_wall_planes
        self._wall_u = np.stack([np.cos(theta), np.sin(theta)], axis=1)

        self._targets = {name: 0.0 for name in self.actuator_names}
        self._target_array = np.zeros(len(self.actuator_names))
        self._act_index = {name: i for i, name in enumerate(self.actuator_names)}

        self._items = self._build_items(float(smell_bound))
        self._by_name = {item.name: item for item in self._items}

        self.energy = 1.0
        self._strength = -1.0
        self.set_energy(initial_energy, force=True)

    # -- item construction -------------------------------------------------
    def _build_items(self, smell_bound: float) -> list[Item]:
        items: list[Item] = []
        if len(self.smell_names) == 1:
            items.append(Item("centre.smell", "sense", 0.0, smell_bound, "smell"))
        else:
            for name in self.smell_names:
                items.append(Item(f"centre.smell.{name}", "sense", 0.0, smell_bound, "smell"))
        items.append(Item("energy.need", "feeling", NEED_LOW, NEED_HIGH, "need"))
        for k in range(self.n_arms):
            for m in range(self.joints_per_arm):
                base = f"arm{k}.j{m + 1}"
                items.append(Item(f"{base}.angle", "sense", -math.pi, math.pi, "rad"))
                items.append(Item(f"{base}.vel", "sense", -50.0, 50.0, "rad/s"))
        for k in range(self.n_arms):
            if len(self.smell_names) == 1:
                items.append(Item(f"arm{k}.smell", "sense", 0.0, smell_bound, "smell"))
            else:
                for name in self.smell_names:
                    items.append(Item(f"arm{k}.smell.{name}", "sense", 0.0, smell_bound, "smell"))
            items.append(Item(f"arm{k}.touch", "sense", 0.0, 1.0, "touch"))
        if self.gut_sense:
            items.append(Item("gut", "sense", 0.0, 1.0, "gut"))
        for k in range(self.n_arms):
            for m in range(self.joints_per_arm):
                base = f"arm{k}.j{m + 1}"
                items.append(
                    Item(f"{base}.target", "actuator", -self.shape.joint_limit, self.shape.joint_limit, "rad")
                )
        return items

    # -- item interface (3.1) ---------------------------------------------
    def items(self) -> list[Item]:
        return list(self._items)

    def read(self) -> dict[str, float]:
        mujoco.mj_kinematics(self.model, self.data)
        q = self.data.qpos
        v = self.data.qvel
        out: dict[str, float] = {}
        for index, jname in enumerate(self.joint_names):
            out[f"{jname}.angle"] = float(q[self._qpos[index]])
            out[f"{jname}.vel"] = float(v[self._dof[index]])

        tips = np.empty((self.n_arms, 2))
        for k, gid in enumerate(self._tip_geoms):
            xmat = self.data.geom_xmat[gid].reshape(3, 3)
            tip = self.data.geom_xpos[gid] + xmat[2] * self._half_seg
            tips[k] = tip[:2]
        centre = np.array(self.centre())
        points = np.vstack([tips, centre[None, :]])
        if len(self.smell_names) > 1:
            if self.smell_at_kinds is None:
                smells_by_kind = np.zeros((points.shape[0], len(self.smell_names)))
            else:
                smells_by_kind = np.asarray(self.smell_at_kinds(points), dtype=float)
            for j, name in enumerate(self.smell_names):
                for k in range(self.n_arms):
                    out[f"arm{k}.smell.{name}"] = float(smells_by_kind[k, j])
                out[f"centre.smell.{name}"] = float(smells_by_kind[self.n_arms, j])
        else:
            if self.smell_at is None:
                smells = np.zeros(points.shape[0])
            else:
                smells = np.asarray(self.smell_at(points), dtype=float).reshape(-1)
            for k in range(self.n_arms):
                out[f"arm{k}.smell"] = float(smells[k])
            out["centre.smell"] = float(smells[self.n_arms])

        wall_distance = self.dish_radius - tips @ self._wall_u.T
        nearest = wall_distance.min(axis=1)
        for k in range(self.n_arms):
            out[f"arm{k}.touch"] = 1.0 if nearest[k] <= self.shape.arm_radius else 0.0

        out["energy.need"] = energy_need(self.energy, self.preset)
        if self.gut_sense:
            out["gut"] = float(self.gut_load)
        return out

    def write(self, targets: dict[str, float]) -> None:
        for name, value in targets.items():
            item = self._by_name.get(name)
            if item is None or item.kind != "actuator":
                raise KeyError(f"not an actuator item: {name!r}")
            v = float(value)
            if not math.isfinite(v):
                raise ValueError(f"target for {name!r} is not finite: {value!r}")
            v = item.low if v < item.low else (item.high if v > item.high else v)
            self._targets[name] = v
            self._target_array[self._act_index[name]] = v

    def targets(self) -> dict[str, float]:
        return dict(self._targets)

    # -- physics glue ------------------------------------------------------
    def apply(self) -> None:
        self.data.ctrl[:] = self._target_array

    def set_energy(self, energy: float, force: bool = False) -> None:
        e = float(energy)
        e = 0.0 if e < 0.0 else (1.0 if e > 1.0 else e)
        self.energy = e
        s = strength(e, self.preset, self.weakness)
        if force or abs(s - self._strength) > 1e-12:
            self._strength = s
            kp = self.shape.kp
            kv = self.shape.kv
            fmax = self.shape.force_limit
            self.model.actuator_gainprm[:, 0] = kp * s
            self.model.actuator_biasprm[:, 1] = -kp * s
            self.model.actuator_biasprm[:, 2] = -kv * s
            self.model.actuator_forcerange[:, 0] = -fmax * s
            self.model.actuator_forcerange[:, 1] = fmax * s

    def effort(self) -> float:
        return float(np.abs(self.data.actuator_force).mean() / self.shape.force_limit)

    # -- state -------------------------------------------------------------
    def centre(self) -> tuple[float, float]:
        return float(self.data.qpos[0]), float(self.data.qpos[1])

    def heading(self) -> float:
        return float(self.data.qpos[2])

    def root_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos[0:3].copy(), self.data.qvel[0:3].copy()

    def set_root_state(self, qpos3: np.ndarray, qvel3: np.ndarray) -> None:
        self.data.qpos[0:3] = qpos3
        self.data.qvel[0:3] = qvel3
        mujoco.mj_forward(self.model, self.data)

    def joint_state(self) -> dict[str, tuple[float, float]]:
        return {
            name: (float(self.data.qpos[self._qpos[i]]), float(self.data.qvel[self._dof[i]]))
            for i, name in enumerate(self.joint_names)
        }

    def set_joint_state(self, name: str, angle: float, velocity: float) -> None:
        i = self.joint_names.index(name)
        self.data.qpos[self._qpos[i]] = angle
        self.data.qvel[self._dof[i]] = velocity
