"""Append-only life record: every advance call, every item, every target.

One JSON object per line, header first.  The file is line-buffered, so a
killed process loses at most the row being written (well inside the 10 s
of world time the design allows). The reader returns numpy arrays keyed
by item name; a body change leaves new names as NaN in earlier rows.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from .world import Config, World

FORMAT = "amoeba-life-record/1"


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }


def _empty_arrays(rows: list[dict], key: str) -> dict[str, np.ndarray]:
    names = sorted({name for row in rows for name in row.get(key, {})})
    out = {}
    for name in names:
        out[name] = np.array(
            [row.get(key, {}).get(name, np.nan) for row in rows], dtype=float
        )
    return out


class Recorder:
    """Writes one row per World.advance call."""

    def __init__(self, path: str | Path, world: World) -> None:
        self.world = world
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # buffering=1 means line buffering in text mode: every row reaches
        # the file immediately, so a killed process cannot lose more than
        # the row it was writing.
        self._file = open(self.path, "w", encoding="utf-8", buffering=1)
        self._last_food_time: float | None = None
        self._write_header()

    def _write_header(self) -> None:
        config: Config = self.world.config
        header = {
            "format": FORMAT,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": config.seed,
            "config": config.as_dict(),
            "git_commit": _git_commit(),
            "versions": _versions(),
            "platform": platform.platform(),
        }
        if config.pulse_patches:
            header["pulse_patch_indices"] = list(
                range(config.n_patches, config.n_patches + config.pulse_patches)
            )
        self._file.write(json.dumps({"header": header}, allow_nan=True) + "\n")
        self._file.flush()

    def note(self) -> None:
        world = self.world
        time = world.time
        x, y = world.body.centre()
        row = {
            "t": time,
            "energy": world.energy,
            "pos": [x, y],
            "heading": world.body.heading(),
            "read": world.read(),
            "targets": world.targets(),
        }
        if world.config.fatigue_weight > 0.0:
            row["fatigue"] = float(world.fatigue)
        interval = world.config.food_record_interval_s
        if self._last_food_time is None or time - self._last_food_time >= interval - 1e-12:
            row["food"] = {
                "amounts": [float(v) for v in world.food.totals()],
                "centres": world.food.centres_list(),
                "kinds": [kind.name for kind in world.config.food_kinds],
                "eaten_by_kind": [float(v) for v in world.food_eaten_by_kind],
                "delivered_by_kind": [float(v) for v in world.food_delivered_by_kind],
                "gut_load": float(world.gut_load),
            }
            self._last_food_time = time
        self._file.write(json.dumps(row, allow_nan=True) + "\n")

    def close(self) -> None:
        self._file.flush()
        self._file.close()


def read_header(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        first = json.loads(f.readline())
    if "header" not in first:
        raise ValueError(f"{path} is not an amoeba life record")
    return first["header"]


def read_record(path: str | Path, t0: float | None = None, t1: float | None = None) -> dict[str, np.ndarray]:
    header = read_header(path)
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = row["t"]
            if t0 is not None and t < t0:
                continue
            if t1 is not None and t > t1:
                continue
            rows.append(row)

    out: dict[str, np.ndarray] = {}
    out["time"] = np.array([row["t"] for row in rows], dtype=float)
    out["energy"] = np.array([row["energy"] for row in rows], dtype=float)
    out["pos.x"] = np.array([row["pos"][0] for row in rows], dtype=float)
    out["pos.y"] = np.array([row["pos"][1] for row in rows], dtype=float)
    out["heading"] = np.array([row["heading"] for row in rows], dtype=float)
    out.update(_empty_arrays(rows, "read"))
    out.update(_empty_arrays(rows, "targets"))
    if any("fatigue" in row for row in rows):
        out["fatigue"] = np.array(
            [row.get("fatigue", np.nan) for row in rows], dtype=float
        )

    n_patches = int(header["config"].get("n_patches", 0)) + int(
        header["config"].get("pulse_patches", 0)
    )
    amounts = np.full((len(rows), n_patches), np.nan)
    centres = np.full((len(rows), n_patches, 2), np.nan)
    for i, row in enumerate(rows):
        food = row.get("food")
        if food:
            amounts[i] = food["amounts"]
            centres[i] = food["centres"]
    out["food.amounts"] = amounts
    out["food.centres"] = centres
    return out
