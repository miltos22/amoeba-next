"""One life of the brain in the stage-0 world, written as JSON lines.

The record keeps the row keys of baselines/life.py so the shared window
statistics can read it:

    header : row, args, config, git_commit, python_version, ... (see
             ``_header``), plus ``brain_config`` and
             ``constants_first_decision``
    window : row, window, t_start, t_end, the aggregate keys, mean_reward,
             mean_abs_td_error, mean_abs_action, frac_targets_at_limit,
             weight_norm_actor, weight_norm_critic, frac_dead_units,
             decisions, wall_seconds, plus tau_policy_*, tau_value_* and
             decisions_per_second
    life   : row, t_start, t_end, the aggregate keys, wall_seconds, steps,
             steps_per_second, decisions, decisions_per_second,
             obs_normalizer_count, mean_reward, mean_abs_td_error,
             mean_abs_action, frac_targets_at_limit, frac_dead_units

The reward is the only option of this job, neg_abs_need.  frac_dead_units
uses the policy head's hidden units (q1 and q2), 2 * hidden of them, as
baselines/life.py uses the hidden activations of its agent.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import subprocess
import sys
import time

import mujoco
import numpy as np

from amoeba.world import Config, World

from .learner import BrainConfig, Learner

SAMPLE_S = 0.5


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_commit(path: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _clock() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def _write_line(fh, obj) -> None:
    fh.write(json.dumps(obj, allow_nan=False, sort_keys=False) + "\n")
    fh.flush()


def aggregate(records, start_pos, food_start) -> dict:
    """The window statistics of experiments/noise_floor/noise_floor.py."""
    n = len(records)
    energies = [r["energy"] for r in records]
    abs_need = [r["abs_need"] for r in records]
    need_pos = [r["need_pos"] for r in records]
    need_neg = [r["need_neg"] for r in records]
    below_half = [1.0 if r["below_half"] else 0.0 for r in records]
    zero = [1.0 if r["zero"] else 0.0 for r in records]
    touch = [1.0 if r["any_touch"] else 0.0 for r in records]
    finite = all(bool(r["finite"]) for r in records)
    path = 0.0
    radii = 0.0
    prev = start_pos
    for r in records:
        x, y = r["pos"]
        path += math.hypot(x - prev[0], y - prev[1])
        radii += math.hypot(x, y)
        prev = r["pos"]
    last = records[-1]["pos"]
    net = math.hypot(last[0] - start_pos[0], last[1] - start_pos[1])
    return {
        "n_samples": n,
        "mean_energy": float(np.mean(energies)),
        "min_energy": float(np.min(energies)),
        "max_energy": float(np.max(energies)),
        "mean_abs_need": float(np.mean(abs_need)),
        "mean_need_pos": float(np.mean(need_pos)),
        "mean_need_neg": float(np.mean(need_neg)),
        "frac_energy_below_half": float(np.mean(below_half)),
        "frac_energy_zero": float(np.mean(zero)),
        "food_eaten": float(records[-1]["food_cum"] - food_start),
        "path_m": float(path),
        "net_m": float(net),
        "mean_radius_m": float(radii / n),
        "frac_any_touch": float(np.mean(touch)),
        "finite": finite,
    }


def _obs_vector(reading, obs_names):
    return np.array([reading[name] for name in obs_names], dtype=float)


def run_life(world: World, brain_config: BrainConfig, life_s: float, window_s: float,
             control_s: float, out_path: str | None = None, log=print) -> dict:
    if life_s <= 0.0 or window_s <= 0.0:
        raise ValueError("life_s and window_s must be positive")
    for name, value in (("life_s", life_s), ("window_s", window_s)):
        ratio = value / SAMPLE_S
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"{name} must be a whole multiple of {SAMPLE_S}")
    if control_s <= 0.0:
        raise ValueError("control_s must be positive")
    timestep = Config().timestep
    ratio = control_s / timestep
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(f"control_s must be a whole multiple of {timestep}")
    per_window = int(round(window_s / SAMPLE_S))
    if per_window < 1:
        raise ValueError("window_s must be at least one sample")

    cfg = world.config
    set_point = cfg.preset_constants.energy_set_point
    items = world.items()
    obs_items = [item for item in items if item.kind != "actuator"]
    actuator_items = [item for item in items if item.kind == "actuator"]
    obs_names = [item.name for item in obs_items]
    learner = Learner(len(obs_items), len(actuator_items), brain_config)

    constants_first = {
        "dt": float(control_s),
        "gamma": brain_config.gamma(control_s),
        "gl": brain_config.trace_decay(control_s),
        "beta": brain_config.beta(control_s),
        "alpha_policy": brain_config.alpha_policy(control_s),
        "alpha_value": brain_config.alpha_value(control_s),
    }
    header = {
        "row": "header",
        "args": {
            "life_s": life_s,
            "window_s": window_s,
            "control_s": control_s,
            "out": out_path,
        },
        "config": cfg.as_dict(),
        "git_commit": _git_commit(_project_root()),
        "python_version": sys.version,
        "python_version_info": list(sys.version_info),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "mujoco_version": mujoco.__version__,
        "torch_version": None,
        "reference_repo": None,
        "reference_repo_branch": None,
        "reference_repo_head": None,
        "reference_files": [],
        "torch_num_threads": None,
        "torch_num_interop_threads": None,
        "torch_seed": None,
        "numpy_seed": int(brain_config.seed),
        "world_seed": int(cfg.seed),
        "control_s": float(control_s),
        "reward": "neg_abs_need",
        "feeling": "real",
        "hyperparameters": {
            "n_units": int(brain_config.n_units),
            "hidden": int(brain_config.hidden),
            "input_gain": brain_config.input_gain,
            "gamma_ref": float(brain_config.gamma_ref),
            "lam_ref": float(brain_config.lam_ref),
            "beta_ref": float(brain_config.beta_ref),
            "alpha_ref_policy": float(brain_config.alpha_ref_policy),
            "alpha_ref_value": float(brain_config.alpha_ref_value),
            "tau_ent": float(brain_config.tau_ent),
            "sparsity": float(brain_config.sparsity),
            "brain_seed": int(brain_config.seed),
        },
        "shuffled_source": None,
        "observation_items": obs_names,
        "actuator_items": [
            {"name": item.name, "low": float(item.low), "high": float(item.high)}
            for item in actuator_items
        ],
        "start_time": _clock(),
        "sample_s": SAMPLE_S,
        "brain_config": brain_config.as_dict(),
        "constants_first_decision": constants_first,
    }

    fh = None
    if out_path is not None:
        fh = open(out_path, "x", encoding="utf-8")
        _write_line(fh, header)

    start_pos = world.body.centre()
    records: list[dict] = []
    current: list[dict] = []
    window_index = 0
    window_start_food = 0.0
    window_start_pos = start_pos
    next_sample_t = SAMPLE_S

    lifetime = {
        "reward_sum": 0.0,
        "td_sum": 0.0,
        "abs_action_sum": 0.0,
        "action_count": 0,
        "at_limit": 0,
        "target_count": 0,
        "decisions": 0,
        "wall": 0.0,
        "dead_units": 0,
        "dead_unit_slots": 0,
    }
    window = {
        "reward_sum": 0.0,
        "td_sum": 0.0,
        "abs_action_sum": 0.0,
        "action_count": 0,
        "at_limit": 0,
        "target_count": 0,
        "decisions": 0,
        "wall": 0.0,
        "h_min": None,
        "h_max": None,
        "tau_policy_sum": 0.0,
        "tau_policy_count": 0,
        "tau_policy_min": None,
        "tau_policy_max": None,
        "tau_value_sum": 0.0,
        "tau_value_count": 0,
        "tau_value_min": None,
        "tau_value_max": None,
    }
    windows = []
    hidden_width = int(brain_config.hidden)

    def tau_stats(prefix, values):
        if values.size == 0:
            return
        window[prefix + "_sum"] += float(values.sum())
        window[prefix + "_count"] += int(values.size)
        lo = float(values.min())
        hi = float(values.max())
        if window[prefix + "_min"] is None or lo < window[prefix + "_min"]:
            window[prefix + "_min"] = lo
        if window[prefix + "_max"] is None or hi > window[prefix + "_max"]:
            window[prefix + "_max"] = hi

    def flush_window():
        nonlocal current, window_start_food, window_start_pos, window_index, window
        if not current:
            return None
        decisions = window["decisions"]
        row = {
            "row": "window",
            "window": window_index,
            "t_start": float(current[0]["t"] - SAMPLE_S),
            "t_end": float(current[-1]["t"]),
        }
        row.update(aggregate(current, window_start_pos, window_start_food))
        total_units = 2 * hidden_width
        if decisions and window["h_min"] is not None:
            varying = int(np.count_nonzero(window["h_max"] - window["h_min"]))
            dead = total_units - varying
            lifetime["dead_units"] += dead
            lifetime["dead_unit_slots"] += total_units
        else:
            dead = 0
        for prefix in ("tau_policy", "tau_value"):
            if window[prefix + "_count"]:
                row[prefix + "_mean"] = float(
                    window[prefix + "_sum"] / window[prefix + "_count"]
                )
                row[prefix + "_min"] = float(window[prefix + "_min"])
                row[prefix + "_max"] = float(window[prefix + "_max"])
            else:
                row[prefix + "_mean"] = None
                row[prefix + "_min"] = None
                row[prefix + "_max"] = None
        wall = float(window["wall"])
        row.update(
            {
                "mean_reward": float(window["reward_sum"] / decisions) if decisions else 0.0,
                "mean_abs_td_error": float(window["td_sum"] / decisions) if decisions else 0.0,
                "mean_abs_action": float(window["abs_action_sum"] / window["action_count"])
                if window["action_count"]
                else 0.0,
                "frac_targets_at_limit": float(window["at_limit"] / window["target_count"])
                if window["target_count"]
                else 0.0,
                "weight_norm_actor": float(learner.policy_group.weight_norm()),
                "weight_norm_critic": float(learner.value_group.weight_norm()),
                "frac_dead_units": float(dead / total_units) if decisions else 0.0,
                "decisions": int(decisions),
                "wall_seconds": wall,
                "decisions_per_second": float(decisions / wall) if wall > 0.0 else 0.0,
            }
        )
        windows.append(row)
        if fh is not None:
            _write_line(fh, row)
        window_start_food = float(current[-1]["food_cum"])
        window_start_pos = current[-1]["pos"]
        window_index += 1
        current = []
        window = {
            "reward_sum": 0.0,
            "td_sum": 0.0,
            "abs_action_sum": 0.0,
            "action_count": 0,
            "at_limit": 0,
            "target_count": 0,
            "decisions": 0,
            "wall": 0.0,
            "h_min": None,
            "h_max": None,
            "tau_policy_sum": 0.0,
            "tau_policy_count": 0,
            "tau_policy_min": None,
            "tau_policy_max": None,
            "tau_value_sum": 0.0,
            "tau_value_count": 0,
            "tau_value_min": None,
            "tau_value_max": None,
        }
        return row

    wall_start = time.perf_counter()
    reading = world.read()
    raw_obs = _obs_vector(reading, obs_names)
    x = learner.observe_initial(raw_obs)
    dt_k = float(control_s)

    while world.time < life_s - 1e-9:
        decision_wall = time.perf_counter()
        action = learner.act(x, dt_k)
        hidden = learner.hidden_activations()
        if window["h_min"] is None:
            window["h_min"] = hidden.copy()
            window["h_max"] = hidden.copy()
        else:
            np.minimum(window["h_min"], hidden, out=window["h_min"])
            np.maximum(window["h_max"], hidden, out=window["h_max"])
        tau = learner.tau_ranges()
        tau_stats("tau_policy", tau["policy"])
        tau_stats("tau_value", tau["value"])

        targets = {}
        at_limit = 0
        for item, value in zip(actuator_items, action):
            target = item.low + (float(value) + 1.0) * 0.5 * (item.high - item.low)
            targets[item.name] = target
            if target <= item.low or target >= item.high:
                at_limit += 1
        t_before = float(world.time)
        world.write(targets)
        world.advance(control_s)
        t_next = float(world.time)
        dt_next = t_next - t_before
        reading_prime = world.read()
        x_next = learner.normalise(_obs_vector(reading_prime, obs_names))
        info = learner.observe(x_next, reading_prime["energy.need"], dt_next)

        reward = info["reward"]
        window["reward_sum"] += reward
        window["td_sum"] += abs(info["delta"])
        window["abs_action_sum"] += float(np.abs(action).sum())
        window["action_count"] += int(np.asarray(action).size)
        window["at_limit"] += at_limit
        window["target_count"] += len(actuator_items)
        window["decisions"] += 1
        window["wall"] += time.perf_counter() - decision_wall
        lifetime["reward_sum"] += reward
        lifetime["td_sum"] += abs(info["delta"])
        lifetime["abs_action_sum"] += float(np.abs(action).sum())
        lifetime["action_count"] += int(np.asarray(action).size)
        lifetime["at_limit"] += at_limit
        lifetime["target_count"] += len(actuator_items)
        lifetime["decisions"] += 1
        lifetime["wall"] += time.perf_counter() - decision_wall

        while next_sample_t <= world.time + 1e-9 and next_sample_t <= life_s + 1e-9:
            energy = float(world.energy)
            x_pos, y_pos = world.body.centre()
            record = {
                "t": float(world.time),
                "energy": energy,
                "abs_need": abs(set_point - energy),
                "need_pos": max(0.0, set_point - energy),
                "need_neg": max(0.0, energy - set_point),
                "below_half": energy < 0.5,
                "zero": energy == 0.0,
                "pos": (x_pos, y_pos),
                "any_touch": any(
                    reading_prime[f"arm{k}.touch"] == 1.0
                    for k in range(world.body.n_arms)
                ),
                "finite": bool(np.isfinite(list(reading_prime.values())).all()),
                "food_cum": float(world.food_eaten),
            }
            records.append(record)
            current.append(record)
            next_sample_t += SAMPLE_S
            while len(current) >= per_window:
                flush_window()

        x = x_next
        dt_k = dt_next

    flush_window()
    wall_seconds = time.perf_counter() - wall_start
    steps = int(round(world.time / cfg.timestep))
    decisions = lifetime["decisions"]
    obs_count = int(learner.obs_count)
    if obs_count != decisions + 1:
        raise AssertionError(
            f"observation normaliser count {obs_count} != decisions + 1 "
            f"({decisions} + 1 = {decisions + 1})"
        )
    life_row = {
        "row": "life",
        "t_start": 0.0,
        "t_end": float(records[-1]["t"]),
    }
    life_row.update(aggregate(records, start_pos, 0.0))
    life_row.update(
        {
            "wall_seconds": float(wall_seconds),
            "steps": steps,
            "steps_per_second": float(steps / wall_seconds),
            "decisions": int(decisions),
            "decisions_per_second": float(decisions / wall_seconds),
            "obs_normalizer_count": int(obs_count),
            "mean_reward": float(lifetime["reward_sum"] / decisions) if decisions else 0.0,
            "mean_abs_td_error": float(lifetime["td_sum"] / decisions) if decisions else 0.0,
            "mean_abs_action": float(
                lifetime["abs_action_sum"] / lifetime["action_count"]
            )
            if lifetime["action_count"]
            else 0.0,
            "frac_targets_at_limit": float(
                lifetime["at_limit"] / lifetime["target_count"]
            )
            if lifetime["target_count"]
            else 0.0,
            "frac_dead_units": float(
                lifetime["dead_units"] / lifetime["dead_unit_slots"]
            )
            if lifetime["dead_unit_slots"]
            else 0.0,
        }
    )
    if fh is not None:
        _write_line(fh, life_row)
        fh.close()
    log(
        f"life: mean_energy={life_row['mean_energy']!r} "
        f"mean_abs_need={life_row['mean_abs_need']!r} path_m={life_row['path_m']!r} "
        f"decisions={decisions} wall_seconds={wall_seconds!r} "
        f"decisions_per_second={life_row['decisions_per_second']!r}"
    )
    return {"header": header, "windows": windows, "life": life_row}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    life = sub.add_parser("life", help="run one life and write JSON lines")
    life.add_argument("--preset", choices=("lean", "rich"), required=True)
    life.add_argument("--n-patches", type=int, required=True)
    life.add_argument("--seed", type=int, required=True)
    life.add_argument("--life-s", type=float, required=True)
    life.add_argument("--window-s", type=float, default=600.0)
    life.add_argument("--control-s", type=float, default=0.1)
    life.add_argument("--gamma", type=float, default=0.99)
    life.add_argument("--out", required=True)
    life.add_argument("--n-units", type=int, default=48)
    life.add_argument("--input-gain", choices=("zoh", "paper"), default="zoh")
    life.add_argument("--brain-seed", type=int, default=0)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "life":
        if os.path.exists(args.out):
            raise SystemExit(f"refusing to overwrite existing file: {args.out}")
        world_cfg = Config(
            seed=args.seed,
            preset=args.preset,
            n_patches=args.n_patches,
            patch_radius=0.25,
            dish_radius=1.5,
        )
        world = World(world_cfg)
        brain_cfg = BrainConfig(
            n_units=args.n_units,
            input_gain=args.input_gain,
            gamma_ref=args.gamma,
            seed=args.brain_seed,
        )
        run_life(
            world,
            brain_cfg,
            args.life_s,
            args.window_s,
            args.control_s,
            out_path=args.out,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
