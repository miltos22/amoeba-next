"""Parameter counts per network from the arrays themselves (design DO 9)."""

from __future__ import annotations

import json

from brain.learner import BrainConfig, Learner

N_OBS = 38
N_ACTIONS = 12
HIDDEN = 128


def plain_network(n_obs, n_out):
    return (n_obs * HIDDEN + HIDDEN) + (HIDDEN * HIDDEN + HIDDEN) + (HIDDEN * n_out + n_out)


def main() -> int:
    rows = []
    for n_units in (0, 32, 48, 64):
        learner = Learner(N_OBS, N_ACTIONS, BrainConfig(n_units=n_units))
        rows.append(
            {
                "n_units": n_units,
                "policy_from_arrays": learner.param_count(learner.policy.params),
                "value_from_arrays": learner.param_count(learner.value.params),
                "policy_plain": plain_network(N_OBS, 2 * N_ACTIONS),
                "value_plain": plain_network(N_OBS, 1),
            }
        )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
