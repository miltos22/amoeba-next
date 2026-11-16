"""Two calibration toys, no world needed (brief DO 5).

Tracking toy: x in [-1, 1] moves by 0.1 * a (clipped), every 200 decisions
x jumps to a random place, reward = -|x|.  The learner must hold the mean
|x| of the last 20 % of 30000 decisions below 0.5 of a random policy's,
and a learner fed another run's rewards must stay above 0.8 of random.

Memory toy: cue +1/-1 in observation coordinate 0 for one decision every
40; 20 decisions later coordinate 1 flashes 1 and the reward is +1 if
sign(a_0) equals the cue, else -1.  For n = 16 and n = 0 the mean reward
at the flashes of the last 20 % of 200000 decisions is reported; n = 16
must be above 0.5 and n = 0 below 0.1.
"""

from __future__ import annotations

import numpy as np

from brain.learner import BrainConfig, Learner

DT = 0.1
TRACK_DECISIONS = 30000
TRACK_UNITS = 0
TRACK_HIDDEN = 64
MEM_DECISIONS = 200000
MEM_UNITS = 16
MEM_HIDDEN = 128
MEM_CONTROL_HIDDEN = 32


def _tracking(n_units, seed, decisions, hidden, reward_stream=None):
    cfg = BrainConfig(n_units=n_units, hidden=hidden, seed=seed)
    learner = Learner(1, 1, cfg)
    jump_rng = np.random.default_rng(1000 + seed)
    x = 0.0
    abs_x = np.empty(decisions)
    rewards = np.empty(decisions)
    x_norm = learner.observe_initial(np.array([x]))
    for k in range(decisions):
        action = learner.act(x_norm, DT)
        action = float(np.clip(action[0], -1.0, 1.0))
        x = float(np.clip(x + 0.1 * action, -1.0, 1.0))
        if (k + 1) % 200 == 0:
            x = float(jump_rng.uniform(-1.0, 1.0))
        reward = -abs(x) if reward_stream is None else float(reward_stream[k])
        abs_x[k] = abs(x)
        rewards[k] = reward
        x_norm = learner.normalise(np.array([x]))
        learner.observe_reward(x_norm, reward, DT)
    return float(np.mean(abs_x[int(0.8 * decisions) :])), rewards


def _random_policy(seed, decisions):
    action_rng = np.random.default_rng(5000 + seed)
    jump_rng = np.random.default_rng(1000 + seed)
    x = 0.0
    abs_x = np.empty(decisions)
    rewards = np.empty(decisions)
    for k in range(decisions):
        action = float(action_rng.uniform(-1.0, 1.0))
        x = float(np.clip(x + 0.1 * action, -1.0, 1.0))
        if (k + 1) % 200 == 0:
            x = float(jump_rng.uniform(-1.0, 1.0))
        abs_x[k] = abs(x)
        rewards[k] = -abs(x)
    return float(np.mean(abs_x[int(0.8 * decisions) :])), rewards


def _memory(n_units, seed, decisions, hidden):
    cfg = BrainConfig(n_units=n_units, hidden=hidden, seed=seed)
    learner = Learner(2, 1, cfg)
    cue_rng = np.random.default_rng(2000 + seed)
    last_cue = 0.0
    flash_at = []
    flash_reward = []
    x_norm = learner.observe_initial(np.array([0.0, 0.0]))
    for k in range(decisions):
        cue = 0.0
        flash = 0.0
        if k % 40 == 0:
            cue = 1.0 if cue_rng.random() < 0.5 else -1.0
            last_cue = cue
        if k % 40 == 20:
            flash = 1.0
        action = learner.act(x_norm, DT)
        reward = 0.0
        if flash:
            reward = 1.0 if np.sign(action[0]) == last_cue else -1.0
            flash_at.append(k)
            flash_reward.append(reward)
        x_norm = learner.normalise(np.array([cue, flash]))
        learner.observe_reward(x_norm, reward, DT)
    at = np.array(flash_at)
    rew = np.array(flash_reward)
    tail = rew[at >= int(0.8 * decisions)]
    return float(np.mean(tail)), int(tail.size)


def test_tracking_toy():
    random_mean, random_rewards = _random_policy(0, TRACK_DECISIONS)
    learner_mean, _ = _tracking(TRACK_UNITS, 0, TRACK_DECISIONS, TRACK_HIDDEN)
    broken_mean, _ = _tracking(
        TRACK_UNITS, 0, TRACK_DECISIONS, TRACK_HIDDEN, reward_stream=random_rewards
    )
    print(
        f"tracking: random_mean={random_mean!r} learner_mean={learner_mean!r} "
        f"broken_mean={broken_mean!r} learner/random={learner_mean / random_mean!r} "
        f"broken/random={broken_mean / random_mean!r}"
    )
    assert learner_mean < 0.5 * random_mean
    assert broken_mean > 0.8 * random_mean


def test_memory_toy():
    mem_mean, mem_count = _memory(MEM_UNITS, 0, MEM_DECISIONS, MEM_HIDDEN)
    control_mean, control_count = _memory(
        0, 0, MEM_DECISIONS, MEM_CONTROL_HIDDEN
    )
    print(
        f"memory: n={MEM_UNITS} mean_flash_reward={mem_mean!r} flashes={mem_count}"
    )
    print(
        f"memory: n=0 mean_flash_reward={control_mean!r} flashes={control_count}"
    )
    assert mem_mean > 0.5
    assert control_mean < 0.1
