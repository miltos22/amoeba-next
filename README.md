# Amoeba Next

One simulated brain, grown through stages that mirror how real brains evolved.

**Status: early and exploratory (November 2026). This is mostly an idea, plus the first pieces of the world it needs. Nothing in this repository is a demonstrated result.**

## The idea

Intelligence in nature was not trained on a data set or toward a score. It grew, in bodies with needs, in a world where actions have consequences. The aim of this project is to grow human-like intelligence the same way: take one simulated brain through stages that mirror how real brains evolved, in a physical (simulated) environment, adding drives, richer bodies, peers and a society step by step. The last stage is language: the organism learns English and communicates by keyboard, after it already has an internal understanding of its world.

## Principles

1. **One organism, one unbroken life.** No populations, no generations, no resets and no death. Running out of energy weakens the body; it never ends the life. Biological limits such as a fixed lifespan are not copied, because they are limits and not sources of intelligence.
2. **Drives, not scores.** The organism has homeostatic drives with set points (the first is energy). There is no task reward and no script that says how to behave: its own feelings and the world shape it.
3. **We build the school; it grows the mind.** What genes or chemistry fix in real organisms may be built in. What real organisms learn or develop must be learned or developed here too: the design supplies only the capability and the conditions.
4. **Consequences of its own actions, across long gaps.** The centre of the design is connecting what happens late to what the organism did earlier, over seconds, minutes and eventually much longer. This is what lets knowledge of what leads to what accumulate; a tight short-range reward loop builds habits instead.
5. **A sense of time.** The brain has units that hold on to the past at many different speeds, so the pattern across them carries how long ago things happened. Every rate is defined in world seconds, and the speed of the world is never tied to the speed of thinking.
6. **No ceilings.** Units, regions, memory, senses and drives are open sets that can grow, and the design may use what a computer can do and a brain cannot (an exact record of its life, snapshots).
7. **Language is a channel, not the source of thought.** Planning and long-range understanding have to develop before language, from wordless machinery.

## Planned stages (a sketch that will change)

1. A simple body in a 2D world with one drive (energy): learn to move and to feed.
2. Time and late consequences: a world it can live well in, where consequences of its actions arrive at many delays.
3. Memory of its own life, and recall as its own act.
4. More drives; more complex bodies; 3D.
5. Peers.
6. A society.
7. Language: English by keyboard.

## Current state (November 2026)

Early and exploratory. Nothing here is a demonstrated result.

1. **Stage-0 world** (`amoeba/`): a planar physics world (MuJoCo) with a soft body of six two-jointed arms, 38 sense readings and 12 joint targets; food patches, half of them fixed and half moving; an energy drive with a set point. No gait or controller is built in.
2. **Late consequences** (`amoeba/`, `experiments/delay_school/`): kinds of food whose effect arrives after a delay, and exact snapshot and restore of the world.
3. **Reference behaviours** (`experiments/noise_floor/`, `experiments/upper_reference/`): still and random bodies and hand-written foragers, used only to calibrate what the world allows.
4. **A first learner** (`brain/`): recurrent units with time constants from 0.2 to 1000 simulated seconds and an online actor-critic with eligibility traces. Its credit reaches back only about ten seconds; it is a starting point, not the design.
5. **A bench for delayed consequences** (`experiments/event_toy/`): a minimal event stream for studying how a learning rule links a late consequence to an earlier choice, with controls that must fail.

**Open problem being worked on:** a learning rule that links consequences across long gaps and passes its own controls.

## Running the tests

Python 3.14, `mujoco` and `numpy`.

```
pip install mujoco numpy pytest
pytest
```

Some scripts refer to a comparison learner (`baselines/`) and to design notes that are not part of this snapshot.

## Licence

No licence has been chosen yet.

## Author

Miltos ([miltos22](https://github.com/miltos22))
