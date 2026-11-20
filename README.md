# Amoeba Next

Amoeba Next is a research project that grows a mind inside a simulated world.
The subject is one organism living one continuous life: it has drives with
innate set points, it senses how far it is from them, and it develops through
stages inside the same world rather than being reset between experiments.
World time is never tied to how often or how fast the organism thinks; sensory
organs integrate physical signals between looks, so the same physical history
teaches the same thing at different rates of thinking. The long-term goal is
learning grounded in lived experience, with language arriving much later
through a keyboard channel rather than from training on text.

## What exists now

* **A 2D dish world.** A thick-fluid dish with a swimming body built on MuJoCo,
  food patches that deplete and regrow, smell fields, an energy drive, food
  kinds whose effects arrive after a delay, and exact snapshot/restore so a
  life can be branched for comparison.
* **The brain side.** Continuous-time learning in experience units: a recurrent
  layer with exact online sensitivities, observation normalisers, sparse
  initialisation, a value/policy network, and a streaming actor-critic learner
  with eligibility traces. A clock-free core rule works on the amounts a sense
  organ accumulated between looks, never on world time or look counts.
* **Reference behaviours and experiment scripts** for foraging, obstacle-free
  swimming, delayed-effect schooling, event-stream toys and an upper-bound
  reference swimmer.
* **`step0a/` — test rigs.** Small, self-contained rigs that check the
  mathematical foundations before they are wired into a life:
  experience time counted inside the sense organ (endogenous pacemaker,
  receptors, exact and censored readouts), a unit record buffer that keeps the
  stream a sense organ produced, and the late-consequence (far-value) readout
  that estimates the utility of a window of that stream.

## Install

Python 3.14 or newer is required.

```
python -m venv .venv
.venv/bin/pip install -e .
```

The dependencies declared in `pyproject.toml` are MuJoCo and NumPy; the brain
tests also use PyTorch.

## Run the tests

```
.venv/bin/python -m pytest -q
```

The `step0a` rigs keep their own tests beside the code:

```
.venv/bin/python -m pytest -q step0a/tests
```

## Author

Miltiadis Argianis Karakitsos
