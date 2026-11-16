"""Known-answer check of world-time constants (brief DO 6).

At dt = ref_s = 0.1 s every converted constant equals the published
per-step constant; at dt = 0.05 s each follows p = dt / ref_s exactly.
"""

from __future__ import annotations

from brain.learner import BrainConfig


def test_constants_at_dt_ref_and_at_half():
    cfg = BrainConfig()
    assert cfg.ref_s == 0.1
    assert cfg.gamma(0.1) == cfg.gamma_ref == 0.99
    assert cfg.trace_decay(0.1) == cfg.gamma_ref * cfg.lam_ref == 0.99 * 0.8
    assert cfg.beta(0.1) == cfg.beta_ref == 0.99995
    assert cfg.alpha_policy(0.1) == cfg.alpha_ref_policy == 1e-4
    assert cfg.alpha_value(0.1) == cfg.alpha_ref_value == 2e-4
    assert cfg.gamma(0.05) == 0.99**0.5
    assert cfg.trace_decay(0.05) == (0.99 * 0.8) ** 0.5
    assert cfg.beta(0.05) == 0.99995**0.5
    assert cfg.alpha_policy(0.05) == 1e-4 * 0.5
    assert cfg.alpha_value(0.05) == 2e-4 * 0.5
    print(
        "dt=0.1:",
        cfg.gamma(0.1),
        cfg.trace_decay(0.1),
        cfg.beta(0.1),
        cfg.alpha_policy(0.1),
        cfg.alpha_value(0.1),
    )
    print(
        "dt=0.05:",
        cfg.gamma(0.05),
        cfg.trace_decay(0.05),
        cfg.beta(0.05),
        cfg.alpha_policy(0.05),
        cfg.alpha_value(0.05),
    )
