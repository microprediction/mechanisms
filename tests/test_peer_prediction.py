"""Tests for peer-prediction mechanisms (Correlated Agreement).

Executable documentation: CA penalises the degenerate constant-report
equilibrium that defeats output agreement, and rewards truthful reporting over
both constant and adversarial (signal-flipping) deviations.
"""

import numpy as np
import pytest

from mechanisms import peer_prediction as pp


def _simulate_signals(n_agents, n_tasks, q=0.8, seed=0):
    """Each task has a hidden binary state; agents observe it correctly w.p. q."""
    rng = np.random.default_rng(seed)
    state = rng.integers(0, 2, n_tasks)
    flip = rng.random((n_agents, n_tasks)) > q
    return state[None, :] ^ flip.astype(int)  # signal = state, flipped w.p. 1-q


def test_ca_penalises_constant_deviator():
    """A constant reporter (the output-agreement exploit) scores below the truthful crowd."""
    deficits = []
    for seed in range(5):
        sig = _simulate_signals(8, 300, q=0.8, seed=seed)
        reports = sig.copy()
        reports[0, :] = 0  # agent 0 ignores its signal and always says 0
        scores = pp.correlated_agreement(reports, n_signals=2)
        deficits.append(np.mean(scores[1:]) - scores[0])
    assert np.mean(deficits) > 0.0  # truthful crowd out-scores the constant deviator


def test_ca_no_free_reward_for_collusion():
    """If everyone reports the same constant, CA pays ~0 (no degenerate equilibrium)."""
    reports = np.zeros((6, 100), dtype=int)
    scores = pp.correlated_agreement(reports, n_signals=2)
    assert np.allclose(scores, 0.0, atol=1e-9)


def test_ca_truthful_beats_flipping():
    """An adversarial agent reporting the opposite of its signal scores below truth."""
    wins = 0
    for seed in range(5):
        sig = _simulate_signals(8, 300, q=0.8, seed=seed)
        reports = sig.copy()
        reports[0, :] = 1 - sig[0, :]  # agent 0 reports the flipped signal
        scores = pp.correlated_agreement(reports, n_signals=2)
        wins += scores[0] < np.mean(scores[1:])
    assert wins >= 4  # truthful crowd beats the flipper in (almost) every run


def test_ca_rewards_informative_truthful_agents():
    """With all agents truthful, every CA score is positive (signals are correlated)."""
    sig = _simulate_signals(8, 400, q=0.85, seed=3)
    scores = pp.correlated_agreement(sig, n_signals=2)
    assert np.all(scores > 0.0)
