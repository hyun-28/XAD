"""Shared synthetic series for the Task B tests (no archive needed)."""
import numpy as np


def sine_with_spike(n=3000, period=50, spike_at=2400, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sin(2 * np.pi * np.arange(n) / period) + 0.05 * rng.standard_normal(n)
    x[spike_at:spike_at + 5] += 4.0
    return x
