"""
Unit tests for the EcoSentinel causal inference engine.
Run with: python3 -m pytest tests/test_causal.py -v

These tests work entirely on synthetic, in-memory data — no live API
calls — so they are fast and deterministic. They test the four pure,
network-free building blocks of causal.py: build_paired_dataset,
run_dowhy_estimate, and compute_causal_probability.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from causal import (
    build_paired_dataset,
    run_dowhy_estimate,
    run_dowhy_refutations,
    compute_causal_probability,
    MIN_PAIRED_OBSERVATIONS,
)


# ── Helper: build a synthetic (X, W, Y) dataset with a known true effect ────

def _make_dataset(beta_x: float, n: int = 20, seed: int = 1) -> pd.DataFrame:
    """
    Y = 10 + beta_x * X + 0.4 * W + noise

    beta_x = 3.0 simulates a STRONG wildfire effect on PM2.5.
    beta_x = 0.0 simulates NO wildfire effect (PM2.5 driven only by wind/noise).

    The final row is forced (noiselessly) to represent "today": a clear
    high-fire-intensity day, so the test has a concrete "current observation"
    to feed into compute_causal_probability.
    """
    rng = np.random.default_rng(seed)
    W = rng.uniform(5, 20, n)
    X = rng.uniform(0, 30, n)
    noise = rng.normal(0, 1.5, n)
    Y = 10 + beta_x * X + 0.4 * W + noise

    df = pd.DataFrame({"X": X, "W": W, "Y": Y})
    last = n - 1
    df.loc[last, "X"] = 40.0
    df.loc[last, "W"] = 12.0
    df.loc[last, "Y"] = 10 + beta_x * 40.0 + 0.4 * 12.0  # noiseless "today"
    return df


# ── DAG construction / estimation does not error ────────────────────────────

def test_dag_construction_does_not_error():
    """CausalModel + identify_effect + estimate_effect should run cleanly
    on a well-formed dataset and return a numeric ATE, not raise or return None."""
    df = _make_dataset(beta_x=2.0, n=20)
    ate = run_dowhy_estimate(df)
    assert ate is not None
    assert isinstance(ate, float)


# ── Graceful degradation: insufficient paired observations ─────────────────

def test_build_paired_dataset_skips_days_missing_wind_or_pm25():
    """A day with PM2.5 data but no wind data must be dropped, not crash."""
    fire_days = {"2026-06-01": [], "2026-06-02": []}
    wind_daily = {"2026-06-01": 10.0}  # only one day has wind
    pm25_daily = {"2026-06-01": [50.0, 52.0], "2026-06-02": [80.0]}  # two days have PM2.5

    df = build_paired_dataset(23.0, 72.6, fire_days, wind_daily, pm25_daily)

    assert len(df) == 1  # only the day with BOTH wind and PM2.5 survives
    assert df.iloc[0]["date"] == "2026-06-01"


def test_build_paired_dataset_defaults_to_zero_fire_intensity():
    """A day with no detected hotspot must get X=0.0, not be dropped or error."""
    fire_days = {"2026-06-01": []}  # no hotspots detected that day
    wind_daily = {"2026-06-01": 10.0}
    pm25_daily = {"2026-06-01": [30.0]}

    df = build_paired_dataset(23.0, 72.6, fire_days, wind_daily, pm25_daily)

    assert len(df) == 1
    assert df.iloc[0]["X"] == 0.0


def test_insufficient_paired_observations_below_threshold():
    """Fewer than MIN_PAIRED_OBSERVATIONS valid days must be detectable,
    so get_causal_attribution knows to fall back to the z-score result."""
    fire_days = {f"2026-06-{d:02d}": [] for d in range(1, 6)}
    wind_daily = {f"2026-06-{d:02d}": 10.0 for d in range(1, 6)}
    pm25_daily = {f"2026-06-{d:02d}": [40.0] for d in range(1, 6)}  # only 5 days

    df = build_paired_dataset(23.0, 72.6, fire_days, wind_daily, pm25_daily)

    assert len(df) == 5
    assert len(df) < MIN_PAIRED_OBSERVATIONS


# ── causal_probability is always a valid probability ─────────────────────────

@pytest.mark.parametrize(
    "ate, x_now, excess",
    [
        (3.0, 40.0, 70.0),     # typical case
        (3.0, 40.0, 1.0),      # tiny excess -> would overshoot 1.0 without clipping
        (0.0, 40.0, 50.0),     # zero effect
        (3.0, 0.0, 50.0),      # no fire today
        (3.0, 40.0, 0.0),      # zero excess (no anomaly to explain)
        (-1.0, 40.0, 50.0),    # a negative ATE should never produce a negative probability
        (1e6, 1e6, 1.0),       # extreme values must still clip to [0, 1]
    ],
)
def test_causal_probability_always_in_bounds(ate, x_now, excess):
    p = compute_causal_probability(ate, x_now, excess)
    assert 0.0 <= p <= 1.0


# ── Sanity check: strong vs weak signal produces the right is_causal verdict ─

def test_strong_signal_produces_is_causal_true():
    """A dataset where PM2.5 is heavily driven by fire intensity (after
    controlling for wind) should yield a high causal probability."""
    df = _make_dataset(beta_x=3.0, n=20)
    ate = run_dowhy_estimate(df)
    assert ate is not None
    assert ate > 2.0  # should recover something close to the true beta_x=3.0

    x_now = df["X"].iloc[-1]
    excess = df["Y"].iloc[-1] - df["Y"].mean()
    probability = compute_causal_probability(ate, x_now, excess)
    is_causal = probability > 0.5

    assert is_causal is True


def test_weak_signal_produces_is_causal_false():
    """A dataset where PM2.5 has NO relationship to fire intensity (only
    wind/noise drive it) should yield a low causal probability."""
    df = _make_dataset(beta_x=0.0, n=20)
    ate = run_dowhy_estimate(df)
    assert ate is not None
    assert abs(ate) < 1.0  # should recover something close to the true beta_x=0.0

    x_now = df["X"].iloc[-1]
    excess = df["Y"].iloc[-1] - df["Y"].mean()
    probability = compute_causal_probability(ate, x_now, excess)
    is_causal = probability > 0.5

    assert is_causal is False


# ── Refutation tests produce valid structure ─────────────────────────────────

def test_refutations_produce_valid_structure():
    """run_dowhy_refutations should return exactly 4 entries (one per refuter),
    each with 'new_ate' and 'passed' keys, on a well-formed dataset."""
    df  = _make_dataset(beta_x=2.0, n=20)
    ate = run_dowhy_estimate(df)
    assert ate is not None

    results = run_dowhy_refutations(df, ate)

    assert len(results) == 4
    for name in ["random_common_cause", "placebo_treatment", "data_subset", "bootstrap"]:
        assert name in results
        assert "new_ate" in results[name]
        assert "passed" in results[name]
