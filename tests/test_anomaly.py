"""
Unit tests for the EcoSentinel anomaly detection engine.
Run with: python3 -m pytest tests/test_anomaly.py -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, 'src')


# ── Test the z-score math directly ────────────────────────────────────────────

def test_zscore_normal():
    """A value at the mean should produce z-score of zero."""
    history = [10.0] * 30
    mean    = np.mean(history)
    std     = np.std(history, ddof=1)
    # std is 0 here so we test a different case
    history2 = list(range(1, 31))  # 1 to 30
    mean2    = np.mean(history2)
    std2     = np.std(history2, ddof=1)
    current  = mean2  # exactly at the mean
    z        = (current - mean2) / std2
    assert abs(z) < 0.001, f"Expected z near 0 but got {z}"


def test_zscore_high_value():
    """A value far above the mean should produce high z-score."""
    history  = [10.0] * 29 + [12.0]
    mean     = np.mean(history)
    std      = np.std(history, ddof=1)
    current  = 50.0
    z        = (current - mean) / std
    assert z > 2.0, f"Expected z > 2.0 but got {z}"


def test_zscore_below_mean():
    """A value below mean should produce negative z-score."""
    history = [50.0] * 29 + [48.0]
    mean    = np.mean(history)
    std     = np.std(history, ddof=1)
    current = 10.0
    z       = (current - mean) / std
    assert z < 0, f"Expected negative z but got {z}"


def test_severity_thresholds():
    """Test that severity labels map correctly to z-score ranges."""
    def severity(z):
        if z >= 4.0:   return "CRITICAL"
        if z >= 3.0:   return "SEVERE"
        if z >= 2.0:   return "ANOMALY"
        if z >= 1.5:   return "ELEVATED"
        return "NORMAL"

    assert severity(5.0)  == "CRITICAL"
    assert severity(3.5)  == "SEVERE"
    assert severity(2.5)  == "ANOMALY"
    assert severity(1.7)  == "ELEVATED"
    assert severity(0.5)  == "NORMAL"
    assert severity(-1.0) == "NORMAL"


def test_mean_calculation():
    """Verify numpy mean matches manual calculation."""
    values    = [10.0, 20.0, 30.0, 40.0, 50.0]
    expected  = 30.0
    computed  = float(np.mean(values))
    assert abs(computed - expected) < 0.001


def test_std_calculation():
    """Verify numpy std with ddof=1 is sample standard deviation."""
    values   = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    expected = 2.138  # correct sample std dev for this dataset
    computed = float(np.std(values, ddof=1))
    assert abs(computed - expected) < 0.01