"""
Unit tests for src/forecast.py

Tests that don't require trained model weights:
  - PM25LSTM forward pass shape and output type
  - normalize / denormalize round-trip
  - _risk_label correctness

Tests that require trained weights (skipped if model missing):
  - predict_next_day returns expected dict shape
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from forecast import (
    PM25LSTM,
    N_FEATURES,
    SEQ_LEN,
    MODEL_PATH,
    normalize,
    denormalize_pm25,
    _risk_label,
)

# ── Model architecture ─────────────────────────────────────────────────────────

def test_lstm_output_shape():
    model = PM25LSTM()
    x = torch.randn(4, SEQ_LEN, N_FEATURES)  # batch of 4
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4,), f"Expected (4,), got {y.shape}"


def test_lstm_output_is_finite():
    model = PM25LSTM()
    x = torch.randn(1, SEQ_LEN, N_FEATURES)
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all(), "Model output contains NaN or Inf"


def test_lstm_single_sample():
    model = PM25LSTM()
    x = torch.randn(1, SEQ_LEN, N_FEATURES)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1,)


# ── Normalization ──────────────────────────────────────────────────────────────

def _make_scaler():
    return {
        "features": ["pm25", "wind_speed_10m", "temperature_2m"],
        "min": [0.0,  0.0, -20.0],
        "max": [500.0, 50.0,  50.0],
    }


def test_normalize_range():
    scaler = _make_scaler()
    data   = np.array([[0.0, 0.0, -20.0], [500.0, 50.0, 50.0]], dtype=np.float32)
    normed = normalize(data, scaler)
    assert normed[0].tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-5)
    assert normed[1].tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-4)


def test_denormalize_roundtrip():
    scaler = _make_scaler()
    original = 123.45
    mn, mx   = scaler["min"][0], scaler["max"][0]
    normed   = (original - mn) / (mx - mn)
    restored = denormalize_pm25(normed, scaler)
    assert abs(restored - original) < 0.01


def test_normalize_midpoint():
    scaler = _make_scaler()
    # PM2.5=250 is midpoint of [0, 500] → should normalize to ~0.5
    data   = np.array([[250.0, 25.0, 15.0]], dtype=np.float32)
    normed = normalize(data, scaler)
    assert abs(normed[0, 0] - 0.5) < 1e-4


# ── Risk labels ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pm25,expected", [
    (5.0,   "Good"),
    (12.0,  "Good"),
    (20.0,  "Moderate"),
    (35.4,  "Moderate"),
    (40.0,  "Unhealthy for Sensitive Groups"),
    (55.4,  "Unhealthy for Sensitive Groups"),
    (100.0, "Unhealthy"),
    (150.4, "Unhealthy"),
    (200.0, "Very Unhealthy"),
    (300.0, "Hazardous"),
])
def test_risk_labels(pm25, expected):
    assert _risk_label(pm25) == expected, f"PM2.5={pm25}: expected {expected!r}"


# ── Integration (requires trained model) ─────────────────────────────────────

@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not trained yet — run scripts/train_forecast.py first",
)
def test_predict_output_shape():
    import asyncio
    from forecast import predict_next_day

    result = asyncio.run(predict_next_day("Seoul"))
    assert "predicted_pm25" in result
    assert "risk_level"     in result
    assert "city"           in result
    assert result["predicted_pm25"] >= 0.0
    assert result["risk_level"] in {
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    }
