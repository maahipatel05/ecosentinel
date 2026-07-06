"""
EcoSentinel Phase 6 — LSTM Forecast Inference
==============================================
Production inference module loaded by the MCP server.
Training happens separately in scripts/train_forecast.py.

Model: stacked LSTM (64 → 32 hidden units)
Input: last 30 days of [PM2.5, wind_speed_10m, temperature_2m]
Output: tomorrow's predicted PM2.5 (µg/m³)

Before calling predict_next_day(), run:
    python3 scripts/train_forecast.py
to generate data/model/lstm_pm25.pt and data/model/scaler.json.
"""

import json
import asyncio
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import torch
import torch.nn as nn

# ── Paths ───────────────────────────────────────────────────────────────────────

MODEL_DIR   = Path(__file__).parent.parent / "data" / "model"
MODEL_PATH  = MODEL_DIR / "lstm_pm25.pt"
SCALER_PATH = MODEL_DIR / "scaler.json"

# ── API endpoints ───────────────────────────────────────────────────────────────

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL     = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL     = "https://geocoding-api.open-meteo.com/v1/search"

# ── Architecture constants — must match train_forecast.py exactly ───────────────

N_FEATURES = 3   # [pm25, wind_speed_10m, temperature_2m]
SEQ_LEN    = 30  # days of history fed as input
HIDDEN1    = 64
HIDDEN2    = 32
DROPOUT    = 0.2


# ── Model definition ────────────────────────────────────────────────────────────

class PM25LSTM(nn.Module):
    """
    Two-layer stacked LSTM for PM2.5 next-day regression.

    Input : (batch, SEQ_LEN, N_FEATURES)
    Output: (batch,)  — predicted PM2.5 for day SEQ_LEN+1
    """

    def __init__(self) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(N_FEATURES, HIDDEN1, batch_first=True)
        self.lstm2 = nn.LSTM(HIDDEN1, HIDDEN2, batch_first=True)
        self.drop  = nn.Dropout(DROPOUT)
        self.fc    = nn.Linear(HIDDEN2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out    = self.drop(out[:, -1, :])
        return self.fc(out).squeeze(-1)


# ── Normalization helpers ────────────────────────────────────────────────────────

def normalize(data: np.ndarray, scaler: dict) -> np.ndarray:
    """Apply min-max normalization using a saved scaler dict."""
    mn = np.array(scaler["min"], dtype=np.float32)
    mx = np.array(scaler["max"], dtype=np.float32)
    return (data - mn) / (mx - mn + 1e-8)


def denormalize_pm25(value: float, scaler: dict) -> float:
    """Reverse min-max normalization for the PM2.5 feature (index 0)."""
    mn, mx = scaler["min"][0], scaler["max"][0]
    return float(value) * (mx - mn) + mn


# ── Model loader (singleton — loaded once at server startup) ────────────────────

_model:  PM25LSTM | None = None
_scaler: dict    | None = None


def _load_model() -> tuple[PM25LSTM, dict]:
    global _model, _scaler
    if _model is not None:
        return _model, _scaler

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\n"
            "Run:  python3 scripts/train_forecast.py"
        )
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}\n"
            "Run:  python3 scripts/train_forecast.py"
        )

    scaler = json.loads(SCALER_PATH.read_text())
    model  = PM25LSTM()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    _model  = model
    _scaler = scaler
    return model, scaler


# ── Data helpers ────────────────────────────────────────────────────────────────

async def _geocode(city: str) -> tuple[float, float, str]:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            GEOCODE_URL,
            params={"name": city, "count": 1, "format": "json"},
        )
        r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise ValueError(f"City not found: {city!r}")
    hit  = results[0]
    name = f"{hit.get('name', city)}, {hit.get('country', '')}"
    return hit["latitude"], hit["longitude"], name


async def _fetch_recent_days(lat: float, lon: float) -> np.ndarray:
    """
    Fetch the last SEQ_LEN days of PM2.5, wind speed, and temperature.
    Returns ndarray of shape (SEQ_LEN, 3): columns [pm25, wind_speed, temp].
    """
    end   = date.today()
    start = end - timedelta(days=SEQ_LEN + 5)  # extra buffer for missing days

    async with httpx.AsyncClient(timeout=30) as client:
        r_aq, r_wx = await asyncio.gather(
            client.get(
                AIR_QUALITY_URL,
                params={
                    "latitude":   lat,
                    "longitude":  lon,
                    "hourly":     "pm2_5",
                    "start_date": start.isoformat(),
                    "end_date":   end.isoformat(),
                    "timezone":   "UTC",
                },
            ),
            client.get(
                WEATHER_URL,
                params={
                    "latitude":   lat,
                    "longitude":  lon,
                    "hourly":     "temperature_2m,wind_speed_10m",
                    "start_date": start.isoformat(),
                    "end_date":   end.isoformat(),
                    "timezone":   "UTC",
                },
            ),
        )

    if r_aq.status_code != 200 or r_wx.status_code != 200:
        raise RuntimeError(
            f"Open-Meteo error: AQ={r_aq.status_code} WX={r_wx.status_code}"
        )

    def _hourly_to_daily(times: list, values: list) -> dict[str, float]:
        by_date: dict[str, list[float]] = {}
        for t, v in zip(times, values):
            if v is None or v < 0:
                continue
            by_date.setdefault(t[:10], []).append(float(v))
        return {d: float(np.mean(vs)) for d, vs in by_date.items()}

    aq_h   = r_aq.json()["hourly"]
    wx_h   = r_wx.json()["hourly"]
    pm25_d = _hourly_to_daily(aq_h["time"], aq_h["pm2_5"])
    wind_d = _hourly_to_daily(wx_h["time"], wx_h["wind_speed_10m"])
    temp_d = _hourly_to_daily(wx_h["time"], wx_h["temperature_2m"])

    common_dates = sorted(set(pm25_d) & set(wind_d) & set(temp_d))
    rows = [[pm25_d[d], wind_d[d], temp_d[d]] for d in common_dates]

    if len(rows) < SEQ_LEN:
        raise ValueError(
            f"Only {len(rows)} complete days available (need {SEQ_LEN}). "
            "Open-Meteo may have a data gap at this location."
        )

    return np.array(rows[-SEQ_LEN:], dtype=np.float32)


# ── Public API ──────────────────────────────────────────────────────────────────

def _risk_label(pm25: float) -> str:
    if pm25 <= 12.0:  return "Good"
    if pm25 <= 35.4:  return "Moderate"
    if pm25 <= 55.4:  return "Unhealthy for Sensitive Groups"
    if pm25 <= 150.4: return "Unhealthy"
    if pm25 <= 250.4: return "Very Unhealthy"
    return "Hazardous"


async def predict_next_day(city: str) -> dict:
    """
    Predict tomorrow's PM2.5 for a city using the trained LSTM.

    Returns:
        {
            "city":           str    — display name from geocoder
            "predicted_pm25": float  — µg/m³, clamped to >= 0
            "risk_level":     str    — EPA category label
            "model":          str    — "LSTM"
            "seq_len":        int    — days of history used
        }

    Raises:
        FileNotFoundError — if model weights haven't been trained yet
        ValueError        — if city not found or data gap
    """
    model, scaler = _load_model()
    lat, lon, display_name = await _geocode(city)
    raw   = await _fetch_recent_days(lat, lon)
    normed = normalize(raw, scaler)

    x = torch.tensor(normed[np.newaxis], dtype=torch.float32)  # (1, 30, 3)
    with torch.no_grad():
        y_norm = model(x).item()

    predicted = max(0.0, round(denormalize_pm25(y_norm, scaler), 1))

    return {
        "city":           display_name,
        "predicted_pm25": predicted,
        "risk_level":     _risk_label(predicted),
        "model":          "LSTM",
        "seq_len":        SEQ_LEN,
    }
