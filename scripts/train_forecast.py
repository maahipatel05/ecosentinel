"""
EcoSentinel Phase 6 — LSTM Training + TimesFM Zero-Shot Comparison
====================================================================
Fetches 1 year of PM2.5 + weather data for 4 cities, trains a stacked LSTM,
evaluates it on a held-out test set, and optionally compares against
Google TimesFM 2.0 zero-shot forecasting.

Run locally:
    python3 scripts/train_forecast.py

Run in Google Colab (recommended for GPU access):
    !pip install torch timesfm
    # Upload this file or copy-paste cells below

Output files (saved to data/model/):
    lstm_pm25.pt   — trained LSTM weights (~200 KB)
    scaler.json    — per-feature min/max normalization values

TimesFM dependency (training/comparison only, not needed for server):
    pip install timesfm
    If not installed, the script skips the zero-shot comparison section.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Import shared model definition from src/ ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from forecast import (
    PM25LSTM,
    N_FEATURES,
    SEQ_LEN,
    HIDDEN1,
    HIDDEN2,
    DROPOUT,
    normalize,
    denormalize_pm25,
    MODEL_DIR,
    MODEL_PATH,
    SCALER_PATH,
)

# ── Training configuration ─────────────────────────────────────────────────────

TRAIN_START  = "2024-01-01"
TRAIN_END    = "2024-12-31"

BATCH_SIZE   = 32
MAX_EPOCHS   = 200
PATIENCE     = 15    # stop if val loss doesn't improve for 15 consecutive epochs
LR           = 1e-3
TRAIN_FRAC   = 0.80
VAL_FRAC     = 0.10  # remaining 10% = test

CITIES = {
    "Delhi":       (28.6517,   77.2219),
    "Los Angeles": (34.0522, -118.2437),
    "Seoul":       (37.5665,  126.9780),
    "Krakow":      (50.0647,   19.9450),
}

AIR_QUALITY_URL   = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ── Optional TimesFM import ────────────────────────────────────────────────────

try:
    import timesfm
    _TIMESFM_AVAILABLE = True
except ImportError:
    _TIMESFM_AVAILABLE = False


# ── Data fetching ──────────────────────────────────────────────────────────────

async def _fetch_city_data(city: str, lat: float, lon: float) -> list[dict]:
    """
    Fetch daily PM2.5, wind_speed_10m, temperature_2m for TRAIN_START → TRAIN_END.
    Only returns days where all three values are present.
    """
    async with httpx.AsyncClient(timeout=60) as client:
        r_aq, r_wx = await asyncio.gather(
            client.get(
                AIR_QUALITY_URL,
                params={
                    "latitude":   lat,
                    "longitude":  lon,
                    "hourly":     "pm2_5",
                    "start_date": TRAIN_START,
                    "end_date":   TRAIN_END,
                    "timezone":   "UTC",
                },
            ),
            client.get(
                WEATHER_ARCHIVE_URL,
                params={
                    "latitude":   lat,
                    "longitude":  lon,
                    "hourly":     "temperature_2m,wind_speed_10m",
                    "start_date": TRAIN_START,
                    "end_date":   TRAIN_END,
                    "timezone":   "UTC",
                },
            ),
        )

    if r_aq.status_code != 200:
        raise RuntimeError(f"{city}: AQ API returned {r_aq.status_code}")
    if r_wx.status_code != 200:
        raise RuntimeError(f"{city}: Weather API returned {r_wx.status_code}")

    def _hourly_to_daily(times, values):
        by_date: dict[str, list[float]] = {}
        for t, v in zip(times, values):
            if v is None or v < 0:
                continue
            by_date.setdefault(t[:10], []).append(float(v))
        return {d: float(np.mean(vs)) for d, vs in by_date.items()}

    aq_h   = r_aq.json()["hourly"]
    wx_h   = r_wx.json()["hourly"]
    pm25_d = _hourly_to_daily(aq_h["time"],  aq_h["pm2_5"])
    wind_d = _hourly_to_daily(wx_h["time"],  wx_h["wind_speed_10m"])
    temp_d = _hourly_to_daily(wx_h["time"],  wx_h["temperature_2m"])

    common = sorted(set(pm25_d) & set(wind_d) & set(temp_d))
    return [
        {"date": d, "pm25": pm25_d[d], "wind": wind_d[d], "temp": temp_d[d]}
        for d in common
    ]


async def fetch_all_cities() -> dict[str, list[dict]]:
    print("\n── Fetching training data ──────────────────────────────────────────")
    all_data = {}
    for city, (lat, lon) in CITIES.items():
        print(f"  {city}...", end=" ", flush=True)
        t0 = time.time()
        rows = await _fetch_city_data(city, lat, lon)
        print(f"{len(rows)} days  ({time.time()-t0:.1f}s)")
        all_data[city] = rows
        await asyncio.sleep(0.5)  # polite pacing between requests
    return all_data


# ── Sliding window dataset ─────────────────────────────────────────────────────

def make_sequences(
    city_data: dict[str, list[dict]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert per-city daily rows into overlapping sliding windows.

    Each window:
      X[i] = days[j : j+SEQ_LEN]   → shape (SEQ_LEN, N_FEATURES)
      y[i] = days[j+SEQ_LEN].pm25  → float

    Returns X shape (N, SEQ_LEN, N_FEATURES) and y shape (N,).
    Windows from different cities do NOT overlap — no cross-city sequences.
    """
    X_all, y_all = [], []
    for city, rows in city_data.items():
        values = np.array(
            [[r["pm25"], r["wind"], r["temp"]] for r in rows],
            dtype=np.float32,
        )
        for i in range(len(values) - SEQ_LEN):
            X_all.append(values[i : i + SEQ_LEN])
            y_all.append(values[i + SEQ_LEN, 0])  # pm25 only as target
    return np.array(X_all), np.array(y_all, dtype=np.float32)


class PM25Dataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def split_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n      = len(X)
    n_tr   = int(n * TRAIN_FRAC)
    n_val  = int(n * VAL_FRAC)
    return (
        X[:n_tr],       y[:n_tr],
        X[n_tr:n_tr+n_val], y[n_tr:n_tr+n_val],
        X[n_tr+n_val:], y[n_tr+n_val:],
    )


def fit_scaler(X_train: np.ndarray) -> dict:
    """Compute per-feature min/max from training sequences only."""
    flat = X_train.reshape(-1, N_FEATURES)
    return {
        "features": ["pm25", "wind_speed_10m", "temperature_2m"],
        "min": flat.min(axis=0).tolist(),
        "max": flat.max(axis=0).tolist(),
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    scaler:  dict,
) -> PM25LSTM:
    X_tr_n = normalize(X_train.reshape(-1, N_FEATURES), scaler).reshape(X_train.shape)
    X_va_n = normalize(X_val.reshape(-1, N_FEATURES),   scaler).reshape(X_val.shape)

    # Normalize targets too (they are PM2.5, feature index 0)
    pm25_min = scaler["min"][0]
    pm25_max = scaler["max"][0]
    y_tr_n = (y_train - pm25_min) / (pm25_max - pm25_min + 1e-8)
    y_va_n = (y_val   - pm25_min) / (pm25_max - pm25_min + 1e-8)

    tr_loader = DataLoader(PM25Dataset(X_tr_n, y_tr_n), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(PM25Dataset(X_va_n, y_va_n), batch_size=BATCH_SIZE)

    model     = PM25LSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    patience_count = 0

    print("\n── Training LSTM ───────────────────────────────────────────────────")
    print(f"  Architecture: LSTM({HIDDEN1}) → LSTM({HIDDEN2}) → FC(1)")
    print(f"  Training sequences : {len(X_train)}")
    print(f"  Validation sequences: {len(X_val)}")
    print(f"  Batch size: {BATCH_SIZE}  |  Max epochs: {MAX_EPOCHS}  |  Patience: {PATIENCE}")
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Status'}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*20}")

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        tr_losses = []
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
        tr_loss = float(np.mean(tr_losses))

        # Validate
        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                va_losses.append(criterion(model(xb), yb).item())
        va_loss = float(np.mean(va_losses))

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_count = 0
            status = "✓ saved"
        else:
            patience_count += 1
            status = f"no improvement ({patience_count}/{PATIENCE})"

        if epoch % 10 == 0 or epoch <= 5 or patience_count >= PATIENCE:
            print(f"  {epoch:>6}  {tr_loss:>12.5f}  {va_loss:>12.5f}  {status}")

        if patience_count >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    # Restore best weights
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    print(f"\n  Best val loss: {best_val_loss:.5f}")
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    model: PM25LSTM,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: dict,
    label: str = "LSTM",
) -> dict:
    """Evaluate model on test set, return metrics dict with real-unit values."""
    X_n = normalize(X_test.reshape(-1, N_FEATURES), scaler).reshape(X_test.shape)
    x   = torch.tensor(X_n, dtype=torch.float32)

    with torch.no_grad():
        preds_norm = model(x).numpy()

    preds = np.array([denormalize_pm25(float(p), scaler) for p in preds_norm])
    preds = np.clip(preds, 0, None)

    mae  = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    mape = float(np.mean(np.abs((y_test - preds) / (y_test + 1e-8))) * 100)
    return {"model": label, "mae": mae, "rmse": rmse, "mape": mape}


def persistence_baseline(X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Persistence: predict tomorrow = today (last value of input window)."""
    preds = X_test[:, -1, 0]  # last day's PM2.5 (feature index 0, unnormalized)
    mae   = float(np.mean(np.abs(y_test - preds)))
    rmse  = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    mape  = float(np.mean(np.abs((y_test - preds) / (y_test + 1e-8))) * 100)
    return {"model": "Persistence (baseline)", "mae": mae, "rmse": rmse, "mape": mape}


def rolling_avg_baseline(X_test: np.ndarray, y_test: np.ndarray, window: int = 7) -> dict:
    """7-day rolling average baseline."""
    preds = X_test[:, -window:, 0].mean(axis=1)
    mae   = float(np.mean(np.abs(y_test - preds)))
    rmse  = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    mape  = float(np.mean(np.abs((y_test - preds) / (y_test + 1e-8))) * 100)
    return {"model": "7-day rolling avg (baseline)", "mae": mae, "rmse": rmse, "mape": mape}


# ── TimesFM zero-shot ──────────────────────────────────────────────────────────

def run_timesfm_comparison(
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    scaler:  dict,
) -> dict | None:
    """
    Run Google TimesFM 2.0 zero-shot forecast on PM2.5 context windows.

    TimesFM receives the raw (unnormalized) PM2.5 history — no training,
    no fine-tuning. This is the zero-shot evaluation from PHASE6.md Part B.3.

    Requires: pip install timesfm
    HuggingFace checkpoint: google/timesfm-2.0-200m-pytorch
    """
    if not _TIMESFM_AVAILABLE:
        print("\n── TimesFM comparison ──────────────────────────────────────────────")
        print("  ⚠️  timesfm not installed. Skipping zero-shot comparison.")
        print("  Install with:  pip install timesfm")
        print("  Then re-run this script to see the full 3-way comparison.")
        return None

    print("\n── TimesFM 2.0 zero-shot ───────────────────────────────────────────")
    print("  Loading google/timesfm-2.0-200m-pytorch from HuggingFace...")
    print("  (First run downloads ~800MB — cached afterwards)")

    try:
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="pytorch",
                per_core_batch_size=32,
                horizon_len=1,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-200m-pytorch",
            ),
        )
    except Exception as e:
        print(f"  ❌ Failed to load TimesFM: {e}")
        print("  Check: https://github.com/google-research/timesfm for API changes")
        return None

    # Feed only PM2.5 history (unnormalized) — TimesFM is a univariate model
    contexts = [X_test[i, :, 0].tolist() for i in range(len(X_test))]

    try:
        forecasts, _ = tfm.forecast(
            inputs=contexts,
            freq=[0] * len(contexts),  # 0 = high frequency (daily)
        )
        preds = np.array([float(f[0]) for f in forecasts])
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        return None

    preds = np.clip(preds, 0, None)
    mae   = float(np.mean(np.abs(y_test - preds)))
    rmse  = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    mape  = float(np.mean(np.abs((y_test - preds) / (y_test + 1e-8))) * 100)
    print(f"  ✅ TimesFM zero-shot complete — {len(preds)} test examples")
    return {"model": "TimesFM 2.0 (zero-shot)", "mae": mae, "rmse": rmse, "mape": mape}


# ── Results table ──────────────────────────────────────────────────────────────

def print_results(metrics_list: list[dict]) -> None:
    print("\n── Evaluation Results ──────────────────────────────────────────────")
    print(f"  {'Model':<34}  {'MAE':>8}  {'RMSE':>8}  {'MAPE':>8}")
    print(f"  {'-'*34}  {'-'*8}  {'-'*8}  {'-'*8}")
    for m in metrics_list:
        if m is None:
            continue
        print(
            f"  {m['model']:<34}  "
            f"{m['mae']:>7.2f}   {m['rmse']:>7.2f}   {m['mape']:>7.1f}%"
        )

    valid = [m for m in metrics_list if m is not None and "baseline" not in m["model"].lower()]
    baselines = [m for m in metrics_list if m is not None and "baseline" in m["model"].lower()]
    if valid and baselines:
        best_model = min(valid, key=lambda m: m["mae"])
        best_baseline = min(baselines, key=lambda m: m["mae"])
        improvement = (best_baseline["mae"] - best_model["mae"]) / best_baseline["mae"] * 100
        print(f"\n  Best model vs best baseline: {improvement:.1f}% lower MAE")
        print(f"  ({best_model['model']}  vs  {best_baseline['model']})")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 68)
    print("EcoSentinel — Phase 6: LSTM Training + TimesFM Comparison")
    print(f"Training window : {TRAIN_START} → {TRAIN_END}")
    print(f"Cities          : {', '.join(CITIES)}")
    print(f"Features        : PM2.5, wind_speed_10m, temperature_2m")
    print(f"Sequence length : {SEQ_LEN} days  →  predict day {SEQ_LEN+1}")
    print("=" * 68)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch data
    city_data = await fetch_all_cities()
    total_days = sum(len(v) for v in city_data.values())
    print(f"\n  Total daily rows fetched: {total_days} across {len(CITIES)} cities")

    # 2. Build sequences
    X, y = make_sequences(city_data)
    print(f"  Sliding windows created : {len(X)}  (shape {X.shape})")

    # 3. Split
    X_tr, y_tr, X_va, y_va, X_te, y_te = split_data(X, y)
    print(f"  Split → train {len(X_tr)} | val {len(X_va)} | test {len(X_te)}")

    # 4. Fit scaler on training data only
    scaler = fit_scaler(X_tr)
    SCALER_PATH.write_text(json.dumps(scaler, indent=2))
    print(f"  Scaler saved → {SCALER_PATH}")

    # 5. Train LSTM
    model = train_lstm(X_tr, y_tr, X_va, y_va, scaler)
    print(f"\n  Model saved → {MODEL_PATH}")

    # 6. Evaluate
    metrics = [
        persistence_baseline(X_te, y_te),
        rolling_avg_baseline(X_te, y_te),
        evaluate(model, X_te, y_te, scaler, label="LSTM (ours)"),
        run_timesfm_comparison(X_te, y_te, scaler),
    ]

    print_results(metrics)

    print("\n── Resume line ─────────────────────────────────────────────────────")
    lstm_m = next((m for m in metrics if m and m["model"].startswith("LSTM")), None)
    pers_m = next((m for m in metrics if m and "Persistence" in m["model"]), None)
    if lstm_m and pers_m:
        improvement = (pers_m["mae"] - lstm_m["mae"]) / pers_m["mae"] * 100
        print(
            f"  LSTM MAE={lstm_m['mae']:.1f} µg/m³ — {improvement:.0f}% lower than "
            f"persistence baseline across {len(CITIES)} global cities."
        )

    print("\n✅ Training complete.\n")


if __name__ == "__main__":
    asyncio.run(main())
