"""
Score a simple ensemble of the LSTM and TimesFM 2.5 zero-shot forecasts.

The ensemble is the plain average of the two models' predictions, with equal
weights and nothing tuned on the test set. Uses the saved predictions only, so
no model needs to be re-run:
    data/eval_results/forecast_test_set.npz      (test set + LSTM predictions)
    data/eval_results/timesfm_zeroshot_preds.npz (TimesFM predictions)

Writes data/eval_results/ensemble_results.json.

Usage (from the project root):
    python scripts/eval_ensemble.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval_results"
CITIES = ["Delhi", "Los Angeles", "Seoul", "Krakow"]


def compute_metrics(y_true, y_pred, label):
    """Same metrics as the training notebooks (predictions clipped at 0)."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float32), 0, None)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return {"model": label, "mae": round(mae, 3), "rmse": round(rmse, 3), "mape": round(mape, 2)}


def main() -> None:
    exam = np.load(RESULTS_DIR / "forecast_test_set.npz")
    tfm = np.load(RESULTS_DIR / "timesfm_zeroshot_preds.npz")

    # Both files must describe the identical test days, in the same order
    assert (exam["date_te"] == tfm["date_te"]).all()
    assert (exam["city_te"] == tfm["city_te"]).all()

    y, city = exam["y_te"], exam["city_te"]
    predictions = {
        "Persistence (same as today)": exam["X_te"][:, -1, 0],
        "LSTM (ours)": exam["preds_lstm"],
        "TimesFM 2.5 (zero-shot)": tfm["preds_tfm"],
        "Ensemble (average of LSTM and TimesFM)": (exam["preds_lstm"] + tfm["preds_tfm"]) / 2,
    }

    overall = [compute_metrics(y, p, name) for name, p in predictions.items()]
    per_city = {
        c: [compute_metrics(y[city == c], p[city == c], name) for name, p in predictions.items()]
        for c in CITIES
    }

    print(f"{'Model':<40} {'MAE':>7} {'RMSE':>7} {'MAPE':>7}")
    for r in overall:
        print(f"{r['model']:<40} {r['mae']:>7.2f} {r['rmse']:>7.2f} {r['mape']:>6.1f}%")

    out = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "equal-weight average of LSTM and TimesFM 2.5 zero-shot predictions; no tuning",
        "test_days": int(len(y)),
        "overall": overall,
        "per_city": per_city,
    }
    path = RESULTS_DIR / "ensemble_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
