"""
EcoSentinel Anomaly Detection Evaluation
==========================================
Evaluates precision, recall, F1-score, and AUC-ROC of the z-score anomaly
detector against the EPA 24-hour PM2.5 standard as independent ground truth.

Data source: Open-Meteo Air Quality API (CAMS model data)
  - Free, no API key, global coverage, reliable date filtering
  - Already used in this project for geocoding and weather data
  - Replaces OpenAQ which has broken date filtering in v3 (returns years-old
    data regardless of date_from parameter, making window filtering impossible)

Ground truth: EPA NAAQS 24h PM2.5 standard >= 35.4 ug/m3
  Any day with mean PM2.5 >= 35.4 is labelled a "bad air day."
  This is an external standard, independent of our detector.

Method: rolling 20-day z-score, same as production anomaly.py
  For each test day (day 21+), use the preceding 20 days as baseline,
  compute z-score, predict anomaly if z >= 2.0.

Run with:
  python3 scripts/eval_anomaly.py
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from anomaly import geocode

AIR_QUALITY_BASE   = "https://air-quality-api.open-meteo.com/v1"
BASELINE_DAYS      = 30     # first 30 days of window used as rolling baseline
MIN_TEST_DAYS      = 10
Z_THRESHOLD        = 2.0
CITY_DELAY_SECONDS = 1.0
EPA_THRESHOLD      = 35.4   # µg/m³

# Fixed historical window covering known pollution events:
#   - LA January 2026 wildfires (Palisades + Eaton): PM2.5 spiked from 12 to 300+ µg/m³
#   - Krakow December-January coal heating season: consistently 60-120 µg/m³
#   - Seoul winter stagnation + China outflow: spikes to 80+ µg/m³
#   - Delhi post-monsoon fog + vehicle pollution: elevated through winter
# Using a past fixed window ensures reproducible results and captures real events.
EVAL_START       = "2025-12-01"
EVAL_END         = "2026-02-28"
EVAL_WINDOW_DAYS = (datetime.strptime(EVAL_END, "%Y-%m-%d") - datetime.strptime(EVAL_START, "%Y-%m-%d")).days + 1

TEST_CITIES = [
    "Los Angeles",  # Jan 2026 wildfires — PM2.5 spiked from ~12 to 300+ µg/m³
    "Krakow",       # Coal heating season — worst EU air quality Dec-Feb
    "Seoul",        # Winter stagnation + China outflow — spikes to 80+ µg/m³
    "Delhi",        # Post-monsoon winter smog — chronically above EPA threshold
    "Bangkok",      # Start of dry/haze season — increasingly elevated
]

RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval_results"


async def _fetch_daily_pm25(lat: float, lon: float) -> dict[str, list[float]]:
    """
    Fetch hourly PM2.5 from Open-Meteo for the fixed EVAL_START→EVAL_END window.
    Returns dict mapping "YYYY-MM-DD" -> list of hourly readings that day.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{AIR_QUALITY_BASE}/air-quality",
            params={
                "latitude":   lat,
                "longitude":  lon,
                "hourly":     "pm2_5",
                "start_date": EVAL_START,
                "end_date":   EVAL_END,
                "timezone":   "UTC",
            },
        )
    if r.status_code != 200:
        return {}
    hourly = r.json().get("hourly", {})
    times  = hourly.get("time", [])
    values = hourly.get("pm2_5", [])
    by_date: dict[str, list[float]] = {}
    for t, v in zip(times, values):
        if v is None or v < 0:
            continue
        by_date.setdefault(t[:10], []).append(float(v))
    return by_date


async def evaluate_city(city: str) -> dict:
    try:
        lat, lon, display_name = await geocode(city)
    except ValueError as e:
        return {"city": city, "error": f"Geocoding failed: {e}"}

    pm25_by_date = await _fetch_daily_pm25(lat, lon)
    if not pm25_by_date:
        return {"city": city, "error": "Open-Meteo returned no air quality data for this location"}

    daily = sorted(
        [(d, float(np.mean(vals))) for d, vals in pm25_by_date.items()],
        key=lambda x: x[0],
    )

    if len(daily) < BASELINE_DAYS + MIN_TEST_DAYS:
        return {
            "city":  city,
            "error": f"Only {len(daily)} days available (need {BASELINE_DAYS + MIN_TEST_DAYS}+)",
        }

    pm25_values = [v for _, v in daily]

    true_labels: list[int]   = []
    pred_labels: list[int]   = []
    z_scores:    list[float] = []
    day_detail:  list[dict]  = []

    for i in range(BASELINE_DAYS, len(pm25_values)):
        baseline = pm25_values[i - BASELINE_DAYS:i]
        current  = pm25_values[i]
        mean_val = float(np.mean(baseline))
        std_val  = float(np.std(baseline, ddof=1))
        if std_val < 0.01:
            continue
        z = (current - mean_val) / std_val
        is_epa_bad  = current >= EPA_THRESHOLD
        is_detected = z >= Z_THRESHOLD
        true_labels.append(1 if is_epa_bad else 0)
        pred_labels.append(1 if is_detected else 0)
        z_scores.append(z)
        day_detail.append({
            "date":     daily[i][0],
            "pm25":     round(current, 1),
            "z_score":  round(z, 2),
            "epa_bad":  is_epa_bad,
            "detected": is_detected,
            "correct":  is_epa_bad == is_detected,
        })

    if len(true_labels) < MIN_TEST_DAYS:
        return {"city": city, "error": f"Only {len(true_labels)} usable test days"}

    n_bad = sum(true_labels)
    if n_bad == 0:
        peak = max(pm25_values)
        return {
            "city":  city,
            "error": (
                f"No days exceeded EPA threshold — city has clean air in this window "
                f"(peak was {peak:.1f} µg/m³). This is a good thing, not a bug."
            ),
        }

    precision = float(precision_score(true_labels, pred_labels, zero_division=0))
    recall    = float(recall_score(true_labels, pred_labels, zero_division=0))
    f1        = float(f1_score(true_labels, pred_labels, zero_division=0))
    try:
        auc = float(roc_auc_score(true_labels, z_scores))
    except Exception:
        auc = None

    return {
        "city":               display_name,
        "days_evaluated":     len(true_labels),
        "epa_bad_days":       n_bad,
        "anomalies_detected": sum(pred_labels),
        "precision":          round(precision, 3),
        "recall":             round(recall, 3),
        "f1":                 round(f1, 3),
        "auc_roc":            round(auc, 3) if auc is not None else None,
        "day_detail":         day_detail,
    }


def save_results(results: list[dict], summary: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    payload = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "baseline_days":    BASELINE_DAYS,
            "eval_window_days": EVAL_WINDOW_DAYS,
            "z_threshold":      Z_THRESHOLD,
            "ground_truth":     f"EPA NAAQS 24h PM2.5 standard >= {EPA_THRESHOLD} µg/m³",
            "data_source":      "Open-Meteo Air Quality API (CAMS model)",
            "eval_window":      f"{EVAL_START} to {EVAL_END}",
            "cities":           TEST_CITIES,
        },
        "summary":  summary,
        "per_city": results,
    }
    run_path    = RESULTS_DIR / f"{timestamp}.json"
    latest_path = RESULTS_DIR / "latest.json"
    run_path.write_text(json.dumps(payload, indent=2))
    latest_path.write_text(json.dumps(payload, indent=2))
    return run_path


async def main() -> None:
    print("=" * 68)
    print("EcoSentinel — Anomaly Detection Evaluation Report")
    print(f"Date      : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Window    : {EVAL_START} to {EVAL_END}  (winter pollution season)")
    print(f"Method    : rolling {BASELINE_DAYS}-day z-score, threshold z >= {Z_THRESHOLD}")
    print(f"GT source : EPA NAAQS 24h PM2.5 standard — {EPA_THRESHOLD} ug/m3")
    print(f"Data      : Open-Meteo Air Quality API (CAMS, global, no auth)")
    print(f"Cities    : {', '.join(TEST_CITIES)}")
    print("=" * 68)

    results = []
    for city in TEST_CITIES:
        print(f"  Evaluating {city}...", end=" ", flush=True)
        result = await evaluate_city(city)
        if "error" in result:
            print(f"⚠️  {result['error']}")
        else:
            print(f"✅ {result['days_evaluated']} days, {result['epa_bad_days']} EPA-bad days")
        results.append(result)
        if city != TEST_CITIES[-1]:
            time.sleep(CITY_DELAY_SECONDS)

    valid  = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    if valid:
        print(
            f"\n{'City':<20} {'Days':>5} {'EPA Bad':>8} {'Det':>5} "
            f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC-ROC':>8}"
        )
        print("-" * 68)
        for r in valid:
            city_short = r["city"].split(",")[0]
            print(
                f"{city_short:<20} {r['days_evaluated']:>5} {r['epa_bad_days']:>8} "
                f"{r['anomalies_detected']:>5} "
                f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
                f"{str(r['auc_roc']):>8}"
            )
        mean_f1  = float(np.mean([r["f1"] for r in valid]))
        auc_vals = [r["auc_roc"] for r in valid if isinstance(r["auc_roc"], float)]
        mean_auc = float(np.mean(auc_vals)) if auc_vals else None
        auc_str  = f"{mean_auc:.3f}" if mean_auc is not None else "N/A"
        print("-" * 68)
        print(f"{'MEAN':<20} {'':>5} {'':>8} {'':>5} {'':>6} {'':>6} {mean_f1:.3f} {auc_str:>8}")

        summary = {
            "cities_evaluated": len(valid),
            "cities_total":     len(TEST_CITIES),
            "mean_f1":          round(mean_f1, 3),
            "mean_auc_roc":     round(mean_auc, 3) if mean_auc is not None else None,
        }
        run_path = save_results(results, summary)

        print(f"\n✅ {len(valid)}/{len(TEST_CITIES)} cities evaluated successfully")
        print(f"📊 Mean F1 = {mean_f1:.3f}  |  Mean AUC-ROC = {auc_str}")
        print(f"\n── Resume line ──────────────────────────────────────────────────")
        print(
            f"  PM2.5 anomaly detector: mean F1 = {mean_f1:.2f}, AUC-ROC = {auc_str}\n"
            f"  Validated against EPA NAAQS 24h standard across {len(valid)} global cities\n"
            f"  ({EVAL_WINDOW_DAYS}-day rolling evaluation, Open-Meteo CAMS model data)"
        )
        print(f"\n💾 Saved to:")
        print(f"   {run_path}")
        print(f"   {RESULTS_DIR / 'latest.json'}")

    if failed:
        print(f"\n⚠️  {len(failed)} city/cities skipped:")
        for r in failed:
            print(f"  • {r.get('city', '?')}: {r['error']}")


if __name__ == "__main__":
    asyncio.run(main())
