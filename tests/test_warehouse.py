"""
Offline tests for src/warehouse.py (BigQuery warehouse).

Covers the pure logic only, so no Google Cloud login or network is needed:
  - new_rows keeps only readings newer than what BigQuery already holds
  - summarize_trends computes averages, worst/cleanest days, week over week
"""

import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from warehouse import new_rows, summarize_trends

LOADED_AT = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


# ── new_rows ───────────────────────────────────────────────────────────────────


def test_new_rows_first_load_keeps_everything():
    measurements = [("2026-10-17T18:30:00Z", 51.8), ("2026-10-18T08:30:00Z", 46.27)]
    rows = new_rows("Ahmedabad, India", 12235671, measurements, None, LOADED_AT)

    assert len(rows) == 2
    assert rows[0]["city"] == "Ahmedabad, India"
    assert rows[0]["sensor_id"] == 12235671
    assert rows[0]["pm25"] == 51.8
    assert rows[0]["measured_at"] == "2026-10-17T18:30:00+00:00"


def test_new_rows_skips_readings_already_in_bigquery():
    latest = datetime(2026, 10, 18, 8, 15, tzinfo=timezone.utc)
    measurements = [
        ("2026-10-18T08:00:00Z", 40.0),  # older than latest: skip
        ("2026-10-18T08:15:00Z", 41.0),  # equal to latest: skip
        ("2026-10-18T08:30:00Z", 46.27),  # newer: keep
    ]
    rows = new_rows("Ahmedabad, India", 1, measurements, latest, LOADED_AT)

    assert [r["pm25"] for r in rows] == [46.27]


def test_new_rows_second_run_adds_nothing():
    measurements = [("2026-10-18T08:15:00Z", 41.0)]
    latest = datetime(2026, 10, 18, 8, 15, tzinfo=timezone.utc)
    assert new_rows("Ahmedabad, India", 1, measurements, latest, LOADED_AT) == []


def test_new_rows_drops_duplicate_timestamps():
    measurements = [("2026-10-18T08:30:00Z", 46.27), ("2026-10-18T08:30:00Z", 46.27)]
    assert len(new_rows("Ahmedabad, India", 1, measurements, None, LOADED_AT)) == 1


# ── summarize_trends ───────────────────────────────────────────────────────────


def day_row(d: date, avg: float, peak: float, readings: int = 96) -> dict:
    return {"day": d, "readings": readings, "avg_pm25": avg, "max_pm25": peak}


def test_summarize_empty_returns_none():
    assert summarize_trends([]) is None


def test_summarize_worst_cleanest_and_peak():
    daily = [
        day_row(date(2026, 9, 24), 59.3, 230.3),
        day_row(date(2026, 10, 4), 33.9, 65.8),
        day_row(date(2026, 10, 2), 67.3, 134.4),
    ]
    daily.sort(key=lambda d: d["day"])
    s = summarize_trends(daily)

    assert s["worst_day"]["day"] == date(2026, 10, 2)
    assert s["cleanest_day"]["day"] == date(2026, 10, 4)
    assert s["peak_reading"]["max_pm25"] == 230.3
    assert s["days_with_data"] == 3


def test_summarize_partial_day_cannot_be_worst_day():
    # Real Ahmedabad case: the window started mid-day on Sep 21, leaving
    # 15 readings of a 4-hour spike (avg 227.2) next to full ~85-reading days
    daily = [
        day_row(date(2026, 9, 21), 227.2, 270.1, readings=15),
        day_row(date(2026, 9, 22), 75.4, 207.8, readings=78),
        day_row(date(2026, 9, 23), 52.4, 112.3, readings=84),
        day_row(date(2026, 9, 24), 59.3, 230.3, readings=86),
    ]
    s = summarize_trends(daily)

    assert s["worst_day"]["day"] == date(2026, 9, 22)
    assert s["peak_reading"]["max_pm25"] == 270.1  # single readings still count


def test_summarize_week_over_week_is_reading_weighted():
    # Previous week averages 50, latest week averages 60: +20%
    previous = [day_row(date(2026, 10, d), 50.0, 80.0) for d in range(1, 8)]
    recent = [day_row(date(2026, 10, d), 60.0, 90.0) for d in range(8, 15)]
    s = summarize_trends(previous + recent)

    assert s["previous_7_avg"] == 50.0
    assert s["recent_7_avg"] == 60.0
    assert s["change_pct"] == 20.0
    assert s["latest_day"] == date(2026, 10, 14)


def test_summarize_overall_avg_weights_days_by_readings():
    # A day with 19 readings should count less than a day with 95
    daily = [
        day_row(date(2026, 9, 26), 100.0, 120.0, readings=19),
        day_row(date(2026, 9, 30), 40.0, 60.0, readings=95),
    ]
    s = summarize_trends(daily)
    assert s["overall_avg"] == round((100 * 19 + 40 * 95) / 114, 1)


def test_summarize_no_previous_week_gives_no_change():
    daily = [day_row(date(2026, 10, d), 50.0, 80.0) for d in range(12, 15)]
    assert summarize_trends(daily)["change_pct"] is None
